# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Point Transformer V3 layers for sparse voxel grids.

Point Transformer V3 (PTv3, Wu et al. 2024) replaces neighborhood search with serialization.
Each block sorts voxels along a space-filling curve and runs attention inside fixed-size
patches of the sorted sequence. A sparse convolution supplies positional information, and a
U-Net of pooling and unpooling stages changes resolution.

The layers mirror Pointcept's reference implementation so that configurations and results
carry over. Voxel features are :class:`~fvdb.JaggedTensor` rows ordered like the voxels of
their :class:`~fvdb.GridBatch`. Serialization state is passed explicitly as a
:class:`~fvdb.GridSerialization` built once per resolution level.

Components:

- :class:`SerializedAttention`: multi-head attention over serialized patches or windows.
- :class:`ConditionalPositionEncoding`: sparse convolution, linear layer and layer norm.
- :class:`PointTransformerV3Block`: position encoding, attention and MLP with residuals.
- :class:`SerializedPooling` / :class:`SerializedUnpooling`: resolution changes by a factor of 2.
- :class:`PointTransformerV3Embedding`: input stem.
- :class:`PointTransformerV3`: the full encoder-decoder network.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
import torch.nn as nn

from fvdb import (
    ConvolutionPlan,
    GridBatch,
    GridSerialization,
    JaggedTensor,
    permute_jagged,
    scaled_dot_product_attention,
    serialize,
)

from .modules import BatchNorm, DropPath, SparseConv3d, _trace_fvdb_nn_forward

NormKind = Literal["batch", "layer"]


class _PointNorm(nn.Module):
    """Batch norm (Pointcept settings) or layer norm over voxel features."""

    def __init__(self, kind: NormKind, channels: int) -> None:
        super().__init__()
        if kind == "batch":
            self.norm: nn.Module = BatchNorm(channels, eps=1e-3, momentum=0.01)
        elif kind == "layer":
            self.norm = nn.LayerNorm(channels)
        else:
            raise ValueError(f"norm must be 'batch' or 'layer', got {kind!r}")

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        if isinstance(self.norm, BatchNorm):
            return self.norm(data, grid)
        return self.norm(data)


@_trace_fvdb_nn_forward
class SerializedAttention(nn.Module):
    """
    Multi-head self-attention over serialized voxels.

    Voxels are reordered along the curve selected by ``order_index``, attended in patches of
    ``patch_size`` (or in a sliding window of ``window_size``), and restored to grid order.
    With both sizes ``0`` every voxel attends to every voxel in its grid.

    .. seealso:: :func:`fvdb.scaled_dot_product_attention` for the attention modes.

    Args:
        channels (int): Feature channels. Must be divisible by ``num_heads``.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch length for patch attention. Default: ``1024``.
        window_size (int): Window length for window attention. Default: ``0``.
        qkv_bias (bool): Add a bias to the query, key and value projection. Default: ``True``.
        qk_scale (float | None): Attention scale. ``None`` uses ``head_dim ** -0.5``. Default: ``None``.
        proj_drop (float): Dropout after the output projection. Default: ``0.0``.
        order_index (int): Which serialization order to use, taken modulo the number of orders.
            Default: ``0``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 1024,
        window_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        proj_drop: float = 0.0,
        order_index: int = 0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if patch_size > 0 and window_size > 0:
            raise ValueError("Set at most one of patch_size and window_size")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.order_index = order_index
        self.scale = qk_scale if qk_scale is not None else (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self) -> str:
        """Return the layer configuration shown in the module representation."""
        return (
            f"channels={self.channels}, num_heads={self.num_heads}, patch_size={self.patch_size}, "
            f"window_size={self.window_size}, order_index={self.order_index}"
        )

    def forward(self, data: JaggedTensor, serialization: GridSerialization) -> JaggedTensor:
        """
        Attend over serialized voxels.

        Args:
            data (JaggedTensor): Voxel features in grid order. Shape: ``(batch_size, num_voxels, channels)``.
            serialization (GridSerialization): Serialization of the grid that ``data`` lives on.

        Returns:
            result (JaggedTensor): Attention output in grid order, with the same shape as ``data``.
        """
        heads, head_dim = self.num_heads, self.channels // self.num_heads
        qkv = permute_jagged(self.qkv(data), serialization.perm(self.order_index))
        q, k, v = qkv.jdata.view(-1, 3, heads, head_dim).unbind(1)
        out = scaled_dot_product_attention(
            qkv.jagged_like(q),
            qkv.jagged_like(k),
            qkv.jagged_like(v),
            self.scale,
            patch_size=self.patch_size,
            window_size=self.window_size,
        )
        out = out.jagged_like(out.jdata.reshape(out.jdata.shape[0], self.channels))
        out = permute_jagged(out, serialization.inv_perm(self.order_index))
        return self.proj_drop(self.proj(out))


@_trace_fvdb_nn_forward
class ConditionalPositionEncoding(nn.Module):
    """
    Conditional position encoding: submanifold sparse convolution, linear layer and layer norm.

    The convolution gives each voxel information about its occupied neighbors, which is how PTv3
    encodes position. The caller adds the output to the input as a residual.

    Args:
        channels (int): Feature channels.
        kernel_size (int): Convolution kernel size. Default: ``3``.
        use_conv (bool): Include the convolution. Without it the module is a linear layer and
            layer norm. Default: ``True``.
    """

    def __init__(self, channels: int, kernel_size: int = 3, use_conv: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = SparseConv3d(channels, channels, kernel_size=kernel_size, bias=True) if use_conv else None
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, data: JaggedTensor, plan: ConvolutionPlan | None) -> JaggedTensor:
        """
        Encode voxel positions.

        Args:
            data (JaggedTensor): Voxel features. Shape: ``(batch_size, num_voxels, channels)``.
            plan (ConvolutionPlan | None): Submanifold plan for the grid of ``data`` with this
                module's kernel size and stride 1. Ignored when ``use_conv`` is ``False``.

        Returns:
            result (JaggedTensor): Encoded features with the same shape as ``data``.
        """
        if self.conv is not None:
            if plan is None:
                raise ValueError("ConditionalPositionEncoding with use_conv=True requires a ConvolutionPlan")
            data = self.conv(data, plan)
        return self.norm(self.linear(data))


@_trace_fvdb_nn_forward
class PointTransformerV3Block(nn.Module):
    """
    A Point Transformer V3 block.

    Applies, each with a residual connection, a conditional position encoding, serialized
    attention, and a two-layer MLP. With ``pre_norm`` the layer norms come before attention and
    the MLP, as in the reference model.

    Args:
        channels (int): Feature channels.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch length for patch attention. Default: ``1024``.
        window_size (int): Window length for window attention. Default: ``0``.
        mlp_ratio (float): Hidden width of the MLP relative to ``channels``. Default: ``4.0``.
        qkv_bias (bool): Add a bias to the query, key and value projection. Default: ``True``.
        qk_scale (float | None): Attention scale. ``None`` uses ``head_dim ** -0.5``. Default: ``None``.
        proj_drop (float): Dropout after projections. Default: ``0.0``.
        drop_path (float): Stochastic depth rate for the attention and MLP branches. Default: ``0.0``.
        order_index (int): Serialization order used by this block's attention. Default: ``0``.
        pre_norm (bool): Normalize before, instead of after, attention and the MLP. Default: ``True``.
        cpe_conv (bool): Include the convolution in the position encoding. Default: ``True``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 1024,
        window_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        order_index: int = 0,
        pre_norm: bool = True,
        cpe_conv: bool = True,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = ConditionalPositionEncoding(channels, use_conv=cpe_conv)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SerializedAttention(
            channels,
            num_heads,
            patch_size=patch_size,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
            order_index=order_index,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )
        self.drop_path = DropPath(drop_path)

    def forward(
        self, data: JaggedTensor, serialization: GridSerialization, cpe_plan: ConvolutionPlan | None
    ) -> JaggedTensor:
        """
        Apply the block.

        Args:
            data (JaggedTensor): Voxel features. Shape: ``(batch_size, num_voxels, channels)``.
            serialization (GridSerialization): Serialization of the grid that ``data`` lives on.
            cpe_plan (ConvolutionPlan | None): Submanifold kernel-3 plan for the grid, or ``None``
                when the position encoding has no convolution.

        Returns:
            result (JaggedTensor): Output features with the same shape as ``data``.
        """
        data = data + self.cpe(data, cpe_plan)

        branch = self.norm1(data) if self.pre_norm else data
        data = data + self.drop_path(self.attn(branch, serialization))
        if not self.pre_norm:
            data = self.norm1(data)

        branch = self.norm2(data) if self.pre_norm else data
        data = data + self.drop_path(self.mlp(branch))
        if not self.pre_norm:
            data = self.norm2(data)
        return data


@_trace_fvdb_nn_forward
class SerializedPooling(nn.Module):
    """
    Downsample by a factor of ``stride`` with a linear projection, max pooling, norm and GELU.

    Voxels are grouped by ``ijk // stride``, as Pointcept groups points by shifting their curve
    codes. Pointcept also starts every sample at coordinate 0, so grids built from Pointcept's
    ``grid_coord`` pool into the same clusters.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels.
        stride (int): Pooling factor. Default: ``2``.
        norm (str): ``"batch"`` or ``"layer"``. Default: ``"batch"``.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, norm: NormKind = "batch") -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = _PointNorm(norm, out_channels)
        self.act = nn.GELU()

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        """
        Pool features onto a coarser grid.

        Args:
            data (JaggedTensor): Voxel features on ``grid``. Shape: ``(batch_size, num_voxels, in_channels)``.
            grid (GridBatch): The fine grid.

        Returns:
            result (JaggedTensor): Pooled features. Shape: ``(batch_size, num_coarse_voxels, out_channels)``.
            coarse_grid (GridBatch): The coarse grid.
        """
        data, coarse_grid = grid.max_pool(self.stride, self.proj(data), stride=self.stride)
        return self.act(self.norm(data, coarse_grid)), coarse_grid


@_trace_fvdb_nn_forward
class SerializedUnpooling(nn.Module):
    """
    Upsample to a stored finer grid and add a projected skip connection.

    Both the coarse features and the skip features pass through a linear layer, norm and GELU.
    Each fine voxel then receives its parent's coarse features plus its own skip features.

    Args:
        in_channels (int): Coarse feature channels.
        skip_channels (int): Skip feature channels.
        out_channels (int): Output feature channels.
        stride (int): Upsampling factor; must match the pooling stride. Default: ``2``.
        norm (str): ``"batch"`` or ``"layer"``. Default: ``"batch"``.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, stride: int = 2, norm: NormKind = "batch"
    ) -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = _PointNorm(norm, out_channels)
        self.proj_skip = nn.Linear(skip_channels, out_channels)
        self.norm_skip = _PointNorm(norm, out_channels)
        self.act = nn.GELU()

    def forward(
        self, data: JaggedTensor, grid: GridBatch, skip_data: JaggedTensor, skip_grid: GridBatch
    ) -> tuple[JaggedTensor, GridBatch]:
        """
        Unpool features onto the skip grid.

        Args:
            data (JaggedTensor): Coarse features on ``grid``. Shape: ``(batch_size, num_voxels, in_channels)``.
            grid (GridBatch): The coarse grid.
            skip_data (JaggedTensor): Features on ``skip_grid``. Shape:
                ``(batch_size, num_fine_voxels, skip_channels)``.
            skip_grid (GridBatch): The fine grid that ``grid`` was pooled from.

        Returns:
            result (JaggedTensor): Fine features. Shape: ``(batch_size, num_fine_voxels, out_channels)``.
            fine_grid (GridBatch): ``skip_grid``.
        """
        data = self.act(self.norm(self.proj(data), grid))
        skip = self.act(self.norm_skip(self.proj_skip(skip_data), skip_grid))
        data, _ = grid.refine(self.stride, data, fine_grid=skip_grid)
        return skip + data, skip_grid


@_trace_fvdb_nn_forward
class PointTransformerV3Embedding(nn.Module):
    """
    Input stem: a submanifold convolution or linear layer, then norm and GELU.

    The reference model uses a kernel-5 submanifold convolution without bias. ``mode="linear"``
    replaces it with a per-voxel linear layer.

    Args:
        in_channels (int): Input feature channels.
        embed_channels (int): Output feature channels.
        mode (str): ``"conv"`` or ``"linear"``. Default: ``"conv"``.
        kernel_size (int): Convolution kernel size for ``mode="conv"``. Default: ``5``.
        norm (str): ``"batch"`` or ``"layer"``. Default: ``"batch"``.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        mode: Literal["conv", "linear"] = "conv",
        kernel_size: int = 5,
        norm: NormKind = "batch",
    ) -> None:
        super().__init__()
        if mode not in ("conv", "linear"):
            raise ValueError(f"mode must be 'conv' or 'linear', got {mode!r}")
        self.mode = mode
        self.kernel_size = kernel_size
        self.stem: nn.Module
        if mode == "conv":
            self.stem = SparseConv3d(in_channels, embed_channels, kernel_size=kernel_size, bias=False)
        else:
            self.stem = nn.Linear(in_channels, embed_channels)
        self.norm = _PointNorm(norm, embed_channels)
        self.act = nn.GELU()

    def forward(self, data: JaggedTensor, grid: GridBatch, plan: ConvolutionPlan | None) -> JaggedTensor:
        """
        Embed input features.

        Args:
            data (JaggedTensor): Input features on ``grid``. Shape: ``(batch_size, num_voxels, in_channels)``.
            grid (GridBatch): The input grid.
            plan (ConvolutionPlan | None): Submanifold plan with this module's kernel size for
                ``mode="conv"``. Ignored for ``mode="linear"``.

        Returns:
            result (JaggedTensor): Embedded features. Shape: ``(batch_size, num_voxels, embed_channels)``.
        """
        if self.mode == "conv":
            if plan is None:
                raise ValueError("PointTransformerV3Embedding with mode='conv' requires a ConvolutionPlan")
            data = self.stem(data, plan)
        else:
            data = self.stem(data)
        return self.act(self.norm(data, grid))


class _Stage(nn.Module):
    """One resolution level: an optional resolution change followed by blocks."""

    resample: nn.Module | None

    def __init__(self, resample: nn.Module | None, blocks: list[PointTransformerV3Block]) -> None:
        super().__init__()
        self.resample = resample
        self.blocks = nn.ModuleList(blocks)

    def run_blocks(
        self, data: JaggedTensor, serialization: GridSerialization, cpe_plan: ConvolutionPlan | None
    ) -> JaggedTensor:
        for block in self.blocks:
            data = block(data, serialization, cpe_plan)
        return data


def _as_tuple(value, name: str, length: int) -> tuple:
    values = tuple(value) if isinstance(value, (list, tuple)) else (value,) * length
    if len(values) != length:
        raise ValueError(f"{name} must have {length} entries, got {len(values)}")
    return values


@_trace_fvdb_nn_forward
class PointTransformerV3(nn.Module):
    """
    Point Transformer V3 encoder-decoder for sparse voxel grids.

    The model embeds input features, runs an encoder of serialized-attention stages separated by
    pooling, and, unless ``cls_mode`` is set, a decoder that unpools back to the input grid with
    skip connections. Argument names and defaults follow Pointcept's ``PT-v3m1`` with its ScanNet
    semantic segmentation config, so Pointcept configs port directly.

    Serialization follows Pointcept. Orders are shuffled at the input when ``shuffle_orders`` is
    set and after every pooling when ``shuffle_pooled_orders`` is set, in training and evaluation.
    Pooled levels continue the curves of the input level. For orderings and pooling clusters that
    match Pointcept exactly, build the grid from coordinates whose minimum is 0 in every grid, as
    Pointcept's ``grid_coord`` is.

    Patch and window attention need PyTorch 2.11 or newer and an SM80+ GPU (see
    :func:`fvdb.scaled_dot_product_attention`). Set every patch size to ``0`` for global attention
    within each grid.

    Args:
        in_channels (int): Input feature channels. Default: ``6``.
        order (str | Sequence[str]): Serialization orders from :data:`fvdb.SERIALIZATION_ORDERS`.
            Block ``i`` of each stage uses order ``i % len(order)``.
            Default: ``("z", "z-trans", "hilbert", "hilbert-trans")``.
        stride (Sequence[int]): Pooling stride between consecutive encoder stages, each a power of
            two. Default: ``(2, 2, 2, 2)``.
        enc_depths (Sequence[int]): Blocks per encoder stage. Default: ``(2, 2, 2, 6, 2)``.
        enc_channels (Sequence[int]): Channels per encoder stage. Default: ``(32, 64, 128, 256, 512)``.
        enc_num_head (Sequence[int]): Attention heads per encoder stage. Default: ``(2, 4, 8, 16, 32)``.
        enc_patch_size (int | Sequence[int]): Patch size per encoder stage. Default: ``1024``.
        dec_depths (Sequence[int]): Blocks per decoder stage, from shallow to deep.
            Default: ``(2, 2, 2, 2)``.
        dec_channels (Sequence[int]): Channels per decoder stage, from shallow to deep.
            Default: ``(64, 64, 128, 256)``.
        dec_num_head (Sequence[int]): Attention heads per decoder stage. Default: ``(4, 4, 8, 16)``.
        dec_patch_size (int | Sequence[int]): Patch size per decoder stage. Default: ``1024``.
        mlp_ratio (float): MLP hidden width relative to the channels. Default: ``4.0``.
        qkv_bias (bool): Add a bias to the query, key and value projections. Default: ``True``.
        qk_scale (float | None): Attention scale. ``None`` uses ``head_dim ** -0.5``. Default: ``None``.
        attn_drop (float): Attention dropout. Only ``0`` is supported, because the fused attention
            kernels have no dropout. Default: ``0.0``.
        proj_drop (float): Dropout after projections. Default: ``0.0``.
        drop_path (float): Largest stochastic depth rate; rates rise linearly over the blocks.
            Default: ``0.3``.
        pre_norm (bool): Normalize before attention and the MLP. Default: ``True``.
        shuffle_orders (bool): Shuffle the orders at the input level. Default: ``True``.
        shuffle_pooled_orders (bool): Shuffle the orders after each pooling, as Pointcept always
            does. Default: ``True``.
        cls_mode (bool): Build only the encoder and return features on the coarsest grid.
            Default: ``False``.
        window_size (int): If positive, use window attention of this size in every block instead
            of patch attention. Default: ``0``.
        embedding_mode (str): ``"conv"`` for the kernel-5 submanifold stem, or ``"linear"``.
            Default: ``"conv"``.
        norm (str): Norm for the stem, pooling and unpooling: ``"batch"`` as in Pointcept, or
            ``"layer"``. Blocks always use layer norm. Default: ``"batch"``.
        cpe_conv (bool): Include the sparse convolution in the position encoding. Default: ``True``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        order: str | Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: int | Sequence[int] = 1024,
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 64, 128, 256),
        dec_num_head: Sequence[int] = (4, 4, 8, 16),
        dec_patch_size: int | Sequence[int] = 1024,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        shuffle_pooled_orders: bool = True,
        cls_mode: bool = False,
        window_size: int = 0,
        embedding_mode: Literal["conv", "linear"] = "conv",
        norm: NormKind = "batch",
        cpe_conv: bool = True,
    ) -> None:
        super().__init__()
        if attn_drop != 0.0:
            raise ValueError("attn_drop > 0 is not supported: the fused attention kernels have no dropout")
        num_stages = len(enc_depths)
        if len(stride) != num_stages - 1:
            raise ValueError(f"stride must have {num_stages - 1} entries, got {len(stride)}")
        for s in stride:
            if s < 2 or s & (s - 1):
                raise ValueError(f"Each stride must be a power of two, got {s}")
        enc_channels = _as_tuple(enc_channels, "enc_channels", num_stages)
        enc_num_head = _as_tuple(enc_num_head, "enc_num_head", num_stages)
        enc_patch_size = _as_tuple(enc_patch_size, "enc_patch_size", num_stages)
        dec_patch_sizes: tuple[int, ...] = ()
        if not cls_mode:
            dec_depths = _as_tuple(dec_depths, "dec_depths", num_stages - 1)
            dec_channels = _as_tuple(dec_channels, "dec_channels", num_stages - 1)
            dec_num_head = _as_tuple(dec_num_head, "dec_num_head", num_stages - 1)
            dec_patch_sizes = _as_tuple(dec_patch_size, "dec_patch_size", num_stages - 1)

        self.order = (order,) if isinstance(order, str) else tuple(order)
        self.stride = tuple(stride)
        self.shuffle_orders = shuffle_orders
        self.shuffle_pooled_orders = shuffle_pooled_orders
        self.cls_mode = cls_mode
        self.cpe_conv = cpe_conv
        self.embedding = PointTransformerV3Embedding(in_channels, enc_channels[0], mode=embedding_mode, norm=norm)

        def make_block(channels: int, heads: int, patch: int, path: float, index: int) -> PointTransformerV3Block:
            return PointTransformerV3Block(
                channels,
                heads,
                patch_size=0 if window_size > 0 else patch,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                proj_drop=proj_drop,
                drop_path=path,
                order_index=index % len(self.order),
                pre_norm=pre_norm,
                cpe_conv=cpe_conv,
            )

        enc_rates = [float(r) for r in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = nn.ModuleList()
        for s in range(num_stages):
            rates = enc_rates[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            down = (
                SerializedPooling(enc_channels[s - 1], enc_channels[s], stride=stride[s - 1], norm=norm)
                if s > 0
                else None
            )
            blocks = [
                make_block(enc_channels[s], enc_num_head[s], enc_patch_size[s], rates[i], i)
                for i in range(enc_depths[s])
            ]
            self.enc.append(_Stage(down, blocks))

        # Decoder stages run from deep to shallow; dec_* arguments are indexed shallow to deep.
        self.dec = nn.ModuleList()
        if not cls_mode:
            dec_rates = [float(r) for r in torch.linspace(0, drop_path, sum(dec_depths))]
            channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(num_stages - 1)):
                rates = dec_rates[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])][::-1]
                up = SerializedUnpooling(channels[s + 1], enc_channels[s], channels[s], stride=stride[s], norm=norm)
                blocks = [
                    make_block(channels[s], dec_num_head[s], dec_patch_sizes[s], rates[i], i)
                    for i in range(dec_depths[s])
                ]
                self.dec.append(_Stage(up, blocks))

    def _cpe_plan(self, grid: GridBatch) -> ConvolutionPlan | None:
        if not self.cpe_conv:
            return None
        return ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid, target_grid=grid)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        """
        Run the network.

        The output lives on ``grid`` with ``dec_channels[0]`` channels, or in ``cls_mode`` on the
        coarsest grid with ``enc_channels[-1]`` channels.

        Args:
            data (JaggedTensor): Input features on ``grid``. Shape: ``(batch_size, num_voxels, in_channels)``.
            grid (GridBatch): The input grid.

        Returns:
            result (JaggedTensor): Output features.
            out_grid (GridBatch): The grid of ``result``.
        """
        serialization = serialize(grid, self.order, shuffle=self.shuffle_orders)
        stem_plan = None
        if self.embedding.mode == "conv":
            k = self.embedding.kernel_size
            stem_plan = ConvolutionPlan.from_grid_batch(kernel_size=k, stride=1, source_grid=grid, target_grid=grid)
        data = self.embedding(data, grid, stem_plan)

        skips: list[tuple[JaggedTensor, GridBatch, GridSerialization, ConvolutionPlan | None]] = []
        cpe_plan = None
        for s, stage in enumerate(cast(Sequence[_Stage], self.enc)):
            if stage.resample is not None:
                skips.append((data, grid, serialization, cpe_plan))
                data, coarse = stage.resample(data, grid)
                serialization = serialization.pooled(coarse, self.stride[s - 1], shuffle=self.shuffle_pooled_orders)
                grid = coarse
            cpe_plan = self._cpe_plan(grid)
            data = stage.run_blocks(data, serialization, cpe_plan)

        for stage in cast(Sequence[_Stage], self.dec):
            assert stage.resample is not None
            skip_data, skip_grid, serialization, cpe_plan = skips.pop()
            data, grid = stage.resample(data, grid, skip_data, skip_grid)
            data = stage.run_blocks(data, serialization, cpe_plan)
        return data, grid
