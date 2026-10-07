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
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from fvdb import (
    ConvolutionPlan,
    GridBatch,
    GridSerialization,
    JaggedTensor,
    permute_jagged,
    scaled_dot_product_attention,
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
