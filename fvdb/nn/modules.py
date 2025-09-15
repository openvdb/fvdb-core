# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
from fvdb.types import (
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    to_Vec3i,
    to_Vec3iBroadcastable,
)
from torch.profiler import record_function

import fvdb
from fvdb import Grid, GridBatch, JaggedTensor, SparseConvPackInfo


def fvnn_module(module):
    # Register class as a module in fvdb.nn
    old_forward = module.forward

    def _forward(self, *args, **kwargs):
        with record_function(repr(self)):
            return old_forward(self, *args, **kwargs)

    module.forward = _forward
    return module


def _vec_is_all(v: torch.Tensor, i: int | float) -> bool:
    return bool(torch.all(torch.eq(v, i)).item())


def _is_same(grid1: GridBatch | Grid, grid2: GridBatch | Grid) -> bool:
    if isinstance(grid1, GridBatch):
        if isinstance(grid2, GridBatch):
            return grid1.is_same(grid2)
        else:
            return False
    elif isinstance(grid1, Grid):
        if isinstance(grid2, Grid):
            return grid1.is_same(grid2)
        else:
            return False
    else:
        raise TypeError(f"Unsupported type for is_same(): {type(grid1)} and {type(grid2)}")


# ------------------------------------------------------------------------------------------------


class _AvgPoolBase(nn.Module):
    r"""Base class for average pooling operations.

    Args:
        kernel_size: the size of the window to take average over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.
    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self.kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"


@fvnn_module
class AvgPool(_AvgPoolBase):
    r"""Applies a 3D average pooling over an input signal.

    Args:
        kernel_size: the size of the window to take average over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        return fine_grid.avg_pool(self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid)


@fvnn_module
class AvgPoolSingle(_AvgPoolBase):
    r"""Applies a 3D average pooling over an input signal. This is a single grid version of the AvgPool layer.

    Args:
        kernel_size: the size of the window to take average over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def forward(
        self, fine_data: torch.Tensor, fine_grid: Grid, coarse_grid: Grid | None = None
    ) -> tuple[torch.Tensor, Grid]:
        return fine_grid.avg_pool(self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid)


# ------------------------------------------------------------------------------------------------


class _MaxPoolBase(nn.Module):
    r"""Base class for max pooling operations.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self.kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"


@fvnn_module
class MaxPool(_MaxPoolBase):
    r"""Applies a 3D max pooling over an input signal.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        new_coarse_data, new_coarse_grid = fine_grid.max_pool(
            self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid
        )

        # TODO(chorvath): If this is desired behavior, build into GridBatch directly.
        new_coarse_data.jdata[torch.isinf(new_coarse_data.jdata)] = 0.0

        return new_coarse_data, new_coarse_grid


@fvnn_module
class MaxPoolSingle(_MaxPoolBase):
    r"""Applies a 3D max pooling over an input signal. This is a single grid version of the MaxPool layer.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def forward(
        self, fine_data: torch.Tensor, fine_grid: Grid, coarse_grid: Grid | None = None
    ) -> tuple[torch.Tensor, Grid]:
        new_coarse_data, new_coarse_grid = fine_grid.max_pool(
            self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid
        )

        # TODO(chorvath): If this is desired behavior, build into Grid directly.
        new_coarse_data[torch.isinf(new_coarse_data)] = 0.0

        return new_coarse_data, new_coarse_grid


# ------------------------------------------------------------------------------------------------


class _UpsamplingNearestBase(nn.Module):
    r"""Upsamples the input by a given scale factor using nearest upsampling.

    Args:
        scale_factor: the upsampling factor
    """

    def __init__(self, scale_factor: NumericMaxRank1):
        super().__init__()
        self.scale_factor = to_Vec3iBroadcastable(scale_factor, value_constraint=ValueConstraint.POSITIVE)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"


@fvnn_module
class UpsamplingNearest(_UpsamplingNearestBase):
    r"""Upsamples the input by a given scale factor using nearest upsampling.

    Args:
        scale_factor: the upsampling factor
    """

    def forward(
        self,
        coarse_data: JaggedTensor,
        coarse_grid: GridBatch,
        mask: JaggedTensor | None = None,
        fine_grid: GridBatch | None = None,
    ) -> tuple[JaggedTensor, GridBatch]:
        return coarse_grid.subdivide(self.scale_factor, coarse_data, mask, fine_grid=fine_grid)


@fvnn_module
class UpsamplingNearestSingle(_UpsamplingNearestBase):
    r"""Upsamples the input by a given scale factor using nearest upsampling. This is a single grid version of the UpsamplingNearest layer.

    Args:
        scale_factor: the upsampling factor
    """

    def forward(
        self,
        coarse_data: torch.Tensor,
        coarse_grid: Grid,
        mask: torch.Tensor | None = None,
        fine_grid: Grid | None = None,
    ) -> tuple[torch.Tensor, Grid]:
        return coarse_grid.subdivide(self.scale_factor, coarse_data, mask, fine_grid=fine_grid)


# ------------------------------------------------------------------------------------------------


class _FillFromGridBase(nn.Module):
    r"""
    Fill the content of input vdb-tensor to another grid.

    Args:
        default_value: the default value to fill in the new grid.
    """

    def __init__(self, default_value: float = 0.0) -> None:
        super().__init__()
        self.default_value = default_value


@fvnn_module
class FillFromGrid(_FillFromGridBase):
    r"""
    Fill the content of input vdb-tensor to another grid.

    Args:
        default_value: the default value to fill in the new grid.
    """

    def forward(
        self,
        data: JaggedTensor,
        grid: GridBatch,
        other_grid: GridBatch | None = None,
    ) -> tuple[JaggedTensor, GridBatch]:
        return (
            (other_grid.inject_from(grid, data, default_value=self.default_value), other_grid)
            if other_grid
            else (data, grid)
        )


@fvnn_module
class FillFromGridSingle(_FillFromGridBase):
    r"""
    Fill the content of input vdb-tensor to another grid. This is a single grid version of the FillFromGrid layer.

    Args:
        default_value: the default value to fill in the new grid.
    """

    def forward(
        self,
        data: torch.Tensor,
        grid: Grid,
        other_grid: Grid | None = None,
    ) -> tuple[torch.Tensor, Grid]:
        return (
            (other_grid.inject_from(grid, data, default_value=self.default_value), other_grid)
            if other_grid
            else (data, grid)
        )


# ------------------------------------------------------------------------------------------------


class _SparseConv3dBase(nn.Module):
    r"""Base class for SparseConv3d. Applies a 3D convolution over an input signal composed of several input
    planes, by performing a sparse convolution on the underlying VDB grid.

    Args:
        in_channels: number of channels in the input tensor
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution. Default value is 1
        bias: if ``True``, adds a learnable bias to the output. Default: ``True``
        transposed: if ``True``, uses a transposed convolution operator
    """

    CUTLASS_SUPPORTED_CHANNELS = [
        (32, 64),
        (64, 128),
        (128, 256),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (128, 64),
        (64, 32),
        (256, 128),
        (384, 256),
        (192, 128),
        (256, 512),
        (512, 256),
        (512, 512),
    ]

    """
    Backend for performing convolutions:
      - "default": for now it is 'igemm_mode1'
      - "legacy": the old slow implementation
      - "me": MinkowskiEngine implementation
      - "halo": 10x10x10 halo buffer implementation, stride 1, kernel 3
      - "cutlass": 4x4x6 cutlass implementation, stride 1, kernel 3, forward only, limited channels support
      - "lggs": kernel optimized for sparse structures
      - "igemm_mode0": unsorted
      - "igemm_mode1": sorted + split=1
      - "igemm_mode2": sorted + split=3
      - "dense": dense convolution
    """
    backend: str = "default"
    allow_tf32: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: NumericMaxRank1 = 3,
        stride: NumericMaxRank1 = 1,
        bias: bool = True,
        transposed: bool = False,
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)
        self.transposed = transposed

        if self.transposed:
            # Only change kernel size instead of module dict
            out_channels, in_channels = in_channels, out_channels

        self.kernel_volume = math.prod(self.kernel_size)
        if self.kernel_volume > 1:
            # Weight tensor is of shape (Do, Di, K0, K1, K2), but the underlying data is (K2, K1, K0, Di, Do)
            #   so we don't need to make a copy of the permuted tensor within the conv kernel.
            weight_shape = [out_channels, in_channels] + self.kernel_size.tolist()
            weight = torch.zeros(*weight_shape[::-1]).permute(4, 3, 2, 1, 0)
            self.weight = nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if not _vec_is_all(self.stride, 1):
            s += f", stride={self.stride}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        return s

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt((self.out_channels if self.transposed else self.in_channels) * self.kernel_volume)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def _dispatch_conv(
        self,
        in_feature: JaggedTensor | torch.Tensor,
        in_grid: GridBatch | Grid,
        in_kmap: SparseConvPackInfo | None,
        out_grid: GridBatch | Grid | None,
    ) -> tuple[GridBatch | Grid, JaggedTensor | torch.Tensor, SparseConvPackInfo | None]:

        backend = self.backend

        sm_arch = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10
        # tf32 requires compute capability >= 8.0 (Ampere)
        if self.allow_tf32 and self.weight.is_cuda:
            assert (
                sm_arch >= 8
            ), "TF32 requires GPU with compute capability >= 8.0. Please set fvdb.nn.SparseConv3d.allow_tf32 = False."

        # bf16 requires compute capability >= 8.0 (Ampere)
        if self.weight.is_cuda and self.weight.dtype == torch.bfloat16:
            assert sm_arch >= 8, "BF16 requires GPU with compute capability >= 8.0."

        # float16 requires compute capability >= 7.5 (Turing)
        if self.weight.is_cuda and self.weight.dtype == torch.float16:
            assert sm_arch >= 7.5, "FP16 requires GPU with compute capability >= 7.5."

        # cutlass, lggs, halo backends require compute capability >= 8.0 (Ampere)
        if backend in ["cutlass", "lggs", "halo"]:
            assert (
                torch.cuda.get_device_capability()[0] >= 8
            ), "cutlass, LGGS and Halo backends require GPU with compute capability >= 8.0."

        if backend == "cutlass" and (
            (not self.weight.is_cuda) or (self.in_channels, self.out_channels) not in self.CUTLASS_SUPPORTED_CHANNELS
        ):
            print(
                f"Cutlass backend does not support {self.in_channels} -> {self.out_channels} convolutions, falling back to default"
            )
            backend = "default"

        if backend == "lggs" and ((self.in_channels, self.out_channels) not in [(128, 128)]):
            print("LGGS backend only supports 128 to 128 convolution, falling back to default")
            backend = "default"

        if backend == "default":
            if (not self.weight.is_cuda) or in_feature.dtype == torch.float64:
                backend = "legacy"
            else:
                backend = "igemm_mode1"

        if backend == "halo" and _vec_is_all(self.stride, 1) and _vec_is_all(self.kernel_size, 3):
            return self._dispatch_halo(in_feature, in_grid, out_grid)  # type: ignore

        elif backend == "dense" and _vec_is_all(self.stride, 1):
            return self._dispatch_dense(in_feature, in_grid, out_grid)  # type: ignore

        else:
            return self._dispatch_default(in_feature, in_grid, out_grid, in_kmap)  # type: ignore

    def _build_kmap_and_convert_backend(self, kmap: fvdb.SparseConvPackInfo, backend: str) -> fvdb.ConvPackBackend:
        if backend in ["legacy", "me"]:
            kmap.build_gather_scatter(backend == "me")
            return fvdb.ConvPackBackend.GATHER_SCATTER

        elif backend == "cutlass":
            kmap.build_cutlass(benchmark=False)
            return fvdb.ConvPackBackend.CUTLASS

        elif backend == "igemm_mode0":
            kmap.build_implicit_gemm(
                sorted=False, split_mask_num=1, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32
            )
            return fvdb.ConvPackBackend.IGEMM

        elif backend == "igemm_mode1":
            kmap.build_implicit_gemm(
                sorted=True, split_mask_num=1, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32
            )
            return fvdb.ConvPackBackend.IGEMM

        elif backend == "igemm_mode2":
            kmap.build_implicit_gemm(
                sorted=True, split_mask_num=3, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32
            )
            return fvdb.ConvPackBackend.IGEMM

        elif backend == "lggs":
            kmap.build_lggs()
            return fvdb.ConvPackBackend.LGGS

        else:
            raise NotImplementedError(f"Backend {backend} is not supported")


@fvnn_module
class SparseConv3d(_SparseConv3dBase):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes, by performing a sparse convolution on the underlying VDB grid.

    Args:
        in_channels: number of channels in the input tensor
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution. Default value is 1
        bias: if ``True``, adds a learnable bias to the output. Default: ``True``
        transposed: if ``True``, uses a transposed convolution operator
    """

    def _dispatch_halo(
        self,
        in_feature: JaggedTensor,
        in_grid: GridBatch,
        out_grid: GridBatch | None,
    ) -> tuple[GridBatch, JaggedTensor, SparseConvPackInfo | None]:
        assert out_grid is None or _is_same(in_grid, out_grid)
        return in_grid, in_grid.sparse_conv_halo(in_feature, self.weight, 8), None

    def _dispatch_dense(
        self,
        in_feature: JaggedTensor,
        in_grid: GridBatch,
        out_grid: GridBatch | None,
    ) -> tuple[GridBatch, JaggedTensor, SparseConvPackInfo | None]:
        assert out_grid is None or _is_same(in_grid, out_grid)
        min_coord = in_grid.ijk.jdata.min(dim=0).values
        # BWHDC -> BCDHW
        dense_feature = in_grid.write_to_dense_czyx(in_feature, min_coord=min_coord)
        dense_feature = torch.nn.functional.conv3d(dense_feature, self.weight, padding=1, stride=1)
        # BCDHW -> BWHDC
        dense_feature = dense_feature.contiguous()
        dense_feature = in_grid.read_from_dense_czyx(dense_feature, dense_origins=min_coord)

        return in_grid, dense_feature, None

    def _dispatch_default(
        self,
        in_feature: JaggedTensor,
        in_grid: GridBatch,
        in_kmap: SparseConvPackInfo | None,
        out_grid: GridBatch | None,
    ) -> tuple[GridBatch, JaggedTensor, SparseConvPackInfo | None]:
        # Fallback to the default implementation
        can_cache = _vec_is_all(self.stride, 1) and (out_grid is None or _is_same(in_grid, out_grid))

        if in_kmap is not None and in_kmap.kernel_size == self.kernel_size and can_cache:
            kmap, out_grid = in_kmap, in_grid
        else:
            if self.transposed:
                assert out_grid is not None
                kmap, _ = out_grid.sparse_conv_kernel_map(self.kernel_size, self.stride, in_grid)
            else:
                kmap, out_grid = in_grid.sparse_conv_kernel_map(self.kernel_size, self.stride, out_grid)

        out_kmap = kmap if can_cache else None

        backend = self._build_kmap_and_convert_backend(kmap, self.backend)

        if not self.transposed:
            out_feature = kmap.sparse_conv_3d(in_feature, self.weight, backend)
        else:
            out_feature = kmap.sparse_transpose_conv_3d(in_feature, self.weight, backend)

        return out_grid, out_feature, out_kmap

    def forward(
        self,
        data: JaggedTensor,
        grid: GridBatch,
        out_grid: GridBatch | None = None,
        kmap: SparseConvPackInfo | None = None,
    ) -> tuple[JaggedTensor, GridBatch, SparseConvPackInfo | None]:

        if _vec_is_all(self.kernel_size, 1) and _vec_is_all(self.stride, 1):
            if out_grid is not None or kmap is not None:
                raise ValueError("out_grid and kmap must be None when kernel_size and stride are all 1")
            out_data = data.jdata.matmul(self.weight.transpose(0, 1))
            out_jagged_data = data.jagged_like(out_data)
            return out_jagged_data, grid, None
        else:
            if kmap is not None:
                if not _is_same(kmap.source_grid, grid):
                    raise ValueError("kmap.source_grid must be the same as grid")

                if out_grid is not None:
                    if not _is_same(out_grid, kmap.target_grid):
                        raise ValueError("out_grid must be the same as kmap.target_grid if not None")
                else:
                    out_grid = kmap.target_grid

                if kmap.kernel_size != self.kernel_size:
                    raise ValueError("kmap.kernel_size must be the same as kernel_size")

                if kmap.stride != self.stride:
                    raise ValueError("kmap.stride must be the same as stride")

            out_grid, out_data, out_kmap = self._dispatch_conv(data, grid, kmap, out_grid)  # type: ignore

        assert out_grid is not None, "Failed to compute output grid. This is a bug in the implementation."
        assert isinstance(out_data, JaggedTensor), "out_data must be a JaggedTensor"
        assert isinstance(out_grid, GridBatch), "out_grid must be a GridBatch"
        assert out_kmap is None or isinstance(
            out_kmap, SparseConvPackInfo
        ), "out_kmap must be a SparseConvPackInfo or None"

        if self.bias is not None:
            out_data.jdata = out_data.jdata + self.bias

        return out_data, out_grid, out_kmap


@fvnn_module
class SparseConv3dSingle(_SparseConv3dBase):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes, by performing a sparse convolution on the underlying VDB grid. This version supports single grid input and output.

    Args:
        in_channels: number of channels in the input tensor
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution. Default value is 1
        bias: if ``True``, adds a learnable bias to the output. Default: ``True``
        transposed: if ``True``, uses a transposed convolution operator
    """

    def _dispatch_halo(
        self,
        in_feature: torch.Tensor,
        in_grid: Grid,
        out_grid: Grid | None,
    ) -> tuple[Grid, torch.Tensor, SparseConvPackInfo | None]:
        assert out_grid is None or _is_same(in_grid, out_grid)
        return in_grid, in_grid.sparse_conv_halo(in_feature, self.weight, 8), None

    def _dispatch_dense(
        self,
        in_feature: torch.Tensor,
        in_grid: Grid,
        out_grid: Grid | None,
    ) -> tuple[Grid, torch.Tensor, SparseConvPackInfo | None]:
        assert out_grid is None or _is_same(in_grid, out_grid)
        min_coord = in_grid.ijk.min(dim=0).values
        # BWHDC -> BCDHW
        dense_feature = in_grid.write_to_dense_czyx(in_feature, min_coord=min_coord)
        dense_feature = torch.nn.functional.conv3d(dense_feature, self.weight, padding=1, stride=1)
        # BCDHW -> BWHDC
        dense_feature = dense_feature.contiguous()
        dense_feature = in_grid.read_from_dense_czyx(dense_feature, dense_origin=min_coord)

        return in_grid, dense_feature, None

    def _dispatch_default(
        self,
        in_feature: torch.Tensor,
        in_grid: Grid,
        in_kmap: SparseConvPackInfo | None,
        out_grid: Grid | None,
    ) -> tuple[Grid, torch.Tensor, SparseConvPackInfo | None]:
        # Fallback to the default implementation
        can_cache = _vec_is_all(self.stride, 1) and (out_grid is None or _is_same(in_grid, out_grid))

        if in_kmap is not None and in_kmap.kernel_size == self.kernel_size and can_cache:
            kmap, out_grid = in_kmap, in_grid
        else:
            if self.transposed:
                assert out_grid is not None
                kmap, _ = out_grid.sparse_conv_kernel_map(self.kernel_size, self.stride, in_grid)
            else:
                kmap, out_grid = in_grid.sparse_conv_kernel_map(self.kernel_size, self.stride, out_grid)

        out_kmap = kmap if can_cache else None

        backend = self._build_kmap_and_convert_backend(kmap, self.backend)

        if not self.transposed:
            out_feature = kmap.sparse_conv_3d(in_feature, self.weight, backend)
        else:
            out_feature = kmap.sparse_transpose_conv_3d(in_feature, self.weight, backend)

        return out_grid, out_feature.jdata, out_kmap

    def forward(
        self,
        data: torch.Tensor,
        grid: Grid,
        out_grid: Grid | None = None,
        kmap: SparseConvPackInfo | None = None,
    ) -> tuple[torch.Tensor, Grid, SparseConvPackInfo | None]:

        if _vec_is_all(self.kernel_size, 1) and _vec_is_all(self.stride, 1):
            if out_grid is not None or kmap is not None:
                raise ValueError("out_grid and kmap must be None when kernel_size and stride are all 1")
            out_data = data.matmul(self.weight.transpose(0, 1))
            return out_data, grid, None
        else:
            if kmap is not None:
                if not _is_same(kmap.source_grid_single, grid):
                    raise ValueError("kmap.source_grid must be the same as grid")

                if out_grid is not None:
                    if not _is_same(out_grid, kmap.target_grid_single):
                        raise ValueError("out_grid must be the same as kmap.target_grid if not None")
                else:
                    out_grid = kmap.target_grid_single

                if kmap.kernel_size != self.kernel_size:
                    raise ValueError("kmap.kernel_size must be the same as kernel_size")

                if kmap.stride != self.stride:
                    raise ValueError("kmap.stride must be the same as stride")

            out_grid, out_data, out_kmap = self._dispatch_conv(data, grid, kmap, out_grid)  # type: ignore

        assert out_grid is not None, "Failed to compute output grid. This is a bug in the implementation."
        assert isinstance(out_data, torch.Tensor), "out_data must be a torch.Tensor"
        assert isinstance(out_grid, Grid), "out_grid must be a Grid"
        assert out_kmap is None or isinstance(
            out_kmap, SparseConvPackInfo
        ), "out_kmap must be a SparseConvPackInfo or None"

        if self.bias is not None:
            out_data = out_data + self.bias

        return out_data, out_grid, out_kmap


# ------------------------------------------------------------------------------------------------


@fvnn_module
class GroupNorm(nn.GroupNorm):
    r"""Applies Group Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.GroupNorm` for detailed information.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"
        num_batches = grid.grid_count

        flat_data, flat_offsets = data.jdata, data.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[b] : flat_offsets[b + 1]]
            if feat.size(0) != 0:
                feat = feat.transpose(0, 1).contiguous().reshape(1, num_channels, -1)
                feat = super().forward(feat)
                feat = feat.reshape(num_channels, -1).transpose(0, 1)

                result_data[flat_offsets[b] : flat_offsets[b + 1]] = feat

        return grid.jagged_like(result_data)


@fvnn_module
class GroupNormSingle(nn.GroupNorm):
    r"""Applies Group Normalization over a torch.Tensor/Grid.
    See :class:`~torch.nn.GroupNorm` for detailed information.
    """

    def forward(self, data: torch.Tensor, grid: Grid) -> torch.Tensor:
        num_channels = data.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"

        result_data = torch.empty_like(data)

        feat = data
        if feat.size(0) != 0:
            feat = feat.transpose(0, 1).contiguous().reshape(1, num_channels, -1)
            feat = super().forward(feat)
            feat = feat.reshape(num_channels, -1).transpose(0, 1)

            result_data[0 : data.size(0)] = feat

        return result_data


# ------------------------------------------------------------------------------------------------


@fvnn_module
class BatchNorm(nn.BatchNorm1d):
    r"""Applies Batch Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.BatchNorm1d` for detailed information.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)


# BatchNorm is meaningless for batch size == 1 (Grid instead of GridBatch)


# ------------------------------------------------------------------------------------------------


@fvnn_module
class SyncBatchNorm(nn.SyncBatchNorm):
    r"""Applies distributed Batch Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.SyncBatchNorm` for detailed information.

    Only supports :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`fvdb.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        """Layer forward pass.

        Args:
            input: input JaggedTensor/GridBatch.

        Returns:
            Output JaggedTensor/GridBatch with batch norm applied to the feature dimension, across all ranks.
        """
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)

    @classmethod
    def convert_sync_batchnorm(cls, module: nn.Module, process_group: Any = None) -> nn.Module:
        r"""Helper function to convert
        :attr:`fvdb.nn.BatchNorm` layer in the model to
        :attr:`fvdb.nn.SyncBatchNorm` layer.

        Args:
            module: Module for which all :attr:`fvdb.nn.BatchNorm` layers will be converted to
                :attr:`fvdb.nn.SyncBatchNorm` layers.
            process_group: process group to scope synchronization, default is the whole world.

        Returns:
            The original module with the converted :attr:`fvdb.nn.SyncBatchNorm` layers.

        Example::

            >>> # Network with fvdb.nn.SyncBatchNorm layer
            >>> module = fvdb.nn.Sequential(
            >>>            fvdb.nn.Linear(20, 100),
            >>>            fvdb.nn.BatchNorm(100)
            >>>          )
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = fvdb.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, BatchNorm):
            module_output = cls(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


# SyncBatchNorm is meaningless for batch size == 1 (Grid instead of GridBatch)


# ------------------------------------------------------------------------------------------------

# Non-linear Activations


@fvnn_module
class ElementwiseMixin:
    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        res = super().forward(data.jdata)  # type: ignore
        return grid.jagged_like(res)


@fvnn_module
class ElementwiseMixinSingle:
    def forward(self, data: torch.Tensor, grid: Grid) -> torch.Tensor:
        return super().forward(data)  # type: ignore


class ELU(ElementwiseMixin, nn.ELU):
    r"""
    Applies the Exponential Linear Unit function element-wise:
    .. math::
    \text{ELU}(x) = \begin{cases}
    x, & \text{ if } x > 0\\
    \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
    \end{cases}
    """


class CELU(ElementwiseMixin, nn.CELU):
    r"""
    Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    """


class GELU(ElementwiseMixin, nn.GELU):
    r"""
    Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    """


class Linear(ElementwiseMixin, nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    """


class ReLU(ElementwiseMixin, nn.ReLU):
    r"""
    Applies the rectified linear unit function element-wise: :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """


class LeakyReLU(ElementwiseMixin, nn.LeakyReLU):
    r"""
    Applies the element-wise function: :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    """


class SELU(ElementwiseMixin, nn.SELU):
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = \lambda \left\{
    \begin{array}{lr}
    x, & \text{if } x > 0 \\
    \text{negative\_slope} \times e^x - \text{negative\_slope}, & \text{otherwise }
    \end{array}
    \right.`
    """


class SiLU(ElementwiseMixin, nn.SiLU):
    r"""
    Applies element-wise, :math:`\text{SiLU}(x) = x * \sigma(x)`, where :math:`\sigma(x)` is the sigmoid function.
    """


class Tanh(ElementwiseMixin, nn.Tanh):
    r"""
    Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}`
    """


class Sigmoid(ElementwiseMixin, nn.Sigmoid):
    r"""
    Applies element-wise, :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """


# Dropout Layers


class Dropout(ElementwiseMixin, nn.Dropout):
    r"""
    During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`
    using samples from a Bernoulli distribution. The elements to zero are randomized on every forward call.
    """


class ELUSingle(ElementwiseMixinSingle, nn.ELU):
    r"""
    Applies the Exponential Linear Unit function element-wise:
    .. math::
    \text{ELU}(x) = \begin{cases}
    x, & \text{ if } x > 0\\
    \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
    \end{cases}
    """


class CELUSingle(ElementwiseMixinSingle, nn.CELU):
    r"""
    Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    """


class GELUSingle(ElementwiseMixinSingle, nn.GELU):
    r"""
    Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    """


class LinearSingle(ElementwiseMixinSingle, nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    """


class ReLUSingle(ElementwiseMixinSingle, nn.ReLU):
    r"""
    Applies the rectified linear unit function element-wise: :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """


class LeakyReLUSingle(ElementwiseMixinSingle, nn.LeakyReLU):
    r"""
    Applies the element-wise function: :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    """


class SELUSingle(ElementwiseMixinSingle, nn.SELU):
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = \lambda \left\{
    \begin{array}{lr}
    x, & \text{if } x > 0 \\
    \text{negative\_slope} \times e^x - \text{negative\_slope}, & \text{otherwise }
    \end{array}
    \right.`
    """


class SiLUSingle(ElementwiseMixinSingle, nn.SiLU):
    r"""
    Applies element-wise, :math:`\text{SiLU}(x) = x * \sigma(x)`, where :math:`\sigma(x)` is the sigmoid function.
    """


class TanhSingle(ElementwiseMixinSingle, nn.Tanh):
    r"""
    Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}`
    """


class SigmoidSingle(ElementwiseMixinSingle, nn.Sigmoid):
    r"""
    Applies element-wise, :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """


# Dropout Layers


class DropoutSingle(ElementwiseMixinSingle, nn.Dropout):
    r"""
    During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`
    using samples from a Bernoulli distribution. The elements to zero are randomized on every forward call.
    """


# ------------------------------------------------------------------------------------------------
# Sequential
# torch.nn.Sequential is designed around layers which take a single input and produce a single output.
# fvdb layers always take a JaggedTensor as the first argument, but sometimes take a GridBatch
# as the second argument, and sometimes a GridBatch as the third argument.
# Sometimes they return just a JaggedTensor, sometimes a tuple of (JaggedTensor, GridBatch),
# and sometimes a tuple of (JaggedTensor, GridBatch, SparseConvPackInfo).


@fvnn_module
class Sequential(nn.Module):
    r"""A sequential container for fvdb neural network modules.

    This container properly handles the different input/output signatures of fvdb modules:
    - Modules returning JaggedTensor only implicitly use the same GridBatch as input
    - Modules returning (JaggedTensor, GridBatch) update the GridBatch for subsequent layers
    - SparseConvPackInfo is ignored as this Sequential is for convenience when structural changes aren't needed

    Args:
        *args: Variable length argument list of modules to be added to the sequential container.
               Can be modules or an OrderedDict of modules.

    Example::
        >>> # Simple sequential with activation
        >>> seq = fvdb.nn.Sequential(
        ...     fvdb.nn.SparseConv3d(64, 128, 3),
        ...     fvdb.nn.BatchNorm(128),
        ...     fvdb.nn.ReLU()
        ... )
        >>> out_data, out_grid = seq(data, grid)

        >>> # Sequential with pooling (changes grid structure)
        >>> seq = fvdb.nn.Sequential(
        ...     fvdb.nn.SparseConv3d(32, 64, 3),
        ...     fvdb.nn.ReLU(),
        ...     fvdb.nn.MaxPool(2)
        ... )
        >>> out_data, out_grid = seq(data, grid)
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            # Handle OrderedDict or regular dict
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # Handle individual modules passed as arguments
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(dict(list(self.named_children())[idx]))
        else:
            return list(self.children())[idx]

    def __len__(self):
        return len(self._modules)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        """Forward pass through all modules in sequence.

        Args:
            data: Input jagged tensor
            grid: Input grid batch

        Returns:
            Tuple of (output_data, output_grid)
        """
        current_data = data
        current_grid = grid

        for module in self:
            result = module(current_data, current_grid)

            if isinstance(result, tuple):
                if len(result) == 2:
                    # (JaggedTensor, GridBatch)
                    current_data, current_grid = result
                elif len(result) == 3:
                    # (JaggedTensor, GridBatch, SparseConvPackInfo | None)
                    # Ignore SparseConvPackInfo as mentioned in requirements
                    current_data, current_grid, _ = result
                else:
                    raise ValueError(f"Unexpected return tuple length {len(result)} from module {module}")
            else:
                # JaggedTensor only - implicitly uses same GridBatch
                current_data = result
                # current_grid remains unchanged

        return current_data, current_grid


# ------------------------------------------------------------------------------------------------
# SequentialSingle
# torch.nn.Sequential is designed around layers which take a single input and produce a single output.
# fvdb layers always take a torch.Tensor as the first argument, but sometimes take a Grid
# as the second argument, and sometimes a Grid as the third argument.
# Sometimes they return just a torch.Tensor, sometimes a tuple of (torch.Tensor, Grid),
# and sometimes a tuple of (torch.Tensor, Grid, SparseConvPackInfo).


@fvnn_module
class SequentialSingle(nn.Module):
    r"""A sequential container for fvdb neural network modules.
    This is a single grid version of the Sequential layer.

    This container properly handles the different input/output signatures of fvdb modules:
    - Modules returning torch.Tensor only implicitly use the same Grid as input
    - Modules returning (torch.Tensor, Grid) update the Grid for subsequent layers
    - SparseConvPackInfo is ignored as this Sequential is for convenience when structural changes aren't needed

    Args:
        *args: Variable length argument list of modules to be added to the sequential container.
               Can be modules or an OrderedDict of modules.

    Example::
        >>> # Simple sequential with activation
        >>> seq = fvdb.nn.SequentialSingle(
        ...     fvdb.nn.SparseConv3d(64, 128, 3),
        ...     fvdb.nn.BatchNorm(128),
        ...     fvdb.nn.ReLU()
        ... )
        >>> out_data, out_grid = seq(data, grid)

        >>> # Sequential with pooling (changes grid structure)
        >>> seq = fvdb.nn.SequentialSingle(
        ...     fvdb.nn.SparseConv3d(32, 64, 3),
        ...     fvdb.nn.ReLU(),
        ...     fvdb.nn.MaxPool(2)
        ... )
        >>> out_data, out_grid = seq(data, grid)
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            # Handle OrderedDict or regular dict
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # Handle individual modules passed as arguments
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(dict(list(self.named_children())[idx]))
        else:
            return list(self.children())[idx]

    def __len__(self):
        return len(self._modules)

    def forward(self, data: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, Grid]:
        """Forward pass through all modules in sequence.

        Args:
            data: Input torch.Tensor
            grid: Input grid

        Returns:
            Tuple of (output_data, output_grid)
        """
        current_data = data
        current_grid = grid

        for module in self:
            result = module(current_data, current_grid)

            if isinstance(result, tuple):
                if len(result) == 2:
                    # (torch.Tensor, Grid)
                    current_data, current_grid = result
                elif len(result) == 3:
                    # (torch.Tensor, Grid, SparseConvPackInfo | None)
                    # Ignore SparseConvPackInfo as mentioned in requirements
                    current_data, current_grid, _ = result
                else:
                    raise ValueError(f"Unexpected return tuple length {len(result)} from module {module}")
            else:
                # torch.Tensor only - implicitly uses same Grid
                current_data = result
                # current_grid remains unchanged

        return current_data, current_grid
