# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Sparse convolution information and utilities for FVDB.

This module provides the SparseConvPackInfo class for managing sparse 3D convolution
operations on GridBatch objects.
"""

from __future__ import annotations

from typing import overload

import torch

from . import GridBatch, _parse_device_string
from ._Cpp import ConvPackBackend
from ._Cpp import GridBatch as GridBatchCpp
from ._Cpp import JaggedTensor
from ._Cpp import SparseConvPackInfo as SparseConvPackInfoCpp
from .types import (
    JaggedTensorOrTensor,
    Vec3iOrScalar,
    is_JaggedTensorOrTensor,
    is_Vec3iOrScalar,
)


class SparseConvPackInfo:
    """
    Information for sparse 3D convolution operations on grid batches.

    This class stores precomputed mappings and data structures needed to efficiently
    perform sparse convolutions on GridBatch objects. It supports multiple backends
    including CUTLASS, gather-scatter, and implicit GEMM implementations.

    The class handles the mapping between input and output voxels for convolution
    operations, taking into account the sparse nature of the grids.
    """

    @overload
    def __init__(
        self,
        kernel_size: Vec3iOrScalar,
        stride: Vec3iOrScalar,
        source_grid: GridBatch,
        target_grid: GridBatch | None = None,
    ) -> None: ...
    @overload
    def __init__(self, *, impl: SparseConvPackInfoCpp) -> None: ...

    def __init__(
        self,
        # These are None only to support the case in which the impl is provided.
        kernel_size: Vec3iOrScalar | None = None,
        stride: Vec3iOrScalar | None = None,
        source_grid: GridBatch | None = None,
        # allowed to be None in normal per-element construction
        target_grid: GridBatch | None = None,
        # If the impl is provided, the other parameters must be None.
        # This is a PRIVATE API, only used internally by this class, should not be used by users.
        impl: SparseConvPackInfoCpp | None = None,
    ):
        """
        Initialize sparse convolution information.

        This can be initialized either by providing convolution parameters or by
        wrapping an existing C++ implementation.

        Args:
            kernel_size (int or 3-tuple): Size of the convolution kernel.
                REQUIRED for non-private API.
            stride (int or 3-tuple): Stride of the convolution.
                REQUIRED for non-private API.
            source_grid (GridBatch): The input grid for convolution.
                REQUIRED for non-private API.
            target_grid (GridBatch): The output grid for convolution.
                OPTIONAL If None, it will be computed based on kernel_size and stride.

        PRIVATE API:
            impl (SparseConvPackInfoCpp): Existing C++ implementation to wrap.
                REQUIRED for private API.
        """
        if impl is not None:
            if not (kernel_size is None and stride is None and source_grid is None and target_grid is None):
                raise ValueError("kernel_size, stride, source_grid, and target_grid must be None when impl is provided")
            self._impl = impl
        else:
            # All parameters must be provided if impl is not, except for target_grid.
            if kernel_size is None or stride is None or source_grid is None:
                raise ValueError("kernel_size, stride, and source_grid must be provided when impl is None")

            source_impl = None
            if source_grid is not None:
                # Import here to avoid circular dependency
                from .grid_batch import GridBatch

                if isinstance(source_grid, GridBatch):
                    source_impl = source_grid._impl
                else:
                    raise TypeError(f"Unsupported type for source_grid: {type(source_grid)}")

            target_impl = None
            if target_grid is not None:
                # Import here to avoid circular dependency
                from .grid_batch import GridBatch

                if isinstance(target_grid, GridBatch):
                    target_impl = target_grid._impl
                else:
                    raise TypeError(f"Unsupported type for target_grid: {type(target_grid)}")

            if not is_Vec3iOrScalar(kernel_size) or not is_Vec3iOrScalar(stride):
                raise TypeError(
                    f"kernel_size and stride must be of type Vec3iOrScalar, but got {type(kernel_size)} and {type(stride)}"
                )

            self._impl = SparseConvPackInfoCpp(kernel_size, stride, source_impl, target_impl)  # type: ignore

    def build_cutlass(self, benchmark: bool = False) -> None:
        """
        Build CUTLASS backend for sparse convolution operations.

        Prepares the CUTLASS (CUDA Templates for Linear Algebra Subroutines) backend
        for performing sparse convolutions. CUTLASS provides highly optimized CUDA
        kernels for tensor operations.

        Args:
            benchmark (bool): Whether to run benchmarking to select optimal kernels.
                Default is False.

        Returns:
            None  (this modifies the SparseConvPackInfo in place)
        """
        self._impl.build_cutlass(benchmark)

    def build_gather_scatter(self, use_me: bool = False) -> None:
        """
        Build gather-scatter backend for sparse convolution operations.

        Prepares the gather-scatter backend which implements sparse convolutions
        using explicit gather and scatter operations. This is typically more
        memory efficient but may be slower than specialized backends.

        Args:
            use_me (bool): Whether to use memory-efficient implementation.
                Default is False.

        Returns:
            None  (this modifies the SparseConvPackInfo in place)
        """
        self._impl.build_gather_scatter(use_me)

    def build_implicit_gemm(
        self,
        sorted: bool = False,
        split_mask_num: int = 1,
        training: bool = False,
        split_mask_num_bwd: int = 1,
        use_tf32: bool = False,
    ) -> None:
        """
        Build implicit GEMM backend for sparse convolution operations.

        Prepares the implicit GEMM (General Matrix Multiply) backend which reformulates
        sparse convolutions as matrix multiplications. This can be highly efficient
        on modern GPUs with tensor cores.

        Args:
            sorted (bool): Whether to sort operations for better memory access patterns.
                Default is False.
            split_mask_num (int): Number of mask splits for forward pass optimization.
                Default is 1.
            training (bool): Whether to prepare for training (enables backward pass).
                Default is False.
            split_mask_num_bwd (int): Number of mask splits for backward pass optimization.
                Default is 1.
            use_tf32 (bool): Whether to use TensorFloat-32 precision on compatible hardware.
                Default is False.

        Returns:
            None  (this modifies the SparseConvPackInfo in place)
        """
        self._impl.build_implicit_gemm(sorted, split_mask_num, training, split_mask_num_bwd, use_tf32)

    def build_lggs(self) -> None:
        """
        Build LGGS (Locally Grouped Gather-Scatter) backend for sparse convolution operations.

        Prepares the LGGS backend which optimizes sparse convolutions by grouping
        nearby operations to improve memory locality and computational efficiency.

        Returns:
            None  (this modifies the SparseConvPackInfo in place)
        """
        self._impl.build_lggs()

    def cpu(self) -> "SparseConvPackInfo":
        """
        Move SparseConvPackInfo to CPU device.

        Returns:
            SparseConvPackInfo: A new SparseConvPackInfo instance on CPU device.
        """
        return SparseConvPackInfo(impl=self._impl.cpu())

    def cuda(self) -> "SparseConvPackInfo":
        """
        Move SparseConvPackInfo to CUDA device.

        Returns:
            SparseConvPackInfo: A new SparseConvPackInfo instance on CUDA device.
        """
        return SparseConvPackInfo(impl=self._impl.cuda())

    def sparse_conv_3d(
        self,
        input: JaggedTensorOrTensor,
        weights: torch.Tensor,
        backend: ConvPackBackend = ConvPackBackend.GATHER_SCATTER,
    ) -> JaggedTensor:
        """
        Perform sparse 3D convolution.

        Applies a 3D convolution operation on sparse grid data using the precomputed
        mappings in this object.

        Args:
            input (JaggedTensor or torch.Tensor): Input features for each voxel.
                Shape: (total_input_voxels, in_channels).
            weights (torch.Tensor): Convolution weights.
                Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
            backend (ConvPackBackend): Which backend implementation to use.
                Default is GATHER_SCATTER.

        Returns:
            JaggedTensor: Output features after convolution.
                Shape: (total_output_voxels, out_channels).
        """
        return self._impl.sparse_conv_3d(input, weights, backend)

    def sparse_transpose_conv_3d(
        self,
        input: JaggedTensorOrTensor,
        weights: torch.Tensor,
        backend: ConvPackBackend = ConvPackBackend.GATHER_SCATTER,
    ) -> JaggedTensor:
        """
        Perform sparse 3D transpose convolution (deconvolution).

        Applies a 3D transpose convolution operation on sparse grid data using the
        precomputed mappings in this object.

        Args:
            input (JaggedTensor or torch.Tensor): Input features for each voxel.
                Shape: (total_input_voxels, in_channels).
            weights (torch.Tensor): Convolution weights.
                Shape: (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
            backend (ConvPackBackend): Which backend implementation to use.
                Default is GATHER_SCATTER.

        Returns:
            JaggedTensor: Output features after transpose convolution.
                Shape: (total_output_voxels, out_channels).
        """
        return self._impl.sparse_transpose_conv_3d(input, weights, backend)

    def to(self, to_device: torch.device | str) -> "SparseConvPackInfo":
        """
        Move SparseConvPackInfo to a target device.

        Args:
            to_device (torch.device or str): Target device to move to.
                Can be a torch.device object or device string (e.g., "cuda", "cpu").

        Returns:
            SparseConvPackInfo: A new SparseConvPackInfo instance on the target device.
        """
        if isinstance(to_device, str):
            to_device = _parse_device_string(to_device)
        return SparseConvPackInfo(impl=self._impl.to(to_device))

    # Properties that wrap GridBatch
    @property
    def source_grid(self) -> GridBatch:
        from .grid_batch import GridBatch

        return GridBatch(impl=self._impl.source_grid)

    @property
    def target_grid(self) -> GridBatch:
        from .grid_batch import GridBatch

        return GridBatch(impl=self._impl.target_grid)

    # Properties that don't need wrapping
    @property
    def block_kernel_in_idx(self) -> torch.Tensor | None:
        return self._impl.block_kernel_in_idx

    @property
    def block_kernel_ranges(self) -> torch.Tensor | None:
        return self._impl.block_kernel_ranges

    @property
    def block_kernel_rel_out_idx(self) -> torch.Tensor | None:
        return self._impl.block_kernel_rel_out_idx

    @property
    def halo_index_buffer(self) -> torch.Tensor | None:
        return self._impl.halo_index_buffer

    @property
    def kernel_size(self) -> tuple:
        return self._impl.kernel_size

    @property
    def neighborhood_map(self) -> torch.Tensor | None:
        return self._impl.neighborhood_map

    @property
    def neighborhood_sizes(self) -> torch.Tensor | None:
        return self._impl.neighborhood_sizes

    @property
    def out_in_map(self) -> torch.Tensor | None:
        return self._impl.out_in_map

    @property
    def out_in_map_bwd(self) -> torch.Tensor | None:
        return self._impl.out_in_map_bwd

    @property
    def output_index_buffer(self) -> torch.Tensor | None:
        return self._impl.output_index_buffer

    @property
    def reduced_sorted_mask(self) -> torch.Tensor | None:
        return self._impl.reduced_sorted_mask

    @property
    def reorder_loc(self) -> torch.Tensor | None:
        return self._impl.reorder_loc

    @property
    def reorder_loc_bwd(self) -> torch.Tensor | None:
        return self._impl.reorder_loc_bwd

    @property
    def reorder_out_in_map(self) -> torch.Tensor | None:
        return self._impl.reorder_out_in_map

    @property
    def reorder_out_in_map_bwd(self) -> torch.Tensor | None:
        return self._impl.reorder_out_in_map_bwd

    @property
    def sorted_mask(self) -> torch.Tensor | None:
        return self._impl.sorted_mask

    @property
    def sorted_mask_bwd_d(self) -> torch.Tensor | None:
        return self._impl.sorted_mask_bwd_d

    @property
    def sorted_mask_bwd_w(self) -> torch.Tensor | None:
        return self._impl.sorted_mask_bwd_w

    @property
    def stride(self) -> tuple:
        return self._impl.stride

    @property
    def use_me(self) -> bool:
        return self._impl.use_me

    @property
    def use_tf32(self) -> bool:
        return self._impl.use_tf32

    # Expose underlying implementation for compatibility
    @property
    def _sparseconvpackinfo(self):
        # Access underlying SparseConvPackInfoCpp - use sparingly during migration
        return self._impl
