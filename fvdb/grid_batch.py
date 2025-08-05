# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Sparse grid batch data structure and operations for FVDB.

This module provides the core GridBatch class for managing sparse voxel grids:

Classes:
- GridBatch: A batch of sparse voxel grids with support for efficient operations

Class-methods for creating GridBatch objects from various sources:
- from_zero_grids: Create a grid batch with grid-count = 0.
- from_empty: Create one or more grids with zero voxels.
- from_dense: Create from dense grid dimensions
- from_ijk: Create from explicit voxel coordinates
- from_mesh: Create from triangle meshes
- from_points: Create from point clouds
- from_nearest_voxels_to_points: Create from nearest voxels to points

Module-level functions for loading and saving grid batches:
- load_gridbatch/save_gridbatch: Load and save grid batches to/from .nvdb files

GridBatch supports operations like convolution, pooling, interpolation, ray casting,
mesh extraction, and coordinate transformations on sparse voxel data.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Sequence, cast, overload

import numpy as np
import torch

from . import _parse_device_string
from ._Cpp import ConvPackBackend
from ._Cpp import GridBatch as GridBatchCpp
from ._Cpp import JaggedTensor
from .types import (
    DeviceIdentifier,
    GridBatchIndex,
    NumericMaxRank1,
    NumericMaxRank2,
    resolve_device,
    to_NonNegativeVec3i,
    to_PositiveVec3fBatch,
    to_PositiveVec3i,
    to_Vec3f,
    to_Vec3fBatch,
    to_Vec3i,
    to_Vec3iBatch,
)

if TYPE_CHECKING:
    from .grid import Grid
    from .sparse_conv_pack_info import SparseConvPackInfo


class GridBatch:
    """
    A batch of sparse voxel grids with support for efficient operations.

    GridBatch represents a collection of sparse 3D voxel grids that can be processed
    together efficiently on GPU. Each grid in the batch can have different resolutions,
    origins, and voxel sizes. The class provides methods for common operations like
    sampling, convolution, pooling, and other operations.

    A GridBatch may contain zero grids, in which case it has no voxel sizes nor origins
    that can be queried. It may also contain one or more empty grids, which means grids that
    have zero voxels. An empty grid still has a voxel size and origin, which can be queried.

    The grids are stored in a sparse format where only active (non-empty) voxels are
    allocated, making it memory efficient for representing large volumes with sparse
    occupancy.

    Note:
        For creating grid batches with actual content, use the classmethods:
        - GridBatch.from_dense() for dense data
        - GridBatch.from_dense_axis_aligned_bounds() for dense defined by bounds
        - GridBatch.from_grid() for building from a Grid() instance
        - GridBatch.from_ijk() for voxel coordinates
        - GridBatch.from_mesh() for triangle meshes
        - GridBatch.from_nearest_voxels_to_points() for nearest voxel mapping
        - GridBatch.from_points() for point clouds
        - GridBatch.from_zero_grids() for zero grids
        - GridBatch.from_zero_voxels() for one or more empty grids (zero voxels)

        The GridBatch constructor is for internal use only, always use the classmethods.

    Attributes:
        max_grids_per_batch (int): Maximum number of grids that can be stored in a single batch.
    """

    # Class variable
    max_grids_per_batch: int = GridBatchCpp.max_grids_per_batch

    def __init__(self, *, impl: GridBatchCpp):
        """
        Constructor for internal use only. - use the Grid.from_* classmethods instead.
        """
        self._impl = impl

    # ============================================================
    #                  GridBatch from_* constructors
    # ============================================================

    @classmethod
    def from_dense(
        cls,
        num_grids: int,
        dense_dims: NumericMaxRank1,
        ijk_min: NumericMaxRank1 = 0,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        mask: torch.Tensor | None = None,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        A dense grid has a voxel for every coordinate in an axis-aligned box of Vec3,
        which can in turn be mapped to a world-space box.

        for each grid in the batch, the dense grid is defined by:
        - dense_dims: the size of the dense grid (shape [3,] = [W, H, D])
        - ijk_min: the minimum voxel index for each grid in the batch (Vec3i)
        - voxel_sizes: the world-space size of each voxel (Vec3d or scalar)
        - origins: the world-space coordinate of the 0,0,0 voxel of each grid
        - mask: indicates which voxels are "active" in the resulting grid.

        The voxel sizes and world space origins can be per-grid or per-batch.
        The ijk-min and sizes are the same for all grids in the batch.
        The mask is the same for all grids in the batch.

        Args:
            num_grids (int): Number of grids to populate.
            dense_dims (NumericMaxRank1): Dimensions of the dense grid, for all grids in the batch
                broadcastable to shape (3,), integer dtype
            ijk_min (NumericMaxRank1): Minimum voxel index for the grid, for all grids in the batch
                broadcastable to shape (3,), integer dtype
            voxel_sizes (NumericMaxRank2): World space size of each voxel, per-grid
                broadcastable to shape (num_grids, 3), floating dtype
            origins (NumericMaxRank2): World space coordinate of the 0,0,0 voxel of the grid, per-grid
                broadcastable to shape (num_grids, 3), floating dtype
            mask (torch.Tensor | None): Mask to apply to the grid,
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from mask, or uses "cpu" if mask is None.

        Returns:
            GridBatch: A new GridBatch object.
        """
        resolved_device = resolve_device(device, inherit_from=mask)

        dense_dims = to_PositiveVec3i(dense_dims)
        ijk_min = to_Vec3i(ijk_min)
        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_dense_grid(num_grids, dense_dims, ijk_min, voxel_sizes, origins, mask)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_dense_axis_aligned_bounds(
        cls,
        num_grids: int,
        dense_dims: NumericMaxRank1,
        bounds_min: NumericMaxRank1 = 0,
        bounds_max: NumericMaxRank1 = 1,
        voxel_center: bool = False,
        device: DeviceIdentifier = "cpu",
    ) -> "GridBatch":
        dense_dims = to_PositiveVec3i(dense_dims)
        bounds_min = to_Vec3f(bounds_min)
        bounds_max = to_Vec3f(bounds_max)

        if torch.any(bounds_max <= bounds_min):
            raise ValueError("bounds_max must be greater than bounds_min in all axes")

        if voxel_center:
            voxel_size = (bounds_max - bounds_min) / (dense_dims.to(torch.float64) - 1.0)
            origin = bounds_min
        else:
            voxel_size = (bounds_max - bounds_min) / dense_dims.to(torch.float64)
            origin = bounds_min + 0.5 * voxel_size

        return cls.from_dense(num_grids, dense_dims=dense_dims, voxel_sizes=voxel_size, origins=origin, device=device)

    @classmethod
    def from_grid(cls, grid: "Grid") -> "GridBatch":
        """
        Create a grid batch of batch size 1 from a single grid.

        Args:
            grid (Grid): The grid to create the grid batch from.

        Returns:
            GridBatch: A new GridBatch object.
        """
        return cls(impl=grid._impl)

    @classmethod
    def from_ijk(
        cls,
        ijk: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        Create a grid batch from voxel coordinates.

        Args:
            ijk (torch.Tensor): Voxel coordinates to populate.
                Shape: (num_voxels, 3) with integer coordinates.
            voxel_size (NumericMaxRank1): Size of each voxel,
                broadcastable to shape (3,), floating dtype
            origin (NumericMaxRank1): Origin of the grid,
                broadcastable to shape (3,), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from ijk.

        Returns:
            Grid: A new Grid object.
        """
        resolved_device = resolve_device(device, inherit_from=ijk)

        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_ijk(ijk, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_mesh(
        cls,
        mesh_vertices: JaggedTensor,
        mesh_faces: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        Create a grid batch from triangle meshes.

        Args:
            mesh_vertices (JaggedTensor): Vertices of the mesh.
                Shape: (batch_size, num_vertices, 3).
            mesh_faces (JaggedTensor): Faces of the mesh.
                Shape: (batch_size, num_faces, 3).
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid
                broadcastable to shape (batch_size, 3), floating dtype
            origins (NumericMaxRank2): Origin of the grid, per-grid
                broadcastable to shape (batch_size, 3), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from mesh_vertices.

        Returns:
            GridBatch: A new GridBatch object.
        """
        resolved_device = resolve_device(device, inherit_from=mesh_vertices)

        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_mesh(mesh_vertices, mesh_faces, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        Create a grid batch from the nearest voxels to a set of points.

        Args:
            points (JaggedTensor): Points to populate the grid from, per-grid
                Shape: (batch_size, num_points, 3).
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid
                broadcastable to shape (batch_size, 3), floating dtype
            origins (NumericMaxRank2): Origin of the grid, per-grid
                broadcastable to shape (batch_size, 3), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from points.

        Returns:
            GridBatch: A new GridBatch object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_nearest_voxels_to_points(points, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_points(
        cls,
        points: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        Create a grid batch from a point cloud.

        Args:
            points (JaggedTensor): Points to populate the grid from, per-grid
                Shape: (batch_size, num_points, 3).
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid
                broadcastable to shape (batch_size, 3), floating dtype
            origins (NumericMaxRank2): Origin of the grid, per-grid
                broadcastable to shape (num_grids, 3), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from points.

        Returns:
            GridBatch: A new GridBatch object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_points(points, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_zero_grids(cls, device: DeviceIdentifier = "cpu") -> "GridBatch":
        """
        Create a new GridBatch with zero grids. It retains its device identifier, but
        has no other information like voxel size or origin or bounding box. It will report
        grid_count == 0
        """
        return cls(impl=GridBatchCpp(device=resolve_device(device)))

    @classmethod
    def from_zero_voxels(
        cls, device: DeviceIdentifier = "cpu", voxel_sizes: NumericMaxRank2 = 1, origins: NumericMaxRank2 = 0
    ) -> "GridBatch":
        """
        Create a GridBatch with one or more zero-voxel grids on a specific device.

        An zero-voxel grid batch does not mean there are zero grids. It means that the grids have
        zero voxels. This constructor will create as many zero-voxel grids as the batch size
        of voxel_sizes and origins, defaulting to 1 grid, though for that case, you should use
        the single-grid Grid constructor instead.

        Args:
            device (DeviceIdentifier): The device to create the GridBatch on.
                Can be a string (e.g., "cuda", "cpu")or a torch.device object. Defaults to "cpu".
            voxel_sizes (NumericMaxRank2): The default size per voxel,
                broadcastable to shape (num_grids, 3), floating dtype
            origins (NumericMaxRank2): The default origin of the grid,
                broadcastable to shape (num_grids, 3), floating dtype


        Returns:
            GridBatch: A new zero-voxel GridBatch object.

        Examples:
            >>> grid_batch = GridBatch.from_zero_voxels("cuda", 1, 0)  # string
            >>> grid_batch = GridBatch.from_zero_voxels(torch.device("cuda:0"), 1, 0)  # device directly
            >>> grid_batch = GridBatch.from_zero_voxels(voxel_sizes=1, origins=0)  # defaults to CPU
        """
        resolved_device = resolve_device(device)
        voxel_sizes = to_PositiveVec3fBatch(voxel_sizes)
        origins = to_Vec3fBatch(origins)
        grid_batch_impl = GridBatchCpp(voxel_sizes=voxel_sizes, grid_origins=origins, device=resolved_device)
        return cls(impl=grid_batch_impl)

    # ============================================================
    #                Regular Instance Methods Begin
    # ============================================================

    def avg_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: JaggedTensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Downsample grid data using average pooling.

        Performs average pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the average of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid.
                broadcastable to shape (3,), integer dtype
            data (JaggedTensor): The voxel data to pool. Shape should be
                (B,total_voxels, channels).
            stride (NumericMaxRank1): The stride to use when pooling. If 0 (default),
                broadcastable to shape (3,), integer dtype
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The pooled data as a JaggedTensor
                - The coarse GridBatch containing the pooled structure
        """
        pool_factor = to_PositiveVec3i(pool_factor)
        stride = to_NonNegativeVec3i(stride)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.avg_pool(pool_factor, data, stride, coarse_grid_impl)

        return result_data, GridBatch(impl=cast(GridBatchCpp, result_grid_impl))

    def bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the bounding box of a specific grid in the batch.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: A tensor of shape (2, 3) containing the minimum and maximum
                coordinates of the bounding box in index space.
        """
        # There's a quirk with zero-voxel grids that we handle here.
        if self.has_zero_voxels_at(bi):
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.bbox_at(bi)

    def clip(
        self, features: JaggedTensor, ijk_min: NumericMaxRank2, ijk_max: NumericMaxRank2
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Clip the grid to a bounding box and return clipped features.

        Creates a new grid containing only the voxels that fall within the specified
        bounding box range [ijk_min, ijk_max] for each grid in the batch.

        Args:
            features (JaggedTensor): The voxel features to clip.
                Shape should be (num_grids, total_voxels, channels).
            ijk_min (NumericMaxRank2): Minimum bounds in index space for each grid.
                broadcastable to shape (num_grids, 3), integer dtype
            ijk_max (NumericMaxRank2): Maximum bounds in index space for each grid.
                broadcastable to shape (num_grids, 3), integer dtype

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The clipped features as a JaggedTensor
                - A new GridBatch containing only voxels within the bounds
        """
        ijk_min = to_Vec3iBatch(ijk_min)
        ijk_max = to_Vec3iBatch(ijk_max)

        result_features, result_grid_impl = self._impl.clip(features, ijk_min, ijk_max)
        return result_features, GridBatch(impl=result_grid_impl)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank2,
        ijk_max: NumericMaxRank2,
    ) -> "GridBatch":
        """
        Return a batch of grids representing the clipped version of this batch of grids.
        Each voxel `[i, j, k]` in the input batch is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            ijk_min (NumericMaxRank2): Index space minimum bound of the clip region.
            ijk_max (NumericMaxRank2): Index space maximum bound of the clip region.

        Returns:
            clipped_grid (GridBatch): A GridBatch representing the clipped version of this grid batch.
        """
        ijk_min = to_Vec3iBatch(ijk_min)
        ijk_max = to_Vec3iBatch(ijk_max)

        return GridBatch(impl=self._impl.clipped_grid(ijk_min, ijk_max))

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> "GridBatch":
        """
        Return a grid representing the coarsened version of this batch of grid.
        Each voxel `[i, j, k]` in the input is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            coarsening_factor (NumericMaxRank1): The factor by which to coarsen the grid.
                broadcastable to shape (3,), integer dtype

        Returns:
            coarsened_grid (GridBatch): A GridBatch representing the coarsened version of this grid batch.
        """
        coarsening_factor = to_PositiveVec3i(coarsening_factor)

        return GridBatch(impl=self._impl.coarsened_grid(coarsening_factor))

    def contiguous(self) -> "GridBatch":
        """
        Return a contiguous copy of the grid batch.

        Ensures that the underlying data is stored contiguously in memory,
        which can improve performance for subsequent operations.

        Returns:
            GridBatch: A new GridBatch with contiguous memory layout.
        """
        return GridBatch(impl=self._impl.contiguous())

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 0) -> "GridBatch":
        """
        Return a batch of grids representing the convolution of this batch with a given kernel.

        Args:
            kernel_size (NumericMaxRank1): The size of the kernel to convolve with.
                broadcastable to shape (3,), integer dtype
            stride (NumericMaxRank1): The stride to use when convolving.
                broadcastable to shape (3,), integer dtype

        Returns:
            conv_grid (GridBatch): A GridBatch representing the convolution of this grid batch.
        """
        kernel_size = to_PositiveVec3i(kernel_size)
        stride = to_NonNegativeVec3i(stride)

        return GridBatch(impl=self._impl.conv_grid(kernel_size, stride))

    def coords_in_active_voxel(self, ijk: JaggedTensor) -> JaggedTensor:
        """
        Check if voxel coordinates are in active voxels.

        Args:
            ijk (JaggedTensor): Voxel coordinates to check.
                Shape: (batch_size, num_queries, 3) with integer coordinates.

        Returns:
            JaggedTensor: Boolean mask indicating which coordinates correspond to
                active voxels. Shape: (batch_size, num_queries).
        """
        return self._impl.coords_in_active_voxel(ijk)

    def cpu(self) -> "GridBatch":
        """
        Move the grid batch to CPU.

        Returns:
            GridBatch: A new GridBatch on CPU device.
        """
        return GridBatch(impl=self._impl.cpu())

    def cubes_in_grid(
        self, cube_centers: JaggedTensor, cube_min: NumericMaxRank1 = 0, cube_max: NumericMaxRank1 = 0
    ) -> JaggedTensor:
        """
        Check if axis-aligned cubes are fully contained within the grid.

        Tests whether cubes defined by their centers and bounds are completely inside
        the active voxels of the grid.

        Args:
            cube_centers (JaggedTensor): Centers of the cubes in world coordinates.
                Shape: (batch_size, num_cubes, 3).
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds.
                broadcastable to shape (3,), floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds.
                broadcastable to shape (3,), floating dtype

        Returns:
            JaggedTensor: Boolean mask indicating which cubes are fully contained in the grid.
                Shape: (batch_size, num_cubes).
        """
        cube_min = to_Vec3f(cube_min)
        cube_max = to_Vec3f(cube_max)

        return self._impl.cubes_in_grid(cube_centers, cube_min, cube_max)

    def cubes_intersect_grid(
        self, cube_centers: JaggedTensor, cube_min: NumericMaxRank1 = 0, cube_max: NumericMaxRank1 = 0
    ) -> JaggedTensor:
        """
        Check if axis-aligned cubes intersect with the grid.

        Tests whether cubes defined by their centers and bounds have any intersection
        with the active voxels of the grid.

        Args:
            cube_centers (JaggedTensor): Centers of the cubes in world coordinates.
                Shape: (batch_size, num_cubes, 3).
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds.
                broadcastable to shape (3,), floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds.
                broadcastable to shape (3,), floating dtype

        Returns:
            JaggedTensor: Boolean mask indicating which cubes intersect the grid.
                Shape: (batch_size, num_cubes).
        """
        cube_min = to_Vec3f(cube_min)
        cube_max = to_Vec3f(cube_max)

        return self._impl.cubes_intersect_grid(cube_centers, cube_min, cube_max)

    def cuda(self) -> "GridBatch":
        """
        Move the grid batch to CUDA device.

        Returns:
            GridBatch: A new GridBatch on CUDA device.
        """
        return GridBatch(impl=self._impl.cuda())

    def cum_voxels_at(self, bi: int) -> int:
        """
        Get the cumulative number of voxels up to and including a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            int: The cumulative number of voxels up to and including grid bi.
        """
        return self._impl.cum_voxels_at(bi)

    def dilated_grid(self, dilation: int) -> "GridBatch":
        """
        Return the grid dilated by a given number of voxels.

        Args:
            dilation (int): The dilation radius in voxels.

        Returns:
            GridBatch: A new GridBatch with dilated active regions.
        """
        return GridBatch(impl=self._impl.dilated_grid(dilation))

    def dual_bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the dual bounding box of a specific grid in the batch.

        The dual grid has voxel centers at the corners of the primal grid voxels.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: A tensor of shape (2, 3) containing the minimum and maximum
                coordinates of the dual bounding box in index space.
        """
        if self.has_zero_voxels_at(bi):
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.dual_bbox_at(bi)

    def dual_grid(self, exclude_border: bool = False) -> "GridBatch":
        """
        Return the dual grid where voxel centers correspond to corners of the primal grid.

        The dual grid is useful for staggered grid discretizations and finite difference operations.

        Args:
            exclude_border (bool): If True, excludes border voxels that would extend beyond
                the primal grid bounds. Default is False.

        Returns:
            GridBatch: A new GridBatch representing the dual grid.
        """
        return GridBatch(
            impl=self._impl.dual_grid(exclude_border),
        )

    def grid_to_world(self, ijk: JaggedTensor) -> JaggedTensor:
        """
        Convert grid (index) coordinates to world coordinates.

        Transforms voxel indices to their corresponding positions in world space
        using the grid's origin and voxel size.

        Args:
            ijk (JaggedTensor): Grid coordinates to convert.
                Shape: (batch_size, num_points, 3). Can be fractional for interpolation.

        Returns:
            JaggedTensor: World coordinates. Shape: (batch_size,num_points, 3).
        """
        return self._impl.grid_to_world(ijk)

    def has_same_address_and_grid_count(self, other: Any) -> bool:
        """
        Check if two grid batches have the same address and grid count.
        """
        if isinstance(other, (GridBatch, GridBatchCpp)):
            return self.address == other.address and self.grid_count == other.grid_count
        else:
            return False

    def has_zero_voxels_at(self, bi: int) -> bool:
        """
        Check if a specific grid in the batch is empty, which means it has zero voxels.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            bool: True if the grid is empty, False otherwise.
        """
        return self.num_voxels_at(bi) == 0

    def ijk_to_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Convert voxel coordinates to linear indices.

        Maps 3D voxel coordinates to their corresponding linear indices in the sparse storage.
        Returns -1 for coordinates that don't correspond to active voxels.

        Args:
            ijk (JaggedTensor): Voxel coordinates to convert.
                Shape: (batch_size, num_queries, 3) with integer coordinates.
            cumulative (bool): If True, returns cumulative indices across the entire batch.
                If False, returns per-grid indices. Default is False.

        Returns:
            JaggedTensor: Linear indices for each coordinate, or -1 if not active.
                Shape: (batch_size, num_queries).
        """
        return self._impl.ijk_to_index(ijk, cumulative)

    def ijk_to_inv_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Get inverse permutation for ijk_to_index.

        Args:
            ijk (JaggedTensor): Voxel coordinates to convert.
                Shape: (batch_size, num_queries, 3) with integer coordinates.
            cumulative (bool): If True, returns cumulative indices across the entire batch.
                If False, returns per-grid indices. Default is False.

        Returns:
            JaggedTensor: Inverse permutation for ijk_to_index.
                Shape: (batch_size, num_queries).
        """
        return self._impl.ijk_to_inv_index(ijk, cumulative)

    def inject_from(
        self,
        src_grid: "GridBatch",
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ) -> JaggedTensor:
        """
        Inject data from the source grid to this grid.
        This method copies sidecar data for voxels in the source grid to a sidecar corresponding to voxels in this grid.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        If you pass in the destination data (`dst`), it will be modified in-place.
        If `dst` is None, a new JaggedTensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            dst_grid (GridBatch): The destination grid to inject data into.
            src (JaggedTensor): Source data from this grid.
                Shape: (batch_size, -1, *).
            dst (JaggedTensor | None): Optional destination data to be modified in-place.
                Shape: (batch_size, -1, *) or None.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.

        Returns:
            JaggedTensor: The destination sidecar data after injection.
        """
        if dst is None:
            dst_shape = [self.total_voxels]
            dst_shape.extend(src.eshape)
            dst = self.jagged_like(torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device))

        if dst.eshape != src.eshape:
            raise ValueError(
                f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}"
            )

        src_grid._impl.inject_to(self._impl, src, dst)

        return dst

    def inject_from_ijk(
        self,
        src_ijk: JaggedTensor,
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ):
        """
        Inject data from source voxel coordinates to a sidecar for this grid.

        Args:
            src_ijk (JaggedTensor): Voxel coordinates in index space from which to copy data.
                Shape: (B, num_src_voxels, 3).
            src (JaggedTensor): Source data to inject. Must match the shape of the destination.
                Shape: (B, num_src_voxels, *).
            dst (JaggedTensor | None): Optional destination data to be modified in-place.
                If None, a new JaggedTensor will be created with the same element shape as src
                and filled with `default_value` for any voxels that do not have corresponding data in `src`.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                Default is 0.
        """

        if not isinstance(src_ijk, JaggedTensor):
            raise TypeError(f"src_ijk must be a JaggedTensor, but got {type(src_ijk)}")

        if not isinstance(src, JaggedTensor):
            raise TypeError(f"src must be a JaggedTensor, but got {type(src)}")

        if dst is None:
            dst_shape = [self.total_voxels]
            dst_shape.extend(src.eshape)
            dst = self.jagged_like(torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device))
        else:
            if not isinstance(dst, JaggedTensor):
                raise TypeError(f"dst must be a JaggedTensor, but got {type(dst)}")

        if dst.eshape != src.eshape:
            raise ValueError(
                f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}"
            )

        src_idx = self.ijk_to_index(src_ijk, cumulative=True).jdata
        src_mask = src_idx >= 0
        src_idx = src_idx[src_mask]
        dst.jdata[src_idx] = src.jdata[src_mask]
        return dst

    def inject_to(
        self,
        dst_grid: "GridBatch",
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ) -> JaggedTensor:
        """
        Inject data from this grid to a destination grid.
        This method copies sidecar data for voxels in this grid to a sidecar corresponding to voxels in the destination grid.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        If you pass in the destination data (`dst`), it will be modified in-place.
        If `dst` is None, a new JaggedTensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            dst_grid (GridBatch): The destination grid to inject data into.
            src (JaggedTensor): Source data from this grid.
                Shape: (batch_size, -1, *).
            dst (JaggedTensor | None): Optional destination data to be modified in-place.
                Shape: (batch_size, -1, *) or None.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.

        Returns:
            JaggedTensor: The destination sidecar data after injection.
        """
        if dst is None:
            dst_shape = [dst_grid.total_voxels]
            dst_shape.extend(src.eshape)
            dst = dst_grid.jagged_like(
                torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
            )

        if dst.eshape != src.eshape:
            raise ValueError(
                f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}"
            )
        self._impl.inject_to(dst_grid._impl, src, dst)
        return dst

    def integrate_tsdf(
        self,
        truncation_distance: float,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        tsdf: JaggedTensor,
        weights: JaggedTensor,
        depth_images: torch.Tensor,
        weight_images: torch.Tensor | None = None,
    ) -> tuple["GridBatch", JaggedTensor, JaggedTensor]:
        """
        Integrate depth images into a Truncated Signed Distance Function (TSDF) volume.

        Updates the TSDF values and weights in the voxel grid by integrating new depth
        observations from multiple camera viewpoints. This is commonly used for 3D
        reconstruction from RGB-D sensors.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values (in world units).
            projection_matrices (torch.Tensor): Camera projection matrices.
                Shape: (batch_size, 4, 4).
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: (batch_size, 4, 4).
            tsdf (JaggedTensor): Current TSDF values for each voxel.
                Shape: (batch_size,total_voxels, 1).
            weights (JaggedTensor): Current integration weights for each voxel.
                Shape: (batch_size, total_voxels, 1).
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: (batch_size, height, width).
            weight_images (torch.Tensor, optional): Weight of each depth sample in the images.
                Shape: (batch_size, height, width). If None, defaults to uniform weights.

        Returns:
            tuple[GridBatch, JaggedTensor, JaggedTensor]: A tuple containing:
                - Updated GridBatch with potentially expanded voxels
                - Updated TSDF values as JaggedTensor
                - Updated weights as JaggedTensor
        """

        result_grid_impl, result_jagged_1, result_jagged_2 = self._impl.integrate_tsdf(
            truncation_distance,
            projection_matrices,
            cam_to_world_matrices,
            tsdf,
            weights,
            depth_images,
            weight_images,
        )

        return (
            GridBatch(impl=result_grid_impl),
            result_jagged_1,
            result_jagged_2,
        )

    def integrate_tsdf_with_features(
        self,
        truncation_distance: float,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        tsdf: JaggedTensor,
        features: JaggedTensor,
        weights: JaggedTensor,
        depth_images: torch.Tensor,
        feature_images: torch.Tensor,
        weight_images: torch.Tensor | None = None,
    ) -> tuple["GridBatch", JaggedTensor, JaggedTensor, JaggedTensor]:
        """
        Integrate depth and feature images into TSDF volume with features.

        Similar to integrate_tsdf but also integrates feature observations (e.g., color)
        along with the depth information. This is useful for colored 3D reconstruction.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values (in world units).
            projection_matrices (torch.Tensor): Camera projection matrices.
                Shape: (batch_size, 4, 4).
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: (batch_size, 4, 4).
            tsdf (JaggedTensor): Current TSDF values for each voxel.
                Shape: (batch_size, total_voxels, 1).
            features (JaggedTensor): Current feature values for each voxel.
                Shape: (batch_size, total_voxels, feature_dim).
            weights (JaggedTensor): Current integration weights for each voxel.
                Shape: (batch_size, total_voxels, 1).
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: (batch_size, height, width).
            feature_images (torch.Tensor): Feature images (e.g., RGB) from cameras.
                Shape: (batch_size, height, width, feature_dim).
            weight_images (torch.Tensor, optional): Weight of each depth sample in the images.
                Shape: (batch_size, height, width). If None, defaults to uniform weights.

        Returns:
            tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]: A tuple containing:
                - Updated GridBatch with potentially expanded voxels
                - Updated TSDF values as JaggedTensor
                - Updated weights as JaggedTensor
                - Updated features as JaggedTensor
        """
        result_grid_impl, result_jagged_1, result_jagged_2, result_jagged_3 = self._impl.integrate_tsdf_with_features(
            truncation_distance,
            projection_matrices,
            cam_to_world_matrices,
            tsdf,
            features,
            weights,
            depth_images,
            feature_images,
            weight_images,
        )

        return (
            GridBatch(impl=result_grid_impl),
            result_jagged_1,
            result_jagged_2,
            result_jagged_3,
        )

    def is_contiguous(self) -> bool:
        """
        Check if the grid batch data is stored contiguously in memory.

        Returns:
            bool: True if the data is contiguous, False otherwise.
        """
        return self._impl.is_contiguous()

    def is_same(self, other: "GridBatch") -> bool:
        """
        Check if two grid batches have the same structure.

        Compares the voxel structure, dimensions, and origins of two grid batches.

        Args:
            other (GridBatch): The other grid batch to compare with.

        Returns:
            bool: True if the grids have identical structure, False otherwise.
        """
        return self._impl.is_same(other._impl)

    def jagged_like(self, data: torch.Tensor) -> JaggedTensor:
        """
        Create a JaggedTensor with the same jagged structure as this grid batch.

        Useful for creating feature tensors that match the grid's voxel layout.

        Args:
            data (torch.Tensor): Dense data to convert to jagged format.
                Shape: (total_voxels, channels).

        Returns:
            JaggedTensor: Data in jagged format matching the grid structure.
        """
        return self._impl.jagged_like(data)

    def marching_cubes(
        self, field: JaggedTensor, level: float = 0.0
    ) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
        """
        Extract isosurface mesh using the marching cubes algorithm.

        Generates a triangle mesh representing the isosurface at the specified level
        from a scalar field defined on the voxels.

        Args:
            field (JaggedTensor): Scalar field values at each voxel.
                Shape: (batch_size, total_voxels, 1).
            level (float): The isovalue to extract the surface at. Default is 0.0.

        Returns:
            tuple[JaggedTensor, JaggedTensor, JaggedTensor]: A tuple containing:
                - Vertex positions of the mesh. Shape: (batch_size, num_vertices, 3).
                - Triangle face indices. Shape: (batch_size, num_faces, 3).
                - Vertex normals (computed from gradients). Shape: (batch_size, num_vertices, 3).
        """
        return self._impl.marching_cubes(field, level)

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: JaggedTensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Downsample grid data using max pooling.

        Performs max pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the maximum of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid.
                broadcastable to shape (3,), integer dtype
            data (JaggedTensor): The voxel data to pool. Shape should be
                (batch_size, total_voxels, channels).
            stride (NumericMaxRank1): The stride to use when pooling. If 0 (default),
                stride equals pool_factor. If an int, the same stride is used for all dimensions.
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The pooled data as a JaggedTensor
                - The coarse GridBatch containing the pooled structure
        """
        pool_factor = to_PositiveVec3i(pool_factor)
        stride = to_NonNegativeVec3i(stride)

        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.max_pool(pool_factor, data, stride, coarse_grid_impl)

        return result_data, GridBatch(impl=result_grid_impl)

    def merged_grid(self, other: "GridBatch") -> "GridBatch":
        """
        Return a grid batch that is the union of this grid batch with another.

        Merges two grid batches by taking the union of their active voxels.
        The grids must have compatible dimensions and transforms.

        Args:
            other (GridBatch): The other grid batch to merge with.

        Returns:
            GridBatch: A new GridBatch containing the union of active voxels from both grids.
        """
        return GridBatch(impl=self._impl.merged_grid(other._impl))

    def neighbor_indexes(self, ijk: JaggedTensor, extent: int, bitshift: int = 0) -> JaggedTensor:
        """
        Get indices of neighbors in N-ring neighborhood.

        Finds the linear indices of all voxels within a specified neighborhood ring
        around the given voxel coordinates.

        Args:
            ijk (JaggedTensor): Voxel coordinates to find neighbors for.
                Shape: (batch_size, num_queries, 3) with integer coordinates.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): Bit shift value for encoding. Default is 0.

        Returns:
            JaggedTensor: Linear indices of neighboring voxels.
        """
        return self._impl.neighbor_indexes(ijk, extent, bitshift)

    def num_voxels_at(self, bi: int) -> int:
        """
        Get the number of active voxels in a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            int: Number of active voxels in the specified grid.
        """
        return self._impl.num_voxels_at(bi)

    def pruned_grid(self, mask: JaggedTensor) -> "GridBatch":
        """
        Return a pruned grid based on a boolean mask.

        Creates a new grid containing only the voxels where the mask is True.

        Args:
            mask (JaggedTensor): Boolean mask for each voxel.
                Shape: (batch_size, total_voxels,).

        Returns:
            GridBatch: A new GridBatch containing only voxels where mask is True.
        """
        return GridBatch(impl=self._impl.pruned_grid(mask))

    def origin_at(self, bi: int) -> torch.Tensor:
        """
        Get the world-space origin of a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: The origin coordinates in world space. Shape: (3,).
        """
        return self._impl.origin_at(bi)

    def points_in_active_voxel(self, points: JaggedTensor) -> JaggedTensor:
        """
        Check if world-space points are located within active voxels.

        Tests whether the given points fall within voxels that are active in the grid.

        Args:
            points (JaggedTensor): World-space points to test.
                Shape: (batch_size, num_points, 3).

        Returns:
            JaggedTensor: Boolean mask indicating which points are in active voxels.
                Shape: (batch_size, num_points,).
        """
        return self._impl.points_in_active_voxel(points)

    def ray_implicit_intersection(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        grid_scalars: JaggedTensor,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """
        Find ray intersections with implicit surface defined by grid scalars.

        Computes intersection points between rays and an implicit surface defined by
        scalar values stored in the grid voxels (e.g., signed distance function).

        Args:
            ray_origins (JaggedTensor): Starting points of rays in world space.
                Shape: (batch_size,num_rays, 3).
            ray_directions (JaggedTensor): Direction vectors of rays.
                Shape: (batch_size, num_rays, 3). Should be normalized.
            grid_scalars (JaggedTensor): Scalar field values at each voxel.
                Shape: (batch_size, total_voxels, 1).
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            JaggedTensor: Intersection information for each ray.
        """
        return self._impl.ray_implicit_intersection(ray_origins, ray_directions, grid_scalars, eps)

    def read_from_dense(self, dense_data: torch.Tensor, dense_origins: NumericMaxRank1 = 0) -> JaggedTensor:
        """
        Read values from a dense tensor into sparse grid structure.

        Extracts values from a dense tensor at locations corresponding to active voxels
        in the sparse grid. Useful for converting dense data to sparse representation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from.
                Shape: (batch_size, channels, depth, height, width) or
                       (batch_size, depth, height, width, channels).
            dense_origins (NumericMaxRank1): Origin of the dense tensor in grid index space.
                broadcastable to shape (3,), integer dtype. Default is (0, 0, 0).

        Returns:
            JaggedTensor: Values from the dense tensor at active voxel locations.
                Shape: (batch_size, total_voxels, channels).
        """
        dense_origins = to_Vec3i(dense_origins)

        return self._impl.read_from_dense(dense_data, dense_origins)

    def sample_bezier(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """
        Sample voxel features at arbitrary points using Bézier interpolation.

        Interpolates voxel data at continuous world-space positions using cubic Bézier
        interpolation. This provides smoother interpolation than trilinear but is more
        computationally expensive.

        Args:
            points (JaggedTensor): World-space points to sample at.
                Shape: (batch_size, num_points, 3).
            voxel_data (JaggedTensor): Features stored at each voxel.
                Shape: (batch_size, total_voxels, channels).

        Returns:
            JaggedTensor: Interpolated features at each point.
                Shape: (batch_size, num_points, channels).
        """
        return self._impl.sample_bezier(points, voxel_data)

    def sample_bezier_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample voxel features and their gradients using Bézier interpolation.

        Similar to sample_bezier but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (JaggedTensor): World-space points to sample at.
                Shape: (batch_size, num_points, 3).
            voxel_data (JaggedTensor): Features stored at each voxel.
                Shape: (batch_size, total_voxels, channels).

        Returns:
            tuple[JaggedTensor, JaggedTensor]: A tuple containing:
                - Interpolated features at each point. Shape: (batch_size, num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (batch_size, num_points, 3, channels).
        """
        return self._impl.sample_bezier_with_grad(points, voxel_data)

    def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """
        Sample voxel features at arbitrary points using trilinear interpolation.

        Interpolates voxel data at continuous world-space positions using trilinear
        interpolation from the 8 nearest voxels. Points outside the grid return zero.

        Args:
            points (JaggedTensor): World-space points to sample at.
                Shape: (batch_size, num_points, 3).
            voxel_data (JaggedTensor): Features stored at each voxel.
                Shape: (batch_size, total_voxels, channels).

        Returns:
            JaggedTensor: Interpolated features at each point.
                Shape: (batch_size, num_points, channels).
        """
        return self._impl.sample_trilinear(points, voxel_data)

    def sample_trilinear_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample voxel features and their gradients using trilinear interpolation.

        Similar to sample_trilinear but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (JaggedTensor): World-space points to sample at.
                Shape: (batch_size, num_points, 3).
            voxel_data (JaggedTensor): Features stored at each voxel.
                Shape: (batch_size, total_voxels, channels).

        Returns:
            tuple[JaggedTensor, JaggedTensor]: A tuple containing:
                - Interpolated features at each point. Shape: (batch_size, num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (batch_size, num_points, 3, channels).
        """
        return self._impl.sample_trilinear_with_grad(points, voxel_data)

    def segments_along_rays(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        max_segments: int,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """
        Enumerate segments along rays.

        Args:
            ray_origins (JaggedTensor): Origin of each ray.
                Shape: (batch_size, num_rays, 3).
            ray_directions (JaggedTensor): Direction of each ray.
                Shape: (batch_size, num_rays, 3).
            max_segments (int): Maximum number of segments to enumerate.
            eps (float): Small epsilon value to avoid numerical issues.

        Returns:
            A JaggedTensor containing the samples along the rays. i.e. a JaggedTensor
            with lshape [[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]] and eshape
            (2,) or (1,) representing the start and end distance of each sample or the midpoint
            of each sample if return_midpoints is true
        """
        return self._impl.segments_along_rays(ray_origins, ray_directions, max_segments, eps)

    def sparse_conv_halo(self, input: JaggedTensor, weight: torch.Tensor, variant: int = 8) -> JaggedTensor:
        """
        Perform sparse convolution with halo exchange optimization.

        Applies sparse convolution using halo exchange to efficiently handle boundary
        conditions in distributed or multi-block sparse grids.

        Args:
            input (JaggedTensor): Input features for each voxel.
                Shape: (batch_size, total_voxels, in_channels).
            weight (torch.Tensor): Convolution weights.
            variant (int): Variant of the halo implementation to use.
                Default is 8.

        Returns:
            JaggedTensor: Output features after convolution.
        """
        return self._impl.sparse_conv_halo(input, weight, variant)

    def sparse_conv_kernel_map(
        self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1, target_grid: "GridBatch | None" = None
    ) -> tuple["SparseConvPackInfo", "GridBatch"]:
        """
        Map sparse convolution kernel to target grid.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel.
                broadcastable to shape (3,), integer dtype
            stride (NumericMaxRank1): Stride of the convolution.
                broadcastable to shape (3,), integer dtype
            target_grid (GridBatch | None): Target grid to map the kernel to.
                If None, the kernel is mapped to the current grid.

        Returns:
            tuple[SparseConvPackInfo, GridBatch]: A tuple containing:
                - SparseConvPackInfo: Information about the sparse convolution kernel.
                - GridBatch: The target grid.
        """
        # Import here to avoid circular dependency
        from .sparse_conv_pack_info import SparseConvPackInfo

        kernel_size = to_PositiveVec3i(kernel_size)
        stride = to_NonNegativeVec3i(stride)

        target_impl = target_grid._impl if target_grid is not None else None

        sparse_impl, grid_impl = self._impl.sparse_conv_kernel_map(kernel_size, stride, target_impl)
        return (
            SparseConvPackInfo(impl=sparse_impl),
            GridBatch(impl=grid_impl),
        )

    def splat_bezier(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """
        Splat point features onto voxels using Bézier interpolation.

        Distributes features from point locations onto the surrounding voxels using
        cubic Bézier interpolation weights. This provides smoother distribution than
        trilinear splatting but is more computationally expensive.

        Args:
            points (JaggedTensor): World-space positions of points.
                Shape: (batch_size, num_points, 3).
            points_data (JaggedTensor): Features to splat from each point.
                Shape: (batch_size, num_points, channels).

        Returns:
            JaggedTensor: Accumulated features at each voxel after splatting.
                Shape: (batch_size, total_voxels, channels).
        """
        return self._impl.splat_bezier(points, points_data)

    def splat_trilinear(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """
        Splat point features onto voxels using trilinear interpolation.

        Distributes features from point locations onto the surrounding 8 voxels using
        trilinear interpolation weights. This is the standard method for converting
        point-based data to voxel grids.

        Args:
            points (JaggedTensor): World-space positions of points.
                Shape: (batch_size, num_points, 3).
            points_data (JaggedTensor): Features to splat from each point.
                Shape: (batch_size, num_points, channels).

        Returns:
            JaggedTensor: Accumulated features at each voxel after splatting.
                Shape: (batch_size, total_voxels, channels).
        """
        return self._impl.splat_trilinear(points, points_data)

    def subdivide(
        self,
        subdiv_factor: NumericMaxRank1,
        data: JaggedTensor,
        mask: JaggedTensor | None = None,
        fine_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Subdivide grid using nearest neighbor interpolation.

        Increases the resolution of the grid by the specified subdivision factor,
        filling in new voxels using nearest neighbor interpolation of the existing data.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to subdivide the grid.
                broadcastable to shape (3,), integer dtype
            data (JaggedTensor): Voxel data to subdivide.
                Shape: (batch_size, total_voxels, channels).
            mask (JaggedTensor | None): Boolean mask indicating which voxels to subdivide.
                If None, all voxels are subdivided.
            fine_grid (GridBatch | None): Pre-allocated fine grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The subdivided data as a JaggedTensor
                - The fine GridBatch containing the subdivided structure
        """
        subdiv_factor = to_PositiveVec3i(subdiv_factor)
        fine_grid_impl = fine_grid._impl if fine_grid else None
        result_data, result_grid_impl = self._impl.subdivide(subdiv_factor, data, mask, fine_grid_impl)
        return result_data, GridBatch(impl=result_grid_impl)

    def subdivided_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: JaggedTensor | None = None,
    ) -> "GridBatch":
        """
        Return a subdivided version of the grid structure.

        Creates a new grid with higher resolution by subdividing existing voxels.
        Only the grid structure is returned, not the data.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to subdivide the grid.
                broadcastable to shape (3,), integer dtype
            mask (JaggedTensor | None): Boolean mask indicating which voxels to subdivide.
                If None, all voxels are subdivided.

        Returns:
            GridBatch: A new GridBatch with subdivided structure.
        """
        subdiv_factor = to_PositiveVec3i(subdiv_factor)
        return GridBatch(impl=self._impl.subdivided_grid(subdiv_factor, mask=mask))

    def to(self, target: "str | torch.device | torch.Tensor | JaggedTensor | GridBatch") -> "GridBatch":
        """
        Move grid batch to a target device or match device of target object.

        Args:
            target: Target to determine device. Can be:
                - str: Device string (e.g., "cuda", "cpu")
                - torch.device: PyTorch device object
                - torch.Tensor: Match device of this tensor
                - JaggedTensor: Match device of this JaggedTensor
                - GridBatch: Match device of this GridBatch

        Returns:
            GridBatch: A new GridBatch on the target device.
        """
        if isinstance(target, str):
            device = _parse_device_string(target)
            return GridBatch(impl=self._impl.to(device))
        elif isinstance(target, torch.device):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, torch.Tensor):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, JaggedTensor):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, GridBatch):
            return GridBatch(impl=self._impl.to(target._impl))
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")

    def uniform_ray_samples(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        t_min: JaggedTensor,
        t_max: JaggedTensor,
        step_size: float,
        cone_angle: float = 0.0,
        include_end_segments: bool = True,
        return_midpoints: bool = False,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """
        Generate uniform samples along rays within the grid.

        Creates sample points at regular intervals along rays, but only for segments
        that intersect with active voxels. Useful for volume rendering and ray marching.

        Args:
            ray_origins (JaggedTensor): Starting points of rays in world space.
                Shape: (batch_size, num_rays, 3).
            ray_directions (JaggedTensor): Direction vectors of rays (should be normalized).
                Shape: (batch_size, num_rays, 3).
            t_min (JaggedTensor): Minimum distance along rays to start sampling.
                Shape: (batch_size, num_rays).
            t_max (JaggedTensor): Maximum distance along rays to stop sampling.
                Shape: (batch_size, num_rays).
            step_size (float): Distance between samples along each ray.
            cone_angle (float): Cone angle for cone tracing (in radians). Default is 0.0.
            include_end_segments (bool): Whether to include partial segments at ray ends.
                Default is True.
            return_midpoints (bool): Whether to return segment midpoints instead of start points.
                Default is False.
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            JaggedTensor: containing the samples along the rays. i.e. a JaggedTensor
              with lshape [[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]] and eshape
              (2,) or (1,) representing the start and end distance of each sample or the midpoint
              of each sample if return_midpoints is true.
        """
        return self._impl.uniform_ray_samples(
            ray_origins,
            ray_directions,
            t_min,
            t_max,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )

    def voxel_size_at(self, bi: int) -> torch.Tensor:
        """
        Get voxel size at a specific grid index.

        Args:
            bi (int): Grid index.

        Returns:
            torch.Tensor: Voxel size at the specified grid index.
                Shape: (3,).
        """
        return self._impl.voxel_size_at(bi)

    def voxels_along_rays(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = True,
        cumulative: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Enumerate voxels intersected by rays.

        Finds all active voxels that are intersected by the given rays using a
        DDA (Digital Differential Analyzer) algorithm.

        Args:
            ray_origins (JaggedTensor): Starting points of rays in world space.
                Shape: (batch_size, num_rays, 3).
            ray_directions (JaggedTensor): Direction vectors of rays (should be normalized).
                Shape: (batch_size, num_rays, 3).
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Epsilon value for numerical stability. Default is 0.0.
            return_ijk (bool): Whether to return voxel indices. If False, returns
                linear indices instead. Default is True.
            cumulative (bool): Whether to return cumulative indices across the batch.
                Default is False.

        Returns:
            A pair of JaggedTensors containing the voxels (or voxel indices) intersected by the rays. i.e.:
                - voxels: A JaggedTensor with lshape [[V_{0,0}, ..., V_{0,N_0}], ..., [V_{B,0}, ..., V_{B,N_B}]]
                          and eshape (3,) or (,) containing the ijk coordinates or indices of the voxels
                - times: A JaggedTensor with lshape [[T_{0,0}, ..., T_{0,N_0}], ..., [T_{B,0}, ..., T_{B,N_B}]]
                          and eshape (2,) containinng the entry and exit distance along the ray of each voxel
        """
        return self._impl.voxels_along_rays(ray_origins, ray_directions, max_voxels, eps, return_ijk, cumulative)

    def world_to_grid(self, points: JaggedTensor) -> JaggedTensor:
        """
        Convert world coordinates to grid (index) coordinates.

        Transforms positions in world space to their corresponding voxel indices
        using the grid's origin and voxel size. The resulting coordinates can be
        fractional for use in interpolation.

        Args:
            points (JaggedTensor): World-space positions to convert.
                Shape: (batch_size, num_points, 3).

        Returns:
            JaggedTensor: Grid coordinates. Shape: (batch_size, num_points, 3).
                Can contain fractional values.
        """
        return self._impl.world_to_grid(points)

    def write_to_dense(
        self,
        sparse_data: JaggedTensor,
        min_coord: NumericMaxRank2 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Write sparse voxel data to a dense tensor.

        Creates a dense tensor and fills it with values from the sparse grid.
        Voxels not present in the sparse grid are filled with zeros.

        Args:
            sparse_data (JaggedTensor): Sparse voxel features to write.
                Shape: (batch_size, total_voxels, channels).
            min_coord (NumericMaxRank2 | None): Minimum coordinates for each grid in the batch.
                (broadcastable to shape (batch_size, 3), integer dtype)
                If None, computed from the grid bounds.
            grid_size (NumericMaxRank1 | None): Size of the output dense tensor.
                (broadcastable to shape (3,), integer dtype)
                If None, computed to fit all active voxels.

        Returns:
            torch.Tensor: Dense tensor containing the sparse data.
                Shape: (batch_size, channels, depth, height, width).
        """
        min_coord = to_Vec3iBatch(min_coord) if min_coord is not None else None
        grid_size = to_Vec3i(grid_size) if grid_size is not None else None

        return self._impl.write_to_dense(sparse_data, min_coord, grid_size)

    # ============================================================
    #                Indexing and Special Functions
    # ============================================================

    # Index methods
    def index_int(self, bi: int | np.integer) -> "GridBatch":
        """
        Get a subset of grids from the batch using integer indexing.

        Args:
            bi (int | np.integer): Grid index.

        Returns:
            GridBatch: A new GridBatch containing the selected grid.
        """
        return GridBatch(impl=self._impl.index_int(int(bi)))

    def index_list(self, indices: list[bool] | list[int]) -> "GridBatch":
        """
        Get a subset of grids from the batch using list indexing.

        Args:
            indices (list[bool] | list[int]): List of indices.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_list(indices))

    def index_slice(self, s: slice) -> "GridBatch":
        """
        Get a subset of grids from the batch using slicing.

        Args:
            s (slice): Slicing object.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_slice(s))

    def index_tensor(self, indices: torch.Tensor) -> "GridBatch":
        """
        Get a subset of grids from the batch using tensor indexing.

        Args:
            indices (torch.Tensor): Tensor of indices.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_tensor(indices))

    # Special methods
    def __getitem__(self, index: GridBatchIndex) -> "GridBatch":
        """
        Get a subset of grids from the batch using indexing.

        Supports integer indexing, slicing, list indexing, and boolean/integer tensor indexing.

        Args:
            index: Index to select grids. Can be:
                - int: Select a single grid
                - slice: Select a range of grids
                - list[int] or list[bool]: Select specific grids
                - torch.Tensor: Boolean or integer tensor for advanced indexing

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        if isinstance(index, (int, np.integer)):
            return self.index_int(int(index))
        elif isinstance(index, slice):
            return self.index_slice(index)
        elif isinstance(index, list):
            return self.index_list(index)
        elif isinstance(index, torch.Tensor):
            return self.index_tensor(index)
        else:
            raise TypeError(f"index must be a GridBatchIndex, but got {type(index)}")

    def __iter__(self) -> Iterator["GridBatch"]:
        """
        Iterate over individual grids in the batch.

        Yields:
            GridBatch: Single-grid batches for each grid in the batch.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """
        Get the number of grids in the batch.

        Returns:
            int: Number of grids in this batch.
        """
        return self._impl.grid_count

    # ============================================================
    #                        Properties
    # ============================================================

    # Properties
    @property
    def address(self) -> int:
        return self._impl.address

    @property
    def all_have_zero_voxels(self) -> bool:
        return self.has_zero_grids or self.total_voxels == 0

    @property
    def any_have_zero_voxels(self) -> bool:
        if self.has_zero_grids:
            return True
        else:
            return bool(torch.any(self.num_voxels == 0).item())

    @property
    def bboxes(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 2, 3), dtype=torch.int32, device=self.device)
        else:
            if self.all_have_zero_voxels:
                return torch.zeros((self.grid_count, 2, 3), dtype=torch.int32, device=self.device)
            elif self.any_have_zero_voxels:
                bboxes = self._impl.bbox

                fixed_bboxes = []
                for i in range(self.grid_count):
                    if self.num_voxels[i] == 0:
                        fixed_bboxes.append(torch.zeros((2, 3), dtype=torch.int32, device=self.device))
                    else:
                        fixed_bboxes.append(bboxes[i])

                return torch.stack(fixed_bboxes, dim=0)
            else:
                return self._impl.bbox

    @property
    def cum_voxels(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.cum_voxels

    @property
    def device(self) -> torch.device:
        return self._impl.device

    @property
    def dual_bboxes(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 2, 3), dtype=torch.int32, device=self.device)
        else:
            if self.all_have_zero_voxels:
                return torch.zeros((self.grid_count, 2, 3), dtype=torch.int32, device=self.device)
            elif self.any_have_zero_voxels:
                bboxes = self._impl.dual_bbox

                fixed_bboxes = []
                for i in range(self.grid_count):
                    if self.num_voxels[i] == 0:
                        fixed_bboxes.append(torch.zeros((2, 3), dtype=torch.int32, device=self.device))
                    else:
                        fixed_bboxes.append(bboxes[i])

                return torch.stack(fixed_bboxes, dim=0)
            else:
                return self._impl.dual_bbox

    @property
    def grid_count(self) -> int:
        return self._impl.grid_count

    @property
    def grid_to_world_matrices(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self._impl.grid_to_world_matrices

    @property
    def has_zero_grids(self) -> bool:
        return self.grid_count == 0

    @property
    def ijk(self) -> JaggedTensor:
        return self._impl.ijk

    @property
    def jidx(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        else:
            return self._impl.jidx

    @property
    def joffsets(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.joffsets

    @property
    def num_bytes(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_bytes

    @property
    def num_leaf_nodes(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_leaf_nodes

    @property
    def num_voxels(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_voxels

    @property
    def origins(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self._impl.origins

    @property
    def total_bbox(self) -> torch.Tensor:
        if self.has_zero_grids or self.all_have_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.total_bbox

    @property
    def total_bytes(self) -> int:
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_bytes

    @property
    def total_leaf_nodes(self) -> int:
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_leaf_nodes

    @property
    def total_voxels(self) -> int:
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_voxels

    @property
    def viz_edge_network(self) -> tuple[JaggedTensor, JaggedTensor]:
        return self._impl.viz_edge_network

    @property
    def voxel_sizes(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self._impl.voxel_sizes

    @property
    def world_to_grid_matrices(self) -> torch.Tensor:
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self._impl.world_to_grid_matrices

    # Expose underlying implementation for compatibility
    @property
    def _gridbatch(self):
        # Access underlying GridBatchCpp - use sparingly during migration
        return self._impl


# Load and save functions
@overload
def load_gridbatch(
    path: str,
    *,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_gridbatch(
    path: str,
    *,
    indices: list[int],
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_gridbatch(
    path: str,
    *,
    index: int,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_gridbatch(
    path: str,
    *,
    names: list[str],
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_gridbatch(
    path: str,
    *,
    name: str,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


def load_gridbatch(
    path: str,
    *,
    indices: list[int] | None = None,
    index: int | None = None,
    names: list[str] | None = None,
    name: str | None = None,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]:
    """Load a grid batch from a .nvdb file.

    Args:
        path: The path to the .nvdb file to load
        indices: Optional list of indices to load from the file (mutually exclusive with other selectors)
        index: Optional single index to load from the file (mutually exclusive with other selectors)
        names: Optional list of names to load from the file (mutually exclusive with other selectors)
        name: Optional single name to load from the file (mutually exclusive with other selectors)
        device: Which device to load the grid batch on
        verbose: If set to true, print information about the loaded grids

    Returns:
        A tuple (gridbatch, data, names) where gridbatch is a GridBatch containing the loaded
        grids, data is a JaggedTensor containing the data of the grids, and names is a list of
        strings containing the name of each grid
    """
    from ._Cpp import load as _load

    device = resolve_device(device)

    # Check that only one selector is provided
    selectors = [indices is not None, index is not None, names is not None, name is not None]
    if sum(selectors) > 1:
        raise ValueError("Only one of indices, index, names, or name can be specified")

    # Call the appropriate overload
    if indices is not None:
        grid_impl, data, names_out = _load(path, indices, device, verbose)
    elif index is not None:
        grid_impl, data, names_out = _load(path, index, device, verbose)
    elif names is not None:
        grid_impl, data, names_out = _load(path, names, device, verbose)
    elif name is not None:
        grid_impl, data, names_out = _load(path, name, device, verbose)
    else:
        # Load all grids
        grid_impl, data, names_out = _load(path, device, verbose)

    # Wrap the GridBatch implementation with the Python wrapper
    return GridBatch(impl=grid_impl), data, names_out


def save_gridbatch(
    path: str,
    grid_batch: GridBatch,
    data: JaggedTensor | None = None,
    names: list[str] | str | None = None,
    name: str | None = None,
    compressed: bool = False,
    verbose: bool = False,
) -> None:
    """
    Save a grid batch and optional voxel data to a .nvdb file.

    Saves sparse grids in the NanoVDB format, which can be loaded by other
    applications that support OpenVDB/NanoVDB.

    Args:
        path (str): The file path to save to. Should have .nvdb extension.
        grid_batch (GridBatch): The grid batch to save.
        data (JaggedTensor | None): Voxel data to save with the grids.
            Shape: (batch_size, total_voxels, channels). If None, only grid structure is saved.
        names (list[str] | str | None): Names for each grid in the batch.
            If a single string, it's used as the name for all grids.
        name (str | None): Alternative way to specify a single name for all grids.
            Takes precedence over names parameter.
        compressed (bool): Whether to compress the data using Blosc compression.
            Default is False.
        verbose (bool): Whether to print information about the saved grids.
            Default is False.

    Note:
        The parameters 'names' and 'name' are mutually exclusive ways to specify
        grid names. Use 'name' for a single name applied to all grids, or 'names'
        for individual names per grid.
    """
    from ._Cpp import save as _save

    # Handle the overloaded signature - if name is provided, use it
    if name is not None:
        _save(path, grid_batch._impl, data, name, compressed, verbose)
    elif names is not None:
        if isinstance(names, str):
            # Handle case where names is actually a single name
            _save(path, grid_batch._impl, data, names, compressed, verbose)
        else:
            # Handle case where names is a list
            _save(path, grid_batch._impl, data, names, compressed, verbose)
    else:
        # Default case with empty names list
        _save(path, grid_batch._impl, data, [], compressed, verbose)
