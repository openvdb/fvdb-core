# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Single sparse grid data structure and operations for FVDB.

This module provides the core Grid class for managing a single sparse voxel grid:

Classes:
- Grid: A single sparse voxel grid with support for efficient operations

Class-methods for creating Grid objects from various sources:
- Grid.from_dense() for dense data
- Grid.from_dense_axis_aligned_bounds() for dense defined by axis-aligned bounds
- Grid.from_grid_batch() for a single grid from a grid batch
- Grid.from_ijk() for voxel coordinates
- Grid.from_mesh() for triangle meshes
- Grid.from_nearest_voxels_to_points() for nearest voxel mapping
- Grid.from_points() for point clouds
- Grid.from_zero_voxels() for a single grid with zero voxels

Module-level functions for loading and saving grids:
- load_grid/save_grid: Load and save grids to/from .nvdb files

Grid supports operations like convolution, pooling, interpolation, ray casting,
mesh extraction, and coordinate transformations on sparse voxel data.
"""

from collections.abc import Iterator
from types import ClassMethodDescriptorType
from typing import TYPE_CHECKING, Any, Sequence, cast, overload

import numpy as np
import torch

from . import _parse_device_string
from ._Cpp import ConvPackBackend
from ._Cpp import GridBatch as GridBatchCpp
from ._Cpp import JaggedTensor
from .types import (
    DeviceIdentifier,
    NumericMaxRank1,
    ValueConstraint,
    resolve_device,
    to_Vec3f,
    to_Vec3fBatch,
    to_Vec3fBatchBroadcastable,
    to_Vec3fBroadcastable,
    to_Vec3i,
    to_Vec3iBroadcastable,
)

if TYPE_CHECKING:
    from .grid_batch import GridBatch
    from .sparse_conv_pack_info import SparseConvPackInfo


class Grid:
    """
    A single sparse voxel grid with support for efficient operations.

    Grid represents a single sparse 3D voxel grid that can be processed
    efficiently on a GPU. The class provides methods for common operations like
    sampling, convolution, pooling, and other operations.

    A Grid cannot be a nonexistent (grid_count==0) grid, for that you'd need a
    GridBatch with batch_size=0. However, a Grid can have zero voxels.

    The grid is stored in a sparse format where only active (non-empty) voxels are
    allocated, making it memory efficient for representing large volumes with sparse
    occupancy.

    Note:
        For creating grids with actual content, use the classmethods:
        - Grid.from_dense() for dense data
        - Grid.from_dense_axis_aligned_bounds() for dense defined by axis-aligned bounds
        - Grid.from_grid_batch() for a single grid from a grid batch
        - Grid.from_ijk() for voxel coordinates
        - Grid.from_mesh() for triangle meshes
        - Grid.from_nearest_voxels_to_points() for nearest voxel mapping
        - Grid.from_points() for point clouds
        - Grid.from_zero_voxels() for a single grid with zero voxels

        The Grid constructor is for internal use only, always use the classmethods.
    """

    def __init__(self, *, impl: GridBatchCpp):
        """
        Constructor for internal use only. - use the Grid.from_* classmethods instead.
        """
        self._impl = impl

    # ============================================================
    #                  Grid from_* constructors
    # ============================================================

    @classmethod
    def from_dense(
        cls,
        dense_dims: NumericMaxRank1,
        ijk_min: NumericMaxRank1 = 0,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        mask: torch.Tensor | None = None,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        A dense grid has a voxel for every coordinate in an axis-aligned box of Vec3,
        which can in turn be mapped to a world-space box.

        The dense grid is defined by:
        - dense_dims: the size of the dense grid (shape [3,] = [W, H, D])
        - ijk_min: the minimum voxel index for the grid (Vec3i)
        - voxel_size: the world-space size of each voxel (Vec3d or scalar)
        - origin: the world-space coordinate of the 0,0,0 voxel of the grid
        - mask: indicates which voxels are "active" in the resulting grid.

        Args:
            dense_dims (NumericMaxRank1): Dimensions of the dense grid,
                broadcastable to shape (3,), integer dtype
            ijk_min (NumericMaxRank1): Minimum voxel index for the grid,
                broadcastable to shape (3,), integer dtype
            voxel_size (NumericMaxRank1): World space size of each voxel,
                broadcastable to shape (3,), floating dtype
            origin (NumericMaxRank1): World space coordinate of the 0,0,0 voxel of the grid,
                broadcastable to shape (3,), floating dtype
            mask (torch.Tensor | None): Mask to apply to the grid,
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from mask, or uses "cpu" if mask is None.

        Returns:
            Grid: A new Grid object.
        """
        resolved_device = resolve_device(device, inherit_from=mask)

        dense_dims = to_Vec3i(dense_dims, value_constraint=ValueConstraint.POSITIVE)
        ijk_min = to_Vec3i(ijk_min)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_dense_grid(1, dense_dims, ijk_min, voxel_size, origin, mask)
        return cls(impl=grid_impl)

    @classmethod
    def from_dense_axis_aligned_bounds(
        cls,
        dense_dims: NumericMaxRank1,
        bounds_min: NumericMaxRank1 = 0,
        bounds_max: NumericMaxRank1 = 1,
        voxel_center: bool = False,
        device: DeviceIdentifier = "cpu",
    ) -> "Grid":
        dense_dims = to_Vec3iBroadcastable(dense_dims, value_constraint=ValueConstraint.POSITIVE)
        bounds_min = to_Vec3fBroadcastable(bounds_min)
        bounds_max = to_Vec3fBroadcastable(bounds_max)

        if torch.any(bounds_max <= bounds_min):
            raise ValueError("bounds_max must be greater than bounds_min in all axes")

        if voxel_center:
            voxel_size = (bounds_max - bounds_min) / (dense_dims.to(torch.float64) - 1.0)
            origin = to_Vec3f(bounds_min)
        else:
            voxel_size = (bounds_max - bounds_min) / dense_dims.to(torch.float64)
            origin = to_Vec3f(bounds_min + 0.5 * voxel_size)

        return cls.from_dense(dense_dims=dense_dims, voxel_size=voxel_size, origin=origin, device=device)

    @classmethod
    def from_grid_batch(cls, grid_batch: "GridBatch", index: int = 0) -> "Grid":
        """
        Create a single grid from one grid in a grid batch.
        If the given grid batch is empty, the returned grid will be empty
        with the same empty voxel size and origin.

        Args:
            grid_batch (GridBatch): The grid batch to create the grid from.
            index (int): The index of the grid to create. Defaults to 0.

        Returns:
            Grid: A new Grid object.
        """
        grid_impl = grid_batch.index_int(index)._impl
        assert grid_impl is not None
        assert grid_impl.grid_count == 1
        return cls(impl=grid_impl)

    @classmethod
    def from_ijk(
        cls,
        ijk: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        Create a grid from voxel coordinates.

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

        jagged_ijk = JaggedTensor(ijk)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_ijk(jagged_ijk, voxel_size, origin)
        return cls(impl=grid_impl)

    @classmethod
    def from_mesh(
        cls,
        mesh_vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        Create a grid from a triangle mesh.

        Args:
            mesh_vertices (torch.Tensor): Vertices of the mesh.
                Shape: (num_vertices, 3).
            mesh_faces (torch.Tensor): Faces of the mesh.
                Shape: (num_faces, 3).
            voxel_size (NumericMaxRank1): Size of each voxel,
                broadcastable to shape (3,), floating dtype
            origin (NumericMaxRank1): Origin of the grid,
                broadcastable to shape (3,), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from mesh_vertices.

        Returns:
            Grid: A new Grid object.
        """
        resolved_device = resolve_device(device, inherit_from=mesh_vertices)

        jagged_mesh_vertices = JaggedTensor(mesh_vertices)
        jagged_mesh_faces = JaggedTensor(mesh_faces)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_mesh(jagged_mesh_vertices, jagged_mesh_faces, voxel_size, origin)
        return cls(impl=grid_impl)

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        Create a grid from the nearest voxels to a set of points.

        Args:
            points (torch.Tensor): Points to populate the grid from.
                Shape: (num_points, 3).
            voxel_size (NumericMaxRank1): Size of each voxel,
                broadcastable to shape (3,), floating dtype
            origin (NumericMaxRank1): Origin of the grid,
                broadcastable to shape (3,), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from points.

        Returns:
            Grid: A new Grid object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        jagged_points = JaggedTensor(points)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_nearest_voxels_to_points(jagged_points, voxel_size, origin)
        return cls(impl=grid_impl)

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        Create a grid from a point cloud.

        Args:
            points (torch.Tensor): Points to populate the grid from.
                Shape: (num_points, 3).
            voxel_size (NumericMaxRank1): Size of each voxel,
                broadcastable to shape (3,), floating dtype
            origin (NumericMaxRank1): Origin of the grid,
                broadcastable to shape (3,), floating dtype
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to None, which inherits from points.

        Returns:
            Grid: A new Grid object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        jagged_points = JaggedTensor(points)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_points(jagged_points, voxel_size, origin)
        return cls(impl=grid_impl)

    @classmethod
    def from_zero_voxels(
        cls,
        device: DeviceIdentifier = "cpu",
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> "Grid":
        """
        Create a new Grid with zero voxels on a specific device.

        Args:
            device: The device to create the Grid on. Can be a string (e.g., "cuda", "cpu")
                or a torch.device object. Defaults to "cpu".
            voxel_size (NumericMaxRank1): Size of each voxel,
                broadcastable to shape (3,), floating dtype
                Defaults to 1.
            origin (NumericMaxRank1): Origin of the grid,
                broadcastable to shape (3,), floating dtype
                Defaults to 0.

        Returns:
            Grid: A new Grid object with zero voxels.

        Examples:
            >>> grid = Grid.from_zero_voxels("cuda", 1, 0)  # string
            >>> grid = Grid.from_zero_voxels(torch.device("cuda:0"), 1, 0)  # device directly
            >>> grid = Grid.from_zero_voxels(voxel_size=1, origin=0)  # defaults to CPU
        """
        resolved_device = resolve_device(device)
        voxel_size = to_Vec3fBatch(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3fBatch(origin)
        grid_impl = GridBatchCpp(voxel_sizes=voxel_size, grid_origins=origin, device=resolved_device)
        return cls(impl=grid_impl)

    # ============================================================
    #                Regular Instance Methods Begin
    # ============================================================

    def avg_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: torch.Tensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "Grid | None" = None,
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Downsample grid data using average pooling.

        Performs average pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the average of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid.
                broadcastable to shape (3,), integer dtype
            data (torch.Tensor): The voxel data to pool. Shape should be (total_voxels, channels).
            stride (NumericMaxRank1): The stride to use when pooling. If 0 (default),
                broadcastable to shape (3,), integer dtype
            coarse_grid (Grid, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[torch.Tensor, Grid]: A tuple containing:
                - The pooled data as a torch.Tensor
                - The coarse Grid containing the pooled structure
        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.avg_pool(pool_factor, jagged_data, stride, coarse_grid_impl)
        return result_data.jdata, Grid(impl=cast(GridBatchCpp, result_grid_impl))

    def clip(
        self, features: torch.Tensor, ijk_min: NumericMaxRank1, ijk_max: NumericMaxRank1
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Clip the grid to a bounding box and return clipped features.

        Creates a new grid containing only the voxels that fall within the specified
        bounding box range [ijk_min, ijk_max].

        Args:
            features (torch.Tensor): The voxel features to clip. Shape should be (total_voxels, channels).
            ijk_min (NumericMaxRank1): Minimum bounds in index space,
                broadcastable to shape (3,), integer dtype
            ijk_max (NumericMaxRank1): Maximum bounds in index space,
                broadcastable to shape (3,), integer dtype

        Returns:
            tuple[torch.Tensor, Grid]: A tuple containing:
                - The clipped features as a torch.Tensor
                - A new Grid containing only voxels within the bounds
        """
        jagged_features = JaggedTensor(features)
        ijk_min = to_Vec3iBroadcastable(ijk_min)
        ijk_max = to_Vec3iBroadcastable(ijk_max)

        result_features, result_grid_impl = self._impl.clip(jagged_features, ijk_min, ijk_max)
        return result_features.jdata, Grid(impl=result_grid_impl)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank1,
        ijk_max: NumericMaxRank1,
    ) -> "Grid":
        """
        Return a grid representing the clipped version of this grid.
        Each voxel `[i, j, k]` in the input grid is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            ijk_min (NumericMaxRank1): Index space minimum bound of the clip region,
                broadcastable to shape (3,), integer dtype
            ijk_max (NumericMaxRank1): Index space maximum bound of the clip region,
                broadcastable to shape (3,), integer dtype

        Returns:
            clipped_grid (Grid): A Grid representing the clipped version of this grid.
        """
        ijk_min = to_Vec3iBroadcastable(ijk_min)
        ijk_max = to_Vec3iBroadcastable(ijk_max)
        return Grid(impl=self._impl.clipped_grid(ijk_min, ijk_max))

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> "Grid":
        """
        Return a grid representing the coarsened version of this grid.
        Each voxel `[i, j, k]` in the input is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            coarsening_factor (NumericMaxRank1): The factor by which to coarsen the grid,
                broadcastable to shape (3,), integer dtype

        Returns:
            coarsened_grid (Grid): A Grid representing the coarsened version of this grid.
        """
        coarsening_factor = to_Vec3iBroadcastable(coarsening_factor, value_constraint=ValueConstraint.POSITIVE)
        return Grid(impl=self._impl.coarsened_grid(coarsening_factor))

    def contiguous(self) -> "Grid":
        """
        Return a contiguous copy of the grid.

        Ensures that the underlying data is stored contiguously in memory,
        which can improve performance for subsequent operations.

        Returns:
            Grid: A new Grid with contiguous memory layout.
        """
        return Grid(impl=self._impl.contiguous())

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 0) -> "Grid":
        """
        Return a grid representing the convolution of this grid with a given kernel.

        Args:
            kernel_size (NumericMaxRank1): The size of the kernel to convolve with,
                broadcastable to shape (3,), integer dtype
            stride (NumericMaxRank1): The stride to use when convolving,
                broadcastable to shape (3,), integer dtype

        Returns:
            conv_grid (Grid): A Grid representing the convolution of this grid.
        """
        kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        return Grid(impl=self._impl.conv_grid(kernel_size, stride))

    def coords_in_active_voxel(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Check if voxel coordinates are in active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates to check.
                Shape: (num_queries, 3) with integer coordinates.

        Returns:
            torch.Tensor: Boolean mask indicating which coordinates correspond to
                active voxels. Shape: (num_queries,).
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.coords_in_active_voxel(jagged_ijk).jdata

    def cpu(self) -> "Grid":
        """
        Move the grid to CPU.

        Returns:
            Grid: A new Grid on CPU device.
        """
        return Grid(impl=self._impl.cpu())

    def cubes_in_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """
        Check if axis-aligned cubes are fully contained within the grid.

        Tests whether cubes defined by their centers and bounds are completely inside
        the active voxels of the grid.

        Args:
            cube_centers (torch.Tensor): Centers of the cubes in world coordinates.
                Shape: (num_cubes, 3).
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds,
                broadcastable to shape (3,), floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds,
                broadcastable to shape (3,), floating dtype

        Returns:
            torch.Tensor: Boolean mask indicating which cubes are fully contained in the grid.
                Shape: (num_cubes,).
        """
        jagged_cube_centers = JaggedTensor(cube_centers)
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)

        return self._impl.cubes_in_grid(jagged_cube_centers, cube_min, cube_max).jdata

    def cubes_intersect_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """
        Check if axis-aligned cubes intersect with the grid.

        Tests whether cubes defined by their centers and bounds have any intersection
        with the active voxels of the grid.

        Args:
            cube_centers (torch.Tensor): Centers of the cubes in world coordinates.
                Shape: (num_cubes, 3).
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds,
                broadcastable to shape (3,), floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds,
                broadcastable to shape (3,), floating dtype

        Returns:
            torch.Tensor: Boolean mask indicating which cubes intersect the grid.
                Shape: (num_cubes,).
        """
        jagged_cube_centers = JaggedTensor(cube_centers)
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)
        return self._impl.cubes_intersect_grid(jagged_cube_centers, cube_min, cube_max).jdata

    def cuda(self) -> "Grid":
        """
        Move the grid to CUDA device.

        Returns:
            Grid: A new Grid on CUDA device.
        """
        return Grid(impl=self._impl.cuda())

    def dilated_grid(self, dilation: int) -> "Grid":
        """
        Return the grid dilated by a given number of voxels.

        Args:
            dilation (int): The dilation radius in voxels.

        Returns:
            Grid: A new Grid with dilated active regions.
        """
        return Grid(impl=self._impl.dilated_grid(dilation))

    def dual_grid(self, exclude_border: bool = False) -> "Grid":
        """
        Return the dual grid where voxel centers correspond to corners of the primal grid.

        The dual grid is useful for staggered grid discretizations and finite difference operations.

        Args:
            exclude_border (bool): If True, excludes border voxels that would extend beyond
                the primal grid bounds. Default is False.

        Returns:
            Grid: A new Grid representing the dual grid.
        """
        return Grid(impl=self._impl.dual_grid(exclude_border))

    def grid_to_world(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Convert grid (index) coordinates to world coordinates.

        Transforms voxel indices to their corresponding positions in world space
        using the grid's origin and voxel size.

        Args:
            ijk (torch.Tensor): Grid coordinates to convert.
                Shape: (num_points, 3). Can be fractional for interpolation.

        Returns:
            torch.Tensor: World coordinates. Shape: (num_points, 3).
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.grid_to_world(jagged_ijk).jdata

    def has_same_address_and_grid_count(self, other: Any) -> bool:
        """
        Check if two grid batches have the same address and grid count.
        """
        if isinstance(other, Grid):
            return self.address == other.address
        elif isinstance(other, GridBatchCpp):
            return self.address == other.address and self._impl.grid_count == other.grid_count
        else:
            return False

    def ijk_to_index(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Convert voxel coordinates to linear indices.

        Maps 3D voxel coordinates to their corresponding linear indices in the sparse storage.
        Returns -1 for coordinates that don't correspond to active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates to convert.
                Shape: (num_queries, 3) with integer coordinates.

        Returns:
            torch.Tensor: Linear indices for each coordinate, or -1 if not active.
                Shape: (num_queries,).
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.ijk_to_index(jagged_ijk).jdata

    def ijk_to_inv_index(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Get inverse permutation for ijk_to_index.

        Args:
            ijk (torch.Tensor): Voxel coordinates to convert.
                Shape: (num_queries, 3) with integer coordinates.

        Returns:
            torch.Tensor: Inverse permutation for ijk_to_index.
                Shape: (num_queries,).
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.ijk_to_inv_index(jagged_ijk).jdata

    def inject_from(
        self,
        src_grid: "Grid",
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """
        Inject data from the source grid to this grid.
        This method copies sidecar data for voxels in the source grid to a sidecar corresponding to voxels in this grid.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        If you pass in the destination data (`dst`), it will be modified in-place.
        If `dst` is None, a new Tensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            dst_grid (Grid): The destination grid to inject data into.
            src (torch.Tensor): Source data from this grid.
                This must be a Tensor with shape (-1, *).
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape (-1, *) or None.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.

        Returns:
            torch.Tensor: The destination sidecar data after injection.
        """
        jagged_src = JaggedTensor(src)

        if dst is None:
            dst_shape = [self.num_voxels, *src.shape[1:]] if src.dim() > 1 else [self.num_voxels]
            dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)

        jagged_dst = JaggedTensor(dst)

        src_grid._impl.inject_to(self._impl, jagged_src, jagged_dst)

        return jagged_dst.jdata

    def inject_from_ijk(
        self,
        src_ijk: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ):
        """

        Inject data from source voxel coordinates to this grid.

        This method copies sidecar data for voxels at specified indices in the source grid
        to a sidecar corresponding to voxels in this grid.

        If you pass in the destination data (`dst`), it will be modified in-place.

        If `dst` is None, a new Tensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            src_ijk (torch.Tensor): Source voxel coordinates in index space.
                Shape: (num_src_voxels, 3) with integer coordinates.
            src (torch.Tensor): Source data from the source grid.
                This must be a Tensor with shape (-1, *).
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape (-1, *) or None.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.
        """

        if not isinstance(src_ijk, torch.Tensor):
            raise TypeError(f"src_ijk must be a torch.Tensor, but got {type(src_ijk)}")

        if not isinstance(src, torch.Tensor):
            raise TypeError(f"src must be a torch.Tensor, but got {type(src)}")

        if dst is None:
            dst_shape = [self.num_voxels, *src.shape[1:]] if src.dim() > 1 else [self.num_voxels]
            dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
        else:
            if not isinstance(dst, torch.Tensor):
                raise TypeError(f"dst must be a torch.Tensor, but got {type(dst)}")
        if src_ijk.dim() != 2 or src_ijk.shape[1] != 3:
            raise ValueError(f"src_ijk must have shape (num_src_voxels, 3), but got {src_ijk.shape}")

        if src_ijk.dtype != torch.int32 and src_ijk.dtype != torch.int64:
            raise ValueError(f"src_ijk must have integer dtype, but got {src_ijk.dtype}")

        if src_ijk.device != src.device:
            raise ValueError(f"src_ijk must be on the same device as src, but got {src_ijk.device} and {src.device}")

        if src_ijk.shape[0] != src.shape[0]:
            raise ValueError(
                f"src_ijk and src must have the same number of elements, but got {src_ijk.shape[0]} and {src.shape[0]}"
            )
        if dst.shape[0] != self.num_voxels:
            raise ValueError(
                f"dst must have the same number of elements as the grid, "
                f"but got {dst.shape[0]} and {self.num_voxels}"
            )
        if dst.shape[1:] != src.shape[1:]:
            raise ValueError(
                f"dst must have the same shape as src except for the first dimension, "
                f"but got {dst.shape[1:]} and {src.shape[1:]}"
            )
        src_idx = self.ijk_to_index(src_ijk)
        src_mask = src_idx >= 0
        src_idx = src_idx[src_mask]
        dst[src_idx] = src[src_mask]

        return dst

    def inject_to(
        self,
        dst_grid: "Grid",
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """
        Inject data from this grid to a destination grid.
        This method copies sidecar data for voxels in this grid to a sidecar corresponding to
        voxels in the destination grid.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        If you pass in the destination data (`dst`), it will be modified in-place.
        If `dst` is None, a new Tensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            dst_grid (Grid): The destination grid to inject data into.
            src (torch.Tensor): Source data from this grid.
                This must be a Tensor with shape (-1, *).
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape (-1, *) or None.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.

        Returns:
            torch.Tensor: The destination sidecar data after injection.
        """
        jagged_src = JaggedTensor(src)

        if dst is None:
            dst_shape = [dst_grid.num_voxels, *src.shape[1:]] if src.dim() > 1 else [dst_grid.num_voxels]
            jagged_dst = JaggedTensor(
                torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
            )
        else:
            jagged_dst = JaggedTensor(dst)
        self._impl.inject_to(dst_grid._impl, jagged_src, jagged_dst)
        return jagged_dst.jdata

    def integrate_tsdf(
        self,
        truncation_distance: float,
        projection_matrix: torch.Tensor,
        cam_to_world_matrix: torch.Tensor,
        tsdf: torch.Tensor,
        weights: torch.Tensor,
        depth_image: torch.Tensor,
        weight_image: torch.Tensor | None = None,
    ) -> tuple["Grid", torch.Tensor, torch.Tensor]:
        """
        Integrate depth images into a Truncated Signed Distance Function (TSDF) volume.

        Updates the TSDF values and weights in the voxel grid by integrating new depth
        observations from multiple camera viewpoints. This is commonly used for 3D
        reconstruction from RGB-D sensors.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values (in world units).
            projection_matrix (torch.Tensor): Camera projection matrix.
                Shape: (4, 4).
            cam_to_world_matrix (torch.Tensor): Camera to world transformation matrix.
                Shape: (4, 4).
            tsdf (torch.Tensor): Current TSDF values for each voxel.
                Shape: (total_voxels, 1).
            weights (torch.Tensor): Current integration weights for each voxel.
                Shape: (total_voxels, 1).
            depth_image (torch.Tensor): Depth image from cameras.
                Shape: (height, width).
            weight_image (torch.Tensor, optional): Weight of each depth sample in the image.
                Shape: (height, width). If None, defaults to uniform weights.

        Returns:
            tuple[Grid, torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated Grid with potentially expanded voxels
                - Updated TSDF values as torch.Tensor
                - Updated weights as torch.Tensor
        """
        if cam_to_world_matrix.shape != (4, 4):
            raise ValueError(f"cam_to_world_matrix must have shape (4, 4), but got {cam_to_world_matrix.shape}")
        if projection_matrix.shape != (3, 3):
            raise ValueError(f"projection_matrix must have shape (3, 3), but got {projection_matrix.shape}")

        if tsdf.dim() != 1:
            if tsdf.dim() != 2 or tsdf.shape[1] != 1:
                raise ValueError(f"tsdf must have shape (N, 1) or (N,), but got {tsdf.shape}")

        if tsdf.shape[0] != weights.shape[0]:
            raise ValueError(
                f"tsdf and weights must have the same number of elements, "
                f"but got {tsdf.shape[0]} and {weights.shape[0]}"
            )

        if weights.dim() != 1:
            if weights.dim() != 2 or weights.shape[1] != 1:
                raise ValueError(f"weights must have shape (N, 1) or (N,), but got {weights.shape}")

        if depth_image.dim() != 2:
            if depth_image.dim() != 3 or depth_image.shape[2] != 1:
                raise ValueError(f"depth_image must have shape (height, width), but got {depth_image.shape}")

        if weight_image is not None:
            if weight_image.dim() != 2:
                if weight_image.dim() != 3 or weight_image.shape[2] != 1:
                    raise ValueError(
                        f"weight_image must have shape (height, width) or (height, width, 1), but got {weight_image.shape}"
                    )

            if weight_image.shape[:2] != depth_image.shape[:2]:
                raise ValueError(
                    f"weight_image must have the same shape as depth_image, "
                    f"but got {weight_image.shape[:2]} and {depth_image.shape[:2]}"
                )

        jagged_tsdf = JaggedTensor(tsdf)

        jagged_weights = JaggedTensor(weights)

        result_grid_impl, result_jagged_1, result_jagged_2 = self._impl.integrate_tsdf(
            truncation_distance,
            projection_matrix.unsqueeze(0),
            cam_to_world_matrix.unsqueeze(0),
            jagged_tsdf,
            jagged_weights,
            depth_image.unsqueeze(0),
            weight_image.unsqueeze(0) if weight_image is not None else None,
        )

        return Grid(impl=result_grid_impl), result_jagged_1.jdata, result_jagged_2.jdata

    def integrate_tsdf_with_features(
        self,
        truncation_distance: float,
        projection_matrix: torch.Tensor,
        cam_to_world_matrix: torch.Tensor,
        tsdf: torch.Tensor,
        features: torch.Tensor,
        weights: torch.Tensor,
        depth_image: torch.Tensor,
        feature_image: torch.Tensor,
        weight_image: torch.Tensor | None = None,
    ) -> tuple["Grid", torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate depth and feature images into TSDF volume with features.

        Similar to integrate_tsdf but also integrates feature observations (e.g., color)
        along with the depth information. This is useful for colored 3D reconstruction.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values.
            projection_matrix (torch.Tensor): Camera projection matrix.
                Shape: (3, 3).
            cam_to_world_matrix (torch.Tensor): Camera to world transformation matrix.
                Shape: (4, 4).
            tsdf (torch.Tensor): Current TSDF values for each voxel.
                Shape: (total_voxels, 1).
            features (torch.Tensor): Current feature values for each voxel.
                Shape: (total_voxels, feature_dim).
            weights (torch.Tensor): Current integration weights for each voxel.
                Shape: (total_voxels, 1).
            depth_image (torch.Tensor): Depth image from cameras.
                Shape: (height, width).
            feature_image (torch.Tensor): Feature image (e.g., RGB) from cameras.
                Shape: (height, width, feature_dim).
            weight_image (torch.Tensor, optional): Weight of each depth sample in the image.
                Shape: (height, width). If None, defaults to uniform weights.

        Returns:
            tuple[Grid, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated Grid with potentially expanded voxels
                - Updated TSDF values as torch.Tensor
                - Updated weights as torch.Tensor
                - Updated features as torch.Tensor
        """
        if cam_to_world_matrix.shape != (4, 4):
            raise ValueError(f"cam_to_world_matrix must have shape (4, 4), but got {cam_to_world_matrix.shape}")
        if projection_matrix.shape != (3, 3):
            raise ValueError(f"projection_matrix must have shape (3, 3), but got {projection_matrix.shape}")

        if tsdf.dim() != 1:
            if tsdf.dim() != 2 or tsdf.shape[1] != 1:
                raise ValueError(f"tsdf must have shape (N, 1) or (N,), but got {tsdf.shape}")

        if weights.dim() != 1:
            if weights.dim() != 2 or weights.shape[1] != 1:
                raise ValueError(f"weights must have shape (N, 1) or (N,), but got {weights.shape}")

        if features.dim() != 2:
            raise ValueError(f"features must have shape (N, feature_dim), but got {features.shape}")

        if features.shape[0] != tsdf.shape[0]:
            raise ValueError(
                f"features must have the same number of voxels as tsdf, "
                f"but got {features.shape[0]} and {tsdf.shape[0]}"
            )
        if weights.shape[0] != tsdf.shape[0]:
            raise ValueError(
                f"weights must have the same number of voxels as tsdf, "
                f"but got {weights.shape[0]} and {tsdf.shape[0]}"
            )

        if depth_image.dim() != 2:
            if depth_image.dim() != 3 or depth_image.shape[2] != 1:
                raise ValueError(f"depth_image must have shape (height, width), but got {depth_image.shape}")

        if feature_image.dim() != 3 or feature_image.shape[2] < 1:
            raise ValueError(
                f"feature_image must have shape (height, width, feature_dim), " f"but got {feature_image.shape}"
            )
        if feature_image.shape[:2] != depth_image.shape[:2]:
            raise ValueError(
                f"feature_image must have the same shape as depth_image, "
                f"but got {feature_image.shape[:2]} and {depth_image.shape[:2]}"
            )
        if feature_image.shape[2] != features.shape[1]:
            raise ValueError(
                f"feature_image's last dimension must match features' second dimension, "
                f"but got {feature_image.shape[2]} and {features.shape[1]}"
            )

        if weight_image is not None:
            if weight_image.dim() != 2:
                if weight_image.dim() != 3 or weight_image.shape[2] != 1:
                    raise ValueError(
                        f"weight_image must have shape (height, width) or (height, width, 1), but got {weight_image.shape}"
                    )

            if weight_image.shape[:2] != depth_image.shape[:2]:
                raise ValueError(
                    f"weight_image must have the same shape as depth_image, "
                    f"but got {weight_image.shape[:2]} and {depth_image.shape[:2]}"
                )

        jagged_tsdf = JaggedTensor(tsdf)
        jagged_weights = JaggedTensor(weights)
        jagged_features = JaggedTensor(features)

        result_grid_impl, result_jagged_1, result_jagged_2, result_jagged_3 = self._impl.integrate_tsdf_with_features(
            truncation_distance,
            projection_matrix.unsqueeze(0),
            cam_to_world_matrix.unsqueeze(0),
            jagged_tsdf,
            jagged_features,
            jagged_weights,
            depth_image.unsqueeze(0),
            feature_image.unsqueeze(0),
            weight_image.unsqueeze(0) if weight_image is not None else None,
        )

        return Grid(impl=result_grid_impl), result_jagged_1.jdata, result_jagged_2.jdata, result_jagged_3.jdata

    def is_contiguous(self) -> bool:
        """
        Check if the grid data is stored contiguously in memory.

        Returns:
            bool: True if the data is contiguous, False otherwise.
        """
        return self._impl.is_contiguous()

    def is_same(self, other: "Grid") -> bool:
        """
        Check if two grids have the same structure.

        Compares the voxel structure, dimensions, and origins of two grids.

        Args:
            other (Grid): The other grid to compare with.

        Returns:
            bool: True if the grids have identical structure, False otherwise.
        """
        return self._impl.is_same(other._impl)

    def marching_cubes(
        self, field: torch.Tensor, level: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract isosurface mesh using the marching cubes algorithm.

        Generates a triangle mesh representing the isosurface at the specified level
        from a scalar field defined on the voxels.

        Args:
            field (torch.Tensor): Scalar field values at each voxel.
                Shape: (total_voxels, 1).
            level (float): The isovalue to extract the surface at. Default is 0.0.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Vertex positions of the mesh. Shape: (num_vertices, 3).
                - Triangle face indices. Shape: (num_faces, 3).
                - Vertex normals (computed from gradients). Shape: (num_vertices, 3).
        """
        jagged_field = JaggedTensor(field)
        verts, indices, normals = self._impl.marching_cubes(jagged_field, level)
        return verts.jdata, indices.jdata, normals.jdata

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: torch.Tensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "Grid | None" = None,
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Downsample grid data using max pooling.

        Performs max pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the maximum of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid,
                broadcastable to shape (3,), integer dtype
            data (torch.Tensor): The voxel data to pool. Shape should be
                (total_voxels, channels).
            stride (NumericMaxRank1): The stride to use when pooling,
                broadcastable to shape (3,), integer dtype
            coarse_grid (Grid, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[torch.Tensor, Grid]: A tuple containing:
                - The pooled data as a torch.Tensor
                - The coarse Grid containing the pooled structure
        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.max_pool(pool_factor, jagged_data, stride, coarse_grid_impl)
        return result_data.jdata, Grid(impl=result_grid_impl)

    def merged_grid(self, other: "Grid") -> "Grid":
        """
        Return a grid that is the union of this grid with another.

        Merges two grids by taking the union of their active voxels.
        The grids must have compatible dimensions and transforms.

        Args:
            other (Grid): The other grid batch to merge with.

        Returns:
            Grid: A new Grid containing the union of active voxels from both grids.
        """
        return Grid(impl=self._impl.merged_grid(other._impl))

    def neighbor_indexes(self, ijk: torch.Tensor, extent: int, bitshift: int = 0) -> torch.Tensor:
        """
        Get indices of neighbors in N-ring neighborhood.

        Finds the linear indices of all voxels within a specified neighborhood ring
        around the given voxel coordinates.

        Args:
            ijk (torch.Tensor): Voxel coordinates to find neighbors for.
                Shape: (num_queries, 3) with integer coordinates.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): Bit shift value for encoding. Default is 0.

        Returns:
            torch.Tensor: Linear indices of neighboring voxels.
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.neighbor_indexes(jagged_ijk, extent, bitshift).jdata

    def points_in_active_voxel(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if world-space points are located within active voxels.

        Tests whether the given points fall within voxels that are active in the grid.

        Args:
            points (torch.Tensor): World-space points to test.
                Shape: (num_points, 3).

        Returns:
            torch.Tensor: Boolean mask indicating which points are in active voxels.
                Shape: (num_points,).
        """
        jagged_points = JaggedTensor(points)
        return self._impl.points_in_active_voxel(jagged_points).jdata

    def pruned_grid(self, mask: torch.Tensor) -> "Grid":
        """
        Return a pruned grid based on a boolean mask.

        Creates a new grid containing only the voxels where the mask is True.

        Args:
            mask (torch.Tensor): Boolean mask for each voxel.
                Shape: (total_voxels,).

        Returns:
            GridBatch: A new GridBatch containing only voxels where mask is True.
        """
        jagged_mask = JaggedTensor(mask)
        return Grid(impl=self._impl.pruned_grid(jagged_mask))

    def ray_implicit_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        grid_scalars: torch.Tensor,
        eps: float = 0.0,
    ) -> torch.Tensor:
        """
        Find ray intersections with implicit surface defined by grid scalars.

        Computes intersection points between rays and an implicit surface defined by
        scalar values stored in the grid voxels (e.g., signed distance function).

        Args:
            ray_origins (JaggedTensor or torch.Tensor): Starting points of rays in world space.
                Shape: (num_rays, 3).
            ray_directions (JaggedTensor or torch.Tensor): Direction vectors of rays.
                Shape: (num_rays, 3). Should be normalized.
            grid_scalars (JaggedTensor or torch.Tensor): Scalar field values at each voxel.
                Shape: (total_voxels, 1).
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            JaggedTensor: Intersection information for each ray.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)
        jagged_grid_scalars = JaggedTensor(grid_scalars)

        return self._impl.ray_implicit_intersection(
            jagged_ray_origins, jagged_ray_directions, jagged_grid_scalars, eps
        ).jdata

    def rays_intersect_voxels(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, eps: float = 0.0
    ) -> torch.Tensor:
        """
        Return a boolean tensor indicating which rays intersect this Grid.

        Args:
            ray_origins (torch.Tensor): an [N, 3] shaped tensor of ray origins
            ray_directions (torch.Tensor): an [N, 3] shaped tensor of ray directions
            eps (float): a small value to avoid numerical instability

        Returns:
            torch.Tensor: a boolean tensor of shape [N,] indicating which rays intersect the grid.
                _i.e._ `rays_intersect_voxels(ray_origins, ray_directions, eps)[i]` is `True` if the
                ray corresponding to `ray_origins[i]`, `ray_directions[i]` intersects with this Grid.
        """
        _, ray_times = self.voxels_along_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            max_voxels=1,
            eps=eps,
            return_ijk=False,
        )
        return (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0

    def read_from_dense_xyzc(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """
        Read values from a dense tensor into sparse grid structure.

        Data is read from a dense tensor with XYZC order shape:
        - [dense_size_x, dense_size_y, dense_size_z, channels*]

        Extracts values from a dense tensor at locations corresponding to active voxels
        in the sparse grid. Useful for converting dense data to sparse representation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from.
                Shape: (dense_size_x, dense_size_y, dense_size_z, channels*).
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                grid index space, broadcastable to shape (3,), integer dtype

        Returns:
            torch.Tensor: Values from the dense tensor at active voxel locations.
                Shape: (total_voxels, channels).
        """
        dense_origin = to_Vec3i(dense_origin)
        return self._impl.read_from_dense_xyzc(dense_data.unsqueeze(0), dense_origin).jdata

    def read_from_dense_czyx(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """
        Read values from a dense tensor into sparse grid structure.

        Data is read from a dense tensor with CZYX order shape:
        - [channels*, dense_size_z, dense_size_y, dense_size_x]

        Extracts values from a dense tensor at locations corresponding to active voxels
        in the sparse grid. Useful for converting dense data to sparse representation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from.
                Shape: (channels*, dense_size_z, dense_size_y, dense_size_x).
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                grid index space, broadcastable to shape (3,), integer dtype

        Returns:
            torch.Tensor: Values from the dense tensor at active voxel locations.
                Shape: (total_voxels, channels).
        """
        dense_origin = to_Vec3i(dense_origin)
        return self._impl.read_from_dense_czyx(dense_data.unsqueeze(0), dense_origin).jdata

    def sample_bezier(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """
        Sample voxel features at arbitrary points using Bézier interpolation.

        Interpolates voxel data at continuous world-space positions using cubic Bézier
        interpolation. This provides smoother interpolation than trilinear but is more
        computationally expensive.

        Args:
            points (torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            torch.Tensor: Interpolated features at each point.
                Shape: (num_points, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)
        return self._impl.sample_bezier(jagged_points, jagged_voxel_data).jdata

    def sample_bezier_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample voxel features and their gradients using Bézier interpolation.

        Similar to sample_bezier but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Interpolated features at each point. Shape: (num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (num_points, 3, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        result_data, result_grad = self._impl.sample_bezier_with_grad(jagged_points, jagged_voxel_data)
        return result_data.jdata, result_grad.jdata

    def sample_trilinear(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """
        Sample voxel features at arbitrary points using trilinear interpolation.

        Interpolates voxel data at continuous world-space positions using trilinear
        interpolation from the 8 nearest voxels. Points outside the grid return zero.

        Args:
            points (torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            torch.Tensor: Interpolated features at each point.
                Shape: (num_points, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_trilinear(jagged_points, jagged_voxel_data).jdata

    def sample_trilinear_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample voxel features and their gradients using trilinear interpolation.

        Similar to sample_trilinear but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Interpolated features at each point. Shape: (num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (num_points, 3, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        result_data, result_grad = self._impl.sample_trilinear_with_grad(jagged_points, jagged_voxel_data)
        return result_data.jdata, result_grad.jdata

    def segments_along_rays(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, max_segments: int, eps: float = 0.0
    ) -> JaggedTensor:
        """
        Enumerate segments along rays.

        Args:
            ray_origins (torch.Tensor): Origin of each ray.
                Shape: (num_rays, 3).
            ray_directions (torch.Tensor): Direction of each ray.
                Shape: (num_rays, 3).
            max_segments (int): Maximum number of segments to enumerate.
            eps (float): Small epsilon value to avoid numerical issues.

        Returns:
            A torch.Tensor containing the samples along the rays. i.e. a torch.Tensor
            with lshape [S_{0}, ..., S_{N_0}] and eshape (2,) or (1,)
            representing the start and end distance of each sample or the midpoint
            of each sample if return_midpoints is true.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)

        return self._impl.segments_along_rays(jagged_ray_origins, jagged_ray_directions, max_segments, eps)[0]

    def sparse_conv_halo(self, input: torch.Tensor, weight: torch.Tensor, variant: int = 8) -> torch.Tensor:
        """
        Perform sparse convolution with halo exchange optimization.

        Applies sparse convolution using halo exchange to efficiently handle boundary
        conditions in distributed or multi-block sparse grids.

        Args:
            input (torch.Tensor): Input features for each voxel.
                Shape: (total_voxels, in_channels).
            weight (torch.Tensor): Convolution weights.
            variant (int): Variant of the halo implementation to use. Default is 8.

        Returns:
            torch.Tensor: Output features after convolution.
        """
        jagged_input = JaggedTensor(input)
        return self._impl.sparse_conv_halo(jagged_input, weight, variant).jdata

    def sparse_conv_kernel_map(
        self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1, target_grid: "Grid | None" = None
    ) -> tuple["SparseConvPackInfo", "Grid"]:
        """
        Map sparse convolution kernel to target grid.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel,
                broadcastable to shape (3,), integer dtype
            stride (NumericMaxRank1): Stride of the convolution,
                broadcastable to shape (3,), integer dtype
            target_grid (Grid | None): Target grid to map the kernel to.
                If None, the kernel is mapped to the current grid.

        Returns:
            tuple[SparseConvPackInfo, Grid]: A tuple containing:
                - SparseConvPackInfo: Information about the sparse convolution kernel.
                - Grid: The target grid.
        """
        # Import here to avoid circular dependency
        from .sparse_conv_pack_info import SparseConvPackInfo

        kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        target_impl = target_grid._impl if target_grid is not None else None

        sparse_impl, grid_impl = self._impl.sparse_conv_kernel_map(kernel_size, stride, target_impl)
        return (SparseConvPackInfo(impl=sparse_impl), Grid(impl=grid_impl))

    def splat_bezier(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """
        Splat point features onto voxels using Bézier interpolation.

        Distributes features from point locations onto the surrounding voxels using
        cubic Bézier interpolation weights. This provides smoother distribution than
        trilinear splatting but is more computationally expensive.

        Args:
            points (torch.Tensor): World-space positions of points.
                Shape: (num_points, 3).
            points_data (torch.Tensor): Features to splat from each point.
                Shape: (num_points, channels).

        Returns:
            torch.Tensor: Accumulated features at each voxel after splatting.
                Shape: (total_voxels, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_points_data = JaggedTensor(points_data)

        return self._impl.splat_bezier(jagged_points, jagged_points_data).jdata

    def splat_trilinear(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """
        Splat point features onto voxels using trilinear interpolation.

        Distributes features from point locations onto the surrounding 8 voxels using
        trilinear interpolation weights. This is the standard method for converting
        point-based data to voxel grids.

        Args:
            points (torch.Tensor): World-space positions of points.
                Shape: (num_points, 3).
            points_data (torch.Tensor): Features to splat from each point.
                Shape: (num_points, channels).

        Returns:
            torch.Tensor: Accumulated features at each voxel after splatting.
                Shape: (total_voxels, channels).
        """
        jagged_points = JaggedTensor(points)
        jagged_points_data = JaggedTensor(points_data)

        return self._impl.splat_trilinear(jagged_points, jagged_points_data).jdata

    def subdivide(
        self,
        subdiv_factor: NumericMaxRank1,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        fine_grid: "Grid | None" = None,
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Subdivide grid using nearest neighbor interpolation.

        Increases the resolution of the grid by the specified subdivision factor,
        filling in new voxels using nearest neighbor interpolation of the existing data.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to subdivide the grid,
                broadcastable to shape (3,), integer dtype
            data (torch.Tensor): Voxel data to subdivide.
                Shape: (total_voxels, channels).
            mask (torch.Tensor, optional): Boolean mask indicating which
                voxels to subdivide. If None, all voxels are subdivided.
            fine_grid (Grid, optional): Pre-allocated fine grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[torch.Tensor, Grid]: A tuple containing:
                - The subdivided data as a torch.Tensor
                - The fine Grid containing the subdivided structure
        """
        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        jagged_mask = JaggedTensor(mask) if mask is not None else None

        fine_grid_impl = fine_grid._impl if fine_grid else None

        result_data, result_grid_impl = self._impl.subdivide(subdiv_factor, jagged_data, jagged_mask, fine_grid_impl)
        return result_data.jdata, Grid(impl=result_grid_impl)

    def subdivided_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: torch.Tensor | None = None,
    ) -> "Grid":
        """
        Return a subdivided version of the grid structure.

        Creates a new grid with higher resolution by subdividing existing voxels.
        Only the grid structure is returned, not the data.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to subdivide the grid,
                broadcastable to shape (3,), integer dtype
            mask (torch.Tensor, optional): Boolean mask indicating which
                voxels to subdivide. If None, all voxels are subdivided.

        Returns:
            Grid: A new Grid with subdivided structure.
        """

        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_mask = JaggedTensor(mask) if mask is not None else None

        return Grid(impl=self._impl.subdivided_grid(subdiv_factor, mask=jagged_mask))

    def to(self, target: "str | torch.device | torch.Tensor | JaggedTensor | Grid") -> "Grid":
        """
        Move grid batch to a target device or match device of target object.

        Args:
            target: Target to determine device. Can be:
                - str: Device string (e.g., "cuda", "cpu")
                - torch.device: PyTorch device object
                - torch.Tensor: Match device of this tensor
                - JaggedTensor: Match device of this JaggedTensor
                - Grid: Match device of this Grid

        Returns:
            Grid: A new Grid on the target device.
        """
        if isinstance(target, str):
            device = _parse_device_string(target)
            return Grid(impl=self._impl.to(device))
        elif isinstance(target, torch.device):
            return Grid(impl=self._impl.to(target))
        elif isinstance(target, torch.Tensor):
            return Grid(impl=self._impl.to(target))
        elif isinstance(target, JaggedTensor):
            return Grid(impl=self._impl.to(target))
        elif isinstance(target, Grid):
            return Grid(impl=self._impl.to(target._impl))
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")

    def uniform_ray_samples(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
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
            ray_origins (torch.Tensor): Starting points of rays in world space.
                Shape: (num_rays, 3).
            ray_directions (torch.Tensor): Direction vectors of rays (should be normalized).
                Shape: (num_rays, 3).
            t_min (torch.Tensor): Minimum distance along rays to start sampling.
                Shape: (num_rays,).
            t_max (torch.Tensor): Maximum distance along rays to stop sampling.
                Shape: (num_rays,).
            step_size (float): Distance between samples along each ray.
            cone_angle (float): Cone angle for cone tracing (in radians). Default is 0.0.
            include_end_segments (bool): Whether to include partial segments at ray ends.
                Default is True.
            return_midpoints (bool): Whether to return segment midpoints instead of start points.
                Default is False.
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            torch.Tensor: containing the samples along the rays. i.e. a torch.Tensor
              with lshape [S_{0}, ..., S_{N_0}] and eshape (2,) or (1,) representing the
              start and end distance of each sample or the midpoint
              of each sample if return_midpoints is true.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)
        jagged_t_min = JaggedTensor(t_min)
        jagged_t_max = JaggedTensor(t_max)

        return self._impl.uniform_ray_samples(
            jagged_ray_origins,
            jagged_ray_directions,
            jagged_t_min,
            jagged_t_max,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )[0]

    def voxels_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = True,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Enumerate voxels intersected by rays.

        Finds all active voxels that are intersected by the given rays using a
        DDA (Digital Differential Analyzer) algorithm.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space.
                Shape: (num_rays, 3).
            ray_directions (torch.Tensor): Direction vectors of rays (should be normalized).
                Shape: (num_rays, 3).
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Epsilon value for numerical stability. Default is 0.0.
            return_ijk (bool): Whether to return voxel indices. If False, returns
                linear indices instead. Default is True.

        Returns:
            A pair of torch.Tensors containing the voxels (or voxel indices) intersected by the rays. i.e.:
                - voxels: A torch.Tensor with lshape [V_{0}, ..., V_{N_0}]
                          and eshape (3,) or (,) containing the ijk coordinates or indices of the voxels
                - times: A torch.Tensor with lshape [T_{0}, ..., T_{N_0}]
                          and eshape (2,) containinng the entry and exit distance along the ray of each voxel
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)

        voxels, times = self._impl.voxels_along_rays(
            jagged_ray_origins, jagged_ray_directions, max_voxels, eps, return_ijk, True
        )
        return voxels[0], times[0]

    def world_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid (index) coordinates.

        Transforms positions in world space to their corresponding voxel indices
        using the grid's origin and voxel size. The resulting coordinates can be
        fractional for use in interpolation.

        Args:
            points (torch.Tensor): World-space positions to convert.
                Shape: (num_points, 3).

        Returns:
            torch.Tensor: Grid coordinates. Shape: (num_points, 3).
                Can contain fractional values.
        """
        jagged_points = JaggedTensor(points)
        return self._impl.world_to_grid(jagged_points).jdata

    def write_to_dense_xyzc(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Write sparse voxel data to a dense tensor.

        Data is written to a dense tensor with XYZC order shape:
        - [dense_size_x, dense_size_y, dense_size_z, channels*]

        Creates a dense tensor and fills it with values from the sparse grid.
        Voxels not present in the sparse grid are filled with zeros.

        Args:
            sparse_data (JaggedTensor or torch.Tensor): Sparse voxel features to write.
                Shape: (total_voxels, channels).
            min_coord (NumericMaxRank1|None, optional): Minimum coordinates the grid
                If None, computed from the grid bounds.
                broadcastable to shape (3,), integer dtype
            grid_size (NumericMaxRank1|None, optional): Size of the output dense tensor.
                If None, computed to fit all active voxels.
                broadcastable to shape (3,), integer dtype

        Returns:
            torch.Tensor: Dense tensor containing the sparse data.
                Shape: (dense_size_x, dense_size_y, dense_size_z, channels*)
        """
        jagged_sparse_data = JaggedTensor(sparse_data)
        min_coord = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        grid_size = (
            to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE)
            if grid_size is not None
            else None
        )
        return self._impl.write_to_dense_xyzc(jagged_sparse_data, min_coord, grid_size).squeeze(0)

    def write_to_dense_czyx(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Write sparse voxel data to a dense tensor.

        Data is written to a dense tensor with CZYX order shape:
        - [channels*, dense_size_z, dense_size_y, dense_size_x]

        Creates a dense tensor and fills it with values from the sparse grid.
        Voxels not present in the sparse grid are filled with zeros.

        Args:
            sparse_data (JaggedTensor or torch.Tensor): Sparse voxel features to write.
                Shape: (total_voxels, channels).
            min_coord (NumericMaxRank1|None, optional): Minimum coordinates the grid
                If None, computed from the grid bounds.
                broadcastable to shape (3,), integer dtype
            grid_size (NumericMaxRank1|None, optional): Size of the output dense tensor.
                If None, computed to fit all active voxels.
                broadcastable to shape (3,), integer dtype

        Returns:
            torch.Tensor: Dense tensor containing the sparse data.
                Shape: (channels*, dense_size_z, dense_size_y, dense_size_x)
        """
        jagged_sparse_data = JaggedTensor(sparse_data)
        min_coord = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        grid_size = (
            to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE)
            if grid_size is not None
            else None
        )
        return self._impl.write_to_dense_czyx(jagged_sparse_data, min_coord, grid_size).squeeze(0)

    # ============================================================
    #                        Properties
    # ============================================================

    # Properties
    @property
    def address(self) -> int:
        return self._impl.address

    @property
    def bbox(self) -> torch.Tensor:
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.bbox_at(0)

    @property
    def device(self) -> torch.device:
        return self._impl.device

    @property
    def dual_bbox(self) -> torch.Tensor:
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.dual_bbox_at(0)

    @property
    def grid_to_world_matrix(self) -> torch.Tensor:
        return self._impl.grid_to_world_matrices[0]

    @property
    def has_zero_voxels(self) -> bool:
        return self.num_voxels == 0

    @property
    def ijk(self) -> torch.Tensor:
        return self._impl.ijk.jdata

    def serialize_encode(self, order_type: str = "z") -> torch.Tensor:
        """
        Return the space-filling curve codes for active voxels in this grid.

        Space-filling curves provide a mapping from 3D coordinates to 1D integers,
        preserving spatial locality. This is useful for serialization, sorting, and
        spatial data structures.

        Args:
            order_type (str): The type of ordering to use:
                - "z": Regular Z-order curve (xyz bit interleaving)
                - "z-trans": Transposed Z-order curve (zyx bit interleaving)
                - "hilbert": Regular Hilbert curve (xyz)
                - "hilbert-trans": Transposed Hilbert curve (zyx)

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the space-filling curve codes for each active voxel.
        """
        return self._impl.serialize_encode(order_type).jdata

    def encode_morton(self) -> torch.Tensor:
        """
        Return Morton codes (Z-order curve) for active voxels in this grid.

        Morton codes use xyz bit interleaving to create a space-filling curve that
        preserves spatial locality. This is useful for serialization, sorting, and
        spatial data structures.

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the Morton codes for each active voxel.
        """
        return self._impl.encode_morton().jdata

    def encode_morton_zyx(self) -> torch.Tensor:
        """
        Return transposed Morton codes (Z-order curve) for active voxels in this grid.

        Transposed Morton codes use zyx bit interleaving to create a space-filling curve.
        This variant can provide better spatial locality for certain access patterns.

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the transposed Morton codes for each active voxel.
        """
        return self._impl.encode_morton_zyx().jdata

    def encode_hilbert(self) -> torch.Tensor:
        """
        Return Hilbert curve codes for active voxels in this grid.

        Hilbert curves provide better spatial locality than Morton codes by ensuring
        that nearby points in 3D space are also nearby in the 1D curve ordering.

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the Hilbert codes for each active voxel.
        """
        return self._impl.encode_hilbert().jdata

    def encode_hilbert_zyx(self) -> torch.Tensor:
        """
        Return transposed Hilbert curve codes for active voxels in this grid.

        Transposed Hilbert curves use zyx ordering instead of xyz. This variant can
        provide better spatial locality for certain access patterns.

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the transposed Hilbert codes for each active voxel.
        """
        return self._impl.encode_hilbert_zyx().jdata

    def permute(self, order_type: str = "z") -> torch.Tensor:
        """
        Get permutation indices to sort voxels by spatial order.

        This method computes space-filling curve codes for all active voxels and returns the indices
        that would sort them according to the specified ordering. This is useful for
        spatially coherent data access patterns and cache optimization.

        Args:
            order_type (str): The type of spatial ordering to use:
                - "z": Regular Z-order curve (xyz bit interleaving, default)
                - "z-trans": Transposed Z-order curve (zyx bit interleaving)
                - "hilbert": Regular Hilbert curve (xyz)
                - "hilbert-trans": Transposed Hilbert curve (zyx)

        Returns:
            torch.Tensor: A tensor of shape `[num_voxels, 1]` containing
                the permutation indices. Use these indices to reorder voxel data for spatial coherence.
        """
        # Get space-filling curve codes using the appropriate encode method
        if order_type == "z":
            curve_codes = self.encode_morton()
        elif order_type == "z-trans":
            curve_codes = self.encode_morton_zyx()
        elif order_type == "hilbert":
            curve_codes = self.encode_hilbert()
        elif order_type == "hilbert-trans":
            curve_codes = self.encode_hilbert_zyx()
        else:
            raise ValueError(
                f"Invalid order_type: {order_type}. Valid options are 'z', 'z-trans', 'hilbert', or 'hilbert-trans'."
            )

        # Get the curve codes as a flat tensor
        curve_data = curve_codes.squeeze(-1)  # Shape: [num_voxels]

        # Sort and get indices
        _, indices = torch.sort(curve_data, dim=0)

        # Return indices as a tensor with shape [num_voxels, 1]
        return indices.unsqueeze(-1)

    @property
    def num_bytes(self) -> int:
        return self._impl.total_bytes

    @property
    def num_leaf_nodes(self) -> int:
        return self._impl.total_leaf_nodes

    @property
    def num_voxels(self) -> int:
        return self._impl.total_voxels

    @property
    def origin(self) -> torch.Tensor:
        return self._impl.origin_at(0)

    @property
    def viz_edge_network(self) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self._impl.viz_edge_network
        return a.jdata, b.jdata

    @property
    def voxel_size(self) -> torch.Tensor:
        return self._impl.voxel_size_at(0)

    @property
    def world_to_grid_matrix(self) -> torch.Tensor:
        return self._impl.world_to_grid_matrices[0]

    # Expose underlying implementation for compatibility
    @property
    def _gridbatch(self):
        # Access underlying GridBatchCpp - use sparingly during migration
        return self._impl


# Load and save functions
@overload
def load_grid(
    path: str,
    *,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[Grid, torch.Tensor, str]: ...


@overload
def load_grid(
    path: str,
    *,
    index: int,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[Grid, torch.Tensor, str]: ...


@overload
def load_grid(
    path: str,
    *,
    name: str,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[Grid, torch.Tensor, str]: ...


def load_grid(
    path: str,
    *,
    index: int | None = None,
    name: str | None = None,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[Grid, torch.Tensor, str]:
    """Load a grid from a .nvdb file.

    Args:
        path: The path to the .nvdb file to load
        index: Optional single index to load from the file (mutually exclusive with other selectors)
        name: Optional single name to load from the file (mutually exclusive with other selectors)
        device: Which device to load the grid on
        verbose: If set to true, print information about the loaded grid

    Returns:
        A tuple (grid, data, name) where grid is a Grid containing the loaded
        grid, data is a torch.Tensor containing the data of the grid, and name is the name of the grid
    """
    from ._Cpp import load as _load

    resolved_device = resolve_device(device)

    # Check that only one selector is provided
    selectors = [index is not None, name is not None]
    if sum(selectors) > 1:
        raise ValueError("Only one of index or name can be specified")

    # Call the appropriate overload
    if index is not None:
        grid_impl, data, names_out = _load(path, index, resolved_device, verbose)
    elif name is not None:
        grid_impl, data, names_out = _load(path, name, resolved_device, verbose)
    else:
        # Load the first grid
        grid_impl, data, names_out = _load(path, 0, resolved_device, verbose)

    # Wrap the Grid implementation with the Python wrapper
    return Grid(impl=grid_impl.index_int(0)), data.jdata, names_out[0]


def save_grid(
    path: str,
    grid: Grid,
    data: torch.Tensor | None = None,
    name: str | None = None,
    compressed: bool = False,
    verbose: bool = False,
) -> None:
    """
    Save a grid and optional voxel data to a .nvdb file.

    Saves sparse grid in the NanoVDB format, which can be loaded by other
    applications that support OpenVDB/NanoVDB.

    Args:
        path (str): The file path to save to. Should have .nvdb extension.
        grid (Grid): The grid to save.
        data (torch.Tensor, optional): Voxel data to save with the gris.
            Shape: (total_voxels, channels). If None, only grid structure is saved.
        name (str, optional): Optional name for the grid
        compressed (bool): Whether to compress the data using Blosc compression.
            Default is False.
        verbose (bool): Whether to print information about the saved grids.
            Default is False.
    """
    from ._Cpp import save as _save

    jagged_data = JaggedTensor(data) if data is not None else None

    # Handle the overloaded signature - if name is provided, use it
    if name is not None:
        _save(path, grid._impl, jagged_data, name, compressed, verbose)
    else:
        # Default case with empty names list
        _save(path, grid._impl, jagged_data, [], compressed, verbose)
