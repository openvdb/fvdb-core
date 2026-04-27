# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Single sparse voxel grid data structure and operations for FVDB.

This module provides the core :class:`Grid` class for managing a single sparse voxel grid.

A :class:`Grid` wraps a :class:`~fvdb._fvdb_cpp.GridBatchData` with ``grid_count == 1``.
Every method delegates to the corresponding ``functional.*_single`` function, and
properties return plain :class:`torch.Tensor` or scalar values rather than
:class:`~fvdb.JaggedTensor`.

Class-methods for creating Grid objects from various sources:

- :meth:`Grid.from_ijk`: from explicit voxel coordinates
- :meth:`Grid.from_points`: from point clouds
- :meth:`Grid.from_mesh`: from triangle meshes
- :meth:`Grid.from_dense`: from dense data
- :meth:`Grid.from_dense_axis_aligned_bounds`: from dense data defined by axis-aligned bounds
- :meth:`Grid.from_nearest_voxels_to_points`: from nearest voxels to points
- :meth:`Grid.from_zero_voxels`: for an empty grid with zero voxels
- :meth:`Grid.from_nanovdb` / :meth:`Grid.save_nanovdb`: Load and save grids to/from .nvdb files
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, overload

import torch

from ._fvdb_cpp import GridBatchData
from .jagged_tensor import JaggedTensor
from .types import DeviceIdentifier, NumericMaxRank1

if TYPE_CHECKING:
    from .grid_batch import GridBatch


class Grid:
    """A single sparse voxel grid backed by a C++ :class:`GridBatchData` with ``grid_count == 1``.

    :class:`Grid` represents a single sparse 3D voxel grid that can be processed
    efficiently on GPU. The class provides methods for common operations like
    sampling, convolution, pooling, dilation, union, etc. It also provides more
    advanced features such as marching cubes, TSDF fusion, and fast ray marching.

    A :class:`Grid` does not store voxel data itself, but rather the structure
    (or topology) of the sparse voxel grid. Voxel data (e.g., features, colors,
    densities) are stored separately as :class:`torch.Tensor` associated with the
    grid. This separation allows for flexibility in the type and number of channels
    of data with which a grid can be used to index into.

    When using a :class:`Grid`, there are three important coordinate systems:

    - **World Space**: The continuous 3D coordinate system in which the grid exists.
    - **Voxel Space**: The discrete voxel index system, where each voxel is
      identified by its integer indices ``(i, j, k)``.
    - **Index Space**: The linear indexing of active voxels in the grid's internal
      storage.

    .. note::

        The grid is stored in a sparse format using
        `NanoVDB <https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb>`_
        where only active (non-empty) voxels are allocated, making it extremely
        memory efficient for representing large volumes with sparse occupancy.

    .. note::

        The :class:`Grid` constructor is for internal use only. To create a
        :class:`Grid` with actual content, use the classmethods:

        - :meth:`from_ijk`: from explicit voxel coordinates
        - :meth:`from_points`: from point clouds
        - :meth:`from_mesh`: from triangle meshes
        - :meth:`from_dense`: from dense data
        - :meth:`from_dense_axis_aligned_bounds`: from dense data defined by
          axis-aligned bounds
        - :meth:`from_nearest_voxels_to_points`: from nearest voxels to points
        - :meth:`from_zero_voxels`: for a grid with zero voxels

    """

    __slots__ = ("data",)

    def __init__(self, *, data: GridBatchData) -> None:
        """
        Constructor for internal use only -- use the ``Grid.from_*`` classmethods instead.
        """
        if data.grid_count != 1:
            raise ValueError(f"Grid requires grid_count == 1, got {data.grid_count}")
        object.__setattr__(self, "data", data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Grid is immutable")

    def __getstate__(self) -> dict:
        return {"data": self.data}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "data", state["data"])

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
    ) -> Grid:
        """Create a dense :class:`Grid` with a voxel for every coordinate in an axis-aligned box.

        Args:
            dense_dims (NumericMaxRank1): Dimensions of the dense grid,
                broadcastable to shape ``(3,)``, integer dtype.
            ijk_min (NumericMaxRank1): Minimum voxel index for the grid,
                broadcastable to shape ``(3,)``, integer dtype.
            voxel_size (NumericMaxRank1): World-space size of each voxel,
                broadcastable to shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): World-space coordinate of the center of the
                ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype.
            mask (torch.Tensor | None): Boolean mask with shape ``(W, H, D)``
                selecting active voxels.
            device (DeviceIdentifier | None): Device to create the grid on.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        from . import functional

        return functional.grid_from_dense(dense_dims, ijk_min, voxel_size, origin, mask, device)

    @classmethod
    def from_dense_axis_aligned_bounds(
        cls,
        dense_dims: NumericMaxRank1,
        bounds_min: NumericMaxRank1 = 0,
        bounds_max: NumericMaxRank1 = 1,
        voxel_center: bool = False,
        device: DeviceIdentifier = "cpu",
    ) -> Grid:
        """Create a dense :class:`Grid` defined by axis-aligned bounds in world space.

        Args:
            dense_dims (NumericMaxRank1): Dimensions of the dense grid,
                broadcastable to shape ``(3,)``, integer dtype.
            bounds_min (NumericMaxRank1): Minimum world-space bounds,
                broadcastable to shape ``(3,)``, floating dtype.
            bounds_max (NumericMaxRank1): Maximum world-space bounds,
                broadcastable to shape ``(3,)``, floating dtype.
            voxel_center (bool): Whether the bounds correspond to voxel centers
                (``True``) or edges (``False``). Defaults to ``False``.
            device (DeviceIdentifier): Device to create the grid on.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        from . import functional

        return functional.grid_from_dense_axis_aligned_bounds(dense_dims, bounds_min, bounds_max, voxel_center, device)

    @classmethod
    def from_ijk(
        cls,
        ijk: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> Grid:
        """Create a :class:`Grid` from voxel coordinates.

        If multiple voxels map to the same coordinate, only one voxel will be
        created at that coordinate.

        Args:
            ijk (torch.Tensor): Voxel coordinates of shape ``(num_voxels, 3)``
                with integer dtype.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to
                shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): World-space position of the center of the
                ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        from . import functional

        return functional.grid_from_ijk(ijk, voxel_size, origin)

    @classmethod
    def from_mesh(
        cls,
        mesh_vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> Grid:
        """Create a :class:`Grid` by voxelizing the surface of a triangle mesh.

        Args:
            mesh_vertices (torch.Tensor): Vertices of shape ``(num_vertices, 3)``.
            mesh_faces (torch.Tensor): Faces of shape ``(num_faces, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to
                shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): World-space position of the center of the
                ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            grid (Grid): A new :class:`Grid` with voxels covering the mesh surface.
        """
        from . import functional

        return functional.grid_from_mesh(mesh_vertices, mesh_faces, voxel_size, origin)

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> Grid:
        """Create a :class:`Grid` from a point cloud.

        Args:
            points (torch.Tensor): Points of shape ``(num_points, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to
                shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): World-space position of the center of the
                ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        from . import functional

        return functional.grid_from_points(points, voxel_size, origin)

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> Grid:
        """Create a :class:`Grid` by adding the eight nearest voxels to every point.

        Args:
            points (torch.Tensor): Points of shape ``(num_points, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to
                shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): World-space position of the center of the
                ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        from . import functional

        return functional.grid_from_nearest_voxels_to_points(points, voxel_size, origin)

    @classmethod
    def from_zero_voxels(
        cls,
        device: DeviceIdentifier = "cpu",
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> Grid:
        """Create a :class:`Grid` with zero voxels on a specific device.

        Args:
            device (DeviceIdentifier): Device to create the grid on.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to
                shape ``(3,)``, floating dtype.
            origin (NumericMaxRank1): Origin of the grid, broadcastable to
                shape ``(3,)``, floating dtype.

        Returns:
            grid (Grid): A new :class:`Grid` object with zero voxels.
        """
        from . import functional

        return functional.grid_from_zero_voxels(device, voxel_size, origin)

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str | pathlib.Path,
        *,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> tuple[Grid, torch.Tensor, str]: ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str | pathlib.Path,
        *,
        index: int,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> tuple[Grid, torch.Tensor, str]: ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str | pathlib.Path,
        *,
        name: str,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> tuple[Grid, torch.Tensor, str]: ...

    @classmethod
    def from_nanovdb(
        cls,
        path: str | pathlib.Path,
        *,
        index: int = 0,
        name: str | None = None,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> tuple[Grid, torch.Tensor, str]:
        """Load a single :class:`Grid` from a ``.nvdb`` file.

        Args:
            path (str | pathlib.Path): Path to the ``.nvdb`` file.
            index (int): Index of the grid to load from the file.
            name (str | None): Name of the grid to load (mutually exclusive
                with ``index``).
            device (DeviceIdentifier): Device to load the grid onto.
            verbose (bool): If ``True``, print information about the loaded grid.

        Returns:
            grid (Grid): The loaded :class:`Grid`.
            data (torch.Tensor): Voxel data with shape ``(num_voxels, channels*)``.
            name (str): Name of the loaded grid.
        """
        from . import functional

        return functional.load_nanovdb_single(
            str(path) if isinstance(path, pathlib.Path) else path,
            index=index,
            name=name,
            device=device,
            verbose=verbose,
        )

    # ============================================================
    #                         Properties
    # ============================================================

    @property
    def device(self) -> torch.device:
        """The :class:`torch.device` where this :class:`Grid` is stored.

        Returns:
            device (torch.device): The device of the grid.
        """
        return self.data.device

    @property
    def num_voxels(self) -> int:
        """The number of active voxels in this :class:`Grid`.

        Returns:
            num_voxels (int): Number of active voxels.
        """
        return self.data.total_voxels

    @property
    def voxel_size(self) -> torch.Tensor:
        """The world-space size of each voxel in this :class:`Grid`.

        Returns:
            voxel_size (torch.Tensor): Shape ``(3,)``.
        """
        return self.data.voxel_size_at(0).to(self.device)

    @property
    def origin(self) -> torch.Tensor:
        """The world-space origin of this :class:`Grid`, i.e. the center of voxel ``(0,0,0)``.

        Returns:
            origin (torch.Tensor): Shape ``(3,)``.
        """
        return self.data.origin_at(0).to(self.device)

    @property
    def bbox(self) -> torch.Tensor:
        """The voxel-space bounding box of this :class:`Grid`.

        Returns:
            bbox (torch.Tensor): Shape ``(2, 3)`` with ``[[min_i, min_j, min_k],
                [max_i, max_j, max_k]]``. Returns a zero tensor if the grid has
                zero voxels.
        """
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        return self.data.bbox_at(0).to(self.device)

    @property
    def dual_bbox(self) -> torch.Tensor:
        """The voxel-space bounding box of the dual of this :class:`Grid`.

        The dual grid has voxel centers at the corners of this grid's voxels.

        .. seealso:: :attr:`bbox`, :meth:`dual_grid`

        Returns:
            dual_bbox (torch.Tensor): Shape ``(2, 3)``. Returns a zero tensor if
                the grid has zero voxels.
        """
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        return self.data.dual_bbox_at(0).to(self.device)

    @property
    def ijk(self) -> torch.Tensor:
        """The voxel coordinates of every active voxel, in index order.

        Returns:
            ijk (torch.Tensor): Shape ``(num_voxels, 3)``.
        """
        from . import functional

        return functional.active_grid_coords_single(self)

    @property
    def voxel_to_world_matrix(self) -> torch.Tensor:
        """The voxel-to-world transformation matrix.

        Returns:
            voxel_to_world_matrix (torch.Tensor): Shape ``(4, 4)``.
        """
        return self.data.voxel_to_world_matrix_at(0)

    @property
    def world_to_voxel_matrix(self) -> torch.Tensor:
        """The world-to-voxel transformation matrix.

        Returns:
            world_to_voxel_matrix (torch.Tensor): Shape ``(4, 4)``.
        """
        return self.data.world_to_voxel_matrix_at(0)

    @property
    def num_bytes(self) -> int:
        """The size in bytes this :class:`Grid` occupies in memory.

        Returns:
            num_bytes (int): Size in bytes.
        """
        return self.data.num_bytes_at(0)

    @property
    def num_leaf_nodes(self) -> int:
        """The number of leaf nodes in the NanoVDB tree for this :class:`Grid`.

        Returns:
            num_leaf_nodes (int): Number of leaf nodes.
        """
        return self.data.num_leaves_at(0)

    @property
    def is_contiguous(self) -> bool:
        """Whether the grid data is stored contiguously in memory.

        Returns:
            is_contiguous (bool): ``True`` if contiguous.
        """
        return self.data.is_contiguous

    @property
    def has_zero_voxels(self) -> bool:
        """``True`` if this :class:`Grid` has zero active voxels.

        Returns:
            has_zero_voxels (bool): Whether the grid is empty.
        """
        return self.data.total_voxels == 0

    @property
    def morton(self) -> torch.Tensor:
        """Morton codes (Z-order curve, xyz interleaving) for active voxels.

        Returns:
            morton (torch.Tensor): Shape ``(num_voxels,)``.
        """
        from . import functional

        return functional.morton_single(self)

    @property
    def morton_zyx(self) -> torch.Tensor:
        """Transposed Morton codes (zyx interleaving) for active voxels.

        Returns:
            morton_zyx (torch.Tensor): Shape ``(num_voxels,)``.
        """
        from . import functional

        return functional.morton_zyx_single(self)

    @property
    def hilbert(self) -> torch.Tensor:
        """Hilbert curve codes for active voxels.

        Returns:
            hilbert (torch.Tensor): Shape ``(num_voxels,)``.
        """
        from . import functional

        return functional.hilbert_single(self)

    @property
    def hilbert_zyx(self) -> torch.Tensor:
        """Transposed Hilbert curve codes (zyx) for active voxels.

        Returns:
            hilbert_zyx (torch.Tensor): Shape ``(num_voxels,)``.
        """
        from . import functional

        return functional.hilbert_zyx_single(self)

    # ============================================================
    #                  Coordinate Transforms
    # ============================================================

    def voxel_to_world(self, ijk: torch.Tensor) -> torch.Tensor:
        """Transform voxel-space coordinates to world-space positions.

        .. seealso:: :meth:`world_to_voxel`, :attr:`voxel_to_world_matrix`

        Args:
            ijk (torch.Tensor): Voxel-space coordinates of shape ``(N, 3)``.
                Can be fractional for interpolation.

        Returns:
            world_coords (torch.Tensor): World-space coordinates of shape ``(N, 3)``.
        """
        from . import functional

        return functional.voxel_to_world_single(self, ijk)

    def world_to_voxel(self, points: torch.Tensor) -> torch.Tensor:
        """Convert world-space coordinates to voxel-space coordinates.

        .. seealso:: :meth:`voxel_to_world`, :attr:`world_to_voxel_matrix`

        Args:
            points (torch.Tensor): World-space positions of shape ``(N, 3)``.

        Returns:
            voxel_coords (torch.Tensor): Voxel-space coordinates of shape ``(N, 3)``.
                Can contain fractional values.
        """
        from . import functional

        return functional.world_to_voxel_single(self, points)

    # ============================================================
    #                        Queries
    # ============================================================

    def ijk_to_index(self, ijk: torch.Tensor, cumulative: bool = False) -> torch.Tensor:
        """Convert voxel-space coordinates to linear index-space.

        Returns ``-1`` for coordinates that do not correspond to active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates of shape ``(N, 3)`` with
                integer dtype.
            cumulative (bool): If ``True``, return cumulative indices. For a
                single grid this is equivalent to the default.

        Returns:
            indices (torch.Tensor): Linear indices of shape ``(N,)``.
        """
        from . import functional

        return functional.ijk_to_index_single(self, ijk, cumulative)

    def ijk_to_inv_index(self, ijk: torch.Tensor, cumulative: bool = False) -> torch.Tensor:
        """Get inverse permutation of :meth:`ijk_to_index`.

        For each voxel in the grid, return the index in the input ``ijk`` tensor
        that maps to it, or ``-1`` if no such coordinate exists.

        Args:
            ijk (torch.Tensor): Voxel coordinates of shape ``(N, 3)`` with
                integer dtype.
            cumulative (bool): If ``True``, return cumulative indices.

        Returns:
            inv_map (torch.Tensor): Inverse permutation of shape ``(N,)``.
        """
        from . import functional

        return functional.ijk_to_inv_index_single(self, ijk, cumulative)

    def points_in_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Check if world-space points are located within active voxels.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.

        Returns:
            mask (torch.Tensor): Boolean mask of shape ``(N,)``.
        """
        from . import functional

        return functional.points_in_grid_single(self, points)

    def coords_in_grid(self, ijk: torch.Tensor) -> torch.Tensor:
        """Check which voxel-space coordinates correspond to active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates of shape ``(N, 3)`` with
                integer dtype.

        Returns:
            mask (torch.Tensor): Boolean mask of shape ``(N,)``.
        """
        from . import functional

        return functional.coords_in_grid_single(self, ijk)

    def cubes_in_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """Test whether cubes are fully contained within active voxels.

        Args:
            cube_centers (torch.Tensor): World-space centers of shape ``(N, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube
                bounds, broadcastable to shape ``(3,)``.
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube
                bounds, broadcastable to shape ``(3,)``.

        Returns:
            mask (torch.Tensor): Boolean mask of shape ``(N,)``.
        """
        from . import functional

        return functional.cubes_in_grid_single(self, cube_centers, cube_min, cube_max)

    def cubes_intersect_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """Test whether cubes intersect any active voxels.

        Args:
            cube_centers (torch.Tensor): World-space centers of shape ``(N, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center.
            cube_max (NumericMaxRank1): Maximum offsets from center.

        Returns:
            mask (torch.Tensor): Boolean mask of shape ``(N,)``.
        """
        from . import functional

        return functional.cubes_intersect_grid_single(self, cube_centers, cube_min, cube_max)

    def neighbor_indexes(self, ijk: torch.Tensor, extent: int, bitshift: int = 0) -> torch.Tensor:
        """Get linear indices of neighboring voxels in an N-ring neighborhood.

        Args:
            ijk (torch.Tensor): Voxel coordinates of shape ``(N, 3)`` with
                integer dtype.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): Bit shift applied to input coordinates before
                querying. Default is ``0``.

        Returns:
            neighbor_indexes (torch.Tensor): Shape ``(N, K)`` where ``K`` is the
                number of neighbors per voxel. ``-1`` for inactive neighbors.
        """
        from . import functional

        return functional.neighbor_indexes_single(self, ijk, extent, bitshift)

    # ============================================================
    #                  Topology (return Grid)
    # ============================================================

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> Grid:
        """Return a coarsened version of this :class:`Grid`.

        Args:
            coarsening_factor (NumericMaxRank1): Factor by which to coarsen,
                broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            coarsened_grid (Grid): A new coarsened :class:`Grid`.
        """
        from . import functional

        return functional.coarsened_grid_single(self, coarsening_factor)

    def refined_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: torch.Tensor | None = None,
    ) -> Grid:
        """Return a refined (subdivided) version of this :class:`Grid`.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to refine,
                broadcastable to shape ``(3,)``, integer dtype.
            mask (torch.Tensor | None): Boolean mask of shape
                ``(num_voxels,)`` indicating which voxels to refine. If
                ``None``, all voxels are refined.

        Returns:
            refined_grid (Grid): A new refined :class:`Grid`.
        """
        from . import functional

        return functional.refined_grid_single(self, subdiv_factor, mask)

    def dual_grid(self, exclude_border: bool = False) -> Grid:
        """Return the dual grid whose voxel centers correspond to corners of
        this :class:`Grid`.

        Args:
            exclude_border (bool): If ``True``, exclude border voxels that
                extend beyond the primal grid bounds.

        Returns:
            dual_grid (Grid): A new :class:`Grid` representing the dual grid.
        """
        from . import functional

        return functional.dual_grid_single(self, exclude_border)

    def dilated_grid(self, dilation: int) -> Grid:
        """Return a dilated version of this :class:`Grid`.

        Args:
            dilation (int): Dilation radius in voxels.

        Returns:
            dilated_grid (Grid): A new :class:`Grid` with dilated active regions.
        """
        from . import functional

        return functional.dilated_grid_single(self, dilation)

    def merged_grid(self, other: Grid) -> Grid:
        """Return the union of this :class:`Grid` with another.

        Args:
            other (Grid): The other :class:`Grid` to merge with.

        Returns:
            merged_grid (Grid): A new :class:`Grid` containing the union of
                active voxels from both grids.
        """
        from . import functional

        return functional.merged_grid_single(self, other)

    def pruned_grid(self, mask: torch.Tensor) -> Grid:
        """Return a pruned :class:`Grid` keeping only voxels where ``mask`` is ``True``.

        Args:
            mask (torch.Tensor): Boolean mask of shape ``(num_voxels,)``.

        Returns:
            pruned_grid (Grid): A new :class:`Grid` with pruned voxels.
        """
        from . import functional

        return functional.pruned_grid_single(self, mask)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank1,
        ijk_max: NumericMaxRank1,
    ) -> Grid:
        """Return a :class:`Grid` clipped to the region ``[ijk_min, ijk_max]``.

        Args:
            ijk_min (NumericMaxRank1): Minimum voxel bounds, broadcastable to
                shape ``(3,)``, integer dtype.
            ijk_max (NumericMaxRank1): Maximum voxel bounds, broadcastable to
                shape ``(3,)``, integer dtype.

        Returns:
            clipped_grid (Grid): A new :class:`Grid` containing only voxels
                within the specified bounds.
        """
        from . import functional

        return functional.clipped_grid_single(self, ijk_min, ijk_max)

    def clip(
        self,
        features: torch.Tensor,
        ijk_min: NumericMaxRank1,
        ijk_max: NumericMaxRank1,
    ) -> tuple[torch.Tensor, Grid]:
        """Clip this :class:`Grid` and its features to the region ``[ijk_min, ijk_max]``.

        Args:
            features (torch.Tensor): Voxel features of shape
                ``(num_voxels, channels*)``.
            ijk_min (NumericMaxRank1): Minimum voxel bounds.
            ijk_max (NumericMaxRank1): Maximum voxel bounds.

        Returns:
            clipped_features (torch.Tensor): Clipped features.
            clipped_grid (Grid): A new :class:`Grid` with only voxels in bounds.
        """
        from . import functional

        return functional.clip_single(self, features, ijk_min, ijk_max)

    def contiguous(self) -> Grid:
        """Return a contiguous copy of this :class:`Grid`.

        Returns:
            grid (Grid): A contiguous copy of this :class:`Grid`.
        """
        from . import functional

        return functional.contiguous_single(self)

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> Grid:
        """Return the output :class:`Grid` for a convolution with the given kernel.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel,
                broadcastable to shape ``(3,)``, integer dtype.
            stride (NumericMaxRank1): Convolution stride, broadcastable to
                shape ``(3,)``, integer dtype.

        Returns:
            conv_grid (Grid): A new :class:`Grid` representing the convolution
                output topology.
        """
        from . import functional

        return functional.conv_grid_single(self, kernel_size, stride)

    def conv_transpose_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> Grid:
        """Return the output :class:`Grid` for a transposed convolution.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel,
                broadcastable to shape ``(3,)``, integer dtype.
            stride (NumericMaxRank1): Convolution stride, broadcastable to
                shape ``(3,)``, integer dtype.

        Returns:
            conv_transpose_grid (Grid): A new :class:`Grid` representing the
                transposed convolution output topology.
        """
        from . import functional

        return functional.conv_transpose_grid_single(self, kernel_size, stride)

    # ============================================================
    #                  Sampling / Splatting
    # ============================================================

    def sample_nearest(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """Sample voxel data at world-space points using nearest-neighbor lookup.

        For each query point the 8 nearest voxel centers are checked and the value of the closest active one is returned.
        Points where none of the 8 surrounding voxel centers are active return zero.

        .. note:: Supports backpropagation w.r.t. ``voxel_data``.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.
            voxel_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.

        Returns:
            sampled_data (torch.Tensor): Shape ``(N, channels*)``.

        .. seealso:: :meth:`GridBatch.sample_nearest`, :meth:`sample_trilinear`
        """
        from . import functional

        return functional.sample_nearest_single(self, points, voxel_data)

    def sample_trilinear(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """Sample voxel data at world-space points using trilinear interpolation.

        .. note:: Supports backpropagation. Samples outside the grid return zero.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.
            voxel_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Shape ``(N, channels*)``.
        """
        from . import functional

        return functional.sample_trilinear_single(self, points, voxel_data)

    def sample_trilinear_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample voxel data using trilinear interpolation and return spatial gradients.

        .. note:: Supports backpropagation. Samples outside the grid return zero.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.
            voxel_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Shape ``(N, channels*)``.
            gradients (torch.Tensor): Spatial gradients of shape
                ``(N, 3, channels*)``.
        """
        from . import functional

        return functional.sample_trilinear_with_grad_single(self, points, voxel_data)

    def sample_bezier(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """Sample voxel data at world-space points using Bezier interpolation.

        .. note:: Supports backpropagation. Samples outside the grid return zero.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.
            voxel_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Shape ``(N, channels*)``.
        """
        from . import functional

        return functional.sample_bezier_single(self, points, voxel_data)

    def sample_bezier_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample voxel data using Bezier interpolation and return spatial gradients.

        .. note:: Supports backpropagation. Samples outside the grid return zero.

        Args:
            points (torch.Tensor): World-space points of shape ``(N, 3)``.
            voxel_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Shape ``(N, channels*)``.
            gradients (torch.Tensor): Spatial gradients of shape
                ``(N, 3, channels*)``.
        """
        from . import functional

        return functional.sample_bezier_with_grad_single(self, points, voxel_data)

    def splat_trilinear(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """Splat point data into voxels using trilinear interpolation.

        Each point distributes its data to surrounding voxels using trilinear
        interpolation weights.

        .. note:: Supports backpropagation.

        Args:
            points (torch.Tensor): World-space point positions of shape
                ``(N, 3)``.
            points_data (torch.Tensor): Data to splat of shape
                ``(N, channels*)``.

        Returns:
            splatted_features (torch.Tensor): Accumulated features of shape
                ``(num_voxels, channels*)``.
        """
        from . import functional

        return functional.splat_trilinear_single(self, points, points_data)

    def splat_bezier(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """Splat point data into voxels using Bezier interpolation.

        Each point distributes its data to surrounding voxels using cubic Bezier
        interpolation weights.

        .. note:: Supports backpropagation.

        Args:
            points (torch.Tensor): World-space point positions of shape
                ``(N, 3)``.
            points_data (torch.Tensor): Data to splat of shape
                ``(N, channels*)``.

        Returns:
            splatted_features (torch.Tensor): Accumulated features of shape
                ``(num_voxels, channels*)``.
        """
        from . import functional

        return functional.splat_bezier_single(self, points, points_data)

    # ============================================================
    #                        Pooling
    # ============================================================

    def avg_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: torch.Tensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: Grid | None = None,
    ) -> tuple[torch.Tensor, Grid]:
        """Apply average pooling to voxel data.

        .. note:: Supports backpropagation.

        Args:
            pool_factor (NumericMaxRank1): Downsample factor, broadcastable to
                shape ``(3,)``, integer dtype.
            data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels)``.
            stride (NumericMaxRank1): Pooling stride. If ``0``, equals
                ``pool_factor``.
            coarse_grid (Grid | None): Pre-allocated coarse :class:`Grid`. If
                ``None``, a new one is created.

        Returns:
            pooled_data (torch.Tensor): Pooled data of shape
                ``(coarse_num_voxels, channels)``.
            coarse_grid (Grid): Coarse :class:`Grid` after pooling.
        """
        from . import functional

        return functional.avg_pool_single(self, pool_factor, data, stride, coarse_grid)

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: torch.Tensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: Grid | None = None,
    ) -> tuple[torch.Tensor, Grid]:
        """Apply max pooling to voxel data.

        .. note:: Supports backpropagation.

        Args:
            pool_factor (NumericMaxRank1): Downsample factor, broadcastable to
                shape ``(3,)``, integer dtype.
            data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels)``.
            stride (NumericMaxRank1): Pooling stride. If ``0``, equals
                ``pool_factor``.
            coarse_grid (Grid | None): Pre-allocated coarse :class:`Grid`. If
                ``None``, a new one is created.

        Returns:
            pooled_data (torch.Tensor): Pooled data of shape
                ``(coarse_num_voxels, channels)``.
            coarse_grid (Grid): Coarse :class:`Grid` after pooling.
        """
        from . import functional

        return functional.max_pool_single(self, pool_factor, data, stride, coarse_grid)

    def refine(
        self,
        subdiv_factor: NumericMaxRank1,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        refined: Grid | None = None,
    ) -> tuple[torch.Tensor, Grid]:
        """Refine (upsample) voxel data into a higher-resolution :class:`Grid`.

        For each voxel ``(i, j, k)`` in this grid, copies its data to the
        subdivided voxels in the fine grid.

        .. note:: Supports backpropagation.

        Args:
            subdiv_factor (NumericMaxRank1): Refinement factor, broadcastable
                to shape ``(3,)``, integer dtype.
            data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels)``.
            mask (torch.Tensor | None): Boolean mask of shape
                ``(num_voxels,)`` indicating which voxels to refine.
            refined (Grid | None): Pre-allocated fine :class:`Grid`. If
                ``None``, a new one is created.

        Returns:
            refined_data (torch.Tensor): Refined data for the fine grid.
            fine_grid (Grid): The fine :class:`Grid`.
        """
        from . import functional

        return functional.refine_single(self, subdiv_factor, data, mask, refined)

    # ============================================================
    #                      Ray Operations
    # ============================================================

    def voxels_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """Enumerate voxels intersected by rays using DDA traversal.

        Args:
            ray_origins (torch.Tensor): Ray origins of shape ``(N, 3)``.
            ray_directions (torch.Tensor): Ray directions of shape ``(N, 3)``.
            max_voxels (int): Maximum voxels to return per ray.
            eps (float): Epsilon for numerical stability.
            return_ijk (bool): If ``True``, return voxel ``(i,j,k)`` coordinates
                instead of linear indices.

        Returns:
            voxels (JaggedTensor): Per-ray voxel indices (or coordinates if
                ``return_ijk``).
            distances (JaggedTensor): Per-ray entry/exit distances of shape
                ``(num_rays, num_voxels_per_ray, 2)``.
        """
        from . import functional

        return functional.voxels_along_rays_single(self, ray_origins, ray_directions, max_voxels, eps, return_ijk)

    def segments_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_segments: int,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """Return continuous segments of ray traversal through this :class:`Grid`.

        Each segment is a ``(t_start, t_end)`` pair of distances along the ray.

        Args:
            ray_origins (torch.Tensor): Ray origins of shape ``(N, 3)``.
            ray_directions (torch.Tensor): Ray directions of shape ``(N, 3)``.
            max_segments (int): Maximum segments to return per ray.
            eps (float): Epsilon for numerical stability.

        Returns:
            segments (JaggedTensor): Per-ray segments with element shape ``(2,)``.
        """
        from . import functional

        return functional.segments_along_rays_single(self, ray_origins, ray_directions, max_segments, eps)

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
        """Generate uniformly spaced samples along rays intersecting this :class:`Grid`.

        Args:
            ray_origins (torch.Tensor): Ray origins of shape ``(N, 3)``.
            ray_directions (torch.Tensor): Ray directions of shape ``(N, 3)``.
            t_min (torch.Tensor): Minimum distances of shape ``(N,)``.
            t_max (torch.Tensor): Maximum distances of shape ``(N,)``.
            step_size (float): Distance between samples.
            cone_angle (float): Cone angle for adaptive sampling (radians).
            include_end_segments (bool): Include partial segments at ray ends.
            return_midpoints (bool): Return midpoints instead of start points.
            eps (float): Epsilon for numerical stability.

        Returns:
            samples (JaggedTensor): Per-ray sample distances.
        """
        from . import functional

        return functional.uniform_ray_samples_single(
            self,
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

    def ray_implicit_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        grid_scalars: torch.Tensor,
        eps: float = 0.0,
    ) -> torch.Tensor:
        """Find ray intersections with an implicit surface defined by voxel scalars.

        The implicit surface is defined by the zero level-set of
        ``grid_scalars``.

        Args:
            ray_origins (torch.Tensor): Ray origins of shape ``(N, 3)``.
            ray_directions (torch.Tensor): Ray directions of shape ``(N, 3)``.
            grid_scalars (torch.Tensor): Scalar field of shape
                ``(num_voxels, 1)``.
            eps (float): Epsilon for numerical stability.

        Returns:
            intersection_distances (torch.Tensor): Distance along each ray,
                or ``-1`` if no intersection. Shape ``(N,)``.
        """
        from . import functional

        return functional.ray_implicit_intersection_single(self, ray_origins, ray_directions, grid_scalars, eps)

    def rays_intersect_voxels(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        eps: float = 0.0,
    ) -> torch.Tensor:
        """Check whether rays hit any voxels in this :class:`Grid`.

        Args:
            ray_origins (torch.Tensor): Ray origins of shape ``(N, 3)``.
            ray_directions (torch.Tensor): Ray directions of shape ``(N, 3)``.
            eps (float): Epsilon for numerical stability.

        Returns:
            hit_mask (torch.Tensor): Boolean tensor of shape ``(N,)``.
        """
        from . import functional

        return functional.rays_intersect_voxels_single(self, ray_origins, ray_directions, eps)

    # ============================================================
    #                  Dense Conversion
    # ============================================================

    def inject_from_dense_cminor(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """Inject values from a dense tensor (XYZC order) into sparse voxel data.

        ``dense_data`` has shape ``(1, X, Y, Z, C*)``.

        .. note:: Supports backpropagation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from, shape
                ``(1, X, Y, Z, C*)``.
            dense_origin (NumericMaxRank1): Origin of the dense tensor in voxel
                space, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            sparse_data (torch.Tensor): Shape ``(num_voxels, channels*)``.
        """
        from . import functional

        return functional.inject_from_dense_cminor_single(self, dense_data, dense_origin)

    def inject_from_dense_cmajor(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """Inject values from a dense tensor (CXYZ order) into sparse voxel data.

        ``dense_data`` has shape ``(1, C*, X, Y, Z)``.

        .. note:: Supports backpropagation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from, shape
                ``(1, C*, X, Y, Z)``.
            dense_origin (NumericMaxRank1): Origin of the dense tensor in voxel
                space, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            sparse_data (torch.Tensor): Shape ``(num_voxels, channels*)``.
        """
        from . import functional

        return functional.inject_from_dense_cmajor_single(self, dense_data, dense_origin)

    def inject_to_dense_cminor(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """Write sparse voxel data into a dense tensor in XYZC order.

        .. note:: Supports backpropagation.

        Args:
            sparse_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.
            min_coord (NumericMaxRank1 | None): Minimum voxel coordinate for the
                output dense tensor. If ``None``, uses the grid's bounding box
                minimum.
            grid_size (NumericMaxRank1 | None): Size of the output dense tensor.
                If ``None``, computed to fit all active voxels.

        Returns:
            dense_data (torch.Tensor): Dense tensor of shape
                ``(X, Y, Z, channels*)``.
        """
        from . import functional

        return functional.inject_to_dense_cminor_single(self, sparse_data, min_coord, grid_size)

    def inject_to_dense_cmajor(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """Write sparse voxel data into a dense tensor in CXYZ order.

        .. note:: Supports backpropagation.

        Args:
            sparse_data (torch.Tensor): Voxel data of shape
                ``(num_voxels, channels*)``.
            min_coord (NumericMaxRank1 | None): Minimum voxel coordinate for the
                output dense tensor. If ``None``, uses the grid's bounding box
                minimum.
            grid_size (NumericMaxRank1 | None): Size of the output dense tensor.
                If ``None``, computed to fit all active voxels.

        Returns:
            dense_data (torch.Tensor): Dense tensor of shape
                ``(channels*, X, Y, Z)``.
        """
        from . import functional

        return functional.inject_to_dense_cmajor_single(self, sparse_data, min_coord, grid_size)

    # ============================================================
    #                      Injection
    # ============================================================

    def inject_from(
        self,
        src_grid: Grid,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """Inject data from ``src_grid`` into this :class:`Grid`.

        Copies sidecar data for voxels shared between the two grids.
        The copy occurs in voxel space; the voxel-to-world transform is not applied.

        .. note:: Supports backpropagation.

        Args:
            src_grid (Grid): Source :class:`Grid` to inject data from.
            src (torch.Tensor): Source data of shape ``(src_grid.num_voxels, *)``.
            dst (torch.Tensor | None): Optional destination data modified
                in-place. Shape ``(self.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Fill value for voxels without
                source data. Used only if ``dst`` is ``None``.

        Returns:
            dst (torch.Tensor): The destination data after injection.
        """
        from . import functional

        return functional.inject_single(self, src_grid, src, dst, default_value)

    def inject_to(
        self,
        dst_grid: Grid,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """Inject data from this :class:`Grid` into ``dst_grid``.

        Copies sidecar data for voxels shared between the two grids.
        The copy occurs in voxel space; the voxel-to-world transform is not applied.

        .. note:: Supports backpropagation.

        Args:
            dst_grid (Grid): Destination :class:`Grid` to inject data into.
            src (torch.Tensor): Source data of shape ``(self.num_voxels, *)``.
            dst (torch.Tensor | None): Optional destination data modified
                in-place. Shape ``(dst_grid.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Fill value for voxels without
                source data. Used only if ``dst`` is ``None``.

        Returns:
            dst (torch.Tensor): The destination data after injection.
        """
        from . import functional

        return functional.inject_single(dst_grid, self, src, dst, default_value)

    def inject_from_ijk(
        self,
        src_ijk: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """Inject data from source voxel coordinates into this :class:`Grid`.

        .. note:: Supports backpropagation.

        Args:
            src_ijk (torch.Tensor): Source voxel coordinates of shape
                ``(num_src_voxels, 3)`` with integer dtype.
            src (torch.Tensor): Source data of shape ``(num_src_voxels, *)``.
            dst (torch.Tensor | None): Optional destination data modified
                in-place. Shape ``(self.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Fill value for voxels without
                source data. Used only if ``dst`` is ``None``.

        Returns:
            dst (torch.Tensor): The destination data after injection.
        """
        from . import functional

        return functional.inject_from_ijk_single(self, src_ijk, src, dst, default_value)

    # ============================================================
    #                    Meshing / TSDF
    # ============================================================

    def marching_cubes(
        self, field: torch.Tensor, level: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract isosurface mesh using marching cubes.

        Args:
            field (torch.Tensor): Scalar field of shape ``(num_voxels, 1)``.
            level (float): Isovalue at which to extract the surface.

        Returns:
            vertices (torch.Tensor): Vertex positions of shape
                ``(num_vertices, 3)``.
            faces (torch.Tensor): Triangle face indices of shape
                ``(num_faces, 3)``.
            normals (torch.Tensor): Vertex normals of shape
                ``(num_vertices, 3)``.
        """
        from . import functional

        return functional.marching_cubes_single(self, field, level)

    def integrate_tsdf(
        self,
        truncation_distance: float,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        tsdf: torch.Tensor,
        weights: torch.Tensor,
        depth_images: torch.Tensor,
        weight_images: torch.Tensor | None = None,
    ) -> tuple[Grid, torch.Tensor, torch.Tensor]:
        """Integrate depth images into a TSDF volume.

        Updates TSDF values and weights by integrating new depth observations
        from camera viewpoints.

        Args:
            truncation_distance (float): Maximum TSDF truncation distance.
            projection_matrices (torch.Tensor): Camera projection matrices.
            cam_to_world_matrices (torch.Tensor): Camera-to-world transforms.
            tsdf (torch.Tensor): Current TSDF values of shape
                ``(num_voxels, 1)``.
            weights (torch.Tensor): Current integration weights of shape
                ``(num_voxels, 1)``.
            depth_images (torch.Tensor): Depth images.
            weight_images (torch.Tensor | None): Optional weight images.

        Returns:
            updated_grid (Grid): Updated :class:`Grid`.
            updated_tsdf (torch.Tensor): Updated TSDF values.
            updated_weights (torch.Tensor): Updated weights.
        """
        from . import functional

        return functional.integrate_tsdf_single(
            self,
            truncation_distance,
            projection_matrices,
            cam_to_world_matrices,
            tsdf,
            weights,
            depth_images,
            weight_images,
        )

    def integrate_tsdf_with_features(
        self,
        truncation_distance: float,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        tsdf: torch.Tensor,
        features: torch.Tensor,
        weights: torch.Tensor,
        depth_images: torch.Tensor,
        feature_images: torch.Tensor,
        weight_images: torch.Tensor | None = None,
    ) -> tuple[Grid, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate depth and feature images into a TSDF volume.

        Similar to :meth:`integrate_tsdf` but also integrates feature
        observations (e.g., color).

        Args:
            truncation_distance (float): Maximum TSDF truncation distance.
            projection_matrices (torch.Tensor): Camera projection matrices.
            cam_to_world_matrices (torch.Tensor): Camera-to-world transforms.
            tsdf (torch.Tensor): Current TSDF values of shape
                ``(num_voxels, 1)``.
            features (torch.Tensor): Current features of shape
                ``(num_voxels, feature_dim)``.
            weights (torch.Tensor): Current integration weights of shape
                ``(num_voxels, 1)``.
            depth_images (torch.Tensor): Depth images.
            feature_images (torch.Tensor): Feature images.
            weight_images (torch.Tensor | None): Optional weight images.

        Returns:
            updated_grid (Grid): Updated :class:`Grid`.
            updated_tsdf (torch.Tensor): Updated TSDF values.
            updated_features (torch.Tensor): Updated features.
            updated_weights (torch.Tensor): Updated weights.
        """
        from . import functional

        return functional.integrate_tsdf_with_features_single(
            self,
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

    # ============================================================
    #                        Device
    # ============================================================

    def cpu(self) -> Grid:
        """Return a copy of this :class:`Grid` on the CPU.

        Returns:
            grid (Grid): A :class:`Grid` on CPU, or ``self`` if already on CPU.
        """
        return self.to("cpu")

    def cuda(self) -> Grid:
        """Return a copy of this :class:`Grid` on CUDA.

        Returns:
            grid (Grid): A :class:`Grid` on CUDA, or ``self`` if already on CUDA.
        """
        return self.to("cuda")

    def to(self, target: str | torch.device | torch.Tensor | JaggedTensor | Grid | GridBatch) -> Grid:
        """Move this :class:`Grid` to the target device.

        Args:
            target: Target device specification. Can be a string, a
                :class:`torch.device`, a :class:`torch.Tensor`, a
                :class:`~fvdb.JaggedTensor`, a :class:`~fvdb.Grid`, or a
                :class:`~fvdb.GridBatch`.

        Returns:
            grid (Grid): A :class:`Grid` on the target device.
        """
        from . import _parse_device_string, functional
        from .grid_batch import GridBatch as GB

        if isinstance(target, str):
            device = _parse_device_string(target)
        elif isinstance(target, torch.device):
            device = target
        elif isinstance(target, torch.Tensor):
            device = target.device
        elif isinstance(target, JaggedTensor):
            device = target.jdata.device
        elif isinstance(target, (Grid, GB)):
            device = target.device
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")
        return functional.clone_grid_single(self, device)

    # ============================================================
    #                          I/O
    # ============================================================

    def save_nanovdb(
        self,
        path: str | pathlib.Path,
        data: torch.Tensor | None = None,
        name: str | None = None,
        compressed: bool = False,
        verbose: bool = False,
    ) -> None:
        """Save this :class:`Grid` and optional voxel data to a ``.nvdb`` file.

        Args:
            path (str | pathlib.Path): File path (should have ``.nvdb`` extension).
            data (torch.Tensor | None): Voxel data of shape
                ``(num_voxels, channels)``. If ``None``, only the grid structure
                is saved.
            name (str | None): Optional name for the grid.
            compressed (bool): Whether to use Blosc compression.
            verbose (bool): Whether to print save information.
        """
        from . import functional

        functional.save_nanovdb_single(
            self,
            str(path) if isinstance(path, pathlib.Path) else path,
            data,
            name,
            compressed,
            verbose,
        )

    # ============================================================
    #                      Edge Network
    # ============================================================

    def edge_network(self, return_voxel_coordinates: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the edge network of this :class:`Grid`.

        Args:
            return_voxel_coordinates (bool): If ``True``, return voxel
                coordinates instead of linear indices.

        Returns:
            edge_a (torch.Tensor): One endpoint of each edge.
            edge_b (torch.Tensor): Other endpoint of each edge.
        """
        from . import functional

        return functional.edge_network_single(self, return_voxel_coordinates)

    # ============================================================
    #                        Utility
    # ============================================================

    def is_same(self, other: Grid) -> bool:
        """Check if two :class:`Grid` objects share the same underlying data.

        Args:
            other (Grid): The other :class:`Grid` to compare with.

        Returns:
            is_same (bool): ``True`` if the grids share underlying data.
        """
        return self.data.is_same(other.data)

    def has_same_address_and_grid_count(self, other: Any) -> bool:
        """Check if this :class:`Grid` has the same underlying data identity
        as another object.

        Args:
            other: Object to compare with.

        Returns:
            result (bool): ``True`` if same address and grid count.
        """
        if isinstance(other, Grid):
            return id(self.data) == id(other.data)
        elif isinstance(other, GridBatchData):
            return id(self.data) == id(other) and other.grid_count == 1
        else:
            return False

    # ============================================================
    #                      Conversion
    # ============================================================

    def to_gridbatch(self) -> GridBatch:
        """Convert this :class:`Grid` to a :class:`~fvdb.GridBatch` with
        ``grid_count == 1``.

        Returns:
            grid_batch (GridBatch): A :class:`~fvdb.GridBatch` wrapping the same
                underlying data.
        """
        from .grid_batch import GridBatch

        return GridBatch(data=self.data)

    # ============================================================
    #                   Special Methods
    # ============================================================

    def __repr__(self) -> str:
        return (
            f"Grid(num_voxels={self.num_voxels}, "
            f"voxel_size={self.voxel_size.tolist()}, "
            f"origin={self.origin.tolist()}, "
            f"device={self.device})"
        )
