# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Single sparse grid data structure and operations for FVDB.

This module provides the core Grid class for managing a single sparse voxel grid:

Classes:
- Grid: A single sparse voxel grid with support for efficient operations

Class-methods for creating Grid objects from various sources:
- :meth:`Grid.from_dense()` for dense data
- :meth:`Grid.from_dense_axis_aligned_bounds()` for dense defined by axis-aligned bounds
- :meth:`Grid.from_grid_batch()` for a single grid from a grid batch
- :meth:`Grid.from_ijk()` for voxel coordinates
- :meth:`Grid.from_mesh()` for triangle meshes
- :meth:`Grid.from_nearest_voxels_to_points()` for nearest voxel mapping
- :meth:`Grid.from_points()` for point clouds
- :meth:`Grid.from_zero_voxels()` for a single grid with zero voxels

Module-level functions for loading and saving grids:
- from_nanovdb/save_nanovdb: Load and save grids to/from .nvdb files

Grid supports operations like convolution, pooling, interpolation, ray casting,
mesh extraction, and coordinate transformations on sparse voxel data.
"""

import pathlib
from typing import TYPE_CHECKING, Any, cast, overload

import torch

from . import _parse_device_string
from ._Cpp import GridBatch as GridBatchCpp
from .jagged_tensor import JaggedTensor
from .types import (
    DeviceIdentifier,
    NumericMaxRank1,
    ValueConstraint,
    resolve_device,
    to_Vec3f,
    to_Vec3fBatch,
    to_Vec3fBroadcastable,
    to_Vec3i,
    to_Vec3iBroadcastable,
)

if TYPE_CHECKING:
    from .grid_batch import GridBatch


class Grid:
    """
    A single sparse voxel grid with support for efficient operations.

    A :class:`Grid` represents a single sparse 3D voxel grid that can be processed
    efficiently on a GPU. The class provides methods for common operations like
    sampling, convolution, pooling, dilation, union, etc. It also provides more advanced features
    such as marching cubes, TSDF fusion, and fast ray marching.

    A :class:`Grid` does not store
    data itself, but rather the structure (or topology) of the sparse voxel grid. Voxel data
    (e.g., features, colors, densities) are stored separately as :class:`torch.Tensor` associated with
    the grid. This separation allows for flexibility in the type and number of channels of data with
    which a grid can be used to index into. This also allows multiple grids to share the same data
    storage if desired.

    When using a :class:`Grid`'s voxel coordinates, there are three important coordinate systems to be aware of:

    - **World Space**: The continuous 3D coordinate system in which the grid exists.
    - **Voxel Space**: The discrete voxel index system, where each voxel is identified by its integer indices (i, j, k).
    - **Index Space**: The linear indexing of active voxels in the grid's internal storage.

    At its core, a :class:`Grid` uses a very fast mapping from voxel space into index space to perform operations on a :class:`torch.Tensor` of
    data associated with the grid. This mapping allows for efficient access and manipulation of voxel data. For example:

    .. code-block:: python

        voxel_coords = torch.tensor([[8, 7, 6], [1, 2, 3], [4, 5, 6]], device="cuda")  # Voxel space coordinates

        # Create a Grid containing the voxels (8, 7, 6), (1, 2, 3), and (4, 5, 6) such that the voxels
        # have a world space size of 1x1x1, and where the [0, 0, 0] voxel in voxel space is at world space origin (0, 0, 0).
        grid = Grid.from_ijk(voxel_coords, voxel_size=1.0, origin=0.0, device="cuda")

        # Create some data associated with the grid - here we have 3 voxels and 2 channels per voxel
        voxel_data = torch.randn(grid.num_voxels, 2, device="cuda")  # Index space data

        # Map voxel space coordinates to index space
        indices = grid.ijk_to_index(voxel_coords)  # Shape: (3,)

        # Access the data for the specified voxel coordinates
        selected_data = voxel_data[indices]  # Shape: (3, 2)


    .. note::

        The grid is stored in a sparse format using `NanoVDB <https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb>`_
        where only active (non-empty) voxels are allocated, making it extremely memory efficient for representing large volumes with sparse
        occupancy.

    .. note::

        A :class:`Grid` cannot be a nonexistent (grid_count==0) grid, for that you'd need a
        :class:`GridBatch` with batch_size=0. However, a :class:`Grid` can have zero voxels.

    .. note::

        The :class:`Grid` constructor is for internal use only, To create a :class:`Grid` with actual content, use the classmethods:

        - :meth:`from_dense()` for dense data
        - :meth:`from_dense_axis_aligned_bounds()` for dense defined by axis-aligned bounds
        - :meth:`from_grid_batch()` for a single grid from a grid batch
        - :meth:`from_ijk()` for voxel coordinates
        - :meth:`from_mesh()` for triangle meshes
        - :meth:`from_nearest_voxels_to_points()` for nearest voxel mapping
        - :meth:`from_points()` for point clouds
        - :meth:`from_zero_voxels()` for a single grid with zero voxels


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
        A dense grid has a voxel for every coordinate in an axis-aligned box.

        The dense grid is defined by:

        - dense_dims: the size of the dense grid (shape ``[3,] = [W, H, D]``)
        - ijk_min: the minimum voxel index for the grid (shape ``[3,] = [i_min, j_min, k_min]``)
        - voxel_size: the world-space size of each voxel (shape ``[3,] = [sx, sy, sz]``)
        - origin: the world-space coordinate of the center of the ``[0,0,0]`` voxel of the grid (shape ``[3,] = [x0, y0, z0]``)
        - mask: indicates which voxels are "active" in the resulting grid.

        Args:
            dense_dims (NumericMaxRank1): Dimensions of the dense grid,
                broadcastable to shape ``(3,)``, integer dtype
            ijk_min (NumericMaxRank1): Minimum voxel index for the grid,
                broadcastable to shape ``(3,)``, integer dtype
            voxel_size (NumericMaxRank1): World space size of each voxel,
                broadcastable to shape ``(3,)``, floating dtype
            origin (NumericMaxRank1): World space coordinate of the center of the ``[0,0,0]`` voxel of the grid,
                broadcastable to shape ``(3,)``, floating dtype
            mask (torch.Tensor | None): Mask to apply to the grid, a :class:`torch.Tensor` with shape ``(W, H, D)`` and boolean dtype.
            device (DeviceIdentifier | None): Device to create the grid on.
                Defaults to ``None``, which inherits the device from
                ``mask``, or uses ``"cpu"`` if ``mask`` is ``None``.

        Returns:
            grid (Grid): A new :class:`Grid` object.
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
        """
        Create a dense grid defined by axis-aligned bounds in world space.

        The grid has voxels spanning ``dense_dims`` with the voxel size and origin set to fit
        within the specified axis-aligned bounding box defined by ``bounds_min`` and ``bounds_max``.

        If ``voxel_center`` is ``True``, the bounds correspond to the centers of the corner voxels.
        If ``voxel_center`` is ``False``, the bounds correspond to the outer edges of the corner voxels.

        Args:
            dense_dims (NumericMaxRank1): Dimensions of the dense grid, broadcastable to shape ``(3,)``, integer dtype
            bounds_min (NumericMaxRank1): Minimum world space bounds of the grid, broadcastable to shape ``(3,)``, floating dtype
            bounds_max (NumericMaxRank1): Maximum world space bounds of the grid, broadcastable to shape ``(3,)``, floating dtype
            voxel_center (bool): Whether the bounds correspond to voxel centers (``True``) or edges (``False``). Defaults to ``False``.
            device (DeviceIdentifier): Device to create the grid on. Defaults to ``"cpu"``.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
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
        Extract a :class:`Grid` from one grid in a :class:`GridBatch`. If ``index`` exceeds the number
        of grids in the batch (minus one), an error is raised.

        .. note::

            The resulting :class:`Grid` will share the same underlying data as the :class:`GridBatch`,
            but have different metadata. Thus, :meth:`is_contiguous()` will return ``False`` on the resulting
            :class:`Grid` if the :class:`GridBatch` contains multiple grids.

        Args:
            grid_batch (GridBatch): The :class:`GridBatch` to extract a :class:`Grid` from.
            index (int): The index of the :class:`Grid` to extract from the :class:`GridBatch`. Defaults to 0.

        Returns:
            grid (Grid): A new :class:`Grid` object matching the index-th grid in the :class:`GridBatch`.
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
        Create a grid from voxel coordinates. If multiple voxels map to the same coordinate,
        only one voxel will be created at that coordinate.

        Args:
            ijk (torch.Tensor): Voxel coordinates to populate. A :class:`torch.Tensor` with shape ``(num_voxels, 3)`` with integer coordinates.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to shape ``(3,)``, floating dtype
            origin (NumericMaxRank1): Origin of the grid. *i.e.* the world-space position of the center of the ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid on. Defaults to None, which inherits the device of ``ijk``.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        resolved_device = resolve_device(device, inherit_from=ijk)

        jagged_ijk = JaggedTensor(ijk)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_ijk(jagged_ijk._impl, voxel_size, origin)
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
        Create a new :class:`Grid` by voxelizing the *surface* of a triangle mesh. *i.e* voxels that intersect
        the surface of the mesh will be contained in the resulting :class:`Grid`.

        .. note:: This method works well but will be made much faster and memory efficient in the next release.

        Args:
            mesh_vertices (torch.Tensor): Vertices of the mesh. A :class:`torch.Tensor` with shape ``(num_vertices, 3)``.
            mesh_faces (torch.Tensor): Faces of the mesh. A :class:`torch.Tensor` with shape ``(num_faces, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to shape ``(3,)``, floating dtype
            origin (NumericMaxRank1): Origin of the grid. *i.e.* the world-space position of the center of the ``[0,0,0]`` voxel, broadcastable to shape ``(3,)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid on. Defaults to ``None``, which inherits the device of ``mesh_vertices``.

        Returns:
            grid (Grid): A new :class:`Grid` object with voxels covering the surface of the input mesh.
        """
        resolved_device = resolve_device(device, inherit_from=mesh_vertices)

        jagged_mesh_vertices = JaggedTensor(mesh_vertices)
        jagged_mesh_faces = JaggedTensor(mesh_faces)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_mesh(jagged_mesh_vertices._impl, jagged_mesh_faces._impl, voxel_size, origin)
        return cls(impl=grid_impl)

    # Load and save functions
    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: pathlib.Path | str,
        *,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[Grid, torch.Tensor, str]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: pathlib.Path | str,
        *,
        index: int,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[Grid, torch.Tensor, str]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: pathlib.Path | str,
        *,
        name: str,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[Grid, torch.Tensor, str]": ...

    @classmethod
    def from_nanovdb(
        cls,
        path: pathlib.Path | str,
        *,
        index: int | None = None,
        name: str | None = None,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[Grid, torch.Tensor, str]":
        """
        Load a :class:`Grid` from a .nvdb file.

        Args:
            path (str): The path to the .nvdb file to load
            index (int | None): Optional single index to load from the file (mutually exclusive with other selectors)
            name (str | None): Optional single name to load from the file (mutually exclusive with other selectors)
            device (DeviceIdentifier): Which device to load the grid on
            verbose (bool): If set to true, print information about the loaded grid

        Returns:
            grid (Grid): The loaded :class:`Grid`.
            data (torch.Tensor): A :class:`torch.Tensor` containing the data associated with the grid, with shape ``(grid.num_voxels, channels*)``.
            name (str): The name of the loaded grid.
        """
        from ._Cpp import load as _load

        if isinstance(path, pathlib.Path):
            path = str(path)

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

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: torch.Tensor,
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "Grid":
        """
        Create a grid by adding the eight nearest voxels to every point in a point cloud.

        Args:
            points (torch.Tensor): Points to populate the grid from. A :class:`torch.Tensor` with shape ``(num_points, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to shape ``(3,)``, floating dtype
            origin (NumericMaxRank1): Origin of the grid, broadcastable to shape ``(3,)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid on. Defaults to ``None``, which inherits the device of ``points``.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        jagged_points = JaggedTensor(points)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_nearest_voxels_to_points(jagged_points._impl, voxel_size, origin)
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
            points (torch.Tensor): Points to populate the grid from. A :class:`torch.Tensor` with shape ``(num_points, 3)``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to shape ``(3,)``, floating dtype
            origin (NumericMaxRank1): Origin of the grid, broadcastable to shape ``(3,)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid on. Defaults to ``None``, which inherits the device of ``points``.

        Returns:
            grid (Grid): A new :class:`Grid` object.
        """
        resolved_device = resolve_device(device, inherit_from=points)

        jagged_points = JaggedTensor(points)
        voxel_size = to_Vec3fBroadcastable(voxel_size, value_constraint=ValueConstraint.POSITIVE)
        origin = to_Vec3f(origin)

        grid_impl = GridBatchCpp(device=resolved_device)
        grid_impl.set_from_points(jagged_points._impl, voxel_size, origin)
        return cls(impl=grid_impl)

    @classmethod
    def from_zero_voxels(
        cls,
        device: DeviceIdentifier = "cpu",
        voxel_size: NumericMaxRank1 = 1,
        origin: NumericMaxRank1 = 0,
    ) -> "Grid":
        """
        Create a new :class:`Grid` with zero voxels on a specific device.

        Args:
            device: The device to create the Grid on. Can be a string (e.g., "cuda", "cpu") or a :class:`torch.device` object. Defaults to ``"cpu"``.
            voxel_size (NumericMaxRank1): Size of each voxel, broadcastable to shape ``(3,)``, floating dtype. Defaults to ``1``.
            origin (NumericMaxRank1): Origin of the grid, broadcastable to shape ``(3,)``, floating dtype. Defaults to ``0``.

        Returns:
            grid (Grid): A new :class:`Grid` object with zero voxels.

        Examples:

        .. code-block:: python

            grid = Grid.from_zero_voxels("cuda", 1, 0)  # string
            grid = Grid.from_zero_voxels(torch.device("cuda:0"), 1, 0)  # device directly
            grid = Grid.from_zero_voxels(voxel_size=1, origin=0)  # defaults to CPU

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
        Apply average pooling to the given data associated with this :class:`Grid` returned as data associated
        with the given ``coarse_grid`` or a newly created coarse :class:`Grid`.

        Performs average pooling on the voxel data, reducing the resolution by the specified
        ``pool_factor``. Each output voxel contains the average of the corresponding input voxels
        within the pooling window. The pooling operation respects the sparse structure of this.
        :class:`Grid` and the given ``coarse_grid``.

        .. note::

            If you pass ``coarse_grid = None``, the returned coarse grid will have its
            voxel size multiplied by the ``pool_factor`` and its origin adjusted accordingly.

        .. note::

            This method supports backpropagation through the pooling operation.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid, broadcastable to shape ``(3,)``, integer dtype
            data (torch.Tensor): The voxel data to pool. A :class:`torch.Tensor` with shape ``(total_voxels, channels)``.
            stride (NumericMaxRank1): The stride to use when pooling. If ``0`` (default), broadcastable to shape ``(3,)``, integer dtype
            coarse_grid (Grid, optional): Pre-allocated coarse grid to use for output. If ``None``, a new :class:`Grid` is created.

        Returns:
            pooled_data (torch.Tensor): A tensor containing the pooled voxel data with shape ``(coarse_total_voxels, channels)``.
            coarse_grid (Grid): A :class:`Grid` object representing the coarse grid topology after pooling. Matches the provided ``coarse_grid`` if given.

        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.avg_pool(pool_factor, jagged_data._impl, stride, coarse_grid_impl)
        return result_data.jdata, Grid(impl=cast(GridBatchCpp, result_grid_impl))

    def clip(
        self, features: torch.Tensor, ijk_min: NumericMaxRank1, ijk_max: NumericMaxRank1
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Creates a new :class:`Grid` containing only the voxels that fall within the specified
        bounding box range ``[ijk_min, ijk_max]``, and returns the corresponding clipped features.

        .. note::

            This method supports backpropagation through the clipping operation.

        Args:
            features (torch.Tensor): The voxel features to clip. A :class:`torch.Tensor` with shape ``(total_voxels, channels)``.
            ijk_min (NumericMaxRank1): Minimum bounds in index space, broadcastable to shape ``(3,)``, integer dtype
            ijk_max (NumericMaxRank1): Maximum bounds in index space, broadcastable to shape ``(3,)``, integer dtype

        Returns:
            clipped_features (torch.Tensor): A tensor containing the clipped voxel features with shape ``(clipped_total_voxels, channels)``.
            clipped_grid (Grid): A new :class:`Grid` object containing only the voxels within the specified bounds.
        """
        jagged_features = JaggedTensor(features)
        ijk_min = to_Vec3iBroadcastable(ijk_min)
        ijk_max = to_Vec3iBroadcastable(ijk_max)

        result_features, result_grid_impl = self._impl.clip(jagged_features._impl, ijk_min, ijk_max)
        return result_features.jdata, Grid(impl=result_grid_impl)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank1,
        ijk_max: NumericMaxRank1,
    ) -> "Grid":
        """
        Return a new :class:`Grid` representing the clipped version of this grid.
        Each voxel ``[i, j, k]`` in the input grid is included in the output if it lies within ``ijk_min`` and ``ijk_max``.

        Args:
            ijk_min (NumericMaxRank1): Index space minimum bound of the clip region,
                broadcastable to shape ``(3,)``, integer dtype
            ijk_max (NumericMaxRank1): Index space maximum bound of the clip region,
                broadcastable to shape ``(3,)``, integer dtype

        Returns:
            clipped_grid (Grid): A :class:`Grid` representing the clipped version of this grid.
        """
        ijk_min = to_Vec3iBroadcastable(ijk_min)
        ijk_max = to_Vec3iBroadcastable(ijk_max)
        return Grid(impl=self._impl.clipped_grid(ijk_min, ijk_max))

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> "Grid":
        """
        Return a :class:`Grid` representing the coarsened version of this grid.
        Each voxel ``[i, j, k]`` in the input is included in the output if it lies within ``ijk_min`` and ``ijk_max``.

        Args:
            coarsening_factor (NumericMaxRank1): The factor by which to coarsen the grid,
                broadcastable to shape ``(3,)``, integer dtype

        Returns:
            coarsened_grid (Grid): A :class:`Grid` representing the coarsened version of this grid.
        """
        coarsening_factor = to_Vec3iBroadcastable(coarsening_factor, value_constraint=ValueConstraint.POSITIVE)
        return Grid(impl=self._impl.coarsened_grid(coarsening_factor))

    def contiguous(self) -> "Grid":
        """
        Return a contiguous copy of the grid.

        .. note::

            This is a no-op since a single :class:`Grid` is always contiguous.  However, this method is provided
            for API consistency with :class:`GridBatch`.

        Returns:
            grid (Grid): The same :class:`Grid` object.
        """
        return Grid(impl=self._impl.contiguous())

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> "Grid":
        """
        Return a :class:`Grid` representing the active voxels at the output of a convolution applied to this :class:`Grid` with a given kernel.

        Args:
            kernel_size (NumericMaxRank1): The size of the kernel to convolve with, broadcastable to shape ``(3,)``, integer dtype
            stride (NumericMaxRank1): The stride to use when convolving,
                broadcastable to shape ``(3,)``, integer dtype

        Returns:
            conv_grid (Grid): A :class:`Grid` representing the set of voxels in the output of the convolution defined by ``kernel_size`` and ``stride``.
        """
        kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE)
        return Grid(impl=self._impl.conv_grid(kernel_size, stride))

    def coords_in_grid(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Check if voxel coordinates are in active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates to check. A :class:`torch.Tensor` with shape ``(num_queries, 3)`` and integer coordinates.

        Returns:
            mask (torch.Tensor): A Boolean mask indicating which coordinates correspond to active voxels. Shape: ``(num_queries,)``.
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.coords_in_grid(jagged_ijk._impl).jdata

    def cpu(self) -> "Grid":
        """
        Make a copy of this :class:`Grid` on the CPU or this :class:`Grid` if it is already on the CPU.

        Returns:
            grid (Grid): A new :class:`Grid` on the CPU device, or this :class:`Grid` if it is already on the CPU.
        """
        return Grid(impl=self._impl.cpu())

    def cubes_in_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """
        Tests whether cubes defined by their centers and bounds are completely inside the active voxels of this :class:`Grid`.

        Args:
            cube_centers (torch.Tensor): Centers of the cubes in world coordinates. A :class:`torch.Tensor` with shape ``(num_cubes, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds, broadcastable to shape ``(3,)``, floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds, broadcastable to shape ``(3,)``, floating dtype

        Returns:
            mask (torch.Tensor): A Boolean mask indicating which cubes are fully contained in the grid. Shape: ``(num_cubes,)``.
        """
        jagged_cube_centers = JaggedTensor(cube_centers)
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)

        return self._impl.cubes_in_grid(jagged_cube_centers._impl, cube_min, cube_max).jdata

    def cubes_intersect_grid(
        self,
        cube_centers: torch.Tensor,
        cube_min: NumericMaxRank1 = 0.0,
        cube_max: NumericMaxRank1 = 0.0,
    ) -> torch.Tensor:
        """
        Tests whether cubes defined by their centers and bounds have any intersection
        with the active voxels of this :class:`Grid`.

        Args:
            cube_centers (torch.Tensor): Centers of the cubes in world coordinates. A :class:`torch.Tensor` with shape ``(num_cubes, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds, broadcastable to shape ``(3,)``, floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds, broadcastable to shape ``(3,)``, floating dtype

        Returns:
            mask (torch.Tensor): A Boolean mask indicating which cubes intersect the grid. Shape: ``(num_cubes,)``.
        """
        jagged_cube_centers = JaggedTensor(cube_centers)
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)
        return self._impl.cubes_intersect_grid(jagged_cube_centers._impl, cube_min, cube_max).jdata

    def cuda(self) -> "Grid":
        """
        Return a copy of this :class:`Grid` on a CUDA device, or this :class:`Grid` if it is already on CUDA.

        Returns:
            grid (Grid): A new :class:`Grid` on CUDA device, or this :class:`Grid` if it is already on CUDA.
        """
        return Grid(impl=self._impl.cuda())

    def dilated_grid(self, dilation: int) -> "Grid":
        """
        Return a new :class:`Grid` that is the result of dilating the current :class:`Grid` by a given number of voxels.

        Args:
            dilation (int): The dilation radius in voxels.

        Returns:
            grid (Grid): A new :class:`Grid` with dilated active regions.
        """
        return Grid(impl=self._impl.dilated_grid(dilation))

    def dual_grid(self, exclude_border: bool = False) -> "Grid":
        """
        Return a new :class:`Grid` whose voxel centers correspond to the corners of this :class:`Grid`.

        The dual grid is useful for staggered grid discretizations and finite difference operations.

        Args:
            exclude_border (bool): If ``True``, excludes border voxels that would extend beyond
                the primal grid bounds. Default is ``False``.

        Returns:
            grid (Grid): A new :class:`Grid` representing the dual grid.
        """
        return Grid(impl=self._impl.dual_grid(exclude_border))

    def voxel_to_world(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Transform a set of grid-space coordinates to their corresponding positions in world space
        using this :class:`Grid`'s origin and voxel size.

        .. seealso::

            :meth:`world_to_voxel` for the inverse transformation, and :meth:`voxel_to_world_matrix` and :meth:`world_to_voxel_matrix` for
            the actual transformation matrices.


        Args:
            ijk (torch.Tensor): A tensor of coordinates to convert. Shape: ``(num_points, 3)``. Can be fractional for interpolation.

        Returns:
            torch.Tensor: World coordinates. Shape: ``(num_points, 3)``.
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.grid_to_world(jagged_ijk._impl).jdata

    def has_same_address_and_grid_count(self, other: Any) -> bool:
        """
        Check if this :class:`Grid` has the same address and grid count as another :class:`Grid`.

        .. note::

            This method is primarily for internal use to compare grids efficiently.

        """
        if isinstance(other, Grid):
            return self.address == other.address
        elif isinstance(other, GridBatchCpp):
            return self.address == other.address and self._impl.grid_count == other.grid_count
        else:
            return False

    def ijk_to_index(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Convert grid-space coordinates to linear index space.

        Maps 3D grid-space coordinates to their corresponding linear indices.
        Returns ``-1`` for coordinates that don't correspond to active voxels.

        Args:
            ijk (torch.Tensor): Voxel coordinates to convert. A :class:`torch.Tensor`
                with shape ``(num_queries, 3)`` with integer coordinates.

        Returns:
            index (torch.Tensor): Linear indices for each coordinate, or ``-1`` if not active. Shape: ``(num_queries,)``.
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.ijk_to_index(jagged_ijk._impl).jdata

    def ijk_to_inv_index(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Get inverse permutation of :meth:`ijk_to_index`. *i.e.* for each voxel
        each index in the grid, return the index in the input ``ijk`` tensor.

        Example:

        .. code-block:: python

            # Create three ijk coordinates
            ijk_coords = torch.tensor([[100,0,10],[1024,1,1],[2,222,2]])

            # Create a grid with 3 voxels at those coordinates
            grid = Grid.from_ijk(ijk_coords)

            # Get the index coordinates of the three voxels
            # Returns [0, 2, 1] meaning
            #   [100,0,10] is voxel 0 in the grid
            #   [1024,1,1] is voxel 2 in the grid
            #   [2,222,2] is voxel 1 in the grid
            index_coords = grid.ijk_to_index(ijk_coords)

            # Now let's say you have another set of ijk coordinates
            query_ijk = torch.tensor([[2,222,2],[100,0,10], [50,50,50], [70, 0, 70]])

            # Returns [1, 0, -1, -1] meaning
            # the voxel in grid's index 0 maps to query_ijk index 1
            # the voxel in grid's index 1 maps to query_ijk index 0
            # the voxel in grid's index 2 does not exist in query_ijk, so -1
            # the voxel in grid's index 3 does not exist in query_ijk, so -1
            inv_index = grid.ijk_to_inv_index(query_ijk)

        Args:
            ijk (torch.Tensor): Voxel coordinates to convert.
                A :class:`torch.Tensor` with shape (num_queries, 3) with
                integer coordinates.

        Returns:
            inv_map (torch.Tensor): Inverse permutation for ijk_to_index.
                A :class:`torch.Tensor` with shape (num_queries,).
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.ijk_to_inv_index(jagged_ijk._impl).jdata

    def inject_from(
        self,
        src_grid: "Grid",
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """
        Inject data associated with the source grid to a :class:`torch.Tensor`
        associated with this grid.

        .. note::

            The copy occurs in voxel space, the voxel-to-world transform is not applied.

        .. note::

            If you pass in destination data, ``dst``, then ``dst`` will be modified in-place.
            If ``dst`` is ``None``, a new :class:`torch.Tensor` will be created with the
            shape ``(self.num_voxels, *src.shape[1:])`` and filled with ``default_value``
            for any voxels that do not have corresponding data in ``src``.

        .. note::

            This method supports backpropagation through the injection operation.

        Args:
            src_grid (Grid): The source :class:`Grid` to inject data from.
            src (torch.Tensor): Source data associated with ``src_grid``.
                This must be a Tensor with shape ``(src_grid.num_voxels, *)``.
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape ``(self.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Value to fill in for voxels that do not
                have corresponding data in ``src``. This is used only if ``dst`` is ``None``.
                Default is ``0``.

        Returns:
            dst (torch.Tensor): The data copied from ``src`` data after injection.

        """
        jagged_src = JaggedTensor(src)

        if dst is None:
            dst_shape = [self.num_voxels, *src.shape[1:]] if src.dim() > 1 else [self.num_voxels]
            dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)

        jagged_dst = JaggedTensor(dst)

        src_grid._impl.inject_to(self._impl, jagged_src._impl, jagged_dst._impl)

        return jagged_dst.jdata

    def inject_from_ijk(
        self,
        src_ijk: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        default_value: float | int | bool = 0,
    ):
        """
        Inject data associated with a set of source voxel coordinates to a
        :class:`torch.Tensor` associated with this grid.

        .. note::

            If you pass in destination data, ``dst``, then ``dst`` will be modified in-place.
            If ``dst`` is ``None``, a new :class:`torch.Tensor` will be created with the
            shape ``(self.num_voxels, *src.shape[1:])`` and filled with ``default_value``
            for any voxels that do not have corresponding data in ``src``.

        .. note::

            This method supports backpropagation through the injection operation.

        Args:
            src_ijk (torch.Tensor): Source voxel coordinates associated with ``src``.
                A :class:`torch.Tensor` with shape ``(num_src_voxels, 3)`` and integer coordinates.
            src (torch.Tensor): Data from the source ijk coordinates ``src_ijk``. A :class:`torch.Tensor`
                with shape ``(src_ijk.shape[0], *)``.
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape ``(self.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Value to fill in for voxels that do not have
                corresponding data in ``src``. This is used only if ``dst`` is ``None``.
                Default is 0.
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
        Inject data associated with this :class:`Grid` to data associated with ``dst_grid``.

        .. note::

            If you pass in destination data, ``dst``, then ``dst`` will be modified in-place.
            If ``dst`` is ``None``, a new :class:`torch.Tensor` will be created with the
            shape ``(self.num_voxels, *src.shape[1:])`` and filled with ``default_value``
            for any voxels that do not have corresponding data in ``src``.

        .. note::

            This method supports backpropagation through the injection operation.

        Args:
            dst_grid (Grid): The destination :class:`Grid` to inject data into.
            src (torch.Tensor): Source data from associated with this :class:`Grid`.
                This must be a Tensor with shape ``(self.num_voxels, *)``.
            dst (torch.Tensor | None): Optional destination data to be modified in-place.
                This must be a Tensor with shape ``(dst_grid.num_voxels, *)`` or ``None``.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in ``src``.
                This is used only if ``dst`` is ``None``. Default is 0.

        Returns:
            dst (torch.Tensor): The destination data associated with ``dst_grid`` data after injection.
        """
        jagged_src = JaggedTensor(src)

        if dst is None:
            dst_shape = [dst_grid.num_voxels, *src.shape[1:]] if src.dim() > 1 else [dst_grid.num_voxels]
            jagged_dst = JaggedTensor(
                torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
            )
        else:
            jagged_dst = JaggedTensor(dst)
        self._impl.inject_to(dst_grid._impl, jagged_src._impl, jagged_dst._impl)
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

        Updates the given TSDF values and weights associated with this :class:`Grid` by integrating new depth
        observations from a given camera viewpoint. This is commonly used for 3D
        reconstruction from RGB-D sensors.

        .. seealso::

            :meth:`integrate_tsdf_with_features` for integrating features along with TSDF values.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values (in world units).
            projection_matrix (torch.Tensor): Camera projection matrix. A tensor-like object with ``shape: (4, 4)``.
            cam_to_world_matrix (torch.Tensor): Camera to world transformation matrix. A tensor-like object with ``shape: (4, 4)``.
            tsdf (torch.Tensor): Current TSDF values for each voxel. A :class:`torch.Tensor` with shape: ``(self.num_voxels, 1)``.
            weights (torch.Tensor): Current integration weights for each voxel.
                A :class:`torch.Tensor` with shape: ``(self.num_voxels, 1)``.
            depth_image (torch.Tensor): Depth image from cameras.
                A :class:`torch.Tensor` with shape: ``(height, width)``.
            weight_image (torch.Tensor, optional): Weight of each depth sample in the image.
                A :class:`torch.Tensor` with shape: ``(height, width)``. If None, defaults to uniform weights.

        Returns:
            new_grid (Grid): Updated :class:`Grid` with potentially expanded voxels.
            new_tsdf (torch.Tensor): Updated TSDF values as a :class:`torch.Tensor` associated with ``new_grid``.
            new_weights (torch.Tensor): Updated weights as a :class:`torch.Tensor` associated with ``new_grid``.
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
            jagged_tsdf._impl,
            jagged_weights._impl,
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
        Integrate depth and feature images into a Truncated Signed Distance Function (TSDF) volume.

        Updates the given TSDF values and weights associated with this :class:`Grid` by integrating new depth
        observations from a given camera viewpoint. This is commonly used for 3D
        reconstruction from RGB-D sensors.

        .. seealso::

            :meth:`integrate_tsdf` for integrating without features along with TSDF values.

        Args:
            truncation_distance (float): Maximum distance to truncate TSDF values (in world units).
            projection_matrix (torch.Tensor): Camera projection matrix. A tensor-like object with ``shape: (4, 4)``.
            cam_to_world_matrix (torch.Tensor): Camera to world transformation matrix. A tensor-like object with ``shape: (4, 4)``.
            features (torch.Tensor): Current feature values associated with each voxel in this :class:`Grid`.
                A :class:`torch.Tensor` with shape ``(total_voxels, feature_dim)``.
            tsdf (torch.Tensor): Current TSDF values for each voxel. A :class:`torch.Tensor` with shape: ``(self.num_voxels, 1)``.
            weights (torch.Tensor): Current integration weights for each voxel.
                A :class:`torch.Tensor` with shape: ``(self.num_voxels, 1)``.
            depth_image (torch.Tensor): Depth image from cameras.
                A :class:`torch.Tensor` with shape: ``(height, width)``.
            feature_image (torch.Tensor): Feature image (e.g., RGB) from cameras.
                A :class:`torch.Tensor` with shape: ``(height, width, feature_dim)``.
            weight_image (torch.Tensor, optional): Weight of each depth sample in the image.
                A :class:`torch.Tensor` with shape: ``(height, width)``. If None, defaults to uniform weights.

        Returns:
            new_grid (Grid): Updated :class:`Grid` with potentially expanded voxels.
            new_tsdf (torch.Tensor): Updated TSDF values as a :class:`torch.Tensor` associated with ``new_grid``
                with shape ``(new_grid.num_voxels, 1)``.
            new_features (torch.Tensor): Updated features as a :class:`torch.Tensor` associated with ``new_grid``
                with shape ``(new_grid.num_voxels, feature_dim)``.
            new_weights (torch.Tensor): Updated weights as a :class:`torch.Tensor` associated with ``new_grid``
                with shape ``(new_grid.num_voxels, 1)``.
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
            jagged_tsdf._impl,
            jagged_features._impl,
            jagged_weights._impl,
            depth_image.unsqueeze(0),
            feature_image.unsqueeze(0),
            weight_image.unsqueeze(0) if weight_image is not None else None,
        )

        return Grid(impl=result_grid_impl), result_jagged_1.jdata, result_jagged_2.jdata, result_jagged_3.jdata

    def is_contiguous(self) -> bool:
        """
        Check if the grid data is stored contiguously in memory. This is generally ``True`` for
        :class:`Grid` since it represents a single grid, though can be ``False`` if you
        constructed the :class:`Grid` using :meth:`from_grid_batch` on a :class:`GridBatch` with more
        than one grid.

        Returns:
            is_contiguous (bool): ``True`` if all the data for this grid is stored contiguously in memory, ``False`` otherwise.
        """
        return self._impl.is_contiguous()

    def is_same(self, other: "Grid") -> bool:
        """
        Check if two :class:`Grid`\'s share the same underlying data in memory.

        Args:
            other (Grid): The other :class:`Grid` to compare with.

        Returns:
            is_same (bool): ``True`` if the grids have the same underlying data in memory, ``False`` otherwise.
        """
        return self._impl.is_same(other._impl)

    def marching_cubes(
        self, field: torch.Tensor, level: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract an isosurface mesh over data associated with this :class:`Grid` using the marching cubes algorithm.
        Generates a triangle mesh representing the isosurface at the specified level from a scalar field defined on the voxels.

        Args:
            field (torch.Tensor): Scalar field values at each voxel in this :class:`Grid`. A :class:`torch.Tensor` with shape
                ``(total_voxels, 1)``.
            level (float): The isovalue to extract the surface at. Default is 0.0.

        Returns:

            vertex_positions (torch.Tensor): Vertex positions of the mesh. Shape: ``(num_vertices, 3)``.
            face_indices (torch.Tensor): Triangle face indices. Shape: ``(num_faces, 3)``.
            vertex_normals (torch.Tensor): Vertex normals (computed from gradients). Shape: ``(num_vertices, 3)``.
        """
        jagged_field = JaggedTensor(field)
        verts, indices, normals = self._impl.marching_cubes(jagged_field._impl, level)
        return verts.jdata, indices.jdata, normals.jdata

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: torch.Tensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "Grid | None" = None,
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Apply max pooling to the given data associated with this :class:`Grid` returned as data associated
        with the given ``coarse_grid`` or a newly created coarse :class:`Grid`.

        Performs max pooling on the voxel data, reducing the resolution by the specified
        ``pool_factor``. Each output voxel contains the max of the corresponding input voxels
        within the pooling window. The pooling operation respects the sparse structure of this.
        :class:`Grid` and the given ``coarse_grid``.

        .. note::

            If you pass ``coarse_grid = None``, the returned coarse grid will have its
            voxel size multiplied by the ``pool_factor`` and its origin adjusted accordingly.

        .. note::

            This method supports backpropagation through the pooling operation.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grid, broadcastable to shape ``(3,)``, integer dtype
            data (torch.Tensor): The voxel data to pool. A :class:`torch.Tensor` with shape ``(total_voxels, channels)``.
            stride (NumericMaxRank1): The stride to use when pooling. If ``0`` (default), broadcastable to shape ``(3,)``, integer dtype
            coarse_grid (Grid, optional): Pre-allocated coarse grid to use for output. If ``None``, a new :class:`Grid` is created.

        Returns:
            pooled_data (torch.Tensor): A tensor containing the pooled voxel data with shape ``(coarse_total_voxels, channels)``.
            coarse_grid (Grid): A :class:`Grid` object representing the coarse grid topology after pooling. Matches the provided ``coarse_grid`` if given.

        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.max_pool(pool_factor, jagged_data._impl, stride, coarse_grid_impl)
        return result_data.jdata, Grid(impl=result_grid_impl)

    def merged_grid(self, other: "Grid") -> "Grid":
        """
        Return a new :class:`Grid` that is the union of this :class:`Grid` with another. The
        voxel-to-world transform of the resulting grid matches that of this :class:`Grid`.

        Args:
            other (Grid): The other :class:`Grid` to compute the union with.

        Returns:
            merged_grid (Grid): A new :class:`Grid` containing the union of active voxels from both grids.
        """
        return Grid(impl=self._impl.merged_grid(other._impl))

    def neighbor_indexes(self, ijk: torch.Tensor, extent: int, bitshift: int = 0) -> torch.Tensor:
        """
        Get indices of neighboring voxels in this :class:`Grid` in an N-ring neighborhood of each
        voxel coordinate in ``ijk``.

        Args:
            ijk (torch.Tensor): Voxel coordinates to find neighbors for.
                A :class:`torch.Tensor` with shape ``(num_queries, 3)`` with integer coordinates.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): An optional bit shift value to provide to each input ijk coordinate.
                *i.e* passing ``bitshift = 2`` is the same as calling ``neighbor_indexes(ijk << 2, extent)``.
                Default is 0.

        Returns:
            neighbor_indices (torch.Tensor): A :class:`torch.Tensor` of shape ``(num_queries, N)``
                containing the linear indices of neighboring voxels for each voxel coordinate in ``ijk``
                in the input. If some neighbors are not active in the grid, their indices will be ``-1``.
        """
        jagged_ijk = JaggedTensor(ijk)
        return self._impl.neighbor_indexes(jagged_ijk._impl, extent, bitshift).jdata

    def points_in_grid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if world-space points are located within active voxels. This method applies the
        world-to-voxel transform of this :class:`Grid` to each point, then checks if the resulting
        voxel coordinates correspond to active voxels.

        Args:
            points (torch.Tensor): World-space points to test. A :class:`torch.Tensor` with shape ``(num_queries, 3)``.

        Returns:
            mask (torch.Tensor): A Boolean mask indicating which points are in active voxels. Shape: ``(num_queries,)``.
        """
        jagged_points = JaggedTensor(points)
        return self._impl.points_in_grid(jagged_points._impl).jdata

    def pruned_grid(self, mask: torch.Tensor) -> "Grid":
        """
        Return a new :class:`Grid` where voxels are pruned based on a boolean mask. ``True``
        values in the mask indicate that the corresponding voxel should be kept, while ``False``
        values indicate that the voxel should be removed.

        Args:
            mask (torch.Tensor): Boolean mask for each voxel. A :class:`torch.Tensor` with shape ``(self.num_voxels,)``.

        Returns:
            pruned_grid (Grid): A new :class:`Grid` containing only voxels at indices where ``mask`` is True.
        """
        jagged_mask = JaggedTensor(mask)
        return Grid(impl=self._impl.pruned_grid(jagged_mask._impl))

    def ray_implicit_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        grid_scalars: torch.Tensor,
        eps: float = 0.0,
    ) -> torch.Tensor:
        """
        Find ray intersections with an implicit surface defined on the voxels of this :class:`Grid`.

        .. note::

            The implicit surface is defined by the zero level-set of the scalar field
            provided in ``grid_scalars``.

        .. note::

            The intersection distances are returned as multiples of the ray direction length.
            If the ray direction is normalized, the distances correspond to Euclidean distances.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space. A :class:`torch.Tensor` with shape ``(num_rays, 3)``.
            ray_directions (torch.Tensor): Direction vectors of rays. A :class:`torch.Tensor` with shape: ``(num_rays, 3)``.
                Note that the intersection distances are returned as a multiple of the ray direction length.
            grid_scalars (torch.Tensor): Scalar field values at each voxel. A :class:`torch.Tensor` with shape: ``(total_voxels, 1)``.
            eps (float): Epsilon value which can improve numerical stability. Default is ``0.0``.

        Returns:
            intersection_distances (torch.Tensor): Intersection distance along each input ray of the zero-level set of the input
                or -1 if no intersection occurs. A :class:`torch.Tensor` with shape ``(num_rays,)``.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)
        jagged_grid_scalars = JaggedTensor(grid_scalars)

        return self._impl.ray_implicit_intersection(
            jagged_ray_origins._impl, jagged_ray_directions._impl, jagged_grid_scalars._impl, eps
        ).jdata

    def rays_intersect_voxels(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, eps: float = 0.0
    ) -> torch.Tensor:
        """
        Given a set of rays, return a boolean :class:`torch.Tensor` indicating which rays intersect this :class:`Grid`.

        Args:
            ray_origins (torch.Tensor): an ``(N, 3)``-shaped tensor of ray origins
            ray_directions (torch.Tensor): an ``(N, 3)``-shaped tensor of ray directions
            eps (float): a small value which can help with numerical stability. Default is ``0.0``.

        Returns:
            rays_intersect (torch.Tensor): a boolean :class:`torch.Tensor` of shape ``(N,)`` indicating which rays intersect the grid.
                *i.e.* ``rays_intersect_voxels(ray_origins, ray_directions, eps)[i]`` is ``True`` if the
                ray corresponding to ``ray_origins[i], ray_directions[i]`` intersects with this Grid.
        """
        _, ray_times = self.voxels_along_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            max_voxels=1,
            eps=eps,
            return_ijk=False,
        )
        return (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0

    def inject_from_dense_cminor(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """
        Inject values from a dense :class:`torch.Tensor` into a :class:`torch.Tensor` associated with this :class:`Grid`.

        This is the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order. *i.e* the
        dense tensor has shape ``[dense_size_x, dense_size_y, dense_size_z, channels*]``.

        .. note::

            This method supports backpropagation through the read operation.

        .. seealso::

            :meth:`inject_from_dense_cmajor` for the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order.

        .. seealso::

            :meth:`inject_to_dense_cminor` for writing data to a dense tensor in C Minor order.

        Args:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` to read from. Shape: ``(dense_size_x, dense_size_y, dense_size_z, channels*)``.
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                voxel space, broadcastable to shape ``(3,)``, integer dtype

        Returns:
            sparse_data (torch.Tensor): Values from the dense tensor at voxel locations active in this :class:`Grid`.
                Shape: ``(self.num_voxels, channels*)``.
        """
        dense_origin = to_Vec3i(dense_origin)
        return self._impl.read_from_dense_cminor(dense_data.unsqueeze(0), dense_origin).jdata

    def inject_from_dense_cmajor(self, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor:
        """
        Inject values from a dense :class:`torch.Tensor` into a :class:`torch.Tensor` associated with this :class:`Grid`.

        This is the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order. *i.e* the
        dense tensor has shape ``[channels*, dense_size_x, dense_size_y, dense_size_z]``.

        .. note::

            This method supports backpropagation through the read operation.

        .. seealso::

            :meth:`inject_from_dense_cminor` for the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order.

        .. seealso::

            :meth:`inject_to_dense_cmajor` for writing data to a dense tensor in "C Major" order.

        Args:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` to read from. Shape: ``(channels*, dense_size_x, dense_size_y, dense_size_z)``.
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                voxel space, broadcastable to shape ``(3,)``, integer dtype

        Returns:
            sparse_data (torch.Tensor): Values from the dense tensor at voxel locations active in this :class:`Grid`.
                Shape: ``(self.num_voxels, channels*)``.
        """
        dense_origin = to_Vec3i(dense_origin)
        return self._impl.read_from_dense_cmajor(dense_data.unsqueeze(0), dense_origin).jdata

    def sample_bezier(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """
        Sample data in a :class:`torch.Tensor` associated with this :class:`Grid` at world-space
        points using Bézier interpolation.

        This method uses Bézier interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grid
            return zero.

        .. seealso::

            :meth:`sample_trilinear` for trilinear interpolation.

        .. seealso::

            :meth:`sample_bezier_with_grad` for Bézier interpolation which also returns spatial gradients.

        Args:
            points (torch.Tensor): World-space points to sample at. A :class:`torch.Tensor` of shape: ``(num_queries, 3)``.
            voxel_data (torch.Tensor): Data associated with each voxel in this :class:`Grid`.
                A :class:`torch.Tensor` of shape ``(total_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Interpolated data at each point. Shape: ``(num_queries, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)
        return self._impl.sample_bezier(jagged_points._impl, jagged_voxel_data._impl).jdata

    def sample_bezier_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample data in a :class:`torch.Tensor` associated with this :class:`Grid` at world-space
        points using Bézier interpolation, and return the sampled values and their spatial gradients at those points.

        This method uses Bézier interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers. It returns both the interpolated data
        and the gradients of the interpolated data with respect to the world coordinates.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grid
            return zero.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. seealso::

            :meth:`sample_bezier` for Bézier interpolation without gradients.

        .. seealso::

            :meth:`sample_trilinear_with_grad` for trilinear interpolation with spatial gradients.

        Args:
            points (torch.Tensor): World-space points to sample at. A :class:`torch.Tensor` of shape: ``(num_queries, 3)``.
            voxel_data (torch.Tensor): Data associated with each voxel in this :class:`Grid`.
                A :class:`torch.Tensor` of shape ``(total_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Interpolated data at each point. Shape: ``(num_queries, channels*)``.
            interpolation_gradients (torch.Tensor): Gradients of the interpolated data with respect to world coordinates.
                This is the spatial gradient of the Bézier interpolation at each point.
                Shape: ``(num_queries, 3, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        result_data, result_grad = self._impl.sample_bezier_with_grad(jagged_points._impl, jagged_voxel_data._impl)
        return result_data.jdata, result_grad.jdata

    def sample_trilinear(self, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor:
        """
        Sample data in a :class:`torch.Tensor` associated with this :class:`Grid` at world-space
        points using trilinear interpolation.

        This method uses trilinear interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grid
            return zero.

        .. seealso::

            :meth:`sample_bezier` for Bézier interpolation.

        .. seealso::

            :meth:`sample_trilinear_with_grad` for trilinear interpolation which also returns spatial gradients.

        Args:
            points (torch.Tensor): World-space points to sample at. A :class:`torch.Tensor` of shape: ``(num_queries, 3)``.
            voxel_data (torch.Tensor): Data associated with each voxel in this :class:`Grid`.
                A :class:`torch.Tensor` of shape ``(total_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Interpolated data at each point. Shape: ``(num_queries, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_trilinear(jagged_points._impl, jagged_voxel_data._impl).jdata

    def sample_trilinear_with_grad(
        self, points: torch.Tensor, voxel_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample data in a :class:`torch.Tensor` associated with this :class:`Grid` at world-space
        points using trilinear interpolation, and return the sampled values and their spatial gradients at those points.

        This method uses trilinear interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers. It returns both the interpolated data
        and the gradients of the interpolated data with respect to the world coordinates.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grid
            return zero.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. seealso::

            :meth:`sample_trilinear` for trilinear interpolation without gradients.

        .. seealso::

            :meth:`sample_bezier_with_grad` for Bézier interpolation with spatial gradients.

        Args:
            points (torch.Tensor): World-space points to sample at. A :class:`torch.Tensor` of shape: ``(num_queries, 3)``.
            voxel_data (torch.Tensor): Data associated with each voxel in this :class:`Grid`.
                A :class:`torch.Tensor` of shape ``(total_voxels, channels*)``.

        Returns:
            interpolated_data (torch.Tensor): Interpolated data at each point. Shape: ``(num_queries, channels*)``.
            interpolation_gradients (torch.Tensor): Gradients of the interpolated data with respect to world coordinates.
                This is the spatial gradient of the Bézier interpolation at each point.
                Shape: ``(num_queries, 3, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_voxel_data = JaggedTensor(voxel_data)

        result_data, result_grad = self._impl.sample_trilinear_with_grad(jagged_points._impl, jagged_voxel_data._impl)
        return result_data.jdata, result_grad.jdata

    def segments_along_rays(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, max_segments: int, eps: float = 0.0
    ) -> JaggedTensor:
        """
        Return segments of continuous ray traversal through this :class:`Grid`. Each segment is
        represented by its start and end distance along the ray. *i.e.* for each ray, the output
        contains a variable number of segments, each defined by a pair of distances ``(t_start, t_end)``,
        where ``t_start`` is the distance when the ray goes from being outside this :class:`Grid` to
        inside, and ``t_end`` is the distance when the ray exits the grid.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space. A :class:`torch.Tensor` with shape ``(num_rays, 3)``.
            ray_directions (torch.Tensor): Direction vectors of rays. A :class:`torch.Tensor` with shape: ``(num_rays, 3)``.
                Note that the intersection distances are returned as a multiple of the ray direction length.
            max_segments (int): Maximum number of segments to return per-ray.
            eps (float): Small epsilon value which can help with numerical stability. Default is ``0.0``.

        Returns:
            segments (JaggedTensor): A :class:`~fvdb.JaggedTensor` containing the segments along each ray. The JaggedTensor has shape:
                ``(num_rays, num_segments_per_ray, 2)``, where ``num_segments_per_ray`` varies per ray up to
                ``max_segments``. Each segment is represented by a pair of distances ``(t_start, t_end)``.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)

        return JaggedTensor(
            impl=self._impl.segments_along_rays(
                jagged_ray_origins._impl, jagged_ray_directions._impl, max_segments, eps
            )[0]
        )

    def sparse_conv_halo(self, input: torch.Tensor, weight: torch.Tensor, variant: int = 8) -> torch.Tensor:
        """
        Perform sparse convolution on an input :class:`torch.Tensor` associated with
        this :class:`Grid` using halo exchange optimization to efficiently handle boundary
        conditions in distributed or multi-block sparse grids.

        .. note::

            Halo convolution only supports convolving when the input and output grid topology match, thus
            this method does not accept an output grid. *i.e.* the output features will be associated with
            this :class:`Grid`.

        Args:
            input (torch.Tensor): Input features for each voxel in this :class:`Grid`.
                Shape: ``(self.num_voxels, in_channels)``.
            weight (torch.Tensor): Convolution weights. Shape ``(out_channels, in_channels, kernel_size_x, kernel_size_y, kernel_size_z)``.
            variant (int): Variant of the halo implementation to use. Default is 8.
                *Note:* This is cryptic on purpose and you should change it only if you know what you're doing.

        Returns:
            out_features (torch.Tensor): Output features with shape ``(self.num_voxels, out_channels)`` after convolution.
        """
        jagged_input = JaggedTensor(input)
        return self._impl.sparse_conv_halo(jagged_input._impl, weight, variant).jdata

    def splat_bezier(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """
        Splat data at a set of input points into a :class:`torch.Tensor` associated with
        this :class:`Grid` using Bézier interpolation. *i.e.* each point distributes its
        data to the surrounding voxels using cubic Bézier interpolation weights.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels.

        .. note::

            This method supports backpropagation through the splatting operation.

        Args:
            points (torch.Tensor): World-space positions of points used to splat data. Shape: ``(num_points, 3)``.
            points_data (torch.Tensor): Data associated with each point to splat into the grid.
                Shape: ``(num_points, channels*)``.

        Returns:
            splatted_features (torch.Tensor): Accumulated features at each voxel after splatting.
                Shape: ``(self.num_voxels, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_points_data = JaggedTensor(points_data)

        return self._impl.splat_bezier(jagged_points._impl, jagged_points_data._impl).jdata

    def splat_trilinear(self, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor:
        """
        Splat data at a set of input points into a :class:`torch.Tensor` associated with
        this :class:`Grid` using trilinear interpolation. *i.e.* each point distributes its
        data to the surrounding voxels using trilinear interpolation weights.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels.

        .. note::

            This method supports backpropagation through the splatting operation.

        Args:
            points (torch.Tensor): World-space positions of points used to splat data. Shape: ``(num_points, 3)``.
            points_data (torch.Tensor): Data associated with each point to splat into the grid.
                Shape: ``(num_points, channels*)``.

        Returns:
            splatted_features (torch.Tensor): Accumulated features at each voxel after splatting.
                Shape: ``(self.num_voxels, channels*)``.
        """
        jagged_points = JaggedTensor(points)
        jagged_points_data = JaggedTensor(points_data)

        return self._impl.splat_trilinear(jagged_points._impl, jagged_points_data._impl).jdata

    def refine(
        self,
        subdiv_factor: NumericMaxRank1,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        fine_grid: "Grid | None" = None,
    ) -> tuple[torch.Tensor, "Grid"]:
        """
        Refine data associated with this :class:`Grid` into a higher-resolution grid by subdividing each voxel.
        *i.e* for each voxel, ``(i, j, k)`` in this :class:`Grid`, copy the data associated with that voxel to
        the voxels ``(subdiv_factor[0]*i + di, subdiv_factor[1]*j + dj, subdiv_factor[2]*k + dk)``
        for ``di, dj, dk`` in ``{0, ..., subdiv_factor - 1}`` in the output data associated with ``fine_grid``, if
        the that voxel exists.

        .. note::

            If you pass ``fine_grid = None``, this method will create a new fine :class:`Grid` with its
            voxel size divided by the ``subdiv_factor`` and its origin adjusted accordingly.

        .. note::

            You can skip copying data at certain voxels in this :class:`Grid` by passing a boolean ``mask`` of shape ``(self.num_voxels,)``.
            Only data at voxels corresponding to ``True`` values in the mask will be refined.

        .. note::

            This method supports backpropagation through the refinement operation.

        .. seealso::

            :meth:`refined_grid` for obtaining a refined version of the grid structure without refining data.

        Args:
            subdiv_factor (NumericMaxRank1): Refinement factor between this :class:`Grid` and the fine grid, broadcastable to shape ``(3,)``, integer dtype
            data (torch.Tensor): Voxel data to refine. A :class:`torch.Tensor` of shape ``(total_voxels, channels)``.
            mask (torch.Tensor, optional): Boolean mask of shape ``(self.num_voxels,)``indicating which voxels in the input :class:`Grid` to refine. If ``None``, data associated with all input voxels are refined.
            fine_grid (Grid, optional): Pre-allocated fine :class:`Grid` to use for output. If ``None``, a new :class:`Grid` is created.

        Returns:
            tuple[torch.Tensor, Grid]: A tuple containing:
                - The refined data as a torch.Tensor
                - The fine Grid containing the refined structure
        """
        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_data = JaggedTensor(data)
        jagged_mask_impl = JaggedTensor(mask)._impl if mask is not None else None

        fine_grid_impl = fine_grid._impl if fine_grid else None

        result_data, result_grid_impl = self._impl.refine(
            subdiv_factor, jagged_data._impl, jagged_mask_impl, fine_grid_impl
        )
        return result_data.jdata, Grid(impl=result_grid_impl)

    def refined_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: torch.Tensor | None = None,
    ) -> "Grid":
        """
        Return a refined version of this :class:`Grid`. *i.e* each voxel in this :class:`Grid` is subdivided
        by the specified ``subdiv_factor`` to create a higher-resolution grid.

        .. note::

            You can skip refining certain voxels in this :class:`Grid` by passing a boolean
            ``mask`` of shape ``(self.num_voxels,)``. Only voxels corresponding to ``True``
            values in the mask will be refined.

        .. seealso::

            :meth:`refine` for copying data from a coarse :class:`Grid` to a refined :class:`Grid`.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to refine this :class:`Grid`, broadcastable to shape ``(3,)``, integer dtype
            mask (torch.Tensor, optional): Boolean mask indicating which voxels to refine. If ``None``, all voxels are refined.

        Returns:
            grid (Grid): A new :class:`Grid` with refined structure.
        """

        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        jagged_mask_impl = JaggedTensor(mask)._impl if mask is not None else None

        return Grid(impl=self._impl.refined_grid(subdiv_factor, mask=jagged_mask_impl))

    def save_nanovdb(
        self,
        path: str | pathlib.Path,
        data: torch.Tensor | None = None,
        name: str | None = None,
        compressed: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Save this :class:`Grid` and optional data associated with it to a .nvdb file.

        The grid is saved in the NanoVDB format, which can be loaded by other
        applications that support OpenVDB/NanoVDB.

        Args:
            path (str | pathlib.Path): The file path to save to. Should have .nvdb extension.
            data (torch.Tensor, optional): Voxel data to save with the grid.
                Shape: ``(self.num_voxels, channels)``. If ``None``, only grid structure is saved.
            name (str, optional): Optional name for the grid
            compressed (bool): Whether to compress the data using Blosc compression.
                Default is ``False``.
            verbose (bool): Whether to print information about the saved grid.
                Default is ``False``.
        """
        from ._Cpp import save as _save

        if isinstance(path, pathlib.Path):
            path = str(path)

        jagged_data_impl = JaggedTensor(data)._impl if data is not None else None

        # Handle the overloaded signature - if name is provided, use it
        if name is not None:
            _save(path, self._impl, jagged_data_impl, name, compressed, verbose)
        else:
            # Default case with empty names list
            _save(path, self._impl, jagged_data_impl, [], compressed, verbose)

    def to(self, target: "str | torch.device | torch.Tensor | JaggedTensor | Grid") -> "Grid":
        """
        Move this :class:`Grid` to a target device or to match the device of an object (*e.g.* another :class:`Grid`, a :class:`JaggedTensor`, a :class:`torch.Tensor`, etc.).

        Args:
            target (str | torch.device | torch.Tensor | JaggedTensor | Grid): Target object to determine the device.

        Returns:
            grid (Grid): A new :class:`Grid` on the target device or this :class:`Grid` if the target device is the same as ``self.device``.
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
        Generate uniformly spaced samples along rays intersecting this :class:`Grid`.

        This method creates sample points at regular intervals along rays, but only for segments
        that intersect with active voxels. The uniform samples start at
        ``ray_origins + ray_directions * t_min`` and end at ``ray_origins + ray_directions * t_max``,
        with spacing defined by ``step_size``, and only include samples which lie within the grid.

        If ``cone_angle`` is greater than zero, the method uses cone tracing to adjust the
        sampling rate based on the distance from the ray origin, allowing for adaptive sampling.

        .. note::

            The returned samples are represented as a :class:`~fvdb.JaggedTensor`, where each
            element contains either the start and end distance of each sample segment along the ray
            or the midpoint of each sample segment if ``return_midpoints`` is ``True``.

        .. note::

            If ``include_end_segments`` is ``True``, partial segments at the start and end of each ray
            that do not fit the full ``step_size`` will be included.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space. A :class:`torch.Tensor` with shape ``(num_rays, 3)``.
            ray_directions (torch.Tensor): Direction vectors of rays. A :class:`torch.Tensor` with shape: ``(num_rays, 3)``.
                Note that the intersection distances are returned as a multiple of the ray direction length.
            t_min (torch.Tensor): Minimum distance along rays to start sampling. A :class:`~torch.Tensor` of shape ``(num_rays,)``.
            t_max (torch.Tensor): Maximum distance along rays to stop sampling. A :class:`~torch.Tensor` of shape ``(num_rays,)``.
            step_size (float): Distance between samples along each ray.
            cone_angle (float): Cone angle for cone tracing (in radians). Default is 0.0.
            include_end_segments (bool): Whether to include partial segments at ray ends. Default is ``True``.
            return_midpoints (bool): Whether to return segment midpoints instead of start points *i.e* if this value is ``True``, the samples will lie halfway between each step. Default is ``False``.
            eps (float): Epsilon value which can improve numerical stability. Default is ``0.0``.

        Returns:
            samples (JaggedTensor): A :class:`~fvdb.JaggedTensor` containing the samples along each ray.
                The :class:`~fvdb.JaggedTensor` has shape: ``(num_rays, num_samples_per_ray,)``,
                where ``num_samples_per_ray`` varies per ray. Each sample in ``samples[r]`` is a
                distance along the ray ``ray_origins + ray_directions * t``.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)
        jagged_t_min = JaggedTensor(t_min)
        jagged_t_max = JaggedTensor(t_max)

        return JaggedTensor(
            impl=self._impl.uniform_ray_samples(
                jagged_ray_origins._impl,
                jagged_ray_directions._impl,
                jagged_t_min._impl,
                jagged_t_max._impl,
                step_size,
                cone_angle,
                include_end_segments,
                return_midpoints,
                eps,
            )[0]
        )

    def voxels_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Enumerate the indices of voxels in this :class:`Grid` intersected by a set of rays in the
        order of their intersection.

        .. note::

            If instead of index coordinates, you want voxel coordinates (*i.e.* ``(i, j, k)``),
            set ``return_ijk=True``.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space. A :class:`torch.Tensor` with shape ``(num_rays, 3)``.
            ray_directions (torch.Tensor): Direction vectors of rays. A :class:`torch.Tensor` with shape: ``(num_rays, 3)``.
                Note that the intersection distances are returned as a multiple of the ray direction length.
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Small epsilon value which can help with numerical stability. Default is ``0.0``.
            return_ijk (bool): Whether to return voxel coordinates instead of index coordinates. If ``False``, returns
                linear indices instead. Default is ``False``.

        Returns:
            voxels (JaggedTensor): The voxel indices (or voxel coordinates) intersected by the rays. This is a
                :class:`~fvdb.JaggedTensor` with shape: ``(num_rays, num_voxels_per_ray,)``,
                where ``num_voxels_per_ray`` varies per ray up to ``max_voxels``. Each element contains either the linear index of the voxel
                or the ``(i, j, k)`` coordinates of the voxel if ``return_ijk=True``.
                *Note:* If ``return_ijk=True``, ``voxels`` will have shape: ``(num_rays, num_voxels_per_ray, 3)``.
            distances (JaggedTensor): The entry and exit distances along each ray for each intersected voxel. This is a
                :class:`~fvdb.JaggedTensor` with shape: ``(num_rays, num_voxels_per_ray, 2)``,
                where ``num_voxels_per_ray`` varies per ray up to ``max_voxels``. Each element contains a pair of distances
                ``(t_entry, t_exit)``, representing where the ray enters and exits the voxel along its direction.
        """
        jagged_ray_origins = JaggedTensor(ray_origins)
        jagged_ray_directions = JaggedTensor(ray_directions)

        voxels, times = self._impl.voxels_along_rays(
            jagged_ray_origins._impl, jagged_ray_directions._impl, max_voxels, eps, return_ijk, True
        )
        return JaggedTensor(impl=voxels[0]), JaggedTensor(impl=times[0])

    def world_to_voxel(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world space coordinates to voxel space coordinates using the
        world-to-voxel transformation of this :class:`Grid`.

        .. note::

            This method supports backpropagation through the transformation operation.

        .. seealso::

            :meth:`voxel_to_world` for the inverse transformation, and :meth:`voxel_to_world_matrix` and :meth:`world_to_voxel_matrix` for
            the actual transformation matrices.

        Args:
            points (torch.Tensor): World-space positions to convert. A :class:`torch.Tensor` with shape ``(num_points, 3)``.

        Returns:
            voxel_points (torch.Tensor): Grid coordinates. A :class:`torch.Tensor` with shape ``(num_points, 3)``.
                Can contain fractional values.
        """
        jagged_points = JaggedTensor(points)
        return self._impl.world_to_grid(jagged_points._impl).jdata

    def inject_to_dense_cminor(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Inject values from :class:`torch.Tensor` associated with this :class:`Grid` into a
        dense :class:`torch.Tensor`.

        This is the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order. *i.e* the
        dense tensor has shape ``[dense_size_x, dense_size_y, dense_size_z, channels*]``.

        This method creates the dense tensor to return, and fills it with values from the sparse grid
        within the range defined by ``min_coord`` and ``grid_size``.
        Voxels not present in the sparse grid are filled with zeros. *.i.e.* this method will copy
        all the voxel values in the range ``[min_coord, min_coord + grid_size)`` into a dense tensor
        of shape ``[dense_size_x, dense_size_y, dense_size_z, channels*]``, such that ``min_coord``
        maps to index ``(0, 0, 0)`` in the dense tensor, and ``min_coord + grid_size - 1`` maps to index
        ``(dense_size_x - 1, dense_size_y - 1, dense_size_z - 1)`` in the dense tensor.

        .. note::

            This method supports backpropagation through the write operation.

        .. seealso::

            :meth:`inject_from_dense_cmajor` for reading from a dense tensor in "C Major" order,
            which assumes the dense tensor has shape ``[channels*, dense_size_x, dense_size_y, dense_size_z]``.

        .. seealso::

            :meth:`inject_to_dense_cmajor` for writing to a dense tensor in "C Major" order.

        Args:
            sparse_data (torch.Tensor): A :class:`torch.Tensor` of data associated with this :class:`Grid` with
                shape ``(self.num_voxels, channels*)``.
            min_coord (NumericMaxRank1|None): Minimum voxel coordinate to read from the :class:`Grid`
                into the output dense tensor, broadcastable to shape ``(3,)``, integer dtype, or ``None``.
                If set to ``None``, this will be the minimum voxel coordinate of this :class:`Grid`'s bounding box.
            grid_size (NumericMaxRank1|None): Size of the output dense tensor, broadcastable to
                shape ``(3,)``, integer dtype, or ``None``. If ``None``, computed to fit all active
                voxels starting from ``min_coord``. *i.e.* if ``min_coord`` is ``(2, 2, 2)`` and the
                maximum active voxel in the grid is ``(5, 5, 5)``, the computed ``grid_size`` will be ``(4, 4, 4)``.

        Returns:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` containing the sparse data with
                shape ``(dense_size_x, dense_size_y, dense_size_z, channels*)``.
        """
        jagged_sparse_data = JaggedTensor(sparse_data)
        min_coord = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        grid_size = (
            to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE)
            if grid_size is not None
            else None
        )

        return self._impl.write_to_dense_cminor(jagged_sparse_data._impl, min_coord, grid_size).squeeze(0)

    def inject_to_dense_cmajor(
        self,
        sparse_data: torch.Tensor,
        min_coord: NumericMaxRank1 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Write values from :class:`torch.Tensor` associated with this :class:`Grid` into a
        dense :class:`torch.Tensor`.

        This is the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order. *i.e* the
        dense tensor has shape ``[channels*, dense_size_x, dense_size_y, dense_size_z]``.

        This method creates the dense tensor to return, and fills it with values from the sparse grid
        within the range defined by ``min_coord`` and ``grid_size``.
        Voxels not present in the sparse grid are filled with zeros. *.i.e.* this method will copy
        all the voxel values in the range ``[min_coord, min_coord + grid_size)`` into a dense tensor
        of shape ``[dense_size_x, dense_size_y, dense_size_z, channels*]``, such that ``min_coord``
        maps to index ``(0, 0, 0)`` in the dense tensor, and ``min_coord + grid_size - 1`` maps to index
        ``(dense_size_x - 1, dense_size_y - 1, dense_size_z - 1)`` in the dense tensor.

        .. note::

            This method supports backpropagation through the write operation.

        .. seealso::

            :meth:`inject_from_dense_cmajor` for reading from a dense tensor in "C Major" order,
            which assumes the dense tensor has shape ``[channels*, dense_size_x, dense_size_y, dense_size_z]``.

        .. seealso::

            :meth:`inject_to_dense_cminor` for writing to a dense tensor in "C Minor" order.

        Args:
            sparse_data (torch.Tensor): A :class:`torch.Tensor` of data associated with this :class:`Grid` with
                shape ``(self.num_voxels, channels*)``.
            min_coord (NumericMaxRank1|None): Minimum voxel coordinate to read from the :class:`Grid`
                into the output dense tensor, broadcastable to shape ``(3,)``, integer dtype, or ``None``.
                If set to ``None``, this will be the minimum voxel coordinate of this :class:`Grid`'s bounding box.
            grid_size (NumericMaxRank1|None): Size of the output dense tensor, broadcastable to
                shape ``(3,)``, integer dtype, or ``None``. If ``None``, computed to fit all active
                voxels starting from ``min_coord``. *i.e.* if ``min_coord`` is ``(2, 2, 2)`` and the
                maximum active voxel in the grid is ``(5, 5, 5)``, the computed ``grid_size`` will be ``(4, 4, 4)``.

        Returns:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` containing the sparse data with
                shape ``(channels*, dense_size_x, dense_size_y, dense_size_z)``.
        """
        jagged_sparse_data = JaggedTensor(sparse_data)
        min_coord = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        grid_size = (
            to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE)
            if grid_size is not None
            else None
        )

        return self._impl.write_to_dense_cmajor(jagged_sparse_data._impl, min_coord, grid_size).squeeze(0)

    # ============================================================
    #                        Properties
    # ============================================================

    # Properties
    @property
    def address(self) -> int:
        """
        The address of the underlying C++ NanoVDB grid object.

        Returns:

            address (int): The memory address of the underlying C++ NanoVDB grid.
        """
        return self._impl.address

    @property
    def bbox(self) -> torch.Tensor:
        """

        The voxel-space bounding box of this :class:`Grid`.

        .. note::

            The bounding box is inclusive of the minimum voxel and the the maximum voxel.

            *e.g.* if you have a grid with a single voxel at index ``(0, 0, 0)``, the bounding box will be
            ``[[0, 0, 0], [0, 0, 0]]``.

            *e.g.* if you have a grid with voxels at indices ``(0, 0, 0)`` and ``(1, 1, 1)``, the bounding box will be
            ``[[0, 0, 0], [1, 1, 1]]``.

        Returns:
            bbox (torch.Tensor): A ``(2, 3)``-shaped tensor representing the minimum and maximum
                voxel indices of the bounding box. If the grid has zero voxels, returns a zero tensor.
        """
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.bbox_at(0)

    @property
    def device(self) -> torch.device:
        """
        Return the :class:`torch.device` where this :class:`Grid` is stored.

        Returns:
            device (torch.device): The device of the grid.
        """
        return self._impl.device

    @property
    def dual_bbox(self) -> torch.Tensor:
        """
        Return the voxel space bounding box of the dual of this :class:`Grid`. *i.e.* the bounding box
        of the :class:`Grid` whose voxel centers correspond to voxel corners in this :class:`Grid`.

        .. seealso::

            :meth:`bbox` for the bounding box of this :class:`Grid`, and :meth:`dual_grid` for computing the
            dual grid itself.

        .. note::

            The bounding box is inclusive of the minimum voxel and the the maximum voxel.

            *e.g.* if you have a grid with a single voxel at index ``(0, 0, 0)``, the dual grid will contain voxels
            at indices ``(0, 0, 0), (0, 0, 1), (0, 1, 0), ..., (1, 1, 1)``, and the bounding box will be
            ``[[0, 0, 0], [1, 1, 1]]``.

        Returns:
            dual_bbox (torch.Tensor): A ``(2, 3)``-shaped tensor representing the minimum and maximum
                voxel indices of the dual bounding box. If the grid has zero voxels, returns a zero tensor.

        """
        if self.has_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.dual_bbox_at(0)

    @property
    def voxel_to_world_matrix(self) -> torch.Tensor:
        """
        The voxel-to-world transformation matrix for this :class:`Grid`, which
        transforms voxel space coordinates to world space coordinates.

        Returns:
            voxel_to_world_matrix (torch.Tensor): A ``(4, 4)``-shaped tensor representing the
                voxel-to-world transformation matrix.
        """
        return self._impl.grid_to_world_matrices[0]

    @property
    def has_zero_voxels(self) -> bool:
        """
        ``True`` if this :class:`Grid` has zero active voxels, ``False`` otherwise.

        Returns:
            has_zero_voxels (bool): Whether the grid has zero active voxels.
        """
        return self.num_voxels == 0

    @property
    def ijk(self) -> torch.Tensor:
        """
        The voxel coordinates of every active voxel in this :class:`Grid`, in index order.

        Returns:
            ijk (torch.Tensor): A ``(num_voxels, 3)``-shaped tensor containing the
                voxel coordinates of each active voxel in index order.
        """
        return self._impl.ijk.jdata

    @property
    def num_bytes(self) -> int:
        """
        The size in bytes this :class:`Grid` occupies in memory.

        Returns:
            num_bytes (int): The size in bytes of the grid.
        """
        return self._impl.total_bytes

    @property
    def num_leaf_nodes(self) -> int:
        """
        The number of leaf nodes in the NanoVDB underlying this :class:`Grid`.

        Returns:
            num_leaf_nodes (int): The number of leaf nodes in the grid.
        """
        return self._impl.total_leaf_nodes

    @property
    def num_voxels(self) -> int:
        """
        The number of active voxels in this :class:`Grid`.

        Returns:
            num_voxels (int): The number of active voxels in the grid.
        """
        return self._impl.total_voxels

    @property
    def origin(self) -> torch.Tensor:
        """
        The world-space origin of this :class:`Grid`. *i.e.* the world-space position of the
        center of the voxel at ``(0, 0, 0)`` in voxel space.

        Returns:
            origin (torch.Tensor): A ``(3,)``-shaped tensor representing the world-space origin.
        """
        return self._impl.origin_at(0)

    @property
    def voxel_size(self) -> torch.Tensor:
        """
        The world-space size of each voxel in this :class:`Grid`.

        Returns:
            voxel_size (torch.Tensor): A ``(3,)``-shaped tensor representing the size of each voxel.
        """
        return self._impl.voxel_size_at(0)

    @property
    def world_to_voxel_matrix(self) -> torch.Tensor:
        """
        The world-to-voxel transformation matrix for this :class:`Grid`, which
        transforms world space coordinates to voxel space coordinates.

        Returns:
            world_to_voxel_matrix (torch.Tensor): A ``(4, 4)``-shaped tensor representing the
                world-to-voxel transformation matrix.
        """
        return self._impl.world_to_grid_matrices[0]

    # Expose underlying implementation for compatibility
    @property
    def _gridbatch(self):
        # Access underlying GridBatchCpp - use sparingly during migration
        return self._impl
