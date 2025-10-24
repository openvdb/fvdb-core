# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Batch of sparse grids data structure and operations for FVDB.

This module provides the core GridBatch class for managing batches of sparse voxel grids:

Classes:
- GridBatch: A batch of sparse voxel grids with support for efficient operations

Class-methods for creating GridBatch objects from various sources:

- :meth:`GridBatch.from_zero_grids()`: for an empty grid batch where grid-count = 0.
- :meth:`GridBatch.from_zero_voxels()`: for a grid batch where each grid has zero voxels.
- :meth:`GridBatch.from_dense()`: for a grid batch where each grid is dense data
- :meth:`GridBatch.from_dense_axis_aligned_bounds()`: for a grid batch where each grid is dense data defined by axis-aligned bounds
- :meth:`GridBatch.from_grid()`: for a grid batch from a single :class:`Grid` instance
- :meth:`GridBatch.from_ijk()`: for a grid batch from explicit voxel coordinates
- :meth:`GridBatch.from_mesh()`: for a grid batch from triangle meshes
- :meth:`GridBatch.from_points()`: for a grid batch from point clouds
- :meth:`GridBatch.from_nearest_voxels_to_points()`: for a grid batch from nearest voxels to points

Class/Instance-methods for loading and saving grids:
- from_nanovdb/save_nanovdb: Load and save grid batches to/from .nvdb files

GridBatch supports operations like convolution, pooling, interpolation, ray casting,
mesh extraction, and coordinate transformations on sparse voxel data.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import torch

from . import _parse_device_string
from ._Cpp import GridBatch as GridBatchCpp
from .jagged_tensor import JaggedTensor
from .types import (
    DeviceIdentifier,
    GridBatchIndex,
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    resolve_device,
    to_Vec3fBatch,
    to_Vec3fBatchBroadcastable,
    to_Vec3fBroadcastable,
    to_Vec3i,
    to_Vec3iBatchBroadcastable,
    to_Vec3iBroadcastable,
)

if TYPE_CHECKING:
    from .grid import Grid


class GridBatch:
    """
    A batch of sparse voxel grids with support for efficient operations.

    :class:`GridBatch` represents a collection of sparse 3D voxel grids that can be processed
    together efficiently on GPU. Each grid in the batch can have different resolutions,
    origins, and voxel sizes. The class provides methods for common operations like
    sampling, convolution, pooling, dilation, union, etc. It also provides more advanced features
    such as marching cubes, TSDF fusion, and fast ray marching.

    A :class:`GridBatch` can be thought of as a mini-batch of :class:`Grid` instances and,
    like the :class:`Grid`, does not collect the sparse voxel grids' data but only collects
    their structure (or topology). Voxel data (e.g., features, colors, densities) for the
    collection of grids is stored separately as an :class:`JaggedTensor` associated with
    the :class:`GridBatch`. This separation allows for flexibility in the type and number of
    channels of data with which a grid can be used to index into. This also allows multiple grids to
    share the same data storage if desired.

    When using a :class:`GridBatch`, there are three important coordinate systems
    to be aware of:

    - **World Space**: The continuous 3D coordinate system in which each grid in the batch exists.
    - **Voxel Space**: The discrete voxel index system of each grid in the batch, where each voxel is identified by its integer indices (i, j, k).
    - **Index Space**: The linear indexing of active voxels in each grid's internal storage.


    At its core, a :class:`GridBatch` uses a very fast mapping from each grid's voxel space into
    index space to perform operations on a :class:`fvdb.JaggedTensor` of data associated with the
    grids in the batch. This mapping allows for efficient access and manipulation of voxel data.
    For example:

    .. code-block:: python

        voxel_coords = torch.tensor([[8, 7, 6], [1, 2, 3], [4, 5, 6]], device="cuda")  # Voxel space coordinates
        batch_voxel_coords = fvdb.JaggedTensor(
            [voxel_coords, voxel_coords + 44, voxel_coords - 44]
        )  # Voxel space coordinates for 3 grids in the batch

        # Create a GridBatch containing 3 grids with the 3 sets of voxel coordinates such that the voxels
        # have a world space size of 1x1x1, and where the [0, 0, 0] voxel in voxel space of each grid is at world space origin (0, 0, 0).
        grid_batch = fvdb.GridBatch.from_ijk(batch_voxel_coords, voxel_sizes=1.0, origins=0.0, device="cuda")

        # Create some data associated with the grids - here we have 9 voxels and 2 channels per voxel
        voxel_data = torch.randn(grid_batch.total_voxels, 2, device="cuda")  # Index space data

        # Map voxel space coordinates to index space
        indices = grid_batch.ijk_to_index(batch_voxel_coords, cumulative=True).jdata  # Shape: (9,)

        # Access the data for the specified voxel coordinates
        selected_data = voxel_data[indices]  # Shape: (9, 2)

    .. note::

        A :class:`GridBatch` may contain zero grids, in which case it has no voxel sizes nor origins
        that can be queried. It may also contain one or more empty grids, which means grids that
        have zero voxels. An empty grid still has a voxel size and origin, which can be queried.

    .. note::

        The grids are stored in a sparse format using `NanoVDB <https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb>`_
        where only active (non-empty) voxels are allocated, making it extremely memory efficient for representing large volumes with sparse
        occupancy.

    .. note::

        The :class:`GridBatch` constructor is for internal use only. To create a :class:`GridBatch` with actual content, use the classmethods:

            - :meth:`from_zero_grids()`: for an empty grid batch where grid-count = 0.
            - :meth:`from_zero_voxels()`: for a grid batch where each grid has zero voxels.
            - :meth:`from_dense()`: for a grid batch where each grid is dense data
            - :meth:`from_dense_axis_aligned_bounds()`: for a grid batch where each grid is dense data defined by axis-aligned bounds
            - :meth:`from_grid()`: for a grid batch from a single :class:`Grid` instance
            - :meth:`from_ijk()`: for a grid batch from explicit voxel coordinates
            - :meth:`from_mesh()`: for a grid batch from triangle meshes
            - :meth:`from_points()`: for a grid batch from point clouds
            - :meth:`from_nearest_voxels_to_points()`: for a grid batch from nearest voxels to points

    Attributes:
        max_grids_per_batch (int): Maximum number of grids that can be stored in a single :class:`fvdb.GridBatch`.


    """

    # Class variable
    max_grids_per_batch: int = GridBatchCpp.max_grids_per_batch

    def __init__(self, *, impl: GridBatchCpp):
        """
        Constructor for internal use only. - use the GridBatch.from_* classmethods instead.
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
        Create a batch of dense grids.

        A dense grid has a voxel for every coordinate in an axis-aligned box.

        For each grid in the batch, the dense grid is defined by:

        - dense_dims: the size of the dense grids (shape ``[3,] = [W, H, D]``)
        - ijk_min: the minimum voxel index for the grid (shape ``[3,] = [i_min, j_min, k_min]``)
        - voxel_sizes: the world-space size of each voxel (shape ``[3,] = [sx, sy, sz]``)
        - origins: the world-space coordinate of the center of the ``[0,0,0]`` voxel of the grid (shape ``[3,] = [x0, y0, z0]``)
        - mask: indicates which voxels are "active" in the resulting grids.

        .. note::

            ``voxel_sizes`` and ``origins`` may be provided per-grid or broadcast across the batch.
            ``ijk_min`` and ``dense_dims`` apply to all grids in the batch.
            ``mask`` applies to all grids.

        Args:
            num_grids (int): Number of grids to create.
            dense_dims (NumericMaxRank1): Dimensions of the dense grid for all grids in the batch,
                broadcastable to shape ``(3,)``, integer dtype.
            ijk_min (NumericMaxRank1): Minimum voxel index for the grids, for all grids in the batch
                broadcastable to shape ``(3,)``, integer dtype.
            voxel_sizes (NumericMaxRank2): World-space size of each voxel, per-grid; broadcastable to
                shape ``(num_grids, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space coordinate of the center of the ``[0,0,0]`` voxel, per-grid;
                broadcastable to shape ``(num_grids, 3)``, floating dtype.
            mask (torch.Tensor | None): Optional boolean mask with shape ``(W, H, D)`` selecting active voxels.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``, which
                inherits from ``mask`` if provided, otherwise uses ``"cpu"``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` object.

        Examples:

            .. code-block:: python

                grid_batch = fvdb.GridBatch.from_dense(
                     num_grids=5,
                     dense_dims=[10, 10, 10],
                     voxel_sizes=[1.0, 1.0, 1.0],
                     origins=[0.0, 0.0, 0.0],
                     mask=None,
                     device="cuda",
                )
                grid_batch.grid_count # 5
                grid_batch.voxel_sizes == tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
                grid_batch.origins == tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        """
        resolved_device = resolve_device(device, inherit_from=mask)

        dense_dims = to_Vec3iBroadcastable(dense_dims, value_constraint=ValueConstraint.POSITIVE)
        ijk_min = to_Vec3i(ijk_min)
        voxel_sizes = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
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
        """
        Create a :class:`fvdb.GridBatch` representing a batch of dense grids defined by axis-aligned bounds.

        The resulting grids have voxels spanning ``dense_dims`` with voxel sizes and origins
        computed to fit within the world-space box ``[bounds_min, bounds_max]``.

        If ``voxel_center`` is ``True``, the bounds correspond to the centers of the corner voxels.
        If ``voxel_center`` is ``False``, the bounds correspond to the outer edges of the corner voxels.

        Args:
            num_grids (int): Number of grids to create.
            dense_dims (NumericMaxRank1): Dimensions of the dense grids, broadcastable to shape ``(3,)``, integer dtype.
            bounds_min (NumericMaxRank1): Minimum world-space coordinate for all grids, broadcastable to shape ``(3,)``, floating dtype.
            bounds_max (NumericMaxRank1): Maximum world-space coordinate for all grids, broadcastable to shape ``(3,)``, floating dtype.
            voxel_center (bool): Whether the bounds correspond to voxel centers (``True``) or edges (``False``). Defaults to ``False``.
            device (DeviceIdentifier): Device to create the grids on. Defaults to ``"cpu"``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                grid_batch = fvdb.GridBatch.from_dense_axis_aligned_bounds(
                    num_grids=5,
                    dense_dims=[10, 10, 10],
                    bounds_min=[-1.0, -1.0, -1.0],
                    bounds_max=[1.0, 1.0, 1.0],
                    voxel_center=False,
                    device="cuda",
                )
                grid_batch.grid_count # 5
                grid_batch.voxel_sizes # tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
                grid_batch.origins # tensor([[-.9, -.9, -.9], [-.9, -.9, -.9], [-.9, -.9, -.9], [-.9, -.9, -.9], [-.9, -.9, -.9]])
        """

        dense_dims = to_Vec3iBroadcastable(dense_dims, value_constraint=ValueConstraint.POSITIVE)
        bounds_min = to_Vec3fBroadcastable(bounds_min)
        bounds_max = to_Vec3fBroadcastable(bounds_max)

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
        Create a :class:`fvdb.GridBatch` of batch size 1 from a single :class:`fvdb.Grid`.

        Args:
            grid (Grid): The :class:`fvdb.Grid` to create the grid batch from.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                grid = fvdb.Grid.from_ijk(
                    ijk=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]),
                    voxel_size=[1.0, 1.0, 1.0],
                    origin=[0.0, 0.0, 0.0],
                    device="cuda",
                )
                grid_batch = fvdb.GridBatch.from_grid(grid)
                grid_batch.grid_count # 1
                grid_batch.ijk.jdata == tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

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
        Create a batch of grids from voxel-space coordinates. If multiple voxels in a grid map to the same coordinate,
        only one voxel will be created at that coordinate.

        Args:
            ijk (JaggedTensor): Per-grid voxel coordinates to populate. Shape: ``(batch_size, num_voxels, 3)``
                with integer coordinates.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``,
                floating dtype
            origins (NumericMaxRank2): World-space coordinate of the center of the ``[0,0,0]`` voxel, per-grid;
                broadcastable to shape ``(batch_size, 3)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``, which
                inherits from ``ijk``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                ijk = fvdb.JaggedTensor(torch.tensor([
                    [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
                ]))
                grid_batch = fvdb.GridBatch.from_ijk(ijk=ijk, voxel_sizes=[1.0, 1.0, 1.0], origins=[0.0, 0.0, 0.0])
                grid_batch.grid_count # 1
                grid_batch.ijk.jdata == tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

        """
        resolved_device = resolve_device(device, inherit_from=ijk)

        voxel_sizes = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_ijk(ijk._impl, voxel_sizes, origins)
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
        Create a :class:`fvdb.GridBatch` by voxelizing the *surface* of a set of triangle meshes. *i.e.* voxels that intersect
        the surface of the meshes will be contained in the resulting :class:`fvdb.GridBatch`.

        Args:
            mesh_vertices (JaggedTensor): Per-grid mesh vertex positions. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_vertices, 3)``.
            mesh_faces (JaggedTensor): Per-grid mesh face indices. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_faces, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``,
                floating dtype
            origins (NumericMaxRank2): World-space coordinate of the center of the ``[0,0,0]`` voxel, per-grid;
                broadcastable to shape ``(batch_size, 3)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``, which
                inherits from ``mesh_vertices``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object with voxels covering the surfaces of the input meshes.

        Examples:

            .. code-block:: python

                mesh_vertices = fvdb.JaggedTensor(torch.tensor([
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
                ]))
                mesh_faces = fvdb.JaggedTensor(torch.tensor([
                    [0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6],
                    [0, 1, 4], [1, 5, 4], [2, 3, 6], [3, 7, 6],
                    [0, 2, 4], [2, 6, 4], [1, 3, 5], [3, 7, 5]
                ]))
                grid_batch = fvdb.GridBatch.from_mesh(mesh_vertices, mesh_faces, voxel_sizes=[1.0, 1.0, 1.0], origins=[0.0, 0.0, 0.0])
                grid_batch.grid_count # 1
                grid_batch.ijk.jdata == tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

        """
        resolved_device = resolve_device(device, inherit_from=mesh_vertices)

        voxel_sizes = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_mesh(mesh_vertices._impl, mesh_faces._impl, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    # Load and save functions
    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        indices: list[int],
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        index: int,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        names: list[str],
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]": ...

    @overload
    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        name: str,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]": ...

    @classmethod
    def from_nanovdb(
        cls,
        path: str,
        *,
        indices: list[int] | None = None,
        index: int | None = None,
        names: list[str] | None = None,
        name: str | None = None,
        device: DeviceIdentifier = "cpu",
        verbose: bool = False,
    ) -> "tuple[GridBatch, JaggedTensor, list[str]]":
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
            grid_batch (GridBatch): A :class:`fvdb.GridBatch` containing the loaded grids.
            data (JaggedTensor): A :class:`fvdb.JaggedTensor` containing the data of the grids.
            names (list[str]): A list of strings containing the name of each grid.
        """
        from ._Cpp import load as _load

        device = resolve_device(device)

        # Check that only one selector is provided
        selectors = [indices is not None, index is not None, names is not None, name is not None]
        if sum(selectors) > 1:
            raise ValueError("Only one of indices, index, names, or name can be specified")

        # Call the appropriate overload
        if indices is not None:
            grid_impl, data_impl, names_out = _load(path, indices, device, verbose)
        elif index is not None:
            grid_impl, data_impl, names_out = _load(path, index, device, verbose)
        elif names is not None:
            grid_impl, data_impl, names_out = _load(path, names, device, verbose)
        elif name is not None:
            grid_impl, data_impl, names_out = _load(path, name, device, verbose)
        else:
            # Load all grids
            grid_impl, data_impl, names_out = _load(path, device, verbose)

        # Wrap the GridBatch implementation with the Python wrapper
        return cls(impl=grid_impl), JaggedTensor(impl=data_impl), names_out

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """
        Create grids by adding the eight nearest voxels to every input point.

        Args:
            points (JaggedTensor): Per-grid point positions to populate the grid from. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``,
                floating dtype
            origins (NumericMaxRank2): World-space coordinate of the center of the ``[0,0,0]`` voxel, per-grid;
                broadcastable to shape ``(batch_size, 3)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``, which
                inherits from ``points``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                points = fvdb.JaggedTensor(torch.tensor([
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
                ]))
                grid_batch = fvdb.GridBatch.from_nearest_voxels_to_points(points, voxel_sizes=[1.0, 1.0, 1.0], origins=[0.0, 0.0, 0.0])
                grid_batch.grid_count # 1
                grid_batch.ijk.jdata == tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                  [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                                  [0, 2, 0], [0, 2, 1], [0, 2, 2],
                                                  [1, 0, 0], [1, 0, 1], [1, 0, 2],
                                                  [1, 1, 0], [1, 1, 1], [1, 1, 2],
                                                  [1, 2, 0], [1, 2, 1], [1, 2, 2],
                                                  [2, 0, 0], [2, 0, 1], [2, 0, 2],
                                                  [2, 1, 0], [2, 1, 1], [2, 1, 2],
                                                  [2, 2, 0], [2, 2, 1], [2, 2, 2]], dtype=torch.int32)


        """
        resolved_device = resolve_device(device, inherit_from=points)

        voxel_sizes = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_nearest_voxels_to_points(points._impl, voxel_sizes, origins)
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
        Create a batch of grids from a batch of point clouds.

        Args:
            points (JaggedTensor): Per-grid point positions to populate the grid from. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``,
                floating dtype
            origins (NumericMaxRank2): World-space coordinate of the center of the ``[0,0,0]`` voxel, per-grid;
                broadcastable to shape ``(batch_size, 3)``, floating dtype
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``, which
                inherits from ``points``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                points = fvdb.JaggedTensor(torch.tensor([
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
                ]))
                grid_batch = fvdb.GridBatch.from_points(points, voxel_sizes=[1.0, 1.0, 1.0], origins=[0.0, 0.0, 0.0])
                grid_batch.grid_count # 1
                grid_batch.ijk.jdata == tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

        """
        resolved_device = resolve_device(device, inherit_from=points)

        voxel_sizes = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
        origins = to_Vec3fBatch(origins)

        grid_batch_impl = GridBatchCpp(resolved_device)
        grid_batch_impl.set_from_points(points._impl, voxel_sizes, origins)
        return cls(impl=grid_batch_impl)

    @classmethod
    def from_zero_grids(cls, device: DeviceIdentifier = "cpu") -> "GridBatch":
        """
        Create a :class:`fvdb.GridBatch` with zero grids. It retains its device identifier, but
        has no other information like voxel size or origin or bounding box. It will report
        ``grid_count == 0``.

        Args:
            device (DeviceIdentifier): The device to create the :class:`fvdb.GridBatch` on.
                Can be a string (e.g., ``"cuda"``, ``"cpu"``) or a :class:`torch.device` object. Defaults to ``"cpu"``.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object.

        Examples:

            .. code-block:: python

                grid_batch = fvdb.GridBatch.from_zero_grids("cuda")
                grid_batch.grid_count # 0

        """
        return cls(impl=GridBatchCpp(device=resolve_device(device)))

    @classmethod
    def from_zero_voxels(
        cls, device: DeviceIdentifier = "cpu", voxel_sizes: NumericMaxRank2 = 1, origins: NumericMaxRank2 = 0
    ) -> "GridBatch":
        """
        Create a :class:`fvdb.GridBatch` with one or more zero-voxel grids on a specific device.

        A zero-voxel grid batch does not mean there are zero grids. It means that the grids have
        zero voxels. This constructor will create as many zero-voxel grids as the batch size
        of ``voxel_sizes`` and ``origins``, defaulting to 1 grid, though for that case, you should use
        the single-grid :class:`fvdb.Grid` constructor instead.

        Args:
            device (DeviceIdentifier): The device to create the :class:`fvdb.GridBatch` on.
                Can be a string (e.g., ``"cuda"``, ``"cpu"``) or a :class:`torch.device` object. Defaults to ``"cpu"``.
            voxel_sizes (NumericMaxRank2): The default size per voxel,
                broadcastable to shape ``(num_grids, 3)``, floating dtype
            origins (NumericMaxRank2): The default origin of the grid,
                broadcastable to shape ``(num_grids, 3)``, floating dtype


        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` object with zero-voxel grids.

        Examples:

            .. code-block:: python

                grid_batch = GridBatch.from_zero_voxels("cuda", 1, 0)  # string
                grid_batch = GridBatch.from_zero_voxels(torch.device("cuda:0"), 1, 0)  # device directly
                grid_batch = GridBatch.from_zero_voxels(voxel_sizes=1, origins=0)  # defaults to CPU

        """
        resolved_device = resolve_device(device)
        voxel_sizes = to_Vec3fBatch(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
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
        Apply average pooling to the given data associated with this :class:`GridBatch` returned as data associated
        with the given ``coarse_grid`` or a newly created coarse :class:`GridBatch`.

        Performs average pooling on the voxel data, reducing the resolution by the specified
        ``pool_factor``. Each output voxel contains the average of the corresponding input voxels
        within the pooling window. The pooling operation respects the sparse structure of this
        :class:`GridBatch` and the given ``coarse_grid``.

        .. note::

            If you pass ``coarse_grid = None``, the returned coarse grid batch will have its
            voxel sizes multiplied by the ``pool_factor`` and origins adjusted accordingly.

        .. note::

            This method supports backpropagation through the pooling operation.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grids, broadcastable to shape ``(3,)``, integer dtype
            data (JaggedTensor): The voxel data to pool. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels)``.
            stride (NumericMaxRank1): The stride to use when pooling. If ``0`` (default), stride equals ``pool_factor``, broadcastable to shape ``(3,)``, integer dtype
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid batch to use for output. If ``None``, a new :class:`GridBatch` is created.

        Returns:
            pooled_data (JaggedTensor): A :class:`fvdb.JaggedTensor` containing the pooled voxel data with shape ``(batch_size, coarse_total_voxels, channels)``.
            coarse_grid (GridBatch): A :class:`GridBatch` object representing the coarse grid batch topology after pooling. Matches the provided ``coarse_grid`` if given.
        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data_impl, result_grid_impl = self._impl.avg_pool(pool_factor, data._impl, stride, coarse_grid_impl)

        return JaggedTensor(impl=result_data_impl), GridBatch(impl=cast(GridBatchCpp, result_grid_impl))

    def bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the bounding box of the bi^th grid in the batch.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            bbox (torch.Tensor): A tensor of shape ``(2, 3)`` where
                ``bbox = [[bmin_i, bmin_j, bmin_k], [bmax_i, bmax_j, bmax_k]]`` is the half-open
                bounding box such that ``bmin <= ijk < bmax`` for all active voxels ``ijk`` in the
                ``bi``-th grid.
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
        Creates a new :class:`fvdb.GridBatch` containing only the voxels that fall within the specified
        bounding box range ``[ijk_min, ijk_max]`` for each grid in the batch, and returns the corresponding clipped features.

        .. note::

            This method supports backpropagation through the clipping operation.

        Args:
            features (JaggedTensor): The voxel features to clip. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels)``.
            ijk_min (NumericMaxRank2): Minimum bounds in voxel space for each grid, broadcastable to shape ``(batch_size, 3)``, integer dtype
            ijk_max (NumericMaxRank2): Maximum bounds in voxel space for each grid, broadcastable to shape ``(batch_size, 3)``, integer dtype

        Returns:
            clipped_features (JaggedTensor): A :class:`fvdb.JaggedTensor` containing the clipped voxel features with shape ``(batch_size, clipped_total_voxels, channels)``.
            clipped_grid (GridBatch): A new :class:`fvdb.GridBatch` containing only voxels within the specified bounds for each grid.
        """
        ijk_min = to_Vec3iBatchBroadcastable(ijk_min)
        ijk_max = to_Vec3iBatchBroadcastable(ijk_max)

        result_features_impl, result_grid_impl = self._impl.clip(features._impl, ijk_min, ijk_max)
        return JaggedTensor(impl=result_features_impl), GridBatch(impl=result_grid_impl)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank2,
        ijk_max: NumericMaxRank2,
    ) -> "GridBatch":
        """
        Return a new :class:`GridBatch` representing the clipped version of this batch of grids.
        Each voxel ``[i, j, k]`` in each grid of the input batch is included in the output if it lies within ``ijk_min`` and ``ijk_max`` for that grid.

        Args:
            ijk_min (NumericMaxRank2): Voxel space minimum bound of the clip region for each grid, broadcastable to shape ``(batch_size, 3)``, integer dtype
            ijk_max (NumericMaxRank2): Voxel space maximum bound of the clip region for each grid, broadcastable to shape ``(batch_size, 3)``, integer dtype

        Returns:
            clipped_grid (GridBatch): A :class:`GridBatch` representing the clipped version of this grid batch.
        """
        ijk_min = to_Vec3iBatchBroadcastable(ijk_min)
        ijk_max = to_Vec3iBatchBroadcastable(ijk_max)

        return GridBatch(impl=self._impl.clipped_grid(ijk_min, ijk_max))

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> "GridBatch":
        """
        Return a :class:`GridBatch` representing the coarsened version of this batch of grids.
        Each voxel ``[i, j, k]`` in the input that satisfies ``i % coarsening_factor[0] == 0``,
        ``j % coarsening_factor[1] == 0``, and ``k % coarsening_factor[2] == 0`` is included in the output.

        Args:
            coarsening_factor (NumericMaxRank1): The factor by which to coarsen each grid, broadcastable to shape ``(3,)``, integer dtype

        Returns:
            coarsened_grid (GridBatch): A :class:`GridBatch` representing the coarsened version of this grid batch.
        """
        coarsening_factor = to_Vec3iBroadcastable(coarsening_factor, value_constraint=ValueConstraint.POSITIVE)

        return GridBatch(impl=self._impl.coarsened_grid(coarsening_factor))

    def contiguous(self) -> "GridBatch":
        """
        Return a contiguous copy of the grid batch.

        Ensures that the underlying data is stored contiguously in memory,
        which can improve performance for subsequent operations.

        Returns:
            grid_batch (GridBatch): A new GridBatch with contiguous memory layout.
        """
        return GridBatch(impl=self._impl.contiguous())

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> "GridBatch":
        """
        Return a :class:`GridBatch` representing the active voxels at the output of a convolution
        applied to this batch with a given kernel.

        Args:
            kernel_size (NumericMaxRank1): Size of the kernel to convolve with, broadcastable to shape ``(3,)``, integer dtype.
            stride (NumericMaxRank1): Stride to use when convolving, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            conv_grid (GridBatch): A GridBatch representing the convolution of this grid batch.
        """
        kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE)

        return GridBatch(impl=self._impl.conv_grid(kernel_size, stride))

    def coords_in_grid(self, ijk: JaggedTensor) -> JaggedTensor:
        """
        Check which voxel-space coordinates lie on active voxels for each grid.

        Args:
            ijk (JaggedTensor): Per-grid voxel coordinates to test. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries, 3)`` with integer dtype.

        Returns:
            mask (JaggedTensor): Boolean mask per-grid indicating which coordinates map to active voxels. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries)``.
        """
        return JaggedTensor(impl=self._impl.coords_in_grid(ijk._impl))

    def cpu(self) -> "GridBatch":
        """
        Move the grid batch to CPU.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` on CPU device.
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
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_cubes, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds,
                broadcastable to shape ``(3,)``, floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds,
                broadcastable to shape ``(3,)``, floating dtype

        Returns:
            mask (JaggedTensor): Boolean mask indicating which cubes are fully contained in the grid.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_cubes)``.
        """
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)

        return JaggedTensor(impl=self._impl.cubes_in_grid(cube_centers._impl, cube_min, cube_max))

    def cubes_intersect_grid(
        self, cube_centers: JaggedTensor, cube_min: NumericMaxRank1 = 0, cube_max: NumericMaxRank1 = 0
    ) -> JaggedTensor:
        """
        Check if axis-aligned cubes intersect with the grid.

        Tests whether cubes defined by their centers and bounds have any intersection
        with the active voxels of the grid.

        Args:
            cube_centers (JaggedTensor): Centers of the cubes in world coordinates.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_cubes, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center defining cube bounds,
                broadcastable to shape ``(3,)``, floating dtype
            cube_max (NumericMaxRank1): Maximum offsets from center defining cube bounds,
                broadcastable to shape ``(3,)``, floating dtype

        Returns:
            mask (JaggedTensor): Boolean mask indicating which cubes intersect the grid.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_cubes)``.
        """
        cube_min = to_Vec3fBroadcastable(cube_min)
        cube_max = to_Vec3fBroadcastable(cube_max)

        return JaggedTensor(impl=self._impl.cubes_intersect_grid(cube_centers._impl, cube_min, cube_max))

    def cuda(self) -> "GridBatch":
        """
        Move the grid batch to CUDA device.

        Returns:
            grid_batch (GridBatch): A new :class:`fvdb.GridBatch` on CUDA device.
        """
        return GridBatch(impl=self._impl.cuda())

    def cum_voxels_at(self, bi: int) -> int:
        """
        Get the cumulative number of voxels up to and including a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            cum_voxels (int): The cumulative number of voxels up to and including grid ``bi``.
        """
        return self._impl.cum_voxels_at(bi)

    def dilated_grid(self, dilation: int) -> "GridBatch":
        """
        Return the grid dilated by a given number of voxels.

        Args:
            dilation (int): The dilation radius in voxels.

        Returns:
            dilated_grid (GridBatch): A new :class:`fvdb.GridBatch` with dilated active regions.
        """
        return GridBatch(impl=self._impl.dilated_grid(dilation))

    def dual_bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the dual bounding box of a specific grid in the batch.

        The dual grid has voxel centers at the corners of the primal grid voxels.

        .. seealso::

            :meth:`dual_grid` to compute the actual dual grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            dual_bbox (torch.Tensor): A tensor of shape ``(2, 3)`` containing the minimum and maximum
                coordinates of the dual bounding box in voxel space.
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
            exclude_border (bool): If ``True``, excludes border voxels that would extend beyond
                the primal grid bounds. Default is ``False``.

        Returns:
            dual_grid (GridBatch): A new :class:`fvdb.GridBatch` representing the dual grid.
        """
        return GridBatch(
            impl=self._impl.dual_grid(exclude_border),
        )

    def voxel_to_world(self, ijk: JaggedTensor) -> JaggedTensor:
        """
        Transform a set of voxel-space coordinates to their corresponding positions in world space
        using each grid's origin and voxel size.

        .. seealso::

            :meth:`world_to_voxel` for the inverse transformation, and :attr:`voxel_to_world_matrices` and :attr:`world_to_voxel_matrices` for
            the actual transformation matrices.

        Args:
            ijk (JaggedTensor): A :class:`fvdb.JaggedTensor` of coordinates to convert. Shape: ``(batch_size, num_points, 3)``. Can be fractional for interpolation.

        Returns:
            world_coords (JaggedTensor): World coordinates. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
        """
        return JaggedTensor(impl=self._impl.grid_to_world(ijk._impl))

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
            is_empty (bool): True if the grid is empty, False otherwise.
        """
        return self.num_voxels_at(bi) == 0

    def ijk_to_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Convert voxel-space coordinates to linear index-space for each grid.

        Maps 3D voxel space coordinates to their corresponding linear indices.
        Returns ``-1`` for coordinates that don't correspond to active voxels.

        Args:
            ijk (JaggedTensor): Per-grid voxel coordinates to convert. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries, 3)`` with integer dtype.
            cumulative (bool): If ``True``, return indices cumulative across the whole batch; otherwise per-grid.

        Returns:
            indices (JaggedTensor): Linear indices for each coordinate, or ``-1`` if not active.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries)``.
        """
        assert isinstance(ijk, JaggedTensor), "ijk must be a JaggedTensor"
        return JaggedTensor(impl=self._impl.ijk_to_index(ijk._impl, cumulative))

    def ijk_to_inv_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Get inverse permutation of :meth:`ijk_to_index`. *i.e.* for each voxel in each grid,
        return the index in the input ``ijk`` tensor.

        Args:
            ijk (JaggedTensor): Voxel coordinates to convert. A :class:`fvdb.JaggedTensor`
                with shape ``(batch_size, num_queries, 3)`` with integer coordinates.
            cumulative (bool): If ``True``, returns cumulative indices across the entire batch.
                If ``False``, returns per-grid indices. Default is ``False``.

        Returns:
            inv_map (JaggedTensor): Inverse permutation for :meth:`ijk_to_index`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries)``.
        """
        return JaggedTensor(impl=self._impl.ijk_to_inv_index(ijk._impl, cumulative))

    def inject_from(
        self,
        src_grid: "GridBatch",
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ) -> JaggedTensor:
        """
        Inject data associated with the source grid batch to a :class:`fvdb.JaggedTensor`
        associated with this grid batch.

        .. note::

            The copy occurs in voxel space, the voxel-to-world transform is not applied.

        .. note::

            If you pass in destination data, ``dst``, then ``dst`` will be modified in-place.
            If ``dst`` is ``None``, a new :class:`fvdb.JaggedTensor` will be created with the
            shape ``(self.grid_count, self.total_voxels, *src.eshape)`` and filled with ``default_value``
            for any voxels that do not have corresponding data in ``src``.

        .. note::

            This method supports backpropagation through the injection operation.

        Args:
            src_grid (GridBatch): The source :class:`fvdb.GridBatch` to inject data from.
            src (JaggedTensor): Source data associated with ``src_grid``.
                This must be a :class:`fvdb.JaggedTensor` with shape ``(batch_size, src_grid.total_voxels, *)``.
            dst (JaggedTensor | None): Optional destination data to be modified in-place.
                This must be a :class:`fvdb.JaggedTensor` with shape ``(batch_size, self.total_voxels, *)`` or ``None``.
            default_value (float | int | bool): Value to fill in for voxels that do not
                have corresponding data in ``src``. This is used only if ``dst`` is ``None``.
                Default is ``0``.

        Returns:
            dst (JaggedTensor): The data copied from ``src`` data after injection.
        """
        if dst is None:
            dst_shape = [self.total_voxels]
            dst_shape.extend(src.eshape)
            dst = self.jagged_like(torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device))

        if dst.eshape != src.eshape:
            raise ValueError(
                f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}"
            )

        src_grid._impl.inject_to(self._impl, src._impl, dst._impl)

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
            src_ijk (JaggedTensor): Voxel coordinates in voxel space from which to copy data.
                Shape: ``(B, num_src_voxels, 3)``.
            src (JaggedTensor): Source data to inject. Must match the shape of the destination.
                Shape: ``(B, num_src_voxels, *)``.
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

        The copy occurs in "voxel-space", the voxel-to-world transform is not applied.

        If you pass in the destination data (`dst`), it will be modified in-place.
        If `dst` is None, a new JaggedTensor will be created with the same element shape as src
        and filled with `default_value` for any voxels that do not have corresponding data in `src`.

        Args:
            dst_grid (GridBatch): The destination grid to inject data into.
            src (JaggedTensor): Source data from this grid.
                Shape: ``(batch_size, -1, *)``.
            dst (JaggedTensor | None): Optional destination data to be modified in-place.
                Shape: ``(batch_size, -1, *)`` or ``None``.
            default_value (float | int | bool): Value to fill in for voxels that do not have corresponding data in `src`.
                This is used only if `dst` is None. Default is 0.

        Returns:
            dst (JaggedTensor): The destination sidecar data after injection.
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
        self._impl.inject_to(dst_grid._impl, src._impl, dst._impl)
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
                Shape: ``(batch_size, 3, 3)``.
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: ``(batch_size, 4, 4)``.
            tsdf (JaggedTensor): Current TSDF values for each voxel.
                Shape: ``(batch_size, total_voxels, 1)``.
            weights (JaggedTensor): Current integration weights for each voxel.
                Shape: ``(batch_size, total_voxels, 1)``.
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: ``(batch_size, height, width)``.
            weight_images (torch.Tensor, optional): Weight of each depth sample in the images.
                Shape: ``(batch_size, height, width)``. If None, defaults to uniform weights.

        Returns:
            updated_grid (GridBatch): Updated GridBatch with potentially expanded voxels.
            updated_tsdf (JaggedTensor): Updated TSDF values as JaggedTensor.
            updated_weights (JaggedTensor): Updated weights as JaggedTensor.
        """

        result_grid_impl, result_jagged_1_impl, result_jagged_2_impl = self._impl.integrate_tsdf(
            truncation_distance,
            projection_matrices,
            cam_to_world_matrices,
            tsdf._impl,
            weights._impl,
            depth_images,
            weight_images,
        )

        return (
            GridBatch(impl=result_grid_impl),
            JaggedTensor(impl=result_jagged_1_impl),
            JaggedTensor(impl=result_jagged_2_impl),
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
                Shape: ``(batch_size, 3, 3)``.
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: ``(batch_size, 4, 4)``.
            tsdf (JaggedTensor): Current TSDF values for each voxel.
                Shape: ``(batch_size, total_voxels, 1)``.
            features (JaggedTensor): Current feature values for each voxel.
                Shape: ``(batch_size, total_voxels, feature_dim)``.
            weights (JaggedTensor): Current integration weights for each voxel.
                Shape: ``(batch_size, total_voxels, 1)``.
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: ``(batch_size, height, width)``.
            feature_images (torch.Tensor): Feature images (e.g., RGB) from cameras.
                Shape: ``(batch_size, height, width, feature_dim)``.
            weight_images (torch.Tensor, optional): Weight of each depth sample in the images.
                Shape: ``(batch_size, height, width)``. If None, defaults to uniform weights.

        Returns:
            updated_grid (GridBatch): Updated GridBatch with potentially expanded voxels.
            updated_tsdf (JaggedTensor): Updated TSDF values as JaggedTensor.
            updated_weights (JaggedTensor): Updated weights as JaggedTensor.
            updated_features (JaggedTensor): Updated features as JaggedTensor.
        """
        result_grid_impl, result_jagged_1_impl, result_jagged_2_impl, result_jagged_3_impl = (
            self._impl.integrate_tsdf_with_features(
                truncation_distance,
                projection_matrices,
                cam_to_world_matrices,
                tsdf._impl,
                features._impl,
                weights._impl,
                depth_images,
                feature_images,
                weight_images,
            )
        )

        return (
            GridBatch(impl=result_grid_impl),
            JaggedTensor(impl=result_jagged_1_impl),
            JaggedTensor(impl=result_jagged_2_impl),
            JaggedTensor(impl=result_jagged_3_impl),
        )

    def is_contiguous(self) -> bool:
        """
        Check if the grid batch data is stored contiguously in memory.

        Returns:
            is_contiguous (bool): True if the data is contiguous, False otherwise.
        """
        return self._impl.is_contiguous()

    def is_same(self, other: "GridBatch") -> bool:
        """
        Check if two grid batches share the same underlying data in memory.

        Args:
            other (GridBatch): The other grid batch to compare with.

        Returns:
            is_same (bool): True if the grid batches have the same underlying data in memory, False otherwise.
        """
        return self._impl.is_same(other._impl)

    def jagged_like(self, data: torch.Tensor) -> JaggedTensor:
        """
        Create a JaggedTensor with the same jagged structure as this grid batch.

        Useful for creating feature tensors that match the grid's voxel layout.

        Args:
            data (torch.Tensor): Dense data to convert to jagged format.
                Shape: ``(total_voxels, channels)``.

        Returns:
            jagged_data (JaggedTensor): Data in jagged format matching the grid structure.
        """
        return JaggedTensor(impl=self._impl.jagged_like(data))

    def marching_cubes(
        self, field: JaggedTensor, level: float = 0.0
    ) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
        """
        Extract isosurface meshes over data associated with this :class:`GridBatch` using the marching cubes algorithm.
        Generates triangle meshes representing the isosurface at the specified level from a scalar field defined on the voxels.

        Args:
            field (JaggedTensor): Scalar field values at each voxel in this :class:`GridBatch`. A :class:`fvdb.JaggedTensor` with shape
                ``(batch_size, total_voxels, 1)``.
            level (float): The isovalue to extract the surface at. Default is ``0.0``.

        Returns:
            vertex_positions (JaggedTensor): Vertex positions of the meshes. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_vertices, 3)``.
            face_indices (JaggedTensor): Triangle face indices. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_faces, 3)``.
            vertex_normals (JaggedTensor): Vertex normals (computed from gradients). A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_vertices, 3)``.
        """
        result_vertices_impl, result_indices_impl, result_normals_impl = self._impl.marching_cubes(field._impl, level)
        return (
            JaggedTensor(impl=result_vertices_impl),
            JaggedTensor(impl=result_indices_impl),
            JaggedTensor(impl=result_normals_impl),
        )

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: JaggedTensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Apply max pooling to the given data associated with this :class:`GridBatch` returned as data associated
        with the given ``coarse_grid`` or a newly created coarse :class:`GridBatch`.

        Performs max pooling on the voxel data, reducing the resolution by the specified
        ``pool_factor``. Each output voxel contains the maximum of the corresponding input voxels
        within the pooling window. The pooling operation respects the sparse structure of this
        :class:`GridBatch` and the given ``coarse_grid``.

        .. note::

            If you pass ``coarse_grid = None``, the returned coarse grid batch will have its
            voxel sizes multiplied by the ``pool_factor`` and origins adjusted accordingly.

        .. note::

            This method supports backpropagation through the pooling operation.

        Args:
            pool_factor (NumericMaxRank1): The factor by which to downsample the grids, broadcastable to shape ``(3,)``, integer dtype
            data (JaggedTensor): The voxel data to pool. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels)``.
            stride (NumericMaxRank1): The stride to use when pooling. If ``0`` (default), stride equals ``pool_factor``, broadcastable to shape ``(3,)``, integer dtype
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid batch to use for output. If ``None``, a new :class:`GridBatch` is created.

        Returns:
            pooled_data (JaggedTensor): A :class:`fvdb.JaggedTensor` containing the pooled voxel data with shape ``(batch_size, coarse_total_voxels, channels)``.
            coarse_grid (GridBatch): A :class:`GridBatch` object representing the coarse grid batch topology after pooling. Matches the provided ``coarse_grid`` if given.
        """
        pool_factor = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)

        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data_impl, result_grid_impl = self._impl.max_pool(pool_factor, data._impl, stride, coarse_grid_impl)

        return JaggedTensor(impl=result_data_impl), GridBatch(impl=result_grid_impl)

    def merged_grid(self, other: "GridBatch") -> "GridBatch":
        """
        Return a grid batch that is the union of this grid batch with another.

        Merges two grid batches by taking the union of their active voxels.
        The grids must have compatible dimensions and transforms.

        Args:
            other (GridBatch): The other grid batch to merge with.

        Returns:
            merged_grid (GridBatch): A new GridBatch containing the union of active voxels from both grids.
        """
        return GridBatch(impl=self._impl.merged_grid(other._impl))

    def neighbor_indexes(self, ijk: JaggedTensor, extent: int, bitshift: int = 0) -> JaggedTensor:
        """
        Get indexes of neighboring voxels in this :class:`GridBatch` in an N-ring neighborhood of each
        voxel coordinate in ``ijk``.

        Args:
            ijk (JaggedTensor): Voxel coordinates to find neighbors for.
                Shape: ``(batch_size, num_queries, 3)`` with integer coordinates.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): Bit shift value for encoding. Default is 0.

        Returns:
            neighbor_indexes (JaggedTensor): A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_queries, N)``
                containing the linear indexes of neighboring voxels for each voxel coordinate in ``ijk``
                in the input. If some neighbors are not active in the grid, their indexes will be ``-1``.
        """
        return JaggedTensor(impl=self._impl.neighbor_indexes(ijk._impl, extent, bitshift))

    def num_voxels_at(self, bi: int) -> int:
        """
        Get the number of active voxels in a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            num_voxels (int): Number of active voxels in the specified grid.
        """
        return self._impl.num_voxels_at(bi)

    def pruned_grid(self, mask: JaggedTensor) -> "GridBatch":
        """
        Return a pruned grid based on a boolean mask.

        Creates a new grid containing only the voxels where the mask is True.

        Args:
            mask (JaggedTensor): Boolean mask for each voxel.
                Shape: ``(batch_size, total_voxels,)``.

        Returns:
            pruned_grid (GridBatch): A new GridBatch containing only voxels where mask is True.
        """
        return GridBatch(impl=self._impl.pruned_grid(mask._impl))

    def origin_at(self, bi: int) -> torch.Tensor:
        """
        Get the world-space origin of a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            origin (torch.Tensor): The origin coordinates in world space. Shape: ``(3,)``.
        """
        return self._impl.origin_at(bi)

    def points_in_grid(self, points: JaggedTensor) -> JaggedTensor:
        """
        Check if world-space points are located within active voxels.

        Tests whether the given points fall within voxels that are active in the grid.

        Args:
            points (JaggedTensor): World-space points to test.
                Shape: ``(batch_size, num_points, 3)``.

        Returns:
            mask (JaggedTensor): Boolean mask indicating which points are in active voxels.
                Shape: ``(batch_size, num_points,)``.
        """
        return JaggedTensor(impl=self._impl.points_in_grid(points._impl))

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
                Shape: ``(batch_size, num_rays, 3)``.
            ray_directions (JaggedTensor): Direction vectors of rays.
                Shape: ``(batch_size, num_rays, 3)``. Should be normalized.
            grid_scalars (JaggedTensor): Scalar field values at each voxel.
                Shape: ``(batch_size, total_voxels, 1)``.
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            intersections (JaggedTensor): Intersection information for each ray.
        """
        return JaggedTensor(
            impl=self._impl.ray_implicit_intersection(ray_origins._impl, ray_directions._impl, grid_scalars._impl, eps)
        )

    def inject_from_dense_cminor(self, dense_data: torch.Tensor, dense_origins: NumericMaxRank1 = 0) -> JaggedTensor:
        """
        Inject values from a dense :class:`torch.Tensor` into a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch`.

        This is the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order. *i.e.* the
        dense tensor has shape ``[batch_size, dense_size_x, dense_size_y, dense_size_z, channels*]``.

        .. note::

            This method supports backpropagation through the read operation.

        .. seealso::

            :meth:`inject_from_dense_cmajor` for the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order.

        .. seealso::

            :meth:`inject_to_dense_cminor` for writing data to a dense tensor in C Minor order.

        Args:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` to read from. Shape: ``(batch_size, dense_size_x, dense_size_y, dense_size_z, channels*)``.
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                voxel space, broadcastable to shape ``(3,)``, integer dtype. Default is ``(0, 0, 0)``.

        Returns:
            sparse_data (JaggedTensor): Values from the dense tensor at voxel locations active in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.
        """
        dense_origins = to_Vec3i(dense_origins)

        return JaggedTensor(impl=self._impl.read_from_dense_cminor(dense_data, dense_origins))

    def inject_from_dense_cmajor(self, dense_data: torch.Tensor, dense_origins: NumericMaxRank1 = 0) -> JaggedTensor:
        """
        Inject values from a dense :class:`torch.Tensor` into a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch`.

        This is the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order. *i.e.* the
        dense tensor has shape ``[batch_size, channels*, dense_size_x, dense_size_y, dense_size_z]``.

        .. note::

            This method supports backpropagation through the read operation.

        .. seealso::

            :meth:`inject_from_dense_cminor` for the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order.

        .. seealso::

            :meth:`inject_to_dense_cmajor` for writing data to a dense tensor in "C Major" order.

        Args:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` to read from. Shape: ``(batch_size, channels*, dense_size_x, dense_size_y, dense_size_z)``.
            dense_origins (NumericMaxRank1, optional): Origin of the dense tensor in
                voxel space, broadcastable to shape ``(3,)``, integer dtype. Default is ``(0, 0, 0)``.

        Returns:
            sparse_data (JaggedTensor): Values from the dense tensor at voxel locations active in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.
        """
        dense_origins = to_Vec3i(dense_origins)

        return JaggedTensor(impl=self._impl.read_from_dense_cmajor(dense_data, dense_origins))

    def sample_bezier(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """
        Sample data in a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` at world-space
        points using Bézier interpolation.

        This method uses Bézier interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grids
            return zero.

        .. seealso::

            :meth:`sample_trilinear` for trilinear interpolation.

        .. seealso::

            :meth:`sample_bezier_with_grad` for Bézier interpolation which also returns spatial gradients.

        Args:
            points (JaggedTensor): World-space points to sample at. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_data (JaggedTensor): Data associated with each voxel in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated data at each point. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.
        """
        return JaggedTensor(impl=self._impl.sample_bezier(points._impl, voxel_data._impl))

    def sample_bezier_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample data in a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` at world-space
        points using Bézier interpolation, and return the sampled values and their spatial gradients at those points.

        This method uses Bézier interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers. It returns both the interpolated data
        and the gradients of the interpolated data with respect to the world coordinates.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grids
            return zero.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. seealso::

            :meth:`sample_bezier` for Bézier interpolation without gradients.

        .. seealso::

            :meth:`sample_trilinear_with_grad` for trilinear interpolation with spatial gradients.

        Args:
            points (JaggedTensor): World-space points to sample at. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_data (JaggedTensor): Data associated with each voxel in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated data at each point. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.
            interpolation_gradients (JaggedTensor): Gradients of the interpolated data with respect to world coordinates.
                This is the spatial gradient of the Bézier interpolation at each point.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3, channels*)``.
        """
        result_data_impl, result_grad_impl = self._impl.sample_bezier_with_grad(points._impl, voxel_data._impl)
        return JaggedTensor(impl=result_data_impl), JaggedTensor(impl=result_grad_impl)

    def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """
        Sample data in a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` at world-space
        points using trilinear interpolation.

        This method uses trilinear interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grids
            return zero.

        .. seealso::

            :meth:`sample_bezier` for Bézier interpolation.

        .. seealso::

            :meth:`sample_trilinear_with_grad` for trilinear interpolation which also returns spatial gradients.

        Args:
            points (JaggedTensor): World-space points to sample at. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_data (JaggedTensor): Data associated with each voxel in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated data at each point. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.
        """
        return JaggedTensor(impl=self._impl.sample_trilinear(points._impl, voxel_data._impl))

    def sample_trilinear_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample data in a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` at world-space
        points using trilinear interpolation, and return the sampled values and their spatial gradients at those points.

        This method uses trilinear interpolation to interpolate data values at arbitrary continuous
        positions in world space, based on values defined at voxel centers. It returns both the interpolated data
        and the gradients of the interpolated data with respect to the world coordinates.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels. Samples outside the grids
            return zero.

        .. note::

            This method supports backpropagation through the interpolation operation.

        .. seealso::

            :meth:`sample_trilinear` for trilinear interpolation without gradients.

        .. seealso::

            :meth:`sample_bezier_with_grad` for Bézier interpolation with spatial gradients.

        Args:
            points (JaggedTensor): World-space points to sample at. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            voxel_data (JaggedTensor): Data associated with each voxel in this :class:`GridBatch`.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated data at each point. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.
            interpolation_gradients (JaggedTensor): Gradients of the interpolated data with respect to world coordinates.
                This is the spatial gradient of the trilinear interpolation at each point.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3, channels*)``.
        """
        result_data_impl, result_grad_impl = self._impl.sample_trilinear_with_grad(points._impl, voxel_data._impl)
        return JaggedTensor(impl=result_data_impl), JaggedTensor(impl=result_grad_impl)

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
                Shape: ``(batch_size, num_rays, 3)``.
            ray_directions (JaggedTensor): Direction of each ray.
                Shape: ``(batch_size, num_rays, 3)``.
            max_segments (int): Maximum number of segments to enumerate.
            eps (float): Small epsilon value to avoid numerical issues.

        Returns:
            ray_segments (JaggedTensor): A JaggedTensor containing the samples along the rays with lshape
                ``[[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]]`` and eshape
                ``(2,)`` representing the start and end distance of each segment.
        """
        return JaggedTensor(
            impl=self._impl.segments_along_rays(ray_origins._impl, ray_directions._impl, max_segments, eps)
        )

    def sparse_conv_halo(self, input: JaggedTensor, weight: torch.Tensor, variant: int = 8) -> JaggedTensor:
        """
        Perform sparse convolution with halo exchange optimization.

        Applies sparse convolution using halo exchange to efficiently handle boundary
        conditions in distributed or multi-block sparse grids.

        Args:
            input (JaggedTensor): Input features for each voxel.
                Shape: ``(batch_size, total_voxels, in_channels)``.
            weight (torch.Tensor): Convolution weights.
            variant (int): Variant of the halo implementation to use. Currently ``8`` and ``64``
                are supported. Default is ``8``.

        Returns:
            output (JaggedTensor): Output features after convolution.

        .. note::
            Currently only 3x3x3 kernels are supported.
        """
        return JaggedTensor(impl=self._impl.sparse_conv_halo(input._impl, weight, variant))

    def splat_bezier(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """
        Splat data at a set of input points into a :class:`fvdb.JaggedTensor` associated with
        this :class:`GridBatch` using Bézier interpolation. *i.e.* each point distributes its
        data to the surrounding voxels using cubic Bézier interpolation weights.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels.

        .. note::

            This method supports backpropagation through the splatting operation.

        Args:
            points (JaggedTensor): World-space positions of points used to splat data. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            points_data (JaggedTensor): Data associated with each point to splat into the grids.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.

        Returns:
            splatted_features (JaggedTensor): Accumulated features at each voxel after splatting.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.
        """
        return JaggedTensor(impl=self._impl.splat_bezier(points._impl, points_data._impl))

    def splat_trilinear(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """
        Splat data at a set of input points into a :class:`fvdb.JaggedTensor` associated with
        this :class:`GridBatch` using trilinear interpolation. *i.e.* each point distributes its
        data to the surrounding voxels using trilinear interpolation weights.

        .. note::

            This method assumes that the voxel data is defined at the centers of voxels.

        .. note::

            This method supports backpropagation through the splatting operation.

        Args:
            points (JaggedTensor): World-space positions of points used to splat data. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
            points_data (JaggedTensor): Data associated with each point to splat into the grids.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, channels*)``.

        Returns:
            splatted_features (JaggedTensor): Accumulated features at each voxel after splatting.
                A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels*)``.
        """
        return JaggedTensor(impl=self._impl.splat_trilinear(points._impl, points_data._impl))

    def refine(
        self,
        subdiv_factor: NumericMaxRank1,
        data: JaggedTensor,
        mask: JaggedTensor | None = None,
        fine_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Refine data associated with this :class:`GridBatch` into higher-resolution grids by subdividing each voxel.
        *i.e.* for each voxel, ``(i, j, k)`` in each grid of this :class:`GridBatch`, copy the data associated with that voxel to
        the voxels ``(subdiv_factor[0]*i + di, subdiv_factor[1]*j + dj, subdiv_factor[2]*k + dk)``
        for ``di, dj, dk`` in ``{0, ..., subdiv_factor - 1}`` in the output data associated with ``fine_grid``, if
        that voxel exists in the fine grid.

        .. note::

            If you pass ``fine_grid = None``, this method will create a new fine :class:`GridBatch` with its
            voxel sizes divided by the ``subdiv_factor`` and origins adjusted accordingly.

        .. note::

            You can skip copying data at certain voxels in this :class:`GridBatch` by passing a boolean ``mask``.
            Only data at voxels corresponding to ``True`` values in the mask will be refined.

        .. note::

            This method supports backpropagation through the refinement operation.

        .. seealso::

            :meth:`refined_grid` for obtaining a refined version of the grid structure without refining data.

        Args:
            subdiv_factor (NumericMaxRank1): Refinement factor between this :class:`GridBatch` and the fine grid batch, broadcastable to shape ``(3,)``, integer dtype
            data (JaggedTensor): Voxel data to refine. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, channels)``.
            mask (JaggedTensor, optional): Boolean mask indicating which voxels in the input grids to refine. If ``None``, data associated with all input voxels are refined.
            fine_grid (GridBatch, optional): Pre-allocated fine :class:`GridBatch` to use for output. If ``None``, a new :class:`GridBatch` is created.

        Returns:
            refined_data (JaggedTensor): The refined data as a :class:`fvdb.JaggedTensor`
            fine_grid (GridBatch): The fine :class:`GridBatch` containing the refined structure
        """
        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        fine_grid_impl = fine_grid._impl if fine_grid else None
        mask_impl = mask._impl if mask else None
        result_data_impl, result_grid_impl = self._impl.refine(subdiv_factor, data._impl, mask_impl, fine_grid_impl)
        return JaggedTensor(impl=result_data_impl), GridBatch(impl=result_grid_impl)

    def refined_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: JaggedTensor | None = None,
    ) -> "GridBatch":
        """
        Return a refined version of this :class:`GridBatch`. *i.e.* each voxel in each grid is subdivided
        by the specified ``subdiv_factor`` to create higher-resolution grids.

        .. note::

            You can skip refining certain voxels in this :class:`GridBatch` by passing a boolean
            ``mask``. Only voxels corresponding to ``True`` values in the mask will be refined.

        .. seealso::

            :meth:`refine` for copying data from a coarse :class:`GridBatch` to a refined :class:`GridBatch`.

        Args:
            subdiv_factor (NumericMaxRank1): Factor by which to refine each grid in the batch, broadcastable to shape ``(3,)``, integer dtype
            mask (JaggedTensor, optional): Boolean mask indicating which voxels to refine. If ``None``, all voxels are refined.

        Returns:
            refined_grid (GridBatch): A new :class:`GridBatch` with refined structure.
        """
        subdiv_factor = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
        return GridBatch(impl=self._impl.refined_grid(subdiv_factor, None if mask is None else mask._impl))

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
            grid_batch (GridBatch): A new GridBatch on the target device.
        """
        if isinstance(target, str):
            device = _parse_device_string(target)
            return GridBatch(impl=self._impl.to(device))
        elif isinstance(target, torch.device):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, torch.Tensor):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, JaggedTensor):
            return GridBatch(impl=self._impl.to(target._impl))
        elif isinstance(target, GridBatch):
            return GridBatch(impl=self._impl.to(target._impl))
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")

    def save_nanovdb(
        self,
        path: str,
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
            data (JaggedTensor | None): Voxel data to save with the grids.
                Shape: ``(batch_size, total_voxels, channels)``. If ``None``, only grid structure is saved.
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
        data_impl = data._impl if data else None
        if name is not None:
            _save(path, self._impl, data_impl, name, compressed, verbose)
        elif names is not None:
            if isinstance(names, str):
                # Handle case where names is actually a single name
                _save(path, self._impl, data_impl, names, compressed, verbose)
            else:
                # Handle case where names is a list
                _save(path, self._impl, data_impl, names, compressed, verbose)
        else:
            # Default case with empty names list
            _save(path, self._impl, data_impl, [], compressed, verbose)

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
        Generate uniformly spaced samples along rays intersecting the grids.

        Creates sample points at regular intervals along rays, but only for segments
        that intersect with active voxels. Useful for volume rendering and ray marching.

        Args:
            ray_origins (JaggedTensor): Starting points of rays in world space.
                Shape: ``(batch_size, num_rays, 3)``.
            ray_directions (JaggedTensor): Direction vectors of rays (should be normalized).
                Shape: ``(batch_size, num_rays, 3)``.
            t_min (JaggedTensor): Minimum distance along rays to start sampling.
                Shape: ``(batch_size, num_rays)``.
            t_max (JaggedTensor): Maximum distance along rays to stop sampling.
                Shape: ``(batch_size, num_rays)``.
            step_size (float): Distance between samples along each ray.
            cone_angle (float): Cone angle for cone tracing (in radians). Default is 0.0.
            include_end_segments (bool): Whether to include partial segments at ray ends.
                Default is True.
            return_midpoints (bool): Whether to return segment midpoints instead of start points.
                Default is False.
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            ray_samples (JaggedTensor): Ray samples containing the samples along the rays. A :class:`fvdb.JaggedTensor`
                with lshape ``[[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]]`` and eshape
                ``(2,)`` or ``(1,)`` representing the start and end distance of each sample or the midpoint
                of each sample if ``return_midpoints`` is ``True``.
        """
        return JaggedTensor(
            impl=self._impl.uniform_ray_samples(
                ray_origins._impl,
                ray_directions._impl,
                t_min._impl,
                t_max._impl,
                step_size,
                cone_angle,
                include_end_segments,
                return_midpoints,
                eps,
            )
        )

    def voxel_size_at(self, bi: int) -> torch.Tensor:
        """
        Get voxel size at a specific grid index.

        Args:
            bi (int): Grid index.

        Returns:
            voxel_size (torch.Tensor): Voxel size at the specified grid index.
                Shape: ``(3,)``.
        """
        return self._impl.voxel_size_at(bi)

    def rays_intersect_voxels(
        self, ray_origins: JaggedTensor, ray_directions: JaggedTensor, eps: float = 0.0
    ) -> JaggedTensor:
        """
        Return a boolean JaggedTensor recording whether a set of rays hit any voxels in this gridbatch.

        Args:
            ray_origins (JaggedTensor): A `JaggedTensor` of ray origins (one set of rays per grid in the batch).
                _i.e._ a `JaggedTensor` of the form `[ray_o0, ..., ray_oB]` where `ray_oI` has shape `[N_I, 3]`.
            ray_directions (JaggedTensor): A `JaggedTensor` of ray directions (one set of rays per grid in the batch).
                _i.e._ a `JaggedTensor` of the form `[ray_d0, ..., ray_dB]` where `ray_dI` has shape `[N_I, 3]`.
            eps (float): Epsilon value to skip intersections whose length is less than this value for
                numerical stability. Default is 0.0.
        Returns:
            hit_mask (JaggedTensor): A :class:`fvdb.JaggedTensor` indicating whether each ray hit a voxel.
                *i.e.* a boolean :class:`fvdb.JaggedTensor` of the form ``[hit_0, ..., hit_B]`` where ``hit_I`` has shape ``[N_I]``.
        """
        _, ray_times = self.voxels_along_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            max_voxels=1,
            eps=eps,
            return_ijk=False,
            cumulative=False,
        )

        did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
        return ray_origins.jagged_like(did_hit)

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
                Shape: ``(batch_size, num_rays, 3)``.
            ray_directions (JaggedTensor): Direction vectors of rays (should be normalized).
                Shape: ``(batch_size, num_rays, 3)``.
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Epsilon value for numerical stability. Default is 0.0.
            return_ijk (bool): Whether to return voxel indices. If False, returns
                linear indices instead. Default is True.
            cumulative (bool): Whether to return cumulative indices across the batch.
                Default is False.

        Returns:
            voxels (JaggedTensor): A JaggedTensor with lshape ``[[V_{0,0}, ..., V_{0,N_0}], ..., [V_{B,0}, ..., V_{B,N_B}]]``
                and eshape ``(3,)`` or ``(,)`` containing the ijk coordinates or indices of the voxels intersected by the rays.
            times (JaggedTensor): A JaggedTensor with lshape ``[[T_{0,0}, ..., T_{0,N_0}], ..., [T_{B,0}, ..., T_{B,N_B}]]``
                and eshape ``(2,)`` containing the entry and exit distance along the ray of each voxel.
        """
        result_voxels_impl, result_times_impl = self._impl.voxels_along_rays(
            ray_origins._impl, ray_directions._impl, max_voxels, eps, return_ijk, cumulative
        )
        return JaggedTensor(impl=result_voxels_impl), JaggedTensor(impl=result_times_impl)

    def world_to_voxel(self, points: JaggedTensor) -> JaggedTensor:
        """
        Convert world-space coordinates to voxel-space coordinates using each grid's transform.

        .. note::

            This method supports backpropagation through the transformation operation.

        .. seealso::

            :meth:`voxel_to_world` for the inverse transformation, and :attr:`voxel_to_world_matrices` and :attr:`world_to_voxel_matrices` for
            the actual transformation matrices.

        Args:
            points (JaggedTensor): Per-grid world-space positions to convert. Shape: ``(batch_size, num_points, 3)``.

        Returns:
            voxel_points (JaggedTensor): Grid coordinates. A :class:`fvdb.JaggedTensor` with shape ``(batch_size, num_points, 3)``.
                Can contain fractional values.
        """
        return JaggedTensor(impl=self._impl.world_to_grid(points._impl))

    def inject_to_dense_cminor(
        self,
        sparse_data: JaggedTensor,
        min_coord: NumericMaxRank2 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Inject values from a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` into a
        dense :class:`torch.Tensor`.

        This is the "C Minor" (channels minor) version, which assumes the ``dense_data`` is in XYZC order. *i.e.* the
        dense tensor has shape ``[batch_size, dense_size_x, dense_size_y, dense_size_z, channels*]``.

        This method creates the dense tensor to return, and fills it with values from the sparse grids
        within the range defined by ``min_coord`` and ``grid_size``.
        Voxels not present in the sparse grids are filled with zeros.

        .. note::

            This method supports backpropagation through the write operation.

        .. seealso::

            :meth:`inject_from_dense_cminor` for reading from a dense tensor in "C Minor" order,
            which assumes the dense tensor has shape ``[batch_size, dense_size_x, dense_size_y, dense_size_z, channels*]``.

        .. seealso::

            :meth:`inject_to_dense_cmajor` for writing to a dense tensor in "C Major" order.

        Args:
            sparse_data (JaggedTensor): A :class:`fvdb.JaggedTensor` of data associated with this :class:`GridBatch` with
                shape ``(batch_size, total_voxels, channels*)``.
            min_coord (NumericMaxRank2|None): Minimum voxel coordinate to read from each grid in the batch
                into the output dense tensor, broadcastable to shape ``(batch_size, 3)``, integer dtype, or ``None``.
                If set to ``None``, this will be the minimum voxel coordinate of each grid's bounding box.
            grid_size (NumericMaxRank1|None): Size of the output dense tensor, broadcastable to
                shape ``(3,)``, integer dtype, or ``None``. If ``None``, computed to fit all active
                voxels starting from ``min_coord``.

        Returns:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` containing the sparse data with
                shape ``(batch_size, dense_size_x, dense_size_y, dense_size_z, channels*)``.
        """
        min_coord = to_Vec3iBatchBroadcastable(min_coord) if min_coord is not None else None
        grid_size = to_Vec3iBroadcastable(grid_size) if grid_size is not None else None

        return self._impl.write_to_dense_cminor(sparse_data._impl, min_coord, grid_size)

    def inject_to_dense_cmajor(
        self,
        sparse_data: JaggedTensor,
        min_coord: NumericMaxRank2 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """
        Inject values from a :class:`fvdb.JaggedTensor` associated with this :class:`GridBatch` into a
        dense :class:`torch.Tensor`.

        This is the "C Major" (channels major) version, which assumes the ``dense_data`` is in CXYZ order. *i.e.* the
        dense tensor has shape ``[batch_size, channels*, dense_size_x, dense_size_y, dense_size_z]``.

        This method creates the dense tensor to return, and fills it with values from the sparse grids
        within the range defined by ``min_coord`` and ``grid_size``.
        Voxels not present in the sparse grids are filled with zeros.

        .. note::

            This method supports backpropagation through the write operation.

        .. seealso::

            :meth:`inject_from_dense_cmajor` for reading from a dense tensor in "C Major" order,
            which assumes the dense tensor has shape ``[batch_size, channels*, dense_size_x, dense_size_y, dense_size_z]``.

        .. seealso::

            :meth:`inject_to_dense_cminor` for writing to a dense tensor in "C Minor" order.

        Args:
            sparse_data (JaggedTensor): A :class:`fvdb.JaggedTensor` of data associated with this :class:`GridBatch` with
                shape ``(batch_size, total_voxels, channels*)``.
            min_coord (NumericMaxRank2|None): Minimum voxel coordinate to read from each grid in the batch
                into the output dense tensor, broadcastable to shape ``(batch_size, 3)``, integer dtype, or ``None``.
                If set to ``None``, this will be the minimum voxel coordinate of each grid's bounding box.
            grid_size (NumericMaxRank1|None): Size of the output dense tensor, broadcastable to
                shape ``(3,)``, integer dtype, or ``None``. If ``None``, computed to fit all active
                voxels starting from ``min_coord``.

        Returns:
            dense_data (torch.Tensor): Dense :class:`torch.Tensor` containing the sparse data with
                shape ``(batch_size, channels*, dense_size_x, dense_size_y, dense_size_z)``.
        """
        min_coord = to_Vec3iBatchBroadcastable(min_coord) if min_coord is not None else None
        grid_size = to_Vec3iBroadcastable(grid_size) if grid_size is not None else None

        return self._impl.write_to_dense_cmajor(sparse_data._impl, min_coord, grid_size)

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
            grid_batch (GridBatch): A new GridBatch containing the selected grid.
        """
        return GridBatch(impl=self._impl.index_int(int(bi)))

    def index_list(self, indices: list[bool] | list[int]) -> "GridBatch":
        """
        Get a subset of grids from the batch using list indexing.

        Args:
            indices (list[bool] | list[int]): List of indices.

        Returns:
            grid_batch (GridBatch): A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_list(indices))

    def index_slice(self, s: slice) -> "GridBatch":
        """
        Get a subset of grids from the batch using slicing.

        Args:
            s (slice): Slicing object.

        Returns:
            grid_batch (GridBatch): A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_slice(s))

    def index_tensor(self, indices: torch.Tensor) -> "GridBatch":
        """
        Get a subset of grids from the batch using tensor indexing.

        Args:
            indices (torch.Tensor): Tensor of indices.

        Returns:
            grid_batch (GridBatch): A new GridBatch containing the selected grids.
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
            grid_batch (GridBatch): A new GridBatch containing the selected grids.
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
            grid_batch (GridBatch): Single-grid batches for each grid in the batch.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """
        Get the number of grids in the batch.

        Returns:
            length (int): Number of grids in this batch.
        """
        return self._impl.grid_count

    # ============================================================
    #                        Properties
    # ============================================================

    # Properties
    @property
    def address(self) -> int:
        """
        The address of the underlying C++ NanoVDB grid batch object.

        Returns:
            address (int): The memory address of the underlying C++ object.
        """
        return self._impl.address

    @property
    def all_have_zero_voxels(self) -> bool:
        """
        ``True`` if all grids in this :class:`GridBatch` have zero active voxels, ``False`` otherwise.

        .. note::

            This returns ``True`` if the batch has zero grids or if all grids have zero voxels.

        Returns:
            all_have_zero_voxels (bool): Whether all grids have zero active voxels.
        """
        return self.has_zero_grids or self.total_voxels == 0

    @property
    def any_have_zero_voxels(self) -> bool:
        """
        ``True`` if at least one grid in this :class:`GridBatch` has zero active voxels, ``False`` otherwise.

        .. note::

            This returns ``True`` if the batch has zero grids or if any grid has zero voxels.

        Returns:
            any_have_zero_voxels (bool): Whether any grid has zero active voxels.
        """
        if self.has_zero_grids:
            return True
        else:
            return bool(torch.any(self.num_voxels == 0).item())

    @property
    def bboxes(self) -> torch.Tensor:
        """
        The voxel-space bounding boxes of each grid in this :class:`GridBatch`.

        .. note::

            The bounding boxes are inclusive of the minimum voxel and the maximum voxel.

            *e.g.* if a grid has a single voxel at index ``(0, 0, 0)``, its bounding box will be
            ``[[0, 0, 0], [0, 0, 0]]``.

            *e.g.* if a grid has voxels at indices ``(0, 0, 0)`` and ``(1, 1, 1)``, its bounding box will be
            ``[[0, 0, 0], [1, 1, 1]]``.

        Returns:
            bboxes (torch.Tensor): A ``(grid_count, 2, 3)``-shaped tensor where each entry represents
                the minimum and maximum voxel indices of the bounding box for each grid.
                If a grid has zero voxels, its bounding box is a zero tensor.
        """
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
        """
        The cumulative number of voxels up to and including each grid in this :class:`GridBatch`.

        .. note::

            This is useful for indexing into flattened data structures where all voxels
            from all grids are concatenated together.

        Returns:
            cum_voxels (torch.Tensor): A ``(grid_count,)``-shaped tensor where each element
                represents the cumulative sum of voxels up to and including that grid.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.cum_voxels

    @property
    def device(self) -> torch.device:
        """
        The :class:`torch.device` where this :class:`GridBatch` is stored.

        Returns:
            device (torch.device): The device of the batch.
        """
        return self._impl.device

    @property
    def dual_bboxes(self) -> torch.Tensor:
        """
        The voxel-space bounding boxes of the dual of each grid in this :class:`GridBatch`.
        *i.e.* the bounding boxes of the grids whose voxel centers correspond to voxel corners
        in the original grids.

        .. seealso::

            :attr:`bboxes` for the bounding boxes of the grids in this :class:`GridBatch`,
            and :meth:`dual_grid` for computing the dual grids.

        .. note::

            The bounding boxes are inclusive of the minimum voxel and the maximum voxel.

            *e.g.* if a grid has a single voxel at index ``(0, 0, 0)``, the dual grid will contain voxels
            at indices ``(0, 0, 0), (0, 0, 1), (0, 1, 0), ..., (1, 1, 1)``, and the bounding box will be
            ``[[0, 0, 0], [1, 1, 1]]``.

        Returns:
            dual_bboxes (torch.Tensor): A ``(grid_count, 2, 3)``-shaped tensor where each entry represents
                the minimum and maximum voxel indices of the dual bounding box for each grid.
                If a grid has zero voxels, its dual bounding box is a zero tensor.
        """
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
        """
        The number of grids in this :class:`GridBatch`.

        Returns:
            count (int): Number of grids.
        """
        return self._impl.grid_count

    @property
    def voxel_to_world_matrices(self) -> torch.Tensor:
        """
        The voxel-to-world transformation matrices for each grid in this :class:`GridBatch`,
        which transform voxel space coordinates to world space coordinates.

        Returns:
            voxel_to_world_matrices (torch.Tensor): A ``(grid_count, 4, 4)``-shaped tensor where
                each ``(4, 4)`` matrix represents the voxel-to-world transformation for a grid.
        """
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self._impl.grid_to_world_matrices

    @property
    def has_zero_grids(self) -> bool:
        """
        ``True`` if this :class:`GridBatch` contains zero grids, ``False`` otherwise.

        Returns:
            has_zero_grids (bool): Whether the batch has zero grids.
        """
        return self.grid_count == 0

    @property
    def ijk(self) -> JaggedTensor:
        """
        The voxel coordinates of every active voxel in each grid of this :class:`GridBatch`, in index order.

        Returns:
            ijk (JaggedTensor): A :class:`fvdb.JaggedTensor` with shape ``(batch_size, total_voxels, 3)`` containing the
                voxel coordinates of each active voxel in index order for each grid.
        """
        return JaggedTensor(impl=self._impl.ijk)

    @property
    def jidx(self) -> torch.Tensor:
        """
        The jagged index tensor indicating which grid each voxel belongs to.

        .. note::

            This property is part of the :class:`fvdb.JaggedTensor` structure and is useful for
            operations that need to know the grid index for each voxel.

        Returns:
            jidx (torch.Tensor): A ``(total_voxels,)``-shaped integer tensor where each element
                is the grid index (0 to grid_count-1) that the voxel at that position belongs to.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        else:
            return self._impl.jidx

    @property
    def joffsets(self) -> torch.Tensor:
        """
        The jagged offset tensor indicating the start index of voxels for each grid.

        .. note::

            This property is part of the :class:`fvdb.JaggedTensor` structure. The offsets
            define the boundaries between grids in a flattened voxel array.

        Returns:
            joffsets (torch.Tensor): A ``(grid_count + 1,)``-shaped integer tensor where
                ``joffsets[i]`` is the starting index of voxels for grid ``i`` in a flattened
                array, and ``joffsets[i+1] - joffsets[i]`` is the number of voxels in grid ``i``.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.joffsets

    @property
    def num_bytes(self) -> torch.Tensor:
        """
        The size in bytes each grid in this :class:`GridBatch` occupies in memory.

        Returns:
            num_bytes (torch.Tensor): A ``(grid_count,)``-shaped tensor containing the
                size in bytes of each grid.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_bytes

    @property
    def num_leaf_nodes(self) -> torch.Tensor:
        """
        The number of leaf nodes in the NanoVDB for each grid in this :class:`GridBatch`.

        Returns:
            num_leaf_nodes (torch.Tensor): A ``(grid_count,)``-shaped tensor containing the
                number of leaf nodes in each grid.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_leaf_nodes

    @property
    def num_voxels(self) -> torch.Tensor:
        """
        The number of active voxels in each grid of this :class:`GridBatch`.

        Returns:
            num_voxels (torch.Tensor): A ``(grid_count,)``-shaped tensor containing the
                number of active voxels in each grid.
        """
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self._impl.num_voxels

    @property
    def origins(self) -> torch.Tensor:
        """
        The world-space origin of each grid. The origin is the center of the ``[0,0,0]`` voxel.

        Returns:
            origins (torch.Tensor): A ``(grid_count, 3)``-shaped tensor of origins.
        """
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self._impl.origins

    @property
    def total_bbox(self) -> torch.Tensor:
        """
        The voxel-space bounding box that encompasses all grids in this :class:`GridBatch`.

        .. note::

            The bounding box is inclusive of the minimum voxel and the maximum voxel across all grids.

        Returns:
            total_bbox (torch.Tensor): A ``(2, 3)``-shaped tensor representing the minimum and
                maximum voxel indices of the bounding box that encompasses all grids in the batch.
                If all grids have zero voxels, returns a zero tensor.
        """
        if self.has_zero_grids or self.all_have_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self._impl.total_bbox

    @property
    def total_bytes(self) -> int:
        """
        The total size in bytes all grids in this :class:`GridBatch` occupy in memory.

        Returns:
            total_bytes (int): The total size in bytes of all grids in the batch.
        """
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_bytes

    @property
    def total_leaf_nodes(self) -> int:
        """
        The total number of leaf nodes in the NanoVDB across all grids in this :class:`GridBatch`.

        Returns:
            total_leaf_nodes (int): The total number of leaf nodes across all grids.
        """
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_leaf_nodes

    @property
    def total_voxels(self) -> int:
        """
        The total number of active voxels across all grids in the batch.

        Returns:
            total_voxels (int): Total active voxel count.
        """
        if self.has_zero_grids:
            return 0
        else:
            return self._impl.total_voxels

    @property
    def voxel_sizes(self) -> torch.Tensor:
        """
        The world-space voxel size of each grid in the batch.

        Returns:
            voxel_sizes (torch.Tensor): A ``(grid_count, 3)``-shaped tensor of voxel sizes.
        """
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self._impl.voxel_sizes

    @property
    def world_to_voxel_matrices(self) -> torch.Tensor:
        """
        The world-to-voxel transformation matrices for each grid in this :class:`GridBatch`,
        which transform world space coordinates to voxel space coordinates.

        Returns:
            world_to_voxel_matrices (torch.Tensor): A ``(grid_count, 4, 4)``-shaped tensor where
                each ``(4, 4)`` matrix represents the world-to-voxel transformation for a grid.
        """
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self._impl.world_to_grid_matrices

    # Expose underlying implementation for compatibility
    @property
    def _gridbatch(self):
        # Access underlying GridBatchCpp - use sparingly during migration
        return self._impl
