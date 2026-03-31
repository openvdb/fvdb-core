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
from typing import Any, Sequence, overload

import numpy as np
import torch

from . import _fvdb_cpp, _parse_device_string
from .jagged_tensor import JaggedTensor
from .types import (
    DeviceIdentifier,
    GridBatchIndex,
    NumericMaxRank1,
    NumericMaxRank2,
)


class GridBatch:
    """
    A batch of sparse voxel grids with support for efficient operations.

    :class:`GridBatch` represents a collection of sparse 3D voxel grids that can be processed
    together efficiently on GPU. Each grid in the batch can have different resolutions,
    origins, and voxel sizes. The class provides methods for common operations like
    sampling, convolution, pooling, dilation, union, etc. It also provides more advanced features
    such as marching cubes, TSDF fusion, and fast ray marching.

    A :class:`GridBatch` can be thought of as a mini-batch of sparse grids and
    does not collect the sparse voxel grids' data but only collects
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
            - :meth:`from_ijk()`: for a grid batch from explicit voxel coordinates
            - :meth:`from_mesh()`: for a grid batch from triangle meshes
            - :meth:`from_points()`: for a grid batch from point clouds
            - :meth:`from_nearest_voxels_to_points()`: for a grid batch from nearest voxels to points
            - :meth:`from_cat()`: for a grid batch from concatenating other grids and grid batches

    Attributes:
        max_grids_per_batch (int): Maximum number of grids that can be stored in a single :class:`fvdb.GridBatch`.


    """

    #: :meta private: # NOTE: This is here for sphinx to not complain that the attribute is double defined in the class and in the class documentation.
    max_grids_per_batch: int = _fvdb_cpp.GridBatchData.MAX_GRIDS_PER_BATCH

    def __init__(self, *, data: "_fvdb_cpp.GridBatchData") -> None:
        """Internal constructor -- use ``GridBatch.from_*`` classmethods instead.

        Args:
            data (_fvdb_cpp.GridBatchData): The underlying C++ grid batch data object.
        """
        self.data = data

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
        """Create a grid batch of dense grids from dimensions and an optional mask.

        Args:
            num_grids (int): Number of grids to create.
            dense_dims (NumericMaxRank1): Dimensions of the dense grid, broadcastable to shape ``(3,)``, integer dtype.
            ijk_min (NumericMaxRank1): Minimum voxel index for all grids, broadcastable to shape ``(3,)``, integer dtype.
            voxel_sizes (NumericMaxRank2): World-space size of each voxel, per-grid; broadcastable to
                shape ``(num_grids, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin of each grid, per-grid; broadcastable to
                shape ``(num_grids, 3)``, floating dtype.
            mask (torch.Tensor | None): Optional boolean mask with shape ``(W, H, D)`` selecting active voxels.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` of dense grids.

        .. seealso:: :meth:`Grid.from_dense`
        """
        from . import functional

        return functional.gridbatch_from_dense(num_grids, dense_dims, ijk_min, voxel_sizes, origins, mask, device)

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
        """Create a grid batch of dense grids defined by axis-aligned world-space bounds.

        Args:
            num_grids (int): Number of grids to create.
            dense_dims (NumericMaxRank1): Dimensions of the dense grids, broadcastable to shape ``(3,)``, integer dtype.
            bounds_min (NumericMaxRank1): Minimum world-space coordinate, broadcastable to shape ``(3,)``, floating dtype.
            bounds_max (NumericMaxRank1): Maximum world-space coordinate, broadcastable to shape ``(3,)``, floating dtype.
            voxel_center (bool): Whether bounds correspond to voxel centers (``True``) or edges (``False``).
            device (DeviceIdentifier): Device to create the grids on. Defaults to ``"cpu"``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` of dense grids.

        .. seealso:: :meth:`Grid.from_dense_axis_aligned_bounds`
        """

        from . import functional

        return functional.gridbatch_from_dense_axis_aligned_bounds(
            num_grids, dense_dims, bounds_min, bounds_max, voxel_center, device
        )

    @classmethod
    def from_ijk(
        cls,
        ijk: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """Create a grid batch from explicit voxel-space coordinates.

        Args:
            ijk (JaggedTensor): Per-grid voxel coordinates. Shape: ``(batch_size, num_voxels_for_grid_b, 3)``, integer dtype.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin of each grid, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with the specified voxel coordinates.

        .. seealso:: :meth:`Grid.from_ijk`
        """
        from . import functional

        return functional.gridbatch_from_ijk(ijk, voxel_sizes, origins, device)

    @classmethod
    def from_mesh(
        cls,
        mesh_vertices: JaggedTensor,
        mesh_faces: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """Create a grid batch by voxelizing the surface of triangle meshes.

        Args:
            mesh_vertices (JaggedTensor): Per-grid mesh vertex positions. Shape: ``(batch_size, num_vertices_for_grid_b, 3)``.
            mesh_faces (JaggedTensor): Per-grid mesh face indices. Shape: ``(batch_size, num_faces_for_grid_b, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin of each grid, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with voxels covering mesh surfaces.

        .. seealso:: :meth:`Grid.from_mesh`
        """
        from . import functional

        return functional.gridbatch_from_mesh(mesh_vertices, mesh_faces, voxel_sizes, origins, device)

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
            path (str): Path to the .nvdb file to load.
            indices (list[int] | None): Optional list of grid indices to load.
            index (int | None): Optional single grid index to load.
            names (list[str] | None): Optional list of grid names to load.
            name (str | None): Optional single grid name to load.
            device (DeviceIdentifier): Device to load the grid batch on. Defaults to ``"cpu"``.
            verbose (bool): If ``True``, print information about the loaded grids.

        Returns:
            grid_batch (GridBatch): A :class:`GridBatch` containing the loaded grids.
            data (JaggedTensor): A :class:`JaggedTensor` containing voxel data.
            names (list[str]): Names of each loaded grid.

        .. seealso:: :meth:`Grid.from_nanovdb`
        """
        from . import functional

        if indices is not None:
            return functional.load_nanovdb(path, indices=indices, device=device, verbose=verbose)
        elif index is not None:
            return functional.load_nanovdb(path, index=index, device=device, verbose=verbose)
        elif names is not None:
            return functional.load_nanovdb(path, names=names, device=device, verbose=verbose)
        elif name is not None:
            return functional.load_nanovdb(path, name=name, device=device, verbose=verbose)
        else:
            return functional.load_nanovdb(path, device=device, verbose=verbose)

    @classmethod
    def from_nearest_voxels_to_points(
        cls,
        points: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """Create a grid batch by adding the eight nearest voxels to every input point.

        Args:
            points (JaggedTensor): Per-grid world-space point positions. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin of each grid, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with voxels surrounding each point.

        .. seealso:: :meth:`Grid.from_nearest_voxels_to_points`
        """
        from . import functional

        return functional.gridbatch_from_nearest_voxels_to_points(points, voxel_sizes, origins, device)

    @classmethod
    def from_points(
        cls,
        points: JaggedTensor,
        voxel_sizes: NumericMaxRank2 = 1,
        origins: NumericMaxRank2 = 0,
        device: DeviceIdentifier | None = None,
    ) -> "GridBatch":
        """Create a grid batch from point clouds by voxelizing each point's location.

        Args:
            points (JaggedTensor): Per-grid world-space point positions. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_sizes (NumericMaxRank2): Size of each voxel, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin of each grid, per-grid; broadcastable to shape ``(batch_size, 3)``, floating dtype.
            device (DeviceIdentifier | None): Device to create the grid batch on. Defaults to ``None``.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with one voxel per occupied point location.

        .. seealso:: :meth:`Grid.from_points`
        """
        from . import functional

        return functional.gridbatch_from_points(points, voxel_sizes, origins, device)

    @classmethod
    def from_zero_grids(cls, device: DeviceIdentifier = "cpu") -> "GridBatch":
        """Create an empty grid batch containing zero grids.

        Args:
            device (DeviceIdentifier): Device to create the grid batch on. Defaults to ``"cpu"``.

        Returns:
            grid_batch (GridBatch): A new empty :class:`GridBatch` with ``grid_count == 0``.
        """
        from . import functional

        return functional.gridbatch_from_zero_grids(device)

    @classmethod
    def from_zero_voxels(
        cls, device: DeviceIdentifier = "cpu", voxel_sizes: NumericMaxRank2 = 1, origins: NumericMaxRank2 = 0
    ) -> "GridBatch":
        """Create a grid batch with one or more grids that each have zero active voxels.

        Args:
            device (DeviceIdentifier): Device to create the grid batch on. Defaults to ``"cpu"``.
            voxel_sizes (NumericMaxRank2): Voxel size per grid, broadcastable to shape ``(num_grids, 3)``, floating dtype.
            origins (NumericMaxRank2): World-space origin per grid, broadcastable to shape ``(num_grids, 3)``, floating dtype.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with zero-voxel grids.

        .. seealso:: :meth:`Grid.from_zero_voxels`
        """
        from . import functional

        return functional.gridbatch_from_zero_voxels(device, voxel_sizes, origins)

    @classmethod
    def from_cat(cls, grids: "Sequence[GridBatch]") -> "GridBatch":
        """Create a grid batch by concatenating a sequence of grid batches along the batch dimension.

        Args:
            grids (Sequence[GridBatch]): Grid batches to concatenate.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing all grids from the inputs.
        """
        from . import functional

        return functional.concatenate_grids(grids)

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
        """Apply average pooling to voxel data associated with this grid batch.

        Supports backpropagation.

        Args:
            pool_factor (NumericMaxRank1): Downsample factor, broadcastable to shape ``(3,)``, integer dtype.
            data (JaggedTensor): Voxel data to pool. Shape: ``(batch_size, total_voxels, channels)``.
            stride (NumericMaxRank1): Pooling stride; if ``0``, equals ``pool_factor``. Broadcastable to shape ``(3,)``, integer dtype.
            coarse_grid (GridBatch | None): Optional pre-allocated coarse grid batch for output.

        Returns:
            pooled_data (JaggedTensor): Pooled voxel data. Shape: ``(batch_size, coarse_total_voxels, channels)``.
            coarse_grid (GridBatch): The coarse grid batch topology after pooling.

        .. seealso:: :meth:`Grid.avg_pool`
        """
        from . import functional

        return functional.avg_pool_batch(self, pool_factor, data, stride, coarse_grid)

    def bbox_at(self, bi: int) -> torch.Tensor:
        """Get the voxel-space bounding box of a specific grid in this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            bbox (torch.Tensor): Bounding box of shape ``(2, 3)`` as ``[[bmin_i, bmin_j, bmin_k], [bmax_i, bmax_j, bmax_k]]``.
        """
        # There's a quirk with zero-voxel grids that we handle here.
        if self.has_zero_voxels_at(bi):
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self.data.bbox_at(bi)

    def clip(
        self, features: JaggedTensor, ijk_min: NumericMaxRank2, ijk_max: NumericMaxRank2
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """Clip voxels and their features to a bounding box range for this grid batch.

        Supports backpropagation.

        Args:
            features (JaggedTensor): Voxel features to clip. Shape: ``(batch_size, total_voxels, channels)``.
            ijk_min (NumericMaxRank2): Minimum voxel-space bounds, broadcastable to shape ``(batch_size, 3)``, integer dtype.
            ijk_max (NumericMaxRank2): Maximum voxel-space bounds, broadcastable to shape ``(batch_size, 3)``, integer dtype.

        Returns:
            clipped_features (JaggedTensor): Clipped voxel features. Shape: ``(batch_size, clipped_total_voxels, channels)``.
            clipped_grid (GridBatch): A new :class:`GridBatch` containing only voxels within bounds.

        .. seealso:: :meth:`Grid.clip`
        """
        from . import functional

        return functional.clip_batch(self, features, ijk_min, ijk_max)

    def clipped_grid(
        self,
        ijk_min: NumericMaxRank2,
        ijk_max: NumericMaxRank2,
    ) -> "GridBatch":
        """Return a new grid batch clipped to a voxel-space bounding box for this grid batch.

        Args:
            ijk_min (NumericMaxRank2): Minimum voxel-space bounds, broadcastable to shape ``(batch_size, 3)``, integer dtype.
            ijk_max (NumericMaxRank2): Maximum voxel-space bounds, broadcastable to shape ``(batch_size, 3)``, integer dtype.

        Returns:
            clipped_grid (GridBatch): A new :class:`GridBatch` containing only voxels within bounds.

        .. seealso:: :meth:`Grid.clipped_grid`
        """
        from . import functional

        return functional.clipped_grid_batch(self, ijk_min, ijk_max)

    def coarsened_grid(self, coarsening_factor: NumericMaxRank1) -> "GridBatch":
        """Return a coarsened version of this grid batch by keeping every N-th voxel.

        Args:
            coarsening_factor (NumericMaxRank1): Coarsening factor, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            coarsened_grid (GridBatch): A new coarsened :class:`GridBatch`.

        .. seealso:: :meth:`Grid.coarsened_grid`
        """
        from . import functional

        return functional.coarsened_grid_batch(self, coarsening_factor)

    def contiguous(self) -> "GridBatch":
        """Return a contiguous copy of this grid batch in memory.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` with contiguous memory layout.

        .. seealso:: :meth:`Grid.contiguous`
        """
        from . import functional

        return functional.contiguous_batch(self)

    def conv_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> "GridBatch":
        """Return the output grid topology for a convolution applied to this grid batch.

        Args:
            kernel_size (NumericMaxRank1): Convolution kernel size, broadcastable to shape ``(3,)``, integer dtype.
            stride (NumericMaxRank1): Convolution stride, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            conv_grid (GridBatch): A :class:`GridBatch` representing the convolution output topology.

        .. seealso:: :meth:`Grid.conv_grid`
        """
        from . import functional

        return functional.conv_grid_batch(self, kernel_size, stride)

    def conv_transpose_grid(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> "GridBatch":
        """Return the output grid topology for a transposed convolution applied to this grid batch.

        Args:
            kernel_size (NumericMaxRank1): Convolution kernel size, broadcastable to shape ``(3,)``, integer dtype.
            stride (NumericMaxRank1): Convolution stride, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            conv_transpose_grid (GridBatch): A :class:`GridBatch` representing the transposed convolution output topology.

        .. seealso:: :meth:`Grid.conv_transpose_grid`
        """
        from . import functional

        return functional.conv_transpose_grid_batch(self, kernel_size, stride)

    def coords_in_grid(self, ijk: JaggedTensor) -> JaggedTensor:
        """Check which voxel-space coordinates lie on active voxels in this grid batch.

        Args:
            ijk (JaggedTensor): Voxel coordinates to test. Shape: ``(batch_size, num_queries_for_grid_b, 3)``, integer dtype.

        Returns:
            mask (JaggedTensor): Boolean mask indicating active voxel hits. Shape: ``(batch_size, num_queries_for_grid_b)``.

        .. seealso:: :meth:`Grid.coords_in_grid`
        """
        from . import functional

        return functional.coords_in_grid_batch(self, ijk)

    def cpu(self) -> "GridBatch":
        """Move this grid batch to CPU.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` on CPU device.

        .. seealso:: :meth:`Grid.cpu`
        """
        return self.to("cpu")

    def cubes_in_grid(
        self, cube_centers: JaggedTensor, cube_min: NumericMaxRank1 = 0, cube_max: NumericMaxRank1 = 0
    ) -> JaggedTensor:
        """Check if axis-aligned cubes are fully contained within active voxels of this grid batch.

        Args:
            cube_centers (JaggedTensor): Cube centers in world coordinates. Shape: ``(batch_size, num_cubes_for_grid_b, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to shape ``(3,)``, floating dtype.
            cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            mask (JaggedTensor): Boolean mask of fully contained cubes. Shape: ``(batch_size, num_cubes_for_grid_b)``.

        .. seealso:: :meth:`Grid.cubes_in_grid`
        """
        from . import functional

        return functional.cubes_in_grid_batch(self, cube_centers, cube_min, cube_max)

    def cubes_intersect_grid(
        self, cube_centers: JaggedTensor, cube_min: NumericMaxRank1 = 0, cube_max: NumericMaxRank1 = 0
    ) -> JaggedTensor:
        """Check if axis-aligned cubes intersect any active voxels in this grid batch.

        Args:
            cube_centers (JaggedTensor): Cube centers in world coordinates. Shape: ``(batch_size, num_cubes_for_grid_b, 3)``.
            cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to shape ``(3,)``, floating dtype.
            cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to shape ``(3,)``, floating dtype.

        Returns:
            mask (JaggedTensor): Boolean mask of intersecting cubes. Shape: ``(batch_size, num_cubes_for_grid_b)``.

        .. seealso:: :meth:`Grid.cubes_intersect_grid`
        """
        from . import functional

        return functional.cubes_intersect_grid_batch(self, cube_centers, cube_min, cube_max)

    def cuda(self) -> "GridBatch":
        """Move this grid batch to CUDA device.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` on CUDA device.

        .. seealso:: :meth:`Grid.cuda`
        """
        return self.to("cuda")

    def cum_voxels_at(self, bi: int) -> int:
        """Get the cumulative voxel count up to and including a specific grid in this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            cum_voxels (int): Cumulative number of voxels up to and including grid ``bi``.
        """
        return self.data.cum_voxels_at(bi)

    def dilated_grid(self, dilation: int) -> "GridBatch":
        """Return a dilated version of this grid batch by expanding active regions.

        Args:
            dilation (int): Dilation radius in voxels.

        Returns:
            dilated_grid (GridBatch): A new :class:`GridBatch` with dilated active regions.

        .. seealso:: :meth:`Grid.dilated_grid`
        """
        from . import functional

        return functional.dilated_grid_batch(self, dilation)

    def dual_bbox_at(self, bi: int) -> torch.Tensor:
        """Get the dual voxel-space bounding box of a specific grid in this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            dual_bbox (torch.Tensor): Dual bounding box of shape ``(2, 3)``.
        """
        if self.has_zero_voxels_at(bi):
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self.data.dual_bbox_at(bi)

    def dual_grid(self, exclude_border: bool = False) -> "GridBatch":
        """Return the dual grid of this grid batch where voxel centers correspond to primal voxel corners.

        Args:
            exclude_border (bool): If ``True``, excludes border voxels beyond primal grid bounds.

        Returns:
            dual_grid (GridBatch): A new :class:`GridBatch` representing the dual grid.

        .. seealso:: :meth:`Grid.dual_grid`
        """
        from . import functional

        return functional.dual_grid_batch(self, exclude_border)

    def voxel_to_world(self, ijk: JaggedTensor) -> JaggedTensor:
        """Transform voxel-space coordinates to world-space positions for this grid batch.

        Supports backpropagation.

        Args:
            ijk (JaggedTensor): Voxel-space coordinates to convert. Shape: ``(batch_size, num_points_for_grid_b, 3)``.

        Returns:
            world_coords (JaggedTensor): World-space coordinates. Shape: ``(batch_size, num_points_for_grid_b, 3)``.

        .. seealso:: :meth:`Grid.voxel_to_world`
        """
        from . import functional

        return functional.voxel_to_world_batch(self, ijk)

    def has_same_address_and_grid_count(self, other: Any) -> bool:
        """Check if another object shares the same underlying data address and grid count as this grid batch.

        Args:
            other (Any): Object to compare with.

        Returns:
            result (bool): ``True`` if both address and grid count match, ``False`` otherwise.

        .. seealso:: :meth:`Grid.has_same_address_and_grid_count`
        """
        if isinstance(other, GridBatch):
            return self.address == other.address and self.grid_count == other.grid_count
        else:
            return False

    def has_zero_voxels_at(self, bi: int) -> bool:
        """Check if a specific grid in this grid batch has zero active voxels.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            is_empty (bool): ``True`` if the grid has zero voxels, ``False`` otherwise.
        """
        return self.num_voxels_at(bi) == 0

    def ijk_to_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """Convert voxel-space coordinates to linear indices for this grid batch.

        Args:
            ijk (JaggedTensor): Voxel coordinates to convert. Shape: ``(batch_size, num_queries_for_grid_b, 3)``, integer dtype.
            cumulative (bool): If ``True``, return batch-cumulative indices; otherwise per-grid.

        Returns:
            indices (JaggedTensor): Linear indices, or ``-1`` for inactive voxels. Shape: ``(batch_size, num_queries_for_grid_b)``.

        .. seealso:: :meth:`Grid.ijk_to_index`
        """
        from . import functional

        return functional.ijk_to_index_batch(self, ijk, cumulative)

    def ijk_to_inv_index(self, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor:
        """Get the inverse permutation of :meth:`ijk_to_index` for this grid batch.

        Args:
            ijk (JaggedTensor): Voxel coordinates to convert. Shape: ``(batch_size, num_queries_for_grid_b, 3)``, integer dtype.
            cumulative (bool): If ``True``, return batch-cumulative indices; otherwise per-grid.

        Returns:
            inv_map (JaggedTensor): Inverse permutation indices. Shape: ``(batch_size, num_queries_for_grid_b)``.

        .. seealso:: :meth:`Grid.ijk_to_inv_index`
        """
        from . import functional

        return functional.ijk_to_inv_index_batch(self, ijk, cumulative)

    def inject_from(
        self,
        src_grid: "GridBatch",
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ) -> JaggedTensor:
        """Inject data from a source grid batch into this grid batch in voxel space.

        Supports backpropagation.

        Args:
            src_grid (GridBatch): Source grid batch to inject data from.
            src (JaggedTensor): Source data. Shape: ``(batch_size, src_grid.total_voxels, *)``.
            dst (JaggedTensor | None): Optional destination data modified in-place, or ``None`` to create new.
            default_value (float | int | bool): Fill value for unmapped voxels when ``dst`` is ``None``.

        Returns:
            dst (JaggedTensor): Data after injection into this grid batch's topology.

        .. seealso:: :meth:`Grid.inject_from`
        """
        from . import functional

        return functional.inject_batch(self, src_grid, src, dst, default_value)

    def inject_from_ijk(
        self,
        src_ijk: JaggedTensor,
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ):
        """Inject data from explicit voxel coordinates into this grid batch.

        Supports backpropagation.

        Args:
            src_ijk (JaggedTensor): Source voxel coordinates. Shape: ``(batch_size, num_src_voxels, 3)``.
            src (JaggedTensor): Source data to inject. Shape: ``(batch_size, num_src_voxels, *)``.
            dst (JaggedTensor | None): Optional destination data modified in-place, or ``None`` to create new.
            default_value (float | int | bool): Fill value for unmapped voxels when ``dst`` is ``None``.

        Returns:
            dst (JaggedTensor): Data after injection into this grid batch's topology.

        .. seealso:: :meth:`Grid.inject_from_ijk`
        """
        from . import functional

        return functional.inject_from_ijk_batch(self, src_ijk, src, dst, default_value)

    def inject_to(
        self,
        dst_grid: "GridBatch",
        src: JaggedTensor,
        dst: JaggedTensor | None = None,
        default_value: float | int | bool = 0,
    ) -> JaggedTensor:
        """Inject data from this grid batch into a destination grid batch in voxel space.

        Supports backpropagation.

        Args:
            dst_grid (GridBatch): Destination grid batch to inject data into.
            src (JaggedTensor): Source data from this grid batch. Shape: ``(batch_size, total_voxels, *)``.
            dst (JaggedTensor | None): Optional destination data modified in-place, or ``None`` to create new.
            default_value (float | int | bool): Fill value for unmapped voxels when ``dst`` is ``None``.

        Returns:
            dst (JaggedTensor): Data after injection into the destination grid batch's topology.

        .. seealso:: :meth:`Grid.inject_to`
        """
        from . import functional

        return functional.inject_batch(dst_grid, self, src, dst, default_value)

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
        """Integrate depth images into a TSDF volume for this grid batch.

        Args:
            truncation_distance (float): Maximum TSDF truncation distance in world units.
            projection_matrices (torch.Tensor): Camera projection matrices. Shape: ``(batch_size, 3, 3)``.
            cam_to_world_matrices (torch.Tensor): Camera-to-world transforms. Shape: ``(batch_size, 4, 4)``.
            tsdf (JaggedTensor): Current TSDF values. Shape: ``(batch_size, total_voxels, 1)``.
            weights (JaggedTensor): Current integration weights. Shape: ``(batch_size, total_voxels, 1)``.
            depth_images (torch.Tensor): Depth images. Shape: ``(batch_size, height, width)``.
            weight_images (torch.Tensor | None): Per-pixel weights, or ``None`` for uniform.

        Returns:
            updated_grid (GridBatch): Updated :class:`GridBatch` with potentially expanded voxels.
            updated_tsdf (JaggedTensor): Updated TSDF values.
            updated_weights (JaggedTensor): Updated integration weights.

        .. seealso:: :meth:`Grid.integrate_tsdf`
        """

        from . import functional

        return functional.integrate_tsdf_batch(
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
        tsdf: JaggedTensor,
        features: JaggedTensor,
        weights: JaggedTensor,
        depth_images: torch.Tensor,
        feature_images: torch.Tensor,
        weight_images: torch.Tensor | None = None,
    ) -> tuple["GridBatch", JaggedTensor, JaggedTensor, JaggedTensor]:
        """Integrate depth and feature images into a TSDF volume for this grid batch.

        Args:
            truncation_distance (float): Maximum TSDF truncation distance in world units.
            projection_matrices (torch.Tensor): Camera projection matrices. Shape: ``(batch_size, 3, 3)``.
            cam_to_world_matrices (torch.Tensor): Camera-to-world transforms. Shape: ``(batch_size, 4, 4)``.
            tsdf (JaggedTensor): Current TSDF values. Shape: ``(batch_size, total_voxels, 1)``.
            features (JaggedTensor): Current feature values. Shape: ``(batch_size, total_voxels, feature_dim)``.
            weights (JaggedTensor): Current integration weights. Shape: ``(batch_size, total_voxels, 1)``.
            depth_images (torch.Tensor): Depth images. Shape: ``(batch_size, height, width)``.
            feature_images (torch.Tensor): Feature images (e.g., RGB). Shape: ``(batch_size, height, width, feature_dim)``.
            weight_images (torch.Tensor | None): Per-pixel weights, or ``None`` for uniform.

        Returns:
            updated_grid (GridBatch): Updated :class:`GridBatch` with potentially expanded voxels.
            updated_tsdf (JaggedTensor): Updated TSDF values.
            updated_weights (JaggedTensor): Updated integration weights.
            updated_features (JaggedTensor): Updated per-voxel features.

        .. seealso:: :meth:`Grid.integrate_tsdf_with_features`
        """
        from . import functional

        return functional.integrate_tsdf_with_features_batch(
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

    def is_contiguous(self) -> bool:
        """Check if this grid batch is stored contiguously in memory.

        Returns:
            is_contiguous (bool): ``True`` if data is contiguous, ``False`` otherwise.

        .. seealso:: :meth:`Grid.is_contiguous`
        """
        return self.data.is_contiguous

    def is_same(self, other: "GridBatch") -> bool:
        """Check if another grid batch shares the same underlying data in memory as this grid batch.

        Args:
            other (GridBatch): Grid batch to compare with.

        Returns:
            is_same (bool): ``True`` if both share the same underlying data, ``False`` otherwise.

        .. seealso:: :meth:`Grid.is_same`
        """
        return self.data.is_same(other.data)

    def jagged_like(self, data: torch.Tensor) -> JaggedTensor:
        """Create a :class:`JaggedTensor` with the same jagged structure as this grid batch.

        Args:
            data (torch.Tensor): Dense data to wrap. Shape: ``(total_voxels, channels)``.

        Returns:
            jagged_data (JaggedTensor): Data in jagged format matching this grid batch's structure.
        """
        return JaggedTensor(impl=self.data.jagged_like(data))

    def marching_cubes(
        self, field: JaggedTensor, level: float = 0.0
    ) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
        """Extract isosurface meshes from a scalar field on this grid batch using marching cubes.

        Args:
            field (JaggedTensor): Scalar field values per voxel. Shape: ``(batch_size, total_voxels, 1)``.
            level (float): Isovalue at which to extract the surface.

        Returns:
            vertex_positions (JaggedTensor): Mesh vertex positions. Shape: ``(batch_size, num_vertices_for_grid_b, 3)``.
            face_indices (JaggedTensor): Triangle face indices. Shape: ``(batch_size, num_faces_for_grid_b, 3)``.
            vertex_normals (JaggedTensor): Vertex normals. Shape: ``(batch_size, num_vertices_for_grid_b, 3)``.

        .. seealso:: :meth:`Grid.marching_cubes`
        """
        from . import functional

        return functional.marching_cubes_batch(self, field, level)

    def max_pool(
        self,
        pool_factor: NumericMaxRank1,
        data: JaggedTensor,
        stride: NumericMaxRank1 = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """Apply max pooling to voxel data associated with this grid batch.

        Supports backpropagation.

        Args:
            pool_factor (NumericMaxRank1): Downsample factor, broadcastable to shape ``(3,)``, integer dtype.
            data (JaggedTensor): Voxel data to pool. Shape: ``(batch_size, total_voxels, channels)``.
            stride (NumericMaxRank1): Pooling stride; if ``0``, equals ``pool_factor``. Broadcastable to shape ``(3,)``, integer dtype.
            coarse_grid (GridBatch | None): Optional pre-allocated coarse grid batch for output.

        Returns:
            pooled_data (JaggedTensor): Pooled voxel data. Shape: ``(batch_size, coarse_total_voxels, channels)``.
            coarse_grid (GridBatch): The coarse grid batch topology after pooling.

        .. seealso:: :meth:`Grid.max_pool`
        """
        from . import functional

        return functional.max_pool_batch(self, pool_factor, data, stride, coarse_grid)

    def merged_grid(self, other: "GridBatch") -> "GridBatch":
        """Return the union of this grid batch with another grid batch.

        Args:
            other (GridBatch): Grid batch to merge with.

        Returns:
            merged_grid (GridBatch): A new :class:`GridBatch` containing the union of active voxels.

        .. seealso:: :meth:`Grid.merged_grid`
        """
        from . import functional

        return functional.merged_grid_batch(self, other)

    def neighbor_indexes(self, ijk: JaggedTensor, extent: int, bitshift: int = 0) -> JaggedTensor:
        """Get N-ring neighbor indices for voxel coordinates in this grid batch.

        Args:
            ijk (JaggedTensor): Voxel coordinates to find neighbors for. Shape: ``(batch_size, num_queries_for_grid_b, 3)``, integer dtype.
            extent (int): Neighborhood ring size.
            bitshift (int): Bit shift value for encoding.

        Returns:
            neighbor_indexes (JaggedTensor): Neighbor linear indices (``-1`` for inactive). Shape: ``(batch_size, num_queries_for_grid_b, N)``.

        .. seealso:: :meth:`Grid.neighbor_indexes`
        """
        from . import functional

        return functional.neighbor_indexes_batch(self, ijk, extent, bitshift)

    def num_voxels_at(self, bi: int) -> int:
        """Get the number of active voxels in a specific grid of this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            num_voxels (int): Number of active voxels in the specified grid.
        """
        return self.data.num_voxels_at(bi)

    def pruned_grid(self, mask: JaggedTensor) -> "GridBatch":
        """Return a pruned version of this grid batch keeping only masked voxels.

        Args:
            mask (JaggedTensor): Boolean mask per voxel. Shape: ``(batch_size, total_voxels)``.

        Returns:
            pruned_grid (GridBatch): A new :class:`GridBatch` containing only voxels where mask is ``True``.

        .. seealso:: :meth:`Grid.pruned_grid`
        """
        from . import functional

        return functional.pruned_grid_batch(self, mask)

    def origin_at(self, bi: int) -> torch.Tensor:
        """Get the world-space origin of a specific grid in this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            origin (torch.Tensor): Origin coordinates in world space. Shape: ``(3,)``.
        """
        return self.data.origin_at(bi)

    def points_in_grid(self, points: JaggedTensor) -> JaggedTensor:
        """Check if world-space points lie within active voxels of this grid batch.

        Args:
            points (JaggedTensor): World-space points to test. Shape: ``(batch_size, num_points_for_grid_b, 3)``.

        Returns:
            mask (JaggedTensor): Boolean mask of points in active voxels. Shape: ``(batch_size, num_points_for_grid_b)``.

        .. seealso:: :meth:`Grid.points_in_grid`
        """
        from . import functional

        return functional.points_in_grid_batch(self, points)

    def ray_implicit_intersection(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        grid_scalars: JaggedTensor,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """Find ray intersections with an implicit surface defined by scalar voxel data in this grid batch.

        Args:
            ray_origins (JaggedTensor): Ray origins in world space. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            ray_directions (JaggedTensor): Ray directions (should be normalized). Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            grid_scalars (JaggedTensor): Scalar field values per voxel. Shape: ``(batch_size, total_voxels, 1)``.
            eps (float): Epsilon for numerical stability.

        Returns:
            intersections (JaggedTensor): Intersection information for each ray.

        .. seealso:: :meth:`Grid.ray_implicit_intersection`
        """
        from . import functional

        return functional.ray_implicit_intersection_batch(self, ray_origins, ray_directions, grid_scalars, eps)

    def inject_from_dense_cminor(self, dense_data: torch.Tensor, dense_origins: NumericMaxRank1 = 0) -> JaggedTensor:
        """Inject values from a dense XYZC-ordered tensor into a :class:`JaggedTensor` for this grid batch.

        Supports backpropagation.

        Args:
            dense_data (torch.Tensor): Dense tensor in XYZC order. Shape: ``(batch_size, X, Y, Z, channels*)``.
            dense_origins (NumericMaxRank1): Dense tensor origin in voxel space, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            sparse_data (JaggedTensor): Values at active voxel locations. Shape: ``(batch_size, total_voxels, channels*)``.

        .. seealso:: :meth:`Grid.inject_from_dense_cminor`
        """
        from . import functional

        return functional.inject_from_dense_cminor_batch(self, dense_data, dense_origins)

    def inject_from_dense_cmajor(self, dense_data: torch.Tensor, dense_origins: NumericMaxRank1 = 0) -> JaggedTensor:
        """Inject values from a dense CXYZ-ordered tensor into a :class:`JaggedTensor` for this grid batch.

        Supports backpropagation.

        Args:
            dense_data (torch.Tensor): Dense tensor in CXYZ order. Shape: ``(batch_size, channels*, X, Y, Z)``.
            dense_origins (NumericMaxRank1): Dense tensor origin in voxel space, broadcastable to shape ``(3,)``, integer dtype.

        Returns:
            sparse_data (JaggedTensor): Values at active voxel locations. Shape: ``(batch_size, total_voxels, channels*)``.

        .. seealso:: :meth:`Grid.inject_from_dense_cmajor`
        """
        from . import functional

        return functional.inject_from_dense_cmajor_batch(self, dense_data, dense_origins)

    def sample_bezier(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """Sample voxel data at world-space points using Bezier interpolation on this grid batch.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space sample points. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_data (JaggedTensor): Per-voxel data. Shape: ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated values. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.

        .. seealso:: :meth:`Grid.sample_bezier`
        """
        from . import functional

        return functional.sample_bezier_batch(self, points, voxel_data)

    def sample_bezier_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """Sample voxel data and spatial gradients at world-space points using Bezier interpolation on this grid batch.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space sample points. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_data (JaggedTensor): Per-voxel data. Shape: ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated values. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.
            interpolation_gradients (JaggedTensor): Spatial gradients. Shape: ``(batch_size, num_points_for_grid_b, 3, channels*)``.

        .. seealso:: :meth:`Grid.sample_bezier_with_grad`
        """
        from . import functional

        return functional.sample_bezier_with_grad_batch(self, points, voxel_data)

    def sample_trilinear(self, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor:
        """Sample voxel data at world-space points using trilinear interpolation on this grid batch.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space sample points. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_data (JaggedTensor): Per-voxel data. Shape: ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated values. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.

        .. seealso:: :meth:`Grid.sample_trilinear`
        """
        from . import functional

        return functional.sample_trilinear_batch(self, points, voxel_data)

    def sample_trilinear_with_grad(
        self, points: JaggedTensor, voxel_data: JaggedTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """Sample voxel data and spatial gradients at world-space points using trilinear interpolation on this grid batch.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space sample points. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            voxel_data (JaggedTensor): Per-voxel data. Shape: ``(batch_size, total_voxels, channels*)``.

        Returns:
            interpolated_data (JaggedTensor): Interpolated values. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.
            interpolation_gradients (JaggedTensor): Spatial gradients. Shape: ``(batch_size, num_points_for_grid_b, 3, channels*)``.

        .. seealso:: :meth:`Grid.sample_trilinear_with_grad`
        """
        from . import functional

        return functional.sample_trilinear_with_grad_batch(self, points, voxel_data)

    def segments_along_rays(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        max_segments: int,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """Enumerate ray-voxel intersection segments for this grid batch.

        Args:
            ray_origins (JaggedTensor): Ray origins in world space. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            ray_directions (JaggedTensor): Ray directions. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            max_segments (int): Maximum number of segments per ray.
            eps (float): Epsilon for numerical stability.

        Returns:
            ray_segments (JaggedTensor): Segment start/end distances with eshape ``(2,)``.

        .. seealso:: :meth:`Grid.segments_along_rays`
        """
        from . import functional

        return functional.segments_along_rays_batch(self, ray_origins, ray_directions, max_segments, eps)

    def splat_bezier(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """Splat point data onto voxels of this grid batch using Bezier interpolation weights.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space point positions. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            points_data (JaggedTensor): Data to splat per point. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.

        Returns:
            splatted_features (JaggedTensor): Accumulated voxel features. Shape: ``(batch_size, total_voxels, channels*)``.

        .. seealso:: :meth:`Grid.splat_bezier`
        """
        from . import functional

        return functional.splat_bezier_batch(self, points, points_data)

    def splat_trilinear(self, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor:
        """Splat point data onto voxels of this grid batch using trilinear interpolation weights.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space point positions. Shape: ``(batch_size, num_points_for_grid_b, 3)``.
            points_data (JaggedTensor): Data to splat per point. Shape: ``(batch_size, num_points_for_grid_b, channels*)``.

        Returns:
            splatted_features (JaggedTensor): Accumulated voxel features. Shape: ``(batch_size, total_voxels, channels*)``.

        .. seealso:: :meth:`Grid.splat_trilinear`
        """
        from . import functional

        return functional.splat_trilinear_batch(self, points, points_data)

    def refine(
        self,
        subdiv_factor: NumericMaxRank1,
        data: JaggedTensor,
        mask: JaggedTensor | None = None,
        fine_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """Refine voxel data into higher-resolution grids by subdividing each voxel of this grid batch.

        Supports backpropagation.

        Args:
            subdiv_factor (NumericMaxRank1): Subdivision factor, broadcastable to shape ``(3,)``, integer dtype.
            data (JaggedTensor): Voxel data to refine. Shape: ``(batch_size, total_voxels, channels)``.
            mask (JaggedTensor | None): Optional boolean mask selecting which voxels to refine.
            fine_grid (GridBatch | None): Optional pre-allocated fine grid batch for output.

        Returns:
            refined_data (JaggedTensor): Refined voxel data on the fine grid.
            fine_grid (GridBatch): The fine :class:`GridBatch` containing the refined structure.

        .. seealso:: :meth:`Grid.refine`
        """
        from . import functional

        return functional.refine_batch(self, subdiv_factor, data, mask, fine_grid)

    def refined_grid(
        self,
        subdiv_factor: NumericMaxRank1,
        mask: JaggedTensor | None = None,
    ) -> "GridBatch":
        """Return a refined version of this grid batch by subdividing each voxel.

        Args:
            subdiv_factor (NumericMaxRank1): Subdivision factor, broadcastable to shape ``(3,)``, integer dtype.
            mask (JaggedTensor | None): Optional boolean mask selecting which voxels to refine.

        Returns:
            refined_grid (GridBatch): A new higher-resolution :class:`GridBatch`.

        .. seealso:: :meth:`Grid.refined_grid`
        """
        from . import functional

        return functional.refined_grid_batch(self, subdiv_factor, mask)

    def to(self, target: "str | torch.device | torch.Tensor | JaggedTensor | GridBatch") -> "GridBatch":
        """Move this grid batch to a target device or match the device of a target object.

        Args:
            target (str | torch.device | torch.Tensor | JaggedTensor | GridBatch): Device or object whose device to match.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` on the target device.

        .. seealso:: :meth:`Grid.to`
        """
        from . import functional

        if isinstance(target, str):
            device = _parse_device_string(target)
        elif isinstance(target, torch.device):
            device = target
        elif isinstance(target, torch.Tensor):
            device = target.device
        elif isinstance(target, JaggedTensor):
            device = target.jdata.device
        elif isinstance(target, GridBatch):
            device = target.device
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")
        return functional.clone_grid_batch(self, device)

    def save_nanovdb(
        self,
        path: str,
        data: JaggedTensor | None = None,
        names: list[str] | str | None = None,
        name: str | None = None,
        compressed: bool = False,
        verbose: bool = False,
    ) -> None:
        """Save this grid batch and optional voxel data to a .nvdb file.

        Args:
            path (str): File path to save to (should have .nvdb extension).
            data (JaggedTensor | None): Optional voxel data to save. Shape: ``(batch_size, total_voxels, channels)``.
            names (list[str] | str | None): Names for each grid, or a single name for all.
            name (str | None): Single name for all grids (takes precedence over ``names``).
            compressed (bool): Whether to use Blosc compression.
            verbose (bool): Whether to print information about saved grids.

        .. seealso:: :meth:`Grid.save_nanovdb`
        """
        from . import functional

        functional.save_nanovdb(self, path, data, names, name, compressed, verbose)

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
        """Generate uniformly spaced samples along rays intersecting active voxels of this grid batch.

        Args:
            ray_origins (JaggedTensor): Ray origins in world space. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            ray_directions (JaggedTensor): Ray directions (should be normalized). Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            t_min (JaggedTensor): Minimum ray distance. Shape: ``(batch_size, num_rays_for_grid_b)``.
            t_max (JaggedTensor): Maximum ray distance. Shape: ``(batch_size, num_rays_for_grid_b)``.
            step_size (float): Distance between samples.
            cone_angle (float): Cone angle for cone tracing in radians.
            include_end_segments (bool): Whether to include partial end segments.
            return_midpoints (bool): If ``True``, return midpoints instead of start/end pairs.
            eps (float): Epsilon for numerical stability.

        Returns:
            ray_samples (JaggedTensor): Sample distances with eshape ``(2,)`` or ``(1,)`` if ``return_midpoints``.

        .. seealso:: :meth:`Grid.uniform_ray_samples`
        """
        from . import functional

        return functional.uniform_ray_samples_batch(
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

    def voxel_size_at(self, bi: int) -> torch.Tensor:
        """Get the voxel size of a specific grid in this grid batch.

        Args:
            bi (int): Batch index of the grid.

        Returns:
            voxel_size (torch.Tensor): Voxel size. Shape: ``(3,)``.
        """
        return self.data.voxel_size_at(bi)

    def rays_intersect_voxels(
        self, ray_origins: JaggedTensor, ray_directions: JaggedTensor, eps: float = 0.0
    ) -> JaggedTensor:
        """Check whether rays hit any active voxels in this grid batch.

        Args:
            ray_origins (JaggedTensor): Ray origins in world space. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            ray_directions (JaggedTensor): Ray directions. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            eps (float): Epsilon for numerical stability.

        Returns:
            hit_mask (JaggedTensor): Boolean mask indicating ray hits. Shape: ``(batch_size, num_rays_for_grid_b)``.

        .. seealso:: :meth:`Grid.rays_intersect_voxels`
        """
        from . import functional

        return functional.rays_intersect_voxels_batch(self, ray_origins, ray_directions, eps)

    def voxels_along_rays(
        self,
        ray_origins: JaggedTensor,
        ray_directions: JaggedTensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = True,
        cumulative: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """Enumerate active voxels intersected by rays in this grid batch via DDA traversal.

        Args:
            ray_origins (JaggedTensor): Ray origins in world space. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            ray_directions (JaggedTensor): Ray directions. Shape: ``(batch_size, num_rays_for_grid_b, 3)``.
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Epsilon for numerical stability.
            return_ijk (bool): If ``True``, return ijk coordinates; otherwise linear indices.
            cumulative (bool): If ``True``, return batch-cumulative indices.

        Returns:
            voxels (JaggedTensor): Voxel coordinates (eshape ``(3,)``) or linear indices per ray hit.
            times (JaggedTensor): Entry/exit distances along each ray with eshape ``(2,)``.

        .. seealso:: :meth:`Grid.voxels_along_rays`
        """
        from . import functional

        return functional.voxels_along_rays_batch(
            self, ray_origins, ray_directions, max_voxels, eps, return_ijk, cumulative
        )

    def world_to_voxel(self, points: JaggedTensor) -> JaggedTensor:
        """Transform world-space coordinates to voxel-space coordinates for this grid batch.

        Supports backpropagation.

        Args:
            points (JaggedTensor): World-space positions to convert. Shape: ``(batch_size, num_points_for_grid_b, 3)``.

        Returns:
            voxel_points (JaggedTensor): Voxel-space coordinates (may be fractional). Shape: ``(batch_size, num_points_for_grid_b, 3)``.

        .. seealso:: :meth:`Grid.world_to_voxel`
        """
        from . import functional

        return functional.world_to_voxel_batch(self, points)

    def inject_to_dense_cminor(
        self,
        sparse_data: JaggedTensor,
        min_coord: NumericMaxRank2 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """Inject sparse voxel data from this grid batch into a dense XYZC-ordered tensor.

        Supports backpropagation.

        Args:
            sparse_data (JaggedTensor): Sparse voxel data. Shape: ``(batch_size, total_voxels, channels*)``.
            min_coord (NumericMaxRank2 | None): Minimum voxel coordinate per grid, or ``None`` for auto.
            grid_size (NumericMaxRank1 | None): Output dense grid size, or ``None`` for auto.

        Returns:
            dense_data (torch.Tensor): Dense tensor in XYZC order. Shape: ``(batch_size, X, Y, Z, channels*)``.

        .. seealso:: :meth:`Grid.inject_to_dense_cminor`
        """
        from . import functional

        return functional.inject_to_dense_cminor_batch(self, sparse_data, min_coord, grid_size)

    def inject_to_dense_cmajor(
        self,
        sparse_data: JaggedTensor,
        min_coord: NumericMaxRank2 | None = None,
        grid_size: NumericMaxRank1 | None = None,
    ) -> torch.Tensor:
        """Inject sparse voxel data from this grid batch into a dense CXYZ-ordered tensor.

        Supports backpropagation.

        Args:
            sparse_data (JaggedTensor): Sparse voxel data. Shape: ``(batch_size, total_voxels, channels*)``.
            min_coord (NumericMaxRank2 | None): Minimum voxel coordinate per grid, or ``None`` for auto.
            grid_size (NumericMaxRank1 | None): Output dense grid size, or ``None`` for auto.

        Returns:
            dense_data (torch.Tensor): Dense tensor in CXYZ order. Shape: ``(batch_size, channels*, X, Y, Z)``.

        .. seealso:: :meth:`Grid.inject_to_dense_cmajor`
        """
        from . import functional

        return functional.inject_to_dense_cmajor_batch(self, sparse_data, min_coord, grid_size)

    # ============================================================
    #                Indexing and Special Functions
    # ============================================================

    # Index methods
    def index_int(self, bi: int | np.integer) -> "GridBatch":
        """Select a single grid from this grid batch by integer index.

        Args:
            bi (int | np.integer): Grid index.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing the selected grid.
        """
        from . import functional

        return functional.index_grid(self, int(bi))

    def index_list(self, indices: list[bool] | list[int]) -> "GridBatch":
        """Select grids from this grid batch using a list of indices or booleans.

        Args:
            indices (list[bool] | list[int]): List of grid indices or boolean mask.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing the selected grids.
        """
        from . import functional

        return functional.index_grid(self, indices)

    def index_slice(self, s: slice) -> "GridBatch":
        """Select grids from this grid batch using a slice.

        Args:
            s (slice): Slice object specifying the range.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing the selected grids.
        """
        from . import functional

        return functional.index_grid(self, s)

    def index_tensor(self, indices: torch.Tensor) -> "GridBatch":
        """Select grids from this grid batch using a tensor of indices.

        Args:
            indices (torch.Tensor): Integer or boolean tensor of grid indices.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing the selected grids.
        """
        from . import functional

        return functional.index_grid(self, indices)

    # Special methods
    def __getitem__(self, index: GridBatchIndex) -> "GridBatch":
        """Select grids from this grid batch by index, slice, list, or tensor.

        Args:
            index (GridBatchIndex): Int, slice, list, or tensor index.

        Returns:
            grid_batch (GridBatch): A new :class:`GridBatch` containing the selected grids.
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
        """Iterate over individual grids in this grid batch.

        Yields:
            grid_batch (GridBatch): A single-grid :class:`GridBatch` for each grid.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of grids in this grid batch.

        Returns:
            length (int): Number of grids.
        """
        return self.data.grid_count

    # ============================================================
    #                        Properties
    # ============================================================

    # Properties
    @property
    def address(self) -> int:
        """Unique identifier (int) of the underlying C++ GridBatchData object."""
        return id(self.data)

    @property
    def all_have_zero_voxels(self) -> bool:
        """``True`` (bool) if all grids in this grid batch have zero active voxels."""
        return self.has_zero_grids or self.total_voxels == 0

    @property
    def any_have_zero_voxels(self) -> bool:
        """``True`` (bool) if any grid in this grid batch has zero active voxels."""
        if self.has_zero_grids:
            return True
        else:
            return bool(torch.any(self.num_voxels == 0).item())

    @property
    def bboxes(self) -> torch.Tensor:
        """Voxel-space bounding boxes (torch.Tensor) of shape ``(grid_count, 2, 3)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 2, 3), dtype=torch.int32, device=self.device)
        else:
            if self.all_have_zero_voxels:
                return torch.zeros((self.grid_count, 2, 3), dtype=torch.int32, device=self.device)
            elif self.any_have_zero_voxels:
                bboxes = self.data.bbox.to(self.device)

                fixed_bboxes = []
                for i in range(self.grid_count):
                    if self.num_voxels[i] == 0:
                        fixed_bboxes.append(torch.zeros((2, 3), dtype=torch.int32, device=self.device))
                    else:
                        fixed_bboxes.append(bboxes[i])

                return torch.stack(fixed_bboxes, dim=0)
            else:
                return self.data.bbox.to(self.device)

    @property
    def cum_voxels(self) -> torch.Tensor:
        """Cumulative voxel counts (torch.Tensor) of shape ``(grid_count,)`` for this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self.data.cum_voxels.to(self.device)

    @property
    def device(self) -> torch.device:
        """The :class:`torch.device` where this grid batch is stored."""
        return self.data.device

    @property
    def dual_bboxes(self) -> torch.Tensor:
        """Dual voxel-space bounding boxes (torch.Tensor) of shape ``(grid_count, 2, 3)`` for this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 2, 3), dtype=torch.int32, device=self.device)
        else:
            if self.all_have_zero_voxels:
                return torch.zeros((self.grid_count, 2, 3), dtype=torch.int32, device=self.device)
            elif self.any_have_zero_voxels:
                bboxes = self.data.dual_bbox.to(self.device)

                fixed_bboxes = []
                for i in range(self.grid_count):
                    if self.num_voxels[i] == 0:
                        fixed_bboxes.append(torch.zeros((2, 3), dtype=torch.int32, device=self.device))
                    else:
                        fixed_bboxes.append(bboxes[i])

                return torch.stack(fixed_bboxes, dim=0)
            else:
                return self.data.dual_bbox.to(self.device)

    @property
    def grid_count(self) -> int:
        """Number of grids (int) in this grid batch."""
        return self.data.grid_count

    @property
    def voxel_to_world_matrices(self) -> torch.Tensor:
        """Voxel-to-world transformation matrices (torch.Tensor) of shape ``(grid_count, 4, 4)`` for this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self.data.voxel_to_world_matrices.to(dtype=torch.float32, device=self.device)

    @property
    def has_zero_grids(self) -> bool:
        """``True`` (bool) if this grid batch contains zero grids."""
        return self.grid_count == 0

    @property
    def ijk(self) -> JaggedTensor:
        """Active voxel coordinates (:class:`JaggedTensor`) of shape ``(batch_size, total_voxels, 3)`` in index order."""
        from . import functional

        return functional.active_grid_coords_batch(self)

    def morton(self, offset: NumericMaxRank1 | None = None) -> JaggedTensor:
        """Return xyz Morton codes (Z-order curve) for active voxels in this grid batch.

        Args:
            offset (NumericMaxRank1 | None): Optional coordinate offset before encoding.

        Returns:
            codes (JaggedTensor): Morton codes. Shape: ``(batch_size, num_voxels_for_grid_b, 1)``.

        .. seealso:: :meth:`Grid.morton`
        """
        from . import functional

        return functional.morton_batch(self, offset)

    def morton_zyx(self, offset: NumericMaxRank1 | None = None) -> JaggedTensor:
        """Return zyx Morton codes (transposed Z-order curve) for active voxels in this grid batch.

        Args:
            offset (NumericMaxRank1 | None): Optional coordinate offset before encoding.

        Returns:
            codes (JaggedTensor): Transposed Morton codes. Shape: ``(batch_size, num_voxels_for_grid_b, 1)``.

        .. seealso:: :meth:`Grid.morton_zyx`
        """
        from . import functional

        return functional.morton_zyx_batch(self, offset)

    def hilbert(self, offset: NumericMaxRank1 | None = None) -> JaggedTensor:
        """Return Hilbert curve codes for active voxels in this grid batch.

        Args:
            offset (NumericMaxRank1 | None): Optional coordinate offset before encoding.

        Returns:
            codes (JaggedTensor): Hilbert codes. Shape: ``(batch_size, num_voxels_for_grid_b, 1)``.

        .. seealso:: :meth:`Grid.hilbert`
        """
        from . import functional

        return functional.hilbert_batch(self, offset)

    def hilbert_zyx(self, offset: NumericMaxRank1 | None = None) -> JaggedTensor:
        """Return zyx Hilbert curve codes (transposed) for active voxels in this grid batch.

        Args:
            offset (NumericMaxRank1 | None): Optional coordinate offset before encoding.

        Returns:
            codes (JaggedTensor): Transposed Hilbert codes. Shape: ``(batch_size, num_voxels_for_grid_b, 1)``.

        .. seealso:: :meth:`Grid.hilbert_zyx`
        """
        from . import functional

        return functional.hilbert_zyx_batch(self, offset)

    @property
    def jidx(self) -> torch.Tensor:
        """Per-voxel grid index (torch.Tensor) of shape ``(total_voxels,)`` for this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        else:
            return self.data.jidx

    @property
    def joffsets(self) -> torch.Tensor:
        """Jagged offset tensor (torch.Tensor) of shape ``(grid_count + 1,)`` defining grid boundaries."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self.data.joffsets

    @property
    def num_bytes(self) -> torch.Tensor:
        """Memory size in bytes (torch.Tensor) of shape ``(grid_count,)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self.data.num_bytes.to(self.device)

    @property
    def num_leaf_nodes(self) -> torch.Tensor:
        """NanoVDB leaf node count (torch.Tensor) of shape ``(grid_count,)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        data = self.data
        result = data.num_leaves
        return result.to(self.device)

    @property
    def num_voxels(self) -> torch.Tensor:
        """Active voxel count (torch.Tensor) of shape ``(grid_count,)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            return self.data.num_voxels.to(self.device)

    @property
    def origins(self) -> torch.Tensor:
        """World-space origins (torch.Tensor) of shape ``(grid_count, 3)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self.data.origins.to(dtype=torch.float32, device=self.device)

    @property
    def total_bbox(self) -> torch.Tensor:
        """Voxel-space bounding box (torch.Tensor) of shape ``(2, 3)`` encompassing all grids in this grid batch."""
        if self.has_zero_grids or self.all_have_zero_voxels:
            return torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        else:
            return self.data.total_bbox.to(self.device)

    @property
    def total_bytes(self) -> int:
        """Total memory size in bytes (int) of all grids in this grid batch."""
        if self.has_zero_grids:
            return 0
        else:
            return self.data.total_bytes

    @property
    def total_leaf_nodes(self) -> int:
        """Total NanoVDB leaf node count (int) across all grids in this grid batch."""
        if self.has_zero_grids:
            return 0
        else:
            return self.data.total_leaves

    @property
    def total_voxels(self) -> int:
        """Total active voxel count (int) across all grids in this grid batch."""
        if self.has_zero_grids:
            return 0
        else:
            return self.data.total_voxels

    @property
    def voxel_sizes(self) -> torch.Tensor:
        """World-space voxel sizes (torch.Tensor) of shape ``(grid_count, 3)`` for each grid in this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            return self.data.voxel_sizes.to(dtype=torch.float32, device=self.device)

    @property
    def world_to_voxel_matrices(self) -> torch.Tensor:
        """World-to-voxel transformation matrices (torch.Tensor) of shape ``(grid_count, 4, 4)`` for this grid batch."""
        if self.has_zero_grids:
            return torch.empty((0, 4, 4), dtype=torch.float32, device=self.device)
        else:
            return self.data.world_to_voxel_matrices.to(dtype=torch.float32, device=self.device)


def gcat(grids: "Sequence[GridBatch]") -> GridBatch:
    """Concatenate a sequence of grid batches into a single grid batch along the batch dimension.

    Args:
        grids (Sequence[GridBatch]): Grid batches to concatenate.

    Returns:
        grid_batch (GridBatch): A new :class:`GridBatch` containing all grids from the inputs.
    """
    return GridBatch.from_cat(grids)
