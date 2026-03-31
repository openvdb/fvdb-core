# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for spatial queries on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..jagged_tensor import JaggedTensor
from .. import _fvdb_cpp
from ..types import NumericMaxRank1, to_Vec3fBroadcastable


def _to_vec3f_list(v: NumericMaxRank1) -> list[float]:
    t = to_Vec3fBroadcastable(v)
    if t.dim() == 0:
        t = t.expand(3)
    return t.tolist()


if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
# points_in_grid
# ---------------------------------------------------------------------------

def points_in_grid_batch(grid: GridBatch, points: JaggedTensor) -> JaggedTensor:
    """Check if world-space points are located within active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        points (JaggedTensor): World-space points, shape ``(B, -1, 3)``.

    Returns:
        mask (JaggedTensor): Boolean mask indicating which points are in active voxels.

    .. seealso:: :func:`points_in_grid_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.points_in_grid(grid.data, points._impl))


def points_in_grid_single(grid: Grid, points: torch.Tensor) -> torch.Tensor:
    """Check if world-space points are located within active voxels of a single grid.

    Args:
        grid (Grid): The single grid to test against.
        points (torch.Tensor): World-space points, shape ``(N, 3)``.

    Returns:
        mask (torch.Tensor): Boolean mask indicating which points are in active voxels.

    .. seealso:: :func:`points_in_grid_batch`
    """
    jt = JaggedTensor(points)
    return _fvdb_cpp.points_in_grid(grid.data, jt._impl).jdata


# ---------------------------------------------------------------------------
# coords_in_grid
# ---------------------------------------------------------------------------

def coords_in_grid_batch(grid: GridBatch, ijk: JaggedTensor) -> JaggedTensor:
    """Check which voxel-space coordinates lie on active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        ijk (JaggedTensor): Voxel coordinates with integer dtype.

    Returns:
        mask (JaggedTensor): Boolean mask indicating which coordinates correspond to active voxels.

    .. seealso:: :func:`coords_in_grid_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.coords_in_grid(grid.data, ijk._impl))


def coords_in_grid_single(grid: Grid, ijk: torch.Tensor) -> torch.Tensor:
    """Check which voxel-space coordinates lie on active voxels of a single grid.

    Args:
        grid (Grid): The single grid to test against.
        ijk (torch.Tensor): Voxel coordinates with integer dtype.

    Returns:
        mask (torch.Tensor): Boolean mask indicating which coordinates correspond to active voxels.

    .. seealso:: :func:`coords_in_grid_batch`
    """
    jt = JaggedTensor(ijk)
    return _fvdb_cpp.coords_in_grid(grid.data, jt._impl).jdata


# ---------------------------------------------------------------------------
# cubes_in_grid
# ---------------------------------------------------------------------------

def cubes_in_grid_batch(
    grid: GridBatch,
    cube_centers: JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> JaggedTensor:
    """Check if axis-aligned cubes are fully contained within active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        cube_centers (JaggedTensor): World-space cube centers, shape ``(B, -1, 3)``.
        cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        mask (JaggedTensor): Boolean mask indicating which cubes are fully contained.

    .. seealso:: :func:`cubes_in_grid_single`
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    return JaggedTensor(impl=_fvdb_cpp.cubes_in_grid(grid.data, cube_centers._impl, cmin, cmax))


def cubes_in_grid_single(
    grid: Grid,
    cube_centers: torch.Tensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor:
    """Check if axis-aligned cubes are fully contained within active voxels of a single grid.

    Args:
        grid (Grid): The single grid to test against.
        cube_centers (torch.Tensor): World-space cube centers, shape ``(N, 3)``.
        cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        mask (torch.Tensor): Boolean mask indicating which cubes are fully contained.

    .. seealso:: :func:`cubes_in_grid_batch`
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    jt = JaggedTensor(cube_centers)
    return _fvdb_cpp.cubes_in_grid(grid.data, jt._impl, cmin, cmax).jdata


# ---------------------------------------------------------------------------
# cubes_intersect_grid
# ---------------------------------------------------------------------------

def cubes_intersect_grid_batch(
    grid: GridBatch,
    cube_centers: JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> JaggedTensor:
    """Check if axis-aligned cubes intersect any active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        cube_centers (JaggedTensor): World-space cube centers, shape ``(B, -1, 3)``.
        cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        mask (JaggedTensor): Boolean mask indicating which cubes intersect the grid.

    .. seealso:: :func:`cubes_intersect_grid_single`
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    return JaggedTensor(impl=_fvdb_cpp.cubes_intersect_grid(grid.data, cube_centers._impl, cmin, cmax))


def cubes_intersect_grid_single(
    grid: Grid,
    cube_centers: torch.Tensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor:
    """Check if axis-aligned cubes intersect any active voxels of a single grid.

    Args:
        grid (Grid): The single grid to test against.
        cube_centers (torch.Tensor): World-space cube centers, shape ``(N, 3)``.
        cube_min (NumericMaxRank1): Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max (NumericMaxRank1): Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        mask (torch.Tensor): Boolean mask indicating which cubes intersect the grid.

    .. seealso:: :func:`cubes_intersect_grid_batch`
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    jt = JaggedTensor(cube_centers)
    return _fvdb_cpp.cubes_intersect_grid(grid.data, jt._impl, cmin, cmax).jdata


# ---------------------------------------------------------------------------
# ijk_to_index
# ---------------------------------------------------------------------------

def ijk_to_index_batch(
    grid: GridBatch,
    ijk: JaggedTensor,
    cumulative: bool = False,
) -> JaggedTensor:
    """Convert voxel-space coordinates to linear indices in a grid batch.

    Args:
        grid (GridBatch): The grid batch to index into.
        ijk (JaggedTensor): Voxel coordinates with integer dtype.
        cumulative (bool): If ``True``, return indices cumulative across the batch.

    Returns:
        indices (JaggedTensor): Linear indices (``-1`` for inactive coordinates).

    .. seealso:: :func:`ijk_to_index_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.ijk_to_index(grid.data, ijk._impl, cumulative))


def ijk_to_index_single(
    grid: Grid,
    ijk: torch.Tensor,
    cumulative: bool = False,
) -> torch.Tensor:
    """Convert voxel-space coordinates to linear indices in a single grid.

    Args:
        grid (Grid): The single grid to index into.
        ijk (torch.Tensor): Voxel coordinates with integer dtype.
        cumulative (bool): If ``True``, return indices cumulative across the batch.

    Returns:
        indices (torch.Tensor): Linear indices (``-1`` for inactive coordinates).

    .. seealso:: :func:`ijk_to_index_batch`
    """
    jt = JaggedTensor(ijk)
    return _fvdb_cpp.ijk_to_index(grid.data, jt._impl, cumulative).jdata


# ---------------------------------------------------------------------------
# ijk_to_inv_index
# ---------------------------------------------------------------------------

def ijk_to_inv_index_batch(
    grid: GridBatch,
    ijk: JaggedTensor,
    cumulative: bool = False,
) -> JaggedTensor:
    """Get the inverse permutation of :func:`ijk_to_index_batch` for a grid batch.

    Args:
        grid (GridBatch): The grid batch to index into.
        ijk (JaggedTensor): Voxel coordinates with integer dtype.
        cumulative (bool): If ``True``, return indices cumulative across the batch.

    Returns:
        indices (JaggedTensor): Inverse permutation indices.

    .. seealso:: :func:`ijk_to_inv_index_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.ijk_to_inv_index(grid.data, ijk._impl, cumulative))


def ijk_to_inv_index_single(
    grid: Grid,
    ijk: torch.Tensor,
    cumulative: bool = False,
) -> torch.Tensor:
    """Get the inverse permutation of :func:`ijk_to_index_single` for a single grid.

    Args:
        grid (Grid): The single grid to index into.
        ijk (torch.Tensor): Voxel coordinates with integer dtype.
        cumulative (bool): If ``True``, return indices cumulative across the batch.

    Returns:
        indices (torch.Tensor): Inverse permutation indices.

    .. seealso:: :func:`ijk_to_inv_index_batch`
    """
    jt = JaggedTensor(ijk)
    return _fvdb_cpp.ijk_to_inv_index(grid.data, jt._impl, cumulative).jdata


# ---------------------------------------------------------------------------
# neighbor_indexes
# ---------------------------------------------------------------------------

def neighbor_indexes_batch(
    grid: GridBatch,
    ijk: JaggedTensor,
    extent: int,
    bitshift: int = 0,
) -> JaggedTensor:
    """Get linear indices of neighboring voxels in an N-ring neighborhood for a grid batch.

    Args:
        grid (GridBatch): The grid batch to query.
        ijk (JaggedTensor): Voxel coordinates with integer dtype.
        extent (int): Neighborhood ring size.
        bitshift (int): Optional bit shift applied to input coordinates. Default ``0``.

    Returns:
        indices (JaggedTensor): Neighbor indices; ``-1`` for inactive neighbors.

    .. seealso:: :func:`neighbor_indexes_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.neighbor_indexes(grid.data, ijk._impl, extent, bitshift))


def neighbor_indexes_single(
    grid: Grid,
    ijk: torch.Tensor,
    extent: int,
    bitshift: int = 0,
) -> torch.Tensor:
    """Get linear indices of neighboring voxels in an N-ring neighborhood for a single grid.

    Args:
        grid (Grid): The single grid to query.
        ijk (torch.Tensor): Voxel coordinates with integer dtype.
        extent (int): Neighborhood ring size.
        bitshift (int): Optional bit shift applied to input coordinates. Default ``0``.

    Returns:
        indices (torch.Tensor): Neighbor indices; ``-1`` for inactive neighbors.

    .. seealso:: :func:`neighbor_indexes_batch`
    """
    jt = JaggedTensor(ijk)
    return _fvdb_cpp.neighbor_indexes(grid.data, jt._impl, extent, bitshift).jdata


# ---------------------------------------------------------------------------
# active_grid_coords
# ---------------------------------------------------------------------------

def active_grid_coords_batch(grid: GridBatch) -> JaggedTensor:
    """Return the voxel coordinates of every active voxel in a grid batch, in index order.

    Args:
        grid (GridBatch): The grid batch to query.

    Returns:
        ijk (JaggedTensor): Voxel coordinates, shape ``(B, -1, 3)``.

    .. seealso:: :func:`active_grid_coords_single`
    """
    return JaggedTensor(impl=_fvdb_cpp.active_grid_coords(grid.data))


def active_grid_coords_single(grid: Grid) -> torch.Tensor:
    """Return the voxel coordinates of every active voxel in a single grid, in index order.

    Args:
        grid (Grid): The single grid to query.

    Returns:
        ijk (torch.Tensor): Voxel coordinates, shape ``(N, 3)``.

    .. seealso:: :func:`active_grid_coords_batch`
    """
    return _fvdb_cpp.active_grid_coords(grid.data).jdata
