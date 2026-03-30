# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for spatial queries on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor
from .. import _fvdb_cpp
from ..types import NumericMaxRank1, to_Vec3fBroadcastable


def _to_vec3f_list(v: NumericMaxRank1) -> list[float]:
    t = to_Vec3fBroadcastable(v)
    if t.dim() == 0:
        t = t.expand(3)
    return t.tolist()
from ._dispatch import _get_grid_data

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


@overload
def points_in_grid(grid: GridBatch, points: torch.Tensor) -> torch.Tensor: ...


@overload
def points_in_grid(grid: GridBatch, points: JaggedTensor) -> JaggedTensor: ...


def points_in_grid(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Check if world-space points are located within active voxels.

    Args:
        grid: The grid to test against.
        points: World-space points.
            For a single grid: shape ``(N, 3)``.
            For a batch: :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.

    Returns:
        Boolean mask indicating which points are in active voxels.
    """
    grid_data = _get_grid_data(grid)
    if isinstance(points, torch.Tensor):
        jt = JaggedTensor(points)
        return _fvdb_cpp.points_in_grid(grid_data, jt._impl).jdata
    return JaggedTensor(impl=_fvdb_cpp.points_in_grid(grid_data, points._impl))


@overload
def coords_in_grid(grid: GridBatch, ijk: torch.Tensor) -> torch.Tensor: ...


@overload
def coords_in_grid(grid: GridBatch, ijk: JaggedTensor) -> JaggedTensor: ...


def coords_in_grid(
    grid: GridBatch,
    ijk: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Check which voxel-space coordinates lie on active voxels.

    Args:
        grid: The grid to test against.
        ijk: Voxel coordinates with integer dtype.

    Returns:
        Boolean mask indicating which coordinates correspond to active voxels.
    """
    grid_data = _get_grid_data(grid)
    if isinstance(ijk, torch.Tensor):
        jt = JaggedTensor(ijk)
        return _fvdb_cpp.coords_in_grid(grid_data, jt._impl).jdata
    return JaggedTensor(impl=_fvdb_cpp.coords_in_grid(grid_data, ijk._impl))


@overload
def cubes_in_grid(
    grid: GridBatch,
    cube_centers: torch.Tensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor: ...


@overload
def cubes_in_grid(
    grid: GridBatch,
    cube_centers: JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> JaggedTensor: ...


def cubes_in_grid(
    grid: GridBatch,
    cube_centers: torch.Tensor | JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor | JaggedTensor:
    """
    Check if axis-aligned cubes are fully contained within active voxels.

    Args:
        grid: The grid to test against.
        cube_centers: World-space cube centers.
        cube_min: Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max: Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        Boolean mask indicating which cubes are fully contained.
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    grid_data = _get_grid_data(grid)
    if isinstance(cube_centers, torch.Tensor):
        jt = JaggedTensor(cube_centers)
        return _fvdb_cpp.cubes_in_grid(grid_data, jt._impl, cmin, cmax).jdata
    return JaggedTensor(impl=_fvdb_cpp.cubes_in_grid(grid_data, cube_centers._impl, cmin, cmax))


@overload
def cubes_intersect_grid(
    grid: GridBatch,
    cube_centers: torch.Tensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor: ...


@overload
def cubes_intersect_grid(
    grid: GridBatch,
    cube_centers: JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> JaggedTensor: ...


def cubes_intersect_grid(
    grid: GridBatch,
    cube_centers: torch.Tensor | JaggedTensor,
    cube_min: NumericMaxRank1 = 0.0,
    cube_max: NumericMaxRank1 = 0.0,
) -> torch.Tensor | JaggedTensor:
    """
    Check if axis-aligned cubes intersect any active voxels.

    Args:
        grid: The grid to test against.
        cube_centers: World-space cube centers.
        cube_min: Minimum offsets from center, broadcastable to ``(3,)``.
        cube_max: Maximum offsets from center, broadcastable to ``(3,)``.

    Returns:
        Boolean mask indicating which cubes intersect the grid.
    """
    cmin = _to_vec3f_list(cube_min)
    cmax = _to_vec3f_list(cube_max)
    grid_data = _get_grid_data(grid)
    if isinstance(cube_centers, torch.Tensor):
        jt = JaggedTensor(cube_centers)
        return _fvdb_cpp.cubes_intersect_grid(grid_data, jt._impl, cmin, cmax).jdata
    return JaggedTensor(impl=_fvdb_cpp.cubes_intersect_grid(grid_data, cube_centers._impl, cmin, cmax))


@overload
def ijk_to_index(grid: GridBatch, ijk: torch.Tensor, cumulative: bool = False) -> torch.Tensor: ...


@overload
def ijk_to_index(grid: GridBatch, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor: ...


def ijk_to_index(
    grid: GridBatch,
    ijk: torch.Tensor | JaggedTensor,
    cumulative: bool = False,
) -> torch.Tensor | JaggedTensor:
    """
    Convert voxel-space coordinates to linear indices. Returns ``-1`` for
    coordinates that do not correspond to active voxels.

    Args:
        grid: The grid to index into.
        ijk: Voxel coordinates with integer dtype.
        cumulative: If ``True``, return indices cumulative across the batch.

    Returns:
        Linear indices (or ``-1`` for inactive coordinates).
    """
    grid_data = _get_grid_data(grid)
    if isinstance(ijk, torch.Tensor):
        jt = JaggedTensor(ijk)
        return _fvdb_cpp.ijk_to_index(grid_data, jt._impl, cumulative).jdata
    return JaggedTensor(impl=_fvdb_cpp.ijk_to_index(grid_data, ijk._impl, cumulative))


@overload
def neighbor_indexes(grid: GridBatch, ijk: torch.Tensor, extent: int, bitshift: int = 0) -> torch.Tensor: ...


@overload
def neighbor_indexes(grid: GridBatch, ijk: JaggedTensor, extent: int, bitshift: int = 0) -> JaggedTensor: ...


def neighbor_indexes(
    grid: GridBatch,
    ijk: torch.Tensor | JaggedTensor,
    extent: int,
    bitshift: int = 0,
) -> torch.Tensor | JaggedTensor:
    """
    Get linear indices of neighboring voxels in an N-ring neighborhood.

    Args:
        grid: The grid to query.
        ijk: Voxel coordinates with integer dtype.
        extent: Neighborhood ring size.
        bitshift: Optional bit shift applied to input coordinates. Default ``0``.

    Returns:
        Neighbor indices; ``-1`` for inactive neighbors.
    """
    grid_data = _get_grid_data(grid)
    if isinstance(ijk, torch.Tensor):
        jt = JaggedTensor(ijk)
        return _fvdb_cpp.neighbor_indexes(grid_data, jt._impl, extent, bitshift).jdata
    return JaggedTensor(impl=_fvdb_cpp.neighbor_indexes(grid_data, ijk._impl, extent, bitshift))


@overload
def ijk_to_inv_index(grid: GridBatch, ijk: torch.Tensor, cumulative: bool = False) -> torch.Tensor: ...


@overload
def ijk_to_inv_index(grid: GridBatch, ijk: JaggedTensor, cumulative: bool = False) -> JaggedTensor: ...


def ijk_to_inv_index(
    grid: GridBatch,
    ijk: torch.Tensor | JaggedTensor,
    cumulative: bool = False,
) -> torch.Tensor | JaggedTensor:
    """
    Get inverse permutation of :func:`ijk_to_index`. For each voxel in each
    grid, return the index in the input ``ijk`` tensor.

    Args:
        grid: The grid to index into.
        ijk: Voxel coordinates with integer dtype.
        cumulative: If ``True``, return indices cumulative across the batch.

    Returns:
        Inverse permutation indices.
    """
    grid_data = _get_grid_data(grid)
    if isinstance(ijk, torch.Tensor):
        jt = JaggedTensor(ijk)
        return _fvdb_cpp.ijk_to_inv_index(grid_data, jt._impl, cumulative).jdata
    return JaggedTensor(impl=_fvdb_cpp.ijk_to_inv_index(grid_data, ijk._impl, cumulative))


def active_grid_coords(grid: GridBatch) -> JaggedTensor:
    """
    Return the voxel coordinates of every active voxel in the grid, in index order.

    Args:
        grid: The grid to query.

    Returns:
        A :class:`~fvdb.JaggedTensor` of shape ``(B, -1, 3)`` with the voxel
        coordinates.
    """
    grid_data = _get_grid_data(grid)
    return JaggedTensor(impl=_fvdb_cpp.active_grid_coords(grid_data))
