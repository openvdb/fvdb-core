# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for creating GridBatch objects from various sources."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import (
    DeviceIdentifier,
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    resolve_device,
    to_Vec3fBatch,
    to_Vec3fBatchBroadcastable,
    to_Vec3fBroadcastable,
    to_Vec3i,
    to_Vec3iBroadcastable,
)
if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


def _wrap_grid(cpp_impl):
    from ..grid_batch import GridBatch

    return GridBatch(data=cpp_impl)


def _to_vec3d_batch(t: torch.Tensor, batch_size: int | None = None) -> list[list[float]]:
    """Convert a broadcastable tensor to list[list[float]] for C++ bindings.

    If *batch_size* is given the result is broadcast-expanded to that many rows.
    """
    t = t.to(torch.float64)
    if t.dim() == 0:
        v = t.item()
        row = [v, v, v]
        n = batch_size if batch_size is not None else 1
        return [row] * n
    if t.dim() == 1:
        row = t.tolist()
        n = batch_size if batch_size is not None else 1
        return [row] * n
    # t.dim() >= 2
    if batch_size is not None and t.size(0) == 1 and batch_size > 1:
        return t.expand(batch_size, -1).tolist()
    return t.tolist()


# ---------------------------------------------------------------------------
#  Grid creation from data
# ---------------------------------------------------------------------------


def gridbatch_from_dense(
    num_grids: int,
    dense_dims: NumericMaxRank1,
    ijk_min: NumericMaxRank1 = 0,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    mask: torch.Tensor | None = None,
    device: DeviceIdentifier | None = None,
) -> GridBatch:
    """
    Create a batch of dense grids.

    Args:
        num_grids: Number of grids to create.
        dense_dims: Dimensions of the dense grid, broadcastable to ``(3,)``, integer dtype.
        ijk_min: Minimum voxel index, broadcastable to ``(3,)``, integer dtype.
        voxel_sizes: Voxel size per grid, broadcastable to ``(num_grids, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(num_grids, 3)``, floating dtype.
        mask: Optional boolean mask ``(W, H, D)`` selecting active voxels.
        device: Device to create on. Defaults to mask's device or ``"cpu"``.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    resolved_device = resolve_device(device, inherit_from=mask)

    dense_dims_t = to_Vec3iBroadcastable(dense_dims, value_constraint=ValueConstraint.POSITIVE)
    ijk_min_t = to_Vec3i(ijk_min)
    voxel_sizes_t = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)

    grid_data = _fvdb_cpp.gridbatch_from_dense(
        num_grids, dense_dims_t.tolist(), ijk_min_t.tolist(),
        _to_vec3d_batch(voxel_sizes_t, num_grids), _to_vec3d_batch(origins_t, num_grids),
        mask, str(resolved_device))
    return _wrap_grid(grid_data)


def gridbatch_from_dense_axis_aligned_bounds(
    num_grids: int,
    dense_dims: NumericMaxRank1,
    bounds_min: NumericMaxRank1 = 0,
    bounds_max: NumericMaxRank1 = 1,
    voxel_center: bool = False,
    device: DeviceIdentifier = "cpu",
) -> GridBatch:
    """
    Create a batch of dense grids defined by axis-aligned world-space bounds.

    Voxel sizes and origins are computed to fit ``dense_dims`` voxels within
    ``[bounds_min, bounds_max]``.

    Args:
        num_grids: Number of grids to create.
        dense_dims: Dimensions of the dense grids, broadcastable to ``(3,)``, integer dtype.
        bounds_min: Minimum world-space coordinate, broadcastable to ``(3,)``, floating dtype.
        bounds_max: Maximum world-space coordinate, broadcastable to ``(3,)``, floating dtype.
        voxel_center: Whether bounds correspond to voxel centers (``True``) or edges (``False``).
        device: Device to create on. Defaults to ``"cpu"``.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    dense_dims_t = to_Vec3iBroadcastable(dense_dims, value_constraint=ValueConstraint.POSITIVE)
    bounds_min_t = to_Vec3fBroadcastable(bounds_min)
    bounds_max_t = to_Vec3fBroadcastable(bounds_max)

    if torch.any(bounds_max_t <= bounds_min_t):
        raise ValueError("bounds_max must be greater than bounds_min in all axes")

    if voxel_center:
        voxel_size = (bounds_max_t - bounds_min_t) / (dense_dims_t.to(torch.float64) - 1.0)
        origin = bounds_min_t
    else:
        voxel_size = (bounds_max_t - bounds_min_t) / dense_dims_t.to(torch.float64)
        origin = bounds_min_t + 0.5 * voxel_size

    return gridbatch_from_dense(num_grids, dense_dims=dense_dims_t, voxel_sizes=voxel_size, origins=origin, device=device)


def gridbatch_from_ijk(
    ijk: JaggedTensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> GridBatch:
    """
    Create a batch of grids from voxel-space coordinates.

    Args:
        ijk: Per-grid voxel coordinates. Shape: ``(B, -1, 3)`` with integer dtype.
        voxel_sizes: Voxel size per grid, broadcastable to ``(B, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(B, 3)``, floating dtype.
        device: Device to create on. Defaults to ijk's device.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    resolved_device = resolve_device(device, inherit_from=ijk)

    voxel_sizes_t = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)

    n = ijk.num_tensors
    grid_data = _fvdb_cpp.gridbatch_from_ijk(
        ijk._impl, _to_vec3d_batch(voxel_sizes_t, n), _to_vec3d_batch(origins_t, n))
    return _wrap_grid(grid_data)


def gridbatch_from_mesh(
    mesh_vertices: JaggedTensor,
    mesh_faces: JaggedTensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> GridBatch:
    """
    Create a grid batch by voxelizing triangle mesh surfaces.

    Args:
        mesh_vertices: Per-grid vertex positions. Shape: ``(B, -1, 3)``.
        mesh_faces: Per-grid face indices. Shape: ``(B, -1, 3)``.
        voxel_sizes: Voxel size per grid, broadcastable to ``(B, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(B, 3)``, floating dtype.
        device: Device to create on. Defaults to mesh_vertices' device.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    resolved_device = resolve_device(device, inherit_from=mesh_vertices)

    voxel_sizes_t = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)

    n = mesh_vertices.num_tensors
    grid_data = _fvdb_cpp.gridbatch_from_mesh(
        mesh_vertices._impl, mesh_faces._impl,
        _to_vec3d_batch(voxel_sizes_t, n), _to_vec3d_batch(origins_t, n))
    return _wrap_grid(grid_data)


def gridbatch_from_nearest_voxels_to_points(
    points: JaggedTensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> GridBatch:
    """
    Create grids by adding the eight nearest voxels to every input point.

    Args:
        points: Per-grid point positions. Shape: ``(B, -1, 3)``.
        voxel_sizes: Voxel size per grid, broadcastable to ``(B, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(B, 3)``, floating dtype.
        device: Device to create on. Defaults to points' device.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    resolved_device = resolve_device(device, inherit_from=points)

    voxel_sizes_t = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)

    n = points.num_tensors
    grid_data = _fvdb_cpp.gridbatch_from_nearest_voxels_to_points(
        points._impl, _to_vec3d_batch(voxel_sizes_t, n), _to_vec3d_batch(origins_t, n))
    return _wrap_grid(grid_data)


def gridbatch_from_points(
    points: JaggedTensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> GridBatch:
    """
    Create a batch of grids from point clouds.

    Args:
        points: Per-grid point positions. Shape: ``(B, -1, 3)``.
        voxel_sizes: Voxel size per grid, broadcastable to ``(B, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(B, 3)``, floating dtype.
        device: Device to create on. Defaults to points' device.

    Returns:
        A new :class:`~fvdb.GridBatch`.
    """
    resolved_device = resolve_device(device, inherit_from=points)

    voxel_sizes_t = to_Vec3fBatchBroadcastable(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)

    n = points.num_tensors
    grid_data = _fvdb_cpp.gridbatch_from_points(
        points._impl, _to_vec3d_batch(voxel_sizes_t, n), _to_vec3d_batch(origins_t, n))
    return _wrap_grid(grid_data)


# ---------------------------------------------------------------------------
#  Empty grid creation
# ---------------------------------------------------------------------------


def gridbatch_from_zero_grids(device: DeviceIdentifier = "cpu") -> GridBatch:
    """
    Create a grid batch with zero grids.

    Args:
        device: Device to create on. Defaults to ``"cpu"``.

    Returns:
        A new empty :class:`~fvdb.GridBatch` with ``grid_count == 0``.
    """
    return _wrap_grid(_fvdb_cpp.create_from_empty(str(resolve_device(device))))


def gridbatch_from_zero_voxels(
    device: DeviceIdentifier = "cpu",
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
) -> GridBatch:
    """
    Create a grid batch with one or more zero-voxel grids.

    Args:
        device: Device to create on. Defaults to ``"cpu"``.
        voxel_sizes: Voxel size per grid, broadcastable to ``(num_grids, 3)``, floating dtype.
        origins: Origin per grid, broadcastable to ``(num_grids, 3)``, floating dtype.

    Returns:
        A new :class:`~fvdb.GridBatch` with zero-voxel grids.
    """
    resolved_device = resolve_device(device)
    voxel_sizes_t = to_Vec3fBatch(voxel_sizes, value_constraint=ValueConstraint.POSITIVE)
    origins_t = to_Vec3fBatch(origins)
    return _wrap_grid(_fvdb_cpp.create_from_empty(
        str(resolved_device),
        _to_vec3d_batch(voxel_sizes_t),
        _to_vec3d_batch(origins_t)))


# ---------------------------------------------------------------------------
#  Concatenation
# ---------------------------------------------------------------------------


def concatenate_grids(grids: Sequence[GridBatch]) -> GridBatch:
    """
    Concatenate a sequence of grid batches into one.

    Args:
        grids: Grid batches to concatenate.

    Returns:
        A new :class:`~fvdb.GridBatch` containing all grids.
    """
    from ..grid_batch import GridBatch as GB

    grid_datas = []
    for grid in grids:
        if not isinstance(grid, GB):
            raise TypeError(f"Expected GridBatch, got {type(grid)}")
        grid_datas.append(grid.data)
    return _wrap_grid(_fvdb_cpp.concatenate_grids(grid_datas))


# ---------------------------------------------------------------------------
#  Single-grid constructors (Grid + torch.Tensor)
# ---------------------------------------------------------------------------


def _wrap_single_grid(cpp_impl):
    from ..grid import Grid

    return Grid(data=cpp_impl)


def grid_from_dense(
    dense_dims: NumericMaxRank1,
    ijk_min: NumericMaxRank1 = 0,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    mask: torch.Tensor | None = None,
    device: DeviceIdentifier | None = None,
) -> Grid:
    """Create a single dense grid."""
    gb = gridbatch_from_dense(1, dense_dims, ijk_min, voxel_sizes, origins, mask, device)
    return _wrap_single_grid(gb.data)


def grid_from_dense_axis_aligned_bounds(
    dense_dims: NumericMaxRank1,
    bounds_min: NumericMaxRank1 = 0,
    bounds_max: NumericMaxRank1 = 1,
    voxel_center: bool = False,
    device: DeviceIdentifier = "cpu",
) -> Grid:
    """Create a single dense grid defined by axis-aligned world-space bounds."""
    gb = gridbatch_from_dense_axis_aligned_bounds(1, dense_dims, bounds_min, bounds_max, voxel_center, device)
    return _wrap_single_grid(gb.data)


def grid_from_ijk(
    ijk: torch.Tensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> Grid:
    """Create a single grid from voxel-space coordinates."""
    jt = JaggedTensor(ijk)
    gb = gridbatch_from_ijk(jt, voxel_sizes, origins, device)
    return _wrap_single_grid(gb.data)


def grid_from_mesh(
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> Grid:
    """Create a single grid by voxelizing a triangle mesh surface."""
    verts_jt = JaggedTensor(mesh_vertices)
    faces_jt = JaggedTensor(mesh_faces)
    gb = gridbatch_from_mesh(verts_jt, faces_jt, voxel_sizes, origins, device)
    return _wrap_single_grid(gb.data)


def grid_from_nearest_voxels_to_points(
    points: torch.Tensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> Grid:
    """Create a single grid by adding the eight nearest voxels to every input point."""
    jt = JaggedTensor(points)
    gb = gridbatch_from_nearest_voxels_to_points(jt, voxel_sizes, origins, device)
    return _wrap_single_grid(gb.data)


def grid_from_points(
    points: torch.Tensor,
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
    device: DeviceIdentifier | None = None,
) -> Grid:
    """Create a single grid from a point cloud."""
    jt = JaggedTensor(points)
    gb = gridbatch_from_points(jt, voxel_sizes, origins, device)
    return _wrap_single_grid(gb.data)


def grid_from_zero_voxels(
    device: DeviceIdentifier = "cpu",
    voxel_sizes: NumericMaxRank2 = 1,
    origins: NumericMaxRank2 = 0,
) -> Grid:
    """Create a single grid with zero voxels."""
    gb = gridbatch_from_zero_voxels(device, voxel_sizes, origins)
    return _wrap_single_grid(gb.data)
