# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for ray operations on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Batch variants (GridBatch + JaggedTensor)
# ---------------------------------------------------------------------------


def voxels_along_rays_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Enumerate voxels intersected by rays using a DDA traversal.

    Returns ``(voxels, distances)`` where ``voxels`` contains ijk coordinates
    or linear indices and ``distances`` contains ``(t_entry, t_exit)`` pairs.
    """
    grid_data = grid.data
    result = _fvdb_cpp.voxels_along_rays(
        grid_data, ray_origins._impl, ray_directions._impl, max_voxels, eps, return_ijk, cumulative
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


def segments_along_rays_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """Return continuous segments of ray traversal through the grid.

    Each segment is a ``(t_start, t_end)`` pair of distances along the ray.
    """
    grid_data = grid.data
    return JaggedTensor(
        impl=_fvdb_cpp.segments_along_rays(grid_data, ray_origins._impl, ray_directions._impl, max_segments, eps)
    )


def uniform_ray_samples_batch(
    grid: GridBatch,
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
    """Generate uniformly spaced samples along rays that intersect active voxels."""
    grid_data = grid.data
    return JaggedTensor(
        impl=_fvdb_cpp.uniform_ray_samples(
            grid_data,
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


def ray_implicit_intersection_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    grid_scalars: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor:
    """Find ray intersections with an implicit surface defined by grid scalars.

    Returns intersection distance along each ray, or ``-1`` if no intersection.
    """
    grid_data = grid.data
    result_impl = _fvdb_cpp.ray_implicit_intersection(
        grid_data, ray_origins._impl, ray_directions._impl, grid_scalars._impl, eps
    )
    return JaggedTensor(impl=result_impl)


def rays_intersect_voxels_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor:
    """Check whether rays hit any voxels in the grid.

    Returns a boolean :class:`~fvdb.JaggedTensor` indicating whether each ray
    hit a voxel.
    """
    _, ray_times = voxels_along_rays_batch(
        grid,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        max_voxels=1,
        eps=eps,
        return_ijk=False,
        cumulative=False,
    )
    did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
    return ray_origins.jagged_like(did_hit)


# ---------------------------------------------------------------------------
#  Single variants (Grid + torch.Tensor)
# ---------------------------------------------------------------------------


def voxels_along_rays_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Enumerate voxels intersected by rays using a DDA traversal for a single grid.

    Returns ``(voxels, distances)`` as :class:`~fvdb.JaggedTensor` (per-ray jagged).
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    result = _fvdb_cpp.voxels_along_rays(
        grid_data, origins_jt._impl, directions_jt._impl, max_voxels, eps, return_ijk, False
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


def segments_along_rays_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """Return continuous segments of ray traversal through a single grid.

    Each segment is a ``(t_start, t_end)`` pair of distances along the ray.
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    return JaggedTensor(
        impl=_fvdb_cpp.segments_along_rays(grid_data, origins_jt._impl, directions_jt._impl, max_segments, eps)
    )


def uniform_ray_samples_single(
    grid: Grid,
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
    """Generate uniformly spaced samples along rays that intersect active voxels in a single grid."""
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    t_min_jt = JaggedTensor(t_min)
    t_max_jt = JaggedTensor(t_max)
    return JaggedTensor(
        impl=_fvdb_cpp.uniform_ray_samples(
            grid_data,
            origins_jt._impl,
            directions_jt._impl,
            t_min_jt._impl,
            t_max_jt._impl,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )
    )


def ray_implicit_intersection_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_scalars: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Find ray intersections with an implicit surface for a single grid.

    Returns intersection distance along each ray, or ``-1`` if no intersection.
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    scalars_jt = JaggedTensor(grid_scalars)
    result_impl = _fvdb_cpp.ray_implicit_intersection(
        grid_data, origins_jt._impl, directions_jt._impl, scalars_jt._impl, eps
    )
    return result_impl.jdata


def rays_intersect_voxels_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Check whether rays hit any voxels in a single grid.

    Returns a boolean tensor indicating whether each ray hit a voxel.
    """
    _, ray_times = voxels_along_rays_single(
        grid,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        max_voxels=1,
        eps=eps,
        return_ijk=False,
    )
    did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
    return did_hit
