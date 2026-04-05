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
    """Enumerate voxels intersected by rays using a DDA traversal on a grid batch.

    Args:
        grid (GridBatch): The grid batch to trace through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        max_voxels (int): Maximum number of voxels to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.
        return_ijk (bool): If ``True``, return voxel coordinates instead of linear indices.
        cumulative (bool): If ``True``, return cumulative indices across the batch.

    Returns:
        voxels (JaggedTensor): Voxel coordinates or linear indices per ray hit.
        distances (JaggedTensor): ``(t_entry, t_exit)`` pairs per ray hit.

    .. seealso:: :func:`voxels_along_rays_single`
    """
    grid_data = grid.data
    result = _fvdb_cpp.voxels_along_rays(
        grid_data,
        ray_origins._impl,
        ray_directions._impl,
        max_voxels,
        eps,
        return_ijk,
        cumulative,
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


def segments_along_rays_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """Return continuous segments of ray traversal through a grid batch.

    Args:
        grid (GridBatch): The grid batch to trace through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        max_segments (int): Maximum number of segments to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        segments (JaggedTensor): ``(t_start, t_end)`` pairs per ray segment.

    .. seealso:: :func:`segments_along_rays_single`
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
    """Generate uniformly spaced samples along rays that intersect active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to sample through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        t_min (JaggedTensor): Minimum ray distances per ray.
        t_max (JaggedTensor): Maximum ray distances per ray.
        step_size (float): Distance between consecutive samples.
        cone_angle (float): Cone angle for mip-mapping. Default ``0.0``.
        include_end_segments (bool): Include segment endpoints. Default ``True``.
        return_midpoints (bool): Return midpoints instead of boundaries. Default ``False``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        samples (JaggedTensor): Sample distances along each ray.

    .. seealso:: :func:`uniform_ray_samples_single`
    """
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
    """Find ray intersections with an implicit surface defined by grid scalars on a grid batch.

    Args:
        grid (GridBatch): The grid batch defining the implicit surface topology.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        grid_scalars (JaggedTensor): Per-voxel scalar values defining the implicit surface.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        distances (JaggedTensor): Intersection distance per ray, or ``-1`` if no intersection.

    .. seealso:: :func:`ray_implicit_intersection_single`
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
    """Check whether rays hit any voxels in a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        hit (JaggedTensor): Boolean mask indicating whether each ray hit a voxel.

    .. seealso:: :func:`rays_intersect_voxels_single`
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
    """Enumerate voxels intersected by rays using a DDA traversal on a single grid.

    Args:
        grid (Grid): The single grid to trace through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        max_voxels (int): Maximum number of voxels to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.
        return_ijk (bool): If ``True``, return voxel coordinates instead of linear indices.

    Returns:
        voxels (JaggedTensor): Voxel coordinates or linear indices per ray hit.
        distances (JaggedTensor): ``(t_entry, t_exit)`` pairs per ray hit.

    .. seealso:: :func:`voxels_along_rays_batch`
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    result = _fvdb_cpp.voxels_along_rays(
        grid_data,
        origins_jt._impl,
        directions_jt._impl,
        max_voxels,
        eps,
        return_ijk,
        False,
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

    Args:
        grid (Grid): The single grid to trace through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        max_segments (int): Maximum number of segments to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        segments (JaggedTensor): ``(t_start, t_end)`` pairs per ray segment.

    .. seealso:: :func:`segments_along_rays_batch`
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
    """Generate uniformly spaced samples along rays that intersect active voxels of a single grid.

    Args:
        grid (Grid): The single grid to sample through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        t_min (torch.Tensor): Minimum ray distances per ray.
        t_max (torch.Tensor): Maximum ray distances per ray.
        step_size (float): Distance between consecutive samples.
        cone_angle (float): Cone angle for mip-mapping. Default ``0.0``.
        include_end_segments (bool): Include segment endpoints. Default ``True``.
        return_midpoints (bool): Return midpoints instead of boundaries. Default ``False``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        samples (JaggedTensor): Sample distances along each ray.

    .. seealso:: :func:`uniform_ray_samples_batch`
    """
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
    """Find ray intersections with an implicit surface defined by grid scalars on a single grid.

    Args:
        grid (Grid): The single grid defining the implicit surface topology.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        grid_scalars (torch.Tensor): Per-voxel scalar values defining the implicit surface.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        distances (torch.Tensor): Intersection distance per ray, or ``-1`` if no intersection.

    .. seealso:: :func:`ray_implicit_intersection_batch`
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

    Args:
        grid (Grid): The single grid to test against.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        hit (torch.Tensor): Boolean mask indicating whether each ray hit a voxel.

    .. seealso:: :func:`rays_intersect_voxels_batch`
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
