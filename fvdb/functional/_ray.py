# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for ray operations on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ._dispatch import _get_grid_data, _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


@overload
def voxels_along_rays(
    grid: GridBatch,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]: ...


@overload
def voxels_along_rays(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]: ...


def voxels_along_rays(
    grid: GridBatch,
    ray_origins: torch.Tensor | JaggedTensor,
    ray_directions: torch.Tensor | JaggedTensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]:
    """
    Enumerate voxels intersected by rays using a DDA traversal.

    Args:
        grid: The grid to traverse.
        ray_origins: Ray start positions in world space.
        ray_directions: Ray direction vectors.
        max_voxels: Maximum number of voxels to return per ray.
        eps: Epsilon for numerical stability.
        return_ijk: If ``True``, return voxel coordinates; otherwise linear indices.
        cumulative: If ``True``, return batch-cumulative indices.

    Returns:
        A tuple ``(voxels, distances)`` where ``voxels`` contains ijk coordinates
        or linear indices, and ``distances`` contains ``(t_entry, t_exit)`` pairs.
    """
    grid_data, (ray_origins_jt, ray_directions_jt), _ = _prepare_args(grid, ray_origins, ray_directions)
    assert ray_origins_jt is not None and ray_directions_jt is not None
    result = _fvdb_cpp.voxels_along_rays(
        grid_data, ray_origins_jt._impl, ray_directions_jt._impl, max_voxels, eps, return_ijk, cumulative
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


@overload
def segments_along_rays(
    grid: GridBatch,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor: ...


@overload
def segments_along_rays(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor: ...


def segments_along_rays(
    grid: GridBatch,
    ray_origins: torch.Tensor | JaggedTensor,
    ray_directions: torch.Tensor | JaggedTensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """
    Return continuous segments of ray traversal through the grid. Each segment
    is a ``(t_start, t_end)`` pair of distances along the ray.

    Args:
        grid: The grid to traverse.
        ray_origins: Ray start positions in world space.
        ray_directions: Ray direction vectors.
        max_segments: Maximum number of segments per ray.
        eps: Epsilon for numerical stability.

    Returns:
        A :class:`~fvdb.JaggedTensor` of segments with eshape ``(2,)``.
    """
    grid_data, (ray_origins_jt, ray_directions_jt), _ = _prepare_args(grid, ray_origins, ray_directions)
    assert ray_origins_jt is not None and ray_directions_jt is not None
    return JaggedTensor(
        impl=_fvdb_cpp.segments_along_rays(grid_data, ray_origins_jt._impl, ray_directions_jt._impl, max_segments, eps)
    )


@overload
def uniform_ray_samples(
    grid: GridBatch,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
    step_size: float,
    cone_angle: float = 0.0,
    include_end_segments: bool = True,
    return_midpoints: bool = False,
    eps: float = 0.0,
) -> JaggedTensor: ...


@overload
def uniform_ray_samples(
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
) -> JaggedTensor: ...


def uniform_ray_samples(
    grid: GridBatch,
    ray_origins: torch.Tensor | JaggedTensor,
    ray_directions: torch.Tensor | JaggedTensor,
    t_min: torch.Tensor | JaggedTensor,
    t_max: torch.Tensor | JaggedTensor,
    step_size: float,
    cone_angle: float = 0.0,
    include_end_segments: bool = True,
    return_midpoints: bool = False,
    eps: float = 0.0,
) -> JaggedTensor:
    """
    Generate uniformly spaced samples along rays that intersect active voxels.

    Args:
        grid: The grid to intersect.
        ray_origins: Ray start positions in world space.
        ray_directions: Ray direction vectors.
        t_min: Minimum distance along rays.
        t_max: Maximum distance along rays.
        step_size: Distance between samples.
        cone_angle: Cone angle for cone tracing (radians).
        include_end_segments: Include partial segments at ray ends.
        return_midpoints: Return midpoints instead of segment bounds.
        eps: Epsilon for numerical stability.

    Returns:
        A :class:`~fvdb.JaggedTensor` of sample distances.
    """
    grid_data, (ray_origins_jt, ray_directions_jt, t_min_jt, t_max_jt), _ = _prepare_args(
        grid, ray_origins, ray_directions, t_min, t_max
    )
    assert ray_origins_jt is not None and ray_directions_jt is not None
    assert t_min_jt is not None and t_max_jt is not None
    return JaggedTensor(
        impl=_fvdb_cpp.uniform_ray_samples(
            grid_data,
            ray_origins_jt._impl,
            ray_directions_jt._impl,
            t_min_jt._impl,
            t_max_jt._impl,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )
    )


@overload
def ray_implicit_intersection(
    grid: GridBatch,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_scalars: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor: ...


@overload
def ray_implicit_intersection(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    grid_scalars: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor: ...


def ray_implicit_intersection(
    grid: GridBatch,
    ray_origins: torch.Tensor | JaggedTensor,
    ray_directions: torch.Tensor | JaggedTensor,
    grid_scalars: torch.Tensor | JaggedTensor,
    eps: float = 0.0,
) -> torch.Tensor | JaggedTensor:
    """
    Find ray intersections with an implicit surface defined by grid scalars
    (e.g. a signed distance function).

    Args:
        grid: The grid containing the scalar field.
        ray_origins: Ray start positions in world space.
        ray_directions: Ray direction vectors (should be normalized).
        grid_scalars: Scalar field values at each voxel.
        eps: Epsilon for numerical stability.

    Returns:
        Intersection distance along each ray, or ``-1`` if no intersection.
    """
    grid_data, (ray_origins_jt, ray_directions_jt, grid_scalars_jt), unwrap = _prepare_args(
        grid, ray_origins, ray_directions, grid_scalars
    )
    assert ray_origins_jt is not None and ray_directions_jt is not None and grid_scalars_jt is not None
    result_impl = _fvdb_cpp.ray_implicit_intersection(
        grid_data, ray_origins_jt._impl, ray_directions_jt._impl, grid_scalars_jt._impl, eps
    )
    return unwrap(result_impl.jdata)


@overload
def rays_intersect_voxels(
    grid: GridBatch,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    eps: float = 0.0,
) -> JaggedTensor: ...


@overload
def rays_intersect_voxels(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor: ...


def rays_intersect_voxels(
    grid: GridBatch,
    ray_origins: torch.Tensor | JaggedTensor,
    ray_directions: torch.Tensor | JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor:
    """
    Check whether rays hit any voxels in the grid.

    Args:
        grid: The grid to test against.
        ray_origins: Ray start positions in world space.
        ray_directions: Ray direction vectors.
        eps: Epsilon for numerical stability.

    Returns:
        A boolean :class:`~fvdb.JaggedTensor` indicating whether each ray hit
        a voxel.
    """
    # Normalize to JaggedTensor for the internal call
    if isinstance(ray_origins, torch.Tensor):
        ray_origins_jt = JaggedTensor(ray_origins)
    else:
        ray_origins_jt = ray_origins
    if isinstance(ray_directions, torch.Tensor):
        ray_directions_jt = JaggedTensor(ray_directions)
    else:
        ray_directions_jt = ray_directions

    _, ray_times = voxels_along_rays(
        grid,
        ray_origins=ray_origins_jt,
        ray_directions=ray_directions_jt,
        max_voxels=1,
        eps=eps,
        return_ijk=False,
        cumulative=False,
    )

    did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
    return ray_origins_jt.jagged_like(did_hit)
