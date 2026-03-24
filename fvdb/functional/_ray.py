# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for ray operations on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


@overload
def voxels_along_rays(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]: ...


@overload
def voxels_along_rays(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = True,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]: ...


def voxels_along_rays(
    grid: Grid | GridBatch,
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
        cumulative: If ``True``, return batch-cumulative indices (GridBatch only).

    Returns:
        A tuple ``(voxels, distances)`` where ``voxels`` contains ijk coordinates
        or linear indices, and ``distances`` contains ``(t_entry, t_exit)`` pairs.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_o = JaggedTensor(ray_origins)
        jt_d = JaggedTensor(ray_directions)
        voxels_list, times_list = grid.data.voxels_along_rays(jt_o._impl, jt_d._impl, max_voxels, eps, return_ijk, True)
        return JaggedTensor(impl=voxels_list[0]), JaggedTensor(impl=times_list[0])
    rv, rt = grid.data.voxels_along_rays(ray_origins._impl, ray_directions._impl, max_voxels, eps, return_ijk, cumulative)
    return JaggedTensor(impl=rv), JaggedTensor(impl=rt)


@overload
def segments_along_rays(
    grid: Grid,
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
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_o = JaggedTensor(ray_origins)
        jt_d = JaggedTensor(ray_directions)
        return JaggedTensor(impl=grid.data.segments_along_rays(jt_o._impl, jt_d._impl, max_segments, eps)[0])
    return JaggedTensor(impl=grid.data.segments_along_rays(ray_origins._impl, ray_directions._impl, max_segments, eps))


@overload
def uniform_ray_samples(
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
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_o = JaggedTensor(ray_origins)
        jt_d = JaggedTensor(ray_directions)
        jt_mn = JaggedTensor(t_min)
        jt_mx = JaggedTensor(t_max)
        return JaggedTensor(
            impl=grid.data.uniform_ray_samples(
                jt_o._impl, jt_d._impl, jt_mn._impl, jt_mx._impl,
                step_size, cone_angle, include_end_segments, return_midpoints, eps,
            )[0]
        )
    return JaggedTensor(
        impl=grid.data.uniform_ray_samples(
            ray_origins._impl, ray_directions._impl, t_min._impl, t_max._impl,
            step_size, cone_angle, include_end_segments, return_midpoints, eps,
        )
    )


@overload
def ray_implicit_intersection(
    grid: Grid,
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
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_o = JaggedTensor(ray_origins)
        jt_d = JaggedTensor(ray_directions)
        jt_s = JaggedTensor(grid_scalars)
        return grid.data.ray_implicit_intersection(jt_o._impl, jt_d._impl, jt_s._impl, eps).jdata
    return JaggedTensor(
        impl=grid.data.ray_implicit_intersection(ray_origins._impl, ray_directions._impl, grid_scalars._impl, eps)
    )
