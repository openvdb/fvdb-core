# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Helpers for tile intersection and RenderSettings construction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ... import _fvdb_cpp as _C
from ..._fvdb_cpp import RenderMode, RenderSettings


def build_render_settings(
    image_width: int,
    image_height: int,
    near: float = 0.01,
    far: float = 1e10,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    sh_degree_to_use: int = -1,
    render_mode: str = "rgb",
) -> RenderSettings:
    """Build a RenderSettings object for use with functional splat ops.

    Args:
        image_width: Width of the output image in pixels.
        image_height: Height of the output image in pixels.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        tile_size: Tile size for tiled rasterization.
        radius_clip: Minimum projected radius; Gaussians below this are culled.
        eps_2d: Epsilon for 2D projection numerical stability.
        antialias: Whether to apply antialiasing.
        sh_degree_to_use: SH degree to use (-1 for all available).
        render_mode: One of ``"rgb"``, ``"depth"``, ``"rgbd"``.

    Returns:
        A ``RenderSettings`` object.
    """
    settings = _C.RenderSettings()
    settings.image_width = image_width
    settings.image_height = image_height
    settings.near_plane = near
    settings.far_plane = far
    settings.tile_size = tile_size
    settings.radius_clip = radius_clip
    settings.eps_2d = eps_2d
    settings.antialias = antialias
    settings.sh_degree_to_use = sh_degree_to_use

    mode_map = {
        "rgb": RenderMode.RGB,
        "depth": RenderMode.DEPTH,
        "rgbd": RenderMode.RGBD,
    }
    settings.render_mode = mode_map[render_mode.lower()]

    return settings


@dataclass(frozen=True)
class TileIntersection:
    """Result of tile-based Gaussian culling.

    Identifies which Gaussians overlap which screen-space tiles, producing
    sorted lists for the tiled rasterizer.

    Attributes:
        tile_offsets: Per-tile start offsets into ``tile_gaussian_ids``.
        tile_gaussian_ids: Sorted Gaussian indices per tile.
    """

    tile_offsets: torch.Tensor
    tile_gaussian_ids: torch.Tensor


def intersect_tiles(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    num_cameras: int,
    tile_size: int,
    image_width: int,
    image_height: int,
) -> TileIntersection:
    """Compute tile-Gaussian intersections for tiled rasterization.

    This is a composable pipeline stage (non-differentiable), meant to be
    called after :func:`~fvdb.functional.splat.project_to_2d`.

    Args:
        means2d: ``[C, N, 2]`` Projected 2D means.
        radii: ``[C, N]`` int32 projected radii.
        depths: ``[C, N]`` Depths along camera z-axis.
        num_cameras: Number of cameras ``C``.
        tile_size: Tile size in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        A :class:`TileIntersection` with tile offsets and sorted Gaussian IDs.
    """
    num_tiles_w = math.ceil(image_width / tile_size)
    num_tiles_h = math.ceil(image_height / tile_size)
    tile_offsets, tile_gaussian_ids = _C.gsplat_tile_intersection(
        means2d, radii, depths, num_cameras, tile_size, num_tiles_h, num_tiles_w
    )
    return TileIntersection(tile_offsets=tile_offsets, tile_gaussian_ids=tile_gaussian_ids)
