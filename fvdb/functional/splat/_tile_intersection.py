# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Helpers for constructing RenderSettings from Python."""
from __future__ import annotations

from ... import _fvdb_cpp as _C


def build_render_settings(
    image_width: int,
    image_height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    sh_degree_to_use: int = -1,
    render_mode: str = "rgb",
) -> _C.RenderSettings:
    """Build a RenderSettings object for use with functional splat ops.

    Args:
        image_width: Width of the output image in pixels.
        image_height: Height of the output image in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
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
    settings.near_plane = near_plane
    settings.far_plane = far_plane
    settings.tile_size = tile_size
    settings.radius_clip = radius_clip
    settings.eps_2d = eps_2d
    settings.antialias = antialias
    settings.sh_degree_to_use = sh_degree_to_use

    mode_map = {"rgb": _C.RenderMode.RGB, "depth": _C.RenderMode.DEPTH, "rgbd": _C.RenderMode.RGBD}
    settings.render_mode = mode_map[render_mode.lower()]

    return settings
