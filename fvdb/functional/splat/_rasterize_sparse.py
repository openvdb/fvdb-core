# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse Gaussian rasterization."""
from __future__ import annotations

from typing import Optional

import torch

from ... import _fvdb_cpp as _C
from ...jagged_tensor import JaggedTensor
from ._tile_intersection import build_render_settings


def sparse_render(
    pixels_to_render: JaggedTensor,
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    render_mode: str = "rgb",
    camera_model: _C.CameraModel = _C.CameraModel.PINHOLE,
    projection_method: _C.ProjectionMethod = _C.ProjectionMethod.AUTO,
    distortion_coeffs: Optional[torch.Tensor] = None,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Render Gaussians at specified sparse pixel locations.

    Full pipeline: projects Gaussians, rasterizes at the given pixels, and
    handles duplicate pixel deduplication/scatter-back. Pure functional
    interface with no in-place mutation.

    Supports backpropagation through the C++ autograd.

    Args:
        pixels_to_render: JaggedTensor of ``[C, num_pixels, 2]`` pixel coordinates.
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        logit_opacities: ``[N]`` Logit opacities.
        sh0: ``[N, 1, D]`` Degree-0 SH coefficients.
        shN: ``[N, K-1, D]`` Higher-degree SH coefficients.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection matrices.
        image_width: Full image width (for tile computation).
        image_height: Full image height (for tile computation).
        near_plane: Near clipping plane.
        far_plane: Far clipping plane.
        sh_degree_to_use: SH degree (-1 for all).
        tile_size: Tile size for tiled rasterization.
        radius_clip: Minimum projected radius.
        eps_2d: Epsilon for numerical stability.
        antialias: Whether to apply antialiasing.
        render_mode: One of ``"rgb"``, ``"depth"``, ``"rgbd"``.
        camera_model: Camera distortion model.
        projection_method: Projection method (AUTO, ANALYTIC, or UNSCENTED).
        distortion_coeffs: ``[C, 12]`` Optional distortion coefficients.
        backgrounds: ``[C, D]`` Optional per-camera backgrounds.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_features, rendered_alphas) as JaggedTensors.
    """
    settings = build_render_settings(
        image_width=image_width,
        image_height=image_height,
        near_plane=near_plane,
        far_plane=far_plane,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps_2d=eps_2d,
        antialias=antialias,
        sh_degree_to_use=sh_degree_to_use,
        render_mode=render_mode,
    )
    return _C.gsplat_sparse_render(
        pixels_to_render._impl, means, quats, log_scales, logit_opacities, sh0, shN,
        world_to_camera_matrices, projection_matrices, settings,
        camera_model, projection_method, distortion_coeffs,
        backgrounds, masks,
    )
