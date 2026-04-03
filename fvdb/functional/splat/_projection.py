# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for Gaussian projection (analytic and camera-dispatched)."""
from __future__ import annotations

from typing import Optional

import torch

from ... import _fvdb_cpp as _C
from ._tile_intersection import build_render_settings


def project_gaussians(
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
) -> _C.ProjectedGaussianSplats:
    """Project 3D Gaussians onto 2D image planes using analytic projection.

    This is a pure functional interface -- no in-place mutation. Supports
    backpropagation through the C++ autograd.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        logit_opacities: ``[N]`` Logit opacities.
        sh0: ``[N, 1, D]`` Degree-0 SH coefficients.
        shN: ``[N, K-1, D]`` Higher-degree SH coefficients.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection/intrinsic matrices.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        near_plane: Near clipping plane.
        far_plane: Far clipping plane.
        sh_degree_to_use: SH degree (-1 for all).
        tile_size: Tile size for tiled rasterization.
        radius_clip: Minimum projected radius.
        eps_2d: Epsilon for numerical stability.
        antialias: Whether to apply antialiasing.
        render_mode: One of ``"rgb"``, ``"depth"``, ``"rgbd"``.
        camera_model: Camera distortion model.

    Returns:
        A :class:`ProjectedGaussianSplats` containing 2D projections, tile
        intersection data, and evaluated render quantities.
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
    return _C.gsplat_project_gaussians_analytic(
        means, quats, log_scales, logit_opacities, sh0, shN,
        world_to_camera_matrices, projection_matrices, settings, camera_model,
    )


def project_gaussians_for_camera(
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
) -> _C.ProjectedGaussianSplats:
    """Project 3D Gaussians, dispatching between analytic and UT projection.

    Validates camera arguments and resolves the projection method automatically
    (analytic for pinhole/orthographic, unscented transform for OpenCV distortion).

    This is a pure functional interface -- no in-place mutation. Supports
    backpropagation through the C++ autograd.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        logit_opacities: ``[N]`` Logit opacities.
        sh0: ``[N, 1, D]`` Degree-0 SH coefficients.
        shN: ``[N, K-1, D]`` Higher-degree SH coefficients.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection/intrinsic matrices.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
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
        distortion_coeffs: ``[C, 12]`` Optional OpenCV distortion coefficients.

    Returns:
        A :class:`ProjectedGaussianSplats` containing 2D projections, tile
        intersection data, and evaluated render quantities.
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
    return _C.gsplat_project_gaussians_for_camera(
        means, quats, log_scales, logit_opacities, sh0, shN,
        world_to_camera_matrices, projection_matrices, settings,
        camera_model, projection_method, distortion_coeffs,
    )
