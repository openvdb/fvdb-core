# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Composite rendering pipelines that chain projection + rasterization."""
from __future__ import annotations


import torch

from ...enums import CameraModel, ProjectionMethod
from ._projection import project_gaussians_for_camera
from ._rasterize import rasterize_from_projected
from ._rasterize_from_world import rasterize_from_world
from ._tile_intersection import build_render_settings


def render_images(
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
    near: float = 0.01,
    far: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB images from Gaussian splats. Projects then rasterizes.

    Returns:
        Tuple of (images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use,
        tile_size,
        radius_clip,
        eps_2d,
        antialias,
        render_mode="rgb",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    return rasterize_from_projected(
        state,
        tile_size,
        image_width,
        image_height,
        0,
        0,
        backgrounds,
        masks,
    )


def render_depths(
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
    near: float = 0.01,
    far: float = 1e10,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render depth maps from Gaussian splats. Projects then rasterizes.

    Returns:
        Tuple of (depths ``[C, H, W, 1]``, alphas ``[C, H, W, 1]``).
    """
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use=-1,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps_2d=eps_2d,
        antialias=antialias,
        render_mode="depth",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    return rasterize_from_projected(
        state,
        tile_size,
        image_width,
        image_height,
        0,
        0,
        backgrounds,
        masks,
    )


def render_images_and_depths(
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
    near: float = 0.01,
    far: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB+depth from Gaussian splats. Projects then rasterizes.

    Returns:
        Tuple of (images_and_depths ``[C, H, W, D+1]``, alphas ``[C, H, W, 1]``).
    """
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use,
        tile_size,
        radius_clip,
        eps_2d,
        antialias,
        render_mode="rgbd",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    return rasterize_from_projected(
        state,
        tile_size,
        image_width,
        image_height,
        0,
        0,
        backgrounds,
        masks,
    )


def render_images_from_world(
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
    near: float = 0.01,
    far: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB images with world-space geometry gradients.

    Returns:
        Tuple of (images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    C = world_to_camera_matrices.size(0)
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use,
        tile_size,
        radius_clip,
        eps_2d,
        antialias,
        render_mode="rgb",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    dc = (
        distortion_coeffs
        if distortion_coeffs is not None
        else torch.empty(C, 0, device=means.device, dtype=means.dtype)
    )
    return rasterize_from_world(
        means,
        quats,
        log_scales,
        state,
        world_to_camera_matrices,
        projection_matrices,
        dc,
        camera_model,
        image_width,
        image_height,
        tile_size,
        backgrounds,
        masks,
    )


def render_depths_from_world(
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
    near: float = 0.01,
    far: float = 1e10,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render depth maps with world-space geometry gradients.

    Returns:
        Tuple of (depths ``[C, H, W, 1]``, alphas ``[C, H, W, 1]``).
    """
    C = world_to_camera_matrices.size(0)
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use=-1,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps_2d=eps_2d,
        antialias=antialias,
        render_mode="depth",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    dc = (
        distortion_coeffs
        if distortion_coeffs is not None
        else torch.empty(C, 0, device=means.device, dtype=means.dtype)
    )
    return rasterize_from_world(
        means,
        quats,
        log_scales,
        state,
        world_to_camera_matrices,
        projection_matrices,
        dc,
        camera_model,
        image_width,
        image_height,
        tile_size,
        backgrounds,
        masks,
    )


def render_images_and_depths_from_world(
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
    near: float = 0.01,
    far: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB+depth with world-space geometry gradients.

    Returns:
        Tuple of (images_and_depths ``[C, H, W, D+1]``, alphas ``[C, H, W, 1]``).
    """
    C = world_to_camera_matrices.size(0)
    state = project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        image_width,
        image_height,
        near,
        far,
        sh_degree_to_use,
        tile_size,
        radius_clip,
        eps_2d,
        antialias,
        render_mode="rgbd",
        camera_model=camera_model,
        projection_method=projection_method,
        distortion_coeffs=distortion_coeffs,
    )
    dc = (
        distortion_coeffs
        if distortion_coeffs is not None
        else torch.empty(C, 0, device=means.device, dtype=means.dtype)
    )
    return rasterize_from_world(
        means,
        quats,
        log_scales,
        state,
        world_to_camera_matrices,
        projection_matrices,
        dc,
        camera_model,
        image_width,
        image_height,
        tile_size,
        backgrounds,
        masks,
    )
