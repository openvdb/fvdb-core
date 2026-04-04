# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for world-space Gaussian rasterization with geometry gradients."""
from __future__ import annotations

from typing import Any, Optional, cast

import torch

from ... import _fvdb_cpp as _C


# ---------------------------------------------------------------------------
#  Autograd function (raw dispatch wrapper)
# ---------------------------------------------------------------------------


class _RasterizeFromWorldFn(torch.autograd.Function):
    """Python autograd wrapper for the from-world Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        log_scales: torch.Tensor,  # [N, 3]
        features: torch.Tensor,  # [C, N, D]
        opacities: torch.Tensor,  # [C, N]
        world_to_cam_start: torch.Tensor,  # [C, 4, 4]
        world_to_cam_end: torch.Tensor,  # [C, 4, 4]
        projection_matrices: torch.Tensor,  # [C, 3, 3]
        distortion_coeffs: torch.Tensor,  # [C, K]
        rolling_shutter_type: _C.RollingShutterType,
        camera_model: _C.CameraModel,
        settings: _C.RenderSettings,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        backgrounds: Optional[torch.Tensor],  # [C, D] or None
        masks: Optional[torch.Tensor],  # [C, tileH, tileW] or None
    ):
        result = _C.gsplat_rasterize_from_world_fwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            rolling_shutter_type,
            camera_model,
            settings,
            tile_offsets,
            tile_gaussian_ids,
            backgrounds,
            masks,
        )
        rendered_features = result[0]
        rendered_alphas = result[1]
        last_ids = result[2]

        to_save = [
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
        ]
        if backgrounds is not None:
            to_save.append(backgrounds)
            ctx.has_backgrounds = True
        else:
            ctx.has_backgrounds = False
        if masks is not None:
            to_save.append(masks)
            ctx.has_masks = True
        else:
            ctx.has_masks = False
        ctx.save_for_backward(*to_save)

        ctx.image_width = settings.image_width
        ctx.image_height = settings.image_height
        ctx.image_origin_w = settings.image_origin_w
        ctx.image_origin_h = settings.image_origin_h
        ctx.tile_size = settings.tile_size
        ctx.rolling_shutter_type = rolling_shutter_type
        ctx.camera_model = camera_model

        return rendered_features, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_features = grad_outputs[0]
        d_loss_d_rendered_alphas = grad_outputs[1]
        if d_loss_d_rendered_features is not None:
            d_loss_d_rendered_features = d_loss_d_rendered_features.contiguous()
        if d_loss_d_rendered_alphas is not None:
            d_loss_d_rendered_alphas = d_loss_d_rendered_alphas.contiguous()

        saved = ctx.saved_tensors
        means = saved[0]
        quats = saved[1]
        log_scales = saved[2]
        features = saved[3]
        opacities = saved[4]
        world_to_cam_start = saved[5]
        world_to_cam_end = saved[6]
        projection_matrices = saved[7]
        distortion_coeffs = saved[8]
        tile_offsets = saved[9]
        tile_gaussian_ids = saved[10]
        rendered_alphas = saved[11]
        last_ids = saved[12]

        backgrounds: Optional[torch.Tensor] = None
        masks: Optional[torch.Tensor] = None
        opt_idx = 13
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]
            opt_idx += 1

        # Reconstruct RenderSettings for backward
        settings = _C.RenderSettings()
        settings.image_width = ctx.image_width
        settings.image_height = ctx.image_height
        settings.image_origin_w = ctx.image_origin_w
        settings.image_origin_h = ctx.image_origin_h
        settings.tile_size = ctx.tile_size

        assert d_loss_d_rendered_features is not None
        assert d_loss_d_rendered_alphas is not None
        result = _C.gsplat_rasterize_from_world_bwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            cast(_C.RollingShutterType, ctx.rolling_shutter_type),
            cast(_C.CameraModel, ctx.camera_model),
            settings,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_loss_d_rendered_features,
            d_loss_d_rendered_alphas,
            backgrounds,
            masks,
        )
        d_means = result[0]
        d_quats = result[1]
        d_log_scales = result[2]
        d_features = result[3]
        d_opacities = result[4]

        # Order: means, quats, log_scales, features, opacities,
        #   world_to_cam_start, world_to_cam_end, projection_matrices, distortion_coeffs,
        #   rolling_shutter_type, camera_model, settings,
        #   tile_offsets, tile_gaussian_ids, backgrounds, masks
        return (
            d_means,
            d_quats,
            d_log_scales,
            d_features,
            d_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_from_world(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    projected_state: _C.ProjectedGaussianSplats,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    camera_model: _C.CameraModel,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians from world-space with geometry gradients.

    Unlike :func:`rasterize_from_projected`, this path computes gradients with
    respect to the 3D Gaussian geometry (means, quats, log_scales) during
    backpropagation, enabling world-space optimization.

    Supports backpropagation through the C++ autograd.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        projected_state: Pre-projected state from :func:`project_gaussians_for_camera`.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection matrices.
        distortion_coeffs: ``[C, K]`` Distortion coefficients (empty ``[C, 0]`` for none).
        camera_model: Camera distortion model.
        image_width: Output image width.
        image_height: Output image height.
        tile_size: Tile size for tiled rasterization.
        backgrounds: ``[C, D]`` Optional per-camera backgrounds.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    return _C.gsplat_rasterize_from_world(
        means,
        quats,
        log_scales,
        projected_state,
        world_to_camera_matrices,
        projection_matrices,
        distortion_coeffs,
        camera_model,
        image_width,
        image_height,
        tile_size,
        backgrounds,
        masks,
    )
