# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense Gaussian rasterization (Stage 4).

Provides both screen-space and world-space rasterization paths:

- **Screen-space** (``rasterize_screen_space_gaussians``): operates on
  pre-projected 2D Gaussians; used with the analytic projection pipeline.
- **World-space** (``rasterize_world_space_gaussians``): reprojects from 3D
  geometry during backpropagation so that gradients flow through the Gaussian
  means, quats, and log_scales; used with the UT projection pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from .. import _fvdb_cpp as _C
from ..enums import CameraModel, RollingShutterType
from .._fvdb_cpp import RenderSettings

if TYPE_CHECKING:
    from ._gaussian_projection import ProjectedGaussians
    from ._gaussian_tile_intersection import GaussianTileIntersection


# ---------------------------------------------------------------------------
#  Internal opacity computation helper
# ---------------------------------------------------------------------------


def _compute_opacities(logit_opacities: torch.Tensor, projected: ProjectedGaussians) -> torch.Tensor:
    """``[N]`` logit_opacities -> ``[C, N]`` opacities with optional compensation."""
    return compute_gaussian_opacities(logit_opacities, projected)


def compute_gaussian_opacities(logit_opacities: torch.Tensor, projected: ProjectedGaussians) -> torch.Tensor:
    """Convert logit opacities to per-camera opacities with optional compensation.

    Args:
        logit_opacities: ``[N]`` pre-sigmoid opacity logits.
        projected: :class:`ProjectedGaussians` from Stage 1.

    Returns:
        ``[C, N]`` opacities (sigmoid-activated, optionally compensated).
    """
    C = projected.means2d.shape[0]
    opacities = torch.sigmoid(logit_opacities).unsqueeze(0).expand(C, -1)
    if projected.compensations is not None:
        opacities = opacities * projected.compensations
    return opacities


# ---------------------------------------------------------------------------
#  Crop validation
# ---------------------------------------------------------------------------


def _validate_crop(
    crop: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Validate and clamp a ``(origin_x, origin_y, width, height)`` crop rect.

    Returns the clamped ``(origin_x, origin_y, width, height)`` that lies
    within ``[0, image_width) x [0, image_height)``.

    Raises:
        ValueError: If any component is negative or if the clamped region
            has zero area.
    """
    ox, oy, w, h = crop
    if ox < 0 or oy < 0:
        raise ValueError(f"Crop origin must be non-negative, got ({ox}, {oy})")
    if w <= 0 or h <= 0:
        raise ValueError(f"Crop size must be positive, got ({w}, {h})")
    # Clamp so the crop doesn't extend beyond the projected image.
    w = min(w, image_width - ox)
    h = min(h, image_height - oy)
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Crop region (origin=({ox}, {oy}), size=({crop[2]}, {crop[3]})) "
            f"has no overlap with the {image_width}x{image_height} image"
        )
    return ox, oy, w, h


# ===========================================================================
#  Screen-space rasterization
# ===========================================================================


class _RasterizeScreenSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the dense Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,  # [C, N, 2]
        conics: torch.Tensor,  # [C, N, 3]
        colors: torch.Tensor,  # [C, N, D]
        opacities: torch.Tensor,  # [N]
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,  # [C, tile_height, tile_width]
        tile_gaussian_ids: torch.Tensor,  # [n_isects]
        absgrad: bool,
        backgrounds: torch.Tensor | None,  # [C, D] or None
        masks: torch.Tensor | None,  # [C, tileH, tileW] or None
    ):
        result = _C.rasterize_screen_space_gaussians_fwd(
            means2d,
            conics,
            colors,
            opacities,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            backgrounds,
            masks,
        )
        rendered_colors = result[0]
        rendered_alphas = result[1]
        last_ids = result[2]

        to_save = [
            means2d,
            conics,
            colors,
            opacities,
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

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        return rendered_colors, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_colors = grad_outputs[0]
        d_loss_d_rendered_alphas = grad_outputs[1]
        if d_loss_d_rendered_colors is not None:
            d_loss_d_rendered_colors = d_loss_d_rendered_colors.contiguous()
        if d_loss_d_rendered_alphas is not None:
            d_loss_d_rendered_alphas = d_loss_d_rendered_alphas.contiguous()

        saved = ctx.saved_tensors
        means2d = saved[0]
        conics = saved[1]
        colors = saved[2]
        opacities = saved[3]
        tile_offsets = saved[4]
        tile_gaussian_ids = saved[5]
        rendered_alphas = saved[6]
        last_ids = saved[7]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 8
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]
            opt_idx += 1

        assert d_loss_d_rendered_colors is not None
        assert d_loss_d_rendered_alphas is not None
        result = _C.rasterize_screen_space_gaussians_bwd(
            means2d,
            conics,
            colors,
            opacities,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_loss_d_rendered_colors,
            d_loss_d_rendered_alphas,
            ctx.absgrad,
            -1,
            backgrounds,
            masks,
        )
        # result: (dMean2dAbs, dMeans2d, dConics, dColors, dOpacities)
        d_means2d = result[1]
        d_conics = result[2]
        d_colors = result[3]
        d_opacities = result[4]

        return (
            d_means2d,
            d_conics,
            d_colors,
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
        )


def rasterize_screen_space_gaussians(
    projected: ProjectedGaussians,
    features: torch.Tensor,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize screen-space Gaussians to produce images and alpha maps (Stage 4).

    Computes opacities internally from ``logit_opacities`` (sigmoid + optional
    antialiasing compensation from ``projected``).

    Differentiable via Python autograd.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        features: ``[C, N, D]`` Render features from Stage 2.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        tiles: :class:`GaussianTileIntersection` from Stage 3.
        backgrounds: ``[C, D]`` Optional per-camera background colours.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.
        crop: Optional ``(origin_x, origin_y, width, height)`` tuple defining a
            sub-region of the projected image to rasterize.  When ``None``
            (the default), the full image is rendered.  The crop region is
            clamped to the projected image bounds so it is safe to specify a
            region that partially or fully extends beyond the image.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``)
        where H and W are the crop dimensions (or the full image dimensions
        when ``crop`` is ``None``).
    """
    opacities = _compute_opacities(logit_opacities, projected)

    if crop is not None:
        ox, oy, w, h = _validate_crop(crop, tiles.image_width, tiles.image_height)
    else:
        ox, oy, w, h = 0, 0, tiles.image_width, tiles.image_height

    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeScreenSpaceGaussiansFn.apply(
            projected.means2d,
            projected.conics,
            features,
            opacities,
            w,
            h,
            ox,
            oy,
            tiles.tile_size,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            False,  # absgrad
            backgrounds,
            masks,
        ),
    )


# ===========================================================================
#  World-space rasterization
# ===========================================================================


class _RasterizeWorldSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for world-space Gaussian rasterization forward/backward dispatch."""

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
        rolling_shutter_type: RollingShutterType,
        camera_model: CameraModel,
        settings: RenderSettings,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        backgrounds: torch.Tensor | None,  # [C, D] or None
        masks: torch.Tensor | None,  # [C, tileH, tileW] or None
    ):
        result = _C.rasterize_world_space_gaussians_fwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            _C.RollingShutterType(rolling_shutter_type),
            _C.CameraModel(camera_model),
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

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 13
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]
            opt_idx += 1

        settings = _C.RenderSettings()
        settings.image_width = ctx.image_width
        settings.image_height = ctx.image_height
        settings.image_origin_w = ctx.image_origin_w
        settings.image_origin_h = ctx.image_origin_h
        settings.tile_size = ctx.tile_size

        assert d_loss_d_rendered_features is not None
        assert d_loss_d_rendered_alphas is not None
        result = _C.rasterize_world_space_gaussians_bwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            _C.RollingShutterType(ctx.rolling_shutter_type),
            _C.CameraModel(ctx.camera_model),
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


def rasterize_world_space_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    projected: ProjectedGaussians,
    features: torch.Tensor,
    logit_opacities: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    camera_model: CameraModel,
    tiles: GaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians from world-space with geometry gradients (Stage 4).

    Unlike :func:`rasterize_screen_space_gaussians`, this path computes gradients
    with respect to the 3D Gaussian geometry (means, quats, log_scales) during
    backpropagation, enabling world-space optimization for the UT projection
    pipeline.

    Computes opacities internally from ``logit_opacities`` (sigmoid + optional
    antialiasing compensation from ``projected``).

    Differentiable via Python autograd.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        projected: :class:`ProjectedGaussians` from Stage 1.
        features: ``[C, N, D]`` Render features from Stage 2.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection matrices.
        distortion_coeffs: ``[C, K]`` Distortion coefficients (empty ``[C, 0]`` for none).
        camera_model: Camera distortion model.
        tiles: :class:`GaussianTileIntersection` from Stage 3.
        backgrounds: ``[C, D]`` Optional per-camera backgrounds.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    opacities = _compute_opacities(logit_opacities, projected)

    settings = _C.RenderSettings()
    settings.image_width = tiles.image_width
    settings.image_height = tiles.image_height
    settings.tile_size = tiles.tile_size

    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeWorldSpaceGaussiansFn.apply(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_camera_matrices,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            RollingShutterType.NONE,
            camera_model,
            settings,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            backgrounds,
            masks,
        ),
    )
