# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense Gaussian rasterization."""
from __future__ import annotations

from typing import Any, cast

import torch

from ... import _fvdb_cpp as _C
from ._projected_gaussians import ProjectedGaussians


# ---------------------------------------------------------------------------
#  Autograd function (raw dispatch wrapper)
# ---------------------------------------------------------------------------


class _RasterizeDenseFn(torch.autograd.Function):
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
        result = _C.gsplat_rasterize_fwd(
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
        result = _C.gsplat_rasterize_bwd(
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

        # Order: means2d, conics, colors, opacities,
        #   image_width, image_height, image_origin_w, image_origin_h,
        #   tile_size, tile_offsets, tile_gaussian_ids, absgrad,
        #   backgrounds, masks
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


def rasterize_from_projected(
    projected_gaussians: ProjectedGaussians,
    tile_size: int = 16,
    crop_width: int = -1,
    crop_height: int = -1,
    crop_origin_w: int = -1,
    crop_origin_h: int = -1,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize pre-projected Gaussians to produce images and alpha maps.

    Supports backpropagation through the Python autograd.

    Args:
        projected_gaussians: Pre-projected state from :func:`project_gaussians`.
        tile_size: Tile size for tiled rasterization.
        crop_width: Crop width (-1 for full image).
        crop_height: Crop height (-1 for full image).
        crop_origin_w: Crop origin W (-1 for no crop).
        crop_origin_h: Crop origin H (-1 for no crop).
        backgrounds: ``[C, D]`` Optional per-camera background colors.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    p = projected_gaussians
    return _C.gsplat_render_crop_from_projected(
        p.means2d,
        p.conics,
        p.render_quantities,
        p.opacities,
        p.tile_offsets,
        p.tile_gaussian_ids,
        p.image_width,
        p.image_height,
        tile_size,
        crop_width,
        crop_height,
        crop_origin_w,
        crop_origin_h,
        backgrounds,
        masks,
    )


def rasterize_dense(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians from raw decomposed tensors.

    This is the final stage of the decomposed rendering pipeline. It takes
    the raw outputs from :func:`~fvdb.functional.splat.project_to_2d`,
    :func:`~fvdb.functional.splat.compute_opacities`,
    :func:`~fvdb.functional.splat.prepare_render_features`, and
    :func:`~fvdb.functional.splat.intersect_tiles` and produces rendered
    images. Differentiable via Python autograd.

    Args:
        means2d: ``[C, N, 2]`` Projected 2D means.
        conics: ``[C, N, 3]`` Upper-triangle of inverse 2D covariance.
        features: ``[C, N, D]`` Render features (from SH eval or depths).
        opacities: ``[C, N]`` Per-camera opacities.
        tile_offsets: Per-tile start offsets (from :func:`intersect_tiles`).
        tile_gaussian_ids: Sorted Gaussian IDs per tile (from :func:`intersect_tiles`).
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        tile_size: Tile size for tiled rasterization.
        backgrounds: ``[C, D]`` Optional per-camera background colors.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeDenseFn.apply(
            means2d,
            conics,
            features,
            opacities,
            image_width,
            image_height,
            0,
            0,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            False,  # absgrad
            backgrounds,
            masks,
        ),
    )
