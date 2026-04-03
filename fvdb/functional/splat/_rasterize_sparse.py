# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse Gaussian rasterization."""
from __future__ import annotations

from typing import Any, Optional

import torch

from ... import _fvdb_cpp as _C
from ...jagged_tensor import JaggedTensor
from ._tile_intersection import build_render_settings


# ---------------------------------------------------------------------------
#  Autograd function (raw dispatch wrapper)
# ---------------------------------------------------------------------------


class _RasterizeSparseFn(torch.autograd.Function):
    """Python autograd wrapper for the sparse Gaussian rasterization forward/backward dispatch.

    The complexity here is that the rasterize kernels operate on JaggedTensors,
    but torch.autograd.Function only tracks plain tensors for gradient computation.
    We decompose JaggedTensors into their component tensors (jdata, joffsets,
    jidx, jlidx) for saving, then reconstruct in backward.
    """

    @staticmethod
    def forward(
        ctx,
        # Differentiable inputs (plain tensors)
        means2d: torch.Tensor,           # [C, N, 2]
        conics: torch.Tensor,            # [C, N, 3]
        features: torch.Tensor,          # [C, N, D]
        opacities: torch.Tensor,         # [N]
        # Non-differentiable inputs
        pixels_to_render: JaggedTensor,   # JaggedTensor [C, num_pixels, 2]
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        active_tiles: torch.Tensor,
        tile_pixel_mask: torch.Tensor,
        tile_pixel_cumsum: torch.Tensor,
        pixel_map: torch.Tensor,
        absgrad: bool,
        backgrounds: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ):
        # The pybind fwd takes a C++ JaggedTensor directly
        result = _C.gsplat_rasterize_sparse_fwd(
            pixels_to_render._impl,
            means2d, conics, features, opacities,
            image_width, image_height, image_origin_w, image_origin_h,
            tile_size, tile_offsets, tile_gaussian_ids,
            active_tiles, tile_pixel_mask, tile_pixel_cumsum, pixel_map,
            backgrounds, masks,
        )
        # result is a tuple of 3 C++ JaggedTensors: (renderedColors, renderedAlphas, lastIds)
        rendered_colors_jt = JaggedTensor(impl=result[0])
        rendered_alphas_jt = JaggedTensor(impl=result[1])
        last_ids_jt = JaggedTensor(impl=result[2])

        # Extract JaggedTensor metadata for saving
        joffsets = pixels_to_render.joffsets
        jidx = pixels_to_render.jidx
        jlidx = pixels_to_render.jlidx

        to_save = [
            means2d,                          # 0
            conics,                           # 1
            features,                         # 2
            opacities,                        # 3
            tile_offsets,                     # 4
            tile_gaussian_ids,                # 5
            pixels_to_render.jdata,           # 6
            rendered_colors_jt.jdata,         # 7
            rendered_alphas_jt.jdata,         # 8
            last_ids_jt.jdata,                # 9
            joffsets,                         # 10
            jidx,                             # 11
            jlidx,                            # 12
            active_tiles,                     # 13
            tile_pixel_mask,                  # 14
            tile_pixel_cumsum,                # 15
            pixel_map,                        # 16
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
        ctx.num_outer_lists = len(pixels_to_render)

        # Return the jdata tensors (plain tensors) for autograd tracking
        return rendered_colors_jt.jdata, rendered_alphas_jt.jdata

    @staticmethod
    def backward(ctx: Any, d_loss_d_rendered_features_jdata, d_loss_d_rendered_alphas_jdata):
        if d_loss_d_rendered_features_jdata is not None:
            d_loss_d_rendered_features_jdata = d_loss_d_rendered_features_jdata.contiguous()
        if d_loss_d_rendered_alphas_jdata is not None:
            d_loss_d_rendered_alphas_jdata = d_loss_d_rendered_alphas_jdata.contiguous()

        saved = ctx.saved_tensors
        means2d = saved[0]
        conics = saved[1]
        features = saved[2]
        opacities = saved[3]
        tile_offsets = saved[4]
        tile_gaussian_ids = saved[5]
        pixels_jdata = saved[6]
        rendered_colors_jdata = saved[7]
        rendered_alphas_jdata = saved[8]
        last_ids_jdata = saved[9]
        joffsets = saved[10]
        jidx = saved[11]
        jlidx = saved[12]
        active_tiles = saved[13]
        tile_pixel_mask = saved[14]
        tile_pixel_cumsum = saved[15]
        pixel_map = saved[16]

        backgrounds: Optional[torch.Tensor] = None
        masks: Optional[torch.Tensor] = None
        opt_idx = 17
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]
            opt_idx += 1

        # Reconstruct JaggedTensors from saved components
        pixels_jt = JaggedTensor(
            impl=_C.JaggedTensor.from_data_offsets_and_list_ids(pixels_jdata, joffsets, jlidx)
        )
        rendered_alphas_jt = pixels_jt.jagged_like(rendered_alphas_jdata)
        last_ids_jt = pixels_jt.jagged_like(last_ids_jdata)
        d_loss_d_rendered_features_jt = pixels_jt.jagged_like(d_loss_d_rendered_features_jdata)
        d_loss_d_rendered_alphas_jt = pixels_jt.jagged_like(d_loss_d_rendered_alphas_jdata)

        result = _C.gsplat_rasterize_sparse_bwd(
            pixels_jt._impl,
            means2d, conics, features, opacities,
            ctx.image_width, ctx.image_height,
            ctx.image_origin_w, ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets, tile_gaussian_ids,
            rendered_alphas_jt._impl, last_ids_jt._impl,
            d_loss_d_rendered_features_jt._impl,
            d_loss_d_rendered_alphas_jt._impl,
            active_tiles, tile_pixel_mask, tile_pixel_cumsum, pixel_map,
            ctx.absgrad, -1,
            backgrounds, masks,
        )
        # result: (dMean2dAbs, dMeans2d, dConics, dColors, dOpacities)
        d_means2d = result[1]
        d_conics = result[2]
        d_colors = result[3]
        d_opacities = result[4]

        # Order: means2d, conics, features, opacities,
        #   pixels_to_render, image_width, image_height, image_origin_w, image_origin_h,
        #   tile_size, tile_offsets, tile_gaussian_ids, active_tiles,
        #   tile_pixel_mask, tile_pixel_cumsum, pixel_map, absgrad,
        #   backgrounds, masks
        return (
            d_means2d, d_conics, d_colors, d_opacities,
            None, None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None,
        )


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
