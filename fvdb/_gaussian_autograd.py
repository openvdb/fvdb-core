# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Python torch.autograd.Function wrappers for Gaussian splatting dispatch functions.

This module replaces the C++ autograd layer (detail/autograd/) with equivalent
Python autograd wrappers that call the raw dispatch forward/backward functions
exposed through pybind11. This is the first step in the migration from C++ to
Python autograd.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from . import _fvdb_cpp as _C
from ._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from .jagged_tensor import JaggedTensor

# ---------------------------------------------------------------------------
#  Projection (analytic)
# ---------------------------------------------------------------------------


class _ProjectGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the analytic Gaussian projection forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        calc_compensations: bool,
        ortho: bool,
        accum_grad_norms: torch.Tensor | None = None,
        accum_step_counts: torch.Tensor | None = None,
        accum_max_radii: torch.Tensor | None = None,
    ):
        result = _C.project_gaussians_analytic_fwd(
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            calc_compensations,
            ortho,
        )
        radii: torch.Tensor = result[0]
        means2d: torch.Tensor = result[1]
        depths: torch.Tensor = result[2]
        conics: torch.Tensor = result[3]
        compensations: torch.Tensor | None = result[4] if calc_compensations else None

        to_save = [means, quats, log_scales, world_to_cam, projection_matrices, radii, conics]
        if compensations is not None:
            to_save.append(compensations)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.calc_compensations = calc_compensations
        ctx.ortho = ortho
        ctx.accum_grad_norms = accum_grad_norms
        ctx.accum_step_counts = accum_step_counts
        ctx.accum_max_radii = accum_max_radii

        if compensations is not None:
            return radii, means2d, depths, conics, compensations
        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        maybe_grad_comp = grad_outputs[4:]
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        grad_compensations: torch.Tensor | None = None
        if ctx.calc_compensations and maybe_grad_comp:
            gc = maybe_grad_comp[0]
            if gc is not None:
                grad_compensations = gc.contiguous()

        saved = ctx.saved_tensors
        means = saved[0]
        quats = saved[1]
        log_scales = saved[2]
        world_to_cam = saved[3]
        projection_matrices = saved[4]
        radii = saved[5]
        conics = saved[6]
        compensations = saved[7] if ctx.calc_compensations else None

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        d_means, _, d_quats, d_scales, d_w2c = _C.project_gaussians_analytic_bwd(
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            compensations,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            radii,
            conics,
            grad_means2d,
            grad_depths,
            grad_conics,
            grad_compensations,
            ctx.needs_input_grad[3],
            ctx.ortho,
            ctx.accum_grad_norms,
            ctx.accum_max_radii,
            ctx.accum_step_counts,
        )

        return (
            d_means,
            d_quats,
            d_scales,
            d_w2c,
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
            None,
        )


# ---------------------------------------------------------------------------
#  Projection (analytic, jagged)
# ---------------------------------------------------------------------------


class _ProjectGaussiansJaggedFn(torch.autograd.Function):
    """Python autograd wrapper for the jagged Gaussian projection dispatch."""

    @staticmethod
    def forward(
        ctx,
        g_sizes: torch.Tensor,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        c_sizes: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        ortho: bool,
    ):
        result = _C.project_gaussians_analytic_jagged_fwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            ortho,
        )
        radii, means2d, depths, conics, compensations = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )

        ctx.save_for_backward(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            radii,
            conics,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.ortho = ortho

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        # grad_outputs[4] is grad_compensations -- the jagged backward dispatch
        # does not consume it, so we ignore it here.
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        g_sizes, means, quats, scales, c_sizes = ctx.saved_tensors[:5]
        world_to_cam, projection_matrices, radii, conics = ctx.saved_tensors[5:]

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        d_means, _, d_quats, d_scales, d_w2c = _C.project_gaussians_analytic_jagged_bwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            radii,
            conics,
            grad_means2d,
            grad_depths,
            grad_conics,
            ctx.needs_input_grad[5],
            ctx.ortho,
        )

        return (
            None,
            d_means,
            d_quats,
            d_scales,
            None,
            d_w2c,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
#  Spherical harmonics evaluation
# ---------------------------------------------------------------------------


class _EvaluateGaussianSHFn(torch.autograd.Function):
    """Python autograd wrapper for the SH evaluation forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        sh_degree_to_use: int,
        num_cameras: int,
        view_dirs: torch.Tensor | None,
        sh0_coeffs: torch.Tensor,
        shN_coeffs: torch.Tensor | None,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        render_quantities = _C.eval_gaussian_sh_fwd(
            sh_degree_to_use,
            num_cameras,
            view_dirs,
            sh0_coeffs,
            shN_coeffs,
            radii,
        )

        to_save = [radii]
        if view_dirs is not None:
            to_save.append(view_dirs)
        if shN_coeffs is not None:
            to_save.append(shN_coeffs)
        ctx.save_for_backward(*to_save)
        ctx.sh_degree_to_use = sh_degree_to_use
        ctx.num_cameras = num_cameras
        ctx.num_gaussians = sh0_coeffs.size(0)
        ctx.has_view_dirs = view_dirs is not None
        ctx.has_shN = shN_coeffs is not None

        return render_quantities

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_colors = grad_outputs[0]
        if d_loss_d_colors is None:
            return (None, None, None, None, None, None)
        d_loss_d_colors = d_loss_d_colors.contiguous()

        saved = list(ctx.saved_tensors)
        radii = saved[0]
        view_dirs = saved[1] if ctx.has_view_dirs else None
        shN_coeffs = saved[-1] if ctx.has_shN else None

        compute_d_loss_d_view_dirs = view_dirs is not None and view_dirs.requires_grad

        d_sh0, d_shN, d_view_dirs = _C.eval_gaussian_sh_bwd(
            ctx.sh_degree_to_use,
            ctx.num_cameras,
            ctx.num_gaussians,
            view_dirs,
            shN_coeffs,
            d_loss_d_colors,
            radii,
            compute_d_loss_d_view_dirs,
        )

        return (None, None, d_view_dirs, d_sh0, d_shN, None)


# ---------------------------------------------------------------------------
#  Dense screen-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeScreenSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the dense Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        absgrad: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
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

        to_save = [means2d, conics, colors, opacities, tile_offsets, tile_gaussian_ids, rendered_alphas, last_ids]
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
        means2d, conics, colors, opacities = saved[0], saved[1], saved[2], saved[3]
        tile_offsets, tile_gaussian_ids = saved[4], saved[5]
        rendered_alphas, last_ids = saved[6], saved[7]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 8
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

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


# ---------------------------------------------------------------------------
#  Sparse screen-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeScreenSpaceGaussiansSparseFn(torch.autograd.Function):
    """Python autograd wrapper for the sparse Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        pixels_to_render: JaggedTensor,
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
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        result = _C.rasterize_screen_space_gaussians_sparse_fwd(
            pixels_to_render._impl,
            means2d,
            conics,
            features,
            opacities,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            backgrounds,
            masks,
        )
        rendered_colors_jt = JaggedTensor(impl=result[0])
        rendered_alphas_jt = JaggedTensor(impl=result[1])
        last_ids_jt = JaggedTensor(impl=result[2])

        joffsets = pixels_to_render.joffsets
        jidx = pixels_to_render.jidx
        jlidx = pixels_to_render.jlidx

        to_save = [
            means2d,
            conics,
            features,
            opacities,
            tile_offsets,
            tile_gaussian_ids,
            pixels_to_render.jdata,
            rendered_colors_jt.jdata,
            rendered_alphas_jt.jdata,
            last_ids_jt.jdata,
            joffsets,
            jidx,
            jlidx,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
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

        return rendered_colors_jt.jdata, rendered_alphas_jt.jdata

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_features_jdata = grad_outputs[0]
        d_loss_d_rendered_alphas_jdata = grad_outputs[1]
        if d_loss_d_rendered_features_jdata is not None:
            d_loss_d_rendered_features_jdata = d_loss_d_rendered_features_jdata.contiguous()
        if d_loss_d_rendered_alphas_jdata is not None:
            d_loss_d_rendered_alphas_jdata = d_loss_d_rendered_alphas_jdata.contiguous()

        saved = ctx.saved_tensors
        means2d, conics, features, opacities = saved[0], saved[1], saved[2], saved[3]
        tile_offsets, tile_gaussian_ids = saved[4], saved[5]
        pixels_jdata = saved[6]
        rendered_alphas_jdata = saved[8]
        last_ids_jdata = saved[9]
        joffsets, jidx, jlidx = saved[10], saved[11], saved[12]
        active_tiles = saved[13]
        tile_pixel_mask, tile_pixel_cumsum, pixel_map = saved[14], saved[15], saved[16]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 17
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

        pixels_jt = JaggedTensor(impl=_C.JaggedTensor.from_data_offsets_and_list_ids(pixels_jdata, joffsets, jlidx))
        rendered_alphas_jt = pixels_jt.jagged_like(rendered_alphas_jdata)
        last_ids_jt = pixels_jt.jagged_like(last_ids_jdata)
        assert d_loss_d_rendered_features_jdata is not None
        assert d_loss_d_rendered_alphas_jdata is not None
        d_loss_d_rendered_features_jt = pixels_jt.jagged_like(d_loss_d_rendered_features_jdata)
        d_loss_d_rendered_alphas_jt = pixels_jt.jagged_like(d_loss_d_rendered_alphas_jdata)

        result = _C.rasterize_screen_space_gaussians_sparse_bwd(
            pixels_jt._impl,
            means2d,
            conics,
            features,
            opacities,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas_jt._impl,
            last_ids_jt._impl,
            d_loss_d_rendered_features_jt._impl,
            d_loss_d_rendered_alphas_jt._impl,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            ctx.absgrad,
            -1,
            backgrounds,
            masks,
        )
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
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
#  World-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeWorldSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for world-space Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        world_to_cam_start: torch.Tensor,
        world_to_cam_end: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        rolling_shutter_type: int,
        camera_model: int,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
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

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
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
        means, quats, log_scales = saved[0], saved[1], saved[2]
        features, opacities = saved[3], saved[4]
        world_to_cam_start, world_to_cam_end = saved[5], saved[6]
        projection_matrices, distortion_coeffs = saved[7], saved[8]
        tile_offsets, tile_gaussian_ids = saved[9], saved[10]
        rendered_alphas, last_ids = saved[11], saved[12]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 13
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

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
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
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
            None,
            None,
            None,
            None,
        )
