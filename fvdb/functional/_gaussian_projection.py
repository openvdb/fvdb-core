# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for Gaussian projection (analytic and camera-dispatched)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from .. import _fvdb_cpp as _C
from ..enums import CameraModel, ProjectionMethod


# ---------------------------------------------------------------------------
#  Autograd functions (raw dispatch wrappers)
# ---------------------------------------------------------------------------


class _ProjectGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the Gaussian projection forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        log_scales: torch.Tensor,  # [N, 3]
        world_to_cam: torch.Tensor,  # [C, 4, 4]
        projection_matrices: torch.Tensor,  # [C, 3, 3]
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

        to_save = [
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            radii,
            conics,
        ]
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
        grad_radii = grad_outputs[0]
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        maybe_grad_comp = grad_outputs[4:]
        # Make gradients contiguous (required by the CUDA backward kernel)
        if grad_radii is not None:
            grad_radii = grad_radii.contiguous()
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

        # --- Direct in-place accumulation ---
        # The kernel uses gpuAtomicAdd/atomicMax which are inherently accumulative,
        # so we pass the persistent accumulators directly — no temporary tensors needed.
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
            ctx.needs_input_grad[3],  # world_to_cam requires_grad
            ctx.ortho,
            ctx.accum_grad_norms,
            ctx.accum_max_radii,
            ctx.accum_step_counts,
        )

        # Return None for all non-differentiable inputs
        # Order: means, quats, log_scales, world_to_cam, projection_matrices,
        #   image_width, image_height, eps2d, near_plane, far_plane, min_radius_2d,
        #   calc_compensations, ortho, accum_grad_norms, accum_step_counts, accum_max_radii
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
        radii, means2d, depths, conics = result[0], result[1], result[2], result[3]

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

        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_radii = grad_outputs[0]
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        if grad_radii is not None:
            grad_radii = grad_radii.contiguous()
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
            ctx.needs_input_grad[5],  # world_to_cam requires_grad
            ctx.ortho,
        )

        # g_sizes, means, quats, scales, c_sizes, world_to_cam, projection_matrices,
        # image_width, image_height, eps2d, near_plane, far_plane, min_radius_2d, ortho
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


def _resolve_projection_method(
    camera_model: CameraModel,
    projection_method: ProjectionMethod,
) -> ProjectionMethod:
    """Resolve AUTO -> ANALYTIC or UNSCENTED based on camera model."""
    if projection_method != ProjectionMethod.AUTO:
        return projection_method
    if camera_model in (CameraModel.PINHOLE, CameraModel.ORTHOGRAPHIC):
        return ProjectionMethod.ANALYTIC
    return ProjectionMethod.UNSCENTED


@dataclass(frozen=True)
class ProjectedGaussians:
    """Result of geometric projection of 3D Gaussians onto 2D image planes.

    This is the output of Stage 1 (``project_gaussians``) and serves as input
    to all subsequent pipeline stages.  Contains only the raw geometric
    projection outputs -- no opacity computation, SH evaluation, or tile
    intersection data.

    Attributes:
        radii: ``[C, N]`` int32 projected radii (<=0 means culled).
        means2d: ``[C, N, 2]`` Projected 2D means.
        depths: ``[C, N]`` Depths along camera z-axis.
        conics: ``[C, N, 3]`` Upper-triangle of inverse 2D covariance.
        compensations: ``[C, N]`` Antialiasing compensation factors, or ``None``
            when antialiasing is disabled.
        image_width: Image width in pixels (carried forward to tile intersection).
        image_height: Image height in pixels (carried forward to tile intersection).
    """

    radii: torch.Tensor
    means2d: torch.Tensor
    depths: torch.Tensor
    conics: torch.Tensor
    compensations: torch.Tensor | None
    image_width: int
    image_height: int


def project_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    eps_2d: float = 0.3,
    near: float = 0.01,
    far: float = 1e10,
    radius_clip: float = 0.0,
    antialias: bool = False,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    accum_grad_norms: torch.Tensor | None = None,
    accum_step_counts: torch.Tensor | None = None,
    accum_max_radii: torch.Tensor | None = None,
) -> ProjectedGaussians:
    """Geometric projection of 3D Gaussians onto 2D image planes (Stage 1).

    Dispatches between analytic projection (differentiable) and unscented
    transform projection (forward-only) based on ``projection_method``.

    Accumulator tensors (``accum_grad_norms``, ``accum_step_counts``,
    ``accum_max_radii``) are only used with analytic projection; they are
    ignored for UT projection.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection/intrinsic matrices.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        eps_2d: Epsilon for 2D projection numerical stability.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        radius_clip: Minimum projected radius for culling.
        antialias: Whether to compute antialiasing compensations.
        camera_model: Camera distortion model.
        projection_method: ``AUTO`` resolves to ``ANALYTIC`` for pinhole/ortho,
            ``UNSCENTED`` for distortion-based camera models.
        distortion_coeffs: ``[C, 12]`` Optional OpenCV distortion coefficients
            (required for UT projection with distortion-based cameras).
        accum_grad_norms: ``[N]`` Persistent accumulator for gradient norms
            (analytic only; mutated in-place during backward).
        accum_step_counts: ``[N]`` Persistent accumulator for gradient step
            counts (analytic only; mutated in-place during backward).
        accum_max_radii: ``[N]`` Persistent accumulator for max projected radii
            (analytic only; mutated in-place during backward).

    Returns:
        A :class:`ProjectedGaussians` containing radii, means2d, depths, conics,
        compensations, image_width, and image_height.
    """
    resolved = _resolve_projection_method(camera_model, projection_method)

    if resolved == ProjectionMethod.ANALYTIC:
        ortho = camera_model == CameraModel.ORTHOGRAPHIC
        proj_result = cast(
            tuple[torch.Tensor, ...],
            _ProjectGaussiansFn.apply(
                means,
                quats,
                log_scales,
                world_to_camera_matrices,
                projection_matrices,
                image_width,
                image_height,
                eps_2d,
                near,
                far,
                radius_clip,
                antialias,
                ortho,
                accum_grad_norms,
                accum_step_counts,
                accum_max_radii,
            ),
        )
        radii = proj_result[0]
        means2d = proj_result[1]
        depths = proj_result[2]
        conics = proj_result[3]
        compensations: torch.Tensor | None = proj_result[4] if antialias else None
    else:
        C = world_to_camera_matrices.size(0)
        dc = (
            distortion_coeffs
            if distortion_coeffs is not None
            else torch.empty(C, 0, device=means.device, dtype=means.dtype)
        )
        radii, means2d, depths, conics, compensations_raw = _C.project_gaussians_ut_fwd(
            means,
            quats,
            log_scales,
            world_to_camera_matrices,
            projection_matrices,
            dc,
            _C.CameraModel(camera_model),
            image_width,
            image_height,
            eps_2d,
            near,
            far,
            radius_clip,
            antialias,
        )
        compensations = compensations_raw if antialias else None

    return ProjectedGaussians(
        radii=radii,
        means2d=means2d,
        depths=depths,
        conics=conics,
        compensations=compensations,
        image_width=image_width,
        image_height=image_height,
    )
