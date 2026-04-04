# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for Gaussian projection (analytic and camera-dispatched)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from ... import _fvdb_cpp as _C
from ._tile_intersection import build_render_settings


# ---------------------------------------------------------------------------
#  Autograd functions (raw dispatch wrappers)
# ---------------------------------------------------------------------------


class _ProjectGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the Gaussian projection forward/backward dispatch.

    Uses the **fresh-zeros pattern** for accumulator handling: each backward call
    creates fresh zero tensors, passes them to the kernel (which fills them with
    per-call contributions via atomicAdd), then explicitly accumulates the deltas
    into the persistent accumulator tensors owned by the caller.

    This keeps the dispatch effectively pure per-call and makes cross-iteration
    accumulation visible in Python.
    """

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
        # Non-differentiable accumulator refs (may be None)
        accum_grad_norms: torch.Tensor | None,
        accum_step_counts: torch.Tensor | None,
        accum_max_radii: torch.Tensor | None,
    ):
        result = _C.gsplat_projection_fwd(
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
        # Save accumulator refs (non-differentiable, not part of grad graph)
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
        # Make gradients contiguous (matching C++ autograd behavior)
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
        d_means, _, d_quats, d_scales, d_w2c = _C.gsplat_projection_bwd(
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
        result = _C.gsplat_projection_jagged_fwd(
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
        d_means, _, d_quats, d_scales, d_w2c = _C.gsplat_projection_jagged_bwd(
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
    near: float = 0.01,
    far: float = 1e10,
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
        near: Near clipping plane.
        far: Far clipping plane.
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
        near=near,
        far=far,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps_2d=eps_2d,
        antialias=antialias,
        sh_degree_to_use=sh_degree_to_use,
        render_mode=render_mode,
    )
    return _C.gsplat_project_gaussians_analytic(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        settings,
        camera_model,
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
    near: float = 0.01,
    far: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    render_mode: str = "rgb",
    camera_model: _C.CameraModel = _C.CameraModel.PINHOLE,
    projection_method: _C.ProjectionMethod = _C.ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
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
        near: Near clipping plane.
        far: Far clipping plane.
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
        near=near,
        far=far,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps_2d=eps_2d,
        antialias=antialias,
        sh_degree_to_use=sh_degree_to_use,
        render_mode=render_mode,
    )
    return _C.gsplat_project_gaussians_for_camera(
        means,
        quats,
        log_scales,
        logit_opacities,
        sh0,
        shN,
        world_to_camera_matrices,
        projection_matrices,
        settings,
        camera_model,
        projection_method,
        distortion_coeffs,
    )


# ---------------------------------------------------------------------------
#  Decomposed projection: raw geometric projection as a composable stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawProjection:
    """Result of raw geometric projection of 3D Gaussians onto 2D image planes.

    Contains the low-level projection outputs before opacity computation,
    SH evaluation, or tile intersection. This is the first stage in the
    decomposed rendering pipeline.

    Attributes:
        radii: ``[C, N]`` int32 projected radii (<=0 means culled).
        means2d: ``[C, N, 2]`` Projected 2D means.
        depths: ``[C, N]`` Depths along camera z-axis.
        conics: ``[C, N, 3]`` Upper-triangle of inverse 2D covariance.
        compensations: ``[C, N]`` Antialiasing compensation factors, or ``None``
            when antialiasing is disabled.
    """

    radii: torch.Tensor
    means2d: torch.Tensor
    depths: torch.Tensor
    conics: torch.Tensor
    compensations: torch.Tensor | None


def project_to_2d(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    eps_2d: float,
    near: float,
    far: float,
    radius_clip: float,
    antialias: bool,
    camera_model: _C.CameraModel,
    accum_grad_norms: torch.Tensor | None = None,
    accum_step_counts: torch.Tensor | None = None,
    accum_max_radii: torch.Tensor | None = None,
) -> RawProjection:
    """Raw geometric projection of 3D Gaussians onto 2D image planes.

    This is a composable pipeline stage: it performs only the geometric
    projection (no opacity computation, SH evaluation, or tile intersection).
    Differentiable via Python autograd.

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
        accum_grad_norms: ``[N]`` Persistent accumulator for gradient norms.
            Mutated in-place during the backward pass via CUDA ``atomicAdd``.
        accum_step_counts: ``[N]`` Persistent accumulator for gradient step counts.
            Mutated in-place during the backward pass via CUDA ``atomicAdd``.
        accum_max_radii: ``[N]`` Persistent accumulator for max projected radii.
            Mutated in-place during the backward pass via CUDA ``atomicMax``.

    Returns:
        A :class:`RawProjection` containing radii, means2d, depths, conics,
        and compensations.
    """
    ortho = camera_model == _C.CameraModel.ORTHOGRAPHIC
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
    radii: torch.Tensor = proj_result[0]
    means2d: torch.Tensor = proj_result[1]
    depths: torch.Tensor = proj_result[2]
    conics: torch.Tensor = proj_result[3]
    compensations: torch.Tensor | None = proj_result[4] if antialias else None

    return RawProjection(
        radii=radii,
        means2d=means2d,
        depths=depths,
        conics=conics,
        compensations=compensations,
    )
