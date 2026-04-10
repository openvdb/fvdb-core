# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for spherical harmonics evaluation (Stage 2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from .. import _fvdb_cpp as _C
from ..enums import GaussianRenderMode

if TYPE_CHECKING:
    from ._gaussian_projection import ProjectedGaussians


# ---------------------------------------------------------------------------
#  Autograd function (raw dispatch wrapper)
# ---------------------------------------------------------------------------


class _EvaluateGaussianSHFn(torch.autograd.Function):
    """Python autograd wrapper for the SH evaluation forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        sh_degree_to_use: int,
        num_cameras: int,
        view_dirs: torch.Tensor,  # [C, N, 3] or empty
        sh0_coeffs: torch.Tensor,  # [N, 1, D]
        shN_coeffs: torch.Tensor,  # [N, K-1, D] or empty
        radii: torch.Tensor,  # [C, N]
    ) -> torch.Tensor:
        render_quantities = _C.eval_gaussian_sh_fwd(
            sh_degree_to_use,
            num_cameras,
            view_dirs,
            sh0_coeffs,
            shN_coeffs,
            radii,
        )

        ctx.save_for_backward(view_dirs, shN_coeffs, radii)
        ctx.sh_degree_to_use = sh_degree_to_use
        ctx.num_cameras = num_cameras
        ctx.num_gaussians = sh0_coeffs.size(0)

        return render_quantities

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_colors = grad_outputs[0]
        if d_loss_d_colors is None:
            return (None, None, None, None, None, None)
        d_loss_d_colors = d_loss_d_colors.contiguous()

        view_dirs, shN_coeffs, radii = ctx.saved_tensors

        compute_d_loss_d_view_dirs = view_dirs.numel() > 0 and view_dirs.requires_grad

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

        # Order: sh_degree_to_use, num_cameras, view_dirs, sh0_coeffs, shN_coeffs, radii
        return (None, None, d_view_dirs, d_sh0, d_shN, None)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def evaluate_gaussian_sh(
    means: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projected: ProjectedGaussians,
    sh_degree_to_use: int = -1,
    render_mode: GaussianRenderMode = GaussianRenderMode.FEATURES,
) -> torch.Tensor:
    """Evaluate per-Gaussian render features from SH coefficients (Stage 2).

    Computes view-dependent features based on ``render_mode``:

    - ``FEATURES``: evaluates spherical harmonics to produce view-dependent
      colours (or any per-Gaussian feature encoded as SH coefficients).
    - ``DEPTH``: returns depths as a single-channel feature (no SH evaluation).
    - ``FEATURES_AND_DEPTH``: concatenates SH-evaluated features with depths.

    Differentiable via Python autograd.

    Args:
        means: ``[N, 3]`` Gaussian means (used to compute view directions when
            the SH degree is > 0).
        sh0: ``[N, 1, D]`` Degree-0 SH coefficients.
        shN: ``[N, K-1, D]`` Higher-degree SH coefficients.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projected: :class:`ProjectedGaussians` from Stage 1 (provides
            ``radii`` and ``depths``).
        sh_degree_to_use: SH degree to use (-1 for all available).
        render_mode: Which quantities to produce.

    Returns:
        ``[C, N, D]`` Render features (``D`` depends on render mode).
    """
    radii = projected.radii
    depths = projected.depths

    if render_mode == GaussianRenderMode.DEPTH:
        return depths.unsqueeze(-1)

    K = shN.size(1) + 1
    C = world_to_camera_matrices.size(0)
    actual_sh_degree = sh_degree_to_use if sh_degree_to_use >= 0 else int(math.sqrt(K)) - 1

    if actual_sh_degree == 0:
        view_dirs = means.new_empty(0)
    else:
        cam_to_world = torch.linalg.inv(world_to_camera_matrices)
        camera_pos = cam_to_world[:, :3, 3]
        view_dirs = means[None, :, :] - camera_pos[:, None, :]

    features = cast(
        torch.Tensor,
        _EvaluateGaussianSHFn.apply(actual_sh_degree, C, view_dirs, sh0, shN, radii),
    )

    if render_mode == GaussianRenderMode.FEATURES_AND_DEPTH:
        features = torch.cat([features, depths.unsqueeze(-1)], -1)

    return features
