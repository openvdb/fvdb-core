# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for spherical harmonics evaluation."""
from __future__ import annotations

from typing import Any

import torch

from ... import _fvdb_cpp as _C


# ---------------------------------------------------------------------------
#  Autograd function (raw dispatch wrapper)
# ---------------------------------------------------------------------------


class _EvalSHFn(torch.autograd.Function):
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
        render_quantities = _C.gsplat_sh_eval_fwd(
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
        if d_loss_d_colors is not None:
            d_loss_d_colors = d_loss_d_colors.contiguous()

        view_dirs, shN_coeffs, radii = ctx.saved_tensors

        assert d_loss_d_colors is not None
        compute_d_loss_d_view_dirs = view_dirs.numel() > 0 and view_dirs.requires_grad

        d_sh0, d_shN, d_view_dirs = _C.gsplat_sh_eval_bwd(
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


def evaluate_spherical_harmonics(
    means: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    sh_degree_to_use: int,
    world_to_camera_matrices: torch.Tensor,
    per_gaussian_projected_radii: torch.Tensor,
) -> torch.Tensor:
    """Evaluate spherical harmonics for Gaussian splats.

    Computes view-dependent features (typically RGB colors) from SH coefficients.
    Supports backpropagation through the C++ autograd.

    Args:
        means: ``[N, 3]`` Gaussian means (used to compute view directions when sh_degree > 0).
        sh0: ``[N, 1, D]`` Degree-0 SH coefficients.
        shN: ``[N, K-1, D]`` Higher-degree SH coefficients.
        sh_degree_to_use: SH degree to use (-1 for all available).
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        per_gaussian_projected_radii: ``[C, N]`` Projected radii (int32). Points with
            radii <= 0 output zeros (used to skip invisible Gaussians).

    Returns:
        ``[C, N, D]`` Evaluated SH features.
    """
    import math

    K = shN.size(1) + 1
    C = world_to_camera_matrices.size(0)
    actual_sh_degree = sh_degree_to_use if sh_degree_to_use >= 0 else int(math.sqrt(K)) - 1
    if actual_sh_degree == 0:
        view_dirs = torch.zeros(C, means.size(0), 3, device=means.device, dtype=means.dtype)
    else:
        cam_to_world = torch.linalg.inv(world_to_camera_matrices)
        camera_pos = cam_to_world[:, :3, 3]
        view_dirs = means[None, :, :] - camera_pos[:, None, :]
    result: Any = _EvalSHFn.apply(actual_sh_degree, C, view_dirs, sh0, shN, per_gaussian_projected_radii)
    assert isinstance(result, torch.Tensor)
    return result
