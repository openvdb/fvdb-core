# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for spherical harmonics evaluation."""
from __future__ import annotations

import torch

from ... import _fvdb_cpp as _C


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
    return _C.gsplat_eval_sh(
        means, sh0, shN, sh_degree_to_use, world_to_camera_matrices, per_gaussian_projected_radii
    )
