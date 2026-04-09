# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for MCMC-based Gaussian operations.

``relocate_gaussians`` and ``add_noise_to_gaussian_means`` dispatch directly
to C++ free functions via pybind.
"""
from __future__ import annotations

import torch

from .. import _fvdb_cpp as _C


def relocate_gaussians(
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    ratios: torch.Tensor,
    binomial_coeffs: torch.Tensor,
    n_max: int,
    min_opacity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relocate dead Gaussians to high-gradient regions (MCMC strategy).

    Args:
        log_scales: ``[N, 3]`` log-scale parameters.
        logit_opacities: ``[N]`` pre-sigmoid opacity logits.
        ratios: ``[N]`` relocation ratios.
        binomial_coeffs: ``[N]`` binomial sampling coefficients.
        n_max: Maximum number of relocation candidates.
        min_opacity: Minimum opacity threshold for liveness.

    Returns:
        Tuple of (logit_opacities_new ``[N]``, log_scales_new ``[N, 3]``).
    """
    return _C.relocate_gaussians(
        log_scales,
        logit_opacities,
        ratios,
        binomial_coeffs,
        n_max,
        min_opacity,
    )


def add_noise_to_gaussian_means(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    quats: torch.Tensor,
    noise_scale: float,
    t: float = 0.005,
    k: float = 100.0,
) -> None:
    """Add scale-dependent noise to Gaussian positions in-place.

    Args:
        means: ``[N, 3]`` Gaussian centres (mutated in-place).
        log_scales: ``[N, 3]`` log-scale parameters.
        logit_opacities: ``[N]`` pre-sigmoid opacity logits.
        quats: ``[N, 4]`` quaternion rotations.
        noise_scale: Noise scale factor.
        t: Noise scaling parameter. Defaults to ``0.005``.
        k: Noise scaling parameter. Defaults to ``100.0``.
    """
    _C.add_noise_to_gaussian_means(
        means,
        log_scales,
        logit_opacities,
        quats,
        noise_scale,
        t,
        k,
    )
