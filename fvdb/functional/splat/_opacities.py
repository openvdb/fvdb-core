# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for computing Gaussian opacities."""
from __future__ import annotations

import torch


def compute_opacities(
    logit_opacities: torch.Tensor,
    num_cameras: int,
    compensations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-camera opacities from logit opacities.

    Applies sigmoid activation, repeats across cameras, and optionally
    multiplies by antialiasing compensation factors.

    Args:
        logit_opacities: ``[N]`` Logit (pre-sigmoid) opacities.
        num_cameras: Number of cameras ``C``.
        compensations: ``[C, N]`` Antialiasing compensation factors, or ``None``
            if antialiasing is disabled.

    Returns:
        ``[C, N]`` Per-camera opacities.
    """
    opacities = torch.sigmoid(logit_opacities).repeat(num_cameras, 1)
    if compensations is not None:
        opacities = opacities * compensations
    return opacities
