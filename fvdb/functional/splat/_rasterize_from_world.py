# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for world-space Gaussian rasterization with geometry gradients."""
from __future__ import annotations

from typing import Optional

import torch

from ... import _fvdb_cpp as _C


def rasterize_from_world(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    projected_state: _C.ProjectedGaussianSplats,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    camera_model: _C.CameraModel,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians from world-space with geometry gradients.

    Unlike :func:`rasterize_from_projected`, this path computes gradients with
    respect to the 3D Gaussian geometry (means, quats, log_scales) during
    backpropagation, enabling world-space optimization.

    Supports backpropagation through the C++ autograd.

    Args:
        means: ``[N, 3]`` Gaussian means.
        quats: ``[N, 4]`` Quaternion rotations.
        log_scales: ``[N, 3]`` Log scale factors.
        projected_state: Pre-projected state from :func:`project_gaussians_for_camera`.
        world_to_camera_matrices: ``[C, 4, 4]`` World-to-camera transforms.
        projection_matrices: ``[C, 3, 3]`` Projection matrices.
        distortion_coeffs: ``[C, K]`` Distortion coefficients (empty ``[C, 0]`` for none).
        camera_model: Camera distortion model.
        image_width: Output image width.
        image_height: Output image height.
        tile_size: Tile size for tiled rasterization.
        backgrounds: ``[C, D]`` Optional per-camera backgrounds.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    return _C.gsplat_rasterize_from_world(
        means, quats, log_scales, projected_state,
        world_to_camera_matrices, projection_matrices, distortion_coeffs,
        camera_model, image_width, image_height, tile_size,
        backgrounds, masks,
    )
