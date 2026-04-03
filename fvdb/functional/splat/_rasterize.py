# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense Gaussian rasterization."""
from __future__ import annotations

from typing import Optional

import torch

from ... import _fvdb_cpp as _C


def rasterize_from_projected(
    projected_gaussians: _C.ProjectedGaussianSplats,
    tile_size: int = 16,
    crop_width: int = -1,
    crop_height: int = -1,
    crop_origin_w: int = -1,
    crop_origin_h: int = -1,
    backgrounds: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize pre-projected Gaussians to produce images and alpha maps.

    Supports backpropagation through the C++ autograd.

    Args:
        projected_gaussians: Pre-projected state from :func:`project_gaussians`.
        tile_size: Tile size for tiled rasterization.
        crop_width: Crop width (-1 for full image).
        crop_height: Crop height (-1 for full image).
        crop_origin_w: Crop origin W (-1 for no crop).
        crop_origin_h: Crop origin H (-1 for no crop).
        backgrounds: ``[C, D]`` Optional per-camera background colors.
        masks: ``[C, tileH, tileW]`` Optional per-tile masks.

    Returns:
        Tuple of (rendered_images ``[C, H, W, D]``, alphas ``[C, H, W, 1]``).
    """
    return _C.gsplat_render_crop_from_projected(
        projected_gaussians, tile_size, crop_width, crop_height,
        crop_origin_w, crop_origin_h, backgrounds, masks,
    )
