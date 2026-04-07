# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for querying contributing Gaussians per pixel."""
from __future__ import annotations


import torch

from ... import _fvdb_cpp as _C
from ...jagged_tensor import JaggedTensor
from ..._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from ..._fvdb_cpp import RenderSettings
from ._projected_gaussians import ProjectedGaussians, SparseProjectedGaussians


def render_num_contributing_gaussians(
    projected_state: ProjectedGaussians,
    settings: RenderSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the number of contributing Gaussians per pixel (dense).

    Args:
        projected_state: Pre-projected Gaussian state.
        settings: Render settings.

    Returns:
        Tuple of (num_contributing ``[C, H, W]``, weights ``[C, H, W]``).
    """
    p = projected_state
    return _C.gsplat_render_num_contributing(
        p.means2d,
        p.conics,
        p.opacities,
        p.tile_offsets,
        p.tile_gaussian_ids,
        settings,
    )


def render_contributing_gaussian_ids(
    projected_state: ProjectedGaussians,
    settings: RenderSettings,
    num_contributing_gaussians: torch.Tensor | None = None,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Get the IDs of contributing Gaussians per pixel (dense).

    Args:
        projected_state: Pre-projected Gaussian state.
        settings: Render settings.
        num_contributing_gaussians: Optional pre-computed count tensor.

    Returns:
        Tuple of (gaussian_ids, weights) as JaggedTensors.
    """
    p = projected_state
    return _C.gsplat_render_contributing_ids(
        p.means2d,
        p.conics,
        p.opacities,
        p.tile_offsets,
        p.tile_gaussian_ids,
        settings,
        num_contributing_gaussians,
    )


def sparse_render_num_contributing_gaussians(
    sparse_state: SparseProjectedGaussians,
    pixels_to_render: JaggedTensor,
    settings: RenderSettings,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Count the number of contributing Gaussians per pixel (sparse).

    Args:
        sparse_state: Pre-projected sparse Gaussian state.
        pixels_to_render: JaggedTensor of pixel coordinates.
        settings: Render settings.

    Returns:
        Tuple of (num_contributing, weights) as JaggedTensors.
    """
    s = sparse_state
    return _C.gsplat_sparse_render_num_contributing(
        s.means2d,
        s.conics,
        s.opacities,
        s.tile_offsets,
        s.tile_gaussian_ids,
        s.active_tiles,
        s.active_tile_mask,
        s.tile_pixel_mask,
        s.tile_pixel_cumsum,
        s.pixel_map,
        s.inverse_indices,
        s.unique_pixels_to_render._impl,
        s.has_duplicates,
        pixels_to_render._impl,
        settings,
    )


def sparse_render_contributing_gaussian_ids(
    sparse_state: SparseProjectedGaussians,
    pixels_to_render: JaggedTensor,
    settings: RenderSettings,
    num_contributing_gaussians: JaggedTensor | None = None,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Get the IDs of contributing Gaussians per pixel (sparse).

    Args:
        sparse_state: Pre-projected sparse Gaussian state.
        pixels_to_render: JaggedTensor of pixel coordinates.
        settings: Render settings.
        num_contributing_gaussians: Optional pre-computed count JaggedTensor.

    Returns:
        Tuple of (gaussian_ids, weights) as JaggedTensors.
    """
    s = sparse_state
    jt_arg = num_contributing_gaussians._impl if num_contributing_gaussians is not None else None
    return _C.gsplat_sparse_render_contributing_ids(
        s.means2d,
        s.conics,
        s.opacities,
        s.tile_offsets,
        s.tile_gaussian_ids,
        s.active_tiles,
        s.active_tile_mask,
        s.tile_pixel_mask,
        s.tile_pixel_cumsum,
        s.pixel_map,
        s.inverse_indices,
        s.unique_pixels_to_render._impl,
        s.has_duplicates,
        pixels_to_render._impl,
        settings,
        jt_arg,
    )
