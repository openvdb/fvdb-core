# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for analysing contributing Gaussians per pixel."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp as _C
from ..jagged_tensor import JaggedTensor
from .._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from .._fvdb_cpp import RenderSettings

if TYPE_CHECKING:
    from ._gaussian_projection import ProjectedGaussians
    from ._gaussian_tile_intersection import GaussianTileIntersection, SparseGaussianTileIntersection


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _build_settings_for_query(tiles: GaussianTileIntersection) -> RenderSettings:
    """Construct a minimal ``RenderSettings`` from tile intersection data."""
    settings = _C.RenderSettings()
    settings.image_width = tiles.image_width
    settings.image_height = tiles.image_height
    settings.tile_size = tiles.tile_size
    return settings


def _build_settings_for_query_sparse(sparse_tiles: SparseGaussianTileIntersection) -> RenderSettings:
    """Construct a minimal ``RenderSettings`` from sparse tile intersection data."""
    settings = _C.RenderSettings()
    settings.image_width = sparse_tiles.image_width
    settings.image_height = sparse_tiles.image_height
    settings.tile_size = sparse_tiles.tile_size
    return settings


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def count_contributing_gaussians(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the number of contributing Gaussians per pixel (dense).

    Non-differentiable analysis function.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        tiles: :class:`GaussianTileIntersection` from Stage 3.

    Returns:
        Tuple of (num_contributing ``[C, H, W]``, weights ``[C, H, W]``).
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query(tiles)
    return _C.count_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        settings,
    )


def identify_contributing_gaussians(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    num_contributing: torch.Tensor | None = None,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Get the IDs of contributing Gaussians per pixel (dense).

    Non-differentiable analysis function.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        tiles: :class:`GaussianTileIntersection` from Stage 3.
        num_contributing: Optional pre-computed count tensor (from
            :func:`count_contributing_gaussians`).

    Returns:
        Tuple of (gaussian_ids, weights) as JaggedTensors.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query(tiles)
    return _C.identify_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        settings,
        num_contributing,
    )


def count_contributing_gaussians_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Count the number of contributing Gaussians per pixel (sparse).

    Non-differentiable analysis function.  Gets ``pixels_to_render`` from
    ``sparse_tiles.pixels_to_render``.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        sparse_tiles: :class:`SparseGaussianTileIntersection` from Stage 3.

    Returns:
        Tuple of (num_contributing, weights) as JaggedTensors.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query_sparse(sparse_tiles)
    return _C.count_contributing_gaussians_sparse(
        projected.means2d,
        projected.conics,
        opacities,
        sparse_tiles.tile_offsets,
        sparse_tiles.tile_gaussian_ids,
        sparse_tiles.active_tiles,
        sparse_tiles.active_tile_mask,
        sparse_tiles.tile_pixel_mask,
        sparse_tiles.tile_pixel_cumsum,
        sparse_tiles.pixel_map,
        sparse_tiles.inverse_indices,
        sparse_tiles.unique_pixels._impl,
        sparse_tiles.has_duplicates,
        sparse_tiles.pixels_to_render._impl,
        settings,
    )


def identify_contributing_gaussians_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
    num_contributing: JaggedTensor | None = None,
) -> tuple[JaggedTensorCpp, JaggedTensorCpp]:
    """Get the IDs of contributing Gaussians per pixel (sparse).

    Non-differentiable analysis function.  Gets ``pixels_to_render`` from
    ``sparse_tiles.pixels_to_render``.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        sparse_tiles: :class:`SparseGaussianTileIntersection` from Stage 3.
        num_contributing: Optional pre-computed count JaggedTensor.

    Returns:
        Tuple of (gaussian_ids, weights) as JaggedTensors.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query_sparse(sparse_tiles)
    jt_arg = num_contributing._impl if num_contributing is not None else None
    return _C.identify_contributing_gaussians_sparse(
        projected.means2d,
        projected.conics,
        opacities,
        sparse_tiles.tile_offsets,
        sparse_tiles.tile_gaussian_ids,
        sparse_tiles.active_tiles,
        sparse_tiles.active_tile_mask,
        sparse_tiles.tile_pixel_mask,
        sparse_tiles.tile_pixel_cumsum,
        sparse_tiles.pixel_map,
        sparse_tiles.inverse_indices,
        sparse_tiles.unique_pixels._impl,
        sparse_tiles.has_duplicates,
        sparse_tiles.pixels_to_render._impl,
        settings,
        jt_arg,
    )
