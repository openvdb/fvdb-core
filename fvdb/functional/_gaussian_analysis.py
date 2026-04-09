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

    render_pixels = sparse_tiles.unique_pixels if sparse_tiles.has_duplicates else sparse_tiles.pixels_to_render
    result = _C.count_contributing_gaussians_sparse(
        projected.means2d,
        projected.conics,
        opacities,
        sparse_tiles.tile_offsets,
        sparse_tiles.tile_gaussian_ids,
        render_pixels._impl,
        sparse_tiles.active_tiles,
        sparse_tiles.tile_pixel_mask,
        sparse_tiles.tile_pixel_cumsum,
        sparse_tiles.pixel_map,
        settings,
    )

    if sparse_tiles.has_duplicates:
        inv = sparse_tiles.inverse_indices
        jt0, jt1 = result
        pixels_impl = sparse_tiles.pixels_to_render._impl
        return (
            pixels_impl.jagged_like(jt0.jdata().index_select(0, inv)),
            pixels_impl.jagged_like(jt1.jdata().index_select(0, inv)),
        )
    return result


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

    render_pixels = sparse_tiles.unique_pixels if sparse_tiles.has_duplicates else sparse_tiles.pixels_to_render

    # When dedup happened, num_contributing is in original (duplicated) space but the
    # kernel expects unique-pixel space.  Pick one representative per group.
    jt_arg: JaggedTensorCpp | None = None
    if num_contributing is not None:
        nc_impl = num_contributing._impl if isinstance(num_contributing, JaggedTensor) else num_contributing
        if sparse_tiles.has_duplicates:
            inv = sparse_tiles.inverse_indices
            rep_idx = torch.empty(render_pixels._impl.rsize(0), dtype=torch.long, device=inv.device)
            rep_idx.scatter_(
                0, inv, torch.arange(sparse_tiles.pixels_to_render._impl.rsize(0), dtype=torch.long, device=inv.device)
            )
            unique_data = nc_impl.jdata().index_select(0, rep_idx)
            jt_arg = render_pixels._impl.jagged_like(unique_data)
        else:
            jt_arg = nc_impl

    result = _C.identify_contributing_gaussians_sparse(
        projected.means2d,
        projected.conics,
        opacities,
        sparse_tiles.tile_offsets,
        sparse_tiles.tile_gaussian_ids,
        render_pixels._impl,
        sparse_tiles.active_tiles,
        sparse_tiles.tile_pixel_mask,
        sparse_tiles.tile_pixel_cumsum,
        sparse_tiles.pixel_map,
        settings,
        jt_arg,
    )

    if sparse_tiles.has_duplicates:
        inv = sparse_tiles.inverse_indices
        jt0, jt1 = result
        pixels_impl = sparse_tiles.pixels_to_render._impl
        return (
            pixels_impl.jagged_like(jt0.jdata().index_select(0, inv)),
            pixels_impl.jagged_like(jt1.jdata().index_select(0, inv)),
        )
    return result
