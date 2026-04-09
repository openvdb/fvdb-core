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


def _wrap_jagged_pair(
    a: JaggedTensorCpp, b: JaggedTensorCpp
) -> tuple[JaggedTensor, JaggedTensor]:
    return JaggedTensor(impl=a), JaggedTensor(impl=b)


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
    top_k_contributors: int = 0,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Get the IDs of contributing Gaussians per pixel (dense).

    Non-differentiable analysis function.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        tiles: :class:`GaussianTileIntersection` from Stage 3.
        num_contributing: Optional pre-computed count tensor (from
            :func:`count_contributing_gaussians`).
        top_k_contributors: If > 0, return only the top-k most opaque
            contributors per pixel.

    Returns:
        Tuple of (gaussian_ids, weights) as :class:`~fvdb.JaggedTensor`.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query(tiles)
    settings.num_depth_samples = top_k_contributors

    if top_k_contributors <= 0 and num_contributing is None:
        num_contributing, _ = count_contributing_gaussians(projected, logit_opacities, tiles)

    ids_impl, weights_impl = _C.identify_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        settings,
        num_contributing,
    )
    return _wrap_jagged_pair(ids_impl, weights_impl)


def count_contributing_gaussians_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Count the number of contributing Gaussians per pixel (sparse).

    Non-differentiable analysis function.  Gets ``pixels_to_render`` from
    ``sparse_tiles.pixels_to_render``.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        sparse_tiles: :class:`SparseGaussianTileIntersection` from Stage 3.

    Returns:
        Tuple of (num_contributing, weights) as :class:`~fvdb.JaggedTensor`.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query_sparse(sparse_tiles)

    render_pixels = sparse_tiles.unique_pixels if sparse_tiles.has_duplicates else sparse_tiles.pixels_to_render
    jt0_impl, jt1_impl = _C.count_contributing_gaussians_sparse(
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
        pixels_impl = sparse_tiles.pixels_to_render._impl
        jt0_impl = pixels_impl.jagged_like(jt0_impl.jdata.index_select(0, inv))
        jt1_impl = pixels_impl.jagged_like(jt1_impl.jdata.index_select(0, inv))
    return _wrap_jagged_pair(jt0_impl, jt1_impl)


def identify_contributing_gaussians_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
    num_contributing: JaggedTensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Get the IDs of contributing Gaussians per pixel (sparse).

    Non-differentiable analysis function.  Gets ``pixels_to_render`` from
    ``sparse_tiles.pixels_to_render``.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        sparse_tiles: :class:`SparseGaussianTileIntersection` from Stage 3.
        num_contributing: Optional pre-computed count :class:`~fvdb.JaggedTensor`.

    Returns:
        Tuple of (gaussian_ids, weights) as :class:`~fvdb.JaggedTensor`.
    """
    from ._gaussian_rasterization import _compute_opacities

    opacities = _compute_opacities(logit_opacities, projected)
    settings = _build_settings_for_query_sparse(sparse_tiles)

    render_pixels = sparse_tiles.unique_pixels if sparse_tiles.has_duplicates else sparse_tiles.pixels_to_render

    # Resolve num_contributing into unique-pixel space for the kernel
    jt_arg: JaggedTensorCpp
    if num_contributing is None:
        jt_arg, _ = _C.count_contributing_gaussians_sparse(
            projected.means2d, projected.conics, opacities,
            sparse_tiles.tile_offsets, sparse_tiles.tile_gaussian_ids,
            render_pixels._impl,
            sparse_tiles.active_tiles,
            sparse_tiles.tile_pixel_mask, sparse_tiles.tile_pixel_cumsum,
            sparse_tiles.pixel_map, settings,
        )
    else:
        nc_impl: JaggedTensorCpp = num_contributing._impl if isinstance(num_contributing, JaggedTensor) else num_contributing  # type: ignore[assignment]
        if sparse_tiles.has_duplicates:
            inv = sparse_tiles.inverse_indices
            rep_idx = torch.empty(render_pixels._impl.rsize(0), dtype=torch.long, device=inv.device)
            rep_idx.scatter_(
                0, inv, torch.arange(sparse_tiles.pixels_to_render._impl.rsize(0), dtype=torch.long, device=inv.device)
            )
            unique_data = nc_impl.jdata.index_select(0, rep_idx)
            jt_arg = render_pixels._impl.jagged_like(unique_data)
        else:
            jt_arg = nc_impl

    ids_impl, weights_impl = _C.identify_contributing_gaussians_sparse(
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
        pixels_impl = sparse_tiles.pixels_to_render._impl
        ids_impl = pixels_impl.jagged_like(ids_impl.jdata.index_select(0, inv))
        weights_impl = pixels_impl.jagged_like(weights_impl.jdata.index_select(0, inv))
    return _wrap_jagged_pair(ids_impl, weights_impl)
