# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for analysing contributing Gaussians per pixel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp as _C
from .._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ._gaussian_projection import ProjectedGaussians
    from ._gaussian_tile_intersection import (
        GaussianTileIntersection,
        SparseGaussianTileIntersection,
    )


def _crop_jagged_pixel_pair(
    ids: JaggedTensor,
    weights: JaggedTensor,
    image_width: int,
    image_height: int,
    ox: int,
    oy: int,
    cw: int,
    ch: int,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Crop a pair of pixel-indexed ``JaggedTensor`` s (ldim=2) to a rectangular sub-region.

    The inputs must have structure: ``C`` cameras, each with ``H * W`` pixel
    rows in row-major order, each pixel with a variable number of samples.
    """
    C = len(ids)
    device = ids.jdata.device
    pixels_per_camera = cw * ch

    # Flat indices of crop pixels within one camera: [ch * cw]
    ys = torch.arange(oy, oy + ch, device=device)
    xs = torch.arange(ox, ox + cw, device=device)
    cam_pixel_offsets = (ys[:, None] * image_width + xs[None, :]).reshape(-1)

    # Extend to all cameras: [C * ch * cw]
    cam_starts = torch.arange(C, device=device) * (image_height * image_width)
    all_pixel_indices = (cam_starts[:, None] + cam_pixel_offsets[None, :]).reshape(-1)

    # Both JaggedTensors share the same jagged structure so we reuse offsets.
    offsets = ids.joffsets  # [C*H*W + 1]
    starts = offsets[all_pixel_indices]
    ends = offsets[all_pixel_indices + 1]
    lengths = ends - starts

    new_offsets = torch.zeros(len(all_pixel_indices) + 1, dtype=torch.int64, device=device)
    torch.cumsum(lengths, dim=0, out=new_offsets[1:])
    total = int(new_offsets[-1].item())

    if total > 0:
        pixel_for_sample = torch.repeat_interleave(
            torch.arange(len(all_pixel_indices), device=device, dtype=torch.long),
            lengths,
        )
        within_pixel = torch.arange(total, device=device, dtype=torch.long) - new_offsets[pixel_for_sample]
        gather_idx = (starts[pixel_for_sample] + within_pixel).long()
        new_ids_data = ids.jdata[gather_idx]
        new_weights_data = weights.jdata[gather_idx]
    else:
        new_ids_data = ids.jdata[:0]
        new_weights_data = weights.jdata[:0]

    cam_ids = torch.arange(C, device=device, dtype=torch.int32).repeat_interleave(pixels_per_camera)
    inner_ids = torch.arange(pixels_per_camera, device=device, dtype=torch.int32).repeat(C)
    list_ids = torch.stack([cam_ids, inner_ids], dim=1)

    return (
        JaggedTensor.from_data_offsets_and_list_ids(new_ids_data, new_offsets, list_ids),
        JaggedTensor.from_data_offsets_and_list_ids(new_weights_data, new_offsets.clone(), list_ids.clone()),
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def count_contributing_gaussians(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the number of contributing Gaussians per pixel (dense).

    Non-differentiable analysis function.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        logit_opacities: ``[N]`` Pre-sigmoid opacities.
        tiles: :class:`GaussianTileIntersection` from Stage 3.
        crop: Optional ``(origin_x, origin_y, width, height)`` tuple defining a
            sub-region to analyse.  When ``None`` (the default), the full image
            is used.  The crop region is clamped to the projected image bounds.

    Returns:
        Tuple of (num_contributing ``[C, H, W]``, weights ``[C, H, W]``) where
        H and W are the crop dimensions (or the full image when ``crop`` is
        ``None``).
    """
    from ._gaussian_rasterization import _compute_opacities, _validate_crop

    opacities = _compute_opacities(logit_opacities, projected)
    num, weights = _C.count_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        tiles.image_width,
        tiles.image_height,
        0,
        0,
        tiles.tile_size,
    )

    if crop is not None:
        ox, oy, w, h = _validate_crop(crop, tiles.image_width, tiles.image_height)
        num = num[:, oy : oy + h, ox : ox + w]
        weights = weights[:, oy : oy + h, ox : ox + w]

    return num, weights


def identify_contributing_gaussians(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    num_contributing: torch.Tensor | None = None,
    top_k_contributors: int = 0,
    crop: tuple[int, int, int, int] | None = None,
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
        crop: Optional ``(origin_x, origin_y, width, height)`` tuple defining a
            sub-region to analyse.  When ``None`` (the default), the full image
            is used.  The crop region is clamped to the projected image bounds.

    Returns:
        Tuple of (gaussian_ids, weights) as :class:`~fvdb.JaggedTensor`.
        Each JaggedTensor has structure ``C`` cameras, each with ``H * W``
        pixel rows (crop dimensions when ``crop`` is provided), each pixel
        with a variable number of contributing Gaussian samples.
    """
    from ._gaussian_rasterization import _compute_opacities, _validate_crop

    opacities = _compute_opacities(logit_opacities, projected)

    if top_k_contributors <= 0 and num_contributing is None:
        num_contributing, _ = count_contributing_gaussians(projected, logit_opacities, tiles)

    ids_impl, weights_impl = _C.identify_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        tiles.image_width,
        tiles.image_height,
        0,
        0,
        tiles.tile_size,
        top_k_contributors,
        num_contributing,
    )
    ids, weights = JaggedTensor(impl=ids_impl), JaggedTensor(impl=weights_impl)

    if crop is not None:
        ox, oy, w, h = _validate_crop(crop, tiles.image_width, tiles.image_height)
        ids, weights = _crop_jagged_pixel_pair(ids, weights, tiles.image_width, tiles.image_height, ox, oy, w, h)

    return ids, weights


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
        sparse_tiles.image_width,
        sparse_tiles.image_height,
        0,
        0,
        sparse_tiles.tile_size,
    )

    if sparse_tiles.has_duplicates:
        inv = sparse_tiles.inverse_indices
        pixels_impl = sparse_tiles.pixels_to_render._impl
        jt0_impl = pixels_impl.jagged_like(jt0_impl.jdata.index_select(0, inv))
        jt1_impl = pixels_impl.jagged_like(jt1_impl.jdata.index_select(0, inv))
    return JaggedTensor(impl=jt0_impl), JaggedTensor(impl=jt1_impl)


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

    render_pixels = sparse_tiles.unique_pixels if sparse_tiles.has_duplicates else sparse_tiles.pixels_to_render

    # Resolve num_contributing into unique-pixel space for the kernel
    jt_arg: JaggedTensorCpp
    if num_contributing is None:
        jt_arg, _ = _C.count_contributing_gaussians_sparse(
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
            sparse_tiles.image_width,
            sparse_tiles.image_height,
            0,
            0,
            sparse_tiles.tile_size,
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
        sparse_tiles.image_width,
        sparse_tiles.image_height,
        0,
        0,
        sparse_tiles.tile_size,
        -1,
        jt_arg,
    )

    if sparse_tiles.has_duplicates:
        inv = sparse_tiles.inverse_indices
        pixels_impl = sparse_tiles.pixels_to_render._impl
        ids_impl = pixels_impl.jagged_like(ids_impl.jdata.index_select(0, inv))
        weights_impl = pixels_impl.jagged_like(weights_impl.jdata.index_select(0, inv))
    return JaggedTensor(impl=ids_impl), JaggedTensor(impl=weights_impl)
