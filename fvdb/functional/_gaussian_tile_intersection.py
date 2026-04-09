# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Helpers for tile intersection and RenderSettings construction (Stage 3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp as _C
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ._gaussian_projection import ProjectedGaussians


@dataclass(frozen=True)
class GaussianTileIntersection:
    """Result of tile-based Gaussian culling for dense rasterization.

    Identifies which Gaussians overlap which screen-space tiles, producing
    sorted lists for the tiled rasterizer.

    Attributes:
        tile_offsets: Per-tile start offsets into ``tile_gaussian_ids``.
        tile_gaussian_ids: Sorted Gaussian indices per tile.
        tile_size: Tile side length in pixels.
        image_width: Image width in pixels (copied from ``ProjectedGaussians``).
        image_height: Image height in pixels (copied from ``ProjectedGaussians``).
    """

    tile_offsets: torch.Tensor
    tile_gaussian_ids: torch.Tensor
    tile_size: int
    image_width: int
    image_height: int


@dataclass(frozen=True)
class SparseGaussianTileIntersection:
    """Result of sparse tile-based Gaussian culling for sparse rasterization.

    Bundles the sparse tile layout, deduplicated pixel information, and tile
    intersection data needed by ``rasterize_screen_space_gaussians_sparse``.

    Attributes:
        tile_offsets: Sparse tile offsets.
        tile_gaussian_ids: Sparse tile Gaussian IDs.
        unique_pixels: Deduplicated pixel coordinates as a :class:`JaggedTensor`.
        inverse_indices: Mapping back to original (possibly duplicated) pixels.
        has_duplicates: Whether the pixel set contained duplicates (and
            ``inverse_indices`` is valid).
        pixels_to_render: Original pixel coordinates passed by the caller.
        active_tiles: ``[num_active_tiles]`` Indices of tiles with at least one pixel.
        active_tile_mask: ``[C, TH, TW]`` Boolean mask of active tiles.
        tile_pixel_mask: ``[num_active_tiles, words_per_tile]`` Per-tile pixel bitmask.
        tile_pixel_cumsum: ``[num_active_tiles]`` Cumulative pixel counts per tile.
        pixel_map: ``[num_active_pixels]`` Pixel-to-tile mapping.
        tile_size: Tile side length in pixels.
        image_width: Image width in pixels (copied from ``ProjectedGaussians``).
        image_height: Image height in pixels (copied from ``ProjectedGaussians``).
    """

    tile_offsets: torch.Tensor
    tile_gaussian_ids: torch.Tensor
    unique_pixels: JaggedTensor
    inverse_indices: torch.Tensor
    has_duplicates: bool
    pixels_to_render: JaggedTensor
    active_tiles: torch.Tensor
    active_tile_mask: torch.Tensor
    tile_pixel_mask: torch.Tensor
    tile_pixel_cumsum: torch.Tensor
    pixel_map: torch.Tensor
    tile_size: int
    image_width: int
    image_height: int


def _find_unique_pixels(
    pixels_to_render: JaggedTensor,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[JaggedTensor, torch.Tensor, bool]:
    """Deduplicate pixel coordinates within a JaggedTensor.

    Given a JaggedTensor of ``[row, col]`` pixel coordinates (possibly with
    duplicates across or within batches), returns a new JaggedTensor containing
    only unique pixels, an inverse-index tensor mapping original positions to
    unique positions, and a boolean indicating whether any duplicates were found.

    Args:
        pixels_to_render: JaggedTensor with ``rshape = (..., 2)`` holding
            ``[row, col]`` coordinates.
        image_width: Image width (used for key encoding). Inferred if ``None``.
        image_height: Image height (used for key encoding). Inferred if ``None``.

    Returns:
        Tuple of ``(unique_pixels, inverse_indices, has_duplicates)``.
    """
    jdata = pixels_to_render.jdata
    total_pixels = jdata.size(0)

    if total_pixels == 0:
        empty_inv = torch.empty(0, dtype=torch.long, device=jdata.device)
        return pixels_to_render, empty_inv, False

    device = jdata.device
    jidx = pixels_to_render.jidx

    if image_width is None:
        image_width = int(jdata[:, 1].max().item()) + 1
    if image_height is None:
        image_height = int(jdata[:, 0].max().item()) + 1
    num_pixels_per_image = image_height * image_width

    rows = jdata[:, 0].long()
    cols = jdata[:, 1].long()

    single_list = jidx.numel() == 0
    if single_list:
        keys = rows * image_width + cols
    else:
        keys = jidx.long() * num_pixels_per_image + rows * image_width + cols

    sorted_keys, sort_perm = keys.sort()

    is_group_start = torch.ones(total_pixels, dtype=torch.bool, device=device)
    if total_pixels > 1:
        is_group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]

    first_in_sorted = is_group_start.nonzero(as_tuple=False).squeeze(1)

    group_ids = is_group_start.long().cumsum(0) - 1
    num_unique = group_ids[-1].item() + 1

    if num_unique == total_pixels:
        return pixels_to_render, torch.arange(total_pixels, dtype=torch.long, device=device), False

    inverse_indices = torch.empty(total_pixels, dtype=torch.long, device=device)
    inverse_indices[sort_perm] = group_ids

    unique_orig_indices = sort_perm[first_in_sorted]
    unique_jdata = jdata[unique_orig_indices]

    if single_list:
        unique_batch_idx = torch.zeros(num_unique, dtype=torch.long, device=device)
    else:
        unique_batch_idx = jidx.long()[unique_orig_indices]

    num_lists = len(pixels_to_render)
    counts_per_list = torch.bincount(unique_batch_idx, minlength=num_lists)
    new_offsets = torch.zeros(num_lists + 1, dtype=torch.long, device=device)
    new_offsets[1:] = counts_per_list.cumsum(0)

    empty_lidx = torch.empty((0, 1), dtype=torch.int32, device=device)
    unique_impl = _C.JaggedTensor.from_data_offsets_and_list_ids(unique_jdata, new_offsets, empty_lidx)
    unique_pixels = JaggedTensor(impl=unique_impl)

    return unique_pixels, inverse_indices, True


def intersect_gaussian_tiles(
    projected: ProjectedGaussians,
    tile_size: int = 16,
) -> GaussianTileIntersection:
    """Compute tile-Gaussian intersections for tiled rasterization (Stage 3).

    Non-differentiable.  ``num_cameras``, ``image_width``, and ``image_height``
    are derived from ``projected`` -- not passed explicitly.

    Args:
        projected: :class:`ProjectedGaussians` from Stage 1.
        tile_size: Tile side length in pixels (default 16).

    Returns:
        A :class:`GaussianTileIntersection` with tile offsets, sorted Gaussian
        IDs, and the tile/image dimensions.
    """
    image_width = projected.image_width
    image_height = projected.image_height
    num_cameras = projected.means2d.shape[0]
    num_tiles_w = math.ceil(image_width / tile_size)
    num_tiles_h = math.ceil(image_height / tile_size)
    tile_offsets, tile_gaussian_ids = _C.intersect_gaussian_tiles(
        projected.means2d, projected.radii, projected.depths,
        num_cameras, tile_size, num_tiles_h, num_tiles_w,
    )
    return GaussianTileIntersection(
        tile_offsets=tile_offsets,
        tile_gaussian_ids=tile_gaussian_ids,
        tile_size=tile_size,
        image_width=image_width,
        image_height=image_height,
    )


def intersect_gaussian_tiles_sparse(
    pixels_to_render: JaggedTensor,
    projected: ProjectedGaussians,
    tile_size: int = 16,
) -> SparseGaussianTileIntersection:
    """Compute sparse tile-Gaussian intersections for sparse rasterization (Stage 3).

    Fuses pixel deduplication, sparse tile layout computation, and sparse tile
    intersection into a single call.  Non-differentiable.

    Args:
        pixels_to_render: :class:`JaggedTensor` of ``[C, num_pixels, 2]`` pixel
            coordinates (may contain duplicates).
        projected: :class:`ProjectedGaussians` from Stage 1.
        tile_size: Tile side length in pixels (default 16).

    Returns:
        A :class:`SparseGaussianTileIntersection` bundling all sparse tile
        layout and intersection data.
    """
    image_width = projected.image_width
    image_height = projected.image_height
    num_cameras = projected.means2d.shape[0]
    num_tiles_w = math.ceil(image_width / tile_size)
    num_tiles_h = math.ceil(image_height / tile_size)

    unique_pixels, inverse_indices, has_duplicates = _find_unique_pixels(pixels_to_render)

    active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = (
        _C.build_sparse_gaussian_tile_layout(tile_size, num_tiles_w, num_tiles_h, unique_pixels._impl)
    )

    sparse_tile_offsets, sparse_tile_gaussian_ids = _C.intersect_gaussian_tiles_sparse(
        projected.means2d,
        projected.radii,
        projected.depths,
        active_tile_mask,
        active_tiles,
        num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
    )

    return SparseGaussianTileIntersection(
        tile_offsets=sparse_tile_offsets,
        tile_gaussian_ids=sparse_tile_gaussian_ids,
        unique_pixels=unique_pixels,
        inverse_indices=inverse_indices,
        has_duplicates=has_duplicates,
        pixels_to_render=pixels_to_render,
        active_tiles=active_tiles,
        active_tile_mask=active_tile_mask,
        tile_pixel_mask=tile_pixel_mask,
        tile_pixel_cumsum=tile_pixel_cumsum,
        pixel_map=pixel_map,
        tile_size=tile_size,
        image_width=image_width,
        image_height=image_height,
    )
