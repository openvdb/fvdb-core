# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Unified projection result types for the Gaussian splatting pipeline.

``ProjectedGaussians`` is the single canonical type representing a fully-projected
set of Gaussians (all four pipeline stages complete: geometric projection, opacity
computation, SH evaluation, tile intersection).  Both the OO layer and the
functional API use this type.

``SparseProjectedGaussians`` extends it with the additional tile/pixel structures
needed for sparse rendering.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ...enums import CameraModel, ProjectionMethod
from ...jagged_tensor import JaggedTensor


@dataclass(frozen=True)
class ProjectedGaussians:
    """Result of the full 4-stage Gaussian splatting projection pipeline.

    All four stages are complete: raw geometric projection (stage 1), opacity
    computation (stage 2), spherical harmonics / feature evaluation (stage 3),
    and tile intersection (stage 4).  This is the input type for all rasterization
    and query operations.

    Attributes:
        means2d: ``[C, N, 2]`` Projected 2D means in pixel units.
        conics: ``[C, N, 3]`` Upper-triangle of the inverse 2D covariance.
        render_quantities: ``[C, N, D]`` Per-Gaussian render features (SH colors
            or depths, depending on the render mode).
        depths: ``[C, N]`` Depths along camera z-axis.
        opacities: ``[C, N]`` Per-camera opacities (sigmoid-activated, with
            optional antialiasing compensation applied).
        radii: ``[C, N]`` int32 projected radii (<=0 means culled).
        tile_offsets: Per-tile start offsets into ``tile_gaussian_ids``.
        tile_gaussian_ids: Sorted Gaussian indices per tile.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        eps_2d: Epsilon used for 2D projection numerical stability.
        antialias: Whether antialiasing compensation was applied.
        sh_degree_to_use: SH degree used for feature evaluation.
        min_radius_2d: Minimum projected radius used for culling.
        camera_model: Camera distortion model used during projection.
        projection_method: Projection method used (ANALYTIC or UNSCENTED).
    """

    means2d: torch.Tensor
    conics: torch.Tensor
    render_quantities: torch.Tensor
    depths: torch.Tensor
    opacities: torch.Tensor
    radii: torch.Tensor
    tile_offsets: torch.Tensor
    tile_gaussian_ids: torch.Tensor
    image_width: int
    image_height: int
    near_plane: float
    far_plane: float
    eps_2d: float
    antialias: bool
    sh_degree_to_use: int
    min_radius_2d: float
    camera_model: CameraModel
    projection_method: ProjectionMethod


@dataclass(frozen=True)
class SparseProjectedGaussians(ProjectedGaussians):
    """Projection result with sparse tile/pixel structures for sparse rendering.

    Extends :class:`ProjectedGaussians` with the additional bookkeeping tensors
    produced by the sparse projection pipeline.

    Attributes:
        active_tiles: ``[num_active_tiles]`` Indices of tiles with at least one
            Gaussian.
        active_tile_mask: ``[C, TH, TW]`` Boolean mask of active tiles.
        tile_pixel_mask: ``[num_active_tiles, words_per_tile]`` Per-tile pixel
            bitmask.
        tile_pixel_cumsum: ``[num_active_tiles]`` Cumulative pixel counts per
            active tile.
        pixel_map: ``[num_active_pixels]`` Map from active pixel index to output
            index.
        inverse_indices: ``[total_pixels]`` Scatter-back indices for deduplication.
        unique_pixels_to_render: Deduplicated pixels as a :class:`JaggedTensor`.
        has_duplicates: Whether the pixel set contained duplicates (and
            ``unique_pixels_to_render`` / ``inverse_indices`` are valid).
    """

    active_tiles: torch.Tensor
    active_tile_mask: torch.Tensor
    tile_pixel_mask: torch.Tensor
    tile_pixel_cumsum: torch.Tensor
    pixel_map: torch.Tensor
    inverse_indices: torch.Tensor
    unique_pixels_to_render: JaggedTensor
    has_duplicates: bool
