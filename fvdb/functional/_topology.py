# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for grid topology operations (coarsen, refine, dual, dilate, etc.)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..jagged_tensor import JaggedTensor
from .. import _fvdb_cpp
from ..types import NumericMaxRank1, NumericMaxRank2, ValueConstraint, to_Vec3i, to_Vec3iBatchBroadcastable

if TYPE_CHECKING:
    from ..grid_batch import GridBatch
    from ..grid import Grid


def _wrap_grid(cpp_impl):
    from ..grid_batch import GridBatch

    return GridBatch(data=cpp_impl)


def _wrap_single_grid(cpp_impl):
    from ..grid import Grid

    return Grid(data=cpp_impl)


# ---------------------------------------------------------------------------
#  Grid structure derivation
# ---------------------------------------------------------------------------


def coarsened_grid_batch(
    grid: GridBatch,
    coarsening_factor: NumericMaxRank1,
) -> GridBatch:
    """Return a coarsened version of a grid batch.

    Args:
        grid (GridBatch): The grid batch to coarsen.
        coarsening_factor (NumericMaxRank1): Factor per axis, broadcastable to ``(3,)``.

    Returns:
        result (GridBatch): The coarsened grid batch.

    .. seealso:: :func:`coarsened_grid_single`
    """
    cf = to_Vec3i(coarsening_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.coarsen_grid(grid.data, cf))


def coarsened_grid_single(
    grid: Grid,
    coarsening_factor: NumericMaxRank1,
) -> Grid:
    """Return a coarsened version of a single grid.

    Args:
        grid (Grid): The single grid to coarsen.
        coarsening_factor (NumericMaxRank1): Factor per axis, broadcastable to ``(3,)``.

    Returns:
        result (Grid): The coarsened grid.

    .. seealso:: :func:`coarsened_grid_batch`
    """
    cf = to_Vec3i(coarsening_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.coarsen_grid(grid.data, cf))


def refined_grid_batch(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    mask: JaggedTensor | None = None,
) -> GridBatch:
    """Return a refined (subdivided) version of a grid batch.

    Args:
        grid (GridBatch): The grid batch to refine.
        subdiv_factor (NumericMaxRank1): Subdivision factor per axis, broadcastable to ``(3,)``.
        mask (JaggedTensor | None): Optional boolean mask selecting voxels to refine.

    Returns:
        result (GridBatch): The refined grid batch.

    .. seealso:: :func:`refined_grid_single`
    """
    sf = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    if mask is not None:
        m = mask._impl
    else:
        m = None
    return _wrap_grid(_fvdb_cpp.upsample_grid(grid.data, sf, m))


def refined_grid_single(
    grid: Grid,
    subdiv_factor: NumericMaxRank1,
    mask: torch.Tensor | None = None,
) -> Grid:
    """Return a refined (subdivided) version of a single grid.

    Args:
        grid (Grid): The single grid to refine.
        subdiv_factor (NumericMaxRank1): Subdivision factor per axis, broadcastable to ``(3,)``.
        mask (torch.Tensor | None): Optional boolean mask selecting voxels to refine.

    Returns:
        result (Grid): The refined grid.

    .. seealso:: :func:`refined_grid_batch`
    """
    sf = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    if mask is not None:
        m = JaggedTensor(mask)._impl
    else:
        m = None
    return _wrap_single_grid(_fvdb_cpp.upsample_grid(grid.data, sf, m))


def dual_grid_batch(grid: GridBatch, exclude_border: bool = False) -> GridBatch:
    """Return the dual grid of a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        exclude_border (bool): If ``True``, exclude border voxels from the dual.

    Returns:
        result (GridBatch): The dual grid batch.

    .. seealso:: :func:`dual_grid_single`
    """
    return _wrap_grid(_fvdb_cpp.dual_grid(grid.data, exclude_border))


def dual_grid_single(grid: Grid, exclude_border: bool = False) -> Grid:
    """Return the dual grid of a single grid.

    Args:
        grid (Grid): The single grid.
        exclude_border (bool): If ``True``, exclude border voxels from the dual.

    Returns:
        result (Grid): The dual grid.

    .. seealso:: :func:`dual_grid_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.dual_grid(grid.data, exclude_border))


def dilated_grid_batch(grid: GridBatch, dilation: int) -> GridBatch:
    """Return a dilated version of a grid batch.

    Args:
        grid (GridBatch): The grid batch to dilate.
        dilation (int): Number of voxels to dilate by.

    Returns:
        result (GridBatch): The dilated grid batch.

    .. seealso:: :func:`dilated_grid_single`
    """
    return _wrap_grid(_fvdb_cpp.dilate_grid(grid.data, [dilation] * grid.data.grid_count))


def dilated_grid_single(grid: Grid, dilation: int) -> Grid:
    """Return a dilated version of a single grid.

    Args:
        grid (Grid): The single grid to dilate.
        dilation (int): Number of voxels to dilate by.

    Returns:
        result (Grid): The dilated grid.

    .. seealso:: :func:`dilated_grid_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.dilate_grid(grid.data, [dilation] * grid.data.grid_count))


def merged_grid_batch(grid: GridBatch, other: GridBatch) -> GridBatch:
    """Return the union of two grid batches.

    Args:
        grid (GridBatch): The first grid batch.
        other (GridBatch): The second grid batch.

    Returns:
        result (GridBatch): Grid batch containing the union of active voxels.

    .. seealso:: :func:`merged_grid_single`
    """
    return _wrap_grid(_fvdb_cpp.merge_grids(grid.data, other.data))


def merged_grid_single(grid: Grid, other: Grid) -> Grid:
    """Return the union of two single grids.

    Args:
        grid (Grid): The first single grid.
        other (Grid): The second single grid.

    Returns:
        result (Grid): Grid containing the union of active voxels.

    .. seealso:: :func:`merged_grid_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.merge_grids(grid.data, other.data))


def pruned_grid_batch(
    grid: GridBatch,
    mask: JaggedTensor,
) -> GridBatch:
    """Return a grid batch containing only voxels where ``mask`` is True.

    Args:
        grid (GridBatch): The grid batch to prune.
        mask (JaggedTensor): Boolean mask selecting voxels to keep.

    Returns:
        result (GridBatch): The pruned grid batch.

    .. seealso:: :func:`pruned_grid_single`
    """
    return _wrap_grid(_fvdb_cpp.prune_grid(grid.data, mask._impl))


def pruned_grid_single(
    grid: Grid,
    mask: torch.Tensor,
) -> Grid:
    """Return a single grid containing only voxels where ``mask`` is True.

    Args:
        grid (Grid): The single grid to prune.
        mask (torch.Tensor): Boolean mask selecting voxels to keep.

    Returns:
        result (Grid): The pruned grid.

    .. seealso:: :func:`pruned_grid_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.prune_grid(grid.data, JaggedTensor(mask)._impl))


def _normalize_clip_bounds(
    grid_data: _fvdb_cpp.GridBatchData,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> tuple[list, list]:
    """Normalize clip bounds, expanding 1D inputs to (grid_count, 3)."""
    ijk_min_t = to_Vec3iBatchBroadcastable(ijk_min)
    ijk_max_t = to_Vec3iBatchBroadcastable(ijk_max)
    if ijk_min_t.dim() == 1:
        ijk_min_t = ijk_min_t.unsqueeze(0).expand(grid_data.grid_count, 3)
    if ijk_max_t.dim() == 1:
        ijk_max_t = ijk_max_t.unsqueeze(0).expand(grid_data.grid_count, 3)
    return ijk_min_t.tolist(), ijk_max_t.tolist()


def clipped_grid_batch(
    grid: GridBatch,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> GridBatch:
    """Return a grid batch clipped to the voxel-space range ``[ijk_min, ijk_max]``.

    Args:
        grid (GridBatch): The grid batch to clip.
        ijk_min (NumericMaxRank2): Minimum voxel coordinate bound.
        ijk_max (NumericMaxRank2): Maximum voxel coordinate bound.

    Returns:
        result (GridBatch): The clipped grid batch.

    .. seealso:: :func:`clipped_grid_single`
    """
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    return _wrap_grid(_fvdb_cpp.clip_grid(grid.data, mn, mx))


def clipped_grid_single(
    grid: Grid,
    ijk_min: NumericMaxRank1,
    ijk_max: NumericMaxRank1,
) -> Grid:
    """Return a single grid clipped to the voxel-space range ``[ijk_min, ijk_max]``.

    Args:
        grid (Grid): The single grid to clip.
        ijk_min (NumericMaxRank1): Minimum voxel coordinate bound.
        ijk_max (NumericMaxRank1): Maximum voxel coordinate bound.

    Returns:
        result (Grid): The clipped grid.

    .. seealso:: :func:`clipped_grid_batch`
    """
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    return _wrap_single_grid(_fvdb_cpp.clip_grid(grid.data, mn, mx))


def clip_batch(
    grid: GridBatch,
    features: JaggedTensor,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> tuple[JaggedTensor, GridBatch]:
    """Clip a grid batch and its features to the voxel-space range ``[ijk_min, ijk_max]``.

    Supports backpropagation on features.

    Args:
        grid (GridBatch): The grid batch to clip.
        features (JaggedTensor): Per-voxel feature data.
        ijk_min (NumericMaxRank2): Minimum voxel coordinate bound.
        ijk_max (NumericMaxRank2): Maximum voxel coordinate bound.

    Returns:
        clipped_features (JaggedTensor): Features for the clipped voxels.
        clipped_grid (GridBatch): The clipped grid batch.

    .. seealso:: :func:`clip_single`
    """
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    result_features_impl, result_grid_impl = _fvdb_cpp.clip_grid_features_with_mask(grid.data, features._impl, mn, mx)
    return JaggedTensor(impl=result_features_impl), _wrap_grid(result_grid_impl)


def clip_single(
    grid: Grid,
    features: torch.Tensor,
    ijk_min: NumericMaxRank1,
    ijk_max: NumericMaxRank1,
) -> tuple[torch.Tensor, Grid]:
    """Clip a single grid and its features to the voxel-space range ``[ijk_min, ijk_max]``.

    Supports backpropagation on features.

    Args:
        grid (Grid): The single grid to clip.
        features (torch.Tensor): Per-voxel feature data.
        ijk_min (NumericMaxRank1): Minimum voxel coordinate bound.
        ijk_max (NumericMaxRank1): Maximum voxel coordinate bound.

    Returns:
        clipped_features (torch.Tensor): Features for the clipped voxels.
        clipped_grid (Grid): The clipped grid.

    .. seealso:: :func:`clip_batch`
    """
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    jt = JaggedTensor(features)
    result_features_impl, result_grid_impl = _fvdb_cpp.clip_grid_features_with_mask(grid.data, jt._impl, mn, mx)
    return JaggedTensor(impl=result_features_impl).jdata, _wrap_single_grid(result_grid_impl)


def contiguous_batch(grid: GridBatch) -> GridBatch:
    """Return a contiguous copy of a grid batch.

    Args:
        grid (GridBatch): The grid batch.

    Returns:
        result (GridBatch): A contiguous copy of the grid batch.

    .. seealso:: :func:`contiguous_single`
    """
    return _wrap_grid(_fvdb_cpp.make_contiguous(grid.data))


def contiguous_single(grid: Grid) -> Grid:
    """Return a contiguous copy of a single grid.

    Args:
        grid (Grid): The single grid.

    Returns:
        result (Grid): A contiguous copy of the grid.

    .. seealso:: :func:`contiguous_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.make_contiguous(grid.data))


def clone_grid_batch(grid: GridBatch, device: torch.device) -> GridBatch:
    """Clone a grid batch to the specified device.

    Args:
        grid (GridBatch): The grid batch to clone.
        device (torch.device): Target device.

    Returns:
        result (GridBatch): A clone of the grid batch on the target device.

    .. seealso:: :func:`clone_grid_single`
    """
    return _wrap_grid(_fvdb_cpp.clone_grid(grid.data, device))


def clone_grid_single(grid: Grid, device: torch.device) -> Grid:
    """Clone a single grid to the specified device.

    Args:
        grid (Grid): The single grid to clone.
        device (torch.device): Target device.

    Returns:
        result (Grid): A clone of the grid on the target device.

    .. seealso:: :func:`clone_grid_batch`
    """
    return _wrap_single_grid(_fvdb_cpp.clone_grid(grid.data, device))


# ---------------------------------------------------------------------------
#  Convolution output grids
# ---------------------------------------------------------------------------


def conv_grid_batch(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """Return the output grid for a convolution on a grid batch.

    Args:
        grid (GridBatch): The input grid batch.
        kernel_size (NumericMaxRank1): Convolution kernel size, broadcastable to ``(3,)``.
        stride (NumericMaxRank1): Convolution stride, broadcastable to ``(3,)``.

    Returns:
        result (GridBatch): The output grid batch for the convolution.

    .. seealso:: :func:`conv_grid_single`
    """
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_grid(grid.data, ks, st))


def conv_grid_single(
    grid: Grid,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid:
    """Return the output grid for a convolution on a single grid.

    Args:
        grid (Grid): The input single grid.
        kernel_size (NumericMaxRank1): Convolution kernel size, broadcastable to ``(3,)``.
        stride (NumericMaxRank1): Convolution stride, broadcastable to ``(3,)``.

    Returns:
        result (Grid): The output grid for the convolution.

    .. seealso:: :func:`conv_grid_batch`
    """
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.conv_grid(grid.data, ks, st))


def conv_transpose_grid_batch(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """Return the output grid for a transposed convolution on a grid batch.

    Args:
        grid (GridBatch): The input grid batch.
        kernel_size (NumericMaxRank1): Kernel size, broadcastable to ``(3,)``.
        stride (NumericMaxRank1): Stride, broadcastable to ``(3,)``.

    Returns:
        result (GridBatch): The output grid batch for the transposed convolution.

    .. seealso:: :func:`conv_transpose_grid_single`
    """
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_transpose_grid(grid.data, ks, st))


def conv_transpose_grid_single(
    grid: Grid,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid:
    """Return the output grid for a transposed convolution on a single grid.

    Args:
        grid (Grid): The input single grid.
        kernel_size (NumericMaxRank1): Kernel size, broadcastable to ``(3,)``.
        stride (NumericMaxRank1): Stride, broadcastable to ``(3,)``.

    Returns:
        result (Grid): The output grid for the transposed convolution.

    .. seealso:: :func:`conv_transpose_grid_batch`
    """
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.conv_transpose_grid(grid.data, ks, st))


# ---------------------------------------------------------------------------
#  Space-filling curves
# ---------------------------------------------------------------------------


def morton_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return Morton (Z-order) codes for active voxels in a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (JaggedTensor): Morton codes per active voxel.

    .. seealso:: :func:`morton_single`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton", offset.tolist()))


def morton_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return Morton (Z-order) codes for active voxels in a single grid.

    Args:
        grid (Grid): The single grid.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (torch.Tensor): Morton codes per active voxel.

    .. seealso:: :func:`morton_batch`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton", offset.tolist())).jdata


def morton_zyx_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return transposed Morton codes (zyx interleaving) for a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (JaggedTensor): Transposed Morton codes per active voxel.

    .. seealso:: :func:`morton_zyx_single`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton_zyx", offset.tolist()))


def morton_zyx_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return transposed Morton codes (zyx interleaving) for a single grid.

    Args:
        grid (Grid): The single grid.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (torch.Tensor): Transposed Morton codes per active voxel.

    .. seealso:: :func:`morton_zyx_batch`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton_zyx", offset.tolist())).jdata


def hilbert_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return Hilbert curve codes for active voxels in a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (JaggedTensor): Hilbert codes per active voxel.

    .. seealso:: :func:`hilbert_single`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert", offset.tolist()))


def hilbert_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return Hilbert curve codes for active voxels in a single grid.

    Args:
        grid (Grid): The single grid.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (torch.Tensor): Hilbert codes per active voxel.

    .. seealso:: :func:`hilbert_batch`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert", offset.tolist())).jdata


def hilbert_zyx_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return transposed Hilbert codes (zyx ordering) for a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (JaggedTensor): Transposed Hilbert codes per active voxel.

    .. seealso:: :func:`hilbert_zyx_single`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert_zyx", offset.tolist()))


def hilbert_zyx_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return transposed Hilbert codes (zyx ordering) for a single grid.

    Args:
        grid (Grid): The single grid.
        offset (torch.Tensor | NumericMaxRank1 | None): Coordinate offset before encoding.

    Returns:
        codes (torch.Tensor): Transposed Hilbert codes per active voxel.

    .. seealso:: :func:`hilbert_zyx_batch`
    """
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert_zyx", offset.tolist())).jdata


# ---------------------------------------------------------------------------
#  Edge network
# ---------------------------------------------------------------------------


def edge_network_batch(grid: GridBatch, return_voxel_coordinates: bool = False) -> tuple[JaggedTensor, JaggedTensor]:
    """Return the edge network of a grid batch.

    Args:
        grid (GridBatch): The grid batch.
        return_voxel_coordinates (bool): If ``True``, return voxel coordinates instead of indices.

    Returns:
        sources (JaggedTensor): Source node indices or coordinates for each edge.
        targets (JaggedTensor): Target node indices or coordinates for each edge.

    .. seealso:: :func:`edge_network_single`
    """
    a, b = _fvdb_cpp.grid_edge_network(grid.data, return_voxel_coordinates)
    return JaggedTensor(impl=a), JaggedTensor(impl=b)


def edge_network_single(grid: Grid, return_voxel_coordinates: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the edge network of a single grid.

    Args:
        grid (Grid): The single grid.
        return_voxel_coordinates (bool): If ``True``, return voxel coordinates instead of indices.

    Returns:
        sources (torch.Tensor): Source node indices or coordinates for each edge.
        targets (torch.Tensor): Target node indices or coordinates for each edge.

    .. seealso:: :func:`edge_network_batch`
    """
    a, b = _fvdb_cpp.grid_edge_network(grid.data, return_voxel_coordinates)
    return JaggedTensor(impl=a).jdata, JaggedTensor(impl=b).jdata
