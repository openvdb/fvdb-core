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
    """Return a coarsened version of the grid batch."""
    cf = to_Vec3i(coarsening_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.coarsen_grid(grid.data, cf))


def coarsened_grid_single(
    grid: Grid,
    coarsening_factor: NumericMaxRank1,
) -> Grid:
    """Return a coarsened version of the single grid."""
    cf = to_Vec3i(coarsening_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.coarsen_grid(grid.data, cf))


def refined_grid_batch(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    mask: JaggedTensor | None = None,
) -> GridBatch:
    """Return a refined (subdivided) version of the grid batch."""
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
    """Return a refined (subdivided) version of the single grid."""
    sf = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    if mask is not None:
        m = JaggedTensor(mask)._impl
    else:
        m = None
    return _wrap_single_grid(_fvdb_cpp.upsample_grid(grid.data, sf, m))


def dual_grid_batch(grid: GridBatch, exclude_border: bool = False) -> GridBatch:
    """Return the dual grid of a grid batch."""
    return _wrap_grid(_fvdb_cpp.dual_grid(grid.data, exclude_border))


def dual_grid_single(grid: Grid, exclude_border: bool = False) -> Grid:
    """Return the dual grid of a single grid."""
    return _wrap_single_grid(_fvdb_cpp.dual_grid(grid.data, exclude_border))


def dilated_grid_batch(grid: GridBatch, dilation: int) -> GridBatch:
    """Return a dilated version of the grid batch."""
    return _wrap_grid(_fvdb_cpp.dilate_grid(grid.data, [dilation] * grid.data.grid_count))


def dilated_grid_single(grid: Grid, dilation: int) -> Grid:
    """Return a dilated version of the single grid."""
    return _wrap_single_grid(_fvdb_cpp.dilate_grid(grid.data, [dilation] * grid.data.grid_count))


def merged_grid_batch(grid: GridBatch, other: GridBatch) -> GridBatch:
    """Return the union of two grid batches."""
    return _wrap_grid(_fvdb_cpp.merge_grids(grid.data, other.data))


def merged_grid_single(grid: Grid, other: Grid) -> Grid:
    """Return the union of two single grids."""
    return _wrap_single_grid(_fvdb_cpp.merge_grids(grid.data, other.data))


def pruned_grid_batch(
    grid: GridBatch,
    mask: JaggedTensor,
) -> GridBatch:
    """Return a grid batch containing only voxels where ``mask`` is True."""
    return _wrap_grid(_fvdb_cpp.prune_grid(grid.data, mask._impl))


def pruned_grid_single(
    grid: Grid,
    mask: torch.Tensor,
) -> Grid:
    """Return a single grid containing only voxels where ``mask`` is True."""
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
    """Return a grid batch clipped to ``[ijk_min, ijk_max]``."""
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    return _wrap_grid(_fvdb_cpp.clip_grid(grid.data, mn, mx))


def clipped_grid_single(
    grid: Grid,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> Grid:
    """Return a single grid clipped to ``[ijk_min, ijk_max]``."""
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    return _wrap_single_grid(_fvdb_cpp.clip_grid(grid.data, mn, mx))


def clip_batch(
    grid: GridBatch,
    features: JaggedTensor,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> tuple[JaggedTensor, GridBatch]:
    """Clip a grid batch and its features to ``[ijk_min, ijk_max]``."""
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    result_features_impl, result_grid_impl = _fvdb_cpp.clip_grid_features_with_mask(
        grid.data, features._impl, mn, mx
    )
    return JaggedTensor(impl=result_features_impl), _wrap_grid(result_grid_impl)


def clip_single(
    grid: Grid,
    features: torch.Tensor,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> tuple[torch.Tensor, Grid]:
    """Clip a single grid and its features to ``[ijk_min, ijk_max]``."""
    mn, mx = _normalize_clip_bounds(grid.data, ijk_min, ijk_max)
    jt = JaggedTensor(features)
    result_features_impl, result_grid_impl = _fvdb_cpp.clip_grid_features_with_mask(
        grid.data, jt._impl, mn, mx
    )
    return JaggedTensor(impl=result_features_impl).jdata, _wrap_single_grid(result_grid_impl)


def contiguous_batch(grid: GridBatch) -> GridBatch:
    """Return a contiguous copy of the grid batch."""
    return _wrap_grid(_fvdb_cpp.make_contiguous(grid.data))


def contiguous_single(grid: Grid) -> Grid:
    """Return a contiguous copy of the single grid."""
    return _wrap_single_grid(_fvdb_cpp.make_contiguous(grid.data))


def clone_grid_batch(grid: GridBatch, device: torch.device) -> GridBatch:
    """Clone a grid batch to the specified device."""
    return _wrap_grid(_fvdb_cpp.clone_grid(grid.data, device))


def clone_grid_single(grid: Grid, device: torch.device) -> Grid:
    """Clone a single grid to the specified device."""
    return _wrap_single_grid(_fvdb_cpp.clone_grid(grid.data, device))


# ---------------------------------------------------------------------------
#  Convolution output grids
# ---------------------------------------------------------------------------


def conv_grid_batch(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """Return the output grid for a convolution on a grid batch."""
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_grid(grid.data, ks, st))


def conv_grid_single(
    grid: Grid,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid:
    """Return the output grid for a convolution on a single grid."""
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.conv_grid(grid.data, ks, st))


def conv_transpose_grid_batch(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """Return the output grid for a transposed convolution on a grid batch."""
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_transpose_grid(grid.data, ks, st))


def conv_transpose_grid_single(
    grid: Grid,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid:
    """Return the output grid for a transposed convolution on a single grid."""
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_single_grid(_fvdb_cpp.conv_transpose_grid(grid.data, ks, st))


# ---------------------------------------------------------------------------
#  Space-filling curves
# ---------------------------------------------------------------------------


def morton_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return Morton (Z-order) codes for active voxels in a grid batch."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton", offset.tolist()))


def morton_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return Morton (Z-order) codes for active voxels in a single grid."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton", offset.tolist())).jdata


def morton_zyx_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return transposed Morton codes (zyx interleaving) for a grid batch."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton_zyx", offset.tolist()))


def morton_zyx_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return transposed Morton codes (zyx interleaving) for a single grid."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton_zyx", offset.tolist())).jdata


def hilbert_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return Hilbert curve codes for active voxels in a grid batch."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert", offset.tolist()))


def hilbert_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return Hilbert curve codes for active voxels in a single grid."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert", offset.tolist())).jdata


def hilbert_zyx_batch(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """Return transposed Hilbert codes (zyx ordering) for a grid batch."""
    grid_data = grid.data
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert_zyx", offset.tolist()))


def hilbert_zyx_single(grid: Grid, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor:
    """Return transposed Hilbert codes (zyx ordering) for a single grid."""
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
    """Return the edge network of a grid batch."""
    a, b = _fvdb_cpp.grid_edge_network(grid.data, return_voxel_coordinates)
    return JaggedTensor(impl=a), JaggedTensor(impl=b)


def edge_network_single(grid: Grid, return_voxel_coordinates: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the edge network of a single grid."""
    a, b = _fvdb_cpp.grid_edge_network(grid.data, return_voxel_coordinates)
    return JaggedTensor(impl=a).jdata, JaggedTensor(impl=b).jdata
