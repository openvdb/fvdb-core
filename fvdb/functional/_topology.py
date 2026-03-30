# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for grid topology operations (coarsen, refine, dual, dilate, etc.)."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor
from .. import _fvdb_cpp
from ..types import NumericMaxRank1, NumericMaxRank2, ValueConstraint, to_Vec3i, to_Vec3iBatchBroadcastable
from ._dispatch import _get_grid_data

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


def _wrap_grid(cpp_impl):
    from ..grid_batch import GridBatch

    return GridBatch(data=cpp_impl)


# ---------------------------------------------------------------------------
#  Grid structure derivation
# ---------------------------------------------------------------------------


def coarsened_grid(
    grid: GridBatch,
    coarsening_factor: NumericMaxRank1,
) -> GridBatch:
    """
    Return a coarsened version of the grid, keeping only voxels whose
    coordinates are divisible by ``coarsening_factor``.

    Args:
        grid: The grid to coarsen.
        coarsening_factor: Factor per axis, broadcastable to ``(3,)``, integer dtype.

    Returns:
        A new grid with coarsened structure.
    """
    grid_data = _get_grid_data(grid)
    cf = to_Vec3i(coarsening_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.coarsen_grid(grid_data, cf))


def refined_grid(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    mask: torch.Tensor | JaggedTensor | None = None,
) -> GridBatch:
    """
    Return a refined (subdivided) version of the grid. Each voxel is
    subdivided by ``subdiv_factor``. An optional boolean ``mask`` selects which
    voxels to refine.

    Args:
        grid: The grid to refine.
        subdiv_factor: Factor per axis, broadcastable to ``(3,)``, integer dtype.
        mask: Optional boolean mask selecting voxels to refine.

    Returns:
        A new grid with refined structure.

    .. seealso:: :func:`~fvdb.functional.refine` in ``_pooling`` for refining *data*.
    """
    grid_data = _get_grid_data(grid)
    sf = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = JaggedTensor(mask)
        m = mask._impl
    else:
        m = None
    return _wrap_grid(_fvdb_cpp.upsample_grid(grid_data, sf, m))


def dual_grid(grid: GridBatch, exclude_border: bool = False) -> GridBatch:
    """
    Return the dual grid whose voxel centers correspond to corners of the
    primal grid.

    Args:
        grid: The primal grid.
        exclude_border: Exclude border voxels that extend beyond primal bounds.

    Returns:
        A new dual grid.
    """
    grid_data = _get_grid_data(grid)
    return _wrap_grid(_fvdb_cpp.dual_grid(grid_data, exclude_border))


def dilated_grid(grid: GridBatch, dilation: int) -> GridBatch:
    """
    Return a grid dilated by ``dilation`` voxels.

    Args:
        grid: The grid to dilate.
        dilation: Dilation radius in voxels.

    Returns:
        A new dilated grid.
    """
    grid_data = _get_grid_data(grid)
    return _wrap_grid(_fvdb_cpp.dilate_grid(grid_data, [dilation] * grid_data.grid_count))


def merged_grid(grid: GridBatch, other: GridBatch) -> GridBatch:
    """
    Return the union of two grids (merged active voxels).

    Args:
        grid: First grid.
        other: Second grid to merge with.

    Returns:
        A new grid containing the union of active voxels.
    """
    grid_data = _get_grid_data(grid)
    other_data = _get_grid_data(other)
    return _wrap_grid(_fvdb_cpp.merge_grids(grid_data, other_data))


@overload
def pruned_grid(grid: GridBatch, mask: torch.Tensor) -> GridBatch: ...


@overload
def pruned_grid(grid: GridBatch, mask: JaggedTensor) -> GridBatch: ...


def pruned_grid(
    grid: GridBatch,
    mask: torch.Tensor | JaggedTensor,
) -> GridBatch:
    """
    Return a grid containing only voxels where ``mask`` is ``True``.

    Args:
        grid: The grid to prune.
        mask: Boolean mask for each voxel.

    Returns:
        A new pruned grid.
    """
    grid_data = _get_grid_data(grid)
    if isinstance(mask, torch.Tensor):
        mask = JaggedTensor(mask)
    return _wrap_grid(_fvdb_cpp.prune_grid(grid_data, mask._impl))


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


def clipped_grid(
    grid: GridBatch,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> GridBatch:
    """
    Return a grid containing only voxels within ``[ijk_min, ijk_max]``.

    Args:
        grid: The grid to clip.
        ijk_min: Minimum voxel-space bounds, broadcastable to ``(B, 3)``, integer dtype.
        ijk_max: Maximum voxel-space bounds, broadcastable to ``(B, 3)``, integer dtype.

    Returns:
        A new clipped grid.
    """
    grid_data = _get_grid_data(grid)
    mn, mx = _normalize_clip_bounds(grid_data, ijk_min, ijk_max)
    return _wrap_grid(_fvdb_cpp.clip_grid(grid_data, mn, mx))


def clip(
    grid: GridBatch,
    features: JaggedTensor,
    ijk_min: NumericMaxRank2,
    ijk_max: NumericMaxRank2,
) -> tuple[JaggedTensor, GridBatch]:
    """
    Clip a grid and its associated features to ``[ijk_min, ijk_max]``.

    Supports backpropagation through the clipping operation.

    Args:
        grid: The grid to clip.
        features: Voxel features to clip alongside the grid.
        ijk_min: Minimum voxel-space bounds, broadcastable to ``(B, 3)``, integer dtype.
        ijk_max: Maximum voxel-space bounds, broadcastable to ``(B, 3)``, integer dtype.

    Returns:
        A tuple ``(clipped_features, clipped_grid)``.
    """
    grid_data = _get_grid_data(grid)
    mn, mx = _normalize_clip_bounds(grid_data, ijk_min, ijk_max)
    result_features_impl, result_grid_impl = _fvdb_cpp.clip_grid_features_with_mask(
        grid_data, features._impl, mn, mx
    )
    return JaggedTensor(impl=result_features_impl), _wrap_grid(result_grid_impl)


def contiguous(grid: GridBatch) -> GridBatch:
    """
    Return a contiguous copy of the grid batch.

    Args:
        grid: The grid to make contiguous.

    Returns:
        A new grid with contiguous memory layout.
    """
    return _wrap_grid(_fvdb_cpp.make_contiguous(_get_grid_data(grid)))


def clone_grid(grid: GridBatch, device: torch.device) -> GridBatch:
    """
    Clone a grid to the specified device.

    Args:
        grid: The grid to clone.
        device: Target device.

    Returns:
        A new grid on the target device.
    """
    return _wrap_grid(_fvdb_cpp.clone_grid(_get_grid_data(grid), device))


# ---------------------------------------------------------------------------
#  Convolution output grids
# ---------------------------------------------------------------------------


def conv_grid(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """
    Return the grid representing active voxels at the output of a convolution
    with the given kernel size and stride.

    Args:
        grid: The input grid.
        kernel_size: Kernel size, broadcastable to ``(3,)``, integer dtype.
        stride: Stride, broadcastable to ``(3,)``, integer dtype.

    Returns:
        Output grid for the convolution.
    """
    grid_data = _get_grid_data(grid)
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_grid(grid_data, ks, st))


def conv_transpose_grid(
    grid: GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> GridBatch:
    """
    Return the grid representing active voxels at the output of a transposed
    convolution with the given kernel size and stride.

    Args:
        grid: The input grid.
        kernel_size: Kernel size, broadcastable to ``(3,)``, integer dtype.
        stride: Stride, broadcastable to ``(3,)``, integer dtype.

    Returns:
        Output grid for the transposed convolution.
    """
    grid_data = _get_grid_data(grid)
    ks = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE).tolist()
    st = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE).tolist()
    return _wrap_grid(_fvdb_cpp.conv_transpose_grid(grid_data, ks, st))


# ---------------------------------------------------------------------------
#  Space-filling curves
# ---------------------------------------------------------------------------


def morton(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """
    Return Morton (Z-order) codes for active voxels. Uses xyz bit interleaving.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.
            ``None`` defaults to ``-min(ijk)``.

    Returns:
        Morton codes for each active voxel.
    """
    grid_data = _get_grid_data(grid)
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton", offset.tolist()))


def morton_zyx(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """
    Return transposed Morton codes (zyx bit interleaving) for active voxels.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Transposed Morton codes for each active voxel.
    """
    grid_data = _get_grid_data(grid)
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "morton_zyx", offset.tolist()))


def hilbert(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """
    Return Hilbert curve codes for active voxels. Better spatial locality than
    Morton codes.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Hilbert codes for each active voxel.
    """
    grid_data = _get_grid_data(grid)
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert", offset.tolist()))


def hilbert_zyx(grid: GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> JaggedTensor:
    """
    Return transposed Hilbert curve codes (zyx ordering) for active voxels.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Transposed Hilbert codes for each active voxel.
    """
    grid_data = _get_grid_data(grid)
    if offset is None:
        offset = -torch.min(_fvdb_cpp.active_grid_coords(grid_data).jdata, dim=0).values
    elif not isinstance(offset, torch.Tensor):
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=_fvdb_cpp.serialize_encode(grid_data, "hilbert_zyx", offset.tolist()))


# ---------------------------------------------------------------------------
#  Edge network
# ---------------------------------------------------------------------------


def edge_network(grid: GridBatch, return_voxel_coordinates: bool = False) -> tuple[JaggedTensor, JaggedTensor]:
    """
    Return the edge network of the grid (pairs of adjacent voxels).

    Args:
        grid: The grid to query.
        return_voxel_coordinates: If ``True``, return voxel coordinates;
            otherwise return linear indices.

    Returns:
        A tuple of two :class:`~fvdb.JaggedTensor` objects representing the
        edge network.
    """
    grid_data = _get_grid_data(grid)
    a, b = _fvdb_cpp.grid_edge_network(grid_data, return_voxel_coordinates)
    return JaggedTensor(impl=a), JaggedTensor(impl=b)
