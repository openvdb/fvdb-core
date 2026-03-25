# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for pooling and refinement operations on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import torch

from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, ValueConstraint, to_Vec3iBroadcastable
from ._dispatch import _prepare_args

if TYPE_CHECKING:
    from .._fvdb_cpp import GridBatch as GridBatchCpp
    from ..grid_batch import GridBatch


@overload
def max_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch]: ...


@overload
def max_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]: ...


def max_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor | JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch] | tuple[JaggedTensor, GridBatch]:
    """
    Apply max pooling to voxel data, reducing resolution by ``pool_factor``.

    Each output voxel contains the maximum of the corresponding input voxels
    within the pooling window. If ``coarse_grid`` is ``None``, a new coarse grid
    is created with voxel sizes multiplied by ``pool_factor``.

    This function supports backpropagation.

    Args:
        grid: The fine grid whose data to pool.
        pool_factor: Downsample factor, broadcastable to ``(3,)``, integer dtype.
        data: Voxel data to pool.
        stride: Pooling stride. If ``0`` (default), equals ``pool_factor``.
        coarse_grid: Optional pre-allocated coarse grid for output.

    Returns:
        A tuple ``(pooled_data, coarse_grid)``.

    .. seealso:: :func:`avg_pool`
    """
    from ..grid_batch import GridBatch as GB

    pool_factor_t = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
    stride_t = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
    coarse_impl: GridBatchCpp | None = coarse_grid.data if coarse_grid else None

    grid_data, (data,), unwrap = _prepare_args(grid, data)
    rd, rg = grid_data.max_pool(pool_factor_t, data._impl, stride_t, coarse_impl)
    return unwrap(rd), GB(data=rg)


@overload
def avg_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch]: ...


@overload
def avg_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]: ...


def avg_pool(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor | JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch] | tuple[JaggedTensor, GridBatch]:
    """
    Apply average pooling to voxel data, reducing resolution by ``pool_factor``.

    Each output voxel contains the average of the corresponding input voxels
    within the pooling window.

    This function supports backpropagation.

    Args:
        grid: The fine grid whose data to pool.
        pool_factor: Downsample factor, broadcastable to ``(3,)``, integer dtype.
        data: Voxel data to pool.
        stride: Pooling stride. If ``0`` (default), equals ``pool_factor``.
        coarse_grid: Optional pre-allocated coarse grid for output.

    Returns:
        A tuple ``(pooled_data, coarse_grid)``.

    .. seealso:: :func:`max_pool`
    """
    from ..grid_batch import GridBatch as GB

    pool_factor_t = to_Vec3iBroadcastable(pool_factor, value_constraint=ValueConstraint.POSITIVE)
    stride_t = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.NON_NEGATIVE)
    coarse_impl: GridBatchCpp | None = coarse_grid.data if coarse_grid else None

    grid_data, (data,), unwrap = _prepare_args(grid, data)
    rd, rg = grid_data.avg_pool(pool_factor_t, data._impl, stride_t, coarse_impl)
    return unwrap(rd), GB(data=cast("GridBatchCpp", rg))


@overload
def refine(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    fine_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch]: ...


@overload
def refine(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    data: JaggedTensor,
    mask: JaggedTensor | None = None,
    fine_grid: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]: ...


def refine(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    data: torch.Tensor | JaggedTensor,
    mask: torch.Tensor | JaggedTensor | None = None,
    fine_grid: GridBatch | None = None,
) -> tuple[torch.Tensor, GridBatch] | tuple[JaggedTensor, GridBatch]:
    """
    Refine (upsample) voxel data by subdividing each voxel by ``subdiv_factor``.

    For each voxel ``(i, j, k)``, copies its data to all sub-voxels in the fine grid.
    If ``fine_grid`` is ``None``, a new fine grid is created with voxel sizes
    divided by ``subdiv_factor``.

    This function supports backpropagation.

    Args:
        grid: The coarse grid whose data to refine.
        subdiv_factor: Subdivision factor, broadcastable to ``(3,)``, integer dtype.
        data: Voxel data to refine.
        mask: Optional boolean mask selecting which voxels to refine.
        fine_grid: Optional pre-allocated fine grid for output.

    Returns:
        A tuple ``(refined_data, fine_grid)``.
    """
    from ..grid_batch import GridBatch as GB

    subdiv_t = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
    fine_impl: GridBatchCpp | None = fine_grid.data if fine_grid else None

    grid_data, (data,), unwrap = _prepare_args(grid, data)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = JaggedTensor(mask)
        mask_impl = mask._impl
    else:
        mask_impl = None
    rd, rg = grid_data.refine(subdiv_t, data._impl, mask_impl, fine_impl)
    return unwrap(rd), GB(data=rg)
