# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for pooling and refinement operations on sparse grids."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, ValueConstraint, to_Vec3i
from ._dispatch import _get_grid_data, _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Autograd functions
# ---------------------------------------------------------------------------


class _MaxPoolFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, fine_grid_data, coarse_grid_data, factor_list, stride_list):
        ctx.save_for_backward(data)
        ctx.fine_grid_data = fine_grid_data
        ctx.coarse_grid_data = coarse_grid_data
        ctx.factor_list = factor_list
        ctx.stride_list = stride_list
        return _fvdb_cpp.max_pool(fine_grid_data, coarse_grid_data, data, factor_list, stride_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        (data,) = ctx.saved_tensors
        grad = _fvdb_cpp.max_pool_bwd(
            ctx.coarse_grid_data, ctx.fine_grid_data, data, grad_output, ctx.factor_list, ctx.stride_list
        )
        return grad, None, None, None, None


class _AvgPoolFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, fine_grid_data, coarse_grid_data, factor_list, stride_list):
        ctx.save_for_backward(data)
        ctx.fine_grid_data = fine_grid_data
        ctx.coarse_grid_data = coarse_grid_data
        ctx.factor_list = factor_list
        ctx.stride_list = stride_list
        return _fvdb_cpp.avg_pool(fine_grid_data, coarse_grid_data, data, factor_list, stride_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        (data,) = ctx.saved_tensors
        grad = _fvdb_cpp.avg_pool_bwd(
            ctx.coarse_grid_data, ctx.fine_grid_data, data, grad_output, ctx.factor_list, ctx.stride_list
        )
        return grad, None, None, None, None


class _RefineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, coarse_grid_data, fine_grid_data, factor_list):
        ctx.save_for_backward(data)
        ctx.coarse_grid_data = coarse_grid_data
        ctx.fine_grid_data = fine_grid_data
        ctx.factor_list = factor_list
        return _fvdb_cpp.refine(coarse_grid_data, fine_grid_data, data, factor_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        (data,) = ctx.saved_tensors
        grad = _fvdb_cpp.refine_bwd(ctx.fine_grid_data, ctx.coarse_grid_data, grad_output, data, ctx.factor_list)
        return grad, None, None, None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


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

    is_flat = isinstance(data, torch.Tensor)
    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data, (data_jt,), unwrap = _prepare_args(grid, data)
    assert data_jt is not None
    if coarse_grid is not None:
        coarse_grid_data = _get_grid_data(coarse_grid)
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)
    result = cast(torch.Tensor, _MaxPoolFn.apply(data_jt.jdata, grid_data, coarse_grid_data, factor_list, stride_list))
    coarse_gb = GB(data=coarse_grid_data)
    if is_flat:
        return result, coarse_gb
    return coarse_gb.jagged_like(result), coarse_gb


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

    is_flat = isinstance(data, torch.Tensor)
    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data, (data_jt,), unwrap = _prepare_args(grid, data)
    assert data_jt is not None
    if coarse_grid is not None:
        coarse_grid_data = _get_grid_data(coarse_grid)
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)
    result = cast(torch.Tensor, _AvgPoolFn.apply(data_jt.jdata, grid_data, coarse_grid_data, factor_list, stride_list))
    coarse_gb = GB(data=coarse_grid_data)
    if is_flat:
        return result, coarse_gb
    return coarse_gb.jagged_like(result), coarse_gb


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

    is_flat = isinstance(data, torch.Tensor)
    factor_list = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()

    grid_data, (data_jt,), unwrap = _prepare_args(grid, data)
    assert data_jt is not None
    if fine_grid is not None:
        fine_grid_data = _get_grid_data(fine_grid)
    else:
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = JaggedTensor(mask)
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list, mask._impl)
        else:
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list)
    result = cast(torch.Tensor, _RefineFn.apply(data_jt.jdata, grid_data, fine_grid_data, factor_list))
    fine_gb = GB(data=fine_grid_data)
    if is_flat:
        return result, fine_gb
    return fine_gb.jagged_like(result), fine_gb
