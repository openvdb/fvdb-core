# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for pooling and refinement operations on sparse grids."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, ValueConstraint, to_Vec3i

if TYPE_CHECKING:
    from ..grid import Grid
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
#  Batch API  (GridBatch + JaggedTensor)
# ---------------------------------------------------------------------------


def max_pool_batch(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]:
    """Max-pool voxel data on a grid batch, reducing resolution by *pool_factor*.

    Supports backpropagation.

    Args:
        grid (GridBatch): The fine grid batch.
        pool_factor (NumericMaxRank1): Pooling factor per axis, broadcastable to ``(3,)``.
        data (JaggedTensor): Per-voxel feature data on the fine grid.
        stride (NumericMaxRank1): Pooling stride per axis. Default ``0`` uses *pool_factor*.
        coarse_grid (GridBatch | None): Optional pre-computed coarse grid batch.

    Returns:
        pooled_data (JaggedTensor): Pooled feature data on the coarse grid.
        coarse_grid (GridBatch): The coarse grid batch.

    .. seealso:: :func:`max_pool_single`
    """
    from ..grid_batch import GridBatch as GB

    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data = grid.data
    if coarse_grid is not None:
        coarse_grid_data = coarse_grid.data
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)

    result = cast(
        torch.Tensor,
        _MaxPoolFn.apply(data.jdata, grid_data, coarse_grid_data, factor_list, stride_list),
    )
    coarse_gb = GB(data=coarse_grid_data)
    return coarse_gb.jagged_like(result), coarse_gb


def max_pool_single(
    grid: Grid,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: Grid | None = None,
) -> tuple[torch.Tensor, Grid]:
    """Max-pool voxel data on a single grid, reducing resolution by *pool_factor*.

    Supports backpropagation.

    Args:
        grid (Grid): The fine single grid.
        pool_factor (NumericMaxRank1): Pooling factor per axis, broadcastable to ``(3,)``.
        data (torch.Tensor): Per-voxel feature data on the fine grid.
        stride (NumericMaxRank1): Pooling stride per axis. Default ``0`` uses *pool_factor*.
        coarse_grid (Grid | None): Optional pre-computed coarse grid.

    Returns:
        pooled_data (torch.Tensor): Pooled feature data on the coarse grid.
        coarse_grid (Grid): The coarse grid.

    .. seealso:: :func:`max_pool_batch`
    """
    from ..grid import Grid as G

    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data = grid.data
    if coarse_grid is not None:
        coarse_grid_data = coarse_grid.data
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)

    data_jt = JaggedTensor(data)
    result = cast(
        torch.Tensor,
        _MaxPoolFn.apply(data_jt.jdata, grid_data, coarse_grid_data, factor_list, stride_list),
    )
    return result, G(data=coarse_grid_data)


def avg_pool_batch(
    grid: GridBatch,
    pool_factor: NumericMaxRank1,
    data: JaggedTensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]:
    """Average-pool voxel data on a grid batch, reducing resolution by *pool_factor*.

    Supports backpropagation.

    Args:
        grid (GridBatch): The fine grid batch.
        pool_factor (NumericMaxRank1): Pooling factor per axis, broadcastable to ``(3,)``.
        data (JaggedTensor): Per-voxel feature data on the fine grid.
        stride (NumericMaxRank1): Pooling stride per axis. Default ``0`` uses *pool_factor*.
        coarse_grid (GridBatch | None): Optional pre-computed coarse grid batch.

    Returns:
        pooled_data (JaggedTensor): Pooled feature data on the coarse grid.
        coarse_grid (GridBatch): The coarse grid batch.

    .. seealso:: :func:`avg_pool_single`
    """
    from ..grid_batch import GridBatch as GB

    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data = grid.data
    if coarse_grid is not None:
        coarse_grid_data = coarse_grid.data
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)

    result = cast(
        torch.Tensor,
        _AvgPoolFn.apply(data.jdata, grid_data, coarse_grid_data, factor_list, stride_list),
    )
    coarse_gb = GB(data=coarse_grid_data)
    return coarse_gb.jagged_like(result), coarse_gb


def avg_pool_single(
    grid: Grid,
    pool_factor: NumericMaxRank1,
    data: torch.Tensor,
    stride: NumericMaxRank1 = 0,
    coarse_grid: Grid | None = None,
) -> tuple[torch.Tensor, Grid]:
    """Average-pool voxel data on a single grid, reducing resolution by *pool_factor*.

    Supports backpropagation.

    Args:
        grid (Grid): The fine single grid.
        pool_factor (NumericMaxRank1): Pooling factor per axis, broadcastable to ``(3,)``.
        data (torch.Tensor): Per-voxel feature data on the fine grid.
        stride (NumericMaxRank1): Pooling stride per axis. Default ``0`` uses *pool_factor*.
        coarse_grid (Grid | None): Optional pre-computed coarse grid.

    Returns:
        pooled_data (torch.Tensor): Pooled feature data on the coarse grid.
        coarse_grid (Grid): The coarse grid.

    .. seealso:: :func:`avg_pool_batch`
    """
    from ..grid import Grid as G

    factor_list = to_Vec3i(pool_factor, value_constraint=ValueConstraint.POSITIVE).tolist()
    stride_list = to_Vec3i(stride, value_constraint=ValueConstraint.NON_NEGATIVE).tolist()

    grid_data = grid.data
    if coarse_grid is not None:
        coarse_grid_data = coarse_grid.data
    else:
        coarse_factor = [s if s > 0 else f for s, f in zip(stride_list, factor_list)]
        coarse_grid_data = _fvdb_cpp.coarsen_grid(grid_data, coarse_factor)

    data_jt = JaggedTensor(data)
    result = cast(
        torch.Tensor,
        _AvgPoolFn.apply(data_jt.jdata, grid_data, coarse_grid_data, factor_list, stride_list),
    )
    return result, G(data=coarse_grid_data)


def refine_batch(
    grid: GridBatch,
    subdiv_factor: NumericMaxRank1,
    data: JaggedTensor,
    mask: JaggedTensor | None = None,
    refined: GridBatch | None = None,
) -> tuple[JaggedTensor, GridBatch]:
    """Refine (upsample) voxel data by subdividing each voxel on a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The coarse grid batch.
        subdiv_factor (NumericMaxRank1): Subdivision factor per axis, broadcastable to ``(3,)``.
        data (JaggedTensor): Per-voxel feature data on the coarse grid.
        mask (JaggedTensor | None): Optional boolean mask selecting voxels to refine.
        refined (GridBatch | None): Optional pre-computed fine grid batch.

    Returns:
        refined_data (JaggedTensor): Refined feature data on the fine grid.
        fine_grid (GridBatch): The fine grid batch.

    .. seealso:: :func:`refine_single`
    """
    from ..grid_batch import GridBatch as GB

    factor_list = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()

    grid_data = grid.data
    if refined is not None:
        fine_grid_data = refined.data
    else:
        if mask is not None:
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list, mask._impl)
        else:
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list)

    result = cast(torch.Tensor, _RefineFn.apply(data.jdata, grid_data, fine_grid_data, factor_list))
    fine_gb = GB(data=fine_grid_data)
    return fine_gb.jagged_like(result), fine_gb


def refine_single(
    grid: Grid,
    subdiv_factor: NumericMaxRank1,
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    refined: Grid | None = None,
) -> tuple[torch.Tensor, Grid]:
    """Refine (upsample) voxel data by subdividing each voxel on a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The coarse single grid.
        subdiv_factor (NumericMaxRank1): Subdivision factor per axis, broadcastable to ``(3,)``.
        data (torch.Tensor): Per-voxel feature data on the coarse grid.
        mask (torch.Tensor | None): Optional boolean mask selecting voxels to refine.
        refined (Grid | None): Optional pre-computed fine grid.

    Returns:
        refined_data (torch.Tensor): Refined feature data on the fine grid.
        fine_grid (Grid): The fine grid.

    .. seealso:: :func:`refine_batch`
    """
    from ..grid import Grid as G

    factor_list = to_Vec3i(subdiv_factor, value_constraint=ValueConstraint.POSITIVE).tolist()

    grid_data = grid.data
    if refined is not None:
        fine_grid_data = refined.data
    else:
        if mask is not None:
            mask_jt = JaggedTensor(mask)
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list, mask_jt._impl)
        else:
            fine_grid_data = _fvdb_cpp.upsample_grid(grid_data, factor_list)

    data_jt = JaggedTensor(data)
    result = cast(torch.Tensor, _RefineFn.apply(data_jt.jdata, grid_data, fine_grid_data, factor_list))
    return result, G(data=fine_grid_data)
