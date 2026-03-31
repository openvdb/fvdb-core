# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse grid interpolation: sampling and splatting."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Autograd functions
# ---------------------------------------------------------------------------


class _SampleTrilinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        result_list = _fvdb_cpp.sample_trilinear(grid_data, pts_impl, voxel_data)
        return result_list[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad_data = _fvdb_cpp.splat_trilinear(ctx.grid_data, ctx.pts_impl, grad_output)
        return grad_data, None, None


class _SplatTrilinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.splat_trilinear(grid_data, pts_impl, points_data)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        result_list = _fvdb_cpp.sample_trilinear(ctx.grid_data, ctx.pts_impl, grad_output)
        return result_list[0], None, None


class _SampleBezierFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        result_list = _fvdb_cpp.sample_bezier(grid_data, pts_impl, voxel_data)
        return result_list[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad_data = _fvdb_cpp.splat_bezier(ctx.grid_data, ctx.pts_impl, grad_output)
        return grad_data, None, None


class _SplatBezierFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.splat_bezier(grid_data, pts_impl, points_data)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        result_list = _fvdb_cpp.sample_bezier(ctx.grid_data, ctx.pts_impl, grad_output)
        return result_list[0], None, None


class _SampleTrilinearWithGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        ctx.save_for_backward(voxel_data)
        result_list = _fvdb_cpp.sample_trilinear_with_grad(grid_data, pts_impl, voxel_data)
        return result_list[0], result_list[1]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_features, grad_grad_features = grad_outputs
        assert grad_features is not None and grad_grad_features is not None
        (voxel_data,) = ctx.saved_tensors
        grad_data = _fvdb_cpp.sample_trilinear_with_grad_bwd(
            ctx.grid_data, ctx.pts_impl, voxel_data, grad_features, grad_grad_features
        )
        return grad_data, None, None


class _SampleBezierWithGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        ctx.save_for_backward(voxel_data)
        result_list = _fvdb_cpp.sample_bezier_with_grad(grid_data, pts_impl, voxel_data)
        return result_list[0], result_list[1]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_features, grad_grad_features = grad_outputs
        assert grad_features is not None and grad_grad_features is not None
        (voxel_data,) = ctx.saved_tensors
        grad_data = _fvdb_cpp.sample_bezier_with_grad_bwd(
            ctx.grid_data, ctx.pts_impl, grad_features, grad_grad_features, voxel_data
        )
        return grad_data, None, None


# ---------------------------------------------------------------------------
#  Public API -- sample_trilinear
# ---------------------------------------------------------------------------


def sample_trilinear_batch(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor,
) -> JaggedTensor:
    """Sample voxel data at world-space points using trilinear interpolation (batched)."""
    result = cast(torch.Tensor, _SampleTrilinearFn.apply(voxel_data.jdata, grid.data, points._impl))
    return points.jagged_like(result)


def sample_trilinear_single(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor,
) -> torch.Tensor:
    """Sample voxel data at world-space points using trilinear interpolation (single grid)."""
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(torch.Tensor, _SampleTrilinearFn.apply(vd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- sample_trilinear_with_grad
# ---------------------------------------------------------------------------


def sample_trilinear_with_grad_batch(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Sample with trilinear interpolation and return spatial gradients (batched)."""
    rd, rg = cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleTrilinearWithGradFn.apply(voxel_data.jdata, grid.data, points._impl),
    )
    return points.jagged_like(rd), points.jagged_like(rg)


def sample_trilinear_with_grad_single(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with trilinear interpolation and return spatial gradients (single grid)."""
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleTrilinearWithGradFn.apply(vd_jt.jdata, grid.data, pts_jt._impl),
    )


# ---------------------------------------------------------------------------
#  Public API -- sample_bezier
# ---------------------------------------------------------------------------


def sample_bezier_batch(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor,
) -> JaggedTensor:
    """Sample voxel data at world-space points using Bezier interpolation (batched)."""
    result = cast(torch.Tensor, _SampleBezierFn.apply(voxel_data.jdata, grid.data, points._impl))
    return points.jagged_like(result)


def sample_bezier_single(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor,
) -> torch.Tensor:
    """Sample voxel data at world-space points using Bezier interpolation (single grid)."""
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(torch.Tensor, _SampleBezierFn.apply(vd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- sample_bezier_with_grad
# ---------------------------------------------------------------------------


def sample_bezier_with_grad_batch(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Sample with Bezier interpolation and return spatial gradients (batched)."""
    rd, rg = cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleBezierWithGradFn.apply(voxel_data.jdata, grid.data, points._impl),
    )
    return points.jagged_like(rd), points.jagged_like(rg)


def sample_bezier_with_grad_single(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with Bezier interpolation and return spatial gradients (single grid)."""
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleBezierWithGradFn.apply(vd_jt.jdata, grid.data, pts_jt._impl),
    )


# ---------------------------------------------------------------------------
#  Public API -- splat_trilinear
# ---------------------------------------------------------------------------


def splat_trilinear_batch(
    grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor,
) -> JaggedTensor:
    """Splat point data into voxels using trilinear weights (batched)."""
    result = cast(torch.Tensor, _SplatTrilinearFn.apply(points_data.jdata, grid.data, points._impl))
    return JaggedTensor(impl=grid.data.jagged_like(result))


def splat_trilinear_single(
    grid: Grid, points: torch.Tensor, points_data: torch.Tensor,
) -> torch.Tensor:
    """Splat point data into voxels using trilinear weights (single grid)."""
    pts_jt = JaggedTensor(points)
    pd_jt = JaggedTensor(points_data)
    return cast(torch.Tensor, _SplatTrilinearFn.apply(pd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- splat_bezier
# ---------------------------------------------------------------------------


def splat_bezier_batch(
    grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor,
) -> JaggedTensor:
    """Splat point data into voxels using Bezier weights (batched)."""
    result = cast(torch.Tensor, _SplatBezierFn.apply(points_data.jdata, grid.data, points._impl))
    return JaggedTensor(impl=grid.data.jagged_like(result))


def splat_bezier_single(
    grid: Grid, points: torch.Tensor, points_data: torch.Tensor,
) -> torch.Tensor:
    """Splat point data into voxels using Bezier weights (single grid)."""
    pts_jt = JaggedTensor(points)
    pd_jt = JaggedTensor(points_data)
    return cast(torch.Tensor, _SplatBezierFn.apply(pd_jt.jdata, grid.data, pts_jt._impl))
