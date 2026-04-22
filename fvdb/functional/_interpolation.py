# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse grid interpolation: sampling and splatting."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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


class _SampleNearestFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl):
        result_list = _fvdb_cpp.sample_nearest(grid_data, pts_impl, voxel_data)
        sampled_values = result_list[0]
        selected_indices = result_list[1]
        ctx.save_for_backward(selected_indices)
        ctx.voxel_shape = voxel_data.shape
        ctx.dtype = voxel_data.dtype
        ctx.device = voxel_data.device
        return sampled_values

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        (selected_indices,) = ctx.saved_tensors
        grad_voxel_data = torch.zeros(ctx.voxel_shape, dtype=ctx.dtype, device=ctx.device)
        valid = selected_indices >= 0
        if valid.any():
            grad_voxel_data.index_add_(0, selected_indices[valid], grad_output[valid])
        return grad_voxel_data, None, None


# ---------------------------------------------------------------------------
#  Public API -- sample_nearest
# ---------------------------------------------------------------------------


def sample_nearest_batch(
    grid: GridBatch,
    points: JaggedTensor,
    voxel_data: JaggedTensor,
) -> JaggedTensor:
    """Sample voxel data at world-space points using nearest-neighbor lookup for a grid batch.

    For each query point the 8 nearest voxel centers are checked and the value of the closest active one is returned.
    When two or more active corners are equidistant, the corner encountered first in the stencil's zig-zag
    traversal order wins (the same ordering used by ``sample_trilinear``).
    Points where none of the 8 surrounding voxel centers are active return zero.

    Supports backpropagation w.r.t. ``voxel_data``.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space query points, shape ``(B, -1, 3)``.
        voxel_data (JaggedTensor): Per-voxel feature data.

    Returns:
        result (JaggedTensor): Sampled values at each query point.

    .. seealso:: :func:`sample_nearest_single`
    """
    result = cast(torch.Tensor, _SampleNearestFn.apply(voxel_data.jdata, grid.data, points._impl))
    return points.jagged_like(result)


def sample_nearest_single(
    grid: Grid,
    points: torch.Tensor,
    voxel_data: torch.Tensor,
) -> torch.Tensor:
    """Sample voxel data at world-space points using nearest-neighbor lookup for a single grid.

    For each query point the 8 nearest voxel centers are checked and the value of the closest active one is returned.
    When two or more active corners are equidistant, the corner encountered first in the stencil's zig-zag
    traversal order wins (the same ordering used by ``sample_trilinear``).
    Points where none of the 8 surrounding voxel centers are active return zero.

    Supports backpropagation w.r.t. ``voxel_data``.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space query points, shape ``(N, 3)``.
        voxel_data (torch.Tensor): Per-voxel feature data.

    Returns:
        result (torch.Tensor): Sampled values at each query point.

    .. seealso:: :func:`sample_nearest_batch`
    """
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(torch.Tensor, _SampleNearestFn.apply(vd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- sample_trilinear
# ---------------------------------------------------------------------------


def sample_trilinear_batch(
    grid: GridBatch,
    points: JaggedTensor,
    voxel_data: JaggedTensor,
) -> JaggedTensor:
    """Sample voxel data at world-space points using trilinear interpolation for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space query points, shape ``(B, -1, 3)``.
        voxel_data (JaggedTensor): Per-voxel feature data.

    Returns:
        result (JaggedTensor): Interpolated values at each query point.

    .. seealso:: :func:`sample_trilinear_single`
    """
    result = cast(torch.Tensor, _SampleTrilinearFn.apply(voxel_data.jdata, grid.data, points._impl))
    return points.jagged_like(result)


def sample_trilinear_single(
    grid: Grid,
    points: torch.Tensor,
    voxel_data: torch.Tensor,
) -> torch.Tensor:
    """Sample voxel data at world-space points using trilinear interpolation for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space query points, shape ``(N, 3)``.
        voxel_data (torch.Tensor): Per-voxel feature data.

    Returns:
        result (torch.Tensor): Interpolated values at each query point.

    .. seealso:: :func:`sample_trilinear_batch`
    """
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(torch.Tensor, _SampleTrilinearFn.apply(vd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- sample_trilinear_with_grad
# ---------------------------------------------------------------------------


def sample_trilinear_with_grad_batch(
    grid: GridBatch,
    points: JaggedTensor,
    voxel_data: JaggedTensor,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Sample with trilinear interpolation and return spatial gradients for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space query points, shape ``(B, -1, 3)``.
        voxel_data (JaggedTensor): Per-voxel feature data.

    Returns:
        values (JaggedTensor): Interpolated values at each query point.
        gradients (JaggedTensor): Spatial gradients at each query point.

    .. seealso:: :func:`sample_trilinear_with_grad_single`
    """
    rd, rg = cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleTrilinearWithGradFn.apply(voxel_data.jdata, grid.data, points._impl),
    )
    return points.jagged_like(rd), points.jagged_like(rg)


def sample_trilinear_with_grad_single(
    grid: Grid,
    points: torch.Tensor,
    voxel_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with trilinear interpolation and return spatial gradients for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space query points, shape ``(N, 3)``.
        voxel_data (torch.Tensor): Per-voxel feature data.

    Returns:
        values (torch.Tensor): Interpolated values at each query point.
        gradients (torch.Tensor): Spatial gradients at each query point.

    .. seealso:: :func:`sample_trilinear_with_grad_batch`
    """
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
    grid: GridBatch,
    points: JaggedTensor,
    voxel_data: JaggedTensor,
) -> JaggedTensor:
    """Sample voxel data at world-space points using Bezier interpolation for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space query points, shape ``(B, -1, 3)``.
        voxel_data (JaggedTensor): Per-voxel feature data.

    Returns:
        result (JaggedTensor): Interpolated values at each query point.

    .. seealso:: :func:`sample_bezier_single`
    """
    result = cast(torch.Tensor, _SampleBezierFn.apply(voxel_data.jdata, grid.data, points._impl))
    return points.jagged_like(result)


def sample_bezier_single(
    grid: Grid,
    points: torch.Tensor,
    voxel_data: torch.Tensor,
) -> torch.Tensor:
    """Sample voxel data at world-space points using Bezier interpolation for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space query points, shape ``(N, 3)``.
        voxel_data (torch.Tensor): Per-voxel feature data.

    Returns:
        result (torch.Tensor): Interpolated values at each query point.

    .. seealso:: :func:`sample_bezier_batch`
    """
    pts_jt = JaggedTensor(points)
    vd_jt = JaggedTensor(voxel_data)
    return cast(torch.Tensor, _SampleBezierFn.apply(vd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- sample_bezier_with_grad
# ---------------------------------------------------------------------------


def sample_bezier_with_grad_batch(
    grid: GridBatch,
    points: JaggedTensor,
    voxel_data: JaggedTensor,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Sample with Bezier interpolation and return spatial gradients for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space query points, shape ``(B, -1, 3)``.
        voxel_data (JaggedTensor): Per-voxel feature data.

    Returns:
        values (JaggedTensor): Interpolated values at each query point.
        gradients (JaggedTensor): Spatial gradients at each query point.

    .. seealso:: :func:`sample_bezier_with_grad_single`
    """
    rd, rg = cast(
        tuple[torch.Tensor, torch.Tensor],
        _SampleBezierWithGradFn.apply(voxel_data.jdata, grid.data, points._impl),
    )
    return points.jagged_like(rd), points.jagged_like(rg)


def sample_bezier_with_grad_single(
    grid: Grid,
    points: torch.Tensor,
    voxel_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with Bezier interpolation and return spatial gradients for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space query points, shape ``(N, 3)``.
        voxel_data (torch.Tensor): Per-voxel feature data.

    Returns:
        values (torch.Tensor): Interpolated values at each query point.
        gradients (torch.Tensor): Spatial gradients at each query point.

    .. seealso:: :func:`sample_bezier_with_grad_batch`
    """
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
    grid: GridBatch,
    points: JaggedTensor,
    points_data: JaggedTensor,
) -> JaggedTensor:
    """Splat point data into voxels using trilinear weights for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space point positions, shape ``(B, -1, 3)``.
        points_data (JaggedTensor): Per-point feature data to splat.

    Returns:
        result (JaggedTensor): Accumulated voxel data.

    .. seealso:: :func:`splat_trilinear_single`
    """
    result = cast(torch.Tensor, _SplatTrilinearFn.apply(points_data.jdata, grid.data, points._impl))
    return JaggedTensor(impl=grid.data.jagged_like(result))


def splat_trilinear_single(
    grid: Grid,
    points: torch.Tensor,
    points_data: torch.Tensor,
) -> torch.Tensor:
    """Splat point data into voxels using trilinear weights for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space point positions, shape ``(N, 3)``.
        points_data (torch.Tensor): Per-point feature data to splat.

    Returns:
        result (torch.Tensor): Accumulated voxel data.

    .. seealso:: :func:`splat_trilinear_batch`
    """
    pts_jt = JaggedTensor(points)
    pd_jt = JaggedTensor(points_data)
    return cast(torch.Tensor, _SplatTrilinearFn.apply(pd_jt.jdata, grid.data, pts_jt._impl))


# ---------------------------------------------------------------------------
#  Public API -- splat_bezier
# ---------------------------------------------------------------------------


def splat_bezier_batch(
    grid: GridBatch,
    points: JaggedTensor,
    points_data: JaggedTensor,
) -> JaggedTensor:
    """Splat point data into voxels using Bezier weights for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        points (JaggedTensor): World-space point positions, shape ``(B, -1, 3)``.
        points_data (JaggedTensor): Per-point feature data to splat.

    Returns:
        result (JaggedTensor): Accumulated voxel data.

    .. seealso:: :func:`splat_bezier_single`
    """
    result = cast(torch.Tensor, _SplatBezierFn.apply(points_data.jdata, grid.data, points._impl))
    return JaggedTensor(impl=grid.data.jagged_like(result))


def splat_bezier_single(
    grid: Grid,
    points: torch.Tensor,
    points_data: torch.Tensor,
) -> torch.Tensor:
    """Splat point data into voxels using Bezier weights for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        points (torch.Tensor): World-space point positions, shape ``(N, 3)``.
        points_data (torch.Tensor): Per-point feature data to splat.

    Returns:
        result (torch.Tensor): Accumulated voxel data.

    .. seealso:: :func:`splat_bezier_batch`
    """
    pts_jt = JaggedTensor(points)
    pd_jt = JaggedTensor(points_data)
    return cast(torch.Tensor, _SplatBezierFn.apply(pd_jt.jdata, grid.data, pts_jt._impl))
