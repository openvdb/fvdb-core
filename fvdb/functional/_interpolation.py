# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse grid interpolation: sampling and splatting."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ._dispatch import _prepare_args

if TYPE_CHECKING:
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
    def backward(ctx, grad_output):
        grad_data = _fvdb_cpp.splat_trilinear(ctx.grid_data, ctx.pts_impl, grad_output)
        return grad_data, None, None


class _SplatTrilinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.splat_trilinear(grid_data, pts_impl, points_data)

    @staticmethod
    def backward(ctx, grad_output):
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
    def backward(ctx, grad_output):
        grad_data = _fvdb_cpp.splat_bezier(ctx.grid_data, ctx.pts_impl, grad_output)
        return grad_data, None, None


class _SplatBezierFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_data, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.splat_bezier(grid_data, pts_impl, points_data)

    @staticmethod
    def backward(ctx, grad_output):
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
    def backward(ctx, grad_features, grad_grad_features):
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
    def backward(ctx, grad_features, grad_grad_features):
        (voxel_data,) = ctx.saved_tensors
        grad_data = _fvdb_cpp.sample_bezier_with_grad_bwd(
            ctx.grid_data, ctx.pts_impl, grad_features, grad_grad_features, voxel_data
        )
        return grad_data, None, None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


@overload
def sample_trilinear(grid: GridBatch, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor: ...


@overload
def sample_trilinear(grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor: ...


def sample_trilinear(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    voxel_data: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Sample data associated with ``grid`` at world-space ``points`` using trilinear interpolation.

    Interpolates data values at arbitrary continuous positions in world space,
    based on values defined at voxel centers. Samples outside the grid return zero.

    This function supports backpropagation through the interpolation operation.

    Args:
        grid: The grid structure to sample from.
        points: World-space points to sample at.
            For a single grid: shape ``(num_points, 3)``.
            For a batch: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.
        voxel_data: Data associated with each voxel.
            For a single grid: shape ``(total_voxels, channels*)``.
            For a batch: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, channels*)``.

    Returns:
        Interpolated data at each point, same type and batch structure as ``points``.

    .. seealso:: :func:`sample_bezier`, :func:`sample_trilinear_with_grad`
    """
    grid_data, (points, voxel_data), unwrap = _prepare_args(grid, points, voxel_data)
    result = _SampleTrilinearFn.apply(voxel_data.jdata, grid_data, points._impl)
    return unwrap(result)


@overload
def sample_trilinear_with_grad(
    grid: GridBatch, points: torch.Tensor, voxel_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def sample_trilinear_with_grad(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor
) -> tuple[JaggedTensor, JaggedTensor]: ...


def sample_trilinear_with_grad(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    voxel_data: torch.Tensor | JaggedTensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[JaggedTensor, JaggedTensor]:
    """
    Sample data using trilinear interpolation, also returning spatial gradients.

    Returns both the interpolated data and the gradients of the interpolated data
    with respect to the world coordinates at each sample point.

    This function supports backpropagation.

    Args:
        grid: The grid structure to sample from.
        points: World-space points to sample at.
        voxel_data: Data associated with each voxel.

    Returns:
        A tuple ``(interpolated_data, interpolation_gradients)``.
        Gradient shape has an extra ``3`` dimension for the spatial gradient.

    .. seealso:: :func:`sample_trilinear`, :func:`sample_bezier_with_grad`
    """
    grid_data, (points, voxel_data), unwrap = _prepare_args(grid, points, voxel_data)
    rd, rg = _SampleTrilinearWithGradFn.apply(voxel_data.jdata, grid_data, points._impl)
    return unwrap(rd), unwrap(rg)


@overload
def sample_bezier(grid: GridBatch, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor: ...


@overload
def sample_bezier(grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor: ...


def sample_bezier(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    voxel_data: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Sample data associated with ``grid`` at world-space ``points`` using Bezier interpolation.

    Uses cubic Bezier interpolation to interpolate data values at arbitrary continuous
    positions in world space. Samples outside the grid return zero.

    This function supports backpropagation.

    Args:
        grid: The grid structure to sample from.
        points: World-space points to sample at.
        voxel_data: Data associated with each voxel.

    Returns:
        Interpolated data at each point, same type and batch structure as ``points``.

    .. seealso:: :func:`sample_trilinear`, :func:`sample_bezier_with_grad`
    """
    grid_data, (points, voxel_data), unwrap = _prepare_args(grid, points, voxel_data)
    result = _SampleBezierFn.apply(voxel_data.jdata, grid_data, points._impl)
    return unwrap(result)


@overload
def sample_bezier_with_grad(
    grid: GridBatch, points: torch.Tensor, voxel_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def sample_bezier_with_grad(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor
) -> tuple[JaggedTensor, JaggedTensor]: ...


def sample_bezier_with_grad(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    voxel_data: torch.Tensor | JaggedTensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[JaggedTensor, JaggedTensor]:
    """
    Sample data using Bezier interpolation, also returning spatial gradients.

    This function supports backpropagation.

    Args:
        grid: The grid structure to sample from.
        points: World-space points to sample at.
        voxel_data: Data associated with each voxel.

    Returns:
        A tuple ``(interpolated_data, interpolation_gradients)``.

    .. seealso:: :func:`sample_bezier`, :func:`sample_trilinear_with_grad`
    """
    grid_data, (points, voxel_data), unwrap = _prepare_args(grid, points, voxel_data)
    rd, rg = _SampleBezierWithGradFn.apply(voxel_data.jdata, grid_data, points._impl)
    return unwrap(rd), unwrap(rg)


@overload
def splat_trilinear(grid: GridBatch, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor: ...


@overload
def splat_trilinear(grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor: ...


def splat_trilinear(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    points_data: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Splat point data into voxels using trilinear interpolation weights.

    Each point distributes its data to the surrounding voxels using trilinear
    interpolation weights. This is the adjoint of :func:`sample_trilinear`.

    This function supports backpropagation.

    Args:
        grid: The grid structure to splat into.
        points: World-space positions of points. Same type conventions as :func:`sample_trilinear`.
        points_data: Data associated with each point.

    Returns:
        Accumulated features at each voxel after splatting.

    .. seealso:: :func:`splat_bezier`, :func:`sample_trilinear`
    """
    is_flat = isinstance(points, torch.Tensor)
    grid_data, (points, points_data), unwrap = _prepare_args(grid, points, points_data)
    result = _SplatTrilinearFn.apply(points_data.jdata, grid_data, points._impl)
    if is_flat:
        return result
    return grid.jagged_like(result)


@overload
def splat_bezier(grid: GridBatch, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor: ...


@overload
def splat_bezier(grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor: ...


def splat_bezier(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
    points_data: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Splat point data into voxels using cubic Bezier interpolation weights.

    This is the adjoint of :func:`sample_bezier`.

    This function supports backpropagation.

    Args:
        grid: The grid structure to splat into.
        points: World-space positions of points.
        points_data: Data associated with each point.

    Returns:
        Accumulated features at each voxel after splatting.

    .. seealso:: :func:`splat_trilinear`, :func:`sample_bezier`
    """
    is_flat = isinstance(points, torch.Tensor)
    grid_data, (points, points_data), unwrap = _prepare_args(grid, points, points_data)
    result = _SplatBezierFn.apply(points_data.jdata, grid_data, points._impl)
    if is_flat:
        return result
    return grid.jagged_like(result)
