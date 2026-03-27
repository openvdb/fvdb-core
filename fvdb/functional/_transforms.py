# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for coordinate transforms between voxel and world space."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ._dispatch import _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


class _VoxelToWorldFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_jdata, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.voxel_to_world(grid_data, pts_impl, True)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad_jt = ctx.pts_impl.jagged_like(grad_output)
        return _fvdb_cpp.voxel_to_world_bwd(ctx.grid_data, grad_jt, True), None, None


class _WorldToVoxelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_jdata, grid_data, pts_impl):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        return _fvdb_cpp.world_to_voxel(grid_data, pts_impl, True)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad_jt = ctx.pts_impl.jagged_like(grad_output)
        return _fvdb_cpp.world_to_voxel_bwd(ctx.grid_data, grad_jt, True), None, None


@overload
def voxel_to_world(grid: GridBatch, ijk: torch.Tensor) -> torch.Tensor: ...


@overload
def voxel_to_world(grid: GridBatch, ijk: JaggedTensor) -> JaggedTensor: ...


def voxel_to_world(
    grid: GridBatch,
    ijk: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Transform voxel-space coordinates to world-space positions using the grid's
    origin and voxel size.

    This function supports backpropagation.

    Args:
        grid: The grid whose transform to use.
        ijk: Voxel-space coordinates (can be fractional).
            For a single grid: shape ``(N, 3)``.
            For a batch: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.

    Returns:
        World-space coordinates, same type as ``ijk``.

    .. seealso:: :func:`world_to_voxel`
    """
    grid_data, (ijk_jt,), unwrap = _prepare_args(grid, ijk)
    assert ijk_jt is not None
    result = cast(torch.Tensor, _VoxelToWorldFn.apply(ijk_jt.jdata, grid_data, ijk_jt._impl))
    return unwrap(result)


@overload
def world_to_voxel(grid: GridBatch, points: torch.Tensor) -> torch.Tensor: ...


@overload
def world_to_voxel(grid: GridBatch, points: JaggedTensor) -> JaggedTensor: ...


def world_to_voxel(
    grid: GridBatch,
    points: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Convert world-space coordinates to voxel-space coordinates using the grid's
    transform. Result can contain fractional values.

    This function supports backpropagation.

    Args:
        grid: The grid whose transform to use.
        points: World-space positions.
            For a single grid: shape ``(N, 3)``.
            For a batch: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.

    Returns:
        Voxel-space coordinates, same type as ``points``.

    .. seealso:: :func:`voxel_to_world`
    """
    grid_data, (points_jt,), unwrap = _prepare_args(grid, points)
    assert points_jt is not None
    result = cast(torch.Tensor, _WorldToVoxelFn.apply(points_jt.jdata, grid_data, points_jt._impl))
    return unwrap(result)
