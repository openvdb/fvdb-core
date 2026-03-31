# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for coordinate transforms between voxel and world space."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
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


# ---------------------------------------------------------------------------
#  Batch variants (GridBatch + JaggedTensor)
# ---------------------------------------------------------------------------


def voxel_to_world_batch(grid: GridBatch, ijk: JaggedTensor) -> JaggedTensor:
    """Transform voxel-space coordinates to world-space positions for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch whose transforms to use.
        ijk (JaggedTensor): Voxel-space coordinates, shape ``(B, -1, 3)``.

    Returns:
        result (JaggedTensor): World-space coordinates, shape ``(B, -1, 3)``.

    .. seealso:: :func:`voxel_to_world_single`
    """
    result = cast(torch.Tensor, _VoxelToWorldFn.apply(ijk.jdata, grid.data, ijk._impl))
    return ijk.jagged_like(result)


def world_to_voxel_batch(grid: GridBatch, points: JaggedTensor) -> JaggedTensor:
    """Transform world-space coordinates to voxel-space positions for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch whose transforms to use.
        points (JaggedTensor): World-space positions, shape ``(B, -1, 3)``.

    Returns:
        result (JaggedTensor): Voxel-space coordinates, shape ``(B, -1, 3)``.

    .. seealso:: :func:`world_to_voxel_single`
    """
    result = cast(torch.Tensor, _WorldToVoxelFn.apply(points.jdata, grid.data, points._impl))
    return points.jagged_like(result)


# ---------------------------------------------------------------------------
#  Single variants (Grid + torch.Tensor)
# ---------------------------------------------------------------------------


def voxel_to_world_single(grid: Grid, ijk: torch.Tensor) -> torch.Tensor:
    """Transform voxel-space coordinates to world-space positions for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid whose transform to use.
        ijk (torch.Tensor): Voxel-space coordinates, shape ``(N, 3)``.

    Returns:
        result (torch.Tensor): World-space coordinates, shape ``(N, 3)``.

    .. seealso:: :func:`voxel_to_world_batch`
    """
    ijk_jt = JaggedTensor(ijk)
    return cast(torch.Tensor, _VoxelToWorldFn.apply(ijk_jt.jdata, grid.data, ijk_jt._impl))


def world_to_voxel_single(grid: Grid, points: torch.Tensor) -> torch.Tensor:
    """Transform world-space coordinates to voxel-space positions for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid whose transform to use.
        points (torch.Tensor): World-space positions, shape ``(N, 3)``.

    Returns:
        result (torch.Tensor): Voxel-space coordinates, shape ``(N, 3)``.

    .. seealso:: :func:`world_to_voxel_batch`
    """
    pts_jt = JaggedTensor(points)
    return cast(torch.Tensor, _WorldToVoxelFn.apply(pts_jt.jdata, grid.data, pts_jt._impl))
