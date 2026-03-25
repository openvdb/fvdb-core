# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for coordinate transforms between voxel and world space."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor
from ._dispatch import _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


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
    grid_data, (ijk,), unwrap = _prepare_args(grid, ijk)
    return unwrap(grid_data.voxel_to_world(ijk._impl))


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
    grid_data, (points,), unwrap = _prepare_args(grid, points)
    return unwrap(grid_data.world_to_voxel(points._impl))
