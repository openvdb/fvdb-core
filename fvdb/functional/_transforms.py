# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for coordinate transforms between voxel and world space."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


@overload
def voxel_to_world(grid: Grid, ijk: torch.Tensor) -> torch.Tensor: ...


@overload
def voxel_to_world(grid: GridBatch, ijk: JaggedTensor) -> JaggedTensor: ...


def voxel_to_world(
    grid: Grid | GridBatch,
    ijk: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Transform voxel-space coordinates to world-space positions using the grid's
    origin and voxel size.

    This function supports backpropagation.

    Args:
        grid: The grid whose transform to use.
        ijk: Voxel-space coordinates (can be fractional).
            For :class:`~fvdb.Grid`: shape ``(N, 3)``.
            For :class:`~fvdb.GridBatch`: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.

    Returns:
        World-space coordinates, same type as ``ijk``.

    .. seealso:: :func:`world_to_voxel`
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_ijk = JaggedTensor(ijk)
        return grid._impl.grid_to_world(jt_ijk._impl).jdata
    return JaggedTensor(impl=grid._impl.grid_to_world(ijk._impl))


@overload
def world_to_voxel(grid: Grid, points: torch.Tensor) -> torch.Tensor: ...


@overload
def world_to_voxel(grid: GridBatch, points: JaggedTensor) -> JaggedTensor: ...


def world_to_voxel(
    grid: Grid | GridBatch,
    points: torch.Tensor | JaggedTensor,
) -> torch.Tensor | JaggedTensor:
    """
    Convert world-space coordinates to voxel-space coordinates using the grid's
    transform. Result can contain fractional values.

    This function supports backpropagation.

    Args:
        grid: The grid whose transform to use.
        points: World-space positions.
            For :class:`~fvdb.Grid`: shape ``(N, 3)``.
            For :class:`~fvdb.GridBatch`: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.

    Returns:
        Voxel-space coordinates, same type as ``points``.

    .. seealso:: :func:`voxel_to_world`
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        return grid._impl.world_to_grid(jt_pts._impl).jdata
    return JaggedTensor(impl=grid._impl.world_to_grid(points._impl))
