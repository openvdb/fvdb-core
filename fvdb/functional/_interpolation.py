# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for sparse grid interpolation: sampling and splatting."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


@overload
def sample_trilinear(grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor: ...


@overload
def sample_trilinear(grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor: ...


def sample_trilinear(
    grid: Grid | GridBatch,
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
            For :class:`~fvdb.Grid`: shape ``(num_points, 3)``.
            For :class:`~fvdb.GridBatch`: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, 3)``.
        voxel_data: Data associated with each voxel.
            For :class:`~fvdb.Grid`: shape ``(total_voxels, channels*)``.
            For :class:`~fvdb.GridBatch`: a :class:`~fvdb.JaggedTensor` with shape ``(B, -1, channels*)``.

    Returns:
        Interpolated data at each point, same type and batch structure as ``points``.

    .. seealso:: :func:`sample_bezier`, :func:`sample_trilinear_with_grad`
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(voxel_data)
        return grid.data.sample_trilinear(jt_pts._impl, jt_dat._impl).jdata
    return JaggedTensor(impl=grid.data.sample_trilinear(points._impl, voxel_data._impl))


@overload
def sample_trilinear_with_grad(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def sample_trilinear_with_grad(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor
) -> tuple[JaggedTensor, JaggedTensor]: ...


def sample_trilinear_with_grad(
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(voxel_data)
        rd, rg = grid.data.sample_trilinear_with_grad(jt_pts._impl, jt_dat._impl)
        return rd.jdata, rg.jdata
    rd, rg = grid.data.sample_trilinear_with_grad(points._impl, voxel_data._impl)
    return JaggedTensor(impl=rd), JaggedTensor(impl=rg)


@overload
def sample_bezier(grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor) -> torch.Tensor: ...


@overload
def sample_bezier(grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor) -> JaggedTensor: ...


def sample_bezier(
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(voxel_data)
        return grid.data.sample_bezier(jt_pts._impl, jt_dat._impl).jdata
    return JaggedTensor(impl=grid.data.sample_bezier(points._impl, voxel_data._impl))


@overload
def sample_bezier_with_grad(
    grid: Grid, points: torch.Tensor, voxel_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def sample_bezier_with_grad(
    grid: GridBatch, points: JaggedTensor, voxel_data: JaggedTensor
) -> tuple[JaggedTensor, JaggedTensor]: ...


def sample_bezier_with_grad(
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(voxel_data)
        rd, rg = grid.data.sample_bezier_with_grad(jt_pts._impl, jt_dat._impl)
        return rd.jdata, rg.jdata
    rd, rg = grid.data.sample_bezier_with_grad(points._impl, voxel_data._impl)
    return JaggedTensor(impl=rd), JaggedTensor(impl=rg)


@overload
def splat_trilinear(grid: Grid, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor: ...


@overload
def splat_trilinear(grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor: ...


def splat_trilinear(
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(points_data)
        return grid.data.splat_trilinear(jt_pts._impl, jt_dat._impl).jdata
    return JaggedTensor(impl=grid.data.splat_trilinear(points._impl, points_data._impl))


@overload
def splat_bezier(grid: Grid, points: torch.Tensor, points_data: torch.Tensor) -> torch.Tensor: ...


@overload
def splat_bezier(grid: GridBatch, points: JaggedTensor, points_data: JaggedTensor) -> JaggedTensor: ...


def splat_bezier(
    grid: Grid | GridBatch,
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
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_pts = JaggedTensor(points)
        jt_dat = JaggedTensor(points_data)
        return grid.data.splat_bezier(jt_pts._impl, jt_dat._impl).jdata
    return JaggedTensor(impl=grid.data.splat_bezier(points._impl, points_data._impl))
