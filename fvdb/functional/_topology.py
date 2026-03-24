# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for grid topology operations (coarsen, refine, dual, dilate, etc.)."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, ValueConstraint, to_Vec3i, to_Vec3iBroadcastable

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Grid structure derivation
# ---------------------------------------------------------------------------


@overload
def coarsened_grid(grid: Grid, coarsening_factor: NumericMaxRank1) -> Grid: ...


@overload
def coarsened_grid(grid: GridBatch, coarsening_factor: NumericMaxRank1) -> GridBatch: ...


def coarsened_grid(
    grid: Grid | GridBatch,
    coarsening_factor: NumericMaxRank1,
) -> Grid | GridBatch:
    """
    Return a coarsened version of the grid, keeping only voxels whose
    coordinates are divisible by ``coarsening_factor``.

    Args:
        grid: The grid to coarsen.
        coarsening_factor: Factor per axis, broadcastable to ``(3,)``, integer dtype.

    Returns:
        A new grid with coarsened structure, same type as ``grid``.
    """
    from ..grid import Grid

    cf = to_Vec3iBroadcastable(coarsening_factor, value_constraint=ValueConstraint.POSITIVE)
    impl = grid._impl.coarsened_grid(cf)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


@overload
def refined_grid(grid: Grid, subdiv_factor: NumericMaxRank1, mask: torch.Tensor | None = None) -> Grid: ...


@overload
def refined_grid(grid: GridBatch, subdiv_factor: NumericMaxRank1, mask: JaggedTensor | None = None) -> GridBatch: ...


def refined_grid(
    grid: Grid | GridBatch,
    subdiv_factor: NumericMaxRank1,
    mask: torch.Tensor | JaggedTensor | None = None,
) -> Grid | GridBatch:
    """
    Return a refined (subdivided) version of the grid. Each voxel is
    subdivided by ``subdiv_factor``. An optional boolean ``mask`` selects which
    voxels to refine.

    Args:
        grid: The grid to refine.
        subdiv_factor: Factor per axis, broadcastable to ``(3,)``, integer dtype.
        mask: Optional boolean mask selecting voxels to refine.

    Returns:
        A new grid with refined structure, same type as ``grid``.

    .. seealso:: :func:`~fvdb.functional.refine` in ``_pooling`` for refining *data*.
    """
    from ..grid import Grid

    sf = to_Vec3iBroadcastable(subdiv_factor, value_constraint=ValueConstraint.POSITIVE)
    if isinstance(grid, Grid):
        m = JaggedTensor(mask)._impl if mask is not None else None
        return Grid(impl=grid._impl.refined_grid(sf, mask=m))
    m = mask._impl if mask is not None else None
    from ..grid_batch import GridBatch as GB

    return GB(impl=grid._impl.refined_grid(sf, m))


@overload
def dual_grid(grid: Grid, exclude_border: bool = False) -> Grid: ...


@overload
def dual_grid(grid: GridBatch, exclude_border: bool = False) -> GridBatch: ...


def dual_grid(grid: Grid | GridBatch, exclude_border: bool = False) -> Grid | GridBatch:
    """
    Return the dual grid whose voxel centers correspond to corners of the
    primal grid.

    Args:
        grid: The primal grid.
        exclude_border: Exclude border voxels that extend beyond primal bounds.

    Returns:
        A new dual grid, same type as ``grid``.
    """
    from ..grid import Grid

    impl = grid._impl.dual_grid(exclude_border)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


@overload
def dilated_grid(grid: Grid, dilation: int) -> Grid: ...


@overload
def dilated_grid(grid: GridBatch, dilation: int) -> GridBatch: ...


def dilated_grid(grid: Grid | GridBatch, dilation: int) -> Grid | GridBatch:
    """
    Return a grid dilated by ``dilation`` voxels.

    Args:
        grid: The grid to dilate.
        dilation: Dilation radius in voxels.

    Returns:
        A new dilated grid, same type as ``grid``.
    """
    from ..grid import Grid

    impl = grid._impl.dilated_grid(dilation)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


@overload
def merged_grid(grid: Grid, other: Grid) -> Grid: ...


@overload
def merged_grid(grid: GridBatch, other: GridBatch) -> GridBatch: ...


def merged_grid(grid: Grid | GridBatch, other: Grid | GridBatch) -> Grid | GridBatch:
    """
    Return the union of two grids (merged active voxels).

    Args:
        grid: First grid.
        other: Second grid to merge with.

    Returns:
        A new grid containing the union of active voxels, same type as ``grid``.
    """
    from ..grid import Grid

    impl = grid._impl.merged_grid(other._impl)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


@overload
def pruned_grid(grid: Grid, mask: torch.Tensor) -> Grid: ...


@overload
def pruned_grid(grid: GridBatch, mask: JaggedTensor) -> GridBatch: ...


def pruned_grid(
    grid: Grid | GridBatch,
    mask: torch.Tensor | JaggedTensor,
) -> Grid | GridBatch:
    """
    Return a grid containing only voxels where ``mask`` is ``True``.

    Args:
        grid: The grid to prune.
        mask: Boolean mask for each voxel.

    Returns:
        A new pruned grid, same type as ``grid``.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_m = JaggedTensor(mask)
        return Grid(impl=grid._impl.pruned_grid(jt_m._impl))
    from ..grid_batch import GridBatch as GB

    return GB(impl=grid._impl.pruned_grid(mask._impl))


# ---------------------------------------------------------------------------
#  Convolution output grids
# ---------------------------------------------------------------------------


@overload
def conv_grid(grid: Grid, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> Grid: ...


@overload
def conv_grid(grid: GridBatch, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> GridBatch: ...


def conv_grid(
    grid: Grid | GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid | GridBatch:
    """
    Return the grid representing active voxels at the output of a convolution
    with the given kernel size and stride.

    Args:
        grid: The input grid.
        kernel_size: Kernel size, broadcastable to ``(3,)``, integer dtype.
        stride: Stride, broadcastable to ``(3,)``, integer dtype.

    Returns:
        Output grid for the convolution, same type as ``grid``.
    """
    from ..grid import Grid

    ks = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
    st = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE)
    impl = grid._impl.conv_grid(ks, st)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


@overload
def conv_transpose_grid(grid: Grid, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> Grid: ...


@overload
def conv_transpose_grid(grid: GridBatch, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 = 1) -> GridBatch: ...


def conv_transpose_grid(
    grid: Grid | GridBatch,
    kernel_size: NumericMaxRank1,
    stride: NumericMaxRank1 = 1,
) -> Grid | GridBatch:
    """
    Return the grid representing active voxels at the output of a transposed
    convolution with the given kernel size and stride.

    Args:
        grid: The input grid.
        kernel_size: Kernel size, broadcastable to ``(3,)``, integer dtype.
        stride: Stride, broadcastable to ``(3,)``, integer dtype.

    Returns:
        Output grid for the transposed convolution, same type as ``grid``.
    """
    from ..grid import Grid

    ks = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
    st = to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE)
    impl = grid._impl.conv_transpose_grid(ks, st)
    if isinstance(grid, Grid):
        return Grid(impl=impl)
    from ..grid_batch import GridBatch as GB

    return GB(impl=impl)


# ---------------------------------------------------------------------------
#  Space-filling curves
# ---------------------------------------------------------------------------


@overload
def morton(grid: Grid, offset: torch.Tensor | None = None) -> torch.Tensor: ...


@overload
def morton(grid: GridBatch, offset: NumericMaxRank1 | None = None) -> JaggedTensor: ...


def morton(grid: Grid | GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None) -> torch.Tensor | JaggedTensor:
    """
    Return Morton (Z-order) codes for active voxels. Uses xyz bit interleaving.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.
            ``None`` defaults to ``-min(ijk)``.

    Returns:
        Morton codes for each active voxel.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        if offset is None:
            offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
        return grid._impl.morton(offset).jdata
    if offset is None:
        offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
    else:
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=grid._impl.morton(offset))


@overload
def morton_zyx(grid: Grid, offset: torch.Tensor | None = None) -> torch.Tensor: ...


@overload
def morton_zyx(grid: GridBatch, offset: NumericMaxRank1 | None = None) -> JaggedTensor: ...


def morton_zyx(
    grid: Grid | GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None
) -> torch.Tensor | JaggedTensor:
    """
    Return transposed Morton codes (zyx bit interleaving) for active voxels.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Transposed Morton codes for each active voxel.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        if offset is None:
            offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
        return grid._impl.morton_zyx(offset).jdata
    if offset is None:
        offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
    else:
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=grid._impl.morton_zyx(offset))


@overload
def hilbert(grid: Grid, offset: torch.Tensor | None = None) -> torch.Tensor: ...


@overload
def hilbert(grid: GridBatch, offset: NumericMaxRank1 | None = None) -> JaggedTensor: ...


def hilbert(
    grid: Grid | GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None
) -> torch.Tensor | JaggedTensor:
    """
    Return Hilbert curve codes for active voxels. Better spatial locality than
    Morton codes.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Hilbert codes for each active voxel.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        if offset is None:
            offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
        return grid._impl.hilbert(offset).jdata
    if offset is None:
        offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
    else:
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=grid._impl.hilbert(offset))


@overload
def hilbert_zyx(grid: Grid, offset: torch.Tensor | None = None) -> torch.Tensor: ...


@overload
def hilbert_zyx(grid: GridBatch, offset: NumericMaxRank1 | None = None) -> JaggedTensor: ...


def hilbert_zyx(
    grid: Grid | GridBatch, offset: torch.Tensor | NumericMaxRank1 | None = None
) -> torch.Tensor | JaggedTensor:
    """
    Return transposed Hilbert curve codes (zyx ordering) for active voxels.

    Args:
        grid: The grid to encode.
        offset: Offset applied to voxel coordinates before encoding.

    Returns:
        Transposed Hilbert codes for each active voxel.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        if offset is None:
            offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
        return grid._impl.hilbert_zyx(offset).jdata
    if offset is None:
        offset = -torch.min(grid._impl.ijk.jdata, dim=0).values
    else:
        offset = to_Vec3i(offset)
    return JaggedTensor(impl=grid._impl.hilbert_zyx(offset))


# ---------------------------------------------------------------------------
#  Edge network
# ---------------------------------------------------------------------------


def edge_network(grid: Grid | GridBatch, return_voxel_coordinates: bool = False) -> tuple[JaggedTensor, JaggedTensor]:
    """
    Return the edge network of the grid (pairs of adjacent voxels).

    Args:
        grid: The grid to query.
        return_voxel_coordinates: If ``True``, return voxel coordinates;
            otherwise return linear indices.

    Returns:
        A tuple of two :class:`~fvdb.JaggedTensor` objects representing the
        edge network.
    """
    a, b = grid._impl.viz_edge_network(return_voxel_coordinates)
    return JaggedTensor(impl=a), JaggedTensor(impl=b)
