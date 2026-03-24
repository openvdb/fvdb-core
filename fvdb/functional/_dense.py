# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense <-> sparse grid data transfer and grid-to-grid injection."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, NumericMaxRank2, ValueConstraint, to_Vec3i, to_Vec3iBatchBroadcastable, to_Vec3iBroadcastable

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Dense -> Sparse  (inject_from_dense)
# ---------------------------------------------------------------------------


@overload
def inject_from_dense_cminor(grid: Grid, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor: ...


@overload
def inject_from_dense_cminor(
    grid: GridBatch, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0
) -> JaggedTensor: ...


def inject_from_dense_cminor(
    grid: Grid | GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> torch.Tensor | JaggedTensor:
    """
    Inject values from a dense tensor (XYZC order) into sparse voxel data.

    For :class:`~fvdb.Grid`, ``dense_data`` has shape ``(X, Y, Z, C*)``.
    For :class:`~fvdb.GridBatch`, ``dense_data`` has shape ``(B, X, Y, Z, C*)``.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        dense_data: Dense tensor to read from.
        dense_origin: Origin of the dense tensor in voxel space, broadcastable to ``(3,)``.

    Returns:
        Sparse data at active voxel locations.

    .. seealso:: :func:`inject_from_dense_cmajor`, :func:`inject_to_dense_cminor`
    """
    from ..grid import Grid

    origin = to_Vec3i(dense_origin)
    if isinstance(grid, Grid):
        return grid._impl.read_from_dense_cminor(dense_data.unsqueeze(0), origin).jdata
    return JaggedTensor(impl=grid._impl.read_from_dense_cminor(dense_data, origin))


@overload
def inject_from_dense_cmajor(grid: Grid, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0) -> torch.Tensor: ...


@overload
def inject_from_dense_cmajor(
    grid: GridBatch, dense_data: torch.Tensor, dense_origin: NumericMaxRank1 = 0
) -> JaggedTensor: ...


def inject_from_dense_cmajor(
    grid: Grid | GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> torch.Tensor | JaggedTensor:
    """
    Inject values from a dense tensor (CXYZ order) into sparse voxel data.

    For :class:`~fvdb.Grid`, ``dense_data`` has shape ``(C*, X, Y, Z)``.
    For :class:`~fvdb.GridBatch`, ``dense_data`` has shape ``(B, C*, X, Y, Z)``.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        dense_data: Dense tensor to read from.
        dense_origin: Origin of the dense tensor in voxel space, broadcastable to ``(3,)``.

    Returns:
        Sparse data at active voxel locations.

    .. seealso:: :func:`inject_from_dense_cminor`, :func:`inject_to_dense_cmajor`
    """
    from ..grid import Grid

    origin = to_Vec3i(dense_origin)
    if isinstance(grid, Grid):
        return grid._impl.read_from_dense_cmajor(dense_data.unsqueeze(0), origin).jdata
    return JaggedTensor(impl=grid._impl.read_from_dense_cmajor(dense_data, origin))


# ---------------------------------------------------------------------------
#  Sparse -> Dense  (inject_to_dense)
# ---------------------------------------------------------------------------


@overload
def inject_to_dense_cminor(
    grid: Grid,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


@overload
def inject_to_dense_cminor(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


def inject_to_dense_cminor(
    grid: Grid | GridBatch,
    sparse_data: torch.Tensor | JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """
    Write sparse voxel data into a dense tensor (XYZC order).

    Voxels not present in the grid are filled with zeros.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        sparse_data: Sparse voxel data to write.
        min_coord: Minimum voxel coordinate for the dense tensor origin.
            ``None`` defaults to the grid's bounding box minimum.
        grid_size: Size of the output dense tensor. ``None`` computes it to fit all active voxels.

    Returns:
        Dense tensor. For :class:`~fvdb.Grid`: shape ``(X, Y, Z, C*)``.
        For :class:`~fvdb.GridBatch`: shape ``(B, X, Y, Z, C*)``.

    .. seealso:: :func:`inject_to_dense_cmajor`, :func:`inject_from_dense_cminor`
    """
    from ..grid import Grid

    gs = to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE) if grid_size is not None else None

    if isinstance(grid, Grid):
        jt_sd = JaggedTensor(sparse_data)
        mc = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        return grid._impl.write_to_dense_cminor(jt_sd._impl, mc, gs).squeeze(0)
    mc = to_Vec3iBatchBroadcastable(min_coord) if min_coord is not None else None
    return grid._impl.write_to_dense_cminor(sparse_data._impl, mc, gs)


@overload
def inject_to_dense_cmajor(
    grid: Grid,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


@overload
def inject_to_dense_cmajor(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


def inject_to_dense_cmajor(
    grid: Grid | GridBatch,
    sparse_data: torch.Tensor | JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """
    Write sparse voxel data into a dense tensor (CXYZ order).

    Voxels not present in the grid are filled with zeros.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        sparse_data: Sparse voxel data to write.
        min_coord: Minimum voxel coordinate for the dense tensor origin.
        grid_size: Size of the output dense tensor.

    Returns:
        Dense tensor. For :class:`~fvdb.Grid`: shape ``(C*, X, Y, Z)``.
        For :class:`~fvdb.GridBatch`: shape ``(B, C*, X, Y, Z)``.

    .. seealso:: :func:`inject_to_dense_cminor`, :func:`inject_from_dense_cmajor`
    """
    from ..grid import Grid

    gs = to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE) if grid_size is not None else None

    if isinstance(grid, Grid):
        jt_sd = JaggedTensor(sparse_data)
        mc = to_Vec3iBroadcastable(min_coord) if min_coord is not None else None
        return grid._impl.write_to_dense_cmajor(jt_sd._impl, mc, gs).squeeze(0)
    mc = to_Vec3iBatchBroadcastable(min_coord) if min_coord is not None else None
    return grid._impl.write_to_dense_cmajor(sparse_data._impl, mc, gs)


# ---------------------------------------------------------------------------
#  Grid-to-grid injection
# ---------------------------------------------------------------------------


@overload
def inject(
    dst_grid: Grid,
    src_grid: Grid,
    src: torch.Tensor,
    dst: torch.Tensor | None = None,
    default_value: float | int | bool = 0,
) -> torch.Tensor: ...


@overload
def inject(
    dst_grid: GridBatch,
    src_grid: GridBatch,
    src: JaggedTensor,
    dst: JaggedTensor | None = None,
    default_value: float | int | bool = 0,
) -> JaggedTensor: ...


def inject(
    dst_grid: Grid | GridBatch,
    src_grid: Grid | GridBatch,
    src: torch.Tensor | JaggedTensor,
    dst: torch.Tensor | JaggedTensor | None = None,
    default_value: float | int | bool = 0,
) -> torch.Tensor | JaggedTensor:
    """
    Inject data from ``src_grid`` into ``dst_grid`` in voxel space.

    Copies sidecar data for voxels shared between the two grids. If ``dst`` is
    ``None``, a new tensor/JaggedTensor filled with ``default_value`` is created.
    If ``dst`` is provided it is modified in-place.

    This function supports backpropagation.

    Args:
        dst_grid: Destination grid.
        src_grid: Source grid.
        src: Source data associated with ``src_grid``.
        dst: Optional destination data (modified in-place). ``None`` allocates a new tensor.
        default_value: Fill value for voxels without corresponding source data.

    Returns:
        The destination data after injection.
    """
    from ..grid import Grid

    if isinstance(dst_grid, Grid):
        jt_src = JaggedTensor(src)
        if dst is None:
            eshape = list(src.shape[1:]) if src.dim() > 1 else []
            dst_shape = [dst_grid.num_voxels] + eshape
            raw_dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
        else:
            raw_dst = dst
        jt_dst = JaggedTensor(raw_dst)
        src_grid._impl.inject_to(dst_grid._impl, jt_src._impl, jt_dst._impl)
        return jt_dst.jdata

    jt_src = src
    if dst is None:
        dst_shape_list: list[int] = [dst_grid.total_voxels]
        dst_shape_list.extend(src.eshape)
        raw_dst_t = torch.full(dst_shape_list, fill_value=default_value, dtype=src.dtype, device=src.device)
        jt_dst_b = dst_grid.jagged_like(raw_dst_t)
    else:
        jt_dst_b = dst
    if jt_dst_b.eshape != jt_src.eshape:
        raise ValueError(f"src and dst must have the same element shape, got src: {jt_src.eshape}, dst: {jt_dst_b.eshape}")
    src_grid._impl.inject_to(dst_grid._impl, jt_src._impl, jt_dst_b._impl)
    return jt_dst_b
