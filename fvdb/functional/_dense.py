# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense <-> sparse grid data transfer and grid-to-grid injection."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import NumericMaxRank1, NumericMaxRank2, ValueConstraint, to_Vec3i, to_Vec3iBatchBroadcastable, to_Vec3iBroadcastable
from ._dispatch import _get_grid_data, _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Autograd functions
# ---------------------------------------------------------------------------


class _InjectFromDenseCminorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_data, grid_data, origins):
        ctx.grid_data = grid_data
        ctx.origins = origins
        ctx.dense_shape = dense_data.shape
        return _fvdb_cpp.inject_from_dense_cminor(grid_data, dense_data, origins)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grid_size = list(ctx.dense_shape[1:4])
        grad = _fvdb_cpp.inject_to_dense_cminor(ctx.grid_data, grad_output, ctx.origins, grid_size)
        return grad.view(ctx.dense_shape), None, None


class _InjectFromDenseCmajorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_data, grid_data, origins):
        ctx.grid_data = grid_data
        ctx.origins = origins
        ctx.dense_shape = dense_data.shape
        return _fvdb_cpp.inject_from_dense_cmajor(grid_data, dense_data, origins)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grid_size = list(ctx.dense_shape[-3:])
        grad = _fvdb_cpp.inject_to_dense_cmajor(ctx.grid_data, grad_output, ctx.origins, grid_size)
        return grad.view(ctx.dense_shape), None, None


class _InjectToDenseCminorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_data, grid_data, origins, grid_size_list):
        ctx.grid_data = grid_data
        ctx.origins = origins
        ctx.sparse_shape = sparse_data.shape
        return _fvdb_cpp.inject_to_dense_cminor(grid_data, sparse_data, origins, grid_size_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad = _fvdb_cpp.inject_from_dense_cminor(ctx.grid_data, grad_output, ctx.origins)
        return grad.view(ctx.sparse_shape), None, None, None


class _InjectToDenseCmajorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_data, grid_data, origins, grid_size_list):
        ctx.grid_data = grid_data
        ctx.origins = origins
        ctx.sparse_shape = sparse_data.shape
        return _fvdb_cpp.inject_to_dense_cmajor(grid_data, sparse_data, origins, grid_size_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        assert grad_output is not None
        grad = _fvdb_cpp.inject_from_dense_cmajor(ctx.grid_data, grad_output, ctx.origins)
        return grad.view(ctx.sparse_shape), None, None, None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _resolve_dense_params(grid_data, sparse_data, min_coord, grid_size):
    """Compute origins tensor and grid_size list for inject_to_dense ops.

    When ``min_coord`` or ``grid_size`` is ``None``, defaults are derived from
    the grid's total bounding box (mirroring the C++ autograd logic).
    """
    bbox = grid_data.total_bbox
    if min_coord is not None:
        mc = to_Vec3iBatchBroadcastable(min_coord).to(device=sparse_data.jdata.device)
        if mc.dim() == 0:
            mc = mc.unsqueeze(0).expand(3)
        if mc.dim() == 1 and mc.size(0) == 3:
            mc = mc.unsqueeze(0).expand(grid_data.grid_count, 3)
        origins = mc.to(torch.int32)
    else:
        origins = bbox[0].to(torch.int32).unsqueeze(0).expand(grid_data.grid_count, 3).to(sparse_data.jdata.device)

    if grid_size is not None:
        gs_t = to_Vec3iBroadcastable(grid_size, value_constraint=ValueConstraint.POSITIVE)
        if gs_t.dim() == 0:
            gs_t = gs_t.expand(3)
        gs_list = gs_t.tolist()
    else:
        bbox_min = bbox[0]
        bbox_max = bbox[1]
        gs_list = (bbox_max - bbox_min + 1).tolist()

    return origins, gs_list


# ---------------------------------------------------------------------------
#  Dense -> Sparse  (inject_from_dense)
# ---------------------------------------------------------------------------


def inject_from_dense_cminor(
    grid: GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> JaggedTensor:
    """
    Inject values from a dense tensor (XYZC order) into sparse voxel data.

    ``dense_data`` has shape ``(B, X, Y, Z, C*)``.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        dense_data: Dense tensor to read from.
        dense_origin: Origin of the dense tensor in voxel space, broadcastable to ``(3,)``.

    Returns:
        Sparse data at active voxel locations as a :class:`~fvdb.JaggedTensor`.

    .. seealso:: :func:`inject_from_dense_cmajor`, :func:`inject_to_dense_cminor`
    """
    grid_data = _get_grid_data(grid)
    origin = to_Vec3i(dense_origin).unsqueeze(0).expand(grid_data.grid_count, 3).to(dtype=torch.int32, device=dense_data.device).contiguous()
    result = cast(torch.Tensor, _InjectFromDenseCminorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[4:])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return grid.jagged_like(result)


def inject_from_dense_cmajor(
    grid: GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> JaggedTensor:
    """
    Inject values from a dense tensor (CXYZ order) into sparse voxel data.

    ``dense_data`` has shape ``(B, C*, X, Y, Z)``.

    This function supports backpropagation.

    Args:
        grid: The grid structure.
        dense_data: Dense tensor to read from.
        dense_origin: Origin of the dense tensor in voxel space, broadcastable to ``(3,)``.

    Returns:
        Sparse data at active voxel locations as a :class:`~fvdb.JaggedTensor`.

    .. seealso:: :func:`inject_from_dense_cminor`, :func:`inject_to_dense_cmajor`
    """
    grid_data = _get_grid_data(grid)
    origin = to_Vec3i(dense_origin).unsqueeze(0).expand(grid_data.grid_count, 3).to(dtype=torch.int32, device=dense_data.device).contiguous()
    result = cast(torch.Tensor, _InjectFromDenseCmajorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[1:-3])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return grid.jagged_like(result)


# ---------------------------------------------------------------------------
#  Sparse -> Dense  (inject_to_dense)
# ---------------------------------------------------------------------------


@overload
def inject_to_dense_cminor(
    grid: GridBatch,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


@overload
def inject_to_dense_cminor(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


def inject_to_dense_cminor(
    grid: GridBatch,
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
        Dense tensor with shape ``(B, X, Y, Z, C*)``.

    .. seealso:: :func:`inject_to_dense_cmajor`, :func:`inject_from_dense_cminor`
    """
    grid_data, (sparse_data_jt,), _ = _prepare_args(grid, sparse_data)
    assert sparse_data_jt is not None
    origins, gs_list = _resolve_dense_params(grid_data, sparse_data_jt, min_coord, grid_size)
    result = cast(torch.Tensor, _InjectToDenseCminorFn.apply(sparse_data_jt.jdata, grid_data, origins, gs_list))
    feature_shape = list(sparse_data_jt.jdata.shape[1:])
    return result.view([grid.grid_count] + gs_list + feature_shape)


@overload
def inject_to_dense_cmajor(
    grid: GridBatch,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


@overload
def inject_to_dense_cmajor(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor: ...


def inject_to_dense_cmajor(
    grid: GridBatch,
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
        Dense tensor with shape ``(B, C*, X, Y, Z)``.

    .. seealso:: :func:`inject_to_dense_cminor`, :func:`inject_from_dense_cmajor`
    """
    grid_data, (sparse_data_jt,), _ = _prepare_args(grid, sparse_data)
    assert sparse_data_jt is not None
    origins, gs_list = _resolve_dense_params(grid_data, sparse_data_jt, min_coord, grid_size)
    result = cast(torch.Tensor, _InjectToDenseCmajorFn.apply(sparse_data_jt.jdata, grid_data, origins, gs_list))
    feature_shape = list(sparse_data_jt.jdata.shape[1:])
    return result.view([grid.grid_count] + feature_shape + gs_list)


# ---------------------------------------------------------------------------
#  Grid-to-grid injection
# ---------------------------------------------------------------------------


class _InjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_jdata, dst_jdata, dst_grid_data, src_grid_data, dst_jt_impl, src_jt_impl):
        ctx.dst_grid_data = dst_grid_data
        ctx.src_grid_data = src_grid_data
        ctx.dst_jt_impl = dst_jt_impl
        ctx.src_jt_impl = src_jt_impl
        ctx.mark_dirty(dst_jdata)
        src_contig = src_jdata.contiguous()
        src_contig_jt = src_jt_impl.jagged_like(src_contig)
        _fvdb_cpp.inject_op(dst_grid_data, src_grid_data, dst_jt_impl, src_contig_jt)
        return dst_jdata

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        (grad_dst_out,) = grad_outputs
        assert grad_dst_out is not None
        grad_src = torch.zeros_like(ctx.src_jt_impl.jdata)
        grad_dst = grad_dst_out.clone().contiguous()

        grad_src_jt = ctx.src_jt_impl.jagged_like(grad_src)
        grad_dst_jt = ctx.dst_jt_impl.jagged_like(grad_dst)
        _fvdb_cpp.inject_op(ctx.src_grid_data, ctx.dst_grid_data, grad_src_jt, grad_dst_jt)

        zeros = torch.zeros([1] * grad_src.dim(), dtype=grad_src.dtype, device=grad_src.device).expand_as(grad_src)
        zeros_jt = ctx.src_jt_impl.jagged_like(zeros)
        _fvdb_cpp.inject_op(ctx.dst_grid_data, ctx.src_grid_data, grad_dst_jt, zeros_jt)

        return grad_src_jt.jdata, grad_dst_jt.jdata, None, None, None, None


@overload
def inject(
    dst_grid: GridBatch,
    src_grid: GridBatch,
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
    dst_grid: GridBatch,
    src_grid: GridBatch,
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
    is_flat = isinstance(src, torch.Tensor)
    dst_grid_data = _get_grid_data(dst_grid)
    src_grid_data = _get_grid_data(src_grid)

    if is_flat:
        assert isinstance(src, torch.Tensor)
        jt_src = JaggedTensor(src)
        if dst is None:
            eshape = list(src.shape[1:]) if src.dim() > 1 else []
            dst_shape = [dst_grid.total_voxels] + eshape
            raw_dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
        else:
            assert isinstance(dst, torch.Tensor)
            raw_dst = dst
        jt_dst = JaggedTensor(raw_dst)
        dst_out = cast(torch.Tensor, _InjectFn.apply(src, jt_dst.jdata, dst_grid_data, src_grid_data, jt_dst._impl, jt_src._impl))
        return dst_out

    assert isinstance(src, JaggedTensor)
    jt_src = src
    if dst is None:
        dst_shape_list: list[int] = [dst_grid.total_voxels]
        dst_shape_list.extend(src.eshape)
        raw_dst_t = torch.full(dst_shape_list, fill_value=default_value, dtype=src.dtype, device=src.device)
        jt_dst_b = dst_grid.jagged_like(raw_dst_t)
    else:
        assert isinstance(dst, JaggedTensor)
        jt_dst_b = dst
    if jt_dst_b.eshape != jt_src.eshape:
        raise ValueError(f"src and dst must have the same element shape, got src: {jt_src.eshape}, dst: {jt_dst_b.eshape}")
    _InjectFn.apply(jt_src.jdata, jt_dst_b.jdata, dst_grid_data, src_grid_data, jt_dst_b._impl, jt_src._impl)
    return jt_dst_b


def inject_from_ijk(
    grid: GridBatch,
    src_ijk: JaggedTensor,
    src: JaggedTensor,
    dst: JaggedTensor | None = None,
    default_value: float | int | bool = 0,
) -> JaggedTensor:
    """
    Inject data from source voxel coordinates to a sidecar for the given grid.

    This function supports backpropagation through the injection operation.

    Args:
        grid: The destination grid.
        src_ijk: Voxel coordinates from which to copy data.
            Shape: ``(B, num_src_voxels, 3)``.
        src: Source data to inject. Shape: ``(B, num_src_voxels, *)``.
        dst: Optional destination data (modified in-place). If ``None``, a new
            :class:`~fvdb.JaggedTensor` is created filled with ``default_value``.
        default_value: Fill value for voxels without source data. Default ``0``.

    Returns:
        The destination data after injection.
    """
    from . import _query

    if not isinstance(src_ijk, JaggedTensor):
        raise TypeError(f"src_ijk must be a JaggedTensor, but got {type(src_ijk)}")

    if not isinstance(src, JaggedTensor):
        raise TypeError(f"src must be a JaggedTensor, but got {type(src)}")

    grid_data = _get_grid_data(grid)
    if dst is None:
        dst_shape: list[int] = [grid.total_voxels]
        dst_shape.extend(src.eshape)
        dst = JaggedTensor(impl=grid_data.jagged_like(
            torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
        ))
    else:
        if not isinstance(dst, JaggedTensor):
            raise TypeError(f"dst must be a JaggedTensor, but got {type(dst)}")

    if dst.eshape != src.eshape:
        raise ValueError(
            f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}"
        )

    src_idx = _query.ijk_to_index(grid, src_ijk, cumulative=True).jdata
    src_mask = src_idx >= 0
    src_idx = src_idx[src_mask]
    dst.jdata[src_idx] = src.jdata[src_mask]
    return dst
