# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for dense <-> sparse grid data transfer and grid-to-grid injection."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import (
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    to_Vec3i,
    to_Vec3iBatchBroadcastable,
    to_Vec3iBroadcastable,
)

if TYPE_CHECKING:
    from ..grid import Grid
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


class _InjectFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        src_jdata,
        dst_jdata,
        dst_grid_data,
        src_grid_data,
        dst_jt_impl,
        src_jt_impl,
    ):
        ctx.dst_grid_data = dst_grid_data
        ctx.src_grid_data = src_grid_data
        ctx.dst_jt_impl = dst_jt_impl
        ctx.src_jt_impl = src_jt_impl
        # Clone dst so the op is out-of-place.  mark_dirty on non-contiguous
        # views disconnects _InjectFnBackward from the autograd graph (PyTorch
        # inserts AsStridedBackward → CopySlices that bypasses the custom
        # backward entirely).  Cloning avoids this and keeps gradients correct.
        dst_out = dst_jdata.clone().contiguous()
        dst_out_jt = dst_jt_impl.jagged_like(dst_out)
        src_contig = src_jdata.contiguous()
        src_contig_jt = src_jt_impl.jagged_like(src_contig)
        _fvdb_cpp.inject_op(dst_grid_data, src_grid_data, dst_out_jt, src_contig_jt)
        return dst_out

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


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _resolve_dense_params(grid_data, sparse_data, min_coord, grid_size):
    """Compute origins tensor and grid_size list for inject_to_dense ops."""
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
#  Dense -> Sparse  (inject_from_dense) -- batch variants
# ---------------------------------------------------------------------------


def inject_from_dense_cminor_batch(
    grid: GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> JaggedTensor:
    """Inject values from a dense tensor (XYZC order) into sparse voxel data for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        dense_data (torch.Tensor): Dense input tensor, shape ``(B, X, Y, Z, C*)``.
        dense_origin (NumericMaxRank1): Voxel-space origin of the dense tensor.

    Returns:
        result (JaggedTensor): Sparse voxel data extracted from the dense tensor.

    .. seealso:: :func:`inject_from_dense_cminor_single`
    """
    grid_data = grid.data
    origin = (
        to_Vec3i(dense_origin)
        .unsqueeze(0)
        .expand(grid_data.grid_count, 3)
        .to(dtype=torch.int32, device=dense_data.device)
        .contiguous()
    )
    result = cast(torch.Tensor, _InjectFromDenseCminorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[4:])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return JaggedTensor(impl=grid_data.jagged_like(result))


def inject_from_dense_cmajor_batch(
    grid: GridBatch,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> JaggedTensor:
    """Inject values from a dense tensor (CXYZ order) into sparse voxel data for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        dense_data (torch.Tensor): Dense input tensor, shape ``(B, C*, X, Y, Z)``.
        dense_origin (NumericMaxRank1): Voxel-space origin of the dense tensor.

    Returns:
        result (JaggedTensor): Sparse voxel data extracted from the dense tensor.

    .. seealso:: :func:`inject_from_dense_cmajor_single`
    """
    grid_data = grid.data
    origin = (
        to_Vec3i(dense_origin)
        .unsqueeze(0)
        .expand(grid_data.grid_count, 3)
        .to(dtype=torch.int32, device=dense_data.device)
        .contiguous()
    )
    result = cast(torch.Tensor, _InjectFromDenseCmajorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[1:-3])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return JaggedTensor(impl=grid_data.jagged_like(result))


# ---------------------------------------------------------------------------
#  Dense -> Sparse  (inject_from_dense) -- single variants
# ---------------------------------------------------------------------------


def inject_from_dense_cminor_single(
    grid: Grid,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> torch.Tensor:
    """Inject values from a dense tensor (XYZC order) into sparse voxel data for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        dense_data (torch.Tensor): Dense input tensor, shape ``(1, X, Y, Z, C*)``.
        dense_origin (NumericMaxRank1): Voxel-space origin of the dense tensor.

    Returns:
        result (torch.Tensor): Sparse voxel data extracted from the dense tensor.

    .. seealso:: :func:`inject_from_dense_cminor_batch`
    """
    grid_data = grid.data
    origin = (
        to_Vec3i(dense_origin)
        .unsqueeze(0)
        .expand(grid_data.grid_count, 3)
        .to(dtype=torch.int32, device=dense_data.device)
        .contiguous()
    )
    result = cast(torch.Tensor, _InjectFromDenseCminorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[4:])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return result


def inject_from_dense_cmajor_single(
    grid: Grid,
    dense_data: torch.Tensor,
    dense_origin: NumericMaxRank1 = 0,
) -> torch.Tensor:
    """Inject values from a dense tensor (CXYZ order) into sparse voxel data for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        dense_data (torch.Tensor): Dense input tensor, shape ``(1, C*, X, Y, Z)``.
        dense_origin (NumericMaxRank1): Voxel-space origin of the dense tensor.

    Returns:
        result (torch.Tensor): Sparse voxel data extracted from the dense tensor.

    .. seealso:: :func:`inject_from_dense_cmajor_batch`
    """
    grid_data = grid.data
    origin = (
        to_Vec3i(dense_origin)
        .unsqueeze(0)
        .expand(grid_data.grid_count, 3)
        .to(dtype=torch.int32, device=dense_data.device)
        .contiguous()
    )
    result = cast(torch.Tensor, _InjectFromDenseCmajorFn.apply(dense_data, grid_data, origin))
    feature_shape = list(dense_data.shape[1:-3])
    if feature_shape:
        result = result.view(result.shape[0], *feature_shape)
    return result


# ---------------------------------------------------------------------------
#  Sparse -> Dense  (inject_to_dense) -- batch variants
# ---------------------------------------------------------------------------


def inject_to_dense_cminor_batch(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """Write sparse voxel data into a dense tensor (XYZC order) for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        sparse_data (JaggedTensor): Per-voxel feature data.
        min_coord (NumericMaxRank1 | NumericMaxRank2 | None): Minimum voxel coordinate for the dense grid.
        grid_size (NumericMaxRank1 | None): Size of the dense grid, broadcastable to ``(3,)``.

    Returns:
        result (torch.Tensor): Dense tensor, shape ``(B, X, Y, Z, C*)``.

    .. seealso:: :func:`inject_to_dense_cminor_single`
    """
    grid_data = grid.data
    origins, gs_list = _resolve_dense_params(grid_data, sparse_data, min_coord, grid_size)
    result = cast(
        torch.Tensor,
        _InjectToDenseCminorFn.apply(sparse_data.jdata, grid_data, origins, gs_list),
    )
    feature_shape = list(sparse_data.jdata.shape[1:])
    return result.view([grid_data.grid_count] + gs_list + feature_shape)


def inject_to_dense_cmajor_batch(
    grid: GridBatch,
    sparse_data: JaggedTensor,
    min_coord: NumericMaxRank1 | NumericMaxRank2 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """Write sparse voxel data into a dense tensor (CXYZ order) for a grid batch.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        sparse_data (JaggedTensor): Per-voxel feature data.
        min_coord (NumericMaxRank1 | NumericMaxRank2 | None): Minimum voxel coordinate for the dense grid.
        grid_size (NumericMaxRank1 | None): Size of the dense grid, broadcastable to ``(3,)``.

    Returns:
        result (torch.Tensor): Dense tensor, shape ``(B, C*, X, Y, Z)``.

    .. seealso:: :func:`inject_to_dense_cmajor_single`
    """
    grid_data = grid.data
    origins, gs_list = _resolve_dense_params(grid_data, sparse_data, min_coord, grid_size)
    result = cast(
        torch.Tensor,
        _InjectToDenseCmajorFn.apply(sparse_data.jdata, grid_data, origins, gs_list),
    )
    feature_shape = list(sparse_data.jdata.shape[1:])
    return result.view([grid_data.grid_count] + feature_shape + gs_list)


# ---------------------------------------------------------------------------
#  Sparse -> Dense  (inject_to_dense) -- single variants
# ---------------------------------------------------------------------------


def inject_to_dense_cminor_single(
    grid: Grid,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """Write sparse voxel data into a dense tensor (XYZC order) for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        sparse_data (torch.Tensor): Per-voxel feature data.
        min_coord (NumericMaxRank1 | None): Minimum voxel coordinate for the dense grid.
        grid_size (NumericMaxRank1 | None): Size of the dense grid, broadcastable to ``(3,)``.

    Returns:
        result (torch.Tensor): Dense tensor, shape ``(1, X, Y, Z, C*)``.

    .. seealso:: :func:`inject_to_dense_cminor_batch`
    """
    grid_data = grid.data
    sparse_jt = JaggedTensor(sparse_data)
    origins, gs_list = _resolve_dense_params(grid_data, sparse_jt, min_coord, grid_size)
    result = cast(
        torch.Tensor,
        _InjectToDenseCminorFn.apply(sparse_jt.jdata, grid_data, origins, gs_list),
    )
    feature_shape = list(sparse_jt.jdata.shape[1:])
    return result.view([grid_data.grid_count] + gs_list + feature_shape)


def inject_to_dense_cmajor_single(
    grid: Grid,
    sparse_data: torch.Tensor,
    min_coord: NumericMaxRank1 | None = None,
    grid_size: NumericMaxRank1 | None = None,
) -> torch.Tensor:
    """Write sparse voxel data into a dense tensor (CXYZ order) for a single grid.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        sparse_data (torch.Tensor): Per-voxel feature data.
        min_coord (NumericMaxRank1 | None): Minimum voxel coordinate for the dense grid.
        grid_size (NumericMaxRank1 | None): Size of the dense grid, broadcastable to ``(3,)``.

    Returns:
        result (torch.Tensor): Dense tensor, shape ``(1, C*, X, Y, Z)``.

    .. seealso:: :func:`inject_to_dense_cmajor_batch`
    """
    grid_data = grid.data
    sparse_jt = JaggedTensor(sparse_data)
    origins, gs_list = _resolve_dense_params(grid_data, sparse_jt, min_coord, grid_size)
    result = cast(
        torch.Tensor,
        _InjectToDenseCmajorFn.apply(sparse_jt.jdata, grid_data, origins, gs_list),
    )
    feature_shape = list(sparse_jt.jdata.shape[1:])
    return result.view([grid_data.grid_count] + feature_shape + gs_list)


# ---------------------------------------------------------------------------
#  Grid-to-grid injection -- batch variant
# ---------------------------------------------------------------------------


def inject_batch(
    dst_grid: GridBatch,
    src_grid: GridBatch,
    src: JaggedTensor,
    dst: JaggedTensor | None = None,
    default_value: float | int | bool = 0,
) -> JaggedTensor:
    """Inject data from ``src_grid`` into ``dst_grid`` in voxel space for grid batches.

    Supports backpropagation.

    Args:
        dst_grid (GridBatch): The destination grid batch.
        src_grid (GridBatch): The source grid batch.
        src (JaggedTensor): Source per-voxel data.
        dst (JaggedTensor | None): Optional destination buffer; created with *default_value* if ``None``.
        default_value (float | int | bool): Fill value for unmatched voxels. Default ``0``.

    Returns:
        result (JaggedTensor): Destination data with injected values.

    .. seealso:: :func:`inject_single`
    """
    dst_grid_data = dst_grid.data
    src_grid_data = src_grid.data
    jt_src = src
    if dst is None:
        dst_shape: list[int] = [dst_grid_data.total_voxels]
        dst_shape.extend(src.eshape)
        raw_dst_t = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
        jt_dst = JaggedTensor(impl=dst_grid_data.jagged_like(raw_dst_t))
    else:
        jt_dst = dst
    if jt_dst.eshape != jt_src.eshape:
        raise ValueError(
            f"src and dst must have the same element shape, got src: {jt_src.eshape}, dst: {jt_dst.eshape}"
        )
    if jt_dst.jdata.requires_grad and jt_dst.jdata.is_leaf:
        raise RuntimeError(
            "inject: destination tensor is a leaf variable that requires grad. "
            "Use a non-leaf tensor (e.g. dst = dst * 1.0) or detach it first."
        )
    dst_out = cast(
        torch.Tensor,
        _InjectFn.apply(
            jt_src.jdata,
            jt_dst.jdata,
            dst_grid_data,
            src_grid_data,
            jt_dst._impl,
            jt_src._impl,
        ),
    )
    jt_dst.jdata = dst_out
    return jt_dst


# ---------------------------------------------------------------------------
#  Grid-to-grid injection -- single variant
# ---------------------------------------------------------------------------


def inject_single(
    dst_grid: Grid,
    src_grid: Grid,
    src: torch.Tensor,
    dst: torch.Tensor | None = None,
    default_value: float | int | bool = 0,
) -> torch.Tensor:
    """Inject data from ``src_grid`` into ``dst_grid`` in voxel space for single grids.

    Supports backpropagation.

    Args:
        dst_grid (Grid): The destination single grid.
        src_grid (Grid): The source single grid.
        src (torch.Tensor): Source per-voxel data.
        dst (torch.Tensor | None): Optional destination buffer; created with *default_value* if ``None``.
        default_value (float | int | bool): Fill value for unmatched voxels. Default ``0``.

    Returns:
        result (torch.Tensor): Destination data with injected values.

    .. seealso:: :func:`inject_batch`
    """
    dst_grid_data = dst_grid.data
    src_grid_data = src_grid.data
    jt_src = JaggedTensor(src)
    if dst is None:
        eshape = list(src.shape[1:]) if src.dim() > 1 else []
        dst_shape = [dst_grid_data.total_voxels] + eshape
        raw_dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
    else:
        raw_dst = dst
    if raw_dst.requires_grad and raw_dst.is_leaf:
        raise RuntimeError(
            "inject: destination tensor is a leaf variable that requires grad. "
            "Use a non-leaf tensor (e.g. dst = dst * 1.0) or detach it first."
        )
    jt_dst = JaggedTensor(raw_dst)
    dst_out = cast(
        torch.Tensor,
        _InjectFn.apply(
            jt_src.jdata,
            jt_dst.jdata,
            dst_grid_data,
            src_grid_data,
            jt_dst._impl,
            jt_src._impl,
        ),
    )
    # Copy result back into the caller's tensor so in-place semantics are
    # preserved even when the caller doesn't capture the return value.
    if dst is not None:
        dst.data.copy_(dst_out.data)
    return dst_out


# ---------------------------------------------------------------------------
#  inject_from_ijk -- batch variant
# ---------------------------------------------------------------------------


def inject_from_ijk_batch(
    grid: GridBatch,
    src_ijk: JaggedTensor,
    src: JaggedTensor,
    dst: JaggedTensor | None = None,
    default_value: float | int | bool = 0,
) -> JaggedTensor:
    """Inject data from source voxel coordinates into a grid batch's voxel data.

    Supports backpropagation.

    Args:
        grid (GridBatch): The grid batch to inject into.
        src_ijk (JaggedTensor): Source voxel coordinates, shape ``(B, -1, 3)``.
        src (JaggedTensor): Source per-voxel data.
        dst (JaggedTensor | None): Optional destination buffer; created with *default_value* if ``None``.
        default_value (float | int | bool): Fill value for unmatched voxels. Default ``0``.

    Returns:
        result (JaggedTensor): Destination data with injected values.

    .. seealso:: :func:`inject_from_ijk_single`
    """
    from . import _query

    if not isinstance(src_ijk, JaggedTensor):
        raise TypeError(f"src_ijk must be a JaggedTensor, but got {type(src_ijk)}")
    if not isinstance(src, JaggedTensor):
        raise TypeError(f"src must be a JaggedTensor, but got {type(src)}")

    grid_data = grid.data
    if dst is None:
        dst_shape: list[int] = [grid_data.total_voxels]
        dst_shape.extend(src.eshape)
        dst = JaggedTensor(
            impl=grid_data.jagged_like(
                torch.full(
                    dst_shape,
                    fill_value=default_value,
                    dtype=src.dtype,
                    device=src.device,
                )
            )
        )
    else:
        if not isinstance(dst, JaggedTensor):
            raise TypeError(f"dst must be a JaggedTensor, but got {type(dst)}")

    if dst.eshape != src.eshape:
        raise ValueError(f"src and dst must have the same element shape, but got src: {src.eshape}, dst: {dst.eshape}")

    src_idx = _query.ijk_to_index_batch(grid, src_ijk, cumulative=True).jdata
    src_mask = src_idx >= 0
    src_idx = src_idx[src_mask]
    dst.jdata[src_idx] = src.jdata[src_mask]
    return dst


# ---------------------------------------------------------------------------
#  inject_from_ijk -- single variant
# ---------------------------------------------------------------------------


def inject_from_ijk_single(
    grid: Grid,
    src_ijk: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor | None = None,
    default_value: float | int | bool = 0,
) -> torch.Tensor:
    """Inject data from source voxel coordinates into a single grid's voxel data.

    Supports backpropagation.

    Args:
        grid (Grid): The single grid to inject into.
        src_ijk (torch.Tensor): Source voxel coordinates, shape ``(N, 3)``.
        src (torch.Tensor): Source per-voxel data.
        dst (torch.Tensor | None): Optional destination buffer; created with *default_value* if ``None``.
        default_value (float | int | bool): Fill value for unmatched voxels. Default ``0``.

    Returns:
        result (torch.Tensor): Destination data with injected values.

    .. seealso:: :func:`inject_from_ijk_batch`
    """
    from . import _query

    grid_data = grid.data
    if dst is None:
        eshape = list(src.shape[1:]) if src.dim() > 1 else []
        dst_shape = [grid_data.total_voxels] + eshape
        raw_dst = torch.full(dst_shape, fill_value=default_value, dtype=src.dtype, device=src.device)
    else:
        raw_dst = dst

    src_idx = _query.ijk_to_index_single(grid, src_ijk, cumulative=True)
    src_mask = src_idx >= 0
    src_idx = src_idx[src_mask]
    raw_dst[src_idx] = src[src_mask]
    return raw_dst
