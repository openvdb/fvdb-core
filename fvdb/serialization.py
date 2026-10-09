# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Space-filling-curve serialization of voxels for serialized (patch) attention.

Serialization sorts each grid's voxels along a space-filling curve so that voxels close in
space end up close in sequence. Point Transformer V3 (PTv3) groups the sorted sequence into
fixed-size patches and runs attention within each patch.

The order names and axis conventions follow Pointcept's PTv3 implementation, so a model
trained with one produces the same voxel orderings in the other.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from . import _fvdb_cpp
from .grid_batch import GridBatch
from .jagged_tensor import JaggedTensor

SERIALIZATION_ORDERS: tuple[str, ...] = ("vdb", "z", "z-trans", "hilbert", "hilbert-trans")
"""Supported serialization order names."""

OffsetMode = Literal["grid", "batch"]

_ALIASES = {"identity": "vdb", "morton": "z", "zorder": "z", "z_order": "z"}
_MAX_DEPTH = 21  # bits per axis of fvdb's Morton and Hilbert encoders
_TRANS = (1, 0, 2)  # Pointcept "-trans" orders swap x and y

# Column order of ijk for fvdb.hilbert to equal Pointcept's Hilbert code at a depth d.
# The curve's orientation cycles with (21 - d) % 3.
_HILBERT_AXES = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}
_MORTON_AXES = (2, 1, 0)


def _normalize_order(order: str) -> str:
    name = _ALIASES.get(order, order)
    if name not in SERIALIZATION_ORDERS:
        raise ValueError(
            f"Unknown serialization order {order!r}. Expected one of {SERIALIZATION_ORDERS}. "
            "Note that GridBatch.morton_zyx/hilbert_zyx swap x and z, which differs from the "
            "Pointcept '-trans' orders (x and y swapped) used here."
        )
    return name


def _encoder_axes(order: str, depth: int) -> tuple[int, ...]:
    """Columns of ijk to pass to fvdb's encoder so codes equal Pointcept's at ``depth``."""
    axes = _MORTON_AXES if order.startswith("z") else _HILBERT_AXES[(_MAX_DEPTH - depth) % 3]
    if order.endswith("-trans"):
        axes = tuple(_TRANS[a] for a in axes)
    return axes


def _grid_index(grid: GridBatch) -> torch.Tensor | None:
    """Per-voxel grid index, or None for a single grid, where fvdb leaves ``jidx`` empty."""
    return grid.jidx.long() if grid.grid_count > 1 else None


def _check_offset_mode(offset_mode: str) -> None:
    if offset_mode not in ("grid", "batch"):
        raise ValueError(f"offset_mode must be 'grid' or 'batch', got {offset_mode!r}")


def _local_ijk(grid: GridBatch, offset_mode: OffsetMode) -> torch.Tensor:
    """Voxel coordinates shifted to be non-negative, in the grid's native voxel order."""
    ijk = grid.ijk.jdata
    if offset_mode == "grid":
        # Per-grid minimum, as Pointcept starts each sample at the origin.
        # data.bbox is host metadata; rows of empty grids are never gathered.
        mins = grid.data.bbox[:, 0].to(device=ijk.device, dtype=ijk.dtype)
        jidx = _grid_index(grid)
        return ijk - (mins[0] if jidx is None else mins[jidx])
    return ijk - grid.data.total_bbox[0].to(device=ijk.device, dtype=ijk.dtype)


def _max_local_coord(grid: GridBatch, offset_mode: OffsetMode) -> int:
    """Largest coordinate after the offset, from host metadata. Empty grids have an inverted bbox."""
    if offset_mode == "grid":
        bbox = grid.data.bbox.long()
        return max(0, int((bbox[:, 1] - bbox[:, 0]).max()))
    bbox = grid.data.total_bbox.long()
    return max(0, int((bbox[1] - bbox[0]).max()))


def serialization_depth(grid: GridBatch, *, offset_mode: OffsetMode = "grid") -> int:
    """Curve depth Pointcept would choose for a grid batch.

    Pointcept sets the depth to ``(max_coord + 1).bit_length()`` over the whole batch, where
    ``max_coord`` is the largest coordinate after each sample is shifted to start at 0. The
    Hilbert curve's orientation depends on the depth, so batches with different extents can
    order the same scene differently, exactly as in Pointcept.

    Args:
        grid (GridBatch): The grid batch.
        offset_mode (str): See :func:`serialization_codes`. Defaults to ``"grid"``.

    Returns:
        depth (int): Bits per axis of the curve.
    """
    _check_offset_mode(offset_mode)
    if grid.total_voxels == 0:
        return 1
    return (_max_local_coord(grid, offset_mode) + 1).bit_length()


def _resolve_depth(grid: GridBatch, offset_mode: OffsetMode, depth: int | None, code_shift: int) -> int:
    if code_shift < 0:
        raise ValueError(f"code_shift must be non-negative, got {code_shift}")
    if depth is None:
        depth = serialization_depth(grid, offset_mode=offset_mode) + code_shift
    if not 0 < depth <= _MAX_DEPTH:
        raise ValueError(f"depth must be in [1, {_MAX_DEPTH}], got {depth}")
    if grid.total_voxels > 0 and _max_local_coord(grid, offset_mode) >= 1 << max(0, depth - code_shift):
        raise ValueError(
            f"Coordinates span more than 2**(depth - code_shift) = 2**{depth - code_shift} voxels; "
            "increase depth or use the depth of the finest level."
        )
    return depth


def _packed_key_shift(grid: GridBatch, depth: int, code_shift: int) -> int | None:
    """Bit shift that packs the grid index above the curve code, or None if it does not fit.

    Codes are below ``2 ** (3 * (depth - code_shift))``.
    """
    shift = 3 * max(1, depth - code_shift)
    batch_bits = max(1, (grid.grid_count - 1).bit_length())
    return shift if shift + batch_bits <= 63 else None


def _segmented_argsort(code: torch.Tensor, jidx: torch.Tensor | None, shift: int | None) -> torch.Tensor:
    """Argsort ``code`` within each grid, keeping grids contiguous and in batch order."""
    if jidx is None:
        return torch.argsort(code, stable=True)
    if shift is not None:
        return torch.argsort((jidx << shift) | code, stable=True)

    # Two stable passes: sort by code, then stably by grid index.
    by_code = torch.argsort(code, stable=True)
    by_grid = torch.argsort(jidx[by_code], stable=True)
    return by_code[by_grid]


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device))
    return inv


def _codes(grid: GridBatch, order: str, offset_mode: OffsetMode, depth: int, code_shift: int) -> torch.Tensor:
    local = _local_ijk(grid, offset_mode) << code_shift
    ijk = local[:, list(_encoder_axes(order, depth))].contiguous()
    code = _fvdb_cpp.morton(ijk) if order.startswith("z") else _fvdb_cpp.hilbert(ijk)
    return code >> (3 * code_shift)


def serialization_codes(
    grid: GridBatch,
    order: str,
    *,
    offset_mode: OffsetMode = "grid",
    depth: int | None = None,
    code_shift: int = 0,
) -> torch.Tensor:
    """Compute space-filling-curve codes for the active voxels of a grid batch.

    Codes equal Pointcept's ``encode(grid_coord, depth=depth, order=order)`` for the same
    coordinates. For a pooled grid, Pointcept derives codes from the finest level by dropping
    three bits per halving. Pass the finest level's ``depth`` and the number of halvings as
    ``code_shift`` to reproduce that.

    Args:
        grid (GridBatch): The grid batch.
        order (str): One of :data:`SERIALIZATION_ORDERS`. ``"vdb"`` returns the voxel index,
            which leaves voxels in the grid's native order.
        offset_mode (str): ``"grid"`` shifts each grid by its own minimum coordinate, which makes
            the order invariant to translating a grid. ``"batch"`` shifts every grid by the batch
            minimum. Defaults to ``"grid"``.
        depth (int | None): Curve depth in bits per axis, at most 21. ``None`` uses
            :func:`serialization_depth` plus ``code_shift``. Defaults to ``None``.
        code_shift (int): Number of halvings since the level ``depth`` refers to. Defaults to ``0``.

    Returns:
        codes (torch.Tensor): Non-negative ``int64`` codes. Shape: ``(total_voxels,)``.
    """
    order = _normalize_order(order)
    _check_offset_mode(offset_mode)
    n = grid.total_voxels
    if order == "vdb" or n == 0:
        return torch.arange(n, dtype=torch.int64, device=grid.device)
    depth = _resolve_depth(grid, offset_mode, depth, code_shift)
    return _codes(grid, order, offset_mode, depth, code_shift)


def serialization_perm(
    grid: GridBatch,
    order: str,
    *,
    offset_mode: OffsetMode = "grid",
    depth: int | None = None,
    code_shift: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the permutation that sorts each grid's voxels along a space-filling curve.

    ``data.jdata[perm]`` is sorted along the curve within each grid, and grids stay contiguous
    and in batch order.

    Args:
        grid (GridBatch): The grid batch.
        order (str): One of :data:`SERIALIZATION_ORDERS`.
        offset_mode (str): See :func:`serialization_codes`. Defaults to ``"grid"``.
        depth (int | None): See :func:`serialization_codes`. Defaults to ``None``.
        code_shift (int): See :func:`serialization_codes`. Defaults to ``0``.

    Returns:
        perm (torch.Tensor): ``int64`` sorting permutation. Shape: ``(total_voxels,)``.
        inv_perm (torch.Tensor): The inverse permutation. Shape: ``(total_voxels,)``.
    """
    ser = serialize(grid, order, offset_mode=offset_mode, depth=depth, code_shift=code_shift)
    return ser.perms[0], ser.inv_perms[0]


@dataclass(frozen=True)
class GridSerialization:
    """Serialization permutations for one grid batch under several curve orders.

    Build with :func:`serialize`. Every attention block at one resolution level shares this
    object and selects an order by index, so the permutations are computed once per level.
    """

    orders: tuple[str, ...]
    """Order names, in their current (possibly shuffled) sequence."""
    perms: tuple[torch.Tensor, ...]
    """Per-order permutations, each of shape ``(total_voxels,)``."""
    inv_perms: tuple[torch.Tensor, ...]
    """Per-order inverse permutations."""
    codes: tuple[torch.Tensor, ...] | None
    """Per-order curve codes, if requested."""
    joffsets: torch.Tensor
    """Offsets of the grid batch these permutations belong to."""
    depth: int
    """Curve depth of the finest level."""
    code_shift: int
    """Number of halvings from the finest level to this grid."""
    offset_mode: OffsetMode
    """Coordinate offset mode used for the codes."""

    def __len__(self) -> int:
        return len(self.orders)

    def order(self, index: int) -> str:
        """Name of the order at ``index``, taken modulo the number of orders."""
        return self.orders[index % len(self.orders)]

    def perm(self, index: int) -> torch.Tensor:
        """Permutation for the order at ``index``, taken modulo the number of orders."""
        return self.perms[index % len(self.orders)]

    def inv_perm(self, index: int) -> torch.Tensor:
        """Inverse permutation for the order at ``index``, taken modulo the number of orders."""
        return self.inv_perms[index % len(self.orders)]

    def shuffled(self, generator: torch.Generator | None = None) -> GridSerialization:
        """Return a copy with the orders randomly reordered.

        Args:
            generator (torch.Generator | None): Optional CPU generator for the shuffle.

        Returns:
            GridSerialization: The shuffled serialization. Tensors are shared, not copied.
        """
        idx = torch.randperm(len(self.orders), generator=generator).tolist()
        return dataclasses.replace(
            self,
            orders=tuple(self.orders[i] for i in idx),
            perms=tuple(self.perms[i] for i in idx),
            inv_perms=tuple(self.inv_perms[i] for i in idx),
            codes=None if self.codes is None else tuple(self.codes[i] for i in idx),
        )

    def matches(self, data: JaggedTensor) -> bool:
        """Whether ``data`` has the jagged layout these permutations were built for."""
        return torch.equal(self.joffsets, data.joffsets)

    def pooled(
        self,
        coarse_grid: GridBatch,
        stride: int = 2,
        *,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> GridSerialization:
        """Serialize a grid pooled from this one, continuing this level's curves.

        Coarse codes are the fine codes with three bits dropped per halving, as in Pointcept's
        serialized pooling. The coarse grid should be ``ijk // stride`` of this level's grid.

        Args:
            coarse_grid (GridBatch): The pooled grid.
            stride (int): Pooling stride, a power of two. Defaults to ``2``.
            shuffle (bool): Randomly reorder the orders. Defaults to ``False``.
            generator (torch.Generator | None): Optional CPU generator for the shuffle.

        Returns:
            GridSerialization: Serialization of ``coarse_grid`` with the same orders.
        """
        if stride < 1 or stride & (stride - 1):
            raise ValueError(f"stride must be a power of two, got {stride}")
        return serialize(
            coarse_grid,
            self.orders,
            offset_mode=self.offset_mode,
            depth=self.depth,
            code_shift=self.code_shift + (stride - 1).bit_length(),
            shuffle=shuffle,
            generator=generator,
            keep_codes=self.codes is not None,
        )


def serialize(
    grid: GridBatch,
    orders: str | Sequence[str] = ("z", "z-trans"),
    *,
    offset_mode: OffsetMode = "grid",
    depth: int | None = None,
    code_shift: int = 0,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    keep_codes: bool = False,
) -> GridSerialization:
    """Serialize a grid batch under one or more space-filling-curve orders.

    Args:
        grid (GridBatch): The grid batch.
        orders (str | Sequence[str]): Order name or names from :data:`SERIALIZATION_ORDERS`.
            Defaults to ``("z", "z-trans")``.
        offset_mode (str): See :func:`serialization_codes`. Defaults to ``"grid"``.
        depth (int | None): See :func:`serialization_codes`. Defaults to ``None``.
        code_shift (int): See :func:`serialization_codes`. Defaults to ``0``.
        shuffle (bool): Randomly reorder the orders, as PTv3 does each forward pass.
            Defaults to ``False``.
        generator (torch.Generator | None): Optional CPU generator for the shuffle.
        keep_codes (bool): Keep the curve codes on the result. Defaults to ``False``.

    Returns:
        GridSerialization: Permutations for each order.
    """
    names = (orders,) if isinstance(orders, str) else tuple(orders)
    if not names:
        raise ValueError("serialize requires at least one order")
    names = tuple(_normalize_order(o) for o in names)
    _check_offset_mode(offset_mode)

    n = grid.total_voxels
    depth = _resolve_depth(grid, offset_mode, depth, code_shift) if n > 0 else (depth or 1)
    shift = _packed_key_shift(grid, depth, code_shift)
    jidx = _grid_index(grid) if n > 0 else None
    perms, inv_perms, codes = [], [], []
    for name in names:
        if name == "vdb" or n == 0:
            code = torch.arange(n, dtype=torch.int64, device=grid.device)
            perm = code
        else:
            code = _codes(grid, name, offset_mode, depth, code_shift)
            perm = _segmented_argsort(code, jidx, shift)
        perms.append(perm)
        inv_perms.append(_inverse_permutation(perm))
        if keep_codes:
            codes.append(code)

    result = GridSerialization(
        orders=names,
        perms=tuple(perms),
        inv_perms=tuple(inv_perms),
        codes=tuple(codes) if keep_codes else None,
        joffsets=grid.joffsets,
        depth=depth,
        code_shift=code_shift,
        offset_mode=offset_mode,
    )
    return result.shuffled(generator) if shuffle else result


def permute_jagged(data: JaggedTensor, perm: torch.Tensor) -> JaggedTensor:
    """Reorder the elements of a JaggedTensor with a grid-preserving permutation.

    Args:
        data (JaggedTensor): Data to reorder. Shape: ``(batch_size, num_elements, *)``.
        perm (torch.Tensor): Permutation from :func:`serialization_perm` or
            :class:`GridSerialization`. Shape: ``(total_elements,)``.

    Returns:
        JaggedTensor: ``data`` with ``jdata`` reordered. The jagged layout is unchanged.
    """
    return data.jagged_like(data.jdata.index_select(0, perm))
