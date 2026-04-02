# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for grid batch indexing operations."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import torch

from .. import _fvdb_cpp
from ._dispatch import _get_grid_data

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


def _wrap_grid(cpp_impl):
    from ..grid_batch import GridBatch

    return GridBatch(data=cpp_impl)


@overload
def index_grid_batch(grid: GridBatch, index: int) -> GridBatch: ...


@overload
def index_grid_batch(grid: GridBatch, index: slice) -> GridBatch: ...


@overload
def index_grid_batch(grid: GridBatch, index: list[bool]) -> GridBatch: ...


@overload
def index_grid_batch(grid: GridBatch, index: list[int]) -> GridBatch: ...


@overload
def index_grid_batch(grid: GridBatch, index: torch.Tensor) -> GridBatch: ...


def index_grid_batch(
    grid: GridBatch,
    index: int | np.integer | slice | list[bool] | list[int] | torch.Tensor,
) -> GridBatch:
    """
    Select a subset of grids from a batch using indexing.

    Supports integer indexing, slicing, list indexing, and boolean/integer
    tensor indexing.

    Args:
        grid: The grid batch to index into.
        index: Index to select grids. Can be:
            - ``int``: Select a single grid.
            - ``slice``: Select a range of grids.
            - ``list[int]`` or ``list[bool]``: Select specific grids.
            - ``torch.Tensor``: Boolean or integer tensor for advanced indexing.

    Returns:
        A new :class:`~fvdb.GridBatch` containing the selected grids.
    """
    grid_data = _get_grid_data(grid)

    if isinstance(index, (int, np.integer)):
        return _wrap_grid(_fvdb_cpp.index_grid_batch_int(grid_data, int(index)))
    elif isinstance(index, slice):
        start, stop, step = index.indices(grid_data.grid_count)
        return _wrap_grid(_fvdb_cpp.index_grid_batch_slice(grid_data, start, stop, step))
    elif isinstance(index, list):
        if len(index) > 0 and isinstance(index[0], bool):
            return _wrap_grid(_fvdb_cpp.index_grid_batch_bool_list(grid_data, cast(list[bool], index)))
        return _wrap_grid(_fvdb_cpp.index_grid_batch_int64_list(grid_data, cast(list[int], index)))
    elif isinstance(index, torch.Tensor):
        return _wrap_grid(_fvdb_cpp.index_grid_batch_tensor(grid_data, index))
    else:
        raise TypeError(f"Unsupported index type: {type(index)}")
