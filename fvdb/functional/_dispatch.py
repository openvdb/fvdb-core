# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Input normalization helpers for the functional API.

These helpers follow the same pattern as ``to_Vec3iBroadcastable`` and friends
in :mod:`fvdb.types`: accept a wide input type, normalize it to a canonical
form, and let the function body operate on a single code path.

The canonical form for voxel-data arguments is always ``JaggedTensor``.  When
the caller passes plain ``torch.Tensor`` (single-grid convenience), the helper
wraps it; when the caller passes ``JaggedTensor``, it passes through.  The
returned ``unwrap`` callable reverses the conversion on the result so the
caller gets back the same type they passed in.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from .._fvdb_cpp import GridBatchData
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


def _get_grid_data(grid: GridBatch) -> GridBatchData:
    """Extract the GridBatchData C++ object from a GridBatch."""
    return grid.data


def _prepare_args(
    grid: GridBatch,
    *data_args: torch.Tensor | JaggedTensor | None,
) -> tuple[GridBatchData, tuple[JaggedTensor | None, ...], Callable]:
    """
    Normalize ``Tensor | JaggedTensor`` data arguments for the batched C++ path.

    Args:
        grid: A :class:`~fvdb.GridBatch`.
        *data_args: Variable-length data arguments that are either
            ``torch.Tensor`` (single-grid path) or ``JaggedTensor`` (batched
            path).  ``None`` values are passed through unchanged.

    Returns:
        A 3-tuple ``(grid_data, normed_args, unwrap)`` where:

        - **grid_data** is the C++ ``GridBatchData`` handle.
        - **normed_args** is a tuple of ``JaggedTensor | None`` with every
          ``Tensor`` wrapped as a single-element ``JaggedTensor``.
        - **unwrap(raw_tensor)** converts a raw ``torch.Tensor`` result back to
          the type matching the original input: returned as-is when the inputs
          were flat, wrapped in a ``JaggedTensor`` (using the first data arg's
          jagged structure) otherwise.
    """
    grid_data = _get_grid_data(grid)

    first = next((a for a in data_args if a is not None), None)
    is_flat = first is not None and isinstance(first, torch.Tensor)

    if is_flat:
        normed = tuple(JaggedTensor(a) if isinstance(a, torch.Tensor) else a for a in data_args)

        def _unwrap_flat(result: torch.Tensor) -> torch.Tensor:
            return result

        return grid_data, normed, _unwrap_flat

    assert isinstance(first, JaggedTensor), "At least one non-None JaggedTensor argument is required"
    proto: JaggedTensor = first

    def _unwrap_jagged(result: torch.Tensor) -> JaggedTensor:
        return proto.jagged_like(result)

    # In the non-flat branch all non-None args are JaggedTensor (plain Tensors
    # would have taken the is_flat path above), so the downcast is safe.
    normed = cast(tuple[JaggedTensor | None, ...], data_args)
    return grid_data, normed, _unwrap_jagged


def _prepare_grid(grid: GridBatch) -> tuple[GridBatchData, Callable[[GridBatchData], GridBatch]]:
    """
    Prepare a grid argument for topology ops that return a new grid.

    Returns:
        A 2-tuple ``(grid_data, unwrap_grid)`` where **unwrap_grid** wraps the
        C++ result back into a :class:`~fvdb.GridBatch`.
    """
    from ..grid_batch import GridBatch

    grid_data = _get_grid_data(grid)

    def _unwrap_grid(cpp_impl: GridBatchData) -> GridBatch:
        return GridBatch(data=cpp_impl)

    return grid_data, _unwrap_grid
