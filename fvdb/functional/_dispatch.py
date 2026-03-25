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

from typing import TYPE_CHECKING, Any, Callable

import torch

from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from .._fvdb_cpp import GridBatch as GridBatchCpp


def _prepare_args(
    grid: Any,
    *data_args: torch.Tensor | JaggedTensor | None,
) -> tuple[GridBatchCpp, tuple[JaggedTensor | None, ...], Callable]:
    """
    Normalize ``Tensor | JaggedTensor`` data arguments for the batched C++ path.

    Args:
        grid: A :class:`~fvdb.GridBatch` (only type after Grid retirement).
        *data_args: Variable-length data arguments that are either
            ``torch.Tensor`` (single-grid path) or ``JaggedTensor`` (batched
            path).  ``None`` values are passed through unchanged.

    Returns:
        A 3-tuple ``(grid_data, normed_args, unwrap)`` where:

        - **grid_data** is the C++ grid handle (``grid.data``).
        - **normed_args** is a tuple of ``JaggedTensor | None`` with every
          ``Tensor`` wrapped as a single-element ``JaggedTensor``.
        - **unwrap(cpp_jt)** converts a C++ ``JaggedTensor`` result back to the
          type matching the original input: ``Tensor`` via ``.jdata`` when the
          inputs were flat, ``JaggedTensor(impl=...)`` otherwise.
    """
    first = next((a for a in data_args if a is not None), None)
    is_flat = first is not None and isinstance(first, torch.Tensor)

    if is_flat:
        normed = tuple(JaggedTensor(a) if isinstance(a, torch.Tensor) else a for a in data_args)

        def _unwrap_flat(cpp_jt: Any) -> torch.Tensor:
            return cpp_jt.jdata

        return grid.data, normed, _unwrap_flat

    normed = data_args

    def _unwrap_jagged(cpp_jt: Any) -> JaggedTensor:
        return JaggedTensor(impl=cpp_jt)

    return grid.data, normed, _unwrap_jagged


def _prepare_grid(grid: Any) -> tuple[GridBatchCpp, Callable]:
    """
    Prepare a grid argument for topology ops that return a new grid.

    Returns:
        A 2-tuple ``(grid_data, unwrap_grid)`` where **unwrap_grid** wraps the
        C++ result back into a :class:`~fvdb.GridBatch`.
    """
    from ..grid_batch import GridBatch

    def _unwrap_grid(cpp_impl: Any) -> GridBatch:
        return GridBatch(data=cpp_impl)

    return grid.data, _unwrap_grid
