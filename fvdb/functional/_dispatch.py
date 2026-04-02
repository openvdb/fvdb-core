# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Helpers for the functional API.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._fvdb_cpp import GridBatchData

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


def _get_grid_data(grid: GridBatch) -> GridBatchData:
    """Extract the GridBatchData C++ object from a GridBatch."""
    return grid.data
