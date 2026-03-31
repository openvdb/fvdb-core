# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Single sparse voxel grid wrapper."""
from __future__ import annotations

from dataclasses import dataclass

from ._fvdb_cpp import GridBatchData


@dataclass(frozen=True)
class Grid:
    """A single sparse voxel grid. Thin frozen wrapper around GridBatchData with grid_count == 1."""

    data: GridBatchData

    def __post_init__(self):
        if self.data.grid_count != 1:
            raise ValueError(f"Grid requires grid_count == 1, got {self.data.grid_count}")
