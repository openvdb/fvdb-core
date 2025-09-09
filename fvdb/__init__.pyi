# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

if torch.cuda.is_available():
    torch.cuda.init()

from .types import JaggedTensorOrTensor

def _parse_device_string(device_string: str | torch.device) -> torch.device: ...

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn
from ._Cpp import (
    ConvPackBackend,
    JaggedTensor,
    config,
    gaussian_render_jagged,
    jempty,
    jones,
    jrand,
    jrandn,
    jzeros,
    scaled_dot_product_attention,
    volume_render,
)
from .gaussian_splatting import GaussianSplat3d
from .grid import Grid, load_grid, save_grid
from .grid_batch import GridBatch, load_gridbatch, save_gridbatch
from .sparse_conv_pack_info import SparseConvPackInfo

@overload
def jcat(grid_batches: Sequence[GridBatch]) -> GridBatch: ...
@overload
def jcat(jagged_tensors: Sequence[JaggedTensorOrTensor], dim: int | None = None) -> JaggedTensor: ...
@overload
def jcat(jagged_tensors: Sequence[JaggedTensor], dim: int | None = None) -> JaggedTensor: ...
@overload
def jcat(vdb_tensors: Sequence[nn.VDBTensor], dim: int | None = None) -> nn.VDBTensor: ...

__all__ = [
    "GridBatch",
    "JaggedTensor",
    "SparseConvPackInfo",
    "ConvPackBackend",
    "GaussianSplat3d",
    "load_gridbatch",
    "save_gridbatch",
    "jcat",
    "scaled_dot_product_attention",
    "config",
    "jrand",
    "jrandn",
    "jones",
    "jzeros",
    "jempty",
    "volume_render",
    "gaussian_render_jagged",
    "Grid",
    "load_grid",
    "save_grid",
]
