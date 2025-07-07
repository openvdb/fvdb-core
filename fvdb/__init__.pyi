# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from .types import JaggedTensorOrTensor

if torch.cuda.is_available():
    torch.cuda.init()

def _parse_tensor_or_sequence(tensor_or_sequence: torch.Tensor | Sequence, name: str = "") -> torch.Tensor: ...
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
from .grid_batch import (
    GridBatch,
    gridbatch_from_dense,
    gridbatch_from_ijk,
    gridbatch_from_mesh,
    gridbatch_from_nearest_voxels_to_points,
    gridbatch_from_points,
    load,
    save,
)
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
    "gridbatch_from_ijk",
    "gridbatch_from_points",
    "gridbatch_from_nearest_voxels_to_points",
    "gridbatch_from_dense",
    "gridbatch_from_mesh",
    "load",
    "jcat",
    "scaled_dot_product_attention",
    "config",
    "save",
    "jrand",
    "jrandn",
    "jones",
    "jzeros",
    "jempty",
    "volume_render",
    "gaussian_render_jagged",
]
