# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

if torch.cuda.is_available():
    torch.cuda.init()

def _parse_device_string(device_string: str | torch.device) -> torch.device: ...

# Make these available without an explicit submodule import
# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn, utils, viz
from ._Cpp import ConvPackBackend, config, volume_render
from .convolution_plan import ConvolutionPlan
from .enums import ProjectionType, ShOrderingMode
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats
from .grid import Grid, load_grid, save_grid
from .grid_batch import GridBatch, load_gridbatch, save_gridbatch
from .jagged_tensor import JaggedTensor, jempty, jones, jrand, jrandn, jzeros

def scaled_dot_product_attention(
    query: JaggedTensor, key: JaggedTensor, value: JaggedTensor, scale: float
) -> JaggedTensor: ...
def gaussian_render_jagged(
    means: JaggedTensor,
    quats: JaggedTensor,
    scales: JaggedTensor,
    opacities: JaggedTensor,
    sh_coeffs: JaggedTensor,
    viewmats: JaggedTensor,
    Ks: JaggedTensor,
    image_width: int,
    image_height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    antialias: bool = False,
    render_depth_channel: bool = False,
    return_debug_info: bool = False,
    render_depth_only: bool = False,
    ortho: bool = False,
) -> JaggedTensor: ...
@overload
def jcat(grid_batches: Sequence[GridBatch]) -> GridBatch: ...
@overload
def jcat(jagged_tensors: Sequence[JaggedTensor], dim: int | None = None) -> JaggedTensor: ...

__all__ = [
    "GridBatch",
    "JaggedTensor",
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "ConvolutionPlan",
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
    "viz",
    "nn",
    "utils",
]
