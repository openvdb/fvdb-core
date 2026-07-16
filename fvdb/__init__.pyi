# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.init()

def _parse_device_string(device_string: str | torch.device) -> torch.device: ...

# Make these available without an explicit submodule import
# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn, utils, viz
from ._fvdb_cpp import NanoVDBGridMetadata, config, hilbert, morton
from ._volume_render import volume_render
from .attention import scaled_dot_product_attention
from .convolution_plan import ConvolutionPlan
from .enums import SmoothingMode
from .grid import Grid
from .grid_batch import GridBatch, gcat
from .jagged_tensor import JaggedTensor, jcat
from .torch_jagged import (
    add,
    all,
    amax,
    amin,
    any,
    argmax,
    argmin,
    ceil,
    clamp,
    eq,
    exp,
    floor,
    floor_divide,
    ge,
    gt,
    le,
    log,
    lt,
    maximum,
    mean,
    minimum,
    mul,
    nan_to_num,
    ne,
    norm,
    pow,
    relu,
    relu_,
    remainder,
    round,
    sigmoid,
    sqrt,
    std,
    sub,
    sum,
    tanh,
    true_divide,
    var,
    where,
)

__all__ = [
    # Core classes
    "GridBatch",
    "Grid",
    "JaggedTensor",
    "ConvolutionPlan",
    "SmoothingMode",
    "Grid",
    "NanoVDBGridMetadata",
    # JaggedTensor operations
    # Concatenation of jagged tensors or grid/grid batches
    "jcat",
    "gcat",
    # Morton/Hilbert operations
    "morton",
    "hilbert",
    # Specialized operations
    "scaled_dot_product_attention",
    "volume_render",
    # Torch-compatible functions (work with both Tensor and JaggedTensor)
    "relu",
    "relu_",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "nan_to_num",
    "clamp",
    "add",
    "sub",
    "mul",
    "true_divide",
    "floor_divide",
    "remainder",
    "pow",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "sum",
    "mean",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "all",
    "any",
    "norm",
    "var",
    "std",
    # Config
    "config",
    # Submodules
    "viz",
    "nn",
    "utils",
]
