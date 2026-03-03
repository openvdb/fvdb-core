# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import ctypes
import importlib.util as _importlib_util
import pathlib
from typing import Sequence

import torch

if torch.cuda.is_available():
    torch.cuda.init()


def _parse_device_string(device_or_device_string: str | torch.device) -> torch.device:
    """
    Parses a device string and returns a torch.device object. For CUDA devices
    without an explicit index, uses the current CUDA device. If the input is a torch.device
    object, it is returned unmodified.

     Args:
         device_string (str | torch.device):
             A device string (e.g., "cpu", "cuda", "cuda:0") or a torch.device object.
             If a string is provided, it should be a valid device identifier.

     Returns:
         torch.device: The parsed device object with proper device index set if a string is passed
         in otherwise returns the input torch.device object.
    """
    if isinstance(device_or_device_string, torch.device):
        return device_or_device_string
    if not isinstance(device_or_device_string, str):
        raise TypeError(f"Expected a string or torch.device, but got {type(device_or_device_string)}")
    device = torch.device(device_or_device_string)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


# Load NanoVDB Editor shared libraries so symbols are globally available before importing the pybind module.
# This helps the dynamic linker resolve dependencies like libpnanovdb*.so when loading fvdb's extensions.
_spec = _importlib_util.find_spec("nanovdb_editor")
if _spec is not None and _spec.origin is not None:
    try:
        _libdir = pathlib.Path(_spec.origin).parent / "lib"
        for _so in sorted(_libdir.glob("libpnanovdb*.so")):
            try:
                ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                print(f"Failed to load {_so} from {_libdir}")
                pass
    except Exception:
        print("Failed to load nanovdb_editor from", _libdir)
        pass

# isort: off
from ._fvdb_cpp import scaled_dot_product_attention as _scaled_dot_product_attention_cpp
from ._fvdb_cpp import gaussian_render_jagged as _gaussian_render_jagged_cpp
from ._fvdb_cpp import evaluate_spherical_harmonics
from ._fvdb_cpp import (
    config,
    volume_render,
    morton,
    hilbert,
)

# Import JaggedTensor from jagged_tensor.py
from .jagged_tensor import JaggedTensor, jcat
from .grid_batch import GridBatch, gcat
from .grid import Grid


def scaled_dot_product_attention(
    query: JaggedTensor, key: JaggedTensor, value: JaggedTensor, scale: float
) -> JaggedTensor:
    return JaggedTensor(impl=_scaled_dot_product_attention_cpp(query._impl, key._impl, value._impl, scale))


def gaussian_render_jagged(
    means: JaggedTensor,  # [N1 + N2 + ..., 3]
    quats: JaggedTensor,  # [N1 + N2 + ..., 4]
    scales: JaggedTensor,  # [N1 + N2 + ..., 3]
    opacities: JaggedTensor,  # [N1 + N2 + ...]
    sh_coeffs: JaggedTensor,  # [N1 + N2 + ..., K, 3]
    viewmats: JaggedTensor,  # [C1 + C2 + ..., 4, 4]
    Ks: JaggedTensor,  # [C1 + C2 + ..., 3, 3]
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
    ortho: bool = False,
    backgrounds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    return _gaussian_render_jagged_cpp(
        means=means._impl,
        quats=quats._impl,
        scales=scales._impl,
        opacities=opacities._impl,
        sh_coeffs=sh_coeffs._impl,
        viewmats=viewmats._impl,
        Ks=Ks._impl,
        image_width=image_width,
        image_height=image_height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree_to_use=sh_degree_to_use,
        tile_size=tile_size,
        radius_clip=radius_clip,
        eps2d=eps2d,
        antialias=antialias,
        render_depth_channel=render_depth_channel,
        return_debug_info=return_debug_info,
        ortho=ortho,
        backgrounds=backgrounds,
    )


from .convolution_plan import ConvolutionPlan
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats
from .enums import ProjectionType, ShOrderingMode

# Import torch-compatible functions that work with both Tensor and JaggedTensor
from .torch_jagged import (
    # Unary operations
    relu,
    relu_,
    sigmoid,
    tanh,
    exp,
    log,
    sqrt,
    floor,
    ceil,
    round,
    nan_to_num,
    clamp,
    # Binary operations
    add,
    sub,
    mul,
    true_divide,
    floor_divide,
    remainder,
    pow,
    maximum,
    minimum,
    # Comparisons
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    where,
    # Reductions
    sum,
    mean,
    amax,
    amin,
    argmax,
    argmin,
    all,
    any,
    norm,
    var,
    std,
)

# The following import needs to come after all classes and functions are defined
# in order to avoid a circular dependency error.
# Make these available without an explicit submodule import
from . import nn, utils, viz
from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

__all__ = [
    # Core classes
    "GridBatch",
    "Grid",
    "JaggedTensor",
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "ProjectionType",
    "ShOrderingMode",
    "ConvolutionPlan",
    # Concatenation of jagged tensors or grid/grid batches
    "jcat",
    "gcat",
    # Morton/Hilbert operations
    "morton",
    "hilbert",
    # Specialized operations
    "scaled_dot_product_attention",
    "volume_render",
    "gaussian_render_jagged",
    "evaluate_spherical_harmonics",
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
