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
from ._Cpp import jcat as _jcat_cpp
from ._Cpp import ConvPackBackend
from ._Cpp import scaled_dot_product_attention as _scaled_dot_product_attention_cpp
from ._Cpp import (
    config,
    volume_render,
    gaussian_render_jagged,
)

# Import JaggedTensor from jagged_tensor.py
from .jagged_tensor import (
    JaggedTensor,
    jrand,
    jrandn,
    jones,
    jzeros,
    jempty,
)

# Import GridBatch and gridbatch_from_* functions from grid_batch.py
from .grid_batch import (
    GridBatch,
    load_gridbatch,
    save_gridbatch,
)


from .grid import (
    Grid,
    load_grid,
    save_grid,
)

from .convolution_plan import ConvolutionPlan
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats
from .enums import ProjectionType, ShOrderingMode

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
# Make these available without an explicit submodule import
from . import nn, viz, utils

# isort: on


def jcat(things_to_cat, dim=None):
    if len(things_to_cat) == 0:
        raise ValueError("Cannot concatenate empty list")
    if isinstance(things_to_cat[0], GridBatch):
        if dim is not None:
            raise ValueError("GridBatch concatenation does not support dim argument")
        # Extract the C++ implementations from the GridBatch wrappers
        cpp_grids = [g._gridbatch for g in things_to_cat]
        cpp_result = _jcat_cpp(cpp_grids)
        # Wrap the result back in a GridBatch
        return GridBatch(impl=cpp_result)
    elif isinstance(things_to_cat[0], JaggedTensor):
        return _Cpp.jcat([thing._impl for thing in things_to_cat], dim)
    else:
        raise TypeError("jcat() can only cat GridBatch or JaggedTensor")


def scaled_dot_product_attention(
    query: JaggedTensor, key: JaggedTensor, value: JaggedTensor, scale: float
) -> JaggedTensor:
    return JaggedTensor(impl=_scaled_dot_product_attention_cpp(query._impl, key._impl, value._impl, scale))


from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

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
