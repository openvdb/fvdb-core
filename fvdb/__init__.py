# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import ctypes
import importlib.util as _importlib_util
import pathlib

import torch

if torch.cuda.is_available():
    torch.cuda.init()


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
from . import _Cpp  # Import the module to use in jcat
from ._Cpp import JaggedTensor, ConvPackBackend
from ._Cpp import (
    scaled_dot_product_attention,
    config,
    jrand,
    jrandn,
    jones,
    jzeros,
    jempty,
    volume_render,
    gaussian_render_jagged,
)

# Import GridBatch and gridbatch_from_* functions from grid_batch.py
from ._grid_batch import (
    GridBatch,
    load_gridbatch,
    save_gridbatch,
)


from ._grid import (
    Grid,
    load_grid,
    save_grid,
)

from ._convolution_plan import ConvolutionPlan
from ._gaussian_splat_3d import GaussianSplat3d

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn

# isort: on


def jcat(things_to_cat, dim=None):
    if len(things_to_cat) == 0:
        raise ValueError("Cannot concatenate empty list")
    if isinstance(things_to_cat[0], GridBatch):
        if dim is not None:
            raise ValueError("GridBatch concatenation does not support dim argument")
        # Extract the C++ implementations from the GridBatch wrappers
        cpp_grids = [g._gridbatch for g in things_to_cat]
        cpp_result = _Cpp.jcat(cpp_grids)
        # Wrap the result back in a GridBatch
        return GridBatch(impl=cpp_result)
    elif isinstance(things_to_cat[0], JaggedTensor):
        return _Cpp.jcat(things_to_cat, dim)
    else:
        raise TypeError("jcat() can only cat GridBatch, JaggedTensor, or VDBTensor")


from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

__all__ = [
    "ConvPackBackend",
    "GridBatch",
    "JaggedTensor",
    "GaussianSplat3d",
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
]
