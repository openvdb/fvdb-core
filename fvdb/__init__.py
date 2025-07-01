# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.init()


def _parse_device_string(device_string: str) -> torch.device:
    """
    Python equivalent of the C++ parseDeviceString function.

    Parses a device string and returns a torch.device object. For CUDA devices
    without an explicit index, uses the current CUDA device.

    Args:
        device_string (str): A device string (e.g., "cpu", "cuda", "cuda:0").

    Returns:
        torch.device: The parsed device object with proper device index set.
    """
    device = torch.device(device_string)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


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
from .grid_batch import (
    GridBatch,
    gridbatch_from_ijk,
    gridbatch_from_points,
    gridbatch_from_nearest_voxels_to_points,
    gridbatch_from_dense,
    gridbatch_from_mesh,
    load,
    save,
)

# Import SparseConvPackInfo from sparse_conv_pack_info.py
from .sparse_conv_pack_info import SparseConvPackInfo
from .gaussian_splatting import GaussianSplat3d

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
    elif isinstance(things_to_cat[0], nn.VDBTensor):
        if dim == 0:
            raise ValueError("VDBTensor concatenation does not support dim=0")
        grids = [t.grid for t in things_to_cat]
        data = [t.data for t in things_to_cat]
        # Check if grids contain wrapped GridBatch objects
        if grids and isinstance(grids[0], GridBatch):
            # Extract C++ implementations
            cpp_grids = [g._gridbatch for g in grids]
            grid_result = _Cpp.jcat(cpp_grids) if dim == None else cpp_grids[0]
            # Wrap back in GridBatch if concatenated
            if dim == None:
                grid_result = GridBatch(impl=grid_result)
            else:
                grid_result = grids[0]
        else:
            grid_result = GridBatch(impl=_Cpp.jcat(grids)) if dim == None else grids[0]
        return nn.VDBTensor(grid_result, _Cpp.jcat(data, dim))
    else:
        raise ValueError("jcat() can only cat GridBatch, JaggedTensor, or VDBTensor")


from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

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
