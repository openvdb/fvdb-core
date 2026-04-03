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
from ._fvdb_cpp import evaluate_spherical_harmonics as _evaluate_spherical_harmonics_cpp
from ._fvdb_cpp import (
    config,
    volume_render,
    morton,
    hilbert,
)

# Import JaggedTensor from jagged_tensor.py
from .jagged_tensor import JaggedTensor, jcat
from .grid import Grid
from .grid_batch import GridBatch, gcat


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
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    import math

    from . import _fvdb_cpp as _C
    from .functional.splat._projection import _ProjectGaussiansJaggedFn
    from .functional.splat._rasterize import _RasterizeDenseFn
    from .functional.splat._sh import _EvalSHFn

    ccz = viewmats.jdata.shape[0]  # total number of cameras
    device = means.jdata.device
    dtype = means.jdata.dtype

    # ---- Step 1: Compute batch indices (camera_ids, gaussian_ids) ----
    # g_sizes is [N1, N2, ...], c_sizes is [C1, C2, ...]
    g_sizes = means.joffsets[1:] - means.joffsets[:-1]
    c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]

    # camera_ids: for each element in the expanded (C_i * N_i) layout, which camera it belongs to
    tt = g_sizes.repeat_interleave(c_sizes)
    camera_ids = torch.arange(ccz, device=device, dtype=torch.int32).repeat_interleave(tt, 0)

    # gaussian_ids: for each element, which gaussian (in the flat jdata) it belongs to
    dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
    dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
    shifts = dd0[1:] - dd1[:-1]
    shifts = torch.cat([torch.tensor([0], device=device), shifts])
    shifts_cumsum = shifts.cumsum(0)
    gaussian_ids = torch.arange(camera_ids.shape[0], device=device, dtype=torch.int32)
    gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, 0)

    # ---- Step 2: Jagged projection (differentiable via Python autograd) ----
    radii, means2d, depths, conics = _ProjectGaussiansJaggedFn.apply(
        g_sizes, means.jdata, quats.jdata, scales.jdata, c_sizes,
        viewmats.jdata, Ks.jdata, image_width, image_height, eps2d,
        near_plane, far_plane, radius_clip, ortho,
    )

    # ---- Step 3: Gather opacities for the expanded layout ----
    opacities_batched = opacities.jdata[gaussian_ids]  # [M]

    # ---- Debug info (populated before SH eval so we capture projection outputs) ----
    debug_info: dict[str, torch.Tensor] = {}
    if return_debug_info:
        debug_info["camera_ids"] = camera_ids
        debug_info["gaussian_ids"] = gaussian_ids
        debug_info["radii"] = radii
        debug_info["means2d"] = means2d
        debug_info["depths"] = depths
        debug_info["conics"] = conics
        debug_info["opacities"] = opacities_batched

    # ---- Step 4: Compute render features (SH eval or depth) ----
    nnz = camera_ids.shape[0]
    D = sh_coeffs.jdata.shape[-1]  # feature dimension (typically 3 for RGB)

    # sh_coeffs.jdata is [total_N, K, D]; permute to [K, total_N, D] then gather
    sh_coeffs_batched = sh_coeffs.jdata.permute(1, 0, 2)[:, gaussian_ids, :]  # [K, nnz, D]
    K = sh_coeffs_batched.shape[0]
    actual_sh_degree = sh_degree_to_use if sh_degree_to_use >= 0 else int(math.sqrt(K)) - 1

    if actual_sh_degree == 0:
        sh0 = sh_coeffs_batched[0].unsqueeze(0)  # [1, nnz, D]
        features = _EvalSHFn.apply(
            actual_sh_degree, 1,
            torch.zeros(1, nnz, 3, device=device, dtype=dtype),
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            torch.empty(nnz, 0, D, device=device, dtype=dtype),
            radii.unsqueeze(0),  # [1, nnz]
        )
    else:
        sh0 = sh_coeffs_batched[0].unsqueeze(0)  # [1, nnz, D]
        shN = sh_coeffs_batched[1:]  # [K-1, nnz, D]
        cam_to_world = torch.linalg.inv(viewmats.jdata)  # [ccz, 4, 4]
        dirs = means.jdata[gaussian_ids] - cam_to_world[camera_ids, :3, 3]  # [nnz, 3]
        features = _EvalSHFn.apply(
            actual_sh_degree, 1,
            dirs.unsqueeze(0),  # [1, nnz, 3]
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            shN.permute(1, 0, 2),  # [nnz, K-1, D]
            radii.unsqueeze(0),  # [1, nnz]
        )
    features = features.squeeze(0)  # [C=1, nnz, D] -> [nnz, D]

    if render_depth_channel:
        features = torch.cat([features, depths[gaussian_ids].unsqueeze(-1)], -1)

    # ---- Step 5: Tile intersection (non-differentiable) ----
    num_tiles_w = math.ceil(image_width / tile_size)
    num_tiles_h = math.ceil(image_height / tile_size)
    tile_offsets, tile_gaussian_ids = _C.gsplat_tile_intersection(
        means2d, radii, depths, ccz, tile_size, num_tiles_h, num_tiles_w,
        camera_ids=camera_ids,
    )

    if return_debug_info:
        debug_info["tile_offsets"] = tile_offsets
        debug_info["tile_gaussian_ids"] = tile_gaussian_ids

    # ---- Step 6: Rasterize (differentiable via Python autograd) ----
    images, alphas = _RasterizeDenseFn.apply(
        means2d, conics, features, opacities_batched.contiguous(),
        image_width, image_height, 0, 0, tile_size,
        tile_offsets, tile_gaussian_ids, False, backgrounds, masks,
    )

    return images, alphas, debug_info


def evaluate_spherical_harmonics(
    sh_degree: int,
    num_cameras: int,
    sh0: torch.Tensor,
    radii: torch.Tensor,
    shN: torch.Tensor | None = None,
    view_directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics (differentiable via Python autograd)."""
    from .functional.splat._sh import _EvalSHFn

    if sh_degree > 0:
        if view_directions is None:
            raise ValueError("view_directions must be provided when sh_degree > 0")
        if shN is None:
            raise ValueError("shN must be provided when sh_degree > 0")
    view_dirs = view_directions if view_directions is not None else torch.zeros(
        num_cameras, sh0.size(0), 3, device=sh0.device, dtype=sh0.dtype)
    shN_val = shN if shN is not None else torch.empty(
        sh0.size(0), 0, sh0.size(2), device=sh0.device, dtype=sh0.dtype)
    return _EvalSHFn.apply(sh_degree, num_cameras, view_dirs, sh0, shN_val, radii)


from .convolution_plan import ConvolutionPlan
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats
from .enums import CameraModel, ProjectionMethod, RollingShutterType, ShOrderingMode

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
from . import functional, nn, utils, version, viz
from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

__all__ = [
    # Core classes
    "Grid",
    "GridBatch",
    "JaggedTensor",
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "CameraModel",
    "ProjectionMethod",
    "RollingShutterType",
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
    "functional",
    "version",
    "viz",
    "nn",
    "utils",
]
