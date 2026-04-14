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
from ._fvdb_cpp import (
    config,
    volume_render,
    morton,
    hilbert,
)
from . import _fvdb_cpp as _C

import math

# Import JaggedTensor from jagged_tensor.py
from .jagged_tensor import JaggedTensor, jcat
from .grid import Grid
from .grid_batch import GridBatch, gcat
from .grid import Grid
from .attention import scaled_dot_product_attention

from ._gaussian_autograd import (
    _ProjectGaussiansJaggedFn,
    _EvaluateGaussianSHFn,
    _RasterizeScreenSpaceGaussiansFn,
)


# TODO: Make a batched class to encapsulate this jagged rendering pipeline.
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
    """Render Gaussian splats with jagged (variable-length) batched inputs.

    This function composes differentiable projection, SH evaluation, tile intersection,
    and rasterization stages, each backed by Python ``torch.autograd.Function`` wrappers
    around the underlying CUDA/CPU dispatch kernels.

    Args:
        means: Jagged tensor of Gaussian centers ``[sum(N_i), 3]``.
        quats: Jagged tensor of Gaussian quaternions ``[sum(N_i), 4]``.
        scales: Jagged tensor of Gaussian scales ``[sum(N_i), 3]``.
        opacities: Jagged tensor of Gaussian opacities ``[sum(N_i)]``.
        sh_coeffs: Jagged tensor of SH coefficients ``[sum(N_i), K, D]``.
        viewmats: Jagged tensor of world-to-camera matrices ``[sum(C_i), 4, 4]``.
        Ks: Jagged tensor of intrinsic matrices ``[sum(C_i), 3, 3]``.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        sh_degree_to_use: SH degree to evaluate (``-1`` means use all available bases).
        tile_size: Rasterization tile size in pixels.
        radius_clip: Minimum 2-D radius for projected Gaussians.
        eps2d: Epsilon added to 2-D covariance diagonal for numerical stability.
        antialias: Whether to apply antialiasing compensation to opacities.
        render_depth_channel: If ``True``, append a depth channel to the rendered colors.
        return_debug_info: If ``True``, return intermediate tensors in the debug dict.
        ortho: Use orthographic projection.
        backgrounds: Optional per-camera background colors ``[total_cameras, D, H, W]``.
        masks: Optional per-camera masks ``[total_cameras, 1, H, W]``.

    Returns:
        A tuple ``(rendered_images, rendered_alphas, debug_info)`` where
        ``rendered_images`` has shape ``[total_cameras, D, H, W]`` and
        ``rendered_alphas`` has shape ``[total_cameras, 1, H, W]``.
    """
    ccz = viewmats.jdata.size(0)  # total cameras across all batches

    # --- Build cross-batch index arrays ---
    # TODO: This indexing logic is convoluted but there is no better way without
    # custom CUDA kernels.  Given Gaussians with shape [sum(N_i), ...] and cameras
    # with shape [sum(C_i), ...], we compute the cross-product of each batch's
    # Gaussians with that batch's cameras, producing a flat tensor of shape
    # [sum(C_i * N_i), ...].  We need to track two index arrays:
    #   camera_ids:   shape [sum(C_i * N_i)], values in [0, sum(C_i))
    #   gaussian_ids: shape [sum(C_i * N_i)], values in [0, sum(N_i))
    # g_sizes: [N1, N2, ...], c_sizes: [C1, C2, ...]
    g_sizes = means.joffsets[1:] - means.joffsets[:-1]
    c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]

    # camera_ids: flat index into viewmats.jdata for each (gaussian, camera) pair
    tt = g_sizes.repeat_interleave(c_sizes)
    camera_ids = torch.arange(ccz, device=means.device, dtype=torch.int32).repeat_interleave(tt, 0)

    # gaussian_ids: flat index into means.jdata for each pair
    dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
    dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
    shifts = dd0[1:] - dd1[:-1]
    shifts = torch.cat([torch.tensor([0], device=means.device), shifts])
    shifts_cumsum = shifts.cumsum(0)
    gaussian_ids = torch.arange(camera_ids.size(0), device=means.device, dtype=torch.int32)
    gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, 0)

    # --- Differentiable projection ---
    radii, means2d, depths, conics, compensations = _ProjectGaussiansJaggedFn.apply(
        g_sizes,
        means.jdata,
        quats.jdata,
        scales.jdata,
        c_sizes,
        viewmats.jdata,
        Ks.jdata,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        ortho,
    )

    # Gather opacities per (gaussian, camera) pair
    opacities_batched = opacities.jdata[gaussian_ids]
    if antialias:
        opacities_batched = opacities_batched * compensations

    debug_info: dict[str, torch.Tensor] = {}
    if return_debug_info:
        debug_info["camera_ids"] = camera_ids
        debug_info["gaussian_ids"] = gaussian_ids
        debug_info["radii"] = radii
        debug_info["means2d"] = means2d
        debug_info["depths"] = depths
        debug_info["conics"] = conics
        debug_info["opacities"] = opacities_batched

    # --- Differentiable SH evaluation ---
    K = sh_coeffs.jdata.size(-2)
    actual_sh_degree = int(math.sqrt(K) - 1) if sh_degree_to_use < 0 else sh_degree_to_use

    # Permute [total_G, K, D] → [K, total_G, D], then gather by gaussian_ids → [K, nnz, D]
    sh_coeffs_batched = sh_coeffs.jdata.permute(1, 0, 2)[:, gaussian_ids, :]

    if actual_sh_degree == 0:
        sh0 = sh_coeffs_batched[0, :, :].unsqueeze(0)  # [1, nnz, D]
        render_quantities = _EvaluateGaussianSHFn.apply(
            actual_sh_degree,
            1,
            None,
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            None,
            radii.unsqueeze(0),  # [1, nnz]
        )
    else:
        sh0 = sh_coeffs_batched[0, :, :].unsqueeze(0)  # [1, nnz, D]
        shN = sh_coeffs_batched[1:, :, :]  # [K-1, nnz, D]
        # FIXME (Francis): Compute view directions in the kernel instead of
        # materializing a large tensor here.  Doing so would require updating
        # the current backward pass as well.
        camtoworlds = torch.linalg.inv(viewmats.jdata)  # [ccz, 4, 4]
        # NOTE: dirs are not normalized here; the SH evaluation kernel normalizes
        # them internally.
        dirs = means.jdata[gaussian_ids] - camtoworlds[camera_ids, :3, 3]
        render_quantities = _EvaluateGaussianSHFn.apply(
            actual_sh_degree,
            1,
            dirs.unsqueeze(0),  # [1, nnz, 3]
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            shN.permute(1, 0, 2),  # [nnz, K-1, D]
            radii.unsqueeze(0),  # [1, nnz]
        )
    render_quantities = render_quantities.squeeze(0)  # [nnz, D]

    if render_depth_channel:
        render_quantities = torch.cat([render_quantities, depths[gaussian_ids].unsqueeze(-1)], dim=-1)

    # --- Non-differentiable tile intersection ---
    num_tiles_h = math.ceil(image_height / tile_size)
    num_tiles_w = math.ceil(image_width / tile_size)
    tile_offsets, tile_gaussian_ids_t = _C.intersect_gaussian_tiles(
        means2d, radii, depths, ccz, tile_size, num_tiles_h, num_tiles_w, camera_ids
    )
    if return_debug_info:
        debug_info["tile_offsets"] = tile_offsets
        debug_info["tile_gaussian_ids"] = tile_gaussian_ids_t

    # --- Differentiable rasterization ---
    rendered_images, rendered_alphas = _RasterizeScreenSpaceGaussiansFn.apply(
        means2d,
        conics,
        render_quantities,
        opacities_batched.contiguous(),
        image_width,
        image_height,
        0,  # image_origin_w
        0,  # image_origin_h
        tile_size,
        tile_offsets,
        tile_gaussian_ids_t,
        False,  # absgrad
        backgrounds,
        masks,
    )

    return rendered_images, rendered_alphas, debug_info


from .convolution_plan import ConvolutionPlan
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats
from .enums import CameraModel, ProjectionMethod, RollingShutterType, ShOrderingMode
from ._gaussian_autograd import _EvaluateGaussianSHFn


def evaluate_spherical_harmonics(
    sh_degree: int,
    num_cameras: int,
    sh0: torch.Tensor,
    radii: torch.Tensor,
    shN: torch.Tensor | None = None,
    view_directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics to compute view-dependent features/colors.

    Args:
        sh_degree: Degree of spherical harmonics to use (0-3 typically).
        num_cameras: Number of camera views (C).
        sh0: DC term coefficients with shape [N, 1, D].
        radii: Projected radii with shape [C, N] (int32). Points with radii <= 0
               will output zeros.
        shN: Higher-order SH coefficients with shape [N, K-1, D] where
             K = (sh_degree+1)^2. Required when sh_degree > 0.
        view_directions: Unnormalized view directions with shape [C, N, 3].
                         Required when sh_degree > 0.

    Returns:
        Tensor of shape [C, N, D] containing the evaluated features/colors.
    """
    if sh_degree > 0 and view_directions is None:
        raise ValueError("view_directions is required when sh_degree > 0")
    return _EvaluateGaussianSHFn.apply(sh_degree, num_cameras, view_directions, sh0, shN, radii)


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
