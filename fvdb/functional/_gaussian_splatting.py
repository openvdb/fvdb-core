# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Flat functional wrappers over the Gaussian splatting CUDA kernels.

Each function here wraps one compiled kernel entry point one to one. None of them build an autograd
graph: forward and backward kernels are exposed as separate functions, and differentiable
composition (``torch.autograd.Function`` classes, pipeline dataclasses) lives in downstream packages
such as ``fvdb-reality-capture``. Unlike the grid operations in :mod:`fvdb.functional`, these
functions have no ``_batch`` / ``_single`` variants.

Conventions shared by every function:

- ``C`` is the number of cameras, ``N`` the number of Gaussians, ``D`` the feature channel count.
- Pixel coordinates have their origin at the top-left corner of the image; x grows to the right
  and y grows downward.
- ``pixels_to_render`` is a :class:`~fvdb.JaggedTensor` with one ``[P_c, 2]`` list of ``(row, col)``
  integer pixel coordinates per camera. A plain ``[C, P, 2]`` tensor is also accepted; 2-D tensors are
  rejected because ``[P, 2]`` and ``[C, 2]`` cannot be told apart. Coordinates must lie inside the
  image and be unique per camera; the layout kernel raises ``ValueError`` otherwise.
- The sparse rasterization kernels are compiled for ``tile_size == 16``; the sparse wrappers reject
  any other value. Dense kernels accept any tile size.
- Outputs the kernel does not produce for the given arguments come back as ``None``.
- Sparse functions given a layout with no active tiles return empty results without launching a
  kernel, so a selection emptied by filtering flows through the pipeline.
- Camera enums are accepted as :class:`~fvdb.CameraModel`, :class:`~fvdb.RollingShutterType`,
  their integer values, or the bound C++ enum members.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .. import _fvdb_cpp
from ..enums import CameraModel, RollingShutterType, _to_cpp_enum
from ..jagged_tensor import JaggedTensor

# The sparse rasterization kernels use a 16x16 block scan to index active pixels within a tile.
_SPARSE_TILE_SIZE = 16

# ---------------------------------------------------------------------------
#  Conversion and validation helpers
# ---------------------------------------------------------------------------


def _to_cpp_camera_model(camera_model: CameraModel | int) -> "_fvdb_cpp.CameraModel":
    """Convert a :class:`fvdb.CameraModel`, its int value, or the bound C++ member to the C++ enum."""
    return _to_cpp_enum(CameraModel, _fvdb_cpp.CameraModel, camera_model)


def _to_cpp_rolling_shutter(rolling_shutter_type: RollingShutterType | int) -> "_fvdb_cpp.RollingShutterType":
    """Convert a :class:`fvdb.RollingShutterType`, its int value, or the bound C++ member to the C++ enum."""
    return _to_cpp_enum(RollingShutterType, _fvdb_cpp.RollingShutterType, rolling_shutter_type)


def _jagged_impl(value: JaggedTensor, name: str) -> "_fvdb_cpp.JaggedTensor":
    """Return the C++ implementation object behind a public :class:`fvdb.JaggedTensor`."""
    if not isinstance(value, JaggedTensor):
        raise TypeError(f"{name} must be a fvdb.JaggedTensor, got {type(value).__name__}")
    return value._impl


def as_pixel_jagged(value: JaggedTensor | torch.Tensor) -> JaggedTensor:
    """Normalize a ``pixels_to_render`` selection to a JaggedTensor with one ``[P_c, 2]`` list per camera.

    The sparse tile-layout and rasterization wrappers call this on their input; callers that build
    their own sparse pipeline (deduplicating pixels first, say) can call it to get the same checks.
    Only shapes and dtypes are checked here; coordinate bounds and uniqueness are checked by the
    layout kernel at the synchronization it performs anyway.

    Args:
        value (JaggedTensor | torch.Tensor): ``(row, col)`` integer pixels, either a JaggedTensor with
            one ``[P_c, 2]`` list per camera or a dense ``[C, P, 2]`` tensor with ``C > 0``.

    Returns:
        pixels (JaggedTensor): The selection as a JaggedTensor. A JaggedTensor input is returned as is.

    Raises:
        TypeError: If ``value`` is neither a JaggedTensor nor a Tensor, or its coordinates are not
            ``int32`` or ``int64``.
        ValueError: If the shape is not ``[C, P, 2]`` (tensor) or ``[P, 2]`` per list (JaggedTensor), or
            a JaggedTensor nests lists (``ldim != 1``).
    """
    if isinstance(value, torch.Tensor):
        if value.dim() != 3 or value.shape[0] == 0 or value.shape[2] != 2:
            raise ValueError(f"pixels_to_render tensor must have shape [C, P, 2] with C > 0, got {tuple(value.shape)}")
        value = JaggedTensor(list(value.unbind(0)))
    elif not isinstance(value, JaggedTensor):
        raise TypeError(f"pixels_to_render must be a fvdb.JaggedTensor or torch.Tensor, got {type(value).__name__}")
    if value.ldim != 1:
        raise ValueError(f"pixels_to_render must have one list per camera (ldim == 1), got ldim == {value.ldim}")
    coords = value.jdata
    if coords.dim() != 2 or coords.shape[1] != 2:
        raise ValueError(f"pixels_to_render elements must be (row, col) pairs, got jdata shape {tuple(coords.shape)}")
    if coords.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"pixels_to_render must be int32 or int64, got {coords.dtype}")
    return value


def _empty_like_pixels(pixels: JaggedTensor, element_shape: tuple[int, ...], dtype: torch.dtype) -> JaggedTensor:
    """An empty per-pixel result with the same camera structure as an empty ``pixels`` selection."""
    return pixels.jagged_like(torch.empty((0, *element_shape), dtype=dtype, device=pixels.device))


def _check_sparse_tile_size(tile_size: int) -> None:
    """Reject tile sizes the sparse rasterization kernels are not compiled for.

    The rasterizers' ``cub::BlockScan`` is fixed at 16x16 threads. The layout op alone handles other
    sizes, but it is checked here too so that a layout built here always fits the rasterizers.
    """
    if tile_size != _SPARSE_TILE_SIZE:
        raise ValueError(f"sparse Gaussian rasterization requires tile_size == {_SPARSE_TILE_SIZE}, got {tile_size}")


def _sparse_prologue(
    tile_size: int,
    pixels_to_render: JaggedTensor | torch.Tensor,
    active_tiles: torch.Tensor,
    pixel_map: torch.Tensor,
) -> tuple[JaggedTensor, bool]:
    """Shared entry checks for the sparse rasterization wrappers.

    Returns the normalized pixel selection and whether the layout is empty. An empty layout must
    come with an empty selection; otherwise the arguments disagree and the call is rejected rather
    than silently returning zeros.
    """
    _check_sparse_tile_size(tile_size)
    pixels = as_pixel_jagged(pixels_to_render)
    empty = active_tiles.numel() == 0
    if empty and (pixels.jdata.shape[0] != 0 or pixel_map.numel() != 0):
        raise ValueError(
            "active_tiles is empty but pixels_to_render or pixel_map is not; "
            "the sparse layout does not match the pixel selection"
        )
    return pixels, empty


def _wrap(impl: "_fvdb_cpp.JaggedTensor") -> JaggedTensor:
    """Wrap a C++ jagged result in the public :class:`fvdb.JaggedTensor`."""
    return JaggedTensor(impl=impl)


# ---------------------------------------------------------------------------
#  Projection
# ---------------------------------------------------------------------------


def project_gaussians_analytic_fwd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    near: float,
    far: float,
    min_radius_2d: float,
    calc_compensations: bool,
    ortho: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project 3D Gaussians to 2D screen space with the analytic (EWA) projection.

    Discarded Gaussians (clipped by the near/far planes or smaller than ``min_radius_2d``) have
    both radii set to zero; their other outputs are left uninitialized.

    Args:
        means (torch.Tensor): Gaussian centers in world space, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian rotations as quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors per axis, shape ``[N, 3]``.
        world_to_cam_matrices (torch.Tensor): World-to-camera transforms, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        eps2d (float): Blur added to the 2D covariance for numerical stability.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        min_radius_2d (float): Gaussians whose projected radius is at most this value are discarded.
        calc_compensations (bool): Whether to compute the anti-aliasing compensation factors.
        ortho (bool): Use an orthographic projection instead of perspective.

    Returns:
        radii (torch.Tensor): Per-axis projected radii, shape ``[C, N, 2]``, ``int32``.
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        depths (torch.Tensor): View-space depths, shape ``[C, N]``.
        conics (torch.Tensor): Inverse 2D covariances as ``(a, b, c)``, shape ``[C, N, 3]``.
        compensations (torch.Tensor | None): Compensation factors, shape ``[C, N]``, or ``None`` when
            ``calc_compensations`` is ``False``.
    """
    return _fvdb_cpp.project_gaussians_analytic_fwd(
        means,
        quats,
        log_scales,
        world_to_cam_matrices,
        projection_matrices,
        image_width,
        image_height,
        eps2d,
        near,
        far,
        min_radius_2d,
        calc_compensations,
        ortho,
    )


def project_gaussians_analytic_bwd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    compensations: torch.Tensor | None,
    image_width: int,
    image_height: int,
    eps2d: float,
    radii: torch.Tensor,
    conics: torch.Tensor,
    d_loss_d_means2d: torch.Tensor,
    d_loss_d_depths: torch.Tensor,
    d_loss_d_conics: torch.Tensor,
    d_loss_d_compensations: torch.Tensor | None,
    world_to_cam_matrices_requires_grad: bool,
    ortho: bool,
    out_normalized_d_loss_d_means2d_norm_accum: torch.Tensor | None = None,
    out_normalized_max_radii_accum: torch.Tensor | None = None,
    out_gradient_step_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward pass of :func:`project_gaussians_analytic_fwd`.

    The optional ``out_*`` accumulators gather per-Gaussian statistics for densification strategies
    (split, clone, prune). The kernel updates them in place only when both
    ``out_normalized_d_loss_d_means2d_norm_accum`` and ``out_gradient_step_counts`` are given;
    ``out_normalized_max_radii_accum`` is updated only alongside those two.

    Args:
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        world_to_cam_matrices (torch.Tensor): World-to-camera transforms, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        compensations (torch.Tensor | None): Forward compensation factors, shape ``[C, N]``, or ``None``.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        eps2d (float): Blur used in the forward pass.
        radii (torch.Tensor): Forward radii, shape ``[C, N, 2]``.
        conics (torch.Tensor): Forward conics, shape ``[C, N, 3]``.
        d_loss_d_means2d (torch.Tensor): Loss gradient w.r.t. ``means2d``, shape ``[C, N, 2]``.
        d_loss_d_depths (torch.Tensor): Loss gradient w.r.t. ``depths``, shape ``[C, N]``.
        d_loss_d_conics (torch.Tensor): Loss gradient w.r.t. ``conics``, shape ``[C, N, 3]``.
        d_loss_d_compensations (torch.Tensor | None): Loss gradient w.r.t. ``compensations``, shape
            ``[C, N]``, or ``None``.
        world_to_cam_matrices_requires_grad (bool): Whether to compute the camera-matrix gradient.
        ortho (bool): Whether the forward pass used an orthographic projection.
        out_normalized_d_loss_d_means2d_norm_accum (torch.Tensor | None): Optional ``[N]`` float
            accumulator of image-normalized 2D mean gradient norms.
        out_normalized_max_radii_accum (torch.Tensor | None): Optional ``[N]`` ``int32`` accumulator of
            the maximum projected radius in pixels.
        out_gradient_step_counts (torch.Tensor | None): Optional ``[N]`` ``int32`` accumulator of
            gradient step counts.

    Returns:
        d_loss_d_means (torch.Tensor): Gradient w.r.t. ``means``, shape ``[N, 3]``.
        d_loss_d_covars (torch.Tensor | None): Gradient w.r.t. precomputed 3D covariances. ``None``
            when the covariance is derived from ``quats`` and ``log_scales``, which is always the case
            through this binding.
        d_loss_d_quats (torch.Tensor): Gradient w.r.t. ``quats``, shape ``[N, 4]``.
        d_loss_d_log_scales (torch.Tensor): Gradient w.r.t. ``log_scales``, shape ``[N, 3]``.
        d_loss_d_world_to_cam_matrices (torch.Tensor | None): Gradient w.r.t. ``world_to_cam_matrices``,
            shape ``[C, 4, 4]``, or ``None`` when not requested.
    """
    return _fvdb_cpp.project_gaussians_analytic_bwd(
        means,
        quats,
        log_scales,
        world_to_cam_matrices,
        projection_matrices,
        compensations,
        image_width,
        image_height,
        eps2d,
        radii,
        conics,
        d_loss_d_means2d,
        d_loss_d_depths,
        d_loss_d_conics,
        d_loss_d_compensations,
        world_to_cam_matrices_requires_grad,
        ortho,
        out_normalized_d_loss_d_means2d_norm_accum,
        out_normalized_max_radii_accum,
        out_gradient_step_counts,
    )


def project_gaussians_analytic_jagged_fwd(
    g_sizes: torch.Tensor,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    c_sizes: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    near: float,
    far: float,
    min_radius_2d: float,
    ortho: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Analytic projection for a batch of scenes with varying Gaussian and camera counts.

    Scene ``b`` owns ``g_sizes[b]`` consecutive Gaussians and ``c_sizes[b]`` consecutive cameras.
    Every Gaussian is projected into every camera of its own scene, producing ``M`` packed
    projections in total. Unlike :func:`project_gaussians_analytic_fwd`, ``scales`` are linear.

    Args:
        g_sizes (torch.Tensor): Gaussians per scene, shape ``[B]``.
        means (torch.Tensor): Gaussian centers for all scenes, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        scales (torch.Tensor): Linear scale factors per axis, shape ``[N, 3]``.
        c_sizes (torch.Tensor): Cameras per scene, shape ``[B]``.
        world_to_cam_matrices (torch.Tensor): World-to-camera transforms for all scenes, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics for all scenes, shape ``[C, 3, 3]``.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        eps2d (float): Blur added to the 2D covariance for numerical stability.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        min_radius_2d (float): Gaussians whose projected radius is at most this value are discarded.
        ortho (bool): Use an orthographic projection instead of perspective.

    Returns:
        radii (torch.Tensor): Per-axis projected radii, shape ``[M, 2]``, ``int32``.
        means2d (torch.Tensor): Projected 2D centers, shape ``[M, 2]``.
        depths (torch.Tensor): View-space depths, shape ``[M]``.
        conics (torch.Tensor): Inverse 2D covariances as ``(a, b, c)``, shape ``[M, 3]``.
        compensations (torch.Tensor | None): Compensation factors, shape ``[M]``, or ``None``.
    """
    return _fvdb_cpp.project_gaussians_analytic_jagged_fwd(
        g_sizes,
        means,
        quats,
        scales,
        c_sizes,
        world_to_cam_matrices,
        projection_matrices,
        image_width,
        image_height,
        eps2d,
        near,
        far,
        min_radius_2d,
        ortho,
    )


def project_gaussians_analytic_jagged_bwd(
    g_sizes: torch.Tensor,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    c_sizes: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    radii: torch.Tensor,
    conics: torch.Tensor,
    d_loss_d_means2d: torch.Tensor,
    d_loss_d_depths: torch.Tensor,
    d_loss_d_conics: torch.Tensor,
    world_to_cam_matrices_requires_grad: bool,
    ortho: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward pass of :func:`project_gaussians_analytic_jagged_fwd`.

    Args:
        g_sizes (torch.Tensor): Gaussians per scene, shape ``[B]``.
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        scales (torch.Tensor): Linear scale factors, shape ``[N, 3]``.
        c_sizes (torch.Tensor): Cameras per scene, shape ``[B]``.
        world_to_cam_matrices (torch.Tensor): World-to-camera transforms, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        eps2d (float): Blur used in the forward pass.
        radii (torch.Tensor): Forward radii, shape ``[M, 2]``.
        conics (torch.Tensor): Forward conics, shape ``[M, 3]``.
        d_loss_d_means2d (torch.Tensor): Loss gradient w.r.t. ``means2d``, shape ``[M, 2]``.
        d_loss_d_depths (torch.Tensor): Loss gradient w.r.t. ``depths``, shape ``[M]``.
        d_loss_d_conics (torch.Tensor): Loss gradient w.r.t. ``conics``, shape ``[M, 3]``.
        world_to_cam_matrices_requires_grad (bool): Whether to compute the camera-matrix gradient.
        ortho (bool): Whether the forward pass used an orthographic projection.

    Returns:
        d_loss_d_means (torch.Tensor): Gradient w.r.t. ``means``, shape ``[N, 3]``.
        d_loss_d_covars (torch.Tensor | None): Gradient w.r.t. precomputed 3D covariances, ``None``
            through this binding.
        d_loss_d_quats (torch.Tensor): Gradient w.r.t. ``quats``, shape ``[N, 4]``.
        d_loss_d_scales (torch.Tensor): Gradient w.r.t. ``scales``, shape ``[N, 3]``.
        d_loss_d_world_to_cam_matrices (torch.Tensor | None): Gradient w.r.t. ``world_to_cam_matrices``,
            shape ``[C, 4, 4]``, or ``None`` when not requested.
    """
    return _fvdb_cpp.project_gaussians_analytic_jagged_bwd(
        g_sizes,
        means,
        quats,
        scales,
        c_sizes,
        world_to_cam_matrices,
        projection_matrices,
        image_width,
        image_height,
        eps2d,
        radii,
        conics,
        d_loss_d_means2d,
        d_loss_d_depths,
        d_loss_d_conics,
        world_to_cam_matrices_requires_grad,
        ortho,
    )


def project_gaussians_ut_fwd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_cam_matrices_start: torch.Tensor,
    world_to_cam_matrices_end: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    camera_model: CameraModel | int,
    image_width: int,
    image_height: int,
    eps2d: float,
    near: float,
    far: float,
    min_radius_2d: float,
    calc_compensations: bool,
    rolling_shutter_type: RollingShutterType | int = RollingShutterType.NONE,
    ut_alpha: float = 0.1,
    ut_beta: float = 2.0,
    ut_kappa: float = 0.0,
    ut_in_image_margin: float = 0.1,
    ut_require_all_sigma_points_in_image: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project 3D Gaussians to 2D screen space with the unscented transform (UT).

    Each Gaussian is represented by seven sigma points that are pushed through the full camera
    model, including OpenCV distortion and rolling shutter, and the 2D mean and covariance are
    rebuilt from the projected points. There is no separate backward kernel; use
    :func:`rasterize_world_space_gaussians_bwd` to differentiate through this path.

    Args:
        means (torch.Tensor): Gaussian centers in world space, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        world_to_cam_matrices_start (torch.Tensor): World-to-camera transforms at the start of the
            exposure, shape ``[C, 4, 4]``.
        world_to_cam_matrices_end (torch.Tensor): World-to-camera transforms at the end of the
            exposure, shape ``[C, 4, 4]``. Equal to the start matrices without rolling shutter.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        distortion_coeffs (torch.Tensor): Packed OpenCV distortion coefficients, shape ``[C, 12]``,
            or an empty ``[C, 0]`` tensor for :attr:`~fvdb.CameraModel.PINHOLE` and
            :attr:`~fvdb.CameraModel.ORTHOGRAPHIC`.
        camera_model (CameraModel | int): Camera model used for projection.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        eps2d (float): Blur added to the 2D covariance for numerical stability.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        min_radius_2d (float): Gaussians whose projected radius is at most this value are discarded.
        calc_compensations (bool): Whether to compute the anti-aliasing compensation factors.
        rolling_shutter_type (RollingShutterType | int): Rolling shutter policy.
        ut_alpha (float): UT spread parameter.
        ut_beta (float): UT prior-knowledge parameter (``2`` is optimal for Gaussians).
        ut_kappa (float): UT secondary scaling parameter.
        ut_in_image_margin (float): Margin, as a fraction of the image size, within which sigma
            points still count as inside the image.
        ut_require_all_sigma_points_in_image (bool): Discard a Gaussian unless every sigma point
            projects inside the (margin-extended) image.

    Returns:
        radii (torch.Tensor): Per-axis projected radii, shape ``[C, N, 2]``, ``int32``.
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        depths (torch.Tensor): View-space depths, shape ``[C, N]``.
        conics (torch.Tensor): Inverse 2D covariances as ``(a, b, c)``, shape ``[C, N, 3]``.
        compensations (torch.Tensor | None): Compensation factors, shape ``[C, N]``, or ``None`` when
            ``calc_compensations`` is ``False``.
    """
    return _fvdb_cpp.project_gaussians_ut_fwd(
        means,
        quats,
        log_scales,
        world_to_cam_matrices_start,
        world_to_cam_matrices_end,
        projection_matrices,
        distortion_coeffs,
        _to_cpp_camera_model(camera_model),
        image_width,
        image_height,
        eps2d,
        near,
        far,
        min_radius_2d,
        calc_compensations,
        _to_cpp_rolling_shutter(rolling_shutter_type),
        ut_alpha,
        ut_beta,
        ut_kappa,
        ut_in_image_margin,
        ut_require_all_sigma_points_in_image,
    )


# ---------------------------------------------------------------------------
#  Spherical harmonics
# ---------------------------------------------------------------------------


def evaluate_spherical_harmonics_fwd(
    sh_degree_to_use: int,
    num_cameras: int,
    means: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    camera_ids: torch.Tensor,
    gaussian_ids: torch.Tensor,
    sh0_coeffs: torch.Tensor,
    sh_n_coeffs: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Evaluate view-dependent spherical harmonics into per-camera, per-Gaussian features.

    Dense mode: ``camera_ids`` and ``gaussian_ids`` are empty, every Gaussian is evaluated for every
    camera, and coefficients are indexed by Gaussian. Packed mode: ``num_cameras`` must be ``1`` and
    the inputs describe ``M`` work items directly. ``camera_ids[i]`` and ``gaussian_ids[i]`` select the
    camera matrix and the mean used for work item ``i``'s view direction, while ``sh0_coeffs[i]``,
    ``sh_n_coeffs[i]`` and ``radii[0, i]`` are read by work item, so coefficients must already be
    gathered into work-item order. Work items whose ``radii`` are not positive on both axes yield
    zero features.

    Args:
        sh_degree_to_use (int): Highest SH degree to evaluate, ``0`` to ``3``.
        num_cameras (int): Number of cameras ``C`` in dense mode; must be ``1`` in packed mode.
        means (torch.Tensor): Gaussian centers in world space, shape ``[N, 3]``.
        world_to_cam_matrices (torch.Tensor): Rigid world-to-camera transforms, shape ``[C, 4, 4]``
            (any number of matrices in packed mode, indexed by ``camera_ids``).
        camera_ids (torch.Tensor): Packed-mode camera matrix index per work item, ``int32`` shape
            ``[M]``, or an empty tensor.
        gaussian_ids (torch.Tensor): Packed-mode index into ``means`` per work item, ``int32`` shape
            ``[M]``, or an empty tensor.
        sh0_coeffs (torch.Tensor): Degree-0 coefficients, shape ``[N, 1, D]`` dense or ``[M, 1, D]``
            packed.
        sh_n_coeffs (torch.Tensor): Higher-degree coefficients, shape ``[N, K-1, D]`` dense or
            ``[M, K-1, D]`` packed, with ``K = (sh_degree_to_use + 1)**2``, or an empty tensor when
            ``sh_degree_to_use == 0``.
        radii (torch.Tensor): Projected per-axis radii, shape ``[C, N, 2]`` dense or ``[1, M, 2]``
            packed.

    Returns:
        features (torch.Tensor): Evaluated features, shape ``[C, N, D]`` dense or ``[1, M, D]`` packed.
    """
    return _fvdb_cpp.evaluate_spherical_harmonics_fwd(
        sh_degree_to_use,
        num_cameras,
        means,
        world_to_cam_matrices,
        camera_ids,
        gaussian_ids,
        sh0_coeffs,
        sh_n_coeffs,
        radii,
    )


def evaluate_spherical_harmonics_bwd(
    sh_degree_to_use: int,
    num_cameras: int,
    num_gaussians: int,
    means: torch.Tensor,
    world_to_cam_matrices: torch.Tensor,
    camera_ids: torch.Tensor,
    gaussian_ids: torch.Tensor,
    sh_n_coeffs: torch.Tensor,
    d_loss_d_colors: torch.Tensor,
    radii: torch.Tensor,
    compute_d_loss_d_means: bool,
    compute_d_loss_d_world_to_cam_matrices: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Backward pass of :func:`evaluate_spherical_harmonics_fwd`.

    In packed mode the same work-item layout applies: coefficient gradients come back in work-item
    order (``[M, ...]``), while ``d_loss_d_means`` and ``d_loss_d_world_to_cam_matrices`` are
    scattered onto the ``means`` rows and camera matrices selected by ``gaussian_ids`` and
    ``camera_ids``.

    Args:
        sh_degree_to_use (int): SH degree used in the forward pass.
        num_cameras (int): Number of cameras ``C``; ``1`` in packed mode.
        num_gaussians (int): Number of Gaussians ``N`` in dense mode, or work items ``M`` in packed mode.
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        world_to_cam_matrices (torch.Tensor): Rigid world-to-camera transforms, shape ``[C, 4, 4]``.
        camera_ids (torch.Tensor): Packed-mode camera matrix indices, ``int32`` shape ``[M]``, or empty.
        gaussian_ids (torch.Tensor): Packed-mode indices into ``means``, ``int32`` shape ``[M]``, or empty.
        sh_n_coeffs (torch.Tensor): Higher-degree coefficients used in the forward pass.
        d_loss_d_colors (torch.Tensor): Loss gradient w.r.t. the forward features, shape ``[C, N, D]``
            dense or ``[1, M, D]`` packed.
        radii (torch.Tensor): Projected per-axis radii used in the forward pass, shape ``[C, N, 2]``
            dense or ``[1, M, 2]`` packed.
        compute_d_loss_d_means (bool): Whether to compute the gradient w.r.t. ``means``.
        compute_d_loss_d_world_to_cam_matrices (bool): Whether to compute the gradient w.r.t.
            ``world_to_cam_matrices``.

    Returns:
        d_loss_d_sh0_coeffs (torch.Tensor): Gradient w.r.t. the degree-0 coefficients, shape ``[N, 1, D]``.
        d_loss_d_sh_n_coeffs (torch.Tensor | None): Gradient w.r.t. the higher-degree coefficients,
            shape ``[N, K-1, D]``, or ``None`` when ``sh_degree_to_use == 0``.
        d_loss_d_means (torch.Tensor | None): Gradient w.r.t. ``means``, shape ``[N, 3]``, or ``None``
            when not requested.
        d_loss_d_world_to_cam_matrices (torch.Tensor | None): Gradient w.r.t. the camera matrices,
            shape ``[C, 4, 4]``, or ``None`` when not requested.
    """
    return _fvdb_cpp.evaluate_spherical_harmonics_bwd(
        sh_degree_to_use,
        num_cameras,
        num_gaussians,
        means,
        world_to_cam_matrices,
        camera_ids,
        gaussian_ids,
        sh_n_coeffs,
        d_loss_d_colors,
        radii,
        compute_d_loss_d_means,
        compute_d_loss_d_world_to_cam_matrices,
    )


# ---------------------------------------------------------------------------
#  Tile intersection
# ---------------------------------------------------------------------------


def intersect_gaussian_tiles(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    num_cameras: int,
    tile_size: int,
    num_tiles_h: int,
    num_tiles_w: int,
    camera_ids: torch.Tensor | None = None,
    conics: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin projected Gaussians into image tiles, sorted by camera, tile and depth.

    Inputs may be dense (``[C, N, ...]``) or packed (``[M, ...]`` with ``camera_ids``). Passing
    ``conics`` and ``opacities`` enables a tighter per-tile culling test.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]`` or ``[M, 2]``.
        radii (torch.Tensor): Per-axis projected radii, shape ``[C, N, 2]`` or ``[M, 2]``.
        depths (torch.Tensor): View-space depths, shape ``[C, N]`` or ``[M]``.
        num_cameras (int): Number of cameras ``C``.
        tile_size (int): Tile side length in pixels.
        num_tiles_h (int): Number of tiles along the image height.
        num_tiles_w (int): Number of tiles along the image width.
        camera_ids (torch.Tensor | None): Packed-mode camera index per Gaussian, shape ``[M]``.
        conics (torch.Tensor | None): Inverse 2D covariances for tighter culling, shape matching ``means2d``.
        opacities (torch.Tensor | None): Opacities for tighter culling, shape matching ``depths``.

    Returns:
        tile_offsets (torch.Tensor): Start index into ``tile_gaussian_ids`` per tile, shape
            ``[C, num_tiles_h, num_tiles_w]``.
        tile_gaussian_ids (torch.Tensor): Flattened Gaussian index per intersection, shape ``[n_isects]``.
    """
    return _fvdb_cpp.intersect_gaussian_tiles(
        means2d,
        radii,
        depths,
        num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        camera_ids,
        conics,
        opacities,
    )


def intersect_gaussian_tiles_sparse(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    tile_mask: torch.Tensor,
    active_tiles: torch.Tensor,
    num_cameras: int,
    tile_size: int,
    num_tiles_h: int,
    num_tiles_w: int,
    camera_ids: torch.Tensor | None = None,
    conics: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin projected Gaussians into the active tiles of a sparse pixel set.

    ``tile_mask`` and ``active_tiles`` come from :func:`build_sparse_gaussian_tile_layout`.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]`` or ``[M, 2]``.
        radii (torch.Tensor): Per-axis projected radii, shape ``[C, N, 2]`` or ``[M, 2]``.
        depths (torch.Tensor): View-space depths, shape ``[C, N]`` or ``[M]``.
        tile_mask (torch.Tensor): Boolean mask of active tiles, shape ``[C, num_tiles_h, num_tiles_w]``.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        num_cameras (int): Number of cameras ``C``.
        tile_size (int): Tile side length in pixels.
        num_tiles_h (int): Number of tiles along the image height.
        num_tiles_w (int): Number of tiles along the image width.
        camera_ids (torch.Tensor | None): Packed-mode camera index per Gaussian, shape ``[M]``.
        conics (torch.Tensor | None): Inverse 2D covariances for tighter culling.
        opacities (torch.Tensor | None): Opacities for tighter culling.

    Returns:
        tile_offsets (torch.Tensor): Start index into ``tile_gaussian_ids`` per active tile, with a
            trailing end offset, shape ``[AT + 1]``.
        tile_gaussian_ids (torch.Tensor): Flattened Gaussian index per intersection, shape ``[n_isects]``.
    """
    return _fvdb_cpp.intersect_gaussian_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask,
        active_tiles,
        num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        camera_ids,
        conics,
        opacities,
    )


def build_sparse_gaussian_tile_layout(
    tile_size: int,
    num_tiles_h: int,
    num_tiles_w: int,
    pixels_to_render: JaggedTensor | torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the tile bookkeeping needed to rasterize an arbitrary set of pixels.

    Coordinates must lie inside the ``image_height`` by ``image_width`` image and each pixel may
    appear only once per camera; the kernel raises ``ValueError`` otherwise. The sparse rasterizers
    trust the layout and do not check pixel bounds again. Let ``AT``
    be the number of tiles that contain at least one requested pixel and ``AP`` the total pixel count.
    An empty selection yields ``AT = 0`` outputs of the same dtypes.

    Args:
        tile_size (int): Tile side length in pixels. Must be ``16``.
        num_tiles_h (int): Number of tiles along the image height.
        num_tiles_w (int): Number of tiles along the image width.
        pixels_to_render (JaggedTensor | torch.Tensor): Integer ``(row, col)`` pixel coordinates, one
            ``[P_c, 2]`` list per camera (``int32`` or ``int64``).
        image_width (int): Image width in pixels. Must fit the tile grid, i.e.
            ``num_tiles_w == ceil(image_width / tile_size)``.
        image_height (int): Image height in pixels. Must fit the tile grid, i.e.
            ``num_tiles_h == ceil(image_height / tile_size)``.

    Returns:
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        active_tile_mask (torch.Tensor): Boolean mask of active tiles, shape ``[C, num_tiles_h, num_tiles_w]``.
        tile_pixel_mask (torch.Tensor): Per-tile ``uint64`` bitmask of requested pixels in raster order,
            shape ``[AT, words_per_tile]``.
        tile_pixel_cumsum (torch.Tensor): Inclusive cumulative count of requested pixels per active
            tile, shape ``[AT]``.
        pixel_map (torch.Tensor): Output slot for the ``k``-th requested pixel of each active tile,
            shape ``[AP]``.
    """
    _check_sparse_tile_size(tile_size)
    if num_tiles_h <= 0 or num_tiles_w <= 0:
        raise ValueError(f"num_tiles_h and num_tiles_w must be positive, got {num_tiles_h} and {num_tiles_w}")
    pixels = as_pixel_jagged(pixels_to_render)
    return _fvdb_cpp.build_sparse_gaussian_tile_layout(
        tile_size, num_tiles_w, num_tiles_h, pixels._impl, image_width, image_height
    )


# ---------------------------------------------------------------------------
#  Rasterization
# ---------------------------------------------------------------------------


def rasterize_screen_space_gaussians_fwd(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    num_shared_channels_override: int = -1,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha-blend projected 2D Gaussians into dense images.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances as ``(a, b, c)``, shape ``[C, N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features to blend, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles`.
        num_shared_channels_override (int): Shared-memory channel count, one of ``16``, ``32`` or
            ``64``, or ``-1`` to pick automatically.
        backgrounds (torch.Tensor | None): Per-camera background features, shape ``[C, D]``.
        masks (torch.Tensor | None): Per-tile boolean render mask, shape ``[C, num_tiles_h, num_tiles_w]``.

    Returns:
        rendered_features (torch.Tensor): Blended features, shape ``[C, image_height, image_width, D]``.
        rendered_alphas (torch.Tensor): Accumulated alpha, shape ``[C, image_height, image_width, 1]``.
        last_ids (torch.Tensor): Tile-relative index of the last blended intersection per pixel, or
            ``-1``, shape ``[C, image_height, image_width]``.
    """
    return _fvdb_cpp.rasterize_screen_space_gaussians_fwd(
        means2d,
        conics,
        features,
        opacities,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        num_shared_channels_override,
        backgrounds,
        masks,
    )


def rasterize_screen_space_gaussians_bwd(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    rendered_alphas: torch.Tensor,
    last_ids: torch.Tensor,
    d_loss_d_rendered_features: torch.Tensor,
    d_loss_d_rendered_alphas: torch.Tensor,
    abs_grad: bool,
    num_shared_channels_override: int = -1,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of :func:`rasterize_screen_space_gaussians_fwd`.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        tile_offsets (torch.Tensor): Per-tile start offsets used in the forward pass.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids used in the forward pass.
        rendered_alphas (torch.Tensor): Forward alphas, shape ``[C, image_height, image_width, 1]``.
        last_ids (torch.Tensor): Forward last-intersection indices, shape ``[C, image_height, image_width]``.
        d_loss_d_rendered_features (torch.Tensor): Loss gradient w.r.t. the rendered features.
        d_loss_d_rendered_alphas (torch.Tensor): Loss gradient w.r.t. the rendered alphas.
        abs_grad (bool): Also accumulate the absolute value of the 2D mean gradient.
        num_shared_channels_override (int): Shared-memory channel count, or ``-1`` for automatic.
        backgrounds (torch.Tensor | None): Per-camera background features used in the forward pass.
        masks (torch.Tensor | None): Per-tile boolean render mask used in the forward pass.

    Returns:
        d_loss_d_means2d_abs (torch.Tensor | None): Absolute 2D mean gradient, shape ``[C, N, 2]``, or
            ``None`` when ``abs_grad`` is ``False``.
        d_loss_d_means2d (torch.Tensor): Gradient w.r.t. ``means2d``, shape ``[C, N, 2]``.
        d_loss_d_conics (torch.Tensor): Gradient w.r.t. ``conics``, shape ``[C, N, 3]``.
        d_loss_d_features (torch.Tensor): Gradient w.r.t. ``features``, shape ``[C, N, D]``.
        d_loss_d_opacities (torch.Tensor): Gradient w.r.t. ``opacities``, shape ``[C, N]``.
    """
    return _fvdb_cpp.rasterize_screen_space_gaussians_bwd(
        means2d,
        conics,
        features,
        opacities,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        rendered_alphas,
        last_ids,
        d_loss_d_rendered_features,
        d_loss_d_rendered_alphas,
        abs_grad,
        num_shared_channels_override,
        backgrounds,
        masks,
    )


def rasterize_screen_space_gaussians_sparse_fwd(
    pixels_to_render: JaggedTensor | torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_pixel_mask: torch.Tensor,
    tile_pixel_cumsum: torch.Tensor,
    pixel_map: torch.Tensor,
    num_shared_channels_override: int = -1,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
    """Alpha-blend projected 2D Gaussians at an arbitrary set of pixels.

    The tile bookkeeping arguments come from :func:`build_sparse_gaussian_tile_layout` and the
    intersections from :func:`intersect_gaussian_tiles_sparse`. Outputs are jagged with the same
    per-camera structure as ``pixels_to_render``.

    Args:
        pixels_to_render (JaggedTensor | torch.Tensor): Integer ``(row, col)`` pixel coordinates, one
            list per camera.
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length. Must be ``16``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles_sparse`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles_sparse`.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        tile_pixel_mask (torch.Tensor): Per-tile bitmask of requested pixels, shape ``[AT, words_per_tile]``.
        tile_pixel_cumsum (torch.Tensor): Cumulative requested-pixel count per active tile, shape ``[AT]``.
        pixel_map (torch.Tensor): Output slot per requested pixel, shape ``[AP]``.
        num_shared_channels_override (int): Shared-memory channel count, or ``-1`` for automatic.
        backgrounds (torch.Tensor | None): Per-camera background features, shape ``[C, D]``.
        masks (torch.Tensor | None): Per-tile boolean render mask, shape ``[C, num_tiles_h, num_tiles_w]``.

    Returns:
        rendered_features (JaggedTensor): Blended features per requested pixel, element shape ``[D]``.
        rendered_alphas (JaggedTensor): Accumulated alpha per requested pixel, element shape ``[1]``.
        last_ids (JaggedTensor): Tile-relative index of the last blended intersection per requested
            pixel, or ``-1``.
    """
    pixels, empty = _sparse_prologue(tile_size, pixels_to_render, active_tiles, pixel_map)
    if empty:
        return (
            _empty_like_pixels(pixels, (features.shape[-1],), features.dtype),
            _empty_like_pixels(pixels, (1,), features.dtype),
            _empty_like_pixels(pixels, (), torch.int32),
        )
    result = _fvdb_cpp.rasterize_screen_space_gaussians_sparse_fwd(
        pixels._impl,
        means2d,
        conics,
        features,
        opacities,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        num_shared_channels_override,
        backgrounds,
        masks,
    )
    return _wrap(result[0]), _wrap(result[1]), _wrap(result[2])


def rasterize_screen_space_gaussians_sparse_bwd(
    pixels_to_render: JaggedTensor | torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    rendered_alphas: JaggedTensor,
    last_ids: JaggedTensor,
    d_loss_d_rendered_features: JaggedTensor,
    d_loss_d_rendered_alphas: JaggedTensor,
    active_tiles: torch.Tensor,
    tile_pixel_mask: torch.Tensor,
    tile_pixel_cumsum: torch.Tensor,
    pixel_map: torch.Tensor,
    abs_grad: bool,
    num_shared_channels_override: int = -1,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of :func:`rasterize_screen_space_gaussians_sparse_fwd`.

    Args:
        pixels_to_render (JaggedTensor | torch.Tensor): Pixel coordinates used in the forward pass.
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length. Must be ``16``.
        tile_offsets (torch.Tensor): Per-tile start offsets used in the forward pass.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids used in the forward pass.
        rendered_alphas (JaggedTensor): Forward alphas per requested pixel.
        last_ids (JaggedTensor): Forward last-intersection indices per requested pixel.
        d_loss_d_rendered_features (JaggedTensor): Loss gradient w.r.t. the rendered features, jagged
            like ``pixels_to_render``.
        d_loss_d_rendered_alphas (JaggedTensor): Loss gradient w.r.t. the rendered alphas, jagged like
            ``pixels_to_render``.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        tile_pixel_mask (torch.Tensor): Per-tile bitmask of requested pixels.
        tile_pixel_cumsum (torch.Tensor): Cumulative requested-pixel count per active tile.
        pixel_map (torch.Tensor): Output slot per requested pixel.
        abs_grad (bool): Also accumulate the absolute value of the 2D mean gradient.
        num_shared_channels_override (int): Shared-memory channel count, or ``-1`` for automatic.
        backgrounds (torch.Tensor | None): Per-camera background features used in the forward pass.
        masks (torch.Tensor | None): Per-tile boolean render mask used in the forward pass.

    Returns:
        d_loss_d_means2d_abs (torch.Tensor | None): Absolute 2D mean gradient, shape ``[C, N, 2]``, or
            ``None`` when ``abs_grad`` is ``False``.
        d_loss_d_means2d (torch.Tensor): Gradient w.r.t. ``means2d``, shape ``[C, N, 2]``.
        d_loss_d_conics (torch.Tensor): Gradient w.r.t. ``conics``, shape ``[C, N, 3]``.
        d_loss_d_features (torch.Tensor): Gradient w.r.t. ``features``, shape ``[C, N, D]``.
        d_loss_d_opacities (torch.Tensor): Gradient w.r.t. ``opacities``, shape ``[C, N]``.
    """
    pixels, empty = _sparse_prologue(tile_size, pixels_to_render, active_tiles, pixel_map)
    jagged_args = (
        _jagged_impl(rendered_alphas, "rendered_alphas"),
        _jagged_impl(last_ids, "last_ids"),
        _jagged_impl(d_loss_d_rendered_features, "d_loss_d_rendered_features"),
        _jagged_impl(d_loss_d_rendered_alphas, "d_loss_d_rendered_alphas"),
    )
    if empty:
        return (
            torch.zeros_like(means2d) if abs_grad else None,
            torch.zeros_like(means2d),
            torch.zeros_like(conics),
            torch.zeros_like(features),
            torch.zeros_like(opacities),
        )
    return _fvdb_cpp.rasterize_screen_space_gaussians_sparse_bwd(
        pixels._impl,
        means2d,
        conics,
        features,
        opacities,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        *jagged_args,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        abs_grad,
        num_shared_channels_override,
        backgrounds,
        masks,
    )


def rasterize_world_space_gaussians_fwd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    world_to_cam_matrices_start: torch.Tensor,
    world_to_cam_matrices_end: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    rolling_shutter_type: RollingShutterType | int,
    camera_model: CameraModel | int,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize 3D Gaussians directly by evaluating them along per-pixel rays.

    This is the dense rendering path for the unscented-transform pipeline. Tile intersections are
    still computed from a 2D projection, for example :func:`project_gaussians_ut_fwd` followed by
    :func:`intersect_gaussian_tiles`.

    Args:
        means (torch.Tensor): Gaussian centers in world space, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        world_to_cam_matrices_start (torch.Tensor): World-to-camera transforms at exposure start, shape ``[C, 4, 4]``.
        world_to_cam_matrices_end (torch.Tensor): World-to-camera transforms at exposure end, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        distortion_coeffs (torch.Tensor): Packed OpenCV distortion coefficients, shape ``[C, 12]`` or ``[C, 0]``.
        rolling_shutter_type (RollingShutterType | int): Rolling shutter policy.
        camera_model (CameraModel | int): Camera model used for ray generation.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles`.
        backgrounds (torch.Tensor | None): Per-camera background features, shape ``[C, D]``.
        masks (torch.Tensor | None): Per-tile boolean render mask, shape ``[C, num_tiles_h, num_tiles_w]``.

    Returns:
        rendered_features (torch.Tensor): Blended features, shape ``[C, image_height, image_width, D]``.
        rendered_alphas (torch.Tensor): Accumulated alpha, shape ``[C, image_height, image_width, 1]``.
        last_ids (torch.Tensor): Tile-relative index of the last blended intersection per pixel, or
            ``-1``, shape ``[C, image_height, image_width]``.
    """
    return _fvdb_cpp.rasterize_world_space_gaussians_fwd(
        means,
        quats,
        log_scales,
        features,
        opacities,
        world_to_cam_matrices_start,
        world_to_cam_matrices_end,
        projection_matrices,
        distortion_coeffs,
        _to_cpp_rolling_shutter(rolling_shutter_type),
        _to_cpp_camera_model(camera_model),
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        backgrounds,
        masks,
    )


def rasterize_world_space_gaussians_bwd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    features: torch.Tensor,
    opacities: torch.Tensor,
    world_to_cam_matrices_start: torch.Tensor,
    world_to_cam_matrices_end: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    rolling_shutter_type: RollingShutterType | int,
    camera_model: CameraModel | int,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    rendered_alphas: torch.Tensor,
    last_ids: torch.Tensor,
    d_loss_d_rendered_features: torch.Tensor,
    d_loss_d_rendered_alphas: torch.Tensor,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of :func:`rasterize_world_space_gaussians_fwd`.

    Args:
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        features (torch.Tensor): Per-camera, per-Gaussian features, shape ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        world_to_cam_matrices_start (torch.Tensor): World-to-camera transforms at exposure start, shape ``[C, 4, 4]``.
        world_to_cam_matrices_end (torch.Tensor): World-to-camera transforms at exposure end, shape ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, shape ``[C, 3, 3]``.
        distortion_coeffs (torch.Tensor): Packed OpenCV distortion coefficients, shape ``[C, 12]`` or ``[C, 0]``.
        rolling_shutter_type (RollingShutterType | int): Rolling shutter policy used in the forward pass.
        camera_model (CameraModel | int): Camera model used in the forward pass.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        tile_offsets (torch.Tensor): Per-tile start offsets used in the forward pass.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids used in the forward pass.
        rendered_alphas (torch.Tensor): Forward alphas, shape ``[C, image_height, image_width, 1]``.
        last_ids (torch.Tensor): Forward last-intersection indices, shape ``[C, image_height, image_width]``.
        d_loss_d_rendered_features (torch.Tensor): Loss gradient w.r.t. the rendered features.
        d_loss_d_rendered_alphas (torch.Tensor): Loss gradient w.r.t. the rendered alphas.
        backgrounds (torch.Tensor | None): Per-camera background features used in the forward pass.
        masks (torch.Tensor | None): Per-tile boolean render mask used in the forward pass.

    Returns:
        d_loss_d_means (torch.Tensor): Gradient w.r.t. ``means``, shape ``[N, 3]``.
        d_loss_d_quats (torch.Tensor): Gradient w.r.t. ``quats``, shape ``[N, 4]``.
        d_loss_d_log_scales (torch.Tensor): Gradient w.r.t. ``log_scales``, shape ``[N, 3]``.
        d_loss_d_features (torch.Tensor): Gradient w.r.t. ``features``, shape ``[C, N, D]``.
        d_loss_d_opacities (torch.Tensor): Gradient w.r.t. ``opacities``, shape ``[C, N]``.
    """
    return _fvdb_cpp.rasterize_world_space_gaussians_bwd(
        means,
        quats,
        log_scales,
        features,
        opacities,
        world_to_cam_matrices_start,
        world_to_cam_matrices_end,
        projection_matrices,
        distortion_coeffs,
        _to_cpp_rolling_shutter(rolling_shutter_type),
        _to_cpp_camera_model(camera_model),
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        tile_offsets,
        tile_gaussian_ids,
        rendered_alphas,
        last_ids,
        d_loss_d_rendered_features,
        d_loss_d_rendered_alphas,
        backgrounds,
        masks,
    )


# ---------------------------------------------------------------------------
#  Analysis (non-differentiable)
# ---------------------------------------------------------------------------


def rasterize_num_contributing_gaussians(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the Gaussians that contribute non-negligible opacity to each pixel.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles`.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.

    Returns:
        num_contributing (torch.Tensor): Contributing Gaussian count per pixel, shape
            ``[C, image_height, image_width]``, ``int32``.
        alphas (torch.Tensor): Accumulated alpha per pixel, shape ``[C, image_height, image_width]``.
    """
    return _fvdb_cpp.rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
    )


def rasterize_num_contributing_gaussians_sparse(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    pixels_to_render: JaggedTensor | torch.Tensor,
    active_tiles: torch.Tensor,
    tile_pixel_mask: torch.Tensor,
    tile_pixel_cumsum: torch.Tensor,
    pixel_map: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Count contributing Gaussians at an arbitrary set of pixels.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles_sparse`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles_sparse`.
        pixels_to_render (JaggedTensor | torch.Tensor): Integer ``(row, col)`` pixel coordinates, one
            list per camera.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        tile_pixel_mask (torch.Tensor): Per-tile bitmask of requested pixels.
        tile_pixel_cumsum (torch.Tensor): Cumulative requested-pixel count per active tile.
        pixel_map (torch.Tensor): Output slot per requested pixel.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length. Must be ``16``.

    Returns:
        num_contributing (JaggedTensor): Contributing Gaussian count per requested pixel, ``int32``.
        alphas (JaggedTensor): Accumulated alpha per requested pixel, one scalar per pixel.
    """
    pixels, empty = _sparse_prologue(tile_size, pixels_to_render, active_tiles, pixel_map)
    if empty:
        return _empty_like_pixels(pixels, (), torch.int32), _empty_like_pixels(pixels, (), opacities.dtype)
    result = _fvdb_cpp.rasterize_num_contributing_gaussians_sparse(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        pixels._impl,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
    )
    return _wrap(result[0]), _wrap(result[1])


def rasterize_contributing_gaussian_ids(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    num_depth_samples: int,
    num_contributing_gaussians: torch.Tensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """List the Gaussians that contribute to each pixel, front to back, with their blend weights.

    ``num_depth_samples`` selects the mode. A positive value records at most that many of the most
    visible contributors per pixel and ignores ``num_contributing_gaussians``. Zero or a negative
    value records every contributor and then requires ``num_contributing_gaussians``, the counts
    from :func:`rasterize_num_contributing_gaussians`, to size the output.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles`.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        num_depth_samples (int): Top-K mode when positive; all-contributors mode when ``<= 0``.
        num_contributing_gaussians (torch.Tensor | None): Per-pixel counts, shape
            ``[C, image_height, image_width]``. Required in all-contributors mode, ignored otherwise.

    Returns:
        gaussian_ids (JaggedTensor): Contributing Gaussian indices per pixel, ``int32``.
        weights (JaggedTensor): Blend weight of each listed Gaussian.
    """
    result = _fvdb_cpp.rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        num_depth_samples,
        num_contributing_gaussians,
    )
    return _wrap(result[0]), _wrap(result[1])


def rasterize_contributing_gaussian_ids_sparse(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    pixels_to_render: JaggedTensor | torch.Tensor,
    active_tiles: torch.Tensor,
    tile_pixel_mask: torch.Tensor,
    tile_pixel_cumsum: torch.Tensor,
    pixel_map: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    num_depth_samples: int,
    num_contributing_gaussians: JaggedTensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """List contributing Gaussians, with blend weights, at an arbitrary set of pixels.

    Mode selection follows :func:`rasterize_contributing_gaussian_ids`: a positive
    ``num_depth_samples`` keeps the top-K contributors per pixel, while ``<= 0`` returns every
    contributor and requires ``num_contributing_gaussians``.

    The results nest cameras, then pixels, then contributors (``ldim == 2``). A selection with no
    pixels cannot be expressed that way, so an empty layout returns ``ldim == 1`` results with one
    empty list per camera; :func:`fvdb.jcat` will not mix the two forms.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles_sparse`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles_sparse`.
        pixels_to_render (JaggedTensor | torch.Tensor): Integer ``(row, col)`` pixel coordinates, one
            list per camera.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        tile_pixel_mask (torch.Tensor): Per-tile bitmask of requested pixels.
        tile_pixel_cumsum (torch.Tensor): Cumulative requested-pixel count per active tile.
        pixel_map (torch.Tensor): Output slot per requested pixel.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length. Must be ``16``.
        num_depth_samples (int): Top-K mode when positive; all-contributors mode when ``<= 0``.
        num_contributing_gaussians (JaggedTensor | None): Per-pixel counts from
            :func:`rasterize_num_contributing_gaussians_sparse`. Required in all-contributors mode,
            ignored otherwise.

    Returns:
        gaussian_ids (JaggedTensor): Contributing Gaussian indices per requested pixel, ``int32``.
        weights (JaggedTensor): Blend weight of each listed Gaussian.
    """
    pixels, empty = _sparse_prologue(tile_size, pixels_to_render, active_tiles, pixel_map)
    counts = (
        None
        if num_contributing_gaussians is None
        else _jagged_impl(num_contributing_gaussians, "num_contributing_gaussians")
    )
    if empty:
        return _empty_like_pixels(pixels, (), torch.int32), _empty_like_pixels(pixels, (), opacities.dtype)
    result = _fvdb_cpp.rasterize_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        pixels._impl,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        num_depth_samples,
        counts,
    )
    return _wrap(result[0]), _wrap(result[1])


def rasterize_top_contributing_gaussian_ids(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    num_depth_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Record the ``num_depth_samples`` most visible Gaussians per pixel with their blend weights.

    Slots without a contributor hold id ``-1`` and weight ``0``.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles`.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length used for ``tile_offsets``.
        num_depth_samples (int): Number of contributors recorded per pixel.

    Returns:
        gaussian_ids (torch.Tensor): Top contributor indices, shape
            ``[C, image_height, image_width, num_depth_samples]``, ``int32``.
        weights (torch.Tensor): Blend weight of each recorded contributor, same shape.
    """
    return _fvdb_cpp.rasterize_top_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        num_depth_samples,
    )


def rasterize_top_contributing_gaussian_ids_sparse(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_gaussian_ids: torch.Tensor,
    pixels_to_render: JaggedTensor | torch.Tensor,
    active_tiles: torch.Tensor,
    tile_pixel_mask: torch.Tensor,
    tile_pixel_cumsum: torch.Tensor,
    pixel_map: torch.Tensor,
    image_width: int,
    image_height: int,
    image_origin_w: int,
    image_origin_h: int,
    tile_size: int,
    num_depth_samples: int,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Record the most visible Gaussians, with blend weights, at an arbitrary set of pixels.

    Args:
        means2d (torch.Tensor): Projected 2D centers, shape ``[C, N, 2]``.
        conics (torch.Tensor): Inverse 2D covariances, shape ``[C, N, 3]``.
        opacities (torch.Tensor): Per-camera, per-Gaussian opacities, shape ``[C, N]``.
        tile_offsets (torch.Tensor): Per-tile start offsets from :func:`intersect_gaussian_tiles_sparse`.
        tile_gaussian_ids (torch.Tensor): Per-intersection Gaussian ids from :func:`intersect_gaussian_tiles_sparse`.
        pixels_to_render (JaggedTensor | torch.Tensor): Integer ``(row, col)`` pixel coordinates, one
            list per camera.
        active_tiles (torch.Tensor): Flattened indices of the active tiles, shape ``[AT]``.
        tile_pixel_mask (torch.Tensor): Per-tile bitmask of requested pixels.
        tile_pixel_cumsum (torch.Tensor): Cumulative requested-pixel count per active tile.
        pixel_map (torch.Tensor): Output slot per requested pixel.
        image_width (int): Width of the render window in pixels.
        image_height (int): Height of the render window in pixels.
        image_origin_w (int): Horizontal pixel offset of the render window.
        image_origin_h (int): Vertical pixel offset of the render window.
        tile_size (int): Tile side length. Must be ``16``.
        num_depth_samples (int): Number of contributors recorded per pixel.

    Returns:
        gaussian_ids (JaggedTensor): Top contributor indices per requested pixel, element shape
            ``[num_depth_samples]``, ``int32``.
        weights (JaggedTensor): Blend weight of each recorded contributor, same structure.
    """
    pixels, empty = _sparse_prologue(tile_size, pixels_to_render, active_tiles, pixel_map)
    if empty:
        return (
            _empty_like_pixels(pixels, (num_depth_samples,), torch.int32),
            _empty_like_pixels(pixels, (num_depth_samples,), opacities.dtype),
        )
    result = _fvdb_cpp.rasterize_top_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        tile_offsets,
        tile_gaussian_ids,
        pixels._impl,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        image_origin_w,
        image_origin_h,
        tile_size,
        num_depth_samples,
    )
    return _wrap(result[0]), _wrap(result[1])


# ---------------------------------------------------------------------------
#  MCMC densification
# ---------------------------------------------------------------------------


def mcmc_relocate_gaussians(
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    ratios: torch.Tensor,
    binomial_coeffs: torch.Tensor,
    n_max: int,
    min_opacity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the opacity and scale a Gaussian should take when split into ``ratios`` copies.

    Implements the relocation rule of MCMC-based densification so that the sum of the copies
    matches the original Gaussian's appearance.

    Args:
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        ratios (torch.Tensor): Replication count per Gaussian, ``int32`` shape ``[N]``.
        binomial_coeffs (torch.Tensor): Binomial coefficient table, shape ``[n_max, n_max]``.
        n_max (int): Largest supported replication count.
        min_opacity (float): Lower clamp applied to the relocated opacity.

    Returns:
        logit_opacities_new (torch.Tensor): Relocated logit opacities, shape ``[N]``.
        log_scales_new (torch.Tensor): Relocated log scales, shape ``[N, 3]``.
    """
    return _fvdb_cpp.mcmc_relocate_gaussians(log_scales, logit_opacities, ratios, binomial_coeffs, n_max, min_opacity)


def mcmc_add_noise_to_means(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    quats: torch.Tensor,
    noise_scale: float,
    t: float,
    k: float,
) -> None:
    """Perturb Gaussian centers in place with covariance-shaped noise, damped by opacity.

    Args:
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``, modified in place.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        noise_scale (float): Overall noise magnitude.
        t (float): Opacity threshold of the damping sigmoid.
        k (float): Sharpness of the damping sigmoid.
    """
    _fvdb_cpp.mcmc_add_noise_to_means(means, log_scales, logit_opacities, quats, noise_scale, t, k)


# ---------------------------------------------------------------------------
#  PLY I/O
# ---------------------------------------------------------------------------


def save_gaussian_ply(
    filename: str,
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save Gaussian parameters, and optional metadata, to a PLY file.

    Args:
        filename (str): Output path.
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        sh0 (torch.Tensor): Degree-0 SH coefficients, shape ``[N, 1, D]``.
        shN (torch.Tensor): Higher-degree SH coefficients, shape ``[N, K-1, D]``.
        metadata (Mapping[str, Any] | None): Extra key-value pairs to store. Values may be ``str``,
            ``int``, ``float`` or ``torch.Tensor``.
    """
    _fvdb_cpp.save_gaussian_ply(
        filename, means, quats, log_scales, logit_opacities, sh0, shN, None if metadata is None else dict(metadata)
    )


def load_gaussian_ply(
    filename: str,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load Gaussian parameters, and any stored metadata, from a PLY file.

    Args:
        filename (str): Path of the PLY file.
        device (torch.device | str): Device the loaded tensors are moved to.

    Returns:
        means (torch.Tensor): Gaussian centers, shape ``[N, 3]``.
        quats (torch.Tensor): Gaussian quaternions, shape ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, shape ``[N, 3]``.
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        sh0 (torch.Tensor): Degree-0 SH coefficients, shape ``[N, 1, D]``.
        shN (torch.Tensor): Higher-degree SH coefficients, shape ``[N, K-1, D]``.
        metadata (dict[str, Any]): Stored metadata, empty if the file has none.
    """
    return _fvdb_cpp.load_gaussian_ply(filename, torch.device(device))
