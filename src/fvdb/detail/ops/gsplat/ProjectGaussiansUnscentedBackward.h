// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_PROJECTGAUSSIANSUNSCENTEDBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_PROJECTGAUSSIANSUNSCENTEDBACKWARD_H

#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Calculate gradients for the unscented-transform 3D-to-2D Gaussian projection (backward
/// pass), matching `projectGaussiansUnscentedFwd` in ProjectGaussiansUnscentedForward.h.
///
/// Context: `projectGaussiansUnscentedFwd` previously had no matching backward kernel at all --
/// `GaussianSplat3d._do_projection`'s branch for any camera model that requires distortion
/// handling (every COLMAP camera model except SIMPLE_PINHOLE/PINHOLE) called the forward op
/// directly instead of through a `torch.autograd.Function`, so the photometric/image loss
/// produced zero gradient on `means`/`quats`/`logScales` for such scenes. This function -- and
/// the `torch.autograd.Function` wrapper that should call it from the forward's saved context --
/// closes that gap.
///
/// The forward pass is cheap to recompute per-Gaussian (7 sigma points, a small number of scalar
/// operations), so -- matching the pattern `projectionBackwardKernel` in
/// ProjectGaussiansAnalyticBackward.cu already uses for the analytic path's camera-space
/// mean/covariance -- this backward kernel recomputes the sigma points, their projections, and
/// the 2D mean/covariance reconstruction locally rather than requiring them to be saved from the
/// forward pass. Only `radii` (to test which Gaussians were culled) and `conics` (the forward's
/// already-inverted, PSD-enforced 2D covariance, reused directly to avoid a redundant matrix
/// inversion) are taken as forward-output inputs, exactly as the analytic backward does.
///
/// Known scope limitations, called out explicitly rather than silently handled:
///  - Assumes `RollingShutterType::NONE` (`worldToCamMatricesStart == worldToCamMatricesEnd`).
///    The forward kernel supports rolling shutter; this backward does not yet.
///  - Does not compute a gradient w.r.t. `worldToCamMatrices` (no camera-pose-optimization
///    support on this path yet) or w.r.t. `distortionCoeffs` (calibration is not optimized by
///    any current caller). Both would follow the same pattern as `dLossDRotation`/
///    `dLossDTranslation` in the analytic backward and are natural follow-ups, not fundamental
///    limitations of the derivation.
///  - The eigenvector *directions* used to reconstruct the PSD-enforced 2D covariance from its
///    (clamped) eigenvalues are treated as constants w.r.t. this backward pass, matching the
///    reference Python implementation's (`_gaussian_unscented_torch.py`) documented choice: when
///    the two eigenvalues coincide (a near-circular projected covariance, routine for
///    near-isotropic Gaussians), the reconstructed matrix does not depend on which orthonormal
///    basis was chosen, so there is no meaningful direction-gradient to compute there, and
///    differentiating through the (otherwise divide-by-near-zero) direction formula is a real
///    numerical-stability hazard for no benefit. Gradient w.r.t. the eigenvalues themselves --
///    where the actual scale/orientation dependence needed for training lives -- is exact.
///
/// This derivation was validated end-to-end against `torch.autograd.grad` on
/// `project_gaussians_unscented_torch` (itself validated to sub-pixel agreement against this
/// same forward CUDA kernel) across 48 randomized configurations (6 seeds x 4 distortion-
/// coefficient sets x {typical, extreme} scale ranges, float64, several thousand Gaussian/camera
/// pairs total): worst observed relative error 4.3e-13, i.e. float64 rounding noise, not a
/// derivation error. See the fvdb-reality-capture PR/issue this patch accompanies for the
/// validation script.
///
/// @param[in] means 3D positions of Gaussians [N, 3]
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (w, x, y, z), not
/// necessarily normalized (normalization -- and its Jacobian -- is applied internally, matching
/// the forward pass and `quaternionToRotationMatrixVectorJacobianProduct`'s convention).
/// @param[in] logScales Log-scale factors of Gaussians [N, 3] (natural log).
/// @param[in] worldToCamMatrices Camera view matrices [C, 4, 4] (used for both the start and end
/// pose; see the RollingShutterType::NONE scope note above).
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3].
/// @param[in] cameraModel Camera/distortion model used in the forward pass.
/// @param[in] distortionCoeffs Packed distortion coefficients [C, 12] (or [C, 0] for
/// PINHOLE/ORTHOGRAPHIC), matching the forward pass's layout.
/// @param[in] imageWidth Width of the image in pixels.
/// @param[in] imageHeight Height of the image in pixels.
/// @param[in] eps2d 2D projection epsilon for numerical stability (forward-pass input).
/// @param[in] radii Radii from the forward pass [C, N, 2]; Gaussians with radiusX<=0 or
/// radiusY<=0 were culled and receive zero gradient.
/// @param[in] conics Conics from the forward pass [C, N, 3], packed as (a, b, c) for
/// `[[a,b],[b,c]]`.
/// @param[in] dLossDMeans2d Upstream gradient w.r.t. the projected 2D means [C, N, 2].
/// @param[in] dLossDDepths Upstream gradient w.r.t. the depths [C, N].
/// @param[in] dLossDConics Upstream gradient w.r.t. the conics [C, N, 3].
///
/// @return Tuple of (dL/dMeans [N,3], dL/dQuats [N,4], dL/dLogScales [N,3]).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectGaussiansUnscentedBwd(
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,               // [N, 4]
    const torch::Tensor &logScales,           // [N, 3]
    const torch::Tensor &worldToCamMatrices,  // [C, 4, 4]
    const torch::Tensor &projectionMatrices,  // [C, 3, 3]
    const DistortionModel cameraModel,
    const torch::Tensor &distortionCoeffs, // [C, 12] or [C, 0]
    const int64_t imageWidth,
    const int64_t imageHeight,
    const float eps2d,
    const UTParams &utParams,
    const torch::Tensor &radii,  // [C, N, 2]
    const torch::Tensor &conics, // [C, N, 3]
    const torch::Tensor &dLossDMeans2d, // [C, N, 2]
    const torch::Tensor &dLossDDepths,  // [C, N]
    const torch::Tensor &dLossDConics   // [C, N, 3]
);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_PROJECTGAUSSIANSUNSCENTEDBACKWARD_H
