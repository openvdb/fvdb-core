// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSBACKWARD_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Spherical harmonics evaluation backward pass
///
/// This function computes the vector-Jacobian product between the output gradients and the
/// Jacobian of the spherical harmonics forward operation.
///
/// @param[in] shDegreeToUse Degree of spherical harmonics used in the forward pass
/// @param[in] numCameras Number of cameras used in the forward pass
/// @param[in] numGaussians Number of Gaussians used in the forward pass
/// @param[in] means Gaussian means in world space [G, 3], where G is N in dense mode
/// @param[in] worldToCamMatrices Rigid world-to-camera matrices [V, 4, 4], where V is C in dense
/// mode. The matrices must be rigid transforms with orthonormal rotation blocks.
/// @param[in] cameraIds Camera index for each packed work item, or empty for unpacked evaluation
/// @param[in] gaussianIds Gaussian index for each packed work item, or empty for unpacked
/// evaluation
/// @param[in] shNCoeffs Spherical harmonic coefficients [N, K-1, D] (packed) or [K-1, N, D]
/// (unpacked) where K depends on sh_degree_to_use
/// @param[in] dLossDColors Gradients of the loss function with respect to output colors [N, 3]
/// - ∂L/∂colors
/// @param[in] radii Per-axis projected radii [C, N, 2] used in the forward pass; a gaussian
/// is masked unless both axes are positive.
/// @param[in] computeDLossDMeans Whether to compute gradients with respect to means
/// @param[in] computeDLossDWorldToCamMatrices Whether to compute gradients with respect to
/// world-to-camera matrices
///
/// @return std::tuple containing gradients of the loss function with respect to:
///         - SH coefficients [N, K, 3] - ∂L/∂sh_coeffs
///         - Gaussian means [G, 3] - ∂L/∂means (if requested, otherwise empty tensor)
///         - World-to-camera matrices [V, 4, 4] - ∂L/∂viewmats (if requested, otherwise empty
///         tensor)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evaluateSphericalHarmonicsBwd(const int64_t shDegreeToUse,
                              const int64_t numCameras,
                              const int64_t numGaussians,
                              const torch::Tensor &means,
                              const torch::Tensor &worldToCamMatrices,
                              const torch::Tensor &cameraIds,
                              const torch::Tensor &gaussianIds,
                              const torch::Tensor &shNCoeffs,
                              const torch::Tensor &dLossDColors,
                              const torch::Tensor &radii,
                              const bool computeDLossDMeans,
                              const bool computeDLossDWorldToCamMatrices);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSBACKWARD_H
