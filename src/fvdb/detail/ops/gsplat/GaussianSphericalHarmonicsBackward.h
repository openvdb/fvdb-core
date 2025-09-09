// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSBACKWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSBACKWARD_H

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
/// @param[in] viewDirs Direction vectors [N, 3] (packed) or [C, N, 3] (unpacked) used in the
/// forward pass
/// @param[in] shNCoeffs Spherical harmonic coefficients [N, K-1, D] (packed) or [K-1, N, D]
/// (unpacked) where K depends on sh_degree_to_use
/// @param[in] dLossDColors Gradients of the loss function with respect to output colors [N, 3]
/// - ∂L/∂colors
/// @param[in] radii radii [N] (packed) or [C, N] (unpacked) used in the forward pass for
/// level-of-detail
/// @param[in] computeDLossDViewDirs Whether to compute gradients with respect to direction
/// vectors
///
/// @return std::tuple containing gradients of the loss function with respect to:
///         - SH coefficients [N, K, 3] - ∂L/∂sh_coeffs
///         - Direction vectors [N, 3] - ∂L/∂dirs (if compute_v_dirs is true, otherwise empty
///         tensor)
template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward(const int64_t shDegreeToUse,
                                   const int64_t numCameras,
                                   const int64_t numGaussians,
                                   const torch::Tensor &viewDirs,  // [N, 3]
                                   const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                   const torch::Tensor &dLossDColors,
                                   const torch::Tensor &radii,     // [N]
                                   const bool computeDLossDViewDirs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSBACKWARD_H
