// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSFORWARD_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Evaluate spherical harmonics functions to compute features/colors.
///
/// This function computes the features/colors for points in 3D space using spherical harmonics
/// (SH) representation. Spherical harmonics provide an efficient way to represent view-dependent
/// appearance for Gaussian Splatting and other rendering techniques. The output features are not
/// limited to RGB colors; they can have any number of channels.
///
/// @param[in] shDegreeToUse Degree of spherical harmonics to use (0-3 typically, higher degrees
/// provide more detail)
/// @param[in] numCameras Number of cameras used for rendering
/// @param[in] viewDirs Direction vectors [N, 3] (packed) or [C, N, 3] (unpacked) normalized to unit
/// length, representing view directions
/// @param[in] sh0Coeffs Spherical harmonic coefficients [N, 1, D] (packed) or
/// [1, N, D] (unpacked), where D is the number of feature channels
/// @param[in] shNCoeffs Higher order spherical harmonic coefficients [N, K-1, D] (packed) or
/// [K-1, N, D] (unpacked), where K depends on sh_degree_to_use (K=(sh_degree_to_use+1)²)
/// @param[in] radii radii [N] (packed) or [C, N] (unpacked) for view-dependent level-of-detail
/// control
///
/// @return Features/colors [N, D] computed from the spherical harmonics evaluation
torch::Tensor sphericalHarmonicsForward(const int64_t shDegreeToUse,
                                        const int64_t numCameras,
                                        const torch::Tensor &viewDirs,  // [C, N, 3]
                                        const torch::Tensor &sh0Coeffs, // [1, N, D]
                                        const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                        const torch::Tensor &radii      // [C, N]
);

/// @brief Evaluate spherical harmonics to compute view-dependent features/colors.
///
/// Computes per-camera, per-Gaussian features using spherical harmonics (SH) representation.
/// Internally derives view directions from the camera matrices and Gaussian means, then dispatches
/// to the SH forward kernel. When @p shDegreeToUse is 0, view directions are not needed.
/// The output features are not limited to RGB colors; they can have any number of channels.
///
/// @param[in] means              Gaussian mean positions [N, 3]
/// @param[in] sh0                Degree-0 SH coefficients [N, 1, D] where D is number of channels
/// @param[in] shN                Higher-degree SH coefficients [N, K-1, D] where
///                               K = (shDegreeToUse+1)²
/// @param[in] shDegreeToUse      SH degree to use (0-3 typically, -1 to use all available degrees)
/// @param[in] worldToCameraMatrices Camera extrinsics [C, 4, 4]
/// @param[in] perGaussianProjectedRadii Projected radii [C, N] for view-dependent level-of-detail
/// @return Evaluated SH features [C, N, D]
torch::Tensor evalSphericalHarmonics(const torch::Tensor &means,
                                     const torch::Tensor &sh0,
                                     const torch::Tensor &shN,
                                     int64_t shDegreeToUse,
                                     const torch::Tensor &worldToCameraMatrices,
                                     const torch::Tensor &perGaussianProjectedRadii);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSFORWARD_H
