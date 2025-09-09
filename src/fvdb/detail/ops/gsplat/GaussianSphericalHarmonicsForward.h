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
template <torch::DeviceType>
torch::Tensor dispatchSphericalHarmonicsForward(const int64_t shDegreeToUse,
                                                const int64_t numCameras,
                                                const torch::Tensor &viewDirs,  // [C, N, 3]
                                                const torch::Tensor &sh0Coeffs, // [1, N, D]
                                                const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                                const torch::Tensor &radii      // [C, N]
);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPHERICALHARMONICSFORWARD_H
