// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSFORWARD_H
#define FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSFORWARD_H

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
/// @param[in] means Gaussian means in world space [N, 3]
/// @param[in] worldToCamMatrices Rigid world-to-camera matrices [C, 4, 4]. The matrices must be
/// rigid transforms with orthonormal rotation blocks.
/// @param[in] cameraIds Camera index for each packed work item, or empty for unpacked evaluation
/// @param[in] gaussianIds Gaussian index for each packed work item, or empty for unpacked
/// evaluation
/// @param[in] sh0Coeffs Spherical harmonic coefficients [N, 1, D] (packed) or
/// [1, N, D] (unpacked), where D is the number of feature channels
/// @param[in] shNCoeffs Higher order spherical harmonic coefficients [N, K-1, D] (packed) or
/// [K-1, N, D] (unpacked), where K depends on sh_degree_to_use (K=(sh_degree_to_use+1)²)
/// @param[in] radii Per-axis projected radii [C, N, 2]. A gaussian is masked unless both axes
/// are positive.
///
/// @return Features/colors [N, D] computed from the spherical harmonics evaluation
torch::Tensor evaluateSphericalHarmonicsFwd(const int64_t shDegreeToUse,
                                            const int64_t numCameras,
                                            const torch::Tensor &means,              // [N, 3]
                                            const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                            const torch::Tensor &cameraIds,          // [N] or empty
                                            const torch::Tensor &gaussianIds,        // [N] or empty
                                            const torch::Tensor &sh0Coeffs,          // [1, N, D]
                                            const torch::Tensor &shNCoeffs,          // [N, K-1, D]
                                            const torch::Tensor &radii               // [C, N, 2]
);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_EVALUATESPHERICALHARMONICSFORWARD_H
