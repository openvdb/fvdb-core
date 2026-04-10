// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCOMPUTENANINFMASK_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCOMPUTENANINFMASK_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Create a mask identifying NaN or Inf values in Gaussian parameters
///
/// This function examines jagged tensors containing Gaussian parameters and creates a mask
/// that identifies any NaN (Not a Number) or Inf (Infinity) values. This is important for
/// numerical stability in Gaussian Splatting algorithms, allowing invalid Gaussians to be
/// filtered out before rendering.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians as a jagged tensor [C, N, 3]
/// @param[in] quats Quaternion rotations of Gaussians as a jagged tensor [C, N, 4]
/// @param[in] logScales Scale factors of Gaussians as a jagged tensor [C, N, 3]
/// @param[in] logitOpacities Opacity values of Gaussians as a jagged tensor [N]
/// @param[in] sh0 Constant term (degree 0) spherical harmonic coefficients as a jagged tensor
/// @param[in] shN Higher degree spherical harmonic coefficients as a jagged tensor
///
/// @return A jagged tensor mask where True indicates valid values (no NaN/Inf) and False indicates
/// invalid values
template <torch::DeviceType>
fvdb::JaggedTensor dispatchGaussianNanInfMask(const fvdb::JaggedTensor &means,
                                              const fvdb::JaggedTensor &quats,
                                              const fvdb::JaggedTensor &logScales,
                                              const fvdb::JaggedTensor &logitOpacities,
                                              const fvdb::JaggedTensor &sh0,
                                              const fvdb::JaggedTensor &shN);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCOMPUTENANINFMASK_H
