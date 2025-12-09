// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRELOCATION_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRELOCATION_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Relocate Gaussians by adjusting opacity and scale based on replication ratio.
///
/// @param opacities Input opacities [N]
/// @param scales Input scales [N, 3]
/// @param ratios Replication ratios per Gaussian [N] (int32)
/// @param binomialCoeffs Binomial coefficients table [nMax, nMax]
/// @param nMax Maximum replication ratio (size of binomial table)
///
/// @return tuple of (opacitiesNew [N], scalesNew [N, 3])
template <torch::DeviceType DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianRelocation(const torch::Tensor &opacities,      // [N]
                           const torch::Tensor &scales,         // [N, 3]
                           const torch::Tensor &ratios,         // [N]
                           const torch::Tensor &binomialCoeffs, // [nMax, nMax]
                           const int nMax);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRELOCATION_H
