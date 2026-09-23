// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_FUSEDL1SSIM_H
#define FVDB_DETAIL_OPS_GSPLAT_FUSEDL1SSIM_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// Aggregate loss and the three SSIM derivative tensors saved for backward.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMCUDA(const torch::Tensor &img1,
                const torch::Tensor &img2,
                double ssim_weight,
                double C1,
                double C2,
                bool train);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMPrivateUse1(const torch::Tensor &img1,
                       const torch::Tensor &img2,
                       double ssim_weight,
                       double C1,
                       double C2,
                       bool train);

torch::Tensor fusedL1SSIMBackwardCUDA(const torch::Tensor &img1,
                                      const torch::Tensor &img2,
                                      const torch::Tensor &grad_loss,
                                      const torch::Tensor &dm_dmu1,
                                      const torch::Tensor &dm_dsigma1_sq,
                                      const torch::Tensor &dm_dsigma12,
                                      double ssim_weight,
                                      double C1,
                                      double C2);
torch::Tensor fusedL1SSIMBackwardPrivateUse1(const torch::Tensor &img1,
                                             const torch::Tensor &img2,
                                             const torch::Tensor &grad_loss,
                                             const torch::Tensor &dm_dmu1,
                                             const torch::Tensor &dm_dsigma1_sq,
                                             const torch::Tensor &dm_dsigma12,
                                             double ssim_weight,
                                             double C1,
                                             double C2);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_FUSEDL1SSIM_H
