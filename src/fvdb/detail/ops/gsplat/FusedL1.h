// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_FUSEDL1_H
#define FVDB_DETAIL_OPS_GSPLAT_FUSEDL1_H

#include <torch/types.h>

#include <optional>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor fusedL1CUDA(const torch::Tensor &img1, const torch::Tensor &img2);
torch::Tensor fusedL1PrivateUse1(const torch::Tensor &img1, const torch::Tensor &img2);

std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>>
fusedL1BackwardCUDA(const torch::Tensor &img1,
                    const torch::Tensor &img2,
                    const torch::Tensor &dL_dloss,
                    bool need_img1,
                    bool need_img2);

std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>>
fusedL1BackwardPrivateUse1(const torch::Tensor &img1,
                           const torch::Tensor &img2,
                           const torch::Tensor &dL_dloss,
                           bool need_img1,
                           bool need_img2);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_FUSEDL1_H
