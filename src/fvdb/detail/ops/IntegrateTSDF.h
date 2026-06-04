// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INTEGRATETSDF_H
#define FVDB_DETAIL_OPS_INTEGRATETSDF_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <optional>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDF(const c10::intrusive_ptr<GridBatchData> grid,
              const double truncationMargin,
              const torch::Tensor &projectionMatrices,
              const torch::Tensor &camToWorldMatrices,
              const JaggedTensor &tsdf,
              const JaggedTensor &weights,
              const torch::Tensor &depthImages,
              const std::optional<torch::Tensor> &weightImages);

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                          const double truncationMargin,
                          const torch::Tensor &projectionMatrices,
                          const torch::Tensor &camToWorldMatrices,
                          const JaggedTensor &tsdf,
                          const JaggedTensor &features,
                          const JaggedTensor &weights,
                          const torch::Tensor &depthImages,
                          const torch::Tensor &featureImages,
                          const std::optional<torch::Tensor> &weightImages);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INTEGRATETSDF_H
