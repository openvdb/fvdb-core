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

/// @brief Integrates N depth frames in one call, growing the topology
///        frame by frame in a persistent accumulator.
///
/// With `carveFreeSpace`, each frame updates every visible voxel, so the
/// result equals calling `integrateTSDF` N times (bit-identically). Without
/// it, each frame updates only its truncation shell, which is cheaper on
/// large grids but leaves visible free space in front of the surface
/// uncarved.
///
/// Requires `grid->batchSize() == 1`. The N dimension is carried on
/// `depthImages.size(0)` and must match `projectionMatrices.size(0)`
/// and `camToWorldMatrices.size(0)`.
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFBatch(const c10::intrusive_ptr<GridBatchData> grid,
                   const double truncationMargin,
                   const torch::Tensor &projectionMatrices,
                   const torch::Tensor &camToWorldMatrices,
                   const JaggedTensor &tsdf,
                   const JaggedTensor &weights,
                   const torch::Tensor &depthImages,
                   const std::optional<torch::Tensor> &weightImages,
                   bool carveFreeSpace);

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFBatchWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                               const double truncationMargin,
                               const torch::Tensor &projectionMatrices,
                               const torch::Tensor &camToWorldMatrices,
                               const JaggedTensor &tsdf,
                               const JaggedTensor &features,
                               const JaggedTensor &weights,
                               const torch::Tensor &depthImages,
                               const torch::Tensor &featureImages,
                               const std::optional<torch::Tensor> &weightImages,
                               bool carveFreeSpace);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INTEGRATETSDF_H
