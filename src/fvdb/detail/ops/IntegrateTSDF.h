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

/// @brief Batched depth-image TSDF integration — builds the full union
///        topology ONCE over all N frames, then runs N sequential
///        integrate passes against that fixed topology.
///
/// Semantically equivalent to calling `integrateTSDF` N times in a row
/// (verified bit-identically in the unit test), but avoids the per-
/// frame `buildPointTruncationShell + mergeGrids` cost that dominates
/// the per-frame wall-clock on small scenes.
///
/// For the paper's RGB-D comparison this is the natural idiom: all
/// frames are known up-front, topology is built once, then the fusion
/// kernel runs at fixed topology — the sparse-topology-as-tensor
/// analog of Open3D's lazy block-hashed allocation.
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
                   const std::optional<torch::Tensor> &weightImages);

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
                               const std::optional<torch::Tensor> &weightImages);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INTEGRATETSDF_H
