// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLENEAREST_H
#define FVDB_DETAIL_OPS_SAMPLENEAREST_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Sample features on the grid batch using nearest-neighbor lookup.
///
/// Data is assumed to be defined at voxel centers.  For each query
/// point, the 8 nearest voxel centers are checked and the value of
/// the closest active one is returned.  When none of the 8 are active
/// the output is zero (matching sample_trilinear boundary behaviour).
///
/// @return A two-element vector: [sampled_values, selected_indices].
///         selected_indices contains the cumulative linear index of the
///         chosen voxel for each point, or -1 when no active corner exists.
std::vector<torch::Tensor> sampleNearest(const GridBatchData &batchHdl,
                                         const JaggedTensor &points,
                                         const torch::Tensor &gridData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLENEAREST_H
