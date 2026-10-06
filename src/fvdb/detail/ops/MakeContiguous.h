// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
#define FVDB_DETAIL_OPS_MAKECONTIGUOUS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> makeContiguous(c10::intrusive_ptr<GridBatchData> input);

/// @brief The batch's logical grids as fresh storage on @p device (the batch's own device by
///        default): a contiguous batch's storage is copied whole (an empty batch's too, so its
///        one voxel-less grid comes along); anything else has its logical grids compacted end to
///        end in one pass, each header stamped with its new (index, count), so a view over
///        storage on another device is compacted and moved by the same copies. The one place
///        the "is it already contiguous" question is asked for a whole-batch copy.
GridStorage contiguousGridStorage(const GridBatchData &input,
                                  std::optional<torch::Device> device = std::nullopt);

/// @brief Logical grid @p i alone, as single-grid storage on the batch's device.
GridStorage cloneGridStorageAt(const GridBatchData &input, int64_t i);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
