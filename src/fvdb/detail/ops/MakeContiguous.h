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

/// @brief The batch's logical grids compacted end to end into fresh storage on @p device (the
///        batch's own device by default), each header stamped with its new (index, count), in
///        one pass: a view over storage on another device is compacted and moved by the same
///        copies. Correct for sliced and non-contiguous views, which share storage holding more
///        grids than they select.
GridStorage contiguousGridStorage(const GridBatchData &input,
                                  std::optional<torch::Device> device = std::nullopt);

/// @brief Logical grid @p i alone, as single-grid storage on the batch's device.
GridStorage cloneGridStorageAt(const GridBatchData &input, int64_t i);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
