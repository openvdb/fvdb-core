// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
#define FVDB_DETAIL_OPS_MAKECONTIGUOUS_H

#include <fvdb/GridBatchData.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> makeContiguous(c10::intrusive_ptr<GridBatchData> input);

/// @brief The batch's logical grids compacted end to end into fresh storage on the batch's
///        device, each header stamped with its new (index, count). Correct for sliced and
///        non-contiguous views, which share storage holding more grids than they select.
GridStorage contiguousGridStorage(const GridBatchData &input);

/// @brief The same, as a dual-space TorchDeviceBuffer handle, for the builders that still work
///        in that type (#770 step 5 removes it).
nanovdb::GridHandle<TorchDeviceBuffer> contiguousGridHandle(const GridBatchData &input);

/// @brief Logical grid @p i alone as a single-grid TorchDeviceBuffer handle (same caveat).
nanovdb::GridHandle<TorchDeviceBuffer> cloneGridHandleAt(const GridBatchData &input, int64_t i);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
