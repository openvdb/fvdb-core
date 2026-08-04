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

// Compact the selected grids of a (possibly sliced/non-contiguous) batch into a fresh contiguous
// GridHandle: a per-grid byte copy plus an mGridIndex/mGridCount header fixup (and checksum
// disable), O(bytes) with no radix sort. This is the cheap, correct way to realize an
// "identity" / whole-copy result on a view -- unlike nanoGridHandle().copy(), which would pull in
// the sibling grids a slice excludes.
nanovdb::GridHandle<TorchDeviceBuffer> contiguousGridHandle(const GridBatchData &input);

// Copy the i-th *logical* grid (by byte offset) into a standalone single-grid handle with
// mGridIndex=0 / mGridCount=1, suitable for mergeGridHandles. Correct for sliced views.
nanovdb::GridHandle<TorchDeviceBuffer> cloneGridHandleAt(const GridBatchData &input, int64_t i);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MAKECONTIGUOUS_H
