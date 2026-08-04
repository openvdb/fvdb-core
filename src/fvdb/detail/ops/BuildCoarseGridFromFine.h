// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDCOARSEGRIDFROMFINE_H
#define FVDB_DETAIL_OPS_BUILDCOARSEGRIDFROMFINE_H

#include <fvdb/GridBatchData.h>
#include <fvdb/TorchDeviceBuffer.h>

#include <nanovdb/GridHandle.h>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> buildCoarseGridFromFine(const GridBatchData &fineGridBatch,
                                                          const nanovdb::Coord branchingFactor);

// CUDA-only: build the coarsened-grid topology handle for `factor` -- leaf-mask CoarsenGrid passes
// for uniform power-of-two factors, coordinate-list fallback otherwise. Exposed so buildGridForConv
// can reuse it for its (kernel_size == 1 || stride == kernel_size) coarsening short circuit.
nanovdb::GridHandle<TorchDeviceBuffer>
coarseGridHandleFromFineCUDA(const GridBatchData &fineGridBatch, const nanovdb::Coord &factor);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDCOARSEGRIDFROMFINE_H
