// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H
#define FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H

#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/JaggedTensor.h>

#include <nanovdb/NanoVDB.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData>
buildFineGridFromCoarse(const GridBatchData &coarseBatchHdl,
                        const nanovdb::Coord subdivisionFactor,
                        const std::optional<JaggedTensor> &subdivMask);

JaggedTensor fineIJKForCoarseGrid(const GridBatchData &batchHdl,
                                  nanovdb::Coord upsamplingFactor,
                                  const std::optional<JaggedTensor> &maybeMask);

// Build the subdivided (fine) grid topology for `factor` as one storage on the coarse batch's
// device -- RefineGrid passes for uniform power-of-two factors (an optional per-coarse-voxel mask
// is applied via PruneGrid first), coordinate-list fallback otherwise.
GridStorage fineGridStorageFromCoarseCUDA(const GridBatchData &coarseBatchHdl,
                                          const nanovdb::Coord &factor,
                                          const std::optional<JaggedTensor> &mask);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H
