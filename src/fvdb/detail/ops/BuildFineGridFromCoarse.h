// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H
#define FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse(const GridBatchImpl &coarseBatchHdl,
                                const nanovdb::Coord subdivisionFactor,
                                const std::optional<JaggedTensor> &subdivMask);

template <torch::DeviceType>
JaggedTensor dispatchFineIJKForCoarseGrid(const GridBatchImpl &batchHdl,
                                          nanovdb::Coord upsamplingFactor,
                                          const std::optional<JaggedTensor> &maybeMask);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDFINEGRIDFROMCOARSE_H
