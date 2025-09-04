// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDDENSEGRID_H
#define FVDB_DETAIL_OPS_BUILDDENSEGRID_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense(int64_t batchSize,
                                nanovdb::Coord origin,
                                nanovdb::Coord size,
                                torch::Device device,
                                const std::optional<torch::Tensor> &maybeMask);
} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDDENSEGRID_H
