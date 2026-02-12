// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFORCONVTRANSPOSE_H
#define FVDB_DETAIL_OPS_BUILDGRIDFORCONVTRANSPOSE_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose(const GridBatchImpl &baseBatchHdl,
                                  const nanovdb::Coord &kernelSize,
                                  const nanovdb::Coord &stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFORCONVTRANSPOSE_H
