// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file BuildGridForConvTranspose.h
/// @brief Builds output grid topology for transposed sparse convolution.

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

/// @brief Build the NanoVDB grid for a transposed convolution output.
///
/// Given an input grid batch and convolution parameters, constructs the
/// output grid whose active voxels are all coordinates reachable by the
/// transposed convolution kernel from any active input voxel.
///
/// @tparam DeviceType  torch::kCPU or torch::kCUDA.
/// @param baseBatchHdl  Input grid batch providing the source topology.
/// @param kernelSize    Spatial kernel dimensions [k0, k1, k2].
/// @param stride        Convolution stride [s0, s1, s2].
/// @return A NanoVDB grid handle backed by a TorchDeviceBuffer.
template <torch::DeviceType DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConvTranspose(const GridBatchImpl &baseBatchHdl,
                                  const nanovdb::Coord &kernelSize,
                                  const nanovdb::Coord &stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFORCONVTRANSPOSE_H
