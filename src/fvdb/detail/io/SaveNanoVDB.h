// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_SAVENANOVDB_H
#define FVDB_DETAIL_IO_SAVENANOVDB_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace io {

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatchData &gridBatchData,
       const std::optional<JaggedTensor> &maybeData = std::nullopt,
       const std::vector<std::string> &names        = {});

// Build an ONINDEX NanoVDB buffer with float32 values appended as blind metadata slot 0.
// ValueOnIndex reserves index 0 for background and stores active voxel indices as 1..N, so the
// blind payload stores a background float first, followed by the user values. This lets editor
// shaders read via: val_addr = blind_data_base + val_index * sizeof(float).
// floatValues must still be a 1D float32 JaggedTensor with one entry per active voxel.
nanovdb::GridHandle<nanovdb::HostBuffer> toNVDBWithBlindFloat(const GridBatchData &gridBatchData,
                                                              const JaggedTensor &floatValues);

void saveNVDB(const std::string &path,
              const GridBatchData &gridBatchData,
              const std::optional<JaggedTensor> &maybeData,
              const std::vector<std::string> &names = {},
              bool compressed                       = false,
              bool verbose                          = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_SAVENANOVDB_H
