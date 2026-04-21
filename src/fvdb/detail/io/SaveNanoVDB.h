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
       const std::optional<JaggedTensor> maybeData = std::nullopt,
       const std::vector<std::string> &names       = {});

void saveNVDB(const std::string &path,
              const GridBatchData &gridBatchData,
              const std::optional<JaggedTensor> maybeData,
              const std::vector<std::string> &names = {},
              bool compressed                       = false,
              bool verbose                          = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_SAVENANOVDB_H
