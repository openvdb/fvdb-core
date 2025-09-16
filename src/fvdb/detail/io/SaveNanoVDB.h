// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_SAVENANOVDB_H
#define FVDB_DETAIL_IO_SAVENANOVDB_H

#include <fvdb/GridBatch.h>
#include <fvdb/Types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace io {

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatch &gridBatch,
       const std::optional<JaggedTensor> maybeData = std::nullopt,
       const std::vector<std::string> &names       = {});

void saveNVDB(const std::string &path,
              const GridBatch &gridBatch,
              const std::optional<JaggedTensor> maybeData,
              const std::vector<std::string> &names = {},
              bool compressed                       = false,
              bool verbose                          = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_SAVENANOVDB_H
