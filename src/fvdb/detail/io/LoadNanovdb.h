// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_LOADNANOVDB_H
#define FVDB_DETAIL_IO_LOADNANOVDB_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace io {

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
fromNVDB(nanovdb::GridHandle<nanovdb::HostBuffer> &handle,
         const std::optional<torch::Device> maybeDevice = std::optional<torch::Device>());

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
fromNVDB(const std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> &handles,
         const std::optional<torch::Device> maybeDevice = std::optional<torch::Device>());

/// @brief Load a vector of grid handles matching the user-specified indices
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path,
         const std::vector<uint64_t> &indices,
         const torch::Device &device,
         bool verbose);

/// @brief Load a vector of grid handles matching the user-specified names
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path,
         const std::vector<std::string> &names,
         const torch::Device &device,
         bool verbose);

/// @brief Load a vector of grid handles
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path, const torch::Device &device, bool verbose);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_LOADNANOVDB_H
