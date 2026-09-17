// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDSTORAGE_H
#define FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDSTORAGE_H
#include <fvdb/GridStorage.h>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <torch/types.h>

#include <memory>
#include <utility>
#include <vector>

namespace fvdb {
namespace detail {

/// @brief Storage holding @p numGrids valid, voxel-less ValueOnIndex grids on @p device: what an
///        empty GridBatchData points at, so every accessor still finds a well-formed header.
inline GridStorage
createEmptyGridStorage(torch::Device device, size_t numGrids = 1) {
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> gridHandles;
    gridHandles.reserve(numGrids);
    for (size_t i = 0; i < numGrids; ++i) {
        using GridT            = nanovdb::ValueOnIndex;
        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(0.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();
        proxyGridAccessor.merge();
        gridHandles.push_back(
            nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                *proxyGrid, 0u, false, false));
    }
    GridStorage host(numGrids == 1 ? std::move(gridHandles[0]) : nanovdb::mergeGrids(gridHandles));
    return device.is_cpu() ? std::move(host) : host.to(device);
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDSTORAGE_H
