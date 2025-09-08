// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H
#define FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H
#include <fvdb/detail/TorchDeviceBuffer.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace fvdb {
namespace detail {

inline nanovdb::GridHandle<TorchDeviceBuffer>
createEmptyGridHandle(torch::Device device, size_t numGrids = 1) {
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(numGrids);
    for (size_t i = 0; i < numGrids; ++i) {
        using GridT            = nanovdb::ValueOnIndex;
        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(0.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        proxyGridAccessor.merge();
        gridHandles.push_back(nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false));
    }
    auto ret = numGrids == 1 ? std::move(gridHandles[0]) : nanovdb::mergeGrids(gridHandles);
    ret.buffer().to(device);
    return ret;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H
