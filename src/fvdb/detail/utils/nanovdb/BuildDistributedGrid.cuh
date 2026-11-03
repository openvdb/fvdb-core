// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_BUILDDISTRIBUTEDGRID_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_BUILDDISTRIBUTEDGRID_CUH

#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#if CCCL_DEVICE_MERGE_SUPPORTED

#include <nanovdb/cuda/DeviceMesh.h>
#include <nanovdb/cuda/HandleStorage.h>
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>

#include <c10/cuda/CUDAException.h>

#include <cstddef>

namespace fvdb::detail {

/// @brief Builds one grid from @p count voxel coordinates in PrivateUse1 memory across every
///        device, and returns it in storage allocated from @p proto.
///
///        DistributedPointsToGrid writes the grid from every device in the mesh, so it builds into
///        a UnifiedGridBuffer. The result is then copied on @p stream into a DeviceGridBuffer,
///        the type GridStorage holds, and @p stream is synchronized so the unified source can be
///        freed.
template <typename GridT>
GridStorage::DeviceHandle
buildDistributedGrid(nanovdb::Coord *coords,
                     size_t count,
                     const DeviceGridBuffer &proto,
                     cudaStream_t stream) {
    nanovdb::cuda::DeviceMesh mesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<GridT> converter(mesh);
    const auto built = converter.template getHandle<nanovdb::Coord *, UnifiedGridBuffer>(
        coords, count, UnifiedGridBuffer());
    auto handle = nanovdb::cuda::copyTo<DeviceGridBuffer>(built, stream, &proto);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return handle;
}

} // namespace fvdb::detail

#endif // CCCL_DEVICE_MERGE_SUPPORTED

#endif // FVDB_DETAIL_UTILS_NANOVDB_BUILDDISTRIBUTEDGRID_CUH
