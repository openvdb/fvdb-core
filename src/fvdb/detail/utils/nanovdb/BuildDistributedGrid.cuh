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
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>

#include <cstddef>

namespace fvdb::detail {

/// @brief Builds one grid from @p count voxel coordinates in PrivateUse1 memory across every
///        device. DistributedPointsToGrid writes the grid from every device in the mesh, so it
///        builds into a UnifiedGridBuffer, which GridStorage holds as is. The build synchronizes
///        the mesh before returning.
template <typename GridT>
GridStorage::UnifiedHandle
buildDistributedGrid(nanovdb::Coord *coords, size_t count) {
    nanovdb::cuda::DeviceMesh mesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<GridT> converter(mesh);
    return converter.template getHandle<nanovdb::Coord *, UnifiedGridBuffer>(
        coords, count, UnifiedGridBuffer());
}

} // namespace fvdb::detail

#endif // CCCL_DEVICE_MERGE_SUPPORTED

#endif // FVDB_DETAIL_UTILS_NANOVDB_BUILDDISTRIBUTEDGRID_CUH
