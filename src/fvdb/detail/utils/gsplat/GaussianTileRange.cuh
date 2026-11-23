// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANTILERANGE_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANTILERANGE_CUH

#include <cuda/std/tuple>

#include <cstdint>

namespace fvdb::detail::ops {

template <typename TileOffsetsAccessor>
inline __device__ cuda::std::tuple<int64_t, int64_t>
tileGaussianRange(const TileOffsetsAccessor &tileOffsets,
                  const int64_t totalIntersections,
                  const uint32_t numCameras,
                  const uint32_t numTilesH,
                  const uint32_t numTilesW,
                  const uint32_t cameraId,
                  const uint32_t tileRow,
                  const uint32_t tileCol) {
    const int64_t firstIntersection = tileOffsets[cameraId][tileRow][tileCol];
    const bool isLastTile =
        cameraId == numCameras - 1 && tileRow == numTilesH - 1 && tileCol == numTilesW - 1;
    if (isLastTile) {
        return {firstIntersection, totalIntersections};
    }

    uint32_t nextCameraId = cameraId;
    uint32_t nextTileRow  = tileRow;
    uint32_t nextTileCol  = tileCol + 1;
    if (nextTileCol == numTilesW) {
        nextTileCol = 0;
        ++nextTileRow;
    }
    if (nextTileRow == numTilesH) {
        nextTileRow = 0;
        ++nextCameraId;
    }
    return {firstIntersection, tileOffsets[nextCameraId][nextTileRow][nextTileCol]};
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANTILERANGE_CUH
