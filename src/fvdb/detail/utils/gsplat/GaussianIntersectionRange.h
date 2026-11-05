// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANINTERSECTIONRANGE_H
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANINTERSECTIONRANGE_H

#include <c10/macros/Macros.h>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cstdint>

namespace fvdb::detail::ops {

struct GaussianIntersectionRange {
    uint32_t offset;
    uint32_t count;
};

// Return the Gaussians in cameras overlapping the half-open tile-key range [keyStart, keyEnd).
// totalTiles is positive. For dense inputs, numGaussians is the count per camera; for packed
// inputs it is the total count, and cameraJIdx must be sorted in nondecreasing camera order.
// The total projected Gaussian count must fit in uint32_t.
inline C10_HOST_DEVICE GaussianIntersectionRange
gaussianRangeForTileKeys(const int64_t keyStart,
                         const int64_t keyEnd,
                         const uint32_t totalTiles,
                         const uint32_t numGaussians,
                         const int32_t *cameraJIdx = nullptr) {
    if (keyStart == keyEnd || numGaussians == 0) {
        return {0, 0};
    }
    const uint32_t cameraStart = keyStart / totalTiles;
    const uint32_t cameraEnd   = (keyEnd - 1) / totalTiles + 1;
    if (cameraJIdx == nullptr) {
        return {cameraStart * numGaussians, (cameraEnd - cameraStart) * numGaussians};
    }
    const auto begin =
        thrust::lower_bound(thrust::seq, cameraJIdx, cameraJIdx + numGaussians, cameraStart);
    const auto end = thrust::lower_bound(thrust::seq, begin, cameraJIdx + numGaussians, cameraEnd);
    return {static_cast<uint32_t>(begin - cameraJIdx), static_cast<uint32_t>(end - begin)};
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANINTERSECTIONRANGE_H
