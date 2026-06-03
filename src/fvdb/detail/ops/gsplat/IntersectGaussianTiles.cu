// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/BuildSparseGaussianTileLayout.h>
#include <fvdb/detail/ops/gsplat/IntersectGaussianTiles.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/util/cuda/Util.h>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/cub.cuh>
#include <cuda/std/functional>

#define FVDB_CUB_WRAPPER(func, ...)                                             \
    do {                                                                        \
        size_t tempStorageBytes = 0;                                            \
        C10_CUDA_CHECK(func(nullptr, tempStorageBytes, __VA_ARGS__));           \
        auto &cachingAllocator = *::c10::cuda::CUDACachingAllocator::get();     \
        auto tempStorage       = cachingAllocator.allocate(tempStorageBytes);   \
        C10_CUDA_CHECK(func(tempStorage.get(), tempStorageBytes, __VA_ARGS__)); \
    } while (false)

// Like FVDB_CUB_WRAPPER but allocates the scratch with cudaMallocAsync/cudaFreeAsync on `stream`
// rather than the caching allocator, so it is stream-ordered and resident on the current device.
// Used by the multi-GPU path. `stream` is appended to the CUB call automatically, so __VA_ARGS__
// holds only the CUB arguments before the stream.
#define FVDB_CUB_WRAPPER_ASYNC(stream, func, ...)                                 \
    do {                                                                          \
        size_t tempStorageBytes = 0;                                              \
        void *tempStorage       = nullptr;                                        \
        C10_CUDA_CHECK(func(tempStorage, tempStorageBytes, __VA_ARGS__, stream)); \
        C10_CUDA_CHECK(cudaMallocAsync(&tempStorage, tempStorageBytes, stream));  \
        C10_CUDA_CHECK(func(tempStorage, tempStorageBytes, __VA_ARGS__, stream)); \
        C10_CUDA_CHECK(cudaFreeAsync(tempStorage, stream));                       \
    } while (false)

#include <thrust/binary_search.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

#define NUM_THREADS DEFAULT_BLOCK_DIM

// For row `i` of camera `cidx`'s tile rect, compute the [jStart, jEnd) column sub-range whose
// linear tile-keys L = cidx * totalTiles + i * numTilesW + j fall within [keyStart, keyEnd).
// Returns {jStart, jEnd, done}: jStart >= jEnd means the row has no in-range tiles, and `done` is
// set when this row (and hence all later rows, since L increases with i) starts at/above keyEnd,
// so the caller can stop iterating.
__device__ inline std::tuple<int32_t, int32_t, bool>
rowKeyColumnRange(int32_t cidx,
                  uint32_t i,
                  uint32_t numTilesW,
                  uint32_t totalTiles,
                  uint2 tileMin,
                  uint2 tileMax,
                  int64_t keyStart,
                  int64_t keyEnd) {
    const int64_t rowKeyOffset =
        static_cast<int64_t>(cidx) * totalTiles + static_cast<int64_t>(i) * numTilesW;
    const bool done = (rowKeyOffset + tileMin.x) >= keyEnd;
    const int32_t jStart =
        static_cast<int32_t>(max(static_cast<int64_t>(tileMin.x), keyStart - rowKeyOffset));
    const int32_t jEnd =
        static_cast<int32_t>(min(static_cast<int64_t>(tileMax.x), keyEnd - rowKeyOffset));
    return {jStart, jEnd, done};
}

// Compute the number of 2d image tiles intersected by a set of 2D projected Gaussians.
//
// The input is a set of 2D ellipses (axis-aligned bounding boxes) with depths approximating the
// projection of 3D gaussians onto the image plane. Each input is encoded as a tuple:
// (mean_u, mean_v, radius_u, radius_v, depth)
// where (mean_u, mean_v) are the image-space center, (radius_u, radius_v) are the per-axis
// pixel radii of the AABB, and depth is the (world-space) depth of the Gaussian.
//
// The output is a set of counts of the number of tiles each Gaussian intersects.
//
template <typename T, typename CountT>
__global__ __launch_bounds__(NUM_THREADS) void
countTilesPerGaussian(const uint32_t gaussianOffset,
                      const uint32_t gaussianCount,
                      const uint32_t numGaussiansPerCamera,
                      const uint32_t tileSize,
                      const uint32_t numTilesW,
                      const uint32_t numTilesH,
                      const uint32_t totalTiles,
                      const int64_t keyStart,                        // tile-key range start
                      const int64_t keyEnd,                          // tile-key range end
                      const T *__restrict__ means2d,                 // [C, N, 2] or [M, 2]
                      const int32_t *__restrict__ radii,             // [C, N, 2] or [M, 2]
                      const bool *__restrict__ tileMask,             // [C, H, W] or nullptr
                      const int32_t *__restrict__ cameraJIdx,        // NULL or [M]
                      CountT *__restrict__ outNumTilesPerGaussian) { // [ C * N ] or [ M ]
    // parallelize over gaussianCount
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gaussianCount;
         idx += blockDim.x * gridDim.x) {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = at::opmath_type<T>;

        auto gidx         = idx + gaussianOffset;
        const OpT radiusU = radii[gidx * 2 + 0];
        const OpT radiusV = radii[gidx * 2 + 1];
        if (radiusU <= 0 || radiusV <= 0) {
            outNumTilesPerGaussian[gidx] = static_cast<CountT>(0);
        } else {
            using vec2f = float2;

            const vec2f mean2d    = *reinterpret_cast<const vec2f *>(means2d + gidx * 2);
            const OpT tileRadiusU = radiusU / static_cast<OpT>(tileSize);
            const OpT tileRadiusV = radiusV / static_cast<OpT>(tileSize);
            const OpT tileMeanU   = mean2d.x / static_cast<OpT>(tileSize);
            const OpT tileMeanV   = mean2d.y / static_cast<OpT>(tileSize);

            // tile_min is inclusive, tile_max is exclusive
            uint2 tileMin, tileMax;
            tileMin.x = min(max(0, (uint32_t)floor(tileMeanU - tileRadiusU)), numTilesW);
            tileMin.y = min(max(0, (uint32_t)floor(tileMeanV - tileRadiusV)), numTilesH);
            tileMax.x = min(max(0, (uint32_t)ceil(tileMeanU + tileRadiusU)), numTilesW);
            tileMax.y = min(max(0, (uint32_t)ceil(tileMeanV + tileRadiusV)), numTilesH);

            const int32_t cidx = (cameraJIdx == nullptr)
                                     ? static_cast<int32_t>(gidx / numGaussiansPerCamera)
                                     : cameraJIdx[gidx];

            // Count only the tiles whose linear tile-key falls in this device's range
            // [keyStart, keyEnd), via the in-range column sub-range of each row of the tile rect.
            CountT numTiles = 0;
            for (uint32_t i = tileMin.y; i < tileMax.y; ++i) {
                const auto [jStart, jEnd, done] = rowKeyColumnRange(
                    cidx, i, numTilesW, totalTiles, tileMin, tileMax, keyStart, keyEnd);
                if (done) {
                    break;
                }
                if (jStart >= jEnd) {
                    continue;
                }
                if (tileMask) {
                    for (int32_t j = jStart; j < jEnd; ++j) {
                        if (tileMask[cidx * numTilesH * numTilesW + i * numTilesW + j]) {
                            numTiles++;
                        }
                    }
                } else {
                    numTiles += static_cast<CountT>(jEnd - jStart);
                }
            }
            outNumTilesPerGaussian[gidx] = numTiles;
        }
    }
}

// Encode a float depth into a int64_t
__device__ inline int64_t
encodeDepth(float depth) {
    int32_t ret;
    std::memcpy(&ret, &depth, sizeof(depth));
    return static_cast<int64_t>(ret);
}

// Encode a camera index, tile index, and depth into a int64_t
// The layout is (from least to most significant bits):
// [32 bits] depth
// [tile_id_bits] tile_idx
// [camera_id_bits] cidx
__device__ inline int64_t
encodeCamTileDepthKey(int64_t cidx, int64_t tileIdx, int64_t encodedDepth, int32_t tileIdBits) {
    int64_t cidxEnc = cidx << tileIdBits;
    return ((cidxEnc | tileIdx) << 32) | encodedDepth;
}

// Decode a camera index and tile index from a int64_t
// The layout is (from least to most significant bits):
// [32 bits] depth (ignored and dropped)
// [tileIdBits] tileIdx
// [camera_id_bits] cidx
__device__ inline std::tuple<int32_t, int32_t>
decodeCamTileKey(int64_t packedCamTileDepthKey, int32_t tileIdBits) {
    int32_t tileKey = packedCamTileDepthKey >> 32;
    int32_t cidxEnc = tileKey >> tileIdBits;
    int32_t tileIdx = tileKey & ((1 << tileIdBits) - 1);
    return {cidxEnc, tileIdx};
}

// Compute a set of intersections between the 2D projected Gaussians and each tile to be
// rendered on screen.
//
// The input is a set of 2D AABBs with depths approximating the projection of 3D gaussians
// onto the image plane. Each input is encoded as a tuple:
// (mean_u, mean_v, radius_u, radius_v, depth) where (mean_u, mean_v) are the image-space center,
// (radius_u, radius_v) are the per-axis pixel radii of the AABB, and depth is the (world-space)
// depth of the Gaussian.
//
// The output is a set of gaussian/tile intersections where each intersection is parameterized
// as: a tuple key (camera_id, tile_id, depth) gaussian_id value indexing into
// means2d/radii/depths
//
// The key (camera_id, tile_id, depth) is packed into 64 bits and identifies which camera, and
// tile this intersection corresponds to, and the depth of the Gaussian at this intersection.
// (we'll use this to sort the intersections into tiles and by depth).
// The value is the index of the Gaussian in the input arrays.
//
template <typename T>
__global__ __launch_bounds__(NUM_THREADS) void
computeGaussianTileIntersections(
    const uint32_t numCameras,
    const uint32_t numGaussiansPerCamera,
    const uint32_t gaussianOffset,
    const uint32_t gaussianCount,
    const uint32_t tileSize,
    const uint32_t numTilesW,
    const uint32_t numTilesH,
    const uint32_t tileIdBits,
    const uint32_t totalTiles,
    const int64_t keyStart,                          // tile-key range start
    const int64_t keyEnd,                            // tile-key range end
    const T *__restrict__ means2d,                   // [C, N, 2] or [M, 2]
    const int32_t *__restrict__ radii,               // [C, N, 2] or [M, 2]
    const T *__restrict__ depths,                    // [C, N]    or [M]
    const int32_t *__restrict__ cumTilesPerGaussian, // [ C * N ] or [ M ]
    const bool *__restrict__ tileMask,               // [C, H, W] or nullptr
    const int32_t *__restrict__ cameraJIdx,          // NULL or [M]
    int64_t *__restrict__ intersectionKeys,          // [ C * N * numTiles ] or [ M * numTiles ]
    int32_t *__restrict__ intersectionValues) {      // [ C * N * numTiles ] or [ M * numTiles ]

    // parallelize over total_gaussians
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gaussianCount;
         idx += blockDim.x * gridDim.x) {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = at::opmath_type<T>;

        auto gidx = idx + gaussianOffset;
        // Get the camera id from the batch indices or use the camera index directly
        const int32_t cidx =
            cameraJIdx == nullptr ? gidx / numGaussiansPerCamera : cameraJIdx[gidx];

        const OpT radiusU = radii[gidx * 2 + 0];
        const OpT radiusV = radii[gidx * 2 + 1];
        if (radiusU > 0 && radiusV > 0) {
            using vec2f = float2;

            const vec2f mean2d    = *reinterpret_cast<const vec2f *>(means2d + 2 * gidx);
            const OpT tileRadiusU = radiusU / static_cast<OpT>(tileSize);
            const OpT tileRadiusV = radiusV / static_cast<OpT>(tileSize);
            const OpT tileMeanU   = mean2d.x / static_cast<OpT>(tileSize);
            const OpT tileMeanV   = mean2d.y / static_cast<OpT>(tileSize);

            // tile_min is inclusive, tile_max is exclusive
            uint2 tileMin, tileMax;
            tileMin.x = min(max(0, (uint32_t)floor(tileMeanU - tileRadiusU)), numTilesW);
            tileMin.y = min(max(0, (uint32_t)floor(tileMeanV - tileRadiusV)), numTilesH);
            tileMax.x = min(max(0, (uint32_t)ceil(tileMeanU + tileRadiusU)), numTilesW);
            tileMax.y = min(max(0, (uint32_t)ceil(tileMeanV + tileRadiusV)), numTilesH);

            // If you use float64, we're casting you to float32 so we can
            // pack the depth into the key. In principle this loses precision,
            // in practice it's fine.
            const float depth = depths[gidx];

            // Suppose you're using tile_id_bits = 22, then the output for this intersection is
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            // which we pack into an int64_t
            const int64_t depthEnc = encodeDepth(depth);

            // For each tile this Gaussian intersects whose tile-key L = cidx * totalTiles +
            // tileIdx falls in this device's range [keyStart, keyEnd), write out an intersection
            // tuple {(camera_id | tile_id | depth), gaussian_id} (int64_t, int32_t). The write
            // index is local to this device's intersection slice (intersectionKeys /
            // intersectionValues are already offset to the start of that slice), and matches the
            // per-device count computed by countTilesPerGaussian.
            int64_t curIsect = (gidx == 0) ? 0 : cumTilesPerGaussian[gidx - 1];
            for (int32_t i = tileMin.y; i < tileMax.y; ++i) {
                const auto [jStart, jEnd, done] = rowKeyColumnRange(
                    cidx, i, numTilesW, totalTiles, tileMin, tileMax, keyStart, keyEnd);
                if (done) {
                    break;
                }
                for (int32_t j = jStart; j < jEnd; ++j) {
                    // Skip if tile is masked out
                    if (tileMask && !tileMask[cidx * numTilesH * numTilesW + i * numTilesW + j]) {
                        continue;
                    }
                    const int64_t tileIdx = (i * numTilesW + j); // Needs to fit in tileIdBits bits
                    const int64_t packedCamIdxAndTileIdx =
                        encodeCamTileDepthKey(cidx, tileIdx, depthEnc, tileIdBits);
                    intersectionKeys[curIsect]   = packedCamIdxAndTileIdx;
                    intersectionValues[curIsect] = gidx;
                    curIsect += 1;
                }
            }
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS) void
computeTileOffsetsSparse(const uint32_t numIntersections,
                         const uint32_t numTiles,
                         const uint32_t tileIdBits,
                         const int64_t *__restrict__ sortedIntersectionKeys,
                         const int32_t *__restrict__ activeTiles,
                         const uint32_t numActiveTiles,
                         int32_t *__restrict__ outOffsets) { // [C, n_tiles] or [num_active_tiles]

    // parallelize over active tiles
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx <= numActiveTiles;
         idx += blockDim.x * gridDim.x) {
        if (idx == numActiveTiles) {
            outOffsets[idx] = numIntersections;
        } else {
            // get the first intersection for this tile
            const int64_t tileStartIdxDense = activeTiles[idx];
            const int64_t camIdx            = tileStartIdxDense / numTiles;
            const int64_t tileIdx           = tileStartIdxDense % numTiles;

            auto depthEnc = encodeDepth(0.0f); // Don't care what depth

            const int64_t packedCamIdxAndTileIdx =
                encodeCamTileDepthKey(camIdx, tileIdx, depthEnc, tileIdBits);

            // search in sortedIntersectionKeys for the start of the range matching
            // packedCamIdxAndTileIdx

            auto compareKeysIgnoreDepth = [tileIdBits](int64_t lhs, int64_t rhs) {
                auto [lhsCamIdx, lhsTileIdx] = decodeCamTileKey(lhs, tileIdBits);
                auto [rhsCamIdx, rhsTileIdx] = decodeCamTileKey(rhs, tileIdBits);
                return (lhsCamIdx < rhsCamIdx) ||
                       ((lhsCamIdx == rhsCamIdx) && (lhsTileIdx < rhsTileIdx));
            };

            auto const tileStart = thrust::lower_bound(thrust::seq,
                                                       sortedIntersectionKeys,
                                                       sortedIntersectionKeys + numIntersections,
                                                       packedCamIdxAndTileIdx,
                                                       compareKeysIgnoreDepth) -
                                   sortedIntersectionKeys;

            outOffsets[idx] = tileStart;
        }
    }
}

// Given a set of intersections between 2D projected Gaussians and image tiles sorted by
// tile_id, camera_id, and depth, compute the a range of Gaussians that intersect each tile
// encoded as an offset into the sorted intersection array.
// i.e. gaussians[out_offsets[c, i, j]:out_offsets[c, i, j+1]] are the Gaussians that
// intersect tile (i, j) in camera c.
__global__ __launch_bounds__(NUM_THREADS) void
computeTileOffsets(const uint32_t offset,
                   const uint32_t count,
                   const uint32_t numIntersections,
                   const uint32_t numCameras,
                   const uint32_t numTiles,
                   const uint32_t tileIdBits,
                   const int64_t *__restrict__ sortedIntersectionKeys,
                   int32_t *__restrict__ outOffsets) { // [C, numTiles]
    // sortedIntersectionKeys is [(cidx_0 | tidx_0 | depth_0), ..., (cidx_N | tidx_N |
    // depth_N)] where cidx_i = camera index, tidx_i = tile index, depth_i = depth of the
    // gaussian at the intersection, lexographically sorted.
    //
    // The output is a set of offsets into the sorted_intersection array, such that
    // if offset_cij = offsets[cid][ti][tj], and offset_cij+1 = offsets[cid][ti][tj+1],
    // then sorted_intersections[offset_cij:offset_cij+1] are the intersections for the
    // tile (cid, ti, tj) sorted by depth.

    // Parallelize over intersections
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x) {
        auto isectIdx = idx + offset;
        // Bit-packed key for the camera/tile part of the this intersection
        // i.e. tile_id | camera_id << tile_id_bits
        const int64_t tileKey  = sortedIntersectionKeys[isectIdx] >> 32;
        auto [camIdx, tileIdx] = decodeCamTileKey(sortedIntersectionKeys[isectIdx], tileIdBits);

        // The first intersection for this camera/tile pair
        const int64_t tileStartIdx = camIdx * numTiles + tileIdx;

        if (isectIdx == 0) {
            // The first tile in the first camera writes out 0 as the offset
            // until the first valid tile (inclusive). i.e. tiles before this one
            // have no intersections, so their offset range is [0, 0]
            for (uint32_t i = 0; i < tileStartIdx + 1; ++i) {
                outOffsets[i] = 0;
            }
        }
        if (isectIdx == numIntersections - 1) {
            for (uint32_t i = tileStartIdx + 1; i < numCameras * numTiles; ++i) {
                outOffsets[i] = static_cast<int32_t>(numIntersections);
            }
        }

        if (isectIdx > 0) {
            const int64_t prevTileKey = sortedIntersectionKeys[isectIdx - 1] >> 32;

            if (prevTileKey == tileKey) {
                continue;
            }

            auto [prevCamIdx, prevTileIdx] =
                decodeCamTileKey(sortedIntersectionKeys[isectIdx - 1], tileIdBits);
            const int64_t prevTileStartIdx = prevCamIdx * numTiles + prevTileIdx;

            for (uint32_t i = prevTileStartIdx + 1; i < tileStartIdx + 1; ++i) {
                outOffsets[i] = static_cast<int32_t>(isectIdx);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
intersectGaussianTilesCudaImpl(
    const torch::Tensor &means2d,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                     // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                    // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx,  // NULL or [M]
    const at::optional<torch::Tensor> &tileMask,    // NULL or [C, H, W]
    const at::optional<torch::Tensor> &activeTiles, // NULL or [numActiveTiles]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    const bool isPacked = cameraJIdx.has_value();
    const bool isSparse = activeTiles.has_value();

    TORCH_CHECK(means2d.is_cuda(), "means2d must be a CUDA tensor");
    TORCH_CHECK(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK(depths.is_cuda(), "depths must be a CUDA tensor");

    if (isPacked) {
        TORCH_CHECK(cameraJIdx.value().is_cuda(), "cameraJIdx must be a CUDA tensor");
        TORCH_CHECK_VALUE(means2d.dim() == 2, "means2d must have 2 dimensions (M, 2)");
        TORCH_CHECK_VALUE(means2d.size(1) == 2, "means2d must have 2 points in the last dimension");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have 2 dimensions (M, 2)");
        TORCH_CHECK_VALUE(radii.size(1) == 2, "radii must have 2 components in the last dimension");
        TORCH_CHECK_VALUE(depths.dim() == 1, "depths must have 1 dimension (M)");
        TORCH_CHECK_VALUE(cameraJIdx.value().dim() == 1, "cameraJIdx must have 1 dimension (M)");
        TORCH_CHECK_VALUE(radii.size(0) == means2d.size(0),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(depths.size(0) == means2d.size(0),
                          "depths must have the same number of points as means2d");
        TORCH_CHECK_VALUE(cameraJIdx.value().size(0) == means2d.size(0),
                          "cameraJIdx must have the same number of points as means2d");
    } else {
        TORCH_CHECK_VALUE(means2d.dim() == 3, "means2d must have 3 dimensions (C, N, 2)");
        TORCH_CHECK_VALUE(means2d.size(0) == numCameras,
                          "means2d must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(means2d.size(2) == 2, "means2d must have 2 points in the last dimension");
        TORCH_CHECK_VALUE(radii.dim() == 3, "radii must have 3 dimensions (C, N, 2)");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have numCameras in the first dimension");
        TORCH_CHECK_VALUE(radii.size(1) == means2d.size(1),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(radii.size(2) == 2, "radii must have 2 components in the last dimension");
        TORCH_CHECK_VALUE(depths.dim() == 2, "depths must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(depths.size(0) == numCameras,
                          "depths must have numCameras in the first dimension");
        TORCH_CHECK_VALUE(depths.size(1) == means2d.size(1),
                          "depths must have the same number of points as means2d");
    }

    if (isSparse) {
        TORCH_CHECK(tileMask.value().is_cuda(), "tileMask must be a CUDA tensor");
        TORCH_CHECK_VALUE(tileMask.value().dim() == 3, "tileMask must have 3 dimensions (C, H, W)");
        TORCH_CHECK_VALUE(tileMask.value().size(0) == numCameras,
                          "tile_mask must have numCameras in the first dimension");
        TORCH_CHECK_VALUE(tileMask.value().size(1) == numTilesH,
                          "tileMask must have numTilesH in the second dimension");
        TORCH_CHECK_VALUE(tileMask.value().size(2) == numTilesW,
                          "tileMask must have numTilesW in the third dimension");
        TORCH_CHECK(activeTiles.value().is_cuda(), "activeTiles must be a CUDA tensor");
        TORCH_CHECK_VALUE(activeTiles.value().dim() == 1,
                          "activeTiles must have 1 dimension (numActiveTiles)");
    }

    const uint32_t numGaussians   = isPacked ? means2d.size(0) : means2d.size(1);
    const uint32_t totalGaussians = isPacked ? means2d.size(0) : numCameras * numGaussians;

    // const uint32_t numCameras      = means2d.size(0);
    const uint32_t totalTiles    = numTilesH * numTilesW;
    const uint32_t numTileIdBits = (uint32_t)floor(log2(totalTiles)) + 1;
    const uint32_t numCamIdBits  = (uint32_t)floor(log2(numCameras)) + 1;
    const auto cameraJIdxPtr =
        cameraJIdx.has_value() ? cameraJIdx.value().const_data_ptr<int32_t>() : nullptr;

    const auto deviceGuard = at::cuda::OptionalCUDAGuard(at::device_of(means2d));
    const auto stream      = at::cuda::getCurrentCUDAStream(means2d.device().index());

    const uint32_t numActiveTiles =
        isSparse ? activeTiles.value().size(0) : numCameras * totalTiles;
    const uint32_t outputNumTiles = numActiveTiles + 1;

    auto dims = isSparse ? std::vector<int64_t>({outputNumTiles})
                         : std::vector<int64_t>({numCameras, numTilesH, numTilesW});

    auto outputDims = at::IntArrayRef(dims);

    if (totalGaussians == 0) {
        return std::make_tuple(torch::zeros(outputDims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({0}, means2d.options().dtype(torch::kInt32)));
    }
    using scalar_t = float;

    // Allocate tensor to store the number of tiles each gaussian intersects
    torch::Tensor tilesPerGaussianCumsum =
        torch::empty({totalGaussians}, means2d.options().dtype(torch::kInt32));

    const auto tileMaskPtr =
        tileMask.has_value() ? tileMask.value().const_data_ptr<bool>() : nullptr;

    // Count the number of tiles each Gaussian intersects, store in tiles_per_gaussian_cumsum
    const int NUM_BLOCKS = (totalGaussians + NUM_THREADS - 1) / NUM_THREADS;
    countTilesPerGaussian<scalar_t, int32_t>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(0,
                                                 totalGaussians,
                                                 numGaussians,
                                                 tileSize,
                                                 numTilesW,
                                                 numTilesH,
                                                 totalTiles,
                                                 0,
                                                 static_cast<int64_t>(numCameras) * totalTiles,
                                                 means2d.const_data_ptr<scalar_t>(),
                                                 radii.const_data_ptr<int32_t>(),
                                                 tileMaskPtr,
                                                 cameraJIdxPtr,
                                                 tilesPerGaussianCumsum.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // cumulative sum to get the total number of intersections
    tilesPerGaussianCumsum = torch::cumsum(tilesPerGaussianCumsum, 0, torch::kInt32);

    // Allocate tensors to store the intersections
    const int64_t totalIntersections = tilesPerGaussianCumsum[-1].item<int64_t>();
    if (totalIntersections == 0) {
        return std::make_tuple(torch::zeros(outputDims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({0}, means2d.options().dtype(torch::kInt32)));
    } else {
        torch::Tensor intersectionKeys =
            torch::empty({totalIntersections}, means2d.options().dtype(torch::kInt64));
        torch::Tensor intersectionValues =
            torch::empty({totalIntersections}, means2d.options().dtype(torch::kInt32));

        // Compute the set of intersections between each projected Gaussian and each tile,
        // store them in intersection_keys and intersection_values
        // where intersection_keys encodes (camera_id, tile_id, depth) and intersection_values
        // encodes the index of the Gaussian in the input arrays.
        computeGaussianTileIntersections<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            numCameras,
            numGaussians,
            0,
            totalGaussians,
            tileSize,
            numTilesW,
            numTilesH,
            numTileIdBits,
            totalTiles,
            0,
            static_cast<int64_t>(numCameras) * totalTiles,
            means2d.const_data_ptr<scalar_t>(),
            radii.const_data_ptr<int32_t>(),
            depths.const_data_ptr<scalar_t>(),
            tilesPerGaussianCumsum.const_data_ptr<int32_t>(),
            tileMaskPtr,
            cameraJIdxPtr,
            intersectionKeys.data_ptr<int64_t>(),
            intersectionValues.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Sort the intersections by their key so intersections within the same tile are grouped
        // together and sorted by their depth (near to far).
        {
            // Allocate tensors to store the sorted intersections
            torch::Tensor keysSorted = torch::empty_like(intersectionKeys);
            torch::Tensor valsSorted = torch::empty_like(intersectionValues);

            // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
            // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(intersectionKeys.data_ptr<int64_t>(),
                                              keysSorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_vals(intersectionValues.data_ptr<int32_t>(),
                                              valsSorted.data_ptr<int32_t>());

            const int32_t numBits = 32 + numCamIdBits + numTileIdBits;
            FVDB_CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
                             d_keys,
                             d_vals,
                             totalIntersections,
                             0,
                             numBits,
                             stream);

            // DoubleBuffer swaps the pointers if the keys were sorted in the input buffer
            // so we need to grab the right buffer.
            if (d_keys.selector == 1) {
                intersectionKeys = keysSorted;
            }
            if (d_vals.selector == 1) {
                intersectionValues = valsSorted;
            }
        }

        if (isSparse) {
            // Compute a joffsets tensor that stores the offsets into the sorted Gaussian
            // intersections
            torch::Tensor tileJOffsets =
                torch::empty({outputNumTiles}, means2d.options().dtype(torch::kInt32));
            const int NUM_BLOCKS_2 = (outputNumTiles + NUM_THREADS - 1) / NUM_THREADS;
            computeTileOffsetsSparse<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                totalIntersections,
                totalTiles,
                numTileIdBits,
                intersectionKeys.const_data_ptr<int64_t>(),
                activeTiles.value().const_data_ptr<int32_t>(),
                numActiveTiles,
                tileJOffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(tileJOffsets, intersectionValues);
        } else {
            // Compute a joffsets tensor that stores the offsets into the sorted Gaussian
            // intersections
            torch::Tensor tileJOffsets = torch::empty({numCameras, numTilesH, numTilesW},
                                                      means2d.options().dtype(torch::kInt32));
            const int NUM_BLOCKS_2     = (totalIntersections + NUM_THREADS - 1) / NUM_THREADS;
            computeTileOffsets<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                0,
                totalIntersections,
                totalIntersections,
                numCameras,
                totalTiles,
                numTileIdBits,
                intersectionKeys.const_data_ptr<int64_t>(),
                tileJOffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(tileJOffsets, intersectionValues);
        }
    }
}

namespace {

[[maybe_unused]] __global__ void
sleepKernel() {
    __nanosleep(1000000U); // Puts the calling thread to sleep
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>
intersectGaussianTilesPrivateUse1Impl(
    const torch::Tensor &means2d,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                     // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                    // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx,  // NULL or [M]
    const at::optional<torch::Tensor> &tileMask,    // NULL or [C, H, W]
    const at::optional<torch::Tensor> &activeTiles, // NULL or [numActiveTiles]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    const bool isPacked = cameraJIdx.has_value();
    const bool isSparse = activeTiles.has_value();

    TORCH_CHECK(means2d.is_privateuseone(), "means2d must be a PrivateUse1 tensor");
    TORCH_CHECK(radii.is_privateuseone(), "radii must be a PrivateUse1 tensor");
    TORCH_CHECK(depths.is_privateuseone(), "depths must be a PrivateUse1 tensor");

    if (isPacked) {
        TORCH_CHECK(cameraJIdx.value().is_privateuseone(),
                    "camera_jidx must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(means2d.dim() == 2, "means2d must have 2 dimensions (M, 2)");
        TORCH_CHECK_VALUE(means2d.size(1) == 2, "means2d must have 2 points in the last dimension");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have 2 dimensions (M, 2)");
        TORCH_CHECK_VALUE(radii.size(1) == 2, "radii must have 2 components in the last dimension");
        TORCH_CHECK_VALUE(depths.dim() == 1, "depths must have 1 dimension (M)");
        TORCH_CHECK_VALUE(cameraJIdx.value().dim() == 1, "cameraJIdx must have 1 dimension (M)");
        TORCH_CHECK_VALUE(radii.size(0) == means2d.size(0),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(depths.size(0) == means2d.size(0),
                          "depths must have the same number of points as means2d");
        TORCH_CHECK_VALUE(cameraJIdx.value().size(0) == means2d.size(0),
                          "cameraJIdx must have the same number of points as means2d");
    } else {
        TORCH_CHECK_VALUE(means2d.dim() == 3, "means2d must have 3 dimensions (C, N, 2)");
        TORCH_CHECK_VALUE(means2d.size(0) == numCameras,
                          "means2d must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(means2d.size(2) == 2, "means2d must have 2 points in the last dimension");
        TORCH_CHECK_VALUE(radii.dim() == 3, "radii must have 3 dimensions (C, N, 2)");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(radii.size(1) == means2d.size(1),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(radii.size(2) == 2, "radii must have 2 components in the last dimension");
        TORCH_CHECK_VALUE(depths.dim() == 2, "depths must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(depths.size(0) == numCameras,
                          "depths must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(depths.size(1) == means2d.size(1),
                          "depths must have the same number of points as means2d");
    }

    if (isSparse) {
        TORCH_CHECK(tileMask.value().is_privateuseone(), "tileMask must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(tileMask.value().dim() == 3, "tileMask must have 3 dimensions (C, H, W)");
        TORCH_CHECK_VALUE(tileMask.value().size(0) == numCameras,
                          "tileMask must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(tileMask.value().size(1) == numTilesH,
                          "tileMask must have numTilesH in the second dimension");
        TORCH_CHECK_VALUE(tileMask.value().size(2) == numTilesW,
                          "tileMask must have numTilesW in the third dimension");
        TORCH_CHECK(activeTiles.value().is_privateuseone(),
                    "activeTiles must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(activeTiles.value().dim() == 1,
                          "activeTiles must have 1 dimension (numActiveTiles)");
    }

    const uint32_t numGaussians   = isPacked ? means2d.size(0) : means2d.size(1);
    const uint32_t totalGaussians = isPacked ? means2d.size(0) : numCameras * numGaussians;

    // const uint32_t numCameras      = means2d.size(0);
    const uint32_t totalTiles    = numTilesH * numTilesW;
    const uint32_t numTileIdBits = (uint32_t)floor(log2(totalTiles)) + 1;
    const auto cameraJIdxPtr =
        cameraJIdx.has_value() ? cameraJIdx.value().const_data_ptr<int32_t>() : nullptr;

    const uint32_t numActiveTiles =
        isSparse ? activeTiles.value().size(0) : numCameras * totalTiles;
    const uint32_t outputNumTiles = numActiveTiles + 1;

    auto dims = isSparse ? std::vector<int64_t>({outputNumTiles})
                         : std::vector<int64_t>({numCameras, numTilesH, numTilesW});

    auto outputDims = at::IntArrayRef(dims);

    if (totalGaussians == 0) {
        return std::make_tuple(torch::zeros(outputDims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({0}, means2d.options().dtype(torch::kInt32)));
    }
    using scalar_t = float;

    const auto tileMaskPtr =
        tileMask.has_value() ? tileMask.value().const_data_ptr<bool>() : nullptr;

    const int deviceCount       = static_cast<int>(c10::cuda::device_count());
    const int64_t totalTileKeys = static_cast<int64_t>(numCameras) * totalTiles;

    // Give each device a contiguous range of the tile-key space [0, numCameras * totalTiles). The
    // linear tile-key L = cidx * totalTiles + tileIdx is monotonic in (camera, tile) -- the high
    // bits of the sort key -- so every key on device d is less than every key on device d+1. Each
    // device can then sort its own intersections independently and their concatenation is globally
    // sorted, with no cross-device merge.

    // For each device, count the in-range tiles per Gaussian, then scan in place to get the
    // per-Gaussian write offsets. Each device gets its own buffer (not a single
    // [deviceCount, totalGaussians] tensor, which the multi-GPU allocator would stripe) so its row
    // stays device-local; count and scan share the per-device stream, so no merge is needed, and a
    // single-device CUB scan avoids torch::cumsum dispatching to the multi-GPU cumsum.
    std::vector<int32_t *> deviceTilesPerGaussianCumsum(deviceCount, nullptr);

    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        C10_CUDA_CHECK(cudaMallocAsync(
            &deviceTilesPerGaussianCumsum[deviceId], totalGaussians * sizeof(int32_t), stream));

        size_t deviceKeyOffset, deviceKeyCount;
        std::tie(deviceKeyOffset, deviceKeyCount) = deviceChunk(totalTileKeys, deviceId);
        const int64_t keyStart                    = static_cast<int64_t>(deviceKeyOffset);
        const int64_t keyEnd = static_cast<int64_t>(deviceKeyOffset + deviceKeyCount);

        // Every device scans all Gaussians but counts only the tiles that fall in its tile-key
        // range.
        const int NUM_BLOCKS = (totalGaussians + NUM_THREADS - 1) / NUM_THREADS;
        countTilesPerGaussian<scalar_t, int32_t>
            <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(0,
                                                     totalGaussians,
                                                     numGaussians,
                                                     tileSize,
                                                     numTilesW,
                                                     numTilesH,
                                                     totalTiles,
                                                     keyStart,
                                                     keyEnd,
                                                     means2d.const_data_ptr<scalar_t>(),
                                                     radii.const_data_ptr<int32_t>(),
                                                     tileMaskPtr,
                                                     cameraJIdxPtr,
                                                     deviceTilesPerGaussianCumsum[deviceId]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Inclusive scan of the counts in place to get the per-Gaussian write offsets.
        FVDB_CUB_WRAPPER_ASYNC(stream,
                               cub::DeviceScan::InclusiveSum,
                               deviceTilesPerGaussianCumsum[deviceId],
                               deviceTilesPerGaussianCumsum[deviceId],
                               totalGaussians);
    }

    // Read each device's scan total back to the host. mergeStreams() only orders the device
    // streams relative to each other and does not block the host, so it cannot make the reads below
    // safe; an explicit per-device synchronize is required.
    // deviceIntersectionOffset[d] is the start offset of device d's intersections in the
    // concatenated (globally sorted) output, and deviceIntersectionCount[d] is the number it owns.
    std::vector<int64_t> deviceIntersectionOffset(deviceCount);
    std::vector<int64_t> deviceIntersectionCount(deviceCount);
    int64_t totalIntersections = 0;
    int32_t *deviceTotals      = nullptr;
    C10_CUDA_CHECK(cudaMallocHost(&deviceTotals, deviceCount * sizeof(int32_t)));
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        // The total is the last element of the device's inclusive-scan row. Issuing the copy on the
        // device's stream orders it after the scan; the synchronize below then guarantees it has
        // completed before we read deviceTotals on the host.
        C10_CUDA_CHECK(cudaMemcpyAsync(&deviceTotals[deviceId],
                                       deviceTilesPerGaussianCumsum[deviceId] + totalGaussians - 1,
                                       sizeof(int32_t),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    }
    for (const auto deviceId: c10::irange(deviceCount)) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        deviceIntersectionOffset[deviceId] = totalIntersections;
        deviceIntersectionCount[deviceId]  = deviceTotals[deviceId];
        totalIntersections += deviceTotals[deviceId];
    }
    C10_CUDA_CHECK(cudaFreeHost(deviceTotals));

    if (totalIntersections == 0) {
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaFreeAsync(deviceTilesPerGaussianCumsum[deviceId], stream));
        }
        return std::make_tuple(torch::zeros(outputDims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({0}, means2d.options().dtype(torch::kInt32)));
    } else {
        torch::Tensor intersectionKeys =
            torch::empty({totalIntersections}, means2d.options().dtype(torch::kInt64));
        torch::Tensor intersectionValues =
            torch::empty({totalIntersections}, means2d.options().dtype(torch::kInt32));

        // Compute a joffsets tensor that stores the offsets into the sorted Gaussian intersections
        torch::Tensor tileJOffsets = torch::empty({numCameras, numTilesH, numTilesW},
                                                  means2d.options().dtype(torch::kInt32));

        std::vector<cudaEvent_t> events(deviceCount);
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaEventCreate(&events[deviceId], cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
        }

        // Prefetch each device's slice [intersectionOffset, intersectionOffset + intersectionCount)
        // of the intersection arrays to that device so the emit, sort, and offset kernels below
        // operate on local memory.
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getStreamFromPool(false, deviceId);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));

            const int64_t intersectionOffset = deviceIntersectionOffset[deviceId];
            const int64_t intersectionCount  = deviceIntersectionCount[deviceId];
            if (intersectionCount > 0) {
#if (CUDART_VERSION < 13000)
                sleepKernel<<<1, 1, 0, stream>>>();
                nanovdb::util::cuda::memPrefetchAsync(intersectionKeys.data_ptr<int64_t>() +
                                                          intersectionOffset,
                                                      intersectionCount * sizeof(int64_t),
                                                      deviceId,
                                                      stream);
                nanovdb::util::cuda::memPrefetchAsync(intersectionValues.data_ptr<int32_t>() +
                                                          intersectionOffset,
                                                      intersectionCount * sizeof(int32_t),
                                                      deviceId,
                                                      stream);
#else
                std::vector<void *> prefetchPointers;
                std::vector<size_t> prefetchSizes;
                const cudaMemLocation location = {cudaMemLocationTypeDevice, deviceId};
                std::vector<cudaMemLocation> prefetchLocations = {location};
                std::vector<size_t> prefetchLocationIndices    = {0};

                prefetchPointers.emplace_back(intersectionKeys.data_ptr<int64_t>() +
                                              intersectionOffset);
                prefetchSizes.emplace_back(intersectionCount * sizeof(int64_t));
                prefetchPointers.emplace_back(intersectionValues.data_ptr<int32_t>() +
                                              intersectionOffset);
                prefetchSizes.emplace_back(intersectionCount * sizeof(int32_t));

                sleepKernel<<<1, 1, 0, stream>>>();
                C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPointers.data(),
                                                         prefetchSizes.data(),
                                                         prefetchPointers.size(),
                                                         prefetchLocations.data(),
                                                         prefetchLocationIndices.data(),
                                                         prefetchLocations.size(),
                                                         0,
                                                         stream));
#endif
            }
            C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
        }

        // Compute the intersections owned by each device. Each device scans all Gaussians and
        // writes only the intersections whose tile-key falls in its range, into its own contiguous
        // slice [intersectionOffset, intersectionOffset + intersectionCount) of the global
        // intersection arrays.
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));

            size_t deviceKeyOffset, deviceKeyCount;
            std::tie(deviceKeyOffset, deviceKeyCount) = deviceChunk(totalTileKeys, deviceId);
            const int64_t keyStart                    = static_cast<int64_t>(deviceKeyOffset);
            const int64_t keyEnd = static_cast<int64_t>(deviceKeyOffset + deviceKeyCount);
            const int64_t intersectionOffset = deviceIntersectionOffset[deviceId];

            const int NUM_BLOCKS = (totalGaussians + NUM_THREADS - 1) / NUM_THREADS;
            computeGaussianTileIntersections<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                numCameras,
                numGaussians,
                0,
                totalGaussians,
                tileSize,
                numTilesW,
                numTilesH,
                numTileIdBits,
                totalTiles,
                keyStart,
                keyEnd,
                means2d.const_data_ptr<scalar_t>(),
                radii.const_data_ptr<int32_t>(),
                depths.const_data_ptr<scalar_t>(),
                deviceTilesPerGaussianCumsum[deviceId],
                tileMaskPtr,
                cameraJIdxPtr,
                intersectionKeys.data_ptr<int64_t>() + intersectionOffset,
                intersectionValues.data_ptr<int32_t>() + intersectionOffset);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // The emit kernel is the last reader of this device's cumulative-count buffer; free it
            // on the same stream so the free is ordered after the kernel.
            C10_CUDA_CHECK(cudaFreeAsync(deviceTilesPerGaussianCumsum[deviceId], stream));
        }

        // Each device independently sorts its own slice in place. Because the tile-key ranges are
        // disjoint and ordered, the concatenation of the sorted slices is globally sorted, so no
        // cross-device merge is needed. DeviceMergeSort sorts in place, avoiding the separate
        // output arrays (and their allocations) a radix sort would require. No mergeStreams() is
        // needed before this: each device's sort reads only its own slice, written by that device's
        // emit on the same stream, so it is already ordered after the emit without a global
        // barrier.
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

            const int64_t intersectionOffset = deviceIntersectionOffset[deviceId];
            const int64_t intersectionCount  = deviceIntersectionCount[deviceId];
            if (intersectionCount > 0) {
                FVDB_CUB_WRAPPER_ASYNC(stream,
                                       cub::DeviceMergeSort::SortPairs,
                                       intersectionKeys.data_ptr<int64_t>() + intersectionOffset,
                                       intersectionValues.data_ptr<int32_t>() + intersectionOffset,
                                       intersectionCount,
                                       ::cuda::std::less<int64_t>{});
            }
        }

        mergeStreams();

        TORCH_CHECK(!isSparse, "Sparse tile offsets are not implemented for mGPU");

        // Compute the tile offsets. intersectionKeys is now contiguous and globally sorted, so each
        // device computes the offsets for its own index slice; the kernel's boundary fills read
        // across slice boundaries through unified memory as needed, which also fills the tile
        // ranges of any devices that own no intersections.
        for (const auto deviceId: c10::irange(deviceCount)) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));

            const int64_t intersectionOffset = deviceIntersectionOffset[deviceId];
            const int64_t intersectionCount  = deviceIntersectionCount[deviceId];
            if (intersectionCount > 0) {
                const int NUM_BLOCKS_2 = cuda::ceil_div<int64_t>(intersectionCount, NUM_THREADS);
                computeTileOffsets<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                    intersectionOffset,
                    intersectionCount,
                    totalIntersections,
                    numCameras,
                    totalTiles,
                    numTileIdBits,
                    intersectionKeys.const_data_ptr<int64_t>(),
                    tileJOffsets.data_ptr<int32_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

        mergeStreams();

        return std::make_tuple(tileJOffsets, intersectionValues);
    }
}

} // namespace

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTiles(const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
                               const torch::Tensor &radii,                   // [C, N, 2] or [M, 2]
                               const torch::Tensor &depths,                  // [C, N] or [M]
                               const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                               const uint32_t numCameras,
                               const uint32_t tileSize,
                               const uint32_t numTilesH,
                               const uint32_t numTilesW);

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTilesSparse(const torch::Tensor &means2d,     // [C, N, 2] or [M, 2]
                                     const torch::Tensor &radii,       // [C, N, 2] or [M, 2]
                                     const torch::Tensor &depths,      // [C, N] or [M]
                                     const torch::Tensor &tileMask,    // [C, H, W]
                                     const torch::Tensor &activeTiles, // [num_active_tiles]
                                     const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                     const uint32_t numCameras,
                                     const uint32_t tileSize,
                                     const uint32_t numTilesH,
                                     const uint32_t numTilesW);

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTiles<torch::kCUDA>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return intersectGaussianTilesCudaImpl(means2d,
                                          radii,
                                          depths,
                                          cameraJIdx,
                                          at::nullopt,
                                          at::nullopt,
                                          numCameras,
                                          tileSize,
                                          numTilesH,
                                          numTilesW);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTiles<torch::kPrivateUse1>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return intersectGaussianTilesPrivateUse1Impl(means2d,
                                                 radii,
                                                 depths,
                                                 cameraJIdx,
                                                 at::nullopt,
                                                 at::nullopt,
                                                 numCameras,
                                                 tileSize,
                                                 numTilesH,
                                                 numTilesW);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTiles<torch::kCPU>(
    const torch::Tensor &means2d,                 // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &depths,                  // [C, N] or [nnz]
    const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    TORCH_CHECK(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTilesSparse<torch::kCUDA>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const torch::Tensor &tileMask,                 // [C, H, W]
    const torch::Tensor &activeTiles,              // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return intersectGaussianTilesCudaImpl(means2d,
                                          radii,
                                          depths,
                                          cameraJIdx,
                                          tileMask,
                                          activeTiles,
                                          numCameras,
                                          tileSize,
                                          numTilesH,
                                          numTilesW);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTilesSparse<torch::kPrivateUse1>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const torch::Tensor &tileMask,                 // [C, H, W]
    const torch::Tensor &activeTiles,              // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    // Sparse tile intersection is not implemented for multi-GPU (PrivateUse1)
    // The PrivateUse1 impl already checks for this and throws an appropriate error
    return intersectGaussianTilesPrivateUse1Impl(means2d,
                                                 radii,
                                                 depths,
                                                 cameraJIdx,
                                                 tileMask,
                                                 activeTiles,
                                                 numCameras,
                                                 tileSize,
                                                 numTilesH,
                                                 numTilesW);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchIntersectGaussianTilesSparse<torch::kCPU>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N, 2] or [M, 2]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const torch::Tensor &tileMask,                 // [C, H, W]
    const torch::Tensor &activeTiles,              // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    TORCH_CHECK(false, "CPU implementation not available for sparse tile intersection");
}

std::tuple<torch::Tensor, torch::Tensor>
intersectGaussianTiles(const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
                       const torch::Tensor &radii,                   // [C, N, 2] or [M, 2]
                       const torch::Tensor &depths,                  // [C, N] or [M]
                       const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                       const uint32_t numCameras,
                       const uint32_t tileSize,
                       const uint32_t numTilesH,
                       const uint32_t numTilesW) {
    return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return dispatchIntersectGaussianTiles<DeviceTag>(
            means2d, radii, depths, cameraIds, numCameras, tileSize, numTilesH, numTilesW);
    });
}

std::tuple<torch::Tensor, torch::Tensor>
intersectGaussianTilesSparse(const torch::Tensor &means2d,                 // [C, N, 2] or [M, 2]
                             const torch::Tensor &radii,                   // [C, N, 2] or [M, 2]
                             const torch::Tensor &depths,                  // [C, N] or [M]
                             const torch::Tensor &tileMask,                // [C, H, W]
                             const torch::Tensor &activeTiles,             // [num_active_tiles]
                             const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                             const uint32_t numCameras,
                             const uint32_t tileSize,
                             const uint32_t numTilesH,
                             const uint32_t numTilesW) {
    return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return dispatchIntersectGaussianTilesSparse<DeviceTag>(means2d,
                                                               radii,
                                                               depths,
                                                               tileMask,
                                                               activeTiles,
                                                               cameraIds,
                                                               numCameras,
                                                               tileSize,
                                                               numTilesH,
                                                               numTilesW);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
