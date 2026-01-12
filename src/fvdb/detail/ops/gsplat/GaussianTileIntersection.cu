// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDAGuard.h>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

#define NUM_THREADS 256

#define CUB_WRAPPER(func, ...)                                                \
    do {                                                                      \
        size_t tempStorageBytes = 0;                                          \
        func(nullptr, tempStorageBytes, __VA_ARGS__);                         \
        auto &cachingAllocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto tempStorage       = cachingAllocator.allocate(tempStorageBytes); \
        func(tempStorage.get(), tempStorageBytes, __VA_ARGS__);               \
    } while (false)

// Compute the number of 2d image tiles intersected by a set of 2D projected Gaussians.
//
// The input is a set of 2D circles with depths approximating the projection of 3D gaussians onto
// the image plane. Each input circle is encoded as a tuple:
// (mean_u, mean_v, radius, depth)
// where (mean_u, mean_v) are the image-space center of the circle, radius is its radius (in pixels)
// and depth is the (world-space) depth of the Gaussian this circle is approximating.
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
                      const T *__restrict__ means2d,                 // [C, N, 2] or [M, 2]
                      const int32_t *__restrict__ radii,             // [C, N]    or [M]
                      const bool *__restrict__ tileMask,             // [C, H, W] or nullptr
                      const int32_t *__restrict__ cameraJIdx,        // NULL or [M]
                      CountT *__restrict__ outNumTilesPerGaussian) { // [ C * N ] or [ M ]
    // parallelize over gaussianCount
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gaussianCount;
         idx += blockDim.x * gridDim.x) {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = typename OpType<T>::type;

        auto gidx        = idx + gaussianOffset;
        const OpT radius = radii[gidx];
        if (radius <= 0) {
            outNumTilesPerGaussian[gidx] = static_cast<CountT>(0);
        } else {
            using vec2f = typename Vec2Type<OpT>::type;

            const vec2f mean2d   = *reinterpret_cast<const vec2f *>(means2d + gidx * 2);
            const OpT tileRadius = radius / static_cast<OpT>(tileSize);
            const OpT tileMeanU  = mean2d.x / static_cast<OpT>(tileSize);
            const OpT tileMeanV  = mean2d.y / static_cast<OpT>(tileSize);

            // tile_min is inclusive, tile_max is exclusive
            uint2 tileMin, tileMax;
            tileMin.x = min(max(0, (uint32_t)floor(tileMeanU - tileRadius)), numTilesW);
            tileMin.y = min(max(0, (uint32_t)floor(tileMeanV - tileRadius)), numTilesH);
            tileMax.x = min(max(0, (uint32_t)ceil(tileMeanU + tileRadius)), numTilesW);
            tileMax.y = min(max(0, (uint32_t)ceil(tileMeanV + tileRadius)), numTilesH);

            outNumTilesPerGaussian[gidx] = [&]() {
                if (tileMask) {
                    CountT numTiles    = 0;
                    const int32_t cidx = (cameraJIdx == nullptr)
                                             ? static_cast<int32_t>(gidx / numGaussiansPerCamera)
                                             : cameraJIdx[gidx];
                    // loop min / max range and count number of tiles
                    for (uint32_t i = tileMin.y; i < tileMax.y; ++i) {
                        for (uint32_t j = tileMin.x; j < tileMax.x; ++j) {
                            if (tileMask[cidx * numTilesH * numTilesW + i * numTilesW + j]) {
                                numTiles++;
                            }
                        }
                    }
                    return numTiles;
                } else {
                    // write out number of tiles per gaussian
                    const CountT numTiles =
                        static_cast<CountT>((tileMax.y - tileMin.y) * (tileMax.x - tileMin.x));
                    return numTiles;
                }
            }();
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
// The input is a set of 2D circles with depths approximating the projection of 3D gaussians
// onto the image plane. Each input circle is encoded as a tuple: (mean_u, mean_v, radius,
// depth) where (mean_u, mean_v) are the image-space center of the circle, radius is its radius
// (in pixels) and depth is the (world-space) depth of the Gaussian this circle is
// approximating.
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
    const T *__restrict__ means2d,                   // [C, N, 2] or [M, 2]
    const int32_t *__restrict__ radii,               // [C, N]    or [M]
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
        using OpT = typename OpType<T>::type;

        auto gidx = idx + gaussianOffset;
        // Get the camera id from the batch indices or use the camera index directly
        const int32_t cidx =
            cameraJIdx == nullptr ? gidx / numGaussiansPerCamera : cameraJIdx[gidx];

        const OpT radius = radii[gidx];
        if (radius > 0) {
            using vec2f = typename Vec2Type<OpT>::type;

            const vec2f mean2d   = *reinterpret_cast<const vec2f *>(means2d + 2 * gidx);
            const OpT tileRadius = radius / static_cast<OpT>(tileSize);
            const OpT tileMeanU  = mean2d.x / static_cast<OpT>(tileSize);
            const OpT tileMeanV  = mean2d.y / static_cast<OpT>(tileSize);

            // tile_min is inclusive, tile_max is exclusive
            uint2 tileMin, tileMax;
            tileMin.x = min(max(0, (uint32_t)floor(tileMeanU - tileRadius)), numTilesW);
            tileMin.y = min(max(0, (uint32_t)floor(tileMeanV - tileRadius)), numTilesH);
            tileMax.x = min(max(0, (uint32_t)ceil(tileMeanU + tileRadius)), numTilesW);
            tileMax.y = min(max(0, (uint32_t)ceil(tileMeanV + tileRadius)), numTilesH);

            // If you use float64, we're casting you to float32 so we can
            // pack the depth into the key. In principle this loses precision,
            // in practice it's fine.
            const float depth = depths[gidx];

            // Suppose you're using tile_id_bits = 22, then the output for this intersection is
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            // which we pack into an int64_t
            const int64_t depthEnc = encodeDepth(depth);

            // For each tile this Gaussian intersects, write out an intersection tuple
            // {(camera_id | tile_id | depth), gaussian_id} (int64_t, int32_t)
            int64_t curIsect = (gidx == 0) ? 0 : cumTilesPerGaussian[gidx - 1];
            for (int32_t i = tileMin.y; i < tileMax.y; ++i) {
                for (int32_t j = tileMin.x; j < tileMax.x; ++j) {
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
                         const uint32_t *__restrict__ activeTiles,
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
gaussianTileIntersectionCUDAImpl(
    const torch::Tensor &means2d,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                     // [C, N] or [M]
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
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have 1 dimension (M)");
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
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have numCameras in the first dimension");
        TORCH_CHECK_VALUE(radii.size(1) == means2d.size(1),
                          "radii must have the same number of points as means2d");
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
        cameraJIdx.has_value() ? cameraJIdx.value().data_ptr<int32_t>() : nullptr;

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

    const auto tileMaskPtr = tileMask.has_value() ? tileMask.value().data_ptr<bool>() : nullptr;

    // Count the number of tiles each Gaussian intersects, store in tiles_per_gaussian_cumsum
    const int NUM_BLOCKS = (totalGaussians + NUM_THREADS - 1) / NUM_THREADS;
    countTilesPerGaussian<scalar_t, int32_t>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(0,
                                                 totalGaussians,
                                                 numGaussians,
                                                 tileSize,
                                                 numTilesW,
                                                 numTilesH,
                                                 means2d.data_ptr<scalar_t>(),
                                                 radii.data_ptr<int32_t>(),
                                                 tileMaskPtr,
                                                 cameraJIdxPtr,
                                                 tilesPerGaussianCumsum.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // In place cumulative sum to get the total number of intersections
    torch::cumsum_out(tilesPerGaussianCumsum, tilesPerGaussianCumsum, 0, torch::kInt32);

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
        computeGaussianTileIntersections<scalar_t>
            <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(numCameras,
                                                     numGaussians,
                                                     0,
                                                     totalGaussians,
                                                     tileSize,
                                                     numTilesW,
                                                     numTilesH,
                                                     numTileIdBits,
                                                     means2d.data_ptr<scalar_t>(),
                                                     radii.data_ptr<int32_t>(),
                                                     depths.data_ptr<scalar_t>(),
                                                     tilesPerGaussianCumsum.data_ptr<int32_t>(),
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
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
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
                intersectionKeys.data_ptr<int64_t>(),
                activeTiles.value().data_ptr<uint32_t>(),
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
                intersectionKeys.data_ptr<int64_t>(),
                tileJOffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(tileJOffsets, intersectionValues);
        }
    }
}

/// @brief Implements the merge path binary search algorithm in order to find the median across two
/// sorted input key arrays
template <typename KeyIteratorIn>
__device__ void
mergePath(KeyIteratorIn keys1,
          size_t keys1Count,
          KeyIteratorIn keys2,
          size_t keys2Count,
          ptrdiff_t *key1Intervals,
          ptrdiff_t *key2Intervals,
          int intervalIndex) {
    using KeyType = typename ::cuda::std::iterator_traits<KeyIteratorIn>::value_type;

    const size_t combinedIndex = intervalIndex * (keys1Count + keys2Count) / 2;
    size_t leftTop             = combinedIndex > keys1Count ? keys1Count : combinedIndex;
    size_t rightTop            = combinedIndex > keys1Count ? combinedIndex - keys1Count : 0;
    size_t leftBottom          = rightTop;

    KeyType leftKey;
    KeyType rightKey;
    while (true) {
        ptrdiff_t offset   = (leftTop - leftBottom) / 2;
        ptrdiff_t leftMid  = leftTop - offset;
        ptrdiff_t rightMid = rightTop + offset;

        if (leftMid > keys1Count - 1 || rightMid < 1) {
            leftKey  = 1;
            rightKey = 0;
        } else {
            leftKey  = *(keys1 + leftMid);
            rightKey = *(keys2 + rightMid - 1);
        }

        if (leftKey > rightKey) {
            if (rightMid > keys2Count - 1 || leftMid < 1) {
                leftKey  = 0;
                rightKey = 1;
            } else {
                leftKey  = *(keys1 + leftMid - 1);
                rightKey = *(keys2 + rightMid);
            }

            if (leftKey <= rightKey) {
                *key1Intervals = leftMid;
                *key2Intervals = rightMid;
                break;
            } else {
                leftTop  = leftMid - 1;
                rightTop = rightMid + 1;
            }
        } else {
            leftBottom = leftMid + 1;
        }
    }
}

/// @brief Kernel wrapper for the merge path algorithm
template <typename KeyIteratorIn>
__global__ void
mergePathKernel(KeyIteratorIn keys1,
                size_t keys1Count,
                KeyIteratorIn keys2,
                size_t keys2Count,
                ptrdiff_t *key1Intervals,
                ptrdiff_t *key2Intervals,
                size_t intervalOffset) {
    const unsigned int intervalIndex = threadIdx.x + blockIdx.x * blockDim.x + intervalOffset;
    mergePath(keys1, keys1Count, keys2, keys2Count, key1Intervals, key2Intervals, intervalIndex);
}

template <typename KeyT, typename ValueT, typename NumItemsT>
void
radixSortAsync(KeyT *keysIn,
               KeyT *keysOut,
               ValueT *valuesIn,
               ValueT *valuesOut,
               NumItemsT numItems,
               int beginBit,
               int endBit,
               cudaEvent_t *events) {
    using OffsetT = int64_t;
    using CountT  = int64_t;

    auto hostOptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto itemOffsets = torch::empty({c10::cuda::device_count()}, hostOptions);
    auto itemCounts  = torch::empty({c10::cuda::device_count()}, hostOptions);
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        std::tie(itemOffsets.data_ptr<OffsetT>()[deviceId],
                 itemCounts.data_ptr<CountT>()[deviceId]) = deviceChunk(numItems, deviceId);
    }
    const auto *offsets = itemOffsets.const_data_ptr<OffsetT>();
    const auto *counts  = itemCounts.const_data_ptr<CountT>();

    torch::Tensor deviceMergeIntervals =
        torch::empty({2 * c10::cuda::device_count()},
                     torch::TensorOptions().dtype(torch::kInt64).device(torch::kPrivateUse1));
    auto mergeIntervals = deviceMergeIntervals.data_ptr<OffsetT>();

    // Radix sort the subset of keys assigned to each device in parallel
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        // C10_CUDA_CHECK(cudaEventSynchronize(events[deviceId]));

        const KeyT *deviceKeysIn     = keysIn + offsets[deviceId];
        const ValueT *deviceValuesIn = valuesIn + offsets[deviceId];
        KeyT *deviceKeysOut          = keysOut + offsets[deviceId];
        ValueT *deviceValuesOut      = valuesOut + offsets[deviceId];

        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            deviceKeysIn, counts[deviceId] * sizeof(KeyT), deviceId, stream));
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            deviceValuesIn, counts[deviceId] * sizeof(ValueT), deviceId, stream));
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            deviceKeysOut, counts[deviceId] * sizeof(KeyT), deviceId, stream));
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            deviceValuesOut, counts[deviceId] * sizeof(ValueT), deviceId, stream));

        CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
                    deviceKeysIn,
                    deviceKeysOut,
                    deviceValuesIn,
                    deviceValuesOut,
                    counts[deviceId],
                    beginBit,
                    endBit,
                    stream);
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }

    // TODO: Generalize to numbers of GPUs that aren't powers of two
    // For each pair of devices, merge the local sorts by first computing the median across the two
    // devices followed by merging the elements less than and greater than/equal to the median onto
    // the first and second device of the pair respectively. This avoids the allocating memory for
    // and gathering the values from both devices onto a single device.
    const int log2DeviceCount = log2(c10::cuda::device_count());
    OffsetT *leftIntervals    = mergeIntervals;
    OffsetT *rightIntervals   = mergeIntervals + c10::cuda::device_count();
    for (int deviceExponent = 0; deviceExponent < log2DeviceCount; ++deviceExponent) {
        std::swap(keysIn, keysOut);
        std::swap(valuesIn, valuesOut);
        const int deviceInc   = 1 << deviceExponent;
        const int deviceCount = static_cast<int>(c10::cuda::device_count());

        for (int leftDeviceId = 0; leftDeviceId < deviceCount; leftDeviceId += 2 * deviceInc) {
            const int rightDeviceId = leftDeviceId + deviceInc;

            CountT leftDeviceItemCount = 0;
            for (int deviceId = leftDeviceId; deviceId < rightDeviceId; ++deviceId)
                leftDeviceItemCount += counts[deviceId];

            CountT rightDeviceItemCount = 0;
            for (int deviceId = rightDeviceId; deviceId < rightDeviceId + deviceInc; ++deviceId)
                rightDeviceItemCount += counts[deviceId];

            const KeyT *leftDeviceKeysIn     = keysIn + offsets[leftDeviceId];
            const ValueT *leftDeviceValuesIn = valuesIn + offsets[leftDeviceId];
            const KeyT *rightDeviceKeysIn    = keysIn + offsets[leftDeviceId] + leftDeviceItemCount;
            const ValueT *rightDeviceValuesIn =
                valuesIn + offsets[leftDeviceId] + leftDeviceItemCount;

            // Wait on the prior sort to finish on both devices before computing the median across
            // both devices
            auto mergePathSubfunc = [&](int deviceId, int otherDeviceId, int intervalIndex) {
                C10_CUDA_CHECK(cudaSetDevice(deviceId));

                C10_CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(deviceId),
                                                   events[otherDeviceId]));
                mergePathKernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream(deviceId)>>>(
                    leftDeviceKeysIn,
                    leftDeviceItemCount,
                    rightDeviceKeysIn,
                    rightDeviceItemCount,
                    leftIntervals + deviceId,
                    rightIntervals + deviceId,
                    intervalIndex);
                C10_CUDA_CHECK(
                    cudaEventRecord(events[deviceId], c10::cuda::getCurrentCUDAStream(deviceId)));
            };
            mergePathSubfunc(leftDeviceId, rightDeviceId, 0);
            mergePathSubfunc(rightDeviceId, leftDeviceId, 1);
        }

        for (int leftDeviceId = 0; leftDeviceId < deviceCount; leftDeviceId += 2 * deviceInc) {
            const int rightDeviceId = leftDeviceId + deviceInc;

            CountT leftDeviceItemCount = 0;
            for (int deviceId = leftDeviceId; deviceId < rightDeviceId; ++deviceId)
                leftDeviceItemCount += counts[deviceId];

            CountT rightDeviceItemCount = 0;
            for (int deviceId = rightDeviceId; deviceId < rightDeviceId + deviceInc; ++deviceId)
                rightDeviceItemCount += counts[deviceId];

            const KeyT *leftDeviceKeysIn     = keysIn + offsets[leftDeviceId];
            const ValueT *leftDeviceValuesIn = valuesIn + offsets[leftDeviceId];
            const KeyT *rightDeviceKeysIn    = keysIn + offsets[leftDeviceId] + leftDeviceItemCount;
            const ValueT *rightDeviceValuesIn =
                valuesIn + offsets[leftDeviceId] + leftDeviceItemCount;

            C10_CUDA_CHECK(cudaEventSynchronize(events[leftDeviceId]));
            C10_CUDA_CHECK(cudaEventSynchronize(events[rightDeviceId]));

            // Merge the pairs less than the median to the left device
            {
                C10_CUDA_CHECK(cudaSetDevice(leftDeviceId));
                auto leftStream = c10::cuda::getCurrentCUDAStream(leftDeviceId);

                const KeyT *leftKeysIn     = leftDeviceKeysIn + leftIntervals[leftDeviceId];
                const ValueT *leftValuesIn = leftDeviceValuesIn + leftIntervals[leftDeviceId];
                CountT leftCount = leftIntervals[rightDeviceId] - leftIntervals[leftDeviceId];

                const KeyT *rightKeysIn     = rightDeviceKeysIn + rightIntervals[leftDeviceId];
                const ValueT *rightValuesIn = rightDeviceValuesIn + rightIntervals[leftDeviceId];
                CountT rightCount = rightIntervals[rightDeviceId] - rightIntervals[leftDeviceId];

                OffsetT outputOffset = offsets[leftDeviceId] + leftIntervals[leftDeviceId] +
                                       rightIntervals[leftDeviceId];

                if (leftCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        leftKeysIn, leftCount * sizeof(KeyT), leftDeviceId, leftStream));
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        leftValuesIn, leftCount * sizeof(ValueT), leftDeviceId, leftStream));
                }
                if (rightCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        rightKeysIn, rightCount * sizeof(KeyT), leftDeviceId, leftStream));
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        rightValuesIn, rightCount * sizeof(ValueT), leftDeviceId, leftStream));
                }
                if (auto outputCount = leftCount + rightCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(keysOut + outputOffset,
                                                                         outputCount * sizeof(KeyT),
                                                                         leftDeviceId,
                                                                         leftStream));
                    C10_CUDA_CHECK(
                        nanovdb::util::cuda::memPrefetchAsync(valuesOut + outputOffset,
                                                              outputCount * sizeof(ValueT),
                                                              leftDeviceId,
                                                              leftStream));
                }

                CUB_WRAPPER(cub::DeviceMerge::MergePairs,
                            leftKeysIn,
                            leftValuesIn,
                            leftCount,
                            rightKeysIn,
                            rightValuesIn,
                            rightCount,
                            keysOut + outputOffset,
                            valuesOut + outputOffset,
                            {},
                            leftStream);
                C10_CUDA_CHECK(cudaEventRecord(events[leftDeviceId], leftStream));
            };

            // Merge the pairs greater than/equal to the median to the right device
            {
                C10_CUDA_CHECK(cudaSetDevice(rightDeviceId));
                auto rightStream = c10::cuda::getCurrentCUDAStream(rightDeviceId);

                const KeyT *leftKeysIn     = leftDeviceKeysIn + leftIntervals[rightDeviceId];
                const ValueT *leftValuesIn = leftDeviceValuesIn + leftIntervals[rightDeviceId];
                CountT leftCount           = leftDeviceItemCount - leftIntervals[rightDeviceId];

                const KeyT *rightKeysIn     = rightDeviceKeysIn + rightIntervals[rightDeviceId];
                const ValueT *rightValuesIn = rightDeviceValuesIn + rightIntervals[rightDeviceId];
                CountT rightCount           = rightDeviceItemCount - rightIntervals[rightDeviceId];

                OffsetT outputOffset = offsets[leftDeviceId] + leftIntervals[rightDeviceId] +
                                       rightIntervals[rightDeviceId];

                if (leftCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        leftKeysIn, leftCount * sizeof(KeyT), rightDeviceId, rightStream));
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        leftValuesIn, leftCount * sizeof(ValueT), rightDeviceId, rightStream));
                }
                if (rightCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        rightKeysIn, rightCount * sizeof(KeyT), rightDeviceId, rightStream));
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
                        rightValuesIn, rightCount * sizeof(ValueT), rightDeviceId, rightStream));
                }
                if (auto outputCount = leftCount + rightCount) {
                    C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(keysOut + outputOffset,
                                                                         outputCount * sizeof(KeyT),
                                                                         rightDeviceId,
                                                                         rightStream));
                    C10_CUDA_CHECK(
                        nanovdb::util::cuda::memPrefetchAsync(valuesOut + outputOffset,
                                                              outputCount * sizeof(ValueT),
                                                              rightDeviceId,
                                                              rightStream));
                }

                CUB_WRAPPER(cub::DeviceMerge::MergePairs,
                            leftKeysIn,
                            leftValuesIn,
                            leftCount,
                            rightKeysIn,
                            rightValuesIn,
                            rightCount,
                            keysOut + outputOffset,
                            valuesOut + outputOffset,
                            {},
                            rightStream);
                C10_CUDA_CHECK(cudaEventRecord(events[rightDeviceId], rightStream));
            };
        }

        for (int leftDeviceId = 0; leftDeviceId < deviceCount; leftDeviceId += 2 * deviceInc) {
            const int rightDeviceId = leftDeviceId + deviceInc;
            C10_CUDA_CHECK(cudaEventSynchronize(events[leftDeviceId]));
            C10_CUDA_CHECK(cudaEventSynchronize(events[rightDeviceId]));
        }
    }

    // There is no merging required for a single device so we simply copy the sorted result to the
    // destination array (where the sort would have been merged to).
    if (log2DeviceCount % 2) {
        std::swap(keysIn, keysOut);
        std::swap(valuesIn, valuesOut);
        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

            cudaMemcpyAsync(keysOut + offsets[deviceId],
                            keysIn + offsets[deviceId],
                            counts[deviceId] * sizeof(KeyT),
                            cudaMemcpyDefault,
                            stream);
            cudaMemcpyAsync(valuesOut + offsets[deviceId],
                            valuesIn + offsets[deviceId],
                            counts[deviceId] * sizeof(ValueT),
                            cudaMemcpyDefault,
                            stream);

            cudaEventRecord(events[deviceId], stream);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
gaussianTileIntersectionPrivateUse1Impl(
    const torch::Tensor &means2d,                   // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                     // [C, N] or [M]
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
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have 1 dimension (M)");
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
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(radii.size(1) == means2d.size(1),
                          "radii must have the same number of points as means2d");
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
    const uint32_t numCamIdBits  = (uint32_t)floor(log2(numCameras)) + 1;
    const auto cameraJIdxPtr =
        cameraJIdx.has_value() ? cameraJIdx.value().data_ptr<int32_t>() : nullptr;

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

    const auto tileMaskPtr = tileMask.has_value() ? tileMask.value().data_ptr<bool>() : nullptr;

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        int64_t deviceGaussianOffset, deviceGaussianCount;
        std::tie(deviceGaussianOffset, deviceGaussianCount) = deviceChunk(totalGaussians, deviceId);

        // Count the number of tiles each Gaussian intersects, store in tilesPerGaussianCumsum
        const int NUM_BLOCKS = (deviceGaussianCount + NUM_THREADS - 1) / NUM_THREADS;
        countTilesPerGaussian<scalar_t, int32_t>
            <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(deviceGaussianOffset,
                                                     deviceGaussianCount,
                                                     numGaussians,
                                                     tileSize,
                                                     numTilesW,
                                                     numTilesH,
                                                     means2d.data_ptr<scalar_t>(),
                                                     radii.data_ptr<int32_t>(),
                                                     tileMaskPtr,
                                                     cameraJIdxPtr,
                                                     tilesPerGaussianCumsum.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    mergeStreams();

    // In place cumulative sum to get the total number of intersections
    torch::cumsum_out(tilesPerGaussianCumsum, tilesPerGaussianCumsum, 0, torch::kInt32);

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

        // Allocate tensors to store the sorted intersections
        torch::Tensor keysSorted = torch::empty_like(intersectionKeys);
        torch::Tensor valsSorted = torch::empty_like(intersectionValues);

        std::vector<cudaEvent_t> events(c10::cuda::device_count());

        // Compute a joffsets tensor that stores the offsets into the sorted Gaussian
        // intersections
        torch::Tensor tileJOffsets = torch::empty({numCameras, numTilesH, numTilesW},
                                                  means2d.options().dtype(torch::kInt32));

        // Compute the set of intersections between each projected Gaussian and each tile,
        // store them in intersectionKeys and intersectionValues
        // where intersectionKeys encodes (camera_id, tile_id, depth) and intersectionValues
        // encodes the index of the Gaussian in the input arrays.

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            cudaEventCreate(&events[deviceId], cudaEventDisableTiming);

            int64_t deviceGaussianOffset, deviceGaussianCount;
            std::tie(deviceGaussianOffset, deviceGaussianCount) =
                deviceChunk(totalGaussians, deviceId);

            const int NUM_BLOCKS = (deviceGaussianCount + NUM_THREADS - 1) / NUM_THREADS;
            computeGaussianTileIntersections<scalar_t>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(numCameras,
                                                         numGaussians,
                                                         deviceGaussianOffset,
                                                         deviceGaussianCount,
                                                         tileSize,
                                                         numTilesW,
                                                         numTilesH,
                                                         numTileIdBits,
                                                         means2d.data_ptr<scalar_t>(),
                                                         radii.data_ptr<int32_t>(),
                                                         depths.data_ptr<scalar_t>(),
                                                         tilesPerGaussianCumsum.data_ptr<int32_t>(),
                                                         tileMaskPtr,
                                                         cameraJIdxPtr,
                                                         intersectionKeys.data_ptr<int64_t>(),
                                                         intersectionValues.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
        }

        {
            C10_CUDA_CHECK(cudaSetDevice(0));
            auto stream = c10::cuda::getCurrentCUDAStream(0);
            for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
                C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
            }
            C10_CUDA_CHECK(cudaEventRecord(events[0], stream));
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[0]));
        }

        const int32_t numBits = 32 + numCamIdBits + numTileIdBits;
        radixSortAsync(intersectionKeys.data_ptr<int64_t>(),
                       keysSorted.data_ptr<int64_t>(),
                       intersectionValues.data_ptr<int32_t>(),
                       valsSorted.data_ptr<int32_t>(),
                       totalIntersections,
                       0,
                       numBits,
                       events.data());

        {
            C10_CUDA_CHECK(cudaSetDevice(0));
            auto stream = c10::cuda::getCurrentCUDAStream(0);
            for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
                C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
            }
            C10_CUDA_CHECK(cudaEventRecord(events[0], stream));
        }

        TORCH_CHECK(!isSparse, "Sparse tile offsets are not implemented for mGPU");

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[0]));

            int64_t deviceIntersectionOffset, deviceIntersectionCount;
            std::tie(deviceIntersectionOffset, deviceIntersectionCount) =
                deviceChunk(totalIntersections, deviceId);

            const int NUM_BLOCKS_2 = (deviceIntersectionCount + NUM_THREADS - 1) / NUM_THREADS;
            computeTileOffsets<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                deviceIntersectionOffset,
                deviceIntersectionCount,
                totalIntersections,
                numCameras,
                totalTiles,
                numTileIdBits,
                keysSorted.data_ptr<int64_t>(),
                tileJOffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            cudaEventDestroy(events[deviceId]);
        }

        mergeStreams();

        return std::make_tuple(tileJOffsets, valsSorted);
    }
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection<torch::kCUDA>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N] or [M]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return gaussianTileIntersectionCUDAImpl(means2d,
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
dispatchGaussianTileIntersection<torch::kPrivateUse1>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N] or [M]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return gaussianTileIntersectionPrivateUse1Impl(means2d,
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
dispatchGaussianTileIntersection<torch::kCPU>(
    const torch::Tensor &means2d,                 // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                   // [C, N] or [nnz]
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
dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N] or [M]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const torch::Tensor &tileMask,                 // [C, H, W]
    const torch::Tensor &activeTiles,              // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return gaussianTileIntersectionCUDAImpl(means2d,
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
dispatchGaussianTileIntersectionSparse<torch::kCPU>(
    const torch::Tensor &means2d,                  // [C, N, 2] or [M, 2]
    const torch::Tensor &radii,                    // [C, N] or [M]
    const torch::Tensor &depths,                   // [C, N] or [M]
    const torch::Tensor &tileMask,                 // [C, H, W]
    const torch::Tensor &activeTiles,              // [num_active_tiles]
    const at::optional<torch::Tensor> &cameraJIdx, // NULL or [M]
    const uint32_t numCameras,
    const uint32_t tileSize,
    const uint32_t numTilesH,
    const uint32_t numTilesW) {
    FVDB_FUNC_RANGE();
    return gaussianTileIntersectionCUDAImpl(means2d,
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

} // namespace ops
} // namespace detail
} // namespace fvdb
