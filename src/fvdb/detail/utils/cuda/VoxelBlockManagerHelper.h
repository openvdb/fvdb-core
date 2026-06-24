// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_VOXELBLOCKMANAGERHELPER_H
#define FVDB_DETAIL_UTILS_CUDA_VOXELBLOCKMANAGERHELPER_H

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// Shared NanoVDB VoxelBlockManager (VBM) scaffolding used by the VBM-based ops (ReinitializeSdf,
// DualContour, ...). Each op still writes its own per-voxel neighbour read -- they differ in which
// neighbours they touch (6 faces vs the full 3x3x3 box stencil) -- but the grid type, buffer type,
// block width, the build-once handle wrapper, and the per-block decode below are identical across
// all of them.

using OnIndexGridT = nanovdb::NanoGrid<nanovdb::ValueOnIndex>;
using VbmBuffer    = nanovdb::cuda::DeviceBuffer;

// log2 of the VoxelBlockManager block width: each VBM block spans 2^9 = 512 active voxels.
static constexpr int kLog2BlockWidth = 9;

// small VBM helper: build once, expose the block count + the firstLeafID/jumpMap device pointers.
struct VBMHelper {
    nanovdb::tools::VoxelBlockManagerHandle<VbmBuffer> handle;
    uint32_t blockCount{0};
    uint64_t firstOffset{0}, valueCount{1};
    VBMHelper(OnIndexGridT *grid, cudaStream_t stream) {
        handle = nanovdb::tools::cuda::buildVoxelBlockManager<kLog2BlockWidth, VbmBuffer>(
            grid, 0, 0, 0, stream);
        blockCount  = (uint32_t)handle.blockCount();
        firstOffset = handle.firstOffset();
        valueCount  = handle.lastOffset() + 1;
    }
    const uint32_t *
    firstLeafID() const {
        return handle.deviceFirstLeafID();
    }
    const uint64_t *
    jumpMap() const {
        return handle.deviceJumpMap();
    }
};

// ------------------------- per-block decode helpers (device) -------------------------
// Per-VBM-block shared decode scratch: the inverse leaf/offset maps for one 2^Log2BlockWidth block.
// Declare one as __shared__ in the kernel and hand it to vbmDecodeBlock.
template <int Log2BlockWidth> struct VbmBlockMaps {
    uint32_t leafIndex[1 << Log2BlockWidth];
    uint16_t voxelOffset[1 << Log2BlockWidth];
};

// Cooperatively decode block blockIdx.x into `maps` (ALL threads in the block must call). Returns
// true if this thread's slot maps to an active voxel; the caller returns on false. The kernel must
// have parameters named grid / firstLeafID / jumpMap / firstOffset (a VBMHelper supplies the latter
// three). After this, maps.leafIndex[threadIdx.x] / maps.voxelOffset[threadIdx.x] locate the voxel.
template <int Log2BlockWidth>
__device__ inline bool
vbmDecodeBlock(const OnIndexGridT *grid,
               const uint32_t *firstLeafID,
               const uint64_t *jumpMap,
               uint64_t firstOffset,
               VbmBlockMaps<Log2BlockWidth> &maps) {
    using VoxelBlockManagerT            = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
    constexpr uint64_t blockWidth       = uint64_t(1) << Log2BlockWidth,
                       jumpMapWordCount = blockWidth / 64;
    VoxelBlockManagerT::template decodeInverseMaps<nanovdb::ValueOnIndex>(
        grid,
        firstLeafID[blockIdx.x],
        &jumpMap[blockIdx.x * jumpMapWordCount],
        firstOffset + blockIdx.x * blockWidth,
        maps.leafIndex,
        maps.voxelOffset);
    return maps.leafIndex[threadIdx.x] != VoxelBlockManagerT::UnusedLeafIndex;
}

// The centre voxel's value-index plus its 6 face-neighbour value-indices (-x,+x,-y,+y,-z,+z;
// 0 = inactive/background).
struct VbmFaceStencil {
    uint64_t centerIndex;
    uint64_t faceIndex[6];
};

// Read this thread's centre + 6-face value indices through a cached accessor. Call only for active
// threads (i.e. after vbmDecodeBlock returned true for this thread).
template <int Log2BlockWidth>
__device__ inline VbmFaceStencil
vbmReadFaceStencil(const OnIndexGridT *grid, const VbmBlockMaps<Log2BlockWidth> &maps) {
    const auto &leaf = grid->tree().template getFirstNode<0>()[maps.leafIndex[threadIdx.x]];
    const nanovdb::Coord centerCoord = leaf.offsetToGlobalCoord(maps.voxelOffset[threadIdx.x]);
    auto accessor                    = grid->getAccessor();
    return {leaf.getValue(maps.voxelOffset[threadIdx.x]),
            {accessor.getValue(centerCoord.offsetBy(-1, 0, 0)),
             accessor.getValue(centerCoord.offsetBy(1, 0, 0)),
             accessor.getValue(centerCoord.offsetBy(0, -1, 0)),
             accessor.getValue(centerCoord.offsetBy(0, 1, 0)),
             accessor.getValue(centerCoord.offsetBy(0, 0, -1)),
             accessor.getValue(centerCoord.offsetBy(0, 0, 1))}};
}

// Full face-stencil kernel preamble: decode this block, and for active threads read the 6-face
// stencil into `out`. Returns false (caller should return) for unused slots. Owns its __shared__
// scratch, so the kernel needs no shared declaration of its own.
template <int Log2BlockWidth>
__device__ inline bool
vbmDecodeFaceStencil(const OnIndexGridT *grid,
                     const uint32_t *firstLeafID,
                     const uint64_t *jumpMap,
                     uint64_t firstOffset,
                     VbmFaceStencil &out) {
    __shared__ VbmBlockMaps<Log2BlockWidth> maps;
    if (!vbmDecodeBlock<Log2BlockWidth>(grid, firstLeafID, jumpMap, firstOffset, maps))
        return false;
    out = vbmReadFaceStencil<Log2BlockWidth>(grid, maps);
    return true;
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_VOXELBLOCKMANAGERHELPER_H
