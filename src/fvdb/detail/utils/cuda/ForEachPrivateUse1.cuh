// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH
#define FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH

#include <fvdb/Config.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAStream.h>

namespace fvdb {

namespace _private {

template <int NumThreads, typename Func, typename... Args>
__global__ void __launch_bounds__(NumThreads)
forEachLeafPrivateUse1Kernel(int64_t leafChannelCount,
                             int64_t leafChannelOffset,
                             fvdb::GridBatchData::Accessor grid,
                             const int32_t channelsPerLeaf,
                             Func func,
                             Args... args) {
    for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < leafChannelCount;
         idx += blockDim.x * gridDim.x) {
        const int64_t leafChannelIdx = idx + leafChannelOffset;
        const int64_t cumLeafIdx     = leafChannelIdx / channelsPerLeaf;
        const int32_t channelIdx     = leafChannelIdx % channelsPerLeaf;

        const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

        func(batchIdx, leafIdx, channelIdx, grid, args...);
    }
}

template <int NumThreads, typename Func, typename... Args>
__global__ void __launch_bounds__(NumThreads)
forEachVoxelPrivateUse1Kernel(int64_t leafVoxelChannelCount,
                              int64_t leafVoxelChannelOffset,
                              fvdb::GridBatchData::Accessor grid,
                              int64_t channelsPerVoxel,
                              Func func,
                              Args... args) {
    constexpr auto VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < leafVoxelChannelCount;
         idx += blockDim.x * gridDim.x) {
        const int64_t leafVoxelChannelIdx   = idx + leafVoxelChannelOffset;
        const int64_t voxelsChannelsPerLeaf = VOXELS_PER_LEAF * channelsPerVoxel;

        const int64_t cumLeafIdx = leafVoxelChannelIdx / voxelsChannelsPerLeaf;
        const int64_t leafVoxelIdx =
            (leafVoxelChannelIdx - cumLeafIdx * voxelsChannelsPerLeaf) / channelsPerVoxel;
        const int64_t channelIdx = leafVoxelChannelIdx - cumLeafIdx * voxelsChannelsPerLeaf -
                                   leafVoxelIdx * channelsPerVoxel;

        const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

        func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
    }
}

template <typename ScalarT, int32_t NDIMS, int NumThreads, typename Func, typename... Args>
__global__ void __launch_bounds__(NumThreads)
forEachJaggedElementChannelPrivateUse1Kernel(int64_t numel,
                                             int64_t offset,
                                             JaggedRAcc64<ScalarT, NDIMS> jaggedAcc,
                                             int64_t channelsPerElement,
                                             Func func,
                                             Args... args) {
    for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < numel * channelsPerElement;
         idx += blockDim.x * gridDim.x) {
        const int64_t elementIdx      = (idx + offset) / channelsPerElement;
        const fvdb::JIdxType batchIdx = jaggedAcc.batchIdx(elementIdx);
        const int64_t channelIdx      = (idx + offset) % channelsPerElement;

        func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
    }
}

template <typename ScalarT, int32_t NDIMS, int NumThreads, typename Func, typename... Args>
__global__ void __launch_bounds__(NumThreads)
forEachTensorElementChannelPrivateUse1Kernel(int64_t numel,
                                             int64_t offset,
                                             TorchRAcc64<ScalarT, NDIMS> tensorAcc,
                                             int64_t channelsPerElement,
                                             Func func,
                                             Args... args) {
    for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < numel * channelsPerElement;
         idx += blockDim.x * gridDim.x) {
        const int64_t elementIdx = (idx + offset) / channelsPerElement;
        const int64_t channelIdx = (idx + offset) % channelsPerElement;

        func(elementIdx, channelIdx, tensorAcc, args...);
    }
}

} // namespace _private

template <int NumThreads = DEFAULT_BLOCK_DIM, typename Func, typename... Args>
void
forEachLeafPrivateUse1(int64_t numChannels,
                       const fvdb::GridBatchData &batchHdl,
                       Func func,
                       Args... args) {
    TORCH_CHECK(batchHdl.device().is_privateuseone(), "Grid batch must be on a PrivateUse1 device");

    const int64_t leafCount = batchHdl.totalLeaves();

    auto batchAccessor = batchHdl.deviceAccessor();

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        size_t deviceLeafOffset, deviceLeafCount;
        std::tie(deviceLeafOffset, deviceLeafCount) =
            fvdb::detail::deviceChunk(leafCount, deviceId);

        const auto deviceLeafChannelCount  = deviceLeafCount * numChannels;
        const auto deviceLeafChannelOffset = deviceLeafOffset * numChannels;

        const int64_t deviceNumBlocks = GET_BLOCKS(deviceLeafChannelCount, NumThreads);
        TORCH_INTERNAL_ASSERT(deviceNumBlocks <
                                  static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
                              "Too many blocks in forEachLeafPrivateUse1");
        if (deviceNumBlocks > 0) {
            _private::forEachLeafPrivateUse1Kernel<NumThreads, Func, Args...>
                <<<deviceNumBlocks, NumThreads, 0, stream>>>(deviceLeafChannelCount,
                                                             deviceLeafChannelOffset,
                                                             batchAccessor,
                                                             numChannels,
                                                             func,
                                                             args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    fvdb::detail::mergeStreams();
}

template <int NumThreads = DEFAULT_BLOCK_DIM, typename Func, typename... Args>
void
forEachVoxelPrivateUse1(int64_t numChannels,
                        const fvdb::GridBatchData &batchHdl,
                        Func func,
                        Args... args) {
    TORCH_CHECK(batchHdl.device().is_privateuseone(), "Grid batch must be on a PrivateUse1 device");
    TORCH_CHECK(!fvdb::Config::global().ultraSparseAccelerationEnabled());

    const int64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
    const int64_t leafCount       = batchHdl.totalLeaves();

    auto batchAccessor = batchHdl.deviceAccessor();

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        size_t deviceLeafOffset, deviceLeafCount;
        std::tie(deviceLeafOffset, deviceLeafCount) =
            fvdb::detail::deviceChunk(leafCount, deviceId);

        const auto deviceLeafVoxelChannelCount  = deviceLeafCount * VOXELS_PER_LEAF * numChannels;
        const auto deviceLeafVoxelChannelOffset = deviceLeafOffset * VOXELS_PER_LEAF * numChannels;

        const int64_t deviceNumBlocks = GET_BLOCKS(deviceLeafVoxelChannelCount, NumThreads);
        TORCH_INTERNAL_ASSERT(deviceNumBlocks <
                                  static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
                              "Too many blocks in forEachVoxelPrivateUse1");

        if (deviceNumBlocks > 0) {
            _private::forEachVoxelPrivateUse1Kernel<NumThreads, Func, Args...>
                <<<deviceNumBlocks, NumThreads, 0, stream>>>(deviceLeafVoxelChannelCount,
                                                             deviceLeafVoxelChannelOffset,
                                                             batchAccessor,
                                                             numChannels,
                                                             func,
                                                             args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    fvdb::detail::mergeStreams();
}

template <typename ScalarT,
          int32_t NDIMS,
          int NumThreads = DEFAULT_BLOCK_DIM,
          typename Func,
          typename... Args>
void
forEachJaggedElementChannelPrivateUse1(int64_t numChannels,
                                       const JaggedTensor &jaggedTensor,
                                       Func func,
                                       Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_privateuseone(),
                "JaggedTensor must be on a PrivateUse1 device");

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        size_t deviceElementOffset, deviceElementCount;
        std::tie(deviceElementOffset, deviceElementCount) =
            fvdb::detail::deviceChunk(jaggedTensor.element_count(), deviceId);

        const int64_t deviceNumBlocks = GET_BLOCKS(deviceElementCount, NumThreads);
        if (deviceNumBlocks > 0) {
            _private::forEachJaggedElementChannelPrivateUse1Kernel<ScalarT,
                                                                   NDIMS,
                                                                   NumThreads,
                                                                   Func,
                                                                   Args...>
                <<<deviceNumBlocks, NumThreads, 0, stream>>>(
                    deviceElementCount,
                    deviceElementOffset,
                    jaggedTensor.packed_accessor64<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                    numChannels,
                    func,
                    args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    fvdb::detail::mergeStreams();
}

template <typename ScalarT,
          int32_t NDIMS,
          int NumThreads = DEFAULT_BLOCK_DIM,
          typename Func,
          typename... Args>
void
forEachTensorElementChannelPrivateUse1(int64_t numChannels,
                                       const torch::Tensor &tensor,
                                       Func func,
                                       Args... args) {
    TORCH_CHECK(tensor.device().is_privateuseone(), "Tensor must be on a PrivateUse1 device");

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        size_t deviceElementOffset, deviceElementCount;
        std::tie(deviceElementOffset, deviceElementCount) =
            fvdb::detail::deviceChunk(tensor.size(0), deviceId);

        const int64_t deviceNumBlocks = GET_BLOCKS(deviceElementCount, NumThreads);
        if (deviceNumBlocks > 0) {
            _private::forEachTensorElementChannelPrivateUse1Kernel<ScalarT,
                                                                   NDIMS,
                                                                   NumThreads,
                                                                   Func,
                                                                   Args...>
                <<<deviceNumBlocks, NumThreads, 0, stream>>>(
                    deviceElementCount,
                    deviceElementOffset,
                    tensor.packed_accessor64<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                    numChannels,
                    func,
                    args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    fvdb::detail::mergeStreams();
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH
