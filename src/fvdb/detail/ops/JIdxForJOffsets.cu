// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/JIdxForJOffsets.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

template <int blockSize>
__global__ __launch_bounds__(blockSize) void
jIdxForJOffsets(TorchRAcc64<fvdb::JOffsetsType, 1> offsets,
                TorchRAcc64<fvdb::JIdxType, 1> outJIdx) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= outJIdx.size(0)) {
        return;
    }

    fvdb::JIdxType left = 0, right = offsets.size(0) - 1;

    while (left <= right) {
        fvdb::JIdxType mid = left + (right - left) / 2;

        // the bin-search should consider when there are some value in offsets
        if (offsets[mid] <= idx) {
            // if offsets[mid] <= idx, just means possible.
            // the case [0, 10, 10, 40], when idx=10, mid=1,
            // what we need is 2, so we need to let left = mid+1
            if ((mid < (offsets.size(0) - 1)) && (idx < offsets[mid + 1])) {
                outJIdx[idx] = mid;
                return;
            } else {
                // if idx >= offsets[mid+1], means target may in the right.
                left = mid + 1;
            }
        } else {
            // if idx < offsets[mid], means target is in the left
            right = mid - 1;
        }
    }
    outJIdx[idx] = -1;
    return;
}

torch::Tensor
jIdxForJOffsetsCUDA(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    torch::Tensor retJIdx = torch::empty({numElements}, options);

    const int NUM_BLOCKS = GET_BLOCKS(numElements, DEFAULT_BLOCK_DIM);
    jIdxForJOffsets<DEFAULT_BLOCK_DIM><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM>>>(
        joffsets.packed_accessor64<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        retJIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retJIdx;
}

template <int blockSize>
__global__ __launch_bounds__(blockSize) void
jIdxFill(fvdb::JOffsetsType start,
         fvdb::JOffsetsType end,
         fvdb::JIdxType i,
         TorchRAcc64<fvdb::JIdxType, 1> outJIdx) {
    for (int64_t idx = start + blockIdx.x * blockDim.x + threadIdx.x; idx < end;
         idx += blockDim.x * gridDim.x) {
        outJIdx[idx] = i;
    }
}

torch::Tensor
jIdxForJOffsetsPrivateUse1(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    torch::Tensor retJIdx = torch::empty({numElements}, options);

    auto joffsets_cpu = joffsets.cpu();
    auto joffsets_acc = joffsets_cpu.accessor<fvdb::JOffsetsType, 1>();

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        size_t deviceJOffsetsStart, deviceJOffsetsCount;
        std::tie(deviceJOffsetsStart, deviceJOffsetsCount) =
            deviceChunk(joffsets.size(0) - 1, deviceId);
        size_t deviceJOffsetsEnd = deviceJOffsetsStart + deviceJOffsetsCount;

        for (auto i = deviceJOffsetsStart; i < deviceJOffsetsEnd; ++i) {
            auto start = joffsets_acc[i];
            auto end   = joffsets_acc[i + 1];
            if (start < end) {
                const int numBlocks = GET_BLOCKS(end - start, DEFAULT_BLOCK_DIM);
                jIdxFill<DEFAULT_BLOCK_DIM><<<numBlocks, DEFAULT_BLOCK_DIM, 0, stream>>>(
                    start,
                    end,
                    static_cast<fvdb::JIdxType>(i),
                    retJIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }

    return retJIdx;
}

torch::Tensor
jIdxForJOffsetsCPU(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(joffsets.size(0));
    auto joffsets_acc = joffsets.accessor<fvdb::JOffsetsType, 1>();
    for (int i = 0; i < joffsets.size(0) - 1; i += 1) {
        auto count = joffsets_acc[i + 1] - joffsets_acc[i];
        batchIdxs.push_back(torch::full({count}, i, options));
    }
    return torch::cat(batchIdxs, 0);
}

torch::Tensor
jIdxForJOffsets(torch::Tensor joffsets, int64_t numElements) {
    if (joffsets.device().is_cuda()) {
        return jIdxForJOffsetsCUDA(joffsets, numElements);
    } else if (joffsets.device().is_privateuseone()) {
        return jIdxForJOffsetsPrivateUse1(joffsets, numElements);
    } else {
        return jIdxForJOffsetsCPU(joffsets, numElements);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
