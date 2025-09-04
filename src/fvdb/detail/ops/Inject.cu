// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/ops/Inject.h>
#include <fvdb/detail/utils/nanovdb/ActiveVoxelIterator.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <ATen/TensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

namespace fvdb::detail::ops {

/// @brief Given a pointer offset into a contiguous nDim-dimensional tensor whose strides are
/// contigStrides, compute the corresponding
///       pointer offset into a tensor of the same shape but with strides `strides`.
/// @param nDim The number of dimensions of the tensor.
/// @param contigPointerOffset The pointer offset into the contiguous data for the contiguous
///     tensor.
/// @param strides The strides of the tensor to return a pointer offset for.
/// @return The pointer offset into the tensor with the given strides.
__device__ int64_t
contigPtrOffsetToStrided(int64_t contigPointerOffset,
                         int64_t nDim,
                         const int64_t *contigStrides,
                         const int64_t *strides) {
    int64_t offset = 0;
    for (int64_t i = 0; i < nDim; ++i) {
        const int64_t idxI = contigPointerOffset / contigStrides[i];

        offset += idxI * strides[i];
        contigPointerOffset %= contigStrides[i];
    }
    return offset;
}

/// @brief Extract the strides of a tensor and the strides of that tensor if it were contiguous (row
///     major) and return them as tensors on the same device as the input.
/// @param tensor The tensor to extract the strides from.
/// @return A tuple containing the strides of the tensor and the dense strides of the tensor.
///         The dense strides are the strides of the tensor if it were contiguous (row major).
///         Both tensors are on the same device as the input tensor.
std::tuple<torch::Tensor, torch::Tensor>
stridesAndContiguousStrides(const torch::Tensor &tensor) {
    auto const options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    auto const tensorSizes   = tensor.sizes();
    auto retContigStrides    = torch::empty({int64_t(tensorSizes.size())}, options);
    auto retContigStridesAcc = retContigStrides.accessor<int64_t, 1>();

    retContigStridesAcc[tensorSizes.size() - 1] = 1;
    for (size_t i = 1; i < tensorSizes.size(); ++i) {
        retContigStridesAcc[tensorSizes.size() - 1 - i] =
            retContigStridesAcc[tensorSizes.size() - i] * tensorSizes[tensorSizes.size() - i];
    }

    auto const tensorStrides = tensor.strides();
    auto retStrides          = torch::empty({int64_t(tensorStrides.size())}, options);
    auto retStridesAcc       = retStrides.accessor<int64_t, 1>();

    for (size_t i = 0; i < tensorStrides.size(); ++i) {
        retStridesAcc[i] = tensorStrides[i];
    }
    return {retContigStrides.to(tensor.device()), retStrides.to(tensor.device())};
}

/// @brief Fucntor to perform injection over an N-dimensional sidecar of a grid.
/// @tparam ValueType The type of the values in the sidecar.
/// @tparam Offset An offseet to apply to all indices in the index grid (default is -1 to ignore the
/// background value)
/// @param srcGrid The source grid to inject from.
/// @param dstGrid The destination grid to inject into.
/// @param numFeaturesPerVoxel The number of features per voxel in the sidecar.
/// @param nDim The number of dimensions of the sidecar tensors
/// @param srcContigStrides The contiguous strides of the sidecar tensor for the source grid as if
/// it were contiguous (row major).
/// @param srcStrides The actual strides of the sidecar tensor for the source grid.
/// @param srcData The data of the source sidecar tensor.
/// @param dstContigStrides The contiguous strides of the sidecar tensor for the destination grid as
/// if it were contiguous (row major).
/// @param dstStrides The actual strides of the sidecar tensor for the destination grid.
/// @param dstData The data of the destination sidecar tensor.
template <typename ValueType, int64_t Offset = -1> struct InjectGridPytorchFunctor {
    // Copies the sidecar data of a (source) grid into the sidecar of an overlapping (destination)
    // grid Intended to be called via nanovdb::util::cuda::operatorKernel blockDim.x is presumed to
    // be the leaf count of the source tree Values of the destination sidecar that do not overlap
    // with the source are left unchanged NOTE: If the source voxels are not a subset of the
    // destination voxels, the injection will be from the intersection of the two active voxel sets
    // into the destination This version presumes a runtime dimension parameter for input features
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void
    operator()(unsigned int srcLeafOffset,
               const nanovdb::OnIndexGrid *srcGrid,
               const nanovdb::OnIndexGrid *dstGrid,
               const std::size_t numFeaturesPerVoxel,
               const int64_t nDim,
               const int64_t *srcContigStrides,
               const int64_t *srcStrides,
               const ValueType *srcData,
               const int64_t *dstContigStrides,
               const int64_t *dstStrides,
               ValueType *dstData) {
        extern __shared__ uint8_t sharedMemory[];
        int64_t *sharedSrcContigStrides = reinterpret_cast<int64_t *>(sharedMemory);
        int64_t *sharedSrcStrides       = sharedSrcContigStrides + nDim;
        int64_t *sharedDstContigStrides = sharedSrcStrides + nDim;
        int64_t *sharedDstStrides       = sharedDstContigStrides + nDim;

        // Copy the source and destination strides into shared memory
        if (threadIdx.x < nDim) {
            sharedSrcContigStrides[threadIdx.x] = srcContigStrides[threadIdx.x];
        } else if (threadIdx.x < 2 * nDim) {
            sharedSrcStrides[threadIdx.x - nDim] = srcStrides[threadIdx.x - nDim];
        } else if (threadIdx.x < 3 * nDim) {
            sharedDstContigStrides[threadIdx.x - 2 * nDim] =
                dstContigStrides[threadIdx.x - 2 * nDim];
        } else if (threadIdx.x < 4 * nDim) {
            sharedDstStrides[threadIdx.x - 3 * nDim] = dstStrides[threadIdx.x - 3 * nDim];
        }
        __syncthreads();

        const auto srcLeafID      = blockIdx.x + srcLeafOffset;
        const auto warpID         = threadIdx.x >> 5;
        const auto threadInWarpID = threadIdx.x & 0x1f;

        const auto &srcTree = srcGrid->tree();
        const auto &dstTree = dstGrid->tree();
        const auto &srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
        auto dstLeafPtr     = dstTree.root().probeLeaf(srcLeaf.origin());
        if (dstLeafPtr) {
            const auto srcWord = srcLeaf.valueMask().words()[warpID];
            const auto dstWord = dstLeafPtr->valueMask().words()[warpID];

            auto srcOffset = srcLeaf.firstOffset();
            if (warpID) {
                srcOffset += (srcLeaf.mPrefixSum >> ((warpID - 1) * 9)) & 0x1ff;
            }
            auto dstOffset = dstLeafPtr->firstOffset();
            if (warpID) {
                dstOffset += (dstLeafPtr->mPrefixSum >> ((warpID - 1) * 9)) & 0x1ff;
            }

            const uint64_t loMask   = 1UL << threadInWarpID;
            const uint64_t hiMask   = 0x100000000UL << threadInWarpID;
            const uint64_t loSrcCnt = nanovdb::util::countOn(srcWord & (loMask - 1UL));
            const uint64_t hiSrcCnt = nanovdb::util::countOn(srcWord & (hiMask - 1UL));
            const uint64_t loDstCnt = nanovdb::util::countOn(dstWord & (loMask - 1UL));
            const uint64_t hiDstCnt = nanovdb::util::countOn(dstWord & (hiMask - 1UL));

            if (loMask & srcWord & dstWord) {
                for (int64_t w = 0; w < numFeaturesPerVoxel; w++) {
                    const int64_t srcContigIndex =
                        numFeaturesPerVoxel * (Offset + int64_t(srcOffset + loSrcCnt)) + w;
                    const int64_t dstContigIndex =
                        numFeaturesPerVoxel * (Offset + int64_t(dstOffset + loDstCnt)) + w;
                    const int64_t srcIndex = contigPtrOffsetToStrided(
                        srcContigIndex, nDim, sharedSrcContigStrides, sharedSrcStrides);
                    const int64_t dstIndex = contigPtrOffsetToStrided(
                        dstContigIndex, nDim, sharedDstContigStrides, sharedDstStrides);
                    dstData[dstIndex] = srcData[srcIndex];
                }
            }
            if (hiMask & srcWord & dstWord) {
                for (int64_t w = 0; w < numFeaturesPerVoxel; w++) {
                    const int64_t srcContigIndex =
                        numFeaturesPerVoxel * (Offset + int64_t(srcOffset + hiSrcCnt)) + w;
                    const int64_t dstContigIndex =
                        numFeaturesPerVoxel * (Offset + int64_t(dstOffset + hiDstCnt)) + w;
                    const int64_t srcIndex = contigPtrOffsetToStrided(
                        srcContigIndex, nDim, sharedSrcContigStrides, sharedSrcStrides);
                    const int64_t dstIndex = contigPtrOffsetToStrided(
                        dstContigIndex, nDim, sharedDstContigStrides, sharedDstStrides);
                    dstData[dstIndex] = srcData[srcIndex];
                }
            }
        }
    }
};

template <>
void
dispatchInject<torch::kCUDA>(const GridBatchImpl &dstGridBatch,
                             const GridBatchImpl &srcGridBatch,
                             JaggedTensor &dst,
                             const JaggedTensor &src) {
    c10::cuda::CUDAGuard deviceGuard(dstGridBatch.device());

    TORCH_CHECK_VALUE(dst.rdim() == src.rdim(),
                      "Source/Destination tensors should have matching dimensions");
    TORCH_CHECK_VALUE(dst.scalar_type() == src.scalar_type(),
                      "Source/Destination tensors should have matching scalar types");

    for (auto i = 1; i < dst.rdim(); i++) {
        TORCH_CHECK_VALUE(dst.rsize(i) == src.rsize(i),
                          "Source/Destination tensors should have matching feature dimensions");
        TORCH_CHECK_VALUE(dst.jdata().stride(i) != 0,
                          "Destination tensor cannot have zero strides");
    }

    int64_t featureDim = 1;
    for (auto j = 1; j < dst.rdim(); j++) {
        featureDim *= dst.rsize(j);
    }

    // Create a grid for each batch item and store the handles
    for (int i = 0; i < dstGridBatch.batchSize(); i += 1) {
        const nanovdb::OnIndexGrid *dstGrid =
            dstGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        const nanovdb::OnIndexGrid *srcGrid =
            srcGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(dstGrid, "Destination grid is null");
        TORCH_CHECK(srcGrid, "Source grid is null");

        torch::Tensor dstI       = dst.index(i).jdata();
        const torch::Tensor srcI = src.index(i).jdata();

        const auto srcLeafCount = srcGridBatch.numLeavesAt(i);
        const at::cuda::CUDAStream stream =
            at::cuda::getCurrentCUDAStream(srcGridBatch.device().index());

        const auto [srcContigStrides, srcStrides] = stridesAndContiguousStrides(srcI);
        const auto [dstContigStrides, dstStrides] = stridesAndContiguousStrides(dstI);

        AT_DISPATCH_V2(
            src.scalar_type(),
            "Inject",
            AT_WRAP([&] {
                constexpr size_t sharedMemSize = 64 * 4 * sizeof(int64_t); // Maximum 64 dimensions
                using Op                       = InjectGridPytorchFunctor<scalar_t>;
                static_assert(sharedMemSize <= Op::MaxThreadsPerBlock * sizeof(int64_t),
                              "Shared memory size exceeds maximum threads per block");
                nanovdb::util::cuda::operatorKernel<Op>
                    <<<srcLeafCount, Op::MaxThreadsPerBlock, sharedMemSize, stream.stream()>>>(
                        0u,
                        srcGrid,
                        dstGrid,
                        featureDim,
                        srcI.dim(),
                        srcContigStrides.const_data_ptr<int64_t>(),
                        srcStrides.const_data_ptr<int64_t>(),
                        srcI.const_data_ptr<scalar_t>(),
                        dstContigStrides.const_data_ptr<int64_t>(),
                        dstStrides.const_data_ptr<int64_t>(),
                        dstI.data_ptr<scalar_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }),
            AT_EXPAND(AT_ALL_TYPES),
            torch::kFloat16);
    }
}

template <>
void
dispatchInject<torch::kPrivateUse1>(const GridBatchImpl &dstGridBatch,
                                    const GridBatchImpl &srcGridBatch,
                                    JaggedTensor &dst,
                                    const JaggedTensor &src) {
    TORCH_CHECK_VALUE(dst.rdim() == src.rdim(),
                      "Source/Destination tensors should have matching dimensions");
    TORCH_CHECK_VALUE(dst.scalar_type() == src.scalar_type(),
                      "Source/Destination tensors should have matching scalar types");

    for (auto i = 1; i < dst.rdim(); i++) {
        TORCH_CHECK_VALUE(dst.rsize(i) == src.rsize(i),
                          "Source/Destination tensors should have matching feature dimensions");
        TORCH_CHECK_VALUE(dst.jdata().stride(i) != 0,
                          "Destination tensor cannot have zero strides");
    }

    int64_t featureDim = 1;
    for (auto j = 1; j < dst.rdim(); j++) {
        featureDim *= dst.rsize(j);
    }

    // Create a grid for each batch item and store the handles
    for (int i = 0; i < dstGridBatch.batchSize(); i += 1) {
        const nanovdb::OnIndexGrid *dstGrid =
            dstGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        const nanovdb::OnIndexGrid *srcGrid =
            srcGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(dstGrid, "Destination grid is null");
        TORCH_CHECK(srcGrid, "Source grid is null");

        torch::Tensor dstI       = dst.index(i).jdata();
        const torch::Tensor srcI = src.index(i).jdata();

        const auto srcLeafCount                   = srcGridBatch.numLeavesAt(i);
        const auto [srcContigStrides, srcStrides] = stridesAndContiguousStrides(srcI);
        const auto [dstContigStrides, dstStrides] = stridesAndContiguousStrides(dstI);

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

            auto deviceSrcLeafCount =
                (srcLeafCount + c10::cuda::device_count() - 1) / c10::cuda::device_count();
            const auto deviceSrcLeafOffset = deviceSrcLeafCount * deviceId;
            deviceSrcLeafCount = std::min(deviceSrcLeafCount, srcLeafCount - deviceSrcLeafOffset);

            AT_DISPATCH_V2(
                src.scalar_type(),
                "Inject",
                AT_WRAP([&] {
                    constexpr size_t sharedMemSize =
                        64 * 4 * sizeof(int64_t); // Maximum 64 dimensions
                    using Op = InjectGridPytorchFunctor<scalar_t>;
                    static_assert(sharedMemSize <= Op::MaxThreadsPerBlock * sizeof(int64_t),
                                  "Shared memory size exceeds maximum threads per block");
                    nanovdb::util::cuda::operatorKernel<Op>
                        <<<deviceSrcLeafCount, Op::MaxThreadsPerBlock, sharedMemSize, stream>>>(
                            deviceSrcLeafOffset,
                            srcGrid,
                            dstGrid,
                            featureDim,
                            srcI.dim(),
                            srcContigStrides.const_data_ptr<int64_t>(),
                            srcStrides.const_data_ptr<int64_t>(),
                            srcI.const_data_ptr<scalar_t>(),
                            dstContigStrides.const_data_ptr<int64_t>(),
                            dstStrides.const_data_ptr<int64_t>(),
                            dstI.data_ptr<scalar_t>());
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }),
                AT_EXPAND(AT_ALL_TYPES),
                torch::kFloat16);
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }
}

template <>
void
dispatchInject<torch::kCPU>(const GridBatchImpl &dstGridBatch,
                            const GridBatchImpl &srcGridBatch,
                            JaggedTensor &dst,
                            const JaggedTensor &src) {
    TORCH_CHECK_VALUE(dst.rdim() == src.rdim(),
                      "Source/Destination tensors should have matching dimensions");
    TORCH_CHECK_VALUE(dst.scalar_type() == src.scalar_type(),
                      "Source/Destination tensors should have matching scalar types");

    for (auto i = 1; i < dst.rdim(); i++) {
        TORCH_CHECK_VALUE(dst.rsize(i) == src.rsize(i),
                          "Source/Destination tensors should have matching feature dimensions");
    }

    const auto [srcJData, dstJData] = [&]() {
        if (src.rdim() == 1) {
            return std::make_tuple(src.jdata().unsqueeze(-1), dst.jdata().unsqueeze(-1));
        } else {
            return std::make_tuple(src.jdata().view({src.rsize(0), -1}),
                                   dst.jdata().view({dst.rsize(0), -1}));
        }
    }();

    const int64_t featureDim = srcJData.size(1);

    AT_DISPATCH_V2(src.scalar_type(),
                   "Inject",
                   AT_WRAP([&] {
                       const auto srcJDataAccessor = srcJData.accessor<scalar_t, 2>();
                       auto dstJDataAccessor       = dstJData.accessor<scalar_t, 2>();

                       for (auto i = 0; i < srcGridBatch.batchSize(); i += 1) {
                           const nanovdb::OnIndexGrid *grid =
                               srcGridBatch.nanoGridHandle().grid<nanovdb::ValueOnIndex>(i);
                           const nanovdb::OnIndexGrid *dstGrid =
                               dstGridBatch.nanoGridHandle().grid<nanovdb::ValueOnIndex>(i);
                           auto dstAccessor           = dstGrid->getAccessor();
                           const int64_t baseSrcIndex = srcGridBatch.cumVoxelsAt(i);
                           const int64_t baseDstIndex = dstGridBatch.cumVoxelsAt(i);
                           for (auto it = ActiveVoxelIterator<-1>(grid->tree()); it.isValid();
                                ++it) {
                               const nanovdb::Coord ijk = it->first;
                               const auto srcIndex      = it->second;

                               const int64_t dstIndex = int64_t(dstAccessor.getValue(ijk)) - 1;
                               if (dstIndex < 0) {
                                   continue; // Skip if the voxel is not in the destination grid
                               }

                               for (int c = 0; c < featureDim; ++c) {
                                   dstJDataAccessor[dstIndex + baseDstIndex][c] =
                                       srcJDataAccessor[srcIndex + baseSrcIndex][c];
                               }
                           }
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES),
                   torch::kFloat16);
}

} // namespace fvdb::detail::ops
