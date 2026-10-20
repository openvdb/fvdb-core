// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/JaggedStructureCheck.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Checks every invariant that position idx of the offsets, indices, and list ids takes part in.
// Empty accessors are skipped. Shared by the CPU loop and the CUDA kernel.
template <typename OffsetsAcc, typename IndicesAcc, typename ListIdsAcc>
__host__ __device__ uint64_t
checkStructureAt(int64_t idx,
                 const OffsetsAcc &offsets,
                 int64_t numElements,
                 const IndicesAcc &indices,
                 int64_t numTensors,
                 const ListIdsAcc &listIds,
                 int64_t numOuterLists) {
    uint64_t failures = 0;

    const int64_t numOffsets = offsets.size(0);
    if (numOffsets > 0) {
        if (idx == 0) {
            if (offsets[0] != 0) {
                failures |= kOffsetsStart;
            }
            if (offsets[numOffsets - 1] != numElements) {
                failures |= kOffsetsEnd;
            }
        }
        if (idx + 1 < numOffsets && offsets[idx] > offsets[idx + 1]) {
            failures |= kOffsetsDecreasing;
        }
    }

    if (idx < indices.size(0)) {
        const int64_t index = indices[idx];
        if (index < 0) {
            failures |= kIndicesNegative;
        }
        if (index >= numTensors) {
            failures |= kIndicesTooLarge;
        }
        if (idx > 0 && indices[idx - 1] > index) {
            failures |= kIndicesDecreasing;
        }
    }

    if (idx < listIds.size(0)) {
        const int64_t outer = listIds[idx][0];
        if (outer < 0) {
            failures |= kOuterIdsNegative;
        }
        if (numOuterLists >= 0 && outer >= numOuterLists) {
            failures |= kOuterIdsTooLarge;
        }

        // Inner ids restart at 0 for each outer list
        int64_t expectedInner = 0;
        if (idx > 0) {
            const int64_t prevOuter = listIds[idx - 1][0];
            if (prevOuter > outer) {
                failures |= kOuterIdsDecreasing;
            } else if (prevOuter == outer) {
                expectedInner = static_cast<int64_t>(listIds[idx - 1][1]) + 1;
            }
        }
        if (listIds[idx][1] != expectedInner) {
            failures |= kInnerIdsNotCounting;
        }
    }

    return failures;
}

// Inputs up to this size are checked by one block, which writes its result once and so needs no
// zeroed buffer and no cross-block atomics
constexpr int64_t kSingleBlockMaxCount = int64_t(1) << 16;

// Writes out[0] = OR of failure bits and, when there are list id rows, out[1] = last outer id. With
// one block, out may point at pinned host memory and is written with plain stores. With several
// blocks, out must be zeroed device memory.
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
checkStructureKernel(int64_t count,
                     TorchRAcc64<JOffsetsType, 1> offsets,
                     int64_t numElements,
                     TorchRAcc64<JIdxType, 1> indices,
                     int64_t numTensors,
                     TorchRAcc64<JLIdxType, 2> listIds,
                     int64_t numOuterLists,
                     int64_t *out) {
    __shared__ unsigned long long blockFailures;
    if (threadIdx.x == 0) {
        blockFailures = 0;
    }
    __syncthreads();

    uint64_t failures = 0;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x) {
        failures |= checkStructureAt(
            idx, offsets, numElements, indices, numTensors, listIds, numOuterLists);
        if (idx == listIds.size(0) - 1) {
            out[1] = listIds[idx][0];
        }
    }
    if (failures != 0) {
        atomicOr(&blockFailures, static_cast<unsigned long long>(failures));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            out[0] = static_cast<int64_t>(blockFailures);
        } else if (blockFailures != 0) {
            atomicOr(reinterpret_cast<unsigned long long *>(&out[0]), blockFailures);
        }
    }
}

// Undefined arguments become empty tensors so both paths can build accessors for them
torch::Tensor
orEmpty(const torch::Tensor &t, c10::ScalarType dtype, int64_t dim, const torch::Device &device) {
    if (t.defined()) {
        return t;
    }
    return dim == 1 ? torch::empty({0}, torch::TensorOptions().dtype(dtype).device(device))
                    : torch::empty({0, 2}, torch::TensorOptions().dtype(dtype).device(device));
}

JaggedStructureCheck
checkJaggedStructureCPU(const torch::Tensor &offsets,
                        int64_t numElements,
                        const torch::Tensor &indices,
                        int64_t numTensors,
                        const torch::Tensor &listIds,
                        int64_t numOuterLists) {
    const auto offsetsAcc = offsets.accessor<JOffsetsType, 1>();
    const auto indicesAcc = indices.accessor<JIdxType, 1>();
    const auto listIdsAcc = listIds.accessor<JLIdxType, 2>();
    const int64_t count   = std::max({offsets.size(0), indices.size(0), listIds.size(0)});

    JaggedStructureCheck ret;
    for (int64_t idx = 0; idx < count; ++idx) {
        ret.failures |= checkStructureAt(
            idx, offsetsAcc, numElements, indicesAcc, numTensors, listIdsAcc, numOuterLists);
    }
    if (listIds.size(0) > 0) {
        ret.lastOuterId = listIdsAcc[listIds.size(0) - 1][0];
    }
    return ret;
}

// Device pointer for pinned host memory, or nullptr when the device cannot address it directly
int64_t *
devicePointerForHost(int64_t *host) {
    void *devicePtr = nullptr;
    if (cudaHostGetDevicePointer(&devicePtr, host, 0) != cudaSuccess) {
        (void)cudaGetLastError(); // Clear the error so later launch checks don't report it
        return nullptr;
    }
    return static_cast<int64_t *>(devicePtr);
}

JaggedStructureCheck
checkJaggedStructureCUDA(const torch::Tensor &offsets,
                         int64_t numElements,
                         const torch::Tensor &indices,
                         int64_t numTensors,
                         const torch::Tensor &listIds,
                         int64_t numOuterLists) {
    const c10::cuda::CUDAGuard deviceGuard(offsets.device());
    const int64_t count = std::max({offsets.size(0), indices.size(0), listIds.size(0)});
    if (count == 0) {
        return {};
    }

    const torch::Tensor result =
        torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    int64_t *resultHost = result.data_ptr<int64_t>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(offsets.device().index()).stream();

    const auto launch = [&](int numBlocks, int64_t *out) {
        checkStructureKernel<<<numBlocks, DEFAULT_BLOCK_DIM, 0, stream>>>(
            count,
            offsets.packed_accessor64<JOffsetsType, 1, torch::RestrictPtrTraits>(),
            numElements,
            indices.packed_accessor64<JIdxType, 1, torch::RestrictPtrTraits>(),
            numTensors,
            listIds.packed_accessor64<JLIdxType, 2, torch::RestrictPtrTraits>(),
            numOuterLists,
            out);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    // One block writes straight into the pinned result, so the readback is just the stream sync
    int64_t *resultDevice =
        count <= kSingleBlockMaxCount ? devicePointerForHost(resultHost) : nullptr;
    if (resultDevice != nullptr) {
        launch(1, resultDevice);
    } else {
        torch::Tensor scratch =
            torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(offsets.device()));
        C10_CUDA_CHECK(cudaMemsetAsync(scratch.data_ptr(), 0, 2 * sizeof(int64_t), stream));
        launch(GET_BLOCKS(count, DEFAULT_BLOCK_DIM), scratch.data_ptr<int64_t>());
        C10_CUDA_CHECK(cudaMemcpyAsync(resultHost,
                                       scratch.data_ptr<int64_t>(),
                                       2 * sizeof(int64_t),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    }
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    JaggedStructureCheck ret;
    ret.failures    = static_cast<uint64_t>(resultHost[0]);
    ret.lastOuterId = listIds.size(0) > 0 ? resultHost[1] : -1;
    return ret;
}

} // namespace

JaggedStructureCheck
checkJaggedStructure(const torch::Tensor &offsets,
                     int64_t numElements,
                     const torch::Tensor &indices,
                     int64_t numTensors,
                     const torch::Tensor &listIds,
                     int64_t numOuterLists) {
    const torch::Device device = offsets.defined()   ? offsets.device()
                                 : indices.defined() ? indices.device()
                                 : listIds.defined() ? listIds.device()
                                                     : torch::Device(torch::kCPU);
    const torch::Tensor o      = orEmpty(offsets, JOffsetsScalarType, 1, device);
    const torch::Tensor i      = orEmpty(indices, JIdxScalarType, 1, device);
    const torch::Tensor l      = orEmpty(listIds, JLIdxScalarType, 2, device);

    if (device.is_cuda()) {
        return checkJaggedStructureCUDA(o, numElements, i, numTensors, l, numOuterLists);
    }
    return checkJaggedStructureCPU(
        o.cpu(), numElements, i.cpu(), numTensors, l.cpu(), numOuterLists);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
