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

__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
checkStructureKernel(int64_t count,
                     TorchRAcc64<JOffsetsType, 1> offsets,
                     int64_t numElements,
                     TorchRAcc64<JIdxType, 1> indices,
                     int64_t numTensors,
                     TorchRAcc64<JLIdxType, 2> listIds,
                     int64_t numOuterLists,
                     TorchRAcc64<int64_t, 1> out) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x) {
        const uint64_t failures = checkStructureAt(
            idx, offsets, numElements, indices, numTensors, listIds, numOuterLists);
        if (failures != 0) {
            atomicOr(reinterpret_cast<unsigned long long *>(&out[0]),
                     static_cast<unsigned long long>(failures));
        }
        if (idx == listIds.size(0) - 1) {
            out[1] = listIds[idx][0];
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

    torch::Tensor out =
        torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64).device(offsets.device()));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(offsets.device().index()).stream();
    checkStructureKernel<<<GET_BLOCKS(count, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
        count,
        offsets.packed_accessor64<JOffsetsType, 1, torch::RestrictPtrTraits>(),
        numElements,
        indices.packed_accessor64<JIdxType, 1, torch::RestrictPtrTraits>(),
        numTensors,
        listIds.packed_accessor64<JLIdxType, 2, torch::RestrictPtrTraits>(),
        numOuterLists,
        out.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const torch::Tensor outCpu = out.cpu();
    const auto acc             = outCpu.accessor<int64_t, 1>();
    JaggedStructureCheck ret;
    ret.failures    = static_cast<uint64_t>(acc[0]);
    ret.lastOuterId = listIds.size(0) > 0 ? acc[1] : -1;
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
