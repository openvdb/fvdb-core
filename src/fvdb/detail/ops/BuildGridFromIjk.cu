// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#if CCCL_DEVICE_MERGE_SUPPORTED
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>
#else
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#endif
#include <nanovdb/HostBuffer.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType> GridStorage dispatchCreateNanoGridFromIJK(const JaggedTensor &ijk);

// NanoVDB's PointsToGrid / DistributedPointsToGrid radix sort casts the per-grid coordinate count
// to int32 (PointsToGrid.cuh:645), which silently corrupts the grid above 2^31 candidates. Reject
// that here rather than return a garbage grid. Takes an already-on-host joffsets accessor so each
// dispatch reuses the host copy it makes anyway -- no extra device sync.
static void
checkCandidateCountsFitInt32(const torch::TensorAccessor<fvdb::JOffsetsType, 1> &joffsetsAcc) {
    for (int64_t gi = 0; gi + 1 < joffsetsAcc.size(0); gi += 1) {
        const int64_t nCoords = joffsetsAcc[gi + 1] - joffsetsAcc[gi];
        TORCH_CHECK(nCoords <= std::numeric_limits<int32_t>::max(),
                    "Grid ",
                    gi,
                    " would be built from ",
                    nCoords,
                    " candidate coordinates, which exceeds the ",
                    std::numeric_limits<int32_t>::max(),
                    "-coordinate limit of the NanoVDB grid builder. Reduce the grid size.");
    }
}

template <>
GridStorage
dispatchCreateNanoGridFromIJK<torch::kCUDA>(const JaggedTensor &ijk) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_cuda(), "device must be cuda");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32 || ijk.scalar_type() == torch::kInt64,
                "ijk must be int32 or int64");

    const torch::Device device = ijk.device();
    c10::cuda::CUDAGuard deviceGuard(device);
    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builder makes.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

    // FIXME: This is slow because we have to copy this data to the host and then build the
    // grids. Ideally we want to do this in a single invocation.
    torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
    auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
    checkCandidateCountsFitInt32(ijkBOffset);
    torch::Tensor ijkData = ijk.jdata();
    if (ijkData.scalar_type() != torch::kInt32) {
        ijkData = ijkData.to(torch::kInt32);
    }
    TORCH_CHECK(ijkData.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijkData.dim() == 2, "ijk must have shape (N, 3)");
    TORCH_CHECK(ijkData.size(1) == 3, "ijk must have shape (N, 3)");

    // Build one grid per batch item, then lay them end to end in one storage.
    GridStorageParts parts(device, stream, ijkBOffset.size(0) - 1);
    for (int i = 0; i < (ijkBOffset.size(0) - 1); i += 1) {
        const int64_t startIdx = ijkBOffset[i];
        const int64_t nVoxels  = ijkBOffset[i + 1] - startIdx;
        const int32_t *dataPtr = ijkData.data_ptr<int32_t>() + 3 * startIdx;

        if (nVoxels == 0) {
            parts.addEmpty();
        } else {
            parts.add(GridStorage(
                nanovdb::tools::cuda::
                    voxelsToGrid<GridT, nanovdb::Coord *, DeviceGridBuffer, BuilderResource>(
                        (nanovdb::Coord *)dataPtr, nVoxels, 1.0, proto, stream),
                device));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return parts.merge();
}

template <>
GridStorage
dispatchCreateNanoGridFromIJK<torch::kPrivateUse1>(const JaggedTensor &ijk) {
    using GridT = nanovdb::ValueOnIndex;

#if CCCL_DEVICE_MERGE_SUPPORTED
    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_privateuseone(), "device must be privateuseone");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32, "ijk must be int32");

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

    const torch::Device device = ijk.device();
    // Unified-memory storage is not stream-ordered; storageStream(PrivateUse1) is the legacy
    // default stream, which is what the storage retains and what the assembler orders against.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // FIXME: This is slow because we have to copy this data to the host and then build the
    // grids. Ideally we want to do this in a single invocation.
    torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
    auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
    checkCandidateCountsFitInt32(ijkBOffset);
    torch::Tensor ijkData = ijk.jdata();
    TORCH_CHECK(ijkData.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijkData.dim() == 2, "ijk must have shape (N, 3)");
    TORCH_CHECK(ijkData.size(1) == 3, "ijk must have shape (N, 3)");

    // The coordinates were written by torch on each device's current stream; the mesh streams
    // the builder runs on are not ordered after them.
    for (const auto device_index: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(device_index).synchronize();
    }

    // Build one grid per batch item, then lay them end to end in one storage.
    GridStorageParts parts(device, stream, ijkBOffset.size(0) - 1);
    for (int i = 0; i < (ijkBOffset.size(0) - 1); i += 1) {
        const int64_t startIdx = ijkBOffset[i];
        const int64_t nVoxels  = ijkBOffset[i + 1] - startIdx;

        if (!nVoxels) {
            parts.addEmpty();
        } else {
            int32_t *dataPtr = ijkData.data_ptr<int32_t>() + ijkData.stride(0) * startIdx;
            auto coordPtr    = reinterpret_cast<nanovdb::Coord *>(dataPtr);

            nanovdb::cuda::DeviceMesh mesh;
            nanovdb::tools::cuda::DistributedPointsToGrid<GridT> converter(mesh);
            auto handle =
                converter.getHandle<nanovdb::Coord *, DeviceGridBuffer>(coordPtr, nVoxels, proto);
            // getHandle allocated the storage on, and synchronized, a stream of `mesh`, which is
            // destroyed at the end of this iteration. Retain the storage stream instead so the
            // buffer never names a dead stream.
            handle.buffer().set_stream(stream);
            parts.add(GridStorage(std::move(handle), device));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return parts.merge();
#else
    TORCH_CHECK(false, "Distributed creation of grids requires CUDA 12.8 or later");
    return GridStorage();
#endif
}

template <>
GridStorage
dispatchCreateNanoGridFromIJK<torch::kCPU>(const JaggedTensor &jaggedCoords) {
    using GridT = nanovdb::ValueOnIndex;

    return AT_DISPATCH_V2(
        jaggedCoords.scalar_type(),
        "buildPaddedGridFromCoords",
        AT_WRAP([&]() {
            using ScalarT = scalar_t;
            jaggedCoords.check_valid();

            static_assert(std::is_integral<ScalarT>::value,
                          "Invalid type for coords, must be integral");

            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            const torch::TensorAccessor<ScalarT, 2> &coordsAcc =
                jaggedCoords.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &coordsBOffsetsAcc =
                jaggedCoords.joffsets().accessor<fvdb::JOffsetsType, 1>();
            checkCandidateCountsFitInt32(coordsBOffsetsAcc);

            std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> batchHandles;
            batchHandles.reserve(coordsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (coordsBOffsetsAcc.size(0) - 1); bi += 1) {
                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = coordsBOffsetsAcc[bi];
                const int64_t end   = coordsBOffsetsAcc[bi + 1];

                for (unsigned ci = start; ci < end; ci += 1) {
                    nanovdb::Coord ijk(coordsAcc[ci][0], coordsAcc[ci][1], coordsAcc[ci][2]);
                    proxyGridAccessor.setValue(ijk, 11);
                }

                proxyGridAccessor.merge();
                batchHandles.push_back(
                    nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                        *proxyGrid, 0u, false, false));
            }

            return GridStorage(batchHandles.size() == 1 ? std::move(batchHandles[0])
                                                        : nanovdb::mergeGrids(batchHandles));
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));
}

GridStorage
_createNanoGridFromIJK(const JaggedTensor &ijk) {
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected coords to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK_VALUE(ijk.rdim() == 2,
                      std::string("Expected ijk to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(ijk.rsize(1) == 3,
                      "Expected 3 dimensional coords but got ijk.rshape[1] = " +
                          std::to_string(ijk.rsize(1)));
    TORCH_CHECK(ijk.num_tensors() == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(ijk.num_outer_lists() <= GridBatchData::MAX_GRIDS_PER_BATCH,
                      "Cannot create a batch of grids with more than ",
                      GridBatchData::MAX_GRIDS_PER_BATCH,
                      " grids in it. ",
                      "You passed in ",
                      ijk.num_outer_lists(),
                      " ijk sets.");
    const int64_t numGrids = ijk.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");

    // The >2^31-candidate overflow guard runs inside each dispatch (checkCandidateCountsFitInt32),
    // reusing the host joffsets copy those paths already make -- no extra device sync here.
    return FVDB_DISPATCH_KERNEL(ijk.device(),
                                [&]() { return dispatchCreateNanoGridFromIJK<DeviceTag>(ijk); });
}

c10::intrusive_ptr<GridBatchData>
createNanoGridFromIJK(const JaggedTensor &ijk,
                      const std::vector<nanovdb::Vec3d> &voxelSizes,
                      const std::vector<nanovdb::Vec3d> &origins) {
    return makeGridBatchData(_createNanoGridFromIJK(ijk), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
