// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/RAIIRawDeviceBuffer.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>
#include <fvdb/detail/utils/nanovdb/PadGrid.cuh>

#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
GridStorage dispatchBuildGridFromNearestVoxelsToPoints(const JaggedTensor &points,
                                                       const std::vector<VoxelCoordTransform> &txs);

namespace {

// Computes floor(transform(point)) -- the base voxel of the cell containing each point. One
// coordinate per point.
template <typename ScalarT>
__device__ void
flooredIjkForPointCallback(int32_t bidx,
                           int32_t eidx,
                           const JaggedRAcc64<ScalarT, 2> points,
                           const VoxelCoordTransform *transforms,
                           TorchRAcc64<int32_t, 2> outIJKData) {
    using MathT                          = typename at::opmath_type<ScalarT>;
    const auto &point                    = points.data()[eidx];
    const VoxelCoordTransform &transform = transforms[bidx];
    const nanovdb::Coord ijk0            = transform
                                    .apply(static_cast<MathT>(point[0]),
                                           static_cast<MathT>(point[1]),
                                           static_cast<MathT>(point[2]))
                                    .floor();
    outIJKData[eidx][0] = ijk0[0];
    outIJKData[eidx][1] = ijk0[1];
    outIJKData[eidx][2] = ijk0[2];
}

JaggedTensor
flooredIjkForPoints(const JaggedTensor &jaggedPoints,
                    const std::vector<VoxelCoordTransform> &transforms) {
    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(jaggedPoints.device());
    torch::Tensor outIJK = torch::empty({jaggedPoints.jdata().size(0), 3}, optsData);

    AT_DISPATCH_V2(
        jaggedPoints.scalar_type(),
        "flooredIjkForPoints",
        AT_WRAP([&] {
            RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(),
                                                                    jaggedPoints.device());
            transformsDVec.setData((VoxelCoordTransform *)transforms.data(), true /* blocking */);
            const VoxelCoordTransform *transformDevPtr = transformsDVec.devicePtr;

            auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();

            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc64<scalar_t, 2> pacc) {
                flooredIjkForPointCallback(bidx, eidx, pacc, transformDevPtr, outIJKAcc);
            };
            forEachJaggedElementChannelCUDA<scalar_t, 2, 1024>(1, jaggedPoints, cb);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
    return jaggedPoints.jagged_like(outIJK);
}

} // namespace

template <>
GridStorage
dispatchBuildGridFromNearestVoxelsToPoints<torch::kCUDA>(
    const JaggedTensor &points, const std::vector<VoxelCoordTransform> &txs) {
    const torch::Device device = points.device();
    c10::cuda::CUDAGuard deviceGuard(device);
    // The grids are built and their storage retained on the device's current torch stream; the
    // prototype carries that stream and the device into every allocation the builder makes.
    const cudaStream_t stream    = storageStream(device);
    const DeviceGridBuffer proto = GridStorage::deviceProto(device, stream);

    // Build the base grid from the single voxel containing each point. Points are unstructured, so
    // one sort is unavoidable. _createNanoGridFromIJK builds on this same storage stream.
    JaggedTensor flooredIjk = flooredIjkForPoints(points, txs);
    GridStorage base        = ops::_createNanoGridFromIJK(flooredIjk);

    // The 8 nearest voxels of a point are floor(p) + {0,1}^3, so the nearest-voxel grid is the base
    // grid padded by one positive octant (the Minkowski sum distributes over the point union).
    const torch::Tensor joffsetsCpu = points.joffsets().cpu();
    const auto joffsetsAcc          = joffsetsCpu.accessor<fvdb::JOffsetsType, 1>();

    // Pad one grid per batch item, then lay them end to end in one storage.
    GridStorageParts parts(device, stream, base.gridCount());
    for (uint32_t i = 0; i < base.gridCount(); i += 1) {
        if (joffsetsAcc[i + 1] - joffsetsAcc[i] == 0) {
            // No points in this batch item -> empty grid (>=1 point always yields >=1 base voxel).
            parts.addEmpty();
            continue;
        }
        nanovdb::OnIndexGrid *grid = base.deviceGridAt<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid, "Grid is null");
        morphology::PadGrid<nanovdb::ValueOnIndex, BuilderResource> op(
            grid, /*positiveOctant=*/true, stream);
        parts.add(GridStorage(op.getHandle(proto), device));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return parts.merge();
}

template <>
GridStorage
dispatchBuildGridFromNearestVoxelsToPoints<torch::kCPU>(
    const JaggedTensor &jaggedPoints, const std::vector<VoxelCoordTransform> &txs) {
    using GridT = nanovdb::ValueOnIndex;

    return AT_DISPATCH_V2(
        jaggedPoints.scalar_type(),
        "buildNearestNeighborGridFromPoints",
        AT_WRAP([&]() {
            using ScalarT    = scalar_t;
            using MathT      = typename at::opmath_type<ScalarT>;
            using Vec3T      = typename nanovdb::math::Vec3<MathT>;
            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            static_assert(is_floating_point_or_half<ScalarT>::value,
                          "Invalid type for points, must be floating point");

            jaggedPoints.check_valid();

            const torch::TensorAccessor<ScalarT, 2> &pointsAcc =
                jaggedPoints.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &pointsBOffsetsAcc =
                jaggedPoints.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> batchHandles;
            batchHandles.reserve(pointsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (pointsBOffsetsAcc.size(0) - 1); bi += 1) {
                const VoxelCoordTransform &tx = txs[bi];

                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = pointsBOffsetsAcc[bi];
                const int64_t end   = pointsBOffsetsAcc[bi + 1];

                for (int64_t pi = start; pi < end; pi += 1) {
                    Vec3T ijk0            = tx.apply(static_cast<MathT>(pointsAcc[pi][0]),
                                          static_cast<MathT>(pointsAcc[pi][1]),
                                          static_cast<MathT>(pointsAcc[pi][2]));
                    nanovdb::Coord ijk000 = ijk0.floor();
                    nanovdb::Coord ijk001 = ijk000 + nanovdb::Coord(0, 0, 1);
                    nanovdb::Coord ijk010 = ijk000 + nanovdb::Coord(0, 1, 0);
                    nanovdb::Coord ijk011 = ijk000 + nanovdb::Coord(0, 1, 1);
                    nanovdb::Coord ijk100 = ijk000 + nanovdb::Coord(1, 0, 0);
                    nanovdb::Coord ijk101 = ijk000 + nanovdb::Coord(1, 0, 1);
                    nanovdb::Coord ijk110 = ijk000 + nanovdb::Coord(1, 1, 0);
                    nanovdb::Coord ijk111 = ijk000 + nanovdb::Coord(1, 1, 1);

                    proxyGridAccessor.setValue(ijk000, 11.0f);
                    proxyGridAccessor.setValue(ijk001, 11.0f);
                    proxyGridAccessor.setValue(ijk010, 11.0f);
                    proxyGridAccessor.setValue(ijk011, 11.0f);
                    proxyGridAccessor.setValue(ijk100, 11.0f);
                    proxyGridAccessor.setValue(ijk101, 11.0f);
                    proxyGridAccessor.setValue(ijk110, 11.0f);
                    proxyGridAccessor.setValue(ijk111, 11.0f);
                }

                proxyGridAccessor.merge();
                batchHandles.push_back(
                    nanovdb::tools::createNanoGrid<ProxyGridT, GridT, nanovdb::HostBuffer>(
                        *proxyGrid, 0u, false, false));
            }

            return GridStorage(batchHandles.size() == 1 ? std::move(batchHandles[0])
                                                        : nanovdb::mergeGrids(batchHandles));
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

c10::intrusive_ptr<GridBatchData>
buildGridFromNearestVoxelsToPoints(const JaggedTensor &points,
                                   const std::vector<nanovdb::Vec3d> &voxelSizes,
                                   const std::vector<nanovdb::Vec3d> &origins) {
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(points.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(points.rsize(1) == 3,
                      "Expected 3 dimensional points but got points.shape[1] = " +
                          std::to_string(points.rsize(1)));
    TORCH_CHECK(points.num_tensors() == points.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(points.num_outer_lists() <= GridBatchData::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchData::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      points.num_outer_lists(),
                      " point sets.");
    const int64_t numGrids = points.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == points.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    std::vector<VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }
    GridStorage storage = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        return dispatchBuildGridFromNearestVoxelsToPoints<DeviceTag>(points, transforms);
    });
    return makeGridBatchData(std::move(storage), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
