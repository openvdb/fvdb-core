// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromMesh.h>
#include <fvdb/detail/ops/IjkForMesh.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>

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
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromMesh(const JaggedTensor &meshVertices,
                          const JaggedTensor &meshFaces,
                          const std::vector<VoxelCoordTransform> &tx);

template <typename ScalarType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromMeshCPU(const JaggedTensor &vertices,
                     const JaggedTensor &triangles,
                     const std::vector<VoxelCoordTransform> &tx) {
    using GridT      = nanovdb::ValueOnIndex;
    using Vec3T      = nanovdb::math::Vec3<ScalarType>;
    using ProxyGridT = nanovdb::tools::build::Grid<float>;

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(vertices.num_outer_lists());

    for (int64_t bidx = 0; bidx < vertices.num_outer_lists(); bidx += 1) {
        const torch::Tensor ti         = triangles.index({bidx}).jdata();
        const torch::Tensor vi         = vertices.index({bidx}).jdata();
        const VoxelCoordTransform &txi = tx[bidx];

        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        // int64_t numSearched = 0;
        // int64_t numFound = 0;
        // For each face, compute the min max voxels
        for (int faceId = 0; faceId < ti.size(0); faceId += 1) {
            const torch::Tensor face         = ti.index({faceId}); // 3
            const torch::Tensor faceVertices = vi.index({face});   // [3, 3]
            torch::TensorAccessor<ScalarType, 2> faceVerticesAcc =
                faceVertices.accessor<ScalarType, 2>();
            const Vec3T v1 = txi.apply(
                Vec3T(faceVerticesAcc[0][0], faceVerticesAcc[0][1], faceVerticesAcc[0][2]));
            const Vec3T v2 = txi.apply(
                Vec3T(faceVerticesAcc[1][0], faceVerticesAcc[1][1], faceVerticesAcc[1][2]));
            const Vec3T v3 = txi.apply(
                Vec3T(faceVerticesAcc[2][0], faceVerticesAcc[2][1], faceVerticesAcc[2][2]));

            const Vec3T e1 = v2 - v1;
            const Vec3T e2 = v3 - v1;
            const ScalarType spacing =
                sqrt(3.0) / 3.0; // This is very conservative spacing but fine for now
            const int32_t numU = ceil((e1.length() + spacing) / spacing);
            const int32_t numV = ceil((e2.length() + spacing) / spacing);

            // numSearched += (numU * numV);
            for (int i = 0; i < numU; i += 1) {
                for (int j = 0; j < numV; j += 1) {
                    ScalarType u = ScalarType(i) / (ScalarType(std::max(numU - 1, 1)));
                    ScalarType v = ScalarType(j) / (ScalarType(std::max(numV - 1, 1)));
                    if (u + v >= 1.0) {
                        u = 1.0 - u;
                        v = 1.0 - v;
                    }
                    const Vec3T p            = v1 + e1 * u + e2 * v;
                    const nanovdb::Coord ijk = p.round();

                    proxyGridAccessor.setValue(ijk, 1.0f);
                    // numFound += 1;
                }
            }
        }

        // std::cerr << "I searched over " << numSearched << " voxels" << std::endl;
        // std::cerr << "I found " << numFound << " voxels" << std::endl;
        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromMesh<torch::kCUDA>(const JaggedTensor &meshVertices,
                                        const JaggedTensor &meshFaces,
                                        const std::vector<VoxelCoordTransform> &tx) {
    JaggedTensor coords = ops::ijkForMesh(meshVertices, meshFaces, tx);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromMesh<torch::kCPU>(const JaggedTensor &meshVertices,
                                       const JaggedTensor &meshFaces,
                                       const std::vector<VoxelCoordTransform> &tx) {
    return AT_DISPATCH_V2(
        meshVertices.scalar_type(),
        "buildGridFromMeshCPU",
        AT_WRAP([&]() { return buildGridFromMeshCPU<scalar_t>(meshVertices, meshFaces, tx); }),
        AT_EXPAND(AT_FLOATING_TYPES));
}

c10::intrusive_ptr<GridBatchData>
buildGridFromMesh(const JaggedTensor &meshVertices,
                  const JaggedTensor &meshFaces,
                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                  const std::vector<nanovdb::Vec3d> &origins) {
    TORCH_CHECK_VALUE(
        meshVertices.device() == meshFaces.device(),
        "meshVertices and meshFaces must be on the same device, but got meshVertices.device() = ",
        meshVertices.device(),
        " and meshFaces.device() = ",
        meshFaces.device());
    TORCH_CHECK_VALUE(
        meshVertices.ldim() == 1,
        "Expected meshVertices to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        meshVertices.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        meshFaces.ldim() == 1,
        "Expected meshFaces to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        meshFaces.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(meshVertices.is_floating_point(),
                     "meshVertices must have a floating point type");
    TORCH_CHECK_VALUE(
        meshVertices.rdim() == 2,
        std::string("Expected meshVertices to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(meshVertices.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(meshVertices.rsize(1) == 3,
                      "Expected 3 dimensional meshVertices but got meshVertices.rshape[1] = " +
                          std::to_string(meshVertices.rsize(1)));
    TORCH_CHECK_TYPE(!meshFaces.is_floating_point(), "meshFaces must have an integer type");
    TORCH_CHECK_VALUE(
        meshFaces.rdim() == 2,
        std::string("Expected meshFaces to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(meshFaces.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(meshFaces.rsize(1) == 3,
                      "Expected 3 dimensional meshFaces but got meshFaces.rshape[1] = " +
                          std::to_string(meshFaces.rsize(1)));
    TORCH_CHECK_VALUE(meshVertices.num_outer_lists() == meshFaces.num_outer_lists(),
                      "Expected same number of vertex and face sets got len(meshVertices) = ",
                      meshVertices.num_outer_lists(),
                      " and len(meshFaces) = ",
                      meshFaces.num_outer_lists());
    const int64_t numGrids = meshVertices.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == meshVertices.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(numGrids <= GridBatchData::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchData::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      numGrids,
                      " mesh sets.");
    std::vector<VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }
    auto handle = FVDB_DISPATCH_KERNEL_DEVICE(meshVertices.device(), [&]() {
        return dispatchBuildGridFromMesh<DeviceTag>(meshVertices, meshFaces, transforms);
    });
    return c10::make_intrusive<GridBatchData>(std::move(handle), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
