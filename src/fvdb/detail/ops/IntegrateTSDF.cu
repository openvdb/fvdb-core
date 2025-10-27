// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/IntegrateTSDF.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/math/Math.h>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/intrusive_ptr.h>

#include <cuda_fp16.h>

#include <optional>

namespace fvdb::detail::ops {

template <typename ScalarType>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
unprojectDepthmapKernel(int64_t imageWidth,
                        int64_t imageHeight,
                        fvdb::TorchRAcc64<ScalarType, 3> invProjMats,
                        fvdb::TorchRAcc64<ScalarType, 3> camToWorldMats,
                        fvdb::TorchRAcc64<ScalarType, 3> depthImages,
                        fvdb::TorchRAcc64<ScalarType, 3> outPoints) {
    using Vec3T = nanovdb::math::Vec3<ScalarType>;
    using Vec4T = nanovdb::math::Vec4<ScalarType>;
    using Mat3T = nanovdb::math::Mat3<ScalarType>;
    using Mat4T = nanovdb::math::Mat4<ScalarType>;

    const auto batchSize = invProjMats.size(0);

    const auto sharedMat3x3NumElements = batchSize * 3 * 3;
    const auto sharedMat4x4NumElements = batchSize * 4 * 4;

    extern __shared__ uint8_t sharedData[];

    Mat3T *sharedInvProjMats    = reinterpret_cast<Mat3T *>(sharedData);
    Mat4T *sharedCamToWorldMats = reinterpret_cast<Mat4T *>(sharedData + batchSize * sizeof(Mat3T));

    // Load view and projection matrices into shared memory
    if (threadIdx.x < sharedMat3x3NumElements) {
        const auto batchIdx                         = threadIdx.x / 9;
        const auto rowIdx                           = (threadIdx.x % 9) / 3;
        const auto colIdx                           = threadIdx.x % 3;
        sharedInvProjMats[batchIdx][rowIdx][colIdx] = invProjMats[batchIdx][rowIdx][colIdx];
    } else if (threadIdx.x < sharedMat3x3NumElements + sharedMat4x4NumElements) {
        const auto baseIdx                             = threadIdx.x - sharedMat3x3NumElements;
        const auto batchIdx                            = baseIdx / 16;
        const auto rowIdx                              = (baseIdx % 16) / 4;
        const auto colIdx                              = baseIdx % 4;
        sharedCamToWorldMats[batchIdx][rowIdx][colIdx] = camToWorldMats[batchIdx][rowIdx][colIdx];
    }

    __syncthreads();

    // Parallelize over all pixels in all images
    const auto problemSize = imageWidth * imageHeight * batchSize;
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < problemSize;
         idx += blockDim.x * gridDim.x) {
        const auto batchIdx = idx / (imageWidth * imageHeight); // [0, batchSize-1]
        const auto pixelIdx = idx % (imageWidth * imageHeight); // [0, imageWidth*imageHeight-1]
        const auto rowIdx   = pixelIdx / imageWidth;            // [0, imageHeight-1]
        const auto colIdx   = pixelIdx % imageWidth;            // [0, imageWidth-1]

        if (rowIdx >= imageHeight || colIdx >= imageWidth) {
            continue;
        }

        const auto depth           = depthImages[batchIdx][rowIdx][colIdx];
        const Vec3T screenSpacePos = {
            static_cast<ScalarType>(colIdx), static_cast<ScalarType>(rowIdx), ScalarType(1)};
        const Vec3T camSpacePos = (sharedInvProjMats[batchIdx] * screenSpacePos) * depth;

        const Vec4T camSpacePosHomogeneous = {
            camSpacePos[0], camSpacePos[1], camSpacePos[2], ScalarType(1)};
        const Vec4T worldSpacePos = sharedCamToWorldMats[batchIdx] * camSpacePosHomogeneous;

        outPoints[batchIdx][pixelIdx][0] = worldSpacePos[0] / worldSpacePos[3];
        outPoints[batchIdx][pixelIdx][1] = worldSpacePos[1] / worldSpacePos[3];
        outPoints[batchIdx][pixelIdx][2] = worldSpacePos[2] / worldSpacePos[3];
    }
}

template <typename T> struct OpType {
    using type = T;
};

template <> struct OpType<c10::Half> {
    using type = float;
};

template <> struct OpType<nv_half> {
    using type = float;
};

template <typename ScalarDataType, typename FeatureScalarDataType = ScalarDataType>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
integrateTSDFKernel(const ScalarDataType truncationMargin,
                    const int64_t imageWidth,
                    const int64_t imageHeight,
                    const bool hasFeatures,
                    const bool hasWeights,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> projMats,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> invProjMats,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> worldToCamMats,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> camToWorldMats,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> depthImages,
                    const fvdb::TorchRAcc64<FeatureScalarDataType, 4> featureImages,
                    const fvdb::TorchRAcc64<ScalarDataType, 3> weightImages,
                    const fvdb::detail::BatchGridAccessor baseGridAcc,
                    const fvdb::detail::BatchGridAccessor unionGridAcc,
                    const fvdb::JaggedRAcc64<ScalarDataType, 1> tsdfAcc,
                    const fvdb::JaggedRAcc64<ScalarDataType, 1> weightsAcc,
                    const fvdb::JaggedRAcc64<FeatureScalarDataType, 2> featuresAcc,
                    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
                    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
                    fvdb::TorchRAcc64<FeatureScalarDataType, 2> outFeaturesAcc) {
    using ScalarType        = OpType<ScalarDataType>::type;
    using FeatureScalarType = OpType<FeatureScalarDataType>::type;

    using GridT        = nanovdb::ValueOnIndex;
    using LeafNodeType = nanovdb::NanoGrid<GridT>::LeafNodeType;
    using Vec3T        = nanovdb::math::Vec3<ScalarType>;
    using Vec4T        = nanovdb::math::Vec4<ScalarType>;
    using Mat3T        = nanovdb::math::Mat3<ScalarType>;
    using Mat4T        = nanovdb::math::Mat4<ScalarType>;

    constexpr uint64_t VOXELS_PER_LEAF = nanovdb::NanoTree<GridT>::LeafNodeType::NUM_VALUES;

    const auto batchSize = projMats.size(0);

    // Grab pointers to the transformation matrices in shared memory
    extern __shared__ uint8_t sharedData[];
    Mat3T *sharedProjMats       = reinterpret_cast<Mat3T *>(sharedData);
    Mat4T *sharedWorldToCamMats = reinterpret_cast<Mat4T *>(sharedData + batchSize * sizeof(Mat3T));
    Mat3T *sharedInvProjMats =
        reinterpret_cast<Mat3T *>(sharedData + batchSize * (sizeof(Mat3T) + sizeof(Mat4T)));
    Mat4T *sharedCamToWorldMats = reinterpret_cast<Mat4T *>(
        sharedData + batchSize * (sizeof(Mat3T) + sizeof(Mat4T) + sizeof(Mat3T)));

    const auto sharedMat3x3NumElements = batchSize * 3 * 3;
    const auto sharedMat4x4NumElements = batchSize * 4 * 4;

    // Load view and projection matrices into shared memory
    if (threadIdx.x < sharedMat3x3NumElements) {
        const auto batchIdx                      = threadIdx.x / 9;
        const auto rowIdx                        = (threadIdx.x % 9) / 3;
        const auto colIdx                        = threadIdx.x % 3;
        sharedProjMats[batchIdx][rowIdx][colIdx] = ScalarType(projMats[batchIdx][rowIdx][colIdx]);
    } else if (threadIdx.x < sharedMat3x3NumElements + sharedMat4x4NumElements) {
        const auto baseIdx  = threadIdx.x - sharedMat3x3NumElements;
        const auto batchIdx = baseIdx / 16;
        const auto rowIdx   = (baseIdx % 16) / 4;
        const auto colIdx   = baseIdx % 4;
        sharedWorldToCamMats[batchIdx][rowIdx][colIdx] =
            ScalarType(worldToCamMats[batchIdx][rowIdx][colIdx]);
    } else if (threadIdx.x < 2 * sharedMat3x3NumElements + sharedMat4x4NumElements) {
        const auto baseIdx  = threadIdx.x - sharedMat3x3NumElements - sharedMat4x4NumElements;
        const auto batchIdx = baseIdx / 9;
        const auto rowIdx   = (baseIdx % 9) / 3;
        const auto colIdx   = baseIdx % 3;
        sharedInvProjMats[batchIdx][rowIdx][colIdx] =
            ScalarType(invProjMats[batchIdx][rowIdx][colIdx]);
    }

    __syncthreads();

    // Parallelize over all voxels in the leaf nodes (whether enabled or not)
    const auto problemSize = unionGridAcc.totalLeaves() * VOXELS_PER_LEAF;
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < problemSize;
         idx += blockDim.x * gridDim.x) {
        // Which leaf we're in relative to all grids in the batch
        const int64_t cumUnionLeafIdx = static_cast<int64_t>(idx / VOXELS_PER_LEAF);

        // Which grid we're in
        const fvdb::JIdxType batchIdx = unionGridAcc.leafBatchIndex(cumUnionLeafIdx);

        // Which leaf we're in relative to the grid
        const int64_t unionLeafIdx = cumUnionLeafIdx - unionGridAcc.leafOffset(batchIdx);

        // Which voxel we're in relative to the leaf
        const int64_t unionLeafVoxelIdx =
            static_cast<int64_t>((idx - cumUnionLeafIdx * VOXELS_PER_LEAF));

        // Get pointers to each grid
        const nanovdb::NanoGrid<GridT> *unionGrid = unionGridAcc.grid(batchIdx);
        const nanovdb::NanoGrid<GridT> *baseGrid  = baseGridAcc.grid(batchIdx);

        // Get the leaf node in the union grid
        const LeafNodeType &unionLeaf = unionGrid->tree().template getFirstNode<0>()[unionLeafIdx];

        // Get the ijk coordinate of the current voxel
        const nanovdb::Coord ijk = unionLeaf.offsetToGlobalCoord(unionLeafVoxelIdx);

        // Which sidecar index the current voxel in the union grid corresponds to
        const int64_t unionWriteOffset =
            unionGridAcc.voxelOffset(batchIdx) +
            static_cast<int64_t>(unionLeaf.getValue(unionLeafVoxelIdx)) - 1;

        // If this is not an active voxel in the union grid, skip it
        if (unionWriteOffset < 0) {
            continue;
        }

        // World space position of the voxel in the union grid
        const Vec3T voxelWorldPos = unionGridAcc.primalTransform(batchIdx).applyInv<ScalarType>(
            ScalarType(ijk[0]), ScalarType(ijk[1]), ScalarType(ijk[2]));

        const Vec4T voxelWorldPosHomogeneous = {
            voxelWorldPos[0], voxelWorldPos[1], voxelWorldPos[2], ScalarType(1.0)};
        const Vec4T voxelPosCamSpace    = sharedWorldToCamMats[batchIdx] * voxelWorldPosHomogeneous;
        const Vec3T voxelPosCamSpace3d  = {voxelPosCamSpace[0] / voxelPosCamSpace[3],
                                           voxelPosCamSpace[1] / voxelPosCamSpace[3],
                                           voxelPosCamSpace[2] / voxelPosCamSpace[3]};
        const Vec3T voxelPosProjSpace   = sharedProjMats[batchIdx] * voxelPosCamSpace3d;
        const Vec3T voxelPosScreenSpace = {voxelPosProjSpace[0] / voxelPosProjSpace[2],
                                           voxelPosProjSpace[1] / voxelPosProjSpace[2],
                                           ScalarType(1.0)};

        const int64_t voxelPosScreenSpaceX = int64_t(voxelPosScreenSpace[0]);
        const int64_t voxelPosScreenSpaceY = int64_t(voxelPosScreenSpace[1]);

        const auto baseGridTreeAccessor = baseGrid->getAccessor();
        const int64_t baseGridOffset    = baseGridAcc.voxelOffset(batchIdx) +
                                       static_cast<int64_t>(baseGridTreeAccessor.getValue(ijk)) - 1;

        const bool voxelInBaseGrid = baseGridOffset >= 0;

        const auto copyOldToNew = [&]() {
            if (voxelInBaseGrid) {
                outWeightsAcc[unionWriteOffset] = weightsAcc.data()[baseGridOffset];
                outTsdfAcc[unionWriteOffset]    = tsdfAcc.data()[baseGridOffset];
                if (hasFeatures) {
                    for (auto i = 0; i < outFeaturesAcc.size(1); ++i) {
                        outFeaturesAcc[unionWriteOffset][i] = featuresAcc.data()[baseGridOffset][i];
                    }
                }
            } else {
                outWeightsAcc[unionWriteOffset] = ScalarDataType(0);
                outTsdfAcc[unionWriteOffset]    = ScalarDataType(0);
                if (hasFeatures) {
                    for (auto i = 0; i < outFeaturesAcc.size(1); ++i) {
                        outFeaturesAcc[unionWriteOffset][i] = ScalarDataType(0);
                    }
                }
            }
        };

        // This voxel is not visible in the image, so just copy whatever value was in the base grid
        const bool voxelIsVisible =
            (voxelPosScreenSpaceX >= 0 && voxelPosScreenSpaceX < imageWidth &&
             voxelPosScreenSpaceY >= 0 && voxelPosScreenSpaceY < imageHeight &&
             voxelPosCamSpace3d[2] > 0.0f);
        if (!voxelIsVisible) {
            copyOldToNew();
            continue;
        }

        const ScalarType pixelDepth =
            ScalarType(depthImages[batchIdx][voxelPosScreenSpaceY][voxelPosScreenSpaceX]);
        const ScalarType voxelDepth                = voxelPosCamSpace3d.length();
        const Vec3T voxelScreenSpacePosHomogeneous = {
            ScalarType(voxelPosScreenSpaceX), ScalarType(voxelPosScreenSpaceY), ScalarType(1.0)};
        const Vec3T unprojectedPixelPosCamSpace =
            (sharedInvProjMats[batchIdx] * voxelScreenSpacePosHomogeneous) * pixelDepth;

        // const ScalarType zDiff = unprojectedPixelPosCamSpace[2] - voxelPosCamSpace3d[2];
        const ScalarType zDiff = pixelDepth - voxelPosCamSpace3d[2];

        // If the voxel is too far behind the point, then it's not visible and we don't know
        // what the value is, so we copy teh value from the base grid
        if (zDiff > -ScalarType(truncationMargin)) {
            const ScalarType oldWeight =
                voxelInBaseGrid ? ScalarType(weightsAcc.data()[baseGridOffset]) : ScalarType(0);
            const ScalarType oldTsdf =
                voxelInBaseGrid ? ScalarType(tsdfAcc.data()[baseGridOffset]) : ScalarType(0);
            const ScalarType tsdf =
                nanovdb::math::Min(ScalarType(1), zDiff / ScalarType(truncationMargin));

            const ScalarType pixelWeight = [&]() {
                if (hasWeights) {
                    return ScalarType(
                        weightImages[batchIdx][voxelPosScreenSpaceY][voxelPosScreenSpaceX]);
                } else {
                    return ScalarType{1};
                }
            }();

            if (pixelWeight <= ScalarType(0)) {
                // If the new weight is zero, we don't update the TSDF or features
                copyOldToNew();
                continue;
            }
            const ScalarType newWeight =
                oldWeight + pixelWeight; // ScalarType(1) + oldWeight * pixelWeight;
            const ScalarType newTsdf        = (oldWeight * oldTsdf + tsdf) / newWeight;
            outTsdfAcc[unionWriteOffset]    = ScalarDataType(newTsdf);
            outWeightsAcc[unionWriteOffset] = ScalarDataType(newWeight);
            if (hasFeatures) {
                for (auto i = 0; i < outFeaturesAcc.size(1); ++i) {
                    const ScalarType pixelFeatureI = ScalarType(
                        featureImages[batchIdx][voxelPosScreenSpaceY][voxelPosScreenSpaceX][i]);
                    const ScalarType oldFeatureI =
                        voxelInBaseGrid ? ScalarType(featuresAcc.data()[baseGridOffset][i])
                                        : ScalarType(0);
                    outFeaturesAcc[unionWriteOffset][i] = FeatureScalarDataType(
                        (oldWeight * oldFeatureI + pixelFeatureI) / newWeight);
                }
            }
        } else {
            copyOldToNew();
            continue;
        }
    }
}

torch::Tensor
unprojectDepthMapToPoints(const torch::Tensor &depthImages,
                          const torch::Tensor &projectionMatrices,
                          const torch::Tensor &invProjectionMatrices,
                          const torch::Tensor &camToWorldMatrices) {
    const int64_t batchSize      = depthImages.size(0);
    const int64_t imageHeight    = depthImages.size(1);
    const int64_t imageWidth     = depthImages.size(2);
    const int64_t pointsPerImage = imageHeight * imageWidth;
    const int64_t numPoints      = batchSize * pointsPerImage;

    torch::Tensor outUnprojectedPoints =
        torch::empty({batchSize, pointsPerImage, 3}, depthImages.options());

    AT_DISPATCH_V2(
        depthImages.scalar_type(),
        "unprojectDepthmapKernel",
        AT_WRAP([&]() {
            using Mat3T = nanovdb::math::Mat3<scalar_t>;
            using Mat4T = nanovdb::math::Mat4<scalar_t>;

            const auto numSharedScalars = batchSize * 3 * 3 + batchSize * 4 * 4;
            const auto problemSize      = std::max(numPoints, numSharedScalars);
            const auto numBlocks        = GET_BLOCKS(problemSize, DEFAULT_BLOCK_DIM);
            const auto sharedSize       = batchSize * (sizeof(Mat3T) + sizeof(Mat4T));

            at::cuda::CUDAStream stream =
                at::cuda::getCurrentCUDAStream(depthImages.device().index());

            if (cudaFuncSetAttribute(unprojectDepthmapKernel<scalar_t>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedSize) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         sharedSize,
                         " bytes), try lowering tile_size.");
            }
            unprojectDepthmapKernel<<<numBlocks, DEFAULT_BLOCK_DIM, sharedSize, stream>>>(
                imageWidth,
                imageHeight,
                invProjectionMatrices.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                camToWorldMatrices.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                depthImages.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                outUnprojectedPoints.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return outUnprojectedPoints;
}

c10::intrusive_ptr<GridBatchImpl>
buildPointGrid(const double truncationMargin,
               const torch::Tensor &unprojectedPoints,
               const GridBatchImpl &grid) {
    std::vector<int64_t> numPadVoxels;
    std::vector<torch::Tensor> jaggedPointsList;
    for (auto i = 0; i < unprojectedPoints.size(0); ++i) {
        jaggedPointsList.push_back(unprojectedPoints[i]);
        const auto minVoxLengthI = grid.voxelSizeAt(i).min();
        const auto numPadVoxelsI = static_cast<int32_t>(ceil(truncationMargin / minVoxLengthI));
        TORCH_CHECK(numPadVoxelsI > 0,
                    "Number of padding voxels must be non-negative, but got ",
                    numPadVoxelsI);
        constexpr int64_t MAX_PAD_VOXELS = 16;
        TORCH_CHECK(numPadVoxelsI < MAX_PAD_VOXELS,
                    "Truncation margin (",
                    truncationMargin,
                    ") is too large for grid with voxel size ",
                    minVoxLengthI,
                    ", resulting in too many padding voxels (",
                    numPadVoxelsI,
                    ") which cannot exceed ",
                    MAX_PAD_VOXELS,
                    ". Use a larger voxel size or a smaller truncation margin.");
        numPadVoxels.push_back(numPadVoxelsI);
    }
    const JaggedTensor jaggedPoints(jaggedPointsList);

    std::vector<nanovdb::Vec3d> voxelSizes;
    std::vector<nanovdb::Vec3d> origins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, origins);
    return GridBatchImpl::createFromPoints(jaggedPoints, voxelSizes, origins)->dilate(numPadVoxels);
}

#define DISPATCH_FEATURE_TYPE(...)                                \
    if (hasFeatures && features.scalar_type() == torch::kUInt8) { \
        using feature_t = uint8_t;                                \
        __VA_ARGS__();                                            \
    } else {                                                      \
        using feature_t = scalar_t;                               \
        __VA_ARGS__();                                            \
    }

std::tuple<JaggedTensor, JaggedTensor, JaggedTensor>
doIntegrate(const float truncationMargin,
            const torch::Tensor &depthImages,
            const torch::Tensor &featureImages,
            const torch::Tensor &weightImages,
            const torch::Tensor &projectionMatrices,
            const torch::Tensor &invProjectionMatrices,
            const torch::Tensor &camToWorldMatrices,
            const torch::Tensor &worldToCamMatrices,
            const GridBatchImpl &unionGrid,
            const GridBatchImpl &baseGrid,
            const JaggedTensor &tsdf,
            const JaggedTensor &weights,
            const JaggedTensor &features) {
    const int64_t batchSize      = depthImages.size(0);
    const int64_t imageHeight    = depthImages.size(1);
    const int64_t imageWidth     = depthImages.size(2);
    const int64_t totalOutVoxels = unionGrid.totalVoxels();
    const int64_t featureDim     = features.rsize(-1);
    const bool hasFeatures       = featureDim > 0;
    const bool hasWeights        = weightImages.size(0) > 0;

    torch::Tensor outWeights = torch::zeros({totalOutVoxels}, weights.jdata().options());
    torch::Tensor outTsdf    = torch::zeros({totalOutVoxels}, tsdf.jdata().options());
    torch::Tensor outFeatures =
        torch::empty({totalOutVoxels, featureDim}, features.jdata().options());

    AT_DISPATCH_V2(
        tsdf.scalar_type(),
        "integrateTSDFKernel",
        AT_WRAP([&]() {
            using shared_scalar_t = typename OpType<scalar_t>::type;
            using SharedMat3T      = nanovdb::math::Mat3<shared_scalar_t>;
            using SharedMat4T      = nanovdb::math::Mat4<shared_scalar_t>;
            constexpr uint64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
            const auto numUnionLeaves          = unionGrid.totalLeaves();
            const auto numSharedScalars        = 2 * batchSize * 3 * 3 + 2 * batchSize * 4 * 4;
            const auto problemSize =
                std::max(numUnionLeaves * VOXELS_PER_LEAF, uint64_t(numSharedScalars));
            const auto sharedMemSize =
                2 * batchSize * sizeof(SharedMat3T) + 2 * batchSize * sizeof(SharedMat4T);
            const auto numBlocks     = GET_BLOCKS(problemSize, DEFAULT_BLOCK_DIM);

            const auto dtype                = tsdf.scalar_type();
            const auto projMatsCasted       = projectionMatrices.to(dtype);
            const auto invProjMatsCasted    = invProjectionMatrices.to(dtype);
            const auto camToWorldMatsCasted = camToWorldMatrices.to(dtype);
            const auto worldToCamMatsCasted = worldToCamMatrices.to(dtype);

            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(tsdf.device().index());

            if (cudaFuncSetAttribute(integrateTSDFKernel<scalar_t>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedMemSize) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         sharedMemSize,
                         " bytes), try lowering tile_size.");
            }

            // Special case for uint8 features
            // We don't need to do anything special here, but we need to ensure that the
            // kernel is called with the correct scalar type.
            DISPATCH_FEATURE_TYPE([&]() {
                integrateTSDFKernel<<<numBlocks, DEFAULT_BLOCK_DIM, sharedMemSize, stream>>>(
                    scalar_t(truncationMargin),
                    imageWidth,
                    imageHeight,
                    hasFeatures,
                    hasWeights,
                    projMatsCasted.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    invProjMatsCasted.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    worldToCamMatsCasted.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    camToWorldMatsCasted.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    depthImages.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    featureImages.packed_accessor64<feature_t, 4, torch::RestrictPtrTraits>(),
                    weightImages.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    baseGrid.deviceAccessor(),
                    unionGrid.deviceAccessor(),
                    tsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    weights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    features.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>(),
                    outTsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    outWeights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    outFeatures.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>());
            });
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return {unionGrid.jaggedTensor(outTsdf),
            unionGrid.jaggedTensor(outWeights),
            unionGrid.jaggedTensor(outFeatures)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
getCameraMatrices(const torch::Tensor &projectionMatrices,
                  const torch::Tensor &camToWorldMatrices) {
    // Maybe make a copy to store the matrices in float32 if they are passed in as float16
    // This is to ensure better numerical stability during the inverse operation
    // and because the inverse operation is not supported for float16 in PyTorch
    torch::Tensor projectionMats = projectionMatrices;
    torch::Tensor camToWorldMats = camToWorldMatrices;
    if (projectionMatrices.scalar_type() == torch::kFloat16) {
        projectionMats = projectionMatrices.to(torch::kFloat32);
    }
    if (camToWorldMatrices.scalar_type() == torch::kFloat16) {
        camToWorldMats = camToWorldMatrices.to(torch::kFloat32);
    }

    const torch::Tensor worldToCamMats    = torch::inverse(camToWorldMats);
    const torch::Tensor invProjectionMats = torch::inverse(projectionMats);

    const auto dtype = projectionMatrices.scalar_type();
    return {projectionMats.to(dtype),
            invProjectionMats.to(dtype),
            camToWorldMats.to(dtype),
            worldToCamMats.to(dtype)};
}

void
checkInputTypes(const JaggedTensor &tsdf,
                const JaggedTensor &weights,
                const std::optional<JaggedTensor> &features,
                const torch::Tensor &depthImages,
                const std::optional<torch::Tensor> &featureImages,
                const std::optional<torch::Tensor> &weightImages,
                const torch::Tensor &projectionMatrices,
                const torch::Tensor &camToWorldMatrices) {
    // We support a few different scalar types for the TSDF, weights, and depth images, and
    // features
    //  - The weights, TSDF, and depth images must all have the same scalar type and be one of
    //    float64, float32, or float16.
    //  - The features must have the same scalar type as the TSDF, or be uint8 (since this is such a
    //    common case for colors).

    // Step 0. Check that TSDf/weights/depthImages all have the same scalar type
    //         and that type is a floating point type
    const auto dtype = tsdf.scalar_type();
    TORCH_CHECK_TYPE(dtype == torch::kFloat32 || dtype == torch::kFloat64 ||
                         dtype == torch::kFloat16,
                     "TSDF values must be of type float32, float64, or float16, but got ",
                     dtype);
    TORCH_CHECK_TYPE(weights.scalar_type() == dtype,
                     "Weights must be of the same type as TSDF values, but got weights.dtype =",
                     weights.scalar_type(),
                     " and tsdf.dtype = ",
                     dtype);
    TORCH_CHECK_TYPE(
        depthImages.scalar_type() == dtype,
        "Depth images must be of the same type as TSDF values, but got depth_images.dtype =",
        depthImages.scalar_type(),
        " and tsdf.dtype = ",
        dtype);

    // Step 1. If the user passes in features, check that their scalar type is either the same
    //         as the TSDF values or uint8 (which is common for RGB colors).
    if (features.has_value()) {
        TORCH_CHECK_TYPE(features.value().scalar_type() == dtype ||
                             features.value().scalar_type() == torch::kUInt8,
                         "Features must be of the same type as TSDF values or uint8, but got "
                         "features.dtype = ",
                         features.value().scalar_type(),
                         " and tsdf.dtype = ",
                         dtype);
        TORCH_CHECK_TYPE(featureImages.has_value(),
                         "Feature images must be provided if features are provided.");
        TORCH_CHECK_TYPE(featureImages.value().scalar_type() == features.value().scalar_type(),
                         "Feature images must be of the same type as features, but got "
                         "feature_images.dtype = ",
                         featureImages.value().scalar_type(),
                         " and features.dtype = ",
                         features.value().scalar_type());
    }
    if (weightImages.has_value()) {
        TORCH_CHECK_TYPE(weightImages.value().scalar_type() == dtype,
                         "Weight images must be of the same type as TSDF values, but got "
                         "weight_images.dtype = ",
                         weightImages.value().scalar_type(),
                         " and tsdf.dtype = ",
                         dtype);
    }

    // Step 3. Check that the projection matrices and camera-to-world matrices
    //         have the same scalar type as the TSDF values
    TORCH_CHECK_TYPE(projectionMatrices.scalar_type() == dtype,
                     "Projection matrices must be of the same type as TSDF values, but got "
                     "projection_matrices.dtype = ",
                     projectionMatrices.scalar_type(),
                     " and tsdf.dtype = ",
                     dtype);
    TORCH_CHECK_TYPE(camToWorldMatrices.scalar_type() == dtype,
                     "Camera-to-world matrices must be of the same type as TSDF values, but "
                     "got cam_to_world_matrices.dtype = ",
                     camToWorldMatrices.scalar_type(),
                     " and tsdf.dtype = ",
                     dtype);
}

void
checkInputSizes(const GridBatchImpl &grid,
                const JaggedTensor &tsdf,
                const JaggedTensor &weights,
                const std::optional<JaggedTensor> &features,
                const torch::Tensor &depthImages,
                const std::optional<torch::Tensor> &featureImages,
                const std::optional<torch::Tensor> &weightImages,
                const torch::Tensor &projectionMatrices,
                const torch::Tensor &camToWorldMatrices) {
    // Step 0. Check that the input tensors have the correct dimensions
    TORCH_CHECK_VALUE(depthImages.dim() == 3 || depthImages.dim() == 4,
                      "Depth images must be of shape (batch_size, image_height, image_width) or "
                      "(batch_size, image_height, image_width, 1), but got ",
                      depthImages.sizes());
    TORCH_CHECK_VALUE(projectionMatrices.dim() == 3 && projectionMatrices.size(1) == 3 &&
                          projectionMatrices.size(2) == 3,
                      "Projection matrices must be of shape (batch_size, 3, 3), but got ",
                      projectionMatrices.sizes());
    TORCH_CHECK_VALUE(camToWorldMatrices.dim() == 3 && camToWorldMatrices.size(1) == 4 &&
                          camToWorldMatrices.size(2) == 4,
                      "Camera-to-world matrices must be of shape (batch_size, 4, 4), but got ",
                      camToWorldMatrices.sizes());
    TORCH_CHECK_VALUE(
        tsdf.rdim() == 1, "TSDF must be a 1D tensor, but got element dimension", tsdf.esizes());
    TORCH_CHECK_VALUE(weights.rdim() == 1,
                      "Weights must be a 1D tensor, but got element dimension",
                      weights.esizes());
    if (features.has_value()) {
        TORCH_CHECK_VALUE(features.value().rdim() == 2,
                          "Features must be a 2D tensor, but got element dimension",
                          features.value().esizes());
        TORCH_CHECK_VALUE(featureImages.has_value(),
                          "Feature images must be provided if features are provided.");
        TORCH_CHECK_VALUE(featureImages.value().dim() == 4 &&
                              featureImages.value().size(3) == features.value().rsize(1),
                          "Feature images must be of shape (batch_size, image_height, "
                          "image_width, num_features), but got ",
                          featureImages.value().sizes());
    }

    if (weightImages.has_value()) {
        TORCH_CHECK_VALUE(weightImages.value().dim() == 3 || weightImages.value().dim() == 4,
                          "Weight images must be of shape (batch_size, image_height, "
                          "image_width) or (batch_size, image_height, image_width, 1), but got ",
                          weightImages.value().sizes());
        TORCH_CHECK_VALUE(weightImages.value().size(0) == depthImages.size(0),
                          "Weight images must have the same batch size as depth images, but got "
                          "weight_images.size(0) = ",
                          weightImages.value().size(0),
                          " and depth_images.size(0) = ",
                          depthImages.size(0));
        TORCH_CHECK_VALUE(weightImages.value().size(1) == depthImages.size(1),
                          "Weight images must have the same height as depth images, but got "
                          "weight_images.size(1) = ",
                          weightImages.value().size(1),
                          " and depth_images.size(1) = ",
                          depthImages.size(1));
        TORCH_CHECK_VALUE(weightImages.value().size(2) == depthImages.size(2),
                          "Weight images must have the same width as depth images, but got "
                          "weight_images.size(2) = ",
                          weightImages.value().size(2),
                          " and depth_images.size(2) = ",
                          depthImages.size(2));
        if (weightImages.value().dim() == 4) {
            TORCH_CHECK_VALUE(weightImages.value().size(3) == 1,
                              "Weight images must have a last dimension of size 1, but got "
                              "weight_images.size(3) = ",
                              weightImages.value().size(3));
        }
    }
    // Step 1. Check that the batch size of the grid matches the batch size of the other tensors
    const int64_t batchSize = grid.batchSize();
    TORCH_CHECK(batchSize == tsdf.num_tensors(),
                "Batch size of grid (",
                batchSize,
                ") must match the number of tensors in tsdf (",
                tsdf.num_tensors(),
                ")");
    TORCH_CHECK(batchSize == weights.num_tensors(),
                "Batch size of grid (",
                batchSize,
                ") must match the number of tensors in weights (",
                weights.num_tensors(),
                ")");
    TORCH_CHECK(batchSize == depthImages.size(0),
                "Batch size of grid (",
                batchSize,
                ") must match the batch size (dim 0) of depth images (",
                depthImages.size(0),
                ")");
    TORCH_CHECK(batchSize == projectionMatrices.size(0),
                "Batch size of grid (",
                batchSize,
                ") must match the batch size (dim 0) of projection matrices (",
                projectionMatrices.size(0),
                ")");
    TORCH_CHECK(batchSize == camToWorldMatrices.size(0),
                "Batch size of grid (",
                batchSize,
                ") must match the batch size (dim 0) of camera-to-world matrices (",
                camToWorldMatrices.size(0),
                ")");
    if (features.has_value()) {
        TORCH_CHECK(batchSize == features.value().num_tensors(),
                    "Batch size of grid (",
                    batchSize,
                    ") must match the number of tensors in features (",
                    features.value().num_tensors(),
                    ")");
        TORCH_CHECK(featureImages.has_value(),
                    "Feature images must be provided if features are provided.");
        TORCH_CHECK(batchSize == featureImages.value().size(0),
                    "Batch size of grid (",
                    batchSize,
                    ") must match the batch size (dim 0) of feature images (",
                    featureImages.value().size(0),
                    ")");
    }

    // Step 2. Check that the feature depth images have the right shape
    //         and that the depth images are either 3D or 4D tensors with the last dimension being 1
    //         (which is the case for single-channel depth images)
    const int64_t imageHeight = depthImages.size(1);
    const int64_t imageWidth  = depthImages.size(2);
    if (depthImages.dim() == 4) {
        TORCH_CHECK(
            depthImages.dim() == 4 && depthImages.size(1) == imageHeight &&
                depthImages.size(2) == imageWidth && depthImages.size(3) == 1,
            "Depth images must be of shape (batch_size, image_height, image_width, 1), but got ",
            depthImages.sizes());
    } else {
        TORCH_CHECK(depthImages.dim() == 3 && depthImages.size(1) == imageHeight &&
                        depthImages.size(2) == imageWidth,
                    "Depth images must be of shape (batch_size, image_height, image_width), but "
                    "got ",
                    depthImages.sizes());
    }
    if (featureImages.has_value()) {
        const auto &featureImage = featureImages.value();
        TORCH_CHECK(featureImage.dim() == 4 && featureImage.size(0) == batchSize &&
                        featureImage.size(1) == imageHeight && featureImage.size(2) == imageWidth,
                    "Feature images must be of shape (batch_size, image_height, image_width, "
                    "num_features), but got ",
                    featureImage.sizes());
    }

    // Step 3. Check that the projection matrices and camera-to-world matrices
    //         have the right sizes
    TORCH_CHECK(projectionMatrices.size(1) == 3 && projectionMatrices.size(2) == 3,
                "Projection matrices must be of shape (batch_size, 3, 3), but got ",
                projectionMatrices.sizes());
    TORCH_CHECK(camToWorldMatrices.size(1) == 4 && camToWorldMatrices.size(2) == 4,
                "Camera-to-world matrices must be of shape (batch_size, 4, 4), but got ",
                camToWorldMatrices.sizes());

    // Step 4. Check that the TSDF and weights have the same number of elements and match the total
    // number of voxels in the grid
    TORCH_CHECK(tsdf.rsize(0) == grid.totalVoxels(),
                "tsdf must have the same number of elements as voxels in the input grid, but got ",
                "tsdf.rsize(0) = ",
                tsdf.rsize(0),
                " and grid.total_voxels = ",
                grid.totalVoxels());
    TORCH_CHECK(
        weights.rsize(0) == grid.totalVoxels(),
        "weights must have the same number of elements as voxels in the input grid, but got ",
        "tsdf.rsize(0) = ",
        weights.rsize(0),
        " and grid.total_voxels = ",
        grid.totalVoxels());
    if (features.has_value()) {
        TORCH_CHECK(features.value().rsize(0) == grid.totalVoxels(),
                    "features must have the same number of elements as voxels in the input grid, "
                    "but got ",
                    "features.rsize(0) = ",
                    features.value().rsize(0),
                    " and grid.total_voxels = ",
                    grid.totalVoxels());
    }
}

std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFImpl(const c10::intrusive_ptr<GridBatchImpl> grid,
                  const double truncationMargin,
                  const torch::Tensor &projectionMatrices,
                  const torch::Tensor &camToWorldMatrices,
                  const JaggedTensor &tsdf,
                  const JaggedTensor &weights,
                  const std::optional<JaggedTensor> &features,
                  const torch::Tensor &depthImages,
                  const std::optional<torch::Tensor> &featureImages,
                  const std::optional<torch::Tensor> &weightImages) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tsdf.jdata()));
    // Check that the input tensors have the correct dimensions and types
    checkInputTypes(tsdf,
                    weights,
                    features,
                    depthImages,
                    featureImages,
                    weightImages,
                    projectionMatrices,
                    camToWorldMatrices);
    checkInputSizes(*grid,
                    tsdf,
                    weights,
                    features,
                    depthImages,
                    featureImages,
                    weightImages,
                    projectionMatrices,
                    camToWorldMatrices);

    // If you passed in depth images with a channel dimension, squeeze it out
    const torch::Tensor squeezedDepthImages =
        depthImages.dim() == 4 ? depthImages.squeeze(-1) : depthImages;

    // Step 0: Inverse camera and projection matrices (using float32 precision for stability if
    // the inputs are float16). We need to compute the inverse of the camera-to-world matrices
    // and the projection matrices to unproject the depth maps to 3D points.
    const auto [projectionMats, invProjectionMats, camToWorldMats, worldToCamMats] =
        getCameraMatrices(projectionMatrices, camToWorldMatrices);

    // Step 1: Unproject the depth maps to 3D pointsauto
    const torch::Tensor unprojectedPoints = unprojectDepthMapToPoints(
        squeezedDepthImages, projectionMats, invProjectionMats, camToWorldMats);

    // Step 2: Build union grid grid from unprojected points and merge into with the old grid
    const auto unionGrid = buildPointGrid(truncationMargin, unprojectedPoints, *grid)->merge(grid);

    // Features are optional. If you don't pass them in, we will use placeholder values which are
    // just empty tensors.
    const auto [featuresValue, featureImagesValue] = [&]() {
        if (features.has_value()) {
            TORCH_CHECK(featureImages.has_value(),
                        "Feature images must be provided if features are provided.");
            return std::make_tuple(features.value(), featureImages.value());
        } else {
            TORCH_CHECK(!featureImages.has_value(),
                        "Feature images must not be provided if features are not provided.");
            const torch::TensorOptions opts              = squeezedDepthImages.options();
            const torch::Tensor placeholderFeatureImages = torch::empty({0, 0, 0, 0}, opts);
            const fvdb::JaggedTensor placeholderFeatures = torch::empty({0, 0}, opts);
            return std::make_tuple(placeholderFeatures, placeholderFeatureImages);
        }
    }();

    const auto weightImagesValue = weightImages.has_value()
                                       ? weightImages.value()
                                       : torch::empty({0, 0, 0}, squeezedDepthImages.options());
    const auto weightImagesSqueezed =
        weightImagesValue.dim() == 4 ? weightImagesValue.squeeze(-1) : weightImagesValue;
    // Step 3: Integrate weights, tsdf values, and feautures into the output tensor
    const auto [outTsdf, outWeights, outFeatures] = doIntegrate(truncationMargin,
                                                                squeezedDepthImages,
                                                                featureImagesValue,
                                                                weightImagesSqueezed,
                                                                projectionMats,
                                                                invProjectionMats,
                                                                camToWorldMats,
                                                                worldToCamMats,
                                                                *unionGrid,
                                                                *grid,
                                                                tsdf,
                                                                weights,
                                                                featuresValue);

    return {unionGrid, outTsdf, outWeights, outFeatures};
}

template <>
std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor, JaggedTensor>
dispatchIntegrateTSDF<torch::kCUDA>(const c10::intrusive_ptr<GridBatchImpl> grid,
                                    const double truncationMargin,
                                    const torch::Tensor &projectionMatrices,
                                    const torch::Tensor &camToWorldMatrices,
                                    const JaggedTensor &tsdf,
                                    const JaggedTensor &weights,
                                    const torch::Tensor &depthImages,
                                    const std::optional<torch::Tensor> &weightImages) {
    const auto [unionGrid, outTsdf, outWeights, outFeatures] = integrateTSDFImpl(grid,
                                                                                 truncationMargin,
                                                                                 projectionMatrices,
                                                                                 camToWorldMatrices,
                                                                                 tsdf,
                                                                                 weights,
                                                                                 std::nullopt,
                                                                                 depthImages,
                                                                                 std::nullopt,
                                                                                 weightImages);
    return {unionGrid, outTsdf, outWeights};
}

template <>
std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor, JaggedTensor>
dispatchIntegrateTSDF<torch::kCPU>(const c10::intrusive_ptr<GridBatchImpl> grid,
                                   const double truncationMargin,
                                   const torch::Tensor &projectionMatrices,
                                   const torch::Tensor &camToWorldMatrices,
                                   const JaggedTensor &tsdf,
                                   const JaggedTensor &weights,
                                   const torch::Tensor &depthImages,
                                   const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "TSDF integration not implemented on the CPU.");
}

template <>
std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor, JaggedTensor, JaggedTensor>
dispatchIntegrateTSDFWithFeatures<torch::kCUDA>(const c10::intrusive_ptr<GridBatchImpl> grid,
                                                const double truncationMargin,
                                                const torch::Tensor &projectionMatrices,
                                                const torch::Tensor &camToWorldMatrices,
                                                const JaggedTensor &tsdf,
                                                const JaggedTensor &features,
                                                const JaggedTensor &weights,
                                                const torch::Tensor &depthImages,
                                                const torch::Tensor &featureImages,
                                                const std::optional<torch::Tensor> &weightImages) {
    return integrateTSDFImpl(grid,
                             truncationMargin,
                             projectionMatrices,
                             camToWorldMatrices,
                             tsdf,
                             weights,
                             features,
                             depthImages,
                             featureImages,
                             weightImages);
}

template <>
std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor, JaggedTensor, JaggedTensor>
dispatchIntegrateTSDFWithFeatures<torch::kCPU>(const c10::intrusive_ptr<GridBatchImpl> grid,
                                               const double truncationMargin,
                                               const torch::Tensor &projectionMatrices,
                                               const torch::Tensor &camToWorldMatrices,
                                               const JaggedTensor &tsdf,
                                               const JaggedTensor &features,
                                               const JaggedTensor &weights,
                                               const torch::Tensor &depthImages,
                                               const torch::Tensor &featureImages,
                                               const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "TSDF integration not implemented on the CPU.");
}
} // namespace fvdb::detail::ops
