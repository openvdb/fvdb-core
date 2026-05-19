// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPointTruncationShell.h>
#include <fvdb/detail/ops/IntegrateTSDF.h>
#include <fvdb/detail/ops/PersistentTSDFState.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/math/Math.h>

#include <ATen/OpMathType.h>
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
                    const fvdb::BatchGridAccessor baseGridAcc,
                    const fvdb::BatchGridAccessor unionGridAcc,
                    const fvdb::JaggedRAcc64<ScalarDataType, 1> tsdfAcc,
                    const fvdb::JaggedRAcc64<ScalarDataType, 1> weightsAcc,
                    const fvdb::JaggedRAcc64<FeatureScalarDataType, 2> featuresAcc,
                    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
                    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
                    fvdb::TorchRAcc64<FeatureScalarDataType, 2> outFeaturesAcc) {
    using ScalarType        = at::opmath_type<ScalarDataType>;
    using FeatureScalarType = at::opmath_type<FeatureScalarDataType>;

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
            const ScalarType newTsdf     = (oldWeight * oldTsdf + pixelWeight * tsdf) / newWeight;
            outTsdfAcc[unionWriteOffset] = ScalarDataType(newTsdf);
            outWeightsAcc[unionWriteOffset] = ScalarDataType(newWeight);
            if (hasFeatures) {
                for (auto i = 0; i < outFeaturesAcc.size(1); ++i) {
                    const ScalarType pixelFeatureI = ScalarType(
                        featureImages[batchIdx][voxelPosScreenSpaceY][voxelPosScreenSpaceX][i]);
                    const ScalarType oldFeatureI =
                        voxelInBaseGrid ? ScalarType(featuresAcc.data()[baseGridOffset][i])
                                        : ScalarType(0);
                    outFeaturesAcc[unionWriteOffset][i] = FeatureScalarDataType(
                        (oldWeight * oldFeatureI + pixelWeight * pixelFeatureI) / newWeight);
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
    const c10::cuda::CUDAGuard device_guard(depthImages.device());

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

c10::intrusive_ptr<GridBatchData>
buildPointGrid(const double truncationMargin,
               const torch::Tensor &unprojectedPoints,
               const GridBatchData &grid) {
    // Pack the [B, N, 3] contiguous-per-batch unprojected-points
    // tensor into a JaggedTensor so we can hit the shared
    // buildPointTruncationShell primitive that the LiDAR integrator
    // also uses. Depth paths always produce equal-N per batch (N = H
    // * W of the input depth image), so the packing is trivial.
    std::vector<torch::Tensor> jaggedPointsList;
    jaggedPointsList.reserve(unprojectedPoints.size(0));
    for (int64_t i = 0; i < unprojectedPoints.size(0); ++i) {
        jaggedPointsList.push_back(unprojectedPoints[i]);
    }
    const JaggedTensor jaggedPoints(jaggedPointsList);

    return buildPointTruncationShell(jaggedPoints, grid, truncationMargin);
}

#define DISPATCH_FEATURE_TYPE(...)                                \
    if (hasFeatures && features.scalar_type() == torch::kUInt8) { \
        using feature_t = uint8_t;                                \
        __VA_ARGS__();                                            \
    } else {                                                      \
        using feature_t = scalar_t;                               \
        __VA_ARGS__();                                            \
    }

// Shell-filtered integrate: two kernels that together do the same
// work as `integrateTSDFKernel` but with a different decomposition.
//
//   1. `injectFromBaseKernel`: walks the BASE grid's leaves, looks each
//      active voxel up in the union grid, and copies old tsdf /
//      weight / features to its new position. Cheap per-thread work
//      (no projection, no depth lookup), and the launch size is
//      `baseGrid.totalLeaves() * 512` rather than `union.totalLeaves()
//      * 512` -- so on late frames where union has accumulated
//      carry-forward voxels that no longer correspond to any current
//      observation, we only pay for the ones that actually need
//      copying.
//
//   2. `integrateShellKernel`: walks the SHELL grid's leaves (i.e.
//      the per-frame truncation-band voxels produced by
//      `buildPointTruncationShell`), looks each active voxel up in
//      the union grid, projects + frustum-checks + applies the TSDF
//      blend. Reads the output buffer (already populated by inject
//      for voxels that were in base) as the "old" value, so
//      read-modify-write is stream-ordered correctly relative to
//      inject.
//
// For a scene that's saturated the union grid, late-frame shell size
// is much smaller than union size (typically ~25% at fine voxel sizes
// on a real RGB-D capture after ~100 frames), so this is a real
// asymptotic win over `integrateTSDFKernel`, which pays projection
// and visibility-check cost on every union voxel every frame.
//
// The legacy single-kernel path (`integrateTSDFKernel`) is still
// available as an ablation via `FVDB_FULL_UNION_INTEGRATE=1`.
template <typename ScalarDataType, typename FeatureScalarDataType = ScalarDataType>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
injectFromBaseKernel(
    const bool hasFeatures,
    const fvdb::BatchGridAccessor baseGridAcc,
    const fvdb::BatchGridAccessor unionGridAcc,
    const fvdb::JaggedRAcc64<ScalarDataType, 1> tsdfAcc,
    const fvdb::JaggedRAcc64<ScalarDataType, 1> weightsAcc,
    const fvdb::JaggedRAcc64<FeatureScalarDataType, 2> featuresAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
    fvdb::TorchRAcc64<FeatureScalarDataType, 2> outFeaturesAcc) {
    using GridT        = nanovdb::ValueOnIndex;
    using LeafNodeType = nanovdb::NanoGrid<GridT>::LeafNodeType;
    constexpr uint64_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<GridT>::LeafNodeType::NUM_VALUES;

    const auto problemSize = baseGridAcc.totalLeaves() * VOXELS_PER_LEAF;
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < problemSize; idx += blockDim.x * gridDim.x) {
        const int64_t cumBaseLeafIdx =
            static_cast<int64_t>(idx / VOXELS_PER_LEAF);
        const fvdb::JIdxType batchIdx =
            baseGridAcc.leafBatchIndex(cumBaseLeafIdx);
        const int64_t baseLeafIdx =
            cumBaseLeafIdx - baseGridAcc.leafOffset(batchIdx);
        const int64_t baseLeafVoxelIdx =
            static_cast<int64_t>(idx - cumBaseLeafIdx * VOXELS_PER_LEAF);

        const nanovdb::NanoGrid<GridT> *baseGrid =
            baseGridAcc.grid(batchIdx);
        const LeafNodeType &baseLeaf =
            baseGrid->tree().template getFirstNode<0>()[baseLeafIdx];
        const int64_t baseVoxelValue = static_cast<int64_t>(
            baseLeaf.getValue(baseLeafVoxelIdx)) - 1;
        if (baseVoxelValue < 0) continue;
        const int64_t baseOffset = baseGridAcc.voxelOffset(batchIdx) +
                                   baseVoxelValue;

        // Look up this ijk in the union grid. Base is guaranteed to
        // be a subset of union (union = merge(shell, base)), so the
        // lookup always succeeds and yields an active voxel.
        const nanovdb::Coord ijk =
            baseLeaf.offsetToGlobalCoord(baseLeafVoxelIdx);
        const nanovdb::NanoGrid<GridT> *unionGrid =
            unionGridAcc.grid(batchIdx);
        const auto unionAcc = unionGrid->getAccessor();
        const int64_t unionOffset = unionGridAcc.voxelOffset(batchIdx) +
            static_cast<int64_t>(unionAcc.getValue(ijk)) - 1;
        if (unionOffset < 0) continue; // defensive; shouldn't happen

        outTsdfAcc[unionOffset]    = tsdfAcc.data()[baseOffset];
        outWeightsAcc[unionOffset] = weightsAcc.data()[baseOffset];
        if (hasFeatures) {
            for (int64_t i = 0; i < outFeaturesAcc.size(1); ++i) {
                outFeaturesAcc[unionOffset][i] =
                    featuresAcc.data()[baseOffset][i];
            }
        }
    }
}

template <typename ScalarDataType, typename FeatureScalarDataType = ScalarDataType>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
integrateShellKernel(
    const ScalarDataType truncationMargin,
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
    const fvdb::BatchGridAccessor shellGridAcc,
    const fvdb::BatchGridAccessor unionGridAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outTsdfAcc,
    fvdb::TorchRAcc64<ScalarDataType, 1> outWeightsAcc,
    fvdb::TorchRAcc64<FeatureScalarDataType, 2> outFeaturesAcc) {
    using ScalarType        = at::opmath_type<ScalarDataType>;
    using FeatureScalarType = at::opmath_type<FeatureScalarDataType>;
    using GridT        = nanovdb::ValueOnIndex;
    using LeafNodeType = nanovdb::NanoGrid<GridT>::LeafNodeType;
    using Vec3T        = nanovdb::math::Vec3<ScalarType>;
    using Vec4T        = nanovdb::math::Vec4<ScalarType>;
    using Mat3T        = nanovdb::math::Mat3<ScalarType>;
    using Mat4T        = nanovdb::math::Mat4<ScalarType>;
    constexpr uint64_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<GridT>::LeafNodeType::NUM_VALUES;

    const auto batchSize = projMats.size(0);

    // Identical shared-memory layout to `integrateTSDFKernel` so the
    // host-side shared-size calculation can be shared.
    extern __shared__ uint8_t sharedData[];
    Mat3T *sharedProjMats       = reinterpret_cast<Mat3T *>(sharedData);
    Mat4T *sharedWorldToCamMats = reinterpret_cast<Mat4T *>(
        sharedData + batchSize * sizeof(Mat3T));
    Mat3T *sharedInvProjMats =
        reinterpret_cast<Mat3T *>(sharedData +
                                  batchSize * (sizeof(Mat3T) + sizeof(Mat4T)));
    Mat4T *sharedCamToWorldMats = reinterpret_cast<Mat4T *>(
        sharedData + batchSize * (sizeof(Mat3T) + sizeof(Mat4T) +
                                  sizeof(Mat3T)));

    const auto sharedMat3x3NumElements = batchSize * 3 * 3;
    const auto sharedMat4x4NumElements = batchSize * 4 * 4;
    if (threadIdx.x < sharedMat3x3NumElements) {
        const auto batchIdx = threadIdx.x / 9;
        const auto rowIdx   = (threadIdx.x % 9) / 3;
        const auto colIdx   = threadIdx.x % 3;
        sharedProjMats[batchIdx][rowIdx][colIdx] =
            ScalarType(projMats[batchIdx][rowIdx][colIdx]);
    } else if (threadIdx.x < sharedMat3x3NumElements + sharedMat4x4NumElements) {
        const auto baseIdx  = threadIdx.x - sharedMat3x3NumElements;
        const auto batchIdx = baseIdx / 16;
        const auto rowIdx   = (baseIdx % 16) / 4;
        const auto colIdx   = baseIdx % 4;
        sharedWorldToCamMats[batchIdx][rowIdx][colIdx] =
            ScalarType(worldToCamMats[batchIdx][rowIdx][colIdx]);
    } else if (threadIdx.x <
               2 * sharedMat3x3NumElements + sharedMat4x4NumElements) {
        const auto baseIdx  = threadIdx.x - sharedMat3x3NumElements -
                              sharedMat4x4NumElements;
        const auto batchIdx = baseIdx / 9;
        const auto rowIdx   = (baseIdx % 9) / 3;
        const auto colIdx   = baseIdx % 3;
        sharedInvProjMats[batchIdx][rowIdx][colIdx] =
            ScalarType(invProjMats[batchIdx][rowIdx][colIdx]);
    }
    __syncthreads();

    // Parallelise over the SHELL's voxels (not the full union). The
    // kernel loads matrices once per block and then only threads whose
    // idx falls inside the shell's 512 * numLeaves range do real
    // work; any thread whose idx is past the shell's total voxel
    // count just exits.
    const auto problemSize = shellGridAcc.totalLeaves() * VOXELS_PER_LEAF;
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < problemSize; idx += blockDim.x * gridDim.x) {
        const int64_t cumShellLeafIdx =
            static_cast<int64_t>(idx / VOXELS_PER_LEAF);
        const fvdb::JIdxType batchIdx =
            shellGridAcc.leafBatchIndex(cumShellLeafIdx);
        const int64_t shellLeafIdx =
            cumShellLeafIdx - shellGridAcc.leafOffset(batchIdx);
        const int64_t shellLeafVoxelIdx = static_cast<int64_t>(
            idx - cumShellLeafIdx * VOXELS_PER_LEAF);

        const nanovdb::NanoGrid<GridT> *shellGrid =
            shellGridAcc.grid(batchIdx);
        const LeafNodeType &shellLeaf =
            shellGrid->tree().template getFirstNode<0>()[shellLeafIdx];
        // Shell leaves can have inactive slots (nanoVDB leaf nodes
        // are fixed 8^3, but only some slots are active).
        const int64_t shellVoxelValue = static_cast<int64_t>(
            shellLeaf.getValue(shellLeafVoxelIdx)) - 1;
        if (shellVoxelValue < 0) continue;

        const nanovdb::Coord ijk =
            shellLeaf.offsetToGlobalCoord(shellLeafVoxelIdx);
        const nanovdb::NanoGrid<GridT> *unionGrid =
            unionGridAcc.grid(batchIdx);
        const auto unionAcc = unionGrid->getAccessor();
        const int64_t unionOffset = unionGridAcc.voxelOffset(batchIdx) +
            static_cast<int64_t>(unionAcc.getValue(ijk)) - 1;
        if (unionOffset < 0) continue;

        // Project voxel to screen, frustum-check, apply TSDF blend.
        const Vec3T voxelWorldPos = unionGridAcc.primalTransform(batchIdx)
            .applyInv<ScalarType>(
                ScalarType(ijk[0]), ScalarType(ijk[1]), ScalarType(ijk[2]));
        const Vec4T voxelWorldPosHomogeneous = {
            voxelWorldPos[0], voxelWorldPos[1], voxelWorldPos[2],
            ScalarType(1.0)};
        const Vec4T voxelPosCamSpace =
            sharedWorldToCamMats[batchIdx] * voxelWorldPosHomogeneous;
        const Vec3T voxelPosCamSpace3d = {
            voxelPosCamSpace[0] / voxelPosCamSpace[3],
            voxelPosCamSpace[1] / voxelPosCamSpace[3],
            voxelPosCamSpace[2] / voxelPosCamSpace[3]};
        const Vec3T voxelPosProjSpace =
            sharedProjMats[batchIdx] * voxelPosCamSpace3d;
        const Vec3T voxelPosScreenSpace = {
            voxelPosProjSpace[0] / voxelPosProjSpace[2],
            voxelPosProjSpace[1] / voxelPosProjSpace[2],
            ScalarType(1.0)};
        const int64_t voxelPosScreenSpaceX =
            int64_t(voxelPosScreenSpace[0]);
        const int64_t voxelPosScreenSpaceY =
            int64_t(voxelPosScreenSpace[1]);

        const bool voxelIsVisible =
            (voxelPosScreenSpaceX >= 0 && voxelPosScreenSpaceX < imageWidth &&
             voxelPosScreenSpaceY >= 0 && voxelPosScreenSpaceY < imageHeight &&
             voxelPosCamSpace3d[2] > 0.0f);
        // Not visible -> the inject pass has already carried the old
        // value forward (or left the slot at zero for shell-only
        // voxels, which is the correct initial state).
        if (!voxelIsVisible) continue;

        const ScalarType pixelDepth = ScalarType(
            depthImages[batchIdx][voxelPosScreenSpaceY][voxelPosScreenSpaceX]);
        const ScalarType zDiff = pixelDepth - voxelPosCamSpace3d[2];
        if (zDiff <= -ScalarType(truncationMargin)) continue;

        const ScalarType pixelWeight = [&]() {
            if (hasWeights) {
                return ScalarType(weightImages[batchIdx][voxelPosScreenSpaceY]
                                             [voxelPosScreenSpaceX]);
            } else {
                return ScalarType{1};
            }
        }();
        if (pixelWeight <= ScalarType(0)) continue;

        const ScalarType tsdf = nanovdb::math::Min(
            ScalarType(1), zDiff / ScalarType(truncationMargin));
        // Read-modify-write: the old value was either written by the
        // inject pass (for voxels in base) or is zero (for shell-only
        // voxels, torch::zeros initialisation). Stream ordering
        // guarantees inject completes before this kernel launches.
        const ScalarType oldWeight   = ScalarType(outWeightsAcc[unionOffset]);
        const ScalarType oldTsdf     = ScalarType(outTsdfAcc[unionOffset]);
        const ScalarType newWeight   = oldWeight + pixelWeight;
        const ScalarType newTsdf     =
            (oldWeight * oldTsdf + pixelWeight * tsdf) / newWeight;
        outTsdfAcc[unionOffset]    = ScalarDataType(newTsdf);
        outWeightsAcc[unionOffset] = ScalarDataType(newWeight);
        if (hasFeatures) {
            for (int64_t i = 0; i < outFeaturesAcc.size(1); ++i) {
                const ScalarType pixelFeatureI = ScalarType(
                    featureImages[batchIdx][voxelPosScreenSpaceY]
                                [voxelPosScreenSpaceX][i]);
                const ScalarType oldFeatureI =
                    ScalarType(outFeaturesAcc[unionOffset][i]);
                outFeaturesAcc[unionOffset][i] = FeatureScalarDataType(
                    (oldWeight * oldFeatureI + pixelWeight * pixelFeatureI) /
                    newWeight);
            }
        }
    }
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
            const GridBatchData &unionGrid,
            const GridBatchData &baseGrid,
            const GridBatchData &shellGrid,
            const JaggedTensor &tsdf,
            const JaggedTensor &weights,
            const JaggedTensor &features) {
    const c10::cuda::CUDAGuard device_guard(tsdf.device());

    const int64_t batchSize      = depthImages.size(0);
    const int64_t imageHeight    = depthImages.size(1);
    const int64_t imageWidth     = depthImages.size(2);
    const int64_t totalOutVoxels = unionGrid.totalVoxels();
    const int64_t featureDim     = features.rsize(-1);
    const bool hasFeatures       = featureDim > 0;
    const bool hasWeights        = weightImages.size(0) > 0;

    // Output tensors are zero-initialised. The shell-filtered integrate
    // kernel has three "continue" branches (voxel not visible, zDiff
    // behind surface, pixelWeight == 0) where it silently leaves the
    // output slot unwritten; for shell voxels NOT in the base grid we
    // need that slot to read as 0 rather than as uninitialised memory,
    // otherwise downstream consumers see |tsdf| > 1 garbage.
    torch::Tensor outWeights =
        torch::zeros({totalOutVoxels}, weights.jdata().options());
    torch::Tensor outTsdf =
        torch::zeros({totalOutVoxels}, tsdf.jdata().options());
    torch::Tensor outFeatures =
        torch::zeros({totalOutVoxels, featureDim},
                     features.jdata().options());

    // `FVDB_FULL_UNION_INTEGRATE=1` opts into the legacy single-kernel
    // path that walks every union voxel and does either copy-forward
    // or integrate per-thread. Default is the two-pass
    // inject + shell-filtered integrate path above.
    const bool force_legacy_integrate = [&]() {
        const char *env = std::getenv("FVDB_FULL_UNION_INTEGRATE");
        return env != nullptr && env[0] == '1';
    }();

    if (force_legacy_integrate) {
        AT_DISPATCH_V2(
            tsdf.scalar_type(),
            "integrateTSDFKernel",
            AT_WRAP([&]() {
                using shared_scalar_t              = at::opmath_type<scalar_t>;
                using SharedMat3T                  = nanovdb::math::Mat3<shared_scalar_t>;
                using SharedMat4T                  = nanovdb::math::Mat4<shared_scalar_t>;
                constexpr uint64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
                const auto numUnionLeaves          = unionGrid.totalLeaves();
                const auto numSharedScalars        = 2 * batchSize * 3 * 3 + 2 * batchSize * 4 * 4;
                const auto problemSize =
                    std::max(numUnionLeaves * VOXELS_PER_LEAF, uint64_t(numSharedScalars));
                const auto sharedMemSize =
                    2 * batchSize * sizeof(SharedMat3T) + 2 * batchSize * sizeof(SharedMat4T);
                const auto numBlocks = GET_BLOCKS(problemSize, DEFAULT_BLOCK_DIM);

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

    // Default: two-pass shell-filtered integrate.
    AT_DISPATCH_V2(
        tsdf.scalar_type(),
        "integrateTSDFShellFiltered",
        AT_WRAP([&]() {
            using shared_scalar_t              = at::opmath_type<scalar_t>;
            using SharedMat3T                  = nanovdb::math::Mat3<shared_scalar_t>;
            using SharedMat4T                  = nanovdb::math::Mat4<shared_scalar_t>;
            constexpr uint64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
            const auto sharedMemSize =
                2 * batchSize * sizeof(SharedMat3T) + 2 * batchSize * sizeof(SharedMat4T);

            const auto dtype                = tsdf.scalar_type();
            const auto projMatsCasted       = projectionMatrices.to(dtype);
            const auto invProjMatsCasted    = invProjectionMatrices.to(dtype);
            const auto camToWorldMatsCasted = camToWorldMatrices.to(dtype);
            const auto worldToCamMatsCasted = worldToCamMatrices.to(dtype);

            at::cuda::CUDAStream stream =
                at::cuda::getCurrentCUDAStream(tsdf.device().index());

            DISPATCH_FEATURE_TYPE([&]() {
                // Pass 1: inject old tsdf / weight / features from base
                // grid to their new positions in union. Skipped when
                // baseGrid is empty (first frame) since there's nothing
                // to carry forward.
                const auto numBaseLeaves = baseGrid.totalLeaves();
                if (numBaseLeaves > 0) {
                    const auto injectProblemSize =
                        numBaseLeaves * VOXELS_PER_LEAF;
                    const auto injectBlocks =
                        GET_BLOCKS(injectProblemSize, DEFAULT_BLOCK_DIM);
                    injectFromBaseKernel<<<injectBlocks, DEFAULT_BLOCK_DIM,
                                          0, stream>>>(
                        hasFeatures,
                        baseGrid.deviceAccessor(),
                        unionGrid.deviceAccessor(),
                        tsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        weights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        features.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>(),
                        outTsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        outWeights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        outFeatures.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>());
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }

                // Pass 2: apply this frame's depth observations to the
                // shell's voxels. Stream ordering guarantees the inject
                // above has completed before we enter the read-modify-
                // write below.
                const auto numShellLeaves = shellGrid.totalLeaves();
                const auto numSharedScalars =
                    2 * batchSize * 3 * 3 + 2 * batchSize * 4 * 4;
                const auto integrateProblemSize = std::max(
                    numShellLeaves * VOXELS_PER_LEAF,
                    uint64_t(numSharedScalars));
                const auto integrateBlocks =
                    GET_BLOCKS(integrateProblemSize, DEFAULT_BLOCK_DIM);

                if (cudaFuncSetAttribute(
                        integrateShellKernel<scalar_t, feature_t>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        sharedMemSize) != cudaSuccess) {
                    AT_ERROR("Failed to set maximum shared memory size (requested ",
                             sharedMemSize, " bytes), try lowering tile_size.");
                }

                integrateShellKernel<<<integrateBlocks, DEFAULT_BLOCK_DIM,
                                       sharedMemSize, stream>>>(
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
                    shellGrid.deviceAccessor(),
                    unionGrid.deviceAccessor(),
                    outTsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    outWeights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    outFeatures.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return {unionGrid.jaggedTensor(outTsdf),
            unionGrid.jaggedTensor(outWeights),
            unionGrid.jaggedTensor(outFeatures)};
}

/// @brief Run `integrateShellKernel` in place against caller-owned
///        sidecar tensors (tsdf / weights / features) whose layout
///        already matches `liveGrid`. This is the kernel-dispatch
///        path used by `integrateTSDFBatchImpl`:
///        `PersistentTSDFState::growFromGrid` has already reallocated
///        + injected the sidecars (or no-op'd on overlap-only shell),
///        so the kernel only needs to read-modify-write the shell's
///        voxels. No alloc, no inject-pass.
///
/// Semantics: identical to `doIntegrate(..., unionGrid=liveGrid,
/// baseGrid=<any-empty-grid>, shellGrid=shellGrid, ...)` except we
/// skip the zero-init + injectFromBaseKernel path since those are
/// no-ops when (a) the output tensors already hold the current
/// accumulator values (post-grow), and (b) the kernel only writes
/// to shell voxels. The legacy `FVDB_FULL_UNION_INTEGRATE=1`
/// ablation is unreachable here -- that path is only exercised via
/// `integrateTSDFImpl` single-frame.
void
doIntegrateShellInPlace(const float truncationMargin,
                        const torch::Tensor &depthImages,
                        const torch::Tensor &featureImages,
                        const torch::Tensor &weightImages,
                        const torch::Tensor &projectionMatrices,
                        const torch::Tensor &invProjectionMatrices,
                        const torch::Tensor &camToWorldMatrices,
                        const torch::Tensor &worldToCamMatrices,
                        const GridBatchData &liveGrid,
                        const GridBatchData &shellGrid,
                        torch::Tensor &tsdf,
                        torch::Tensor &weights,
                        torch::Tensor &features) {
    const c10::cuda::CUDAGuard device_guard(tsdf.device());

    const int64_t batchSize   = depthImages.size(0);
    const int64_t imageHeight = depthImages.size(1);
    const int64_t imageWidth  = depthImages.size(2);
    const int64_t featureDim  = features.size(-1);
    const bool hasFeatures    = featureDim > 0;
    const bool hasWeights     = weightImages.size(0) > 0;

    AT_DISPATCH_V2(
        tsdf.scalar_type(),
        "integrateTSDFShellInPlace",
        AT_WRAP([&]() {
            using shared_scalar_t              = at::opmath_type<scalar_t>;
            using SharedMat3T                  = nanovdb::math::Mat3<shared_scalar_t>;
            using SharedMat4T                  = nanovdb::math::Mat4<shared_scalar_t>;
            constexpr uint64_t VOXELS_PER_LEAF = nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
            const auto sharedMemSize =
                2 * batchSize * sizeof(SharedMat3T) + 2 * batchSize * sizeof(SharedMat4T);

            const auto dtype                = tsdf.scalar_type();
            const auto projMatsCasted       = projectionMatrices.to(dtype);
            const auto invProjMatsCasted    = invProjectionMatrices.to(dtype);
            const auto camToWorldMatsCasted = camToWorldMatrices.to(dtype);
            const auto worldToCamMatsCasted = worldToCamMatrices.to(dtype);

            at::cuda::CUDAStream stream =
                at::cuda::getCurrentCUDAStream(tsdf.device().index());

            DISPATCH_FEATURE_TYPE([&]() {
                const auto numShellLeaves = shellGrid.totalLeaves();
                const auto numSharedScalars =
                    2 * batchSize * 3 * 3 + 2 * batchSize * 4 * 4;
                const auto integrateProblemSize = std::max(
                    numShellLeaves * VOXELS_PER_LEAF,
                    uint64_t(numSharedScalars));
                const auto integrateBlocks =
                    GET_BLOCKS(integrateProblemSize, DEFAULT_BLOCK_DIM);

                if (cudaFuncSetAttribute(
                        integrateShellKernel<scalar_t, feature_t>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        sharedMemSize) != cudaSuccess) {
                    AT_ERROR("Failed to set maximum shared memory size (requested ",
                             sharedMemSize, " bytes), try lowering tile_size.");
                }

                // `integrateShellKernel` reads-modifies-writes
                // `outTsdf / outWeights / outFeatures`, and here we
                // pass the state tensors as both input and output.
                // That's correct: for each shell voxel the kernel
                // reads the current (accumulated) (tsdf, weight)
                // value, computes the new weighted average with this
                // frame's depth observation, and writes the result
                // back -- a classic in-place running-mean update.
                integrateShellKernel<<<integrateBlocks, DEFAULT_BLOCK_DIM,
                                       sharedMemSize, stream>>>(
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
                    shellGrid.deviceAccessor(),
                    liveGrid.deviceAccessor(),
                    tsdf.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    weights.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    features.packed_accessor64<feature_t, 2, torch::RestrictPtrTraits>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
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
checkInputSizes(const GridBatchData &grid,
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

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFImpl(const c10::intrusive_ptr<GridBatchData> grid,
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

    // `FVDB_TSDF_PHASE_PROFILE=1` enables per-step CUDA-event timing of
    // the integrate pipeline. Rows are printed to stderr as a CSV so
    // they can be aggregated across frames by a wrapping script:
    //   [fvdb/tsdf_phase] unproject=X ms  shell=Y ms  merge=Z ms
    //   integrate=W ms  total=T ms  old_voxels=K  new_voxels=M
    // This is invaluable for decomposing the fvdb_leaf vs fvdb_voxel
    // ~15x slowdown (see session journal entry on voxel-shell tuning).
    const bool phaseProfile =
        std::getenv("FVDB_TSDF_PHASE_PROFILE") != nullptr;
    cudaEvent_t evA{}, evB{}, evC{}, evD{}, evE{};
    auto phaseMark = [&](cudaEvent_t &ev) {
        if (phaseProfile) {
            cudaEventCreate(&ev);
            cudaEventRecord(ev);
        }
    };
    phaseMark(evA);

    // If you passed in depth images with a channel dimension, squeeze it out
    const torch::Tensor squeezedDepthImages =
        depthImages.dim() == 4 ? depthImages.squeeze(-1) : depthImages;

    // Step 0: Inverse camera and projection matrices (using float32 precision for stability if
    // the inputs are float16). We need to compute the inverse of the camera-to-world matrices
    // and the projection matrices to unproject the depth maps to 3D points.
    const auto [projectionMats, invProjectionMats, camToWorldMats, worldToCamMats] =
        getCameraMatrices(projectionMatrices, camToWorldMatrices);

    // Step 1: Unproject the depth maps to 3D points.
    //
    // For fp16 inputs we promote the unprojected point cloud to fp32
    // before handing it to `buildPointGrid` because `pointsToIjk`
    // quantises in the caller's dtype -- and fp16 at room-scale
    // magnitudes (5-15 m) has ~0.3-1 mm ULP, which at 5 mm voxels is a
    // nontrivial fraction of a voxel. In practice this was producing
    // 5-20% *more* active voxels for fp16 workloads than fp32
    // (different boundary points rounded to different voxels),
    // partially cancelling the fp16 sidecar memory win. Promoting the
    // ~H*W points to fp32 for the one-shot quantisation adds a few MB
    // of transient memory and no measurable wall time; keeping the
    // sidecar tensors (tsdf / weight / features) in fp16 retains the
    // ~2x GB savings that motivated the fp16 path in the first place.
    const torch::Tensor unprojectedPointsNative = unprojectDepthMapToPoints(
        squeezedDepthImages, projectionMats, invProjectionMats, camToWorldMats);
    const torch::Tensor unprojectedPoints =
        unprojectedPointsNative.scalar_type() == torch::kHalf
            ? unprojectedPointsNative.to(torch::kFloat32)
            : unprojectedPointsNative;
    phaseMark(evB);

    // Step 2: Build union grid grid from unprojected points and merge into with the old grid
    const auto pointGrid = buildPointGrid(truncationMargin, unprojectedPoints, *grid);
    phaseMark(evC);
    const auto unionGrid = ops::mergeGrids(*pointGrid, *grid);
    phaseMark(evD);

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
    // Step 3: Integrate weights, tsdf values, and features into the
    // output tensor. We pass three grids:
    //   - unionGrid: where output sidecars are indexed (size = total
    //     active voxels after this frame's shell has been merged in).
    //   - grid (base): the old accumulated grid, used for
    //     carrying-forward previously-integrated tsdf/weight.
    //   - pointGrid (shell): this frame's truncation-band voxels, which
    //     is the set the integrate kernel actually needs to update
    //     (everything else just needs a copy-forward).
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
                                                                *pointGrid,
                                                                tsdf,
                                                                weights,
                                                                featuresValue);
    phaseMark(evE);
    if (phaseProfile) {
        cudaEventSynchronize(evE);
        float t_unproj = 0.f, t_shell = 0.f, t_merge = 0.f, t_integ = 0.f;
        cudaEventElapsedTime(&t_unproj, evA, evB);
        cudaEventElapsedTime(&t_shell, evB, evC);
        cudaEventElapsedTime(&t_merge, evC, evD);
        cudaEventElapsedTime(&t_integ, evD, evE);
        std::fprintf(
            stderr,
            "[fvdb/tsdf_phase] unproject=%.3f ms  shell=%.3f ms  "
            "merge=%.3f ms  integrate=%.3f ms  total=%.3f ms  "
            "old_vox=%lld  union_vox=%lld  point_vox=%lld\n",
            t_unproj, t_shell, t_merge, t_integ,
            t_unproj + t_shell + t_merge + t_integ,
            (long long)grid->totalVoxels(),
            (long long)unionGrid->totalVoxels(),
            (long long)pointGrid->totalVoxels());
        cudaEventDestroy(evA);
        cudaEventDestroy(evB);
        cudaEventDestroy(evC);
        cudaEventDestroy(evD);
        cudaEventDestroy(evE);
    }

    return {unionGrid, outTsdf, outWeights, outFeatures};
}

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDF(const c10::intrusive_ptr<GridBatchData> grid,
              const double truncationMargin,
              const torch::Tensor &projectionMatrices,
              const torch::Tensor &camToWorldMatrices,
              const JaggedTensor &tsdf,
              const JaggedTensor &weights,
              const torch::Tensor &depthImages,
              const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(grid->device().is_cuda(),
                                "TSDF integration not implemented on the CPU.");
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

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                          const double truncationMargin,
                          const torch::Tensor &projectionMatrices,
                          const torch::Tensor &camToWorldMatrices,
                          const JaggedTensor &tsdf,
                          const JaggedTensor &features,
                          const JaggedTensor &weights,
                          const torch::Tensor &depthImages,
                          const torch::Tensor &featureImages,
                          const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(grid->device().is_cuda(),
                                "TSDF integration not implemented on the CPU.");
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

// -------------------------------------------------------------------------
// Batched depth-image TSDF integration.
//
// Builds the full union-grid topology ONCE over all N frames, then runs
// N sequential calls to the existing `doIntegrate` kernel against that
// fixed topology. Semantically equivalent to calling `integrateTSDF` N
// times (verified bit-identically in the unit test).
//
// The per-frame path pays O(pixels + unionVoxels) of topology rebuild
// every call; the batched path does one topology build over
// N * pixels points, then N kernel launches — so the perf win is
// (N - 1) * (topology_build_ms + merge_ms) per N-frame batch.
// -------------------------------------------------------------------------

namespace {

// Implementation note: an alternative one-shot topology build was
// considered for the batched path -- unproject ALL N frames at once
// and build a single union grid -- but it allocates an
// O(N * pixels) point buffer that is dominated by free-space rays
// at typical fine voxel sizes and high frame counts, and pays a
// union-grid-sized integrate loop on every frame. The incremental
// per-frame loop used by `integrateTSDFBatchImpl` below has the
// same final topology while keeping intermediate working-set size
// bounded.

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFBatchImpl(const c10::intrusive_ptr<GridBatchData> grid,
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

    TORCH_CHECK_VALUE(grid->batchSize() == 1,
                      "integrateTSDFBatch requires a single-scene grid "
                      "(batchSize = 1); got batchSize = ",
                      grid->batchSize(),
                      ". The N dimension is carried on depthImages.size(0).");

    // Squeeze the optional trailing channel dim on depth / weight images
    // so downstream code sees a uniform [N, H, W] shape.
    const torch::Tensor depthImagesSqueezed =
        depthImages.dim() == 4 ? depthImages.squeeze(-1) : depthImages;
    const int64_t N = depthImagesSqueezed.size(0);
    TORCH_CHECK_VALUE(N > 0, "depthImages must have at least one frame");
    TORCH_CHECK_VALUE(projectionMatrices.size(0) == N,
                      "projectionMatrices frame count (",
                      projectionMatrices.size(0),
                      ") must equal depth-image frame count (", N, ")");
    TORCH_CHECK_VALUE(camToWorldMatrices.size(0) == N,
                      "camToWorldMatrices frame count (",
                      camToWorldMatrices.size(0),
                      ") must equal depth-image frame count (", N, ")");

    // --- Incremental per-frame pipeline ------------------------------
    //
    // The batched path grows topology one frame at a time, looping the
    // existing single-frame `integrateTSDFImpl`. This is asymptotically
    // O(N * frustum_voxels_per_frame) rather than the O(N^2) cost of
    // building a static union over all frames up-front (each iteration
    // of the union-then-integrate variant runs the TSDF kernel over
    // every voxel in the union, the vast majority of which are not
    // in-view for any given frame).
    //
    // It also fixes a mesh under-coverage bug the union-then-integrate
    // variant exhibited: voxels in the union that were never visible
    // in any frame stayed at weight=0 and got pruned out by the mesh
    // extractor. In the incremental path, a voxel only enters the
    // grid once some frame's truncation shell has touched it, so by
    // construction every active voxel has at least one real TSDF
    // update.
    const bool profile_batch =
        std::getenv("FVDB_TSDF_BATCH_PROFILE") != nullptr;
    cudaEvent_t evStart{}, evEnd{};
    if (profile_batch) {
        cudaEventCreate(&evStart);
        cudaEventCreate(&evEnd);
        cudaEventRecord(evStart);
    }

    // Feature / weight-image validation (same convention as
    // `integrateTSDFImpl` so per-frame slices pass its checks).
    const bool hasFeatureImages = features.has_value();
    if (hasFeatureImages) {
        TORCH_CHECK(featureImages.has_value(),
                    "Feature images must be provided if features are provided.");
        TORCH_CHECK_VALUE(featureImages.value().size(0) == N,
                          "featureImages frame count (",
                          featureImages.value().size(0),
                          ") must equal depth-image frame count (", N, ")");
    } else {
        TORCH_CHECK(!featureImages.has_value(),
                    "Feature images must not be provided if features are not provided.");
    }
    const bool hasPerFrameWeightImages =
        weightImages.has_value() && weightImages.value().size(0) == N;
    if (weightImages.has_value()) {
        TORCH_CHECK_VALUE(hasPerFrameWeightImages,
                          "weightImages frame count (",
                          weightImages.value().size(0),
                          ") must equal depth-image frame count (", N, ")");
    }

    // Own the accumulator as a `PersistentTSDFState` so the per-frame
    // "grow topology + carry sidecar values forward" step becomes a
    // single `growFromGrid` call that fast-paths to a no-op when the
    // frame's truncation shell is a subset of the current live grid.
    // On bounded-scene trajectories the shell stops introducing new
    // voxels after some warm-up, so post-converge frames skip both
    // the sidecar realloc and the inject-from-base pass entirely,
    // leaving only the shell integrate kernel to run.
    //
    // Equivalence with the pre-refactor path:
    //   - `integrateTSDFImpl` did `zeros(union) + injectFromBase +
    //     integrateShellKernel` each frame. The first two steps are
    //     exactly what `PersistentTSDFState::growFromGrid` performs
    //     (fresh zeros sized to union, then `ops::inject` from live
    //     grid's sidecars). Replacing them with one `growFromGrid`
    //     call is semantically identical -- bit-identical mesh /
    //     tsdf / weight outputs are pinned by
    //     `test_integrate_tsdf_frames_matches_sequential`
    //     (atol=rtol=0).
    //   - The integrate kernel call then becomes
    //     `doIntegrateShellInPlace` on the state's tensors (skips
    //     the alloc + inject since growFromGrid already did it).
    //   - `FVDB_FULL_UNION_INTEGRATE=1` is an opt-in legacy knob
    //     only exercised by the single-frame `integrateTSDFImpl`
    //     path; batched always uses shell-filtered integrate.
    auto featuresStart = hasFeatureImages
                             ? std::make_optional(features.value().jdata())
                             : std::nullopt;
    PersistentTSDFState state(
        grid, tsdf.jdata(), weights.jdata(), featuresStart);

    for (int64_t i = 0; i < N; ++i) {
        const torch::Tensor depth_i =
            depthImagesSqueezed.narrow(0, i, 1).contiguous();
        const torch::Tensor proj_i =
            projectionMatrices.narrow(0, i, 1).contiguous();
        const torch::Tensor c2w_i =
            camToWorldMatrices.narrow(0, i, 1).contiguous();
        const torch::Tensor featImg_i =
            hasFeatureImages
                ? featureImages.value().narrow(0, i, 1).contiguous()
                : torch::empty({0, 0, 0, 0}, depth_i.options());
        const torch::Tensor wImg_i =
            hasPerFrameWeightImages
                ? weightImages.value().narrow(0, i, 1).contiguous()
                : torch::empty({0, 0, 0}, depth_i.options());

        // Rebuild camera matrices for this frame (same helper the
        // single-frame impl uses).
        const auto [projMats, invProjMats, c2wMats, w2cMats] =
            getCameraMatrices(proj_i, c2w_i);

        // Squeeze optional channel dim to keep the single-frame
        // conventions uniform.
        const torch::Tensor depth_i_sq =
            depth_i.dim() == 4 ? depth_i.squeeze(-1) : depth_i;
        const torch::Tensor wImg_i_sq =
            wImg_i.dim() == 4 ? wImg_i.squeeze(-1) : wImg_i;

        // Unproject + build this frame's shell (identical to the
        // single-frame path; see `integrateTSDFImpl` for fp16
        // promote-for-quantise note).
        const torch::Tensor unprojectedNative = unprojectDepthMapToPoints(
            depth_i_sq, projMats, invProjMats, c2wMats);
        const torch::Tensor unprojected =
            unprojectedNative.scalar_type() == torch::kHalf
                ? unprojectedNative.to(torch::kFloat32)
                : unprojectedNative;
        const auto pointGrid = buildPointGrid(
            truncationMargin, unprojected, state.grid());

        // Grow the persistent state: maybe-alloc sidecars, maybe-
        // inject from old layout to new, update grid pointer.
        // No-op when `pointGrid` is a subset of `state.grid()`.
        state.growFromGrid(*pointGrid);

        // Placeholder features tensor when features are disabled --
        // the integrate kernel still takes the argument via its
        // `hasFeatures` flag. Keep the size-matching invariant.
        torch::Tensor featuresRef = state.features();

        doIntegrateShellInPlace(
            truncationMargin,
            depth_i_sq,
            featImg_i,
            wImg_i_sq,
            projMats, invProjMats, c2wMats, w2cMats,
            state.grid(),
            *pointGrid,
            state.tsdf(), state.weights(), featuresRef);
    }

    c10::intrusive_ptr<GridBatchData> accumGrid = state.gridPtr();
    JaggedTensor accumTsdf     = state.tsdfJagged();
    JaggedTensor accumWeights  = state.weightsJagged();
    JaggedTensor accumFeatures = hasFeatureImages
                                     ? state.featuresJagged()
                                     : JaggedTensor();

    if (profile_batch) {
        cudaEventRecord(evEnd);
        cudaEventSynchronize(evEnd);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, evStart, evEnd);
        std::fprintf(
            stderr,
            "[fvdb/tsdf_batch] N=%lld  incremental=%.2f ms  (%.2f ms/frame)  final_voxels=%lld  final_leaves=%lld\n",
            (long long)N, ms, ms / static_cast<float>(N),
            (long long)accumGrid->totalVoxels(),
            (long long)accumGrid->totalLeaves());
        cudaEventDestroy(evStart);
        cudaEventDestroy(evEnd);
    }

    return {accumGrid, accumTsdf, accumWeights, accumFeatures};
}

} // anonymous namespace

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFBatch(const c10::intrusive_ptr<GridBatchData> grid,
                   const double truncationMargin,
                   const torch::Tensor &projectionMatrices,
                   const torch::Tensor &camToWorldMatrices,
                   const JaggedTensor &tsdf,
                   const JaggedTensor &weights,
                   const torch::Tensor &depthImages,
                   const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(grid->device().is_cuda(),
                                "TSDF integration not implemented on the CPU.");
    auto [unionGrid, outTsdf, outWeights, _unusedFeatures] = integrateTSDFBatchImpl(
        grid, truncationMargin, projectionMatrices, camToWorldMatrices,
        tsdf, weights, std::nullopt,
        depthImages, std::nullopt, weightImages);
    return {unionGrid, outTsdf, outWeights};
}

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFBatchWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                               const double truncationMargin,
                               const torch::Tensor &projectionMatrices,
                               const torch::Tensor &camToWorldMatrices,
                               const JaggedTensor &tsdf,
                               const JaggedTensor &features,
                               const JaggedTensor &weights,
                               const torch::Tensor &depthImages,
                               const torch::Tensor &featureImages,
                               const std::optional<torch::Tensor> &weightImages) {
    TORCH_CHECK_NOT_IMPLEMENTED(grid->device().is_cuda(),
                                "TSDF integration not implemented on the CPU.");
    return integrateTSDFBatchImpl(grid, truncationMargin, projectionMatrices,
                                  camToWorldMatrices, tsdf, weights, features,
                                  depthImages, featureImages, weightImages);
}

} // namespace fvdb::detail::ops
