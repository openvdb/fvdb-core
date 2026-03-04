// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH

#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <cuda/std/tuple>

#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb::detail::ops {

// Initialize an accessor for a tensor. The tensor must be a CUDA tensor.
template <typename T, int N>
inline auto
initAccessor(const torch::Tensor &tensor, const std::string &name) {
    TORCH_CHECK(tensor.is_cuda() || tensor.is_privateuseone(),
                "Tensor ",
                name,
                " must be a CUDA or PrivateUse1 tensor");
    return tensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
}

// Initialize an accessor for an optional tensor. The tensor must be a CUDA tensor. If the tensor
// is std::nullopt, return an invalid accessor to a temporary empty tensor. This invalid accessor
// should not be used, it only exists because accessors cannot be default-constructed.
template <typename T, int N>
inline auto
initAccessor(const std::optional<torch::Tensor> &tensor,
             torch::TensorOptions defaultOptions,
             const std::string &name) {
    return initAccessor<T, N>(
        tensor.value_or(torch::empty(std::array<int64_t, N>{}, defaultOptions)), name);
}

// Initialize a jagged accessor for a JaggedTensor. The tensor must be a CUDA tensor.
template <typename T, int N>
inline auto
initJaggedAccessor(const fvdb::JaggedTensor &tensor, const std::string &name) {
    TORCH_CHECK(tensor.is_cuda() || tensor.is_privateuseone(),
                "Tensor ",
                name,
                " must be a CUDA or PrivateUse1 tensor");
    return tensor.packed_accessor64<T, N, torch::RestrictPtrTraits>();
}

/// @brief Common fields and helpers for both forward and backward rasterization kernels
/// @tparam ScalarType The scalar type of the Gaussian
/// @tparam NUM_CHANNELS The number of channels of the Gaussian
/// @tparam IS_PACKED Whether the Gaussian is packed (i.e. linearized across the outer dimensions)
template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED, typename TileIntersectionsT>
struct RasterizeCommonArgs {
    constexpr static size_t NUM_OUTER_DIMS         = IS_PACKED ? 1 : 2;
    constexpr static ScalarType ALPHA_THRESHOLD    = ScalarType{0.999};
    using vec2t                                    = nanovdb::math::Vec2<ScalarType>;
    using vec3t                                    = nanovdb::math::Vec3<ScalarType>;
    template <typename T, int N> using TorchRAcc64 = fvdb::TorchRAcc64<T, N>;
    using ScalarAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS>;
    using VectorAccessor                           = TorchRAcc64<ScalarType, NUM_OUTER_DIMS + 1>;

    uint32_t mNumGaussiansPerCamera;
    RenderWindow2D mRenderWindow;
    TileIntersectionsT mTileIntersections;

    // Common input tensors
    VectorAccessor mMeans2d;                  // [C, N, 2] or [nnz, 2]
    VectorAccessor mConics;                   // [C, N, 3] or [nnz, 3]
    ScalarAccessor mOpacities;                // [C, N] or [nnz]
    // Common optional input tensors
    bool mHasFeatures;
    VectorAccessor mFeatures;                // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    TorchRAcc64<ScalarType, 2> mBackgrounds; // [C, NUM_CHANNELS]
    bool mHasBackgrounds;
    TorchRAcc64<bool, 3> mMasks;             // [C, nTilesH, nTilesW]
    bool mHasMasks;

    RasterizeCommonArgs(
        const torch::Tensor &means2d,                 // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                  // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,               // [C, N] or [nnz]
        const std::optional<torch::Tensor> &features, // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const TileIntersectionsT &tileIntersections,
        const RenderWindow2D &renderWindow)
        : mRenderWindow(renderWindow),
          mTileIntersections(tileIntersections),
          mMeans2d(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(means2d, "means2d")),
          mConics(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(conics, "conics")),
          mOpacities(initAccessor<ScalarType, NUM_OUTER_DIMS>(opacities, "opacities")),
          mHasFeatures(features.has_value()),
          mFeatures(initAccessor<ScalarType, NUM_OUTER_DIMS + 1>(
              features, opacities.options(), "features")),
          mBackgrounds(initAccessor<ScalarType, 2>(backgrounds, means2d.options(), "backgrounds")),
          mHasBackgrounds(backgrounds.has_value()),
          mMasks(initAccessor<bool, 3>(masks, means2d.options().dtype(torch::kBool), "masks")),
          mHasMasks(masks.has_value()) {
        static_assert(NUM_OUTER_DIMS == 1 || NUM_OUTER_DIMS == 2, "NUM_OUTER_DIMS must be 1 or 2");

        mNumGaussiansPerCamera = IS_PACKED ? 0 : mMeans2d.size(1);

        checkInputShapes();
    }

    // Check that the input tensor shapes are valid
    void
    checkInputShapes() {
        const int64_t totalGaussians = IS_PACKED ? mMeans2d.size(0) : 0;

        TORCH_CHECK_VALUE(2 == mMeans2d.size(NUM_OUTER_DIMS), "Bad size for means2d");
        TORCH_CHECK_VALUE(3 == mConics.size(NUM_OUTER_DIMS), "Bad size for conics");

        if constexpr (IS_PACKED) {
            TORCH_CHECK_VALUE(totalGaussians == mMeans2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(totalGaussians == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(totalGaussians == mOpacities.size(0), "Bad size for opacities");
        } else {
            TORCH_CHECK_VALUE(mMeans2d.size(0) == mConics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mMeans2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(mMeans2d.size(0) == mOpacities.size(0), "Bad size for opacities");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mConics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mOpacities.size(1),
                              "Bad size for opacities");
        }

        if (mHasFeatures) {
            TORCH_CHECK_VALUE(NUM_CHANNELS == mFeatures.size(NUM_OUTER_DIMS),
                              "Bad size for features");
            if constexpr (IS_PACKED) {
                TORCH_CHECK_VALUE(totalGaussians == mFeatures.size(0), "Bad size for features");
            } else {
                TORCH_CHECK_VALUE(mMeans2d.size(0) == mFeatures.size(0), "Bad size for features");
                TORCH_CHECK_VALUE(mNumGaussiansPerCamera == mFeatures.size(1),
                                  "Bad size for features");
            }
        }
        if (mHasBackgrounds) {
            TORCH_CHECK_VALUE(mMeans2d.size(0) == mBackgrounds.size(0), "Bad size for backgrounds");
            TORCH_CHECK_VALUE(NUM_CHANNELS == mBackgrounds.size(1), "Bad size for backgrounds");
        }
        if (mHasMasks) {
            TORCH_CHECK_VALUE(mMeans2d.size(0) == mMasks.size(0), "Bad size for masks");
            TORCH_CHECK_VALUE(mTileIntersections.numTilesH() == mMasks.size(1), "Bad size for masks");
            TORCH_CHECK_VALUE(mTileIntersections.numTilesW() == mMasks.size(2), "Bad size for masks");
        }

    }

    // Construct a Gaussian2D object from the input tensors at the given index
    inline __device__ Gaussian2D<ScalarType>
    getGaussian(const uint32_t index) {
        if constexpr (IS_PACKED) {
            return Gaussian2D<ScalarType>(
                index,
                vec2t(mMeans2d[index][0], mMeans2d[index][1]),
                mOpacities[index],
                vec3t(mConics[index][0], mConics[index][1], mConics[index][2]));
        } else {
            auto cid = index / mNumGaussiansPerCamera;
            auto gid = index % mNumGaussiansPerCamera;
            return Gaussian2D<ScalarType>(
                index,
                vec2t(mMeans2d[cid][gid][0], mMeans2d[cid][gid][1]),
                mOpacities[cid][gid],
                vec3t(mConics[cid][gid][0], mConics[cid][gid][1], mConics[cid][gid][2]));
        }
    }

    // Evaluate a Gaussian at a given pixel
    // @return tuple: {gaussianIsValid, delta, exp(-sigma),
    //   alpha = min(ALPHA_THRESHOLD, opacity * exp(-sigma))}
    inline __device__ auto
    evalGaussian(const Gaussian2D<ScalarType> &gaussian,
                 const ScalarType px,
                 const ScalarType py) const {
        const auto delta         = gaussian.delta(px, py);
        const auto sigma         = gaussian.sigma(px, py);
        const auto expMinusSigma = __expf(-sigma);
        const auto alpha         = min(ALPHA_THRESHOLD, gaussian.opacity * expMinusSigma);

        const bool gaussianIsValid = !(sigma < 0 || alpha < 1.f / 255.f);

        return std::make_tuple(gaussianIsValid, delta, expMinusSigma, alpha);
    }

    inline __device__ uint32_t
    numGaussianBatches(int32_t firstGaussianIdInBlock,
                       int32_t lastGaussianIdInBlock,
                       uint32_t blockSize) const {
        return (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;
    }

    inline __device__ uint32_t
    gaussianBatchStartFrontToBack(int32_t firstGaussianIdInBlock,
                                  uint32_t batchIdx,
                                  uint32_t blockSize) const {
        return firstGaussianIdInBlock + blockSize * batchIdx;
    }

    inline __device__ uint32_t
    gaussianBatchSizeFrontToBack(int32_t lastGaussianIdInBlock,
                                 uint32_t batchStart,
                                 uint32_t blockSize) const {
        return min(blockSize, static_cast<uint32_t>(lastGaussianIdInBlock - batchStart));
    }

    template <typename SharedGaussianT>
    inline __device__ void
    loadGaussianBatchFrontToBack(int32_t firstGaussianIdInBlock,
                                 int32_t lastGaussianIdInBlock,
                                 uint32_t batchIdx,
                                 uint32_t blockSize,
                                 uint32_t tidx,
                                 SharedGaussianT *sharedGaussians) const {
        const uint32_t batchStart = gaussianBatchStartFrontToBack(
            firstGaussianIdInBlock, batchIdx, blockSize);
        const uint32_t intersectionIdx = batchStart + tidx;
        if (intersectionIdx < static_cast<uint32_t>(lastGaussianIdInBlock)) {
            const int32_t gaussianIdx = mTileIntersections.gaussianIdAt(intersectionIdx);
            sharedGaussians[tidx]     = getGaussian(gaussianIdx);
        }
    }

    inline __device__ int32_t
    gaussianBatchEndBackToFront(int32_t lastGaussianIdInBlock,
                                uint32_t batchIdx,
                                uint32_t blockSize) const {
        return lastGaussianIdInBlock - 1 - blockSize * batchIdx;
    }

    inline __device__ int32_t
    gaussianBatchSizeBackToFront(int32_t firstGaussianIdInBlock,
                                 int32_t batchEnd,
                                 uint32_t blockSize) const {
        return min(static_cast<int32_t>(blockSize), batchEnd + 1 - firstGaussianIdInBlock);
    }

    template <typename SharedGaussianT>
    inline __device__ bool
    loadGaussianBatchBackToFront(int32_t firstGaussianIdInBlock,
                                 int32_t lastGaussianIdInBlock,
                                 uint32_t batchIdx,
                                 uint32_t blockSize,
                                 uint32_t tidx,
                                 SharedGaussianT *sharedGaussians,
                                 int32_t &outGaussianId,
                                 int32_t &outBatchEnd) const {
        outBatchEnd               = gaussianBatchEndBackToFront(lastGaussianIdInBlock,
                                                  batchIdx,
                                                  blockSize);
        const int32_t intersectionIdx = outBatchEnd - tidx;
        if (intersectionIdx >= firstGaussianIdInBlock) {
            outGaussianId         = mTileIntersections.gaussianIdAt(intersectionIdx);
            sharedGaussians[tidx] = getGaussian(outGaussianId);
            return true;
        }
        outGaussianId = -1;
        return false;
    }

};

} // namespace fvdb::detail::ops
#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZE_CUH
