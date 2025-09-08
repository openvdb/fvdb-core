// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/ops/gsplat/GaussianWarpUtils.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>

#include <cooperative_groups.h>

namespace fvdb::detail::ops {
namespace {

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
struct RasterizeBackwardArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    CommonArgs commonArgs;

    using vec2t          = typename CommonArgs::vec2t;
    using vec3t          = typename CommonArgs::vec3t;
    using VectorAccessor = typename CommonArgs::VectorAccessor;
    using ScalarAccessor = typename CommonArgs::ScalarAccessor;

    constexpr static bool IS_CHUNKED = (NUM_CHANNELS != NUM_SHARED_CHANNELS);
    bool mAbsGrad;

    // These are either a packed tensor accessor or a jagged tensor accessor, depending on the
    // mode (dense or sparse, respectively). In sparse mode, they have dimensions {C, [AP_i, X]},
    // where AP_i is the number of active pixels in the i-th camera image and X is [], null, or
    // [NUM_CHANNELS]. In dense mode, they have dimensions {1, [C, H, W, X]}
    fvdb::JaggedRAcc64<ScalarType, 2> mRenderedAlphas; // {1, [C, H, W, 1]} or {C, [AP_i, 1]}
    fvdb::JaggedRAcc64<int32_t, 1> mLastGaussianIds;   // {1, [C, H, W]} or {C, [AP_i]}
    fvdb::JaggedRAcc64<ScalarType, 2>
        mDLossDRenderedFeatures; // {1, [C, H, W, NUM_CHANNELS]} or {C, [numPixels, NUM_CHANNELS]}
    fvdb::JaggedRAcc64<ScalarType, 2>
        mDLossDRenderedAlphas;   // {1, [C, H, W, 1]} or {C, [numPixels, 1]}

    VectorAccessor mOutDLossDMeans2dAbs; // [C, N, 2] or [nnz, 2]
    VectorAccessor mOutDLossDMeans2d;    // [C, N, 2] or [nnz, 2]
    VectorAccessor mOutDLossDConics;     // [C, N, 3] or [nnz, 3]
    VectorAccessor mOutDLossDFeatures;   // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    ScalarAccessor mOutDLossDOpacities;  // [C, N] or [nnz]

    RasterizeBackwardArgs(
        const torch::Tensor &means2d,    // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,     // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,  // [C, N] or [nnz]
        const torch::Tensor &features,   // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const at::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const torch::Tensor &tileOffsets,          // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds,      // [totalIntersections]
        const fvdb::JaggedTensor &renderedAlphas,  // {C, [AP_i, 1]} or {1, [C, H, W, 1]}
        const fvdb::JaggedTensor &lastGaussianIds, // {C, [AP_i]} or {1, [C, H, W]}
        const fvdb::JaggedTensor
            &dLossDRenderedFeatures, // {C, [AP_i, NUM_CHANNELS]} or {1, [C, H, W, NUM_CHANNELS]}
        const fvdb::JaggedTensor &dLossDRenderedAlphas, // {C, [AP_i, 1]} or {1, [C, H, W, 1]}
        const torch::Tensor &outDLossDMeans2d,          // [C, N, 2] or [nnz, 2]
        const torch::Tensor &outDLossDConics,           // [C, N, 3] or [nnz, 3]
        const torch::Tensor &outDLossDFeatures,  // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const torch::Tensor &outDLossDOpacities, // [C, N] or [nnz]
        const std::optional<torch::Tensor> &outDLossDMeans2dAbs,        // [C, N, 2] or [nnz, 2]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : commonArgs(means2d,
                     conics,
                     opacities,
                     features,
                     backgrounds,
                     masks,
                     imageWidth,
                     imageHeight,
                     imageOriginW,
                     imageOriginH,
                     tileSize,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
          mAbsGrad(outDLossDMeans2dAbs.has_value()),
          mRenderedAlphas(initJaggedAccessor<ScalarType, 2>(renderedAlphas, "renderedAlphas")),
          mLastGaussianIds(initJaggedAccessor<int32_t, 1>(lastGaussianIds, "lastGaussianIds")),
          mDLossDRenderedFeatures(
              initJaggedAccessor<ScalarType, 2>(dLossDRenderedFeatures, "dLossDRenderedFeatures")),
          mDLossDRenderedAlphas(
              initJaggedAccessor<ScalarType, 2>(dLossDRenderedAlphas, "dLossDRenderedAlphas")),
          mOutDLossDMeans2dAbs(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDMeans2dAbs, means2d.options(), "outDLossDMeans2dAbs")),
          mOutDLossDMeans2d(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDMeans2d, "outDLossDMeans2d")),
          mOutDLossDFeatures(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDFeatures, "outDLossDFeatures")),
          mOutDLossDConics(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDConics, "outDLossDConics")),
          mOutDLossDOpacities(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS>(
              outDLossDOpacities, "outDLossDOpacities")) {
        checkInputShapes();
    }

    // Check that the input tensor shapes are valid
    void
    checkInputShapes() {
        const int64_t totalGaussians = IS_PACKED ? commonArgs.mMeans2d.size(0) : 0;

        TORCH_CHECK_VALUE(2 == mOutDLossDMeans2d.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDMeans2d");
        TORCH_CHECK_VALUE(3 == mOutDLossDConics.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDConics");
        TORCH_CHECK_VALUE(NUM_CHANNELS == mOutDLossDFeatures.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDFeatures");
        TORCH_CHECK_VALUE(2 == mOutDLossDMeans2d.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDMeans2d");
        TORCH_CHECK_VALUE(3 == mOutDLossDConics.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDConics");
        TORCH_CHECK_VALUE(NUM_CHANNELS == mOutDLossDFeatures.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDFeatures");

        if constexpr (IS_PACKED) {
            if (mAbsGrad) {
                TORCH_CHECK_VALUE(totalGaussians == mOutDLossDMeans2dAbs.size(0),
                                  "Bad size for outDLossDMeans2dAbs");
                TORCH_CHECK_VALUE(2 == mOutDLossDMeans2dAbs.size(CommonArgs::NUM_OUTER_DIMS),
                                  "Bad size for outDLossDMeans2dAbs");
            }
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDMeans2d.size(0),
                              "Bad size for outDLossDMeans2d");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDConics.size(0),
                              "Bad size for outDLossDConics");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDFeatures.size(0),
                              "Bad size for outDLossDFeatures");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDOpacities.size(0),
                              "Bad size for outDLossDOpacities");
        } else {
            if (mAbsGrad) {
                TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDMeans2dAbs.size(0),
                                  "Bad size for outDLossDMeans2dAbs");
                TORCH_CHECK_VALUE(2 == mOutDLossDMeans2dAbs.size(CommonArgs::NUM_OUTER_DIMS),
                                  "Bad size for outDLossDMeans2dAbs");
            }

            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDMeans2d.size(0),
                              "Bad size for outDLossDMeans2d");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDConics.size(0),
                              "Bad size for outDLossDConics");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDFeatures.size(0),
                              "Bad size for outDLossDFeatures");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDOpacities.size(0),
                              "Bad size for outDLossDOpacities");
            TORCH_CHECK_VALUE(commonArgs.mNumGaussiansPerCamera == mOutDLossDOpacities.size(1),
                              "Bad size for outDLossDOpacities");
        }

        if (commonArgs.mIsSparse) {
            // just check that the number of cameras and number of pixels are correct
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mRenderedAlphas.numTensors(),
                              "Bad size for renderedAlphas");
            TORCH_CHECK_VALUE(commonArgs.mPixelMap.size(0) == mRenderedAlphas.elementCount(),
                              "Bad size for renderedAlphas");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mLastGaussianIds.numTensors(),
                              "Bad size for lastGaussianIds");
            TORCH_CHECK_VALUE(commonArgs.mPixelMap.size(0) == mLastGaussianIds.elementCount(),
                              "Bad size for lastGaussianIds");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mDLossDRenderedFeatures.numTensors(),
                              "Bad size for dLossDRenderedFeatures");
            TORCH_CHECK_VALUE(commonArgs.mPixelMap.size(0) ==
                                  mDLossDRenderedFeatures.elementCount(),
                              "Bad size for dLossDRenderedFeatures");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mDLossDRenderedAlphas.numTensors(),
                              "Bad size for dLossDRenderedAlphas");
            TORCH_CHECK_VALUE(commonArgs.mPixelMap.size(0) == mDLossDRenderedAlphas.elementCount(),
                              "Bad size for dLossDRenderedAlphas");
        } else {
            auto const tensorSize =
                commonArgs.mNumCameras * commonArgs.mImageHeight * commonArgs.mImageWidth;
            TORCH_CHECK_VALUE(tensorSize == mRenderedAlphas.data().size(0),
                              "Bad size for renderedAlphas");
            TORCH_CHECK_VALUE(1 == mRenderedAlphas.data().size(1), "Bad size for renderedAlphas");

            TORCH_CHECK_VALUE(tensorSize == mLastGaussianIds.data().size(0),
                              "Bad size for lastGaussianIds");

            TORCH_CHECK_VALUE(tensorSize == mDLossDRenderedFeatures.data().size(0),
                              "Bad size for dLossDRenderedFeatures");
            TORCH_CHECK_VALUE(NUM_CHANNELS == mDLossDRenderedFeatures.data().size(1),
                              "Bad size for dLossDRenderedFeatures");

            TORCH_CHECK_VALUE(tensorSize == mDLossDRenderedAlphas.data().size(0),
                              "Bad size for dLossDRenderedAlphas");
            TORCH_CHECK_VALUE(1 == mDLossDRenderedAlphas.data().size(1),
                              "Bad size for dLossDRenderedAlphas");
        }
    }

    /// @brief Read the alpha value for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The alpha value for the pixel
    __device__ ScalarType
    readAlpha(uint64_t pixelIndex) {
        return mRenderedAlphas.data()[pixelIndex][0];
    }

    /// @brief Read the last ID for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The last ID for the pixel
    __device__ int32_t
    readLastId(uint64_t pixelIndex) {
        return mLastGaussianIds.data()[pixelIndex];
    }

    /// @brief Read the gradient of the loss with respect to the rendered alpha for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The gradient of the loss with respect to the rendered alpha for the pixel
    __device__ ScalarType
    readDLossDRenderedAlpha(uint64_t pixelIndex) {
        return mDLossDRenderedAlphas.data()[pixelIndex][0];
    }

    /// @brief Read the gradient of the loss with respect to the rendered features for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The gradient of the loss with respect to the rendered features for the pixel
    __device__ ScalarType
    readDLossDRenderedFeature(uint64_t pixelIndex, uint32_t channel) {
        return mDLossDRenderedFeatures.data()[pixelIndex][channel];
    }

    // Fetch the features for a Gaussian into shared memory
    inline __device__ void
    fetchGaussianFeatureIntoSharedMemory(const int32_t g,
                                         const size_t channelStart,
                                         const size_t numChannels,
                                         ScalarType *outFeatures) {
        if constexpr (IS_PACKED) {
            const auto featureAccessor = commonArgs.mFeatures[g];
            for (uint32_t k = 0; k < numChannels; ++k) {
                outFeatures[k] = featureAccessor[k + channelStart];
            }
        } else {
            // colors: [C, N, NUM_CHANNELS]
            // colors[c, n, k] = [c * N * NUM_CHANNELS + n * NUM_CHANNELS + k]
            // g = c * N + n
            const int32_t cid          = g / commonArgs.mNumGaussiansPerCamera;
            const int32_t gid          = g % commonArgs.mNumGaussiansPerCamera;
            const auto featureAccessor = commonArgs.mFeatures[cid][gid];
            if constexpr (IS_CHUNKED) {
                for (auto k = 0; k < numChannels; ++k) {
                    outFeatures[k] = featureAccessor[k + channelStart];
                }
            } else {
#pragma unroll NUM_CHANNELS
                for (auto k = 0; k < NUM_CHANNELS; ++k) {
                    outFeatures[k] = featureAccessor[k];
                }
            }
        }
    }

    /// @brief Get a pointer to per-Gaussian data from a tensor accessor (used for device atomics)
    ///
    /// This assumes that the tensor accessor dimension is at least 2.
    /// @tparam T The type of the data to access
    /// @tparam Accessor The type of the accessor
    /// @param accessor The accessor to the tensor
    /// @param g The index of the Gaussian
    /// @return A pointer to the data for the Gaussian
    template <typename T, typename Accessor>
    inline __device__ T *
    getAccessorPointer(const Accessor &accessor, const int32_t g) const {
        if constexpr (IS_PACKED) {
            return reinterpret_cast<T *>(accessor[g].data());
        } else {
            auto cid = g / commonArgs.mNumGaussiansPerCamera;
            auto gid = g % commonArgs.mNumGaussiansPerCamera;
            return reinterpret_cast<T *>(accessor[cid][gid].data());
        }
    }

    /// @brief Get a pointer to per-Gaussian data from a 1D tensor accessor (used for device
    /// atomics)
    /// @tparam T The type of the data to access
    /// @tparam Accessor The type of the accessor
    /// @param accessor The accessor to the tensor
    /// @param g The index of the Gaussian
    /// @return A pointer to the data for the Gaussian
    template <typename T, typename Accessor>
    inline __device__ T *
    get1DAccessorPointer(const Accessor &accessor, const int32_t g) const {
        // Because GenericPackedTensorAccessor<...,N=1,...>::operator[] returns a reference,
        // we can't take the address of the specific offset. So in the 1D accessor case, we
        // have to use some pointer arithmetic to get the address of the specific offset.
        if constexpr (IS_PACKED) {
            return reinterpret_cast<T *>(accessor.data()) + g;
        } else {
            auto cid = g / commonArgs.mNumGaussiansPerCamera;
            auto gid = g % commonArgs.mNumGaussiansPerCamera;
            return reinterpret_cast<T *>(accessor[cid].data()) + gid;
        }
    }

    /// @brief Atomically accumulate the gradient contribution from this pixel to the gradient of
    /// the loss with respect to the features of this Gaussian
    /// @param g The index of the Gaussian
    /// @param featureGradientContribution The gradient contribution from this pixel to the gradient
    /// for the features of this Gaussian
    /// @param channelStart The starting channel index for the features
    /// @param numChannels The number of channels to accumulate
    inline __device__ void
    atomicAddFeatureGradientContributions(const int32_t g,
                                          const ScalarType *featureGradientContribution,
                                          const size_t channelStart,
                                          const size_t numChannels) {
        auto dLossDFeaturesGaussianPtr = getAccessorPointer<ScalarType>(mOutDLossDFeatures, g);
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                gpuAtomicAdd(dLossDFeaturesGaussianPtr + channelStart + k,
                             featureGradientContribution[k]);
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                gpuAtomicAdd(dLossDFeaturesGaussianPtr + k, featureGradientContribution[k]);
            }
        }
    }

    /// @brief Atomically accumulate the gradient contribution from this pixel to the gradient of
    /// the loss with respect to the means, conics, and opacities of this Gaussian
    /// @param g The index of the Gaussian
    /// @param pixelConicGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the conic of this Gaussian
    /// @param pixelMean2dGradientContribution The gradient contribution from this pixel to the
    inline __device__ void
    atomicAddMeans2dConicsAndOpacitiesGradientContributions(
        const int32_t g,
        const vec3t &pixelConicGradientContribution,
        const vec2t &pixelMean2dGradientContribution,
        const vec2t &pixelMean2dAbsGradientContribution,
        const ScalarType pixelOpacityGradientContribution) {
        auto *dLossDConicsGaussianPtr = getAccessorPointer<vec3t>(mOutDLossDConics, g);
        gpuAtomicAdd(&dLossDConicsGaussianPtr->operator[](0), pixelConicGradientContribution[0]);
        gpuAtomicAdd(&dLossDConicsGaussianPtr->operator[](1), pixelConicGradientContribution[1]);
        gpuAtomicAdd(&dLossDConicsGaussianPtr->operator[](2), pixelConicGradientContribution[2]);

        auto *dLossDMeans2DGaussianPtr = getAccessorPointer<vec2t>(mOutDLossDMeans2d, g);
        gpuAtomicAdd(&dLossDMeans2DGaussianPtr->operator[](0), pixelMean2dGradientContribution[0]);
        gpuAtomicAdd(&dLossDMeans2DGaussianPtr->operator[](1), pixelMean2dGradientContribution[1]);

        if (mAbsGrad) {
            auto *dLossDMeans2dAbsGaussianPtr = getAccessorPointer<vec2t>(mOutDLossDMeans2dAbs, g);
            gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->operator[](0),
                         pixelMean2dAbsGradientContribution[0]);
            gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->operator[](1),
                         pixelMean2dAbsGradientContribution[1]);
        }

        auto *dLossDOpacitiesGaussianPtr = get1DAccessorPointer<ScalarType>(mOutDLossDOpacities, g);
        gpuAtomicAdd(dLossDOpacitiesGaussianPtr, pixelOpacityGradientContribution);
    }

    /// @brief Accumulate the features of this Gaussian into the accumulated features
    /// @param gaussianFeatures The features of this Gaussian
    /// @param fac The factor to multiply the features by
    /// @param numChannels The number of channels to accumulate
    /// @param outAccumFeatures The accumulated features
    inline __device__ void
    accumulateFeaturesStep(const ScalarType *gaussianFeatures,
                           const ScalarType fac,
                           const size_t numChannels,
                           ScalarType *outAccumFeatures) const {
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                outAccumFeatures[k] += gaussianFeatures[k] * fac;
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                outAccumFeatures[k] += gaussianFeatures[k] * fac;
            }
        }
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradient of the loss with
    /// respect to the features of this Gaussian
    /// @param fac The factor to multiply the features by
    /// @param dLossDRenderedFeatures The gradient of the loss with respect to the rendered features
    /// @param outFeatureGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the features of this Gaussian
    inline __device__ void
    calculateFeatureGradientContribution(const ScalarType fac,
                                         const ScalarType *dLossDRenderedFeatures,
                                         ScalarType *outFeatureGradientContribution) const {
#pragma unroll NUM_SHARED_CHANNELS
        for (uint32_t k = 0; k < NUM_SHARED_CHANNELS; ++k) {
            outFeatureGradientContribution[k] = fac * dLossDRenderedFeatures[k];
        }
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradient of the loss with
    /// respect to the alpha value of this Gaussian
    /// @param cameraId The ID of the camera
    /// @param finalTransmittance The final transmittance for the current pixel
    /// @param oneOverOneMinusAlpha The inverse of (1 - alpha)
    /// @param accumTransmittance The accumulated transmittance for the current pixel
    inline __device__ ScalarType
    calculateAlphaGradientContribution(const uint32_t cameraId,
                                       const ScalarType finalTransmittance,
                                       const ScalarType oneOverOneMinusAlpha,
                                       const ScalarType accumTransmittance,
                                       const ScalarType *accumFeature,
                                       const ScalarType *gaussianFeature,
                                       const ScalarType *dLossDRenderedFeature,
                                       const ScalarType dLossDRenderedAlpha,
                                       const size_t numChannels,
                                       const bool includeLastTerm) const {
        ScalarType alphaGradientContribution = ScalarType{0};
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                alphaGradientContribution += (gaussianFeature[k] * accumTransmittance -
                                              accumFeature[k] * oneOverOneMinusAlpha) *
                                             dLossDRenderedFeature[k];
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                alphaGradientContribution += (gaussianFeature[k] * accumTransmittance -
                                              accumFeature[k] * oneOverOneMinusAlpha) *
                                             dLossDRenderedFeature[k];
            }
        }

        if (includeLastTerm) {
            alphaGradientContribution +=
                finalTransmittance * oneOverOneMinusAlpha * dLossDRenderedAlpha;
        }

        // Factor in the contribution from the background to this pixel
        if (commonArgs.mHasBackgrounds) {
            ScalarType accum = ScalarType{0};
            if constexpr (IS_CHUNKED) {
                for (uint32_t k = 0; k < numChannels; ++k) {
                    accum += commonArgs.mBackgrounds[cameraId][k] * dLossDRenderedFeature[k];
                }
            } else {
#pragma unroll NUM_CHANNELS
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    accum += commonArgs.mBackgrounds[cameraId][k] * dLossDRenderedFeature[k];
                }
            }
            if (includeLastTerm) {
                alphaGradientContribution += -finalTransmittance * oneOverOneMinusAlpha * accum;
            }
        }

        return alphaGradientContribution;
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradients of the loss with
    /// respect to the means, conics, and opacities of this Gaussian
    /// @param opac The opacity of this Gaussian
    /// @param vis The visibility of this Gaussian
    /// @param alphaGradientContribution The gradient contribution from this pixel to the gradient
    /// of the loss with respect to the alpha value of this Gaussian
    inline __device__ void
    calculateMeansConicsAndOpacitiesGradientContribution(
        ScalarType opac,
        ScalarType expMinusSigma,
        ScalarType alphaGradientContribution,
        const vec3t &conic,
        const vec2t &delta,
        vec3t &outConicGradientContribution,
        vec2t &outMean2dGradientContribution,
        vec2t &outMean2dAbsGradientContribution,
        ScalarType &outOpacityGradientContribution) const {
        // Contribution from this pixel to sigma for this Gaussian
        const ScalarType sigmaGradientContribution =
            -opac * expMinusSigma * alphaGradientContribution;
        outConicGradientContribution = {
            ScalarType{0.5} * sigmaGradientContribution * delta[0] * delta[0],
            sigmaGradientContribution * delta[0] * delta[1],
            ScalarType{0.5} * sigmaGradientContribution * delta[1] * delta[1]};
        outMean2dGradientContribution = {
            sigmaGradientContribution * (conic[0] * delta[0] + conic[1] * delta[1]),
            sigmaGradientContribution * (conic[1] * delta[0] + conic[2] * delta[1])};
        if (!mAbsGrad) {
            outMean2dAbsGradientContribution = {abs(outMean2dGradientContribution[0]),
                                                abs(outMean2dGradientContribution[1])};
        }
        outOpacityGradientContribution = expMinusSigma * alphaGradientContribution;
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradients of the loss with
    /// respect to the features, means, conics, and opacities of this Gaussian
    /// @param cameraId The ID of the camera
    /// @param gaussian The Gaussian
    /// @param gaussianFeature The features of this Gaussian
    /// @param dLossDRenderedFeature The gradient of the loss with respect to the rendered features
    /// @param dLossDRenderedAlpha The gradient of the loss with respect to the alpha value of this
    /// Gaussian
    /// @param px The x coordinate of the pixel
    /// @param py The y coordinate of the pixel
    /// @param finalTransmittance The final transmittance for the current pixel
    /// @param numChannels The number of channels to accumulate
    /// @param calculateMeansConicsAndOpacitiesGradient Whether to calculate the gradient
    /// contribution from this pixel to the gradients of the loss with respect to the means, conics,
    /// and opacities of this Gaussian
    /// @param accumTransmittance The accumulated transmittance for the current pixel
    /// @param accumFeature The accumulated features for the current pixel
    /// @param outFeatureGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the features of this Gaussian
    /// @param outConicGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the conic of this Gaussian
    /// @param outMean2dGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the 2d mean of this Gaussian
    /// @param outMean2dAbsGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the absolute value of the 2d mean of this Gaussian
    /// @param outOpacityGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the opacity of this Gaussian
    /// @return Whether this Gaussian contributes to this pixel
    inline __device__ void
    calculateGradientContributions(const uint32_t cameraId,
                                   const Gaussian2D<ScalarType> &gaussian,
                                   const vec2t &delta,
                                   const ScalarType expMinusSigma,
                                   const ScalarType alpha,
                                   const ScalarType *gaussianFeature,
                                   const ScalarType *dLossDRenderedFeature,
                                   const ScalarType dLossDRenderedAlpha,
                                   const ScalarType px,
                                   const ScalarType py,
                                   const ScalarType finalTransmittance,
                                   const size_t numChannels,
                                   const bool calculateMeansConicsAndOpacitiesGradient,
                                   ScalarType &accumTransmittance,
                                   ScalarType *accumFeature,
                                   ScalarType *outFeatureGradientContribution,
                                   vec3t &outConicGradientContribution,
                                   vec2t &outMean2dGradientContribution,
                                   vec2t &outMean2dAbsGradientContribution,
                                   ScalarType &outOpacityGradientContribution) const {
        const vec3t conic     = gaussian.conic;
        const ScalarType opac = gaussian.opacity;

        // Compute the transmittance for the current gaussian
        const ScalarType oneOverOneMinusAlpha = ScalarType{1} / (ScalarType{1} - alpha);
        accumTransmittance *= oneOverOneMinusAlpha;

        // Update the contribution of this pixel to the color gradient of the
        // Gaussian
        const ScalarType fac = alpha * accumTransmittance;
        calculateFeatureGradientContribution(
            fac, dLossDRenderedFeature, outFeatureGradientContribution);

        // Contribution from this pixel to the alpha value for this Gaussian
        const ScalarType alphaGradientContribution =
            calculateAlphaGradientContribution(cameraId,
                                               finalTransmittance,
                                               oneOverOneMinusAlpha,
                                               accumTransmittance,
                                               accumFeature,
                                               gaussianFeature,
                                               dLossDRenderedFeature,
                                               dLossDRenderedAlpha,
                                               numChannels,
                                               calculateMeansConicsAndOpacitiesGradient);

        if (opac * expMinusSigma <= commonArgs.ALPHA_THRESHOLD) {
            calculateMeansConicsAndOpacitiesGradientContribution(opac,
                                                                 expMinusSigma,
                                                                 alphaGradientContribution,
                                                                 conic,
                                                                 delta,
                                                                 outConicGradientContribution,
                                                                 outMean2dGradientContribution,
                                                                 outMean2dAbsGradientContribution,
                                                                 outOpacityGradientContribution);
        }

        accumulateFeaturesStep(gaussianFeature, fac, numChannels, accumFeature);
    }

    /// @brief Compute the gradient of the loss with respect to the parameters of the Gaussians
    /// that contribute to this pixel.
    /// @param warp The warp of threads
    /// @param cameraId The ID of the camera
    /// @param i The row index of the pixel
    /// @param j The column index of the pixel
    /// @param firstGaussianIdInBlock The ID of the first Gaussian in the block
    template <size_t WARP_TILE_SIZE>
    inline __device__ void
    volumeRenderTileBackward( // const cooperative_groups::thread_block_tile<WARP_TILE_SIZE> &warp,
        const uint32_t cameraId,
        const uint32_t row,
        const uint32_t col,
        const int32_t firstGaussianIdInBlock,
        const int32_t lastGaussianIdInBlock,
        const uint32_t blockSize,
        const bool pixelIsActive,
        const uint32_t activePixelIndex) {
        extern __shared__ int s[];

        Gaussian2D<ScalarType> *sharedGaussians =
            reinterpret_cast<Gaussian2D<ScalarType> *>(s);               // [blockSize]
        ScalarType *sharedGaussianFeatures =
            reinterpret_cast<ScalarType *>(&sharedGaussians[blockSize]); // [blockSize]

        // To protect against out of bounds access, we clamp the coordinates to the image bounds
        const auto rowClamped = min(row, commonArgs.mImageHeight - 1);
        const auto colClamped = min(col, commonArgs.mImageWidth - 1);

        const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

        // Only access memory if the pixel is active (within image bounds)
        ScalarType finalTransmittance  = ScalarType{1};
        ScalarType dLossDRenderedAlpha = ScalarType{0};
        int32_t lastGaussianId         = 0;

        if (pixelIsActive) {
            // this is the T AFTER the last gaussian in this pixel
            finalTransmittance = ScalarType{1} - readAlpha(pixIdx);

            // Gradient of the loss with respect to the alpha output of the forward pass at this
            // pixel
            dLossDRenderedAlpha = readDLossDRenderedAlpha(pixIdx);

            // ID of the last Gaussian to contribute to this pixel
            lastGaussianId = readLastId(pixIdx);
        }

        namespace cg = cooperative_groups;
        auto block   = cg::this_thread_block();
        const cg::thread_block_tile<WARP_TILE_SIZE> warp =
            cg::tiled_partition<WARP_TILE_SIZE>(block);

        const int32_t lastGaussianIdInWarp = warpMax(lastGaussianId, warp);

        // Process Gaussians in batches of block size (i.e. one Gaussian per thread in the block)
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        constexpr size_t NUM_CHUNKS =
            (NUM_CHANNELS + NUM_SHARED_CHANNELS - 1) / NUM_SHARED_CHANNELS;
        for (size_t chunk = 0; chunk < NUM_CHUNKS; chunk += 1) {
            const size_t channelStart = chunk * NUM_SHARED_CHANNELS;
            const size_t numChannels  = min(NUM_CHANNELS - channelStart, NUM_SHARED_CHANNELS);
            const bool isLastChunk    = chunk == (NUM_CHUNKS - 1);

            ScalarType accumTransmittance = finalTransmittance;

            // the contribution from gaussians behind the current one
            ScalarType accumFeature[NUM_SHARED_CHANNELS] = {ScalarType(0)};

            // Gradient of the loss with respect to the color output of the forward pass at this
            // pixel
            ScalarType dLossDRenderedFeature[NUM_SHARED_CHANNELS];
            if (pixelIsActive) {
                for (auto k = 0; k < numChannels; ++k) {
                    dLossDRenderedFeature[k] = readDLossDRenderedFeature(pixIdx, channelStart + k);
                }
            } else {
                for (auto k = 0; k < numChannels; ++k) {
                    dLossDRenderedFeature[k] = ScalarType{0};
                }
            }
            for (auto k = numChannels; k < NUM_SHARED_CHANNELS; ++k) {
                dLossDRenderedFeature[k] = ScalarType{0};
            }

            for (uint32_t b = 0; b < numBatches; ++b) {
                // resync all threads before writing next batch of shared mem
                __syncthreads();

                // Each thread fetches one gaussian into shared memory.
                // Gaussians are stored in shared memory locations in order of decreasing
                // distance from the camera. Gaussians are processed in batches of size
                // blockSize (i.e. one Gaussian per thread in the block), and batchEnd is the
                // index of the last gaussian. NOTE: These values can be negative so must be
                // int32 instead of uint32
                const int32_t batchEnd = lastGaussianIdInBlock - 1 - blockSize * b;
                const int32_t idx      = batchEnd - tidx;
                if (idx >= firstGaussianIdInBlock) {
                    const int32_t g =
                        commonArgs.mTileGaussianIds[idx]; // Gaussian index in [C * N] or [nnz]
                    sharedGaussians[tidx] = commonArgs.getGaussian(g);
                    ScalarType *feature   = &sharedGaussianFeatures[tidx * NUM_SHARED_CHANNELS];
                    fetchGaussianFeatureIntoSharedMemory(g, channelStart, numChannels, feature);
                }

                // Sync threads so all gaussians for this batch are loaded in shared memory
                __syncthreads();

                // process gaussians in the current batch for this pixel
                // 0 index is the furthest back gaussian in the batch
                // For each Gaussian which contributes to this pixel, compute this pixel's
                // gradient contribution to that Gaussian
                const int32_t batchSize = min(blockSize, batchEnd + 1 - firstGaussianIdInBlock);
                for (uint32_t t = max(0, batchEnd - lastGaussianIdInWarp); t < batchSize; ++t) {
                    // (row, col) coordinates are relative to the specified image origin which may
                    // be a crop so we need to add the origin to get the absolute pixel coordinates
                    const ScalarType px =
                        col + ScalarType(commonArgs.mImageOriginW) + ScalarType{0.5};
                    const ScalarType py =
                        row + ScalarType(commonArgs.mImageOriginH) + ScalarType{0.5};

                    bool valid = pixelIsActive && (batchEnd - t <= lastGaussianId);

                    const auto [gaussianIsValid, delta, expMinusSigma, alpha] = [&]() {
                        if (valid) {
                            return commonArgs.evalGaussian(sharedGaussians[t], px, py);
                        }
                        return std::make_tuple(false, vec2t{}, ScalarType{0}, ScalarType{0});
                    }();

                    valid = valid && gaussianIsValid;

                    // if there are no active thread in this warp, skip this loop
                    if (!warp.any(valid)) {
                        continue;
                    }

                    // How much each pixel contributes to the gradient of the parameters for
                    // this gaussian Initialize to 0 and only set if this pixel is valid
                    ScalarType featureGradientContribution[NUM_SHARED_CHANNELS] = {ScalarType{0}};
                    vec3t conicGradientContribution = {ScalarType{0}, ScalarType{0}, ScalarType{0}};
                    vec2t mean2dGradientContribution       = {ScalarType{0}, ScalarType{0}};
                    vec2t mean2dAbsGradientContribution    = {ScalarType{0}, ScalarType{0}};
                    ScalarType opacityGradientContribution = ScalarType{0};

                    if (valid) {
                        calculateGradientContributions(
                            cameraId,
                            sharedGaussians[t],
                            delta,
                            expMinusSigma,
                            alpha,
                            &sharedGaussianFeatures[t * NUM_SHARED_CHANNELS],
                            dLossDRenderedFeature,
                            dLossDRenderedAlpha,
                            px,
                            py,
                            finalTransmittance,
                            numChannels,
                            isLastChunk,
                            accumTransmittance,
                            accumFeature,
                            featureGradientContribution,
                            conicGradientContribution,
                            mean2dGradientContribution,
                            mean2dAbsGradientContribution,
                            opacityGradientContribution);
                    }

                    // Accumulate the gradient contribution to this Gaussian from every
                    // pixel in the block
                    if constexpr (IS_CHUNKED) {
                        warpSumMut<decltype(warp), ScalarType>(
                            featureGradientContribution, numChannels, warp);
                    } else {
                        warpSumMut<NUM_SHARED_CHANNELS, decltype(warp), ScalarType>(
                            featureGradientContribution, warp);
                    }

                    warpSumMut<decltype(warp), ScalarType>(conicGradientContribution, warp);
                    warpSumMut<decltype(warp), ScalarType>(mean2dGradientContribution, warp);
                    if (mAbsGrad) {
                        warpSumMut<decltype(warp), ScalarType>(mean2dAbsGradientContribution, warp);
                    }
                    warpSumMut<decltype(warp), ScalarType>(opacityGradientContribution, warp);

                    // The first thread in the block accumulates the gradient
                    // contribution from the whole block into the global gradient of
                    // this Gaussian
                    if (warp.thread_rank() == 0) {
                        atomicAddFeatureGradientContributions(sharedGaussians[t].id,
                                                              featureGradientContribution,
                                                              channelStart,
                                                              numChannels);
                        atomicAddMeans2dConicsAndOpacitiesGradientContributions(
                            sharedGaussians[t].id,
                            conicGradientContribution,
                            mean2dGradientContribution,
                            mean2dAbsGradientContribution,
                            opacityGradientContribution);
                    }
                }
            }
        }
    }

    /// @brief Get the block dimensions for the backward pass
    /// @return The block dimensions
    const dim3
    getBlockDim() const {
        return {commonArgs.mTileSize, commonArgs.mTileSize, 1};
    }

    /// @brief Get the grid dimensions for the backward pass
    /// @return The grid dimensions
    const dim3
    getGridDim() const {
        if (commonArgs.mIsSparse) {
            // Sparse mode: only launch blocks for active tiles
            return {static_cast<uint32_t>(commonArgs.mActiveTiles.size(0)), 1, 1};
        } else {
            // Dense mode: launch blocks for all tiles
            const uint32_t tileExtentW =
                (commonArgs.mImageWidth + commonArgs.mTileSize - 1) / commonArgs.mTileSize;
            const uint32_t tileExtentH =
                (commonArgs.mImageHeight + commonArgs.mTileSize - 1) / commonArgs.mTileSize;
            return {commonArgs.mNumCameras, tileExtentH, tileExtentW};
        }
    }
};

/// @brief Compute the gradient of the loss with respect to the parameters of the Gaussians that
/// contribute to each pixel in the image.
/// @param args The arguments for the backward pass
/// @param NUM_CHANNELS The number of channels in the features
/// @param NUM_SHARED_CHANNELS The number of channels to chunk in shared memory
/// @param IS_PACKED Whether the features are packed (i.e. linearized across the outer dimensions)
template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansBackward(
    RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row;
    uint32_t col;

    cuda::std::tie(cameraId, tileRow, tileCol, row, col) =
        commonArgs.mIsSparse ? commonArgs.sparseCoordinates() : commonArgs.denseCoordinates();

    // NOTE: We keep threads which correspond to pixels outside the image bounds around
    //       to load gaussians from global memory, but they do not contribute to the output.

    // pixelInImage: Whether this pixel is inside the image bounds.
    // activePixelIndex: Index of this pixel in the output for the block if it is active (sparse
    // mode only).
    bool pixelInImage{false};
    uint32_t activePixelIndex{0};
    cuda::std::tie(pixelInImage, activePixelIndex) = commonArgs.activePixelIndex(row, col);

    // If the caller provides a per-tile mask and this tile is masked, do nothing and return
    if (commonArgs.mHasMasks && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        return;
    }

    int32_t firstGaussianIdInBlock;
    int32_t lastGaussianIdInBlock;
    cuda::std::tie(firstGaussianIdInBlock, lastGaussianIdInBlock) =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);

    // Compute the backward pass for the current tile starting at pixel (i, j)
    // and containing Gaussians with ids in [firstGaussianIdInBlock, lastGaussianIdInBlock)
    constexpr uint32_t WARP_TILE_SIZE = 32; // TODO (fwilliams): Tune this value
    args.template volumeRenderTileBackward<WARP_TILE_SIZE>(cameraId,
                                                           row,
                                                           col,
                                                           firstGaussianIdInBlock,
                                                           lastGaussianIdInBlock,
                                                           blockDim.x * blockDim.y,
                                                           pixelInImage,
                                                           activePixelIndex);
}

/// @brief Get the shared memory requirements for the backward pass kernel
/// @param numColorChannels The number of color channels
/// @param tileSize The size of the tile
/// @return The shared memory required in bytes
template <typename ScalarType>
size_t
getSharedMemRequirements(const size_t numColorChannels, const size_t tileSize) {
    return tileSize * tileSize *
           (sizeof(Gaussian2D<ScalarType>) + numColorChannels * sizeof(ScalarType));
}

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardWithTemplatedSharedChannels(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,          // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,      // [totalIntersections]
    const fvdb::JaggedTensor &renderedAlphas,  // {C, [AP, 1]} or {1, [C, H, W, 1]}
    const fvdb::JaggedTensor &lastGaussianIds, // {C, [AP]} or {1, [C, H, W]}
    const fvdb::JaggedTensor
        &dLossDRenderedFeatures, // {C, [AP, NUM_CHANNELS]} or {1, [C, H, W, NUM_CHANNELS]}
    const fvdb::JaggedTensor &dLossDRenderedAlphas, // {C, [AP, 1]} or {1, [C, H, W, 1]}
    bool absGrad, // True if we are computing the gradient of the absolute gradient of the means2d
    at::cuda::CUDAStream stream,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    torch::Tensor outDLossDMeans2d   = torch::zeros_like(means2d);
    torch::Tensor outDLossDConics    = torch::zeros_like(conics);
    torch::Tensor outDLossDFeatures  = torch::zeros_like(features);
    torch::Tensor outDLossDOpacities = torch::zeros_like(opacities);
    torch::Tensor outDLossDMeans2dAbs;
    if (absGrad) {
        outDLossDMeans2dAbs = torch::zeros_like(means2d);
    }

    // Just return empty tensors if there are no gaussians, cameras, or intersections
    if (means2d.numel() == 0 || tileGaussianIds.numel() == 0) {
        return std::make_tuple(outDLossDMeans2dAbs,
                               outDLossDMeans2d,
                               outDLossDConics,
                               outDLossDFeatures,
                               outDLossDOpacities);
    }

    const uint32_t C = means2d.size(0);
    const uint32_t H = imageHeight;
    const uint32_t W = imageWidth;

    auto [reshapedRenderedAlphas,
          reshapedLastGaussianIds,
          reshapedDLossDRenderedFeatures,
          reshapedDLossDRenderedAlphas] = [&]() {
        if (!activeTiles.has_value()) {
            // Dense mode. Reshape the JaggedTensor inputs to match sparse mode
            return std::make_tuple(
                fvdb::JaggedTensor(renderedAlphas.jdata().view({C * H * W, 1})),
                fvdb::JaggedTensor(lastGaussianIds.jdata().view({C * H * W})),
                fvdb::JaggedTensor(dLossDRenderedFeatures.jdata().view({C * H * W, NUM_CHANNELS})),
                fvdb::JaggedTensor(dLossDRenderedAlphas.jdata().view({C * H * W, 1})));
        }
        return std::make_tuple(
            renderedAlphas, lastGaussianIds, dLossDRenderedFeatures, dLossDRenderedAlphas);
    }();

    RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args(
        means2d,
        conics,
        opacities,
        features,
        backgrounds,
        masks,
        imageWidth,
        imageHeight,
        imageOriginW,
        imageOriginH,
        tileSize,
        tileOffsets,
        tileGaussianIds,
        reshapedRenderedAlphas,
        reshapedLastGaussianIds,
        reshapedDLossDRenderedFeatures,
        reshapedDLossDRenderedAlphas,
        outDLossDMeans2d,
        outDLossDConics,
        outDLossDFeatures,
        outDLossDOpacities,
        absGrad ? std::make_optional(outDLossDMeans2dAbs) : std::nullopt,
        activeTiles,
        tilePixelMask,
        tilePixelCumsum,
        pixelMap);

    const size_t numChannels =
        (NUM_SHARED_CHANNELS == NUM_CHANNELS) ? NUM_CHANNELS : NUM_SHARED_CHANNELS + 1;
    const size_t sharedMemSize = getSharedMemRequirements<ScalarType>(numChannels, tileSize);

    if (cudaFuncSetAttribute(
            rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedMemSize) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMemSize,
                 " bytes), try lowering tileSize.");
    }
    rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>
        <<<args.getGridDim(), args.getBlockDim(), sharedMemSize, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(outDLossDMeans2dAbs,
                           outDLossDMeans2d,
                           outDLossDConics,
                           outDLossDFeatures,
                           outDLossDOpacities);
}

template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardWithCorrectSharedChannels(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,          // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,      // [totalIntersections]
    const fvdb::JaggedTensor &renderedAlphas,  // {C, [AP, 1]} or {1, [C, H, W, 1]}
    const fvdb::JaggedTensor &lastGaussianIds, // {C, [AP]} or {1, [C, H, W]}
    const fvdb::JaggedTensor
        &dLossDRenderedFeatures, // {C, [AP, NUM_CHANNELS]} or {1, [C, H, W, NUM_CHANNELS]}
    const fvdb::JaggedTensor &dLossDRenderedAlphas, // {C, [AP, 1]} or {1, [C, H, W, 1]}
    const bool absGrad,
    const int64_t numSharedSharedChannelsOverride,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    auto callWithSharedChannels = [&](size_t numSharedChannels) {
        if (numSharedChannels == NUM_CHANNELS) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    NUM_CHANNELS,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                imageWidth,
                imageHeight,
                imageOriginW,
                imageOriginH,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 64) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    64,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                imageWidth,
                imageHeight,
                imageOriginW,
                imageOriginH,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 32) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    32,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                imageWidth,
                imageHeight,
                imageOriginW,
                imageOriginH,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 16) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    16,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                imageWidth,
                imageHeight,
                imageOriginW,
                imageOriginH,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else {
            if (numSharedSharedChannelsOverride > 0) {
                AT_ERROR("Invalid numSharedChannelsOverride. Must be 64, 32, or 16.");
            } else {
                AT_ERROR("Failed to set maximum shared memory size");
            }
        }
    };

    if (numSharedSharedChannelsOverride > 0) {
        return callWithSharedChannels(numSharedSharedChannelsOverride);
    } else {
        cudaDeviceProp deviceProperties;
        if (cudaGetDeviceProperties(&deviceProperties, stream.device_index()) != cudaSuccess) {
            AT_ERROR("Failed to query device properties");
        }
        const size_t maxSharedMemory = deviceProperties.sharedMemPerBlockOptin;

        const size_t sharedMemChannelOptions[4] = {NUM_CHANNELS, 64, 32, 16};
        for (size_t i = 0; i < 4; ++i) {
            const size_t numSharedChannels = sharedMemChannelOptions[i];
            if (getSharedMemRequirements<ScalarType>(numSharedChannels, tileSize) <=
                maxSharedMemory) {
                return callWithSharedChannels(numSharedChannels);
            }
        }
        AT_ERROR("Failed to set maximum shared memory size");
    }
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward<torch::kCUDA>(
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,        // [totalIntersections]
    const torch::Tensor &renderedAlphas,         // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastGaussianIds,        // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, imageHeight, imageWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, imageHeight, imageWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride) {
    uint32_t colorDim   = features.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define __GS__CALL_BWD_(N)                                                          \
    case N: {                                                                       \
        if (isPacked) {                                                             \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, true>(  \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                at::nullopt /*backgrounds*/,                                        \
                at::nullopt /*masks*/,                                              \
                imageWidth,                                                         \
                imageHeight,                                                        \
                imageOriginW,                                                       \
                imageOriginH,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride);                                         \
        } else {                                                                    \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, false>( \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                at::nullopt /*backgrounds*/,                                        \
                at::nullopt /*masks*/,                                              \
                imageWidth,                                                         \
                imageHeight,                                                        \
                imageOriginW,                                                       \
                imageOriginH,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride);                                         \
        }                                                                           \
    }

    switch (colorDim) {
        __GS__CALL_BWD_(1)
        __GS__CALL_BWD_(2)
        __GS__CALL_BWD_(3)
        __GS__CALL_BWD_(4)
        __GS__CALL_BWD_(5)
        __GS__CALL_BWD_(8)
        __GS__CALL_BWD_(9)
        __GS__CALL_BWD_(16)
        __GS__CALL_BWD_(17)
        __GS__CALL_BWD_(32)
        __GS__CALL_BWD_(33)
        __GS__CALL_BWD_(47) // TODO, is this only here to support a gtest?
        __GS__CALL_BWD_(64)
        __GS__CALL_BWD_(65)
        __GS__CALL_BWD_(128)
        __GS__CALL_BWD_(129)
        __GS__CALL_BWD_(192)
        __GS__CALL_BWD_(193)
        __GS__CALL_BWD_(256)
        __GS__CALL_BWD_(257)
        __GS__CALL_BWD_(512)
        __GS__CALL_BWD_(513)
    default: AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward<torch::kCPU>(
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,        // [totalIntersections]
    const torch::Tensor &renderedAlphas,         // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastGaussianIds,        // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, imageHeight, imageWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, imageHeight, imageWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianSparseRasterizeBackward<torch::kCUDA>(
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    const torch::Tensor &means2d,             // [C, N, 2]
    const torch::Tensor &conics,              // [C, N, 3]
    const torch::Tensor &features,            // [C, N, D]
    const torch::Tensor &opacities,           // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,                 // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,             // [n_isects]
    const fvdb::JaggedTensor &renderedAlphas,         // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastGaussianIds,        // [C lists: varying sizes]
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    const torch::Tensor &activeTiles,                 // [AT]
    const torch::Tensor &tilePixelMask,               // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum,             // [AT]
    const torch::Tensor &pixelMap,                    // [AP]
    const bool absGrad,
    const int64_t numSharedChannelsOverride) {
    uint32_t colorDim   = features.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define __GS__CALL_SPARSE_BWD_(N)                                                   \
    case N: {                                                                       \
        if (isPacked) {                                                             \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, true>(  \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                at::nullopt /*backgrounds*/,                                        \
                at::nullopt /*masks*/,                                              \
                imageWidth,                                                         \
                imageHeight,                                                        \
                imageOriginW,                                                       \
                imageOriginH,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride,                                          \
                activeTiles,                                                        \
                tilePixelMask,                                                      \
                tilePixelCumsum,                                                    \
                pixelMap);                                                          \
        } else {                                                                    \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, false>( \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                at::nullopt /*backgrounds*/,                                        \
                at::nullopt /*masks*/,                                              \
                imageWidth,                                                         \
                imageHeight,                                                        \
                imageOriginW,                                                       \
                imageOriginH,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride,                                          \
                activeTiles,                                                        \
                tilePixelMask,                                                      \
                tilePixelCumsum,                                                    \
                pixelMap);                                                          \
        }                                                                           \
    }

    switch (colorDim) {
        __GS__CALL_SPARSE_BWD_(1)
        __GS__CALL_SPARSE_BWD_(2)
        __GS__CALL_SPARSE_BWD_(3)
        __GS__CALL_SPARSE_BWD_(4)
        __GS__CALL_SPARSE_BWD_(5)
        __GS__CALL_SPARSE_BWD_(8)
        __GS__CALL_SPARSE_BWD_(9)
        __GS__CALL_SPARSE_BWD_(16)
        __GS__CALL_SPARSE_BWD_(17)
        __GS__CALL_SPARSE_BWD_(32)
        __GS__CALL_SPARSE_BWD_(33)
        __GS__CALL_SPARSE_BWD_(47)
        __GS__CALL_SPARSE_BWD_(64)
        __GS__CALL_SPARSE_BWD_(65)
        __GS__CALL_SPARSE_BWD_(128)
        __GS__CALL_SPARSE_BWD_(129)
        __GS__CALL_SPARSE_BWD_(192)
        __GS__CALL_SPARSE_BWD_(193)
        __GS__CALL_SPARSE_BWD_(256)
        __GS__CALL_SPARSE_BWD_(257)
        __GS__CALL_SPARSE_BWD_(512)
        __GS__CALL_SPARSE_BWD_(513)
    default: AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianSparseRasterizeBackward<torch::kCPU>(
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    const torch::Tensor &means2d,             // [C, N, 2]
    const torch::Tensor &conics,              // [C, N, 3]
    const torch::Tensor &features,            // [C, N, D]
    const torch::Tensor &opacities,           // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,                 // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,             // [n_isects]
    const fvdb::JaggedTensor &renderedAlphas,         // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastIds,                // [C lists: varying sizes]
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    const torch::Tensor &activeTiles,                 // [AT]
    const torch::Tensor &tilePixelMask,               // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum,             // [AT]
    const torch::Tensor &pixelMap,                    // [AP]
    const bool absGrad,
    const int64_t numSharedChannelsOverride) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
