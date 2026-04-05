// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Standalone structs for Gaussian projection results, used by the projection
// pipeline, rasterization ops, and pybind bindings.
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/types.h>

#include <string>
#include <variant>

namespace fvdb {

/// @brief Type alias for PLY metadata values.
using PlyMetadataTypes = std::variant<std::string, int64_t, double, torch::Tensor>;

/// @brief A set of projected Gaussians that can be used to render images.
struct ProjectedGaussianSplats {
    torch::Tensor perGaussian2dMean;         // [C, N, 2]
    torch::Tensor perGaussianConic;          // [C, N, 3]
    torch::Tensor perGaussianRenderQuantity; // [C, N, 3]
    torch::Tensor perGaussianDepth;          // [C, N, 1]
    torch::Tensor perGaussianOpacity;        // [N] or [C, N] if antialias is true
    torch::Tensor perGaussianRadius;         // [C, N]
    torch::Tensor tileOffsets;               // [C, num_tiles_h, num_tiles_w, 2]
    torch::Tensor tileGaussianIds; // [C, num_tiles_h, num_tiles_w, max_gaussians_per_tile]

    detail::ops::RenderSettings mRenderSettings;
    detail::ops::DistortionModel mCameraModel       = detail::ops::DistortionModel::PINHOLE;
    detail::ops::ProjectionMethod mProjectionMethod = detail::ops::ProjectionMethod::ANALYTIC;

    ssize_t
    imageHeight() const {
        return mRenderSettings.imageHeight;
    }

    ssize_t
    imageWidth() const {
        return mRenderSettings.imageWidth;
    }

    float
    nearPlane() const {
        return mRenderSettings.nearPlane;
    }

    float
    farPlane() const {
        return mRenderSettings.farPlane;
    }

    detail::ops::DistortionModel
    cameraModel() const {
        return mCameraModel;
    }

    detail::ops::ProjectionMethod
    projectionMethod() const {
        return mProjectionMethod;
    }

    int64_t
    shDegreeToUse() const {
        return mRenderSettings.shDegreeToUse;
    }

    float
    minRadius2d() const {
        return mRenderSettings.radiusClip;
    }

    float
    eps2d() const {
        return mRenderSettings.eps2d;
    }

    bool
    antialias() const {
        return mRenderSettings.antialias;
    }

    torch::Tensor
    means2d() const {
        return perGaussian2dMean;
    }

    torch::Tensor
    conics() const {
        return perGaussianConic;
    }

    torch::Tensor
    renderQuantities() const {
        return perGaussianRenderQuantity;
    }

    torch::Tensor
    depths() const {
        return perGaussianDepth;
    }

    torch::Tensor
    opacities() const {
        if (perGaussianOpacity.dim() == 1) {
            const int64_t C = perGaussian2dMean.size(0);
            return perGaussianOpacity.unsqueeze(0).expand({C, -1});
        }
        return perGaussianOpacity;
    }

    torch::Tensor
    radii() const {
        return perGaussianRadius;
    }

    torch::Tensor
    offsets() const {
        return tileOffsets;
    }

    torch::Tensor
    gaussianIds() const {
        return tileGaussianIds;
    }
};

/// @brief A set of projected Gaussians with sparse tile intersection data for sparse rendering.
struct SparseProjectedGaussianSplats : public ProjectedGaussianSplats {
    torch::Tensor activeTiles;         // [num_active_tiles]
    torch::Tensor activeTileMask;      // [C, TH, TW]
    torch::Tensor tilePixelMask;       // [num_active_tiles, words_per_tile]
    torch::Tensor tilePixelCumsum;     // [num_active_tiles]
    torch::Tensor pixelMap;            // [num_active_pixels]

    torch::Tensor inverseIndices;      // [total_pixels]
    JaggedTensor uniquePixelsToRender; // deduplicated pixels
    bool hasDuplicates = false;
};

} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONTYPES_H
