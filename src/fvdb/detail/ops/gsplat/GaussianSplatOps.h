// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>

#include <ATen/core/TensorBody.h>

#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Check whether a camera model uses OpenCV-style distortion.
inline bool
usesOpenCVDistortion(const DistortionModel cameraModel) {
    return cameraModel == DistortionModel::OPENCV_RADTAN_5 ||
           cameraModel == DistortionModel::OPENCV_RATIONAL_8 ||
           cameraModel == DistortionModel::OPENCV_RADTAN_THIN_PRISM_9 ||
           cameraModel == DistortionModel::OPENCV_THIN_PRISM_12;
}

/// @brief Resolve the projection method for a given camera model.
///        AUTO is resolved to UNSCENTED for OpenCV models, and ANALYTIC otherwise.
inline ProjectionMethod
resolveProjectionMethod(const DistortionModel cameraModel,
                        const ProjectionMethod projectionMethod) {
    if (projectionMethod == ProjectionMethod::AUTO) {
        return usesOpenCVDistortion(cameraModel) ? ProjectionMethod::UNSCENTED
                                                 : ProjectionMethod::ANALYTIC;
    }
    return projectionMethod;
}

/// @brief Validate the camera projection arguments (matrix shapes, distortion coefficients, etc.).
void validateCameraProjectionArgs(const torch::Tensor &worldToCameraMatrices,
                                  const torch::Tensor &projectionMatrices,
                                  DistortionModel cameraModel,
                                  ProjectionMethod requestedProjectionMethod,
                                  const std::optional<torch::Tensor> &distortionCoeffs);

/// @brief Validate tensor shapes, devices, and types for Gaussian splat state.
void checkGaussianState(const torch::Tensor &means,
                        const torch::Tensor &quats,
                        const torch::Tensor &logScales,
                        const torch::Tensor &logitOpacities,
                        const torch::Tensor &sh0,
                        const torch::Tensor &shN);

/// @brief Deduplicate pixel coordinates in a JaggedTensor.
///
/// Encodes each pixel as a single int64 key incorporating its batch index and 2D coordinate,
/// sorts keys to find unique groups and builds an inverse mapping. Returns the deduplicated
/// pixels as a new JaggedTensor, the inverse index tensor, and a flag indicating whether any
/// duplicates were found.
///
/// @param pixelsToRender The input JaggedTensor of pixel coordinates [total_pixels, 2]
/// @param imageWidth Width of each image in pixels
/// @param imageHeight Height of each image in pixels
/// @return Tuple of (uniquePixels JaggedTensor, inverseIndices tensor, hasDuplicates bool)
std::tuple<fvdb::JaggedTensor, torch::Tensor, bool>
deduplicatePixels(const fvdb::JaggedTensor &pixelsToRender,
                  int64_t imageWidth,
                  int64_t imageHeight);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATOPS_H
