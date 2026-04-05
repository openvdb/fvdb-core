// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianCameraValidation.h>

namespace fvdb::detail::ops {

void
validateCameraProjectionArgs(const torch::Tensor &worldToCameraMatrices,
                             const torch::Tensor &projectionMatrices,
                             const DistortionModel cameraModel,
                             const ProjectionMethod requestedProjectionMethod,
                             const std::optional<torch::Tensor> &distortionCoeffs) {
    const int64_t C = worldToCameraMatrices.size(0);
    TORCH_CHECK(C > 0, "At least one camera must be provided (got 0)");
    TORCH_CHECK(worldToCameraMatrices.sizes() == torch::IntArrayRef({C, 4, 4}),
                "worldToCameraMatrices must have shape (C, 4, 4)");
    TORCH_CHECK(projectionMatrices.sizes() == torch::IntArrayRef({C, 3, 3}),
                "projectionMatrices must have shape (C, 3, 3)");
    TORCH_CHECK(worldToCameraMatrices.is_contiguous(), "worldToCameraMatrices must be contiguous");
    TORCH_CHECK(projectionMatrices.is_contiguous(), "projectionMatrices must be contiguous");

    const ProjectionMethod resolvedProjectionMethod =
        resolveProjectionMethod(cameraModel, requestedProjectionMethod);

    if (distortionCoeffs.has_value()) {
        TORCH_CHECK(distortionCoeffs->sizes() == torch::IntArrayRef({C, 12}),
                    "distortionCoeffs must have shape (C, 12)");
        TORCH_CHECK(distortionCoeffs->is_contiguous(), "distortionCoeffs must be contiguous");
    }

    if (usesOpenCVDistortion(cameraModel)) {
        TORCH_CHECK(distortionCoeffs.has_value(),
                    "distortionCoeffs must be provided for OpenCV camera models");
        TORCH_CHECK(resolvedProjectionMethod == ProjectionMethod::UNSCENTED,
                    "OpenCV camera models require ProjectionMethod::UNSCENTED or AUTO");
    }
}

void
checkGaussianState(const torch::Tensor &means,
                   const torch::Tensor &quats,
                   const torch::Tensor &logScales,
                   const torch::Tensor &logitOpacities,
                   const torch::Tensor &sh0,
                   const torch::Tensor &shN) {
    const int64_t N = means.size(0); // number of gaussians

    TORCH_CHECK_VALUE(means.sizes() == torch::IntArrayRef({N, 3}), "means must have shape (N, 3)");
    TORCH_CHECK_VALUE(quats.sizes() == torch::IntArrayRef({N, 4}), "quats must have shape (N, 4)");
    TORCH_CHECK_VALUE(logScales.sizes() == torch::IntArrayRef({N, 3}),
                      "scales must have shape (N, 3)");
    TORCH_CHECK_VALUE(logitOpacities.sizes() == torch::IntArrayRef({N}),
                      "opacities must have shape (N)");
    TORCH_CHECK_VALUE(sh0.size(0) == N, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(sh0.size(1) == 1, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(sh0.dim() == 3, "sh0 must have shape (N, 1, D)");
    TORCH_CHECK_VALUE(shN.size(0) == N, "shN must have shape (N, K-1, D)");
    TORCH_CHECK_VALUE(shN.dim() == 3, "shN must have shape (N, K-1, D)");

    TORCH_CHECK_VALUE(means.device() == quats.device(), "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == logScales.device(),
                      "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == logitOpacities.device(),
                      "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == sh0.device(), "All tensors must be on the same device");
    TORCH_CHECK_VALUE(means.device() == shN.device(), "All tensors must be on the same device");

    TORCH_CHECK_VALUE(torch::isFloatingType(means.scalar_type()),
                      "All tensors must be of floating point type");
    TORCH_CHECK_VALUE(means.scalar_type() == quats.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == logScales.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == logitOpacities.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == sh0.scalar_type(),
                      "All tensors must be of the same type");
    TORCH_CHECK_VALUE(means.scalar_type() == shN.scalar_type(),
                      "All tensors must be of the same type");
}

} // namespace fvdb::detail::ops
