// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GAUSSIANSPLAT3D_H
#define FVDB_GAUSSIANSPLAT3D_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <torch/types.h>

namespace fvdb {

/// @brief A class representing a Gaussian splat scene in 3D space.
/// This class is used to store the parameters of the Gaussians in the scene and provides
/// methods to project the Gaussians onto a 2D image plane, render images and depths,
/// and save the scene to a PLY file.
/// The Gaussians are represented by their means, quaternions (for rotation), log scales,
/// logit opacities, and SH coefficients. We use log_scales and logit_opacities since we can
/// optimize these quantities without clipping them to a specific range.
class GaussianSplat3d {
  public:
    /// Magic string prepended to additional metadata properties stored in PLY files
    inline static const std::string PLY_MAGIC = "fvdb_ply_af_8198767135";

    /// We won't allow keys in a PLY file longer than this many characters.
    inline static const size_t MAX_PLY_KEY_LENGTH = 256;

    inline static const std::string PLY_VERSION_STRING = "fvdb_ply 1.0.0";

    using PlyMetadataTypes = std::variant<std::string, int64_t, double, torch::Tensor>;

    GaussianSplat3d(const torch::Tensor &means,
                    const torch::Tensor &quats,
                    const torch::Tensor &logScales,
                    const torch::Tensor &logitOpacities,
                    const torch::Tensor &sh0,
                    const torch::Tensor &shN,
                    const bool accumulateMean2dGradients,
                    const bool accumulateMax2dRadii,
                    const bool detach);

    /// @brief Create a GaussianSplat3d object from a state_dict (similar to Pytorch's nn.Module).
    /// @param stateDict A dictionary containing the state of the GaussianSplat3d object.
    /// @return A GaussianSplat3d object created from the state_dict.
    GaussianSplat3d(const std::unordered_map<std::string, torch::Tensor> &stateDict) {
        loadStateDict(stateDict);
    }

    using ProjectionType = fvdb::detail::ops::ProjectionType;

    /// @brief A set of projected Gaussians that can be used to render images.
    struct ProjectedGaussianSplats {
        torch::Tensor perGaussian2dMean;         // [C, N, 2]
        torch::Tensor perGaussianConic;          // [C, N, 3]
        torch::Tensor perGaussianRenderQuantity; // [C, N, 3]
        torch::Tensor perGaussianDepth;          // [C, N, 1]
        torch::Tensor perGaussianOpacity;        // [N]
        torch::Tensor perGaussianRadius;         // [C, N]
        torch::Tensor tileOffsets;               // [C, num_tiles_h, num_tiles_w, 2]
        torch::Tensor tileGaussianIds; // [C, num_tiles_h, num_tiles_w, max_gaussians_per_tile]

        fvdb::detail::ops::RenderSettings mRenderSettings;

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

        ProjectionType
        projectionType() const {
            return mRenderSettings.projectionType;
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

  public:
    /// @brief Concatenate a vector of GaussianSplat3d objects into a single GaussianSplat3d object.
    /// @param splats A vector of GaussianSplat3d objects to concatenate.
    /// @param accumulateMean2dGradients Whether to accumulate the mean 2D gradients for each
    /// Gaussian.
    ///     For splats that do not have mean2d gradients, zeros will be copied to the means2d
    ///     gradient norm state in the output.
    /// @param accumulateMax2dRadii Whether to accumulate the maximum 2D radii for each Gaussian.
    ///     For splats that do not have 2D radii, zeros will be copied to the radii state in the
    ///     output.
    /// @return A new GaussianSplat3d object that is the concatenation of the input splats.
    static GaussianSplat3d
    cat(const std::vector<GaussianSplat3d> &splats,
        bool accumulateMean2dGradients,
        bool accumulateMax2dRadii,
        bool detach) {
        TORCH_CHECK_VALUE(!splats.empty(), "Cannot concatenate an empty vector of splats");

        std::vector<torch::Tensor> meansVec, quatsVec, logScalesVec, logitOpacitiesVec, sh0Vec,
            shNVec;

        std::vector<torch::Tensor> accStepCountsVec, accMax2dRadiiVec, accNorm2dMeansGradientsVec;

        const auto device = splats[0].device();
        const auto dtype  = splats[0].scalarType();

        for (const auto &splat: splats) {
            TORCH_CHECK_VALUE(splat.device() == device, "All splats must be on the same device");
            TORCH_CHECK_VALUE(splat.scalarType() == dtype, "All splats must be of the same type");

            meansVec.push_back(splat.mMeans);
            quatsVec.push_back(splat.mQuats);
            logScalesVec.push_back(splat.mLogScales);
            logitOpacitiesVec.push_back(splat.mLogitOpacities);
            sh0Vec.push_back(splat.mSh0);
            shNVec.push_back(splat.mShN);

            const auto N = splat.numGaussians();
            if (accumulateMean2dGradients) {
                auto [accNorm2dMeansGradients, accGradientStepCounts] = [&]() {
                    if (splat.mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()) {
                        TORCH_CHECK(
                            splat.mAccumulatedNormalized2dMeansGradientNormsForGrad.numel() == N,
                            "accumulated_mean_2d_gradient_norms_for_grad must have shape (N)");
                        TORCH_CHECK(
                            splat.mAccumulatedNormalized2dMeansGradientNormsForGrad.device() ==
                                splat.device(),
                            "accumulated_mean_2d_gradient_norms_for_grad must be on the same device as "
                            "means");
                        TORCH_CHECK(splat.mGradientStepCountForGrad.defined(),
                                    "gradient_step_counts_for_grad must be non-empty if "
                                    "accumulated_mean_2d_gradient_norms_for_grad is non-empty");
                        TORCH_CHECK(
                            splat.mGradientStepCountForGrad.numel() == N,
                            "accumulated_gradient_step_counts_for_grad must have shape (N)");
                        return std::make_tuple(
                            splat.mAccumulatedNormalized2dMeansGradientNormsForGrad,
                            splat.mGradientStepCountForGrad);
                    } else {
                        return std::make_tuple(torch::zeros({N}, splat.mMeans.options()),
                                               torch::zeros({N}, torch::kInt32).to(splat.device()));
                    }
                }();
                accNorm2dMeansGradientsVec.push_back(accNorm2dMeansGradients);
                accStepCountsVec.push_back(accGradientStepCounts);
            }
            if (accumulateMax2dRadii) {
                if (splat.mAccumulated2dRadiiForGrad.defined()) {
                    TORCH_CHECK(splat.mAccumulated2dRadiiForGrad.numel() == N,
                                "accumulated_max_2d_radii_for_grad must have shape (N)");
                    TORCH_CHECK(
                        splat.mAccumulated2dRadiiForGrad.device() == splat.device(),
                        "accumulated_max_2d_radii_for_grad must be on the same device as means");
                    accMax2dRadiiVec.push_back(splat.mAccumulated2dRadiiForGrad);
                } else {
                    accMax2dRadiiVec.push_back(torch::zeros({N}, torch::kInt32).to(splat.device()));
                }
            }
        }

        torch::Tensor meansCat          = torch::cat(meansVec, 0);
        torch::Tensor quatsCat          = torch::cat(quatsVec, 0);
        torch::Tensor logScalesCat      = torch::cat(logScalesVec, 0);
        torch::Tensor logitOpacitiesCat = torch::cat(logitOpacitiesVec, 0);
        torch::Tensor sh0Cat            = torch::cat(sh0Vec, 0);
        torch::Tensor shNCat            = torch::cat(shNVec, 0);

        auto ret = GaussianSplat3d(meansCat,
                                   quatsCat,
                                   logScalesCat,
                                   logitOpacitiesCat,
                                   sh0Cat,
                                   shNCat,
                                   accumulateMean2dGradients,
                                   accumulateMax2dRadii,
                                   detach);

        if (accumulateMean2dGradients) {
            auto catNorm2dGradMeans = torch::cat(accNorm2dMeansGradientsVec, 0);
            if (detach) {
                catNorm2dGradMeans = catNorm2dGradMeans.detach();
            }
            auto catStepCounts = torch::cat(accStepCountsVec, 0);
            if (detach) {
                catStepCounts = catStepCounts.detach();
            }
            ret.mAccumulatedNormalized2dMeansGradientNormsForGrad = catNorm2dGradMeans;
            ret.mGradientStepCountForGrad                         = catStepCounts;
        }
        if (accumulateMax2dRadii) {
            auto catMax2dRadii = torch::cat(accMax2dRadiiVec, 0);
            if (detach) {
                catMax2dRadii = catMax2dRadii.detach();
            }
            ret.mAccumulated2dRadiiForGrad = catMax2dRadii;
        }

        return ret;
    }

    /// @brief Get the device this Gaussian splat is on.
    /// @return The device of the means tensor.
    torch::Device
    device() const {
        TORCH_CHECK(mMeans.device() == mQuats.device(),
                    "All tensors must be on the same device. Means and quats must match.");
        TORCH_CHECK(mMeans.device() == mLogScales.device(),
                    "All tensors must be on the same device. Means and log scales must match.");
        TORCH_CHECK(
            mMeans.device() == mLogitOpacities.device(),
            "All tensors must be on the same device. Means and logit opacities must match.");
        TORCH_CHECK(mMeans.device() == mSh0.device(),
                    "All tensors must be on the same device. Means and SH0 must match.");
        TORCH_CHECK(mMeans.device() == mShN.device(),
                    "All tensors must be on the same device. Means and SHN must match.");
        return mMeans.device();
    }

    /// @brief Get the scalar type of the tensors in this Gaussian splat.
    /// @return The scalar type of the means tensor.
    ///         All tensors are expected to have the same scalar type.
    torch::ScalarType
    scalarType() const {
        TORCH_CHECK(mMeans.scalar_type() == mQuats.scalar_type(),
                    "All tensors must be of the same type. Means and quats must match.");
        TORCH_CHECK(mMeans.scalar_type() == mLogScales.scalar_type(),
                    "All tensors must be of the same type. Means and log scales must match.");
        TORCH_CHECK(mMeans.scalar_type() == mLogitOpacities.scalar_type(),
                    "All tensors must be of the same type. Means and logit opacities must match.");
        TORCH_CHECK(mMeans.scalar_type() == mSh0.scalar_type(),
                    "All tensors must be of the same type. Means and SH0 must match.");
        TORCH_CHECK(mMeans.scalar_type() == mShN.scalar_type(),
                    "All tensors must be of the same type. Means and SHN must match.");
        return mMeans.scalar_type();
    }

    /// @brief Return the means of the Gaussians in this scene.
    /// @return An [N, 3]-shaped tensor representing the means of the Gaussians in this scenes.
    torch::Tensor
    means() const {
        return mMeans;
    }

    /// @brief Return the quaternions of the Gaussians in this scene which define the rotation
    ///        component of the covariance of each Gaussian (in the form [x, y, z, w]).
    /// @return An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this scene.
    torch::Tensor
    quats() const {
        return mQuats;
    }

    /// @brief Return the log of the scales of the Gaussians in this scene.
    /// @return An [N]-shaped tensor representing the log of the scales of the
    ///         Gaussians in this scene.
    torch::Tensor
    logScales() const {
        return mLogScales;
    }

    /// @brief Return the logit (inverse of Sigmoid) of the opacities of the Gaussians in this
    ///        scene.
    /// @return An [N]-shaped tensor representing the logit of the opacities of the
    ///         Gaussians in this scene.
    torch::Tensor
    logitOpacities() const {
        return mLogitOpacities;
    }

    /// @brief Return the diffuse SH coefficients of the Gaussians in this scene
    /// @return An [N, 1, D]-shaped tensor representing the diffuse SH coefficients of the
    /// Gaussians in this scene.
    torch::Tensor
    sh0() const {
        return mSh0;
    }

    /// @brief Return the directionally-dependent SH coefficients of the Gaussians in this scene
    /// @return A [N, K-1, D]-shaped tensor representing the directionally-dependent SH
    ///         coefficients of the Gaussians in this scene.
    torch::Tensor
    shN() const {
        return mShN;
    }

    /// @brief Return the scales of the Gaussians in this scene.
    /// @return An [N, 3]-shaped tensor representing the scales of the Gaussians in this scene.
    ///         (i.e. exp(logScales)).
    torch::Tensor
    scales() const {
        return torch::exp(mLogScales);
    }

    /// @brief Return the opacities of the Gaussians in this scene.
    /// @return An [N]-shaped tensor representing the opacities of the Gaussians in this scene.
    ///         (i.e. sigmoid(logitOpacities)).
    torch::Tensor
    opacities() const {
        return torch::sigmoid(mLogitOpacities);
    }

    int64_t
    shDegree() const {
        // The SH degree is determined by the number of SH coefficients in shN.
        // If shN is empty, we return -1 to indicate that no SH coefficients are used.
        const auto K        = mShN.size(1) + 1; // number of SH bases
        const auto shDegree = static_cast<int64_t>(std::sqrt(K) - 1);
        return shDegree;
    }

    /// @brief Return a copy of this GaussianSplat3d object with the same parameters, but detached
    ///        from the computation graph.
    /// @return A new instance of GaussianSplat3d with the same parameters, but detached from the
    ///         computation graph.
    GaussianSplat3d
    detach() const {
        return GaussianSplat3d(mMeans.detach(),
                               mQuats.detach(),
                               mLogScales.detach(),
                               mLogitOpacities.detach(),
                               mSh0.detach(),
                               mShN.detach(),
                               mAccumulateMean2dGradients,
                               mAccumulateMax2dRadii,
                               false);
    }

    /// @brief Return a copy of this GaussianSplat3d object with the same parameters, but moved to
    ///        the specified device and dtype, or return *this if the device and dtype match this.
    /// @param device The device to move the tensors to.
    /// @param dtype The data type to convert the tensors to.
    /// @return A new instance of GaussianSplat3d with the same parameters, but moved to the
    ///         specified device and dtype, or *this if the device and dtype match this.
    GaussianSplat3d
    to(torch::Device device, torch::ScalarType dtype) {
        if (this->device() == device && this->scalarType() == dtype) {
            return *this; // No need to copy if already on the right device and type
        } else {
            auto ret = GaussianSplat3d(
                mMeans.to(device, dtype),
                mQuats.to(device, dtype),
                mLogScales.to(device, dtype),
                mLogitOpacities.to(device, dtype),
                mSh0.to(device, dtype),
                mShN.to(device, dtype),
                mAccumulateMean2dGradients,
                mAccumulateMax2dRadii,
                false // Detach is false since we are copying the data (not detaching it)
            );
            if (mAccumulated2dRadiiForGrad.defined()) {
                ret.mAccumulated2dRadiiForGrad = mAccumulated2dRadiiForGrad.to(device);
            }
            if (mAccumulatedNormalized2dMeansGradientNormsForGrad.defined()) {
                ret.mAccumulatedNormalized2dMeansGradientNormsForGrad =
                    mAccumulatedNormalized2dMeansGradientNormsForGrad.to(device, dtype);
            }
            if (mGradientStepCountForGrad.defined()) {
                ret.mGradientStepCountForGrad = mGradientStepCountForGrad.to(device);
            }
            return ret;
        }
    }

    /// @brief Detach the parameters of this GaussianSplat3d object in place.
    ///        This will detach the parameters from the computation graph, allowing them to be
    ///        modified without affecting the gradients of the original tensors.
    void
    detachInPlace() {
        mMeans.detach_();
        mQuats.detach_();
        mLogScales.detach_();
        mLogitOpacities.detach_();
        mSh0.detach_();
        mShN.detach_();
    }

    /// @brief Set the log of the opacities of the Gaussians in this scene.
    /// @param logitOpacities An [N]-shaped tensor representing the log of the opacities of the
    ///                     Gaussians in this scene.
    void
    setLogitOpacities(const torch::Tensor &logitOpacities) {
        TORCH_CHECK_VALUE(logitOpacities.sizes() == mLogitOpacities.sizes(),
                          "logit_opacities must have the same shape as the current opacities");
        TORCH_CHECK_VALUE(
            logitOpacities.device() == mLogitOpacities.device(),
            "logit_opacities must be on the same device as the current logit_opacities");
        mLogitOpacities = logitOpacities;
    }

    /// @brief Set the log of the scales of the Gaussians in this scene.
    /// @param logScales An [N, 3]-shaped tensor representing the log of the scales of the
    void
    setLogScales(const torch::Tensor &logScales) {
        TORCH_CHECK_VALUE(logScales.sizes() == mLogScales.sizes(),
                          "log_scales must have the same shape as the current scales");
        TORCH_CHECK_VALUE(logScales.device() == mLogScales.device(),
                          "log_scales must be on the same device as the current log_scales");
        mLogScales = logScales;
    }

    /// @brief Set the quaternions of the Gaussians in this scene which define the rotation
    ///        component of the covariance of each Gaussian (in the form [x, y, z, w]).
    /// @param quats An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this
    ///              scene.
    void
    setQuats(const torch::Tensor &quats) {
        TORCH_CHECK_VALUE(quats.sizes() == mQuats.sizes(),
                          "quats must have the same shape as the current quats");
        TORCH_CHECK_VALUE(quats.device() == mQuats.device(),
                          "quats must be on the same device as the current quats");
        mQuats = quats;
    }

    /// @brief Set the means of the Gaussians in this scene.
    /// @param means An [N, 3]-shaped tensor representing the means of the Gaussians in this scene.
    void
    setMeans(const torch::Tensor &means) {
        TORCH_CHECK_VALUE(means.sizes() == mMeans.sizes(),
                          "means must have the same shape as the current means");
        TORCH_CHECK_VALUE(means.device() == mMeans.device(),
                          "means must be on the same device as the current means");
        mMeans = means;
    }

    /// @brief Set the diffuse SH coefficients of the Gaussians in this scene.
    /// @param sh0 An [N, 1, D]-shaped tensor representing the diffuse SH coefficients of the
    ///            Gaussians in this scene.
    void
    setSh0(const torch::Tensor &sh0) {
        TORCH_CHECK_VALUE(sh0.sizes() == mSh0.sizes(),
                          "sh0 must have the same shape as the current sh0");
        TORCH_CHECK_VALUE(sh0.device() == mSh0.device(),
                          "sh0 must be on the same device as the current sh0");
        mSh0 = sh0;
    }

    /// @brief Set the directionally-dependent SH coefficients of the Gaussians in this scene.
    /// @param shN A [N, K-1, D]-shaped tensor representing the directionally-dependent SH
    ///            coefficients of the Gaussians in this scene.
    void
    setShN(const torch::Tensor &shN) {
        TORCH_CHECK_VALUE(shN.sizes() == mShN.sizes(),
                          "shN must have the same shape as the current shN");
        TORCH_CHECK_VALUE(shN.device() == mShN.device(),
                          "shN must be on the same device as the current shN");
        mShN = shN;
    }

    /// @brief Return whether to track the maximum 2D radii of each Gaussian over backward passes
    ///        of projection.
    /// @return True if the maximum 2D radii are tracked, false otherwise.
    /// @note This is used by some optimizers to decide whether to split/delete/duplicate Gaussians.
    ///       If this is set to true, the maximum 2D radii will be accumulated during the
    ///       backward pass of projection.
    bool
    accumulateMax2dRadii() const {
        return mAccumulateMax2dRadii;
    }

    /// @brief Set whether to accumulate the maximum 2D radii of each Gaussian over backward passes
    ///        of projection.
    /// @param accumulateMax2dRadii Whether to accumulate the maximum 2D radii of each Gaussian
    ///                             over backward passes of projection.
    /// @note This is used by some optimizers to decide whether to split/delete/duplicate Gaussians.
    ///       If this is set to true, the maximum 2D radii will be accumulated during the
    ///       backward pass of projection.
    void
    setAccumulateMax2dRadii(bool accumulateMax2dRadii) {
        mAccumulateMax2dRadii = accumulateMax2dRadii;
    }

    /// @brief Return whether to accumulate the means 2D gradients of each Gaussian over backward
    ///        passes of projection.
    /// @return True if the means 2D gradients are accumulated, false otherwise.
    /// @note This is used by some optimizers to decide whether to split/delete/duplicate Gaussians.
    ///       If this is set to true, the average norm of the gradient of projected means
    ///       for each Gaussian will be accumulated during the backward pass of projection.
    ///       This is useful for some optimization techniques.
    bool
    accumulateMean2dGradients() const {
        return mAccumulateMean2dGradients;
    }

    /// @brief Set whether to accumulate the means 2D gradients of each Gaussian over backward
    ///        passes of projection.
    /// @param accumulateMean2dGradients Whether to accumulate the means 2D gradients
    ///                                   of each Gaussian over backward passes of projection.
    /// @note This is used by some optimizers to decide whether to split/delete/duplicate Gaussians.
    ///       If this is set to true, the average norm of the gradient
    ///       of projected means for each Gaussian will be accumulated during the backward pass
    ///       of projection. This is useful for some optimization techniques.
    void
    setAccumulateMean2dGradients(bool accumulateMean2dGradients) {
        mAccumulateMean2dGradients = accumulateMean2dGradients;
    }

    /// @brief Return true if all tensors tracked by this object require gradients.
    /// @return True if all tensors tracked by this object require gradients, false otherwise.
    /// @note This function checks if all tensors are leaf tensors and have requires_grad set to
    /// true.
    ///       If any of the tensors are non-leaf tensors, this function will return false.
    ///       If you want to check if the tensors require gradients individually, you can use
    ///       the `requires_grad()` method on each tensor directly.
    /// @note If you want to ensure all tensors are leaf tensors, you can create a
    ///       new GaussianSplat3d object with the `detach` flag set to `True` when
    ///       creating the object.
    bool
    requiresGrad() const {
        return mMeans.requires_grad() && mQuats.requires_grad() && mLogScales.requires_grad() &&
               mLogitOpacities.requires_grad() && mSh0.requires_grad() && mShN.requires_grad();
    }

    /// @brief Set requires_grad on all tensors managed by this object.
    /// @param requiresGrad Whether the tensors should require gradients.
    /// @note This function will throw an error if any of the tensors are non-leaf tensors.
    ///       If you want to set requires_grad on specific tensors, set them on the tensors directly
    ///       instead of using this function.
    /// @note If you want to ensure all tensors are leaf tensors, you can call .detach() or create a
    ///       new GaussianSplat3d object with the `detach` flag set to `True` when
    ///       creating the object.
    void
    setRequiresGrad(bool requiresGrad) {
        TORCH_CHECK_VALUE(
            mMeans.is_leaf(),
            "Cannot set requires_grad of means which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        TORCH_CHECK_VALUE(
            mQuats.is_leaf(),
            "Cannot set requires_grad of quats which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        TORCH_CHECK_VALUE(
            mLogScales.is_leaf(),
            "Cannot set requires_grad of log_scales which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        TORCH_CHECK_VALUE(
            mLogitOpacities.is_leaf(),
            "Cannot set requires_grad of logit_opacities which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        TORCH_CHECK_VALUE(
            mSh0.is_leaf(),
            "Cannot set requires_grad of sh0 which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        TORCH_CHECK_VALUE(
            mShN.is_leaf(),
            "Cannot set requires_grad of shN which is a non-leaf tensor. "
            "Call .detach() on this object or create a new GaussianSplat3d object with leaf tensors.");

        mMeans.requires_grad_(requiresGrad);
        mQuats.requires_grad_(requiresGrad);
        mLogScales.requires_grad_(requiresGrad);
        mLogitOpacities.requires_grad_(requiresGrad);
        mSh0.requires_grad_(requiresGrad);
        mShN.requires_grad_(requiresGrad);
    }

    /// @brief Set the data of the GaussianSplat3d object from the given tensors.
    /// @param means An [N, 3]-shaped tensor representing the means of the Gaussians in this scene.
    /// @param quats An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this
    ///              scene.
    /// @param logScales An [N, 3]-shaped tensor representing the log of the scales of the
    ///                  Gaussians in this scene.
    /// @param logitOpacities An [N]-shaped tensor representing the logit of the opacities of the
    ///                     Gaussians in this scene.
    /// @param sh0 An [N, 1, D]-shaped tensor representing the diffuse SH coefficients of the
    ///            Gaussians in this scene.
    /// @param shN A [N, K-1, D]-shaped tensor representing the directionally-dependent SH
    ///            coefficients of the Gaussians in this scene.
    void setState(const torch::Tensor &means,
                  const torch::Tensor &quats,
                  const torch::Tensor &logScales,
                  const torch::Tensor &logitOpacities,
                  const torch::Tensor &sh0,
                  const torch::Tensor &shN);

    /// @brief Return the number of Gaussians in the scene.
    /// @return The number of Gaussians in the scene.
    int64_t
    numGaussians() const {
        return mMeans.size(0);
    }

    /// @brief Return the number of SH basis coeffients used in the scene.
    /// @return The number of SH bases used in the scene.
    int64_t
    numShBases() const {
        return mShN.size(1) + 1;
    }

    /// @brief Return the number of channels used in the scene (e.g. 3 for RGB colors).
    /// @return The number of channels used in the scene.
    int64_t
    numChannels() const {
        return mShN.size(2);
    }

    /// @brief Return the accumulated gradient norms of projected Gaussians in this
    ///        scene across backward passes.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the accumulated gradient norms of projected
    ///         Gaussians in this scene across backward passes or an empty tensor if
    ///         accumulateMean2dGradients is false.
    torch::Tensor
    accumulated2dMeansGradientNormsForGrad() const {
        return mAccumulatedNormalized2dMeansGradientNormsForGrad;
    }

    /// @brief Return the accumulated maximum 2D radii of projected Gaussians in this
    ///        scene across backward passes.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the accumulated maximum 2D radii of projected
    ///         Gaussians in this scene across backward passes or an empty tensor if
    ///         accumulateMax2dRadii is false.
    torch::Tensor
    accumulatedMax2dRadiiForGrad() const {
        return mAccumulated2dRadiiForGrad;
    }

    /// @brief Return the backward passes used to accumulate each Gaussian during optimization.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the backward passes used to accumulate each
    ///         Gaussian during optimization or an empty tensor if accumulateMean2dGradients is
    ///         false.
    torch::Tensor
    gradientStepCountsForGrad() const {
        return mGradientStepCountForGrad;
    }

    /// @brief Reset the gradient statistics of the Gaussians in this scene.
    ///        See @ref accumulated2dMeansGradientNormsForGrad, @ref gradientStepCountsForGrad,
    ///        @ref accumulatedMax2dRadiiForGrad.
    /// @note This function is only valid if requiresGrad is true.
    void
    resetAccumulatedGradientState() {
        if (mAccumulateMean2dGradients) {
            mAccumulatedNormalized2dMeansGradientNormsForGrad = torch::Tensor();
            mGradientStepCountForGrad                         = torch::Tensor();
        }
        if (mAccumulateMax2dRadii) {
            mAccumulated2dRadiiForGrad = torch::Tensor();
        }
    }

    /// @brief Return the state of the GaussianSplat3d object as a dictionary (similar to Pytorch's
    /// nn.Module).
    /// @return A dictionary containing the state of the GaussianSplat3d object.
    std::unordered_map<std::string, torch::Tensor> stateDict() const;

    /// @brief Load the state of the GaussianSplat3d object from a state_dict (similar to Pytorch's
    /// nn.Module).
    /// @param stateDict A dictionary containing the state of the GaussianSplat3d object.
    void loadStateDict(const std::unordered_map<std::string, torch::Tensor> &stateDict);

    /// @brief Precompute the projected Gaussians to be re-used for rendering images (e.g. if you
    /// want to render multiple images with the same camera settings or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render images with @ref
    /// renderFromProjectedGaussians
    ProjectedGaussianSplats projectGaussiansForImages(const torch::Tensor &worldToCameraMatrices,
                                                      const torch::Tensor &projectionMatrices,
                                                      size_t imageWidth,
                                                      size_t imageHeight,
                                                      const float near,
                                                      const float far,
                                                      const ProjectionType projectionType,
                                                      const int64_t shDegreeToUse,
                                                      const float minRadius2d,
                                                      const float eps2d,
                                                      const bool antialias);

    /// @brief Precompute the projected Gaussians to be re-used for rendering depths (e.g. if
    /// you want to render multiple depth maps with the same camera settings or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render depths with @ref
    /// renderFromProjectedGaussians
    ProjectedGaussianSplats projectGaussiansForDepths(const torch::Tensor &worldToCameraMatrices,
                                                      const torch::Tensor &projectionMatrices,
                                                      size_t imageWidth,
                                                      size_t imageHeight,
                                                      const float near,
                                                      const float far,
                                                      const ProjectionType projectionType,
                                                      const float minRadius2d,
                                                      const float eps2d,
                                                      const bool antialias);

    /// @brief Precompute the projected Gaussians to be re-used for rendering images and depths
    /// (e.g. if you want to render multiple images and depth maps with the same camera settings
    /// or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render images and depths with
    /// @ref renderFromProjectedGaussians
    ProjectedGaussianSplats
    projectGaussiansForImagesAndDepths(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       size_t imageWidth,
                                       size_t imageHeight,
                                       const float near,
                                       const float far,
                                       const ProjectionType projectionType,
                                       const int64_t shDegreeToUse,
                                       const float minRadius2d,
                                       const float eps2d,
                                       const bool antialias);

    /// @brief Save this scene and optional training metadata to a PLY file with the given filename
    /// @param filename The path to save the PLY file to
    /// @param metadata An optional dictionary of training metadata to include in the PLY file. The
    /// keys are strings and the values are either strings, int64s, doubles, or tensors
    void savePly(const std::string &filename,
                 std::optional<std::unordered_map<std::string, PlyMetadataTypes>> metadata) const;

    /// @brief Load a PLY file's means, quats, scales, opacities, and SH coefficients as the state
    /// of this GaussianSplat3d object
    /// @param filename Filename of the PLY file
    /// @param device Device to transfer the loaded tensors to
    /// @return The loaded GaussianSplat3d class, and a dictionary of metadata (can be empty if no
    //  metadata was saved in the PLY file). The metadata keys are strings and the values are either
    //  strings, int64s, doubles, or tensors.
    static std::tuple<GaussianSplat3d, std::unordered_map<std::string, PlyMetadataTypes>>
    fromPly(const std::string &filename, torch::Device device = torch::kCPU);

    /// @brief Render using precomputed projected Gaussians (see
    /// @ref projectGaussiansForImages, @ref projectGaussiansForDepths,
    /// @ref projectGaussiansForImagesAndDepths).
    /// Optionally lets you render a cropped image by specifying the crop width, height, and origin.
    /// @param projectedGaussians ProjectedGaussianSplats object obtained from @ref
    /// projectGaussiansForImages, @ref projectGaussiansForDepths, or @ref
    /// projectGaussiansForImagesAndDepths
    /// @param cropWidth Width of the cropped image (use -1 for no cropping)
    /// @param cropHeight Height of the cropped image (use -1 for no cropping)
    /// @param cropOriginW Origin of the cropped image in the width dimension (use -1 for no
    /// cropping)
    /// @param cropOriginH Origin of the cropped image in the height dimension (use -1 for no
    /// cropping)
    /// @param tileSize Size of the tiles used for rendering
    /// @return Tuple of two tensors:
    ///     images: A [C, H, W, D|1|D+1] tensor containing the the rendered image
    ///             (or depth or image and depth) for each camera
    ///     alphas: A [C, H, W, 1] tensor containing the alpha values of the rendered images
    std::tuple<torch::Tensor, torch::Tensor>
    renderFromProjectedGaussians(const GaussianSplat3d::ProjectedGaussianSplats &projectedGaussians,
                                 const ssize_t cropWidth   = -1,
                                 const ssize_t cropHeight  = -1,
                                 const ssize_t cropOriginW = -1,
                                 const ssize_t cropOriginH = -1,
                                 const size_t tileSize     = 16);

    /// @brief Render images of this Gaussian splat scene from the given camera matrices and
    /// projection matrices.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     images: A [C, H, W, D] tensor containing the the rendered image for each camera
    ///     alphas: A [C, H, W, 1] tensor containing the alpha values of the rendered images
    std::tuple<torch::Tensor, torch::Tensor>
    renderImages(const torch::Tensor &worldToCameraMatrices,
                 const torch::Tensor &projectionMatrices,
                 const size_t imageWidth,
                 const size_t imageHeight,
                 const float near,
                 const float far,
                 const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                 const int64_t shDegreeToUse         = -1,
                 const size_t tileSize               = 16,
                 const float minRadius2d             = 0.0,
                 const float eps2d                   = 0.3,
                 const bool antialias                = false);

    /// @brief Render depths of this Gaussian splat scene from the given camera matrices and
    /// projection matrices.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     images: A [C, H, W, 1] tensor containing the the rendered depths for each camera
    ///     alphas: A [C, H, W, 1] tensor containing the alpha values of the rendered depths
    std::tuple<torch::Tensor, torch::Tensor>
    renderDepths(const torch::Tensor &worldToCameraMatrices,
                 const torch::Tensor &projectionMatrices,
                 const size_t imageWidth,
                 const size_t imageHeight,
                 const float near,
                 const float far,
                 const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                 const size_t tileSize               = 16,
                 const float minRadius2d             = 0.0,
                 const float eps2d                   = 0.3,
                 const bool antialias                = false);

    std::tuple<torch::Tensor, torch::Tensor>
    renderImagesAndDepths(const torch::Tensor &worldToCameraMatrices,
                          const torch::Tensor &projectionMatrices,
                          const size_t imageWidth,
                          const size_t imageHeight,
                          const float near,
                          const float far,
                          const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                          const int64_t shDegreeToUse         = -1,
                          const size_t tileSize               = 16,
                          const float minRadius2d             = 0.0,
                          const float eps2d                   = 0.3,
                          const bool antialias                = false);

    std::tuple<JaggedTensor, JaggedTensor>
    sparseRenderImages(const fvdb::JaggedTensor &pixelsToRender,
                       const torch::Tensor &worldToCameraMatrices,
                       const torch::Tensor &projectionMatrices,
                       const size_t imageWidth,
                       const size_t imageHeight,
                       const float near,
                       const float far,
                       const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                       const int64_t shDegreeToUse         = -1,
                       const size_t tileSize               = 16,
                       const float minRadius2d             = 0.0,
                       const float eps2d                   = 0.3,
                       const bool antialias                = false);

    std::tuple<JaggedTensor, JaggedTensor>
    sparseRenderDepths(const fvdb::JaggedTensor &pixelsToRender,
                       const torch::Tensor &worldToCameraMatrices,
                       const torch::Tensor &projectionMatrices,
                       const size_t imageWidth,
                       const size_t imageHeight,
                       const float near,
                       const float far,
                       const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                       const size_t tileSize               = 16,
                       const float minRadius2d             = 0.0,
                       const float eps2d                   = 0.3,
                       const bool antialias                = false);

    std::tuple<JaggedTensor, JaggedTensor>
    sparseRenderImagesAndDepths(const fvdb::JaggedTensor &pixelsToRender,
                                const torch::Tensor &worldToCameraMatrices,
                                const torch::Tensor &projectionMatrices,
                                const size_t imageWidth,
                                const size_t imageHeight,
                                const float near,
                                const float far,
                                const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                                const int64_t shDegreeToUse         = -1,
                                const size_t tileSize               = 16,
                                const float minRadius2d             = 0.0,
                                const float eps2d                   = 0.3,
                                const bool antialias                = false);

    /// @brief Render the number of contributing Gaussians for each pixel in the image.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     num_contributing_gaussians: A [C, H, W] tensor containing the number of contributing
    ///                                 Gaussians for each pixel for each camera
    ///     alphas: A [C, H, W] tensor containing the alpha values of the rendered images
    std::tuple<torch::Tensor, torch::Tensor> renderNumContributingGaussians(
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const size_t imageWidth,
        const size_t imageHeight,
        const float near,
        const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const size_t tileSize               = 16,
        const float minRadius2d             = 0.0,
        const float eps2d                   = 0.3,
        const bool antialias                = false);

    /// @brief Render the number of contributing Gaussians for each pixel in the image.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of two tensors:
    ///     num_contributing_gaussians: A [P1 + P2 + ..., 1] jagged tensor containing the number of
    ///     contributing
    ///                                 Gaussians for each pixel for each camera
    ///     alphas: A [P1 + P2 + ..., 1] jagged tensor containing the composited alpha value of the
    ///     pixels
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
    sparseRenderNumContributingGaussians(const fvdb::JaggedTensor &pixelsToRender,
                                         const torch::Tensor &worldToCameraMatrices,
                                         const torch::Tensor &projectionMatrices,
                                         const size_t imageWidth,
                                         const size_t imageHeight,
                                         const float near,
                                         const float far,
                                         const ProjectionType projectionType,
                                         const size_t tileSize,
                                         const float minRadius2d,
                                         const float eps2d,
                                         const bool antialias);

    /// @brief Render the IDs of the gaussians that are the top K contributors to the rendered
    /// pixels and the value of the weighted contribution to the rendered pixels.  If the size of
    /// `numSamples`(i.e. K) is greater than the number of contributing samples for a pixel, the
    /// remaining samples' weights are filled with zeros and the IDs are filled with -1.
    /// @param numSamples Requested number of top K contributing samples per pixel
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     ids: A [C, H, W, K] tensor containing the the IDs of the top K contributors to the
    ///          rendered pixel for each camera
    ///     weights: A [C, H, W, K] tensor containing the weights of the top K contributors to the
    ///              rendered pixel for each camera. The weights are normalized to sum to 1 if the
    ///              list is exahustive of all contributing samples.
    std::tuple<torch::Tensor, torch::Tensor> renderTopContributingGaussianIds(
        const int numSamples,
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const size_t imageWidth,
        const size_t imageHeight,
        const float near,
        const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const size_t tileSize               = 16,
        const float minRadius2d             = 0.0,
        const float eps2d                   = 0.3,
        const bool antialias                = false);

    /// @brief Render the IDs of the gaussians that are the top K contributors to the rendered
    /// pixels and the value of the weighted contribution to the rendered pixels.  If the size of
    /// `numSamples`(i.e. K) is greater than the number of contributing samples for a pixel, the
    /// remaining samples' weights are filled with zeros and the IDs are filled with -1.  This
    /// function will render only a sparse subset of the pixels in the overall image, as specified
    /// by the `pixelsToRender` parameter.
    /// @param numSamples Requested number of top K contributing samples per pixel
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     ids: A [P1 + P2 + ..., K] jagged tensor containing the the IDs of the top K contributors
    ///     to the
    ///          rendered pixel for each camera
    ///     weights: A [P1 + P2 + ..., K] jagged tensor containing the weights of the top K
    ///     contributors to the
    ///              rendered pixel for each camera. The weights are normalized to sum to 1 if the
    ///              list is exahustive of all contributing samples.
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> sparseRenderTopContributingGaussianIds(
        const int numSamples,
        const fvdb::JaggedTensor &pixelsToRender,
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const size_t imageWidth,
        const size_t imageHeight,
        const float near,
        const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const size_t tileSize               = 16,
        const float minRadius2d             = 0.0,
        const float eps2d                   = 0.3,
        const bool antialias                = false);

    /// @brief Render the IDs of the gaussians that are the contributors to the rendered images'
    /// pixels and the value of their weighted contributions to the rendered pixels.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @param topKContributors If > 0, uses the efficient top-K kernel to return only the top K
    ///        contributors per pixel. If 0 (default), returns all contributing Gaussians.
    /// @return Tuple of two JaggedTensors:
    ///     ids: A [[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ...
    ///         CNP(imageWidth * imageHeight), 1]] jagged tensor containing the IDs of the
    ///         contributing Gaussians of each rendered pixel for each camera. The IDs are in
    ///         row-major order.
    ///     weights: A [[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ...
    ///         CNP(imageWidth * imageHeight), 1]] jagged tensor containing the weights of the
    ///         contributing Gaussians of each rendered pixel for each camera. The weights are in
    ///         row-major order and sum to 1 for each pixel if that pixel is opaque (alpha=1).
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
    renderContributingGaussianIds(const torch::Tensor &worldToCameraMatrices,
                                  const torch::Tensor &projectionMatrices,
                                  const size_t imageWidth,
                                  const size_t imageHeight,
                                  const float near,
                                  const float far,
                                  const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                                  const size_t tileSize               = 16,
                                  const float minRadius2d             = 0.0,
                                  const float eps2d                   = 0.3,
                                  const bool antialias                = false,
                                  const int topKContributors          = 0);

    /// @brief Render the IDs of the gaussians that are the contributors to the rendered images'
    /// pixels and the value of their weighted contributions to the rendered pixels.  This
    /// function will render only a sparse subset of the pixels in the overall image, as specified
    /// by the `pixelsToRender` parameter.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @param topKContributors If > 0, uses the efficient top-K kernel to return only the top K
    ///        contributors per pixel. If 0 (default), returns all contributing Gaussians.
    /// @return Tuple of two JaggedTensors:
    ///     ids: A [[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]] jagged tensor
    ///          containing the IDs of the contributing Gaussians of each rendered pixel for each
    ///          camera. The IDs are in row-major order.
    ///     weights: A [[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]] jagged
    ///              tensor containing the weights of the contributing Gaussians of each rendered
    ///              pixel for each camera. The weights are in row-major order and sum to 1 for each
    ///              pixel if that pixel is opaque (alpha=1).
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> sparseRenderContributingGaussianIds(
        const fvdb::JaggedTensor &pixelsToRender,
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const size_t imageWidth,
        const size_t imageHeight,
        const float near,
        const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const size_t tileSize               = 16,
        const float minRadius2d             = 0.0,
        const float eps2d                   = 0.3,
        const bool antialias                = false,
        const int topKContributors          = 0);

    /// @brief Select a subset of the Gaussians in this scene based on the given slice.
    /// @param begin The start index of the slice (inclusive)
    /// @param end The end index of the slice (exclusive)
    /// @param step The step size of the slice
    /// @return A new GaussianSplat3d object containing only the selected Gaussians.
    GaussianSplat3d sliceSelect(const int64_t begin, const int64_t end, const int64_t step) const;

    /// @brief Select a subset of the Gaussians in this scene based on the given indices.
    /// @param indices A 1D tensor of indices in the range [0, numGaussians-1] to select from the
    //  Gaussians in this scene.
    /// @return A new GaussianSplat3d object containing only the selected Gaussians.
    GaussianSplat3d indexSelect(const torch::Tensor &indices) const;

    /// @brief Select a subset of the Gaussians in this scene based on the given mask.
    /// @param mask A 1D boolean tensor of shape [N] where N is the number of Gaussians in this
    ///             scene. The mask indicates which Gaussians to select.
    /// @return A new GaussianSplat3d object containing only the selected Gaussians.
    ///         The mask must have the same length as the number of Gaussians in this scene.
    GaussianSplat3d maskSelect(const torch::Tensor &mask) const;

    /// @brief Assign new Gaussians to a subset of the Gaussians in this scene based on the given
    /// indices.
    /// @param indices A 1D tensor of indices in the range [0, numGaussians-1] to assign new
    /// Gaussians to.
    /// @param other A GaussianSplat3d object containing the new Gaussians to assign.
    void indexSet(const torch::Tensor &indices, const GaussianSplat3d &other);

    /// @brief Assign new Gaussians to a subset of the Gaussians in this scene based on the given
    /// slice.
    /// @param begin The start index of the slice (inclusive)
    /// @param end The end index of the slice (exclusive)
    /// @param step The step size of the slice
    /// @param other A GaussianSplat3d object containing the new Gaussians to assign.
    ///             The mask must have the same length as the number of Gaussians in this scene.
    void sliceSet(const int64_t begin,
                  const int64_t end,
                  const int64_t step,
                  const GaussianSplat3d &other);

    /// @brief Assign new Gaussians to a subset of the Gaussians in this scene based on the given
    /// mask.
    /// @param mask A 1D boolean tensor of shape [N] where N is the number of Gaussians in this
    ///             scene. The mask indicates which Gaussians to assign.
    /// @param other A GaussianSplat3d object containing the new Gaussians to assign.
    ///             The mask must have the same length as the number of Gaussians in this scene.
    void maskSet(const torch::Tensor &mask, const GaussianSplat3d &other);

  private:
    torch::Tensor mMeans;          // [N, 3]
    torch::Tensor mQuats;          // [N, 4]
    torch::Tensor mLogScales;      // [N, 3]
    torch::Tensor mLogitOpacities; // [N]
    torch::Tensor mSh0;            // [N, 1, D]
    torch::Tensor mShN;            // [N, K-1, D]

    // Used for subdivision during optimization
    torch::Tensor mAccumulatedNormalized2dMeansGradientNormsForGrad; // [N]
    torch::Tensor mAccumulated2dRadiiForGrad;                        // [N]
    torch::Tensor mGradientStepCountForGrad;                         // [N]
    bool mAccumulateMean2dGradients = false;
    bool mAccumulateMax2dRadii      = false;

    static void checkState(const torch::Tensor &means,
                           const torch::Tensor &quats,
                           const torch::Tensor &logScales,
                           const torch::Tensor &logitOpacities,
                           const torch::Tensor &sh0,
                           const torch::Tensor &shN);

    ProjectedGaussianSplats projectGaussiansImpl(const torch::Tensor &worldToCameraMatrices,
                                                 const torch::Tensor &projectionMatrices,
                                                 const fvdb::detail::ops::RenderSettings &settings);

    std::tuple<torch::Tensor, torch::Tensor>
    renderCropFromProjectedGaussiansImpl(const ProjectedGaussianSplats &state,
                                         const size_t tileSize,
                                         const ssize_t cropWidth,
                                         const ssize_t cropHeight,
                                         const ssize_t cropOriginW,
                                         const ssize_t cropOriginH);

    /// @brief Implements index set with a tensor of booleans or integer indices
    /// @param indexOrMask A 1D tensor of indices in the range [0, numGaussians-1] or a boolean mask
    ///                    of shape [N] where N is the number of Gaussians in this scene.
    /// @param other A GaussianSplat3d object containing the new Gaussians to assign.
    ///              The mask must have the same length as the number of Gaussians in this scene.
    void tensorIndexSetImpl(const torch::Tensor &indexOrMask, const GaussianSplat3d &other);

    /// @brief Implements indexing with a tensor of booleans or integer indices
    /// @param indexOrMask A 1D tensor of indices in the range [0, numGaussians-1] or a boolean mask
    ///                    of shape [N] where N is the number of Gaussians in this scene.
    /// @return A new GaussianSplat3d object containing only the selected Gaussians.
    ///         The mask must have the same length as the number of Gaussians in this scene.
    GaussianSplat3d tensorIndexGetImpl(const torch::Tensor &indexOrMask) const;

    /// @brief Render the scene described by the Gaussian splats from the specified views.
    ///        This function returns a single render quantity (RGB, depth, RGB+D) and
    ///        single alpha value per pixel.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of (render quantity, alpha value)
    std::tuple<torch::Tensor, torch::Tensor>
    renderImpl(const torch::Tensor &worldToCameraMatrices,
               const torch::Tensor &projectionMatrices,
               const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Render the scene described by the Gaussian splats at the specified pixels in the
    ///        specified views. This function returns a single render quantity (RGB, depth, RGB+D)
    ///        and single alpha value per pixel.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of (render quantity, alpha value)
    std::tuple<JaggedTensor, JaggedTensor>
    sparseRenderImpl(const JaggedTensor &pixelsToRender,
                     const torch::Tensor &worldToCameraMatrices,
                     const torch::Tensor &projectionMatrices,
                     const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Render the number of contributing Gaussians for each pixel in the image.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of two tensors:
    ///     num_contributing_gaussians: A [B, H, W] tensor containing the number of contributing
    ///                                 Gaussians for each pixel for each camera
    std::tuple<torch::Tensor, torch::Tensor>
    renderNumContributingGaussiansImpl(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Render the number of contributing Gaussians for each pixel in the image.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of two tensors:
    ///     num_contributing_gaussians: A [P1 + P2 + ..., 1] jagged tensor containing the number of
    ///     contributing
    ///                                 Gaussians for each pixel for each camera
    ///     alphas: A [P1 + P2 + ..., 1] jagged tensor containing the composited alpha value of the
    ///     pixels
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
    sparseRenderNumContributingGaussiansImpl(const fvdb::JaggedTensor &pixelsToRender,
                                             const torch::Tensor &worldToCameraMatrices,
                                             const torch::Tensor &projectionMatrices,
                                             const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Render the gaussian splatting scene
    ///         For every pixel being rendered, this function returns multiple samples in depth of
    ///         the gaussian IDs and multiple samples of the weighted alpha values. The number of
    ///         samples per pixel is determined by the sampling parameters in the settings. If
    ///         the size of the requested number of samples is greater than the number of
    ///         contributing samples for a pixel, the remaining samples' weights are filled with
    ///         zeros and the IDs are filled with -1.  The samples are ordered front to back in
    ///         their depth ordering from camera.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of two tensors:
    ///     ids: A [B, H, W, K] tensor containing the the IDs of the top K contributors to the
    ///          rendered pixel for each camera
    ///     weights: A [B, H, W, K] tensor containing the weights of the top K contributors to the
    ///              rendered pixel for each camera. The weights are normalized to sum to the alpha
    ///              value of the final rendered pixel if the list is exahustive of all contributing
    ///              samples.
    std::tuple<torch::Tensor, torch::Tensor>
    renderTopContributingGaussianIdsImpl(const torch::Tensor &worldToCameraMatrices,
                                         const torch::Tensor &projectionMatrices,
                                         const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Sparse render the gaussian splatting scene
    ///         For every pixel being rendered, this function returns multiple samples in depth of
    ///         the gaussian IDs and multiple samples of the weighted alpha values. The number of
    ///         samples per pixel is determined by the sampling parameters in the settings. If
    ///         the size of the requested number of samples is greater than the number of
    ///         contributing samples for a pixel, the remaining samples' weights are filled with
    ///         zeros and the IDs are filled with -1.  The samples are ordered front to back in
    ///         their depth ordering from camera.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @return Tuple of two tensors:
    ///     ids: A [P1 + P2 + ..., K] jagged tensor containing the the IDs of the top K contributors
    ///     to the
    ///          rendered pixel for each camera
    ///     weights: A [P1 + P2 + ..., K] jagged tensor containing the weights of the top K
    ///     contributors to the
    ///              rendered pixel for each camera. The weights are normalized to sum to the alpha
    ///              value of the final rendered pixel if the list is exahustive of all contributing
    ///              samples.
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor>
    sparseRenderTopContributingGaussianIdsImpl(const fvdb::JaggedTensor &pixelsToRender,
                                               const torch::Tensor &worldToCameraMatrices,
                                               const torch::Tensor &projectionMatrices,
                                               const fvdb::detail::ops::RenderSettings &settings);

    /// @brief Render the gaussian splatting scene
    ///         For every pixel being rendered, this function returns multiple samples in depth of
    ///         the gaussian IDs and multiple samples of the weighted alpha values.  The samples are
    ///         ordered front to back in their depth ordering from camera.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @param maybeNumContributingGaussians [C, H, W] tensor containing the number of contributing
    ///                                      Gaussians for each pixel for each camera. If not
    ///                                      provided, ``settings`` must have ``numDepthSamples``
    ///                                      set to a value greater than 0.
    /// @return Tuple of two JaggedTensors:
    ///     ids: A [[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ...
    ///         CNP(imageWidth * imageHeight), 1]] jagged tensor containing the IDs of the
    ///         contributing Gaussians of each rendered pixel for each camera. The IDs are in
    ///         row-major order.
    ///     weights: A [[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ...
    ///         CNP(imageWidth * imageHeight), 1]] jagged tensor containing the weights of the
    ///         contributing Gaussians of each rendered pixel for each camera. The weights are in
    ///         row-major order and sum to 1 for each pixel if that pixel is opaque (alpha=1).
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> renderContributingGaussianIdsImpl(
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const fvdb::detail::ops::RenderSettings &settings,
        const std::optional<torch::Tensor> &maybeNumContributingGaussians = std::nullopt);

    /// @brief Sparse render the gaussian splatting scene
    ///         For every pixel being rendered, this function returns multiple samples in depth of
    ///         the gaussian IDs and multiple samples of the weighted alpha values. The number of
    ///         samples per pixel is determined by the sampling parameters in the settings. If
    ///         the size of the requested number of samples is greater than the number of
    ///         contributing samples for a pixel, the remaining samples' weights are filled with
    ///         zeros and the IDs are filled with -1.  The samples are ordered front to back in
    ///         their depth ordering from camera.
    /// @param pixelsToRender [P1 + P2 + ..., 2] JaggedTensor of pixels per camera to render.
    /// @param worldToCameraMatrices [C, 4, 4]
    /// @param projectionMatrices [C, 3, 3]
    /// @param settings
    /// @param maybeNumContributingGaussians [C, H, W] tensor containing the number of contributing
    ///                                      Gaussians for each pixel for each camera. If provided,
    ///                                      the kernel will use the top-k path and ignore this
    ///                                      tensor. If not provided, ``settings`` must have
    ///                                      ``numDepthSamples`` set to a value greater than 0.
    /// @return Tuple of two tensors:
    ///     ids: A [P1 + P2 + ..., K] jagged tensor containing the the IDs of the top K contributors
    ///     to the
    ///          rendered pixel for each camera
    ///     weights: A [P1 + P2 + ..., K] jagged tensor containing the weights of the top K
    ///     contributors to the
    ///              rendered pixel for each camera. The weights are normalized to sum to the alpha
    ///              value of the final rendered pixel if the list is exahustive of all contributing
    ///              samples.
    std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor> sparseRenderContributingGaussianIdsImpl(
        const fvdb::JaggedTensor &pixelsToRender,
        const torch::Tensor &worldToCameraMatrices,
        const torch::Tensor &projectionMatrices,
        const fvdb::detail::ops::RenderSettings &settings,
        const std::optional<fvdb::JaggedTensor> &maybeNumContributingGaussians = std::nullopt);

    torch::Tensor evalSphericalHarmonicsImpl(const int64_t shDegreeToUse,
                                             const torch::Tensor &worldToCameraMatrices,
                                             const torch::Tensor &perGaussianProjectedRadii) const;
};

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderJagged(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                     const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                     const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                     const JaggedTensor &opacities, // [N1 + N2 + ...]
                     const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
                     const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                     const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                     const uint32_t image_width,
                     const uint32_t image_height,
                     const float near_plane          = 0.01,
                     const float far_plane           = 1e10,
                     const int sh_degree_to_use      = -1,
                     const int tile_size             = 16,
                     const float radius_clip         = 0.0,
                     const float eps2d               = 0.3,
                     const bool antialias            = false,
                     const bool render_depth_channel = false,
                     const bool return_debug_info    = false,
                     const bool render_depth_only    = false,
                     const bool ortho                = false);

} // namespace fvdb

#endif // FVDB_GAUSSIANSPLAT3D_H
