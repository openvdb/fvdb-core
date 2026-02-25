// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONMODEL_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONMODEL_H

#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>

#include <torch/types.h>

#include <memory>

namespace fvdb::detail::ops {

struct GaussianProjectionConfig {
    torch::Tensor worldToCamMatricesStart; // [C, 4, 4]
    torch::Tensor worldToCamMatricesEnd;   // [C, 4, 4] (same as start for no rolling shutter)
    torch::Tensor projectionMatrices;      // [C, 3, 3]
    torch::Tensor distortionCoeffs;        // [C, 12] for OPENCV_*, [C, 0] otherwise
    CameraModel cameraModel               = CameraModel::PINHOLE;
    RollingShutterType rollingShutterType = RollingShutterType::NONE;
    UTParams utParams{};
};

class GaussianProjectionModel {
  public:
    virtual ~GaussianProjectionModel() = default;
    virtual const GaussianProjectionConfig &config() const = 0;
    virtual bool usesUT() const                            = 0;
    virtual bool isOrthographic() const                    = 0;
};

class ClassicGaussianProjectionModel final : public GaussianProjectionModel {
  public:
    ClassicGaussianProjectionModel(const torch::Tensor &worldToCamMatrices,
                                   const torch::Tensor &projectionMatrices,
                                   const bool ortho)
        : mConfig{worldToCamMatrices,
                  worldToCamMatrices,
                  projectionMatrices,
                  torch::empty({worldToCamMatrices.size(0), 0}, worldToCamMatrices.options()),
                  ortho ? CameraModel::ORTHOGRAPHIC : CameraModel::PINHOLE,
                  RollingShutterType::NONE,
                  UTParams{}},
          mOrthographic(ortho) {}

    const GaussianProjectionConfig &config() const override { return mConfig; }
    bool usesUT() const override { return false; }
    bool isOrthographic() const override { return mOrthographic; }

  private:
    GaussianProjectionConfig mConfig;
    bool mOrthographic;
};

class UTGaussianProjectionModel final : public GaussianProjectionModel {
  public:
    UTGaussianProjectionModel(const torch::Tensor &worldToCamMatricesStart,
                              const torch::Tensor &worldToCamMatricesEnd,
                              const torch::Tensor &projectionMatrices,
                              const torch::Tensor &distortionCoeffs,
                              const RollingShutterType rollingShutterType,
                              const UTParams &utParams,
                              const CameraModel cameraModel)
        : mConfig{worldToCamMatricesStart,
                  worldToCamMatricesEnd,
                  projectionMatrices,
                  distortionCoeffs,
                  cameraModel,
                  rollingShutterType,
                  utParams} {}

    const GaussianProjectionConfig &config() const override { return mConfig; }
    bool usesUT() const override { return true; }
    bool isOrthographic() const override { return mConfig.cameraModel == CameraModel::ORTHOGRAPHIC; }

  private:
    GaussianProjectionConfig mConfig;
};

using GaussianProjectionModelPtr = std::shared_ptr<GaussianProjectionModel>;

inline GaussianProjectionModelPtr
makeClassicProjectionModel(const torch::Tensor &worldToCamMatrices,
                           const torch::Tensor &projectionMatrices,
                           const bool ortho) {
    return std::make_shared<ClassicGaussianProjectionModel>(
        worldToCamMatrices, projectionMatrices, ortho);
}

inline GaussianProjectionModelPtr
makeUTProjectionModel(const torch::Tensor &worldToCamMatricesStart,
                      const torch::Tensor &worldToCamMatricesEnd,
                      const torch::Tensor &projectionMatrices,
                      const torch::Tensor &distortionCoeffs,
                      const RollingShutterType rollingShutterType,
                      const UTParams &utParams,
                      const CameraModel cameraModel) {
    return std::make_shared<UTGaussianProjectionModel>(worldToCamMatricesStart,
                                                       worldToCamMatricesEnd,
                                                       projectionMatrices,
                                                       distortionCoeffs,
                                                       rollingShutterType,
                                                       utParams,
                                                       cameraModel);
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANPROJECTIONMODEL_H
