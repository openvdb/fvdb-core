// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>

#include <torch/script.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cmath>

namespace fvdb::detail::ops {

// We keep these UT tests purely analytic/synthetic: they should not depend on any external test
// data downloads.
struct GaussianProjectionUTTestFixture : public ::testing::Test {
    void
    SetUp() override {
        torch::manual_seed(0);
    }

    torch::Tensor means;                   // [N, 3]
    torch::Tensor quats;                   // [N, 4]
    torch::Tensor scales;                  // [N, 3]
    torch::Tensor worldToCamMatricesStart; // [C, 4, 4]
    torch::Tensor worldToCamMatricesEnd;  // [C, 4, 4]
    torch::Tensor projectionMatrices;     // [C, 3, 3]
    torch::Tensor radialCoeffs;           // [C, 0] or [C, 4] or [C, 6]
    torch::Tensor tangentialCoeffs;        // [C, 0] or [C, 2]
    torch::Tensor thinPrismCoeffs;        // [C, 0] or [C, 3]

    int64_t imageWidth;
    int64_t imageHeight;
    float eps2d;
    float nearPlane;
    float farPlane;
    float minRadius2d;

    UTParams utParams;
};

TEST_F(GaussianProjectionUTTestFixture, CenteredGaussian_NoDistortion_AnalyticMeanAndConic) {
    const int64_t C = 1;

    // World-space Gaussian mean at optical axis (x=y=0), so projected mean should be exactly (cx,cy)
    const float z = 5.0f;
    means         = torch::tensor({{0.0f, 0.0f, z}}, torch::kFloat32);
    // Quaternion is stored as [w,x,y,z] in fvdb kernels
    quats         = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);

    const float sx = 0.2f, sy = 0.3f, sz = 0.4f;
    scales         = torch::tensor({{sx, sy, sz}}, torch::kFloat32);

    worldToCamMatricesStart = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32))
                                  .unsqueeze(0)
                                  .expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 100.0f, fy = 200.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    projectionMatrices[0][0][0] = fx;
    projectionMatrices[0][1][1] = fy;
    projectionMatrices[0][0][2] = cx;
    projectionMatrices[0][1][2] = cy;
    projectionMatrices[0][2][2] = 1.0f;

    radialCoeffs     = torch::zeros({C, 0}, torch::kFloat32);
    tangentialCoeffs = torch::zeros({C, 0}, torch::kFloat32);
    thinPrismCoeffs  = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f; // matches defaults
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f; // interpreted as fraction of image dims
    utParams.requireAllSigmaPointsInImage = true;

    // CUDA
    means                  = means.cuda();
    quats                  = quats.cuda();
    scales                 = scales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices     = projectionMatrices.cuda();
    radialCoeffs           = radialCoeffs.cuda();
    tangentialCoeffs       = tangentialCoeffs.cuda();
    thinPrismCoeffs        = thinPrismCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(
            means, quats, scales, worldToCamMatricesStart, worldToCamMatricesEnd, projectionMatrices,
            RollingShutterType::NONE, utParams, radialCoeffs, tangentialCoeffs, thinPrismCoeffs,
            imageWidth, imageHeight, eps2d, nearPlane, farPlane, minRadius2d, false, false);

    auto means2d_cpu = means2d.cpu();
    auto depths_cpu  = depths.cpu();
    auto conics_cpu  = conics.cpu();
    auto radii_cpu   = radii.cpu();

    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);
    EXPECT_NEAR(depths_cpu[0][0].item<float>(), z, 1e-4f);
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), cx, 1e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), cy, 1e-3f);

    // For u = fx * x/z + cx, v = fy * y/z + cy and mean on optical axis (x=y=0),
    // the projected covariance is exactly:
    // cov_u = (fx*sx/z)^2, cov_v = (fy*sy/z)^2, off-diagonals = 0.
    const float cov_u = (fx * sx / z) * (fx * sx / z);
    const float cov_v = (fy * sy / z) * (fy * sy / z);
    const float cov_u_blur = cov_u + eps2d;
    const float cov_v_blur = cov_v + eps2d;

    const float expected_a = 1.0f / cov_u_blur;
    const float expected_b = 0.0f;
    const float expected_c = 1.0f / cov_v_blur;

    EXPECT_NEAR(conics_cpu[0][0][0].item<float>(), expected_a, 1e-3f);
    EXPECT_NEAR(conics_cpu[0][0][1].item<float>(), expected_b, 1e-3f);
    EXPECT_NEAR(conics_cpu[0][0][2].item<float>(), expected_c, 1e-3f);
}

TEST_F(GaussianProjectionUTTestFixture, OffAxisTinyGaussian_NoDistortion_MeanMatchesPinhole) {
    const int64_t C = 1;

    const float x = 1.0f, y = -2.0f, z = 10.0f;
    means         = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats         = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    // Extremely small Gaussian so UT mean should match the point projection closely
    // (off-axis + perspective nonlinearity can otherwise introduce a tiny UT mean shift).
    scales = torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32);

    worldToCamMatricesStart = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32))
                                  .unsqueeze(0)
                                  .expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 123.0f, fy = 77.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    projectionMatrices[0][0][0] = fx;
    projectionMatrices[0][1][1] = fy;
    projectionMatrices[0][0][2] = cx;
    projectionMatrices[0][1][2] = cy;
    projectionMatrices[0][2][2] = 1.0f;

    radialCoeffs     = torch::zeros({C, 0}, torch::kFloat32);
    tangentialCoeffs = torch::zeros({C, 0}, torch::kFloat32);
    thinPrismCoeffs  = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = true;

    means                  = means.cuda();
    quats                  = quats.cuda();
    scales                 = scales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices     = projectionMatrices.cuda();
    radialCoeffs           = radialCoeffs.cuda();
    tangentialCoeffs       = tangentialCoeffs.cuda();
    thinPrismCoeffs        = thinPrismCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(
            means, quats, scales, worldToCamMatricesStart, worldToCamMatricesEnd, projectionMatrices,
            RollingShutterType::NONE, utParams, radialCoeffs, tangentialCoeffs, thinPrismCoeffs,
            imageWidth, imageHeight, eps2d, nearPlane, farPlane, minRadius2d, false, false);

    auto means2d_cpu = means2d.cpu();
    const float expected_u = fx * (x / z) + cx;
    const float expected_v = fy * (y / z) + cy;
    // UT projects sigma points through a nonlinear pinhole model; even for a very small Gaussian
    // there can be a tiny second-order mean shift. Keep a slightly relaxed tolerance here.
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 2e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 2e-3f);
}

} // namespace fvdb::detail::ops
