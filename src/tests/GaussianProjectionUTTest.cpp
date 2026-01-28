// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>

#include <c10/util/Exception.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/types.h>

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace fvdb::detail::ops {

namespace {

inline std::tuple<float, float>
projectPointWithOpenCVDistortion(const float x,
                                 const float y,
                                 const float z,
                                 const float fx,
                                 const float fy,
                                 const float cx,
                                 const float cy,
                                 const std::vector<float> &radial,     // k1,k2,k3 or k1..k6
                                 const std::vector<float> &tangential, // p1,p2 (or empty)
                                 const std::vector<float> &thinPrism   // s1..s4 (or empty)
) {
    const float xn = x / z;
    const float yn = y / z;

    const float x2 = xn * xn;
    const float y2 = yn * yn;
    const float xy = xn * yn;

    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    float radial_dist = 1.0f;
    if (radial.size() == 3) {
        const float k1 = radial[0];
        const float k2 = radial[1];
        const float k3 = radial[2];
        radial_dist    = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    } else if (radial.size() == 6) {
        const float k1  = radial[0];
        const float k2  = radial[1];
        const float k3  = radial[2];
        const float k4  = radial[3];
        const float k5  = radial[4];
        const float k6  = radial[5];
        const float num = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
        const float den = 1.0f + r2 * (k4 + r2 * (k5 + r2 * k6));
        radial_dist     = (den != 0.0f) ? (num / den) : 0.0f;
    } else if (!radial.empty()) {
        return {std::nanf(""), std::nanf("")};
    }

    float xd = xn * radial_dist;
    float yd = yn * radial_dist;

    const float p1 = tangential.size() >= 1 ? tangential[0] : 0.0f;
    const float p2 = tangential.size() >= 2 ? tangential[1] : 0.0f;
    // OpenCV tangential:
    // x += 2*p1*x*y + p2*(r^2 + 2*x^2)
    // y += p1*(r^2 + 2*y^2) + 2*p2*x*y
    xd += 2.0f * p1 * xy + p2 * (r2 + 2.0f * x2);
    yd += p1 * (r2 + 2.0f * y2) + 2.0f * p2 * xy;

    if (!thinPrism.empty()) {
        if (thinPrism.size() != 4) {
            return {std::nanf(""), std::nanf("")};
        }
        const float s1 = thinPrism[0];
        const float s2 = thinPrism[1];
        const float s3 = thinPrism[2];
        const float s4 = thinPrism[3];
        xd += s1 * r2 + s2 * r4;
        yd += s3 * r2 + s4 * r4;
    }

    const float u = fx * xd + cx;
    const float v = fy * yd + cy;
    return {u, v};
}

} // namespace

// We keep these UT tests purely analytic/synthetic: they should not depend on any external test
// data downloads.
struct GaussianProjectionUTTestFixture : public ::testing::Test {
    void
    SetUp() override {
        torch::manual_seed(0);
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is not available; skipping GaussianProjectionUT tests.";
        }
    }

    torch::Tensor means;                   // [N, 3]
    torch::Tensor quats;                   // [N, 4]
    torch::Tensor logScales;               // [N, 3]
    torch::Tensor worldToCamMatricesStart; // [C, 4, 4]
    torch::Tensor worldToCamMatricesEnd;   // [C, 4, 4]
    torch::Tensor projectionMatrices;      // [C, 3, 3]
    DistortionModel distortionModel = DistortionModel::NONE;
    torch::Tensor distortionCoeffs;        // [C, 12] for OPENCV, or [C, 0] for NONE

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

    // World-space Gaussian mean at optical axis (x=y=0), so projected mean should be exactly
    // (cx,cy)
    const float z = 5.0f;
    means         = torch::tensor({{0.0f, 0.0f, z}}, torch::kFloat32);
    // Quaternion is stored as [w,x,y,z] in fvdb kernels
    quats = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);

    const float sx = 0.2f, sy = 0.3f, sz = 0.4f;
    logScales = torch::log(torch::tensor({{sx, sy, sz}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 100.0f, fy = 200.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

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
    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

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
    const float cov_u      = (fx * sx / z) * (fx * sx / z);
    const float cov_v      = (fy * sy / z) * (fy * sy / z);
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
    means = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    // Extremely small Gaussian so UT mean should match the point projection closely
    // (off-axis + perspective nonlinearity can otherwise introduce a tiny UT mean shift).
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 123.0f, fy = 77.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

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

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto means2d_cpu       = means2d.cpu();
    const float expected_u = fx * (x / z) + cx;
    const float expected_v = fy * (y / z) + cy;
    // UT projects sigma points through a nonlinear pinhole model; even for a very small Gaussian
    // there can be a tiny second-order mean shift. Keep a slightly relaxed tolerance here.
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 2e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 2e-3f);
}

TEST_F(GaussianProjectionUTTestFixture,
       OffAxisTinyGaussian_RadTanDistortion_MeanMatchesOpenCVPoint) {
    const int64_t C = 1;

    const float x = 0.2f, y = -0.1f, z = 2.0f;
    means     = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 500.0f, fy = 450.0f, cx = 400.0f, cy = 300.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    // coefficients chosen to be non-trivial but not extreme
    const float k1 = 0.10f;
    const float k2 = -0.05f;
    const float k3 = 0.01f;
    const float p1 = 0.001f;
    const float p2 = -0.0005f;

    distortionModel = DistortionModel::OPENCV_RADTAN_5;
    // [k1..k6,p1,p2,s1..s4] (use polynomial by setting k4..k6 = 0, and no thin-prism by zeroing s*)
    distortionCoeffs          = torch::zeros({C, 12}, torch::kFloat32);
    auto distortionCoeffsAcc  = distortionCoeffs.accessor<float, 2>();
    distortionCoeffsAcc[0][0] = k1;
    distortionCoeffsAcc[0][1] = k2;
    distortionCoeffsAcc[0][2] = k3;
    distortionCoeffsAcc[0][6] = p1;
    distortionCoeffsAcc[0][7] = p2;

    imageWidth  = 800;
    imageHeight = 600;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = true;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto radii_cpu   = radii.cpu();
    auto means2d_cpu = means2d.cpu();
    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);

    const auto [expected_u, expected_v] =
        projectPointWithOpenCVDistortion(x, y, z, fx, fy, cx, cy, {k1, k2, k3}, {p1, p2}, {});
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 5e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 5e-3f);
}

TEST_F(GaussianProjectionUTTestFixture,
       OffAxisTinyGaussian_RationalDistortion_MeanMatchesOpenCVPoint) {
    const int64_t C = 1;

    const float x = -0.15f, y = 0.12f, z = 3.0f;
    means     = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 600.0f, fy = 550.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    const float k1 = 0.08f;
    const float k2 = -0.02f;
    const float k3 = 0.005f;
    const float k4 = 0.01f;
    const float k5 = -0.004f;
    const float k6 = 0.001f;
    const float p1 = -0.0007f;
    const float p2 = 0.0003f;

    distortionModel           = DistortionModel::OPENCV_RATIONAL_8;
    distortionCoeffs          = torch::zeros({C, 12}, torch::kFloat32);
    auto distortionCoeffsAcc  = distortionCoeffs.accessor<float, 2>();
    distortionCoeffsAcc[0][0] = k1;
    distortionCoeffsAcc[0][1] = k2;
    distortionCoeffsAcc[0][2] = k3;
    distortionCoeffsAcc[0][3] = k4;
    distortionCoeffsAcc[0][4] = k5;
    distortionCoeffsAcc[0][5] = k6;
    distortionCoeffsAcc[0][6] = p1;
    distortionCoeffsAcc[0][7] = p2;

    imageWidth  = 800;
    imageHeight = 600;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = true;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto radii_cpu   = radii.cpu();
    auto means2d_cpu = means2d.cpu();
    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);

    const auto [expected_u, expected_v] = projectPointWithOpenCVDistortion(
        x, y, z, fx, fy, cx, cy, {k1, k2, k3, k4, k5, k6}, {p1, p2}, {});
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 5e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 5e-3f);
}

TEST_F(GaussianProjectionUTTestFixture,
       OffAxisTinyGaussian_ThinPrismDistortion_MeanMatchesOpenCVPoint) {
    const int64_t C = 1;

    const float x = 0.1f, y = 0.08f, z = 2.5f;
    means     = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 700.0f, fy = 650.0f, cx = 500.0f, cy = 400.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    const float k1 = 0.05f;
    const float k2 = -0.01f;
    const float k3 = 0.002f;
    const float k4 = 0.005f;
    const float k5 = -0.001f;
    const float k6 = 0.0005f;
    const float p1 = 0.0002f;
    const float p2 = -0.0004f;
    const float s1 = 0.0008f;
    const float s2 = -0.0003f;
    const float s3 = 0.0005f;
    const float s4 = 0.0001f;

    distortionModel            = DistortionModel::OPENCV_THIN_PRISM_12;
    distortionCoeffs           = torch::zeros({C, 12}, torch::kFloat32);
    auto distortionCoeffsAcc   = distortionCoeffs.accessor<float, 2>();
    distortionCoeffsAcc[0][0]  = k1;
    distortionCoeffsAcc[0][1]  = k2;
    distortionCoeffsAcc[0][2]  = k3;
    distortionCoeffsAcc[0][3]  = k4;
    distortionCoeffsAcc[0][4]  = k5;
    distortionCoeffsAcc[0][5]  = k6;
    distortionCoeffsAcc[0][6]  = p1;
    distortionCoeffsAcc[0][7]  = p2;
    distortionCoeffsAcc[0][8]  = s1;
    distortionCoeffsAcc[0][9]  = s2;
    distortionCoeffsAcc[0][10] = s3;
    distortionCoeffsAcc[0][11] = s4;

    imageWidth  = 1200;
    imageHeight = 900;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = true;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto radii_cpu   = radii.cpu();
    auto means2d_cpu = means2d.cpu();
    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);

    const auto [expected_u, expected_v] = projectPointWithOpenCVDistortion(
        x, y, z, fx, fy, cx, cy, {k1, k2, k3, k4, k5, k6}, {p1, p2}, {s1, s2, s3, s4});
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 5e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 5e-3f);
}

TEST_F(GaussianProjectionUTTestFixture,
       OffAxisTinyGaussian_RadTanThinPrismDistortion_MeanMatchesOpenCVPoint) {
    const int64_t C = 1;

    const float x = 0.07f, y = -0.11f, z = 2.2f;
    means     = torch::tensor({{x, y, z}}, torch::kFloat32);
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 620.0f, fy = 590.0f, cx = 410.0f, cy = 305.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    const float k1 = 0.06f;
    const float k2 = -0.015f;
    const float k3 = 0.003f;
    const float p1 = 0.0004f;
    const float p2 = -0.0002f;
    const float s1 = 0.0007f;
    const float s2 = -0.0003f;
    const float s3 = 0.0005f;
    const float s4 = 0.0001f;

    distortionModel           = DistortionModel::OPENCV_RADTAN_THIN_PRISM_9;
    distortionCoeffs          = torch::zeros({C, 12}, torch::kFloat32);
    auto distortionCoeffsAcc  = distortionCoeffs.accessor<float, 2>();
    distortionCoeffsAcc[0][0] = k1;
    distortionCoeffsAcc[0][1] = k2;
    distortionCoeffsAcc[0][2] = k3;
    // k4..k6 must be 0 for this explicit model
    distortionCoeffsAcc[0][6]  = p1;
    distortionCoeffsAcc[0][7]  = p2;
    distortionCoeffsAcc[0][8]  = s1;
    distortionCoeffsAcc[0][9]  = s2;
    distortionCoeffsAcc[0][10] = s3;
    distortionCoeffsAcc[0][11] = s4;

    imageWidth  = 900;
    imageHeight = 700;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = true;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto radii_cpu   = radii.cpu();
    auto means2d_cpu = means2d.cpu();
    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);

    const auto [expected_u, expected_v] = projectPointWithOpenCVDistortion(
        x, y, z, fx, fy, cx, cy, {k1, k2, k3}, {p1, p2}, {s1, s2, s3, s4});
    EXPECT_NEAR(means2d_cpu[0][0][0].item<float>(), expected_u, 5e-3f);
    EXPECT_NEAR(means2d_cpu[0][0][1].item<float>(), expected_v, 5e-3f);
}

TEST_F(GaussianProjectionUTTestFixture, RadTanThinPrism_RejectsNonZeroK456) {
    const int64_t C = 1;

    means     = torch::tensor({{0.1f, 0.05f, 2.0f}}, torch::kFloat32).cuda();
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32).cuda();
    logScales = torch::log(torch::tensor({{1e-6f, 1e-6f, 1e-6f}}, torch::kFloat32)).cuda();

    worldToCamMatricesStart = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32))
                                  .unsqueeze(0)
                                  .expand({C, 4, 4})
                                  .cuda();
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();
    // NOTE: `.accessor<>()` is a host-side view; only use it on CPU tensors, then move to CUDA.
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = 500.0f;
    projectionMatricesAcc[0][1][1] = 500.0f;
    projectionMatricesAcc[0][0][2] = 320.0f;
    projectionMatricesAcc[0][1][2] = 240.0f;
    projectionMatricesAcc[0][2][2] = 1.0f;
    projectionMatrices             = projectionMatrices.cuda();

    distortionModel           = DistortionModel::OPENCV_RADTAN_THIN_PRISM_9;
    distortionCoeffs          = torch::zeros({C, 12}, torch::kFloat32);
    auto distortionCoeffsAcc  = distortionCoeffs.accessor<float, 2>();
    distortionCoeffsAcc[0][0] = 0.01f;  // k1
    distortionCoeffsAcc[0][3] = 0.1f;   // k4 (invalid for RADTAN_THIN_PRISM_9)
    distortionCoeffsAcc[0][8] = 0.001f; // s1
    distortionCoeffs          = distortionCoeffs.cuda();

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams = UTParams{};

    EXPECT_THROW((dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                                    quats,
                                                                    logScales,
                                                                    worldToCamMatricesStart,
                                                                    worldToCamMatricesEnd,
                                                                    projectionMatrices,
                                                                    RollingShutterType::NONE,
                                                                    utParams,
                                                                    distortionModel,
                                                                    distortionCoeffs,
                                                                    imageWidth,
                                                                    imageHeight,
                                                                    eps2d,
                                                                    nearPlane,
                                                                    farPlane,
                                                                    minRadius2d,
                                                                    false,
                                                                    false)),
                 c10::Error);
}

TEST_F(GaussianProjectionUTTestFixture,
       SigmaPointBehindCamera_HardRejectsEvenWhenNotRequiringAllInImage) {
    const int64_t C = 1;

    // Place the Gaussian mean just in front of the camera, but give it a large Z scale so one
    // UT sigma point crosses behind the camera (z <= 0). The UT kernel should hard-reject such
    // Gaussians (new behavior), regardless of requireAllSigmaPointsInImage.
    const float z = 0.20f;
    means         = torch::tensor({{0.0f, 0.0f, z}}, torch::kFloat32);
    quats         = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);

    // With UT alpha=0.1, gamma ~= sqrt(0.03) ~= 0.173. Choose sz so z - gamma*sz <= 0.
    const float sx = 1e-3f, sy = 1e-3f, sz = 2.0f;
    logScales = torch::log(torch::tensor({{sx, sy, sz}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 500.0f, fy = 500.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.05f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams.alpha                        = 0.1f;
    utParams.beta                         = 2.0f;
    utParams.kappa                        = 0.0f;
    utParams.numSigmaPoints               = 7;
    utParams.inImageMargin                = 0.1f;
    utParams.requireAllSigmaPointsInImage = false;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    // When the UT kernel discards a Gaussian, only radii are defined to be 0; other outputs are
    // undefined (may contain garbage). Only assert radii here.
    auto radii_cpu = radii.cpu();
    EXPECT_EQ(radii_cpu[0][0].item<int32_t>(), 0);
}

TEST_F(GaussianProjectionUTTestFixture,
       SomeSigmaPointsOutOfBoundsButInFront_NotHardRejectedWhenNotRequiringAllInImage) {
    const int64_t C = 1;

    // Centered mean in front of camera; huge X scale so +/-X sigma points project outside the
    // image, but remain in front of the camera. With requireAllSigmaPointsInImage=false this should
    // still produce a valid Gaussian (radii > 0).
    const float z = 5.0f;
    means         = torch::tensor({{0.0f, 0.0f, z}}, torch::kFloat32);
    quats         = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);

    // Choose sx large enough that projected sigma points fall outside image+margin.
    const float sx = 30.0f, sy = 1e-3f, sz = 1e-3f;
    logScales = torch::log(torch::tensor({{sx, sy, sz}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    const float fx = 500.0f, fy = 500.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams                              = UTParams{};
    utParams.requireAllSigmaPointsInImage = false;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto radii_cpu = radii.cpu();
    EXPECT_GT(radii_cpu[0][0].item<int32_t>(), 0);
}

TEST_F(GaussianProjectionUTTestFixture, RollingShutterNone_DepthUsesStartPoseNotCenter) {
    const int64_t C = 1;

    // If RollingShutterType::NONE, projection uses the start pose. Depth culling and outDepths
    // should therefore also use the start pose. This test ensures we don't accidentally use the
    // center pose (t=0.5) when start/end differ.
    const float z = 5.0f;
    means         = torch::tensor({{0.0f, 0.0f, z}}, torch::kFloat32);
    quats         = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales     = torch::log(torch::tensor({{0.2f, 0.2f, 0.2f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();
    // End pose translates camera forward in +z, so p_cam.z is larger at t=1.0.
    auto worldToCamEndAcc     = worldToCamMatricesEnd.accessor<float, 3>();
    worldToCamEndAcc[0][2][3] = 1.0f;

    const float fx = 100.0f, fy = 100.0f, cx = 320.0f, cy = 240.0f;
    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = fx;
    projectionMatricesAcc[0][1][1] = fy;
    projectionMatricesAcc[0][0][2] = cx;
    projectionMatricesAcc[0][1][2] = cy;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams                              = UTParams{};
    utParams.requireAllSigmaPointsInImage = true;

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    const auto [radii, means2d, depths, conics, compensations] =
        dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                          quats,
                                                          logScales,
                                                          worldToCamMatricesStart,
                                                          worldToCamMatricesEnd,
                                                          projectionMatrices,
                                                          RollingShutterType::NONE,
                                                          utParams,
                                                          distortionModel,
                                                          distortionCoeffs,
                                                          imageWidth,
                                                          imageHeight,
                                                          eps2d,
                                                          nearPlane,
                                                          farPlane,
                                                          minRadius2d,
                                                          false,
                                                          false);

    auto depths_cpu = depths.cpu();
    // Start pose is identity, so depth should be exactly z (not z + 0.5).
    EXPECT_NEAR(depths_cpu[0][0].item<float>(), z, 1e-4f);
}

TEST_F(GaussianProjectionUTTestFixture, RejectsNonSevenSigmaPointsOnHost) {
    const int64_t C = 1;

    means     = torch::tensor({{0.0f, 0.0f, 5.0f}}, torch::kFloat32);
    quats     = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, torch::kFloat32);
    logScales = torch::log(torch::tensor({{0.2f, 0.2f, 0.2f}}, torch::kFloat32));

    worldToCamMatricesStart =
        torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).expand({C, 4, 4});
    worldToCamMatricesEnd = worldToCamMatricesStart.clone();

    projectionMatrices = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto projectionMatricesAcc     = projectionMatrices.accessor<float, 3>();
    projectionMatricesAcc[0][0][0] = 100.0f;
    projectionMatricesAcc[0][1][1] = 100.0f;
    projectionMatricesAcc[0][0][2] = 320.0f;
    projectionMatricesAcc[0][1][2] = 240.0f;
    projectionMatricesAcc[0][2][2] = 1.0f;

    distortionModel  = DistortionModel::NONE;
    distortionCoeffs = torch::zeros({C, 0}, torch::kFloat32);

    imageWidth  = 640;
    imageHeight = 480;
    eps2d       = 0.3f;
    nearPlane   = 0.1f;
    farPlane    = 100.0f;
    minRadius2d = 0.0f;

    utParams                = UTParams{};
    utParams.numSigmaPoints = 5; // invalid: kernel supports only 7

    means                   = means.cuda();
    quats                   = quats.cuda();
    logScales               = logScales.cuda();
    worldToCamMatricesStart = worldToCamMatricesStart.cuda();
    worldToCamMatricesEnd   = worldToCamMatricesEnd.cuda();
    projectionMatrices      = projectionMatrices.cuda();
    distortionCoeffs        = distortionCoeffs.cuda();

    EXPECT_THROW((dispatchGaussianProjectionForwardUT<torch::kCUDA>(means,
                                                                    quats,
                                                                    logScales,
                                                                    worldToCamMatricesStart,
                                                                    worldToCamMatricesEnd,
                                                                    projectionMatrices,
                                                                    RollingShutterType::NONE,
                                                                    utParams,
                                                                    distortionModel,
                                                                    distortionCoeffs,
                                                                    imageWidth,
                                                                    imageHeight,
                                                                    eps2d,
                                                                    nearPlane,
                                                                    farPlane,
                                                                    minRadius2d,
                                                                    false,
                                                                    false)),
                 c10::Error);
}

} // namespace fvdb::detail::ops
