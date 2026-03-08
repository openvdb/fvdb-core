// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include <cuda/std/cmath>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace fvdb::detail::ops {
namespace {

using Mat2f = nanovdb::math::Mat2<float>;
using Mat3f = nanovdb::math::Mat3<float>;
using Vec2f = nanovdb::math::Vec2<float>;
using Vec3f = nanovdb::math::Vec3<float>;

inline std::tuple<float, float>
projectPointWithRadTan5Distortion(const float x,
                                  const float y,
                                  const float z,
                                  const float fx,
                                  const float fy,
                                  const float cx,
                                  const float cy,
                                  const float k1,
                                  const float k2,
                                  const float k3,
                                  const float p1,
                                  const float p2) {
    const float xn = x / z;
    const float yn = y / z;
    const float x2 = xn * xn;
    const float y2 = yn * yn;
    const float xy = xn * yn;
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    float xd           = xn * radial;
    float yd           = yn * radial;
    xd += 2.0f * p1 * xy + p2 * (r2 + 2.0f * x2);
    yd += p1 * (r2 + 2.0f * y2) + 2.0f * p2 * xy;

    return {fx * xd + cx, fy * yd + cy};
}

inline __device__ float
maxAbs2(float a, float b) {
    return ::cuda::std::fabs(a - b);
}

inline __device__ float
maxAbsVec2(const Vec2f &a, const Vec2f &b) {
    return ::cuda::std::fmax(maxAbs2(a[0], b[0]), maxAbs2(a[1], b[1]));
}

inline __device__ float
maxAbsVec3(const Vec3f &a, const Vec3f &b) {
    const float dx = ::cuda::std::fabs(a[0] - b[0]);
    const float dy = ::cuda::std::fabs(a[1] - b[1]);
    const float dz = ::cuda::std::fabs(a[2] - b[2]);
    return ::cuda::std::fmax(dx, ::cuda::std::fmax(dy, dz));
}

inline __device__ float
maxAbsMat2(const Mat2f &a, const Mat2f &b) {
    float m = 0.0f;
    m       = ::cuda::std::fmax(m, ::cuda::std::fabs(a[0][0] - b[0][0]));
    m       = ::cuda::std::fmax(m, ::cuda::std::fabs(a[0][1] - b[0][1]));
    m       = ::cuda::std::fmax(m, ::cuda::std::fabs(a[1][0] - b[1][0]));
    m       = ::cuda::std::fmax(m, ::cuda::std::fabs(a[1][1] - b[1][1]));
    return m;
}

inline __device__ float
maxAbsMat3(const Mat3f &a, const Mat3f &b) {
    float m = 0.0f;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m = ::cuda::std::fmax(m, ::cuda::std::fabs(a[r][c] - b[r][c]));
        }
    }
    return m;
}

template <typename CameraT>
__global__ void
compareProjectWorldTo2DKernel(int64_t C,
                              int64_t N,
                              const float *meansWorld,
                              const float *covarsWorld6,
                              CameraT camera,
                              float *outMaxDiffs) {
    const int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
    if (idx >= C * N) {
        return;
    }
    const int64_t cid = idx / N;
    const int64_t gid = idx % N;

    const Vec3f meanWorld(meansWorld[gid * 3], meansWorld[gid * 3 + 1], meansWorld[gid * 3 + 2]);
    const Mat3f covarWorld = loadCovarianceRowMajor6(covarsWorld6 + gid * 6);

    const auto [cov2dNew, mean2dNew, depthNew] =
        camera.projectWorldGaussianTo2D(cid, meanWorld, covarWorld);

    const auto [R, t]                = camera.worldToCamTransform(cid);
    const Vec3f meanCam              = transformPointWorldToCam(R, t, meanWorld);
    const Mat3f covCam               = transformCovarianceWorldToCam(R, covarWorld);
    const auto [cov2dRef, mean2dRef] = camera.projectTo2DGaussian(cid, meanCam, covCam);
    const float depthRef             = meanCam[2];

    outMaxDiffs[idx * 3 + 0] = maxAbsMat2(cov2dNew, cov2dRef);
    outMaxDiffs[idx * 3 + 1] = maxAbsVec2(mean2dNew, mean2dRef);
    outMaxDiffs[idx * 3 + 2] = ::cuda::std::fabs(depthNew - depthRef);
}

template <typename CameraT>
__global__ void
compareProjectWorldTo2DVJPKernel(int64_t C,
                                 int64_t N,
                                 const float *meansWorld,
                                 const float *covarsWorld6,
                                 const float *dLossDCov2d,
                                 const float *dLossDMean2d,
                                 const float *dLossDDepth,
                                 CameraT camera,
                                 float *outMaxDiffs) {
    const int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
    if (idx >= C * N) {
        return;
    }
    const int64_t cid = idx / N;
    const int64_t gid = idx % N;

    const Vec3f meanWorld(meansWorld[gid * 3], meansWorld[gid * 3 + 1], meansWorld[gid * 3 + 2]);
    const Mat3f covarWorld = loadCovarianceRowMajor6(covarsWorld6 + gid * 6);
    const Mat2f dCov2d(dLossDCov2d[idx * 4 + 0],
                       dLossDCov2d[idx * 4 + 1],
                       dLossDCov2d[idx * 4 + 2],
                       dLossDCov2d[idx * 4 + 3]);
    const Vec2f dMean2d(dLossDMean2d[idx * 2 + 0], dLossDMean2d[idx * 2 + 1]);
    const float dDepth = dLossDDepth[idx];

    const auto [dCovWorldNew, dMeanWorldNew, dRotNew, dTransNew] =
        camera.projectWorldGaussianTo2DVJP(cid, meanWorld, covarWorld, dCov2d, dMean2d, dDepth);

    const auto [R, t]   = camera.worldToCamTransform(cid);
    const Vec3f meanCam = transformPointWorldToCam(R, t, meanWorld);
    const Mat3f covCam  = transformCovarianceWorldToCam(R, covarWorld);
    auto [dCovCamRef, dMeanCamRef] =
        camera.projectTo2DGaussianVJP(cid, meanCam, covCam, dCov2d, dMean2d);
    dMeanCamRef[2] += dDepth;

    auto [dRotRef, dTransRef, dMeanWorldRef] =
        transformPointWorldToCamVectorJacobianProduct(R, t, meanWorld, dMeanCamRef);
    auto [dRotCovRef, dCovWorldRef] =
        transformCovarianceWorldToCamVectorJacobianProduct(R, covarWorld, dCovCamRef);
    dRotRef += dRotCovRef;

    outMaxDiffs[idx * 4 + 0] = maxAbsMat3(dCovWorldNew, dCovWorldRef);
    outMaxDiffs[idx * 4 + 1] = maxAbsVec3(dMeanWorldNew, dMeanWorldRef);
    outMaxDiffs[idx * 4 + 2] = maxAbsMat3(dRotNew, dRotRef);
    outMaxDiffs[idx * 4 + 3] = maxAbsVec3(dTransNew, dTransRef);
}

template <typename CameraT>
__global__ void
projectWorldPointToPixelKernel(
    int64_t C, const float *pointsWorld, CameraT camera, float *outPixels, int32_t *outStatuses) {
    const int64_t cid = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
    if (cid >= C) {
        return;
    }

    const Vec3f pointWorld(
        pointsWorld[cid * 3 + 0], pointsWorld[cid * 3 + 1], pointsWorld[cid * 3 + 2]);
    Vec2f pixel(0.0f, 0.0f);
    const auto status      = camera.projectWorldPointToPixel(cid, pointWorld, 0.0f, pixel);
    outPixels[cid * 2 + 0] = pixel[0];
    outPixels[cid * 2 + 1] = pixel[1];
    outStatuses[cid]       = static_cast<int32_t>(status);
}

torch::Tensor
makeWorldToCamMatrices(int64_t C) {
    auto worldToCamCpu =
        torch::zeros({C, 4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = worldToCamCpu.accessor<float, 3>();
    for (int64_t c = 0; c < C; ++c) {
        acc[c][0][0] = 1.0f;
        acc[c][1][1] = 1.0f;
        acc[c][2][2] = 1.0f;
        acc[c][3][3] = 1.0f;
        acc[c][0][3] = 0.03f * static_cast<float>(c + 1);
        acc[c][1][3] = -0.02f * static_cast<float>(c + 1);
        acc[c][2][3] = 0.01f * static_cast<float>(c + 1);
    }
    return worldToCamCpu.cuda();
}

torch::Tensor
makeProjectionMatrices(int64_t C, bool ortho) {
    auto kCpu =
        torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = kCpu.accessor<float, 3>();
    for (int64_t c = 0; c < C; ++c) {
        const float fx = ortho ? (1.2f + 0.1f * c) : (480.0f + 7.0f * c);
        const float fy = ortho ? (1.3f + 0.1f * c) : (470.0f + 5.0f * c);
        const float cx = 320.0f + 0.5f * static_cast<float>(c);
        const float cy = 240.0f + 0.5f * static_cast<float>(c);
        acc[c][0][0]   = fx;
        acc[c][1][1]   = fy;
        acc[c][0][2]   = cx;
        acc[c][1][2]   = cy;
        acc[c][2][2]   = 1.0f;
    }
    return kCpu.cuda();
}

torch::Tensor
makeMeansWorld(int64_t N) {
    auto meansCpu =
        torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = meansCpu.accessor<float, 2>();
    for (int64_t i = 0; i < N; ++i) {
        acc[i][0] = -0.2f + 0.01f * static_cast<float>(i);
        acc[i][1] = 0.15f - 0.008f * static_cast<float>(i);
        acc[i][2] = 2.0f + 0.02f * static_cast<float>(i);
    }
    return meansCpu.cuda();
}

torch::Tensor
makeCovarsWorld6(int64_t N) {
    auto covCpu =
        torch::zeros({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = covCpu.accessor<float, 2>();
    for (int64_t i = 0; i < N; ++i) {
        const float f = 0.001f * static_cast<float>(i);
        acc[i][0]     = 0.05f + f;  // xx
        acc[i][1]     = 0.001f + f; // xy
        acc[i][2]     = -0.0005f;   // xz
        acc[i][3]     = 0.07f + f;  // yy
        acc[i][4]     = 0.0007f;    // yz
        acc[i][5]     = 0.09f + f;  // zz
    }
    return covCpu.cuda();
}

torch::Tensor
makeOpenCVDistortionCoeffs(int64_t C) {
    auto distCpu =
        torch::zeros({C, 12}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = distCpu.accessor<float, 2>();
    for (int64_t c = 0; c < C; ++c) {
        const float s = static_cast<float>(c + 1);
        acc[c][0]     = 0.02f * s;
        acc[c][1]     = -0.004f * s;
        acc[c][2]     = 0.001f * s;
        acc[c][6]     = 0.0015f * s;
        acc[c][7]     = -0.0012f * s;
    }
    return distCpu.cuda();
}

template <typename CameraT>
void
runForwardAndVjpParityChecks(const CameraT &camera,
                             int64_t C,
                             int64_t N,
                             const torch::Tensor &means,
                             const torch::Tensor &covars6) {
    constexpr int kBlock = 256;
    const int64_t total  = C * N;
    const dim3 grid((total + kBlock - 1) / kBlock);
    const dim3 block(kBlock);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    auto fwdDiffs = torch::zeros({total, 3}, means.options());
    compareProjectWorldTo2DKernel<<<grid, block, 0, stream>>>(C,
                                                              N,
                                                              means.data_ptr<float>(),
                                                              covars6.data_ptr<float>(),
                                                              camera,
                                                              fwdDiffs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto dCov2d   = torch::randn({total, 4}, means.options());
    auto dMean2d  = torch::randn({total, 2}, means.options());
    auto dDepth   = torch::randn({total}, means.options());
    auto bwdDiffs = torch::zeros({total, 4}, means.options());
    compareProjectWorldTo2DVJPKernel<<<grid, block, 0, stream>>>(C,
                                                                 N,
                                                                 means.data_ptr<float>(),
                                                                 covars6.data_ptr<float>(),
                                                                 dCov2d.data_ptr<float>(),
                                                                 dMean2d.data_ptr<float>(),
                                                                 dDepth.data_ptr<float>(),
                                                                 camera,
                                                                 bwdDiffs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float maxFwd = fwdDiffs.abs().max().item<float>();
    const float maxBwd = bwdDiffs.abs().max().item<float>();
    EXPECT_LT(maxFwd, 1e-6f);
    EXPECT_LT(maxBwd, 1e-6f);
}

} // namespace

TEST(GaussianCamerasTest, PerspectiveEncapsulatedProjectionAndVJPMatchReferencePath) {
    constexpr int64_t C = 3;
    constexpr int64_t N = 64;
    auto worldToCam     = makeWorldToCamMatrices(C);
    auto projection     = makeProjectionMatrices(C, false);
    auto meansWorld     = makeMeansWorld(N);
    auto covarsWorld6   = makeCovarsWorld6(N);

    auto camera = PerspectiveCamera<float>(
        projection, worldToCam, static_cast<int32_t>(C), 640, 480, 0.01f, 1.0e8f);
    runForwardAndVjpParityChecks(camera, C, N, meansWorld, covarsWorld6);
}

TEST(GaussianCamerasTest, OrthographicEncapsulatedProjectionAndVJPMatchReferencePath) {
    constexpr int64_t C = 2;
    constexpr int64_t N = 64;
    auto worldToCam     = makeWorldToCamMatrices(C);
    auto projection     = makeProjectionMatrices(C, true);
    auto meansWorld     = makeMeansWorld(N);
    auto covarsWorld6   = makeCovarsWorld6(N);

    auto camera = OrthographicCamera<float>(
        projection, worldToCam, static_cast<int32_t>(C), 640, 480, -1.0e8f, 1.0e8f);
    runForwardAndVjpParityChecks(camera, C, N, meansWorld, covarsWorld6);
}

TEST(GaussianCamerasTest, DistortedPerspectiveEncapsulatedProjectionAndVJPMatchReferencePath) {
    constexpr int64_t C = 2;
    auto worldToCam     = makeWorldToCamMatrices(C);
    auto projection     = makeProjectionMatrices(C, false);
    auto distortion     = makeOpenCVDistortionCoeffs(C);
    auto pointsWorldCpu =
        torch::tensor({{-0.18f, 0.10f, 2.5f}, {-0.05f, -0.12f, 2.8f}},
                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto pointsWorld = pointsWorldCpu.cuda();

    auto camera = PerspectiveWithDistortionCamera<float>(worldToCam,
                                                         worldToCam.clone(),
                                                         projection,
                                                         distortion,
                                                         static_cast<uint32_t>(C),
                                                         /*numDistCoeffs=*/12,
                                                         /*imageWidth=*/640,
                                                         /*imageHeight=*/480,
                                                         /*imageOriginW=*/0,
                                                         /*imageOriginH=*/0,
                                                         RollingShutterType::NONE,
                                                         DistortionModel::OPENCV_RADTAN_5);

    using ProjectionVisibility = PerspectiveWithDistortionCamera<float>::ProjectionVisibility;

    auto outPixels =
        torch::zeros({C, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto outStatuses =
        torch::zeros({C}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    constexpr int kBlock = 64;
    const dim3 grid((C + kBlock - 1) / kBlock);
    const dim3 block(kBlock);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    projectWorldPointToPixelKernel<<<grid, block, 0, stream>>>(C,
                                                               pointsWorld.data_ptr<float>(),
                                                               camera,
                                                               outPixels.data_ptr<float>(),
                                                               outStatuses.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto worldToCamCpu  = worldToCam.cpu();
    auto projectionCpu  = projection.cpu();
    auto distortionCpu  = distortion.cpu();
    auto outPixelsCpu   = outPixels.cpu();
    auto outStatusesCpu = outStatuses.cpu();

    auto worldAcc      = worldToCamCpu.accessor<float, 3>();
    auto projectionAcc = projectionCpu.accessor<float, 3>();
    auto distortionAcc = distortionCpu.accessor<float, 2>();
    auto pointAcc      = pointsWorldCpu.accessor<float, 2>();
    auto pixelAcc      = outPixelsCpu.accessor<float, 2>();
    auto statusAcc     = outStatusesCpu.accessor<int32_t, 1>();

    for (int64_t c = 0; c < C; ++c) {
        const float xCam = pointAcc[c][0] + worldAcc[c][0][3];
        const float yCam = pointAcc[c][1] + worldAcc[c][1][3];
        const float zCam = pointAcc[c][2] + worldAcc[c][2][3];
        const auto [expectedU, expectedV] =
            projectPointWithRadTan5Distortion(xCam,
                                              yCam,
                                              zCam,
                                              projectionAcc[c][0][0],
                                              projectionAcc[c][1][1],
                                              projectionAcc[c][0][2],
                                              projectionAcc[c][1][2],
                                              distortionAcc[c][0],
                                              distortionAcc[c][1],
                                              distortionAcc[c][2],
                                              distortionAcc[c][6],
                                              distortionAcc[c][7]);

        EXPECT_EQ(static_cast<ProjectionVisibility>(statusAcc[c]), ProjectionVisibility::InImage);
        EXPECT_NEAR(pixelAcc[c][0], expectedU, 1.0e-4f);
        EXPECT_NEAR(pixelAcc[c][1], expectedV, 1.0e-4f);
    }
}

} // namespace fvdb::detail::ops
