// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cuda/std/cmath>
#include <cuda_runtime_api.h>

namespace fvdb::detail::ops {
namespace {

using Mat2f = nanovdb::math::Mat2<float>;
using Mat3f = nanovdb::math::Mat3<float>;
using Vec2f = nanovdb::math::Vec2<float>;
using Vec3f = nanovdb::math::Vec3<float>;

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

template <typename CameraOpT>
__global__ void
compareProjectWorldTo2DKernel(int64_t C,
                              int64_t N,
                              const float *meansWorld,
                              const float *covarsWorld6,
                              CameraOpT cameraOp,
                              float *outMaxDiffs) {
    const int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
    if (idx >= C * N) {
        return;
    }
    const int64_t cid = idx / N;
    const int64_t gid = idx % N;

    const Vec3f meanWorld(meansWorld[gid * 3], meansWorld[gid * 3 + 1], meansWorld[gid * 3 + 2]);
    const Mat3f covarWorld = loadCovarianceRowMajor6(covarsWorld6 + gid * 6);

    const auto [cov2dNew, mean2dNew, depthNew] = cameraOp.projectWorldGaussianTo2D(cid, meanWorld, covarWorld);

    const auto [R, t]   = cameraOp.worldToCamRt(cid);
    const Vec3f meanCam = transformPointWorldToCam(R, t, meanWorld);
    const Mat3f covCam  = transformCovarianceWorldToCam(R, covarWorld);
    const auto [cov2dRef, mean2dRef] = cameraOp.projectTo2DGaussian(cid, meanCam, covCam);
    const float depthRef             = meanCam[2];

    outMaxDiffs[idx * 3 + 0] = maxAbsMat2(cov2dNew, cov2dRef);
    outMaxDiffs[idx * 3 + 1] = maxAbsVec2(mean2dNew, mean2dRef);
    outMaxDiffs[idx * 3 + 2] = ::cuda::std::fabs(depthNew - depthRef);
}

template <typename CameraOpT>
__global__ void
compareProjectWorldTo2DVJPKernel(int64_t C,
                                 int64_t N,
                                 const float *meansWorld,
                                 const float *covarsWorld6,
                                 const float *dLossDCov2d,
                                 const float *dLossDMean2d,
                                 const float *dLossDDepth,
                                 CameraOpT cameraOp,
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
        cameraOp.projectWorldGaussianTo2DVJP(cid, meanWorld, covarWorld, dCov2d, dMean2d, dDepth);

    const auto [R, t] = cameraOp.worldToCamRt(cid);
    const Vec3f meanCam = transformPointWorldToCam(R, t, meanWorld);
    const Mat3f covCam  = transformCovarianceWorldToCam(R, covarWorld);
    auto [dCovCamRef, dMeanCamRef] = cameraOp.projectTo2DGaussianVJP(cid, meanCam, covCam, dCov2d, dMean2d);
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

torch::Tensor
makeWorldToCamMatrices(int64_t C) {
    auto worldToCamCpu = torch::zeros({C, 4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc           = worldToCamCpu.accessor<float, 3>();
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
    auto kCpu = torch::zeros({C, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc  = kCpu.accessor<float, 3>();
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
    auto meansCpu = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc      = meansCpu.accessor<float, 2>();
    for (int64_t i = 0; i < N; ++i) {
        acc[i][0] = -0.2f + 0.01f * static_cast<float>(i);
        acc[i][1] = 0.15f - 0.008f * static_cast<float>(i);
        acc[i][2] = 2.0f + 0.02f * static_cast<float>(i);
    }
    return meansCpu.cuda();
}

torch::Tensor
makeCovarsWorld6(int64_t N) {
    auto covCpu = torch::zeros({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc    = covCpu.accessor<float, 2>();
    for (int64_t i = 0; i < N; ++i) {
        const float f = 0.001f * static_cast<float>(i);
        acc[i][0]     = 0.05f + f;   // xx
        acc[i][1]     = 0.001f + f;  // xy
        acc[i][2]     = -0.0005f;    // xz
        acc[i][3]     = 0.07f + f;   // yy
        acc[i][4]     = 0.0007f;     // yz
        acc[i][5]     = 0.09f + f;   // zz
    }
    return covCpu.cuda();
}

template <typename CameraOpT>
void
runForwardAndVjpParityChecks(const CameraOpT &cameraOp, int64_t C, int64_t N, const torch::Tensor &means, const torch::Tensor &covars6) {
    constexpr int kBlock = 256;
    const int64_t total  = C * N;
    const dim3 grid((total + kBlock - 1) / kBlock);
    const dim3 block(kBlock);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    auto fwdDiffs = torch::zeros({total, 3}, means.options());
    compareProjectWorldTo2DKernel<<<grid, block, 0, stream>>>(
        C,
        N,
        means.data_ptr<float>(),
        covars6.data_ptr<float>(),
        cameraOp,
        fwdDiffs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto dCov2d   = torch::randn({total, 4}, means.options());
    auto dMean2d  = torch::randn({total, 2}, means.options());
    auto dDepth   = torch::randn({total}, means.options());
    auto bwdDiffs = torch::zeros({total, 4}, means.options());
    compareProjectWorldTo2DVJPKernel<<<grid, block, 0, stream>>>(
        C,
        N,
        means.data_ptr<float>(),
        covars6.data_ptr<float>(),
        dCov2d.data_ptr<float>(),
        dMean2d.data_ptr<float>(),
        dDepth.data_ptr<float>(),
        cameraOp,
        bwdDiffs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float maxFwd = fwdDiffs.abs().max().item<float>();
    const float maxBwd = bwdDiffs.abs().max().item<float>();
    EXPECT_LT(maxFwd, 1e-6f);
    EXPECT_LT(maxBwd, 1e-6f);
}

} // namespace

TEST(GaussianCamerasOpTest, PerspectiveEncapsulatedProjectionAndVJPMatchReferencePath) {
    constexpr int64_t C = 3;
    constexpr int64_t N = 64;
    auto worldToCam      = makeWorldToCamMatrices(C);
    auto projection      = makeProjectionMatrices(C, false);
    auto meansWorld      = makeMeansWorld(N);
    auto covarsWorld6    = makeCovarsWorld6(N);

    auto cameraOp = PerspectiveCameraOp<float>(
        projection, worldToCam, static_cast<int32_t>(C), 640, 480, 0.01f, 1.0e8f);
    runForwardAndVjpParityChecks(cameraOp, C, N, meansWorld, covarsWorld6);
}

TEST(GaussianCamerasOpTest, OrthographicEncapsulatedProjectionAndVJPMatchReferencePath) {
    constexpr int64_t C = 2;
    constexpr int64_t N = 64;
    auto worldToCam      = makeWorldToCamMatrices(C);
    auto projection      = makeProjectionMatrices(C, true);
    auto meansWorld      = makeMeansWorld(N);
    auto covarsWorld6    = makeCovarsWorld6(N);

    auto cameraOp = OrthographicCameraOp<float>(
        projection, worldToCam, static_cast<int32_t>(C), 640, 480, -1.0e8f, 1.0e8f);
    runForwardAndVjpParityChecks(cameraOp, C, N, meansWorld, covarsWorld6);
}

} // namespace fvdb::detail::ops
