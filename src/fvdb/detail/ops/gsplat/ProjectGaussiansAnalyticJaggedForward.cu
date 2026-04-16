// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedForward.h>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/BinSearch.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/WarpReduce.cuh>
#include <fvdb/detail/utils/cuda/math/AffineTransform.cuh>
#include <fvdb/detail/utils/cuda/math/Rotation.cuh>
#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/utils/gsplat/GaussianMath.cuh>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename T, typename Camera>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
jaggedProjectionForwardKernel(const uint32_t B,
                              const int64_t N,
                              const int64_t *__restrict__ gSizes, // [B]
                              const int64_t *__restrict__ cSizes, // [B]
                              const int64_t *__restrict__ gIndex, // [B] start indices
                              const int64_t *__restrict__ cIndex, // [B] start indices
                              const int64_t *__restrict__ nIndex, // [B] start indices
                              const T *__restrict__ means,        // [N, 3]
                              const T *__restrict__ covars,       // [N, 6] optional
                              const T *__restrict__ quats,        // [N, 4] optional
                              const T *__restrict__ scales,       // [N, 3] optional
                              Camera camera,
                              const T eps2d,
                              const T radiusClip,
                              // outputs
                              int32_t *__restrict__ radii,  // [N]
                              T *__restrict__ means2d,      // [N, 2]
                              T *__restrict__ depths,       // [N]
                              T *__restrict__ conics,       // [N, 3]
                              T *__restrict__ compensations // [N] optional
) {
    // parallelize over N.
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // TODO: too many global memory accesses.
    const int64_t bId      = binSearch(nIndex, B, static_cast<int64_t>(idx)); // batch id
    const int64_t idxLocal = idx - nIndex[bId];      // local elem idx within Ci * Ni
    const int64_t cidLocal = idxLocal / gSizes[bId]; // local camera id within Ci
    const int64_t gidLocal = idxLocal % gSizes[bId]; // local gaussian id within Ni
    const int64_t cId      = cidLocal + cIndex[bId]; // camera id
    const int64_t gId      = gidLocal + gIndex[bId]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gId * 3;
    const nanovdb::math::Vec3<T> meanWorldSpace(means[0], means[1], means[2]);
    nanovdb::math::Mat3<T> covar;
    if (covars != nullptr) {
        covars += gId * 6;
        covar = loadCovarianceRowMajor6(covars);
    } else {
        // compute from quaternions and scales
        quats += gId * 4;
        scales += gId * 3;
        nanovdb::math::Vec4<T> quat;
        nanovdb::math::Vec3<T> scale;
        loadQuatScaleFromScalesRowMajor(quats, scales, quat, scale);
        covar = quaternionAndScaleToCovariance<T>(quat, scale);
    }
    auto [covar2d, mean2d, depthCam] = camera.projectWorldGaussianTo2D(cId, meanWorldSpace, covar);
    if (!camera.isDepthVisible(depthCam)) {
        radii[idx] = 0;
        return;
    }

    T compensation;
    const T det = addBlur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    const nanovdb::math::Mat2<T> covar2dInverse = covar2d.inverse();

    // take 3 sigma as the radius (non-differentiable)
    const T radius = radiusFromCovariance2dDet(covar2d, det, T(3));
    // T v2 = b - sqrt(max(0.1f, b * b - det));
    // T radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radiusClip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (camera.isProjectedFootprintOutsideImage(mean2d, radius, radius)) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx]                               = (int32_t)radius;
    means2d[idx * 2]                         = mean2d[0];
    means2d[idx * 2 + 1]                     = mean2d[1];
    depths[idx]                              = depthCam;
    const nanovdb::math::Vec3<T> conicPacked = packConicRowMajor3(covar2dInverse);
    conics[idx * 3]                          = conicPacked[0];
    conics[idx * 3 + 1]                      = conicPacked[1];
    conics[idx * 3 + 2]                      = conicPacked[2];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchProjectGaussiansAnalyticJaggedFwd(const torch::Tensor &gSizes, // [B] gaussian sizes
                                          const torch::Tensor &means,  // [N, 3]
                                          const torch::Tensor &quats,  // [N, 4] optional
                                          const torch::Tensor &scales, // [N, 3] optional
                                          const torch::Tensor &cSizes, // [B] camera sizes
                                          const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                          const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                          const uint32_t imageWidth,
                                          const uint32_t imageHeight,
                                          const float eps2d,
                                          const float nearPlane,
                                          const float farPlane,
                                          const float minRadius2d,
                                          const bool ortho);

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchProjectGaussiansAnalyticJaggedFwd<torch::kCUDA>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,              // [N, 4] optional
    const torch::Tensor &scales,             // [N, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
    const torch::Tensor &projectionMatrices, // [C, 3, 3]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool ortho) {
    FVDB_FUNC_RANGE();
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars = std::nullopt;
    constexpr bool calc_compensations         = false;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));
    TORCH_CHECK(gSizes.is_cuda() && gSizes.is_contiguous(),
                "gSizes must be a contiguous CUDA tensor");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be a contiguous CUDA tensor");
    if (covars.has_value()) {
        TORCH_CHECK(covars.value().is_cuda() && covars.value().is_contiguous(),
                    "covars must be a contiguous CUDA tensor");
    } else {
        TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(),
                    "quats must be a contiguous CUDA tensor");
        TORCH_CHECK(scales.is_cuda() && scales.is_contiguous(),
                    "scales must be a contiguous CUDA tensor");
    }
    TORCH_CHECK(cSizes.is_cuda() && cSizes.is_contiguous(),
                "cSizes must be a contiguous CUDA tensor");
    TORCH_CHECK(worldToCamMatrices.is_cuda() && worldToCamMatrices.is_contiguous(),
                "worldToCamMatrices must be a contiguous CUDA tensor");
    TORCH_CHECK(projectionMatrices.is_cuda() && projectionMatrices.is_contiguous(),
                "projectionMatrices must be a contiguous CUDA tensor");

    // TODO: use inclusive sum
    const uint32_t B     = gSizes.size(0);
    const int64_t C      = worldToCamMatrices.size(0);
    torch::Tensor cIndex = torch::cumsum(cSizes, 0, torch::kInt64) - cSizes;
    torch::Tensor gIndex = torch::cumsum(gSizes, 0, torch::kInt64) - gSizes;
    torch::Tensor nSize  = cSizes * gSizes;            // element size = Ci * Ni
    torch::Tensor nIndex = torch::cumsum(nSize, 0, torch::kInt64);
    const int64_t N      = nIndex[-1].item<int64_t>(); // total number of elements
    nIndex               = nIndex - nSize;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor radii   = torch::empty({N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({N, 2}, means.options());
    torch::Tensor depths  = torch::empty({N}, means.options());
    torch::Tensor conics  = torch::empty({N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({N}, means.options());
    }
    if (N) {
        if (ortho) {
            const auto camera = OrthographicCamera<float>{projectionMatrices,
                                                          worldToCamMatrices,
                                                          static_cast<int32_t>(imageWidth),
                                                          static_cast<int32_t>(imageHeight),
                                                          nearPlane,
                                                          farPlane};
            jaggedProjectionForwardKernel<float, OrthographicCamera<float>>
                <<<GET_BLOCKS(N, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                    B,
                    N,
                    gSizes.data_ptr<int64_t>(),
                    cSizes.data_ptr<int64_t>(),
                    gIndex.data_ptr<int64_t>(),
                    cIndex.data_ptr<int64_t>(),
                    nIndex.data_ptr<int64_t>(),
                    means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    camera,
                    eps2d,
                    minRadius2d,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(),
                    depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        } else {
            const auto camera = PerspectiveCamera<float>{projectionMatrices,
                                                         worldToCamMatrices,
                                                         static_cast<int32_t>(imageWidth),
                                                         static_cast<int32_t>(imageHeight),
                                                         nearPlane,
                                                         farPlane};
            jaggedProjectionForwardKernel<float, PerspectiveCamera<float>>
                <<<GET_BLOCKS(N, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                    B,
                    N,
                    gSizes.data_ptr<int64_t>(),
                    cSizes.data_ptr<int64_t>(),
                    gIndex.data_ptr<int64_t>(),
                    cIndex.data_ptr<int64_t>(),
                    nIndex.data_ptr<int64_t>(),
                    means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    camera,
                    eps2d,
                    minRadius2d,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(),
                    depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchProjectGaussiansAnalyticJaggedFwd<torch::kCPU>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,              // [N, 4] optional
    const torch::Tensor &scales,             // [N, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
    const torch::Tensor &projectionMatrices, // [C, 3, 3]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2d,
    const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
projectGaussiansAnalyticJaggedFwd(const torch::Tensor &gSizes,             // [B] gaussian sizes
                                  const torch::Tensor &means,              // [N, 3]
                                  const torch::Tensor &quats,              // [N, 4] optional
                                  const torch::Tensor &scales,             // [N, 3] optional
                                  const torch::Tensor &cSizes,             // [B] camera sizes
                                  const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                  const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                  const uint32_t imageWidth,
                                  const uint32_t imageHeight,
                                  const float eps2d,
                                  const float nearPlane,
                                  const float farPlane,
                                  const float minRadius2d,
                                  const bool ortho) {
    return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return dispatchProjectGaussiansAnalyticJaggedFwd<DeviceTag>(gSizes,
                                                                    means,
                                                                    quats,
                                                                    scales,
                                                                    cSizes,
                                                                    worldToCamMatrices,
                                                                    projectionMatrices,
                                                                    imageWidth,
                                                                    imageHeight,
                                                                    eps2d,
                                                                    nearPlane,
                                                                    farPlane,
                                                                    minRadius2d,
                                                                    ortho);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
