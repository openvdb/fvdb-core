// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// We repeat this code everywhere in evalShFunctionVJP to compute the gradient of the
// direction and write it out, so pull this into a function.
template <typename T>
__device__ inline void
writeDLossDViewDir(
    T x, T y, T z, T vX, T vY, T vZ, T inorm, typename Vec3Type<T>::type *dLossDViewDir) {
    using vec3t                     = typename Vec3Type<T>::type;
    const T dLossDViewDirDotViewDir = x * vX + y * vY + z * vZ;

    dLossDViewDir->x = (vX - dLossDViewDirDotViewDir * x) * inorm;
    dLossDViewDir->y = (vY - dLossDViewDirDotViewDir * y) * inorm;
    dLossDViewDir->z = (vZ - dLossDViewDirDotViewDir * z) * inorm;
}

template <typename T>
inline __device__ void
evalShFunctionVJP(const int64_t degree,                     // degree of SH to be evaluated
                  const int64_t ci,                         // camera index
                  const int64_t gi,                         // gaussian index
                  const int64_t c,                          // render channel
                  const typename Vec3Type<T>::type &dir,    // [3]
                  const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> shCoeffs,
                  const T *dLossDRenderQuantities,          // [D]
                  torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> dLossDShCoeffs,
                  typename Vec3Type<T>::type *dLossDViewDir // [3] optional
) {
    T dLossDRenderQuantitiesLocal = dLossDRenderQuantities[c];

    gpuAtomicAdd(&dLossDShCoeffs[gi][0][c], T(0.2820947917738781) * dLossDRenderQuantitiesLocal);

    if (degree < 1) {
        return;
    }
    const T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const T x     = dir.x * inorm;
    const T y     = dir.y * inorm;
    const T z     = dir.z * inorm;
    T vX = 0.f, vY = 0.f, vZ = 0.f;

    gpuAtomicAdd(&dLossDShCoeffs[gi][1][c], T(-0.48860251190292) * y * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][2][c], T(0.48860251190292) * z * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][3][c], T(-0.48860251190292) * x * dLossDRenderQuantitiesLocal);

    if (dLossDViewDir != nullptr) {
        vX += T(-0.48860251190292) * shCoeffs[gi][3][c] * dLossDRenderQuantitiesLocal;
        vY += T(-0.48860251190292) * shCoeffs[gi][1][c] * dLossDRenderQuantitiesLocal;
        vZ += T(0.48860251190292) * shCoeffs[gi][2][c] * dLossDRenderQuantitiesLocal;
    }
    if (degree < 2) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T z2     = z * z;
    const T fTmp0B = T(-1.092548430592079) * z;
    const T fC1    = x * x - y * y;
    const T fS1    = 2.f * x * y;
    const T pSH6   = (T(0.9461746957575601) * z2 - T(0.3153915652525201));
    const T pSH7   = fTmp0B * x;
    const T pSH5   = fTmp0B * y;
    const T pSH8   = T(0.5462742152960395) * fC1;
    const T pSH4   = T(0.5462742152960395) * fS1;
    gpuAtomicAdd(&dLossDShCoeffs[gi][4][c], pSH4 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][5][c], pSH5 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][6][c], pSH6 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][7][c], pSH7 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][8][c], pSH8 * dLossDRenderQuantitiesLocal);

    T fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y,
        pSH4_x, pSH4_y;
    if (dLossDViewDir != nullptr) {
        fTmp0B_z = T(-1.092548430592079);
        fC1_x    = T(2.0) * x;
        fC1_y    = T(-2.0) * y;
        fS1_x    = T(2.0) * y;
        fS1_y    = T(2.0) * x;
        pSH6_z   = T(2.0) * T(0.9461746957575601) * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = T(0.5462742152960395) * fC1_x;
        pSH8_y   = T(0.5462742152960395) * fC1_y;
        pSH4_x   = T(0.5462742152960395) * fS1_x;
        pSH4_y   = T(0.5462742152960395) * fS1_y;

        vX += dLossDRenderQuantitiesLocal *
              (pSH4_x * shCoeffs[gi][4][c] + pSH8_x * shCoeffs[gi][8][c] +
               pSH7_x * shCoeffs[gi][7][c]);
        vY += dLossDRenderQuantitiesLocal *
              (pSH4_y * shCoeffs[gi][4][c] + pSH8_y * shCoeffs[gi][8][c] +
               pSH5_y * shCoeffs[gi][5][c]);
        vZ += dLossDRenderQuantitiesLocal *
              (pSH6_z * shCoeffs[gi][6][c] + pSH7_z * shCoeffs[gi][7][c] +
               pSH5_z * shCoeffs[gi][5][c]);
    }

    if (degree < 3) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T fTmp0C = T(-2.285228997322329) * z2 + T(0.4570457994644658);
    const T fTmp1B = T(1.445305721320277) * z;
    const T fC2    = x * fC1 - y * fS1;
    const T fS2    = x * fS1 + y * fC1;
    const T pSH12  = z * (T(1.865881662950577) * z2 - T(1.119528997770346));
    const T pSH13  = fTmp0C * x;
    const T pSH11  = fTmp0C * y;
    const T pSH14  = fTmp1B * fC1;
    const T pSH10  = fTmp1B * fS1;
    const T pSH15  = T(-0.5900435899266435) * fC2;
    const T pSH9   = T(-0.5900435899266435) * fS2;

    gpuAtomicAdd(&dLossDShCoeffs[gi][9][c], pSH9 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][10][c], pSH10 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][11][c], pSH11 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][12][c], pSH12 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][13][c], pSH13 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][14][c], pSH14 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][15][c], pSH15 * dLossDRenderQuantitiesLocal);

    T fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z,
        pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (dLossDViewDir != nullptr) {
        fTmp0C_z = T(-2.285228997322329) * T(2.0) * z;
        fTmp1B_z = T(1.445305721320277);
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = T(3.0) * T(1.865881662950577) * z2 - T(1.119528997770346);
        pSH13_x  = fTmp0C;
        pSH13_z  = fTmp0C_z * x;
        pSH11_y  = fTmp0C;
        pSH11_z  = fTmp0C_z * y;
        pSH14_x  = fTmp1B * fC1_x;
        pSH14_y  = fTmp1B * fC1_y;
        pSH14_z  = fTmp1B_z * fC1;
        pSH10_x  = fTmp1B * fS1_x;
        pSH10_y  = fTmp1B * fS1_y;
        pSH10_z  = fTmp1B_z * fS1;
        pSH15_x  = T(-0.5900435899266435) * fC2_x;
        pSH15_y  = T(-0.5900435899266435) * fC2_y;
        pSH9_x   = T(-0.5900435899266435) * fS2_x;
        pSH9_y   = T(-0.5900435899266435) * fS2_y;

        const T cSH9  = shCoeffs[gi][9][c];
        const T cSH10 = shCoeffs[gi][10][c];
        const T cSH11 = shCoeffs[gi][11][c];
        const T cSH12 = shCoeffs[gi][12][c];
        const T cSH13 = shCoeffs[gi][13][c];
        const T cSH14 = shCoeffs[gi][14][c];
        const T cSH15 = shCoeffs[gi][15][c];

        vX += dLossDRenderQuantitiesLocal * (pSH9_x * cSH9 + pSH15_x * cSH15 + pSH10_x * cSH10 +
                                             pSH14_x * cSH14 + pSH13_x * cSH13);

        vY += dLossDRenderQuantitiesLocal * (pSH9_y * cSH9 + pSH15_y * cSH15 + pSH10_y * cSH10 +
                                             pSH14_y * cSH14 + pSH11_y * cSH11);

        vZ += dLossDRenderQuantitiesLocal * (pSH12_z * cSH12 + pSH13_z * cSH13 + pSH11_z * cSH11 +
                                             pSH14_z * cSH14 + pSH10_z * cSH10);
    }

    if (degree < 4) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T fTmp0D = z * (T(-4.683325804901025) * z2 + T(2.007139630671868));
    const T fTmp1C = T(3.31161143515146) * z2 - T(0.47308734787878);
    const T fTmp2B = -1.770130769779931f * z;
    const T fC3    = x * fC2 - y * fS2;
    const T fS3    = x * fS2 + y * fC2;
    const T pSH20  = (T(1.984313483298443) * z * pSH12 + T(-1.006230589874905) * pSH6);
    const T pSH21  = fTmp0D * x;
    const T pSH19  = fTmp0D * y;
    const T pSH22  = fTmp1C * fC1;
    const T pSH18  = fTmp1C * fS1;
    const T pSH23  = fTmp2B * fC2;
    const T pSH17  = fTmp2B * fS2;
    const T pSH24  = T(0.6258357354491763) * fC3;
    const T pSH16  = T(0.6258357354491763) * fS3;

    gpuAtomicAdd(&dLossDShCoeffs[gi][16][c], pSH16 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][17][c], pSH17 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][18][c], pSH18 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][19][c], pSH19 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][20][c], pSH20 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][21][c], pSH21 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][22][c], pSH22 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][23][c], pSH23 * dLossDRenderQuantitiesLocal);
    gpuAtomicAdd(&dLossDShCoeffs[gi][24][c], pSH24 * dLossDRenderQuantitiesLocal);

    T fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y,
        pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z,
        pSH17_x, pSH17_y, pSH17_z, pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (dLossDViewDir != nullptr) {
        fTmp0D_z = T(3.0) * T(-4.683325804901025) * z2 + T(2.007139630671868);
        fTmp1C_z = T(2.0) * 3.31161143515146f * z;
        fTmp2B_z = T(-1.770130769779931);
        fC3_x    = fC2 + x * fC2_x - y * fS2_x;
        fC3_y    = x * fC2_y - fS2 - y * fS2_y;
        fS3_x    = fS2 + y * fC2_x + x * fS2_x;
        fS3_y    = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z  = T(1.984313483298443) * (pSH12 + z * pSH12_z) + T(-1.006230589874905) * pSH6_z;
        pSH21_x  = fTmp0D;
        pSH21_z  = fTmp0D_z * x;
        pSH19_y  = fTmp0D;
        pSH19_z  = fTmp0D_z * y;
        pSH22_x  = fTmp1C * fC1_x;
        pSH22_y  = fTmp1C * fC1_y;
        pSH22_z  = fTmp1C_z * fC1;
        pSH18_x  = fTmp1C * fS1_x;
        pSH18_y  = fTmp1C * fS1_y;
        pSH18_z  = fTmp1C_z * fS1;
        pSH23_x  = fTmp2B * fC2_x;
        pSH23_y  = fTmp2B * fC2_y;
        pSH23_z  = fTmp2B_z * fC2;
        pSH17_x  = fTmp2B * fS2_x;
        pSH17_y  = fTmp2B * fS2_y;
        pSH17_z  = fTmp2B_z * fS2;
        pSH24_x  = T(0.6258357354491763) * fC3_x;
        pSH24_y  = T(0.6258357354491763) * fC3_y;
        pSH16_x  = T(0.6258357354491763) * fS3_x;
        pSH16_y  = T(0.6258357354491763) * fS3_y;

        const T cSH16 = shCoeffs[gi][16][c];
        const T cSH17 = shCoeffs[gi][17][c];
        const T cSH18 = shCoeffs[gi][18][c];
        const T cSH19 = shCoeffs[gi][19][c];
        const T cSH20 = shCoeffs[gi][20][c];
        const T cSH21 = shCoeffs[gi][21][c];
        const T cSH22 = shCoeffs[gi][22][c];
        const T cSH23 = shCoeffs[gi][23][c];
        const T cSH24 = shCoeffs[gi][24][c];

        vX += dLossDRenderQuantitiesLocal *
              (pSH16_x * cSH16 + pSH24_x * cSH24 + pSH17_x * cSH17 + pSH23_x * cSH23 +
               pSH18_x * cSH18 + pSH22_x * cSH22 + pSH21_x * cSH21);
        vY += dLossDRenderQuantitiesLocal *
              (pSH16_y * cSH16 + pSH24_y * cSH24 + pSH17_y * cSH17 + pSH23_y * cSH23 +
               pSH18_y * cSH18 + pSH22_y * cSH22 + pSH19_y * cSH19);
        vZ += dLossDRenderQuantitiesLocal *
              (pSH20_z * cSH20 + pSH21_z * cSH21 + pSH19_z * cSH19 + pSH22_z * cSH22 +
               pSH18_z * cSH18 + pSH23_z * cSH23 + pSH17_z * cSH17);

        writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
    }
}

template <typename T>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
computeShBackward(
    const int64_t offset,
    const int64_t count,
    const int64_t C,
    const int64_t N,
    const int64_t K,
    const int64_t D,
    const int64_t shDegreeToUse,
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> viewDirs,    // [C, N, 3]
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> shCoeffs,    // [N, K, D]
    const int *__restrict__ radii,                                                   // [C, N]
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits>
        dLossDRenderQuantities,                                                      // [C, N, D]
    torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> outDLossDShCoeffs, // [N, K, D]
    T *__restrict__ outDLossDViewDirs // [C, N, 3] optional
) {
    // parallelize over C * N * D
    auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= count * C * D) {
        return;
    }

    const auto eid = idx / D;              // cidx * N + gidx
    const auto cid = eid / count;          // camera index
    const auto gid = eid % count + offset; // gaussian index
    const auto c   = idx % D;              // render channel
    if (radii != nullptr && radii[cid * N + gid] <= 0) {
        return;
    }

    using vec3t            = typename Vec3Type<T>::type;
    const bool hasViewDirs = viewDirs.size(0) > 0;
    const vec3t viewDir    = hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[cid][gid].data())
                                         : vec3t{T(0), T(0), T(0)};
    const T *dLossDRenderQuantityPtr = dLossDRenderQuantities[cid][gid].data();

    vec3t dLossDViewDir{T(0), T(0), T(0)};
    vec3t *outDLossDViewDirPtr = outDLossDViewDirs == nullptr ? nullptr : &dLossDViewDir;

    evalShFunctionVJP(shDegreeToUse,
                      cid,
                      gid,
                      c,
                      viewDir,
                      shCoeffs,
                      dLossDRenderQuantityPtr,
                      outDLossDShCoeffs,
                      outDLossDViewDirPtr);
    if (outDLossDViewDirs != nullptr) {
        gpuAtomicAdd(outDLossDViewDirs + (cid * N + gid) * 3, dLossDViewDir.x);
        gpuAtomicAdd(outDLossDViewDirs + (cid * N + gid) * 3 + 1, dLossDViewDir.y);
        gpuAtomicAdd(outDLossDViewDirs + (cid * N + gid) * 3 + 2, dLossDViewDir.z);
    }
}

template <typename T>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
computeShDiffuseOnlyBackward(
    const int64_t offset,
    const int64_t count,
    const int64_t C,
    const int64_t N,
    const int64_t D,
    const torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits>
        dLossDRenderQuantities,                                                     // [C, N, D]
    const int *__restrict__ radii,                                                  // [C, N]
    torch::PackedTensorAccessor64<T, 3, torch::RestrictPtrTraits> outDLossDShCoeffs // [N, K, D]
) {
    // parallelize over C * N * D
    auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= count * C * D) {
        return;
    }

    const auto eid = idx / D;              // cidx * N + gidx
    const auto cid = eid / count;          // camera index
    const auto gid = eid % count + offset; // gaussian index
    const auto c   = idx % D;              // render channel
    if (radii != nullptr && radii[cid * N + gid] <= 0) {
        return;
    }

    gpuAtomicAdd(&outDLossDShCoeffs[gid][0][c],
                 T(0.2820947917738781) * dLossDRenderQuantities[cid][gid][c]);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCUDA>(
    const int64_t shDegreeToUse,
    const int64_t numCameras,
    const int64_t numGaussians,
    const torch::Tensor &viewDirs,               // [C, N, 3]
    const torch::Tensor &shCoeffs,               // [N, K, D]
    const torch::Tensor &dLossDRenderQuantities, // [C, N, D]
    const torch::Tensor &radii,                  // [C, N]
    const bool computeDLossDViewDirs) {
    FVDB_FUNC_RANGE();
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(dLossDRenderQuantities));

    const bool hasViewDirs = viewDirs.defined();
    const bool hasRadii    = radii.defined();

    TORCH_CHECK_VALUE(shCoeffs.dim() == 3, "shCoeffs must have shape [N, K, D]");
    TORCH_CHECK_VALUE(shCoeffs.is_cuda(), "shCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(shCoeffs.size(0) == numGaussians, "shCoeffs must have shape [N, K, D]");

    const int64_t K = shCoeffs.size(1);
    const int64_t N = dLossDRenderQuantities.size(1);
    const int64_t C = numCameras;
    const int64_t D = dLossDRenderQuantities.size(2);

    if (shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(K > 1, "shCoeffs must have K > 1 for shDegreeToUse > 0");
        TORCH_CHECK_VALUE(hasViewDirs, "viewDirs must be defined for shDegreeToUse > 0");
    }

    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have two dimensions with shape [C, N]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1),
                          "radii must have shape [C, N] but got shape = ",
                          radii.sizes());
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    const int64_t TOTAL_ELEMS = C * N * D;
    const int64_t NUM_BLOCKS  = GET_BLOCKS(TOTAL_ELEMS, DEFAULT_BLOCK_DIM);

    if (shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(viewDirs.dim() == 3, "viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(
            N == viewDirs.size(1),
            "shCoeffs must have shape [N, K, D] and viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(viewDirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(viewDirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    at::cuda::CUDAStream stream =
        at::cuda::getCurrentCUDAStream(dLossDRenderQuantities.device().index());

    using scalar_t = float;

    const int *radiiPtr = hasRadii ? radii.data_ptr<int>() : nullptr;

    const auto tensorOptions = dLossDRenderQuantities.options();
    if (shDegreeToUse > 0) {
        torch::Tensor dLossDShCoeffs = torch::zeros({N, K, D}, tensorOptions);
        torch::Tensor dLossDViewDirs;
        if (computeDLossDViewDirs) {
            dLossDViewDirs = torch::zeros_like(viewDirs);
        }
        if (N == 0) {
            return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
        }

        computeShBackward<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
            0,
            N,
            C,
            N,
            K,
            D,
            shDegreeToUse,
            viewDirs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            shCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            dLossDRenderQuantities.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLossDShCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            computeDLossDViewDirs ? dLossDViewDirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
    } else {
        torch::Tensor dLossDShCoeffs = torch::zeros({N, 1, D}, tensorOptions);
        torch::Tensor dLossDViewDirs;
        if (N == 0) {
            return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
        }

        computeShDiffuseOnlyBackward<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
            0,
            N,
            C,
            N,
            D,
            dLossDRenderQuantities.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            dLossDShCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kPrivateUse1>(
    const int64_t shDegreeToUse,
    const int64_t numCameras,
    const int64_t numGaussians,
    const torch::Tensor &viewDirs,               // [C, N, 3]
    const torch::Tensor &shCoeffs,               // [N, K, D]
    const torch::Tensor &dLossDRenderQuantities, // [C, N, D]
    const torch::Tensor &radii,                  // [C, N]
    const bool computeDLossDViewDirs) {
    FVDB_FUNC_RANGE();

    const bool hasViewDirs = viewDirs.defined();
    const bool hasRadii    = radii.defined();

    TORCH_CHECK_VALUE(shCoeffs.dim() == 3, "shCoeffs must have shape [N, K, D]");
    TORCH_CHECK_VALUE(shCoeffs.is_privateuseone(), "shCoeffs must be a PrivateUse1 tensor");
    TORCH_CHECK_VALUE(shCoeffs.size(0) == numGaussians, "shCoeffs must have shape [N, K, D]");

    const int64_t K = shCoeffs.size(1);
    const int64_t N = dLossDRenderQuantities.size(1);
    const int64_t C = numCameras;
    const int64_t D = dLossDRenderQuantities.size(2);

    if (shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(K > 1, "shCoeffs must have K > 1 for shDegreeToUse > 0");
        TORCH_CHECK_VALUE(hasViewDirs, "viewDirs must be defined for shDegreeToUse > 0");
    }

    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have two dimensions with shape [C, N]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1),
                          "radii must have shape [C, N] but got shape = ",
                          radii.sizes());
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_privateuseone(), "radii must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    if (shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(viewDirs.dim() == 3, "viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(
            N == viewDirs.size(1),
            "shCoeffs must have shape [N, K, D] and viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(viewDirs.is_privateuseone(), "dirs must be a PrivateUse1 tensor");
        TORCH_CHECK_VALUE(viewDirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    using scalar_t = float;

    const int *radiiPtr = hasRadii ? radii.data_ptr<int>() : nullptr;

    const auto tensorOptions = dLossDRenderQuantities.options();
    if (shDegreeToUse > 0) {
        torch::Tensor dLossDShCoeffs = torch::zeros({N, K, D}, tensorOptions);
        torch::Tensor dLossDViewDirs;
        if (computeDLossDViewDirs) {
            dLossDViewDirs = torch::empty_like(viewDirs);
        }
        if (N == 0) {
            return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
        }

        std::vector<cudaEvent_t> events(c10::cuda::device_count());
        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaEventCreate(&events[deviceId], cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
        }

        if (computeDLossDViewDirs) {
            for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
                C10_CUDA_CHECK(cudaSetDevice(deviceId));
                auto stream = c10::cuda::getStreamFromPool(false, deviceId);
                C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));

                int64_t elementOffset, elementCount;
                std::tie(elementOffset, elementCount) = deviceChunk(N, deviceId);

#if (CUDART_VERSION < 13000)
                for (int cameraIndex = 0; cameraIndex < C; ++cameraIndex) {
                    nanovdb::util::cuda::memPrefetchAsync(
                        dLossDViewDirs.data_ptr<scalar_t>() +
                            cameraIndex * dLossDViewDirs.stride(0) +
                            elementOffset * dLossDViewDirs.stride(1),
                        elementCount * dLossDViewDirs.stride(1) * sizeof(scalar_t),
                        deviceId,
                        stream);
                }
#else
                std::vector<void *> prefetchPtrs;
                std::vector<size_t> prefetchSizes;
                const cudaMemLocation location = {cudaMemLocationTypeDevice, deviceId};
                std::vector<cudaMemLocation> prefetchLocations = {location};
                std::vector<size_t> prefetchLocationIndices    = {0};

                for (int cameraIndex = 0; cameraIndex < C; ++cameraIndex) {
                    prefetchPtrs.emplace_back(dLossDViewDirs.data_ptr<scalar_t>() +
                                              cameraIndex * dLossDViewDirs.stride(0) +
                                              elementOffset * dLossDViewDirs.stride(1));
                    prefetchSizes.emplace_back(elementCount * dLossDViewDirs.stride(1) *
                                               sizeof(scalar_t));
                }

                C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(prefetchPtrs.data(),
                                                         prefetchSizes.data(),
                                                         prefetchPtrs.size(),
                                                         prefetchLocations.data(),
                                                         prefetchLocationIndices.data(),
                                                         prefetchLocations.size(),
                                                         0,
                                                         stream));
#endif
                for (int cameraIndex = 0; cameraIndex < C; ++cameraIndex) {
                    C10_CUDA_CHECK(
                        cudaMemsetAsync(dLossDViewDirs.data_ptr<scalar_t>() +
                                            cameraIndex * dLossDViewDirs.stride(0) +
                                            elementOffset * dLossDViewDirs.stride(1),
                                        0,
                                        elementCount * dLossDViewDirs.stride(1) * sizeof(scalar_t),
                                        stream));
                }
                C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
            }
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
            C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));

            int64_t elementOffset, elementCount;
            std::tie(elementOffset, elementCount) = deviceChunk(N, deviceId);

            const auto NUM_BLOCKS = GET_BLOCKS(C * elementCount * D, DEFAULT_BLOCK_DIM);

            computeShBackward<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                elementOffset,
                elementCount,
                C,
                N,
                K,
                D,
                shDegreeToUse,
                viewDirs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                shCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                radiiPtr,
                dLossDRenderQuantities.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                dLossDShCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                computeDLossDViewDirs ? dLossDViewDirs.data_ptr<scalar_t>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        mergeStreams();

        return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
    } else {
        torch::Tensor dLossDShCoeffs = torch::zeros({N, 1, D}, tensorOptions);
        torch::Tensor dLossDViewDirs;
        if (N == 0) {
            return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

            int64_t elementOffset, elementCount;
            std::tie(elementOffset, elementCount) = deviceChunk(N, deviceId);

            const auto NUM_BLOCKS = GET_BLOCKS(C * elementCount * D, DEFAULT_BLOCK_DIM);

            computeShDiffuseOnlyBackward<scalar_t><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM, 0, stream>>>(
                elementOffset,
                elementCount,
                C,
                N,
                D,
                dLossDRenderQuantities.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                radiiPtr,
                dLossDShCoeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        mergeStreams();

        return std::make_tuple(dLossDShCoeffs, dLossDViewDirs);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCPU>(
    const int64_t shDegreeToUse,
    const int64_t numCameras,
    const int64_t numGaussians,
    const torch::Tensor &viewDirs,               // [C, N, 3]
    const torch::Tensor &shCoeffs,               // [N, K, D]
    const torch::Tensor &dLossDRenderQuantities, // [C, N, D]
    const torch::Tensor &radii,                  // [C, N]
    const bool computeDLossDViewDirs) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
