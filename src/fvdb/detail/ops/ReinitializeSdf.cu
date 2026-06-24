// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/ReinitializeSdf.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/VoxelBlockManagerHelper.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/math/Math.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// =====================  fused stencil kernels ====================================================
// frozen Peng smoothed sign from a field's value + central-difference gradient.
template <typename ScalarT>
__global__ void
signFusedKernel(const OnIndexGridT *grid,
                const uint32_t *firstLeafID,
                const uint64_t *jumpMap,
                uint64_t firstOffset,
                const ScalarT *field,
                ScalarT voxelSize,
                ScalarT *sign) {
    VbmFaceStencil faces;
    if (!vbmDecodeFaceStencil<kLog2BlockWidth>(grid, firstLeafID, jumpMap, firstOffset, faces))
        return;
    const uint64_t centerIndex = faces.centerIndex;
    const uint64_t *faceIndex  = faces.faceIndex;
    ScalarT xm = field[faceIndex[0]], xp = field[faceIndex[1]], ym = field[faceIndex[2]],
            yp = field[faceIndex[3]], zm = field[faceIndex[4]], zp = field[faceIndex[5]];
    ScalarT gradX = (xp - xm) / (2 * voxelSize), gradY = (yp - ym) / (2 * voxelSize),
            gradZ = (zp - zm) / (2 * voxelSize), phiCenter = field[centerIndex];
    sign[centerIndex] =
        phiCenter / nanovdb::math::Sqrt(phiCenter * phiCenter +
                                        (gradX * gradX + gradY * gradY + gradZ * gradZ) *
                                            voxelSize * voxelSize +
                                        ScalarT(1e-12));
}

// One-sided upwind selection of the squared one-dimensional derivative for the Godunov scheme.
template <typename ScalarT>
__device__ inline ScalarT
upwind(ScalarT backwardDiff, ScalarT forwardDiff, ScalarT sgn) {
    if (sgn > 0) {
        ScalarT backTerm = nanovdb::math::Max(backwardDiff, ScalarT(0));
        ScalarT fwdTerm  = nanovdb::math::Min(forwardDiff, ScalarT(0));
        return nanovdb::math::Max(backTerm * backTerm, fwdTerm * fwdTerm);
    } else {
        ScalarT backTerm = nanovdb::math::Min(backwardDiff, ScalarT(0));
        ScalarT fwdTerm  = nanovdb::math::Max(forwardDiff, ScalarT(0));
        return nanovdb::math::Max(backTerm * backTerm, fwdTerm * fwdTerm);
    }
}

// Godunov RHS: d phi/dt = sign * (1 - |grad phi|), one-sided upwind on the 6 faces.
template <typename ScalarT>
__global__ void
godunovFusedKernel(const OnIndexGridT *grid,
                   const uint32_t *firstLeafID,
                   const uint64_t *jumpMap,
                   uint64_t firstOffset,
                   const ScalarT *field,
                   const ScalarT *sign,
                   ScalarT voxelSize,
                   ScalarT *rhs) {
    VbmFaceStencil faces;
    if (!vbmDecodeFaceStencil<kLog2BlockWidth>(grid, firstLeafID, jumpMap, firstOffset, faces))
        return;
    const uint64_t centerIndex = faces.centerIndex;
    const uint64_t *faceIndex  = faces.faceIndex;
    ScalarT center = field[centerIndex], sgn = sign[centerIndex];
    ScalarT xm = field[faceIndex[0]], xp = field[faceIndex[1]], ym = field[faceIndex[2]],
            yp = field[faceIndex[3]], zm = field[faceIndex[4]], zp = field[faceIndex[5]];
    ScalarT gradMag = nanovdb::math::Sqrt(
        upwind<ScalarT>((center - xm) / voxelSize, (xp - center) / voxelSize, sgn) +
        upwind<ScalarT>((center - ym) / voxelSize, (yp - center) / voxelSize, sgn) +
        upwind<ScalarT>((center - zm) / voxelSize, (zp - center) / voxelSize, sgn));
    rhs[centerIndex] = sgn * (ScalarT(1) - gradMag);
}

// one umbrella-Laplacian smoothing pass: out[centerIndex] = in + weight*(faceMean - in).
// Double-buffered (in != out) so neighbour reads see the pre-pass field.
template <typename ScalarT>
__global__ void
smoothFusedKernel(const OnIndexGridT *grid,
                  const uint32_t *firstLeafID,
                  const uint64_t *jumpMap,
                  uint64_t firstOffset,
                  const ScalarT *in,
                  ScalarT weight,
                  ScalarT *out) {
    VbmFaceStencil faces;
    if (!vbmDecodeFaceStencil<kLog2BlockWidth>(grid, firstLeafID, jumpMap, firstOffset, faces))
        return;
    const uint64_t centerIndex = faces.centerIndex;
    const uint64_t *faceIndex  = faces.faceIndex;
    ScalarT center             = in[centerIndex];
    ScalarT faceMean = (in[faceIndex[0]] + in[faceIndex[1]] + in[faceIndex[2]] + in[faceIndex[3]] +
                        in[faceIndex[4]] + in[faceIndex[5]]) *
                       (ScalarT(1) / ScalarT(6));
    out[centerIndex] = center + weight * (faceMean - center);
}

// =====================  value-indexed kernels (no stencil) =======================================
template <typename ScalarT>
__global__ void
fillKernel(ScalarT *data, int64_t count, ScalarT value) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        data[i] = value;
}

// out = clip(phiBaseCoeff*phiBase + stageCoeff*stage + rhsCoeff*timeStep*rhs, -bandWidth,
// bandWidth) (the TVD-RK combiners). Never writes slot 0.
template <typename ScalarT>
__global__ void
combineKernel(ScalarT *out,
              const ScalarT *phiBase,
              const ScalarT *stage,
              const ScalarT *rhs,
              ScalarT phiBaseCoeff,
              ScalarT stageCoeff,
              ScalarT rhsCoeff,
              ScalarT timeStep,
              ScalarT bandWidth,
              int64_t valueCount) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 1 || i >= valueCount)
        return;
    ScalarT value = phiBaseCoeff * phiBase[i] +
                    (stageCoeff != ScalarT(0) ? stageCoeff * stage[i] : ScalarT(0)) +
                    rhsCoeff * timeStep * rhs[i];
    out[i] = nanovdb::math::Min(nanovdb::math::Max(value, -bandWidth), bandWidth);
}

// Heun final: out = clip(phiBase + 0.5 timeStep (rhs0+rhs1), -bandWidth, bandWidth). Never writes
// slot 0.
template <typename ScalarT>
__global__ void
heunKernel(ScalarT *out,
           const ScalarT *phiBase,
           const ScalarT *rhs0,
           const ScalarT *rhs1,
           ScalarT timeStep,
           ScalarT bandWidth,
           int64_t valueCount) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 1 || i >= valueCount)
        return;
    ScalarT value = phiBase[i] + ScalarT(0.5) * timeStep * (rhs0[i] + rhs1[i]);
    out[i]        = nanovdb::math::Min(nanovdb::math::Max(value, -bandWidth), bandWidth);
}

// Redistance (|grad phi| = 1) + optional de-staircase one grid's value-indexed buffer, in place.
// `phi`/scratch are length `valueCount` with slot 0 holding the +bandWidth background; the
// stencil/combiner kernels never write slot 0 (so inactive-neighbour reads always see the boundary
// value).
template <typename ScalarT>
void
runReinit(OnIndexGridT *grid,
          const VBMHelper &vbm,
          ScalarT *phi,
          ScalarT *sign,
          ScalarT *phiBase,
          ScalarT *stage,
          ScalarT *rhs0,
          ScalarT *rhs1,
          int64_t valueCount,
          ScalarT voxelSize,
          ScalarT bandWidth,
          int band,
          int smooth,
          int order,
          bool taubin,
          int redistanceIters,
          cudaStream_t stream) {
    const uint32_t blockCount   = vbm.blockCount;
    constexpr int blockWidth    = 1 << kLog2BlockWidth;
    const uint32_t *firstLeafID = vbm.firstLeafID();
    const uint64_t *jumpMap     = vbm.jumpMap();
    const uint64_t firstOffset  = vbm.firstOffset;
    const ScalarT timeStep      = ScalarT(0.4) * voxelSize;

    auto godunov = [&](const ScalarT *field, ScalarT *out) {
        if (blockCount) {
            godunovFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(
                grid, firstLeafID, jumpMap, firstOffset, field, sign, voxelSize, out);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    };
    auto redistance = [&](int iters) {
        if (blockCount) {
            signFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(
                grid, firstLeafID, jumpMap, firstOffset, phi, voxelSize, sign);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        for (int it = 0; it < iters; ++it) {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                phiBase, phi, valueCount * sizeof(ScalarT), cudaMemcpyDeviceToDevice, stream));
            godunov(phi, rhs0); // rhs0 = rhs(phi)

            // Every scheme begins with the same forward-Euler step. For RK1 this is the final
            // update (written to phi); for RK2/RK3 it is the first stage (written to `stage`).
            ScalarT *firstStage = (order <= 1) ? phi : stage;
            combineKernel<ScalarT>
                <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                    firstStage,
                    phiBase,
                    nullptr,
                    rhs0,
                    ScalarT(1),
                    ScalarT(0),
                    ScalarT(1),
                    timeStep,
                    bandWidth,
                    valueCount);

            if (order == 2) { // Heun (TVD-RK2)
                godunov(stage, rhs1);
                heunKernel<ScalarT>
                    <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                        phi, phiBase, rhs0, rhs1, timeStep, bandWidth, valueCount);
            } else if (order == 3) { // Shu-Osher TVD-RK3
                godunov(stage, rhs0);
                combineKernel<ScalarT>
                    <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                        stage,
                        phiBase,
                        stage,
                        rhs0,
                        ScalarT(0.75),
                        ScalarT(0.25),
                        ScalarT(0.25),
                        timeStep,
                        bandWidth,
                        valueCount);
                godunov(stage, rhs0);
                combineKernel<ScalarT>
                    <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                        phi,
                        phiBase,
                        stage,
                        rhs0,
                        ScalarT(1.0 / 3.0),
                        ScalarT(2.0 / 3.0),
                        ScalarT(2.0 / 3.0),
                        timeStep,
                        bandWidth,
                        valueCount);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    };

    const int defaultIters = std::max(6, (int)std::lround(2.5 * band) + 2);
    redistance(redistanceIters > 0 ? redistanceIters : defaultIters);

    if (smooth) {
        ScalarT *cur = phi, *other = stage; // ping-pong (stage[0] already = bandWidth)
        auto pass = [&](ScalarT weight) {
            if (blockCount) {
                smoothFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(
                    grid, firstLeafID, jumpMap, firstOffset, cur, weight, other);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            std::swap(cur, other);
        };
        if (taubin) {
            for (int i = 0; i < smooth; ++i) {
                pass(ScalarT(0.5));
                pass(ScalarT(-0.53));
            } // volume-preserving
        } else {
            for (int i = 0; i < smooth; ++i)
                pass(ScalarT(1.0)); // mean-curvature
        }
        if (cur != phi)
            C10_CUDA_CHECK(cudaMemcpyAsync(
                phi, cur, valueCount * sizeof(ScalarT), cudaMemcpyDeviceToDevice, stream));
        redistance(std::max(4, smooth));
    }
}

template <typename ScalarT>
void
reinitializeSdfCuda(const GridBatchData &batchHdl,
                    const torch::Tensor &field,
                    torch::Tensor &out,
                    int band,
                    int redistanceIters,
                    int order,
                    int smooth,
                    bool taubin) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index()).stream();
    auto opts           = torch::TensorOptions().dtype(field.dtype()).device(field.device());

    const ScalarT *fieldPtr = field.data_ptr<ScalarT>();
    ScalarT *outPtr         = out.data_ptr<ScalarT>();

    for (int64_t batchIdx = 0; batchIdx < batchHdl.batchSize(); ++batchIdx) {
        const int64_t numVoxels = batchHdl.numVoxelsAt(batchIdx);
        if (numVoxels == 0)
            continue;
        OnIndexGridT *grid =
            batchHdl.mGridHdl->deviceGrid<nanovdb::ValueOnIndex>((uint32_t)batchIdx);
        const int64_t voxelOffset = batchHdl.cumVoxelsAt(batchIdx);
        const ScalarT voxelSize   = (ScalarT)batchHdl.voxelSizeAt(batchIdx)[0];
        const ScalarT bandWidth = (ScalarT)band * voxelSize; // narrow-band half-width, world units

        VBMHelper vbm(grid, stream);
        const int64_t valueCount = (int64_t)vbm.valueCount;  // numVoxels + 1 (slot 0 = background)

        torch::Tensor phiBuf     = torch::empty({valueCount}, opts);
        torch::Tensor signBuf    = torch::empty({valueCount}, opts);
        torch::Tensor phiBaseBuf = torch::empty({valueCount}, opts);
        torch::Tensor stageBuf   = torch::empty({valueCount}, opts);
        torch::Tensor rhs0Buf    = torch::empty({valueCount}, opts);
        torch::Tensor rhs1Buf = (order == 2) ? torch::empty({valueCount}, opts) : torch::Tensor();
        ScalarT *phi          = phiBuf.data_ptr<ScalarT>();
        ScalarT *sign         = signBuf.data_ptr<ScalarT>();
        ScalarT *phiBase      = phiBaseBuf.data_ptr<ScalarT>();
        ScalarT *stage        = stageBuf.data_ptr<ScalarT>();
        ScalarT *rhs0         = rhs0Buf.data_ptr<ScalarT>();
        ScalarT *rhs1         = (order == 2) ? rhs1Buf.data_ptr<ScalarT>() : nullptr;

        // gather: phi[0] = bandWidth (outside BC), phi[1..] = field; stage[0] = bandWidth (RK slot
        // 0)
        fillKernel<ScalarT>
            <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                phi, valueCount, bandWidth);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fillKernel<ScalarT>
            <<<GET_BLOCKS(valueCount, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM, 0, stream>>>(
                stage, valueCount, bandWidth);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(phi + 1,
                                       fieldPtr + voxelOffset,
                                       numVoxels * sizeof(ScalarT),
                                       cudaMemcpyDeviceToDevice,
                                       stream));

        runReinit<ScalarT>(grid,
                           vbm,
                           phi,
                           sign,
                           phiBase,
                           stage,
                           rhs0,
                           rhs1,
                           valueCount,
                           voxelSize,
                           bandWidth,
                           band,
                           smooth,
                           order,
                           taubin,
                           redistanceIters,
                           stream);

        // scatter: out[voxelOffset..] = phi[1..]
        C10_CUDA_CHECK(cudaMemcpyAsync(outPtr + voxelOffset,
                                       phi + 1,
                                       numVoxels * sizeof(ScalarT),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

} // namespace

JaggedTensor
reinitializeSdf(const GridBatchData &batchHdl,
                const JaggedTensor &field,
                int band,
                int redistanceIters,
                int order,
                int smooth,
                SmoothingMode smoothing) {
    const bool taubin = (smoothing == SmoothingMode::TAUBIN);
    TORCH_CHECK_VALUE(
        field.ldim() == 1,
        "Expected field to have 1 list dimension (a single list of per-voxel values)");
    TORCH_CHECK_TYPE(field.is_floating_point(), "field must have a floating point type");
    TORCH_CHECK_VALUE(field.numel() == batchHdl.totalVoxels(),
                      "field value count does not match the number of voxels in the grid");
    TORCH_CHECK_VALUE(field.num_outer_lists() == batchHdl.batchSize(),
                      "field batch size does not match the grid batch size");
    TORCH_CHECK_VALUE(band >= 1, "band must be >= 1");
    TORCH_CHECK_VALUE(order >= 1 && order <= 3, "order must be 1, 2, or 3");
    TORCH_CHECK_VALUE(smooth >= 0, "smooth must be >= 0");
    batchHdl.checkDevice(field);
    TORCH_CHECK(field.device().is_cuda(),
                "reinitialize_sdf currently requires a CUDA device (VoxelBlockManager solver)");
    TORCH_CHECK_TYPE(field.scalar_type() == torch::kFloat32 ||
                         field.scalar_type() == torch::kFloat64,
                     "reinitialize_sdf supports float32 or float64 fields");

    torch::Tensor fieldJdata = field.jdata().contiguous();
    if (fieldJdata.dim() != 1)
        fieldJdata = fieldJdata.view({-1});

    torch::Tensor out = torch::empty_like(fieldJdata);
    AT_DISPATCH_FLOATING_TYPES(fieldJdata.scalar_type(), "reinitializeSdf", [&] {
        reinitializeSdfCuda<scalar_t>(
            batchHdl, fieldJdata, out, band, redistanceIters, order, smooth, taubin);
    });
    return field.jagged_like(out);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
