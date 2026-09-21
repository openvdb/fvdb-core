// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/detail/ops/ReinitializeSdf.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/math/Math.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

using OnIndexGridT = nanovdb::NanoGrid<nanovdb::ValueOnIndex>;
// VoxelBlockManager metadata (firstLeafID / jumpMap) lives in a single-space device buffer
// over the builders' resource (torch's active CUDA allocator). buildVoxelBlockManager
// allocates it stream-ordered on the reinit stream via createDeviceStorage, and the handle's
// single-space accessors (openvdb #2301) hand the pointers back without an adapter.
using VbmBuffer = BuilderBuffer<std::byte>;

// log2 of the VoxelBlockManager block width: each VBM block spans 2^9 = 512 active voxels.
static constexpr int kLog2BlockWidth = 9;

// ------------------------- VBM fused 6-face stencil preamble -------------------------
// The VBM decode gives the centre coord / value-index for free; we then read just the 6 FACE
// neighbours through a cached ReadAccessor. Yields `centerIndex` (the centre voxel's value index)
// and `faceIndex[6]` (the 6 face-neighbour value indices, -x,+x,-y,+y,-z,+z; 0 =
// inactive/background). Kernels using it take the grid/firstLeafID/jumpMap/firstOffset parameters
// by these exact names. Neighbour VALUES must be read through `faceValue` (below), never
// `field[faceIndex[k]]` directly, so that inactive neighbours get a sign-aware boundary value.
#define VBM_FACES_BEGIN()                                                                        \
    constexpr int blockWidth = 1 << kLog2BlockWidth, jumpMapWordCount = blockWidth / 64;         \
    using VoxelBlockManagerT = nanovdb::tools::cuda::VoxelBlockManager<kLog2BlockWidth>;         \
    __shared__ uint32_t sharedLeafIndex[blockWidth];                                             \
    __shared__ uint16_t sharedVoxelOffset[blockWidth];                                           \
    VoxelBlockManagerT::template decodeInverseMaps<nanovdb::ValueOnIndex>(                       \
        grid,                                                                                    \
        firstLeafID[blockIdx.x],                                                                 \
        &jumpMap[uint64_t(blockIdx.x) * jumpMapWordCount],                                       \
        firstOffset + uint64_t(blockIdx.x) * blockWidth,                                         \
        sharedLeafIndex,                                                                         \
        sharedVoxelOffset);                                                                      \
    if (sharedLeafIndex[threadIdx.x] == VoxelBlockManagerT::UnusedLeafIndex)                     \
        return;                                                                                  \
    const auto &leaf = grid->tree().template getFirstNode<0>()[sharedLeafIndex[threadIdx.x]];    \
    const nanovdb::Coord centerCoord = leaf.offsetToGlobalCoord(sharedVoxelOffset[threadIdx.x]); \
    const uint64_t centerIndex       = leaf.getValue(sharedVoxelOffset[threadIdx.x]);            \
    auto accessor                    = grid->getAccessor();                                      \
    const uint64_t faceIndex[6]      = {accessor.getValue(centerCoord.offsetBy(-1, 0, 0)),       \
                                        accessor.getValue(centerCoord.offsetBy(1, 0, 0)),        \
                                        accessor.getValue(centerCoord.offsetBy(0, -1, 0)),       \
                                        accessor.getValue(centerCoord.offsetBy(0, 1, 0)),        \
                                        accessor.getValue(centerCoord.offsetBy(0, 0, -1)),       \
                                        accessor.getValue(centerCoord.offsetBy(0, 0, 1))};

// Sign-aware read of a face neighbour. An IndexGrid has a single background slot (value index 0),
// so it cannot carry OpenVDB's two-signed inactive tiles (-background inside, +background outside).
// Instead, an inactive neighbour continues the supplied sign: beside a negative voxel it
// is deep interior (-bandWidth), beside a positive voxel deep exterior (+bandWidth). Without this
// the inner edge of a narrow band sees +bandWidth across the face and a phantom interface forms
// one voxel inside the true surface. Godunov updates supply the frozen sign; sign computation and
// smoothing supply the current centre value. Active neighbours use the value-indexed buffer.
template <typename ScalarT>
__device__ inline ScalarT
faceValue(const ScalarT *field, uint64_t index, ScalarT signSource, ScalarT bandWidth) {
    return index ? field[index] : (signSource < ScalarT(0) ? -bandWidth : bandWidth);
}

// Reads the 6 face neighbours, using `signSource` for inactive values.
#define VBM_FACE_VALUES(field, signSource, bandWidth)                                  \
    const ScalarT xm = faceValue<ScalarT>(field, faceIndex[0], signSource, bandWidth), \
                  xp = faceValue<ScalarT>(field, faceIndex[1], signSource, bandWidth), \
                  ym = faceValue<ScalarT>(field, faceIndex[2], signSource, bandWidth), \
                  yp = faceValue<ScalarT>(field, faceIndex[3], signSource, bandWidth), \
                  zm = faceValue<ScalarT>(field, faceIndex[4], signSource, bandWidth), \
                  zp = faceValue<ScalarT>(field, faceIndex[5], signSource, bandWidth)

// =====================  fused stencil kernels ====================================================
// Frozen Peng smoothed sign from a field's value + central-difference gradient, plus the
// Russo-Smereka interface data: a voxel whose value changes sign across an active face is an
// interface cell, and `interfaceDist` holds its signed distance to the zero crossing estimated
// from the largest one-sided slope of the initial field across its active faces. Godunov updates at
// interface cells are replaced by a relaxation toward that distance, which anchors the zero
// crossing; without it the pure Godunov scheme has no steady state on features 1-3 voxels thick and
// erodes them a little more with every iteration. An exactly-0 centre yields sign 0 and distance 0,
// so the RHS vanishes there and such (no-data) voxels are never moved by the redistance; the
// ray-implicit-intersection op relies on exact 0 surviving as a gap marker.
template <typename ScalarT>
__global__ void
signFusedKernel(const OnIndexGridT *grid,
                const uint32_t *firstLeafID,
                const uint64_t *jumpMap,
                uint64_t firstOffset,
                const ScalarT *field,
                ScalarT voxelSize,
                ScalarT bandWidth,
                ScalarT *sign,
                ScalarT *interfaceDist,
                uint8_t *interfaceMask) {
    VBM_FACES_BEGIN();
    const ScalarT phiCenter = field[centerIndex];
    VBM_FACE_VALUES(field, phiCenter, bandWidth);
    ScalarT gradX = (xp - xm) / (2 * voxelSize);
    ScalarT gradY = (yp - ym) / (2 * voxelSize);
    ScalarT gradZ = (zp - zm) / (2 * voxelSize);
    sign[centerIndex] =
        phiCenter / nanovdb::math::Sqrt(phiCenter * phiCenter +
                                        (gradX * gradX + gradY * gradY + gradZ * gradZ) *
                                            voxelSize * voxelSize +
                                        ScalarT(1e-12));

    // Inactive faces read with the centre's sign, so they never form an interface.
    const bool centerNeg   = phiCenter < ScalarT(0);
    const bool isInterface = ((xm < ScalarT(0)) != centerNeg) || ((xp < ScalarT(0)) != centerNeg) ||
                             ((ym < ScalarT(0)) != centerNeg) || ((yp < ScalarT(0)) != centerNeg) ||
                             ((zm < ScalarT(0)) != centerNeg) || ((zp < ScalarT(0)) != centerNeg);
    interfaceMask[centerIndex] = isInterface ? 1 : 0;
    if (isInterface) {
        // Largest one-sided slope over ACTIVE faces only. An inactive face reads as +/-bandWidth
        // and would inflate the slope, pulling a band-edge interface cell toward zero. The central
        // difference is bounded by the larger one-sided difference, so it adds nothing.
        using nanovdb::math::Abs;
        using nanovdb::math::Max;
        const ScalarT faceVals[6] = {xm, xp, ym, yp, zm, zp};
        ScalarT slope             = ScalarT(1e-6) * voxelSize;
        for (int k = 0; k < 6; ++k)
            if (faceIndex[k])
                slope = Max(slope, Abs(faceVals[k] - phiCenter));
        interfaceDist[centerIndex] = voxelSize * phiCenter / slope;
    } else {
        interfaceDist[centerIndex] = ScalarT(0);
    }
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

// Godunov RHS: d phi/dt = sign * (1 - |grad phi|), one-sided upwind on the 6 faces. Interface
// cells instead use the Russo-Smereka subcell update d phi/dt = -(sgn(phi0) |phi| - D) / dx, which
// converges to |phi| = D and keeps the zero crossing where the initial field put it.
template <typename ScalarT>
__global__ void
godunovFusedKernel(const OnIndexGridT *grid,
                   const uint32_t *firstLeafID,
                   const uint64_t *jumpMap,
                   uint64_t firstOffset,
                   const ScalarT *field,
                   const ScalarT *sign,
                   const ScalarT *interfaceDist,
                   const uint8_t *interfaceMask,
                   ScalarT voxelSize,
                   ScalarT bandWidth,
                   ScalarT *rhs) {
    VBM_FACES_BEGIN();
    const ScalarT center = field[centerIndex], sgn = sign[centerIndex];
    if (interfaceMask[centerIndex]) {
        const ScalarT sgn0 =
            sgn > ScalarT(0) ? ScalarT(1) : (sgn < ScalarT(0) ? ScalarT(-1) : ScalarT(0));
        rhs[centerIndex] =
            -(sgn0 * nanovdb::math::Abs(center) - interfaceDist[centerIndex]) / voxelSize;
        return;
    }
    // Match upwind's frozen sign even if an RK stage crosses zero. For clamped stage values,
    // inactive faces then remain downwind instead of introducing a spurious boundary slope.
    VBM_FACE_VALUES(field, sgn, bandWidth);
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
                  ScalarT bandWidth,
                  ScalarT *out) {
    VBM_FACES_BEGIN();
    const ScalarT center = in[centerIndex];
    VBM_FACE_VALUES(in, center, bandWidth);
    ScalarT faceMean = (xm + xp + ym + yp + zm + zp) * (ScalarT(1) / ScalarT(6));
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

// small VBM helper: build once, expose the block count + the firstLeafID/jumpMap device pointers.
struct VBMHelper {
    nanovdb::tools::VoxelBlockManagerHandle<VbmBuffer> handle;
    uint32_t blockCount{0};
    uint64_t firstOffset{0}, valueCount{1};
    VBMHelper(OnIndexGridT *grid, cudaStream_t stream) {
        handle = nanovdb::tools::cuda::buildVoxelBlockManager<kLog2BlockWidth, VbmBuffer>(
            grid, 0, 0, 0, stream);
        blockCount  = (uint32_t)handle.blockCount();
        firstOffset = handle.firstOffset();
        valueCount  = handle.lastOffset() + 1;
    }
    const uint32_t *
    firstLeafID() const {
        return handle.deviceFirstLeafID();
    }
    const uint64_t *
    jumpMap() const {
        return handle.deviceJumpMap();
    }
};

// Redistance (|grad phi| = 1) + optional de-staircase one grid's value-indexed buffer, in place.
// `phi`/scratch are length `valueCount`; slot 0 is the (never written) background slot. Stencil
// kernels do not read slot 0: inactive neighbours are resolved by `faceValue` to
// +/-bandWidth using the frozen sign for Godunov updates and the current centre sign otherwise,
// so a narrow band with an inactive interior is treated as continuing inward, not as exterior.
template <typename ScalarT>
void
runReinit(OnIndexGridT *grid,
          const VBMHelper &vbm,
          ScalarT *phi,
          ScalarT *sign,
          ScalarT *interfaceDist,
          uint8_t *interfaceMask,
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
            godunovFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(grid,
                                                                               firstLeafID,
                                                                               jumpMap,
                                                                               firstOffset,
                                                                               field,
                                                                               sign,
                                                                               interfaceDist,
                                                                               interfaceMask,
                                                                               voxelSize,
                                                                               bandWidth,
                                                                               out);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    };
    auto redistance = [&](int iters) {
        if (blockCount) {
            signFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(grid,
                                                                            firstLeafID,
                                                                            jumpMap,
                                                                            firstOffset,
                                                                            phi,
                                                                            voxelSize,
                                                                            bandWidth,
                                                                            sign,
                                                                            interfaceDist,
                                                                            interfaceMask);
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
        ScalarT *cur = phi, *other = stage; // ping-pong; stencil reads do not use slot 0
        auto pass = [&](ScalarT weight) {
            if (blockCount) {
                smoothFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(
                    grid, firstLeafID, jumpMap, firstOffset, cur, weight, bandWidth, other);
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
        OnIndexGridT *grid        = batchHdl.deviceGridPtrAt(batchIdx);
        const int64_t voxelOffset = batchHdl.cumVoxelsAt(batchIdx);
        const nanovdb::Vec3d &vs  = batchHdl.voxelSizeAt(batchIdx);
        // The eikonal solve uses one voxel size for all three axes; an anisotropic grid would get
        // distances silently scaled by the aspect ratio along y/z. Relative tolerance absorbs
        // float32 -> double round-off in voxel sizes that arrive from Python.
        TORCH_CHECK_VALUE(std::abs(vs[1] - vs[0]) <= 1e-6 * vs[0] &&
                              std::abs(vs[2] - vs[0]) <= 1e-6 * vs[0],
                          "reinitialize_sdf requires isotropic voxels (the eikonal solve uses a "
                          "single voxel size), but grid ",
                          batchIdx,
                          " has voxel_size (",
                          vs[0],
                          ", ",
                          vs[1],
                          ", ",
                          vs[2],
                          ")");
        const ScalarT voxelSize = (ScalarT)vs[0];
        const ScalarT bandWidth = (ScalarT)band * voxelSize; // narrow-band half-width, world units

        VBMHelper vbm(grid, stream);
        const int64_t valueCount = (int64_t)vbm.valueCount;  // numVoxels + 1 (slot 0 = background)

        torch::Tensor phiBuf     = torch::empty({valueCount}, opts);
        torch::Tensor signBuf    = torch::empty({valueCount}, opts);
        torch::Tensor distBuf    = torch::empty({valueCount}, opts);
        torch::Tensor maskBuf    = torch::zeros({valueCount}, opts.dtype(torch::kUInt8));
        torch::Tensor phiBaseBuf = torch::empty({valueCount}, opts);
        torch::Tensor stageBuf   = torch::empty({valueCount}, opts);
        torch::Tensor rhs0Buf    = torch::empty({valueCount}, opts);
        torch::Tensor rhs1Buf  = (order == 2) ? torch::empty({valueCount}, opts) : torch::Tensor();
        ScalarT *phi           = phiBuf.data_ptr<ScalarT>();
        ScalarT *sign          = signBuf.data_ptr<ScalarT>();
        ScalarT *interfaceDist = distBuf.data_ptr<ScalarT>();
        uint8_t *interfaceMask = maskBuf.data_ptr<uint8_t>();
        ScalarT *phiBase       = phiBaseBuf.data_ptr<ScalarT>();
        ScalarT *stage         = stageBuf.data_ptr<ScalarT>();
        ScalarT *rhs0          = rhs0Buf.data_ptr<ScalarT>();
        ScalarT *rhs1          = (order == 2) ? rhs1Buf.data_ptr<ScalarT>() : nullptr;

        // gather: phi[1..] = field. Slot 0 of phi/stage is the background slot: filled with
        // bandWidth for hygiene but never read by the stencil kernels (see faceValue) nor written.
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
                           interfaceDist,
                           interfaceMask,
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
    const auto &data = field.jdata();
    TORCH_CHECK_VALUE(data.dim() == 1 || (data.dim() == 2 && data.size(1) == 1),
                      "field must be a scalar field with shape (N,) or (N, 1), got ",
                      data.sizes());
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
    TORCH_CHECK_VALUE(torch::isfinite(fieldJdata).all().item<bool>(),
                      "field must contain only finite values; leave no-data voxels inactive");
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
