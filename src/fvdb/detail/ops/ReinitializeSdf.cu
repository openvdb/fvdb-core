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
#include <utility>

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
// `field[faceIndex[k] - 1]` directly, so that inactive neighbours get a sign-aware boundary value.
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

// Sign-aware read of a face neighbour. Value buffers are indexed by `valueIndex - 1`: an IndexGrid
// numbers active voxels from 1 and reserves 0 for its single background slot, so a face index of 0
// means inactive. An inactive neighbour continues the supplied sign: beside a negative voxel it is
// deep interior (-bandWidth), beside a positive voxel deep exterior (+bandWidth). Without this the
// inner edge of a narrow band sees +bandWidth across the face and a phantom interface forms one
// voxel inside the true surface. Godunov updates supply the frozen sign; the frozen-data and
// smoothing reads supply the centre value of the field being read.
template <typename ScalarT>
__device__ inline ScalarT
faceValue(const ScalarT *field, uint64_t index, ScalarT signSource, ScalarT bandWidth) {
    return index ? field[index - 1] : (signSource < ScalarT(0) ? -bandWidth : bandWidth);
}

// Reads the 6 face neighbours of `field` into `name`[6] (-x,+x,-y,+y,-z,+z), using `signSource`
// for inactive values.
#define VBM_FACE_VALUES(name, field, signSource, bandWidth)                                  \
    const ScalarT name[6] = {faceValue<ScalarT>(field, faceIndex[0], signSource, bandWidth), \
                             faceValue<ScalarT>(field, faceIndex[1], signSource, bandWidth), \
                             faceValue<ScalarT>(field, faceIndex[2], signSource, bandWidth), \
                             faceValue<ScalarT>(field, faceIndex[3], signSource, bandWidth), \
                             faceValue<ScalarT>(field, faceIndex[4], signSource, bandWidth), \
                             faceValue<ScalarT>(field, faceIndex[5], signSource, bandWidth)}

// =====================  frozen interface data
// ===================================================== Everything the redistance freezes from the
// initial field phi0, recomputed from phi0's 6-face stencil on every RHS evaluation instead of
// being stored (phi0 stays resident for the whole solve, so this costs neighbour reads, not
// memory):
//   * the Peng smoothed sign  phi0 / sqrt(phi0^2 + |grad phi0|^2 dx^2);
//   * whether the voxel is an interface cell (sign change across an active face);
//   * the Russo-Smereka signed distance D to the zero crossing: phi0 * dx over the larger of the
//     central-difference gradient norm and the largest one-sided slope, active faces only (see the
//     note in frozenData).
// An exactly-0 centre yields sign 0 and D 0, so the RHS vanishes there and such (no-data) voxels
// are never moved by the redistance; the ray-implicit-intersection op relies on exact 0 surviving
// as a gap marker.
template <typename ScalarT> struct FrozenData {
    ScalarT sign;
    ScalarT dist;
    bool isInterface;
};

template <typename ScalarT>
__device__ inline FrozenData<ScalarT>
frozenData(const ScalarT *phi0,
           int64_t vi,
           const uint64_t *faceIndex,
           ScalarT voxelSize,
           ScalarT bandWidth) {
    using nanovdb::math::Abs;
    using nanovdb::math::Max;
    const ScalarT c = phi0[vi];
    VBM_FACE_VALUES(f, phi0, c, bandWidth);
    const ScalarT gradX = (f[1] - f[0]) / (2 * voxelSize);
    const ScalarT gradY = (f[3] - f[2]) / (2 * voxelSize);
    const ScalarT gradZ = (f[5] - f[4]) / (2 * voxelSize);

    FrozenData<ScalarT> d;
    // Both terms under the root scale as dx^2, so the 0/0 guard does too.
    const ScalarT dx2 = voxelSize * voxelSize;
    d.sign = c / nanovdb::math::Sqrt(c * c + (gradX * gradX + gradY * gradY + gradZ * gradZ) * dx2 +
                                     ScalarT(1e-10) * dx2);

    // Denominator for D, following Russo-Smereka's 1D max(central, one-sided, eps) in 3D: the
    // Euclidean norm of the central-difference gradient captures oblique surfaces (a single face
    // difference only sees one gradient component and would overestimate D by up to sqrt(3)); the
    // largest one-sided slope takes over at kinks and thin features, where the central difference
    // collapses toward zero. Only active faces contribute. An inactive face reads as +/-bandWidth
    // and would inflate the slope, pulling a band-edge interface cell toward zero. Restricting the
    // slopes to crossing faces was tried and rejected: it picks oblique, shallow faces and shifted
    // zero crossings by up to 0.9 voxels on a real SDF.
    const bool centerNeg = c < ScalarT(0);
    d.isInterface        = false;
    ScalarT slope        = ScalarT(1e-6) * voxelSize;
    ScalarT gradSq       = ScalarT(0);
    for (int axis = 0; axis < 3; ++axis) {
        const int km = 2 * axis, kp = 2 * axis + 1;
        d.isInterface |= ((f[km] < ScalarT(0)) != centerNeg) || ((f[kp] < ScalarT(0)) != centerNeg);
        const bool am = faceIndex[km] != 0, ap = faceIndex[kp] != 0;
        const ScalarT dm = am ? Abs(c - f[km]) : ScalarT(0), dp = ap ? Abs(f[kp] - c) : ScalarT(0);
        slope           = Max(slope, Max(dm, dp));
        const ScalarT g = (am && ap) ? Abs(f[kp] - f[km]) * ScalarT(0.5) : Max(dm, dp);
        gradSq += g * g;
    }
    slope  = Max(slope, nanovdb::math::Sqrt(gradSq));
    d.dist = d.isInterface ? voxelSize * c / slope : ScalarT(0);
    return d;
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

// =====================  fused stencil kernels ====================================================
// One TVD-RK stage, fused with its RHS evaluation:
//     out = clip(baseCoeff*base + stageCoeff*cur + rhsCoeff*timeStep*rhs(cur), -bandWidth,
//     bandWidth)
// rhs(cur) is the Godunov RHS  sign * (1 - |grad cur|)  with one-sided upwinding on the 6 faces,
// or, at interface cells, the Russo-Smereka subcell update  -(sgn(phi0) |cur| - D) / dx  that
// converges to |cur| = D and keeps the zero crossing where phi0 put it.
// `out` must not alias `cur` (neighbours of `cur` are read) but may alias `base` (read at the
// centre only). Sets *nonFinite if phi0 holds a NaN/Inf at this voxel.
template <typename ScalarT>
__global__ void
rkStageFusedKernel(const OnIndexGridT *grid,
                   const uint32_t *firstLeafID,
                   const uint64_t *jumpMap,
                   uint64_t firstOffset,
                   const ScalarT *phi0,
                   const ScalarT *cur,
                   const ScalarT *base,
                   ScalarT baseCoeff,
                   ScalarT stageCoeff,
                   ScalarT rhsCoeff,
                   ScalarT timeStep,
                   ScalarT voxelSize,
                   ScalarT bandWidth,
                   ScalarT *out,
                   int *nonFinite) {
    VBM_FACES_BEGIN();
    const int64_t vi = int64_t(centerIndex) - 1;
    if (!isfinite(phi0[vi]))
        *nonFinite = 1;
    const FrozenData<ScalarT> fz = frozenData<ScalarT>(phi0, vi, faceIndex, voxelSize, bandWidth);
    const ScalarT sgn            = fz.sign;
    const ScalarT center         = cur[vi];

    ScalarT rhs;
    if (fz.isInterface) {
        const ScalarT sgn0 =
            sgn > ScalarT(0) ? ScalarT(1) : (sgn < ScalarT(0) ? ScalarT(-1) : ScalarT(0));
        rhs = -(sgn0 * nanovdb::math::Abs(center) - fz.dist) / voxelSize;
    } else {
        // Match upwind's frozen sign even if an RK stage crosses zero. For clamped stage values,
        // inactive faces then remain downwind instead of introducing a spurious boundary slope.
        VBM_FACE_VALUES(f, cur, sgn, bandWidth);
        const ScalarT gradMag = nanovdb::math::Sqrt(
            upwind<ScalarT>((center - f[0]) / voxelSize, (f[1] - center) / voxelSize, sgn) +
            upwind<ScalarT>((center - f[2]) / voxelSize, (f[3] - center) / voxelSize, sgn) +
            upwind<ScalarT>((center - f[4]) / voxelSize, (f[5] - center) / voxelSize, sgn));
        rhs = sgn * (ScalarT(1) - gradMag);
    }

    ScalarT value = baseCoeff * base[vi] +
                    (stageCoeff != ScalarT(0) ? stageCoeff * center : ScalarT(0)) +
                    rhsCoeff * timeStep * rhs;
    out[vi] = nanovdb::math::Min(nanovdb::math::Max(value, -bandWidth), bandWidth);
}

// one umbrella-Laplacian smoothing pass: out = in + weight*(faceMean - in). Double-buffered
// (in != out) so neighbour reads see the pre-pass field.
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
    const int64_t vi     = int64_t(centerIndex) - 1;
    const ScalarT center = in[vi];
    VBM_FACE_VALUES(f, in, center, bandWidth);
    ScalarT faceMean = (f[0] + f[1] + f[2] + f[3] + f[4] + f[5]) * (ScalarT(1) / ScalarT(6));
    out[vi]          = center + weight * (faceMean - center);
}

// small VBM helper: build once, expose the block count + the firstLeafID/jumpMap device pointers.
struct VBMHelper {
    nanovdb::tools::VoxelBlockManagerHandle<VbmBuffer> handle;
    uint32_t blockCount{0};
    uint64_t firstOffset{0};
    VBMHelper(OnIndexGridT *grid, cudaStream_t stream) {
        handle = nanovdb::tools::cuda::buildVoxelBlockManager<kLog2BlockWidth, VbmBuffer>(
            grid, 0, 0, 0, stream);
        blockCount  = (uint32_t)handle.blockCount();
        firstOffset = handle.firstOffset();
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

// Redistance (|grad phi| = 1) + optional de-staircase one grid's field, in place in `phi`.
//
// Memory: `phi` is the caller's output slice and `field` (the untouched input slice) doubles as
// the frozen phi0, so the solve needs only the scratch below, each `numVoxels` long:
//   * scratchA          always (RK stage / ping-pong)
//   * scratchB          order 3 only (Shu-Osher needs phi^n and two stages live at once)
//   * smoothedPhi0      smooth > 0 only (snapshot of the smoothed field for the re-redistance)
// Inactive neighbours are resolved by `faceValue` to +/-bandWidth using the frozen sign for Godunov
// updates and the current centre sign otherwise, so a narrow band with an inactive interior is
// treated as continuing inward, not as exterior.
template <typename ScalarT>
void
runReinit(OnIndexGridT *grid,
          const VBMHelper &vbm,
          const ScalarT *field,
          ScalarT *phi,
          ScalarT *scratchA,
          ScalarT *scratchB,
          ScalarT *smoothedPhi0,
          int *nonFinite,
          int64_t numVoxels,
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
    const size_t bytes          = size_t(numVoxels) * sizeof(ScalarT);

    auto stage = [&](const ScalarT *phi0,
                     const ScalarT *cur,
                     const ScalarT *base,
                     ScalarT baseCoeff,
                     ScalarT stageCoeff,
                     ScalarT rhsCoeff,
                     ScalarT *out) {
        if (blockCount) {
            rkStageFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(grid,
                                                                               firstLeafID,
                                                                               jumpMap,
                                                                               firstOffset,
                                                                               phi0,
                                                                               cur,
                                                                               base,
                                                                               baseCoeff,
                                                                               stageCoeff,
                                                                               rhsCoeff,
                                                                               timeStep,
                                                                               voxelSize,
                                                                               bandWidth,
                                                                               out,
                                                                               nonFinite);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    };

    auto redistance = [&](const ScalarT *phi0, int iters) {
        const ScalarT one = 1, zero = 0;
        if (order <= 1) { // forward Euler, ping-pong between phi and scratchA
            ScalarT *src = phi, *dst = scratchA;
            for (int it = 0; it < iters; ++it) {
                stage(phi0, src, src, one, zero, one, dst);
                std::swap(src, dst);
            }
            if (src != phi)
                C10_CUDA_CHECK(cudaMemcpyAsync(phi, src, bytes, cudaMemcpyDeviceToDevice, stream));
        } else if (order == 2) { // Heun (TVD-RK2) in SSP form: two buffers
            for (int it = 0; it < iters; ++it) {
                stage(phi0, phi, phi, one, zero, one, scratchA);
                stage(phi0, scratchA, phi, ScalarT(0.5), ScalarT(0.5), ScalarT(0.5), phi);
            }
        } else { // Shu-Osher TVD-RK3: three buffers
            for (int it = 0; it < iters; ++it) {
                stage(phi0, phi, phi, one, zero, one, scratchA);
                stage(phi0, scratchA, phi, ScalarT(0.75), ScalarT(0.25), ScalarT(0.25), scratchB);
                stage(phi0,
                      scratchB,
                      phi,
                      ScalarT(1.0 / 3.0),
                      ScalarT(2.0 / 3.0),
                      ScalarT(2.0 / 3.0),
                      phi);
            }
        }
    };

    // Information travels at most 0.4 dx per sweep and the Peng sign roughly halves that near the
    // interface, so a full band needs about 5*band sweeps; 6*band leaves a convergence margin. The
    // subcell anchor makes extra sweeps harmless, so this errs long (a step input on a band-3
    // sphere converges by ~20 sweeps).
    const int defaultIters = std::max(20, 6 * band);
    const int iters        = redistanceIters > 0 ? redistanceIters : defaultIters;
    redistance(field, iters);

    if (smooth) {
        ScalarT *cur = phi, *other = scratchA; // ping-pong
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
            C10_CUDA_CHECK(cudaMemcpyAsync(phi, cur, bytes, cudaMemcpyDeviceToDevice, stream));
        // The smoothed surface is the new anchor: snapshot it as phi0 for the re-redistance.
        C10_CUDA_CHECK(cudaMemcpyAsync(smoothedPhi0, phi, bytes, cudaMemcpyDeviceToDevice, stream));
        redistance(smoothedPhi0, iters);
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
    torch::Tensor nonFinite = torch::zeros({1}, opts.dtype(torch::kInt32));

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

        const ScalarT *phi0 = fieldPtr + voxelOffset;
        ScalarT *phi        = outPtr + voxelOffset;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            phi, phi0, numVoxels * sizeof(ScalarT), cudaMemcpyDeviceToDevice, stream));

        torch::Tensor scratchA = torch::empty({numVoxels}, opts);
        torch::Tensor scratchB = (order >= 3) ? torch::empty({numVoxels}, opts) : torch::Tensor();
        torch::Tensor smoothed = (smooth > 0) ? torch::empty({numVoxels}, opts) : torch::Tensor();

        runReinit<ScalarT>(grid,
                           vbm,
                           phi0,
                           phi,
                           scratchA.data_ptr<ScalarT>(),
                           (order >= 3) ? scratchB.data_ptr<ScalarT>() : nullptr,
                           (smooth > 0) ? smoothed.data_ptr<ScalarT>() : nullptr,
                           nonFinite.data_ptr<int>(),
                           numVoxels,
                           voxelSize,
                           bandWidth,
                           band,
                           smooth,
                           order,
                           taubin,
                           redistanceIters,
                           stream);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK_VALUE(nonFinite.item<int>() == 0,
                          "field must contain only finite values (grid ",
                          batchIdx,
                          "); leave no-data voxels inactive");
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
