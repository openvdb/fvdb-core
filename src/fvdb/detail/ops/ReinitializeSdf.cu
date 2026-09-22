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

// =====================  frozen interface data ====================================================
// Everything the redistance freezes from the initial field phi0 is recomputed from phi0's 6-face
// stencil on every RHS evaluation instead of being stored (phi0 stays resident for the whole solve,
// so this costs neighbour reads, not memory). A voxel is first classified, then only the quantity
// its branch needs is computed:
//   * interface cells (a strict sign change across an active face; an exactly-0 neighbour is a
//     no-data gap and does not make a crossing) get the Russo-Smereka signed distance D, with a
//     denominator shared with the steepest crossing neighbour (see interfaceDistance);
//   * all other cells get the Peng smoothed sign  phi0 / sqrt(phi0^2 + |grad phi0|^2 dx^2).
// An exactly-0 centre is never an interface cell and has sign 0, so its RHS vanishes and such
// (no-data) voxels are never moved; the ray-implicit-intersection op relies on exact 0 surviving as
// a gap marker.

template <typename ScalarT>
__device__ inline bool
isInterfaceCell(ScalarT phiCenter, const ScalarT *faceValues) {
    bool crossing = false;
    for (int k = 0; k < 6; ++k)
        crossing |= faceValues[k] * phiCenter < ScalarT(0);
    return crossing;
}

template <typename ScalarT>
__device__ inline ScalarT
pengSign(ScalarT phiCenter, const ScalarT *faceValues, ScalarT voxelSize) {
    const ScalarT gradX = (faceValues[1] - faceValues[0]) / (2 * voxelSize);
    const ScalarT gradY = (faceValues[3] - faceValues[2]) / (2 * voxelSize);
    const ScalarT gradZ = (faceValues[5] - faceValues[4]) / (2 * voxelSize);
    // Both terms under the root scale as dx^2, so the 0/0 guard does too.
    const ScalarT voxelSizeSq = voxelSize * voxelSize;
    return phiCenter /
           nanovdb::math::Sqrt(phiCenter * phiCenter +
                               (gradX * gradX + gradY * gradY + gradZ * gradZ) * voxelSizeSq +
                               ScalarT(1e-10) * voxelSizeSq);
}

// Face-neighbour value indices of `coord` in the (-x,+x,-y,+y,-z,+z) order used throughout.
template <typename AccessorT>
__device__ inline void
faceIndicesAt(AccessorT &accessor, const nanovdb::Coord &coord, uint64_t faceIndexOut[6]) {
    faceIndexOut[0] = accessor.getValue(coord.offsetBy(-1, 0, 0));
    faceIndexOut[1] = accessor.getValue(coord.offsetBy(1, 0, 0));
    faceIndexOut[2] = accessor.getValue(coord.offsetBy(0, -1, 0));
    faceIndexOut[3] = accessor.getValue(coord.offsetBy(0, 1, 0));
    faceIndexOut[4] = accessor.getValue(coord.offsetBy(0, 0, -1));
    faceIndexOut[5] = accessor.getValue(coord.offsetBy(0, 0, 1));
}

__device__ inline nanovdb::Coord
faceOffset(int face) {
    const int axis = face / 2, direction = (face & 1) ? 1 : -1;
    return nanovdb::Coord(
        axis == 0 ? direction : 0, axis == 1 ? direction : 0, axis == 2 ? direction : 0);
}

// Denominator of the Russo-Smereka distance for one cell: the larger of its central-difference
// gradient norm and its largest one-sided slope, over active faces, which is Russo-Smereka's 1D
// max(central, one-sided, eps) taken to 3D. The Euclidean norm captures oblique surfaces (a single
// face difference only sees one gradient component and would overestimate D by up to sqrt(3));
// the largest one-sided slope takes over at kinks and thin features, where the central difference
// collapses toward zero. Only active faces contribute. An inactive face reads as +/-bandWidth and
// would inflate the slope, pulling a band-edge interface cell toward zero. Restricting the slopes
// to crossing faces was tried and rejected: it picks oblique, shallow faces and shifted zero
// crossings by up to 0.9 voxels on a real SDF.
template <typename ScalarT>
__device__ inline ScalarT
anchorDenominator(ScalarT phiCenter,
                  const ScalarT *faceValues,
                  const uint64_t *faceIndex,
                  ScalarT voxelSize) {
    using nanovdb::math::Abs;
    using nanovdb::math::Max;
    ScalarT maxSlope          = ScalarT(1e-6) * voxelSize;
    ScalarT centralGradientSq = ScalarT(0);
    for (int axis = 0; axis < 3; ++axis) {
        const int minusFace = 2 * axis, plusFace = 2 * axis + 1;
        const bool minusActive = faceIndex[minusFace] != 0, plusActive = faceIndex[plusFace] != 0;
        const ScalarT backwardDiff =
            minusActive ? Abs(phiCenter - faceValues[minusFace]) : ScalarT(0);
        const ScalarT forwardDiff = plusActive ? Abs(faceValues[plusFace] - phiCenter) : ScalarT(0);
        maxSlope                  = Max(maxSlope, Max(backwardDiff, forwardDiff));
        const ScalarT axisGradient =
            (minusActive && plusActive)
                ? Abs(faceValues[plusFace] - faceValues[minusFace]) * ScalarT(0.5)
                : Max(backwardDiff, forwardDiff);
        centralGradientSq += axisGradient * axisGradient;
    }
    return Max(maxSlope, nanovdb::math::Sqrt(centralGradientSq));
}

// Signed distance from an interface cell's centre to the zero crossing of phi0. The two ends of a
// crossing edge must divide by the same denominator or the interpolated crossing moves, and a
// per-cell denominator cannot guarantee that from six face values alone: the medial cell of a
// 1-voxel oblique slab and the interior cell of a 1-voxel rod present the same faces up to scale
// yet need different distances. So each interface cell also evaluates the denominator of its
// steepest crossing neighbour (a 2-ring read, interface cells only) and uses the larger. On the
// exact SDF of a (1,1,1) slab one voxel thick the per-cell rule anchored the medial layer at -0.5
// instead of -0.2887 and the crossing drifted 0.13 voxels per call; with the shared denominator
// the slab, the 1-3 voxel rods and oblique planes are all fixed points, and five repeated calls on
// a sphere move crossings 0.0025 voxels instead of 0.09.
template <typename ScalarT, typename AccessorT>
__device__ inline ScalarT
interfaceDistance(const ScalarT *phi0,
                  ScalarT phiCenter,
                  const ScalarT *faceValues,
                  const uint64_t *faceIndex,
                  AccessorT &accessor,
                  const nanovdb::Coord &centerCoord,
                  ScalarT voxelSize,
                  ScalarT bandWidth) {
    ScalarT denominator = anchorDenominator<ScalarT>(phiCenter, faceValues, faceIndex, voxelSize);

    int steepestFace      = -1;
    ScalarT steepestSlope = ScalarT(-1);
    for (int face = 0; face < 6; ++face) {
        const ScalarT slope = nanovdb::math::Abs(faceValues[face] - phiCenter);
        if (faceValues[face] * phiCenter < ScalarT(0) && slope > steepestSlope) {
            steepestSlope = slope;
            steepestFace  = face;
        }
    }
    if (steepestFace >= 0) {
        const ScalarT neighbourCenter = faceValues[steepestFace];
        uint64_t neighbourFaceIndex[6];
        faceIndicesAt(accessor, centerCoord + faceOffset(steepestFace), neighbourFaceIndex);
        ScalarT neighbourFaces[6];
        for (int face = 0; face < 6; ++face)
            neighbourFaces[face] =
                faceValue<ScalarT>(phi0, neighbourFaceIndex[face], neighbourCenter, bandWidth);
        denominator =
            nanovdb::math::Max(denominator,
                               anchorDenominator<ScalarT>(
                                   neighbourCenter, neighbourFaces, neighbourFaceIndex, voxelSize));
    }
    return voxelSize * phiCenter / denominator;
}

// One-sided upwind selection of the squared one-dimensional derivative for the Godunov scheme.
template <typename ScalarT>
__device__ inline ScalarT
upwind(ScalarT backwardDiff, ScalarT forwardDiff, ScalarT sign) {
    if (sign > 0) {
        ScalarT backwardTerm = nanovdb::math::Max(backwardDiff, ScalarT(0));
        ScalarT forwardTerm  = nanovdb::math::Min(forwardDiff, ScalarT(0));
        return nanovdb::math::Max(backwardTerm * backwardTerm, forwardTerm * forwardTerm);
    } else {
        ScalarT backwardTerm = nanovdb::math::Min(backwardDiff, ScalarT(0));
        ScalarT forwardTerm  = nanovdb::math::Max(forwardDiff, ScalarT(0));
        return nanovdb::math::Max(backwardTerm * backwardTerm, forwardTerm * forwardTerm);
    }
}

// =====================  fused stencil kernels ====================================================
// One TVD-RK stage, fused with its RHS evaluation:
//     outField = clamp(baseCoeff*baseField + stageCoeff*stageField +
//     rhsCoeff*timeStep*rhs(stageField))
// rhs is the Godunov RHS  sign * (1 - |grad stageField|)  with one-sided upwinding on the 6 faces,
// or, at interface cells, the Russo-Smereka subcell update  (D - stageField) / dx, which relaxes
// the value to D from either side and keeps the zero crossing where phi0 put it.
// `outField` must not alias `stageField` (its neighbours are read) but may alias `baseField` (read
// at the centre only).
template <typename ScalarT>
__global__ void
rkStageFusedKernel(const OnIndexGridT *grid,
                   const uint32_t *firstLeafID,
                   const uint64_t *jumpMap,
                   uint64_t firstOffset,
                   const ScalarT *phi0,
                   const ScalarT *stageField,
                   const ScalarT *baseField,
                   ScalarT baseCoeff,
                   ScalarT stageCoeff,
                   ScalarT rhsCoeff,
                   ScalarT timeStep,
                   ScalarT voxelSize,
                   ScalarT bandWidth,
                   ScalarT *outField) {
    VBM_FACES_BEGIN();
    const int64_t bufferIndex = int64_t(centerIndex) - 1;
    const ScalarT phi0Center  = phi0[bufferIndex];
    const ScalarT stageCenter = stageField[bufferIndex];
    VBM_FACE_VALUES(phi0Faces, phi0, phi0Center, bandWidth);

    ScalarT rhs;
    if (isInterfaceCell<ScalarT>(phi0Center, phi0Faces)) {
        const ScalarT distance = interfaceDistance<ScalarT>(
            phi0, phi0Center, phi0Faces, faceIndex, accessor, centerCoord, voxelSize, bandWidth);
        rhs = (distance - stageCenter) / voxelSize;
    } else {
        const ScalarT frozenSign = pengSign<ScalarT>(phi0Center, phi0Faces, voxelSize);
        // Match upwind's frozen sign even if an RK stage crosses zero. For clamped stage values,
        // inactive faces then remain downwind instead of introducing a spurious boundary slope.
        VBM_FACE_VALUES(stageFaces, stageField, frozenSign, bandWidth);
        const ScalarT gradientMagnitude =
            nanovdb::math::Sqrt(upwind<ScalarT>((stageCenter - stageFaces[0]) / voxelSize,
                                                (stageFaces[1] - stageCenter) / voxelSize,
                                                frozenSign) +
                                upwind<ScalarT>((stageCenter - stageFaces[2]) / voxelSize,
                                                (stageFaces[3] - stageCenter) / voxelSize,
                                                frozenSign) +
                                upwind<ScalarT>((stageCenter - stageFaces[4]) / voxelSize,
                                                (stageFaces[5] - stageCenter) / voxelSize,
                                                frozenSign));
        rhs = frozenSign * (ScalarT(1) - gradientMagnitude);
    }

    const ScalarT combined =
        baseCoeff * baseField[bufferIndex] + stageCoeff * stageCenter + rhsCoeff * timeStep * rhs;
    outField[bufferIndex] = nanovdb::math::Clamp(combined, -bandWidth, bandWidth);
}

// one umbrella-Laplacian smoothing pass: outField = inField + weight*(faceMean - inField).
// Double-buffered (inField != outField) so neighbour reads see the pre-pass field.
template <typename ScalarT>
__global__ void
smoothFusedKernel(const OnIndexGridT *grid,
                  const uint32_t *firstLeafID,
                  const uint64_t *jumpMap,
                  uint64_t firstOffset,
                  const ScalarT *inField,
                  ScalarT weight,
                  ScalarT bandWidth,
                  ScalarT *outField) {
    VBM_FACES_BEGIN();
    const int64_t bufferIndex = int64_t(centerIndex) - 1;
    const ScalarT center      = inField[bufferIndex];
    VBM_FACE_VALUES(faceValues, inField, center, bandWidth);
    ScalarT faceMean = (faceValues[0] + faceValues[1] + faceValues[2] + faceValues[3] +
                        faceValues[4] + faceValues[5]) *
                       (ScalarT(1) / ScalarT(6));
    outField[bufferIndex] = center + weight * (faceMean - center);
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
    const size_t fieldBytes     = size_t(numVoxels) * sizeof(ScalarT);

    auto runStage = [&](const ScalarT *phi0,
                        const ScalarT *stageField,
                        const ScalarT *baseField,
                        ScalarT baseCoeff,
                        ScalarT stageCoeff,
                        ScalarT rhsCoeff,
                        ScalarT *outField) {
        if (blockCount) {
            rkStageFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(grid,
                                                                               firstLeafID,
                                                                               jumpMap,
                                                                               firstOffset,
                                                                               phi0,
                                                                               stageField,
                                                                               baseField,
                                                                               baseCoeff,
                                                                               stageCoeff,
                                                                               rhsCoeff,
                                                                               timeStep,
                                                                               voxelSize,
                                                                               bandWidth,
                                                                               outField);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    };

    auto redistance = [&](const ScalarT *phi0, ScalarT *scratch, int iters) {
        const ScalarT one = 1, zero = 0;
        if (order <= 1) { // forward Euler, ping-pong between phi and scratch
            ScalarT *source = phi, *destination = scratch;
            for (int it = 0; it < iters; ++it) {
                runStage(phi0, source, source, one, zero, one, destination);
                std::swap(source, destination);
            }
            if (source != phi)
                C10_CUDA_CHECK(
                    cudaMemcpyAsync(phi, source, fieldBytes, cudaMemcpyDeviceToDevice, stream));
        } else if (order == 2) { // Heun (TVD-RK2) in SSP form: two buffers
            for (int it = 0; it < iters; ++it) {
                runStage(phi0, phi, phi, one, zero, one, scratch);
                runStage(phi0, scratch, phi, ScalarT(0.5), ScalarT(0.5), ScalarT(0.5), phi);
            }
        } else { // Shu-Osher TVD-RK3: three buffers
            for (int it = 0; it < iters; ++it) {
                runStage(phi0, phi, phi, one, zero, one, scratch);
                runStage(phi0, scratch, phi, ScalarT(0.75), ScalarT(0.25), ScalarT(0.25), scratchB);
                runStage(phi0,
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
    redistance(field, scratchA, iters);

    if (smooth) {
        ScalarT *current = phi, *other = scratchA; // ping-pong
        auto pass = [&](ScalarT weight) {
            if (blockCount) {
                smoothFusedKernel<ScalarT><<<blockCount, blockWidth, 0, stream>>>(
                    grid, firstLeafID, jumpMap, firstOffset, current, weight, bandWidth, other);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            std::swap(current, other);
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
        // The smoothed surface is the new anchor. The result must end up in phi (evolving field)
        // and be kept as phi0; one copy suffices if the buffer it landed in takes one of those
        // roles and the other spare buffer becomes the scratch.
        const ScalarT *anchor;
        ScalarT *scratch;
        if (current == phi) {
            C10_CUDA_CHECK(
                cudaMemcpyAsync(smoothedPhi0, phi, fieldBytes, cudaMemcpyDeviceToDevice, stream));
            anchor  = smoothedPhi0;
            scratch = scratchA;
        } else {
            C10_CUDA_CHECK(
                cudaMemcpyAsync(phi, current, fieldBytes, cudaMemcpyDeviceToDevice, stream));
            anchor  = current;
            scratch = smoothedPhi0;
        }
        // The full sweep count is needed here, not a short fixed floor: smoothing is not a small
        // perturbation on a clamped band. On a 52k-voxel chair, mean-curvature passes moved values
        // by 1.2 (1 pass) to 2.5 (8 passes) voxels, and with 4 post-smoothing sweeps the field was
        // off the converged result by up to 2.8 voxels (mean 0.04-0.26); 20 sweeps brought the mean
        // under 0.005.
        redistance(anchor, scratch, iters);
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
    // NaN and +/-Inf both propagate to the extrema, so two scalars validate the field without
    // materializing a per-voxel mask.
    if (fieldJdata.numel() > 0) {
        auto [fieldMin, fieldMax] = torch::aminmax(fieldJdata);
        TORCH_CHECK_VALUE(torch::isfinite(torch::stack({fieldMin, fieldMax})).all().item<bool>(),
                          "field must contain only finite values; leave no-data voxels inactive");
    }
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
