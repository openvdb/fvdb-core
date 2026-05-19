// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-compact, packed-key marching cubes for fp32/fp16 CUDA. See
// MarchingCubesFast.h for the full algorithm and dtype-coverage notes.
//
// In broad strokes:
//   - Classify kernel writes per-leaf-voxel uint8 vertex counts.
//   - A prefix sum over those counts gives the emit-vertex offsets and
//     compacts the surface voxels, so the emit pass touches only
//     surface voxels rather than every voxel in the grid.
//   - The emit kernel writes one packed int64 key per triangle vertex
//     holding `(batchIdx, vid0, vid1)`, and we dedup the 1-D key
//     vector via `torch::unique` (vs the legacy's 3-column
//     `[nTri*3, 3]` `torch::unique_dim`). The output is unpacked back
//     to `[nV, 3]` to preserve the public legacy contract.

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/VoxelCoordTransform.h>
#include <fvdb/detail/ops/MarchingCubes.h>
#include <fvdb/detail/ops/MarchingCubesFast.h>
#include <fvdb/detail/utils/MarchingCubesData.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <nanovdb/NanoVDB.h>

#include <cuda_runtime.h>
#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

constexpr int64_t MCF_BLOCK_SIZE = 128;

// Packed-key bit layout (must match the unpack in marchingCubesFastImpl).
//
// 30 bits per vid supports up to 2^30 = 1,073,741,824 active voxels per
// batch — comfortably covering a paper-hero 800^3 = 512M voxel grid. An
// earlier 24-bit layout silently truncated pids above 16M, which caused
// vertex-dedup over-merging at 400^3 and 512^3 sweeps (triangles still
// matched but vertex counts drifted by <1%).
//
// Layout:  [bits 63..60 batchIdx] [bits 59..30 vid0] [bits 29..0 vid1]
constexpr int MCF_VID_BITS      = 30;
constexpr int64_t MCF_VID_MASK  = (int64_t{1} << MCF_VID_BITS) - 1;
constexpr int MCF_VID1_SHIFT    = 0;
constexpr int MCF_VID0_SHIFT    = MCF_VID_BITS;
constexpr int MCF_BATCH_SHIFT   = 2 * MCF_VID_BITS;
constexpr int64_t MCF_BATCH_MAX = int64_t{1} << (64 - MCF_BATCH_SHIFT);

__host__ __device__ __forceinline__ int64_t
mcf_pack_key(int32_t batchIdx, int64_t vid0, int64_t vid1) {
    return (static_cast<int64_t>(batchIdx) << MCF_BATCH_SHIFT)
         | ((vid0 & MCF_VID_MASK) << MCF_VID0_SHIFT)
         | ((vid1 & MCF_VID_MASK) << MCF_VID1_SHIFT);
}

// -------------------------------------------------------------------------
// mcfClassifyKernel — same per-thread state as the legacy classify kernel.
//
// Templated on the SDF input scalar type (float or at::Half) so that
// fp16 callers don't need a 2x-size transient fp32 upcast of the input
// buffer: the kernel loads fp16 directly and casts to float on the fly
// via c10::Half's `operator float()` (lowers to a single F2F.F32.F16
// instruction per load on sm_89+). Internal arithmetic is all fp32 to
// keep numerics identical across dtypes — the kernel's per-thread state
// and compile-time-indexed vertex positions need the dynamic range.
// -------------------------------------------------------------------------

template <typename InputT>
__global__ void
mcfClassifyKernel(fvdb::GridBatchData::Accessor batchAcc,
                   const InputT *__restrict__ sdfData,
                   const float level,
                   uint8_t *__restrict__ nVertsPerLv) {
    constexpr uint64_t VOXELS_PER_LEAF =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    const uint64_t lvIdx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                           threadIdx.x;
    const uint64_t totalLeafVoxels =
        static_cast<uint64_t>(batchAcc.totalLeaves()) * VOXELS_PER_LEAF;
    if (lvIdx >= totalLeafVoxels) {
        return;
    }

    const int64_t cumLeafIdx   = static_cast<int64_t>(lvIdx / VOXELS_PER_LEAF);
    const int64_t leafVoxelIdx = static_cast<int64_t>(lvIdx % VOXELS_PER_LEAF);
    const JIdxType batchIdx    = batchAcc.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx      = cumLeafIdx - batchAcc.leafOffset(batchIdx);

    const nanovdb::OnIndexGrid *grid = batchAcc.grid(batchIdx);
    const auto &leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(leafVoxelIdx);

    auto acc                  = grid->getAccessor();
    const int64_t voxelOffset = batchAcc.voxelOffset(batchIdx);

    float sdf_0, sdf_1, sdf_2, sdf_3, sdf_4, sdf_5, sdf_6, sdf_7;

#define MCF_LOAD_CORNER(IDX, DX, DY, DZ)                                     \
    {                                                                         \
        const nanovdb::Coord c = ijk + nanovdb::Coord((DX), (DY), (DZ));      \
        if (!acc.isActive(c)) {                                               \
            nVertsPerLv[lvIdx] = 0;                                           \
            return;                                                           \
        }                                                                     \
        sdf_##IDX = static_cast<float>(                                       \
                        sdfData[voxelOffset + acc.getValue(c) - 1]) -         \
                    level;                                                    \
    }

    MCF_LOAD_CORNER(0, 0, 0, 0)
    MCF_LOAD_CORNER(1, 1, 0, 0)
    MCF_LOAD_CORNER(2, 1, 1, 0)
    MCF_LOAD_CORNER(3, 0, 1, 0)
    MCF_LOAD_CORNER(4, 0, 0, 1)
    MCF_LOAD_CORNER(5, 1, 0, 1)
    MCF_LOAD_CORNER(6, 1, 1, 1)
    MCF_LOAD_CORNER(7, 0, 1, 1)

#undef MCF_LOAD_CORNER

    int cubeType = 0;
    if (sdf_0 < 0.0f) cubeType |= 1;
    if (sdf_1 < 0.0f) cubeType |= 2;
    if (sdf_2 < 0.0f) cubeType |= 4;
    if (sdf_3 < 0.0f) cubeType |= 8;
    if (sdf_4 < 0.0f) cubeType |= 16;
    if (sdf_5 < 0.0f) cubeType |= 32;
    if (sdf_6 < 0.0f) cubeType |= 64;
    if (sdf_7 < 0.0f) cubeType |= 128;

    nVertsPerLv[lvIdx] = static_cast<uint8_t>(
        fvdb::detail::marchingCubesNumVertsTable[cubeType]);
}

// -------------------------------------------------------------------------
// mcfEmitCompactKernel — same iteration order as the legacy emit but writes packed int64
// keys to `flatKeys[nTri*3]` instead of (batchIdx, vid0, vid1) triples.
//
// Templated on SDF input scalar type (float or at::Half) for the same
// zero-copy fp16 reason as `mcfClassifyKernel`. Triangle positions are
// still computed and stored in fp32 — world coordinates can exceed
// fp16's dynamic range in large reality-capture scenes, so keeping the
// output at fp32 matches user expectations. The resulting `retVertices`
// JaggedTensor is downcast to the original input dtype at the end of
// `marchingCubesFast` (a small tensor; far less than the SDF buffer).
// -------------------------------------------------------------------------

template <typename InputT>
__global__ void
mcfEmitCompactKernel(
    fvdb::GridBatchData::Accessor batchAcc,
    const InputT *__restrict__ sdfData,
    const float level,
    const int64_t *__restrict__ surfaceLvIdx,
    const int64_t surfaceCount,
    const int64_t *__restrict__ csumCompact,
    torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits>
        trianglesAcc,
    int64_t *__restrict__ flatKeys) {
    constexpr uint64_t VOXELS_PER_LEAF =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;

    const int64_t tid = (static_cast<int64_t>(blockIdx.x) * blockDim.x) +
                        threadIdx.x;
    if (tid >= surfaceCount) {
        return;
    }

    const int64_t lvIdx        = surfaceLvIdx[tid];
    const int64_t cumLeafIdx   = lvIdx / static_cast<int64_t>(VOXELS_PER_LEAF);
    const int64_t leafVoxelIdx = lvIdx % static_cast<int64_t>(VOXELS_PER_LEAF);
    const JIdxType batchIdx    = batchAcc.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx      = cumLeafIdx - batchAcc.leafOffset(batchIdx);

    const nanovdb::OnIndexGrid *grid = batchAcc.grid(batchIdx);
    const auto &leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(leafVoxelIdx);
    const VoxelCoordTransform transform = batchAcc.primalTransform(batchIdx);

    auto acc                  = grid->getAccessor();
    const int64_t voxelOffset = batchAcc.voxelOffset(batchIdx);

    float sdf_0, sdf_1, sdf_2, sdf_3, sdf_4, sdf_5, sdf_6, sdf_7;
    int64_t pid_0, pid_1, pid_2, pid_3, pid_4, pid_5, pid_6, pid_7;
    float p_0_x, p_0_y, p_0_z;
    float p_1_x, p_1_y, p_1_z;
    float p_2_x, p_2_y, p_2_z;
    float p_3_x, p_3_y, p_3_z;
    float p_4_x, p_4_y, p_4_z;
    float p_5_x, p_5_y, p_5_z;
    float p_6_x, p_6_y, p_6_z;
    float p_7_x, p_7_y, p_7_z;

#define MCF_EMIT_LOAD_CORNER(IDX, DX, DY, DZ)                                \
    {                                                                         \
        const nanovdb::Coord c = ijk + nanovdb::Coord((DX), (DY), (DZ));      \
        if (!acc.isActive(c)) {                                               \
            return;                                                           \
        }                                                                     \
        pid_##IDX = voxelOffset + acc.getValue(c) - 1;                        \
        sdf_##IDX = static_cast<float>(sdfData[pid_##IDX]) - level;           \
        const auto worldP = transform.applyInv(static_cast<float>(c[0]),      \
                                               static_cast<float>(c[1]),      \
                                               static_cast<float>(c[2]));     \
        p_##IDX##_x = static_cast<float>(worldP[0]);                          \
        p_##IDX##_y = static_cast<float>(worldP[1]);                          \
        p_##IDX##_z = static_cast<float>(worldP[2]);                          \
    }

    MCF_EMIT_LOAD_CORNER(0, 0, 0, 0)
    MCF_EMIT_LOAD_CORNER(1, 1, 0, 0)
    MCF_EMIT_LOAD_CORNER(2, 1, 1, 0)
    MCF_EMIT_LOAD_CORNER(3, 0, 1, 0)
    MCF_EMIT_LOAD_CORNER(4, 0, 0, 1)
    MCF_EMIT_LOAD_CORNER(5, 1, 0, 1)
    MCF_EMIT_LOAD_CORNER(6, 1, 1, 1)
    MCF_EMIT_LOAD_CORNER(7, 0, 1, 1)

#undef MCF_EMIT_LOAD_CORNER

    int cubeType = 0;
    if (sdf_0 < 0.0f) cubeType |= 1;
    if (sdf_1 < 0.0f) cubeType |= 2;
    if (sdf_2 < 0.0f) cubeType |= 4;
    if (sdf_3 < 0.0f) cubeType |= 8;
    if (sdf_4 < 0.0f) cubeType |= 16;
    if (sdf_5 < 0.0f) cubeType |= 32;
    if (sdf_6 < 0.0f) cubeType |= 64;
    if (sdf_7 < 0.0f) cubeType |= 128;

    const int edgeConfig = fvdb::detail::marchingCubesEdgeTable[cubeType];
    if (edgeConfig == 0) {
        return;
    }

    float vert_0_x = 0.0f, vert_0_y = 0.0f, vert_0_z = 0.0f;
    float vert_1_x = 0.0f, vert_1_y = 0.0f, vert_1_z = 0.0f;
    float vert_2_x = 0.0f, vert_2_y = 0.0f, vert_2_z = 0.0f;
    float vert_3_x = 0.0f, vert_3_y = 0.0f, vert_3_z = 0.0f;
    float vert_4_x = 0.0f, vert_4_y = 0.0f, vert_4_z = 0.0f;
    float vert_5_x = 0.0f, vert_5_y = 0.0f, vert_5_z = 0.0f;
    float vert_6_x = 0.0f, vert_6_y = 0.0f, vert_6_z = 0.0f;
    float vert_7_x = 0.0f, vert_7_y = 0.0f, vert_7_z = 0.0f;
    float vert_8_x = 0.0f, vert_8_y = 0.0f, vert_8_z = 0.0f;
    float vert_9_x = 0.0f, vert_9_y = 0.0f, vert_9_z = 0.0f;
    float vert_10_x = 0.0f, vert_10_y = 0.0f, vert_10_z = 0.0f;
    float vert_11_x = 0.0f, vert_11_y = 0.0f, vert_11_z = 0.0f;

#define MCF_INTERP_EDGE(IDX, IA, IB)                                         \
    if (edgeConfig & (1 << (IDX))) {                                          \
        const float va = sdf_##IA;                                            \
        const float vb = sdf_##IB;                                            \
        const float ax = p_##IA##_x, ay = p_##IA##_y, az = p_##IA##_z;        \
        const float bx = p_##IB##_x, by = p_##IB##_y, bz = p_##IB##_z;        \
        constexpr float MC_EPS = 1.0e-5f;                                     \
        if (fabsf(va) < MC_EPS) {                                             \
            vert_##IDX##_x = ax; vert_##IDX##_y = ay; vert_##IDX##_z = az;    \
        } else if (fabsf(vb) < MC_EPS) {                                      \
            vert_##IDX##_x = bx; vert_##IDX##_y = by; vert_##IDX##_z = bz;    \
        } else if (fabsf(va - vb) < MC_EPS) {                                 \
            vert_##IDX##_x = ax; vert_##IDX##_y = ay; vert_##IDX##_z = az;    \
        } else {                                                              \
            const float w2 = (0.0f - va) / (vb - va);                         \
            const float w1 = 1.0f - w2;                                       \
            vert_##IDX##_x = ax * w1 + bx * w2;                               \
            vert_##IDX##_y = ay * w1 + by * w2;                               \
            vert_##IDX##_z = az * w1 + bz * w2;                               \
        }                                                                     \
    }

    MCF_INTERP_EDGE(0,  0, 1)
    MCF_INTERP_EDGE(1,  1, 2)
    MCF_INTERP_EDGE(2,  2, 3)
    MCF_INTERP_EDGE(3,  0, 3)
    MCF_INTERP_EDGE(4,  4, 5)
    MCF_INTERP_EDGE(5,  5, 6)
    MCF_INTERP_EDGE(6,  6, 7)
    MCF_INTERP_EDGE(7,  7, 4)
    MCF_INTERP_EDGE(8,  0, 4)
    MCF_INTERP_EDGE(9,  1, 5)
    MCF_INTERP_EDGE(10, 6, 2)
    MCF_INTERP_EDGE(11, 3, 7)

#undef MCF_INTERP_EDGE

    const int64_t triangleBase = csumCompact[tid] / 3;

#define MCF_PICK_VERT_X(vlid)                                                \
    ((vlid) == 0  ? vert_0_x  : (vlid) == 1  ? vert_1_x  :                    \
     (vlid) == 2  ? vert_2_x  : (vlid) == 3  ? vert_3_x  :                    \
     (vlid) == 4  ? vert_4_x  : (vlid) == 5  ? vert_5_x  :                    \
     (vlid) == 6  ? vert_6_x  : (vlid) == 7  ? vert_7_x  :                    \
     (vlid) == 8  ? vert_8_x  : (vlid) == 9  ? vert_9_x  :                    \
     (vlid) == 10 ? vert_10_x : vert_11_x)
#define MCF_PICK_VERT_Y(vlid)                                                \
    ((vlid) == 0  ? vert_0_y  : (vlid) == 1  ? vert_1_y  :                    \
     (vlid) == 2  ? vert_2_y  : (vlid) == 3  ? vert_3_y  :                    \
     (vlid) == 4  ? vert_4_y  : (vlid) == 5  ? vert_5_y  :                    \
     (vlid) == 6  ? vert_6_y  : (vlid) == 7  ? vert_7_y  :                    \
     (vlid) == 8  ? vert_8_y  : (vlid) == 9  ? vert_9_y  :                    \
     (vlid) == 10 ? vert_10_y : vert_11_y)
#define MCF_PICK_VERT_Z(vlid)                                                \
    ((vlid) == 0  ? vert_0_z  : (vlid) == 1  ? vert_1_z  :                    \
     (vlid) == 2  ? vert_2_z  : (vlid) == 3  ? vert_3_z  :                    \
     (vlid) == 4  ? vert_4_z  : (vlid) == 5  ? vert_5_z  :                    \
     (vlid) == 6  ? vert_6_z  : (vlid) == 7  ? vert_7_z  :                    \
     (vlid) == 8  ? vert_8_z  : (vlid) == 9  ? vert_9_z  :                    \
     (vlid) == 10 ? vert_10_z : vert_11_z)
#define MCF_PICK_PID(cid)                                                    \
    ((cid) == 0 ? pid_0 : (cid) == 1 ? pid_1 :                                \
     (cid) == 2 ? pid_2 : (cid) == 3 ? pid_3 :                                \
     (cid) == 4 ? pid_4 : (cid) == 5 ? pid_5 :                                \
     (cid) == 6 ? pid_6 : pid_7)

    for (int i = 0; fvdb::detail::marchingCubesTriTable[cubeType][i] != -1;
         i += 3) {
        const int64_t triangleIdx = triangleBase + i / 3;
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            const int vlid = fvdb::detail::marchingCubesTriTable[cubeType][i + vi];
            trianglesAcc[triangleIdx][vi][0] = MCF_PICK_VERT_X(vlid);
            trianglesAcc[triangleIdx][vi][1] = MCF_PICK_VERT_Y(vlid);
            trianglesAcc[triangleIdx][vi][2] = MCF_PICK_VERT_Z(vlid);

            const int e2i_0 = fvdb::detail::marchingCubesE2iTable[vlid][0];
            const int e2i_1 = fvdb::detail::marchingCubesE2iTable[vlid][1];
            int64_t vid0    = MCF_PICK_PID(e2i_0);
            int64_t vid1    = MCF_PICK_PID(e2i_1);
            if (vid0 < vid1) {
                const int64_t t = vid1;
                vid1            = vid0;
                vid0            = t;
            }
            flatKeys[triangleIdx * 3 + vi] =
                mcf_pack_key(static_cast<int32_t>(batchIdx), vid0, vid1);
        }
    }

#undef MCF_PICK_PID
#undef MCF_PICK_VERT_Z
#undef MCF_PICK_VERT_Y
#undef MCF_PICK_VERT_X
}

// -------------------------------------------------------------------------
// Public entry: marchingCubesFastImpl (templated on SDF input scalar type,
// either float or at::Half — see kernel-level docstrings for rationale).
// -------------------------------------------------------------------------

template <typename InputT>
std::vector<JaggedTensor>
marchingCubesFastImpl(const GridBatchData &batchHdl,
                    const torch::Tensor &sdf,
                    double level) {
    batchHdl.checkDevice(sdf);
    TORCH_CHECK_TYPE(sdf.is_floating_point(),
                     "field must have a floating point type");
    TORCH_CHECK(sdf.dim() == 1,
                "Expected field to have 1 dimension but got ", sdf.dim());

    // Guard against silent pid / batch overflow in the packed key. The
    // 30-bit vid field covers up to 1B active voxels per batch; batch
    // field at bits 60..63 supports up to 16 batches.
    TORCH_CHECK_VALUE(batchHdl.batchSize() < MCF_BATCH_MAX,
                      "marchingCubesFast: batch size ", batchHdl.batchSize(),
                      " exceeds packed-key capacity ", MCF_BATCH_MAX);
    TORCH_CHECK_VALUE(batchHdl.totalVoxels() <= (int64_t{1} << MCF_VID_BITS),
                      "marchingCubesFast: totalVoxels ", batchHdl.totalVoxels(),
                      " exceeds packed-key vid capacity ",
                      int64_t{1} << MCF_VID_BITS,
                      " — widen MCF_VID_BITS or fall back to legacy MC.");

    c10::cuda::CUDAGuard guard(sdf.device());
    at::cuda::CUDAStream stream =
        at::cuda::getCurrentCUDAStream(sdf.device().index());

    const int64_t totalLeaves = batchHdl.totalLeaves();
    constexpr int64_t VOXELS_PER_LEAF =
        nanovdb::OnIndexTree::LeafNodeType::NUM_VALUES;
    const int64_t totalLeafVoxels = totalLeaves * VOXELS_PER_LEAF;

    auto longOpts =
        torch::TensorOptions().dtype(torch::kLong).device(sdf.device());
    auto floatOpts =
        torch::TensorOptions().dtype(torch::kFloat32).device(sdf.device());
    auto byteOpts =
        torch::TensorOptions().dtype(torch::kUInt8).device(sdf.device());

    if (totalLeaves == 0) {
        return marchingCubesLegacy(batchHdl,
                                   JaggedTensor::from_data_indices_and_list_ids(
                                       sdf,
                                       torch::zeros({0},
                                                    torch::TensorOptions()
                                                        .dtype(fvdb::JIdxScalarType)
                                                        .device(sdf.device())),
                                       torch::empty({0, 1},
                                                    torch::TensorOptions()
                                                        .dtype(fvdb::JIdxScalarType)
                                                        .device(sdf.device())),
                                       batchHdl.batchSize()),
                                   level);
    }

    // --- Step 1: classify ---
    torch::Tensor nVertsPerLv = torch::empty({totalLeafVoxels}, byteOpts);
    const int64_t classifyBlocks =
        GET_BLOCKS(totalLeafVoxels, MCF_BLOCK_SIZE);
    mcfClassifyKernel<InputT>
        <<<static_cast<unsigned int>(classifyBlocks),
           static_cast<unsigned int>(MCF_BLOCK_SIZE),
           0, stream.stream()>>>(
        batchHdl.deviceAccessor(),
        sdf.data_ptr<InputT>(),
        static_cast<float>(level),
        nVertsPerLv.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- Step 2: compact ---
    torch::Tensor surfaceLvIdx =
        nVertsPerLv.nonzero().squeeze(-1).contiguous();
    const int64_t surfaceCount = surfaceLvIdx.size(0);

    torch::Tensor nVertsCompact =
        nVertsPerLv.index_select(0, surfaceLvIdx).to(torch::kLong);
    torch::Tensor csumInclusive = torch::cumsum(nVertsCompact, 0);
    const int64_t nTriangles =
        surfaceCount > 0
            ? (csumInclusive.index({-1}).item<int64_t>() / 3)
            : 0;
    torch::Tensor csumCompact = torch::roll(csumInclusive, {1});
    if (surfaceCount > 0) {
        csumCompact.index_put_({0}, 0);
    }

    torch::Tensor triangles = torch::empty({nTriangles, 3, 3}, floatOpts);
    // Single-column packed-key tensor (replaces legacy's [nTri, 3, 3] int64).
    torch::Tensor flatKeys =
        torch::empty({nTriangles * 3}, longOpts);

    if (nTriangles > 0) {
        const int64_t emitBlocks =
            GET_BLOCKS(surfaceCount, MCF_BLOCK_SIZE);
        mcfEmitCompactKernel<InputT>
            <<<static_cast<unsigned int>(emitBlocks),
               static_cast<unsigned int>(MCF_BLOCK_SIZE),
               0, stream.stream()>>>(
            batchHdl.deviceAccessor(),
            sdf.data_ptr<InputT>(),
            static_cast<float>(level),
            surfaceLvIdx.data_ptr<int64_t>(),
            surfaceCount,
            csumCompact.data_ptr<int64_t>(),
            triangles.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
            flatKeys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // --- Step 3: 1-D dedup via torch::_unique (replaces unique_dim) ---
    // at::_unique returns (unique_values, inverse_indices). Smaller input
    // footprint: 8 B/elem vs 24 B/elem for the legacy 3-col key.
    auto unqRet                = at::_unique(flatKeys, /*sorted=*/true,
                                             /*return_inverse=*/true);
    torch::Tensor unqKeys      = std::get<0>(unqRet);
    torch::Tensor unqTriangles = std::get<1>(unqRet);

    // Unpack keys back to [nV, 3] (batchIdx, vid0, vid1) for the public
    // contract. Done purely in Torch ops for device-side execution. Each
    // field is masked explicitly so arithmetic-shift sign-extension on
    // signed int64 can't leak upper bits into the lower fields.
    const int64_t nV = unqKeys.size(0);
    torch::Tensor unqVertIdx;
    if (nV > 0) {
        const torch::Tensor vidMaskT =
            torch::full({}, MCF_VID_MASK, unqKeys.options());
        const torch::Tensor batchMaskT =
            torch::full({}, MCF_BATCH_MAX - 1, unqKeys.options());

        torch::Tensor vid1 = torch::bitwise_and(unqKeys, vidMaskT);
        torch::Tensor vid0 = torch::bitwise_and(
            torch::bitwise_right_shift(unqKeys, MCF_VID_BITS), vidMaskT);
        torch::Tensor bidx = torch::bitwise_and(
            torch::bitwise_right_shift(unqKeys, MCF_BATCH_SHIFT), batchMaskT);
        unqVertIdx = torch::stack({bidx, vid0, vid1}, /*dim=*/1).contiguous();
    } else {
        unqVertIdx = torch::empty({0, 3}, longOpts);
    }

    auto flatTriangles = triangles.view({-1, 3});
    torch::Tensor vertices =
        torch::zeros({nV, 3}, floatOpts);
    if (nV > 0) {
        vertices.index_put_({unqTriangles}, flatTriangles);
    }

    unqTriangles            = unqTriangles.view({-1, 3});
    torch::Tensor vBatchIdx = unqVertIdx.index({torch::indexing::Slice(), 0})
                                  .to(fvdb::JIdxScalarType);
    torch::Tensor tBatchIdx =
        vBatchIdx.index({unqTriangles.index({torch::indexing::Slice(), 0})})
            .to(fvdb::JIdxScalarType);

    JaggedTensor retVertices = JaggedTensor::from_data_indices_and_list_ids(
        vertices, vBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());
    JaggedTensor retTriangles = JaggedTensor::from_data_indices_and_list_ids(
        unqTriangles, tBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());
    JaggedTensor retUniqueVertices =
        JaggedTensor::from_data_indices_and_list_ids(
            unqVertIdx, vBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());

    int64_t cumNumVerts = 0;
    for (int i = 1; i < batchHdl.batchSize(); i += 1) {
        cumNumVerts += retVertices.index({i - 1}).jdata().size(0);
        retTriangles.index({i}).jdata().sub_(cumNumVerts);
    }

    return {retVertices, retTriangles, retUniqueVertices};
}

} // anonymous namespace

std::vector<JaggedTensor>
marchingCubesFast(const GridBatchData &batchHdl,
                const JaggedTensor &field,
                double level) {
    TORCH_CHECK_VALUE(field.ldim() == 1,
                      "Expected field to have 1 list dimension, got ",
                      field.ldim());
    TORCH_CHECK_TYPE(field.is_floating_point(),
                     "field must have a floating point type");
    TORCH_CHECK_VALUE(field.numel() == batchHdl.totalVoxels(),
                      "Value count not match!");
    TORCH_CHECK_VALUE(field.num_outer_lists() == batchHdl.batchSize(),
                      "Batch size not match!");

    torch::Tensor fieldJdata = field.jdata();
    if (fieldJdata.dim() == 0) {
        fieldJdata = fieldJdata.unsqueeze(0);
    }
    if (fieldJdata.dim() != 1) {
        fieldJdata = fieldJdata.squeeze();
    }
    batchHdl.checkDevice(field);

    // CPU and fp64 paths go through the legacy (fully templated) impl.
    // This implementation's kernels are fp32-internal because:
    //   (a) vertex world positions can exceed fp16 dynamic range in
    //       large reality-capture scenes (thousands of meters at ~mm
    //       voxel size);
    //   (b) keeping arithmetic at fp32 gives numerically identical
    //       output across input dtypes — a property the ablation
    //       table's correctness gate relies on.
    // But we do NOT upcast the input buffer. The kernels are templated
    // on the SDF input scalar type (float or at::Half) and cast on the
    // fly per load via c10::Half's `operator float()` — a single
    // F2F.F32.F16 per read on sm_89+. For a fp16 input that means:
    //   - zero extra buffer allocation (no N_voxels * 4B transient);
    //   - half the input DRAM bandwidth of the fp32 path;
    //   - only the final small `retVertices` tensor (nV x 3 floats,
    //     orders of magnitude smaller than the SDF) gets downcast to
    //     fp16 to preserve legacy's output-dtype contract.
    // This matters for fvdb-reality-capture's 500M+ voxel hero runs
    // where a 2 GB fp32 upcast would be painful.
    const bool isCuda = field.device().is_cuda();
    const auto origDtype = fieldJdata.scalar_type();
    const bool supportedDtype =
        (origDtype == torch::kFloat32 || origDtype == torch::kHalf);

    if (!isCuda || !supportedDtype) {
        return marchingCubesLegacy(batchHdl, field, level);
    }

    std::vector<JaggedTensor> outputs =
        (origDtype == torch::kFloat32)
            ? marchingCubesFastImpl<float>(batchHdl, fieldJdata, level)
            : marchingCubesFastImpl<at::Half>(batchHdl, fieldJdata, level);

    if (origDtype != torch::kFloat32) {
        // Only `retVertices` (outputs[0]) is dtype-dependent; it's [nV, 3]
        // and typically orders of magnitude smaller than the SDF input,
        // so this cast is negligible. Triangles (face indices) and
        // unqVertIdx are int64 regardless.
        JaggedTensor &verts = outputs[0];
        verts = JaggedTensor::from_data_indices_and_list_ids(
            verts.jdata().to(origDtype),
            verts.jidx(),
            verts.jlidx(),
            verts.num_outer_lists());
    }
    return outputs;
}

} // namespace ops
} // namespace detail
} // namespace fvdb
