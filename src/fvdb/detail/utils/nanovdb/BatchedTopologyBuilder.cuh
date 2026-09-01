// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Batched leaf-mask topology construction: builds ALL grids of a batch in one pass.
//
// NanoVDB's morphology builders (RefineGrid / CoarsenGrid via TopologyBuilder) are single-grid by
// construction: each getHandle() call performs several stream synchronizations (source-tree
// readback, host-side speculative root refinement, node-count readback for allocation sizing) and
// produces one single-grid handle, which fvdb then merges per batch member -- with another
// synchronization per grid inside nanovdb::cuda::mergeGridHandles. For generated-topology
// workloads that rebuild grids every training iteration (issue #755), that per-member fixed
// overhead -- not the topology size -- dominates wall clock and serializes the GPU.
//
// This header rebuilds the factor-2 refine (subdivision) and coarsen passes so one invocation
// covers the whole batch:
//
//   emit    one kernel over all source leaves of all members emits candidate output leaves as
//           (root-tile sort key, in-tile node key, leaf origin, 512-bit activity mask) slots,
//           segmented per grid;
//   sort    two stable segmented radix sorts put each grid's slots in canonical NanoVDB node
//           order (root tiles by the PointsToGrid offset-shifted key, then upper/lower child
//           offsets); invalid slots sort to the segment tails;
//   dedup   head-flag + scan passes derive the unique leaf/lower/upper nodes, their per-grid
//           counts, and parent linkage (coarsen additionally OR-combines duplicate leaf masks);
//   size    ONE host synchronization reads back the per-grid node counts; per-grid byte offsets
//           are computed on the host and a single output buffer is allocated;
//   build   batched kernels write every grid's GridData/TreeData/RootData (mGridIndex = g,
//           mGridCount = B), root tiles, upper/lower/leaf nodes, leaf mOffset/mPrefixSum, and
//           bounding boxes. Empty members become valid empty grids inline (no host proxy grids).
//
// The mask bit math is NanoVDB's own (RefineLeafMasksFunctor::refineMask /
// CoarsenLeafMasksFunctor::coarsenMask); the header/bbox/prefix-sum stages are transcriptions of
// tools::cuda::TopologyBuilder's functors with (gridIndex, localIndex) indexing. Checksums are
// disabled on the output, matching ops::contiguousGridHandle and mergeGridHandles behavior.
//
// Scratch is allocated through torch (the caching allocator) on the caller's current stream. The
// only stream synchronization per pass is the node-count readback that sizes the output buffer.

#ifndef FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH

#include <fvdb/GridBatchData.h>
#include <fvdb/TorchDeviceBuffer.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Morphology.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace fvdb {
namespace detail {
namespace batched {

using BuildT = nanovdb::ValueOnIndex;
using GridT  = nanovdb::NanoGrid<BuildT>;
using TreeT  = nanovdb::NanoTree<BuildT>;
using RootT  = nanovdb::NanoRoot<BuildT>;
using UpperT = nanovdb::NanoUpper<BuildT>;
using LowerT = nanovdb::NanoLower<BuildT>;
using LeafT  = nanovdb::NanoLeaf<BuildT>;

/// One batched topology pass over the whole batch. `Refine` and `Coarsen` are the factor-2
/// subdivision / coarsening passes; `BoxDilate` is a Minkowski sum with the axis-aligned box
/// [boxLo, boxHi] (components in {-1,0,1}), covering NanoVDB's 26-neighbor DilateGrid
/// (boxLo=-1, boxHi=1) and fvdb's one-sided PadGrid octants ({-1,0}^3 and {0,1}^3). Larger
/// boxes compose from multiple passes (Minkowski sums by boxes compose).
struct TopologyPassSpec {
    enum class Op { Refine, Coarsen, BoxDilate };
    Op op;
    nanovdb::Coord boxLo{0}, boxHi{0}; // BoxDilate only

    static TopologyPassSpec
    refine() {
        return {Op::Refine};
    }
    static TopologyPassSpec
    coarsen() {
        return {Op::Coarsen};
    }
    static TopologyPassSpec
    boxDilate(const nanovdb::Coord &lo, const nanovdb::Coord &hi) {
        return {Op::BoxDilate, lo, hi};
    }
};

/// Device-pointer view of the B source grids of one pass. The grids may live anywhere (a
/// GridBatchData buffer -- including sliced/non-contiguous views -- or the output buffer of a
/// previous pass); only per-member device grid pointers and leaf counts are needed.
struct BatchedTopologySource {
    std::vector<const GridT *> grids; // per-member device grid pointers (host-side vector)
    std::vector<int64_t> leafCounts;  // per-member leaf counts (host-side)
    torch::Device device{torch::kCUDA};
};

/// Result of one batched pass: a single device buffer holding all B output grids back-to-back,
/// plus the host-side layout needed to wrap it in a GridHandle or feed the next pass.
struct BatchedTopologyResult {
    TorchDeviceBuffer buffer;
    std::vector<uint64_t> gridByteOffsets; // B+1 cumulative byte offsets into `buffer`
    std::vector<int64_t> leafCounts;       // per-member output leaf counts
};

// ---------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------

/// Sentinel tile key marking an invalid emission slot (sorts to the end of its segment).
static constexpr uint64_t kInvalidTileKey = ~uint64_t(0);

/// Root-tile sort key: the offset-shifted encoding used by PointsToGrid and RefineGrid::refineRoot
/// (NOT the encoding stored in Tile::key by RootData::CoordToKey). Grids built by every other CUDA
/// path order their root tiles by this key, and elementwise topology comparisons rely on that
/// canonical order.
__device__ inline uint64_t
tileSortKey(const nanovdb::Coord &ijk) {
    static constexpr int64_t kOffset = int64_t(1) << 31;
    return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)) |
           (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) |
           (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);
}

/// In-tile node key ordering leaves by (upper child offset, lower child offset) -- x-major at each
/// level, matching the breadth-first node order every NanoVDB builder produces.
__device__ inline uint32_t
nodeSortKey(const nanovdb::Coord &leafOrigin) {
    return (UpperT::CoordToOffset(leafOrigin) << 12) | LowerT::CoordToOffset(leafOrigin);
}

/// Index of the segment containing element i: upper_bound(offsets, i) - 1 over numSegments+1
/// monotone offsets.
template <typename OffsetT>
__device__ inline int32_t
findSegment(const OffsetT *__restrict__ offsets, int32_t numSegments, OffsetT i) {
    int32_t lo = 0, hi = numSegments - 1; // segment index range
    while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= i) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

/// Per-grid node layout within the shared output buffer; mirrors TopologyBuilder::getBuffer.
struct GridNodes {
    GridT *grid;
    TreeT *tree;
    RootT *root;
    UpperT *upper; // first upper node
    LowerT *lower; // first lower node
    LeafT *leaf;   // first leaf node
    uint32_t numUpper, numLower, numLeaf;
    uint64_t size; // total grid size in bytes
};

struct BuildDeviceArrays {
    uint8_t *dstBase;
    const uint64_t *gridByteOffsets; // [B+1]
    const uint32_t *upperStart;      // [B+1] global upper-node index offsets per grid
    const uint32_t *lowerStart;      // [B+1]
    const uint32_t *leafStart;       // [B+1]
};

__device__ inline GridNodes
gridNodes(const BuildDeviceArrays &a, int32_t g) {
    GridNodes n;
    n.numUpper      = a.upperStart[g + 1] - a.upperStart[g];
    n.numLower      = a.lowerStart[g + 1] - a.lowerStart[g];
    n.numLeaf       = a.leafStart[g + 1] - a.leafStart[g];
    n.size          = a.gridByteOffsets[g + 1] - a.gridByteOffsets[g];
    uint8_t *base   = a.dstBase + a.gridByteOffsets[g];
    uint64_t offset = 0;
    n.grid          = reinterpret_cast<GridT *>(base);
    offset += GridT::memUsage();
    n.tree = reinterpret_cast<TreeT *>(base + offset);
    offset += TreeT::memUsage();
    n.root = reinterpret_cast<RootT *>(base + offset);
    offset += RootT::memUsage(n.numUpper);
    n.upper = reinterpret_cast<UpperT *>(base + offset);
    offset += UpperT::memUsage() * uint64_t(n.numUpper);
    n.lower = reinterpret_cast<LowerT *>(base + offset);
    offset += LowerT::memUsage() * uint64_t(n.numLower);
    n.leaf = reinterpret_cast<LeafT *>(base + offset);
    return n;
}

/// Byte size of one grid given its node counts; must match gridNodes()'s layout.
inline uint64_t
gridByteSize(uint32_t numUpper, uint32_t numLower, uint32_t numLeaf) {
    return GridT::memUsage() + TreeT::memUsage() + RootT::memUsage(numUpper) +
           UpperT::memUsage() * uint64_t(numUpper) + LowerT::memUsage() * uint64_t(numLower) +
           LeafT::DataType::memUsage() * uint64_t(numLeaf);
}

// ---------------------------------------------------------------------------------------------
// Emission kernels: one slot per candidate output leaf, segmented per grid
// ---------------------------------------------------------------------------------------------

struct EmissionArrays {
    uint64_t *tileKey;            // [N] root-tile sort key; kInvalidTileKey for dead slots
    uint32_t *nodeKey;            // [N] (upperChildOffset << 12) | lowerChildOffset
    int32_t *origin;              // [N*3] output leaf origin
    uint64_t *mask;               // [N*8] output leaf 512-bit activity-mask contribution
    const int32_t *segOffsets;    // [B+1] per-grid slot ranges
    const GridT *const *srcGrids; // [B] device pointers to the source grids
    int32_t numSegments;
    int32_t numSlots;
};

/// Refine (factor 2): slot = (source leaf, octant). A source leaf spanning [o, o+7]^3 produces up
/// to 8 fine leaves tiling [2o, 2o+15]^3; the fine mask of octant (bi,bj,bk) is the bit-doubled
/// 4^3 sub-mask of the source mask. Exactly one producer per fine leaf, so no deduplication.
static __global__ void
emitRefinedLeaves(EmissionArrays em) {
    using RefineOp = nanovdb::util::morphology::cuda::RefineLeafMasksFunctor<BuildT>;
    for (int32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < em.numSlots;
         s += gridDim.x * blockDim.x) {
        const int32_t g         = findSegment(em.segOffsets, em.numSegments, s);
        const int32_t localSlot = s - em.segOffsets[g];
        const int32_t leafLocal = localSlot >> 3;
        const int32_t octant    = localSlot & 7;
        const int32_t bi = (octant >> 2) & 1, bj = (octant >> 1) & 1, bk = octant & 1;

        const LeafT &srcLeaf     = em.srcGrids[g]->tree().template getFirstNode<0>()[leafLocal];
        const uint64_t *srcWords = srcLeaf.valueMask().words();

        // Extract the (bi,bj,bk) 4^3 sub-block of the source mask. refineMask() only reads the
        // low-nibble pattern, so the stray high bits from the shift are harmless there, but they
        // must be masked out for the occupancy test.
        nanovdb::Mask<3> fineMask;
        uint64_t *w       = fineMask.words();
        uint64_t occupied = 0;
        for (int i = 0; i < 4; ++i) {
            w[i] = srcWords[i + bi * 4] >> (4 * bk + 32 * bj);
            occupied |= w[i] & 0x000000000f0f0f0fUL;
        }
        if (!occupied) {
            em.tileKey[s] = kInvalidTileKey;
            continue;
        }
        RefineOp::refineMask(fineMask);

        const nanovdb::Coord srcOrigin = srcLeaf.origin();
        const nanovdb::Coord fineOrigin(
            srcOrigin[0] * 2 + 8 * bi, srcOrigin[1] * 2 + 8 * bj, srcOrigin[2] * 2 + 8 * bk);
        em.tileKey[s]        = tileSortKey(fineOrigin);
        em.nodeKey[s]        = nodeSortKey(fineOrigin);
        em.origin[s * 3]     = fineOrigin[0];
        em.origin[s * 3 + 1] = fineOrigin[1];
        em.origin[s * 3 + 2] = fineOrigin[2];
        const uint64_t *fw   = fineMask.words();
        for (int i = 0; i < 8; ++i) {
            em.mask[s * 8 + i] = fw[i];
        }
    }
}

/// Coarsen (factor 2): slot = source leaf. A source leaf collapses to a 4^3 block placed by the
/// parity of the coarsened origin inside one output leaf; up to 8 source leaves contribute to the
/// same output leaf and are deduplicated (mask-OR) downstream.
static __global__ void
emitCoarsenedLeaves(EmissionArrays em) {
    using CoarsenOp = nanovdb::util::morphology::cuda::CoarsenLeafMasksFunctor<BuildT>;
    for (int32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < em.numSlots;
         s += gridDim.x * blockDim.x) {
        const int32_t g         = findSegment(em.segOffsets, em.numSegments, s);
        const int32_t leafLocal = s - em.segOffsets[g];

        const LeafT &srcLeaf = em.srcGrids[g]->tree().template getFirstNode<0>()[leafLocal];
        if (srcLeaf.valueMask().isOff()) { // leaves always have active voxels; defensive
            em.tileKey[s] = kInvalidTileKey;
            continue;
        }

        const nanovdb::Coord coarseOrigin =
            nanovdb::util::morphology::coarsenCoord(srcLeaf.origin());
        // The 4^3 coarse block starts at a multiple of 4, so it lies in a single 8-aligned leaf.
        const nanovdb::Coord dstLeafOrigin(
            coarseOrigin[0] & ~7, coarseOrigin[1] & ~7, coarseOrigin[2] & ~7);
        const int bi = (coarseOrigin[0] & 7) ? 1 : 0;
        const int bj = (coarseOrigin[1] & 7) ? 1 : 0;
        const int bk = (coarseOrigin[2] & 7) ? 1 : 0;

        nanovdb::Mask<3> coarseMask = srcLeaf.valueMask();
        CoarsenOp::coarsenMask(coarseMask);

        uint64_t contribution[8] = {};
        const uint64_t *cw       = coarseMask.words();
        for (int wi = 0; wi < 4; ++wi) {
            contribution[wi + 4 * bi] = cw[wi] << (4 * bk + 32 * bj);
        }

        em.tileKey[s]        = tileSortKey(dstLeafOrigin);
        em.nodeKey[s]        = nodeSortKey(dstLeafOrigin);
        em.origin[s * 3]     = dstLeafOrigin[0];
        em.origin[s * 3 + 1] = dstLeafOrigin[1];
        em.origin[s * 3 + 2] = dstLeafOrigin[2];
        for (int i = 0; i < 8; ++i) {
            em.mask[s * 8 + i] = contribution[i];
        }
    }
}

/// Shifts a leaf-mask word along z by s in [-7,7], dropping bits that leave the leaf.
/// (Mask<3> word layout: word index = local x, byte within word = local y, bit within byte = z.)
__device__ inline uint64_t
shiftWordZ(uint64_t w, int s) {
    if (s > 0) {
        return (w & (0x0101010101010101UL * (0xffu >> s))) << s;
    }
    if (s < 0) {
        return (w >> -s) & (0x0101010101010101UL * (0xffu >> -s));
    }
    return w;
}

/// Shifts a leaf-mask word along y by s in [-7,7]; byte granularity, natural truncation.
__device__ inline uint64_t
shiftWordY(uint64_t w, int s) {
    return s > 0 ? w << (8 * s) : (s < 0 ? w >> (-8 * s) : w);
}

/// BoxDilate: slot = (source leaf, target neighbor-leaf offset db). The contribution of a source
/// leaf S to the target leaf at S.origin + 8*db is U_{o in [boxLo,boxHi]} shift(S, o - 8*db),
/// which factorizes per axis (Minkowski sums of axis-aligned boxes compose); shifts with any
/// component of magnitude >= 8 vanish, so per axis: db==0 keeps all box offsets, db==+-1 keeps
/// only o==+-1 (shift of -+7). Up to 2^3 source leaves contribute to one target leaf per
/// one-sided pass (3^3 for the full box), deduplicated (mask-OR) downstream like coarsen.
static __global__ void
emitBoxDilatedLeaves(EmissionArrays em, nanovdb::Coord boxLo, nanovdb::Coord boxHi) {
    // Per-axis target-offset ranges and fanout.
    int dbLo[3], fan[3];
    for (int a = 0; a < 3; ++a) {
        dbLo[a] = boxLo[a] < 0 ? -1 : 0;
        fan[a]  = (boxHi[a] > 0 ? 1 : 0) - dbLo[a] + 1;
    }
    const int32_t fanout = fan[0] * fan[1] * fan[2];

    for (int32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < em.numSlots;
         s += gridDim.x * blockDim.x) {
        const int32_t g         = findSegment(em.segOffsets, em.numSegments, s);
        const int32_t localSlot = s - em.segOffsets[g];
        const int32_t leafLocal = localSlot / fanout;
        int32_t t               = localSlot % fanout;
        const int dbz           = dbLo[2] + t % fan[2];
        t /= fan[2];
        const int dby = dbLo[1] + t % fan[1];
        const int dbx = dbLo[0] + t / fan[1];

        const LeafT &srcLeaf     = em.srcGrids[g]->tree().template getFirstNode<0>()[leafLocal];
        const uint64_t *srcWords = srcLeaf.valueMask().words();

        // z-stage, then y-stage (per word), then x-stage (word permutation).
        uint64_t wz[8], wy[8], out[8];
        for (int i = 0; i < 8; ++i) {
            uint64_t acc = 0;
            for (int oz = boxLo[2]; oz <= boxHi[2]; ++oz) {
                const int sz = oz - 8 * dbz;
                if (sz > -8 && sz < 8) {
                    acc |= shiftWordZ(srcWords[i], sz);
                }
            }
            wz[i] = acc;
        }
        for (int i = 0; i < 8; ++i) {
            uint64_t acc = 0;
            for (int oy = boxLo[1]; oy <= boxHi[1]; ++oy) {
                const int sy = oy - 8 * dby;
                if (sy > -8 && sy < 8) {
                    acc |= shiftWordY(wz[i], sy);
                }
            }
            wy[i] = acc;
        }
        uint64_t occupied = 0;
        for (int x = 0; x < 8; ++x) {
            uint64_t acc = 0;
            for (int ox = boxLo[0]; ox <= boxHi[0]; ++ox) {
                const int sx  = ox - 8 * dbx; // out[x] gathers wy[x - sx]
                const int src = x - sx;
                if (src >= 0 && src < 8) {
                    acc |= wy[src];
                }
            }
            out[x] = acc;
            occupied |= acc;
        }
        if (!occupied) {
            em.tileKey[s] = kInvalidTileKey;
            continue;
        }

        const nanovdb::Coord srcOrigin = srcLeaf.origin();
        const nanovdb::Coord dstLeafOrigin(
            srcOrigin[0] + 8 * dbx, srcOrigin[1] + 8 * dby, srcOrigin[2] + 8 * dbz);
        em.tileKey[s]        = tileSortKey(dstLeafOrigin);
        em.nodeKey[s]        = nodeSortKey(dstLeafOrigin);
        em.origin[s * 3]     = dstLeafOrigin[0];
        em.origin[s * 3 + 1] = dstLeafOrigin[1];
        em.origin[s * 3 + 2] = dstLeafOrigin[2];
        for (int i = 0; i < 8; ++i) {
            em.mask[s * 8 + i] = out[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sort / dedup kernels
// ---------------------------------------------------------------------------------------------

static __global__ void
iotaKernel(uint32_t *out, int32_t n) {
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = uint32_t(i);
    }
}

static __global__ void
gatherKeys64(const uint64_t *__restrict__ keys,
             const uint32_t *__restrict__ perm,
             uint64_t *__restrict__ out,
             int32_t n) {
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = keys[perm[i]];
    }
}

struct SortedArrays {
    const uint64_t *tileKey; // [N] canonical order, invalid slots at segment tails
    uint32_t *nodeKey;       // [N]
    int32_t *origin;         // [N*3]
    uint64_t *mask;          // [N*8]
};

static __global__ void
gatherSorted(EmissionArrays em, const uint32_t *__restrict__ perm, SortedArrays out) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < em.numSlots;
         j += gridDim.x * blockDim.x) {
        const uint32_t s      = perm[j];
        out.nodeKey[j]        = em.nodeKey[s];
        out.origin[j * 3]     = em.origin[s * 3];
        out.origin[j * 3 + 1] = em.origin[s * 3 + 1];
        out.origin[j * 3 + 2] = em.origin[s * 3 + 2];
        for (int i = 0; i < 8; ++i) {
            out.mask[j * 8 + i] = em.mask[s * 8 + i];
        }
    }
}

/// Marks the first slot of every distinct leaf / lower / upper node within each grid's segment.
static __global__ void
computeHeadFlags(const uint64_t *__restrict__ tileKey,
                 const uint32_t *__restrict__ nodeKey,
                 const int32_t *__restrict__ segOffsets,
                 int32_t numSegments,
                 int32_t n,
                 uint32_t *__restrict__ leafFlag,
                 uint32_t *__restrict__ lowerFlag,
                 uint32_t *__restrict__ upperFlag) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        uint32_t lf = 0, wf = 0, uf = 0;
        if (tileKey[j] != kInvalidTileKey) {
            const int32_t g     = findSegment(segOffsets, numSegments, j);
            const bool first    = (j == segOffsets[g]);
            const bool newTile  = first || tileKey[j] != tileKey[j - 1];
            const bool newLower = newTile || (nodeKey[j] >> 12) != (nodeKey[j - 1] >> 12);
            const bool newLeaf  = newLower || nodeKey[j] != nodeKey[j - 1];
            lf                  = newLeaf ? 1u : 0u;
            wf                  = newLower ? 1u : 0u;
            uf                  = newTile ? 1u : 0u;
        }
        leafFlag[j]  = lf;
        lowerFlag[j] = wf;
        upperFlag[j] = uf;
    }
}

/// Per-grid global node-index offsets, read off the inclusive rank scans at segment boundaries.
static __global__ void
segmentNodeOffsets(const uint32_t *__restrict__ leafRank,
                   const uint32_t *__restrict__ lowerRank,
                   const uint32_t *__restrict__ upperRank,
                   const int32_t *__restrict__ segOffsets,
                   int32_t numSegments,
                   uint32_t *__restrict__ leafStart,
                   uint32_t *__restrict__ lowerStart,
                   uint32_t *__restrict__ upperStart) {
    const int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g > numSegments) {
        return;
    }
    const int32_t boundary = segOffsets[g]; // slots before this boundary belong to grids < g
    leafStart[g]           = boundary == 0 ? 0 : leafRank[boundary - 1];
    lowerStart[g]          = boundary == 0 ? 0 : lowerRank[boundary - 1];
    upperStart[g]          = boundary == 0 ? 0 : upperRank[boundary - 1];
}

/// Records, for every unique node, its head slot and its parent's global node index.
static __global__ void
scatterNodeTables(const uint32_t *__restrict__ leafFlag,
                  const uint32_t *__restrict__ lowerFlag,
                  const uint32_t *__restrict__ upperFlag,
                  const uint32_t *__restrict__ leafRank,
                  const uint32_t *__restrict__ lowerRank,
                  const uint32_t *__restrict__ upperRank,
                  int32_t n,
                  uint32_t *__restrict__ leafHeadSlot,
                  uint32_t *__restrict__ leafParent,
                  uint32_t *__restrict__ lowerHeadSlot,
                  uint32_t *__restrict__ lowerParent,
                  uint32_t *__restrict__ upperHeadSlot) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        if (leafFlag[j]) {
            const uint32_t leafIdx = leafRank[j] - 1;
            leafHeadSlot[leafIdx]  = uint32_t(j);
            leafParent[leafIdx]    = lowerRank[j] - 1;
        }
        if (lowerFlag[j]) {
            const uint32_t lowerIdx = lowerRank[j] - 1;
            lowerHeadSlot[lowerIdx] = uint32_t(j);
            lowerParent[lowerIdx]   = upperRank[j] - 1;
        }
        if (upperFlag[j]) {
            upperHeadSlot[upperRank[j] - 1] = uint32_t(j);
        }
    }
}

/// Coarsen only: OR every duplicate slot's mask contribution into its unique leaf's head slot.
static __global__ void
combineDuplicateMasks(const uint64_t *__restrict__ tileKey,
                      const uint32_t *__restrict__ leafFlag,
                      const uint32_t *__restrict__ leafRank,
                      const uint32_t *__restrict__ leafHeadSlot,
                      int32_t n,
                      uint64_t *__restrict__ mask) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        if (tileKey[j] == kInvalidTileKey || leafFlag[j]) {
            continue;
        }
        const uint32_t head = leafHeadSlot[leafRank[j] - 1];
        for (int i = 0; i < 8; ++i) {
            const uint64_t w = mask[j * 8 + i];
            if (w) {
                nanovdb::util::atomicOr(&mask[head * 8 + i], w);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Build kernels (after the single sizing readback and output allocation)
// ---------------------------------------------------------------------------------------------

/// Copies every source grid's GridData header to its output position (preserving grid name, map,
/// voxel size); the fields reset by initGridTreeRoot are rewritten afterwards.
static __global__ void
copyGridData(const GridT *const *__restrict__ srcGrids, BuildDeviceArrays a, int32_t numGrids) {
    constexpr int32_t kWords = int32_t(sizeof(nanovdb::GridData) / sizeof(uint64_t));
    const int32_t totalWords = numGrids * kWords;
    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < totalWords;
         t += gridDim.x * blockDim.x) {
        const int32_t g     = t / kWords;
        const int32_t w     = t % kWords;
        const uint64_t *src = reinterpret_cast<const uint64_t *>(srcGrids[g]);
        uint64_t *dst       = reinterpret_cast<uint64_t *>(a.dstBase + a.gridByteOffsets[g]);
        dst[w]              = src[w];
    }
}

/// Per-grid transcription of tools::cuda::topology::detail::BuildGridTreeRootFunctor with
/// mGridIndex/mGridCount/mGridSize set for a multi-grid buffer.
static __global__ void
initGridTreeRoot(BuildDeviceArrays a, int32_t numGrids) {
    const int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGrids) {
        return;
    }
    GridNodes n = gridNodes(a, g);

    auto &root       = *n.root;
    root.mTableSize  = n.numUpper;
    root.mBackground = RootT::ValueType(0);
    root.mMinimum = root.mMaximum = RootT::ValueType(0);
    root.mAverage = root.mStdDevi = RootT::FloatType(0);
    root.mBBox                    = nanovdb::CoordBBox();

    auto &tree = *n.tree;
    tree.setRoot(&root);
    if (n.numUpper) {
        tree.setFirstNode(n.upper);
        tree.setFirstNode(n.lower);
        tree.setFirstNode(n.leaf);
    } else {
        tree.setFirstNode<UpperT>(nullptr);
        tree.setFirstNode<LowerT>(nullptr);
        tree.setFirstNode<LeafT>(nullptr);
    }
    tree.mNodeCount[2] = n.numUpper;
    tree.mNodeCount[1] = n.numLower;
    tree.mNodeCount[0] = n.numLeaf;
    tree.mVoxelCount   = 0; // set by finalizeGrids once leaf masks are in place
    tree.mTileCount[2] = tree.mTileCount[1] = tree.mTileCount[0] = 0;

    auto &grid = *n.grid;
    grid.mChecksum.disable();
    grid.mFlags.initMask({nanovdb::GridFlags::IsBreadthFirst});
    grid.mGridIndex           = uint32_t(g);
    grid.mGridCount           = uint32_t(numGrids);
    grid.mGridSize            = n.size;
    grid.mWorldBBox           = nanovdb::Vec3dBBox();
    grid.mVoxelSize           = grid.mMap.getVoxelSize();
    grid.mBlindMetadataOffset = n.size;
    grid.mBlindMetadataCount  = 0u;
    grid.mData1               = 1u;
}

/// One thread per unique upper node: writes its root tile (in canonical tile order) and preamble.
static __global__ void
buildUpperNodes(BuildDeviceArrays a,
                SortedArrays sorted,
                const uint32_t *__restrict__ upperHeadSlot,
                int32_t numGrids,
                int32_t totalUpper) {
    for (int32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < totalUpper;
         u += gridDim.x * blockDim.x) {
        const int32_t g      = findSegment(a.upperStart, numGrids, uint32_t(u));
        const GridNodes n    = gridNodes(a, g);
        const uint32_t local = uint32_t(u) - a.upperStart[g];
        const uint32_t slot  = upperHeadSlot[u];
        const nanovdb::Coord tileOrigin(sorted.origin[slot * 3] & ~4095,
                                        sorted.origin[slot * 3 + 1] & ~4095,
                                        sorted.origin[slot * 3 + 2] & ~4095);
        UpperT &upper = n.upper[local];
        n.root->tile(local)->setChild(tileOrigin, &upper, n.root->data());
        upper.mBBox  = nanovdb::CoordBBox();
        upper.mFlags = (uint64_t)nanovdb::GridFlags::HasBBox;
    }
}

/// One thread per unique lower node: links it under its upper node and writes its preamble.
static __global__ void
buildLowerNodes(BuildDeviceArrays a,
                SortedArrays sorted,
                const uint32_t *__restrict__ lowerHeadSlot,
                const uint32_t *__restrict__ lowerParent,
                int32_t numGrids,
                int32_t totalLower) {
    for (int32_t l = blockIdx.x * blockDim.x + threadIdx.x; l < totalLower;
         l += gridDim.x * blockDim.x) {
        const int32_t g         = findSegment(a.lowerStart, numGrids, uint32_t(l));
        const GridNodes n       = gridNodes(a, g);
        const uint32_t local    = uint32_t(l) - a.lowerStart[g];
        const uint32_t slot     = lowerHeadSlot[l];
        const uint32_t upperOff = sorted.nodeKey[slot] >> 12;
        UpperT &upper           = n.upper[lowerParent[l] - a.upperStart[g]];
        LowerT &lower           = n.lower[local];
        upper.mChildMask.setOnAtomic(upperOff);
        upper.setChild(upperOff, &lower);
        lower.mBBox  = nanovdb::CoordBBox();
        lower.mFlags = (uint64_t)nanovdb::GridFlags::HasBBox;
    }
}

/// One thread per unique leaf: links it under its lower node, writes its activity mask, per-leaf
/// prefix sums, and voxel count.
static __global__ void
buildLeafNodes(BuildDeviceArrays a,
               SortedArrays sorted,
               const uint32_t *__restrict__ leafHeadSlot,
               const uint32_t *__restrict__ leafParent,
               int32_t numGrids,
               int32_t totalLeaf,
               uint64_t *__restrict__ voxelCounts) { // [totalLeaf+1]; element 0 stays 0
    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < totalLeaf;
         t += gridDim.x * blockDim.x) {
        const int32_t g         = findSegment(a.leafStart, numGrids, uint32_t(t));
        const GridNodes n       = gridNodes(a, g);
        const uint32_t local    = uint32_t(t) - a.leafStart[g];
        const uint32_t slot     = leafHeadSlot[t];
        const uint32_t lowerOff = sorted.nodeKey[slot] & 0xFFFu;
        LowerT &lower           = n.lower[leafParent[t] - a.lowerStart[g]];
        LeafT &leaf             = n.leaf[local];
        lower.mChildMask.setOnAtomic(lowerOff);
        lower.setChild(lowerOff, &leaf);
        leaf.mBBoxMin = nanovdb::Coord(
            sorted.origin[slot * 3], sorted.origin[slot * 3 + 1], sorted.origin[slot * 3 + 2]);
        leaf.mFlags = uint8_t(nanovdb::GridFlags::HasBBox);

        uint64_t *dstWords = leaf.mValueMask.words();
        for (int i = 0; i < 8; ++i) {
            dstWords[i] = sorted.mask[slot * 8 + i];
        }

        // Per-leaf voxel count and the 9-bit encoded intra-leaf prefix sums (transcribed from
        // TopologyBuilder's UpdateLeafVoxelCountsAndPrefixSumFunctor).
        uint64_t prefixSum = 0, sum = nanovdb::util::countOn(dstWords[0]);
        prefixSum = sum;
        for (int wi = 1; wi < 7; ++wi) {
            sum += nanovdb::util::countOn(dstWords[wi]);
            prefixSum |= sum << (9 * wi);
        }
        sum += nanovdb::util::countOn(dstWords[7]);
        voxelCounts[t + 1] = sum;
        leaf.mPrefixSum    = prefixSum;
    }
}

/// One thread per leaf: 1-based per-grid value offsets from the global voxel-count scan.
static __global__ void
setLeafOffsets(BuildDeviceArrays a,
               const uint64_t *__restrict__ voxelScan, // [totalLeaf+1] inclusive scan, [0] == 0
               int32_t numGrids,
               int32_t totalLeaf) {
    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < totalLeaf;
         t += gridDim.x * blockDim.x) {
        const int32_t g       = findSegment(a.leafStart, numGrids, uint32_t(t));
        const GridNodes n     = gridNodes(a, g);
        const uint32_t local  = uint32_t(t) - a.leafStart[g];
        n.leaf[local].mOffset = voxelScan[t] - voxelScan[a.leafStart[g]] + 1;
    }
}

static __global__ void
propagateLeafBBox(BuildDeviceArrays a,
                  const uint32_t *__restrict__ leafParent,
                  int32_t numGrids,
                  int32_t totalLeaf) {
    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < totalLeaf;
         t += gridDim.x * blockDim.x) {
        const int32_t g   = findSegment(a.leafStart, numGrids, uint32_t(t));
        const GridNodes n = gridNodes(a, g);
        LeafT &leaf       = n.leaf[uint32_t(t) - a.leafStart[g]];
        LowerT &lower     = n.lower[leafParent[t] - a.lowerStart[g]];
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }
}

static __global__ void
propagateLowerBBox(BuildDeviceArrays a,
                   const uint32_t *__restrict__ lowerParent,
                   int32_t numGrids,
                   int32_t totalLower) {
    for (int32_t l = blockIdx.x * blockDim.x + threadIdx.x; l < totalLower;
         l += gridDim.x * blockDim.x) {
        const int32_t g   = findSegment(a.lowerStart, numGrids, uint32_t(l));
        const GridNodes n = gridNodes(a, g);
        LowerT &lower     = n.lower[uint32_t(l) - a.lowerStart[g]];
        UpperT &upper     = n.upper[lowerParent[l] - a.upperStart[g]];
        upper.mBBox.expandAtomic(lower.bbox());
    }
}

static __global__ void
propagateUpperBBox(BuildDeviceArrays a, int32_t numGrids, int32_t totalUpper) {
    for (int32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < totalUpper;
         u += gridDim.x * blockDim.x) {
        const int32_t g   = findSegment(a.upperStart, numGrids, uint32_t(u));
        const GridNodes n = gridNodes(a, g);
        n.root->mBBox.expandAtomic(n.upper[uint32_t(u) - a.upperStart[g]].bbox());
    }
}

/// Per-grid epilogue: voxel counts, index-space -> world bbox, HasBBox flag.
static __global__ void
finalizeGrids(BuildDeviceArrays a, const uint64_t *__restrict__ voxelScan, int32_t numGrids) {
    const int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGrids) {
        return;
    }
    const GridNodes n = gridNodes(a, g);
    if (n.numLeaf == 0) { // empty grid: keep the empty bbox and defaults
        return;
    }
    const uint64_t voxelCount = voxelScan[a.leafStart[g + 1]] - voxelScan[a.leafStart[g]];
    n.tree->mVoxelCount       = voxelCount;
    n.grid->mData1            = voxelCount + 1;

    nanovdb::CoordBBox bbox = n.root->mBBox;
    bbox.max() += nanovdb::Coord(1);
    n.grid->mFlags.setMaskOn(nanovdb::GridFlags::HasBBox);
    n.grid->mWorldBBox = bbox.transform(n.grid->data()->mMap);
}

// ---------------------------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------------------------

inline BatchedTopologySource
sourceFromGridBatch(const GridBatchData &batch) {
    BatchedTopologySource src;
    src.device              = batch.device();
    const int64_t batchSize = batch.batchSize();
    src.grids.reserve(batchSize);
    src.leafCounts.reserve(batchSize);
    for (int64_t i = 0; i < batchSize; ++i) {
        src.grids.push_back(batch.deviceGridPtrAt(i));
        src.leafCounts.push_back(batch.numLeavesAt(i));
    }
    return src;
}

inline BatchedTopologySource
sourceFromResult(const BatchedTopologyResult &result, const torch::Device &device) {
    BatchedTopologySource src;
    src.device             = device;
    const size_t batchSize = result.leafCounts.size();
    src.grids.reserve(batchSize);
    src.leafCounts      = result.leafCounts;
    const uint8_t *base = result.buffer.deviceData();
    for (size_t i = 0; i < batchSize; ++i) {
        src.grids.push_back(reinterpret_cast<const GridT *>(base + result.gridByteOffsets[i]));
    }
    return src;
}

/// Runs one batched topology pass over all grids of `src`.
/// One cudaStreamSynchronize total (the node-count readback that sizes the output allocation).
inline BatchedTopologyResult
runBatchedTopologyPass(const BatchedTopologySource &src,
                       const TopologyPassSpec &pass,
                       cudaStream_t stream) {
    TORCH_CHECK(src.device.is_cuda(), "batched topology passes require a CUDA device");
    const int32_t numGrids = int32_t(src.grids.size());
    TORCH_CHECK(numGrids > 0, "batched topology pass requires at least one grid");
    int32_t fanout = 1;
    switch (pass.op) {
    case TopologyPassSpec::Op::Refine: fanout = 8; break;
    case TopologyPassSpec::Op::Coarsen: fanout = 1; break;
    case TopologyPassSpec::Op::BoxDilate:
        for (int a = 0; a < 3; ++a) {
            TORCH_CHECK(pass.boxLo[a] >= -1 && pass.boxLo[a] <= 0 && pass.boxHi[a] >= 0 &&
                            pass.boxHi[a] <= 1,
                        "BoxDilate pass components must be in {-1,0} / {0,1}");
            fanout *= (pass.boxHi[a] > 0 ? 1 : 0) - (pass.boxLo[a] < 0 ? -1 : 0) + 1;
        }
        break;
    }

    // Per-grid emission slot ranges (host-known: leaf counts x fanout; no sync needed).
    std::vector<int32_t> segOffsetsHost(numGrids + 1, 0);
    int64_t total = 0;
    for (int32_t g = 0; g < numGrids; ++g) {
        segOffsetsHost[g] = int32_t(total);
        total += src.leafCounts[g] * fanout;
    }
    TORCH_CHECK(total <= std::numeric_limits<int32_t>::max(),
                "batched topology pass: emission slot count ",
                total,
                " exceeds int32 range");
    segOffsetsHost[numGrids] = int32_t(total);
    const int32_t numSlots   = int32_t(total);
    const int64_t allocSlots = std::max(numSlots, 1); // torch::empty({0}) yields a null data_ptr

    const auto byteOpts = torch::TensorOptions().dtype(torch::kUInt8).device(src.device);
    const auto i32Opts  = byteOpts.dtype(torch::kInt32);
    const auto i64Opts  = byteOpts.dtype(torch::kInt64);

    constexpr int32_t kThreads = 256;
    const auto numBlocks = [](int32_t n) { return std::max((n + kThreads - 1) / kThreads, 1); };
    const auto u32       = [](torch::Tensor &t) {
        return reinterpret_cast<uint32_t *>(t.data_ptr<int32_t>());
    };
    const auto u64 = [](torch::Tensor &t) {
        return reinterpret_cast<uint64_t *>(t.data_ptr<int64_t>());
    };

    // Runs a cub function twice: once to query scratch size, once for real.
    const auto callCub = [&](auto fn) {
        size_t tempBytes = 0;
        fn(nullptr, tempBytes);
        torch::Tensor tempStorage =
            torch::empty({std::max<int64_t>(int64_t(tempBytes), 1)}, byteOpts);
        fn(tempStorage.data_ptr<uint8_t>(), tempBytes);
    };

    // Small host-side staging uploaded once per pass: segment offsets + source grid pointers.
    // (Pageable H2D copies are synchronous with the host but do not synchronize the stream.)
    torch::Tensor segOffsetsDev = torch::empty({numGrids + 1}, i32Opts);
    torch::Tensor srcGridsDev   = torch::empty({numGrids}, i64Opts);
    C10_CUDA_CHECK(cudaMemcpyAsync(segOffsetsDev.data_ptr<int32_t>(),
                                   segOffsetsHost.data(),
                                   sizeof(int32_t) * (numGrids + 1),
                                   cudaMemcpyHostToDevice,
                                   stream));
    static_assert(sizeof(const GridT *) == sizeof(int64_t));
    C10_CUDA_CHECK(cudaMemcpyAsync(srcGridsDev.data_ptr<int64_t>(),
                                   src.grids.data(),
                                   sizeof(const GridT *) * numGrids,
                                   cudaMemcpyHostToDevice,
                                   stream));

    // --- Emit candidate output leaves. ---
    torch::Tensor emTileKey = torch::empty({allocSlots}, i64Opts);
    torch::Tensor emNodeKey = torch::empty({allocSlots}, i32Opts);
    torch::Tensor emOrigin  = torch::empty({allocSlots * 3}, i32Opts);
    torch::Tensor emMask    = torch::empty({allocSlots * 8}, i64Opts);

    EmissionArrays em;
    em.tileKey     = u64(emTileKey);
    em.nodeKey     = u32(emNodeKey);
    em.origin      = emOrigin.data_ptr<int32_t>();
    em.mask        = u64(emMask);
    em.segOffsets  = segOffsetsDev.data_ptr<int32_t>();
    em.srcGrids    = reinterpret_cast<const GridT *const *>(srcGridsDev.data_ptr<int64_t>());
    em.numSegments = numGrids;
    em.numSlots    = numSlots;

    if (numSlots > 0) {
        switch (pass.op) {
        case TopologyPassSpec::Op::Refine:
            emitRefinedLeaves<<<numBlocks(numSlots), kThreads, 0, stream>>>(em);
            break;
        case TopologyPassSpec::Op::Coarsen:
            emitCoarsenedLeaves<<<numBlocks(numSlots), kThreads, 0, stream>>>(em);
            break;
        case TopologyPassSpec::Op::BoxDilate:
            emitBoxDilatedLeaves<<<numBlocks(numSlots), kThreads, 0, stream>>>(
                em, pass.boxLo, pass.boxHi);
            break;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // --- Canonical-order segmented sort: stable by node key, then stable by tile key. ---
    torch::Tensor permA   = torch::empty({allocSlots}, i32Opts);
    torch::Tensor permB   = torch::empty({allocSlots}, i32Opts);
    torch::Tensor keysTmp = torch::empty({allocSlots}, i32Opts);
    torch::Tensor keys64A = torch::empty({allocSlots}, i64Opts);
    torch::Tensor keys64B = torch::empty({allocSlots}, i64Opts);

    if (numSlots > 0) {
        iotaKernel<<<numBlocks(numSlots), kThreads, 0, stream>>>(u32(permA), numSlots);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Pass 1: sort by the 27-bit in-tile node key.
        callCub([&](void *temp, size_t &bytes) {
            C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(temp,
                                                                    bytes,
                                                                    em.nodeKey,
                                                                    u32(keysTmp),
                                                                    u32(permA),
                                                                    u32(permB),
                                                                    numSlots,
                                                                    numGrids,
                                                                    em.segOffsets,
                                                                    em.segOffsets + 1,
                                                                    0,
                                                                    27,
                                                                    stream));
        });

        // Pass 2: stable sort by the 64-bit tile key (invalid slots carry ~0 and sort last).
        gatherKeys64<<<numBlocks(numSlots), kThreads, 0, stream>>>(
            em.tileKey, u32(permB), u64(keys64A), numSlots);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        callCub([&](void *temp, size_t &bytes) {
            C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(temp,
                                                                    bytes,
                                                                    u64(keys64A),
                                                                    u64(keys64B),
                                                                    u32(permB),
                                                                    u32(permA),
                                                                    numSlots,
                                                                    numGrids,
                                                                    em.segOffsets,
                                                                    em.segOffsets + 1,
                                                                    0,
                                                                    64,
                                                                    stream));
        });
    }
    // From here on: permA = canonical-order permutation, keys64B = sorted tile keys.

    torch::Tensor sortedNodeKey = torch::empty({allocSlots}, i32Opts);
    torch::Tensor sortedOrigin  = torch::empty({allocSlots * 3}, i32Opts);
    torch::Tensor sortedMask    = torch::empty({allocSlots * 8}, i64Opts);

    SortedArrays sorted;
    sorted.tileKey = u64(keys64B);
    sorted.nodeKey = u32(sortedNodeKey);
    sorted.origin  = sortedOrigin.data_ptr<int32_t>();
    sorted.mask    = u64(sortedMask);

    // --- Dedup: head flags, global node ranks, per-grid offsets, parent linkage. ---
    torch::Tensor leafFlag  = torch::empty({allocSlots}, i32Opts);
    torch::Tensor lowerFlag = torch::empty({allocSlots}, i32Opts);
    torch::Tensor upperFlag = torch::empty({allocSlots}, i32Opts);
    torch::Tensor leafRank  = torch::empty({allocSlots}, i32Opts);
    torch::Tensor lowerRank = torch::empty({allocSlots}, i32Opts);
    torch::Tensor upperRank = torch::empty({allocSlots}, i32Opts);

    torch::Tensor leafHeadSlot  = torch::empty({allocSlots}, i32Opts);
    torch::Tensor leafParent    = torch::empty({allocSlots}, i32Opts);
    torch::Tensor lowerHeadSlot = torch::empty({allocSlots}, i32Opts);
    torch::Tensor lowerParent   = torch::empty({allocSlots}, i32Opts);
    torch::Tensor upperHeadSlot = torch::empty({allocSlots}, i32Opts);

    if (numSlots > 0) {
        gatherSorted<<<numBlocks(numSlots), kThreads, 0, stream>>>(em, u32(permA), sorted);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        computeHeadFlags<<<numBlocks(numSlots), kThreads, 0, stream>>>(sorted.tileKey,
                                                                       sorted.nodeKey,
                                                                       em.segOffsets,
                                                                       numGrids,
                                                                       numSlots,
                                                                       u32(leafFlag),
                                                                       u32(lowerFlag),
                                                                       u32(upperFlag));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const std::pair<torch::Tensor *, torch::Tensor *> scans[3] = {
            {&leafFlag, &leafRank}, {&lowerFlag, &lowerRank}, {&upperFlag, &upperRank}};
        for (const auto &[flags, ranks]: scans) {
            callCub([&](void *temp, size_t &bytes) {
                C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                    temp, bytes, u32(*flags), u32(*ranks), numSlots, stream));
            });
        }

        scatterNodeTables<<<numBlocks(numSlots), kThreads, 0, stream>>>(u32(leafFlag),
                                                                        u32(lowerFlag),
                                                                        u32(upperFlag),
                                                                        u32(leafRank),
                                                                        u32(lowerRank),
                                                                        u32(upperRank),
                                                                        numSlots,
                                                                        u32(leafHeadSlot),
                                                                        u32(leafParent),
                                                                        u32(lowerHeadSlot),
                                                                        u32(lowerParent),
                                                                        u32(upperHeadSlot));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (pass.op != TopologyPassSpec::Op::Refine) { // coarsen/dilate have duplicate producers
            combineDuplicateMasks<<<numBlocks(numSlots), kThreads, 0, stream>>>(sorted.tileKey,
                                                                                u32(leafFlag),
                                                                                u32(leafRank),
                                                                                u32(leafHeadSlot),
                                                                                numSlots,
                                                                                sorted.mask);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    torch::Tensor upperStart = torch::empty({numGrids + 1}, i32Opts);
    torch::Tensor lowerStart = torch::empty({numGrids + 1}, i32Opts);
    torch::Tensor leafStart  = torch::empty({numGrids + 1}, i32Opts);

    segmentNodeOffsets<<<numBlocks(numGrids + 1), kThreads, 0, stream>>>(u32(leafRank),
                                                                         u32(lowerRank),
                                                                         u32(upperRank),
                                                                         em.segOffsets,
                                                                         numGrids,
                                                                         u32(leafStart),
                                                                         u32(lowerStart),
                                                                         u32(upperStart));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- THE synchronization: read back per-grid node offsets, size and allocate the output. ---
    std::vector<uint32_t> leafStartHost(numGrids + 1), lowerStartHost(numGrids + 1),
        upperStartHost(numGrids + 1);
    C10_CUDA_CHECK(cudaMemcpyAsync(leafStartHost.data(),
                                   leafStart.data_ptr<int32_t>(),
                                   sizeof(uint32_t) * (numGrids + 1),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(lowerStartHost.data(),
                                   lowerStart.data_ptr<int32_t>(),
                                   sizeof(uint32_t) * (numGrids + 1),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(upperStartHost.data(),
                                   upperStart.data_ptr<int32_t>(),
                                   sizeof(uint32_t) * (numGrids + 1),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    BatchedTopologyResult result;
    result.gridByteOffsets.resize(numGrids + 1);
    result.leafCounts.resize(numGrids);
    uint64_t totalBytes = 0;
    for (int32_t g = 0; g < numGrids; ++g) {
        result.gridByteOffsets[g] = totalBytes;
        const uint32_t numUpper   = upperStartHost[g + 1] - upperStartHost[g];
        const uint32_t numLower   = lowerStartHost[g + 1] - lowerStartHost[g];
        const uint32_t numLeaf    = leafStartHost[g + 1] - leafStartHost[g];
        result.leafCounts[g]      = numLeaf;
        totalBytes += gridByteSize(numUpper, numLower, numLeaf);
    }
    result.gridByteOffsets[numGrids] = totalBytes;

    const int32_t totalLeaf  = int32_t(leafStartHost[numGrids]);
    const int32_t totalLower = int32_t(lowerStartHost[numGrids]);
    const int32_t totalUpper = int32_t(upperStartHost[numGrids]);

    result.buffer = TorchDeviceBuffer(totalBytes, src.device);
    C10_CUDA_CHECK(cudaMemsetAsync(result.buffer.deviceData(), 0, totalBytes, stream));

    torch::Tensor gridOffDev = torch::empty({numGrids + 1}, i64Opts);
    C10_CUDA_CHECK(cudaMemcpyAsync(gridOffDev.data_ptr<int64_t>(),
                                   result.gridByteOffsets.data(),
                                   sizeof(uint64_t) * (numGrids + 1),
                                   cudaMemcpyHostToDevice,
                                   stream));

    BuildDeviceArrays build;
    build.dstBase         = result.buffer.deviceData();
    build.gridByteOffsets = u64(gridOffDev);
    build.upperStart      = u32(upperStart);
    build.lowerStart      = u32(lowerStart);
    build.leafStart       = u32(leafStart);

    // --- Build all grids. ---
    constexpr int32_t kGridDataWords = int32_t(sizeof(nanovdb::GridData) / sizeof(uint64_t));
    copyGridData<<<numBlocks(numGrids * kGridDataWords), kThreads, 0, stream>>>(
        em.srcGrids, build, numGrids);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    initGridTreeRoot<<<numBlocks(numGrids), kThreads, 0, stream>>>(build, numGrids);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    torch::Tensor voxelCounts = torch::zeros({int64_t(totalLeaf) + 1}, i64Opts);

    if (totalLeaf > 0) {
        buildUpperNodes<<<numBlocks(totalUpper), kThreads, 0, stream>>>(
            build, sorted, u32(upperHeadSlot), numGrids, totalUpper);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        buildLowerNodes<<<numBlocks(totalLower), kThreads, 0, stream>>>(
            build, sorted, u32(lowerHeadSlot), u32(lowerParent), numGrids, totalLower);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        buildLeafNodes<<<numBlocks(totalLeaf), kThreads, 0, stream>>>(build,
                                                                      sorted,
                                                                      u32(leafHeadSlot),
                                                                      u32(leafParent),
                                                                      numGrids,
                                                                      totalLeaf,
                                                                      u64(voxelCounts));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // In-place inclusive scan over elements [1, totalLeaf]; element 0 stays 0.
        callCub([&](void *temp, size_t &bytes) {
            C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                temp, bytes, u64(voxelCounts) + 1, u64(voxelCounts) + 1, totalLeaf, stream));
        });

        setLeafOffsets<<<numBlocks(totalLeaf), kThreads, 0, stream>>>(
            build, u64(voxelCounts), numGrids, totalLeaf);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        propagateLeafBBox<<<numBlocks(totalLeaf), kThreads, 0, stream>>>(
            build, u32(leafParent), numGrids, totalLeaf);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        propagateLowerBBox<<<numBlocks(totalLower), kThreads, 0, stream>>>(
            build, u32(lowerParent), numGrids, totalLower);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        propagateUpperBBox<<<numBlocks(totalUpper), kThreads, 0, stream>>>(
            build, numGrids, totalUpper);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    finalizeGrids<<<numBlocks(numGrids), kThreads, 0, stream>>>(build, u64(voxelCounts), numGrids);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

/// Runs a chained sequence of batched passes over the batch and wraps the final buffer in a
/// GridHandle. `passes` must be non-empty.
inline nanovdb::GridHandle<TorchDeviceBuffer>
batchedTopologyHandle(const GridBatchData &batch,
                      const std::vector<TopologyPassSpec> &passes,
                      cudaStream_t stream) {
    TORCH_CHECK(!passes.empty(), "batchedTopologyHandle requires at least one pass");
    BatchedTopologyResult result =
        runBatchedTopologyPass(sourceFromGridBatch(batch), passes[0], stream);
    for (size_t p = 1; p < passes.size(); ++p) {
        result =
            runBatchedTopologyPass(sourceFromResult(result, batch.device()), passes[p], stream);
    }
    return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(result.buffer));
}

} // namespace batched
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
