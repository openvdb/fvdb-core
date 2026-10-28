// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Batched topology passes: builds ALL grids of a batch in one pass of NanoVDB's batch-capable
// tools::cuda::TopologyBuilder.
//
// NanoVDB's morphology builders (RefineGrid, CoarsenGrid, DilateGrid) build one grid per call, with
// several stream synchronizations each, and fvdb would then lay the single-grid results end to end.
// For generated-topology workloads that rebuild grids every training iteration (issue #755), that
// per-member overhead dominates. This header drives the builder over the whole batch instead:
//
//   roots   one kernel gathers every source grid's upper-node bounding boxes, read back with one
//           host wait; the host derives each output grid's speculative root tiles from them the way
//           NanoVDB's single-grid consumers do;
//   masks   one launch over all source leaves fills the builder's densified child masks, through
//           NanoVDB's Refine/Coarsen internal-node functors or fvdb's BoxDilate marking;
//   build   the builder counts nodes (its one host wait), allocates one buffer, and writes every
//           node; a second launch over all source leaves writes the leaf value masks.
//
// The builder densifies 32^3 lower-node masks (about 16.5 MiB of scratch) per speculative root
// tile, so a pass splits the batch into chunks of consecutive grids holding at most
// FVDB_TOPOLOGY_MAX_TILES tiles (default 64). The chunks of one pass feed the next pass where they
// lie; the last pass's chunks are assembled into one storage, or adopted directly when there is
// one.
//
// Scratch is allocated through BuilderResource on the caller's stream. The builder is cached per
// stream, so its pinned staging is allocated once rather than per pass. Checksums are disabled on
// the output, matching assembleGridStorage (compaction, concatenation, merge).

#ifndef FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH

#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Morphology.cuh>

#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fvdb {
namespace detail {
namespace batched {

using BuildT = nanovdb::ValueOnIndex;
using GridT  = nanovdb::NanoGrid<BuildT>;
using RootT  = nanovdb::NanoRoot<BuildT>;
using UpperT = nanovdb::NanoUpper<BuildT>;
using LowerT = nanovdb::NanoLower<BuildT>;
using LeafT  = nanovdb::NanoLeaf<BuildT>;

using ProcessedTileMap = nanovdb::tools::cuda::topology::detail::ProcessedTileMap<RootT>;
using TopologyBuilderT = nanovdb::tools::cuda::TopologyBuilder<BuildT, BuilderResource>;

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
/// GridBatchData buffer -- including sliced/non-contiguous views -- or the output chunks of a
/// previous pass); only per-member device grid pointers, leaf counts, and a bound on each member's
/// upper-node count are needed.
struct BatchedTopologySource {
    std::vector<const GridT *> grids;   // per-member device grid pointers (host-side vector)
    std::vector<int64_t> leafCounts;    // per-member leaf counts (host-side)
    std::vector<int64_t> upperCapacity; // per-member upper bound on upper nodes (host-side)
    torch::Device device{torch::kCUDA};
};

/// Result of one batched pass: the output grids in one or more chunk buffers, in batch order, plus
/// the host-side layout needed to wrap them in GridStorage or feed the next pass. Each buffer is
/// grid storage proper (allocated through TorchDeviceResource, retaining the pass's stream), so a
/// single-chunk result becomes the batch's storage without another copy.
struct BatchedTopologyResult {
    std::vector<DeviceGridBuffer> chunks; // output buffers; each holds its grids back to back
    std::vector<const GridT *> grids;     // per-member output grid pointers, into `chunks`
    std::vector<uint64_t> gridBytes;      // per-member output byte sizes
    std::vector<int64_t> leafCounts;      // per-member output leaf counts
    std::vector<int64_t> upperCounts;     // per-member output upper-node counts
};

// ---------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------

/// Per-chunk device tables, indexed by member within the chunk.
struct PassTables {
    const GridT *const *src;  // source grids
    GridT *const *dst;        // output grids (set once the output buffer exists)
    const RootT *const *root; // processed roots
    const int64_t *tileBase;  // first processed tile of each grid in the builder's mask arrays
    const int64_t *leafStart; // [numGrids+1] first source leaf of each grid within the chunk
    int32_t numGrids;
};

/// The grid owning chunk-wide source leaf `leaf`: the last grid whose first leaf does not exceed
/// it. Empty grids share their successor's first leaf and are skipped that way.
__device__ inline int32_t
gridOfLeaf(const PassTables &t, int64_t leaf) {
    int32_t lo = 0, hi = t.numGrids;
    while (lo < hi) {
        const int32_t mid = (lo + hi) >> 1;
        if (t.leafStart[mid] <= leaf) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
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

/// The voxels of source leaf mask `src` that land in the target leaf at offset 8*db under a
/// Minkowski sum with the box [boxLo, boxHi]: U_{o in box} shift(src, o - 8*db). The sum
/// factorizes per axis; shifts with any component of magnitude >= 8 vanish, so per axis db==0
/// keeps every box offset and db==+-1 keeps only o==+-1. Returns whether any voxel lands.
__device__ inline bool
boxDilatedContribution(const uint64_t *src,
                       const nanovdb::Coord &boxLo,
                       const nanovdb::Coord &boxHi,
                       const nanovdb::Coord &db,
                       uint64_t out[8]) {
    // z-stage, then y-stage (per word), then x-stage (word permutation).
    uint64_t wz[8], wy[8];
    for (int i = 0; i < 8; ++i) {
        uint64_t acc = 0;
        for (int oz = boxLo[2]; oz <= boxHi[2]; ++oz) {
            const int sz = oz - 8 * db[2];
            if (sz > -8 && sz < 8) {
                acc |= shiftWordZ(src[i], sz);
            }
        }
        wz[i] = acc;
    }
    for (int i = 0; i < 8; ++i) {
        uint64_t acc = 0;
        for (int oy = boxLo[1]; oy <= boxHi[1]; ++oy) {
            const int sy = oy - 8 * db[1];
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
            const int from = x - (ox - 8 * db[0]); // out[x] gathers wy[x - shift]
            if (from >= 0 && from < 8) {
                acc |= wy[from];
            }
        }
        out[x] = acc;
        occupied |= acc;
    }
    return occupied != 0;
}

/// The target-leaf offsets a BoxDilate pass with [boxLo, boxHi] reaches: per axis, -1 when the box
/// extends below 0, 0, and +1 when it extends above 0. A BoxDilate kernel runs one work item per
/// (source leaf, offset) pair rather than looping over offsets per thread; a per-thread loop with
/// these runtime bounds makes each including file take about 20 minutes to compile.
struct BoxNeighbors {
    int lo[3], count[3];

    __hostdev__
    BoxNeighbors(const nanovdb::Coord &boxLo, const nanovdb::Coord &boxHi) {
        for (int a = 0; a < 3; ++a) {
            lo[a]    = boxLo[a] < 0 ? -1 : 0;
            count[a] = (boxHi[a] > 0 ? 1 : 0) - lo[a] + 1;
        }
    }

    __hostdev__ int
    size() const {
        return count[0] * count[1] * count[2];
    }

    /// Offset `i` of size(), z fastest.
    __hostdev__ nanovdb::Coord
    operator[](int i) const {
        const int z = i % count[2];
        i /= count[2];
        const int y = i % count[1];
        return nanovdb::Coord(lo[0] + i / count[1], lo[1] + y, lo[2] + z);
    }
};

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

/// One thread per source grid: its upper-node count, and the bounding box of each upper node at
/// `boxes + 6 * upperStart[g]` (min xyz, then max xyz).
static __global__ void
gatherUpperBBoxes(const GridT *const *grids,
                  const int64_t *upperStart,
                  int32_t numGrids,
                  int32_t *counts,
                  int32_t *boxes) {
    const int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGrids) {
        return;
    }
    const RootT &root      = grids[g]->tree().root();
    const int64_t capacity = upperStart[g + 1] - upperStart[g];
    int32_t n              = 0;
    for (uint32_t t = 0; t < root.tileCount(); ++t) {
        const auto *tile = root.tile(t);
        if (!tile->isChild()) {
            continue;
        }
        if (n < capacity) {
            const nanovdb::CoordBBox b = root.getChild(tile)->bbox();
            int32_t *out               = boxes + 6 * (upperStart[g] + n);
            for (int a = 0; a < 3; ++a) {
                out[a]     = b.min()[a];
                out[3 + a] = b.max()[a];
            }
        }
        ++n;
    }
    counts[g] = n;
}

/// The source leaf of work item `item`, and for BoxDilate its target-leaf offset `db`: BoxDilate
/// runs one item per (source leaf, offset) pair, every other op one item per source leaf.
template <TopologyPassSpec::Op Op>
__device__ inline int64_t
leafOfItem(int64_t item, const BoxNeighbors &neighbors, nanovdb::Coord &db) {
    if constexpr (Op == TopologyPassSpec::Op::BoxDilate) {
        db = neighbors[int(item % neighbors.size())];
        return item / neighbors.size();
    }
    return item;
}

/// One work item per source leaf of the chunk (BoxDilate: per source leaf and target offset):
/// marks the upper- and lower-node children it produces in its grid's densified masks.
template <TopologyPassSpec::Op Op>
static __global__ void
fillInternalMasks(PassTables t,
                  int64_t numItems,
                  nanovdb::Coord boxLo,
                  nanovdb::Coord boxHi,
                  nanovdb::Mask<5> *upperMasks,
                  nanovdb::Mask<4> (*lowerMasks)[nanovdb::Mask<5>::SIZE]) {
    namespace morph = nanovdb::util::morphology::cuda;
    const BoxNeighbors neighbors(boxLo, boxHi);
    for (int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; item < numItems;
         item += int64_t(gridDim.x) * blockDim.x) {
        nanovdb::Coord db;
        const int64_t leaf  = leafOfItem<Op>(item, neighbors, db);
        const int32_t g     = gridOfLeaf(t, leaf);
        const int64_t local = leaf - t.leafStart[g];
        auto *upper         = upperMasks + t.tileBase[g];
        auto *lower         = lowerMasks + t.tileBase[g];
        if constexpr (Op == TopologyPassSpec::Op::Refine) {
            morph::RefineInternalNodesFunctor<BuildT>()(local, t.src[g], t.root[g], upper, lower);
        } else if constexpr (Op == TopologyPassSpec::Op::Coarsen) {
            morph::CoarsenInternalNodesFunctor<BuildT>()(local, t.src[g], t.root[g], upper, lower);
        } else {
            const LeafT &srcLeaf = t.src[g]->tree().template getFirstNode<0>()[local];
            uint64_t contribution[8];
            if (!boxDilatedContribution(
                    srcLeaf.valueMask().words(), boxLo, boxHi, db, contribution)) {
                continue;
            }
            const RootT *root           = t.root[g];
            const nanovdb::Coord origin = srcLeaf.origin() + (db << 3);
            const uint32_t upperChild   = UpperT::CoordToOffset(origin);
            const int64_t tileIndex     = root->probeTile(origin) - root->tile(0);
            upper[tileIndex].setOnAtomic(upperChild);
            lower[tileIndex][upperChild].setOnAtomic(LowerT::CoordToOffset(origin));
        }
    }
}

/// One work item per source leaf of the chunk (BoxDilate: per source leaf and target offset):
/// writes, or ORs in, its voxels in the output grid's leaf value masks.
template <TopologyPassSpec::Op Op>
static __global__ void
fillLeafMasks(PassTables t, int64_t numItems, nanovdb::Coord boxLo, nanovdb::Coord boxHi) {
    namespace morph = nanovdb::util::morphology::cuda;
    const BoxNeighbors neighbors(boxLo, boxHi);
    for (int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; item < numItems;
         item += int64_t(gridDim.x) * blockDim.x) {
        nanovdb::Coord db;
        const int64_t leaf  = leafOfItem<Op>(item, neighbors, db);
        const int32_t g     = gridOfLeaf(t, leaf);
        const int64_t local = leaf - t.leafStart[g];
        if constexpr (Op == TopologyPassSpec::Op::Refine) {
            morph::RefineLeafMasksFunctor<BuildT>()(local, t.src[g], t.dst[g]);
        } else if constexpr (Op == TopologyPassSpec::Op::Coarsen) {
            morph::CoarsenLeafMasksFunctor<BuildT>()(local, t.src[g], t.dst[g]);
        } else {
            const LeafT &srcLeaf = t.src[g]->tree().template getFirstNode<0>()[local];
            uint64_t contribution[8];
            if (!boxDilatedContribution(
                    srcLeaf.valueMask().words(), boxLo, boxHi, db, contribution)) {
                continue;
            }
            const LeafT *dstLeaf = t.dst[g]->tree().root().probeLeaf(srcLeaf.origin() + (db << 3));
            auto &dstMask        = const_cast<nanovdb::Mask<3> &>(dstLeaf->valueMask());
            for (int w = 0; w < 8; ++w) {
                if (contribution[w]) {
                    nanovdb::util::atomicOr(&dstMask.words()[w], contribution[w]);
                }
            }
        }
    }
}

/// One block per grid: copies the source GridData header to the output grid, as NanoVDB's
/// single-grid consumers do before TopologyBuilder::processGridTreeRoot.
static __global__ void
copyGridHeaders(PassTables t) {
    static_assert(sizeof(nanovdb::GridData) % sizeof(uint64_t) == 0);
    constexpr uint32_t kWords = sizeof(nanovdb::GridData) / sizeof(uint64_t);
    const int32_t g           = blockIdx.x;
    const auto *src           = reinterpret_cast<const uint64_t *>(t.src[g]);
    auto *dst                 = reinterpret_cast<uint64_t *>(t.dst[g]);
    for (uint32_t w = threadIdx.x; w < kWords; w += blockDim.x) {
        dst[w] = src[w];
    }
}

// ---------------------------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------------------------

/// The builder for `stream`, created on first use. It is never destroyed, so no device memory is
/// released after CUDA shuts down, and its pinned host staging is allocated once, not per pass.
inline TopologyBuilderT &
cachedTopologyBuilder(cudaStream_t stream) {
    thread_local auto *builders =
        new std::unordered_map<cudaStream_t, std::unique_ptr<TopologyBuilderT>>();
    auto &builder = (*builders)[stream];
    if (!builder) {
        builder = std::make_unique<TopologyBuilderT>(stream);
    }
    return *builder;
}

/// FVDB_TOPOLOGY_MAX_TILES: the most speculative root tiles one builder run takes (default 64,
/// about 1 GiB of densified masks). A grid needing more runs alone.
inline int64_t
maxTilesPerChunk() {
    const char *env = std::getenv("FVDB_TOPOLOGY_MAX_TILES");
    const int64_t n = env != nullptr ? std::atoll(env) : 0;
    return n > 0 ? n : 64;
}

/// Root tiles a grid with index bounding box `bbox` can occupy: a bound on its upper nodes.
inline int64_t
rootTilesOverlapping(const nanovdb::CoordBBox &bbox) {
    if (bbox.empty()) {
        return 0;
    }
    int64_t n = 1;
    for (int a = 0; a < 3; ++a) {
        n *= int64_t(bbox.max()[a] >> 12) - int64_t(bbox.min()[a] >> 12) + 1;
    }
    return n;
}

/// Uploads a small host table to the device on `stream`. Pageable H2D copies are synchronous with
/// the host but do not synchronize the stream.
inline torch::Tensor
uploadTable(const std::vector<int64_t> &host, const torch::Device &device, cudaStream_t stream) {
    torch::Tensor table = torch::empty({std::max<int64_t>(int64_t(host.size()), 1)},
                                       torch::dtype(torch::kInt64).device(device));
    if (!host.empty()) {
        C10_CUDA_CHECK(cudaMemcpyAsync(table.data_ptr<int64_t>(),
                                       host.data(),
                                       sizeof(int64_t) * host.size(),
                                       cudaMemcpyHostToDevice,
                                       stream));
    }
    return table;
}

template <typename T>
inline int64_t
tableEntry(T *ptr) {
    static_assert(sizeof(T *) == sizeof(int64_t));
    return int64_t(reinterpret_cast<uintptr_t>(ptr));
}

inline BatchedTopologySource
sourceFromGridBatch(const GridBatchData &batch) {
    BatchedTopologySource src;
    src.device              = batch.device();
    const int64_t batchSize = batch.batchSize();
    src.grids.reserve(batchSize);
    src.leafCounts.reserve(batchSize);
    src.upperCapacity.reserve(batchSize);
    for (int64_t i = 0; i < batchSize; ++i) {
        const int64_t leaves = batch.numLeavesAt(i);
        src.grids.push_back(batch.deviceGridPtrAt(i));
        src.leafCounts.push_back(leaves);
        src.upperCapacity.push_back(leaves > 0 ? rootTilesOverlapping(batch.bboxAt(i)) : 0);
    }
    return src;
}

inline BatchedTopologySource
sourceFromResult(const BatchedTopologyResult &result, const torch::Device &device) {
    BatchedTopologySource src;
    src.device        = device;
    src.grids         = result.grids;
    src.leafCounts    = result.leafCounts;
    src.upperCapacity = result.upperCounts;
    return src;
}

/// Each source grid's upper-node bounding boxes, read back with one host wait.
inline std::vector<std::vector<nanovdb::CoordBBox>>
readUpperBBoxes(const BatchedTopologySource &src, cudaStream_t stream) {
    const int32_t numGrids = int32_t(src.grids.size());
    std::vector<int64_t> upperStart(numGrids + 1, 0), gridTable(numGrids);
    for (int32_t g = 0; g < numGrids; ++g) {
        upperStart[g + 1] = upperStart[g] + src.upperCapacity[g];
        gridTable[g]      = tableEntry(src.grids[g]);
    }
    torch::Tensor gridsDev      = uploadTable(gridTable, src.device, stream);
    torch::Tensor upperStartDev = uploadTable(upperStart, src.device, stream);

    // One int32 block: the per-grid counts, then 6 ints per upper-node bbox.
    const int64_t words = numGrids + 6 * upperStart[numGrids];
    torch::Tensor dev   = torch::empty({words}, torch::dtype(torch::kInt32).device(src.device));
    torch::Tensor host  = torch::empty({words}, torch::dtype(torch::kInt32).pinned_memory(true));
    gatherUpperBBoxes<<<(numGrids + 127) / 128, 128, 0, stream>>>(
        reinterpret_cast<const GridT *const *>(gridsDev.data_ptr<int64_t>()),
        upperStartDev.data_ptr<int64_t>(),
        numGrids,
        dev.data_ptr<int32_t>(),
        dev.data_ptr<int32_t>() + numGrids);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemcpyAsync(host.data_ptr<int32_t>(),
                                   dev.data_ptr<int32_t>(),
                                   sizeof(int32_t) * words,
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t *counts = host.data_ptr<int32_t>();
    const int32_t *boxes  = counts + numGrids;
    std::vector<std::vector<nanovdb::CoordBBox>> uppers(numGrids);
    for (int32_t g = 0; g < numGrids; ++g) {
        TORCH_CHECK(counts[g] <= src.upperCapacity[g],
                    "batched topology pass: grid ",
                    g,
                    " has ",
                    counts[g],
                    " upper nodes, more than its bound of ",
                    src.upperCapacity[g]);
        for (int32_t u = 0; u < counts[g]; ++u) {
            const int32_t *b = boxes + 6 * (upperStart[g] + u);
            uppers[g].emplace_back(nanovdb::Coord(b[0], b[1], b[2]),
                                   nanovdb::Coord(b[3], b[4], b[5]));
        }
    }
    return uppers;
}

/// The speculative root tiles of one output grid, from its source grid's upper-node bounding
/// boxes, as NanoVDB's RefineGrid, CoarsenGrid and DilateGrid derive them.
inline ProcessedTileMap
processedTiles(const std::vector<nanovdb::CoordBBox> &uppers, const TopologyPassSpec &pass) {
    namespace topo = nanovdb::tools::cuda::topology::detail;
    using nanovdb::util::morphology::coarsenCoord;
    using nanovdb::util::morphology::refineCoord;
    ProcessedTileMap tiles;
    for (const nanovdb::CoordBBox &b: uppers) {
        if (b.empty()) {
            continue;
        }
        switch (pass.op) {
        case TopologyPassSpec::Op::Refine:
            topo::insertProcessedTiles<RootT>(
                tiles, nanovdb::CoordBBox(refineCoord(b.min()), refineCoord(b.max()).offsetBy(1)));
            break;
        case TopologyPassSpec::Op::Coarsen:
            topo::insertProcessedTile<RootT>(tiles, coarsenCoord((b.min() >> 12) << 12));
            break;
        case TopologyPassSpec::Op::BoxDilate:
            topo::insertProcessedTiles<RootT>(
                tiles, nanovdb::CoordBBox(b.min() + pass.boxLo, b.max() + pass.boxHi));
            break;
        }
    }
    return tiles;
}

constexpr int32_t kThreads = 256;

/// Work items of a mask-fill launch over `numLeaves` source leaves (see leafOfItem).
inline int64_t
workItems(int64_t numLeaves, const TopologyPassSpec &pass) {
    return pass.op == TopologyPassSpec::Op::BoxDilate
               ? numLeaves * BoxNeighbors(pass.boxLo, pass.boxHi).size()
               : numLeaves;
}

inline unsigned
itemBlocks(int64_t numItems) {
    return unsigned(std::min<int64_t>((numItems + kThreads - 1) / kThreads, int64_t(1) << 20));
}

template <TopologyPassSpec::Op Op>
inline void
launchFillInternalMasks(const PassTables &t,
                        int64_t numLeaves,
                        const TopologyPassSpec &pass,
                        TopologyBuilderT &builder,
                        cudaStream_t stream) {
    const int64_t numItems = workItems(numLeaves, pass);
    fillInternalMasks<Op>
        <<<itemBlocks(numItems), kThreads, 0, stream>>>(t,
                                                        numItems,
                                                        pass.boxLo,
                                                        pass.boxHi,
                                                        builder.deviceUpperMasks(),
                                                        builder.deviceLowerMasks());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <TopologyPassSpec::Op Op>
inline void
launchFillLeafMasks(const PassTables &t,
                    int64_t numLeaves,
                    const TopologyPassSpec &pass,
                    cudaStream_t stream) {
    const int64_t numItems = workItems(numLeaves, pass);
    fillLeafMasks<Op>
        <<<itemBlocks(numItems), kThreads, 0, stream>>>(t, numItems, pass.boxLo, pass.boxHi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Builds grids [first, last) of `src` in one TopologyBuilder run and appends them to `result`.
inline void
buildChunk(const BatchedTopologySource &src,
           const TopologyPassSpec &pass,
           int32_t first,
           int32_t last,
           const std::vector<ProcessedTileMap> &tiles,
           BatchedTopologyResult &result,
           cudaStream_t stream) {
    namespace topo            = nanovdb::tools::cuda::topology::detail;
    using Op                  = TopologyPassSpec::Op;
    const int32_t n           = last - first;
    TopologyBuilderT &builder = cachedTopologyBuilder(stream);

    // 1. Speculative roots.
    std::vector<uint32_t> tileCounts(n);
    for (int32_t i = 0; i < n; ++i) {
        tileCounts[i] = uint32_t(tiles[first + i].size());
    }
    builder.allocateProcessedRoots(tileCounts);
    for (int32_t i = 0; i < n; ++i) {
        topo::packProcessedRoot(tiles[first + i], builder.hostProcessedRoot(i));
    }
    builder.uploadProcessedRoot(stream);
    builder.allocateInternalMaskBuffers(stream);

    // 2. Densified child masks, one launch over the chunk's source leaves.
    std::vector<int64_t> srcTable(n), rootTable(n), tileBase(n), leafStart(n + 1, 0);
    for (int32_t i = 0; i < n; ++i) {
        srcTable[i]      = tableEntry(src.grids[first + i]);
        rootTable[i]     = tableEntry(builder.deviceProcessedRoot(i));
        tileBase[i]      = builder.data(i)->tileBase;
        leafStart[i + 1] = leafStart[i] + src.leafCounts[first + i];
    }
    const int64_t numLeaves = leafStart[n];
    torch::Tensor srcDev    = uploadTable(srcTable, src.device, stream);
    torch::Tensor rootDev   = uploadTable(rootTable, src.device, stream);
    torch::Tensor tileDev   = uploadTable(tileBase, src.device, stream);
    torch::Tensor leafDev   = uploadTable(leafStart, src.device, stream);
    PassTables t;
    t.src       = reinterpret_cast<const GridT *const *>(srcDev.data_ptr<int64_t>());
    t.dst       = nullptr;
    t.root      = reinterpret_cast<const RootT *const *>(rootDev.data_ptr<int64_t>());
    t.tileBase  = tileDev.data_ptr<int64_t>();
    t.leafStart = leafDev.data_ptr<int64_t>();
    t.numGrids  = n;
    if (numLeaves > 0) {
        switch (pass.op) {
        case Op::Refine:
            launchFillInternalMasks<Op::Refine>(t, numLeaves, pass, builder, stream);
            break;
        case Op::Coarsen:
            launchFillInternalMasks<Op::Coarsen>(t, numLeaves, pass, builder, stream);
            break;
        case Op::BoxDilate:
            launchFillInternalMasks<Op::BoxDilate>(t, numLeaves, pass, builder, stream);
            break;
        }
    }

    // 3. Node counts (the builder's host wait), the output buffer, and every node but the leaf
    //    value masks.
    builder.countNodes(stream);
    const DeviceGridBuffer proto(stream, TorchDeviceResource(src.device), 0, nanovdb::cuda::noInit);
    DeviceGridBuffer buffer = builder.getBuffer(proto, stream);
    std::vector<int64_t> dstTable(n);
    for (int32_t i = 0; i < n; ++i) {
        GridT *dst  = &builder.data(i)->getGrid();
        dstTable[i] = tableEntry(dst);
        result.grids.push_back(dst);
        result.gridBytes.push_back(builder.data(i)->size);
        result.leafCounts.push_back(builder.data(i)->nodeCount[0]);
        result.upperCounts.push_back(builder.data(i)->nodeCount[2]);
    }
    torch::Tensor dstDev = uploadTable(dstTable, src.device, stream);
    t.dst                = reinterpret_cast<GridT *const *>(dstDev.data_ptr<int64_t>());
    copyGridHeaders<<<n, 128, 0, stream>>>(t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    builder.processGridTreeRoot(stream);
    builder.processUpperNodes(stream);
    builder.processLowerNodes(stream);

    // 4. Leaf value masks, one launch over the chunk's source leaves.
    if (numLeaves > 0) {
        switch (pass.op) {
        case Op::Refine: launchFillLeafMasks<Op::Refine>(t, numLeaves, pass, stream); break;
        case Op::Coarsen: launchFillLeafMasks<Op::Coarsen>(t, numLeaves, pass, stream); break;
        case Op::BoxDilate: launchFillLeafMasks<Op::BoxDilate>(t, numLeaves, pass, stream); break;
        }
    }
    builder.processLeafOffsets(stream);
    builder.processBBox(stream);
    builder.postProcessGridTree(stream);
    result.chunks.push_back(std::move(buffer));
}

/// Runs one batched topology pass over all grids of `src`: one host wait for the source roots,
/// then one per chunk for its node counts.
inline BatchedTopologyResult
runBatchedTopologyPass(const BatchedTopologySource &src,
                       const TopologyPassSpec &pass,
                       cudaStream_t stream) {
    TORCH_CHECK(src.device.is_cuda(), "batched topology passes require a CUDA device");
    const int32_t numGrids = int32_t(src.grids.size());
    TORCH_CHECK(numGrids > 0, "batched topology pass requires at least one grid");
    if (pass.op == TopologyPassSpec::Op::BoxDilate) {
        for (int a = 0; a < 3; ++a) {
            TORCH_CHECK(pass.boxLo[a] >= -1 && pass.boxLo[a] <= 0 && pass.boxHi[a] >= 0 &&
                            pass.boxHi[a] <= 1,
                        "BoxDilate pass components must be in {-1,0} / {0,1}");
        }
    }

    const auto uppers = readUpperBBoxes(src, stream);
    std::vector<ProcessedTileMap> tiles(numGrids);
    for (int32_t g = 0; g < numGrids; ++g) {
        tiles[g] = processedTiles(uppers[g], pass);
    }

    // Chunks of consecutive grids, each holding at most maxTilesPerChunk() tiles.
    BatchedTopologyResult result;
    const int64_t maxTiles = maxTilesPerChunk();
    int32_t first          = 0;
    int64_t chunkTiles     = 0;
    for (int32_t g = 0; g < numGrids; ++g) {
        const int64_t gridTiles = int64_t(tiles[g].size());
        if (g > first && chunkTiles + gridTiles > maxTiles) {
            buildChunk(src, pass, first, g, tiles, result, stream);
            first      = g;
            chunkTiles = 0;
        }
        chunkTiles += gridTiles;
    }
    buildChunk(src, pass, first, numGrids, tiles, result, stream);
    return result;
}

/// Runs a chained sequence of batched passes over the batch and wraps the result as the batch's
/// storage. `passes` must be non-empty. A single-chunk result is adopted with the layout the
/// builder computed (each header carries its gridIndex/gridCount), so its kernels may still be in
/// flight on `stream`, which the storage retains; several chunks are assembled into one storage.
inline GridStorage
batchedTopologyStorage(const GridBatchData &batch,
                       const std::vector<TopologyPassSpec> &passes,
                       cudaStream_t stream) {
    TORCH_CHECK(!passes.empty(), "batchedTopologyStorage requires at least one pass");
    BatchedTopologyResult result =
        runBatchedTopologyPass(sourceFromGridBatch(batch), passes[0], stream);
    for (size_t p = 1; p < passes.size(); ++p) {
        result =
            runBatchedTopologyPass(sourceFromResult(result, batch.device()), passes[p], stream);
    }

    const size_t numGrids = result.grids.size();
    if (result.chunks.size() == 1) {
        std::vector<nanovdb::GridHandleMetaData> meta;
        meta.reserve(numGrids);
        uint64_t offset = 0;
        for (size_t g = 0; g < numGrids; ++g) {
            meta.push_back(nanovdb::GridHandleMetaData{
                offset, result.gridBytes[g], nanovdb::GridType::OnIndex});
            offset += result.gridBytes[g];
        }
        return GridStorage(
            makeDeviceHandleFromLayout(std::move(result.chunks.front()), std::move(meta)),
            batch.device());
    }

    GridSpanSource source{batch.device(), stream, {}};
    source.spans.reserve(numGrids);
    for (size_t g = 0; g < numGrids; ++g) {
        source.spans.push_back(
            GridSpan{result.grids[g], result.gridBytes[g], nanovdb::GridType::OnIndex});
    }
    return assembleGridStorage({source}, batch.device(), stream);
}

} // namespace batched
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
