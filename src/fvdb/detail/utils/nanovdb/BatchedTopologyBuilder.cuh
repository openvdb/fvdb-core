// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Batched topology passes: builds all grids of a batch per pass with NanoVDB's batch-capable
// tools::cuda::TopologyBuilder, in as few builder runs as the root-tile cap below allows.
//
// NanoVDB's morphology builders (RefineGrid, CoarsenGrid, DilateGrid, PruneGrid) build one grid per
// call, with several stream synchronizations each, and fvdb would then lay the single-grid results
// end to end.
// For generated-topology workloads that rebuild grids every training iteration (issue #755), that
// per-member overhead dominates. This header drives the builder over the whole batch instead:
//
//   roots   one kernel gathers every source grid's upper-node bounding boxes, read back with one
//           host wait; the host derives each output grid's speculative root tiles from them the way
//           NanoVDB's single-grid consumers do;
//   masks   per builder run, one launch over the run's source leaves fills the builder's densified
//           child masks, through NanoVDB's Refine/Coarsen/Prune internal-node functors or fvdb's
//           BoxDilate marking;
//   build   the builder waits for its node counts (and, from a pass's second run on, for its pinned
//           staging), allocates the run's buffer, and writes every node; a second launch over the
//           run's source leaves writes the leaf value masks.
//
// The builder densifies 32^3 lower-node masks (about 16.5 MiB of scratch) per speculative root
// tile, so a pass splits the batch into chunks of consecutive grids holding at most
// FVDB_TOPOLOGY_MAX_TILES tiles (default 64). The chunks of one pass feed the next pass where they
// lie; the last pass's chunks are assembled into one storage, or adopted directly when there is
// one.
//
// A pass runs on the device's current torch stream. The builder's scratch is allocated through
// BuilderResource and the pass's small device tables as torch tensors, both on that stream. The
// builder is cached per (device, stream), so its pinned staging is allocated once rather than per
// pass; see cachedTopologyBuilder for what it keeps between passes. Checksums are disabled on
// the output, matching assembleGridStorage (compaction, concatenation, merge).

#ifndef FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH

#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceResource.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Morphology.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
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

using LeafMaskT                   = nanovdb::Mask<LeafT::LOG2DIM>;
constexpr uint32_t kLeafMaskWords = LeafMaskT::WORD_COUNT;

using ProcessedTileMap = nanovdb::tools::cuda::topology::detail::ProcessedTileMap<RootT>;
using TopologyBuilderT = nanovdb::tools::cuda::TopologyBuilder<BuildT, BuilderResource>;

/// One batched topology pass over the whole batch. `Refine` and `Coarsen` are the factor-2
/// subdivision / coarsening passes; `BoxDilate` is a Minkowski sum with the axis-aligned box
/// [boxLo, boxHi] (components in {-1,0,1}), covering NanoVDB's 26-neighbor DilateGrid
/// (boxLo=-1, boxHi=1) and fvdb's one-sided pad octants ({-1,0}^3 and {0,1}^3). Larger
/// boxes compose from multiple passes (Minkowski sums by boxes compose). `Prune` keeps the source
/// voxels set in `keepMasks`, one leaf mask per source leaf of the pass, in batch order (the leaves
/// of grid 0, then grid 1, ...); `numKeepMasks` must equal the source's leaf count, and the masks
/// must stay valid on the pass's stream until it ends.
struct TopologyPassSpec {
    enum class Op { Refine, Coarsen, BoxDilate, Prune };
    Op op;
    nanovdb::Coord boxLo{0}, boxHi{0};    // BoxDilate only
    const LeafMaskT *keepMasks = nullptr; // Prune only
    int64_t numKeepMasks       = 0;       // Prune only

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
    static TopologyPassSpec
    prune(const LeafMaskT *keepMasks, int64_t numKeepMasks) {
        TORCH_CHECK(numKeepMasks >= 0 && (numKeepMasks == 0 || keepMasks != nullptr),
                    "TopologyPassSpec::prune: ",
                    numKeepMasks,
                    " keep masks need a non-null pointer, and the count must not be negative");
        TopologyPassSpec spec{Op::Prune};
        spec.keepMasks    = keepMasks;
        spec.numKeepMasks = numKeepMasks;
        return spec;
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
    const LeafMaskT *keep;    // Prune: keep masks of the chunk's source leaves, chunk-leaf indexed
    int32_t numGrids;
};

/// The grid owning leaf `leaf` of leaves laid out grid by grid, where grid g's leaves start at
/// leafStart[g]: the last grid whose first leaf does not exceed it. Empty grids share their
/// successor's first leaf and are skipped that way.
__device__ inline int32_t
gridOfLeaf(const int64_t *leafStart, int32_t numGrids, int64_t leaf) {
    int32_t lo = 0, hi = numGrids;
    while (lo < hi) {
        const int32_t mid = (lo + hi) >> 1;
        if (leafStart[mid] <= leaf) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

/// The grid owning chunk-wide source leaf `leaf`.
__device__ inline int32_t
gridOfLeaf(const PassTables &t, int64_t leaf) {
    return gridOfLeaf(t.leafStart, t.numGrids, leaf);
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
                       uint64_t out[kLeafMaskWords]) {
    static_assert(kLeafMaskWords == 8, "the shifts below assume an 8^3 leaf of eight 64-bit words");
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
        } else if constexpr (Op == TopologyPassSpec::Op::Prune) {
            morph::PruneInternalNodesFunctor<BuildT>()(
                local, t.src[g], t.root[g], t.keep + t.leafStart[g], upper, lower);
        } else {
            const LeafT &srcLeaf = t.src[g]->tree().template getFirstNode<0>()[local];
            uint64_t contribution[kLeafMaskWords];
            if (!boxDilatedContribution(
                    srcLeaf.valueMask().words(), boxLo, boxHi, db, contribution)) {
                continue;
            }
            const RootT *root           = t.root[g];
            const nanovdb::Coord origin = srcLeaf.origin() + (db << LeafT::TOTAL);
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
        } else if constexpr (Op == TopologyPassSpec::Op::Prune) {
            morph::PruneLeafMasksFunctor<BuildT>()(
                local, t.src[g], t.dst[g], t.keep + t.leafStart[g]);
        } else {
            const LeafT &srcLeaf = t.src[g]->tree().template getFirstNode<0>()[local];
            uint64_t contribution[kLeafMaskWords];
            if (!boxDilatedContribution(
                    srcLeaf.valueMask().words(), boxLo, boxHi, db, contribution)) {
                continue;
            }
            const LeafT *dstLeaf =
                t.dst[g]->tree().root().probeLeaf(srcLeaf.origin() + (db << LeafT::TOTAL));
            auto &dstMask = const_cast<LeafMaskT &>(dstLeaf->valueMask());
            for (uint32_t w = 0; w < kLeafMaskWords; ++w) {
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

/// A cached builder and the mutex a builder run holds while it uses it.
/// `builder` is created by the first run that finds it null, and reset by a run that throws.
struct CachedTopologyBuilder {
    std::mutex mutex;
    std::unique_ptr<TopologyBuilderT> builder;
};

/// The process's cache entry for `stream` on `device`. The key needs the device: a null handle
/// names a different default stream on every device, and a builder's event and scratch belong to
/// the device it was created on. Threads share the entry and take its mutex for each builder run.
/// Entries are never destroyed, so no builder is torn down after CUDA shuts down and the pinned
/// host staging is allocated once rather than per pass. Each (device, stream) a pass has run on
/// keeps its builder's pinned staging and small per-grid tables for the life of the process; the
/// densified masks and other per-run scratch are released at the end of every run, and a run that
/// throws drops its builder entirely (see buildChunk).
inline CachedTopologyBuilder &
cachedTopologyBuilder(const torch::Device &device, cudaStream_t stream) {
    TORCH_CHECK(device.is_cuda() && device.has_index(),
                "cachedTopologyBuilder requires an indexed CUDA device, got ",
                device);
    using Key           = std::pair<c10::DeviceIndex, cudaStream_t>;
    static auto *mutex  = new std::mutex();
    static auto *cached = new std::map<Key, std::unique_ptr<CachedTopologyBuilder>>();
    std::lock_guard<std::mutex> lock(*mutex);
    auto &entry = (*cached)[Key{device.index(), stream}];
    if (!entry) {
        entry = std::make_unique<CachedTopologyBuilder>();
    }
    return *entry;
}

/// Drops a cached builder when the run using it throws. A run that fails after allocating, e.g.
/// an out-of-memory error from getBuffer, skips processLowerNodes, which releases the densified
/// masks, so the builder would otherwise hold them for the life of the process. The stream is
/// synchronized first so that the pinned staging is not freed under an in-flight copy; its status
/// is ignored, since the run is already failing.
struct DropBuilderOnFailure {
    CachedTopologyBuilder &cached;
    cudaStream_t stream;
    bool succeeded = false;

    ~DropBuilderOnFailure() {
        if (!succeeded) {
            (void)cudaStreamSynchronize(stream);
            cached.builder.reset();
        }
    }
};

/// FVDB_TOPOLOGY_MAX_TILES: the most speculative root tiles one builder run takes (default 64,
/// about 1 GiB of densified masks). A grid needing more runs alone. Unset or empty uses 64; any
/// other value that is not a positive integer in range throws.
inline int64_t
maxTilesPerChunk() {
    const char *env = std::getenv("FVDB_TOPOLOGY_MAX_TILES");
    if (env == nullptr || *env == '\0') {
        return 64;
    }
    char *end             = nullptr;
    errno                 = 0;
    const long long value = std::strtoll(env, &end, 10);
    TORCH_CHECK(*end == '\0' && errno != ERANGE && value > 0,
                "FVDB_TOPOLOGY_MAX_TILES must be a positive integer, got \"",
                env,
                "\"");
    return int64_t(value);
}

/// Root tiles a grid with index bounding box `bbox` can occupy: a bound on its upper nodes.
inline int64_t
rootTilesOverlapping(const nanovdb::CoordBBox &bbox) {
    if (bbox.empty()) {
        return 0;
    }
    int64_t n = 1;
    for (int a = 0; a < 3; ++a) {
        n *= int64_t(bbox.max()[a] >> UpperT::TOTAL) - int64_t(bbox.min()[a] >> UpperT::TOTAL) + 1;
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
    return int64_t(reinterpret_cast<std::uintptr_t>(ptr));
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
        // Every upper node holds at least one leaf, so `leaves` also bounds the upper nodes; it
        // keeps a grid with a few far-flung voxels from sizing the readback by its bbox volume. An
        // empty bbox on a grid with leaves falls back to `leaves`, so a bad bbox reaches
        // processedTiles' empty-bbox check rather than failing the capacity check here.
        int64_t capacity = 0;
        if (leaves > 0) {
            const nanovdb::CoordBBox bbox = batch.bboxAt(i);
            capacity = bbox.empty() ? leaves : std::min(leaves, rootTilesOverlapping(bbox));
        }
        src.upperCapacity.push_back(capacity);
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

/// Device tables for kernels that run once per leaf of a whole source (all of its grids): each
/// grid's device pointer and the first leaf of each grid, with the leaves laid out grid by grid.
class SourceLeafTables;
inline SourceLeafTables sourceLeafTables(const BatchedTopologySource &src, cudaStream_t stream);

/// Device tables for kernels over every leaf of a source, built only by sourceLeafTables(): each
/// grid's device pointer and each grid's first leaf (numGrids + 1 entries, the last being
/// numLeaves). The accessors throw on tables that do not match numGrids.
class SourceLeafTables {
  public:
    int32_t
    numGrids() const {
        return mNumGrids;
    }
    int64_t
    numLeaves() const {
        return mNumLeaves;
    }
    const GridT *const *
    gridPtrs() const {
        check();
        return reinterpret_cast<const GridT *const *>(mGrids.data_ptr<int64_t>());
    }
    const int64_t *
    leafStartPtr() const {
        check();
        return mLeafStart.data_ptr<int64_t>();
    }

  private:
    friend SourceLeafTables sourceLeafTables(const BatchedTopologySource &src, cudaStream_t stream);
    SourceLeafTables() = default;

    void
    check() const {
        TORCH_CHECK(mGrids.defined() && mLeafStart.defined() && mGrids.is_cuda() &&
                        mLeafStart.is_cuda() && mNumGrids > 0 && mNumLeaves >= 0 &&
                        mGrids.numel() == mNumGrids && mLeafStart.numel() == mNumGrids + 1,
                    "SourceLeafTables: tables do not match ",
                    mNumGrids,
                    " grids");
    }

    torch::Tensor mGrids;     // int64 device grid pointers, one per grid
    torch::Tensor mLeafStart; // int64, numGrids + 1 entries
    int32_t mNumGrids  = 0;
    int64_t mNumLeaves = 0;
};

/// Throws unless `src` holds at least one grid, every grid pointer is non-null, and there is one
/// leaf count and upper-node bound per grid.
inline void
checkTopologySource(const BatchedTopologySource &src) {
    TORCH_CHECK(!src.grids.empty(), "batched topology pass requires at least one grid");
    TORCH_CHECK(src.leafCounts.size() == src.grids.size() &&
                    src.upperCapacity.size() == src.grids.size(),
                "batched topology pass: source has ",
                src.grids.size(),
                " grids but ",
                src.leafCounts.size(),
                " leaf counts and ",
                src.upperCapacity.size(),
                " upper-node bounds");
    for (const GridT *grid: src.grids) {
        TORCH_CHECK(grid != nullptr, "batched topology pass: null source grid");
    }
}

/// Throws unless `pass` can run on `src`: BoxDilate box components in {-1,0} / {0,1}, and for Prune
/// one keep mask per source leaf.
inline void
checkTopologyPass(const BatchedTopologySource &src, const TopologyPassSpec &pass) {
    if (pass.op == TopologyPassSpec::Op::BoxDilate) {
        for (int a = 0; a < 3; ++a) {
            TORCH_CHECK(pass.boxLo[a] >= -1 && pass.boxLo[a] <= 0 && pass.boxHi[a] >= 0 &&
                            pass.boxHi[a] <= 1,
                        "BoxDilate pass components must be in {-1,0} / {0,1}");
        }
    }
    if (pass.op == TopologyPassSpec::Op::Prune) {
        int64_t leaves = 0;
        for (int64_t count: src.leafCounts) {
            leaves += count;
        }
        TORCH_CHECK(pass.numKeepMasks == leaves,
                    "Prune pass has ",
                    pass.numKeepMasks,
                    " keep masks for ",
                    leaves,
                    " source leaves");
        TORCH_CHECK(leaves == 0 || pass.keepMasks != nullptr,
                    "Prune pass requires keep masks for its source leaves");
    }
}

/// Uploads `src`'s grid pointers and per-grid first leaves on `stream` (no host wait).
inline SourceLeafTables
sourceLeafTables(const BatchedTopologySource &src, cudaStream_t stream) {
    checkTopologySource(src);
    const int32_t numGrids = int32_t(src.grids.size());
    std::vector<int64_t> grids(numGrids), leafStart(numGrids + 1, 0);
    for (int32_t g = 0; g < numGrids; ++g) {
        grids[g]         = tableEntry(src.grids[g]);
        leafStart[g + 1] = leafStart[g] + src.leafCounts[g];
    }
    SourceLeafTables tables;
    tables.mGrids     = uploadTable(grids, src.device, stream);
    tables.mLeafStart = uploadTable(leafStart, src.device, stream);
    tables.mNumGrids  = numGrids;
    tables.mNumLeaves = leafStart[numGrids];
    return tables;
}

/// Each source grid's upper-node bounding boxes, read back with one host wait.
inline std::vector<std::vector<nanovdb::CoordBBox>>
readUpperBBoxes(const BatchedTopologySource &src, cudaStream_t stream) {
    checkTopologySource(src);
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
/// boxes, as NanoVDB's RefineGrid, CoarsenGrid and DilateGrid derive them. An upper node with an
/// empty bbox throws: its leaves would still be processed, and the mask fills would look up a
/// root tile that was never added.
inline ProcessedTileMap
processedTiles(const std::vector<nanovdb::CoordBBox> &uppers, const TopologyPassSpec &pass) {
    namespace topo = nanovdb::tools::cuda::topology::detail;
    using nanovdb::util::morphology::coarsenCoord;
    using nanovdb::util::morphology::refineCoord;
    ProcessedTileMap tiles;
    for (const nanovdb::CoordBBox &b: uppers) {
        TORCH_CHECK(!b.empty(),
                    "batched topology pass: source grid has an upper node with no bbox");
        switch (pass.op) {
        case TopologyPassSpec::Op::Refine:
            topo::insertProcessedTiles<RootT>(
                tiles, nanovdb::CoordBBox(refineCoord(b.min()), refineCoord(b.max()).offsetBy(1)));
            break;
        case TopologyPassSpec::Op::Coarsen:
            // The coarsened upper node spans 2048 voxels from coarsenCoord(origin), inside one
            // root tile, so any of its coordinates names that tile. Using b.min() rather than the
            // tile origin keeps coarsenCoord's -n + 1 from overflowing in the lowest root tile.
            topo::insertProcessedTile<RootT>(tiles, coarsenCoord(b.min()));
            break;
        case TopologyPassSpec::Op::BoxDilate:
            topo::insertProcessedTiles<RootT>(
                tiles, nanovdb::CoordBBox(b.min() + pass.boxLo, b.max() + pass.boxHi));
            break;
        case TopologyPassSpec::Op::Prune:
            // Pruning keeps every source root tile speculatively, as NanoVDB's PruneGrid does; the
            // builder drops tiles whose upper node ends up empty.
            topo::insertProcessedTile<RootT>(tiles, b.min());
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
    checkTopologySource(src);
    checkTopologyPass(src, pass);
    TORCH_CHECK(0 <= first && first < last && size_t(last) <= src.grids.size() &&
                    tiles.size() == src.grids.size(),
                "buildChunk: grids [",
                first,
                ", ",
                last,
                ") with ",
                tiles.size(),
                " tile maps do not fit a source of ",
                src.grids.size(),
                " grids");
    namespace topo  = nanovdb::tools::cuda::topology::detail;
    using Op        = TopologyPassSpec::Op;
    const int32_t n = last - first;
    c10::cuda::CUDAGuard deviceGuard(src.device);
    CachedTopologyBuilder &cached = cachedTopologyBuilder(src.device, stream);
    std::lock_guard<std::mutex> builderLock(cached.mutex);
    if (!cached.builder) {
        cached.builder = std::make_unique<TopologyBuilderT>(stream);
    }
    DropBuilderOnFailure dropOnFailure{cached, stream};
    TopologyBuilderT &builder = *cached.builder;

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
    t.keep      = nullptr;
    if (pass.op == Op::Prune) {
        // The keep masks index the pass's source leaves in batch order; this chunk's leaves start
        // after those of grids [0, first).
        int64_t before = 0;
        for (int32_t g = 0; g < first; ++g) {
            before += src.leafCounts[g];
        }
        t.keep = pass.keepMasks + before;
    }
    t.numGrids = n;
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
        case Op::Prune:
            launchFillInternalMasks<Op::Prune>(t, numLeaves, pass, builder, stream);
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
        case Op::Prune: launchFillLeafMasks<Op::Prune>(t, numLeaves, pass, stream); break;
        }
    }
    builder.processLeafOffsets(stream);
    builder.processBBox(stream);
    builder.postProcessGridTree(stream);
    result.chunks.push_back(std::move(buffer));
    dropOnFailure.succeeded = true;
}

/// Runs one batched topology pass over all grids of `src`: one host wait for the source roots,
/// then per chunk one for its node counts and, after the first chunk, one for the builder's pinned
/// staging to come free. `stream` must be `src.device`'s current torch stream:
/// the pass's device tables are torch tensors, allocated and freed on that stream.
inline BatchedTopologyResult
runBatchedTopologyPass(const BatchedTopologySource &src,
                       const TopologyPassSpec &pass,
                       cudaStream_t stream) {
    TORCH_CHECK(src.device.is_cuda() && src.device.has_index(),
                "batched topology passes require an indexed CUDA device, got ",
                src.device);
    c10::cuda::CUDAGuard deviceGuard(src.device);
    TORCH_CHECK(stream == storageStream(src.device),
                "batched topology pass: stream must be the current torch stream of ",
                src.device);
    checkTopologySource(src);
    checkTopologyPass(src, pass);
    const int32_t numGrids = int32_t(src.grids.size());

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

/// Wraps a pass's output as storage. A single-chunk result is adopted with the layout the builder
/// computed (each header carries its gridIndex/gridCount), so its kernels may still be in flight on
/// `stream`, which the storage retains; several chunks are assembled into one storage.
inline GridStorage
storageFromResult(BatchedTopologyResult &&result,
                  const torch::Device &device,
                  cudaStream_t stream) {
    TORCH_CHECK(!result.chunks.empty() && !result.grids.empty() &&
                    result.gridBytes.size() == result.grids.size(),
                "storageFromResult: empty or inconsistent pass result");
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
            makeDeviceHandleFromLayout(std::move(result.chunks.front()), std::move(meta)), device);
    }

    GridSpanSource source{device, stream, {}};
    source.spans.reserve(numGrids);
    for (size_t g = 0; g < numGrids; ++g) {
        source.spans.push_back(
            GridSpan{result.grids[g], result.gridBytes[g], nanovdb::GridType::OnIndex});
    }
    return assembleGridStorage({source}, device, stream);
}

/// Runs a chained sequence of batched passes over the batch and wraps the result as the batch's
/// storage (see storageFromResult). `passes` must be non-empty.
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
    return storageFromResult(std::move(result), batch.device(), stream);
}

} // namespace batched
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_BATCHEDTOPOLOGYBUILDER_CUH
