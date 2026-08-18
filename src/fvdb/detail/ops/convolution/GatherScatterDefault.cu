// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GatherScatterDefault.cu -- Default gather-scatter sparse convolution.
//
// Per kernel offset k, three phases are streamed:
//   1. Gather:       features[gatherIndices[start:end]] -> small buffer   (1 launch per k)
//   2. GEMM:         torch::mm on the gathered slice                       (1 launch per k)
//   3. Scatter-add:  result -> output[scatterIndices[start:end]] (atomic) (1 launch per k)
//
// Buffers are sized to max_pairs_per_k (the largest per-offset segment),
// reducing peak memory from O(total_pairs * C) to O(max_pairs_per_k * C).
//

#include "dispatch/detail/core_types.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"

#include <fvdb/detail/dispatch/AtomicAdd.cuh>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/dispatch/TensorChecks.h>
#include <fvdb/detail/ops/convolution/ConvolutionGeometry.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

using namespace ::dispatch;

// =============================================================================
// Tensor options helpers
// =============================================================================

template <torch::ScalarType Stype>
inline torch::TensorOptions
optsOn(torch::Device device) {
    return torch::dtype(Stype).device(device);
}

static void
checkTopologyPreconditions(GridBatchData const &feature_grid,
                           GridBatchData const &output_grid,
                           nanovdb::Coord kernel_size,
                           nanovdb::Coord stride) {
    TORCH_CHECK(feature_grid.device() == output_grid.device(),
                "feature_grid and output_grid must be on the same device, got ",
                feature_grid.device(),
                " and ",
                output_grid.device());
    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(
            kernel_size[d] > 0, "kernel_size[", d, "] must be positive, got ", kernel_size[d]);
        TORCH_CHECK(stride[d] > 0, "stride[", d, "] must be positive, got ", stride[d]);
    }
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(feature_grid.totalVoxels() <= kInt32Max,
                "feature_grid has ",
                feature_grid.totalVoxels(),
                " voxels, exceeding the int32 index limit (",
                kInt32Max,
                ")");
    TORCH_CHECK(output_grid.totalVoxels() <= kInt32Max,
                "output_grid has ",
                output_grid.totalVoxels(),
                " voxels, exceeding the int32 index limit (",
                kInt32Max,
                ")");
}

// =============================================================================
// Two-pass topology builder via dispatch framework (CPU + CUDA)
// =============================================================================
//
// Two sweeps over active output voxels via forEachActiveVoxel:
//   Sweep 1: Count active pairs per kernel offset k using atomic counters.
//   Prefix sums: Compute CSR offsets via torch::cumsum.
//   Sweep 2: Fill gather/scatter indices using atomic position assignment.

struct twopass_topology_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static GatherScatterDefaultTopology
    op(Tag tg,
       GridBatchData const &feature_grid,
       GridBatchData const &output_grid,
       nanovdb::Coord kernel_size,
       nanovdb::Coord stride,
       ConvDirection direction) {
        ConvolutionGeometry const geometry(kernel_size, stride);
        int64_t const K = geometry.kernelVolume();

        int64_t const featureTotal = feature_grid.totalVoxels();
        int64_t const output_total = output_grid.totalVoxels();

        bool const is_transposed = (direction == ConvDirection::Transposed);
        auto const device        = output_grid.device();

        // ---- Sweep 1: count pairs per offset k ----
        auto counts = torch::zeros({K}, optsOn<torch::kInt32>(device));
        auto guard  = make_device_guard(tg, counts);

        auto feature_acc = dispatch::make_grid_accessor(tg, feature_grid);
        auto *counts_ptr = counts.data_ptr<int32_t>();

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord ijk,
                            int64_t /*voxel_idx*/,
                            GridBatchData::Accessor /*output_acc*/) {
                auto const *feat_grid = feature_acc.grid(batch_idx);
                auto feat_tree_acc    = feat_grid->getAccessor();

                for (int64_t k = 0; k < K; ++k) {
                    nanovdb::Coord probe;
                    if (is_transposed) {
                        if (!geometry.coarseFromFine(ijk, geometry.tapCoord(k), probe))
                            continue;
                    } else {
                        probe = geometry.fineFromCoarse(ijk, geometry.tapCoord(k));
                    }

                    if (feat_tree_acc.isActive(probe)) {
                        dispatch::atomic_fetch_add_i32(tg_inner, &counts_ptr[k], 1);
                    }
                }
            });

        // ---- Prefix sums (device-generic torch ops) ----
        auto offsets_dev = torch::zeros({K + 1}, optsOn<torch::kInt64>(device));
        if (K > 0) {
            offsets_dev.slice(0, 1, K + 1).copy_(torch::cumsum(counts.to(torch::kInt64), 0));
        }
        auto offsets_host   = offsets_dev.cpu();
        auto offsets_acc    = offsets_host.accessor<int64_t, 1>();
        int64_t total_pairs = (K > 0) ? offsets_acc[K] : 0;

        if (total_pairs == 0) {
            return GatherScatterDefaultTopology{
                torch::empty({0}, optsOn<torch::kInt32>(device)),
                torch::empty({0}, optsOn<torch::kInt32>(device)),
                offsets_host,
                featureTotal,
                output_total,
                K,
                0,
                kernel_size,
                stride,
                direction,
            };
        }

        // ---- Sweep 2: fill gather/scatter indices ----
        auto gatherIndices  = torch::empty({total_pairs}, optsOn<torch::kInt32>(device));
        auto scatterIndices = torch::empty({total_pairs}, optsOn<torch::kInt32>(device));
        auto counters       = torch::zeros({K}, optsOn<torch::kInt32>(device));

        auto *counters_ptr = counters.data_ptr<int32_t>();
        auto *offsets_ptr  = offsets_dev.data_ptr<int64_t>();
        auto *gather_ptr   = gatherIndices.data_ptr<int32_t>();
        auto *scatter_ptr  = scatterIndices.data_ptr<int32_t>();

        dispatch::forEachActiveVoxel(
            tg,
            output_grid,
            [=] __hostdev__(Tag tg_inner,
                            JIdxType batch_idx,
                            nanovdb::Coord ijk,
                            int64_t voxel_idx,
                            GridBatchData::Accessor /*output_acc*/) {
                auto const *feat_grid   = feature_acc.grid(batch_idx);
                auto feat_tree_acc      = feat_grid->getAccessor();
                int64_t const feat_base = feature_acc.voxelOffset(batch_idx);

                for (int64_t k = 0; k < K; ++k) {
                    nanovdb::Coord probe;
                    if (is_transposed) {
                        if (!geometry.coarseFromFine(ijk, geometry.tapCoord(k), probe))
                            continue;
                    } else {
                        probe = geometry.fineFromCoarse(ijk, geometry.tapCoord(k));
                    }

                    if (feat_tree_acc.isActive(probe)) {
                        int32_t const feat_flat =
                            static_cast<int32_t>(feat_base + feat_tree_acc.getValue(probe) - 1);
                        int32_t const pos =
                            dispatch::atomic_fetch_add_i32(tg_inner, &counters_ptr[k], 1);
                        int64_t const write_pos = offsets_ptr[k] + pos;
                        gather_ptr[write_pos]   = feat_flat;
                        scatter_ptr[write_pos]  = static_cast<int32_t>(voxel_idx);
                    }
                }
            });

        return GatherScatterDefaultTopology{
            gatherIndices,
            scatterIndices,
            offsets_host,
            featureTotal,
            output_total,
            K,
            total_pairs,
            kernel_size,
            stride,
            direction,
        };
    }

    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space,
                                      GatherScatterDefaultTopology(GridBatchData const &,
                                                                   GridBatchData const &,
                                                                   nanovdb::Coord,
                                                                   nanovdb::Coord,
                                                                   ConvDirection)>;
};

static GatherScatterDefaultTopology
buildTopologyTwoPass(GridBatchData const &feature_grid,
                     GridBatchData const &output_grid,
                     nanovdb::Coord kernel_size,
                     nanovdb::Coord stride,
                     ConvDirection direction) {
    checkTopologyPreconditions(feature_grid, output_grid, kernel_size, stride);

    static auto const table =
        dispatch_table_from_op<twopass_topology_op>("gather_scatter_default_twopass_topology");

    auto const dev = feature_grid.device().type();
    return table.select(dispatch_set{dev})(
        feature_grid, output_grid, kernel_size, stride, direction);
}

// =============================================================================
// Topology builder entry points (two-pass for all devices)
// =============================================================================

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTopology(GridBatchData const &feature_grid,
                                       GridBatchData const &output_grid,
                                       nanovdb::Coord kernel_size,
                                       nanovdb::Coord stride) {
    return buildTopologyTwoPass(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Forward);
}

GatherScatterDefaultTopology
gatherScatterDefaultSparseConvTransposeTopology(GridBatchData const &feature_grid,
                                                GridBatchData const &output_grid,
                                                nanovdb::Coord kernel_size,
                                                nanovdb::Coord stride) {
    return buildTopologyTwoPass(
        feature_grid, output_grid, kernel_size, stride, ConvDirection::Transposed);
}

GatherScatterDefaultTopology
reverseGatherScatterDefaultTopology(GatherScatterDefaultTopology const &topology) {
    TORCH_CHECK(topology.direction == ConvDirection::Forward ||
                    topology.direction == ConvDirection::Transposed,
                "cannot reverse topology with invalid convolution direction ",
                static_cast<int>(topology.direction));
    ConvDirection const reversedDirection = topology.direction == ConvDirection::Forward
                                                ? ConvDirection::Transposed
                                                : ConvDirection::Forward;
    return GatherScatterDefaultTopology{
        topology.scatterIndices,
        topology.gatherIndices,
        topology.offsets,
        topology.outputTotalVoxels,
        topology.featureTotalVoxels,
        topology.kernelVolume,
        topology.totalPairs,
        topology.kernelSize,
        topology.stride,
        reversedDirection,
    };
}

// =============================================================================
// Explicit test/debug topology validation
// =============================================================================

struct topology_validation_coordinates_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType>
    static torch::Tensor
    op(Tag tg, GridBatchData const &grid) {
        auto coordinates = torch::full({grid.totalVoxels(), 4},
                                       std::numeric_limits<int32_t>::min(),
                                       optsOn<torch::kInt32>(grid.device()));
        if (grid.totalVoxels() == 0) {
            return coordinates;
        }

        auto guard           = make_device_guard(tg, coordinates);
        auto *coordinatesPtr = coordinates.data_ptr<int32_t>();
        dispatch::forEachActiveVoxel(tg,
                                     grid,
                                     [=] __hostdev__(Tag,
                                                     JIdxType batchIndex,
                                                     nanovdb::Coord ijk,
                                                     int64_t voxelIndex,
                                                     GridBatchData::Accessor) {
                                         int32_t *row = coordinatesPtr + voxelIndex * 4;
                                         row[0]       = static_cast<int32_t>(batchIndex);
                                         row[1]       = ijk[0];
                                         row[2]       = ijk[1];
                                         row[3]       = ijk[2];
                                     });
        return coordinates;
    }

    using space      = axes<torch_full_device_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, torch::Tensor(GridBatchData const &)>;
};

static torch::Tensor
topologyValidationCoordinates(GridBatchData const &grid) {
    static auto const table = dispatch_table_from_op<topology_validation_coordinates_op>(
        "gather_scatter_default_validation_coordinates");
    return table.select(dispatch_set{grid.device().type()})(grid);
}

void
validateGatherScatterDefaultTopology(GridBatchData const &fineGrid,
                                     GridBatchData const &coarseGrid,
                                     GatherScatterDefaultTopology const &topology) {
    TORCH_CHECK(topology.direction == ConvDirection::Forward ||
                    topology.direction == ConvDirection::Transposed,
                "topology has invalid convolution direction ",
                static_cast<int>(topology.direction));
    TORCH_CHECK(fineGrid.device() == coarseGrid.device(),
                "fine_grid and coarse_grid must be on the same device, got ",
                fineGrid.device(),
                " and ",
                coarseGrid.device());
    TORCH_CHECK(fineGrid.batchSize() == coarseGrid.batchSize(),
                "fine_grid and coarse_grid batch sizes must match, got ",
                fineGrid.batchSize(),
                " and ",
                coarseGrid.batchSize());

    bool const isForward             = topology.direction == ConvDirection::Forward;
    GridBatchData const &featureGrid = isForward ? fineGrid : coarseGrid;
    GridBatchData const &outputGrid  = isForward ? coarseGrid : fineGrid;
    TORCH_CHECK(topology.featureTotalVoxels == featureGrid.totalVoxels(),
                "topology feature voxel count ",
                topology.featureTotalVoxels,
                " does not match its ",
                isForward ? "fine" : "coarse",
                " domain count ",
                featureGrid.totalVoxels());
    TORCH_CHECK(topology.outputTotalVoxels == outputGrid.totalVoxels(),
                "topology output voxel count ",
                topology.outputTotalVoxels,
                " does not match its ",
                isForward ? "coarse" : "fine",
                " domain count ",
                outputGrid.totalVoxels());
    TORCH_CHECK(topology.totalPairs >= 0,
                "topology total pair count must be nonnegative, got ",
                topology.totalPairs);

    ConvolutionGeometry const geometry(topology.kernelSize, topology.stride);
    TORCH_CHECK(topology.kernelVolume == geometry.kernelVolume(),
                "topology kernel volume ",
                topology.kernelVolume,
                " does not match geometry volume ",
                geometry.kernelVolume());

    auto checkIndexTensor = [&](torch::Tensor const &tensor, char const *name) {
        TORCH_CHECK(tensor.defined(), name, " must be defined");
        TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have int32 dtype");
        TORCH_CHECK(tensor.device() == fineGrid.device(),
                    name,
                    " must be on grid device ",
                    fineGrid.device(),
                    ", got ",
                    tensor.device());
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.size(0) == topology.totalPairs,
                    name,
                    " length ",
                    tensor.size(0),
                    " does not match total pair count ",
                    topology.totalPairs);
    };
    checkIndexTensor(topology.gatherIndices, "gather_indices");
    checkIndexTensor(topology.scatterIndices, "scatter_indices");

    TORCH_CHECK(topology.offsets.defined(), "offsets must be defined");
    TORCH_CHECK(topology.offsets.dim() == 1, "offsets must be one-dimensional");
    TORCH_CHECK(topology.offsets.scalar_type() == torch::kInt64, "offsets must have int64 dtype");
    TORCH_CHECK(topology.offsets.device().is_cpu(), "offsets must be stored on CPU");
    TORCH_CHECK(topology.offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(topology.offsets.size(0) == topology.kernelVolume + 1,
                "offset length ",
                topology.offsets.size(0),
                " does not match kernel volume + 1 (",
                topology.kernelVolume + 1,
                ")");

    auto offsets = topology.offsets.accessor<int64_t, 1>();
    TORCH_CHECK(offsets[0] == 0, "offsets must start at zero, got ", offsets[0]);
    for (int64_t tap = 0; tap < topology.kernelVolume; ++tap) {
        TORCH_CHECK(offsets[tap] <= offsets[tap + 1],
                    "offsets must be monotone at tap ",
                    tap,
                    ": ",
                    offsets[tap],
                    " > ",
                    offsets[tap + 1]);
    }
    TORCH_CHECK(offsets[topology.kernelVolume] == topology.totalPairs,
                "final offset ",
                offsets[topology.kernelVolume],
                " does not match total pair count ",
                topology.totalPairs);

    auto gatherHost  = topology.gatherIndices.cpu().contiguous();
    auto scatterHost = topology.scatterIndices.cpu().contiguous();
    auto gather      = gatherHost.accessor<int32_t, 1>();
    auto scatter     = scatterHost.accessor<int32_t, 1>();
    for (int64_t pair = 0; pair < topology.totalPairs; ++pair) {
        TORCH_CHECK(gather[pair] >= 0 && gather[pair] < topology.featureTotalVoxels,
                    "gather index out of range at pair ",
                    pair,
                    ": ",
                    gather[pair]);
        TORCH_CHECK(scatter[pair] >= 0 && scatter[pair] < topology.outputTotalVoxels,
                    "scatter index out of range at pair ",
                    pair,
                    ": ",
                    scatter[pair]);
    }

    auto fineCoordinatesHost   = topologyValidationCoordinates(fineGrid).cpu().contiguous();
    auto coarseCoordinatesHost = topologyValidationCoordinates(coarseGrid).cpu().contiguous();
    auto fineCoordinates       = fineCoordinatesHost.accessor<int32_t, 2>();
    auto coarseCoordinates     = coarseCoordinatesHost.accessor<int32_t, 2>();
    using CoordinateKey        = std::tuple<int32_t, int32_t, int32_t, int32_t>;
    using EdgeKey              = std::tuple<int64_t, int64_t, int64_t>;

    std::map<CoordinateKey, int64_t> fineIndexByCoordinate;
    for (int64_t fineIndex = 0; fineIndex < fineGrid.totalVoxels(); ++fineIndex) {
        CoordinateKey const key{fineCoordinates[fineIndex][0],
                                fineCoordinates[fineIndex][1],
                                fineCoordinates[fineIndex][2],
                                fineCoordinates[fineIndex][3]};
        TORCH_CHECK(std::get<0>(key) >= 0 && std::get<0>(key) < fineGrid.batchSize(),
                    "fine coordinate lookup was not populated at flat index ",
                    fineIndex);
        TORCH_CHECK(fineIndexByCoordinate.emplace(key, fineIndex).second,
                    "fine grid contains duplicate batch/coordinate entries");
    }

    std::set<EdgeKey> actualEdges;
    for (int64_t tap = 0; tap < topology.kernelVolume; ++tap) {
        for (int64_t pair = offsets[tap]; pair < offsets[tap + 1]; ++pair) {
            int64_t const fineIndex   = isForward ? gather[pair] : scatter[pair];
            int64_t const coarseIndex = isForward ? scatter[pair] : gather[pair];
            TORCH_CHECK(fineCoordinates[fineIndex][0] == coarseCoordinates[coarseIndex][0],
                        "edge crosses batch domains at pair ",
                        pair);
            nanovdb::Coord const fineCoordinate(fineCoordinates[fineIndex][1],
                                                fineCoordinates[fineIndex][2],
                                                fineCoordinates[fineIndex][3]);
            nanovdb::Coord const coarseCoordinate(coarseCoordinates[coarseIndex][1],
                                                  coarseCoordinates[coarseIndex][2],
                                                  coarseCoordinates[coarseIndex][3]);
            TORCH_CHECK(fineCoordinate ==
                            geometry.fineFromCoarse(coarseCoordinate, geometry.tapCoord(tap)),
                        "edge does not satisfy canonical fine/coarse geometry at pair ",
                        pair,
                        ", tap ",
                        tap);
            TORCH_CHECK(actualEdges.emplace(fineIndex, coarseIndex, tap).second,
                        "duplicate fine/coarse/tap edge at pair ",
                        pair);
        }
    }

    std::set<EdgeKey> expectedEdges;
    for (int64_t coarseIndex = 0; coarseIndex < coarseGrid.totalVoxels(); ++coarseIndex) {
        int32_t const batch = coarseCoordinates[coarseIndex][0];
        TORCH_CHECK(batch >= 0 && batch < coarseGrid.batchSize(),
                    "coarse coordinate lookup was not populated at flat index ",
                    coarseIndex);
        nanovdb::Coord const coarseCoordinate(coarseCoordinates[coarseIndex][1],
                                              coarseCoordinates[coarseIndex][2],
                                              coarseCoordinates[coarseIndex][3]);
        for (int64_t tap = 0; tap < topology.kernelVolume; ++tap) {
            nanovdb::Coord const fineCoordinate =
                geometry.fineFromCoarse(coarseCoordinate, geometry.tapCoord(tap));
            CoordinateKey const fineKey{
                batch, fineCoordinate[0], fineCoordinate[1], fineCoordinate[2]};
            auto const fineEntry = fineIndexByCoordinate.find(fineKey);
            if (fineEntry != fineIndexByCoordinate.end()) {
                expectedEdges.emplace(fineEntry->second, coarseIndex, tap);
            }
        }
    }
    TORCH_CHECK(actualEdges == expectedEdges,
                "stored topology edge set does not equal the complete canonical relation: got ",
                actualEdges.size(),
                " edges, expected ",
                expectedEdges.size());
}

// =============================================================================
// Gather / scatter-add helpers (tag-dispatched via for_each)
// =============================================================================

template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
gsDefaultGather(Tag tg,
                torch::Tensor src,
                torch::Tensor dst,
                torch::Tensor indices,
                int64_t total_pairs,
                int64_t C) {
    if (total_pairs == 0)
        return;

    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto const *src_ptr = src.data_ptr<scalar_t>();
    auto *dst_ptr       = dst.data_ptr<scalar_t>();
    auto const *idx_ptr = indices.data_ptr<int32_t>();

    for_each(tg, total_pairs * C, [=] __hostdev__(Tag, int64_t idx) {
        int64_t const pair    = idx / C;
        int64_t const c       = idx % C;
        int32_t const src_row = idx_ptr[pair];
        dst_ptr[pair * C + c] = src_ptr[static_cast<int64_t>(src_row) * C + c];
    });
}

template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
void
gsDefaultScatterAdd(Tag tg,
                    torch::Tensor src,
                    torch::Tensor dst,
                    torch::Tensor indices,
                    int64_t total_pairs,
                    int64_t C) {
    if (total_pairs == 0)
        return;

    constexpr auto stype = tag_get<torch::ScalarType>(tg);
    using scalar_t       = torch_scalar_cpp_type_t<stype>;

    auto const *src_ptr = src.data_ptr<scalar_t>();
    auto *dst_ptr       = dst.data_ptr<scalar_t>();
    auto const *idx_ptr = indices.data_ptr<int32_t>();

    for_each(tg, total_pairs * C, [=] __hostdev__(Tag tg_inner, int64_t idx) {
        int64_t const pair    = idx / C;
        int64_t const c       = idx % C;
        int32_t const dst_row = idx_ptr[pair];
        dispatch::atomic_add(
            tg_inner, &dst_ptr[static_cast<int64_t>(dst_row) * C + c], src_ptr[pair * C + c]);
    });
}

// =============================================================================
// Type promotion
// =============================================================================

static torch::ScalarType
promoteFloatTypes(torch::ScalarType a, torch::ScalarType b) {
    return at::result_type(torch::empty({0}, torch::dtype(a)), torch::empty({0}, torch::dtype(b)));
}

// =============================================================================
// CPU-safe matrix multiply
// =============================================================================
//
// torch::mm on CPU does not support float16 or bfloat16.  On those types we
// promote to float32, multiply, and demote.  On CUDA the fast path is always
// taken (cuBLAS supports half types natively via tensor cores).

static void
mmOutSafe(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b) {
    if (a.is_cpu() && (a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kBFloat16)) {
        auto a_f = a.to(torch::kFloat32);
        auto b_f = b.to(torch::kFloat32);
        auto o_f = torch::mm(a_f, b_f);
        out.copy_(o_f.to(out.scalar_type()));
    } else {
        torch::mm_out(out, a, b);
    }
}

// =============================================================================
// Max pairs-per-offset helper
// =============================================================================

static int64_t
maxPairsPerOffset(torch::Tensor const &offsets, int64_t K) {
    auto acc      = offsets.accessor<int64_t, 1>();
    int64_t max_n = 0;
    for (int64_t k = 0; k < K; ++k) {
        max_n = std::max(max_n, acc[k + 1] - acc[k]);
    }
    return max_n;
}

// =============================================================================
// Precondition checks
// =============================================================================

static void
checkConvPreconditions(torch::Tensor features,
                       torch::Tensor weights,
                       GatherScatterDefaultTopology const &topo,
                       char const *name) {
    TORCH_CHECK(features.dim() == 2, name, ": features must be 2D");
    TORCH_CHECK(features.size(0) == topo.featureTotalVoxels,
                name,
                ": features.size(0)=",
                features.size(0),
                " must match featureTotalVoxels=",
                topo.featureTotalVoxels);
    TORCH_CHECK(features.is_floating_point(), name, ": features must be floating point");
    TORCH_CHECK(features.is_contiguous(), name, ": features must be contiguous");

    TORCH_CHECK(weights.dim() == 5, name, ": weights must be 5D [C_out, C_in, k0, k1, k2]");
    TORCH_CHECK(weights.is_floating_point(), name, ": weights must be floating point");
    TORCH_CHECK(features.size(1) == weights.size(1),
                name,
                ": features channels=",
                features.size(1),
                " must match weights C_in=",
                weights.size(1));

    TORCH_CHECK(weights.size(2) == topo.kernelSize[0] && weights.size(3) == topo.kernelSize[1] &&
                    weights.size(4) == topo.kernelSize[2],
                name,
                ": weights spatial dims must match topology kernel_size");

    TORCH_CHECK(features.device() == weights.device(),
                name,
                ": features and weights must be on the same device");
}

// =============================================================================
// Forward convolution (shared by forward and transposed)
// =============================================================================

struct gs_default_conv_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features,
       torch::Tensor weights,
       GatherScatterDefaultTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const O     = topo.outputTotalVoxels;
        int64_t const K     = topo.kernelVolume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.totalPairs;

        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto output = torch::zeros({O, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0)
            return output;

        int64_t const max_n = maxPairsPerOffset(topo.offsets, K);
        auto buf_A          = torch::empty({max_n, C_in}, features.options());
        auto buf_D          = torch::empty({max_n, C_out}, features.options());
        auto off_acc        = topo.offsets.accessor<int64_t, 1>();

        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const end   = off_acc[k + 1];
            int64_t const n_k   = end - start;

            if (n_k == 0)
                continue;

            auto A_k = buf_A.slice(0, 0, n_k);
            auto D_k = buf_D.slice(0, 0, n_k);

            gsDefaultGather(tg, features, A_k, topo.gatherIndices.slice(0, start, end), n_k, C_in);
            mmOutSafe(D_k, A_k, W[k]);
            gsDefaultScatterAdd(
                tg, D_k, output, topo.scatterIndices.slice(0, start, end), n_k, C_out);
        }

        return output;
    }

    using space     = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces = coverage<space>;
    using dispatcher =
        dispatch_table<space,
                       torch::Tensor(
                           torch::Tensor, torch::Tensor, GatherScatterDefaultTopology const &)>;
};

// =============================================================================
// Backward convolution (shared by forward and transposed entry points)
//
// The topology arrays are already oriented for execution. Direction is
// endpoint metadata validated by the wrappers below; this executor neither
// reapplies convolution geometry nor swaps gather/scatter indices.
// =============================================================================

struct gs_default_backward_op {
    template <typename Tag>
        requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType>
    static std::tuple<torch::Tensor, torch::Tensor>
    op(Tag tg,
       torch::Tensor grad_output,
       torch::Tensor features,
       torch::Tensor weights,
       GatherScatterDefaultTopology const &topo) {
        constexpr auto dev = tag_get<torch::DeviceType>(Tag{});

        auto guard = make_device_guard(tag<dev>{}, features);

        int64_t const F     = topo.featureTotalVoxels;
        int64_t const O     = topo.outputTotalVoxels;
        int64_t const K     = topo.kernelVolume;
        int64_t const C_in  = weights.size(1);
        int64_t const C_out = weights.size(0);
        int64_t const TP    = topo.totalPairs;

        auto W = weights.permute({2, 3, 4, 1, 0}).reshape({K, C_in, C_out}).contiguous();
        if (W.scalar_type() != features.scalar_type()) {
            W = W.to(features.scalar_type());
        }

        auto grad_features = torch::zeros({F, C_in}, features.options());

        auto grad_W_flat = torch::zeros({K, C_in, C_out}, features.options());

        if (O == 0 || K == 0 || TP == 0) {
            auto ks           = topo.kernelSize;
            auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                    .permute({4, 3, 0, 1, 2})
                                    .contiguous();
            return {grad_features, grad_weights};
        }

        auto off_acc = topo.offsets.accessor<int64_t, 1>();

        int64_t const max_n = maxPairsPerOffset(topo.offsets, K);
        auto feat_buf       = torch::empty({max_n, C_in}, features.options());
        auto grad_buf       = torch::empty({max_n, C_out}, features.options());
        auto grad_feat_buf  = torch::empty({max_n, C_in}, features.options());

        for (int64_t k = 0; k < K; ++k) {
            int64_t const start = off_acc[k];
            int64_t const end   = off_acc[k + 1];
            int64_t const n_k   = end - start;

            if (n_k == 0)
                continue;

            auto gi_k = topo.gatherIndices.slice(0, start, end);
            auto si_k = topo.scatterIndices.slice(0, start, end);
            auto fb_k = feat_buf.slice(0, 0, n_k);
            auto gb_k = grad_buf.slice(0, 0, n_k);
            auto gf_k = grad_feat_buf.slice(0, 0, n_k);

            gsDefaultGather(tg, features, fb_k, gi_k, n_k, C_in);
            gsDefaultGather(tg, grad_output, gb_k, si_k, n_k, C_out);

            mmOutSafe(gf_k, gb_k, W[k].t());
            gsDefaultScatterAdd(tg, gf_k, grad_features, gi_k, n_k, C_in);

            auto gw_k = grad_W_flat[k];
            mmOutSafe(gw_k, fb_k.t(), gb_k);
        }

        auto ks           = topo.kernelSize;
        auto grad_weights = grad_W_flat.reshape({ks[0], ks[1], ks[2], C_in, C_out})
                                .permute({4, 3, 0, 1, 2})
                                .contiguous();

        return {grad_features, grad_weights};
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<
        space,
        std::tuple<torch::Tensor, torch::Tensor>(
            torch::Tensor, torch::Tensor, torch::Tensor, GatherScatterDefaultTopology const &)>;
};

static std::tuple<torch::Tensor, torch::Tensor>
executeGatherScatterDefaultBackward(torch::Tensor gradOutput,
                                    torch::Tensor features,
                                    torch::Tensor weights,
                                    GatherScatterDefaultTopology const &topology,
                                    torch::ScalarType workingType) {
    static auto const table =
        dispatch_table_from_op<gs_default_backward_op>("gather_scatter_default_shared_backward");
    auto const device = features.device().type();
    return table.select(dispatch_set{device, workingType})(gradOutput, features, weights, topology);
}

// =============================================================================
// Type-erased entry points
// =============================================================================

torch::Tensor
gatherScatterDefaultSparseConv(torch::Tensor features,
                               torch::Tensor weights,
                               GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConv");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterDefaultSparseConv requires topology with direction=Forward");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvBackward(torch::Tensor grad_output,
                                       torch::Tensor features,
                                       torch::Tensor weights,
                                       GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConvBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Forward,
                "gatherScatterDefaultSparseConvBackward requires topology with direction=Forward");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.outputTotalVoxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);
    if (grad_output.scalar_type() != working_st)
        grad_output = grad_output.to(working_st);

    return executeGatherScatterDefaultBackward(grad_output, features, weights, topo, working_st);
}

torch::Tensor
gatherScatterDefaultSparseConvTranspose(torch::Tensor features,
                                        torch::Tensor weights,
                                        GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(features, weights, topo, "gatherScatterDefaultSparseConvTranspose");
    TORCH_CHECK(
        topo.direction == ConvDirection::Transposed,
        "gatherScatterDefaultSparseConvTranspose requires topology with direction=Transposed");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);

    static auto const table =
        dispatch_table_from_op<gs_default_conv_op>("gather_scatter_default_sparse_conv_transpose");

    auto const dev = features.device().type();
    return table.select(dispatch_set{dev, working_st})(features, weights, topo);
}

std::tuple<torch::Tensor, torch::Tensor>
gatherScatterDefaultSparseConvTransposeBackward(torch::Tensor grad_output,
                                                torch::Tensor features,
                                                torch::Tensor weights,
                                                GatherScatterDefaultTopology const &topo) {
    checkConvPreconditions(
        features, weights, topo, "gatherScatterDefaultSparseConvTransposeBackward");
    TORCH_CHECK(topo.direction == ConvDirection::Transposed,
                "gatherScatterDefaultSparseConvTransposeBackward requires direction=Transposed");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == topo.outputTotalVoxels,
                "grad_output shape mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.is_floating_point(), "grad_output must be floating point");

    auto working_st = promoteFloatTypes(features.scalar_type(), weights.scalar_type());
    if (features.scalar_type() != working_st)
        features = features.to(working_st);
    if (grad_output.scalar_type() != working_st)
        grad_output = grad_output.to(working_st);

    return executeGatherScatterDefaultBackward(grad_output, features, weights, topo, working_st);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
