// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/ops/GridEdgeNetwork.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Only dispatches on device -- all output types are fixed (float32
// vertices, int64 edges), freshly allocated (contiguous).
//
// Iteration pattern: forEachActiveVoxel (one thread per leaf-voxel slot).
//
// The old code used forEachVoxelCUDA/CPU with numChannels=1, which is
// the same thread decomposition and the same per-voxel work (emit 8
// vertices + 12 edges).  The callback body is unchanged.

struct grid_edge_network_op {
    template <typename Tag>
    static std::vector<JaggedTensor>
    op(Tag tg, GridBatchImpl const &grid, bool returnVoxelCoordinates) {
        constexpr auto dev = ::dispatch::tag_get<torch::DeviceType>(tg);

        int64_t const totVoxels = grid.totalVoxels();

        auto optsV = torch::TensorOptions().dtype(torch::kFloat32).device(grid.device());
        auto optsE = torch::TensorOptions().dtype(torch::kInt64).device(grid.device());
        auto optsB = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(grid.device());

        torch::Tensor outV     = torch::empty({8 * totVoxels, 3}, optsV);
        torch::Tensor outE     = torch::empty({12 * totVoxels, 2}, optsE);
        torch::Tensor outVBidx = torch::empty({8 * totVoxels}, optsB);
        torch::Tensor outEBidx = torch::empty({12 * totVoxels}, optsB);

        auto v_out =
            ::dispatch::tensor_out<dev, torch::kFloat32, 2, ::dispatch::contiguity::contiguous>(
                outV);
        auto e_out =
            ::dispatch::tensor_out<dev, torch::kInt64, 2, ::dispatch::contiguity::contiguous>(outE);

        // Use raw pointers for batch-index tensors (JIdxType = int32_t).
        JIdxType *vbidx_ptr = outVBidx.data_ptr<JIdxType>();
        JIdxType *ebidx_ptr = outEBidx.data_ptr<JIdxType>();

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            grid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType batchIdx,
                            nanovdb::Coord voxIjk,
                            int64_t voxelIndex,
                            GridBatchImpl::Accessor acc) {
                int64_t const countV = voxelIndex * 8;
                int64_t const countE = voxelIndex * 12;

                VoxelCoordTransform const tx = acc.primalTransform(batchIdx);

                // -- 8 cube-corner vertices --
                for (int idx = 0; idx < 8; ++idx) {
                    int32_t const iz = (idx & 1);
                    int32_t const iy = (idx & 2) >> 1;
                    int32_t const ix = (idx & 4) >> 2;

                    nanovdb::Vec3f xyz =
                        (voxIjk + nanovdb::Coord(ix, iy, iz)).asVec3s() - nanovdb::Vec3f(0.5f);
                    if (!returnVoxelCoordinates) {
                        xyz = tx.applyInv(xyz);
                    }

                    v_out(countV + idx, 0)  = xyz[0];
                    v_out(countV + idx, 1)  = xyz[1];
                    v_out(countV + idx, 2)  = xyz[2];
                    vbidx_ptr[countV + idx] = batchIdx;
                }

                // -- 12 cube edges (vertex indices relative to per-batch base) --
                int64_t const baseOffset = acc.voxelOffset(batchIdx);
                int64_t const eBase      = countV - baseOffset * 8;

                e_out(countE + 0, 0)  = 0 + eBase;
                e_out(countE + 0, 1)  = 1 + eBase;
                e_out(countE + 1, 0)  = 0 + eBase;
                e_out(countE + 1, 1)  = 2 + eBase;
                e_out(countE + 2, 0)  = 0 + eBase;
                e_out(countE + 2, 1)  = 4 + eBase;
                e_out(countE + 3, 0)  = 2 + eBase;
                e_out(countE + 3, 1)  = 3 + eBase;
                e_out(countE + 4, 0)  = 2 + eBase;
                e_out(countE + 4, 1)  = 6 + eBase;
                e_out(countE + 5, 0)  = 3 + eBase;
                e_out(countE + 5, 1)  = 7 + eBase;
                e_out(countE + 6, 0)  = 3 + eBase;
                e_out(countE + 6, 1)  = 1 + eBase;
                e_out(countE + 7, 0)  = 7 + eBase;
                e_out(countE + 7, 1)  = 6 + eBase;
                e_out(countE + 8, 0)  = 6 + eBase;
                e_out(countE + 8, 1)  = 4 + eBase;
                e_out(countE + 9, 0)  = 7 + eBase;
                e_out(countE + 9, 1)  = 5 + eBase;
                e_out(countE + 10, 0) = 5 + eBase;
                e_out(countE + 10, 1) = 4 + eBase;
                e_out(countE + 11, 0) = 1 + eBase;
                e_out(countE + 11, 1) = 5 + eBase;

                for (int i = 0; i < 12; ++i) {
                    ebidx_ptr[countE + i] = batchIdx;
                }
            });

        // FIXME: (@fwilliams) Be smarter about this
        torch::Tensor const outVBidx2 = grid.batchSize() == 1 ? torch::empty({0}, optsB) : outVBidx;
        torch::Tensor const outEBidx2 = grid.batchSize() == 1 ? torch::empty({0}, optsB) : outEBidx;

        return {JaggedTensor::from_data_indices_and_list_ids(
                    outV, outVBidx2, grid.jlidx(), grid.batchSize()),
                JaggedTensor::from_data_indices_and_list_ids(
                    outE, outEBidx2, grid.jlidx(), grid.batchSize())};
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher =
        ::dispatch::dispatch_table<space, std::vector<JaggedTensor>(GridBatchImpl const &, bool)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

std::vector<JaggedTensor>
gridEdgeNetwork(GridBatchImpl const &grid, bool returnVoxelCoordinates) {
    c10::DeviceGuard guard(grid.device());
    grid.checkNonEmptyGrid();

    static auto const table =
        ::dispatch::dispatch_table_from_op<grid_edge_network_op>("gridEdgeNetwork");
    return table.select(::dispatch::dispatch_set{grid.device().type()})(grid,
                                                                        returnVoxelCoordinates);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
