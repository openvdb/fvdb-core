// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/views.h"

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/ops/ActiveVoxelsInBoundsMask.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Dispatch table wiring
// ---------------------------------------------------------------------------
// Only dispatches on device -- no user-supplied tensors whose type or
// contiguity could vary.  Outputs are freshly allocated (contiguous).
//
// Iteration pattern: forEachActiveVoxel (one thread per leaf-voxel slot).
//
// This matches the old forEachVoxelCUDA with numChannels=1: the thread
// decomposition is totalLeaves * VOXELS_PER_LEAF in both cases.
// forEachActiveVoxel already filters to active voxels before calling the
// callback, so the callback only needs the cheap isInside(ijk) test.
//
// The old code had an additional leaf-level early-out --
// maskBbox.hasOverlap(leaf.bbox()) -- that let each thread skip the
// per-voxel work when the entire leaf was outside the bbox.  That check
// was per-thread (not cooperative), but it did avoid the
// offsetToGlobalCoord + isInside work for every voxel in out-of-bounds
// leaves.  The new code omits it because forEachActiveVoxel does not
// expose the leaf reference to the callback.  If profiling shows this
// matters (e.g. small bbox relative to grid), a future forEachLeaf-based
// variant could restore the leaf-level rejection.

struct active_voxels_in_bounds_mask_op {
    template <typename Tag>
    static JaggedTensor
    op(Tag tg, GridBatchImpl const &grid, torch::Tensor batchBboxes) {
        constexpr auto dev = ::dispatch::tag_get<torch::DeviceType>(tg);

        auto out = torch::zeros({static_cast<int64_t>(grid.totalVoxels())},
                                torch::TensorOptions().dtype(torch::kBool).device(grid.device()));
        auto out_v =
            ::dispatch::tensor_out<dev, torch::kBool, 1, ::dispatch::contiguity::contiguous>(out);
        auto bboxes_v =
            ::dispatch::tensor_in<dev, torch::kInt32, 3, ::dispatch::contiguity::contiguous>(
                batchBboxes);

        ::fvdb::detail::dispatch::forEachActiveVoxel(
            tg,
            grid,
            [=] __hostdev__(Tag /*tg*/,
                            JIdxType batchIdx,
                            nanovdb::Coord ijk,
                            int64_t voxelIndex,
                            GridBatchImpl::Accessor /*acc*/) {
                nanovdb::CoordBBox const maskBbox(nanovdb::Coord(bboxes_v(batchIdx, 0, 0),
                                                                 bboxes_v(batchIdx, 0, 1),
                                                                 bboxes_v(batchIdx, 0, 2)),
                                                  nanovdb::Coord(bboxes_v(batchIdx, 1, 0),
                                                                 bboxes_v(batchIdx, 1, 1),
                                                                 bboxes_v(batchIdx, 1, 2)));
                if (maskBbox.isInside(ijk)) {
                    out_v(voxelIndex) = true;
                }
            });

        return grid.jaggedTensor(out);
    }

    using space     = ::dispatch::axes<::dispatch::torch_full_device_axis>;
    using subspaces = ::dispatch::coverage<space>;
    using dispatcher =
        ::dispatch::dispatch_table<space, JaggedTensor(GridBatchImpl const &, torch::Tensor)>;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

JaggedTensor
activeVoxelsInBoundsMask(GridBatchImpl const &grid,
                         std::vector<nanovdb::Coord> const &bboxMins,
                         std::vector<nanovdb::Coord> const &bboxMaxs) {
    grid.checkNonEmptyGrid();

    // Pack per-batch bounding boxes into a [B, 2, 3] int32 tensor on device.
    torch::Tensor batchBboxes =
        torch::empty({grid.batchSize(), 2, 3},
                     torch::TensorOptions().dtype(torch::kInt32).device(grid.device()));
    for (int64_t b = 0; b < grid.batchSize(); ++b) {
        for (int32_t d = 0; d < 3; ++d) {
            batchBboxes[b][0][d] = bboxMins[b][d];
            batchBboxes[b][1][d] = bboxMaxs[b][d];
        }
    }

    static auto const table = ::dispatch::dispatch_table_from_op<active_voxels_in_bounds_mask_op>(
        "activeVoxelsInBoundsMask");
    return table.select(::dispatch::dispatch_set{grid.device().type()})(grid, batchBboxes);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
