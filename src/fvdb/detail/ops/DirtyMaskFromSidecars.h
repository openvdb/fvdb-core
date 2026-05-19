// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_DIRTYMASKFROMSIDECARS_H
#define FVDB_DETAIL_OPS_DIRTYMASKFROMSIDECARS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Compute a "dirty" bool mask on `newGrid` flagging voxels
///        whose sidecar value differs from the corresponding voxel in
///        `oldGrid` (if present), or is absent from `oldGrid` entirely.
///
/// Primitive used by the paper's dirty-region ESDF update pattern
/// (and composable into any user-level change-tracking workflow).
/// Built from `ops::inject`, no new CUDA kernels.
///
/// Semantics per output voxel `v` in `newGrid`:
///
///   - If `v.ijk` is **not** in `oldGrid`: the voxel is new → marked
///     dirty.
///   - If `v.ijk` IS in `oldGrid` at some `w` and
///     `newSidecar[v] == oldSidecar[w]` (elementwise equality across
///     all channels for multi-channel sidecars): not dirty.
///   - Otherwise: dirty.
///
/// Multi-channel sidecars (2-D `[num_voxels, C]`) reduce via
/// "any channel differs" → per-voxel bool.
///
/// Both sidecars must have floating-point dtype in M5; we use the
/// NaN != anything trick to flag "voxel not present in old grid"
/// without needing a separate overlap mask pass (NaN-init the
/// projection target, inject only writes ijk-overlap slots, then
/// `new != projection` gives dirty — NaN comparison is always True
/// so non-overlap slots automatically flag as dirty).
///
/// @param newGrid      Grid whose voxel set we compute the mask on.
/// @param newSidecar   `[newGrid.totalVoxels]` or
///                     `[newGrid.totalVoxels, C]` sidecar on newGrid.
/// @param oldGrid      Baseline grid for comparison.
/// @param oldSidecar   Sidecar on `oldGrid`, same feature-dim as
///                     `newSidecar`.
///
/// @return Bool tensor of shape `[newGrid.totalVoxels]` on the same
///         device as `newSidecar`.
torch::Tensor
dirtyMaskFromSidecars(const GridBatchData &newGrid,
                      const torch::Tensor &newSidecar,
                      const GridBatchData &oldGrid,
                      const torch::Tensor &oldSidecar);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_DIRTYMASKFROMSIDECARS_H
