// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDPOINTTRUNCATIONSHELL_H
#define FVDB_DETAIL_OPS_BUILDPOINTTRUNCATIONSHELL_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <c10/util/intrusive_ptr.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Build a sparse grid covering the truncation shell of a
///        point cloud.
///
/// Composition of two topology primitives that together define the
/// minimal set of voxels a TSDF-fusion pass must touch:
///
///   1. `buildGridFromPoints(points, voxelSize, origin)` — one active
///      voxel per occupied cell in world space.
///   2. `dilateGrid(numPadVoxels)` where
///      `numPadVoxels = ceil(truncationMargin / voxelSize)` — expand
///      by the truncation-band radius so every voxel within
///      `truncationMargin` of any point is active.
///
/// Shared between the depth-image and LiDAR-point TSDF integrators so
/// both paths hit the same paper-relevant topology primitive. This
/// matters for the paper's "topology ops compose as a reusable
/// primitive" claim (both integrators call this function literally).
///
/// `points` is a JaggedTensor `[B, N_i, 3]` — each batch item may
/// have a different number of input points. `grid` is used only for
/// its per-batch voxel sizes + origins (the truncation-shell output
/// has a different active-voxel set).
///
/// `truncationMargin` is the world-space truncation distance. Caller
/// is responsible for ensuring it's positive and fits within the
/// `MAX_PAD_VOXELS = 16` dilation cap; both are enforced via
/// TORCH_CHECK inside.
c10::intrusive_ptr<GridBatchData>
buildPointTruncationShell(const JaggedTensor &points,
                          const GridBatchData &grid,
                          double truncationMargin);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDPOINTTRUNCATIONSHELL_H
