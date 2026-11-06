// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INTEGRATETSDFFROMPOINTS_H
#define FVDB_DETAIL_OPS_INTEGRATETSDFFROMPOINTS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Integrate a batch of LiDAR / range-sensor point clouds into
///        a TSDF volume via per-point ray-walking (no range-image
///        proxy).
///
/// For each point we walk voxels from `sensorOrigins[b]` toward the
/// point endpoint via HDDA over the union-grid topology, updating the
/// TSDF and weight at each active voxel via lock-free atomicAdd in
/// running-sum form. The topology for the new scan is constructed by
/// `buildPointTruncationShell(points, grid, truncationMargin)` — the
/// same primitive the depth-image integrator uses — then merged with
/// the existing grid.
///
/// This mirrors the VDBFusion / nvblox LiDAR integration surface so
/// the cross-library comparison remains apples-to-apples.
///
/// @param grid  The existing grid to integrate into. The output grid
///              is the union of this and the truncation shell of the
///              new points.
/// @param truncationMargin  World-space truncation distance.
/// @param points  JaggedTensor [B, N_i, 3] of world-space point
///                positions. Each batch item may have a different
///                `N_i`.
/// @param sensorOrigins  [B, 3] per-batch sensor origin in world
///                       space (one origin per sweep; per-ray
///                       origins are future work).
/// @param tsdf  JaggedTensor [totalVoxels, 1] — TSDF values on `grid`.
/// @param weights  JaggedTensor [totalVoxels, 1] — integration
///                 weights on `grid`.
/// @param carveFreeSpace  If true, voxels observed to be in front of
///                        the endpoint (outside the truncation band)
///                        get TSDF = +1 and weight = 1. Matches
///                        VDBFusion / nvblox default behaviour.
///
/// @return (newGrid, newTsdf, newWeights) all on the union grid.
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFFromPoints(const c10::intrusive_ptr<GridBatchData> grid,
                        const double truncationMargin,
                        const JaggedTensor &points,
                        const torch::Tensor &sensorOrigins,
                        const JaggedTensor &tsdf,
                        const JaggedTensor &weights,
                        bool carveFreeSpace);

/// @brief Like `integrateTSDFFromPoints` but also blends a per-point
///        feature vector (e.g. RGB colour) into per-voxel features.
///
/// Feature dtype must match `tsdf.scalar_type()` OR be `uint8` (for
/// RGB colours — matches the convention used by the depth-image
/// integrator's `integrateTSDFWithFeatures`).
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor, JaggedTensor>
integrateTSDFFromPointsWithFeatures(const c10::intrusive_ptr<GridBatchData> grid,
                                    const double truncationMargin,
                                    const JaggedTensor &points,
                                    const torch::Tensor &sensorOrigins,
                                    const JaggedTensor &tsdf,
                                    const JaggedTensor &features,
                                    const JaggedTensor &weights,
                                    const JaggedTensor &pointFeatures,
                                    bool carveFreeSpace);

/// @brief Batched version of `integrateTSDFFromPoints`: integrate N
///        LiDAR sweeps into a single persistent TSDF volume without
///        paying the Python<->C++ round-trip overhead each frame.
///
/// Semantics are identical to N sequential calls to
/// `integrateTSDFFromPoints(grid, trunc, points[i], sensorOrigins[i],
/// tsdf, weights, carveFreeSpace)`: the topology grows incrementally
/// frame-by-frame (exactly the same way the per-frame loop does), and
/// the final (grid, tsdf, weights) is the union over all frames'
/// truncation shells with the ray-walk integrated values. Bit-
/// identical to the sequential reference is pinned by
/// `test_integrate_tsdf_from_points_frames_matches_sequential`.
///
/// The win over a Python-level `for` loop is purely the removal of
/// per-frame JaggedTensor / GridBatchData rewrapping + Python
/// dispatch overhead, which is most visible on long outdoor LiDAR
/// trajectories with many short sweeps per second.
///
/// @param grid  Initial grid topology (seed). May be empty, a 1x1x1
///              dense placeholder, or a pre-populated grid from
///              previous calls.
/// @param truncationMargin  World-space truncation distance.
/// @param pointsPerFrame  Per-frame point clouds, `pointsPerFrame[i]`
///                        is `[N_i, 3]` in world frame. Count
///                        determines N.
/// @param sensorOrigins  `[N, 3]` per-frame sensor origin in world
///                       space, same as the single-frame API
///                       accepts a `[batchSize, 3]` tensor.
/// @param tsdf  `[totalVoxels]` TSDF values on `grid`.
/// @param weights  `[totalVoxels]` integration weights on `grid`.
/// @param carveFreeSpace  If true, free-space voxels in front of the
///                        endpoint get TSDF=+1, weight=1. Matches
///                        VDBFusion / nvblox default behaviour.
///
/// @return (newGrid, newTsdf, newWeights) on the final union grid.
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, JaggedTensor>
integrateTSDFFromPointsFrames(const c10::intrusive_ptr<GridBatchData> grid,
                              const double truncationMargin,
                              const std::vector<torch::Tensor> &pointsPerFrame,
                              const torch::Tensor &sensorOrigins,
                              const JaggedTensor &tsdf,
                              const JaggedTensor &weights,
                              bool carveFreeSpace);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INTEGRATETSDFFROMPOINTS_H
