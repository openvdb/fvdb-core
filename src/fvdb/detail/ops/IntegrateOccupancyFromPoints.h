// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_INTEGRATEOCCUPANCYFROMPOINTS_H
#define FVDB_DETAIL_OPS_INTEGRATEOCCUPANCYFROMPOINTS_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Integrate a batch of LiDAR / range-sensor point clouds into
///        a log-odds **occupancy** volume via per-point ray-walking.
///
/// Sister primitive to `integrateTSDFFromPoints`: same shell-allocator
/// (buildPointTruncationShell -> mergeGrids) and same HDDA ray-walk
/// pattern, but the per-voxel update is a Bayesian log-odds
/// accumulation instead of the TSDF's running weighted-average.
///
/// For each ray origin -> endpoint:
///   - Voxels within `truncationMargin` of the endpoint ("hit band")
///     get `log_odds += logOddsHit` per ray that passes through them.
///   - Voxels on the sensor-ray side of the endpoint ("free band")
///     and within `truncationMargin` of the ray get
///     `log_odds += logOddsMiss`.
///   - Voxels beyond the endpoint by more than `truncationMargin`
///     are "unknown" and left alone.
///   - After all rays are processed, `log_odds` is clamped to
///     `[logOddsMin, logOddsMax]`.
///
/// The stored value IS the log-odds. To get probability, apply a
/// sigmoid host-side: `p = 1 / (1 + exp(-log_odds))`. Storing log-
/// odds (rather than probabilities) is the standard choice because
/// Bayesian updates compose as additions in log space and don't
/// require per-update division.
///
/// Paper-framing: this is the paper's fifth application of the
/// nanoVDB topology-op vocabulary (after depth TSDF, LiDAR TSDF, MC,
/// ESDF). Same substrate (voxelsToGrid + mergeGrids + an HDDA ray-
/// walk) with a different per-voxel update rule. Demonstrates the
/// orthogonality claim: nvblox's `OCCUPANCY` vs `TSDF` integrator is
/// a whole-different-allocator distinction; ours is a
/// different-inner-loop distinction.
///
/// **Why ray-walking and not projective-per-pixel** (nvblox's default):
/// nvblox's occupancy integrator projects voxels into the depth
/// frame and updates based on (voxel_depth vs pixel_depth). We use
/// the same ray-walk as our TSDF-from-points integrator instead, to
/// keep the comparison with nvblox LiDAR honest (nvblox also walks
/// rays for LiDAR input). The two yield equivalent probabilities
/// modulo the LiDAR's discretisation-to-range-image step.
///
/// @param grid  The existing grid to integrate into. The output grid
///              is the union of this and the truncation shell of the
///              new points.
/// @param truncationMargin  World-space distance defining the hit
///                          band (voxels within this distance of the
///                          endpoint are "hit"). Also drives the
///                          shell allocator's dilation.
/// @param points  JaggedTensor [B, N_i, 3] of world-space point
///                positions.
/// @param sensorOrigins  [B, 3] per-batch sensor origin in world
///                       space.
/// @param logOdds  JaggedTensor [totalVoxels, 1] — current log-odds
///                 values on `grid`.
/// @param logOddsHit  Increment per ray endpoint observation
///                    (typical: +0.85).
/// @param logOddsMiss  Increment per ray-pass-through observation
///                     (typical: -0.40, negative).
/// @param logOddsMin  Lower clamp bound (typical: -4.0).
/// @param logOddsMax  Upper clamp bound (typical: +4.0).
///
/// @return (newGrid, newLogOdds) on the union grid.
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
integrateOccupancyFromPoints(const c10::intrusive_ptr<GridBatchData> grid,
                             const double truncationMargin,
                             const JaggedTensor &points,
                             const torch::Tensor &sensorOrigins,
                             const JaggedTensor &logOdds,
                             const double logOddsHit,
                             const double logOddsMiss,
                             const double logOddsMin,
                             const double logOddsMax);

/// @brief Batched version of `integrateOccupancyFromPoints`: integrate
///        N LiDAR sweeps into a single persistent occupancy volume.
///
/// Mirrors `integrateTSDFFromPointsFrames` exactly but with log-odds
/// updates instead of running-weighted-avg. The topology grows
/// incrementally frame-by-frame; the final `(grid, logOdds)` is the
/// union over all frames' truncation shells with the log-odds
/// accumulated value.
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
integrateOccupancyFromPointsFrames(const c10::intrusive_ptr<GridBatchData> grid,
                                   const double truncationMargin,
                                   const std::vector<torch::Tensor> &pointsPerFrame,
                                   const torch::Tensor &sensorOrigins,
                                   const JaggedTensor &logOdds,
                                   const double logOddsHit,
                                   const double logOddsMiss,
                                   const double logOddsMin,
                                   const double logOddsMax);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_INTEGRATEOCCUPANCYFROMPOINTS_H
