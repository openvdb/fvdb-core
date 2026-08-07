// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_REINITIALIZESDF_H
#define FVDB_DETAIL_OPS_REINITIALIZESDF_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Laplacian smoothing mode for the SDF reinitialization de-staircase passes.
///        Mirrored by the Python ``fvdb.SmoothingMode`` enum (matching member names).
enum class SmoothingMode : int32_t {
    MEAN_CURVATURE = 0, ///< Mean-curvature flow (umbrella Laplacian, weight 1.0 per pass).
    TAUBIN         = 1, ///< Volume-preserving Taubin smoothing (alternating 0.5 / -0.53 passes).
};

/// @brief Re-initialize a signed per-voxel field into a signed distance field on the *same* grid.
///
/// Redistances the input field to satisfy |grad phi| = 1 (TVD-RK Godunov upwind eikonal solve with
/// a frozen Peng sign), then optionally de-staircases it with mean-curvature or volume-preserving
/// Taubin Laplacian smoothing. The grid topology is unchanged: the returned JaggedTensor has the
/// same per-voxel ordering as the input.
///
/// @param batchHdl    Grid batch defining the sparse topology.
/// @param field       Per-voxel signed field: a floating-point JaggedTensor of shape [B, -1].
/// @param band        Narrow-band half-width in voxels. The field is clamped to [-band*vx, band*vx]
///                    each sweep and the "outside" Dirichlet value for inactive neighbours is
///                    +band*vx.
/// @param redistanceIters  Number of TVD-RK redistancing sweeps. Pass <= 0 to use the default
///                         max(6, round(2.5*band) + 2).
/// @param order       TVD-RK order: 1 (forward Euler), 2 (Heun), or 3 (Shu-Osher).
/// @param smooth      Number of smoothing passes (0 disables smoothing).
/// @param smoothing   Which Laplacian flow each smoothing pass applies (mean-curvature or Taubin).
/// @return A per-voxel SDF: a JaggedTensor of shape [B, -1] with the same dtype, grid, and ordering
///         as @p field.
JaggedTensor reinitializeSdf(const GridBatchData &batchHdl,
                             const JaggedTensor &field,
                             int band,
                             int redistanceIters,
                             int order,
                             int smooth,
                             SmoothingMode smoothing);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_REINITIALIZESDF_H
