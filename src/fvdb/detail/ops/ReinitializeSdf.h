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
/// @param field       Per-voxel signed field with finite values and shape (N,) or (N, 1),
///                    N == totalVoxels. It must represent the intended inside/outside regions and
///                    zero crossings. Magnitudes need not be accurate distances, but affect
///                    convergence, accuracy, and sub-voxel surface location. Invalid shapes and
///                    NaN/Inf values raise ValueError. Leave no-data voxels inactive. A voxel whose
///                    value is exactly 0 has a zero frozen sign and is left at 0 by the redistance
///                    (no-data pass-through), though its signed neighbours see it as an interface.
///                    The surface is where the field changes sign between ACTIVE voxels; a grid
///                    whose active values are all one sign has no surface and comes back as the
///                    constant -/+band*vx. Voxels must be isotropic (a ValueError is raised
///                    otherwise).
/// @param band        Narrow-band half-width in voxels. The field is clamped to [-band*vx, band*vx]
///                    each sweep. Inactive neighbours act as a Dirichlet boundary whose value
///                    uses the adjacent voxel's frozen sign during redistancing and its current
///                    sign during smoothing: -band*vx for negative, +band*vx otherwise. An
///                    IndexGrid has a single background slot, so this is how a narrow band with an
///                    inactive interior (e.g. the output of rebuild_narrow_band) is kept solid
///                    rather than hollow.
/// @param redistanceIters  Number of TVD-RK redistancing sweeps. Pass <= 0 to use the default
///                         max(6, round(2.5*band) + 2).
/// @param order       TVD-RK order: 1 (forward Euler), 2 (Heun), or 3 (Shu-Osher).
/// @param smooth      Number of smoothing passes (0 disables smoothing).
/// @param smoothing   Which Laplacian flow each smoothing pass applies (mean-curvature or Taubin).
/// @return A new per-voxel SDF JaggedTensor on the same grid/ordering as @p field.
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
