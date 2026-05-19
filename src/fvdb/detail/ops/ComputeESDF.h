// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COMPUTEESDF_H
#define FVDB_DETAIL_OPS_COMPUTEESDF_H

#include <fvdb/GridBatchData.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Compute a Euclidean Signed Distance Field (ESDF) from an
///        integrated narrow-band TSDF.
///
/// The ESDF extends the TSDF's narrow-band signed distances outward (and
/// inward) across a wider band via monotone 26-neighbour min-propagation,
/// producing per-voxel world-unit signed distances `d` with
/// `|d| <= max_distance`. This is the paper's **second application** of
/// the nanoVDB topology-op vocabulary (the first being depth/LiDAR TSDF):
///
///   - `dilateGrid`  (once, by `ceil(max_distance / voxel_size) + 1`) to
///     allocate the ESDF support band around the TSDF zero-crossing shell.
///   - A custom VBM-stencil kernel, launched N times, that reads each
///     voxel's 26-neighbourhood and computes
///     `d' = sign(d_n) * (|d_n| + ||offset|| * voxel_size)`
///     against the current value. N = `ceil(max_distance / voxel_size) + 2`
///     is sufficient for 26-connectivity convergence; more-than-needed
///     sweeps are cheap (each voxel's min is monotone, so extra sweeps
///     are no-ops).
///   - `pruneGrid` (once, optional) to drop voxels the wavefront never
///     reached (still at sentinel value). Off by default so the returned
///     grid matches the dilated support and the caller decides whether
///     to prune.
///
/// Seeding: voxels with `weights[v] > weight_threshold` AND
/// `|tsdf[v]| < 1 - eps` (i.e., the TSDF is not saturated at the
/// truncation boundary) are used as wavefront sources with initial
/// distance `tsdf[v] * truncation_distance` (world units). Saturated
/// voxels (|tsdf|==1 after clamping) carry no useful distance
/// information and are filled by the wavefront; unobserved voxels
/// (|weights|==0) likewise.
///
/// Ablation knob: `use_vbm == false` replaces the VBM per-active-voxel
/// iteration with a per-leaf-slot iteration so the two cost models can
/// be compared directly on the same workload. Output is bit-identical
/// (both code paths execute the same
/// `min(d, d_n + ||offset|| * voxel_size)` formula in the same order
/// per voxel).
///
/// @param gridBatch               Input TSDF grid topology (single batch).
/// @param tsdf                    `[totalVoxels]` fp32 normalized TSDF
///                                in `[-1, +1]` (fvdb's `integrate_tsdf*`
///                                convention). Other scalar types fall
///                                back to float by internal cast in M5.
/// @param weights                 `[totalVoxels]` fp32 integration weights.
/// @param truncation_distance     Truncation margin in world units (the
///                                `T` of `tsdf = clip(d_world / T, -1, 1)`).
/// @param max_distance            ESDF support radius in world units.
/// @param weight_threshold        Voxels with `weights <= threshold` are
///                                not used as wavefront sources.
/// @param prune_unreached         If true, drop voxels the wavefront
///                                never reached (still at sentinel).
/// @param use_vbm                 Iteration-pattern ablation knob.
///
/// @return `(esdf_grid, esdf_values)` where `esdf_values` is
///         `[esdf_grid.totalVoxels]` fp32 world-unit signed distances,
///         with `|esdf[i]| <= max_distance + voxel_size` at wavefront
///         terminations.
std::tuple<c10::intrusive_ptr<GridBatchData>, torch::Tensor>
computeESDF(const GridBatchData &gridBatch,
            const torch::Tensor &tsdf,
            const torch::Tensor &weights,
            double truncation_distance,
            double max_distance,
            double weight_threshold,
            bool prune_unreached,
            bool use_vbm);

/// @brief Monotone-incremental ESDF: extend a previous ESDF to cover
///        the current TSDF grid without paying the full-from-scratch
///        wavefront cost on every frame.
///
/// Pattern (the paper's "same primitives, different composition"
/// argument): instead of restarting from a sentinel-filled buffer, we
/// reuse the previous frame's ESDF values as a warm-start for the
/// wavefront. Because the 26-neighbour min-propagation is monotone, a
/// warm-started sweep converges in fewer effective iterations than a
/// cold start -- and even better, previously-converged values in
/// regions the current frame didn't touch are preserved byte-for-byte.
///
/// Composition (exclusively topology-op primitives + the same two
/// kernels as one-shot):
///
///   1. `dilateGrid(gridBatch, K)`  to size the minimum new support.
///   2. `mergeGrids(dilated_support, prevEsdfGrid)` so the output
///      covers BOTH the new support AND the previous ESDF's
///      support (handles the monotonically-growing-scene case
///      cleanly without dropping previously-computed data).
///   3. Allocate `esdf_new[|merged|]` initialized to sentinel.
///   4. `inject(esdfGrid, prevEsdfGrid, esdf_new, prevEsdf)` to copy
///      previous values into their (possibly shifted) positions in
///      the merged grid.
///   5. Seed from current TSDF (same `esdfSeedKernel` as one-shot;
///      overwrites previous value at seed voxels with the current-
///      frame's signed distance, which is correct since seeds are by
///      definition exact).
///   6. Same sweep loop as one-shot (same VBM / per-leaf kernels).
///   7. Same clamp + optional prune.
///
/// **Correctness assumption (monotone only)**: we assume distances
/// decrease monotonically between frames -- i.e. surfaces are added
/// or refined but never removed. This matches standard TSDF-fusion
/// workflows where the sensor adds observations over time. If surfaces
/// disappear (dynamic objects, noise-resolved phantom surfaces), the
/// incremental ESDF can lock in stale-lower distances. For those
/// cases, call `computeESDF` one-shot on a fresh schedule (e.g. every
/// M frames) as a correction pass. See
/// `sessions/2026-04-23_esdf_one_shot.md` section on "the one subtle
/// correctness trap" for the FIESTA-style parent-witness alternative
/// we explicitly chose NOT to implement here.
///
/// When `prevEsdfGrid.totalVoxels() == 0`, falls through to one-shot
/// semantics (useful for the first frame of an incremental session).
///
/// When `dirtyMask.defined()` (non-trivial bool tensor of shape
/// `[gridBatch.totalVoxels()]`): only voxels with
/// `dirtyMask[v] == true` seed the wavefront. This exposes nvblox-
/// style "dirty-region update" cost scaling (proportional to the
/// number of changed voxels, not the full grid) without any
/// library-internal block-dirty state. Combine with
/// `ops::dirtyMaskFromSidecars(newGrid, newWeights, oldGrid,
/// oldWeights)` to derive the mask from a TSDF integration pair.
std::tuple<c10::intrusive_ptr<GridBatchData>, torch::Tensor>
computeESDFIncremental(const GridBatchData &gridBatch,
                       const torch::Tensor &tsdf,
                       const torch::Tensor &weights,
                       const GridBatchData &prevEsdfGrid,
                       const torch::Tensor &prevEsdf,
                       double truncation_distance,
                       double max_distance,
                       double weight_threshold,
                       bool prune_unreached,
                       bool use_vbm,
                       const torch::Tensor &dirtyMask);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COMPUTEESDF_H
