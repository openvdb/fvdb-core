// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_PERSISTENTTSDFSTATE_H
#define FVDB_DETAIL_OPS_PERSISTENTTSDFSTATE_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief A generic grow-on-touch state holder for per-voxel sidecar tensors
///        that ride on top of a monotonically-growing `nanovdb::ValueOnIndex`
///        grid.
///
/// `PersistentTSDFState` pairs a `GridBatchData` (the *live grid*) with a
/// fixed set of sidecar tensors (`tsdf`, `weights`, optional `features`)
/// indexed by the grid's active voxel linear index. Each call to `grow` or
/// `growFromGrid` expands the live grid to the union of its current voxels
/// and the caller-supplied voxel set, reallocates the sidecars, copies
/// surviving voxels' values into their new positions via `ops::inject`,
/// and zero-initialises slots for genuinely new voxels.
///
/// The class is intentionally TSDF-agnostic beyond the sidecar names: the
/// "tsdf/weights/features" triple is the minimum surface area the depth and
/// LiDAR integrators both need. Callers who want to carry extra sidecars
/// can stack additional `PersistentTSDFState`-like wrappers on the same
/// underlying grid.
///
/// Why this class exists (paper-framing note): there are TSDF-fusion
/// workloads in which the output topology naturally persists across
/// observations (canonical incremental RGB-D / LiDAR fusion), and other
/// workloads where two independently-built grids want to be composed
/// one-shot (non-persistent union of attribute fields, runtime-loaded
/// terrain tiles, etc.). The one-shot pattern is served by the existing
/// `mergeGrids` primitive; the persistent pattern is served by this
/// class. Both patterns compose the same nanoVDB `voxelsToGrid +
/// mergeGrids + inject` building blocks -- only the outer shape differs.
class PersistentTSDFState {
  public:
    /// @brief Construct a new state from an initial grid + sidecar tensors.
    ///
    /// The initial grid may have zero voxels (for a from-scratch workflow)
    /// or contain a seed topology. Sidecar tensors must be 1-D/2-D with
    /// `size(0) == grid->totalVoxels()`.
    /// @param grid The initial grid topology (non-null, single-batch preferred).
    /// @param tsdf The initial TSDF sidecar, shape `[totalVoxels]`.
    /// @param weights The initial weight sidecar, shape `[totalVoxels]`.
    /// @param features Optional `[totalVoxels, featureDim]` sidecar; pass
    ///                 `std::nullopt` for no-features workloads.
    PersistentTSDFState(c10::intrusive_ptr<GridBatchData> grid,
                        torch::Tensor tsdf,
                        torch::Tensor weights,
                        std::optional<torch::Tensor> features = std::nullopt);

    // Move-only: like `GridBatchData`, we forbid copy to avoid accidental
    // sidecar-tensor aliasing (the tensors are mutated in-place by the
    // shell-filtered integrate kernels).
    PersistentTSDFState(const PersistentTSDFState &)            = delete;
    PersistentTSDFState &operator=(const PersistentTSDFState &) = delete;
    PersistentTSDFState(PersistentTSDFState &&)                 = default;
    PersistentTSDFState &operator=(PersistentTSDFState &&)      = default;
    ~PersistentTSDFState()                                      = default;

    /// @brief Expand the live grid to include the voxel ijk set in
    ///        `newVoxelIjks`. Fully equivalent to
    ///        `growFromGrid(voxelsToGrid(newVoxelIjks))`.
    /// @param newVoxelIjks A `JaggedTensor` of integer voxel coordinates
    ///                     with element shape `[-1, 3]` and batch size 1.
    void grow(const JaggedTensor &newVoxelIjks);

    /// @brief Expand the live grid to the union of its current voxels and
    ///        `shellGrid`. This is the primary entry point used by the
    ///        depth / LiDAR integrators, both of which have already built
    ///        a shell grid via `buildPointTruncationShell`.
    ///
    /// No-op when `shellGrid.totalVoxels() == 0`.
    /// No-op when the merged grid has the same active-voxel count as the
    /// current live grid (the shell was a subset). In that case the
    /// existing sidecar tensors and grid handle are retained unmodified,
    /// which is the steady-state fast path on bounded-scene trajectories.
    ///
    /// @param shellGrid The shell (or any other) grid whose active voxels
    ///                  should be merged into the live grid.
    void growFromGrid(const GridBatchData &shellGrid);

    /// @brief Drop the live grid and sidecars back to an empty, zero-voxel
    ///        state. Retains the voxel size and origin of the current
    ///        live grid so subsequent `grow()` calls quantise against the
    ///        same coordinate frame.
    void reset();

    /// @brief Current active voxel count in the live grid.
    int64_t
    activeVoxelCount() const {
        return mGrid->totalVoxels();
    }

    /// @brief Access the live grid by reference (stable pointer semantics
    ///        within a single `grow` call; do not retain across grows).
    GridBatchData &
    grid() {
        return *mGrid;
    }
    const GridBatchData &
    grid() const {
        return *mGrid;
    }
    const c10::intrusive_ptr<GridBatchData> &
    gridPtr() const {
        return mGrid;
    }

    torch::Tensor &
    tsdf() {
        return mTsdf;
    }
    const torch::Tensor &
    tsdf() const {
        return mTsdf;
    }

    torch::Tensor &
    weights() {
        return mWeights;
    }
    const torch::Tensor &
    weights() const {
        return mWeights;
    }

    /// @brief Whether a features sidecar is attached.
    bool
    hasFeatures() const {
        return mHasFeatures;
    }

    /// @brief Access the features sidecar. Valid only when `hasFeatures()`.
    torch::Tensor &
    features() {
        return mFeatures;
    }
    const torch::Tensor &
    features() const {
        return mFeatures;
    }

    /// @brief JaggedTensor view of the TSDF sidecar that matches the
    ///        current live grid's batch layout. Convenience wrapper
    ///        around `grid().jaggedTensor(tsdf())` used by callers that
    ///        hand off to the existing JaggedTensor-accepting kernels.
    JaggedTensor
    tsdfJagged() const {
        return mGrid->jaggedTensor(mTsdf);
    }
    JaggedTensor
    weightsJagged() const {
        return mGrid->jaggedTensor(mWeights);
    }
    JaggedTensor
    featuresJagged() const {
        return mGrid->jaggedTensor(mFeatures);
    }

  private:
    c10::intrusive_ptr<GridBatchData> mGrid;
    torch::Tensor mTsdf;
    torch::Tensor mWeights;
    torch::Tensor mFeatures;    // shape `[totalVoxels, 0]` when no features
    bool mHasFeatures = false;
};

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_PERSISTENTTSDFSTATE_H
