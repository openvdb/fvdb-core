// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// DirtyMaskFromSidecars.cu
//
// Standalone utility that computes a per-voxel "dirty" bool mask on
// newGrid from two (grid, sidecar) pairs. Built entirely on top of
// `ops::inject` — no new CUDA kernels, just one inject + one tensor
// comparison.
//
// Paper-framing: this is a 40-LoC C++ helper that the paper cites as
// the backbone of fvdb's dirty-region ESDF update. Contrast nvblox's
// dirty-block tracking, which lives inside the block-hash allocator
// and isn't user-visible. Ours is a torch tensor the user can pass
// to `compute_esdf_incremental` (new `dirty_mask` arg) or compose
// with their own predicates.

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/DirtyMaskFromSidecars.h>
#include <fvdb/detail/ops/Inject.h>

#include <ATen/TensorOperators.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <torch/types.h>

namespace fvdb::detail::ops {

torch::Tensor
dirtyMaskFromSidecars(const GridBatchData &newGrid,
                      const torch::Tensor &newSidecar,
                      const GridBatchData &oldGrid,
                      const torch::Tensor &oldSidecar) {
    TORCH_CHECK_VALUE(newSidecar.is_floating_point(),
                      "dirtyMaskFromSidecars: newSidecar must be "
                      "floating-point (NaN-sentinel trick requires it)");
    TORCH_CHECK_VALUE(oldSidecar.scalar_type() == newSidecar.scalar_type(),
                      "dirtyMaskFromSidecars: newSidecar and oldSidecar "
                      "must share dtype; got ", newSidecar.scalar_type(),
                      " and ", oldSidecar.scalar_type());
    TORCH_CHECK_VALUE(newGrid.device() == oldGrid.device(),
                      "dirtyMaskFromSidecars: newGrid and oldGrid must "
                      "be on the same device");
    TORCH_CHECK_VALUE(newSidecar.device() == newGrid.device(),
                      "dirtyMaskFromSidecars: newSidecar must be on the "
                      "same device as newGrid");
    TORCH_CHECK_VALUE(oldSidecar.device() == oldGrid.device(),
                      "dirtyMaskFromSidecars: oldSidecar must be on the "
                      "same device as oldGrid");
    TORCH_CHECK_VALUE(newSidecar.size(0) == newGrid.totalVoxels(),
                      "dirtyMaskFromSidecars: newSidecar size(0) (",
                      newSidecar.size(0),
                      ") must match newGrid totalVoxels (",
                      newGrid.totalVoxels(), ")");
    TORCH_CHECK_VALUE(oldSidecar.size(0) == oldGrid.totalVoxels(),
                      "dirtyMaskFromSidecars: oldSidecar size(0) (",
                      oldSidecar.size(0),
                      ") must match oldGrid totalVoxels (",
                      oldGrid.totalVoxels(), ")");
    TORCH_CHECK_VALUE(newSidecar.dim() == oldSidecar.dim(),
                      "dirtyMaskFromSidecars: newSidecar and oldSidecar "
                      "must have the same number of dimensions");
    if (newSidecar.dim() > 1) {
        TORCH_CHECK_VALUE(newSidecar.sizes().slice(1) ==
                              oldSidecar.sizes().slice(1),
                          "dirtyMaskFromSidecars: feature dims must match");
    }

    const c10::cuda::CUDAGuard deviceGuard(newSidecar.device());

    // Fast-path: oldGrid is empty. Every voxel in newGrid is "new" →
    // entirely dirty. Avoids calling inject with a zero-voxel source.
    if (oldGrid.totalVoxels() == 0) {
        return torch::ones({newGrid.totalVoxels()},
                           torch::TensorOptions()
                               .dtype(torch::kBool)
                               .device(newSidecar.device()));
    }

    // NaN-init the projection target. `ops::inject` writes only
    // ijk-overlap slots, so non-overlap slots keep their NaN — and
    // NaN comparison with anything returns True, giving us "not in
    // old grid" ⇒ dirty automatically.
    torch::Tensor projected = torch::full(
        newSidecar.sizes(),
        std::nan(""),
        newSidecar.options());

    JaggedTensor projectedJt = newGrid.jaggedTensor(projected);
    JaggedTensor oldJt       = oldGrid.jaggedTensor(oldSidecar);
    ops::inject(newGrid, oldGrid, projectedJt, oldJt);
    // `ops::inject` may swap the underlying tensor reference inside
    // the dst JaggedTensor (see PersistentTSDFState.cu:59-61). Pull
    // the possibly-new tensor back out.
    projected = projectedJt.jdata();

    // Per-voxel, per-channel bool: True if new differs from projected.
    // NaN != anything (even NaN) is True, so non-overlap voxels
    // automatically flag as dirty.
    torch::Tensor diff = projected.ne(newSidecar);

    // Multi-channel: reduce via "any channel differs".
    while (diff.dim() > 1) {
        diff = diff.any(/*dim=*/-1);
    }

    return diff;
}

} // namespace fvdb::detail::ops
