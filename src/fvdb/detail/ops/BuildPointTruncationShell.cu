// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/ops/BuildPointTruncationShell.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Drops NaN / Inf points, which would otherwise quantise to arbitrary voxels.
torch::Tensor
filterFinitePoints(const torch::Tensor &points) {
    return points.index({at::isfinite(points).all(/*dim=*/1)});
}

} // namespace

c10::intrusive_ptr<GridBatchData>
buildPointTruncationShell(const JaggedTensor &points,
                          const GridBatchData &grid,
                          double truncationMargin) {
    TORCH_CHECK_VALUE(
        truncationMargin > 0.0, "truncationMargin must be > 0, got ", truncationMargin);
    TORCH_CHECK_VALUE(points.num_outer_lists() == grid.batchSize(),
                      "points batch size (",
                      points.num_outer_lists(),
                      ") must equal grid batch size (",
                      grid.batchSize(),
                      ")");

    // Per-batch voxel sizes and origins define the world-to-index
    // transform for the new grid.
    std::vector<nanovdb::Vec3d> voxelSizes;
    std::vector<nanovdb::Vec3d> origins;
    grid.gridVoxelSizesAndOrigins(voxelSizes, origins);

    // Per-batch truncation-band radius (in voxels). `ceil(trunc/voxel)`
    // guarantees every voxel within `truncationMargin` of any point is
    // covered; we use the minimum per-batch voxel length so anisotropic
    // grids dilate enough on the shortest axis.
    constexpr int64_t MAX_PAD_VOXELS = 16;
    std::vector<int64_t> numPadVoxels;
    numPadVoxels.reserve(grid.batchSize());
    for (int64_t i = 0; i < grid.batchSize(); ++i) {
        const double minVoxLengthI = grid.voxelSizeAt(i).min();
        // `std::ceil(trunc / voxel)` snaps to the next integer even when
        // the ratio is mathematically exact -- e.g. a user-requested
        // `trunc=0.015`, `voxel=0.005` (ratio 3 exactly) comes out as
        // ~3.000000067 because `Grid.from_dense` internally rounds
        // `voxel_size` to fp32 along the way (observed stored value:
        // `0.00499999988...`). The naive ceil then yields numPad=4
        // where the intended value is 3, dilating one voxel further than
        // asked.
        //
        // Snap to the lower integer when the fractional part is within
        // an fp32-epsilon-scale tolerance. We use ~4 * float32_eps so
        // the check is scale-invariant (works for both 0.015/0.005 and
        // 15.0/5.0) and accepts both the fp32 rounding artifact above
        // and the much smaller fp64 round-off from the `trunc / voxel`
        // division itself. A genuine input like `trunc=0.0151` (which
        // the user really meant to be ceiled to 4) has a fractional
        // part of ~0.02 in ratio space -- 0.02 >> 5e-7 so the legit
        // ceil case is untouched.
        const double ratio        = truncationMargin / minVoxLengthI;
        const double ratioRounded = std::round(ratio);
        const double tol =
            4.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) * std::max(1.0, ratio);
        const double ceilRatio =
            (std::abs(ratio - ratioRounded) <= tol) ? ratioRounded : std::ceil(ratio);
        const auto numPadVoxelsI = static_cast<int32_t>(ceilRatio);
        TORCH_CHECK_VALUE(numPadVoxelsI > 0,
                          "Number of padding voxels must be positive, got ",
                          numPadVoxelsI,
                          " (truncationMargin=",
                          truncationMargin,
                          ", voxelSize=",
                          minVoxLengthI,
                          ")");
        TORCH_CHECK_VALUE(numPadVoxelsI < MAX_PAD_VOXELS,
                          "Truncation margin (",
                          truncationMargin,
                          ") is too large for grid with voxel size ",
                          minVoxLengthI,
                          ", resulting in too many padding voxels (",
                          numPadVoxelsI,
                          ") which cannot exceed ",
                          MAX_PAD_VOXELS,
                          ". Use a larger voxel size or a smaller truncation margin.");
        numPadVoxels.push_back(numPadVoxelsI);
    }

    // Drop non-finite points per batch item, then build one voxel per
    // occupied cell and dilate it by the truncation band. An item with no
    // finite points yields a voxel-less grid.
    const torch::Tensor offsets = points.joffsets().cpu();
    const auto offsetsAcc       = offsets.accessor<JOffsetsType, 1>();
    std::vector<torch::Tensor> validPoints;
    validPoints.reserve(grid.batchSize());
    for (int64_t i = 0; i < grid.batchSize(); ++i) {
        const int64_t start = offsetsAcc[i];
        const int64_t count = offsetsAcc[i + 1] - start;
        validPoints.push_back(filterFinitePoints(points.jdata().narrow(0, start, count)));
    }

    auto pointGrid = buildGridFromPoints(JaggedTensor(validPoints), voxelSizes, origins);
    return dilateGrid(*pointGrid, numPadVoxels);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
