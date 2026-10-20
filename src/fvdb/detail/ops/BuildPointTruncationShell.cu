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

// Drops points that would inflate the shell topology.
//
// `unprojectDepthmapKernel` emits about 1% (Replica room0) of finite but
// far-field coordinates, tens to millions of metres from the scene, where
// the fp32 unprojection loses precision. A fixed clamp cannot serve both
// room-scale depth (~5 m) and outdoor LiDAR (~100 m), so the clamp adapts
// to the scene. It is `kAdaptiveClampHeadroom` times the median of
// `max|coord|` over the finite points. The median ignores the garbage tail, which drags
// p99 to ~50 m on a 4 m room. Replica gets a ~12 m clamp. The floor keeps
// tiny scenes intact, and the 100 km ceiling keeps far-from-origin LiDAR
// trajectories (Mai City, KITTI) while rejecting the 10^4+ m tail.
constexpr double kAdaptiveClampHeadroom = 8.0;
constexpr double kMinAdaptiveClamp      = 4.0;
constexpr double kMaxAdaptiveClamp      = 100000.0;

torch::Tensor
filterValidPoints(const torch::Tensor &points) {
    // Works in the caller's dtype to avoid an fp64 copy. Non-finite points
    // go first so the median only sees real coordinates.
    const torch::Tensor finite_mask = at::isfinite(points).all(/*dim=*/1);
    const torch::Tensor finite_pts  = points.index({finite_mask});
    if (finite_pts.size(0) == 0) {
        return points.new_empty({0, 3});
    }

    // `at::quantile` has a ~2^24-row sort limit, so large inputs are
    // stride-subsampled to 1 M rows, which pins the median to within a
    // centimetre.
    const torch::Tensor max_abs     = std::get<0>(finite_pts.abs().max(/*dim=*/1)); // [N_fin]
    constexpr int64_t kPctSampleCap = 1 << 20;                                      // 1 M
    torch::Tensor max_abs_for_quantile;
    if (max_abs.size(0) > kPctSampleCap) {
        const int64_t stride = (max_abs.size(0) + kPctSampleCap - 1) / kPctSampleCap;
        max_abs_for_quantile =
            max_abs.index({torch::indexing::Slice(0, torch::indexing::None, stride)}).contiguous();
    } else {
        max_abs_for_quantile = max_abs;
    }
    // `at::quantile` takes only float or double.
    const torch::Tensor max_abs_f32 = max_abs_for_quantile.scalar_type() == torch::kHalf
                                          ? max_abs_for_quantile.to(torch::kFloat32)
                                          : max_abs_for_quantile;
    const double median             = at::quantile(max_abs_f32, 0.50).item<double>();
    const double clamp =
        std::min(kMaxAdaptiveClamp, std::max(kMinAdaptiveClamp, kAdaptiveClampHeadroom * median));
    const torch::Tensor bounded_mask = (finite_pts.abs() < clamp).all(/*dim=*/1);

    return finite_pts.index({bounded_mask}).contiguous();
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

    // Drop non-finite and far-field points per batch item, then build one
    // voxel per occupied cell and dilate it by the truncation band. An item
    // with no surviving points yields a voxel-less grid.
    const torch::Tensor offsets = points.joffsets().cpu();
    const auto offsetsAcc       = offsets.accessor<JOffsetsType, 1>();
    std::vector<torch::Tensor> validPoints;
    validPoints.reserve(grid.batchSize());
    for (int64_t i = 0; i < grid.batchSize(); ++i) {
        const int64_t start = offsetsAcc[i];
        const int64_t count = offsetsAcc[i + 1] - start;
        validPoints.push_back(filterValidPoints(points.jdata().narrow(0, start, count)));
    }

    auto pointGrid = buildGridFromPoints(JaggedTensor(validPoints), voxelSizes, origins);
    return dilateGrid(*pointGrid, numPadVoxels);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
