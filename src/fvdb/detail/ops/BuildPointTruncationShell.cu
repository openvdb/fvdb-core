// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/ActiveGridCoords.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/ops/BuildPointTruncationShell.h>

#include <c10/util/Exception.h>
#include <torch/types.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Generate a dense (2*numPad+1)^3 x 3 int32 tensor of integer lattice offsets in
// [-numPad, numPad]^3. Applied as a broadcast add on top of a base-voxel list
// this is equivalent to `numPad` successive NN_FACE_EDGE_VERTEX dilations,
// which is the stencil fvdb uses for truncation-band topology.
// Currently unused because the voxel-path shell build separates the 3-D
// stencil into three 1-D axis expansions; kept for the leaf-shell and
// potential CPU fallback paths.
[[maybe_unused]] torch::Tensor
makeStencilOffsets(int64_t numPad, torch::Device device) {
    const torch::TensorOptions optI32 =
        torch::TensorOptions().dtype(torch::kInt32).device(device);
    const torch::Tensor axis = torch::arange(-numPad, numPad + 1, optI32);
    const auto grid          = at::meshgrid({axis, axis, axis}, "ij");
    return torch::stack(
               {grid[0].flatten(), grid[1].flatten(), grid[2].flatten()}, 1)
        .contiguous(); // [(2k+1)^3, 3]
}

// Dedupe an [N, 3] int32 ijk tensor via lexicographic unique on dim 0.
torch::Tensor
uniqueIjk(const torch::Tensor &ijk) {
    TORCH_CHECK(ijk.dim() == 2 && ijk.size(1) == 3, "uniqueIjk expects [N, 3]");
    const auto uniq = at::unique_dim(ijk, /*dim=*/0, /*sorted=*/false,
                                     /*return_inverse=*/false,
                                     /*return_counts=*/false);
    return std::get<0>(uniq).contiguous();
}

// Tree-merge a list of [N_i, 3] int32 unique-voxel tensors into a single
// deduped unique-voxel tensor. Pairwise-merges the list in log2 rounds,
// which bounds peak transient memory to ~2x the largest partial instead
// of (sum of partials) + scratch that a single final-cat-plus-unique
// would need. Unused since the voxel-granularity path switched to
// packed-key int64 tree-merging (kept for parity with CPU path if it
// is ever restored).
[[maybe_unused]] torch::Tensor
treeMergeUniqueIjk(std::vector<torch::Tensor> shards) {
    while (shards.size() > 1) {
        std::vector<torch::Tensor> next;
        next.reserve((shards.size() + 1) / 2);
        for (size_t i = 0; i + 1 < shards.size(); i += 2) {
            next.push_back(uniqueIjk(
                torch::cat({shards[i], shards[i + 1]}, /*dim=*/0)));
            // Eagerly release input tensors now that they've been merged
            // so torch's caching allocator can reuse their blocks for the
            // next round. Without this the allocator holds the shard
            // memory live until the enclosing vector dies, which roughly
            // doubles peak usage at large N.
            shards[i]     = torch::Tensor();
            shards[i + 1] = torch::Tensor();
        }
        if (shards.size() % 2 == 1) {
            next.push_back(std::move(shards.back()));
        }
        shards = std::move(next);
    }
    return shards.empty() ? torch::Tensor() : std::move(shards[0]);
}

// World-space point filter with scene-adaptive clamp.
//
// Motivation: `unprojectDepthmapKernel` emits a non-trivial fraction
// (~1% for Replica room0) of "garbage" unprojected coordinates — finite
// values ranging from tens of metres to millions of metres — for pixels
// where the float32 inv-projection + cam-to-world matmul chain loses
// precision. Reproducing the exact same depth + pose + intrinsics in fp32
// torch on the same inputs produces 0% garbage, so the issue is specific
// to the CUDA kernel (likely an FMA-accuracy / denormal interaction we
// haven't yet tracked down; see research journal).
//
// A *static* clamp is brittle across workloads: ±10 m is fine for Replica
// but will reject valid LiDAR at ±80 m; ±1024 m keeps the KITTI case but
// readmits enough Replica garbage to blow up the downstream shell by 5x.
//
// Strategy: compute the p99 of `max|coord|` over the finite points, use
// `k_sigma * p99` as the upper bound, and reject everything beyond. p99
// adapts to scene scale (Replica ~5 m, KITTI ~60 m, autonomous-car LiDAR
// ~100 m), and the 4x headroom keeps every realistic "last 1%" point
// inside while still aggressively cutting the far-field garbage tail.
// NaN / Inf are rejected first so the percentile only sees finite values.
//
// `kAdaptiveClampHeadroom` and the hard ceiling are chosen empirically:
//   - On Replica room0, p99 is already polluted by the garbage tail
//     (p99 ~= 50 m even though the real scene is only 4 m wide); using
//     p50 (median) ~= 1.5 m is much more robust. `headroom=8x` over
//     median gives a 12 m clamp for Replica, plenty for any room-scale
//     scene while still rejecting garbage at 50+ m.
//   - The hard ceiling (300 m) protects against workloads where even
//     the median drifts high (e.g. if a whole frame's pixels are
//     garbage); 300 m is larger than any real indoor/outdoor TSDF
//     workload but small enough that the resulting voxel set is still
//     tractable.
//   - The hard floor (4 m) handles tiny scenes / initialisation edge
//     cases so we never clamp below the plausible scene extent.
constexpr double kAdaptiveClampHeadroom = 8.0;
constexpr double kMinAdaptiveClamp      = 4.0;       // never clamp below 4 m
// Raised from the room-scale default (300 m) so outdoor LiDAR
// datasets (Mai City, KITTI, etc.) with trajectories that wander
// far from the world origin don't get their entire per-frame point
// cloud rejected. The filter's primary purpose is to drop the
// fp32-precision garbage tail from `unprojectDepthmapKernel` (10^4-
// 10^38 m coordinates); 100 km is generous for any realistic TSDF
// workload while still well below the fp32 overflow regime. LiDAR
// callers never hit this filter's garbage-rejection branch in
// practice (their inputs are raw Velodyne-style fp32 readings, not
// unprojected from fp32 matrix math), so the cap only matters to
// prevent accidental rejection of legitimate far-from-origin points.
constexpr double kMaxAdaptiveClamp      = 100000.0;

torch::Tensor
filterValidPoints(const torch::Tensor &points) {
    // Runs in fp32 (whatever dtype the caller passed). Keeps peak memory
    // to points.size(0) * sizeof(scalar_t) * 3 bytes rather than the 2x
    // that an intermediate fp64 copy would need.
    //
    // Stage A: reject non-finite (NaN / Inf) so the quantile stage
    // only looks at real-valued finite coordinates.
    const torch::Tensor finite_mask =
        at::isfinite(points).all(/*dim=*/1);
    const torch::Tensor finite_pts = points.index({finite_mask});
    if (finite_pts.size(0) == 0) {
        return points.new_empty({0, 3});
    }

    // Stage B: scene-adaptive clamp. Robust statistic = median of
    // max|coord| across the surviving finite points. The garbage tail
    // that `unprojectDepthmapKernel` emits is a small fraction of the
    // total (~1%) so the median is dominated by genuine scene content
    // -- unlike p99 which gets dragged up by the garbage.
    //
    // `at::quantile` has a ~2^24-row internal sort limit, so we stride-
    // subsample large inputs. 1 M samples pins the median to within a
    // centimetre for any realistic point distribution, for ~20 ms of
    // extra work.
    const torch::Tensor max_abs =
        std::get<0>(finite_pts.abs().max(/*dim=*/1)); // [N_fin]
    constexpr int64_t kPctSampleCap = 1 << 20; // 1 M
    torch::Tensor max_abs_for_quantile;
    if (max_abs.size(0) > kPctSampleCap) {
        const int64_t stride = (max_abs.size(0) + kPctSampleCap - 1) /
                               kPctSampleCap;
        max_abs_for_quantile =
            max_abs.index({torch::indexing::Slice(0, torch::indexing::None,
                                                   stride)})
                .contiguous();
    } else {
        max_abs_for_quantile = max_abs;
    }
    // `at::quantile` does not support fp16 inputs as of PyTorch 2.x
    // (it requires float or double). Promote to fp32 for the single
    // median call -- this is a ~1 M-element tensor at most so the
    // promotion cost is trivial.
    const torch::Tensor max_abs_f32 =
        max_abs_for_quantile.scalar_type() == torch::kHalf
            ? max_abs_for_quantile.to(torch::kFloat32)
            : max_abs_for_quantile;
    const double median =
        at::quantile(max_abs_f32, 0.50).item<double>();
    const double clamp = std::min(
        kMaxAdaptiveClamp,
        std::max(kMinAdaptiveClamp, kAdaptiveClampHeadroom * median));
    const torch::Tensor bounded_mask =
        (finite_pts.abs() < clamp).all(/*dim=*/1);

    if (std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
        std::fprintf(
            stderr,
            "[fvdb] filterValidPoints median=%.3f m -> clamp=%.3f m (finite=%lld -> bounded=%lld)\n",
            median, clamp, (long long)finite_pts.size(0),
            (long long)bounded_mask.sum().item<int64_t>());
    }

    return finite_pts.index({bounded_mask}).contiguous();
}

// Quantise world-space points into integer voxel ijk using the same
// transform (xyz - origin) / voxelSize + round() that fvdb's primal
// `VoxelCoordTransform` applies. Uses the input dtype for the
// `(xyz - origin) / voxelSize` math to avoid a 2x-memory fp32→fp64
// upcast; for typical TSDF settings (2 cm voxels, ~10 m scenes) the
// worst-case rounding error is well under a single voxel, so fp32 is
// sufficient for correctness.
torch::Tensor
pointsToIjk(const torch::Tensor &points,
            const nanovdb::Vec3d &voxelSize,
            const nanovdb::Vec3d &origin) {
    const torch::TensorOptions optSame =
        torch::TensorOptions().dtype(points.scalar_type()).device(points.device());
    const torch::Tensor vs =
        torch::tensor({voxelSize[0], voxelSize[1], voxelSize[2]}, optSame);
    const torch::Tensor og =
        torch::tensor({origin[0], origin[1], origin[2]}, optSame);
    const torch::Tensor ijk_same = ((points - og) / vs).round();
    return ijk_same.to(torch::kInt32).contiguous(); // [N, 3]
}

// Leaf-granularity shell builder (FVDB_LEAF_SHELL=1 fast path).
//
// Compared to the default voxel-granularity build:
//   - Map each unique voxel ijk to its LEAF key: `ijk >> 3` per axis
//     (each nanoVDB leaf is 8^3 voxels).
//   - Dilate at LEAF granularity. A voxel-level dilation radius of
//     `numPad` voxels translates to a leaf-level dilation of
//     `ceil((numPad + 7) / 8)` leaves per axis (worst case when a
//     voxel sits at the far edge of its leaf). So for the typical
//     numPad = 3 case (6 cm truncation at 2 cm voxels), the leaf
//     stencil is just 3^3 = 27 vs the voxel stencil's 7^3 = 343 --
//     a 13x reduction in dilate-and-dedupe work.
//   - Dedupe to unique leaves.
//   - Expand each unique leaf to its 512 voxel ijks (a fixed cartesian
//     `[0, 8)^3` offset, then broadcast-add to the leaf origin).
//   - Hand the 512-voxels-per-leaf ijk set to `_createNanoGridFromIJK`.
//
// The resulting grid is a strict SUPERSET of what the voxel-granularity
// path produces: every voxel within `numPad` of any input point is
// active, AND so are all other voxels in those voxels' leaves. Extra
// voxels cost a little memory (roughly `512 / voxels_per_leaf_hit`)
// but they're a no-op for the downstream TSDF integrate kernel -- they
// stay at weight = 0 and do nothing. In exchange we avoid ~50 dedupe
// passes on multi-million-row tensors, which was the ~60 ms/frame
// bottleneck on Replica (see research journal
// `2026-04-22_topology_ops_feasibility.md`).
//
// Returns `[U_leaves * 512, 3]` int32 voxel ijk tensor ready to hand
// to `_createNanoGridFromIJK`.
torch::Tensor
leafGranularityShell(const torch::Tensor &ijk,
                     int64_t numPad) {
    TORCH_CHECK(ijk.dim() == 2 && ijk.size(1) == 3,
                "leafGranularityShell expects ijk [N, 3]");
    const torch::Device device = ijk.device();
    const torch::TensorOptions optI32 =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Step 1: map each ijk to its LEAF key (ijk >> 3 in floor-arithmetic),
    // then dedupe to UNIQUE LEAVES.
    //
    // This is the crucial ordering: we dedupe BEFORE dilating. For a
    // Replica-scale depth frame, the 816 K quantised ijks collapse down
    // to ~1-2 K unique 8-voxel leaves (a 500x collapse), so the
    // downstream dilate-and-dedupe pass works on a tiny set. The
    // ~2-3 ms dedupe dominates; everything after it is sub-ms.
    const torch::Tensor leaf_key =
        at::div(ijk, 8, /*rounding_mode=*/"floor");
    torch::Tensor unique_leaves_raw = uniqueIjk(leaf_key);
    if (unique_leaves_raw.size(0) == 0) {
        return at::empty({0, 3}, optI32);
    }

    // Step 2: dilate at leaf granularity. Leaf-level dilation
    // half-radius for a voxel-level radius of `numPad`: a voxel anywhere
    // within an 8-wide leaf can reach up to `ceil((numPad + 7) / 8)`
    // leaves away, so the leaf stencil is `(2 * half + 1)^3`. For the
    // typical numPad = 3 case that is 3^3 = 27 (vs the voxel path's
    // 7^3 = 343).
    const int64_t leaf_half = (numPad + 7 + 7) / 8; // ceil((numPad+7)/8)
    const torch::Tensor leaf_axis =
        torch::arange(-leaf_half, leaf_half + 1, optI32);
    const auto leaf_grid =
        at::meshgrid({leaf_axis, leaf_axis, leaf_axis}, "ij");
    const torch::Tensor leaf_stencil =
        torch::stack({leaf_grid[0].flatten(),
                      leaf_grid[1].flatten(),
                      leaf_grid[2].flatten()},
                     1)
            .contiguous();

    // [U_raw, 1, 3] + [1, S_leaf, 3] -> [U_raw * S_leaf, 3]. At typical
    // Replica scale U_raw ~ 1-2 K and S_leaf = 27 so this is ~30-50 K
    // rows, trivial to dedupe.
    const torch::Tensor leaf_expanded =
        (unique_leaves_raw.unsqueeze(1) + leaf_stencil.unsqueeze(0))
            .reshape({-1, 3})
            .contiguous();
    unique_leaves_raw = torch::Tensor(); // free
    const torch::Tensor unique_leaves = uniqueIjk(leaf_expanded);
    if (unique_leaves.size(0) == 0) {
        return at::empty({0, 3}, optI32);
    }

    // Emit all 512 voxels per leaf: leaf origin = leaf_key * 8, and each
    // voxel in the leaf is leaf_origin + (i, j, k) for (i,j,k) in
    // [0, 8)^3.
    const torch::Tensor local_axis = torch::arange(0, 8, optI32);
    const auto local_grid = at::meshgrid({local_axis, local_axis, local_axis}, "ij");
    const torch::Tensor local_offsets =
        torch::stack({local_grid[0].flatten(),
                      local_grid[1].flatten(),
                      local_grid[2].flatten()},
                     1)
            .contiguous();  // [512, 3]

    const torch::Tensor leaf_origins = unique_leaves * 8;  // [U_leaves, 3]
    // [U, 1, 3] + [1, 512, 3] -> [U, 512, 3] -> [U*512, 3]
    const torch::Tensor shell =
        (leaf_origins.unsqueeze(1) + local_offsets.unsqueeze(0))
            .reshape({-1, 3})
            .contiguous();

    if (std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
        std::fprintf(
            stderr,
            "[fvdb] leafGranularityShell input_ijks=%lld leaves=%lld shell_voxels=%lld\n",
            (long long)ijk.size(0),
            (long long)unique_leaves.size(0),
            (long long)shell.size(0));
    }
    return shell;
}

// Build the single-batch truncation-shell grid directly from a list of
// world-space points. Entirely in torch ops (no `buildGridFromPoints` /
// `dilateGrid` call), deduping at two levels and tree-merging the
// per-stencil-chunk partials so that peak transient memory stays at
// `O(U_unique_base_voxels * stencil_chunk * 12 B)` rather than
// `O(N_points)` or `O(pointGrid_tile_count * 16 MB)`.
//
// Env override `FVDB_LEAF_SHELL=1` switches to the leaf-granularity
// fast path (see `leafGranularityShell` above). The leaf path
// over-covers at the sub-leaf scale but avoids the ~50 dedupe-pass
// accumulation that dominates the voxel-granularity path on room-
// scale scenes. Targeted for the phase-1b per-frame fusion pipeline;
// see research journal `2026-04-22_topology_ops_feasibility.md`.
nanovdb::GridHandle<TorchDeviceBuffer>
buildSingleBatchShell(const torch::Tensor &points_b,
                      const nanovdb::Vec3d &voxelSize,
                      const nanovdb::Vec3d &origin,
                      int64_t numPad) {
    TORCH_CHECK(points_b.dim() == 2 && points_b.size(1) == 3,
                "points must be [N, 3]");
    TORCH_CHECK(points_b.device().is_cuda(),
                "fast shell builder is CUDA-only");

    const torch::Device device = points_b.device();

    // `FVDB_SHELL_PHASE_PROFILE=1` decomposes the voxel-shell build into
    // its four sub-steps (filter+quantise, base-dedupe, stencil
    // expand+merge, createGrid). Use to identify which stage to attack
    // next in the shell-build speedup track. One line per frame to
    // stderr.
    const bool phaseProfile =
        std::getenv("FVDB_SHELL_PHASE_PROFILE") != nullptr;
    cudaEvent_t evA{}, evB{}, evC{}, evD{}, evE{};
    auto phaseMark = [&](cudaEvent_t &ev) {
        if (phaseProfile) {
            cudaEventCreate(&ev);
            cudaEventRecord(ev);
        }
    };
    phaseMark(evA);

    // Stage 1: point-level filter + quantise.
    //
    // Work in the caller's dtype (typically fp32) throughout. Converting
    // to fp64 for "precision" doubles peak memory for no benefit in this
    // workload -- the subsequent `.round()` step is trivially exact in
    // fp32 for any realistic voxel size / scene extent combination.
    //
    // Eagerly drop `valid_points` as soon as `ijk_i32` exists, so the
    // [N_valid, 3] fp32 tensor (~1 GB at N=200 frames @ 1200x680) is
    // reclaimed before the dedupe / stencil stages ask for scratch.
    torch::Tensor ijk_i32;
    {
        torch::Tensor valid_points = filterValidPoints(points_b);
        if (valid_points.size(0) == 0) {
            TorchDeviceBuffer emptyBuf(0, device);
            return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(emptyBuf));
        }
        ijk_i32 = pointsToIjk(valid_points, voxelSize, origin);
    } // valid_points drops here
    phaseMark(evB);

    // --- Voxel-granularity shell (default CUDA fast path) ----------
    //
    // The voxel-granularity path uses separable-axis dilation on
    // packed int64 ijk keys followed by `voxelsToGrid`: quantise ->
    // base dedupe -> dilate-X + dedupe -> dilate-Y + dedupe ->
    // dilate-Z + dedupe -> unpack -> voxelsToGrid. Total dedupe work
    // is O(N * 3 * (2r+1)) with each intermediate compressing by
    // 2-3x before the next axis expansion, replacing the ~90
    // `_unique` launches of the old 3D-stencil-chunked tree-merge
    // with 3 launches on progressively larger but still bounded
    // tensors.
    //
    // In our experiments this runs measurably faster and with
    // substantially lower peak memory than the previous chunked-3D-
    // stencil path, especially at fine voxel sizes where the 3D
    // stencil's intermediate buffer is the bottleneck.
    //
    // Opt-in: `FVDB_LEAF_SHELL=1` reverts to the leaf-granularity
    // builder further down. The leaf path over-covers at the
    // sub-leaf scale (allocates all 512 voxels in every touched
    // leaf, mostly weight-zero no-ops) but retains a potential
    // edge for workloads where the scene is so dense that every
    // 8^3 voxel neighborhood is in the truncation band anyway;
    // kept as an ablation knob.
    const bool force_leaf_shell = [&]() {
        const char *env = std::getenv("FVDB_LEAF_SHELL");
        return env != nullptr && env[0] == '1';
    }();

    if (force_leaf_shell) {
        const torch::Tensor leaf_shell =
            leafGranularityShell(ijk_i32, numPad);
        ijk_i32 = torch::Tensor();
        if (leaf_shell.size(0) == 0) {
            TorchDeviceBuffer emptyBuf(0, device);
            return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(emptyBuf));
        }
        if (std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
            std::fprintf(
                stderr,
                "[fvdb] buildSingleBatchShell (leaf path, FVDB_LEAF_SHELL opt-in) shell=%lld\n",
                (long long)leaf_shell.size(0));
        }
        const JaggedTensor shellJT(leaf_shell);
        return _createNanoGridFromIJK(shellJT);
    }

    // Stage 2: separable box dilation in packed int64 keys.
    //
    // The voxel-shell's [-r, r]^3 box dilation is a morphological
    // open-ball operation in the Chebyshev metric; such dilations are
    // separable across axes: dilate_3D(A, r) ==
    // dilate_Z(dilate_Y(dilate_X(A, r), r), r). Doing it separably
    // reduces work from O(N * (2r+1)^3) to O(N * 3 * (2r+1)) with
    // dedup between each axis, which shrinks each stage's working set
    // by ~2-3x before the next axis expands it. At Replica scale
    // (N=800 K, r=3) the total dedup work drops from ~440 M to ~60 M
    // rows, and we replace ~90 `_unique` kernel launches with exactly
    // three.
    //
    // Keys are packed into int64 (21 bits per axis, 20-bit bias) so
    // `_unique` runs on a 1-D tensor rather than row-wise on
    // int32[N, 3]. Stencil offsets are pre-packed the same way so the
    // per-axis expand is a single `[U, 1] + [1, 2r+1]` broadcast add.
    // Final unpack hands int32[F, 3] to `voxelsToGrid` which builds
    // topology in one more sort+RLE pass.
    //
    // `voxelsToGrid` itself will dedupe any input, but feeding it the
    // raw undeduped ~400-600 M voxels directly (we tested) takes ~300
    // ms at 5 mm because CUB radix-sort cost scales near-linearly with
    // input size. The three pre-sorts here are cheap (each works on a
    // progressively larger but still much smaller tensor than the full
    // N * S expansion) and reduce the final voxelsToGrid input to
    // ~10 M unique voxels, which is ~10 ms to turn into a grid.
    constexpr int64_t kPackBias = 1ll << 20;
    constexpr int64_t kPackMask = (1ll << 21) - 1;
    auto packIjk = [&](const torch::Tensor &ijk_i32) -> torch::Tensor {
        const torch::Tensor ijk_i64 = ijk_i32.to(torch::kInt64);
        const torch::Tensor i =
            ijk_i64.select(1, 0).add(kPackBias);
        const torch::Tensor j =
            ijk_i64.select(1, 1).add(kPackBias);
        const torch::Tensor k =
            ijk_i64.select(1, 2).add(kPackBias);
        return (i.bitwise_left_shift(42))
            .bitwise_or(j.bitwise_left_shift(21))
            .bitwise_or(k);
    };
    auto unpackKeys = [&](const torch::Tensor &keys) -> torch::Tensor {
        const torch::Tensor i =
            keys.bitwise_right_shift(42).bitwise_and(kPackMask)
                .sub(kPackBias);
        const torch::Tensor j =
            keys.bitwise_right_shift(21).bitwise_and(kPackMask)
                .sub(kPackBias);
        const torch::Tensor k =
            keys.bitwise_and(kPackMask).sub(kPackBias);
        return torch::stack({i, j, k}, /*dim=*/1)
            .to(torch::kInt32).contiguous();
    };

    // Pack and dedup the base ijks once. Raw N includes substantial
    // aliasing (multiple depth pixels quantising to the same voxel),
    // and deduping here saves work at every subsequent axis-expand.
    //
    // We use `at::_unique` rather than a direct CUB radix-sort + select-
    // unique because `_unique` already calls into CUB under the hood
    // and the per-call allocation overhead is absorbed by torch's
    // caching allocator.
    torch::Tensor keys = std::get<0>(
        at::_unique(packIjk(ijk_i32), /*sorted=*/false,
                    /*return_inverse=*/false));
    ijk_i32 = torch::Tensor();
    phaseMark(evC);

    // Per-axis 1-D stencils of length `2r+1`, pre-packed as int64 so
    // broadcast-add composes directly with the packed base keys. Shift
    // factors (42, 21, 0) mirror the axis-to-bit assignment above.
    const torch::TensorOptions optI64 =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    const torch::Tensor axisOffsets =
        torch::arange(-numPad, numPad + 1, optI64); // [2r+1] signed
    const torch::Tensor stencil_x =
        axisOffsets.bitwise_left_shift(42);
    const torch::Tensor stencil_y =
        axisOffsets.bitwise_left_shift(21);
    const torch::Tensor stencil_z = axisOffsets;

    auto applyAxis = [&](torch::Tensor keys_in,
                         const torch::Tensor &axisStencil) {
        // [U, 1] + [1, 2r+1] -> [U * (2r+1)] -> unique.
        torch::Tensor expanded =
            (keys_in.unsqueeze(1) + axisStencil.unsqueeze(0))
                .flatten().contiguous();
        keys_in = torch::Tensor();
        return std::get<0>(
            at::_unique(expanded, /*sorted=*/false,
                        /*return_inverse=*/false));
    };

    keys = applyAxis(std::move(keys), stencil_x);
    keys = applyAxis(std::move(keys), stencil_y);
    keys = applyAxis(std::move(keys), stencil_z);
    const int64_t F = keys.size(0);
    if (F == 0) {
        TorchDeviceBuffer emptyBuf(0, device);
        return nanovdb::GridHandle<TorchDeviceBuffer>(std::move(emptyBuf));
    }
    const torch::Tensor shell = unpackKeys(keys);
    keys = torch::Tensor();
    phaseMark(evD);

    if (std::getenv("FVDB_NANOVDB_TRACE_ALLOCS")) {
        std::fprintf(
            stderr,
            "[fvdb] buildSingleBatchShell (voxel path, separable) "
            "numPad=%lld shell=%lld\n",
            (long long)numPad, (long long)F);
    }

    const JaggedTensor shellJT(shell);
    auto gridHandle = _createNanoGridFromIJK(shellJT);
    phaseMark(evE);

    if (phaseProfile) {
        cudaEventSynchronize(evE);
        float t_filter = 0, t_base = 0, t_sep = 0, t_grid = 0;
        cudaEventElapsedTime(&t_filter, evA, evB);
        cudaEventElapsedTime(&t_base, evB, evC);
        cudaEventElapsedTime(&t_sep, evC, evD);
        cudaEventElapsedTime(&t_grid, evD, evE);
        std::fprintf(
            stderr,
            "[fvdb/shell_phase] filter+ijk=%.2f ms  base_dedup=%.2f ms "
            " separable_xyz=%.2f ms  createGrid=%.2f ms  total=%.2f "
            "ms  numPad=%lld shell=%lld\n",
            t_filter, t_base, t_sep, t_grid,
            t_filter + t_base + t_sep + t_grid,
            (long long)numPad, (long long)F);
        cudaEventDestroy(evA);
        cudaEventDestroy(evB);
        cudaEventDestroy(evC);
        cudaEventDestroy(evD);
        cudaEventDestroy(evE);
    }
    return gridHandle;
}

} // namespace

c10::intrusive_ptr<GridBatchData>
buildPointTruncationShell(const JaggedTensor &points,
                          const GridBatchData &grid,
                          double truncationMargin) {
    TORCH_CHECK_VALUE(truncationMargin > 0.0,
                      "truncationMargin must be > 0, got ",
                      truncationMargin);
    TORCH_CHECK_VALUE(points.num_outer_lists() == grid.batchSize(),
                      "points batch size (", points.num_outer_lists(),
                      ") must equal grid batch size (", grid.batchSize(), ")");

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
        // where the intended value is 3, inflating the separable-axis
        // stencil from 7 to 9 per axis and wasting 28% of dedup work
        // on expanded voxels nobody asked for.
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
        const double tol = 4.0 * static_cast<double>(
            std::numeric_limits<float>::epsilon()) * std::max(1.0, ratio);
        const double ceilRatio = (std::abs(ratio - ratioRounded) <= tol)
                                     ? ratioRounded
                                     : std::ceil(ratio);
        const auto numPadVoxelsI = static_cast<int32_t>(ceilRatio);
        TORCH_CHECK_VALUE(numPadVoxelsI > 0,
                          "Number of padding voxels must be positive, got ",
                          numPadVoxelsI,
                          " (truncationMargin=", truncationMargin,
                          ", voxelSize=", minVoxLengthI, ")");
        TORCH_CHECK_VALUE(numPadVoxelsI < MAX_PAD_VOXELS,
                          "Truncation margin (", truncationMargin,
                          ") is too large for grid with voxel size ",
                          minVoxLengthI,
                          ", resulting in too many padding voxels (",
                          numPadVoxelsI, ") which cannot exceed ",
                          MAX_PAD_VOXELS,
                          ". Use a larger voxel size or a smaller truncation margin.");
        numPadVoxels.push_back(numPadVoxelsI);
    }

    // CPU and opt-out paths: run the original `buildGridFromPoints +
    // dilateGrid(numPad)` pipeline verbatim. The CUDA fast path below
    // sidesteps it because `dilateGrid` scratch blows up on
    // room-scale scenes.
    const bool isCuda = points.device().is_cuda();
    if (!isCuda ||
        (std::getenv("FVDB_NANOVDB_LEGACY_SHELL") != nullptr &&
         std::getenv("FVDB_NANOVDB_LEGACY_SHELL")[0] == '1')) {
        auto pointGrid = buildGridFromPoints(points, voxelSizes, origins);
        return dilateGrid(*pointGrid, numPadVoxels);
    }

    // --- Fast path (CUDA, N-way union via voxel-level dilation) ---------
    //
    // For each batch item we:
    //
    //   1. Filter out NaN / Inf / far-field garbage at the point level
    //      (`unprojectDepthmapKernel` has a precision quirk that emits
    //      ~1% of its pixels at 10-10^6 m from the scene -- see the
    //      research journal entry for details).
    //   2. Quantise surviving points to integer voxel ijk.
    //   3. Dedupe to unique-base voxels.
    //   4. Stencil-dilate by `[-numPad, numPad]^3`, chunked + tree-merged.
    //   5. Call `voxelsToGrid` once on the final shell voxel set.
    //
    // Per-batch grids are concatenated via `nanovdb::cuda::mergeGridHandles`
    // which is a pure-buffer memcpy (no topology work, no speculative root
    // blow-up).
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    handles.reserve(points.num_outer_lists());

    const torch::Tensor offsetsCpu = points.joffsets().cpu();
    const auto offsets             = offsetsCpu.accessor<JOffsetsType, 1>();
    TORCH_CHECK(offsets.size(0) == grid.batchSize() + 1,
                "joffsets length mismatch: expected ",
                grid.batchSize() + 1, " got ", offsets.size(0));

    const torch::Tensor data = points.jdata();
    for (int64_t i = 0; i < grid.batchSize(); ++i) {
        const int64_t start = offsets[i];
        const int64_t count = offsets[i + 1] - start;
        if (count == 0) {
            TorchDeviceBuffer emptyBuf(0, points.device());
            handles.emplace_back(
                nanovdb::GridHandle<TorchDeviceBuffer>(std::move(emptyBuf)));
            continue;
        }

        const torch::Tensor points_i =
            data.narrow(0, start, count).contiguous();
        handles.push_back(buildSingleBatchShell(
            points_i, voxelSizes[i], origins[i], numPadVoxels[i]));
    }

    nanovdb::GridHandle<TorchDeviceBuffer> mergedHandle;
    if (handles.size() == 1) {
        mergedHandle = std::move(handles[0]);
    } else {
        TorchDeviceBuffer guide(0, points.device());
        mergedHandle = nanovdb::cuda::mergeGridHandles(handles, &guide);
    }
    return makeGridBatchData(std::move(mergedHandle), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
