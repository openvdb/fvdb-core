// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_LAPLACIANSMOOTH_H
#define FVDB_DETAIL_OPS_LAPLACIANSMOOTH_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Umbrella-Laplacian smoothing schedule. Mirrored by the Python ``fvdb.SmoothingMode``
///        enum (matching member names).
enum class SmoothingMode : int32_t {
    MEAN_CURVATURE = 0, ///< One pass of weight 1.0 per iteration. On a signed distance field the
                        ///< umbrella Laplacian approximates mean-curvature flow.
    TAUBIN = 1,         ///< Volume-preserving Taubin smoothing: passes of weight 0.5 then -0.53 per
                        ///< iteration.
};

/// @brief Smooth a per-voxel field with the 6-neighbour umbrella Laplacian.
///
/// Each pass replaces every voxel's value with
///     out = in + weight * (mean of the 6 face-neighbour values - in),
/// reading the pre-pass field (Jacobi update), for every channel independently. The grid topology
/// is unchanged and the result has the same per-voxel ordering and shape as @p field.
///
/// @param batchHdl        Grid batch defining the sparse topology.
/// @param field           Per-voxel field with shape (N,) or (N, C), N == totalVoxels, float32 or
///                        float64.
/// @param iterations      Number of smoothing iterations (>= 0; 0 returns a copy). A Taubin
///                        iteration is two passes.
/// @param mode            Which umbrella-Laplacian schedule each iteration applies.
/// @param boundaryValues  How an inactive face neighbour is read. std::nullopt: as the centre
///                        voxel's own value (zero flux; the face contributes nothing). Otherwise a
///                        1-D floating-point tensor, on the host or the field's device, of one
///                        non-negative value b per grid: the face reads as -b beside a negative
///                        centre value and +b otherwise (sign continuation, for signed fields such
///                        as a narrow-band SDF with b = band * voxel size). Host values are
///                        validated without a device synchronization.
/// @return A new JaggedTensor on the same grid/ordering as @p field.
JaggedTensor laplacianSmooth(const GridBatchData &batchHdl,
                             const JaggedTensor &field,
                             int iterations,
                             SmoothingMode mode,
                             const std::optional<torch::Tensor> &boundaryValues);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_LAPLACIANSMOOTH_H
