// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MARCHINGCUBES_H
#define FVDB_DETAIL_OPS_MARCHINGCUBES_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Public marching-cubes entry point.
///
/// Dispatches to a sparse-compact / packed-key fast variant
/// (`marchingCubesFast`) for float32 and float16 CUDA inputs on batches of
/// up to 8 grids and 2^30 voxels. Its triangles match the legacy
/// implementation, and vertex positions agree to within float32 rounding.
/// fp16 fields are cast to fp32 on load, without a transient fp32 copy.
/// Everything else (fp64, CPU, larger batches) routes to
/// `marchingCubesLegacy`.
std::vector<JaggedTensor>
marchingCubes(const GridBatchData &batchHdl, const JaggedTensor &field, double level);

/// @brief Reference legacy marching-cubes implementation.
///
/// Used as the fallback when `marchingCubes` cannot route to the fast
/// variant (non-float32/float16 inputs, or CPU device). New code
/// should call `marchingCubes` instead — it picks the fast path when
/// eligible and falls back here automatically otherwise.
std::vector<JaggedTensor>
marchingCubesLegacy(const GridBatchData &batchHdl, const JaggedTensor &field, double level);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MARCHINGCUBES_H
