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
/// (`marchingCubesFast`) for float32 and float16 CUDA inputs.
/// `marchingCubesFast` produces bit-identical output to the legacy
/// implementation at fp32 (and numerically-identical output at fp16,
/// since its kernels cast fp16 -> fp32 on load and do all arithmetic
/// in fp32 without allocating a transient fp32 buffer). It is
/// substantially faster and uses substantially less peak memory at
/// large grid sizes. Other dtypes (fp64) and CPU inputs route to
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
marchingCubesLegacy(const GridBatchData &batchHdl,
                    const JaggedTensor &field,
                    double level);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MARCHINGCUBES_H
