// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MARCHINGCUBESFAST_H
#define FVDB_DETAIL_OPS_MARCHINGCUBESFAST_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Sparse-compact, packed-key marching-cubes for fp32/fp16 CUDA.
///
/// This is the variant that `marchingCubes` dispatches to by default
/// for CUDA inputs; `marchingCubesLegacy` is the fallback for
/// unsupported dtype / device combinations.
///
/// The main differences vs `marchingCubesLegacy` are:
///
///   - **Surface-voxel compaction**: a classify pass writes a per-leaf-
///     voxel `nVertsPerLv[uint8_t]` array and a prefix-summed offset
///     table; the emit pass iterates only the surface voxels rather
///     than every voxel in the grid, dropping work and DRAM traffic
///     for sparse SDFs.
///   - **Packed-key dedup**: each triangle vertex is emitted as a
///     single packed int64 key `(batchIdx, vid0, vid1)` and deduped
///     via 1-D `torch::unique`, replacing the legacy's 3-column
///     `[nTri*3, 3]` int64 tensor + `torch::unique_dim`. Cuts the
///     dedup-input footprint ~3x and halves the internal sort temps.
///   - **fp16 fast path**: the classify and emit kernels are
///     templated on the input scalar type so fp16 inputs are loaded
///     and cast to fp32 in-register (single `F2F.F32.F16` per load),
///     avoiding the 2x transient fp32 buffer a naive
///     `sdf.to(kFloat32)` would allocate.
///
/// Packing layout (64-bit key; validated by `TORCH_CHECK_VALUE`
/// guards in the implementation so future scale changes fail loudly):
///
///     key = (batchIdx & 0xF) << 60                // 4 bits, up to 16 batches
///         | (vid0     & 0x3FFFFFFF) << 30         // 30 bits, up to 1B voxels/batch
///         | (vid1     & 0x3FFFFFFF)               // 30 bits, up to 1B voxels/batch
///
/// Dtype / device coverage:
///   - float32 CUDA: native fast path.
///   - float16 CUDA: as described above; only the final `retVertices`
///     tensor (`[nV, 3]` floats, orders of magnitude smaller than the
///     SDF) is downcast to fp16 to preserve the public output-dtype
///     contract.
///   - float64 or CPU: forwarded to `marchingCubesLegacy`, which is
///     fully templated and handles every floating-point dtype.
std::vector<JaggedTensor>
marchingCubesFast(const GridBatchData &batchHdl,
                  const JaggedTensor &field,
                  double level);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MARCHINGCUBESFAST_H