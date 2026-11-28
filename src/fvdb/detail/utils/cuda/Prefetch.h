// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
#define FVDB_DETAIL_UTILS_CUDA_PREFETCH_H

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace fvdb {
namespace detail {

/// Prefetch the memory ranges given by paired pointers and byte sizes to the specified device,
/// ordered on the non-default input stream. On CUDA 13+, discard existing contents to avoid
/// migrating data that will be overwritten; callers must overwrite discarded data before reading
/// it. On older CUDA versions, fall back to individual prefetch calls without discarding contents.
/// Queue a bounded GPU delay before each prefetch call to reduce host submission stalls.
void memDiscardAndPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                                     std::vector<size_t> &prefetchSizes,
                                     int deviceId,
                                     cudaStream_t stream);

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_PREFETCH_H
