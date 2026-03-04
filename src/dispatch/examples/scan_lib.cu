// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "examples/scan_lib.h"

#include <cub/cub.cuh>

#include <stdexcept>
#include <string>

namespace scan_lib {

// =============================================================================
// CUDA implementations (CUB)
// Uses CUB's DeviceScan::InclusiveSum for efficient parallel scan.
// =============================================================================

template <typename T>
size_t
inclusive_scan_cuda_temp_bytes(int64_t n) {
    if (n <= 0)
        return 0;

    size_t temp_bytes = 0;
    // Query CUB for required temporary storage size
    // Pass nullptr for temp storage to get size only
    // Note: CUB's NumItemsT is a template parameter that accepts any integral type,
    // so we can pass int64_t directly without truncation concerns.
    cub::DeviceScan::InclusiveSum(nullptr,                         // d_temp_storage
                                  temp_bytes,                      // temp_storage_bytes (output)
                                  static_cast<T const *>(nullptr), // d_in
                                  static_cast<T *>(nullptr),       // d_out
                                  n                                // num_items
    );
    return temp_bytes;
}

template <typename T>
void
inclusive_scan_cuda(
    T const *in, T *out, int64_t n, void *temp, size_t temp_bytes, cudaStream_t stream) {
    if (n <= 0)
        return;

    // Run CUB inclusive scan
    cudaError_t err = cub::DeviceScan::InclusiveSum(temp,       // d_temp_storage
                                                    temp_bytes, // temp_storage_bytes
                                                    in,         // d_in
                                                    out,        // d_out
                                                    n,          // num_items
                                                    stream      // stream
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUB InclusiveSum failed: ") +
                                 cudaGetErrorString(err));
    }
}

// =============================================================================
// Explicit instantiations
// All types are instantiated here. The "floats are non-deterministic"
// constraint is documented but not enforced at this level.
// =============================================================================

// clang-format off
template size_t inclusive_scan_cuda_temp_bytes<float>(int64_t);
template size_t inclusive_scan_cuda_temp_bytes<double>(int64_t);
template size_t inclusive_scan_cuda_temp_bytes<int32_t>(int64_t);
template size_t inclusive_scan_cuda_temp_bytes<int64_t>(int64_t);

template void inclusive_scan_cuda<float>(float const *, float *, int64_t, void *, size_t, cudaStream_t);
template void inclusive_scan_cuda<double>(double const *, double *, int64_t, void *, size_t, cudaStream_t);
template void inclusive_scan_cuda<int32_t>(int32_t const *, int32_t *, int64_t, void *, size_t, cudaStream_t);
template void inclusive_scan_cuda<int64_t>(int64_t const *, int64_t *, int64_t, void *, size_t, cudaStream_t);
// clang-format on

} // namespace scan_lib
