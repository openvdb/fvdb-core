// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// scan_lib: A foundation library for inclusive scan operations.
//
// This library provides low-level scan primitives with explicit control over
// determinism, parallelism, and memory layout. It is designed to be wrapped
// by higher-level dispatch systems.
//
// DESIGN PRINCIPLES:
// - No switching logic inside implementations; all dispatch happens externally
// - Raw pointer + stride interface (not tied to any tensor library)
// - Explicit instantiation for common types
//
// DETERMINISM NOTES:
// - Serial CPU functions are deterministic for all types
// - Parallel CPU functions are NOT deterministic for floating-point types
//   (due to thread-count-dependent chunking affecting associativity)
// - Parallel CPU functions ARE deterministic for integer types
// - CUDA functions are NOT deterministic for floating-point types
// - CUDA functions ARE deterministic for integer types
//
// IN-PLACE SUPPORT:
// - Serial CPU functions support in-place operation
// - Parallel CPU functions do NOT support in-place operation
// - CUDA functions do NOT support in-place operation (parallel scan needs workspace)
//
// STRIDE SUPPORT:
// - CPU functions support strided data (stride parameter)
// - CUDA functions require contiguous data (no stride parameter)

#ifndef DISPATCH_EXAMPLES_SCAN_LIB_H
#define DISPATCH_EXAMPLES_SCAN_LIB_H

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Forward declare cudaStream_t for header compatibility when not compiling with nvcc
using cudaStream_t = void *;
#endif

namespace scan_lib {

// =============================================================================
// CPU Serial implementations
// Deterministic for all types. Supports strided data. Supports in-place.
// =============================================================================

/// Inclusive scan (prefix sum) - serial, out-of-place.
/// @param in       Input data pointer
/// @param in_stride Stride between input elements (1 for contiguous)
/// @param out      Output data pointer (must not alias input)
/// @param out_stride Stride between output elements (1 for contiguous)
/// @param n        Number of elements
template <typename T>
void inclusive_scan_serial(T const *in, int64_t in_stride, T *out, int64_t out_stride, int64_t n);

/// Inclusive scan (prefix sum) - serial, in-place.
/// @param data   Data pointer (read and written)
/// @param stride Stride between elements (1 for contiguous)
/// @param n      Number of elements
template <typename T> void inclusive_scan_serial_inplace(T *data, int64_t stride, int64_t n);

// =============================================================================
// CPU Parallel implementations (OpenMP)
// NOT deterministic for floating-point types. Supports strided data.
// Does NOT support in-place operation.
// =============================================================================

/// Inclusive scan (prefix sum) - parallel, out-of-place.
/// WARNING: Non-deterministic for floating-point types due to parallel reduction.
/// @param in       Input data pointer
/// @param in_stride Stride between input elements (1 for contiguous)
/// @param out      Output data pointer (must not alias input)
/// @param out_stride Stride between output elements (1 for contiguous)
/// @param n        Number of elements
template <typename T>
void inclusive_scan_parallel(T const *in, int64_t in_stride, T *out, int64_t out_stride, int64_t n);

// =============================================================================
// CUDA implementations (CUB)
// NOT deterministic for floating-point types. Contiguous data only.
// Does NOT support in-place operation.
// =============================================================================

/// Query temporary storage size required for CUDA inclusive scan.
/// Call this first, allocate temp storage, then call inclusive_scan_cuda.
/// @param n Number of elements
/// @return  Required temporary storage size in bytes
template <typename T> size_t inclusive_scan_cuda_temp_bytes(int64_t n);

/// Inclusive scan (prefix sum) - CUDA, out-of-place.
/// WARNING: Non-deterministic for floating-point types due to parallel reduction.
/// @param in         Input data pointer (device memory, contiguous)
/// @param out        Output data pointer (device memory, contiguous, must not alias input)
/// @param n          Number of elements
/// @param temp       Temporary storage pointer (device memory)
/// @param temp_bytes Size of temporary storage (from inclusive_scan_cuda_temp_bytes)
/// @param stream     CUDA stream (nullptr for default stream)
template <typename T>
void inclusive_scan_cuda(
    T const *in, T *out, int64_t n, void *temp, size_t temp_bytes, cudaStream_t stream = nullptr);

// =============================================================================
// Extern template declarations for explicit instantiations
// =============================================================================

// clang-format off
extern template void inclusive_scan_serial<float>(float const *, int64_t, float *, int64_t, int64_t);
extern template void inclusive_scan_serial<double>(double const *, int64_t, double *, int64_t, int64_t);
extern template void inclusive_scan_serial<int32_t>(int32_t const *, int64_t, int32_t *, int64_t, int64_t);
extern template void inclusive_scan_serial<int64_t>(int64_t const *, int64_t, int64_t *, int64_t, int64_t);

extern template void inclusive_scan_serial_inplace<float>(float *, int64_t, int64_t);
extern template void inclusive_scan_serial_inplace<double>(double *, int64_t, int64_t);
extern template void inclusive_scan_serial_inplace<int32_t>(int32_t *, int64_t, int64_t);
extern template void inclusive_scan_serial_inplace<int64_t>(int64_t *, int64_t, int64_t);

extern template void inclusive_scan_parallel<float>(float const *, int64_t, float *, int64_t, int64_t);
extern template void inclusive_scan_parallel<double>(double const *, int64_t, double *, int64_t, int64_t);
extern template void inclusive_scan_parallel<int32_t>(int32_t const *, int64_t, int32_t *, int64_t, int64_t);
extern template void inclusive_scan_parallel<int64_t>(int64_t const *, int64_t, int64_t *, int64_t, int64_t);

extern template size_t inclusive_scan_cuda_temp_bytes<float>(int64_t);
extern template size_t inclusive_scan_cuda_temp_bytes<double>(int64_t);
extern template size_t inclusive_scan_cuda_temp_bytes<int32_t>(int64_t);
extern template size_t inclusive_scan_cuda_temp_bytes<int64_t>(int64_t);

extern template void inclusive_scan_cuda<float>(float const *, float *, int64_t, void *, size_t, cudaStream_t);
extern template void inclusive_scan_cuda<double>(double const *, double *, int64_t, void *, size_t, cudaStream_t);
extern template void inclusive_scan_cuda<int32_t>(int32_t const *, int32_t *, int64_t, void *, size_t, cudaStream_t);
extern template void inclusive_scan_cuda<int64_t>(int64_t const *, int64_t *, int64_t, void *, size_t, cudaStream_t);
// clang-format on

} // namespace scan_lib

#endif // DISPATCH_EXAMPLES_SCAN_LIB_H
