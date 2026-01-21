// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "fvdb/detail/dispatch/example/ScanLib.h"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace scanlib {

// =============================================================================
// CPU Serial implementations
// =============================================================================

template <typename T>
void
inclusive_scan_serial(T const *in, int64_t in_stride, T *out, int64_t out_stride, int64_t n) {
    if (n <= 0)
        return;

    T sum = T{0};
    for (int64_t i = 0; i < n; ++i) {
        sum += in[i * in_stride];
        out[i * out_stride] = sum;
    }
}

template <typename T>
void
inclusive_scan_serial_inplace(T *data, int64_t stride, int64_t n) {
    if (n <= 0)
        return;

    T sum = T{0};
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i * stride];
        data[i * stride] = sum;
    }
}

// =============================================================================
// CPU Parallel implementations (OpenMP)
// Three-phase algorithm:
//   Phase 1: Each thread computes local scan of its chunk
//   Phase 2: Sequential scan of partial sums (O(num_threads))
//   Phase 3: Each thread adds its prefix to its chunk
// =============================================================================

template <typename T>
void
inclusive_scan_parallel(T const *in, int64_t in_stride, T *out, int64_t out_stride, int64_t n) {
    if (n <= 0)
        return;

#ifdef _OPENMP
    // Get number of threads
    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    // For small arrays or single thread, fall back to serial
    if (num_threads == 1 || n < num_threads * 4) {
        inclusive_scan_serial(in, in_stride, out, out_stride, n);
        return;
    }

    // Partial sums for each thread's chunk
    std::vector<T> partial_sums(num_threads);

// Phase 1: Each thread computes local scan of its chunk
#pragma omp parallel
    {
        int tid            = omp_get_thread_num();
        int64_t chunk_size = (n + num_threads - 1) / num_threads;
        int64_t start      = tid * chunk_size;
        int64_t end        = std::min(start + chunk_size, n);

        T sum = T{0};
        for (int64_t i = start; i < end; ++i) {
            sum += in[i * in_stride];
            out[i * out_stride] = sum;
        }
        partial_sums[tid] = sum;
    }

    // Phase 2: Sequential scan of partial sums
    for (int t = 1; t < num_threads; ++t) {
        partial_sums[t] += partial_sums[t - 1];
    }

// Phase 3: Add prefix to each chunk (except first)
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid > 0) {
            int64_t chunk_size = (n + num_threads - 1) / num_threads;
            int64_t start      = tid * chunk_size;
            int64_t end        = std::min(start + chunk_size, n);
            T prefix           = partial_sums[tid - 1];

            for (int64_t i = start; i < end; ++i) {
                out[i * out_stride] += prefix;
            }
        }
    }
#else
    // No OpenMP available, fall back to serial
    inclusive_scan_serial(in, in_stride, out, out_stride, n);
#endif
}

// =============================================================================
// Explicit instantiations
// =============================================================================

// clang-format off
template void inclusive_scan_serial<float>(float const *, int64_t, float *, int64_t, int64_t);
template void inclusive_scan_serial<double>(double const *, int64_t, double *, int64_t, int64_t);
template void inclusive_scan_serial<int32_t>(int32_t const *, int64_t, int32_t *, int64_t, int64_t);
template void inclusive_scan_serial<int64_t>(int64_t const *, int64_t, int64_t *, int64_t, int64_t);

template void inclusive_scan_serial_inplace<float>(float *, int64_t, int64_t);
template void inclusive_scan_serial_inplace<double>(double *, int64_t, int64_t);
template void inclusive_scan_serial_inplace<int32_t>(int32_t *, int64_t, int64_t);
template void inclusive_scan_serial_inplace<int64_t>(int64_t *, int64_t, int64_t);

template void inclusive_scan_parallel<float>(float const *, int64_t, float *, int64_t, int64_t);
template void inclusive_scan_parallel<double>(double const *, int64_t, double *, int64_t, int64_t);
template void inclusive_scan_parallel<int32_t>(int32_t const *, int64_t, int32_t *, int64_t, int64_t);
template void inclusive_scan_parallel<int64_t>(int64_t const *, int64_t, int64_t *, int64_t, int64_t);
// clang-format on

} // namespace scanlib
