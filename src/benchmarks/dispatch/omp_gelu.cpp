// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// OpenMP GELU implementation.
// This file MUST be compiled by g++/clang++ directly (not nvcc) for
// OpenMP pragmas to work correctly.

#include "omp_gelu.h"

#include <ATen/cpu/vec/vec.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>

namespace omp_gelu {

namespace {

// SIMD GELU operation using exp-based formula
// GELU(x) = x / (1 + exp(-2u)) where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// This avoids tanh() which can be slower on some architectures.
template <typename T>
inline at::vec::Vectorized<T>
gelu_simd_op(const at::vec::Vectorized<T> &x) {
    using Vec = at::vec::Vectorized<T>;

    // kAlphaTwo = sqrt(2/pi) * 2 = 1.5957691216...
    const Vec kAlphaTwo(static_cast<T>(1.5957691216057307117597842397375));
    const Vec kBeta(static_cast<T>(0.044715));
    const Vec kOne(static_cast<T>(1.0));

    // Calculate 2u = 2 * sqrt(2/pi) * (x + 0.044715 * x^3)
    Vec x_sq  = x * x;
    Vec inner = at::vec::fmadd(kBeta, x_sq * x, x);
    Vec two_u = kAlphaTwo * inner;

    // Calculate x / (1 + exp(-2u))
    Vec exp_val = two_u.neg().exp();

    return x / (kOne + exp_val);
}

} // namespace

void
gelu_openmp(const float *in_ptr, float *out_ptr, int64_t numel) {
    using Vec                    = at::vec::Vectorized<float>;
    constexpr int64_t vec_len    = Vec::size();
    constexpr int64_t chunk_size = 2048;

    // Calculate number of chunks for OpenMP to distribute
    const int64_t num_chunks = (numel + chunk_size - 1) / chunk_size;

#pragma omp parallel for schedule(static)
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const int64_t chunk_start = chunk_idx * chunk_size;
        const int64_t chunk_end   = std::min(chunk_start + chunk_size, numel);

        int64_t i           = chunk_start;
        const int64_t limit = chunk_end - (chunk_end - chunk_start) % vec_len;

        // Main SIMD loop
        for (; i + vec_len <= limit; i += vec_len) {
            Vec data = Vec::loadu(in_ptr + i);
            data     = gelu_simd_op(data);
            data.store(out_ptr + i);
        }

        // Tail elements
        if (i < chunk_end) {
            const int64_t remaining = chunk_end - i;
            Vec data                = Vec::loadu(in_ptr + i, remaining);
            data                    = gelu_simd_op(data);
            data.store(out_ptr + i, remaining);
        }
    }
}

int
get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // namespace omp_gelu
