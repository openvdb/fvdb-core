// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// OpenMP GELU implementation header.
// The implementation is in omp_gelu.cpp to ensure OpenMP pragmas are
// processed by g++ directly, not nvcc.

#pragma once

#ifndef BENCHMARKS_DISPATCH_OMP_GELU_H
#define BENCHMARKS_DISPATCH_OMP_GELU_H
#include <cstdint>

namespace omp_gelu {

/// @brief GELU using raw OpenMP parallel for with SIMD vectorization.
/// @param in_ptr Input float array
/// @param out_ptr Output float array
/// @param numel Number of elements
void gelu_openmp(const float *in_ptr, float *out_ptr, int64_t numel);

/// @brief Returns the number of OpenMP threads that will be used.
int get_num_threads();

} // namespace omp_gelu

#endif // BENCHMARKS_DISPATCH_OMP_GELU_H
