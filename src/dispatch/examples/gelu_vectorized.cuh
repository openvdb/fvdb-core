// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Vectorized GELU operations for improved memory throughput on GPU.
//
// Uses float4/double2 for 128-bit vectorized memory access.
// CUDA GPUs achieve best bandwidth with 128-bit (16-byte) transactions.
//
#ifndef DISPATCH_EXAMPLES_GELU_VECTORIZED_CUH
#define DISPATCH_EXAMPLES_GELU_VECTORIZED_CUH

#include "examples/gelu_scalar.h"

#include <cstdint>

namespace dispatch_examples {

//------------------------------------------------------------------------------
// Vector type traits
//------------------------------------------------------------------------------

template <typename T> struct vector_traits;

template <> struct vector_traits<float> {
    using vector_type                    = float4;
    static constexpr int64_t vector_size = 4;
};

template <> struct vector_traits<double> {
    using vector_type                    = double2;
    static constexpr int64_t vector_size = 2;
};

// Half types use float4 internally (promoted to float for computation)
template <> struct vector_traits<at::Half> {
    using vector_type                    = float4;
    static constexpr int64_t vector_size = 4;
};

template <> struct vector_traits<at::BFloat16> {
    using vector_type                    = float4;
    static constexpr int64_t vector_size = 4;
};

//------------------------------------------------------------------------------
// Vectorized GELU: float4 (4 floats at once)
//------------------------------------------------------------------------------

__device__ __forceinline__ float4
gelu_float4(float4 v) {
    return make_float4(gelu_scalar(v.x), gelu_scalar(v.y), gelu_scalar(v.z), gelu_scalar(v.w));
}

//------------------------------------------------------------------------------
// Vectorized GELU: double2 (2 doubles at once)
//------------------------------------------------------------------------------

__device__ __forceinline__ double2
gelu_double2(double2 v) {
    return make_double2(gelu_scalar(v.x), gelu_scalar(v.y));
}

//------------------------------------------------------------------------------
// Vectorized load/store helpers
//------------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ typename vector_traits<T>::vector_type
load_vector(T const *ptr) {
    using vec_t = typename vector_traits<T>::vector_type;
    return *reinterpret_cast<vec_t const *>(ptr);
}

template <typename T>
__device__ __forceinline__ void
store_vector(T *ptr, typename vector_traits<T>::vector_type vec) {
    using vec_t                     = typename vector_traits<T>::vector_type;
    *reinterpret_cast<vec_t *>(ptr) = vec;
}

// Specialization for float
template <>
__device__ __forceinline__ float4
load_vector<float>(float const *ptr) {
    return *reinterpret_cast<float4 const *>(ptr);
}

template <>
__device__ __forceinline__ void
store_vector<float>(float *ptr, float4 vec) {
    *reinterpret_cast<float4 *>(ptr) = vec;
}

// Specialization for double
template <>
__device__ __forceinline__ double2
load_vector<double>(double const *ptr) {
    return *reinterpret_cast<double2 const *>(ptr);
}

template <>
__device__ __forceinline__ void
store_vector<double>(double *ptr, double2 vec) {
    *reinterpret_cast<double2 *>(ptr) = vec;
}

//------------------------------------------------------------------------------
// Vectorized GELU dispatch by type
//------------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ typename vector_traits<T>::vector_type
gelu_vector(typename vector_traits<T>::vector_type v);

template <>
__device__ __forceinline__ float4
gelu_vector<float>(float4 v) {
    return gelu_float4(v);
}

template <>
__device__ __forceinline__ double2
gelu_vector<double>(double2 v) {
    return gelu_double2(v);
}

// Half/BFloat16 use float4 with conversion
template <>
__device__ __forceinline__ float4
gelu_vector<at::Half>(float4 v) {
    return gelu_float4(v);
}

template <>
__device__ __forceinline__ float4
gelu_vector<at::BFloat16>(float4 v) {
    return gelu_float4(v);
}

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_GELU_VECTORIZED_CUH
