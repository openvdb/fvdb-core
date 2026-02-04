// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// vectorization_policy.h - Op-defined vectorization strategies
//
// Vectorization is opt-in and op-specific. The op author defines a policy
// that specifies:
//   1. The vector width (elements per SIMD register)
//   2. The vector type for loads/stores
//   3. The vectorized operation itself
//
// The framework (views, for_each) consumes this policy but doesn't decide
// vectorization strategy - that's the op author's responsibility.
//
// Example usage:
//
//   template <torch::DeviceType Dev, torch::ScalarType Stype>
//   struct my_op_vec_policy;
//
//   template <>
//   struct my_op_vec_policy<torch::kCUDA, torch::kFloat32> {
//       static constexpr int64_t width = 4;
//       using vector_type = float4;
//       __device__ static vector_type load(float const* ptr) { ... }
//       __device__ static void store(float* ptr, vector_type v) { ... }
//       __device__ static vector_type apply(vector_type in) { ... }
//   };
//
#ifndef DISPATCH_DISPATCH_VECTORIZATION_POLICY_H
#define DISPATCH_DISPATCH_VECTORIZATION_POLICY_H

#include <cstdint>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Concept: is_vectorization_policy
//------------------------------------------------------------------------------
// A valid vectorization policy must provide:
//   - width: number of elements per vector
//   - vector_type: the SIMD type
//   - load(ptr): load vector from memory
//   - store(ptr, vec): store vector to memory
//   - apply(vec): the vectorized operation

template <typename Policy, typename = void> struct is_vectorization_policy : std::false_type {};

template <typename Policy>
struct is_vectorization_policy<
    Policy,
    std::void_t<decltype(Policy::width),
                typename Policy::vector_type,
                decltype(Policy::load(std::declval<typename Policy::scalar_type const *>())),
                decltype(Policy::store(std::declval<typename Policy::scalar_type *>(),
                                       std::declval<typename Policy::vector_type>())),
                decltype(Policy::apply(std::declval<typename Policy::vector_type>()))>>
    : std::true_type {};

template <typename Policy>
inline constexpr bool is_vectorization_policy_v = is_vectorization_policy<Policy>::value;

//------------------------------------------------------------------------------
// scalar_policy: fallback for non-vectorized ops
//------------------------------------------------------------------------------
// Used when vectorization is not beneficial or not implemented.

template <typename T, typename ApplyFn> struct scalar_policy {
    using scalar_type              = T;
    using vector_type              = T;
    static constexpr int64_t width = 1;

    static T
    load(T const *ptr) {
        return *ptr;
    }

    static void
    store(T *ptr, T val) {
        *ptr = val;
    }

    // apply is provided by the user via ApplyFn
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_VECTORIZATION_POLICY_H
