// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef DISPATCH_CONCRETE_ACCESSOR_H
#define DISPATCH_CONCRETE_ACCESSOR_H

#include "concrete_tensor.h"

#include <ATen/core/TensorAccessor.h>

#include <cstdint>
#include <type_traits>

namespace dispatch {

//-----------------------------------------------------------------------------------
// concrete_tensor CPU accessor
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's accessor.
// Matches any concrete_tensor with CPU device.

template <torch::ScalarType Stype, size_t Rank>
auto
accessor(concrete_tensor<torch::kCPU, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<ScalarT, Rank>();
}

#ifdef __CUDACC__
//-----------------------------------------------------------------------------------
// concrete_tensor CUDA/PrivateUse1 accessors (nvcc only)
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's packed_accessor.
// Matches any concrete_tensor with CUDA or PrivateUse1 device.
// These use RestrictPtrTraits which requires nvcc compilation.

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
accessor(concrete_tensor<torch::kCUDA, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
accessor(concrete_tensor<torch::kPrivateUse1, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}
#endif // __CUDACC__

} // namespace dispatch

#endif // DISPATCH_CONCRETE_ACCESSOR_H
