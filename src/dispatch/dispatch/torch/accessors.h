// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch tensor accessor factories for CPU, CUDA, and PrivateUse1 devices.
//
#ifndef DISPATCH_DISPATCH_TORCH_ACCESSORS_H
#define DISPATCH_DISPATCH_TORCH_ACCESSORS_H

#include "dispatch/torch/types.h"

#include <ATen/core/TensorAccessor.h>

#include <cstdint>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// torch_concrete_tensor CPU accessor
//------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's accessor.
// Matches any torch_concrete_tensor with CPU device.

template <torch::ScalarType Stype, size_t Rank>
auto
torch_accessor(torch_concrete_tensor<torch::kCPU, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<scalar_t, Rank>();
}

#ifdef __CUDACC__
//------------------------------------------------------------------------------
// torch_concrete_tensor CUDA/PrivateUse1 accessors (nvcc only)
//------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's packed_accessor.
// Matches any torch_concrete_tensor with CUDA or PrivateUse1 device.
// These use RestrictPtrTraits which requires nvcc compilation.

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
torch_accessor(torch_concrete_tensor<torch::kCUDA, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<scalar_t, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<scalar_t, Rank, at::RestrictPtrTraits>();
    }
}

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
torch_accessor(torch_concrete_tensor<torch::kPrivateUse1, Stype, Rank> ct) {
    using scalar_t = torch_scalar_cpp_type_t<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<scalar_t, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<scalar_t, Rank, at::RestrictPtrTraits>();
    }
}
#endif // __CUDACC__

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_ACCESSORS_H
