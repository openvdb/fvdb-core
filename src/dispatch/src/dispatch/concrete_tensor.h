// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef DISPATCH_CONCRETE_TENSOR_H
#define DISPATCH_CONCRETE_TENSOR_H

#include <torch/types.h>

namespace dispatch {

template <torch::ScalarType S>
using scalar_cpp_type_t = typename c10::impl::ScalarTypeToCPPType<S>::type;

//-----------------------------------------------------------------------------------
// TENSOR ACCESSOR WRAPPERS
//-----------------------------------------------------------------------------------
// concrete_tensor is parameterized by device and dtype tags,
// not by C++ scalar types. This keeps the interface consistent with the dispatch system.
// Example: concrete_tensor<Tag<torch::kCPU>, Tag<torch::kFloat32>, 2>

// The typename of the value torch::kCPU is torch::DeviceType

template <torch::DeviceType Device, torch::ScalarType Stype, size_t Rank> struct concrete_tensor {
    static constexpr torch::DeviceType DeviceValue     = Device;
    static constexpr torch::ScalarType ScalarTypeValue = Stype;
    using value_type                                   = scalar_cpp_type_t<Stype>;

    torch::Tensor tensor;
    concrete_tensor() = default;
    explicit concrete_tensor(torch::Tensor t) : tensor(t) {
        TORCH_CHECK_VALUE(t.device().type() == DeviceValue, "Device mismatch");
        TORCH_CHECK_VALUE(t.scalar_type() == ScalarTypeValue, "Scalar type mismatch");
    }
};

// Convenience aliases using device tags and dtype enum values
template <torch::ScalarType Stype, size_t Rank>
using cpu_tensor = concrete_tensor<torch::kCPU, Stype, Rank>;

template <torch::ScalarType Stype, size_t Rank>
using cuda_tensor = concrete_tensor<torch::kCUDA, Stype, Rank>;

template <torch::ScalarType Stype, size_t Rank>
using pvt1_tensor = concrete_tensor<torch::kPrivateUse1, Stype, Rank>;

} // namespace dispatch

#endif // DISPATCH_CONCRETE_TENSOR_H
