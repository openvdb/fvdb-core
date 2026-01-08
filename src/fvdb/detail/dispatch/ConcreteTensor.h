// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H

#include <torch/types.h>

namespace fvdb {
namespace dispatch {

template <torch::ScalarType S>
using ScalarCppTypeT = typename c10::impl::ScalarTypeToCPPType<S>::type;

//-----------------------------------------------------------------------------------
// TENSOR ACCESSOR WRAPPERS
//-----------------------------------------------------------------------------------
// ConcreteTensor is parameterized by device and dtype tags (enum values wrapped in Tag<>),
// not by C++ scalar types. This keeps the interface consistent with the dispatch system.
// Example: ConcreteTensor<Tag<torch::kCPU>, Tag<torch::kFloat32>, 2>

// The typename of the value torch::kCPU is torch::DeviceType

template <torch::DeviceType Device, torch::ScalarType Stype, size_t Rank> struct ConcreteTensor {
    static constexpr torch::DeviceType DeviceValue     = Device;
    static constexpr torch::ScalarType ScalarTypeValue = Stype;

    torch::Tensor tensor;
    ConcreteTensor() = default;
    explicit ConcreteTensor(torch::Tensor t) : tensor(t) {
        TORCH_CHECK_VALUE(t.device().type() == DeviceValue, "Device mismatch");
        TORCH_CHECK_VALUE(t.scalar_type() == ScalarTypeValue, "Scalar type mismatch");
    }
};

// Convenience aliases using device tags and dtype enum values
template <torch::ScalarType Stype, size_t Rank>
using CpuTensor = ConcreteTensor<torch::kCPU, Stype, Rank>;

template <torch::ScalarType Stype, size_t Rank>
using CudaTensor = ConcreteTensor<torch::kCUDA, Stype, Rank>;

template <torch::ScalarType Stype, size_t Rank>
using PrivateUse1Tensor = ConcreteTensor<torch::kPrivateUse1, Stype, Rank>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
