// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H

#include "fvdb/detail/dispatch/TorchTags.h"

#include <c10/core/ScalarType.h>

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ScalarCppTypeT: Extract C++ type from a dtype Tag
//-----------------------------------------------------------------------------------
// Given Tag<torch::kFloat32>, yields float.
// This is used by accessor functions to get the actual C++ type for tensor access.

template <typename T> struct ScalarCppType;

template <torch::ScalarType S> struct ScalarCppType<Tag<S>> {
    using type = typename c10::impl::ScalarTypeToCPPType<S>::type;
};

template <typename T> using ScalarCppTypeT = typename ScalarCppType<T>::type;

//-----------------------------------------------------------------------------------
// TENSOR ACCESSOR WRAPPERS
//-----------------------------------------------------------------------------------
// ConcreteTensor is parameterized by device and dtype tags (enum values wrapped in Tag<>),
// not by C++ scalar types. This keeps the interface consistent with the dispatch system.
// Example: ConcreteTensor<Tag<torch::kCPU>, Tag<torch::kFloat32>, 2>

template <typename DeviceTag, typename DtypeTag, size_t Rank> struct ConcreteTensor {
    torch::Tensor tensor;
};

// Convenience aliases using device tags and dtype enum values
template <torch::ScalarType Dtype, size_t Rank>
using CpuTensor = ConcreteTensor<TorchDeviceCpuTag, Tag<Dtype>, Rank>;

template <torch::ScalarType Dtype, size_t Rank>
using CudaTensor = ConcreteTensor<TorchDeviceCudaTag, Tag<Dtype>, Rank>;

template <torch::ScalarType Dtype, size_t Rank>
using PrivateUse1Tensor = ConcreteTensor<TorchDevicePrivateUse1Tag, Tag<Dtype>, Rank>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
