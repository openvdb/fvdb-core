// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSOR_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSOR_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"

#include <ATen/core/TensorAccessor.h>

#include <cstdint>
#include <type_traits>

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CPU accessor
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's accessor.
// Matches any ConcreteTensor with CPU device.

template <torch::ScalarType Stype, size_t Rank>
auto
accessor(ConcreteTensor<torch::kCPU, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<ScalarT, Rank>();
}

#ifdef __CUDACC__
//-----------------------------------------------------------------------------------
// ConcreteTensor CUDA/PrivateUse1 accessors (nvcc only)
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the ScalarType enum to call torch's packed_accessor.
// Matches any ConcreteTensor with CUDA or PrivateUse1 device.
// These use RestrictPtrTraits which requires nvcc compilation.

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<torch::kCUDA, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

template <torch::ScalarType Stype, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<torch::kPrivateUse1, Stype, Rank> ct) {
    using ScalarT = ScalarCppTypeT<Stype>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}
#endif // __CUDACC__

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSOR_H
