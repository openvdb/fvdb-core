// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __CUDACC__
#error "This header must only be included during nvcc compilation"
#endif

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORDEVICE_CUH
#define FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORDEVICE_CUH

#include "fvdb/detail/dispatch/ConcreteTensor.h"

#include <ATen/core/TensorAccessor.h>

#include <cstdint>
#include <type_traits>

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CUDA/PrivateUse1 accessors
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the DtypeTag to call torch's packed_accessor.
// Matches any ConcreteTensor with CUDA or PrivateUse1 device tags.

template <typename DtypeTag, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<TorchDeviceCudaTag, DtypeTag, Rank> ct) {
    using ScalarT = ScalarCppTypeT<DtypeTag>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

template <typename DtypeTag, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<TorchDevicePrivateUse1Tag, DtypeTag, Rank> ct) {
    using ScalarT = ScalarCppTypeT<DtypeTag>;
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORDEVICE_CUH
