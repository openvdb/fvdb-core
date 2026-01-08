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
// ConcreteTensor CUDA/PrivateUse1 specializations
//-----------------------------------------------------------------------------------

template <typename ScalarT, size_t Rank, typename IndexT = int64_t>
auto
accessor(CudaTensor<ScalarT, Rank> ct) {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

template <typename ScalarT, size_t Rank, typename IndexT = int64_t>
auto
accessor(PrivateUse1Tensor<ScalarT, Rank> ct) {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORDEVICE_CUH
