#if 1

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// CUDA specializations for TorchAccessor wrappers.
// This file should only be included by .cu files compiled with nvcc.
//
#ifndef FVDB_DETAIL_UTILS_DISPATCHSPARSE_CUH
#define FVDB_DETAIL_UTILS_DISPATCHSPARSE_CUH

#include "DispatchSparse.h"

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CUDA/PrivateUse1 specializations
//-----------------------------------------------------------------------------------

template <typename ScalarT, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<TorchDeviceCudaTag, ScalarT, Rank> ct) {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}

template <typename ScalarT, size_t Rank, typename IndexT = int64_t>
auto
accessor(ConcreteTensor<TorchDevicePrivateUse1Tag, ScalarT, Rank> ct) {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
        return ct.tensor.template packed_accessor64<ScalarT, Rank, at::RestrictPtrTraits>();
    } else {
        return ct.tensor.template packed_accessor32<ScalarT, Rank, at::RestrictPtrTraits>();
    }
}


} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_DISPATCHSPARSE_CUH

#endif // 0
