// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H

#include "fvdb/detail/dispatch/TorchTags.h"

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// TENSOR ACCESSOR WRAPPERS
//-----------------------------------------------------------------------------------

template <typename DeviceTag, typename ScalarT, size_t Rank> struct ConcreteTensor {
    torch::Tensor tensor;
};

template <typename ScalarT, size_t Rank>
using CpuTensor = ConcreteTensor<TorchDeviceCpuTag, ScalarT, Rank>;

template <typename ScalarT, size_t Rank>
using CudaTensor = ConcreteTensor<TorchDeviceCudaTag, ScalarT, Rank>;

template <typename ScalarT, size_t Rank>
using PrivateUse1Tensor = ConcreteTensor<TorchDevicePrivateUse1Tag, ScalarT, Rank>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSOR_H
