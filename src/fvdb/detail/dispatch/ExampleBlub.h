// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H
#define FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"

#include <cstdint>

namespace fvdb {
namespace dispatch {
namespace example {

// -----------------------------------------------------------------------------
// Example: "blub" operation with sparse coverage
// -----------------------------------------------------------------------------

// -------------------------------------------------------------------------
// The blub_impl variations below represent "real" authoring content. Nothing in them
// is boilerplate beyond what we'd expect to have to write, nothing feels overly repetitive
// or onerous. We are able to separate out variations where it makes sense to do so for
// whatever our kernel/op requires. The examples here are contrived and an unlikely
// set of specializations, done really just to demonstrate and flex the dispatch system.
// Pretty much everything beyond these blub_impl implementations is boilerplate,
// so we want it to be as minimal as possible, and we want to offload as much as possible
// to the dispatch utility headers.
// -------------------------------------------------------------------------

void blub_impl(TorchDeviceCpuTag, CpuTensor<float, 1> in, CpuTensor<int, 1> out);

void blub_impl(TorchDeviceCudaTag, CudaTensor<float, 1> in, CudaTensor<int, 1> out);

template <typename T> void blub_impl(TorchDeviceCpuTag, CpuTensor<T, 1> in, CpuTensor<T, 1> out);

extern template void
    blub_impl<int32_t>(TorchDeviceCpuTag, CpuTensor<int32_t, 1>, CpuTensor<int32_t, 1>);
extern template void
    blub_impl<int64_t>(TorchDeviceCpuTag, CpuTensor<int64_t, 1>, CpuTensor<int64_t, 1>);
extern template void
    blub_impl<torch::Half>(TorchDeviceCpuTag, CpuTensor<torch::Half, 1>, CpuTensor<torch::Half, 1>);
extern template void blub_impl<float>(TorchDeviceCpuTag, CpuTensor<float, 1>, CpuTensor<float, 1>);
extern template void
    blub_impl<double>(TorchDeviceCpuTag, CpuTensor<double, 1>, CpuTensor<double, 1>);

template <typename DeviceTag>
void blub_impl(DeviceTag,
               ConcreteTensor<DeviceTag, double, 1> in,
               ConcreteTensor<DeviceTag, int, 1> out);

extern template void blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                                  ConcreteTensor<TorchDeviceCpuTag, double, 1>,
                                                  ConcreteTensor<TorchDeviceCpuTag, int, 1>);

extern template void blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                                   ConcreteTensor<TorchDeviceCudaTag, double, 1>,
                                                   ConcreteTensor<TorchDeviceCudaTag, int, 1>);

// -----------------------------------------------------------------------------
// The general use "blub" function, which will handle all of the dispatch to the
// blub implementations above. This is the whole reason we're here.
// -----------------------------------------------------------------------------
// Overload: output dtype defaults to input dtype
torch::Tensor blub(torch::Tensor in, torch::ScalarType out_dtype);

// Overload: output dtype defaults to input dtype
inline torch::Tensor
blub(torch::Tensor in) {
    return blub(in, in.scalar_type());
}

} // namespace example
} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H
