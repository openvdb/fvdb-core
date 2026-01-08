// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H
#define FVDB_DETAIL_DISPATCH_EXAMPLEBLUB_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"
#include "fvdb/detail/dispatch/TorchTags.h"

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

// All tensor parameters use enum values for dtype (e.g., torch::kFloat32, not float)
void
blub_impl(TorchDeviceCpuTag, CpuTensor<torch::kFloat32, 1> in, CpuTensor<torch::kInt32, 1> out);

void
blub_impl(TorchDeviceCudaTag, CudaTensor<torch::kFloat32, 1> in, CudaTensor<torch::kInt32, 1> out);

template <torch::ScalarType Dtype>
void blub_impl(TorchDeviceCpuTag, CpuTensor<Dtype, 1> in, CpuTensor<Dtype, 1> out);

extern template void blub_impl<torch::kInt32>(TorchDeviceCpuTag,
                                              CpuTensor<torch::kInt32, 1>,
                                              CpuTensor<torch::kInt32, 1>);
extern template void blub_impl<torch::kInt64>(TorchDeviceCpuTag,
                                              CpuTensor<torch::kInt64, 1>,
                                              CpuTensor<torch::kInt64, 1>);
extern template void blub_impl<torch::kFloat16>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat16, 1>,
                                                CpuTensor<torch::kFloat16, 1>);
extern template void blub_impl<torch::kFloat32>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat32, 1>,
                                                CpuTensor<torch::kFloat32, 1>);
extern template void blub_impl<torch::kFloat64>(TorchDeviceCpuTag,
                                                CpuTensor<torch::kFloat64, 1>,
                                                CpuTensor<torch::kFloat64, 1>);

template <torch::DeviceType DeviceValue>
void blub_impl(Tag<DeviceValue>,
               ConcreteTensor<DeviceValue, torch::kFloat64, 1> in,
               ConcreteTensor<DeviceValue, torch::kInt32, 1> out);

extern template void blub_impl<torch::kCPU>(Tag<torch::kCPU>,
                                            ConcreteTensor<torch::kCPU, torch::kFloat64, 1>,
                                            ConcreteTensor<torch::kCPU, torch::kInt32, 1>);

extern template void blub_impl<torch::kCUDA>(Tag<torch::kCUDA>,
                                             ConcreteTensor<torch::kCUDA, torch::kFloat64, 1>,
                                             ConcreteTensor<torch::kCUDA, torch::kInt32, 1>);

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
