// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "fvdb/detail/dispatch/ExampleBlub.h"

#include "fvdb/detail/dispatch/ConcreteTensorAccessorDevice.cuh"
#include "fvdb/detail/dispatch/ConcreteTensorAccessorHost.h"

namespace fvdb {
namespace dispatch {
namespace example {

// -------------------------------------------------------------------------
// CUDA implementation (requires nvcc for PackedTensorAccessor types)
// All tensor parameters use enum values for dtype.
// -------------------------------------------------------------------------

void
blub_impl(TorchDeviceCudaTag, CudaTensor<torch::kFloat32, 1> in, CudaTensor<torch::kInt32, 1> out) {
    printf("blub_impl(Cuda, kFloat32, kInt32)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

// -------------------------------------------------------------------------
// Device-generic template for double->int conversion
// Template definition here since it uses ConcreteTensor<DeviceTag, DtypeTag, Rank>
// Both CPU and CUDA instantiations provided here to satisfy ODR.
// -------------------------------------------------------------------------

template <typename DeviceTag>
void
blub_impl(DeviceTag,
          ConcreteTensor<DeviceTag, Tag<torch::kFloat64>, 1> in,
          ConcreteTensor<DeviceTag, Tag<torch::kInt32>, 1> out) {
    printf("any-device kFloat64->kInt32 blub_impl(%s)\n", typeid(DeviceTag).name());
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

template void
    blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                 ConcreteTensor<TorchDeviceCpuTag, Tag<torch::kFloat64>, 1>,
                                 ConcreteTensor<TorchDeviceCpuTag, Tag<torch::kInt32>, 1>);
template void
    blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                  ConcreteTensor<TorchDeviceCudaTag, Tag<torch::kFloat64>, 1>,
                                  ConcreteTensor<TorchDeviceCudaTag, Tag<torch::kInt32>, 1>);

} // namespace example
} // namespace dispatch
} // namespace fvdb
