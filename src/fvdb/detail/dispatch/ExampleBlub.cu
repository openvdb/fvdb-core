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
// -------------------------------------------------------------------------

void
blub_impl(TorchDeviceCudaTag, CudaTensor<float, 1> in, CudaTensor<int, 1> out) {
    printf("blub_impl(Cuda, float, int)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

// -------------------------------------------------------------------------
// Device-generic template for double->int conversion
// Template definition here since it uses ConcreteTensor<DeviceTag, ...>
// which resolves to PackedTensorAccessor for CUDA/PrivateUse1 device tags.
// Both CPU and CUDA instantiations provided here to satisfy ODR.
// -------------------------------------------------------------------------

template <typename DeviceTag>
void
blub_impl(DeviceTag,
          ConcreteTensor<DeviceTag, double, 1> in,
          ConcreteTensor<DeviceTag, int, 1> out) {
    printf("any-device double-int-type blub_impl(%s, double, int)\n", typeid(DeviceTag).name());
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

template void blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                           ConcreteTensor<TorchDeviceCpuTag, double, 1>,
                                           ConcreteTensor<TorchDeviceCpuTag, int, 1>);
template void blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                            ConcreteTensor<TorchDeviceCudaTag, double, 1>,
                                            ConcreteTensor<TorchDeviceCudaTag, int, 1>);

} // namespace example
} // namespace dispatch
} // namespace fvdb
