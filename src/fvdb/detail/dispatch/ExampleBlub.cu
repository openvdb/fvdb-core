// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "fvdb/detail/dispatch/ExampleBlub.h"

#include "fvdb/detail/dispatch/ConcreteTensorAccessor.h"
#include "fvdb/detail/dispatch/SparseDispatchTable.h"
#include "fvdb/detail/dispatch/TorchDispatch.h"

namespace fvdb {
namespace dispatch {
namespace example {

// =============================================================================
// IMPLEMENTATIONS
// =============================================================================
// These blub_impl functions represent "real" authoring content - the actual
// kernel/operation logic. Nothing here is boilerplate. We separate variations
// where it makes sense for our specific application needs.

// -----------------------------------------------------------------------------
// CPU-only implementations
// -----------------------------------------------------------------------------

void
blub_impl(TorchDeviceCpuTag, CpuTensor<torch::kFloat32, 1> in, CpuTensor<torch::kInt32, 1> out) {
    printf("blub_impl(Cpu, kFloat32, kInt32)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n", in_accessor.size(0), out_accessor.size(0));
}

template <torch::ScalarType Dtype>
void
blub_impl(TorchDeviceCpuTag, CpuTensor<Dtype, 1> in, CpuTensor<Dtype, 1> out) {
    printf("generic same-type blub_impl(Cpu, dtype=%d)\n", static_cast<int>(Dtype));
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n", in_accessor.size(0), out_accessor.size(0));
}

template void blub_impl<torch::kInt32>(TorchDeviceCpuTag,
                                       CpuTensor<torch::kInt32, 1>,
                                       CpuTensor<torch::kInt32, 1>);
template void blub_impl<torch::kInt64>(TorchDeviceCpuTag,
                                       CpuTensor<torch::kInt64, 1>,
                                       CpuTensor<torch::kInt64, 1>);
template void blub_impl<torch::kFloat16>(TorchDeviceCpuTag,
                                         CpuTensor<torch::kFloat16, 1>,
                                         CpuTensor<torch::kFloat16, 1>);
template void blub_impl<torch::kFloat32>(TorchDeviceCpuTag,
                                         CpuTensor<torch::kFloat32, 1>,
                                         CpuTensor<torch::kFloat32, 1>);
template void blub_impl<torch::kFloat64>(TorchDeviceCpuTag,
                                         CpuTensor<torch::kFloat64, 1>,
                                         CpuTensor<torch::kFloat64, 1>);

// -----------------------------------------------------------------------------
// CUDA implementation
// -----------------------------------------------------------------------------

void
blub_impl(TorchDeviceCudaTag, CudaTensor<torch::kFloat32, 1> in, CudaTensor<torch::kInt32, 1> out) {
    printf("blub_impl(Cuda, kFloat32, kInt32)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

// -----------------------------------------------------------------------------
// Device-generic template (works on CPU and CUDA)
// -----------------------------------------------------------------------------

template <torch::DeviceType DeviceValue>
void
blub_impl(Tag<DeviceValue>,
          ConcreteTensor<DeviceValue, torch::kFloat64, 1> in,
          ConcreteTensor<DeviceValue, torch::kInt32, 1> out) {
    printf("any-device kFloat64->kInt32 blub_impl(%d)\n", static_cast<int>(DeviceValue));
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    printf("  in.size(0)=%ld, out.size(0)=%ld\n",
           (long)in_accessor.size(0),
           (long)out_accessor.size(0));
}

template void blub_impl<torch::kCPU>(Tag<torch::kCPU>,
                                     ConcreteTensor<torch::kCPU, torch::kFloat64, 1>,
                                     ConcreteTensor<torch::kCPU, torch::kInt32, 1>);
template void blub_impl<torch::kCUDA>(Tag<torch::kCUDA>,
                                      ConcreteTensor<torch::kCUDA, torch::kFloat64, 1>,
                                      ConcreteTensor<torch::kCUDA, torch::kInt32, 1>);

// =============================================================================
// DISPATCH TABLE CONFIGURATION
// =============================================================================
// This section defines the dispatch space, invoker, and bindings for blub.
// The goal is to be declarative: specify WHAT combinations are supported,
// and let the infrastructure handle HOW to dispatch.

// -----------------------------------------------------------------------------
// Encoder and error handler
// -----------------------------------------------------------------------------

auto
blub_encode(torch::Tensor in, torch::ScalarType outScalarType) {
    return std::make_tuple(in.device().type(), in.scalar_type(), outScalarType);
}

[[noreturn]] void
blub_unbound_error(torch::DeviceType deviceType,
                   torch::ScalarType inScalarType,
                   torch::ScalarType outScalarType) {
    TORCH_CHECK(false,
                "Blub: Unsupported combination - device: ",
                deviceType,
                ", input dtype: ",
                inScalarType,
                ", output dtype: ",
                outScalarType);
}

// -----------------------------------------------------------------------------
// Axis definitions - the full space of possible dispatch combinations
// -----------------------------------------------------------------------------

using BlubDeviceAxis     = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;
using BlubScalarTypeAxis = TorchScalarTypeAxis<torch::kInt32,
                                               torch::kInt64,
                                               torch::kFloat16,
                                               torch::kFloat32,
                                               torch::kFloat64>;
using BlubAxes           = AxisOuterProduct<BlubDeviceAxis, BlubScalarTypeAxis, BlubScalarTypeAxis>;

// -----------------------------------------------------------------------------
// Invoker - the bridge between runtime dispatch and static implementation
// -----------------------------------------------------------------------------

template <torch::DeviceType DeviceValue,
          torch::ScalarType InScalarType,
          torch::ScalarType OutScalarType>
struct BlubInvoker {
    static torch::Tensor
    invoke(torch::Tensor in, torch::ScalarType outScalarType) {
        // Create output tensor with the specified dtype on the same device
        auto out =
            torch::empty_like(in, torch::TensorOptions().dtype(OutScalarType).device(in.device()));

        // Concretize tensors with compile-time type information
        auto concreteIn  = ConcreteTensor<DeviceValue, InScalarType, 1>(in);
        auto concreteOut = ConcreteTensor<DeviceValue, OutScalarType, 1>(out);

        // Dispatch to the implementation - overload resolution picks the right one
        blub_impl(Tag<DeviceValue>(), concreteIn, concreteOut);

        return out;
    }
};

// -----------------------------------------------------------------------------
// Bindings - declare which combinations are actually implemented
// -----------------------------------------------------------------------------
// Each generator corresponds to one or more blub_impl overloads above.
// The structure mirrors the implementation coverage.
template <auto... Values> using BlubBind = PointGenerator<GetFromInvoke<BlubInvoker>, Values...>;

template <typename Subspace>
using BlubSubspaceBind = SubspaceGenerator<GetFromInvoke<BlubInvoker>, Subspace>;

using BlubBindings = GeneratorList<
    // CPU Float32 -> Int32: specific overload blub_impl(Cpu, Float32, Int32)
    BlubBind<torch::kCPU, torch::kFloat32, torch::kInt32>,

    // CUDA Float32 -> Int32: specific overload blub_impl(Cuda, Float32, Int32)
    BlubBind<torch::kCUDA, torch::kFloat32, torch::kInt32>,

    // CPU same-type: generic template blub_impl<Dtype>(Cpu, in, out)
    BlubBind<torch::kCPU, torch::kInt32, torch::kInt32>,
    BlubBind<torch::kCPU, torch::kInt64, torch::kInt64>,
    BlubBind<torch::kCPU, torch::kFloat16, torch::kFloat16>,
    BlubBind<torch::kCPU, torch::kFloat32, torch::kFloat32>,
    BlubBind<torch::kCPU, torch::kFloat64, torch::kFloat64>,

    // Any-device Float64 -> Int32: device-generic template blub_impl<Device>(...)
    BlubSubspaceBind<AxisOuterProduct<BlubDeviceAxis,
                                      TorchScalarTypeAxis<torch::kFloat64>,
                                      TorchScalarTypeAxis<torch::kInt32>>>>;

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

torch::Tensor
blub(torch::Tensor in, torch::ScalarType outScalarType) {
    static const auto dispatchTable =
        build_dispatcher<BlubAxes, BlubBindings, torch::Tensor, torch::Tensor, torch::ScalarType>();

    return dispatchTable(blub_encode, blub_unbound_error, in, outScalarType);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
