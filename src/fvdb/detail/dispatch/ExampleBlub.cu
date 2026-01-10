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
// DISPATCH INFRASTRUCTURE
// =============================================================================
// Everything below is dispatch machinery. The goal: minimal boilerplate that
// clearly expresses authoring intent.
//
// =============================================================================

// -----------------------------------------------------------------------------
// AXIS DEFINITIONS: The extent of the dispatch space
// -----------------------------------------------------------------------------
// These define WHAT we might dispatch over. The actual sparse coverage
// (which combinations are supported) is declared separately in bindings.

using BlubDeviceAxis = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;
using BlubDtypeAxis  = TorchScalarTypeAxis<torch::kInt32,
                                           torch::kInt64,
                                           torch::kFloat16,
                                           torch::kFloat32,
                                           torch::kFloat64>;

// Full dispatch space: Device × InputDtype × OutputDtype
using BlubAxes = AxisOuterProduct<BlubDeviceAxis, BlubDtypeAxis, BlubDtypeAxis>;

// -----------------------------------------------------------------------------
// TABLE TYPE: Function signature and dispatch table
// -----------------------------------------------------------------------------

using BlubFunctionPtr = torch::Tensor (*)(torch::Tensor, torch::ScalarType);
using BlubTable       = DispatchTable<torch::Tensor(torch::Tensor, torch::ScalarType), BlubAxes>;

// -----------------------------------------------------------------------------
// INVOKER: The actual implementation for each axis value combination
// -----------------------------------------------------------------------------
// This is the bridge between dispatch values and actual implementations.
// One invoker can serve many bindings if they share the same pattern.
// Note: Only invoke() is needed; WithGet adds the get() method automatically.

template <torch::DeviceType Device, torch::ScalarType InDtype, torch::ScalarType OutDtype>
struct BlubInvoker {
    static torch::Tensor
    invoke(torch::Tensor in, torch::ScalarType out_dtype) {
        auto out          = torch::empty_like(in, in.options().dtype(out_dtype));
        auto concrete_in  = ConcreteTensor<Device, InDtype, 1>{in};
        auto concrete_out = ConcreteTensor<Device, OutDtype, 1>{out};
        blub_impl(Tag<Device>{}, concrete_in, concrete_out);
        return out;
    }
};

// Wrap the invoker to add get() - this is what Binding/SubspaceBinding use
using BlubInstantiator = WithGet<BlubInvoker, BlubFunctionPtr>;

// -----------------------------------------------------------------------------
// BINDING ALIASES: Reduce repetition in binding declarations
// -----------------------------------------------------------------------------

template <auto... Values> using BlubBind = Binding<BlubInstantiator::Bind, Values...>;

template <typename Subspace>
using BlubSubspaceBind = SubspaceBinding<BlubInstantiator::Bind, Subspace>;

// -----------------------------------------------------------------------------
// SUBSPACE DEFINITIONS: Rectangular regions of supported combinations
// -----------------------------------------------------------------------------

// Float32 → Int32 on all devices
using Float32ToInt32Subspace = AxisOuterProduct<BlubDeviceAxis,
                                                TorchScalarTypeAxis<torch::kFloat32>,
                                                TorchScalarTypeAxis<torch::kInt32>>;

// Float64 → Int32 on all devices
using Float64ToInt32Subspace = AxisOuterProduct<BlubDeviceAxis,
                                                TorchScalarTypeAxis<torch::kFloat64>,
                                                TorchScalarTypeAxis<torch::kInt32>>;

// -----------------------------------------------------------------------------
// BINDING SPECIFICATION: Declares which combinations are supported
// -----------------------------------------------------------------------------
// This is THE declaration of intent - clean, minimal, and expressive.

// clang-format off
using BlubBindings = BindingList<
    // Same-type CPU bindings (diagonal entries - must be individual)
    BlubBind<torch::kCPU, torch::kInt32, torch::kInt32>,
    BlubBind<torch::kCPU, torch::kInt64, torch::kInt64>,
    BlubBind<torch::kCPU, torch::kFloat16, torch::kFloat16>,
    BlubBind<torch::kCPU, torch::kFloat32, torch::kFloat32>,
    BlubBind<torch::kCPU, torch::kFloat64, torch::kFloat64>,

    // Cross-type conversions (rectangular subspaces - both CPU and CUDA)
    BlubSubspaceBind<Float32ToInt32Subspace>,
    BlubSubspaceBind<Float64ToInt32Subspace>
>;
// clang-format on

// -----------------------------------------------------------------------------
// ERROR HANDLER: Called for unsupported combinations
// -----------------------------------------------------------------------------

BlubFunctionPtr
blub_unsupported(torch::DeviceType /*device*/,
                 torch::ScalarType /*in_dtype*/,
                 torch::ScalarType /*out_dtype*/) {
    return [](torch::Tensor in, torch::ScalarType out_dtype) -> torch::Tensor {
        TORCH_CHECK(false,
                    "blub: unsupported combination - device=",
                    in.device().type(),
                    ", in_dtype=",
                    in.scalar_type(),
                    ", out_dtype=",
                    out_dtype);
    };
}

// -----------------------------------------------------------------------------
// PUBLIC API: The minimal wrapper
// -----------------------------------------------------------------------------

torch::Tensor
blub(torch::Tensor in, torch::ScalarType out_dtype) {
    // Build table once, statically
    static auto const table = build_table<BlubTable, BlubBindings>();

    // Extract runtime values
    auto const device   = in.device().type();
    auto const in_dtype = in.scalar_type();

    // Dispatch with fallback for unsupported combinations
    auto fn_ptr = table.find_or(blub_unsupported, device, in_dtype, out_dtype);
    return fn_ptr(in, out_dtype);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
