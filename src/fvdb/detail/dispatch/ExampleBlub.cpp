// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "fvdb/detail/dispatch/ExampleBlub.h"

#include "fvdb/detail/dispatch/ConcreteTensorAccessorHost.h"
#include "fvdb/detail/dispatch/SparseDispatchTable.h"
#include "fvdb/detail/dispatch/TorchDispatchHost.h"

namespace fvdb {
namespace dispatch {
namespace example {

// -------------------------------------------------------------------------
// CPU-only implementations (no CUDA accessor types needed)
// All tensor parameters use enum values for dtype.
// -------------------------------------------------------------------------

void
blub_impl(TorchDeviceCpuTag, CpuTensor<torch::kFloat32, 1> in, CpuTensor<torch::kInt32, 1> out) {
    printf("blub_impl(Cpu, kFloat32, kInt32)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    // Verify accessors work by accessing size
    printf("  in.size(0)=%ld, out.size(0)=%ld\n", in_accessor.size(0), out_accessor.size(0));
}

template <torch::ScalarType Dtype>
void
blub_impl(TorchDeviceCpuTag, CpuTensor<Dtype, 1> in, CpuTensor<Dtype, 1> out) {
    printf("generic same-type blub_impl(Cpu, dtype=%d)\n", static_cast<int>(Dtype));
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    // Verify accessors work by accessing size
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

// -------------------------------------------------------------------------
// Axis definitions: declare the extent of the dispatch space
// Often, these can just use defaults like "all torch devices", and
// "all torch numeric types".
// All axes use enum values, not C++ types.
// -------------------------------------------------------------------------
using BlubDeviceAxis = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;
using BlubDtypeAxis =
    TorchDtypeAxis<torch::kInt32, torch::kInt64, torch::kFloat16, torch::kFloat32, torch::kFloat64>;
using BlubAxes        = AxisProduct<BlubDeviceAxis, BlubDtypeAxis, BlubDtypeAxis>;
using BlubSignature   = torch::Tensor(torch::Tensor, torch::ScalarType);
using BlubFunctionPtr = BlubSignature *;

// -------------------------------------------------------------------------
// Factory: a template lambda that maps axis types to a function pointer.
// This uses C++20 template lambdas, which is safe here since this file is
// compiled only by the host compiler (not nvcc). NVCC doesn't support
// template lambdas being converted to function pointers when the template
// parameters aren't used in the parameter types.
//
// The factory receives Tag types for each axis:
// - DeviceTag: Tag<torch::kCPU> or Tag<torch::kCUDA>
// - InDtypeTag/OutDtypeTag: Tag<torch::kFloat32>, Tag<torch::kInt32>, etc.
// We extract the actual C++ scalar types using ScalarCppTypeT.
// -------------------------------------------------------------------------

constexpr auto blub_impl_factory =
    []<typename DeviceTag, typename InDtypeTag, typename OutDtypeTag>() -> BlubFunctionPtr {
    // DeviceTag is Tag<torch::kCPU> or Tag<torch::kCUDA>
    // InDtypeTag/OutDtypeTag are Tag<torch::kFloat32>, Tag<torch::kInt32>, etc.
    // ConcreteTensor now takes the dtype tag directly, not the C++ type.

    return [](torch::Tensor in, torch::ScalarType out_dtype) -> torch::Tensor {
        torch::Tensor out = torch::empty_like(in, in.options().dtype(out_dtype));
        auto concrete_in  = ::fvdb::dispatch::ConcreteTensor<DeviceTag, InDtypeTag, 1>{in};
        auto concrete_out = ::fvdb::dispatch::ConcreteTensor<DeviceTag, OutDtypeTag, 1>{out};
        blub_impl(DeviceTag{}, concrete_in, concrete_out);
        return out;
    };
};

// -----------------------------------------------------------------------------
// Operation-specific error handler for unsupported combinations.
// Returns a function pointer that, when called, throws an error.
// -----------------------------------------------------------------------------
BlubFunctionPtr
blub_unsupported(torch::DeviceType device,
                 torch::ScalarType in_dtype,
                 torch::ScalarType out_dtype) {
    // Capture error info and return a lambda that throws when invoked
    // Note: We use a static to avoid dangling captures; the error message
    // is generated at call time from the actual tensor properties anyway.
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

// -------------------------------------------------------------------------
// Declarative binding specification: THIS IS THE INTENT
// The list below declares exactly which value combinations are supported.
// All bindings use enum values, matching the axis definitions.
// -------------------------------------------------------------------------
using BlubBindings = Bindings<
    // Same-type CPU bindings
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kInt32, torch::kInt32>,
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kInt64, torch::kInt64>,
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kFloat16, torch::kFloat16>,
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kFloat32, torch::kFloat32>,
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kFloat64, torch::kFloat64>,
    // Cross-type conversions (float/double -> int)
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kFloat32, torch::kInt32>,
    BindValuesFn<blub_impl_factory, torch::kCUDA, torch::kFloat32, torch::kInt32>,
    BindValuesFn<blub_impl_factory, torch::kCPU, torch::kFloat64, torch::kInt32>,
    BindValuesFn<blub_impl_factory, torch::kCUDA, torch::kFloat64, torch::kInt32>>;

// -----------------------------------------------------------------------------
// The general use "blub" function, which will handle all of the dispatch to the
// blub implementations above. This is the whole reason we're here.
// -----------------------------------------------------------------------------
torch::Tensor
blub(torch::Tensor in, torch::ScalarType out_dtype) {
    static auto const table = make_table<BlubSignature, BlubAxes, BlubBindings>();
    auto const device       = in.device().type();
    auto const in_dtype     = in.scalar_type();

    auto fn_ptr = table.find_or(blub_unsupported, device, in_dtype, out_dtype);
    return fn_ptr(in, out_dtype);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
