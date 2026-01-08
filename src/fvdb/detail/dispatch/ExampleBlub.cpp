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
// -------------------------------------------------------------------------

void
blub_impl(TorchDeviceCpuTag, CpuTensor<float, 1> in, CpuTensor<int, 1> out) {
    printf("blub_impl(Cpu, float, int)\n");
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    // Verify accessors work by accessing size
    printf("  in.size(0)=%ld, out.size(0)=%ld\n", in_accessor.size(0), out_accessor.size(0));
}

template <typename T>
void
blub_impl(TorchDeviceCpuTag, CpuTensor<T, 1> in, CpuTensor<T, 1> out) {
    printf("generic same-type blub_impl(Cpu, %s, %s)\n", typeid(T).name(), typeid(T).name());
    auto in_accessor  = accessor(in);
    auto out_accessor = accessor(out);
    // Verify accessors work by accessing size
    printf("  in.size(0)=%ld, out.size(0)=%ld\n", in_accessor.size(0), out_accessor.size(0));
}

template void blub_impl<int32_t>(TorchDeviceCpuTag, CpuTensor<int32_t, 1>, CpuTensor<int32_t, 1>);
template void blub_impl<int64_t>(TorchDeviceCpuTag, CpuTensor<int64_t, 1>, CpuTensor<int64_t, 1>);
template void
    blub_impl<torch::Half>(TorchDeviceCpuTag, CpuTensor<torch::Half, 1>, CpuTensor<torch::Half, 1>);
template void blub_impl<float>(TorchDeviceCpuTag, CpuTensor<float, 1>, CpuTensor<float, 1>);
template void blub_impl<double>(TorchDeviceCpuTag, CpuTensor<double, 1>, CpuTensor<double, 1>);

// -------------------------------------------------------------------------
// Axis definitions: declare the extent of the dispatch space
// Often, these can just use defaults like "all torch devices", and
// "all torch numeric types".
// -------------------------------------------------------------------------
using BlubDeviceAxis  = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;
using BlubDtypeAxis   = TorchDtypeAxis<int32_t, int64_t, torch::Half, float, double>;
using BlubAxes        = AxisProduct<BlubDeviceAxis, BlubDtypeAxis, BlubDtypeAxis>;
using BlubSignature   = torch::Tensor(torch::Tensor, torch::ScalarType);
using BlubFunctionPtr = BlubSignature *;

// -------------------------------------------------------------------------
// Factory: a template lambda that maps axis types to a function pointer.
// This uses C++20 template lambdas, which is safe here since this file is
// compiled only by the host compiler (not nvcc). NVCC doesn't support
// template lambdas being converted to function pointers when the template
// parameters aren't used in the parameter types.
// -------------------------------------------------------------------------

constexpr auto blub_impl_factory =
    []<typename DeviceTag, typename InDtype, typename OutDtype>() -> BlubFunctionPtr {
    return [](torch::Tensor in, torch::ScalarType out_dtype) -> torch::Tensor {
        torch::Tensor out = torch::empty_like(in, in.options().dtype(out_dtype));
        auto concrete_in  = ::fvdb::dispatch::ConcreteTensor<DeviceTag, InDtype, 1>{in};
        auto concrete_out = ::fvdb::dispatch::ConcreteTensor<DeviceTag, OutDtype, 1>{out};
        blub_impl(DeviceTag{}, concrete_in, concrete_out);
        return out;
    };
};

// -----------------------------------------------------------------------------
// Operation-specific error handler for unsupported combinations.
// Receives axis values (device, in_dtype, out_dtype) followed by function args.
// -----------------------------------------------------------------------------
torch::Tensor
blub_unsupported(torch::DeviceType device,
                 torch::ScalarType in_dtype,
                 torch::ScalarType out_dtype) {
    TORCH_CHECK(false,
                "blub: unsupported combination - device=",
                device,
                ", in_dtype=",
                in_dtype,
                ", out_dtype=",
                out_dtype);
}

// -------------------------------------------------------------------------
// Declarative binding specification: THIS IS THE INTENT
// The list below declares exactly which type combinations are supported.
// -------------------------------------------------------------------------
using BlubBindings = Bindings<
    // Same-type CPU bindings
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, int32_t, int32_t>,
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, int64_t, int64_t>,
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, torch::Half, torch::Half>,
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, float, float>,
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, double, double>,
    // Cross-type conversions (float/double -> int)
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, float, int32_t>,
    BindTypesFn<blub_impl_factory, TorchDeviceCudaTag, float, int32_t>,
    BindTypesFn<blub_impl_factory, TorchDeviceCpuTag, double, int32_t>,
    BindTypesFn<blub_impl_factory, TorchDeviceCudaTag, double, int32_t>>;

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
