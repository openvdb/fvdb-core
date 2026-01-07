#include "fvdb/detail/utils/DispatchSparse.h"

namespace fvdb {
namespace dispatch {
namespace example {

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

void
blub_impl(TorchDeviceCpuTag,
          TorchAccessor64<TorchDeviceCpuTag, float, 1> in,
          TorchAccessor64<TorchDeviceCpuTag, int, 1> out) {
    printf("blub_impl(Cpu, float, int)\n");
}

void
blub_impl(TorchDeviceCudaTag,
          TorchAccessor64<TorchDeviceCudaTag, float, 1> in,
          TorchAccessor64<TorchDeviceCudaTag, int, 1> out) {
    printf("blub_impl(Cuda, float, int)\n");
}

template <typename T>
void
blub_impl(TorchDeviceCpuTag,
          TorchAccessor64<TorchDeviceCpuTag, T, 1> in,
          TorchAccessor64<TorchDeviceCpuTag, T, 1> out) {
    printf("generic same-type blub_impl(Cpu, %s, %s)\n", typeid(T).name(), typeid(T).name());
}

template void blub_impl<int32_t>(TorchDeviceCpuTag,
                                 TorchAccessor64<TorchDeviceCpuTag, int32_t, 1>,
                                 TorchAccessor64<TorchDeviceCpuTag, int32_t, 1>);
template void blub_impl<int64_t>(TorchDeviceCpuTag,
                                 TorchAccessor64<TorchDeviceCpuTag, int64_t, 1>,
                                 TorchAccessor64<TorchDeviceCpuTag, int64_t, 1>);
template void blub_impl<torch::Half>(TorchDeviceCpuTag,
                                     TorchAccessor64<TorchDeviceCpuTag, torch::Half, 1>,
                                     TorchAccessor64<TorchDeviceCpuTag, torch::Half, 1>);
template void blub_impl<float>(TorchDeviceCpuTag,
                               TorchAccessor64<TorchDeviceCpuTag, float, 1>,
                               TorchAccessor64<TorchDeviceCpuTag, float, 1>);
template void blub_impl<double>(TorchDeviceCpuTag,
                                TorchAccessor64<TorchDeviceCpuTag, double, 1>,
                                TorchAccessor64<TorchDeviceCpuTag, double, 1>);

template <typename DeviceTag>
void
blub_impl(DeviceTag,
          TorchAccessor64<DeviceTag, double, 1> in,
          TorchAccessor64<DeviceTag, int, 1> out) {
    printf("any-device double-int-type blub_impl(%s, double, int)\n", typeid(DeviceTag).name());
}

template void blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                           TorchAccessor64<TorchDeviceCpuTag, double, 1>,
                                           TorchAccessor64<TorchDeviceCpuTag, int, 1>);
template void blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                            TorchAccessor64<TorchDeviceCudaTag, double, 1>,
                                            TorchAccessor64<TorchDeviceCudaTag, int, 1>);

//-----------------------------------------------------------------------------------
// The above is the mostly-not-boilerplate (the explicit instantiations are a little bit
// boilerplate-y) The stuff below is what we want to minimize.

// -------------------------------------------------------------------------
// Axis definitions: declare the extent of the dispatch space
// -------------------------------------------------------------------------
using BlubDeviceAxis = TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA>;
using BlubDtypeAxis  = TorchDtypeAxis<int32_t, int64_t, torch::Half, float, double>;
using BlubAxes       = AxisProduct<BlubDeviceAxis, BlubDtypeAxis, BlubDtypeAxis>;
using BlubSignature  = torch::Tensor(torch::Tensor, torch::ScalarType);

// -------------------------------------------------------------------------
// Factory: a constexpr lambda that maps axis types to a function pointer.
// C++20 templated lambdas eliminate the struct/operator() boilerplate.
// -------------------------------------------------------------------------
constexpr auto blub_factory = []<typename DeviceTag, typename InDtype, typename OutDtype>() {
    return [](torch::Tensor in, torch::ScalarType out_dtype) -> torch::Tensor {
        torch::Tensor out = torch::empty_like(in, in.options().dtype(out_dtype));
        auto in_accessor  = makeAccessor64<DeviceTag, InDtype, 1>(in);
        auto out_accessor = makeAccessor64<DeviceTag, OutDtype, 1>(out);
        blub_impl(DeviceTag{}, in_accessor, out_accessor);
        return out;
    };
};

// -----------------------------------------------------------------------------
// Operation-specific error handler for unsupported combinations.
// Receives axis values (device, in_dtype, out_dtype) followed by function args.
// -----------------------------------------------------------------------------
torch::Tensor
blub_unsupported(c10::DeviceType device,
                 c10::ScalarType in_dtype,
                 c10::ScalarType out_dtype,
                 torch::Tensor /*in*/,
                 torch::ScalarType /*out_dtype_arg*/) {
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
    BindFn<blub_factory, TorchDeviceCpuTag, int32_t, int32_t>,
    BindFn<blub_factory, TorchDeviceCpuTag, int64_t, int64_t>,
    BindFn<blub_factory, TorchDeviceCpuTag, torch::Half, torch::Half>,
    BindFn<blub_factory, TorchDeviceCpuTag, float, float>,
    BindFn<blub_factory, TorchDeviceCpuTag, double, double>,
    // Cross-type conversions (float/double -> int)
    BindFn<blub_factory, TorchDeviceCpuTag, float, int32_t>,
    BindFn<blub_factory, TorchDeviceCudaTag, float, int32_t>,
    BindFn<blub_factory, TorchDeviceCpuTag, double, int32_t>,
    BindFn<blub_factory, TorchDeviceCudaTag, double, int32_t>>;

// -----------------------------------------------------------------------------
// The general use "blub" function, which will handle all of the dispatch to the
// blub implementations above. This is the whole reason we're here.
// -----------------------------------------------------------------------------
torch::Tensor
blub(torch::Tensor in, torch::ScalarType out_dtype) {
    static constexpr auto table = make_table<BlubSignature, BlubAxes, BlubBindings>();
    auto const device           = in.device().type();
    auto const in_dtype         = in.scalar_type();
    return table.dispatch_or(blub_unsupported, device, in_dtype, out_dtype, in, out_dtype);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
