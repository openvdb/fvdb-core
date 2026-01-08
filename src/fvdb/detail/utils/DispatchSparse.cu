#if 1

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "fvdb/detail/utils/DispatchSparse.cuh"

namespace fvdb {
namespace dispatch {
namespace example {

// -------------------------------------------------------------------------
// CUDA implementation (requires nvcc for PackedTensorAccessor types)
// -------------------------------------------------------------------------

void
blub_impl(TorchDeviceCudaTag,
          ConcreteTensor<TorchDeviceCudaTag, float, 1> in,
          ConcreteTensor<TorchDeviceCudaTag, int, 1> out) {
    printf("blub_impl(Cuda, float, int)\n");
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
}

template void blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                           ConcreteTensor<TorchDeviceCpuTag, double, 1>,
                                           ConcreteTensor<TorchDeviceCpuTag, int, 1>);
template void blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                            ConcreteTensor<TorchDeviceCudaTag, double, 1>,
                                            ConcreteTensor<TorchDeviceCudaTag, int, 1>);

// -------------------------------------------------------------------------
// Axis definitions: declare the extent of the dispatch space
// -------------------------------------------------------------------------
using BlubDeviceAxis = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;
using BlubDtypeAxis  = TorchDtypeAxis<int32_t, int64_t, torch::Half, float, double>;
using BlubAxes       = AxisProduct<BlubDeviceAxis, BlubDtypeAxis, BlubDtypeAxis>;
using BlubSignature  = torch::Tensor(torch::Tensor, torch::ScalarType);

// -------------------------------------------------------------------------
// Factory: a template struct that maps axis types to a function pointer.
// Must be in .cu because it instantiates ConcreteTensor for CUDA device tags,
// whose specializations require nvcc (defined in DispatchSparse.cuh).
// Note: We use a template struct instead of a template lambda because NVCC
// doesn't support template lambdas being converted to function pointers when
// the template parameters aren't used in the parameter types.
// -------------------------------------------------------------------------
template <typename DeviceTag, typename InDtype, typename OutDtype>
struct BlubFactory {
    static torch::Tensor
    impl(torch::Tensor in, torch::ScalarType out_dtype) {
        torch::Tensor out = torch::empty_like(in, in.options().dtype(out_dtype));
        auto concrete_in  = ::fvdb::dispatch::ConcreteTensor<DeviceTag, InDtype, 1>{in};
        auto concrete_out = ::fvdb::dispatch::ConcreteTensor<DeviceTag, OutDtype, 1>{out};
        blub_impl(DeviceTag{}, concrete_in, concrete_out);
        return out;
    }

    auto
    operator()() const {
        return &impl;
    }
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
    BindAt<BlubFactory, TorchDeviceCpuTag, int32_t, int32_t>,
    BindAt<BlubFactory, TorchDeviceCpuTag, int64_t, int64_t>,
    BindAt<BlubFactory, TorchDeviceCpuTag, torch::Half, torch::Half>,
    BindAt<BlubFactory, TorchDeviceCpuTag, float, float>,
    BindAt<BlubFactory, TorchDeviceCpuTag, double, double>,
    // Cross-type conversions (float/double -> int)
    BindAt<BlubFactory, TorchDeviceCpuTag, float, int32_t>,
    BindAt<BlubFactory, TorchDeviceCudaTag, float, int32_t>,
    BindAt<BlubFactory, TorchDeviceCpuTag, double, int32_t>,
    BindAt<BlubFactory, TorchDeviceCudaTag, double, int32_t>>;

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

#endif // 0
