// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "fvdb/detail/dispatch/example/Functional.h"

#include "fvdb/detail/dispatch/SparseDispatchTable.h"
#include "fvdb/detail/dispatch/Tag.h"
#include "fvdb/detail/dispatch/ValueSpace.h"
#include "fvdb/detail/dispatch/Values.h"
#include "fvdb/detail/dispatch/example/ScanLib.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace dispatch {
namespace example {

// we don't check that the input matches the tag, we rely on the dispatch system working correctly.

// On cpu, for out of place with any stype, contiguity or determinism, we will use the serial
// option.
template <torch::ScalarType stype, Contiguity contiguity, Determinism determinism>
TensorWithNotes
iscan_impl(Tag<torch::kCPU, stype, contiguity, Placement::OutOfPlace, determinism>,
           torch::Tensor input) {
    using T     = typename c10::impl::ScalarTypeToCPPType<stype>::type;
    auto output = torch::empty_like(input);
    scanlib::inclusive_scan_serial<T>(input.data_ptr<T>(),
                                      input.stride(0),
                                      output.data_ptr<T>(),
                                      output.stride(0),
                                      input.size(0));
    return {output, "cpu_serial_out_of_place"};
}

// On cpu, for in place with any stype, contiguity or determinism, we will use the serial option.
template <torch::ScalarType stype, Contiguity contiguity, Determinism determinism>
TensorWithNotes
iscan_impl(Tag<torch::kCPU, stype, contiguity, Placement::InPlace, determinism>,
           torch::Tensor input) {
    using T = typename c10::impl::ScalarTypeToCPPType<stype>::type;
    scanlib::inclusive_scan_serial_inplace<T>(input.data_ptr<T>(), input.stride(0), input.size(0));
    return {input, "cpu_serial_in_place"};
}

// out of place for integer types on cpu will choose parallel option, regardless of determinism.
// This is more constrained than above because of the concept on stype. We will simply say that
// integer types are always constrained to deterministic, which we can fix in the calling code.
template <torch::ScalarType stype, Contiguity contiguity>
    requires IntegerTorchScalarType<stype>
TensorWithNotes
iscan_impl(Tag<torch::kCPU, stype, contiguity, Placement::OutOfPlace, Determinism::Deterministic>,
           torch::Tensor input) {
    using T     = typename c10::impl::ScalarTypeToCPPType<stype>::type;
    auto output = torch::empty_like(input);
    scanlib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                        input.stride(0),
                                        output.data_ptr<T>(),
                                        output.stride(0),
                                        input.size(0));
    return {output, "cpu_parallel_int_deterministic"};
}

// on cpu, for float types and whatever contiguity, with non-deterministic we choose parallel
// option.
template <torch::ScalarType stype, Contiguity contiguity>
    requires FloatTorchScalarType<stype>
TensorWithNotes
iscan_impl(
    Tag<torch::kCPU, stype, contiguity, Placement::OutOfPlace, Determinism::NonDeterministic>,
    torch::Tensor input) {
    using T     = typename c10::impl::ScalarTypeToCPPType<stype>::type;
    auto output = torch::empty_like(input);
    scanlib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                        input.stride(0),
                                        output.data_ptr<T>(),
                                        output.stride(0),
                                        input.size(0));
    return {output, "cpu_parallel_float_nondeterministic"};
}

// on gpu, for contiguous, out-of-place:
// - integer types require deterministic (integers are deterministic on CUDA)
// - float types require non-deterministic (floats are non-deterministic on CUDA)
template <torch::ScalarType stype, Determinism determinism>
    requires CudaScanAllowed<stype, determinism>
TensorWithNotes
iscan_impl(Tag<torch::kCUDA, stype, Contiguity::Contiguous, Placement::OutOfPlace, determinism>,
           torch::Tensor input) {
    using T = typename c10::impl::ScalarTypeToCPPType<stype>::type;

    // Set the current CUDA device to match the input tensor
    c10::cuda::CUDAGuard device_guard(input.device());

    // Get the current CUDA stream for this device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto n = input.size(0);

    auto output = torch::empty_like(input);

    // Query required temp storage size
    auto temp_bytes = scanlib::inclusive_scan_cuda_temp_bytes<T>(n);

    // Allocate temp storage on the same device as input
    auto temp = torch::empty({static_cast<int64_t>(temp_bytes)},
                             torch::dtype(torch::kByte).device(input.device()));

    scanlib::inclusive_scan_cuda<T>(
        input.data_ptr<T>(), output.data_ptr<T>(), n, temp.data_ptr(), temp_bytes, stream);

    return {output, "cuda_cub"};
}

// torch::Tensor inclusiveScanFunctional(torch::Tensor input, Placement placement, Determinism
// determinism) {
//     return inclusiveScan(input, placement, determinism);
// }

// Dispatch space axes - must match the Tag parameters used in iscan_impl
using IscanDeviceAxis      = Values<torch::kCPU, torch::kCUDA>;
using IscanFloatDtypes     = Values<torch::kFloat, torch::kDouble>;
using IscanIntDtypes       = Values<torch::kInt, torch::kLong>;
using IscanDtypeAxis       = Values<torch::kFloat, torch::kDouble, torch::kInt, torch::kLong>;
using IscanContiguityAxis  = Values<Contiguity::Strided, Contiguity::Contiguous>;
using IscanPlacementAxis   = Values<Placement::InPlace, Placement::OutOfPlace>;
using IscanDeterminismAxis = Values<Determinism::Deterministic, Determinism::NonDeterministic>;

// Full dispatch space
using IscanSpace = ValueAxes<IscanDeviceAxis,
                             IscanDtypeAxis,
                             IscanContiguityAxis,
                             IscanPlacementAxis,
                             IscanDeterminismAxis>;

// CPU: floats support both determinism modes, integers only Deterministic (promoted at call site)
using IscanCPUFloatSubspace = ValueAxes<Values<torch::kCPU>,
                                        IscanFloatDtypes,
                                        IscanContiguityAxis,
                                        IscanPlacementAxis,
                                        IscanDeterminismAxis>;

using IscanCPUIntSubspace = ValueAxes<Values<torch::kCPU>,
                                      IscanIntDtypes,
                                      IscanContiguityAxis,
                                      IscanPlacementAxis,
                                      Values<Determinism::Deterministic>>;

// GPU: only Contiguous + OutOfPlace; floats need NonDeterministic, integers need Deterministic
using IscanGPUFloatSubspace = ValueAxes<Values<torch::kCUDA>,
                                        IscanFloatDtypes,
                                        Values<Contiguity::Contiguous>,
                                        Values<Placement::OutOfPlace>,
                                        Values<Determinism::NonDeterministic>>;

using IscanGPUIntSubspace = ValueAxes<Values<torch::kCUDA>,
                                      IscanIntDtypes,
                                      Values<Contiguity::Contiguous>,
                                      Values<Placement::OutOfPlace>,
                                      Values<Determinism::Deterministic>>;

using IscanDispatcher = DispatchTable<IscanSpace, TensorWithNotes(torch::Tensor)>;

TensorWithNotes
inclusiveScanFunctional(torch::Tensor input, Placement placement, Determinism determinism) {
    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == 1, "inclusiveScanFunctional: expected 1D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.size(0) == 0) {
        if (placement == Placement::InPlace) {
            return {input, "empty_in_place"};
        } else {
            return {torch::empty_like(input), "empty_out_of_place"};
        }
    }

    static IscanDispatcher const table{
        IscanDispatcher::from_visitor(
            [](auto coord, torch::Tensor t) { return iscan_impl(coord, t); }),
        IscanCPUFloatSubspace{},
        IscanCPUIntSubspace{},
        IscanGPUFloatSubspace{},
        IscanGPUIntSubspace{}};

    auto const dev        = input.device().type();
    auto const stype      = input.scalar_type();
    auto const contiguity = getContiguity(input);

    // Promote integers to deterministic, since all integer algorithms are deterministic
    if (isIntegerScalarType(stype)) {
        determinism = Determinism::Deterministic;
    }

    try {
        return table(std::make_tuple(dev, stype, contiguity, placement, determinism), input);
    } catch (std::runtime_error const &) {
        TORCH_CHECK_VALUE(false,
                          "inclusiveScanFunctional: unsupported dispatch combination - "
                          "device=",
                          c10::DeviceTypeName(dev),
                          ", dtype=",
                          c10::toString(stype),
                          ", contiguity=",
                          toString(contiguity),
                          ", placement=",
                          toString(placement),
                          ", determinism=",
                          toString(determinism));
    }
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
