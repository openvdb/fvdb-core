// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// Functional.cu - Dispatch via Overloaded Free Functions
// ================================================================================================
//
// This file demonstrates dispatch across a 5-DIMENSIONAL SPACE using overloaded free functions.
// The dispatch coordinates are: Device × Dtype × Contiguity × Placement × Determinism.
//
// THE FULL SPACE would be: 2 devices × 4 dtypes × 2 contiguities × 2 placements × 2 determinisms
//                        = 64 possible instantiations. However, not all combinations are valid:
//   - GPU only supports contiguous, out-of-place operations
//   - Integer types are inherently deterministic
//   - Some algorithms only work with specific constraints
//
// SPARSE INSTANTIATION:
//   Instead of instantiating all 64 points, we define SUBSPACES that cover only valid combinations:
//     - IscanCPUFloatSubspace:  CPU × {float,double} × {any contiguity} × {any placement} × {any
//     determinism}
//     - IscanCPUIntSubspace:    CPU × {int,long} × {any contiguity} × {any placement} ×
//     {Deterministic only}
//     - IscanGPUFloatSubspace:  CUDA × {float,double} × {Contiguous} × {OutOfPlace} ×
//     {NonDeterministic}
//     - IscanGPUIntSubspace:    CUDA × {int,long} × {Contiguous} × {OutOfPlace} × {Deterministic}
//
//   The union of these subspaces covers exactly the supported configurations. Points outside these
//   subspaces will produce a runtime error with a clear message about unsupported coordinates.
//
// THE FUNCTIONAL IDIOM:
//   Define multiple `iscan_impl` free function overloads, each taking a different Tag<...> type.
//   The compiler uses overload resolution to select the most specific match:
//
//   1. Generic fallbacks use broad template parameters:
//        template<stype, contiguity, determinism> iscan_impl(Tag<kCPU, stype, contiguity, ...>,
//        ...)
//
//   2. Specialized versions use `requires` clauses to constrain selection:
//        template<stype, contiguity> requires IntegerTorchScalarType<stype>
//        iscan_impl(Tag<kCPU, stype, contiguity, OutOfPlace, Deterministic>, ...)
//
//   When multiple overloads match, the most constrained one wins. This allows natural expression of
//   "integers use this algorithm" or "non-deterministic floats use that algorithm" without nested
//   if-constexpr chains.
//
// DISPATCH TABLE CONSTRUCTION:
//   The table is built using `from_visitor()` with a lambda that forwards to the overload set:
//     from_visitor([](auto coord, torch::Tensor t) { return iscan_impl(coord, t); })
//   The dispatcher instantiates this lambda for every point in the declared subspaces, and the
//   lambda's call to iscan_impl uses overload resolution to select the right implementation.
//
// WHY FREE FUNCTIONS?
//   - Natural for functional-style code that calls into external libraries
//   - Easy to add new overloads without modifying existing code
//   - Concepts and requires clauses provide fine-grained control over selection
//   - Good when different algorithm variants have very different implementations
//
// Compare with Op.cu which achieves the same result using if-constexpr inside a single struct.
//
// ================================================================================================

#include "fvdb/detail/dispatch/example/Functional.h"

#include "fvdb/detail/dispatch/DispatchTable.h"
#include "fvdb/detail/dispatch/TorchDispatch.h"
#include "fvdb/detail/dispatch/TypesFwd.h"
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

// Dispatch space axes - must match the Tag parameters used in iscan_impl
using IscanIntDtypes = Values<torch::kInt, torch::kLong>;
using IscanDtypeAxis = Values<torch::kFloat, torch::kDouble, torch::kInt, torch::kLong>;

// Full dispatch space
using IscanSpace = ValueAxes<CpuCudaDeviceAxis,
                             IscanDtypeAxis,
                             FullContiguityAxis,
                             FullPlacementAxis,
                             FullDeterminismAxis>;

// CPU: floats support both determinism modes, integers only Deterministic (promoted at call site)
using IscanCPUFloatSubspace = ValueAxes<Values<torch::kCPU>,
                                        BuiltinFloatDtypeAxis,
                                        FullContiguityAxis,
                                        FullPlacementAxis,
                                        FullDeterminismAxis>;

using IscanCPUIntSubspace = ValueAxes<Values<torch::kCPU>,
                                      IscanIntDtypes,
                                      FullContiguityAxis,
                                      FullPlacementAxis,
                                      Values<Determinism::Deterministic>>;

// GPU: only Contiguous + OutOfPlace; floats need NonDeterministic, integers need Deterministic
using IscanGPUFloatSubspace = ValueAxes<Values<torch::kCUDA>,
                                        BuiltinFloatDtypeAxis,
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
    static IscanDispatcher const table{
        IscanDispatcher::from_visitor(
            [](auto coord, torch::Tensor t) { return iscan_impl(coord, t); }),
        IscanCPUFloatSubspace{},
        IscanCPUIntSubspace{},
        IscanGPUFloatSubspace{},
        IscanGPUIntSubspace{}};

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

    auto const dev        = input.device().type();
    auto const stype      = input.scalar_type();
    auto const contiguity = getContiguity(input);

    // Promote integers to deterministic, since all integer algorithms are deterministic
    if (isIntegerScalarType(stype)) {
        determinism = Determinism::Deterministic;
    }

    auto const dispatch_coord = std::make_tuple(dev, stype, contiguity, placement, determinism);
    return torchDispatch("inclusiveScanFunctional", table, dispatch_coord, input);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
