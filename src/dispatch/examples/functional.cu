// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// functional.cu - Dispatch via Overloaded Free Functions
// ================================================================================================
//
// This file demonstrates dispatch across a 5-DIMENSIONAL SPACE using overloaded free functions.
// The dispatch coordinates are: Device × ScalarType × Contiguity × Placement × Determinism.
//
// THE FULL SPACE would be: 2 devices × 4 stypes × 2 contiguities × 2 placements × 2 determinisms
//                        = 64 possible instantiations. However, not all combinations are valid:
//   - GPU only supports contiguous, out-of-place operations
//   - Integer types are inherently deterministic
//   - Some algorithms only work with specific constraints
//
// SPARSE INSTANTIATION:
//   Instead of instantiating all 64 points, we define SUBSPACES that cover only valid combinations:
//     - iscan_cpu_float_subspace:  CPU × {float,double} × {any contiguity} × {any placement} × {any
//     determinism}
//     - iscan_cpu_int_subspace:    CPU × {int,long} × {any contiguity} × {any placement} ×
//     {required only}
//     - iscan_gpu_float_subspace:  CUDA × {float,double} × {contiguous} × {out_of_place} ×
//     {not_required}
//     - iscan_gpu_int_subspace:    CUDA × {int,long} × {contiguous} × {out_of_place} × {required}
//
//   The union of these subspaces covers exactly the supported configurations. Points outside these
//   subspaces will produce a runtime error with a clear message about unsupported coordinates.
//
// THE FUNCTIONAL IDIOM:
//   Define multiple `iscan_impl` free function overloads, each taking a different tag<...> type.
//   The compiler uses overload resolution to select the most specific match:
//
//   1. Generic fallbacks use broad template parameters:
//        template<stype, contiguity, determinism> iscan_impl(tag<kCPU, stype, contiguity, ...>,
//        ...)
//
//   2. Specialized versions use `requires` clauses to constrain selection:
//        template<stype, contiguity> requires torch_integer_stype<stype>
//        iscan_impl(tag<kCPU, stype, contiguity, out_of_place, required>, ...)
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
//   - Enables the "overloaded" idiom for combining multiple lambdas
//
// FUNCTIONAL vs OP - HOW THEY FIND THE IMPLEMENTATION:
//   Both functional.cu and op.cu use the same overload-based dispatch pattern. The only difference
//   is how the dispatch table locates the implementation:
//
//   - functional.cu: Uses `from_visitor(lambda)` where the lambda forwards to free functions.
//   - op.cu: Uses `from_op<Op>()` which calls `Op::op(tag<...>, args...)` directly.
//
//   Choose based on organizational preference: free functions allow more separation and the
//   overloaded idiom, struct-based keeps implementations and type aliases together.
//
// ================================================================================================

#include "examples/functional.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"
#include "examples/scan_lib.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace dispatch_examples {

using namespace dispatch;

// we don't check that the input matches the tag, we rely on the dispatch system working correctly.

// On cpu, for out of place with any stype, contiguity or determinism, we will use the serial
// option.
template <torch::ScalarType stype, contiguity cont, determinism det>
tensor_with_notes
iscan_impl(tag<torch::kCPU, stype, cont, placement::out_of_place, det>, torch::Tensor input) {
    using T     = torch_scalar_cpp_type_t<stype>;
    auto output = torch::empty_like(input);
    scan_lib::inclusive_scan_serial<T>(input.data_ptr<T>(),
                                       input.stride(0),
                                       output.data_ptr<T>(),
                                       output.stride(0),
                                       input.size(0));
    return {output, "cpu_serial_out_of_place"};
}

// On cpu, for in place with any stype, contiguity or determinism, we will use the serial option.
template <torch::ScalarType stype, contiguity cont, determinism det>
tensor_with_notes
iscan_impl(tag<torch::kCPU, stype, cont, placement::in_place, det>, torch::Tensor input) {
    using T = torch_scalar_cpp_type_t<stype>;
    scan_lib::inclusive_scan_serial_inplace<T>(input.data_ptr<T>(), input.stride(0), input.size(0));
    return {input, "cpu_serial_in_place"};
}

// out of place for integer types on cpu will choose parallel option, regardless of determinism.
// This is more constrained than above because of the concept on stype. We will simply say that
// integer types are always constrained to deterministic, which we can fix in the calling code.
template <torch::ScalarType stype, contiguity cont>
    requires torch_integer_stype<stype>
tensor_with_notes
iscan_impl(tag<torch::kCPU, stype, cont, placement::out_of_place, determinism::required>,
           torch::Tensor input) {
    using T     = torch_scalar_cpp_type_t<stype>;
    auto output = torch::empty_like(input);
    scan_lib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                         input.stride(0),
                                         output.data_ptr<T>(),
                                         output.stride(0),
                                         input.size(0));
    return {output, "cpu_parallel_int_deterministic"};
}

// on cpu, for float types and whatever contiguity, with non-deterministic we choose parallel
// option.
template <torch::ScalarType stype, contiguity cont>
    requires torch_float_stype<stype>
tensor_with_notes
iscan_impl(tag<torch::kCPU, stype, cont, placement::out_of_place, determinism::not_required>,
           torch::Tensor input) {
    using T     = torch_scalar_cpp_type_t<stype>;
    auto output = torch::empty_like(input);
    scan_lib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                         input.stride(0),
                                         output.data_ptr<T>(),
                                         output.stride(0),
                                         input.size(0));
    return {output, "cpu_parallel_float_nondeterministic"};
}

// on gpu, for contiguous, out-of-place:
// - integer types require deterministic (integers are deterministic on CUDA)
// - float types require non-deterministic (floats are non-deterministic on CUDA)
template <torch::ScalarType stype, determinism det>
    requires cuda_scan_allowed<stype, det>
tensor_with_notes
iscan_impl(tag<torch::kCUDA, stype, contiguity::contiguous, placement::out_of_place, det>,
           torch::Tensor input) {
    using T = torch_scalar_cpp_type_t<stype>;

    // Set the current CUDA device to match the input tensor
    c10::cuda::CUDAGuard device_guard(input.device());

    // Get the current CUDA stream for this device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto n = input.size(0);

    auto output = torch::empty_like(input);

    // Query required temp storage size
    auto temp_bytes = scan_lib::inclusive_scan_cuda_temp_bytes<T>(n);

    // Allocate temp storage on the same device as input
    auto temp = torch::empty({static_cast<int64_t>(temp_bytes)},
                             torch::dtype(torch::kByte).device(input.device()));

    scan_lib::inclusive_scan_cuda<T>(
        input.data_ptr<T>(), output.data_ptr<T>(), n, temp.data_ptr(), temp_bytes, stream);

    if constexpr (torch_is_integer_stype_v<stype>) {
        return {output, "cuda_int_deterministic"};
    } else {
        return {output, "cuda_float_nondeterministic"};
    }
}

// Dispatch space axes - must match the tag parameters used in iscan_impl
using iscan_int_stypes = axis<torch::kInt, torch::kLong>;
using iscan_stype_axis = axis<torch::kFloat, torch::kDouble, torch::kInt, torch::kLong>;

// Full dispatch space
using iscan_space = axes<torch_cpu_cuda_device_axis,
                         iscan_stype_axis,
                         full_contiguity_axis,
                         full_placement_axis,
                         full_determinism_axis>;

// CPU: floats support both determinism modes, integers only required (promoted at call site)
using iscan_cpu_float_subspace = axes<axis<torch::kCPU>,
                                      torch_builtin_float_stype_axis,
                                      full_contiguity_axis,
                                      full_placement_axis,
                                      full_determinism_axis>;

using iscan_cpu_int_subspace = axes<axis<torch::kCPU>,
                                    iscan_int_stypes,
                                    full_contiguity_axis,
                                    full_placement_axis,
                                    axis<determinism::required>>;

// GPU: only contiguous + out_of_place; floats need not_required, integers need required
using iscan_gpu_float_subspace = axes<axis<torch::kCUDA>,
                                      torch_builtin_float_stype_axis,
                                      axis<contiguity::contiguous>,
                                      axis<placement::out_of_place>,
                                      axis<determinism::not_required>>;

using iscan_gpu_int_subspace = axes<axis<torch::kCUDA>,
                                    iscan_int_stypes,
                                    axis<contiguity::contiguous>,
                                    axis<placement::out_of_place>,
                                    axis<determinism::required>>;

using iscan_dispatcher = dispatch_table<iscan_space, tensor_with_notes(torch::Tensor)>;

tensor_with_notes
inclusive_scan_functional(torch::Tensor input, placement plc, determinism det) {
    static iscan_dispatcher const table{
        iscan_dispatcher::from_visitor(
            [](auto coord, torch::Tensor t) { return iscan_impl(coord, t); }),
        iscan_cpu_float_subspace{},
        iscan_cpu_int_subspace{},
        iscan_gpu_float_subspace{},
        iscan_gpu_int_subspace{}};

    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == 1, "inclusive_scan_functional: expected 1D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.size(0) == 0) {
        if (plc == placement::in_place) {
            return {input, "empty_in_place"};
        } else {
            return {torch::empty_like(input), "empty_out_of_place"};
        }
    }

    auto const dev  = input.device().type();
    auto const st   = input.scalar_type();
    auto const cont = torch_get_contiguity(input);

    // Promote integers to deterministic, since all integer algorithms are deterministic
    if (is_torch_integer_stype(st)) {
        det = determinism::required;
    }

    auto const dispatch_coord = std::make_tuple(dev, st, cont, plc, det);
    return torch_dispatch("inclusive_scan_functional", table, dispatch_coord, input);
}

} // namespace dispatch_examples
