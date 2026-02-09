// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// op.cu - Dispatch via a Struct with Static Method Overloads
// ================================================================================================
//
// This file demonstrates dispatch across a 5-DIMENSIONAL SPACE using a struct with overloaded
// static `op` methods. The dispatch coordinates are the same as functional.cu:
// Device × ScalarType × Contiguity × Placement × Determinism.
//
// THE FULL SPACE would be: 2 devices × 4 stypes × 2 contiguities × 2 placements × 2 determinisms
//                        = 64 possible instantiations.
//
// SPARSE INSTANTIATION:
//   Like functional.cu, we avoid instantiating all 64 points by declaring specific subspaces:
//     - cpu_float_subspace:  CPU × {float,double} × {any contiguity} × {any placement} × {any
//     determinism}
//     - cpu_int_subspace:    CPU × {int,long} × {any contiguity} × {any placement} × {required
//     only}
//     - gpu_float_subspace:  CUDA × {float,double} × {contiguous} × {out_of_place} × {not_required}
//     - gpu_int_subspace:    CUDA × {int,long} × {contiguous} × {out_of_place} × {required}
//
//   These subspaces are passed to the dispatch_table constructor. Only points in their union are
//   instantiated; all others return a clear runtime error. This is crucial for compile times and
//   binary size when working with high-dimensional dispatch spaces.
//
// THE OP STRUCT IDIOM:
//   Define a struct (iscan_op) with:
//     1. Multiple static `op` overloads, each taking a different tag<...> signature
//     2. Type aliases for stypes, the full space, the supported subspaces, and the dispatcher
//
//   The compiler uses overload resolution to select the most specific match, just like with
//   free functions in functional.cu. Use `requires` clauses to constrain which overloads match.
//
// DISPATCH TABLE CONSTRUCTION:
//   The table is built using `from_op<iscan_op>()` which automatically finds and
//   instantiates `iscan_op::op` for every point in the declared subspaces.
//
// OP vs FUNCTIONAL - HOW THEY FIND THE IMPLEMENTATION:
//   Both examples use the same overload-based dispatch pattern with concepts/requires. The only
//   difference is how the dispatch table locates the implementation:
//
//   - op.cu: Uses `from_op<Op>()` which calls `Op::op(tag<...>, args...)` directly.
//     The implementations are static methods in a struct, and all related type aliases
//     (space, subspaces, dispatcher) live together in that struct.
//
//   - functional.cu: Uses `from_visitor(lambda)` where the lambda forwards to free functions.
//     This enables the "overloaded" idiom for combining multiple lambdas, or simply calling
//     into an overload set of free functions.
//
//   Choose based on organizational preference: struct-based keeps everything together,
//   functional-style allows more separation and the overloaded idiom.
//
// Both op.cu and functional.cu wrap the same underlying library (scan_lib, a stand-in for something
// like CUB or Thrust). They produce identical behavior but demonstrate different code organization
// styles.
//
// ================================================================================================

#include "examples/op.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"
#include "examples/scan_lib.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace dispatch_examples {

using namespace dispatch;

struct iscan_op {
    // we don't check that the input matches the tag, we rely on the dispatch system working
    // correctly.

    // On cpu, for out of place with any stype, contiguity or determinism, we will use the serial
    // option.
    template <typename Tag>
        requires with_value<Tag, torch::kCPU> && with_value<Tag, placement::out_of_place> &&
                 with_type<Tag, torch::ScalarType>
    static tensor_with_notes
    op(Tag, torch::Tensor input) {
        using T     = torch_scalar_cpp_type<Tag>;
        auto output = torch::empty_like(input);
        scan_lib::inclusive_scan_serial<T>(input.data_ptr<T>(),
                                           input.stride(0),
                                           output.data_ptr<T>(),
                                           output.stride(0),
                                           input.size(0));
        return {output, "cpu_serial_out_of_place"};
    }

    // On cpu, for in place with any stype, contiguity or determinism, we will use the serial
    // option.
    template <typename Tag>
        requires with_value<Tag, torch::kCPU> && with_value<Tag, placement::in_place> &&
                 with_type<Tag, torch::ScalarType>
    static tensor_with_notes
    op(Tag, torch::Tensor input) {
        using T = torch_scalar_cpp_type<Tag>;
        scan_lib::inclusive_scan_serial_inplace<T>(
            input.data_ptr<T>(), input.stride(0), input.size(0));
        return {input, "cpu_serial_in_place"};
    }

    // out of place for integer types on cpu will choose parallel option.
    // This is more constrained than above: integer stype + out_of_place + deterministic.
    template <typename Tag>
        requires with_value<Tag, torch::kCPU> && with_value<Tag, placement::out_of_place> &&
                 with_value<Tag, determinism::required> && with_type<Tag, torch::ScalarType> &&
                 torch_integer_stype<tag_get<torch::ScalarType, Tag>()>
    static tensor_with_notes
    op(Tag, torch::Tensor input) {
        using T     = torch_scalar_cpp_type<Tag>;
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
    template <typename Tag>
        requires with_value<Tag, torch::kCPU> && with_value<Tag, placement::out_of_place> &&
                 with_value<Tag, determinism::not_required> && with_type<Tag, torch::ScalarType> &&
                 torch_float_stype<tag_get<torch::ScalarType, Tag>()>
    static tensor_with_notes
    op(Tag, torch::Tensor input) {
        using T     = torch_scalar_cpp_type<Tag>;
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
    template <typename Tag>
        requires with_value<Tag, torch::kCUDA> && with_value<Tag, contiguity::contiguous> &&
                 with_value<Tag, placement::out_of_place> && cuda_scan_allowed_tag<Tag>
    static tensor_with_notes
    op(Tag, torch::Tensor input) {
        using T = torch_scalar_cpp_type<Tag>;

        c10::cuda::CUDAGuard device_guard(input.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        auto const n        = input.size(0);
        auto output         = torch::empty_like(input);
        auto temp_bytes     = scan_lib::inclusive_scan_cuda_temp_bytes<T>(n);
        auto temp           = torch::empty({static_cast<int64_t>(temp_bytes)},
                                 torch::dtype(torch::kByte).device(input.device()));

        scan_lib::inclusive_scan_cuda<T>(
            input.data_ptr<T>(), output.data_ptr<T>(), n, temp.data_ptr(), temp_bytes, stream);

        constexpr auto stype = tag_get<torch::ScalarType>(Tag{});
        if constexpr (torch_is_integer_stype_v<stype>) {
            return {output, "cuda_int_deterministic"};
        } else {
            return {output, "cuda_float_nondeterministic"};
        }
    }

    // Dispatch space axes - must match the tag parameters used in op function
    using int_stypes = axis<torch::kInt, torch::kLong>;
    using stype_axis = axis<torch::kFloat, torch::kDouble, torch::kInt, torch::kLong>;

    // Full dispatch space
    using space = axes<torch_cpu_cuda_device_axis,
                       stype_axis,
                       full_contiguity_axis,
                       full_placement_axis,
                       full_determinism_axis>;

    // CPU: floats support both determinism modes, integers only required (promoted at call
    // site)
    using cpu_float_subspace = axes<axis<torch::kCPU>,
                                    torch_builtin_float_stype_axis,
                                    full_contiguity_axis,
                                    full_placement_axis,
                                    full_determinism_axis>;

    using cpu_int_subspace = axes<axis<torch::kCPU>,
                                  int_stypes,
                                  full_contiguity_axis,
                                  full_placement_axis,
                                  axis<determinism::required>>;

    // GPU: only contiguous + out_of_place; floats need not_required, integers need required
    using gpu_float_subspace = axes<axis<torch::kCUDA>,
                                    torch_builtin_float_stype_axis,
                                    axis<contiguity::contiguous>,
                                    axis<placement::out_of_place>,
                                    axis<determinism::not_required>>;

    using gpu_int_subspace = axes<axis<torch::kCUDA>,
                                  int_stypes,
                                  axis<contiguity::contiguous>,
                                  axis<placement::out_of_place>,
                                  axis<determinism::required>>;

    using subspaces =
        coverage<cpu_float_subspace, cpu_int_subspace, gpu_float_subspace, gpu_int_subspace>;

    using dispatcher = dispatch_table<space, tensor_with_notes(torch::Tensor)>;
};

tensor_with_notes
inclusive_scan_op(torch::Tensor input, placement plc, determinism det) {
    static auto const table = dispatch_table_from_op<iscan_op>("inclusive_scan_op");

    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == 1, "inclusive_scan_op: expected 1D tensor, got ", input.dim(), "D");

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

    auto const fn = table.select(dispatch_set{dev, st, cont, plc, det});
    return fn(input);
}

} // namespace dispatch_examples
