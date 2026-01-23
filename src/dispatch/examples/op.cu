// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// op.cu - Dispatch via a Single Struct with if-constexpr
// ================================================================================================
//
// This file demonstrates dispatch across a 5-DIMENSIONAL SPACE using a single struct with one
// heavily-templated `op` function. The dispatch coordinates are the same as functional.cu:
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
//   Define a struct (inclusive_scan_op_t) with:
//     1. A static `op` template that takes tag<device, stype, contiguity, placement, determinism>
//        and uses if-constexpr to branch on the compile-time values.
//     2. Type aliases for stypes, the full space, the supported subspaces, and the dispatcher.
//
//   All algorithm selection logic lives in one function body with compile-time branching:
//     if constexpr (device == kCPU) {
//         if constexpr (placement == in_place) { ... }
//         else if constexpr (torch_integer_stype<stype>) { ... }
//         ...
//     } else { /* CUDA path */ }
//
//   The compiler eliminates dead branches, so each instantiation contains only the relevant code.
//
// DISPATCH TABLE CONSTRUCTION:
//   The table is built using `from_op<inclusive_scan_op_t>()` which automatically finds and
//   instantiates `inclusive_scan_op_t::op` for every point in the declared subspaces.
//
// WHY A SINGLE STRUCT?
//   - All logic is centralized and easy to follow top-to-bottom
//   - Type aliases for space/subspaces live with the implementation
//   - Good when algorithm variants share significant code structure
//   - Easier to see the complete decision tree at a glance
//
// Both op.cu and functional.cu wrap the same underlying library (scan_lib, a stand-in for something
// like CUB or Thrust). They produce identical behavior but demonstrate different code organization
// styles. Choose based on whether your algorithm variants are structurally similar (op) or
// fundamentally different implementations (functional).
//
// ================================================================================================

#include "examples/op.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"
#include "examples/scan_lib.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace dispatch_examples {

using namespace dispatch;

struct inclusive_scan_op_t {
    template <torch::DeviceType device,
              torch::ScalarType stype,
              contiguity cont,
              placement plc,
              determinism det>
    static tensor_with_notes
    op(tag<device, stype, cont, plc, det>, torch::Tensor input) {
        using T      = torch_scalar_cpp_type_t<stype>;
        auto const n = input.size(0);

        if constexpr (device == torch::kCPU) {
            if constexpr (plc == placement::in_place) {
                // CPU in-place: always use serial in-place scan
                scan_lib::inclusive_scan_serial_inplace<T>(
                    input.data_ptr<T>(), input.stride(0), input.size(0));
                return {input, "cpu_serial_in_place"};
            } else {
                // CPU out-of-place: select based on stype and determinism
                if constexpr ((torch_integer_stype<stype>) ||
                              (torch_float_stype<stype> && det == determinism::not_required)) {
                    // Integer types: use parallel scan (integers are always deterministic)
                    auto output = torch::empty_like(input);
                    scan_lib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                                         input.stride(0),
                                                         output.data_ptr<T>(),
                                                         output.stride(0),
                                                         input.size(0));
                    return {output, "cpu_parallel_deterministic"};
                } else {
                    // Fallback: serial out-of-place (floats with deterministic)
                    auto output = torch::empty_like(input);
                    scan_lib::inclusive_scan_serial<T>(input.data_ptr<T>(),
                                                       input.stride(0),
                                                       output.data_ptr<T>(),
                                                       output.stride(0),
                                                       input.size(0));
                    return {output, "cpu_serial_out_of_place"};
                }
            }
        } else {
            c10::cuda::CUDAGuard device_guard(input.device());
            cudaStream_t stream   = at::cuda::getCurrentCUDAStream().stream();
            auto output           = torch::empty_like(input);
            auto const temp_bytes = scan_lib::inclusive_scan_cuda_temp_bytes<T>(n);
            auto temp             = torch::empty({static_cast<int64_t>(temp_bytes)},
                                     torch::dtype(torch::kByte).device(input.device()));
            scan_lib::inclusive_scan_cuda<T>(
                input.data_ptr<T>(), output.data_ptr<T>(), n, temp.data_ptr(), temp_bytes, stream);
            return {output, "cuda_non_deterministic"};
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

    using dispatcher = dispatch_table<space, tensor_with_notes(torch::Tensor)>;
};

tensor_with_notes
inclusive_scan_op(torch::Tensor input, placement plc, determinism det) {
    static inclusive_scan_op_t::dispatcher const table{
        inclusive_scan_op_t::dispatcher::from_op<inclusive_scan_op_t>(),
        inclusive_scan_op_t::cpu_float_subspace{},
        inclusive_scan_op_t::cpu_int_subspace{},
        inclusive_scan_op_t::gpu_float_subspace{},
        inclusive_scan_op_t::gpu_int_subspace{}};

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

    auto const dispatch_coord = std::make_tuple(dev, st, cont, plc, det);
    return torch_dispatch("inclusive_scan_op", table, dispatch_coord, input);
}

} // namespace dispatch_examples
