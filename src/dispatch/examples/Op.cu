// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// Op.cu - Dispatch via a Single Struct with if-constexpr
// ================================================================================================
//
// This file demonstrates dispatch across a 5-DIMENSIONAL SPACE using a single struct with one
// heavily-templated `op` function. The dispatch coordinates are the same as Functional.cu:
// Device × Dtype × Contiguity × Placement × Determinism.
//
// THE FULL SPACE would be: 2 devices × 4 dtypes × 2 contiguities × 2 placements × 2 determinisms
//                        = 64 possible instantiations.
//
// SPARSE INSTANTIATION:
//   Like Functional.cu, we avoid instantiating all 64 points by declaring specific subspaces:
//     - CPUFloatSubspace:  CPU × {float,double} × {any contiguity} × {any placement} × {any
//     determinism}
//     - CPUIntSubspace:    CPU × {int,long} × {any contiguity} × {any placement} × {Deterministic
//     only}
//     - GPUFloatSubspace:  CUDA × {float,double} × {Contiguous} × {OutOfPlace} × {NonDeterministic}
//     - GPUIntSubspace:    CUDA × {int,long} × {Contiguous} × {OutOfPlace} × {Deterministic}
//
//   These subspaces are passed to the DispatchTable constructor. Only points in their union are
//   instantiated; all others return a clear runtime error. This is crucial for compile times and
//   binary size when working with high-dimensional dispatch spaces.
//
// THE OP STRUCT IDIOM:
//   Define a struct (InclusiveScanOp) with:
//     1. A static `op` template that takes Tag<device, stype, contiguity, placement, determinism>
//        and uses if-constexpr to branch on the compile-time values.
//     2. Type aliases for dtypes, the full Space, the supported Subspaces, and the Dispatcher.
//
//   All algorithm selection logic lives in one function body with compile-time branching:
//     if constexpr (device == kCPU) {
//         if constexpr (placement == InPlace) { ... }
//         else if constexpr (IntegerTorchScalarType<stype>) { ... }
//         ...
//     } else { /* CUDA path */ }
//
//   The compiler eliminates dead branches, so each instantiation contains only the relevant code.
//
// DISPATCH TABLE CONSTRUCTION:
//   The table is built using `from_op<InclusiveScanOp>()` which automatically finds and
//   instantiates `InclusiveScanOp::op` for every point in the declared subspaces.
//
// WHY A SINGLE STRUCT?
//   - All logic is centralized and easy to follow top-to-bottom
//   - Type aliases for Space/Subspaces live with the implementation
//   - Good when algorithm variants share significant code structure
//   - Easier to see the complete decision tree at a glance
//
// Both Op.cu and Functional.cu wrap the same underlying library (ScanLib, a stand-in for something
// like CUB or Thrust). They produce identical behavior but demonstrate different code organization
// styles. Choose based on whether your algorithm variants are structurally similar (Op) or
// fundamentally different implementations (Functional).
//
// ================================================================================================

#include "fvdb/detail/dispatch/example/Op.h"

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

struct InclusiveScanOp {
    template <torch::DeviceType device,
              torch::ScalarType stype,
              Contiguity contiguity,
              Placement placement,
              Determinism determinism>
    static TensorWithNotes
    op(Tag<device, stype, contiguity, placement, determinism>, torch::Tensor input) {
        using T      = typename c10::impl::ScalarTypeToCPPType<stype>::type;
        auto const n = input.size(0);

        if constexpr (device == torch::kCPU) {
            if constexpr (placement == Placement::InPlace) {
                // CPU in-place: always use serial in-place scan
                scanlib::inclusive_scan_serial_inplace<T>(
                    input.data_ptr<T>(), input.stride(0), input.size(0));
                return {input, "cpu_serial_in_place"};
            } else {
                // CPU out-of-place: select based on dtype and determinism
                if constexpr ((IntegerTorchScalarType<stype>) ||
                              (FloatTorchScalarType<stype> &&
                               determinism == Determinism::NonDeterministic)) {
                    // Integer types: use parallel scan (integers are always deterministic)
                    auto output = torch::empty_like(input);
                    scanlib::inclusive_scan_parallel<T>(input.data_ptr<T>(),
                                                        input.stride(0),
                                                        output.data_ptr<T>(),
                                                        output.stride(0),
                                                        input.size(0));
                    return {output, "cpu_parallel_deterministic"};
                } else {
                    // Fallback: serial out-of-place (floats with deterministic)
                    auto output = torch::empty_like(input);
                    scanlib::inclusive_scan_serial<T>(input.data_ptr<T>(),
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
            auto const temp_bytes = scanlib::inclusive_scan_cuda_temp_bytes<T>(n);
            auto temp             = torch::empty({static_cast<int64_t>(temp_bytes)},
                                     torch::dtype(torch::kByte).device(input.device()));
            scanlib::inclusive_scan_cuda<T>(
                input.data_ptr<T>(), output.data_ptr<T>(), n, temp.data_ptr(), temp_bytes, stream);
            return {output, "cuda_non_deterministic"};
        }
    }

    // Dispatch space axes - must match the Tag parameters used in op function
    using IntDtypes = Values<torch::kInt, torch::kLong>;
    using DtypeAxis = Values<torch::kFloat, torch::kDouble, torch::kInt, torch::kLong>;

    // Full dispatch space
    using Space = ValueAxes<CpuCudaDeviceAxis,
                            DtypeAxis,
                            FullContiguityAxis,
                            FullPlacementAxis,
                            FullDeterminismAxis>;

    // CPU: floats support both determinism modes, integers only Deterministic (promoted at call
    // site)
    using CPUFloatSubspace = ValueAxes<Values<torch::kCPU>,
                                       BuiltinFloatDtypeAxis,
                                       FullContiguityAxis,
                                       FullPlacementAxis,
                                       FullDeterminismAxis>;

    using CPUIntSubspace = ValueAxes<Values<torch::kCPU>,
                                     IntDtypes,
                                     FullContiguityAxis,
                                     FullPlacementAxis,
                                     Values<Determinism::Deterministic>>;

    // GPU: only Contiguous + OutOfPlace; floats need NonDeterministic, integers need Deterministic
    using GPUFloatSubspace = ValueAxes<Values<torch::kCUDA>,
                                       BuiltinFloatDtypeAxis,
                                       Values<Contiguity::Contiguous>,
                                       Values<Placement::OutOfPlace>,
                                       Values<Determinism::NonDeterministic>>;

    using GPUIntSubspace = ValueAxes<Values<torch::kCUDA>,
                                     IntDtypes,
                                     Values<Contiguity::Contiguous>,
                                     Values<Placement::OutOfPlace>,
                                     Values<Determinism::Deterministic>>;

    using Dispatcher = DispatchTable<Space, TensorWithNotes(torch::Tensor)>;
};

TensorWithNotes
inclusiveScanOp(torch::Tensor input, Placement placement, Determinism determinism) {
    static InclusiveScanOp::Dispatcher const table{
        InclusiveScanOp::Dispatcher::from_op<InclusiveScanOp>(),
        InclusiveScanOp::CPUFloatSubspace{},
        InclusiveScanOp::CPUIntSubspace{},
        InclusiveScanOp::GPUFloatSubspace{},
        InclusiveScanOp::GPUIntSubspace{}};

    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == 1, "inclusiveScanOp: expected 1D tensor, got ", input.dim(), "D");

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
    return torchDispatch("inclusiveScanOp", table, dispatch_coord, input);
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
