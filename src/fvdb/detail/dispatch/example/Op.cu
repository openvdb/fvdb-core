// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "fvdb/detail/dispatch/example/Op.h"

#include "fvdb/detail/dispatch/SparseDispatchTable.h"
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
