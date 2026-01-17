// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// ReLU.cu - A Simple Dispatch Example
// ================================================================================================
//
// This file demonstrates the simplest use of the dispatch system: dispatching across device types
// (CPU/CUDA) and floating-point scalar types (float16, bfloat16, float, double).
//
// THE DISPATCH PATTERN:
//   1. Define a struct (ReluOp) containing static `op` member function templates.
//   2. Each `op` overload takes a Tag<...> as its first argument, which encodes compile-time
//      information about the dispatch coordinate (device type, scalar type, etc.).
//   3. Declare a `Space` type alias that defines all possible dispatch coordinates as a Cartesian
//      product of value axes. Here: CpuCudaDeviceAxis × FullFloatDtypeAxis = 2 × 4 = 8 points.
//   4. Declare a `Dispatcher` type alias that combines the Space with the function signature.
//   5. At the call site, create a static dispatch table using `from_op<ReluOp>()` which
//      automatically instantiates `ReluOp::op` for every point in the Space.
//   6. Call `torchDispatch(name, table, coord, args...)` which looks up the runtime coordinate
//      in the table and invokes the corresponding compile-time instantiation.
//
// HOW TAG DISPATCH WORKS:
//   The Tag<torch::kCPU, stype> and Tag<torch::kCUDA, stype> overloads are distinguished by the
//   device type in the tag. The dispatcher calls the appropriate overload based on runtime device.
//   Within each overload, `stype` is a compile-time constant, allowing type-safe accessor usage.
//
// KEY TYPES:
//   - Tag<DeviceType, ScalarType, ...>: A compile-time coordinate in the dispatch space.
//   - ValueAxes<Axis1, Axis2, ...>: Cartesian product of value sets defining the full space.
//   - DispatchTable<Space, Signature>: Maps runtime coordinates to function pointers.
//   - torchDispatch(): Validates the coordinate and calls the appropriate instantiation.
//
// This example is intentionally minimal. For more complex dispatch with multiple axes, partial
// coverage, and algorithm selection based on constraints, see Functional.cu and Op.cu.
//
// ================================================================================================

#include "fvdb/detail/dispatch/example/ReLU.h"

#include "fvdb/detail/dispatch/SparseDispatchTable.h"
#include "fvdb/detail/dispatch/TorchDispatch.h"
#include "fvdb/detail/dispatch/TypesFwd.h"
#include "fvdb/detail/dispatch/ValueSpace.h"
#include "fvdb/detail/dispatch/Values.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace dispatch {
namespace example {

template <typename T>
__global__ void
relu_kernel(torch::PackedTensorAccessor64<T, 1, torch::RestrictPtrTraits> input,
            torch::PackedTensorAccessor64<T, 1, torch::RestrictPtrTraits> output) {
    int64_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input.size(0)) {
        T const val = input[idx];
        output[idx] = val > T(0) ? val : T(0);
    }
}

struct ReluOp {
    template <torch::ScalarType stype>
    static void
    op(Tag<torch::kCPU, stype>, torch::Tensor input, torch::Tensor output) {
        using T = typename c10::impl::ScalarTypeToCPPType<stype>::type;

        auto input_accessor  = input.accessor<T, 1>();
        auto output_accessor = output.accessor<T, 1>();

        for (int64_t i = 0; i < input.numel(); ++i) {
            T const val        = input_accessor[i];
            output_accessor[i] = val > T(0) ? val : T(0);
        }
    }

    template <torch::ScalarType stype>
    static void
    op(Tag<torch::kCUDA, stype>, torch::Tensor input, torch::Tensor output) {
        using T = typename c10::impl::ScalarTypeToCPPType<stype>::type;

        c10::cuda::CUDAGuard device_guard(input.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        auto input_accessor  = input.packed_accessor64<T, 1, torch::RestrictPtrTraits>();
        auto output_accessor = output.packed_accessor64<T, 1, torch::RestrictPtrTraits>();

        int constexpr block_size = 256;
        int const num_blocks     = (input.numel() + block_size - 1) / block_size;
        relu_kernel<T><<<num_blocks, block_size, 0, stream>>>(input_accessor, output_accessor);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    using Space      = ValueAxes<CpuCudaDeviceAxis, FullFloatDtypeAxis>;
    using Dispatcher = DispatchTable<Space, void(torch::Tensor, torch::Tensor)>;
};

torch::Tensor
relu(torch::Tensor input, Placement placement) {
    static ReluOp::Dispatcher const table{ReluOp::Dispatcher::from_op<ReluOp>(), ReluOp::Space{}};

    // Validate input rank
    TORCH_CHECK_VALUE(input.dim() == 1, "relu: expected 1D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.size(0) == 0) {
        if (placement == Placement::InPlace) {
            return input;
        } else {
            return torch::empty_like(input);
        }
    }

    auto output = (placement == Placement::InPlace) ? input : torch::empty_like(input);

    auto const dev   = input.device().type();
    auto const stype = input.scalar_type();

    auto const dispatch_coord = std::make_tuple(dev, stype);
    torchDispatch("relu", table, dispatch_coord, input, output);

    return output;
}

} // namespace example
} // namespace dispatch
} // namespace fvdb
