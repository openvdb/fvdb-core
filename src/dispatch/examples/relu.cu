// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// relu.cu - A Simple Dispatch Example
// ================================================================================================
//
// This file demonstrates the simplest use of the dispatch system: dispatching across device types
// (CPU/CUDA) and floating-point scalar types (float16, bfloat16, float, double).
//
// THE DISPATCH PATTERN:
//   1. Define a struct (relu_op_t) containing static `op` member function templates.
//   2. Each `op` overload takes a tag<...> as its first argument, which encodes compile-time
//      information about the dispatch coordinate (device type, scalar type, etc.).
//   3. Declare a `space` type alias that defines all possible dispatch coordinates as a Cartesian
//      product of value axes. Here: torch_cpu_cuda_device_axis × torch_full_float_stype_axis = 2 ×
//      4 = 8 points.
//   4. Declare a `dispatcher` type alias that combines the space with the function signature.
//   5. At the call site, create a static dispatch table using `from_op<relu_op_t>()` which
//      automatically instantiates `relu_op_t::op` for every point in the space.
//   6. Call `torch_dispatch(name, table, coord, args...)` which looks up the runtime coordinate
//      in the table and invokes the corresponding compile-time instantiation.
//
// HOW TAG DISPATCH WORKS:
//   The tag<torch::kCPU, stype> and tag<torch::kCUDA, stype> overloads are distinguished by the
//   device type in the tag. The dispatcher calls the appropriate overload based on runtime device.
//   Within each overload, `stype` is a compile-time constant, allowing type-safe accessor usage.
//
// KEY TYPES:
//   - tag<DeviceType, ScalarType, ...>: A compile-time coordinate in the dispatch space.
//   - axes<axis1, axis2, ...>: Cartesian product of value sets defining the full space.
//   - dispatch_table<space, signature>: Maps runtime coordinates to function pointers.
//   - torch_dispatch(): Validates the coordinate and calls the appropriate instantiation.
//
// This example is intentionally minimal. For more complex dispatch with multiple axes, partial
// coverage, and algorithm selection based on constraints, see functional.cu and op.cu.
//
// ================================================================================================
//
// PORTABLE SCALAR OPERATIONS FOR HALF-PRECISION TYPES
// ====================================================
//
// When writing CUDA kernels that must work across float, double, at::Half, and at::BFloat16,
// avoid using:
//
//   - max(x, T(0)) or std::max(x, T(0)):
//     Unqualified max() may not have overloads for at::Half/at::BFloat16, or may resolve to
//     an incorrect overload. std::max requires operator< which these types may not define
//     in device code.
//
//   - Ternary expressions like (x > T(0)) ? x : T(0):
//     at::Half and at::BFloat16 may lack comparison operators (operator>, operator<, etc.)
//     in CUDA device code, causing compilation failures or SFINAE-related instantiation errors.
//
// SOLUTION: Use relu_scalar(x) defined below, with concept-constrained overloads:
//   - half_like (at::Half/at::BFloat16): casts to float for comparison (always supported).
//   - builtin_float_like (float/double): native comparison preserves full precision.
//
// This pattern generalizes: scalar comparisons on half-precision types should cast to float
// for the comparison while preserving the original type for the result.
//
// ================================================================================================

#include "examples/relu.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <type_traits>

namespace dispatch_examples {

using namespace dispatch;

// ------------------------------------------------------------------------------------------------
// relu_scalar: Portable ReLU for all types in torch_full_float_stype_axis.
// See "PORTABLE SCALAR OPERATIONS" above for rationale.
// ------------------------------------------------------------------------------------------------

template <typename T>
concept half_like = std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>;

template <typename T>
concept builtin_float_like = std::is_same_v<T, float> || std::is_same_v<T, double>;

__host__ __device__ inline auto
relu_scalar(half_like auto x) -> decltype(x) {
    // Compare in float; half types may lack comparison operators in device code.
    if (static_cast<float>(x) > 0.0f) {
        return x;
    }
    return {};
}

__host__ __device__ inline auto
relu_scalar(builtin_float_like auto x) -> decltype(x) {
    if (x > 0) {
        return x;
    }
    return {};
}

template <typename T>
__global__ void
relu_kernel(torch::PackedTensorAccessor64<T, 1> input, torch::PackedTensorAccessor64<T, 1> output) {
    int64_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input.size(0)) {
        output[idx] = relu_scalar(input[idx]);
    }
}

struct relu_op_t {
    template <torch::ScalarType stype>
    static void
    op(tag<torch::kCPU, stype>, torch::Tensor input, torch::Tensor output) {
        using T = torch_scalar_cpp_type_t<stype>;

        auto input_accessor  = input.accessor<T, 1>();
        auto output_accessor = output.accessor<T, 1>();

        // A "real" implementation would parallelize this loop on the host, but
        // we're keeping it simple here to illustrate the dispatch pattern.
        for (int64_t i = 0; i < input.numel(); ++i) {
            output_accessor[i] = relu_scalar(input_accessor[i]);
        }
    }

    template <torch::ScalarType stype>
    static void
    op(tag<torch::kCUDA, stype>, torch::Tensor input, torch::Tensor output) {
        using T = torch_scalar_cpp_type_t<stype>;

        c10::cuda::CUDAGuard device_guard(input.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        auto input_accessor  = input.packed_accessor64<T, 1>();
        auto output_accessor = output.packed_accessor64<T, 1>();

        int constexpr block_size = 256;
        int const num_blocks     = (input.numel() + block_size - 1) / block_size;
        relu_kernel<T><<<num_blocks, block_size, 0, stream>>>(input_accessor, output_accessor);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    using space      = axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

torch::Tensor
example_relu_impl(torch::Tensor input, placement plc) {
    static relu_op_t::dispatcher const table{relu_op_t::dispatcher::from_op<relu_op_t>(),
                                             relu_op_t::space{}};

    // Validate input rank
    TORCH_CHECK_VALUE(input.dim() == 1, "example_relu: expected 1D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.size(0) == 0) {
        if (plc == placement::in_place) {
            return input;
        } else {
            return torch::empty_like(input);
        }
    }

    auto output = (plc == placement::in_place) ? input : torch::empty_like(input);

    auto const dev = input.device().type();
    auto const st  = input.scalar_type();

    auto const dispatch_coord = std::make_tuple(dev, st);
    torch_dispatch("example_relu", table, dispatch_coord, input, output);

    return output;
}

torch::Tensor
example_relu(torch::Tensor input) {
    return example_relu_impl(input, placement::out_of_place);
}

torch::Tensor
example_relu_(torch::Tensor input) {
    return example_relu_impl(input, placement::in_place);
}

} // namespace dispatch_examples
