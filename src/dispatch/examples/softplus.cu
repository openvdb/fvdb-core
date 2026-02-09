// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// softplus.cu — for_each + Views Example
// ================================================================================================
//
// Demonstrates the recommended pattern for elementwise ops using dispatch::for_each
// and dispatch::flat_in / dispatch::flat_out views.
//
// CONTRAST WITH relu.cu:
//   relu.cu (the "before"):
//     - Two concept-constrained relu_scalar overloads (half vs builtin float)
//     - A separate __global__ CUDA kernel function
//     - Two op() overloads: one CPU (manual serial loop), one CUDA (manual device guard,
//       stream, grid/block, kernel launch, C10_CUDA_KERNEL_LAUNCH_CHECK)
//     - Only handles 1D contiguous tensors
//
//   softplus.cu (the "after"):
//     - One __hostdev__ scalar function — works on all devices
//     - One impl function — all devices, all scalar types, both contiguities, ANY rank
//     - No manual CUDA kernel, no manual grid/block, no manual device guard
//     - flat_in/flat_out handle arbitrary rank and strided tensors transparently
//     - Contiguity resolved at dispatch time, not in op code
//
// THE PATTERN:
//   1. Write a __hostdev__ scalar function for the computation.
//   2. Write ONE impl function constrained by with_type on the tag's coordinates.
//      Use tag_get to extract compile-time DeviceType, ScalarType, contiguity.
//      Construct flat views from tensors. Call for_each with a lambda using operator[].
//   3. Wire into a dispatch_table with (device, scalar type, contiguity) axes.
//   4. The entry point calls table.select(dispatch_set{...}).
//
// ================================================================================================

#include "examples/softplus.h"

#include "dispatch/dispatch_set.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"

namespace dispatch_examples {

using namespace dispatch;

// ================================================================================================
// Step 1: The scalar function — __hostdev__, works on all devices and all float types.
// ================================================================================================
//
// Uses torch_compute_type_t to promote half types to float for arithmetic.
// This avoids the ambiguous operator> and ternary that half types cause in
// CUDA device code, while being identity for float/double. ONE function for
// all types — no separate overloads needed (unlike relu which needed
// concept-constrained half_like / builtin_float_like overloads).

template <typename T, typename C>
__hostdev__ T
softplus_scalar(T x, T beta, T threshold) {
    C const cx  = static_cast<C>(x);
    C const cb  = static_cast<C>(beta);
    C const cth = static_cast<C>(threshold);
    C const bx  = cb * cx;
    if (bx > cth) {
        return x;
    }
    return static_cast<T>(log1p(exp(bx)) / cb);
}

// ================================================================================================
// Step 2: The impl — ONE function for all devices, all scalar types, both contiguities, ANY rank.
// ================================================================================================
//
// The tag carries three compile-time coordinates:
//   - DeviceType:  selects CPU thread pool vs CUDA grid-stride kernel (via for_each)
//   - ScalarType:  selects the C++ type for views and arithmetic
//   - contiguity:  selects the view specialization (direct pointer vs unravel + stride)
//
// flat_in / flat_out map a flat linear index to the correct memory location regardless
// of the tensor's actual rank. for_each is a rank-1 map; the views hide the shape.
//
// The op author writes ZERO device-specific, contiguity-specific, or rank-specific code.

template <typename Tag>
    requires with_type<Tag, torch::DeviceType> && with_type<Tag, torch::ScalarType> &&
             with_type<Tag, contiguity>
void
softplus_impl(
    Tag tg, torch::Tensor const &input, torch::Tensor &output, double beta_d, double threshold_d) {
    // Extract compile-time coordinates from the tag
    constexpr auto dev    = tag_get<torch::DeviceType>(tg);
    constexpr auto stype  = tag_get<torch::ScalarType>(tg);
    constexpr auto contig = tag_get<contiguity>(tg);

    using scalar_t  = torch_scalar_cpp_type_t<stype>; // storage type (may be half)
    using compute_t = torch_compute_type_t<stype>;    // compute type (float for halves)

    // Device guard: no-op for CPU, sets CUDA device for GPU
    auto guard = make_device_guard(tg, input);

    // Flat views — rank-free. Contiguity selects the specialization:
    //   contiguous: operator[] is just data[flat_idx] (zero overhead, any rank)
    //   strided:    operator[] unravels flat_idx via div/mod, then applies strides
    auto in  = flat_in<dev, stype, contig>(input);
    auto out = flat_out<dev, stype, contig>(output);

    auto const beta      = static_cast<scalar_t>(beta_d);
    auto const threshold = static_cast<scalar_t>(threshold_d);

    // for_each: parallel iteration over [0, numel).
    // softplus_scalar<scalar_t, compute_t> promotes to compute_t for math, stores as scalar_t.
    for_each(tg, input.numel(), [=] __hostdev__(Tag, int64_t idx) {
        out[idx] = softplus_scalar<scalar_t, compute_t>(in[idx], beta, threshold);
    });
}

// ================================================================================================
// Step 3: Dispatch table wiring
// ================================================================================================
//
// softplus_op defines:
//   - space: (device, scalar type, contiguity) = 2 x 4 x 2 = 16 points
//   - subspaces: full coverage (all combinations supported)
//   - dispatcher: function signature for the kernel
//
// The dispatch table instantiates softplus_impl for every point in the space.
// At runtime, table.select(dispatch_set{dev, stype, contig}) picks the right instantiation.

struct softplus_op {
    template <typename Tag>
    static void
    op(Tag tg, torch::Tensor const &input, torch::Tensor &output, double beta, double threshold) {
        softplus_impl(tg, input, output, beta, threshold);
    }

    using space =
        axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using subspaces = coverage<space>;
    using dispatcher =
        dispatch_table<space, void(torch::Tensor const &, torch::Tensor &, double, double)>;
};

// ================================================================================================
// Step 4: Entry points
// ================================================================================================

void
example_softplus_out(torch::Tensor input, torch::Tensor output, double beta, double threshold) {
    static auto const table = dispatch_table_from_op<softplus_op>("example_softplus");

    auto const dev    = input.device().type();
    auto const stype  = input.scalar_type();
    auto const contig = torch_get_contiguity(input);

    table.select(dispatch_set{dev, stype, contig})(input, output, beta, threshold);
}

torch::Tensor
example_softplus(torch::Tensor input, double beta, double threshold) {
    auto output = torch::empty_like(input);
    example_softplus_out(input, output, beta, threshold);
    return output;
}

} // namespace dispatch_examples
