// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CRTP base class and helper for unary element-wise operations.
// Reduces boilerplate for simple ops like relu, gelu, sigmoid, etc.
//
// Supports N-dimensional tensors with configurable iteration policy.
// Uses stride-aware accessors for non-contiguous tensor support.
//
// Performance tuning:
//   GrainSize: Elements per thread (improves ILP). Default 4.
//   BlockDim: Threads per block. Default 256.
//
// For most memory-bound ops, defaults are reasonable. Compute-bound ops
// or ops with high register pressure may benefit from tuning.
//
#ifndef DISPATCH_DISPATCH_TORCH_UNARY_ELEMENTWISE_H
#define DISPATCH_DISPATCH_TORCH_UNARY_ELEMENTWISE_H

#include "dispatch/dispatch_table.h"
#include "dispatch/iteration_policy.h"
#include "dispatch/macros.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/nd_accessor.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <c10/core/DeviceGuard.h>

#include <array>
#include <cstddef>

namespace dispatch {

//------------------------------------------------------------------------------
// CRTP base class for unary element-wise operations
//------------------------------------------------------------------------------
// Derived class must provide:
//   template <typename T>
//   __hostdev__ static T scalar_op(T x);
//
// Template parameters:
//   Derived    - CRTP derived class
//   DeviceAxis - Axis of supported devices (e.g., torch_full_device_axis)
//   StypeAxis  - Axis of supported scalar types (e.g., torch_full_float_stype_axis)
//   Rank       - Tensor rank (1 for 1D, 2 for 2D, etc.)
//   IterPolicy - Iteration policy (row_major or col_major, default: row_major)
//   GrainSize  - Elements per thread for ILP (default: 4)
//   BlockDim   - CUDA threads per block (default: 256)
//
// Usage:
//   struct my_op : unary_elementwise_op<my_op,
//                                        torch_full_device_axis,
//                                        torch_full_float_stype_axis, 1> {
//       template <typename T>
//       __hostdev__ static T scalar_op(T x) { return my_scalar_func(x); }
//   };
//
// For 2D tensors:
//   struct my_2d_op : unary_elementwise_op<my_2d_op,
//                                           torch_full_device_axis,
//                                           torch_full_float_stype_axis, 2> { ... };
//
// With custom tuning:
//   struct my_op : unary_elementwise_op<my_op,
//                                        torch_full_device_axis,
//                                        torch_full_float_stype_axis, 1,
//                                        row_major,
//                                        /*GrainSize=*/8, /*BlockDim=*/128> { ... };

template <typename Derived,
          typename DeviceAxis,
          typename StypeAxis,
          int64_t Rank,
          typename IterPolicy = row_major,
          int64_t GrainSize   = for_each_config::default_grain_size,
          int BlockDim        = for_each_config::default_block_dim>
struct unary_elementwise_op {
    static constexpr int64_t rank = Rank;

    template <torch::DeviceType dev, torch::ScalarType stype>
    static void
    op(tag<dev, stype>, torch::Tensor in_tensor, torch::Tensor out_tensor) {
        // Guard device context for CUDA/PrivateUse1 to ensure correct stream selection
        c10::OptionalDeviceGuard device_guard;
        if constexpr (dev == torch::kCUDA || dev == torch::kPrivateUse1) {
            device_guard.reset_device(in_tensor.device());
        }

        using scalar_t = torch_scalar_cpp_type_t<stype>;

        auto in  = nd_accessor<scalar_t, Rank>::from_tensor(in_tensor);
        auto out = nd_accessor<scalar_t, Rank>::from_tensor(out_tensor);

        // Build shape array
        std::array<int64_t, Rank> shape;
        for (int64_t d = 0; d < Rank; ++d) {
            shape[d] = in_tensor.size(d);
        }

        for_each_nd<Rank, IterPolicy, GrainSize, BlockDim>(
            tag<dev, stype>{},
            shape,
            [in, out] __hostdev__(tag<dev, stype>, std::array<int64_t, Rank> const &idx) mutable {
                out(idx) = Derived::scalar_op(in(idx));
            });
    }

    using space      = axes<DeviceAxis, StypeAxis>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

//------------------------------------------------------------------------------
// Implementation helper for unary element-wise operations
//------------------------------------------------------------------------------
// Handles: validation, empty tensors, output allocation, dispatch.
//
// Usage:
//   torch::Tensor my_func(torch::Tensor input) {
//       return unary_elementwise_impl<my_op>(input, placement::out_of_place, "my_func");
//   }

template <typename Op>
torch::Tensor
unary_elementwise_impl(torch::Tensor input, placement plc, char const *name) {
    static typename Op::dispatcher const table{Op::dispatcher::template from_op<Op>(),
                                               typename Op::space{}};

    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == Op::rank, name, ": expected ", Op::rank, "D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.numel() == 0) {
        if (plc == placement::in_place) {
            return input;
        } else {
            return torch::empty_like(input);
        }
    }

    auto output = (plc == placement::in_place) ? input : torch::empty_like(input);

    auto const dev            = input.device().type();
    auto const st             = input.scalar_type();
    auto const dispatch_coord = std::make_tuple(dev, st);

    torch_dispatch(name, table, dispatch_coord, input, output);

    return output;
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_UNARY_ELEMENTWISE_H
