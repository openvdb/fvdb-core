// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Wrapper for unary element-wise operations.
// Reduces boilerplate for simple ops like relu, gelu, sigmoid, etc.
//
// Supports N-dimensional tensors with configurable iteration policy.
// Dispatches on contiguity for optimized access:
//   - Contiguous tensors: direct data[idx] access (no stride math)
//   - Strided tensors: N-D indexing with stride computation
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
#include "dispatch/torch/accessors.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <c10/core/DeviceGuard.h>

#include <array>
#include <cstddef>

namespace dispatch {

//------------------------------------------------------------------------------
// Wrapper for unary element-wise operations
//------------------------------------------------------------------------------
// ScalarOp must provide:
//   template <typename T>
//   __hostdev__ static T scalar_op(T x);
//
// Template parameters:
//   ScalarOp   - Class providing the scalar_op static method
//   DeviceAxis - Axis of supported devices (e.g., torch_full_device_axis)
//   StypeAxis  - Axis of supported scalar types (e.g., torch_full_float_stype_axis)
//   Rank       - Tensor rank (1 for 1D, 2 for 2D, etc.)
//   IterPolicy - Iteration policy (row_major or col_major, default: row_major)
//   GrainSize  - Elements per thread for ILP (default: 4)
//   BlockDim   - CUDA threads per block (default: 256)
//
// Usage:
//   struct my_scalar_op {
//       template <typename T>
//       __hostdev__ static T scalar_op(T x) { return my_scalar_func(x); }
//   };
//
//   using my_op = unary_elementwise<my_scalar_op,
//                                   torch_full_device_axis,
//                                   torch_full_float_stype_axis, 1>;
//
//   torch::Tensor result = my_op::map(input, placement::out_of_place, "my_op");
//
// For 2D tensors:
//   using my_2d_op = unary_elementwise<my_scalar_op,
//                                      torch_full_device_axis,
//                                      torch_full_float_stype_axis, 2>;
//
// With custom tuning:
//   using my_op = unary_elementwise<my_scalar_op,
//                                   torch_full_device_axis,
//                                   torch_full_float_stype_axis, 1,
//                                   row_major,
//                                   /*GrainSize=*/8, /*BlockDim=*/128>;

template <typename ScalarOp,
          typename DeviceAxis,
          typename StypeAxis,
          int64_t Rank,
          typename IterPolicy = row_major,
          int64_t GrainSize   = for_each_config::default_grain_size,
          int BlockDim        = for_each_config::default_block_dim>
struct unary_elementwise {
    static constexpr int64_t rank = Rank;

    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig>, torch::Tensor in_tensor, torch::Tensor out_tensor) {
        // Guard device context for CUDA/PrivateUse1 to ensure correct stream selection
        c10::OptionalDeviceGuard device_guard;
        if constexpr (dev == torch::kCUDA || dev == torch::kPrivateUse1) {
            device_guard.reset_device(in_tensor.device());
        }

        // Accessor derives value_type internally from stype
        auto in_acc  = accessor<stype, contig, Rank>::from_tensor(in_tensor);
        auto out_acc = accessor<stype, contig, Rank>::from_tensor(out_tensor);

        if constexpr (contig == contiguity::contiguous) {
            // Contiguous path: linear iteration with direct data[idx] access
            for_each<GrainSize, BlockDim>(tag<dev>{},
                                          in_tensor.numel(),
                                          [in_acc, out_acc]
                                          __hostdev__(tag<dev>, int64_t idx) mutable {
                                              out_acc[idx] = ScalarOp::scalar_op(in_acc[idx]);
                                          });
        } else {
            // Strided path: N-D iteration with stride-based access
            std::array<int64_t, Rank> shape;
            for (int64_t d = 0; d < Rank; ++d) {
                shape[d] = in_tensor.size(d);
            }

            for_each_nd<Rank, IterPolicy, GrainSize, BlockDim>(
                tag<dev>{},
                shape,
                [in_acc, out_acc]
                __hostdev__(tag<dev>, std::array<int64_t, Rank> const &idx) mutable {
                    out_acc(idx) = ScalarOp::scalar_op(in_acc(idx));
                });
        }
    }

    using space      = axes<DeviceAxis, StypeAxis, full_contiguity_axis>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;

    //--------------------------------------------------------------------------
    // Map the scalar operation over the input tensor
    //--------------------------------------------------------------------------
    // Handles: validation, empty tensors, output allocation, dispatch.
    static torch::Tensor
    map(torch::Tensor input, placement plc, char const *name) {
        static dispatcher const table{dispatcher::template from_op<unary_elementwise>(), space{}};

        // Validate input rank
        TORCH_CHECK_VALUE(
            input.dim() == rank, name, ": expected ", rank, "D tensor, got ", input.dim(), "D");

        // Handle empty tensor case
        if (input.numel() == 0) {
            return (plc == placement::in_place) ? input : torch::empty_like(input);
        }

        auto output = (plc == placement::in_place) ? input : torch::empty_like(input);

        auto const dev    = input.device().type();
        auto const st     = input.scalar_type();
        auto const contig = combined_contiguity(input, output);

        torch_dispatch(name, table, std::make_tuple(dev, st, contig), input, output);

        return output;
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_TORCH_UNARY_ELEMENTWISE_H
