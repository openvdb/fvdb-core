// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CRTP base class and helper for unary element-wise operations.
// Reduces boilerplate for simple ops like relu, gelu, sigmoid, etc.
//
// Performance tuning:
//   GrainSize: Elements per thread (improves ILP). Default 4.
//   BlockDim: Threads per block. Default 256.
//
// For most memory-bound ops, defaults are reasonable. Compute-bound ops
// or ops with high register pressure may benefit from tuning.
//
#ifndef DISPATCH_DISPATCH_UNARY_ELEMENTWISE_H
#define DISPATCH_DISPATCH_UNARY_ELEMENTWISE_H

#include "dispatch/dispatch_table.h"
#include "dispatch/for_each_torch.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"

#include <cstddef>

namespace dispatch {

//------------------------------------------------------------------------------
// __hostdev__ macro for host/device portability
//------------------------------------------------------------------------------

#ifndef __hostdev__
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif
#endif

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
//   Rank       - Tensor rank (typically 1 for element-wise)
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
// With custom tuning:
//   struct my_op : unary_elementwise_op<my_op,
//                                        torch_full_device_axis,
//                                        torch_full_float_stype_axis, 1,
//                                        /*GrainSize=*/8, /*BlockDim=*/128> { ... };

template <typename Derived,
          typename DeviceAxis,
          typename StypeAxis,
          size_t Rank,
          int64_t GrainSize = for_each_config::default_grain_size,
          int BlockDim      = for_each_config::default_block_dim>
struct unary_elementwise_op {
    template <torch::DeviceType dev, torch::ScalarType stype>
    static void
    op(tag<dev, stype>, torch::Tensor in_tensor, torch::Tensor out_tensor) {
        auto in  = torch_accessor(torch_concrete_tensor<dev, stype, Rank>{in_tensor});
        auto out = torch_accessor(torch_concrete_tensor<dev, stype, Rank>{out_tensor});
        for_each<GrainSize, BlockDim>(tag<dev, stype>{},
                                      in_tensor.numel(),
                                      [in, out] __hostdev__(tag<dev, stype>, int64_t idx) mutable {
                                          out[idx] = Derived::scalar_op(in[idx]);
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

    // Validate input rank (assumes 1D for now, could be parameterized)
    TORCH_CHECK_VALUE(input.dim() == 1, name, ": expected 1D tensor, got ", input.dim(), "D");

    // Handle empty tensor case
    if (input.size(0) == 0) {
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

#endif // DISPATCH_DISPATCH_UNARY_ELEMENTWISE_H
