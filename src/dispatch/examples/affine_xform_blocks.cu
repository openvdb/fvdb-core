// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Block-based affine transformation: y = R @ x + T
//
// This example demonstrates the block-based accessor framework:
//   - Uses element_accessor with structured element shapes (mat3x3, vec3)
//   - Uses for_each_blocks for block-based iteration
//   - Processes Vec elements per thread for ILP
//
// Compared to affine_xform_ternary.cu, this version:
//   - Uses a unified element_accessor instead of separate gather/scatter
//   - Loads/stores blocks of elements at a time
//   - Handles contiguous/strided with a single code path
//

#include "dispatch/dispatch_table.h"
#include "dispatch/macros.h"
#include "dispatch/torch/accessors_C.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <cstdint>

namespace dispatch_examples {

using namespace dispatch;

//------------------------------------------------------------------------------
// Element shapes
//------------------------------------------------------------------------------

using mat3x3 = extents<3, 3>; // 3x3 rotation matrix
using vec3   = extents<3>;    // 3-vector (translation, position)

//------------------------------------------------------------------------------
// Block operation: y = R @ x + t for a block of elements
//------------------------------------------------------------------------------

template <size_t Vec, typename T>
__hostdev__ void
affine_xform_block(block_storage<packed_element<T, 9>, Vec> const &R_blk,
                   block_storage<packed_element<T, 3>, Vec> const &t_blk,
                   block_storage<packed_element<T, 3>, Vec> const &x_blk,
                   block_storage<packed_element<T, 3>, Vec> &y_blk) {
DISPATCH_UNROLL
    for (size_t b = 0; b < Vec; ++b) {
        // y[i] = sum_j(R[i,j] * x[j]) + t[i]
DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            T sum = t_blk[b][i];
DISPATCH_UNROLL
            for (int j = 0; j < 3; ++j) {
                sum += R_blk[b][i * 3 + j] * x_blk[b][j];
            }
            y_blk[b][i] = sum;
        }
    }
}

//------------------------------------------------------------------------------
// Vec width selection based on device and dtype
//------------------------------------------------------------------------------

template <torch::DeviceType dev, torch::ScalarType stype>
consteval size_t
affine_vec_width() {
    if constexpr (dev == torch::kCPU) {
        // CPU: moderate vectorization for cache efficiency
        if constexpr (stype == torch::kFloat32)
            return 4;
        if constexpr (stype == torch::kFloat64)
            return 2;
        return 1;
    } else {
        // CUDA: keep Vec=1 to avoid register spills with large elements
        // (each element is 9+3+3+3 = 18 scalars)
        return 1;
    }
}

//------------------------------------------------------------------------------
// Affine transform operation struct for dispatch
//------------------------------------------------------------------------------

struct affine_xform_blocks {
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig> t,
       torch::Tensor R_tensor,
       torch::Tensor T_tensor,
       torch::Tensor x_tensor,
       torch::Tensor out_tensor) {
        auto guard = make_device_guard(t, R_tensor);

        // Create accessors with appropriate element shapes
        using R_acc_t = element_accessor<dev, stype, contig, 1, mat3x3>;
        using V_acc_t = element_accessor<dev, stype, contig, 1, vec3>;

        auto R_acc = R_acc_t::from_tensor(R_tensor);
        auto t_acc = V_acc_t::from_tensor(T_tensor);
        auto x_acc = V_acc_t::from_tensor(x_tensor);
        auto y_acc = V_acc_t::from_tensor(out_tensor);

        int64_t const N      = x_tensor.size(0);
        constexpr size_t Vec = affine_vec_width<dev, stype>();

        // Block-based iteration
        for_each_blocks<linear_schedule<Vec>>(
            t, N, [=] __hostdev__(tag<dev, stype, contig>, auto blk) mutable {
                // Load blocks of inputs
                auto R_blk = R_acc.load(blk);
                auto t_blk = t_acc.load(blk);
                auto x_blk = x_acc.load(blk);
                auto y_blk = y_acc.prepare(blk);

                // Compute: y = R @ x + t for each element in block
                affine_xform_block(R_blk, t_blk, x_blk, y_blk);

                // Store output block
                y_acc.store(blk, y_blk);
            });
    }

    using space =
        axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using dispatcher =
        dispatch_table<space, void(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>;
};

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

torch::Tensor
example_affine_xform_blocks(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    static affine_xform_blocks::dispatcher const table{
        affine_xform_blocks::dispatcher::from_op<affine_xform_blocks>(),
        affine_xform_blocks::space{}};

    // Validate tensor ranks
    TORCH_CHECK_VALUE(
        R.dim() == 3, "example_affine_xform_blocks: R must be 3D (N, 3, 3), got ", R.dim(), "D");
    TORCH_CHECK_VALUE(
        T.dim() == 2, "example_affine_xform_blocks: T must be 2D (N, 3), got ", T.dim(), "D");
    TORCH_CHECK_VALUE(
        x.dim() == 2, "example_affine_xform_blocks: x must be 2D (N, 3), got ", x.dim(), "D");

    // Validate element shapes
    TORCH_CHECK_VALUE(R.size(1) == 3 && R.size(2) == 3,
                      "example_affine_xform_blocks: R element shape must be (3, 3), got (",
                      R.size(1),
                      ", ",
                      R.size(2),
                      ")");
    TORCH_CHECK_VALUE(T.size(1) == 3,
                      "example_affine_xform_blocks: T element shape must be (3,), got (",
                      T.size(1),
                      ",)");
    TORCH_CHECK_VALUE(x.size(1) == 3,
                      "example_affine_xform_blocks: x element shape must be (3,), got (",
                      x.size(1),
                      ",)");

    // Determine iteration dimension size N
    int64_t const N = x.size(0);

    // Validate broadcast compatibility on iteration dimension
    TORCH_CHECK_VALUE(R.size(0) == N || R.size(0) == 1,
                      "example_affine_xform_blocks: R iteration dim must be ",
                      N,
                      " or 1, got ",
                      R.size(0));
    TORCH_CHECK_VALUE(T.size(0) == N || T.size(0) == 1,
                      "example_affine_xform_blocks: T iteration dim must be ",
                      N,
                      " or 1, got ",
                      T.size(0));

    // Validate all tensors have same dtype
    TORCH_CHECK_VALUE(R.scalar_type() == T.scalar_type() && T.scalar_type() == x.scalar_type(),
                      "example_affine_xform_blocks: all tensors must have same dtype, got R=",
                      R.scalar_type(),
                      ", T=",
                      T.scalar_type(),
                      ", x=",
                      x.scalar_type());

    // Validate all tensors on same device
    TORCH_CHECK_VALUE(R.device() == T.device() && T.device() == x.device(),
                      "example_affine_xform_blocks: all tensors must be on same device");

    // Handle empty case
    if (N == 0) {
        return torch::empty({0, 3}, x.options());
    }

    // Handle broadcasting via expand (stride-0 views, no copy)
    if (R.size(0) == 1 && N > 1)
        R = R.expand({N, 3, 3});
    if (T.size(0) == 1 && N > 1)
        T = T.expand({N, 3});

    // Allocate output
    auto output = torch::empty({N, 3}, x.options());

    // Dispatch
    auto const dev    = x.device().type();
    auto const stype  = x.scalar_type();
    auto const contig = combined_contiguity(R, T, x, output);

    torch_dispatch("example_affine_xform_blocks",
                   table,
                   std::make_tuple(dev, stype, contig),
                   R,
                   T,
                   x,
                   output);

    return output;
}

} // namespace dispatch_examples
