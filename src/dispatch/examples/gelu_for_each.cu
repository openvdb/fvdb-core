// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// gelu_for_each.cu - GELU implementation using the for_each primitive
// ================================================================================================
//
// This example demonstrates using dispatch::for_each with flat views for element-wise
// operations. The implementation is optimized for contiguous GPU tensors using vectorized
// loads/stores (float4/double2) for improved memory throughput.
//
// Dispatch paths:
//   - CUDA contiguous: Vectorized loads (128-bit transactions) + scalar tail
//   - CUDA strided: Scalar access via offset computation
//   - CPU: Scalar access (no vectorization benefit on CPU for this op)
//
// ================================================================================================

#include "examples/gelu_for_each.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"
#include "examples/gelu_scalar.h"

#ifdef __CUDACC__
#include "examples/gelu_vectorized.cuh"
#endif

namespace dispatch_examples {

using namespace dispatch;

//------------------------------------------------------------------------------
// CUDA contiguous: vectorized path (float4/double2)
//------------------------------------------------------------------------------
#ifdef __CUDACC__

template <torch::ScalarType stype>
void
gelu_cuda_contiguous(torch::Tensor input, torch::Tensor output) {
    using scalar_t = torch_scalar_cpp_type_t<stype>;
    using Tag      = tag<torch::kCUDA, stype>;

    scalar_t const *in_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr      = output.data_ptr<scalar_t>();
    int64_t const numel    = input.numel();

    // Vector configuration
    constexpr int64_t vec_size = vector_traits<scalar_t>::vector_size;
    int64_t const num_vecs     = numel / vec_size;
    int64_t const tail_start   = num_vecs * vec_size;
    int64_t const tail_count   = numel - tail_start;

    // Process vectorized portion
    if (num_vecs > 0) {
        for_each(Tag{}, num_vecs, [=] __device__(Tag, int64_t vec_idx) {
            int64_t const offset = vec_idx * vec_size;
            auto vec_in          = load_vector<scalar_t>(in_ptr + offset);
            auto vec_out         = gelu_vector<scalar_t>(vec_in);
            store_vector<scalar_t>(out_ptr + offset, vec_out);
        });
    }

    // Process tail (always less than vec_size elements)
    if (tail_count > 0) {
        for_each(Tag{}, tail_count, [=] __device__(Tag, int64_t i) {
            int64_t const idx = tail_start + i;
            out_ptr[idx]      = gelu_scalar(in_ptr[idx]);
        });
    }
}

#endif // __CUDACC__

//------------------------------------------------------------------------------
// Generic scalar path (strided CUDA, all CPU)
//------------------------------------------------------------------------------

struct gelu_op {
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig>, torch::Tensor input, torch::Tensor output) {
#ifdef __CUDACC__
        // Use vectorized path for contiguous CUDA float/double
        if constexpr (dev == torch::kCUDA && contig == contiguity::contiguous &&
                      (stype == torch::kFloat32 || stype == torch::kFloat64)) {
            gelu_cuda_contiguous<stype>(input, output);
            return;
        }
#endif

        // Scalar fallback for strided CUDA, half types, and all CPU
        using Tag       = tag<dev, stype>;
        using ConstView = flat_const_view<dev, stype>;
        using MutView   = flat_mutable_view<dev, stype>;

        ConstView in{input};
        MutView out{output};
        int64_t const numel = in.numel;

        for_each(
            Tag{}, numel, [=] __hostdev__(Tag, int64_t idx) { out[idx] = gelu_scalar(in[idx]); });
    }

    using space = axes<torch_full_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

// Internal implementation that operates on pre-allocated input/output
void
example_gelu_for_each_kernel(torch::Tensor input, torch::Tensor output) {
    static auto const table = dispatch_table_from_op<gelu_op>();

    if (input.numel() == 0) {
        return;
    }

    auto const dev            = input.device().type();
    auto const st             = input.scalar_type();
    auto const contig         = combined_contiguity(input, output);
    auto const dispatch_coord = std::make_tuple(dev, st, contig);

    torch_dispatch("example_gelu_for_each", table, dispatch_coord, input, output);
}

torch::Tensor
example_gelu_for_each(torch::Tensor input) {
    if (input.numel() == 0) {
        return torch::empty_like(input);
    }
    auto output = torch::empty_like(input);
    example_gelu_for_each_kernel(input, output);
    return output;
}

void
example_gelu_for_each_out(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.sizes() == output.sizes(),
                "example_gelu_for_each_out: input and output must have the same shape");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                "example_gelu_for_each_out: input and output must have the same dtype");
    TORCH_CHECK(input.device() == output.device(),
                "example_gelu_for_each_out: input and output must be on the same device");
    example_gelu_for_each_kernel(input, output);
}

torch::Tensor
example_gelu_for_each_(torch::Tensor input) {
    example_gelu_for_each_kernel(input, input);
    return input;
}

} // namespace dispatch_examples
