// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// gelu_for_each.cu - GELU implementation using the for_each primitive with fragment views
// ================================================================================================
//
// This example demonstrates using dispatch::for_each with fragment views for element-wise
// operations. Fragment views load/store multiple elements at once (16 bytes = 4 floats)
// for improved memory bandwidth and instruction-level parallelism on GPU.
//
// Contiguity dispatch:
//   - contiguous: Vectorized fragment loads (elements_per_frag = 4 for float32 on GPU)
//   - strided: Scalar access via offset computation (elements_per_frag = 1)
//
// The loop structure is uniform for both:
//   1. Fragment loop: processes num_fragments() fragments
//   2. Tail loop: handles remaining elements (only non-empty for contiguous with frag_size > 1)
//
// CPU always uses elements_per_fragment=1 regardless of contiguity.
//
// ================================================================================================

#include "examples/gelu_for_each.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"
#include "examples/gelu_scalar.h"

namespace dispatch_examples {

using namespace dispatch;

struct gelu_op {
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig>, torch::Tensor input, torch::Tensor output) {
        using Tag           = tag<dev, stype>;
        using FragConstView = flat_fragment_const_view<dev, stype, contig>;
        using FragMutView   = flat_fragment_mutable_view<dev, stype, contig>;
        using frag_type     = typename FragConstView::fragment_type;

        constexpr int64_t frag_size = FragConstView::elements_per_frag();

        FragConstView in{input};
        FragMutView out{output};

        // Process complete fragments
        int64_t const num_frags = in.num_fragments();
        for_each(Tag{}, num_frags, [=] __hostdev__(Tag, int64_t frag_idx) {
            frag_type in_frag = in.load_fragment(frag_idx);
            frag_type out_frag;
            DISPATCH_UNROLL
            for (int64_t i = 0; i < frag_size; ++i) {
                out_frag[i] = gelu_scalar(in_frag[i]);
            }
            out.store_fragment(frag_idx, out_frag);
        });

        // Process tail elements (only non-empty when frag_size > 1)
        int64_t const tail_count  = in.tail_count();
        int64_t const tail_offset = in.tail_offset();
        for_each(Tag{}, tail_count, [=] __hostdev__(Tag, int64_t i) {
            int64_t const idx = tail_offset + i;
            out[idx]          = gelu_scalar(in[idx]);
        });
    }

    using space = axes<torch_full_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

torch::Tensor
example_gelu_for_each_impl(torch::Tensor input, placement plc) {
    static auto const table = dispatch_table_from_op<gelu_op>();

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
    auto const contig         = combined_contiguity(input, output);
    auto const dispatch_coord = std::make_tuple(dev, st, contig);

    torch_dispatch("example_gelu_for_each", table, dispatch_coord, input, output);

    return output;
}

torch::Tensor
example_gelu_for_each(torch::Tensor input) {
    return example_gelu_for_each_impl(input, placement::out_of_place);
}

torch::Tensor
example_gelu_for_each_(torch::Tensor input) {
    return example_gelu_for_each_impl(input, placement::in_place);
}

} // namespace dispatch_examples
