// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// gelu_for_each.cu - GELU implementation using the for_each primitive
// ================================================================================================
//
// This example demonstrates using dispatch::for_each for element-wise operations.
// Unlike relu.cu which manually launches kernels, this uses the for_each abstraction
// which handles device dispatch (CPU/CUDA/PrivateUse1) automatically.
//
// The flat_view abstraction handles tensors of any rank and stride, iterating over
// all elements in linear order with correct offset computation.
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
    template <torch::DeviceType dev, torch::ScalarType stype>
    static void
    op(tag<dev, stype>, torch::Tensor input, torch::Tensor output) {
        using Tag = tag<dev, stype>;

        flat_const_view<dev, stype> in{input};
        flat_mutable_view<dev, stype> out{output};

        for_each(Tag{}, in.numel, [=] __hostdev__(Tag, int64_t i) { out[i] = gelu_scalar(in[i]); });
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
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
    auto const dispatch_coord = std::make_tuple(dev, st);

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
