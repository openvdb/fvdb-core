// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "examples/gelu_for_each.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/for_each_torch.h"
#include "dispatch/tag_match.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"
#include "examples/common.h" // for __hostdev__
#include "examples/gelu_scalar.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <type_traits>

namespace dispatch_examples {

using namespace dispatch;

struct gelu_op {
    template <torch::DeviceType dev, torch::ScalarType stype>
    static void
    op(tag<dev, stype>, torch::Tensor in_tensor, torch::Tensor out_tensor) {
        auto in  = torch_accessor(torch_concrete_tensor<dev, stype, 1>{in_tensor});
        auto out = torch_accessor(torch_concrete_tensor<dev, stype, 1>{out_tensor});
        for_each(tag<dev, stype>{},
                 in_tensor.numel(),
                 [in, out] __hostdev__(tag<dev, stype>, int64_t idx) mutable {
                     out[idx] = gelu_scalar(in[idx]);
                 });
    }

    using space      = axes<torch_full_device_axis, torch_full_float_stype_axis>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

torch::Tensor
example_gelu_for_each_impl(torch::Tensor input, placement plc) {
    static gelu_op::dispatcher const table{gelu_op::dispatcher::from_op<gelu_op>(),
                                           gelu_op::space{}};

    // Validate input rank
    TORCH_CHECK_VALUE(
        input.dim() == 1, "example_gelu_for_each: expected 1D tensor, got ", input.dim(), "D");

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
