// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "examples/gelu_for_each.h"

#include "dispatch/torch/unary_elementwise.h"
#include "examples/gelu_scalar.h"

namespace dispatch_examples {

using namespace dispatch;

struct gelu_scalar_op {
    template <typename T>
    __hostdev__ static T
    scalar_op(T x) {
        return gelu_scalar(x);
    }
};

using gelu_op =
    unary_elementwise<gelu_scalar_op, torch_full_device_axis, torch_full_float_stype_axis, 1>;

torch::Tensor
example_gelu_for_each(torch::Tensor input) {
    return gelu_op::map(input, placement::out_of_place, "example_gelu_for_each");
}

torch::Tensor
example_gelu_for_each_(torch::Tensor input) {
    return gelu_op::map(input, placement::in_place, "example_gelu_for_each_");
}

} // namespace dispatch_examples
