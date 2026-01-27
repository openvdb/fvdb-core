// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_COMMON_H
#define DISPATCH_EXAMPLES_COMMON_H

#include "dispatch/macros.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"

#include <torch/types.h>

#include <string>

namespace dispatch_examples {

struct tensor_with_notes {
    torch::Tensor tensor;
    std::string notes;
};

// Concept for CUDA scan: integer+deterministic or float+non-deterministic
template <torch::ScalarType stype, dispatch::determinism det>
concept cuda_scan_allowed =
    (dispatch::torch_is_integer_stype_v<stype> && det == dispatch::determinism::required) ||
    (dispatch::torch_is_float_stype_v<stype> && det == dispatch::determinism::not_required);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_COMMON_H
