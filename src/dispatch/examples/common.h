// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_COMMON_H
#define DISPATCH_EXAMPLES_COMMON_H

#include "dispatch/macros.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"
#include "dispatch/with_value.h"

#include <torch/types.h>

#include <string>

namespace dispatch_examples {

struct tensor_with_notes {
    torch::Tensor tensor;
    std::string notes;
};

// Concept for CUDA scan: integer+deterministic or float+non-deterministic.
// Value-level form (for direct use with known NTTPs):
template <torch::ScalarType stype, dispatch::determinism det>
concept cuda_scan_allowed =
    (dispatch::torch_is_integer_stype_v<stype> && det == dispatch::determinism::required) ||
    (dispatch::torch_is_float_stype_v<stype> && det == dispatch::determinism::not_required);

// Tag-level form (for use in requires clauses on tag-dispatched ops):
template <typename Tag>
concept cuda_scan_allowed_tag = dispatch::with_type<Tag, torch::ScalarType> &&
                                dispatch::with_type<Tag, dispatch::determinism> &&
                                cuda_scan_allowed<dispatch::tag_get<torch::ScalarType, Tag>(),
                                                  dispatch::tag_get<dispatch::determinism, Tag>()>;

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_COMMON_H
