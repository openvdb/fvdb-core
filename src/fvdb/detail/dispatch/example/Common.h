// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H
#define FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H

#include "fvdb/detail/dispatch/TorchDispatch.h"
#include "fvdb/detail/dispatch/TypesFwd.h"

#include <torch/types.h>

#include <concepts>
#include <string>

namespace fvdb {
namespace dispatch {
namespace example {

struct TensorWithNotes {
    torch::Tensor tensor;
    std::string notes;
};

// Concept for CUDA scan: integer+deterministic or float+non-deterministic
template <torch::ScalarType stype, Determinism det>
concept CudaScanAllowed = (is_integer_scalar_type_v<stype> && det == Determinism::Deterministic) ||
                          (is_float_scalar_type_v<stype> && det == Determinism::NonDeterministic);

} // namespace example
} // namespace dispatch
} // namespace fvdb
#endif // FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H
