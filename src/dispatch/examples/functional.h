// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_FUNCTIONAL_H
#define DISPATCH_EXAMPLES_FUNCTIONAL_H

#include "examples/common.h"

#include <torch/types.h>

namespace dispatch_examples {

tensor_with_notes
inclusive_scan_functional(torch::Tensor input, dispatch::placement plc, dispatch::determinism det);

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_FUNCTIONAL_H
