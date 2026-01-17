// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_EXAMPLE_RELU_H
#define FVDB_DETAIL_DISPATCH_EXAMPLE_RELU_H

#include "fvdb/detail/dispatch/example/Common.h"

#include <torch/types.h>

namespace fvdb {
namespace dispatch {
namespace example {

torch::Tensor relu(torch::Tensor input, Placement placement);

} // namespace example
} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_EXAMPLE_RELU_H
