// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_EXAMPLE_FUNCTIONAL_H
#define FVDB_DETAIL_DISPATCH_EXAMPLE_FUNCTIONAL_H

#include "fvdb/detail/dispatch/example/Common.h"

#include <torch/types.h>

namespace fvdb {
namespace dispatch {
namespace example {

TensorWithNotes
inclusiveScanFunctional(torch::Tensor input, Placement placement, Determinism determinism);

} // namespace example
} // namespace dispatch
} // namespace fvdb
#endif // FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H
