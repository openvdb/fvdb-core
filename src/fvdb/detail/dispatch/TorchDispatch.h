// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H
#define FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"
#include "fvdb/detail/dispatch/SparseDispatchTable.h"

#include <torch/types.h>

#include <cstdint>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// TorchDtypeAxis: DispatchAxis from ScalarType enum values
// -----------------------------------------------------------------------------
template <torch::ScalarType... Values> using TorchScalarTypeAxis = ValueOrdering<Values...>;

// -----------------------------------------------------------------------------
// TorchDeviceAxis: DispatchAxis for torch device types
// -----------------------------------------------------------------------------
template <torch::DeviceType... Values> using TorchDeviceDispatchAxis = ValueOrdering<Values...>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H
