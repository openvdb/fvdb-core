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
// TorchScalarTypeAxis: DispatchAxis from ScalarType enum values
// -----------------------------------------------------------------------------
template <torch::ScalarType... Vals> using TorchScalarTypeAxis = SameTypeUniqueValuePack<Vals...>;

// -----------------------------------------------------------------------------
// TorchDeviceDispatchAxis: DispatchAxis for torch device types
// -----------------------------------------------------------------------------
template <torch::DeviceType... Vals>
using TorchDeviceDispatchAxis = SameTypeUniqueValuePack<Vals...>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TORCHDISPATCH_H
