// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GridAccessor.h — Concept-constrained grid accessor construction.
//
// Provides make_grid_accessor(), which selects hostAccessor() or
// deviceAccessor() at compile time based on the dispatch tag's device
// coordinate.  Aligned with dispatch::make_device_guard in
// dispatch/torch/dispatch.h.
//
// GridBatchImpl::Accessor is already a trivially-copyable POD
// (pointers + scalars), safe for CUDA kernel capture.
//
#ifndef FVDB_DETAIL_DISPATCH_GRIDACCESSOR_H
#define FVDB_DETAIL_DISPATCH_GRIDACCESSOR_H

#include "dispatch/torch/dispatch.h"

#include <fvdb/detail/GridBatchImpl.h>

namespace fvdb {
namespace detail {
namespace dispatch {

/// @brief Construct a host grid accessor (CPU path).
inline GridBatchImpl::Accessor
make_grid_accessor(::dispatch::cpu_tag auto, GridBatchImpl const &grid) {
    return grid.hostAccessor();
}

/// @brief Construct a device grid accessor (CUDA / PrivateUse1 path).
inline GridBatchImpl::Accessor
make_grid_accessor(::dispatch::gpu_tag auto, GridBatchImpl const &grid) {
    return grid.deviceAccessor();
}

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_GRIDACCESSOR_H
