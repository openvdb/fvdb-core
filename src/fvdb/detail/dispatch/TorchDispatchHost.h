// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __CUDACC__
#error "This header must not be included during nvcc compilation"
#endif

#ifndef FVDB_DETAIL_DISPATCH_TORCHDISPATCHHOST_H
#define FVDB_DETAIL_DISPATCH_TORCHDISPATCHHOST_H

#include "fvdb/detail/dispatch/SparseDispatchTable.h"

#include <torch/types.h>

#include <cstdint>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// TorchDtypeAxis: DispatchAxis from ScalarType enum values
// -----------------------------------------------------------------------------
// Creates a DispatchAxis from torch ScalarType enum values, automatically
// generating Tag<V> types for each value.
// Usage: TorchDtypeAxis<torch::kFloat32, torch::kFloat64, torch::kInt32>

template <auto... Vs> using TorchDtypeAxis = DispatchAxis<ValueTypePair<Vs, Tag<Vs>>...>;

// Example: A dtype axis for common floating-point types
using FloatDtypeAxis = TorchDtypeAxis<torch::kFloat32, torch::kFloat64>;

// Example: A dtype axis matching the existing DtypeList
using StandardDtypeAxis =
    TorchDtypeAxis<torch::kFloat32, torch::kFloat64, torch::kInt32, torch::kInt64, torch::kBool>;

// -----------------------------------------------------------------------------
// TorchDeviceAxis: DispatchAxis for torch device types
// -----------------------------------------------------------------------------
// Note: ScalarCppTypeT is defined in ConcreteTensor.h for use by accessor functions
// For devices, we don't have distinct C++ types - we need to create tag types
// that wrap the device enum values. The Tag template creates a unique type for
// each enum value.

// Create a DispatchAxis directly from enum values
// Automatically generates Tag<V> types for each value
// Usage: DispatchAxisFromValues<torch::kCPU, torch::kCUDA, torch::kPrivateUse1>
template <auto... Vs> using TorchDeviceDispatchAxis = DispatchAxis<ValueTypePair<Vs, Tag<Vs>>...>;

// Now TorchDeviceAxis can be defined much more simply:
using ExampleTorchDeviceAxis =
    TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA, torch::kPrivateUse1>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TORCHDISPATCHHOST_H
