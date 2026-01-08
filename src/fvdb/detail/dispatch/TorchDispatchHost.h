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
// TorchDtypeAxis: DispatchAxis from a pack of C++ types
// -----------------------------------------------------------------------------
// Demonstrates defining a DispatchAxis from just raw C++ element types,
// automatically associating each with the corresponding torch ScalarType enum.
// Uses c10::CppTypeToScalarType<T> for the mapping (not available in torch:: namespace).

template <typename... Ts>
using TorchDtypeAxis = DispatchAxis<ValueTypePair<c10::CppTypeToScalarType<Ts>::value, Ts>...>;

// Example: A dtype axis for common floating-point types
using FloatDtypeAxis = TorchDtypeAxis<float, double>;

// Example: A dtype axis matching the existing DtypeList
using StandardDtypeAxis = TorchDtypeAxis<float, double, int32_t, int64_t, bool>;

// -----------------------------------------------------------------------------
// TorchDeviceAxis: DispatchAxis for torch device types
// -----------------------------------------------------------------------------
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
