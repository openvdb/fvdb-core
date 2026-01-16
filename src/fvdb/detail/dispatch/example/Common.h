// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H
#define FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H

#include <torch/types.h>

#include <string>

namespace fvdb {
namespace dispatch {
namespace example {

struct TensorWithNotes {
    torch::Tensor tensor;
    std::string notes;
};

enum class Placement { InPlace, OutOfPlace };
enum class Determinism { NonDeterministic, Deterministic };
enum class Contiguity { Strided, Contiguous };

// Stringification helpers for enum values
inline char const *
toString(Placement p) {
    switch (p) {
    case Placement::InPlace: return "InPlace";
    case Placement::OutOfPlace: return "OutOfPlace";
    }
    return "Unknown";
}

inline char const *
toString(Determinism d) {
    switch (d) {
    case Determinism::NonDeterministic: return "NonDeterministic";
    case Determinism::Deterministic: return "Deterministic";
    }
    return "Unknown";
}

inline char const *
toString(Contiguity c) {
    switch (c) {
    case Contiguity::Strided: return "Strided";
    case Contiguity::Contiguous: return "Contiguous";
    }
    return "Unknown";
}

inline Contiguity
getContiguity(torch::Tensor tensor) {
    return tensor.is_contiguous() ? Contiguity::Contiguous : Contiguity::Strided;
}

// Concept to match integer torch::ScalarTypes (compile-time)
template <torch::ScalarType T>
constexpr bool is_integer_scalar_type_v =
    T == torch::kByte || T == torch::kChar || T == torch::kShort || T == torch::kInt ||
    T == torch::kLong || T == torch::kBool;

template <torch::ScalarType T>
concept IntegerTorchScalarType = is_integer_scalar_type_v<T>;

// Runtime check for integer scalar types
inline bool
isIntegerScalarType(torch::ScalarType stype) {
    return stype == torch::kByte || stype == torch::kChar || stype == torch::kShort ||
           stype == torch::kInt || stype == torch::kLong || stype == torch::kBool;
}

// Concept to match floating-point torch::ScalarTypes (compile-time)
template <torch::ScalarType T>
constexpr bool is_float_scalar_type_v =
    T == torch::kFloat || T == torch::kDouble || T == torch::kHalf || T == torch::kBFloat16;

template <torch::ScalarType T>
concept FloatTorchScalarType = is_float_scalar_type_v<T>;

// Runtime check for floating-point scalar types
inline bool
isFloatScalarType(torch::ScalarType stype) {
    return stype == torch::kFloat || stype == torch::kDouble || stype == torch::kHalf ||
           stype == torch::kBFloat16;
}

// Concept for CUDA scan: integer+deterministic or float+non-deterministic
template <torch::ScalarType stype, Determinism det>
concept CudaScanAllowed = (is_integer_scalar_type_v<stype> && det == Determinism::Deterministic) ||
                          (is_float_scalar_type_v<stype> && det == Determinism::NonDeterministic);

} // namespace example
} // namespace dispatch
} // namespace fvdb
#endif // FVDB_DETAIL_DISPATCH_EXAMPLE_COMMON_H
