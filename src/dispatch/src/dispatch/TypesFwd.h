// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TYPESFWD_H
#define FVDB_DETAIL_DISPATCH_TYPESFWD_H

namespace fvdb {
namespace dispatch {

// Minimal Value Pack Struct - a compile-time sequence of non-type template parameters.
// The ValuePack concept and inspection utilities are defined in Values.h.
template <auto... Vs> struct Values {};

template <auto... values> using Tag = Values<values...>;

enum class Placement { InPlace, OutOfPlace };
enum class Determinism { NonDeterministic, Deterministic };
enum class Contiguity { Strided, Contiguous };

using FullPlacementAxis   = Values<Placement::InPlace, Placement::OutOfPlace>;
using FullDeterminismAxis = Values<Determinism::NonDeterministic, Determinism::Deterministic>;
using FullContiguityAxis  = Values<Contiguity::Strided, Contiguity::Contiguous>;

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

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TYPESFWD_H
