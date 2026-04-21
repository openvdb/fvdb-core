// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TYPETRAITS_H
#define FVDB_TYPETRAITS_H

#include <c10/util/Half.h>

#include <type_traits>

namespace fvdb {

/// @brief A helper struct to determine if a type is a floating-point type or a half-precision
/// floating-point type.
/// @tparam T The type to check.
template <class T>
struct is_floating_point_or_half
    : std::integral_constant<bool,
                             // Note: standard floating-point types
                             std::is_same<float, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<double, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<long double, typename std::remove_cv<T>::type>::value
                                 // Note: extended floating-point types (C++23, if supported)
                                 ||
                                 std::is_same<c10::Half, typename std::remove_cv<T>::type>::value> {
};

} // namespace fvdb

#endif // FVDB_TYPETRAITS_H
