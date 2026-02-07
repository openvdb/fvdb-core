// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time boolean type wrappers. By using consteval member functions
// instead of static constexpr variables, we guarantee values exist only at
// compile-time and are never instantiated as addressable objects. This
// addresses a real issue with deeply nested templates and nvcc.
//
#ifndef DISPATCH_DISPATCH_CONSTEVAL_TYPES_H
#define DISPATCH_DISPATCH_CONSTEVAL_TYPES_H

namespace dispatch {

template <bool B> struct consteval_bool_type {
    static consteval bool
    value() {
        return B;
    }
};

using consteval_true_type  = consteval_bool_type<true>;
using consteval_false_type = consteval_bool_type<false>;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_CONSTEVAL_TYPES_H
