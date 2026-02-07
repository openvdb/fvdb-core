// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// dispatch_set: a curly-brace constructible wrapper for runtime dispatch
// coordinates. Communicates at the call site that the enclosed values are
// an unordered set of dispatch attributes, matched to axes by type.
//
// Usage:
//
//     auto fn = table.select(dispatch_set{dev, stype, contig});
//     fn(input, output);
//
#ifndef DISPATCH_DISPATCH_DISPATCH_SET_H
#define DISPATCH_DISPATCH_DISPATCH_SET_H

#include <tuple>

namespace dispatch {

template <typename... Ts> struct dispatch_set {
    std::tuple<Ts...> values;

    constexpr dispatch_set(Ts... args) : values{args...} {}

    template <typename T>
    constexpr T const &
    get() const {
        return std::get<T>(values);
    }
};

// CTAD
template <typename... Ts> dispatch_set(Ts...) -> dispatch_set<Ts...>;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_DISPATCH_SET_H
