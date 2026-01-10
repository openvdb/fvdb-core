// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_PERMUTATIONARRAY_H
#define FVDB_DETAIL_DISPATCH_PERMUTATIONARRAY_H

#include "fvdb/detail/dispatch/AxisOuterProduct.h"
#include "fvdb/detail/dispatch/Values.h"

#include <array>
#include <concepts>
#include <optional>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// PermutationSpace concept
// -----------------------------------------------------------------------------
// A PermutationSpace is a AxisOuterProduct. Given an ordered set of values
// of heterogeneous types, the values are mapped to a contiguous index space
// that spans all possible permutations of the values of each of the given
// dimensional axes from which the AxisOuterProduct is constructed.

template <typename T> struct is_permutation_space : std::false_type {};

template <typename T> inline constexpr bool is_permutation_space_v = is_permutation_space<T>::value;

template <DimensionalAxis... Axes>
struct is_permutation_space<AxisOuterProduct<Axes...>> : std::true_type {};

template <typename T>
concept PermutationSpace = is_permutation_space_v<T>;

// -----------------------------------------------------------------------------
// PermutationArray class
// -----------------------------------------------------------------------------
template <PermutationSpace Space, typename ElementType> struct PermutationArray {
    using space_type   = Space;
    using element_type = ElementType;
    std::array<element_type, Space::size> elements{};

    constexpr std::optional<size_t>
    set(element_type element, auto... values) {
        auto const idx = space_type::index_of_values(values...);
        if (idx.has_value()) {
            elements[idx.value()] = element;
        }
        return idx;
    }

    constexpr std::optional<element_type>
    at(auto... values) const {
        auto const idx = space_type::index_of_values(values...);
        if (!idx.has_value()) {
            return std::nullopt;
        }
        return elements[idx.value()];
    }
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_PERMUTATIONARRAY_H
