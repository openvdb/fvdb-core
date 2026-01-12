// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_INDEXSPACE_H
#define FVDB_DETAIL_DISPATCH_INDEXSPACE_H

#include "fvdb/detail/dispatch/Traits.h"

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

namespace fvdb {
namespace dispatch {

template <size_t... Is> struct IndexSpace {
    static_assert(sizeof...(Is) > 0, "Space only defined for rank > 0");
    static_assert(((Is > 0) && ... && true), "dimensions must be greater than 0");

    static constexpr size_t rank  = sizeof...(Is);
    static constexpr size_t numel = (Is * ... * 1);

    using shape_seq   = std::index_sequence<Is...>;
    using strides_seq = strides_helper_t<shape_seq>;

    static constexpr auto shape   = array_from_indices<shape_seq>::value;
    static constexpr auto strides = array_from_indices<strides_seq>::value;

    using Coord = std::array<size_t, rank>;

    // This will happily compute something for a coord that is out of bounds.
    static constexpr size_t
    linear_index(Coord const &coord) {
        size_t result = 0;
        for (size_t i = 0; i < rank; ++i) {
            result += coord[i] * strides[i];
        }
        return result;
    }

    // This will happily compute something for a linear index that is out of bounds.
    static constexpr Coord
    coord(size_t linear_index) {
        Coord result{};
        for (size_t i = 0; i < rank; ++i) {
            result[i] = (linear_index / strides[i]) % shape[i];
        }
        return result;
    }

    template <size_t LinearIndex> struct LinearIndexCoord {
        static constexpr auto coord_value = IndexSpace::coord(LinearIndex);

        template <size_t... Js>
        static auto
            _make_type(std::index_sequence<Js...>) -> std::index_sequence<coord_value[Js]...>;

        using type = decltype(_make_type(std::make_index_sequence<rank>{}));
    };

    template <typename T> struct VisitorHelper;

    template <size_t... LinearIndices> struct VisitorHelper<std::index_sequence<LinearIndices...>> {
        template <typename Visitor>
        static void
        visit(Visitor &&visitor) {
            (std::invoke(std::forward<Visitor>(visitor),
                         typename LinearIndexCoord<LinearIndices>::type{}),
             ...);
        }
    };

    template <typename Visitor>
    static void
    visit(Visitor &&visitor) {
        VisitorHelper<std::make_index_sequence<numel>>::visit(std::forward<Visitor>(visitor));
    }
};

template <typename Visitor, typename... Spaces>
void
visit_spaces(Visitor &&visitor, Spaces... spaces) {
    (spaces.visit(std::forward<Visitor>(visitor)), ...);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_INDEXSPACE_H
