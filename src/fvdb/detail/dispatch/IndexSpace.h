// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_INDEXSPACE_H
#define FVDB_DETAIL_DISPATCH_INDEXSPACE_H

#include "fvdb/detail/dispatch/Traits.h"

#include <nanovdb/util/Util.h>

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

    using Coord = std::array<size_t, rank>;

    // Returns the shape as an array, computed from function-local storage.
    // This avoids static data member linkage issues in device code.
    __hostdev__ static constexpr Coord
    get_shape() {
        return Coord{Is...};
    }

    // Returns the strides as an array, computed on-demand.
    // Row-major strides: stride[i] = product of shape[i+1..rank-1]
    __hostdev__ static constexpr Coord
    get_strides() {
        constexpr size_t dims[] = {Is...};
        Coord result{};
        for (size_t i = 0; i < rank; ++i) {
            size_t stride = 1;
            for (size_t j = i + 1; j < rank; ++j) {
                stride *= dims[j];
            }
            result[i] = stride;
        }
        return result;
    }

    // This will happily compute something for a coord that is out of bounds.
    __hostdev__ static constexpr size_t
    linear_index(Coord const &coord) {
        constexpr auto strides = get_strides();
        size_t result          = 0;
        for (size_t i = 0; i < rank; ++i) {
            result += coord[i] * strides[i];
        }
        return result;
    }

    // This will happily compute something for a linear index that is out of bounds.
    __hostdev__ static constexpr Coord
    coord(size_t linear_idx) {
        constexpr auto shape   = get_shape();
        constexpr auto strides = get_strides();
        Coord result{};
        for (size_t i = 0; i < rank; ++i) {
            result[i] = (linear_idx / strides[i]) % shape[i];
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
