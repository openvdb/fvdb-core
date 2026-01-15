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

// -----------------------------------------------------------------------------
// Sizes and Point
// -----------------------------------------------------------------------------

template <size_t... Is> using Sizes = std::index_sequence<Is...>;

template <size_t... Is> using Point = std::index_sequence<Is...>;

// -----------------------------------------------------------------------------
// IndexSpace Concept
// -----------------------------------------------------------------------------
//
// IndexSpace: a type representing an n-dimensional index space (e.g., Sizes<...>).
// Corresponds to std::index_sequence specializations.
//
// This concept enables generic code working with multidimensional index spaces.
//

template <typename T>
concept IndexSpace = is_index_sequence<T>::value;

template <typename T>
consteval bool
is_index_space() {
    return IndexSpace<T>;
}

// -----------------------------------------------------------------------------
// IndexPoint Concept
// -----------------------------------------------------------------------------
//
// IndexPoint: a type representing a coordinate/point within an index space
// (e.g., Point<0, 1, 2>). Structurally identical to IndexSpace but semantically
// distinct - an IndexPoint is a single coordinate, not an iterable space.
//

template <typename T>
concept IndexPoint = is_index_sequence<T>::value;

template <typename T>
consteval bool
is_index_point() {
    return IndexPoint<T>;
}

// -----------------------------------------------------------------------------
// Rank, Numel, and Shape
// -----------------------------------------------------------------------------
//
// Rank: the number of dimensions in the index space
// Numel: the total number of elements in the index space
// Shape: the shape of the index space
//

template <typename T> struct Rank;

template <size_t... Is> struct Rank<Sizes<Is...>> {
    static consteval size_t
    value() {
        return sizeof...(Is);
    }
};

template <typename T>
consteval size_t
Rank_v() {
    return Rank<T>::value();
}

template <typename T> struct Numel;

template <size_t... Is> struct Numel<Sizes<Is...>> {
    static consteval size_t
    value() {
        return (Is * ... * 1);
    }
};

// Explicit specialization: empty Sizes has 0 elements
template <> struct Numel<Sizes<>> {
    static consteval size_t
    value() {
        return 0;
    }
};

template <typename T>
consteval size_t
Numel_v() {
    return Numel<T>::value();
}

// -----------------------------------------------------------------------------
// IndexSpace Concept Refinements
// -----------------------------------------------------------------------------
//
// TensorIndexSpace: an index space with more than 0 dimensions
// ScalarIndexSpace: an index space with 0 dimensions
// NonEmptyIndexSpace: an index space with at least 1 element
//

template <typename T>
concept TensorIndexSpace = IndexSpace<T> && Rank_v<T>() > 0;

template <typename T>
concept ScalarIndexSpace = IndexSpace<T> && Rank_v<T>() == 0;

template <typename T>
concept NonEmptyIndexSpace = IndexSpace<T> && Rank_v<T>() > 0 && Numel_v<T>() > 0;

template <typename S, typename T>
concept SameRank = Rank_v<S>() == Rank_v<T>();

// =============================================================================
// Concept Test Helpers (consteval functions to avoid nvcc instantiation issues)
// =============================================================================

template <typename T>
consteval bool
is_tensor_index_space() {
    return TensorIndexSpace<T>;
}

template <typename T>
consteval bool
is_scalar_index_space() {
    return ScalarIndexSpace<T>;
}

template <typename T>
consteval bool
is_non_empty_index_space() {
    return NonEmptyIndexSpace<T>;
}

template <typename S, typename T>
consteval bool
is_same_rank() {
    return SameRank<S, T>;
}

// -----------------------------------------------------------------------------
// PointInBounds - checks if an IndexPoint lies within an IndexSpace
// -----------------------------------------------------------------------------

template <typename Space, typename Pt> struct point_in_bounds {
    static consteval bool
    value() {
        return false;
    }
};

template <> struct point_in_bounds<Sizes<>, Point<>> {
    static consteval bool
    value() {
        return true;
    }
};

template <size_t S0, size_t... TailSs, size_t I0, size_t... TailIs>
struct point_in_bounds<Sizes<S0, TailSs...>, Point<I0, TailIs...>> {
    static consteval bool
    value() {
        return (I0 < S0) && point_in_bounds<Sizes<TailSs...>, Point<TailIs...>>::value();
    }
};

template <typename Space, typename Pt>
consteval bool
point_in_bounds_v() {
    return point_in_bounds<Space, Pt>::value();
}

template <typename Space, typename Pt>
concept PointInBounds = IndexSpace<Space> && IndexPoint<Pt> && Rank_v<Space>() == Rank_v<Pt>() &&
                        point_in_bounds_v<Space, Pt>();

template <typename Space, typename Pt>
consteval bool
is_point_in_bounds() {
    return PointInBounds<Space, Pt>;
}

// -----------------------------------------------------------------------------
// Shape: type trait to extract the shape type from a space
// -----------------------------------------------------------------------------

template <typename T> struct Shape;

template <IndexSpace Space> struct Shape<Space> {
    using type = Space;
};

template <typename T> using Shape_t = typename Shape<T>::type;

// -----------------------------------------------------------------------------
// Prepend type trait for adding a new size as the first dimension of a space
// -----------------------------------------------------------------------------

template <typename T, size_t V> struct Prepend;

template <size_t... Is, size_t V> struct Prepend<Sizes<Is...>, V> {
    using type = Sizes<V, Is...>;
};

template <typename T, size_t V> using Prepend_t = typename Prepend<T, V>::type;

// -----------------------------------------------------------------------------
// PointFromLinearIndex: Compute a coordinate (Point<...>) from a linear index
// -----------------------------------------------------------------------------

template <NonEmptyIndexSpace Space, size_t linearIndex> struct PointFromLinearIndex;

template <size_t I, size_t linearIndex>
    requires NonEmptyIndexSpace<Sizes<I>>
struct PointFromLinearIndex<Sizes<I>, linearIndex> {
    // happily will use indices out of bounds.
    using type = Point<linearIndex>;
};

template <size_t I0, size_t... TailIs, size_t linearIndex>
    requires NonEmptyIndexSpace<Sizes<I0, TailIs...>> && NonEmptyIndexSpace<Sizes<TailIs...>>
struct PointFromLinearIndex<Sizes<I0, TailIs...>, linearIndex> {
    static consteval size_t
    tailNumel() {
        return Numel_v<Sizes<TailIs...>>();
    }

    using type =
        Prepend_t<typename PointFromLinearIndex<Sizes<TailIs...>, linearIndex % tailNumel()>::type,
                  linearIndex / tailNumel()>;
};

template <NonEmptyIndexSpace Space, size_t linearIndex>
using PointFromLinearIndex_t = typename PointFromLinearIndex<Space, linearIndex>::type;

// -----------------------------------------------------------------------------
// LinearIndexFromPoint: Compute a linear index from a coordinate (Point<...>)
// -----------------------------------------------------------------------------

template <NonEmptyIndexSpace Space, IndexPoint Pt> struct LinearIndexFromPoint;

template <size_t S, size_t I>
    requires NonEmptyIndexSpace<Sizes<S>>
struct LinearIndexFromPoint<Sizes<S>, Point<I>> {
    static consteval size_t
    value() {
        return I;
    }
};

template <size_t S0, size_t... TailSs, size_t I0, size_t... TailIs>
    requires NonEmptyIndexSpace<Sizes<S0, TailSs...>> && NonEmptyIndexSpace<Sizes<TailSs...>>
struct LinearIndexFromPoint<Sizes<S0, TailSs...>, Point<I0, TailIs...>> {
    static consteval size_t
    tailNumel() {
        return Numel_v<Sizes<TailSs...>>();
    }

    static consteval size_t
    value() {
        return I0 * tailNumel() + LinearIndexFromPoint<Sizes<TailSs...>, Point<TailIs...>>::value();
    }
};

template <NonEmptyIndexSpace Space, IndexPoint Pt>
consteval size_t
LinearIndexFromPoint_v() {
    return LinearIndexFromPoint<Space, Pt>::value();
}

// -----------------------------------------------------------------------------
// Visitation Utilities for Index Spaces
// -----------------------------------------------------------------------------
//
// Index spaces are traversed in row-major order: the last dimension is
// the fastest-changing. You can visit all coordinates in an index space
// by providing a visitor (callable) to visit_index_space.
//
// Example:
//     using MySpace = Sizes<2, 3>; // 2x3 space
//     visit_index_space([](auto coord) {
//         // coord is a Point<I0, I1> for each valid coordinate pair
//     }, MySpace{});
//
// You can also visit multiple spaces in sequence using visit_index_spaces.
// Each space is visited in order, calling the visitor for every coordinate
// in that space.
//
// Example:
//     visit_index_spaces(visitor, MySpace1{}, MySpace2{});
//
// Note: For each coordinate, the visitor is invoked with a default-constructed
// Point<...> object with constexpr values for that coordinate.
// -----------------------------------------------------------------------------

template <NonEmptyIndexSpace Space, typename LinearPointSeq> struct IndexSpaceVisitHelper;

template <NonEmptyIndexSpace Space, size_t... linearIndices>
struct IndexSpaceVisitHelper<Space, std::index_sequence<linearIndices...>> {
    template <typename Visitor>
    static void
    visit(Visitor &&visitor) {
        // Note: don't forward visitor in fold - it's invoked multiple times
        (std::invoke(visitor, PointFromLinearIndex_t<Space, linearIndices>{}), ...);
    }
};

template <typename Visitor, NonEmptyIndexSpace Space>
void
visit_index_space(Visitor &&visitor, Space) {
    IndexSpaceVisitHelper<Space, std::make_index_sequence<Numel_v<Space>()>>::visit(
        std::forward<Visitor>(visitor));
}

template <typename Visitor, NonEmptyIndexSpace... Spaces>
void
visit_index_spaces(Visitor &&visitor, Spaces... spaces) {
    // Note: don't forward visitor in fold - it's used for each space
    (visit_index_space(visitor, spaces), ...);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_INDEXSPACE_H
