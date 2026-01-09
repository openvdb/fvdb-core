// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
#define FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H

#include "fvdb/detail/dispatch/Tag.h"

#include <array>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <concepts>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// ValueOrdering
// -----------------------------------------------------------------------------

template <auto... Values> struct ValueOrdering {
    using value_type                     = std::common_type_t<decltype(Values)...>;
    static constexpr value_type values[] = {Values...};

    static_assert((std::is_same_v<decltype(Values), std::common_type_t<decltype(Values)...>> &&
                   ...),
                  "All axis values must have the same type");

    // -------------------------------------------------------------------------
    // SIZE
    // -------------------------------------------------------------------------
    // Number of values in the axis
    static constexpr size_t size = sizeof...(Values);

    // -------------------------------------------------------------------------
    // MEMBERSHIP
    // -------------------------------------------------------------------------
    // Check if value v is in the set of values spanned by this axis (constexpr)
    static constexpr bool
    contains_value(value_type v) {
        return ((Values == v) || ...);
    }

    // -------------------------------------------------------------------------
    // INDEXING (runtime value to index)
    // -------------------------------------------------------------------------
    // Get the linear index of value v in the axis
    // Throws if v is not in the axis (compile error if called in constexpr context)
    static constexpr size_t
    index_of_value(value_type v) {
        for (size_t i = 0; i < size; ++i) {
            if (values[i] == v) {
                return i;
            }
        }
        throw std::runtime_error("[ValueOrdering] Value not in ordering");
    }

    // -------------------------------------------------------------------------
    // REVERSE INDEXING (runtime index to value)
    // -------------------------------------------------------------------------
    // Get the value at linear index idx
    // Behavior undefined if idx >= size (for performance; caller must check)
    static constexpr value_type
    value_at_index(size_t idx) {
        return values[idx];
    }
};

// -----------------------------------------------------------------------------
// ValueOrdering from a list of integer values
// -----------------------------------------------------------------------------
// For dispatch over compile-time integer values (e.g., supported channel counts,
// kernel sizes, etc.).

// Create a ValueOrdering directly from integer values
// Usage: IntOrdering<1, 3, 4, 8>
template <int... Values> using IntOrdering = ValueOrdering<Values...>;

// -----------------------------------------------------------------------------
// AxisProduct: Outer product of multiple ValueOrdering types
// -----------------------------------------------------------------------------
// Forms the cartesian product of all axis elements, providing linear indexing
// into the combined space from either runtime values or compile-time types.

template <typename... Axes> struct AxisProduct {
    static constexpr size_t num_axes = sizeof...(Axes);
    static constexpr size_t size     = (Axes::size * ... * 1);

    // Access the Nth axis type
    template <size_t I> using axis_at = std::tuple_element_t<I, std::tuple<Axes...>>;

  private:
    // Compute strides at compile time (row-major: last axis has stride 1)
    static constexpr auto
    compute_strides() {
        std::array<size_t, num_axes> result{};
        if constexpr (num_axes > 0) {
            constexpr size_t sizes[] = {Axes::size...};
            size_t stride            = 1;
            for (size_t i = num_axes; i > 0; --i) {
                result[i - 1] = stride;
                stride *= sizes[i - 1];
            }
        }
        return result;
    }

    static constexpr std::array<size_t, num_axes> strides = compute_strides();

    // Helper for index computation (constexpr-friendly)
    template <size_t... Is, typename ValueTuple>
    static constexpr size_t
    index_of_values_impl(std::index_sequence<Is...>, ValueTuple const &values) {
        return ((axis_at<Is>::index_of_value(std::get<Is>(values)) * strides[Is]) + ... + 0);
    }

    // Helper for checking value types match axis value_types (zips two packs via index sequence)
    template <typename ValueTypesTuple, size_t... Is>
    static constexpr bool
    check_value_types_impl(std::index_sequence<Is...>) {
        return (std::is_same_v<std::tuple_element_t<Is, ValueTypesTuple>,
                               typename axis_at<Is>::value_type> &&
                ...);
    }

  public:
    // -------------------------------------------------------------------------
    // Linear index from a pack of values (one per axis)
    // Can be evaluated at compile-time if all values are constexpr
    // -------------------------------------------------------------------------
    template <typename... ValueTypes>
    static constexpr size_t
    index_of_values(ValueTypes... values) {
        static_assert(sizeof...(ValueTypes) == num_axes, "Must provide one value per axis");
        static_assert(check_value_types_impl<std::tuple<std::decay_t<ValueTypes>...>>(
                          std::make_index_sequence<num_axes>{}),
                      "Provided value types must match the axis value_type at each axis position.");

        return index_of_values_impl(std::make_index_sequence<num_axes>{},
                                    std::make_tuple(values...));
    }
};

// -----------------------------------------------------------------------------
// SparseDispatchTable class
// -----------------------------------------------------------------------------
template <typename FunctionSignature, typename AxesT> struct SparseDispatchTable;

template <typename ReturnType, typename... Args, typename AxesT>
struct SparseDispatchTable<ReturnType(Args...), AxesT> {
    using Axes        = AxesT;
    using FunctionPtr = ReturnType (*)(Args...);
    std::array<FunctionPtr, Axes::size> table_{};

    template <typename... Values>
    FunctionPtr
    find_required(Values... values) const {
        static_assert(sizeof...(Values) == Axes::num_axes, "Must provide one value per axis");
        auto const idx = Axes::index_of_values(values...);
        auto ret       = table_[idx];
        if (ret == nullptr) {
            throw std::runtime_error("No implementation bound for this combination of axis values");
        }
        return ret;
    }

    template <typename Fallback, typename... Values>
    FunctionPtr
    find_or(Fallback &&fallback, Values... values) const {
        static_assert(sizeof...(Values) == Axes::num_axes, "Must provide one value per axis");
        auto const idx = Axes::index_of_values(values...);
        auto ret       = table_[idx];
        if (ret == nullptr) {
            return std::invoke(std::forward<Fallback>(fallback), values...);
        }
        return ret;
    }
};

// Concept for the dispatch table
template <typename T>
concept DispatchTable = requires(T& t, std::size_t idx) {
    typename T::FunctionPtr;  // Must have FunctionPtr type alias
    typename T::Axes;         // Must have Axes type alias
    t.table_[idx];            // Must have indexable table_ member
};

// -----------------------------------------------------------------------------
// The following section requires C++20 template lambdas which nvcc doesn't
// fully support (template lambdas converted to function pointers). This
// section is compiled only by the host compiler.
// -----------------------------------------------------------------------------
#ifndef __CUDACC__

// -----------------------------------------------------------------------------
// Helpers for BindValuesFn: Convert compile-time values to types via axes
// -----------------------------------------------------------------------------

// Get the Nth value from a parameter pack of values
template <size_t I, auto... Vs> struct NthValue;

template <auto First, auto... Rest> struct NthValue<0, First, Rest...> {
    static constexpr auto value = First;
};

template <size_t I, auto First, auto... Rest> struct NthValue<I, First, Rest...> {
    static constexpr auto value = NthValue<I - 1, Rest...>::value;
};

// -----------------------------------------------------------------------------
// BindFn: Bind a constexpr lambda factory (C++20)
// -----------------------------------------------------------------------------

// Bind using enum/integral values instead of types
// The values are converted to their corresponding types via the axes
template <auto Factory, auto... Values> struct BindValuesFn {
    template <typename Table>
    static void
    apply(Table &table) {
        using Axes           = typename Table::Axes;
        constexpr auto index = Axes::index_of_values(Values...);
        table.table_[index]  = Factory.template operator()<Values...>();
    }
};

// -----------------------------------------------------------------------------
// Bindings: A composable list of binding specifications
// -----------------------------------------------------------------------------
// Aggregates multiple Bind specs. Applied via fold expression.

template <typename... Specs> struct Bindings {
    template <typename Table>
    static void
    apply(Table &table) {
        (Specs::apply(table), ...);
    }
};

// -----------------------------------------------------------------------------
// make_table: Construct a dispatch table from a binding specification
// -----------------------------------------------------------------------------
// Pure function: specification in, table out.

template <typename Signature, typename Axes, typename BindingSpec>
auto
make_table() {
    SparseDispatchTable<Signature, Axes> table{};
    BindingSpec::apply(table);
    return table;
}

#endif // !__CUDACC__

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
