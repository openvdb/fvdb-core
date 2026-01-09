// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
#define FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H

#include "fvdb/detail/dispatch/Tag.h"

#include <array>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fvdb {
namespace dispatch {

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
concept DispatchTable = requires(T &t, std::size_t idx) {
    typename T::FunctionPtr; // Must have FunctionPtr type alias
    typename T::Axes;        // Must have Axes type alias
    t.table_[idx];           // Must have indexable table_ member
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
