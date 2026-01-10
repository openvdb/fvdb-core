// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
#define FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H

#include "fvdb/detail/dispatch/AxisOuterProduct.h"

#include <array>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Traits
// =============================================================================

// -----------------------------------------------------------------------------
// is_instantiator_for trait
// -----------------------------------------------------------------------------
// Checks if an Instantiator template, when instantiated with Values..., has a
// static get() method that returns FunctionPtrT.
//
// Usage: is_instantiator_for_v<MyInstantiator, FunctionPtr, val1, val2, ...>

template <template <auto...> typename Instantiator, typename FunctionPtrT, auto... Values>
    struct is_instantiator_for : std::bool_constant < requires {
    { Instantiator<Values...>::get() } -> std::same_as<FunctionPtrT>;
} > {};

template <template <auto...> typename Instantiator, typename FunctionPtrT, auto... Values>
inline constexpr bool is_instantiator_for_v =
    is_instantiator_for<Instantiator, FunctionPtrT, Values...>::value;

// =============================================================================
// Concepts
// =============================================================================

// -----------------------------------------------------------------------------
// DispatchTableConcept: Checks if a type is a valid dispatch table
// -----------------------------------------------------------------------------

template <typename T>
concept DispatchTableConcept = requires(T &t, std::size_t idx) {
    typename T::FunctionPtr;
    typename T::Axes;
    { t.table_[idx] } -> std::convertible_to<typename T::FunctionPtr &>;
} && AxisOuterProductConcept<typename T::Axes>;

// -----------------------------------------------------------------------------
// BindingSpecForConcept: Checks if a type is a valid binding specification for a table
// -----------------------------------------------------------------------------

template <typename Spec, typename Table>
concept BindingSpecForConcept = DispatchTableConcept<Table> && requires(Table &t) {
    { Spec::apply(t) } -> std::same_as<void>;
};

// =============================================================================
// DispatchTable
// =============================================================================
// A simple dispatch table that maps axis value combinations to function pointers.
// This is just data + lookup; construction is handled by free functions.

template <typename FunctionSignature, typename AxesT> struct DispatchTable;

template <typename ReturnType, typename... Args, typename AxesT>
struct DispatchTable<ReturnType(Args...), AxesT> {
    static_assert(is_axis_outer_product_v<AxesT>, "DispatchTable Axes must be an AxisOuterProduct");

    using Axes        = AxesT;
    using FunctionPtr = ReturnType (*)(Args...);

    std::array<FunctionPtr, Axes::size> table_{};

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    template <typename... Values>
    FunctionPtr
    find_required(Values... values) const {
        static_assert(sizeof...(Values) == Axes::num_axes,
                      "Must provide exactly one value per axis");
        auto const idx = Axes::index_of_values(values...);
        static_assert(idx.has_value(), "Values must be valid members of their corresponding axes");
        auto ret = table_[*idx];
        if (ret == nullptr) {
            throw std::runtime_error("No implementation bound for this combination of axis values");
        }
        return ret;
    }

    template <typename Fallback, typename... Values>
    FunctionPtr
    find_or(Fallback &&fallback, Values... values) const {
        static_assert(sizeof...(Values) == Axes::num_axes,
                      "Must provide exactly one value per axis");
        auto const idx = Axes::index_of_values(values...);
        if (!idx.has_value() || table_[*idx] == nullptr) {
            return std::invoke(std::forward<Fallback>(fallback), values...);
        }
        auto ret = table_[*idx];
        return ret;
    }
};

// =============================================================================
// Binding Types
// =============================================================================
// These types describe bindings declaratively. They have a static apply()
// method that performs the actual binding when called.
//
// IMPORTANT: Bindings are checked at apply() time to ensure:
//   1. The subspace is valid for the table's axes (prevents accidental full-space binding)
//   2. The instantiator returns the correct function pointer type

// -----------------------------------------------------------------------------
// Binding: Binds a single value combination
// -----------------------------------------------------------------------------
// Instantiator must be a template with signature:
//   template <auto... Values> struct Inst { static FunctionPtr get(); };

template <template <auto...> typename Instantiator, auto... Values> struct Binding {
    template <typename Table>
        requires DispatchTableConcept<Table>
    static void
    apply(Table &table) {
        // Check that the instantiator returns the correct type
        static_assert(is_instantiator_for_v<Instantiator, typename Table::FunctionPtr, Values...>,
                      "Instantiator::get() must return Table::FunctionPtr");

        // Check that all values are valid for the table's axes
        static_assert(Table::Axes::index_of_values(Values...).has_value(),
                      "Binding values must be valid members of the table's axes");

        constexpr auto idx = Table::Axes::index_of_values(Values...).value();
        table.table_[idx]  = Instantiator<Values...>::get();
    }
};

// -----------------------------------------------------------------------------
// SubspaceBinding: Binds all combinations in a subspace
// -----------------------------------------------------------------------------
// Subspace must be an AxisOuterProduct whose axes are subsets of the table's axes.
// This is checked at apply() time to prevent accidental full-space instantiation.

template <template <auto...> typename Instantiator, typename Subspace> struct SubspaceBinding {
    static_assert(is_axis_outer_product_v<Subspace>,
                  "SubspaceBinding requires Subspace to be an AxisOuterProduct");

  private:
    // Action for for_each_permutation: binds each instantiated value combination
    template <typename InstType, auto... Vals> struct BindAction {
        template <typename Table>
        static void
        apply(Table &table) {
            // These are already validated by the outer apply(), but we include
            // them here for defense-in-depth and clearer error messages.
            static_assert(Table::Axes::index_of_values(Vals...).has_value(),
                          "Subspace values must be valid members of the table's axes");

            constexpr auto idx = Table::Axes::index_of_values(Vals...).value();
            table.table_[idx]  = InstType::get();
        }
    };

  public:
    template <typename Table>
        requires DispatchTableConcept<Table>
    static void
    apply(Table &table) {
        // CRITICAL: Verify the subspace is actually a subspace of the table's axes.
        // This prevents accidentally binding the full permutation space when you
        // meant to bind a subset.
        static_assert(is_subspace_of_v<Subspace, typename Table::Axes>,
                      "SubspaceBinding: Subspace must be a subspace of the table's axes. "
                      "Each axis of the Subspace must be a subset of the corresponding "
                      "axis in the table. This check prevents accidental full-space "
                      "instantiation.");

        for_each_permutation<Subspace, Instantiator, BindAction>::apply(table);
    }
};

// -----------------------------------------------------------------------------
// FillBinding: Fills all combinations in a subspace with a constant value
// -----------------------------------------------------------------------------
// No instantiation per permutation - just stores the same pointer everywhere.
// Useful for filling unsupported combinations with an error handler.

template <typename Subspace, auto FnPtr> struct FillBinding {
    static_assert(is_axis_outer_product_v<Subspace>,
                  "FillBinding requires Subspace to be an AxisOuterProduct");

  private:
    template <auto... Vals> struct FillAction {
        template <typename Table>
        static void
        apply(Table &table) {
            static_assert(Table::Axes::index_of_values(Vals...).has_value(),
                          "Subspace values must be valid members of the table's axes");

            constexpr auto idx = Table::Axes::index_of_values(Vals...).value();
            table.table_[idx]  = FnPtr;
        }
    };

  public:
    template <typename Table>
        requires DispatchTableConcept<Table>
    static void
    apply(Table &table) {
        // Verify the subspace is valid for this table
        static_assert(is_subspace_of_v<Subspace, typename Table::Axes>,
                      "FillBinding: Subspace must be a subspace of the table's axes");

        // Verify the function pointer type matches
        static_assert(std::is_same_v<decltype(FnPtr), typename Table::FunctionPtr>,
                      "FillBinding: FnPtr must be of type Table::FunctionPtr");

        for_each_values<Subspace, FillAction>::apply(table);
    }
};

// -----------------------------------------------------------------------------
// BindingList: Aggregates multiple bindings
// -----------------------------------------------------------------------------

template <typename... BindingTs> struct BindingList {
    template <typename Table>
        requires DispatchTableConcept<Table>
    static void
    apply(Table &table) {
        (BindingTs::apply(table), ...);
    }
};

// -----------------------------------------------------------------------------
// WithGet: Wraps an Invoker (with static invoke()) to add the get() method
// -----------------------------------------------------------------------------
// Use this to avoid writing boilerplate get() methods on instantiators.
// The Invoker must have: static FunctionPtrT invoke(Args...);
//
// Usage:
//   template <auto... Values> struct MyInvoker {
//       static RetT invoke(Args...) { ... }
//   };
//   using MyInstantiator = WithGet<MyInvoker, FunctionPtrT>;
//   // Then use MyInstantiator::Bind in Binding/SubspaceBinding

template <template <auto...> typename Invoker, typename FunctionPtrT> struct WithGet {
    template <auto... Values> struct Bind {
        static FunctionPtrT
        get() {
            return &Invoker<Values...>::invoke;
        }
    };
};

// =============================================================================
// Free Functions for Table Construction
// =============================================================================

// -----------------------------------------------------------------------------
// build_table: Construct a dispatch table from a binding specification
// -----------------------------------------------------------------------------

template <typename Table, typename BindingSpec>
    requires DispatchTableConcept<Table> && BindingSpecForConcept<BindingSpec, Table>
Table
build_table() {
    Table table{};
    BindingSpec::apply(table);
    return table;
}

// -----------------------------------------------------------------------------
// with_bindings: Derive a new table by adding bindings to an existing one
// -----------------------------------------------------------------------------

template <typename BindingSpec, typename Table>
    requires DispatchTableConcept<Table> && BindingSpecForConcept<BindingSpec, Table>
Table
with_bindings(Table const &original) {
    Table result = original;
    BindingSpec::apply(result);
    return result;
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
