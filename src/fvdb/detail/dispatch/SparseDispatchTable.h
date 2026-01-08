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

namespace fvdb {
namespace dispatch {

// A "Dispatch Axis" is a traits type which takes a set of values - they could
// be integers, enums, and a set of types corresponding to those values, and then
// it defines a linear order for them. This will be used to index into a lookup table.
// where the value is given at runtime, and is used to look up a jump table entry.
// The axis needs to answer several questions:
//
// SIZE: (these should be the same, and static asserted as such)
// (compile time) what are the number of values in the axis?
// (compile time) what are the number of types in the axis?
//
// MEMBERSHIP:
// (run time) for ANY enum/int/value V, is it in the set of values spanned by the axis?
// (compile time) for ANY type T, is it in the set of types spanned by the axis?
//
// INDEXING (association):
// (run time) linear index of a value V in the axis, error to ask for non-spanned value
// (compile time) linear index of a type T in the axis, error to ask for a non-spanned type
//
// REVERSE INDEXING (association):
// (run time) value V from a linear index
// (compile time) type T from a linear index

// There should be a generic dispatch axis where the list of types and values is
// provided as template arguments, but then we should also be able to make specific
// dispatch axes for things like torch dtype, torch device, where all that is required
// is the list of C++ types.

enum class DummyEnum { Mup = 18, Spod = 2, Bom = 44 };

using DummyMupTag  = Tag<DummyEnum::Mup>;
using DummySpodTag = Tag<DummyEnum::Spod>;
using DummyBomTag  = Tag<DummyEnum::Bom>;

// -----------------------------------------------------------------------------
// DispatchAxis: Generic Traits for Value-Type Associations
// -----------------------------------------------------------------------------
// Associates a set of runtime values (enums/ints) with corresponding types,
// providing compile-time and runtime lookup in both directions.

// A single value-type pair: directly encodes the association
template <auto V, typename T> struct ValueTypePair {
    static constexpr auto value = V;
    using type                  = T;
};

namespace detail {

// Helper: Get the type at index I from a pack of Pairs
template <size_t I, typename... Pairs> struct TypeAtIndexImpl;

template <typename First, typename... Rest> struct TypeAtIndexImpl<0, First, Rest...> {
    using type = typename First::type;
};

template <size_t I, typename First, typename... Rest> struct TypeAtIndexImpl<I, First, Rest...> {
    using type = typename TypeAtIndexImpl<I - 1, Rest...>::type;
};

// Helper: Check if type T exists in the pack of Pairs
template <typename T, typename... Pairs> struct ContainsTypeImpl : std::false_type {};

template <typename T, typename First, typename... Rest>
struct ContainsTypeImpl<T, First, Rest...>
    : std::bool_constant<std::is_same_v<T, typename First::type> ||
                         ContainsTypeImpl<T, Rest...>::value> {};

// Helper: Get the index of type T in the pack of Pairs
// Uses partial specialization to properly short-circuit template instantiation
template <typename T, typename... Pairs> struct IndexOfTypeImpl;

// Base case: T not found in list - provide a clear compile-time error
template <typename T> struct IndexOfTypeImpl<T> {
    static_assert(sizeof(T) == 0, "Type not found in DispatchAxis");
    static constexpr size_t value = size_t(-1); // Unreachable
};

// Match case: T matches the first pair's type
template <typename T, auto V, typename... Rest>
struct IndexOfTypeImpl<T, ValueTypePair<V, T>, Rest...> {
    static constexpr size_t value = 0;
};

// Recursive case: T doesn't match, continue searching
template <typename T, typename First, typename... Rest> struct IndexOfTypeImpl<T, First, Rest...> {
    static constexpr size_t value = 1 + IndexOfTypeImpl<T, Rest...>::value;
};

} // namespace detail

// The main DispatchAxis template
// Usage: DispatchAxis<ValueTypePair<EnumVal1, Type1>, ValueTypePair<EnumVal2, Type2>, ...>
template <typename... Pairs> struct DispatchAxis {
    // -------------------------------------------------------------------------
    // SIZE (compile-time)
    // -------------------------------------------------------------------------
    // Number of values/types in the axis (inherently equal since each Pair has one of each)
    static constexpr size_t size = sizeof...(Pairs);

    // -------------------------------------------------------------------------
    // MEMBERSHIP (compile-time for types)
    // -------------------------------------------------------------------------
    // Check if type T is in the set of types spanned by this axis
    template <typename T>
    static constexpr bool contains_type = detail::ContainsTypeImpl<T, Pairs...>::value;

    // -------------------------------------------------------------------------
    // MEMBERSHIP (runtime for values)
    // -------------------------------------------------------------------------
    // Check if value v is in the set of values spanned by this axis
    template <typename ValueType>
    static bool
    contains_value(ValueType v) {
        return ((Pairs::value == v) || ...);
    }

    // -------------------------------------------------------------------------
    // INDEXING (compile-time type to index)
    // -------------------------------------------------------------------------
    // Get the linear index of type T in the axis
    // Error if T is not in the axis (will fail to compile)
    template <typename T>
    static constexpr size_t index_of_type = detail::IndexOfTypeImpl<T, Pairs...>::value;

    // -------------------------------------------------------------------------
    // INDEXING (runtime value to index)
    // -------------------------------------------------------------------------
    // Get the linear index of value v in the axis
    // Throws if v is not in the axis (compile error if called in constexpr context)
    template <typename ValueType>
    static constexpr size_t
    index_of_value(ValueType v) {
        // Build a constexpr array of all values for lookup
        using CommonType              = std::common_type_t<decltype(Pairs::value)...>;
        constexpr CommonType values[] = {static_cast<CommonType>(Pairs::value)...};

        for (size_t i = 0; i < size; ++i) {
            if (values[i] == static_cast<CommonType>(v))
                return i;
        }
        throw std::runtime_error("[DispatchAxis] Value not in axis");
    }

    // -------------------------------------------------------------------------
    // REVERSE INDEXING (compile-time index to type)
    // -------------------------------------------------------------------------
    // Get the type at linear index I
    // Error if I >= size (will fail to compile)
    template <size_t I> using type_at_index = typename detail::TypeAtIndexImpl<I, Pairs...>::type;

    // -------------------------------------------------------------------------
    // REVERSE INDEXING (runtime index to value)
    // -------------------------------------------------------------------------
    // Get the value at linear index idx
    // Behavior undefined if idx >= size (for performance; caller must check)
    static constexpr auto
    value_at_index(size_t idx) {
        using CommonType              = std::common_type_t<decltype(Pairs::value)...>;
        constexpr CommonType values[] = {static_cast<CommonType>(Pairs::value)...};
        return values[idx];
    }
};

// -----------------------------------------------------------------------------
// Example: DummyAxis using DummyEnum and Dummy types
// -----------------------------------------------------------------------------
using DummyAxis = DispatchAxis<ValueTypePair<DummyEnum::Mup, DummyMupTag>,
                               ValueTypePair<DummyEnum::Spod, DummySpodTag>,
                               ValueTypePair<DummyEnum::Bom, DummyBomTag>>;

// -----------------------------------------------------------------------------
// IntDispatchAxis: DispatchAxis from a list of integer values
// -----------------------------------------------------------------------------
// For dispatch over compile-time integer values (e.g., supported channel counts,
// kernel sizes, etc.). Each integer becomes a distinct type via IntegralTag.

// Create a DispatchAxis directly from integer values
// Automatically generates IntegralTag<V> types for each value
// Usage: IntDispatchAxis<1, 3, 4, 8>
template <auto... Vs> using IntDispatchAxis = DispatchAxis<ValueTypePair<Vs, IntegralTag<Vs>>...>;

// Example: An axis for supported channel counts
using ExampleChannelAxis = IntDispatchAxis<1, 3, 4, 8, 16, 32>;

// -----------------------------------------------------------------------------
// AxisProduct: Outer product of multiple DispatchAxis types
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

    // Helper for compile-time index computation
    template <typename... Types> struct IndexOfTypesImpl {
        static_assert(sizeof...(Types) == num_axes, "Must provide one type per axis");

        template <size_t... Is>
        static constexpr size_t
        compute(std::index_sequence<Is...>) {
            return ((axis_at<Is>::template index_of_type<
                         std::tuple_element_t<Is, std::tuple<Types...>>> *
                     strides[Is]) +
                    ... + 0);
        }

        static constexpr size_t value = compute(std::make_index_sequence<num_axes>{});
    };

    // Helper for index computation (constexpr-friendly)
    template <size_t... Is, typename ValueTuple>
    static constexpr size_t
    index_of_values_impl(std::index_sequence<Is...>, const ValueTuple &values) {
        return ((axis_at<Is>::index_of_value(std::get<Is>(values)) * strides[Is]) + ... + 0);
    }

  public:
    // -------------------------------------------------------------------------
    // Compile-time: linear index from a pack of types (one per axis)
    // -------------------------------------------------------------------------
    template <typename... Types>
    static constexpr size_t index_of_types = IndexOfTypesImpl<Types...>::value;

    // -------------------------------------------------------------------------
    // Linear index from a pack of values (one per axis)
    // Can be evaluated at compile-time if all values are constexpr
    // -------------------------------------------------------------------------
    template <typename... Values>
    static constexpr size_t
    index_of_values(Values... values) {
        static_assert(sizeof...(Values) == num_axes, "Must provide one value per axis");
        return index_of_values_impl(std::make_index_sequence<num_axes>{},
                                    std::make_tuple(values...));
    }
};

// Example: Product of device, dtype, and channel axes
// Total combinations = 3 devices * 5 dtypes * 6 channels = 90
// using ExampleProductAxis =
//     AxisProduct<TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA, torch::kPrivateUse1>,
//                 TorchDtypeAxis<float, double, int32_t, int64_t, bool>,
//                 IntDispatchAxis<1, 3, 4, 8, 16, 32>>;

// Usage: (compile time or runtime)
// ExampleProductAxis::index_of_types<TorchDeviceCudaTag, float, IntegralTag<4>>
// ExampleProductAxis::index_of_values(torch::kCUDA, torch::ScalarType::Float, 4)

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

// Get the type corresponding to a value in an axis
template <typename Axis, auto V> struct ValueToType {
    static constexpr size_t idx = Axis::index_of_value(V);
    using type                  = typename Axis::template type_at_index<idx>;
};

// -----------------------------------------------------------------------------
// BindFn: Bind a constexpr lambda factory (C++20)
// -----------------------------------------------------------------------------
// Allows factories to be written as constexpr lambdas with template parameters:
//   constexpr auto my_factory = []<typename A, typename B>() { return [...]; };
// Usage: BindTypesFn<my_factory, TypeA, TypeB>

template <auto Factory, typename... Types> struct BindTypesFn {
    template <typename Table>
    static void
    apply(Table &table) {
        constexpr size_t idx = Table::Axes::template index_of_types<Types...>;
        table.table_[idx]    = Factory.template operator()<Types...>();
    }
};

// Bind using enum/integral values instead of types
// The values are converted to their corresponding types via the axes
template <auto Factory, auto... Values> struct BindValuesFn {
    template <typename Table, size_t... Is>
    static void
    apply_impl(Table &table, std::index_sequence<Is...>) {
        using Axes           = typename Table::Axes;
        constexpr size_t idx = Axes::index_of_values(Values...);
        // Convert each value to its corresponding type from the appropriate axis
        table.table_[idx] =
            Factory.template
            operator()<typename ValueToType<typename Axes::template axis_at<Is>,
                                            NthValue<Is, Values...>::value>::type...>();
    }

    template <typename Table>
    static void
    apply(Table &table) {
        apply_impl(table, std::make_index_sequence<sizeof...(Values)>{});
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
