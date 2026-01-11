// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H
#define FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H

#include "fvdb/detail/dispatch/Values.h"

#include <array>
#include <concepts>
#include <optional>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// is_dimensional_axis trait
// -----------------------------------------------------------------------------
// A DimensionalAxis is a SameTypeUniqueValuePack - a set of unique values of
// the same type that can be indexed bidirectionally.

template <typename T> struct is_dimensional_axis : std::false_type {};

template <typename T> inline constexpr bool is_dimensional_axis_v = is_dimensional_axis<T>::value;

template <auto... values>
struct is_dimensional_axis<SameTypeUniqueValuePack<values...>> : std::true_type {};

// -----------------------------------------------------------------------------
// AxisOuterProduct
// -----------------------------------------------------------------------------
// Forms the cartesian product of multiple DimensionalAxis types, providing
// linear indexing into the combined space from runtime values.
//
// Given axes A, B, C with sizes |A|, |B|, |C|, the linear index for values
// (a, b, c) is computed as: a_idx * (|B| * |C|) + b_idx * |C| + c_idx
// (row-major order: last axis varies fastest)
//
// Returns std::nullopt if any value is not found in its corresponding axis.

template <typename... Axes> struct AxisOuterProduct {
    static_assert((is_dimensional_axis_v<Axes> && ...),
                  "AxisOuterProduct requires all Axes to be DimensionalAxis types "
                  "(i.e., SameTypeUniqueValuePack instantiations)");
    static_assert(((Axes::size > 0) && ... && true),
                  "AxisOuterProduct does not support empty axes (size must be > 0)");
    // -------------------------------------------------------------------------
    // Type aliases
    // -------------------------------------------------------------------------
    using axes_tuple_type        = std::tuple<Axes...>;
    using value_types_tuple_type = std::tuple<typename Axes::value_type...>;

    template <size_t I> using axis_at_type = std::tuple_element_t<I, axes_tuple_type>;

    template <size_t I> using value_at_type = std::tuple_element_t<I, value_types_tuple_type>;

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    static constexpr size_t num_axes = sizeof...(Axes);
    static constexpr size_t size     = (Axes::size * ... * 1);

  private:
    // -------------------------------------------------------------------------
    // Stride computation (row-major: last axis has stride 1)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Index computation helper
    // -------------------------------------------------------------------------
    // Looks up each value in its corresponding axis and computes the linear
    // index. Returns nullopt if any lookup fails.
    template <size_t... Is>
    static constexpr std::optional<size_t>
    index_of_values_impl(std::index_sequence<Is...>, value_types_tuple_type const &values) {
        // Look up each value in its axis
        std::array<std::optional<size_t>, num_axes> indices = {
            axis_at_type<Is>::index_of_value(std::get<Is>(values))...};

        // Check if any lookup failed
        for (auto const &idx: indices) {
            if (!idx.has_value()) {
                return std::nullopt;
            }
        }

        // Compute linear index from the looked-up indices
        size_t linear_index = 0;
        for (size_t i = 0; i < num_axes; ++i) {
            linear_index += indices[i].value() * strides[i];
        }
        return linear_index;
    }

    // -------------------------------------------------------------------------
    // Value type checking helper
    // -------------------------------------------------------------------------
    template <typename... ProvidedTypes, size_t... Is>
    static constexpr bool
    check_value_types_impl(std::index_sequence<Is...>) {
        return (std::is_same_v<std::decay_t<std::tuple_element_t<Is, std::tuple<ProvidedTypes...>>>,
                               value_at_type<Is>> &&
                ...);
    }

  public:
    // -------------------------------------------------------------------------
    // index_of_values
    // -------------------------------------------------------------------------
    // Returns the linear index for a set of values (one per axis).
    // Returns std::nullopt if any value is not found in its corresponding axis.
    //
    // Can be evaluated at compile-time if all values are constexpr.
    template <typename... ValueTypes>
    static constexpr std::optional<size_t>
    index_of_values(ValueTypes... values) {
        static_assert(sizeof...(ValueTypes) == num_axes, "Must provide exactly one value per axis");
        static_assert(check_value_types_impl<ValueTypes...>(std::make_index_sequence<num_axes>{}),
                      "Provided value types must match the axis value_type at each position");

        return index_of_values_impl(std::make_index_sequence<num_axes>{},
                                    value_types_tuple_type{values...});
    }

    // -------------------------------------------------------------------------
    // LINEAR INDEX DECODING
    // -------------------------------------------------------------------------
    // These functions provide the reverse of index_of_values: given a linear
    // index, decode it back to axis indices and values. This enables flat
    // iteration over the space using index_sequence<0..Size-1> instead of
    // recursive template expansion.

    // Given a linear index, compute the index within axis I.
    // For row-major order: axis_index = (linear_idx / stride[I]) % axis_size[I]
    template <size_t I>
        requires(I < num_axes)
    static constexpr size_t
    axis_index_at(size_t linear_idx) {
        return (linear_idx / strides[I]) % axis_at_type<I>::size;
    }

    // Compile-time version: compute axis index for a compile-time linear index.
    template <size_t I, size_t LinearIdx>
        requires(I < num_axes && LinearIdx < size)
    static constexpr size_t axis_index_at_v = (LinearIdx / strides[I]) % axis_at_type<I>::size;

  private:
    // Helper to decode linear index to value tuple at compile time
    template <size_t LinearIdx, size_t... Is>
    static constexpr value_types_tuple_type
    values_at_index_impl(std::index_sequence<Is...>) {
        return value_types_tuple_type{
            axis_at_type<Is>::template value_at<axis_index_at_v<Is, LinearIdx>>...};
    }

  public:
    // Decode a compile-time linear index to a tuple of axis values.
    // This is the inverse of index_of_values for valid indices.
    template <size_t LinearIdx>
        requires(LinearIdx < size)
    static constexpr value_types_tuple_type values_at_index =
        values_at_index_impl<LinearIdx>(std::make_index_sequence<num_axes>{});
};

// -----------------------------------------------------------------------------
// is_axis_outer_product trait
// -----------------------------------------------------------------------------
// Checks if a type is an AxisOuterProduct instantiation.

template <typename T> struct is_axis_outer_product : std::false_type {};

template <typename... Axes>
struct is_axis_outer_product<AxisOuterProduct<Axes...>> : std::true_type {};

template <typename T>
inline constexpr bool is_axis_outer_product_v = is_axis_outer_product<T>::value;

template <typename T>
concept AxisOuterProductConcept = is_axis_outer_product_v<T>;

// -----------------------------------------------------------------------------
// is_subspace_of trait
// -----------------------------------------------------------------------------
// Checks if one AxisOuterProduct is a subspace of another. A subspace has the
// same number of axes, and each axis is a subset of the corresponding axis in
// the full space.

template <typename SubSpace, typename FullSpace> struct is_subspace_of : std::false_type {};

// Specialization for same-length axis packs only (uses requires to prevent
// instantiation of fold expression when pack lengths differ)
template <typename... SubAxes, typename... FullAxes>
    requires(sizeof...(SubAxes) == sizeof...(FullAxes))
struct is_subspace_of<AxisOuterProduct<SubAxes...>, AxisOuterProduct<FullAxes...>>
    : std::bool_constant<(is_subset_of_v<SubAxes, FullAxes> && ...)> {};

template <typename Sub, typename Full>
inline constexpr bool is_subspace_of_v = is_subspace_of<Sub, Full>::value;

// -----------------------------------------------------------------------------
// for_each_values
// -----------------------------------------------------------------------------
// Compile-time iteration over all value combinations in an AxisOuterProduct.
// For each permutation (v0, v1, ..., vN), invokes Action<v0, v1, ..., vN>::apply(args...).
//
// Action signature: template <auto... Values> struct Action {
//     static void apply(Args...);
// };

namespace detail {

template <template <auto...> typename Action, typename... Axes> struct ForEachValuesImpl;

// Base case: invoke Action with accumulated values
template <template <auto...> typename Action> struct ForEachValuesImpl<Action> {
    template <auto... Values, typename... Args>
    static void
    apply(Args &&...args) {
        Action<Values...>::apply(std::forward<Args>(args)...);
    }
};

// Recursive case: expand current axis, recurse
template <template <auto...> typename Action, auto... AxisVals, typename... Rest>
struct ForEachValuesImpl<Action, SameTypeUniqueValuePack<AxisVals...>, Rest...> {
    template <auto... Accumulated, typename... Args>
    static void
    apply(Args &&...args) {
        (ForEachValuesImpl<Action, Rest...>::template apply<Accumulated..., AxisVals>(
             std::forward<Args>(args)...),
         ...);
    }
};

} // namespace detail

template <typename Space, template <auto...> typename Action> struct for_each_values;

template <typename... Axes, template <auto...> typename Action>
struct for_each_values<AxisOuterProduct<Axes...>, Action> {
    template <typename... Args>
    static void
    apply(Args &&...args) {
        detail::ForEachValuesImpl<Action, Axes...>::template apply<>(std::forward<Args>(args)...);
    }
};

// -----------------------------------------------------------------------------
// for_each_values_flat
// -----------------------------------------------------------------------------
// Flattened version of for_each_values that uses linear index iteration
// instead of recursive template expansion. This produces O(Space::size)
// template instantiations with constant depth, rather than the O(product of
// axis sizes × recursion depth) instantiations of the recursive version.
//
// The implementation decodes each linear index back to axis values at compile
// time, then invokes Action<values...>::apply(args...).

namespace detail {

// Helper to invoke Action with values decoded from a linear index.
// Uses the axis_index_at_v and value_at machinery to extract compile-time values.
template <typename Space,
          size_t LinearIdx,
          template <auto...>
          typename Action,
          typename AxisIndexSeq>
struct InvokeActionAtIndex;

template <typename Space,
          size_t LinearIdx,
          template <auto...>
          typename Action,
          size_t... AxisIndices>
struct InvokeActionAtIndex<Space, LinearIdx, Action, std::index_sequence<AxisIndices...>> {
    template <typename... Args>
    static void
    apply(Args &&...args) {
        // Decode linear index to per-axis indices, then to values
        // axis_index_at_v<AxisIdx, LinearIdx> gives the index within that axis
        // axis_at_type<AxisIdx>::value_at<idx> gives the value at that index
        Action<Space::template axis_at_type<AxisIndices>::template value_at<
            Space::template axis_index_at_v<AxisIndices, LinearIdx>>...>::
            apply(std::forward<Args>(args)...);
    }
};

// Helper to iterate over all linear indices using a fold expression
template <typename Space, template <auto...> typename Action, typename LinearIndexSeq>
struct ForEachValuesFlatImpl;

template <typename Space, template <auto...> typename Action, size_t... LinearIndices>
struct ForEachValuesFlatImpl<Space, Action, std::index_sequence<LinearIndices...>> {
    template <typename... Args>
    static void
    apply(Args &&...args) {
        // Invoke action for each linear index
        (InvokeActionAtIndex<
             Space,
             LinearIndices,
             Action,
             std::make_index_sequence<Space::num_axes>>::apply(std::forward<Args>(args)...),
         ...);
    }
};

} // namespace detail

template <typename Space, template <auto...> typename Action> struct for_each_values_flat;

template <typename... Axes, template <auto...> typename Action>
struct for_each_values_flat<AxisOuterProduct<Axes...>, Action> {
    using Space = AxisOuterProduct<Axes...>;

    template <typename... Args>
    static void
    apply(Args &&...args) {
        detail::ForEachValuesFlatImpl<Space, Action, std::make_index_sequence<Space::size>>::apply(
            std::forward<Args>(args)...);
    }
};

// -----------------------------------------------------------------------------
// for_each_permutation
// -----------------------------------------------------------------------------
// Compile-time iteration over all value combinations in an AxisOuterProduct.
// For each permutation (v0, v1, ..., vN):
//   1. Instantiates Instantiator<v0, v1, ..., vN>
//   2. Invokes Action<Instantiator<v0, v1, ..., vN>, v0, v1, ..., vN>::apply(args...)
//
// This gives Action access to both the instantiated type AND the compile-time values.
//
// Instantiator signature: template <auto... Values> struct Inst { ... };
// Action signature: template <typename InstType, auto... Values> struct Action {
//     static void apply(Args...);
// };

namespace detail {

template <template <auto...> typename Instantiator,
          template <typename, auto...>
          typename Action,
          typename... Axes>
struct ForEachPermutationImpl;

// Base case: instantiate and invoke Action
template <template <auto...> typename Instantiator, template <typename, auto...> typename Action>
struct ForEachPermutationImpl<Instantiator, Action> {
    template <auto... Values, typename... Args>
    static void
    apply(Args &&...args) {
        Action<Instantiator<Values...>, Values...>::apply(std::forward<Args>(args)...);
    }
};

// Recursive case: expand axis, recurse
template <template <auto...> typename Instantiator,
          template <typename, auto...>
          typename Action,
          auto... AxisVals,
          typename... Rest>
struct ForEachPermutationImpl<Instantiator, Action, SameTypeUniqueValuePack<AxisVals...>, Rest...> {
    template <auto... Accumulated, typename... Args>
    static void
    apply(Args &&...args) {
        (ForEachPermutationImpl<Instantiator, Action, Rest...>::template apply<Accumulated...,
                                                                               AxisVals>(
             std::forward<Args>(args)...),
         ...);
    }
};

} // namespace detail

template <typename Space,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action>
struct for_each_permutation;

template <typename... Axes,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action>
struct for_each_permutation<AxisOuterProduct<Axes...>, Instantiator, Action> {
    template <typename... Args>
    static void
    apply(Args &&...args) {
        detail::ForEachPermutationImpl<Instantiator, Action, Axes...>::template apply<>(
            std::forward<Args>(args)...);
    }
};

// -----------------------------------------------------------------------------
// for_each_permutation_flat
// -----------------------------------------------------------------------------
// Flattened version of for_each_permutation that uses linear index iteration
// instead of recursive template expansion. Like for_each_values_flat, this
// produces O(Space::size) template instantiations with constant depth.
//
// For each linear index:
//   1. Decodes to axis values at compile time
//   2. Instantiates Instantiator<values...>
//   3. Invokes Action<InstType, values...>::apply(args...)

namespace detail {

// Helper to invoke Action with Instantiator at a specific linear index.
template <typename Space,
          size_t LinearIdx,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action,
          typename AxisIndexSeq>
struct InvokePermutationAtIndex;

template <typename Space,
          size_t LinearIdx,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action,
          size_t... AxisIndices>
struct InvokePermutationAtIndex<Space,
                                LinearIdx,
                                Instantiator,
                                Action,
                                std::index_sequence<AxisIndices...>> {
    // Extract the value for each axis at this linear index
    template <size_t AxisIdx>
    static constexpr auto value_for_axis = Space::template axis_at_type<AxisIdx>::template value_at<
        Space::template axis_index_at_v<AxisIdx, LinearIdx>>;

    // The instantiated type for these values
    using instantiated_type = Instantiator<value_for_axis<AxisIndices>...>;

    template <typename... Args>
    static void
    apply(Args &&...args) {
        Action<instantiated_type, value_for_axis<AxisIndices>...>::apply(
            std::forward<Args>(args)...);
    }
};

// Helper to iterate over all linear indices
template <typename Space,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action,
          typename LinearIndexSeq>
struct ForEachPermutationFlatImpl;

template <typename Space,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action,
          size_t... LinearIndices>
struct ForEachPermutationFlatImpl<Space,
                                  Instantiator,
                                  Action,
                                  std::index_sequence<LinearIndices...>> {
    template <typename... Args>
    static void
    apply(Args &&...args) {
        (InvokePermutationAtIndex<
             Space,
             LinearIndices,
             Instantiator,
             Action,
             std::make_index_sequence<Space::num_axes>>::apply(std::forward<Args>(args)...),
         ...);
    }
};

} // namespace detail

template <typename Space,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action>
struct for_each_permutation_flat;

template <typename... Axes,
          template <auto...>
          typename Instantiator,
          template <typename, auto...>
          typename Action>
struct for_each_permutation_flat<AxisOuterProduct<Axes...>, Instantiator, Action> {
    using Space = AxisOuterProduct<Axes...>;

    template <typename... Args>
    static void
    apply(Args &&...args) {
        detail::ForEachPermutationFlatImpl<
            Space,
            Instantiator,
            Action,
            std::make_index_sequence<Space::size>>::apply(std::forward<Args>(args)...);
    }
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_AXISOUTERPRODUCT_H
