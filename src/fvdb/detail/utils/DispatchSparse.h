// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_DISPATCHSPARSE_H
#define FVDB_DETAIL_UTILS_DISPATCHSPARSE_H

#include <torch/types.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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

struct DummyMupT {};
struct DummySpodT {};
struct DummyBomT {};

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
template <typename T, typename... Pairs> struct IndexOfTypeImpl;

template <typename T, typename First, typename... Rest> struct IndexOfTypeImpl<T, First, Rest...> {
    static constexpr size_t value =
        std::is_same_v<T, typename First::type> ? 0 : 1 + IndexOfTypeImpl<T, Rest...>::value;
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
using DummyAxis = DispatchAxis<ValueTypePair<DummyEnum::Mup, DummyMupT>,
                               ValueTypePair<DummyEnum::Spod, DummySpodT>,
                               ValueTypePair<DummyEnum::Bom, DummyBomT>>;

// -----------------------------------------------------------------------------
// TorchDtypeAxis: DispatchAxis from a pack of C++ types
// -----------------------------------------------------------------------------
// Demonstrates defining a DispatchAxis from just raw C++ element types,
// automatically associating each with the corresponding torch ScalarType enum.
// Uses torch's built-in c10::CppTypeToScalarType<T> for the mapping.

template <typename... Ts>
using TorchDtypeAxis = DispatchAxis<ValueTypePair<c10::CppTypeToScalarType<Ts>::value, Ts>...>;

// Example: A dtype axis for common floating-point types
using FloatDtypeAxis = TorchDtypeAxis<float, double>;

// Example: A dtype axis matching the existing DtypeList
using StandardDtypeAxis = TorchDtypeAxis<float, double, int32_t, int64_t, bool>;

// -----------------------------------------------------------------------------
// TorchDeviceAxis: DispatchAxis for torch device types
// -----------------------------------------------------------------------------
// For devices, we don't have distinct C++ types - we need to create tag types
// that wrap the device enum values. The Tag template creates a unique type for
// each enum value.

template <auto V> struct Tag {
    static constexpr auto value = V;
};

// Device tag types - each is a distinct type that carries its device enum value
using TorchDeviceCpuTag         = Tag<c10::kCPU>;
using TorchDeviceCudaTag        = Tag<c10::kCUDA>;
using TorchDevicePrivateUse1Tag = Tag<c10::kPrivateUse1>;

// Create a DispatchAxis directly from enum values
// Automatically generates Tag<V> types for each value
// Usage: DispatchAxisFromValues<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>
template <auto... Vs> using TorchDeviceDispatchAxis = DispatchAxis<ValueTypePair<Vs, Tag<Vs>>...>;

// Now TorchDeviceAxis can be defined much more simply:
using ExampleTorchDeviceAxis = TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>;

// -----------------------------------------------------------------------------
// IntDispatchAxis: DispatchAxis from a list of integer values
// -----------------------------------------------------------------------------
// For dispatch over compile-time integer values (e.g., supported channel counts,
// kernel sizes, etc.). Each integer becomes a distinct type via IntegralTag.

// A tag type that wraps an integral value, inheriting from std::integral_constant
// for standard library compatibility (implicit conversion, ::value, etc.)
template <auto V> struct IntegralTag : std::integral_constant<decltype(V), V> {};

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
using ExampleProductAxis =
    AxisProduct<TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>,
                TorchDtypeAxis<float, double, int32_t, int64_t, bool>,
                IntDispatchAxis<1, 3, 4, 8, 16, 32>>;

// Usage: (compile time or runtime)
// ExampleProductAxis::index_of_types<TorchDeviceCudaTag, float, IntegralTag<4>>
// ExampleProductAxis::index_of_values(c10::kCUDA, c10::ScalarType::Float, 4)

//-----------------------------------------------------------------------------------
// TENSOR ACCESSOR HELPERS
//-----------------------------------------------------------------------------------

template <typename DeviceTag, typename T, size_t N>
using TorchAccessor64 =
    typename std::conditional<DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1,
                              torch::PackedTensorAccessor64<T, N, torch::RestrictPtrTraits>,
                              torch::TensorAccessor<T, N, int64_t>>::type;

template <typename DeviceTag, typename T, size_t N>
using TorchAccessor32 =
    typename std::conditional<DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1,
                              torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>,
                              torch::TensorAccessor<T, N, int32_t>>::type;

template <typename DeviceTag, typename T, size_t N>
TorchAccessor64<DeviceTag, T, N>
makeAccessor64(torch::Tensor &tensor) {
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        return tensor.generic_packed_accessor<T, N, torch::RestrictPtrTraits, int64_t>();
    } else {
        return tensor.accessor<T, N, int64_t>();
    }
}

template <typename DeviceTag, typename T, size_t N>
TorchAccessor32<DeviceTag, T, N>
makeAccessor32(torch::Tensor &tensor) {
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        return tensor.generic_packed_accessor<T, N, torch::RestrictPtrTraits, int32_t>();
    } else {
        return tensor.accessor<T, N, int32_t>();
    }
}

// -----------------------------------------------------------------------------
// The Base DispatchTable class
// -----------------------------------------------------------------------------
template <typename FunctionSignature, typename AxesT> struct BaseDispatchTable;

template <typename ReturnType, typename... Args, typename AxesT>
struct BaseDispatchTable<ReturnType(Args...), AxesT> {
    using Axes        = AxesT;
    using FunctionPtr = ReturnType (*)(Args...);
    std::array<FunctionPtr, Axes::size> table_{};

    // Dispatch with generic error on unbound combination
    template <auto... Values>
    ReturnType
    dispatch(Values... values, Args... args) const {
        static_assert(sizeof...(Values) == Axes::num_axes, "Must provide one value per axis");
        auto const idx = Axes::index_of_values(values...);

        if (table_[idx] == nullptr) {
            throw std::runtime_error("No implementation bound for this combination of axis values");
        }
        return table_[idx](std::forward<Args>(args)...);
    }

    // Dispatch with custom fallback handler for unbound combinations.
    // Fallback receives axis values followed by function args, enabling
    // operation-specific error messages with full runtime context.
    template <typename Fallback, auto... Values>
    ReturnType
    dispatch_or(Fallback &&fallback, Values... values, Args... args) const {
        static_assert(sizeof...(Values) == Axes::num_axes, "Must provide one value per axis");
        auto const idx = Axes::index_of_values(values...);

        if (table_[idx] == nullptr) {
            return std::invoke(
                std::forward<Fallback>(fallback), values..., std::forward<Args>(args)...);
        }
        return table_[idx](std::forward<Args>(args)...);
    }
};

// -----------------------------------------------------------------------------
// BindAt: A declarative binding specification (type-level, no runtime data)
// -----------------------------------------------------------------------------
// Specifies that Factory<Types...>{}() should be bound at the slot for Types...
// Factory is a class template with operator() that returns a function pointer.

template <template <typename...> class Factory, typename... Types> struct BindAt {
    template <typename Table>
    static void
    apply(Table &table) {
        constexpr size_t idx = Table::Axes::template index_of_types<Types...>;
        table.table_[idx]    = Factory<Types...>{}();
    }
};

// -----------------------------------------------------------------------------
// BindFn: Bind a constexpr lambda factory (C++20)
// -----------------------------------------------------------------------------
// Allows factories to be written as constexpr lambdas with template parameters:
//   constexpr auto my_factory = []<typename A, typename B>() { return [...]; };
// Usage: BindFn<my_factory, TypeA, TypeB>

template <auto Factory, typename... Types> struct BindFn {
    template <typename Table>
    static void
    apply(Table &table) {
        constexpr size_t idx = Table::Axes::template index_of_types<Types...>;
        table.table_[idx]    = Factory.template operator()<Types...>();
    }
};

// -----------------------------------------------------------------------------
// Bindings: A composable list of binding specifications
// -----------------------------------------------------------------------------
// Aggregates multiple BindAt specs. Applied via fold expression.

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
    BaseDispatchTable<Signature, Axes> table{};
    BindingSpec::apply(table);
    return table;
}

namespace example {

// -----------------------------------------------------------------------------
// Example: "blub" operation with sparse coverage
// -----------------------------------------------------------------------------

// -------------------------------------------------------------------------
// The blub_impl variations below represent "real" authoring content. Nothing in them
// is boilerplate beyond what we'd expect to have to write, nothing feels overly repetitive
// or onerous. We are able to separate out variations where it makes sense to do so for
// whatever our kernel/op requires. The examples here are contrived and an unlikely
// set of specializations, done really just to demonstrate and flex the dispatch system.
// Pretty much everything beyond these blub_impl implementations is boilerplate,
// so we want it to be as minimal as possible, and we want to offload as much as possible
// to the dispatch utility headers.
// -------------------------------------------------------------------------

void blub_impl(TorchDeviceCpuTag,
               TorchAccessor64<TorchDeviceCpuTag, float, 1> in,
               TorchAccessor64<TorchDeviceCpuTag, int, 1> out);

void blub_impl(TorchDeviceCudaTag,
               TorchAccessor64<TorchDeviceCudaTag, float, 1> in,
               TorchAccessor64<TorchDeviceCudaTag, int, 1> out);

template <typename T>
void blub_impl(TorchDeviceCpuTag,
               TorchAccessor64<TorchDeviceCpuTag, T, 1> in,
               TorchAccessor64<TorchDeviceCpuTag, T, 1> out);

extern template void blub_impl<int32_t>(TorchDeviceCpuTag,
                                        TorchAccessor64<TorchDeviceCpuTag, int32_t, 1>,
                                        TorchAccessor64<TorchDeviceCpuTag, int32_t, 1>);
extern template void blub_impl<int64_t>(TorchDeviceCpuTag,
                                        TorchAccessor64<TorchDeviceCpuTag, int64_t, 1>,
                                        TorchAccessor64<TorchDeviceCpuTag, int64_t, 1>);
extern template void blub_impl<torch::Half>(TorchDeviceCpuTag,
                                            TorchAccessor64<TorchDeviceCpuTag, torch::Half, 1>,
                                            TorchAccessor64<TorchDeviceCpuTag, torch::Half, 1>);
extern template void blub_impl<float>(TorchDeviceCpuTag,
                                      TorchAccessor64<TorchDeviceCpuTag, float, 1>,
                                      TorchAccessor64<TorchDeviceCpuTag, float, 1>);
extern template void blub_impl<double>(TorchDeviceCpuTag,
                                       TorchAccessor64<TorchDeviceCpuTag, double, 1>,
                                       TorchAccessor64<TorchDeviceCpuTag, double, 1>);

template <typename DeviceTag>
void blub_impl(DeviceTag,
               TorchAccessor64<DeviceTag, double, 1> in,
               TorchAccessor64<DeviceTag, int, 1> out);

extern template void blub_impl<TorchDeviceCpuTag>(TorchDeviceCpuTag,
                                                  TorchAccessor64<TorchDeviceCpuTag, double, 1>,
                                                  TorchAccessor64<TorchDeviceCpuTag, int, 1>);
extern template void blub_impl<TorchDeviceCudaTag>(TorchDeviceCudaTag,
                                                   TorchAccessor64<TorchDeviceCudaTag, double, 1>,
                                                   TorchAccessor64<TorchDeviceCudaTag, int, 1>);

// -----------------------------------------------------------------------------
// The general use "blub" function, which will handle all of the dispatch to the
// blub implementations above. This is the whole reason we're here.
// -----------------------------------------------------------------------------
torch::Tensor blub(torch::Tensor in, torch::ScalarType out_dtype = in.scalar_type());

} // namespace example
} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_DISPATCHSPARSE_H
