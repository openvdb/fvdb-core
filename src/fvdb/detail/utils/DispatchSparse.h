// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_DISPATCHSPARSE_H
#define FVDB_DETAIL_UTILS_DISPATCHSPARSE_H

#include <torch/types.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdint>
#include <type_traits>

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


enum class DummyEnum {
    Mup = 18,
    Spod = 2,
    Bom = 44
};

struct DummyMupT {};
struct DummySpodT {};
struct DummyBomT {};

// -----------------------------------------------------------------------------
// DispatchAxis: Generic Traits for Value-Type Associations
// -----------------------------------------------------------------------------
// Associates a set of runtime values (enums/ints) with corresponding types,
// providing compile-time and runtime lookup in both directions.

// A single value-type pair: directly encodes the association
template <auto V, typename T>
struct ValueTypePair {
    static constexpr auto value = V;
    using type = T;
};

namespace detail {

// Helper: Get the type at index I from a pack of Pairs
template <size_t I, typename... Pairs>
struct TypeAtIndexImpl;

template <typename First, typename... Rest>
struct TypeAtIndexImpl<0, First, Rest...> {
    using type = typename First::type;
};

template <size_t I, typename First, typename... Rest>
struct TypeAtIndexImpl<I, First, Rest...> {
    using type = typename TypeAtIndexImpl<I - 1, Rest...>::type;
};

// Helper: Check if type T exists in the pack of Pairs
template <typename T, typename... Pairs>
struct ContainsTypeImpl : std::false_type {};

template <typename T, typename First, typename... Rest>
struct ContainsTypeImpl<T, First, Rest...>
    : std::bool_constant<std::is_same_v<T, typename First::type> ||
                         ContainsTypeImpl<T, Rest...>::value> {};

// Helper: Get the index of type T in the pack of Pairs
template <typename T, typename... Pairs>
struct IndexOfTypeImpl;

template <typename T, typename First, typename... Rest>
struct IndexOfTypeImpl<T, First, Rest...> {
    static constexpr size_t value =
        std::is_same_v<T, typename First::type> ? 0 : 1 + IndexOfTypeImpl<T, Rest...>::value;
};

} // namespace detail

// The main DispatchAxis template
// Usage: DispatchAxis<ValueTypePair<EnumVal1, Type1>, ValueTypePair<EnumVal2, Type2>, ...>
template <typename... Pairs>
struct DispatchAxis {
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
    static bool contains_value(ValueType v) {
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
    static constexpr size_t index_of_value(ValueType v) {
        // Build a constexpr array of all values for lookup
        using CommonType = std::common_type_t<decltype(Pairs::value)...>;
        constexpr CommonType values[] = {static_cast<CommonType>(Pairs::value)...};

        for (size_t i = 0; i < size; ++i) {
            if (values[i] == static_cast<CommonType>(v)) return i;
        }
        throw std::runtime_error("[DispatchAxis] Value not in axis");
    }

    // -------------------------------------------------------------------------
    // REVERSE INDEXING (compile-time index to type)
    // -------------------------------------------------------------------------
    // Get the type at linear index I
    // Error if I >= size (will fail to compile)
    template <size_t I>
    using type_at_index = typename detail::TypeAtIndexImpl<I, Pairs...>::type;

    // -------------------------------------------------------------------------
    // REVERSE INDEXING (runtime index to value)
    // -------------------------------------------------------------------------
    // Get the value at linear index idx
    // Behavior undefined if idx >= size (for performance; caller must check)
    static constexpr auto value_at_index(size_t idx) {
        using CommonType = std::common_type_t<decltype(Pairs::value)...>;
        constexpr CommonType values[] = {static_cast<CommonType>(Pairs::value)...};
        return values[idx];
    }
};

// -----------------------------------------------------------------------------
// Example: DummyAxis using DummyEnum and Dummy types
// -----------------------------------------------------------------------------
using DummyAxis = DispatchAxis<
    ValueTypePair<DummyEnum::Mup, DummyMupT>,
    ValueTypePair<DummyEnum::Spod, DummySpodT>,
    ValueTypePair<DummyEnum::Bom, DummyBomT>
>;

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

template <auto V>
struct Tag {
    static constexpr auto value = V;
};

// Device tag types - each is a distinct type that carries its device enum value
using TorchDeviceCpuTag = Tag<c10::kCPU>;
using TorchDeviceCudaTag = Tag<c10::kCUDA>;
using TorchDevicePrivateUse1Tag = Tag<c10::kPrivateUse1>;

// Create a DispatchAxis directly from enum values
// Automatically generates Tag<V> types for each value
// Usage: DispatchAxisFromValues<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>
template <auto... Vs>
using TorchDeviceDispatchAxis = DispatchAxis<ValueTypePair<Vs, Tag<Vs>>...>;

// Now TorchDeviceAxis can be defined much more simply:
using ExampleTorchDeviceAxis = TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>;

// -----------------------------------------------------------------------------
// IntDispatchAxis: DispatchAxis from a list of integer values
// -----------------------------------------------------------------------------
// For dispatch over compile-time integer values (e.g., supported channel counts,
// kernel sizes, etc.). Each integer becomes a distinct type via IntegralTag.

// A tag type that wraps an integral value, inheriting from std::integral_constant
// for standard library compatibility (implicit conversion, ::value, etc.)
template <auto V>
struct IntegralTag : std::integral_constant<decltype(V), V> {};

// Create a DispatchAxis directly from integer values
// Automatically generates IntegralTag<V> types for each value
// Usage: IntDispatchAxis<1, 3, 4, 8>
template <auto... Vs>
using IntDispatchAxis = DispatchAxis<ValueTypePair<Vs, IntegralTag<Vs>>...>;

// Example: An axis for supported channel counts
using ExampleChannelAxis = IntDispatchAxis<1, 3, 4, 8, 16, 32>;

// -----------------------------------------------------------------------------
// AxisProduct: Outer product of multiple DispatchAxis types
// -----------------------------------------------------------------------------
// Forms the cartesian product of all axis elements, providing linear indexing
// into the combined space from either runtime values or compile-time types.

template <typename... Axes>
struct AxisProduct {
    static constexpr size_t num_axes = sizeof...(Axes);
    static constexpr size_t size = (Axes::size * ... * 1);

    // Access the Nth axis type
    template <size_t I>
    using axis_at = std::tuple_element_t<I, std::tuple<Axes...>>;

private:
    // Compute strides at compile time (row-major: last axis has stride 1)
    static constexpr auto compute_strides() {
        std::array<size_t, num_axes> result{};
        if constexpr (num_axes > 0) {
            constexpr size_t sizes[] = {Axes::size...};
            size_t stride = 1;
            for (size_t i = num_axes; i > 0; --i) {
                result[i - 1] = stride;
                stride *= sizes[i - 1];
            }
        }
        return result;
    }

    static constexpr std::array<size_t, num_axes> strides = compute_strides();

    // Helper for compile-time index computation
    template <typename... Types>
    struct IndexOfTypesImpl {
        static_assert(sizeof...(Types) == num_axes, "Must provide one type per axis");

        template <size_t... Is>
        static constexpr size_t compute(std::index_sequence<Is...>) {
            return ((axis_at<Is>::template index_of_type<
                         std::tuple_element_t<Is, std::tuple<Types...>>> *
                     strides[Is]) +
                    ... + 0);
        }

        static constexpr size_t value = compute(std::make_index_sequence<num_axes>{});
    };

    // Helper for index computation (constexpr-friendly)
    template <size_t... Is, typename ValueTuple>
    static constexpr size_t index_of_values_impl(std::index_sequence<Is...>, const ValueTuple& values) {
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
    static constexpr size_t index_of_values(Values... values) {
        static_assert(sizeof...(Values) == num_axes, "Must provide one value per axis");
        return index_of_values_impl(std::make_index_sequence<num_axes>{},
                                    std::make_tuple(values...));
    }
};

// Example: Product of device, dtype, and channel axes
// Total combinations = 3 devices * 5 dtypes * 6 channels = 90
using ExampleProductAxis = AxisProduct<
    TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA, c10::kPrivateUse1>,
    TorchDtypeAxis<float, double, int32_t, int64_t, bool>,
    IntDispatchAxis<1, 3, 4, 8, 16, 32>
>;

// Usage: (compile time or runtime)
// ExampleProductAxis::index_of_types<TorchDeviceCudaTag, float, IntegralTag<4>>
// ExampleProductAxis::index_of_values(c10::kCUDA, c10::ScalarType::Float, 4)

// -----------------------------------------------------------------------------
// Binding: A self-describing binding of types to a function pointer
// -----------------------------------------------------------------------------
// Carries the axis types and the function pointer. Knows its own index
// when combined with an AxisProduct.

template <typename FnPtr, typename... Types>
struct Binding {
    FnPtr func;

    constexpr explicit Binding(FnPtr f) : func(f) {}

    // Compute index for a given AxisProduct
    template <typename Axes>
    static constexpr size_t index_for() {
        return Axes::template index_of_types<Types...>;
    }
};

// Helper to create a Binding with explicit types
template <typename... Types, typename FnPtr>
constexpr auto make_binding(FnPtr func) {
    return Binding<FnPtr, Types...>{func};
}

// -----------------------------------------------------------------------------
// DispatchTable: Sparse dispatch over an AxisProduct
// -----------------------------------------------------------------------------
// A table of function pointers, populated by Bindings.
// Uses function pointers (not std::function) for zero overhead.

template <typename FunctionSignature, typename AxesT>
struct DispatchTable;

template <typename ReturnType, typename... Args, typename AxesT>
struct DispatchTable<ReturnType(Args...), AxesT> {
    using Axes = AxesT;
    using FunctionPtr = ReturnType (*)(Args...);

private:
    std::array<FunctionPtr, Axes::size> table_;

    // Default handler for unbound combinations
    static ReturnType not_implemented(Args...) {
        throw std::runtime_error(
            "[DispatchTable] No implementation bound for this combination of axis values");
    }

    // Install a single binding
    template <typename FnPtr, typename... Types>
    constexpr void install(Binding<FnPtr, Types...> binding) {
        static_assert(sizeof...(Types) == Axes::num_axes,
                      "Binding must have one type per axis");
        constexpr size_t idx = Binding<FnPtr, Types...>::template index_for<Axes>();
        table_[idx] = binding.func;
    }

public:
    // Default constructor: all slots point to not_implemented
    constexpr DispatchTable() {
        for (auto& entry : table_) {
            entry = &not_implemented;
        }
    }

    // Construct from a list of Bindings
    template <typename... Bindings>
    constexpr explicit DispatchTable(Bindings... bindings) : DispatchTable() {
        (install(bindings), ...);
    }

    // Add a binding after construction
    template <typename... Types>
    constexpr DispatchTable& add(Binding<FunctionPtr, Types...> binding) {
        install(binding);
        return *this;
    }

    // -------------------------------------------------------------------------
    // operator(): Dispatch based on runtime values
    // -------------------------------------------------------------------------
    template <typename... Values>
    ReturnType operator()(Values... values, Args... args) const {
        static_assert(sizeof...(Values) == Axes::num_axes,
                      "Must provide one value per axis");
        size_t idx = Axes::index_of_values(values...);
        return table_[idx](std::forward<Args>(args)...);
    }

    // -------------------------------------------------------------------------
    // is_bound(): Check if a combination is bound (for debugging/testing)
    // -------------------------------------------------------------------------
    template <typename... Types>
    constexpr bool is_bound() const {
        constexpr size_t idx = Axes::template index_of_types<Types...>;
        return table_[idx] != &not_implemented;
    }
};

// Free function to add a binding (alternative syntax)
template <typename Table, typename FnPtr, typename... Types>
constexpr Table& bind(Table& table, Binding<FnPtr, Types...> binding) {
    return table.add(binding);
}

// -----------------------------------------------------------------------------
// Example: "blub" operation with sparse coverage
// -----------------------------------------------------------------------------

#if 0  // Example code - not compiled

// Accessor is assumed to be a tensor accessor parameterized by element type
template <typename T> struct Accessor {
    torch::Tensor tensor;
    explicit Accessor(torch::Tensor t) : tensor(std::move(t)) {}
};

struct BlubOp {
    // -------------------------------------------------------------------------
    // Axis definitions
    // -------------------------------------------------------------------------
    using DeviceAxis = TorchDeviceDispatchAxis<c10::kCPU, c10::kCUDA>;
    using InDtypeAxis = TorchDtypeAxis<float, double>;
    using OutDtypeAxis = TorchDtypeAxis<float, double, int32_t>;
    using Axes = AxisProduct<DeviceAxis, InDtypeAxis, OutDtypeAxis>;
    using Table = DispatchTable<void(torch::Tensor, torch::Tensor), Axes>;

    // -------------------------------------------------------------------------
    // Implementations - overloaded for different type combinations
    // -------------------------------------------------------------------------
    static void impl(TorchDeviceCpuTag, Accessor<float> in, Accessor<int> out) {
        printf("blub_impl(Cpu, float, int)\n");
    }

    template <typename T>
    static void impl(TorchDeviceCpuTag, Accessor<T> in, Accessor<T> out) {
        printf("blub_impl(Cpu, %s, %s)\n", typeid(T).name(), typeid(T).name());
    }

    static void impl(TorchDeviceCudaTag, Accessor<float> in, Accessor<int> out) {
        printf("blub_impl(Cuda, float, int)\n");
    }

    // Fallback for any device with double,double
    template <typename DeviceTag>
    static void impl(DeviceTag, Accessor<double> in, Accessor<double> out) {
        printf("fallback blub_impl(%s, double, double)\n", typeid(DeviceTag).name());
    }

    // -------------------------------------------------------------------------
    // Delegate: bridges runtime tensors to compile-time typed implementation
    // -------------------------------------------------------------------------
    template <typename DeviceTag, typename Tin, typename Tout>
    static void delegate(torch::Tensor in, torch::Tensor out) {
        impl(DeviceTag{}, Accessor<Tin>{in}, Accessor<Tout>{out});
    }

    // -------------------------------------------------------------------------
    // Binding factory: creates a Binding for given types using our delegate
    // -------------------------------------------------------------------------
    template <typename... Types>
    static constexpr auto binding() {
        return make_binding<Types...>(&delegate<Types...>);
    }

    // -------------------------------------------------------------------------
    // Build the dispatch table
    // -------------------------------------------------------------------------
    static auto make_dispatch_table() {
        // Option 1: Construct with all bindings at once
        return Table{
            binding<TorchDeviceCpuTag, float, int32_t>(),
            binding<TorchDeviceCudaTag, float, int32_t>(),
            binding<TorchDeviceCpuTag, double, double>(),
            binding<TorchDeviceCudaTag, double, double>()
        };

        // Option 2: Build incrementally
        // Table table;
        // bind(table, binding<TorchDeviceCpuTag, float, int32_t>());
        // bind(table, binding<TorchDeviceCudaTag, float, int32_t>());
        // return table;
    }
};

void blub(torch::Tensor in, torch::Tensor out) {
    static auto dispatch = BlubOp::make_dispatch_table();

    auto device_type = in.device().type();
    auto in_dtype = in.scalar_type();
    auto out_dtype = out.scalar_type();

    dispatch(device_type, in_dtype, out_dtype, in, out);
}

#endif  // Example code



// -----------------------------------------------------------------------------
// IGNORE BELOW HERE

// -----------------------------------------------------------------------------
// 2. Compile-Time Tags
// -----------------------------------------------------------------------------
// These are passed to your templates to select specific kernel overloads.

struct CpuTag {};
struct CudaTag {};

// -----------------------------------------------------------------------------
// 3. Dispatch Configuration Lists
// -----------------------------------------------------------------------------
// These lists define the "Universes" of types your registry can iterate over.

// A simple container for types
template <typename... Ts> struct TypeList {};

// The specific lists used by the Dispatcher to calculate table sizes/indices.
// The order here determines the index in the lookup table (0, 1, ...).

using DeviceList = TypeList<CpuTag, CudaTag>;

using DtypeList = TypeList<float,   // Maps to Float32
                           double,  // Maps to Float64
                           int32_t, // Maps to Int32
                           int64_t, // Maps to Int64
                           bool     // Maps to Bool
                           >;

// -------------------------------------------------------------------------
// Helper: Compile-Time Lookup of a Type in a TypeList (first found)
// -------------------------------------------------------------------------
template <typename T, typename List> struct IndexOf;

template <typename T, typename... Ts> struct IndexOf<T, TypeList<T, Ts...>> {
    static constexpr size_t value = 0;
};

template <typename T, typename U, typename... Ts> struct IndexOf<T, TypeList<U, Ts...>> {
    static constexpr size_t value = 1 + IndexOf<T, TypeList<Ts...>>::value;
};

// -------------------------------------------------------------------------
// Helper: Strided Index Calculation (N-Dimensional -> 1D)
// -------------------------------------------------------------------------
// Recursively calculates the flat index based on runtime enums and dimension sizes.

// Base case: Last dimension
template <size_t Stride>
size_t
linearize_index() {
    return 0;
}

// Recursive step
// Arg: Current runtime enum value
// List: The TypeList corresponding to this dimension (used to get size)
// Rest: Remaining dimensions
template <size_t Stride, typename List, typename... Lists, typename Enum, typename... Enums>
size_t
linearize_index(Enum val, Enums... rest) {
    // Size of the current dimension
    constexpr size_t DimSize =
        sizeof...(List); // Requires Types in List (hacky but works if List=TypeList<...>)
                         // Better: define SizeOf<TypeList> helper. See below.

    // Stride for the *next* dimension is (CurrentStride / DimSize)
    // This is standard row-major indexing: idx = i*stride + j*stride2 ...
    // Actually, easier approach: Accumulate offsets.

    // Let's do simple accumulation:
    // idx = val + DimSize * (linearize(rest...))
    return static_cast<size_t>(val) + DimSize * linearize_index<0, Lists...>(rest...);
}

// Helper to get size of a TypeList
template <typename List> struct ListSize;
template <typename... Ts> struct ListSize<TypeList<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

// -------------------------------------------------------------------------
// The Variadic Sparse Registry
// -------------------------------------------------------------------------
// DimensionLists... : A pack of TypeLists (e.g. DeviceList, DtypeList, DtypeList)
template <typename Sig, typename Op, typename... DimensionLists> class Registry;

template <typename R, typename... Args, typename Op, typename... DimensionLists>
class Registry<R(Args...), Op, DimensionLists...> {
    Op op;

    // Total table size = Product of all dimension sizes
    static constexpr size_t TableSize = (ListSize<DimensionLists>::value * ... * 1);

    using Trampoline = R (*)(void *, Args...);
    struct Entry {
        Trampoline func = nullptr;
    };
    std::vector<Entry> table;

    // ---------------------------------------------------------------------
    // Internal Thunk
    // ---------------------------------------------------------------------
    // Tags... corresponds to <CudaTag, Float, Int> etc.
    template <typename... Tags>
    static R
    specific_trampoline(void *instance, Args... args) {
        auto *self = static_cast<Registry *>(instance);
        // Expand the tags into the user's lambda
        return self->op(Tags{}..., std::forward<Args>(args)...);
    }

    static R
    not_implemented(void *, Args...) {
        throw std::runtime_error("[Dispatch] Combination not implemented.");
    }

    // Helper to linearize indices at Runtime
    // We use a fold expression to compute the flat index
    // Formula: i + D1*(j + D2*(k ...))
    size_t
    compute_flat_index(auto... enums) {
        size_t idx    = 0;
        size_t stride = 1;

        // Helper lambda to accumulate index
        // We process dimensions right-to-left (inner-to-outer) or left-to-right.
        // Let's do simple: idx = e0 + Size0 * (e1 + Size1 * (e2...))
        // To do this via fold expression requires zipping Enums and Lists.

        // Simpler Loop approach (since N is small):
        size_t values[] = {static_cast<size_t>(enums)...};
        size_t sizes[]  = {ListSize<DimensionLists>::value...};

        for (int i = sizeof...(enums) - 1; i >= 0; --i) {
            idx = values[i] + sizes[i] * idx;
            // Note: Make sure your 'sizes' logic matches your 'bind' logic!
            // If bind uses Lexicographical order (D0, D1, D2), this formula is:
            // idx = val[N] + Size[N] * (val[N-1] + ...)
            // This means the LAST dimension is the major stride.
        }
        return idx;
    }

    // Helper to linearize indices at Compile Time (for Bind)
    template <typename... Tags>
    static constexpr size_t
    compile_time_index() {
        constexpr size_t indices[] = {IndexOf<Tags, DimensionLists>::value...};
        constexpr size_t sizes[]   = {ListSize<DimensionLists>::value...};

        size_t idx = 0;
        for (int i = sizeof...(Tags) - 1; i >= 0; --i) {
            idx = indices[i] + sizes[i] * idx;
        }
        return idx;
    }

  public:
    explicit Registry(Op o) : op(std::move(o)), table(TableSize) {
        for (auto &e: table)
            e.func = &not_implemented;
    }

    // ---------------------------------------------------------------------
    // Variadic Bind
    // ---------------------------------------------------------------------
    // Usage: .bind<CudaTag, float, int>()
    template <typename... Tags>
    Registry &
    bind() {
        static_assert(sizeof...(Tags) == sizeof...(DimensionLists),
                      "Bind must match dimension count");

        constexpr size_t idx = compile_time_index<Tags...>();
        table[idx].func      = &specific_trampoline<Tags...>;
        return *this;
    }

    // ---------------------------------------------------------------------
    // Variadic Call
    // ---------------------------------------------------------------------
    // Usage: dispatch(dev, dtype1, dtype2, args...)
    // Note: The first N arguments must be the Enums.
    template <typename... Enums>
    R
    operator()(Enums... enums, Args... args) {
        static_assert(sizeof...(Enums) == sizeof...(DimensionLists),
                      "Call must match dimension count");

        size_t idx = compute_flat_index(enums...);
        return table[idx].func(this, std::forward<Args>(args)...);
    }
};

// Factory
template <typename Sig, typename... DimensionLists, typename Op>
auto
make_registry(Op &&op) {
    return Registry<Sig, std::decay_t<Op>, DimensionLists...>(std::forward<Op>(op));
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_DISPATCHSPARSE_H
