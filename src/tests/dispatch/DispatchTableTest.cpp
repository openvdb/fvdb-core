// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/SparseDispatchTable.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test Axes Definitions
// =============================================================================

using SingleAxis    = SameTypeUniqueValuePack<10, 20, 30>;
using CharAxis      = SameTypeUniqueValuePack<'a', 'b'>;
using IntAxis       = SameTypeUniqueValuePack<1, 2, 3>;
using BoolAxis      = SameTypeUniqueValuePack<true, false>;

using SingleAxisSpace = AxisOuterProduct<SingleAxis>;
using TwoAxisSpace    = AxisOuterProduct<IntAxis, CharAxis>;
using ThreeAxisSpace  = AxisOuterProduct<IntAxis, CharAxis, BoolAxis>;

// =============================================================================
// Test Function Types
// =============================================================================

// Simple functions with different signatures for testing
int simpleNoArgs() { return 42; }

int simpleOneArg(int x) { return x * 2; }

int simpleTwoArgs(int x, int y) { return x + y; }

std::string stringReturn(int x) { return std::to_string(x); }

// =============================================================================
// FunctionPtr Type Alias Tests
// =============================================================================

TEST(FunctionPtr, TypeAliasIsCorrectForNoArgs) {
    using FPtr = FunctionPtr<int>;
    static_assert(std::is_same_v<FPtr, int(*)()>);

    FPtr ptr = &simpleNoArgs;
    EXPECT_EQ(ptr(), 42);
}

TEST(FunctionPtr, TypeAliasIsCorrectForOneArg) {
    using FPtr = FunctionPtr<int, int>;
    static_assert(std::is_same_v<FPtr, int(*)(int)>);

    FPtr ptr = &simpleOneArg;
    EXPECT_EQ(ptr(5), 10);
}

TEST(FunctionPtr, TypeAliasIsCorrectForMultipleArgs) {
    using FPtr = FunctionPtr<int, int, int>;
    static_assert(std::is_same_v<FPtr, int(*)(int, int)>);

    FPtr ptr = &simpleTwoArgs;
    EXPECT_EQ(ptr(3, 4), 7);
}

TEST(FunctionPtr, TypeAliasWorksWithDifferentReturnTypes) {
    using FPtr = FunctionPtr<std::string, int>;
    static_assert(std::is_same_v<FPtr, std::string(*)(int)>);

    FPtr ptr = &stringReturn;
    EXPECT_EQ(ptr(123), "123");
}

// =============================================================================
// Function Pointer Storage and Retrieval Tests
// =============================================================================

namespace {
// Test functions that return their "identity" based on compile-time values
template <int V> int returnValue(int) { return V; }

template <int V1, char V2> int returnCombined(int x, char c) {
    return V1 * 1000 + static_cast<int>(V2) + x;
}
} // namespace

TEST(FunctionPtrStorage, CanStoreAndRetrieveFunctionPointer) {
    using FPtr = FunctionPtr<int, int>;
    PermutationArrayMap<SingleAxisSpace, FPtr, nullptr> map;

    map.set(std::make_tuple(10), &returnValue<10>);
    map.set(std::make_tuple(20), &returnValue<20>);
    map.set(std::make_tuple(30), &returnValue<30>);

    // Retrieve and call
    auto ptr10 = map.get(std::make_tuple(10));
    auto ptr20 = map.get(std::make_tuple(20));
    auto ptr30 = map.get(std::make_tuple(30));

    ASSERT_NE(ptr10, nullptr);
    ASSERT_NE(ptr20, nullptr);
    ASSERT_NE(ptr30, nullptr);

    EXPECT_EQ(ptr10(0), 10);
    EXPECT_EQ(ptr20(0), 20);
    EXPECT_EQ(ptr30(0), 30);
}

TEST(FunctionPtrStorage, UnsetEntriesReturnNullptr) {
    using FPtr = FunctionPtr<int, int>;
    PermutationArrayMap<SingleAxisSpace, FPtr, nullptr> map;

    map.set(std::make_tuple(10), &returnValue<10>);
    // 20 and 30 not set

    EXPECT_NE(map.get(std::make_tuple(10)), nullptr);
    EXPECT_EQ(map.get(std::make_tuple(20)), nullptr);
    EXPECT_EQ(map.get(std::make_tuple(30)), nullptr);
}

TEST(FunctionPtrStorage, MultiAxisFunctionPointers) {
    using FPtr = FunctionPtr<int, int, char>;
    PermutationArrayMap<TwoAxisSpace, FPtr, nullptr> map;

    map.set(std::make_tuple(1, 'a'), &returnCombined<1, 'a'>);
    map.set(std::make_tuple(2, 'b'), &returnCombined<2, 'b'>);

    auto ptr1a = map.get(std::make_tuple(1, 'a'));
    auto ptr2b = map.get(std::make_tuple(2, 'b'));

    ASSERT_NE(ptr1a, nullptr);
    ASSERT_NE(ptr2b, nullptr);

    // returnCombined<1, 'a'> returns 1*1000 + 'a'(97) + x
    EXPECT_EQ(ptr1a(5, 'x'), 1097 + 5);
    // returnCombined<2, 'b'> returns 2*1000 + 'b'(98) + x
    EXPECT_EQ(ptr2b(10, 'y'), 2098 + 10);
}

// =============================================================================
// Encoder Concept Tests
// =============================================================================

namespace {
// A simple encoder that extracts values from function arguments
struct SingleAxisEncoder {
    static std::tuple<int> encode(int axisValue) {
        return std::make_tuple(axisValue);
    }
};

struct TwoAxisEncoder {
    static std::tuple<int, char> encode(int first, char second, int /*extra*/) {
        return std::make_tuple(first, second);
    }
};

// Invalid encoder - wrong return type
struct InvalidEncoder {
    static int encode(int x) { return x; } // Returns int, not tuple
};
} // namespace

TEST(EncoderConcept, ValidSingleAxisEncoderSatisfiesConcept) {
    static_assert(EncoderConcept<SingleAxisEncoder, SingleAxisSpace, int>);
}

TEST(EncoderConcept, ValidTwoAxisEncoderSatisfiesConcept) {
    static_assert(EncoderConcept<TwoAxisEncoder, TwoAxisSpace, int, char, int>);
}

TEST(EncoderConcept, EncoderWithWrongArgCountFails) {
    // SingleAxisEncoder expects one arg, but we're checking with two
    static_assert(!EncoderConcept<SingleAxisEncoder, SingleAxisSpace, int, int>);
}

TEST(EncoderConcept, EncoderAllowsImplicitConversions) {
    // Implicit conversions are allowed (standard C++ behavior)
    // SingleAxisEncoder::encode takes int, char implicitly converts to int
    static_assert(EncoderConcept<SingleAxisEncoder, SingleAxisSpace, char>);
    // double also converts to int (with truncation)
    static_assert(EncoderConcept<SingleAxisEncoder, SingleAxisSpace, double>);
}

TEST(EncoderConcept, EncoderRejectsNonConvertibleTypes) {
    // std::string cannot convert to int, so this should fail
    static_assert(!EncoderConcept<SingleAxisEncoder, SingleAxisSpace, std::string>);
}

TEST(EncoderConcept, EncoderEncodesCorrectly) {
    auto encoded = SingleAxisEncoder::encode(20);
    EXPECT_EQ(std::get<0>(encoded), 20);

    auto encoded2 = TwoAxisEncoder::encode(2, 'b', 999);
    EXPECT_EQ(std::get<0>(encoded2), 2);
    EXPECT_EQ(std::get<1>(encoded2), 'b');
}

// =============================================================================
// UnboundHandler Concept Tests
// =============================================================================

namespace {
struct ThrowingUnboundHandler {
    [[noreturn]] static void unbound(int value) {
        throw std::runtime_error("Unbound for value: " + std::to_string(value));
    }
};

struct TwoAxisThrowingHandler {
    [[noreturn]] static void unbound(int v1, char v2) {
        throw std::runtime_error("Unbound for: " + std::to_string(v1) + ", " + v2);
    }
};

// For tracking calls without throwing
struct TrackingUnboundHandler {
    static inline int lastValue = 0;
    static inline bool called = false;

    [[noreturn]] static void unbound(int value) {
        lastValue = value;
        called = true;
        throw std::runtime_error("Tracked unbound");
    }

    static void reset() {
        lastValue = 0;
        called = false;
    }
};
} // namespace

TEST(UnboundHandlerConcept, ValidHandlerSatisfiesConcept) {
    static_assert(UnboundHandlerConcept<ThrowingUnboundHandler, SingleAxisSpace>);
    static_assert(UnboundHandlerConcept<TwoAxisThrowingHandler, TwoAxisSpace>);
}

TEST(UnboundHandlerConcept, HandlerThrowsCorrectly) {
    EXPECT_THROW(ThrowingUnboundHandler::unbound(42), std::runtime_error);

    try {
        ThrowingUnboundHandler::unbound(42);
        FAIL() << "Should have thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("42"), std::string::npos);
    }
}

// =============================================================================
// Dispatcher Basic Tests
// =============================================================================

namespace {
// Simple invoker templates for testing
template <int V> struct SimpleInvoker {
    static int invoke(int x) { return V * 10 + x; }
};

template <int V1, char V2> struct TwoAxisInvoker {
    static int invoke(int x, char c) {
        return V1 * 1000 + static_cast<int>(V2) + x + static_cast<int>(c);
    }
};

// Instantiators that return function pointers (the get() pattern)
template <int V> struct SimpleInstantiator {
    static FunctionPtr<int, int> get() {
        return &SimpleInvoker<V>::invoke;
    }
};

template <int V1, char V2> struct TwoAxisInstantiator {
    static FunctionPtr<int, int, char> get() {
        return &TwoAxisInvoker<V1, V2>::invoke;
    }
};
} // namespace

TEST(Dispatcher, TypeAliasesAreCorrect) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    static_assert(std::is_same_v<DispatcherType::axes_type, SingleAxisSpace>);
    static_assert(std::is_same_v<DispatcherType::return_type, int>);
    static_assert(std::is_same_v<DispatcherType::function_ptr_type, int(*)(int)>);
}

TEST(Dispatcher, MapTypeUsesNullptrAsEmptyValue) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;
    using MapType = typename DispatcherType::map_type;

    static_assert(MapType::empty_value == nullptr);
}

TEST(Dispatcher, CanBeDefaultConstructed) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    // All entries should be nullptr
    for (size_t i = 0; i < SingleAxisSpace::size; ++i) {
        EXPECT_EQ(dispatcher.permutation_map.get(i), nullptr);
    }
}

TEST(Dispatcher, ManuallyPopulatedDispatcherWorks) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    dispatcher.permutation_map.set(std::make_tuple(10), &SimpleInvoker<10>::invoke);
    dispatcher.permutation_map.set(std::make_tuple(20), &SimpleInvoker<20>::invoke);
    dispatcher.permutation_map.set(std::make_tuple(30), &SimpleInvoker<30>::invoke);

    // Call through the dispatcher
    EXPECT_EQ(dispatcher(10), 100 + 10); // 10*10 + 10
    EXPECT_EQ(dispatcher(20), 200 + 20); // 20*10 + 20
    EXPECT_EQ(dispatcher(30), 300 + 30); // 30*10 + 30
}

TEST(Dispatcher, ThrowsForUnboundEntry) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    dispatcher.permutation_map.set(std::make_tuple(10), &SimpleInvoker<10>::invoke);
    // 20 and 30 not bound

    EXPECT_EQ(dispatcher(10), 110);
    EXPECT_THROW(dispatcher(20), std::runtime_error);
    EXPECT_THROW(dispatcher(30), std::runtime_error);
}

TEST(Dispatcher, UnboundHandlerReceivesEncodedValues) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, TrackingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    // Nothing bound

    TrackingUnboundHandler::reset();

    try {
        dispatcher(20);
        FAIL() << "Should have thrown";
    } catch (...) {
        // Expected
    }

    EXPECT_TRUE(TrackingUnboundHandler::called);
    EXPECT_EQ(TrackingUnboundHandler::lastValue, 20);
}

// =============================================================================
// build_dispatcher Tests
// =============================================================================

TEST(BuildDispatcher, CreatesDispatcherFromPointGenerators) {
    using Generators = GeneratorList<
        PointGenerator<SimpleInstantiator, 10>,
        PointGenerator<SimpleInstantiator, 20>,
        PointGenerator<SimpleInstantiator, 30>
    >;

    auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    EXPECT_EQ(dispatcher(10), 110);
    EXPECT_EQ(dispatcher(20), 220);
    EXPECT_EQ(dispatcher(30), 330);
}

TEST(BuildDispatcher, CreatesDispatcherFromSubspaceGenerator) {
    using Generators = SubspaceGenerator<SimpleInstantiator, SingleAxisSpace>;

    auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    EXPECT_EQ(dispatcher(10), 110);
    EXPECT_EQ(dispatcher(20), 220);
    EXPECT_EQ(dispatcher(30), 330);
}

TEST(BuildDispatcher, WorksWithTwoAxisSpace) {
    struct TwoAxisEncoderDirect {
        static std::tuple<int, char> encode(int first, char second) {
            return std::make_tuple(first, second);
        }
    };

    using Generators = GeneratorList<
        PointGenerator<TwoAxisInstantiator, 1, 'a'>,
        PointGenerator<TwoAxisInstantiator, 2, 'b'>
    >;

    auto dispatcher = build_dispatcher<
        TwoAxisSpace,
        Generators,
        TwoAxisEncoderDirect,
        TwoAxisThrowingHandler,
        int,
        int, char
    >();

    // TwoAxisInvoker<1, 'a'>::invoke(5, 'x') = 1*1000 + 97 + 5 + 120 = 1222
    EXPECT_EQ(dispatcher(1, 'a'), 1000 + 97 + 1 + 97); // invoker adds x and c to result
    EXPECT_EQ(dispatcher(2, 'b'), 2000 + 98 + 2 + 98);

    // Unbound entry
    EXPECT_THROW(dispatcher(3, 'a'), std::runtime_error);
}

// =============================================================================
// GetFromInvoke Tests (The potentially problematic mechanism)
// =============================================================================

// The GetFromInvoke pattern transforms a template with invoke() into one with get()
// This is currently commented out in the header due to nvcc crashes, but we can test
// the pattern directly to verify the concept works.

namespace {
// An invoker template (what we have)
template <int V> struct MyInvoker {
    static int invoke(int x) { return V * 100 + x; }
};

// Manual implementation of GetFromInvoke for testing
template <template <auto...> typename InvokerTemplate>
struct TestGetFromInvoke {
    template <auto... Values> struct fromInvoke {
        static constexpr auto get() {
            return &InvokerTemplate<Values...>::invoke;
        }
    };
};

// Verify the wrapped type has the expected signature
using WrappedInvoker10 = TestGetFromInvoke<MyInvoker>::template fromInvoke<10>;
} // namespace

TEST(GetFromInvoke, WrappedInvokerReturnsCorrectFunctionPointer) {
    auto ptr = WrappedInvoker10::get();

    static_assert(std::is_same_v<decltype(ptr), int(*)(int)>);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr(5), 1005); // 10*100 + 5
}

TEST(GetFromInvoke, CanBeUsedWithPointGenerator) {
    // Note: fromInvoke is a class template, so we use it directly as a template template arg
    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<int, int>, nullptr>;

    Map map;
    PointGenerator<TestGetFromInvoke<MyInvoker>::fromInvoke, 10>::apply(map);
    PointGenerator<TestGetFromInvoke<MyInvoker>::fromInvoke, 20>::apply(map);
    PointGenerator<TestGetFromInvoke<MyInvoker>::fromInvoke, 30>::apply(map);

    auto ptr10 = map.get(std::make_tuple(10));
    auto ptr20 = map.get(std::make_tuple(20));
    auto ptr30 = map.get(std::make_tuple(30));

    ASSERT_NE(ptr10, nullptr);
    ASSERT_NE(ptr20, nullptr);
    ASSERT_NE(ptr30, nullptr);

    EXPECT_EQ(ptr10(5), 1005);
    EXPECT_EQ(ptr20(5), 2005);
    EXPECT_EQ(ptr30(5), 3005);
}

TEST(GetFromInvoke, CanBeUsedWithSubspaceGenerator) {
    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<int, int>, nullptr>;

    Map map;
    SubspaceGenerator<TestGetFromInvoke<MyInvoker>::fromInvoke, SingleAxisSpace>::apply(map);

    for (int v : {10, 20, 30}) {
        auto ptr = map.get(std::make_tuple(v));
        ASSERT_NE(ptr, nullptr) << "Pointer for value " << v << " is null";
        EXPECT_EQ(ptr(7), v * 100 + 7);
    }
}

TEST(GetFromInvoke, FullDispatcherIntegration) {
    using Generators = SubspaceGenerator<TestGetFromInvoke<MyInvoker>::fromInvoke, SingleAxisSpace>;

    auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    EXPECT_EQ(dispatcher(10), 1010); // 10*100 + 10
    EXPECT_EQ(dispatcher(20), 2020); // 20*100 + 20
    EXPECT_EQ(dispatcher(30), 3030); // 30*100 + 30
}

// =============================================================================
// InvokeToGet Tests (The new cleaner design)
// =============================================================================
// InvokeToGet is the production version that allows clean template aliasing.
// Unlike the nested GetFromInvoke pattern, InvokeToGet takes both the invoker
// template AND the values as template arguments, allowing:
//   template <auto... Vs> using MyInst = InvokeToGet<MyInvoker, Vs...>;

namespace {
// Invoker for InvokeToGet tests
template <int V> struct ITGInvoker {
    static int invoke(int x) { return V * 100 + x; }
};

// The key improvement: users can create clean template aliases
template <auto... Vs>
using ITGInstantiator = InvokeToGet<ITGInvoker, Vs...>;

// Multi-axis version
template <int V1, char V2> struct ITGMultiAxisInvoker {
    static std::string invoke(int x) {
        return std::to_string(V1) + V2 + std::to_string(x);
    }
};

template <auto... Vs>
using ITGMultiAxisInstantiator = InvokeToGet<ITGMultiAxisInvoker, Vs...>;
} // namespace

TEST(InvokeToGet, DirectInstantiationWorks) {
    // Can instantiate directly with explicit values
    auto ptr = InvokeToGet<ITGInvoker, 10>::get();

    static_assert(std::is_same_v<decltype(ptr), int(*)(int)>);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr(5), 1005); // 10*100 + 5
}

TEST(InvokeToGet, TemplateAliasWorks) {
    // The key feature: template alias is usable
    auto ptr10 = ITGInstantiator<10>::get();
    auto ptr20 = ITGInstantiator<20>::get();

    EXPECT_EQ(ptr10(5), 1005);
    EXPECT_EQ(ptr20(5), 2005);
}

TEST(InvokeToGet, CanBeUsedWithPointGenerator) {
    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<int, int>, nullptr>;

    Map map;
    // Use the template alias with PointGenerator
    PointGenerator<ITGInstantiator, 10>::apply(map);
    PointGenerator<ITGInstantiator, 20>::apply(map);
    PointGenerator<ITGInstantiator, 30>::apply(map);

    auto ptr10 = map.get(std::make_tuple(10));
    auto ptr20 = map.get(std::make_tuple(20));
    auto ptr30 = map.get(std::make_tuple(30));

    ASSERT_NE(ptr10, nullptr);
    ASSERT_NE(ptr20, nullptr);
    ASSERT_NE(ptr30, nullptr);

    EXPECT_EQ(ptr10(5), 1005);
    EXPECT_EQ(ptr20(5), 2005);
    EXPECT_EQ(ptr30(5), 3005);
}

TEST(InvokeToGet, CanBeUsedWithSubspaceGenerator) {
    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<int, int>, nullptr>;

    Map map;
    SubspaceGenerator<ITGInstantiator, SingleAxisSpace>::apply(map);

    for (int v : {10, 20, 30}) {
        auto ptr = map.get(std::make_tuple(v));
        ASSERT_NE(ptr, nullptr) << "Pointer for value " << v << " is null";
        EXPECT_EQ(ptr(7), v * 100 + 7);
    }
}

TEST(InvokeToGet, FullDispatcherIntegration) {
    using Generators = SubspaceGenerator<ITGInstantiator, SingleAxisSpace>;

    auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    EXPECT_EQ(dispatcher(10), 1010); // 10*100 + 10
    EXPECT_EQ(dispatcher(20), 2020);
    EXPECT_EQ(dispatcher(30), 3030);
}

TEST(InvokeToGet, PreservesFunctionPointerAddress) {
    // Verify the address is exactly the invoke function
    auto fromInvokeToGet = InvokeToGet<ITGInvoker, 10>::get();
    auto direct = &ITGInvoker<10>::invoke;

    EXPECT_EQ(fromInvokeToGet, direct);
}

TEST(InvokeToGet, MultiAxisTemplateAliasWorks) {
    auto ptr1a = ITGMultiAxisInstantiator<1, 'a'>::get();
    auto ptr2b = ITGMultiAxisInstantiator<2, 'b'>::get();

    EXPECT_EQ(ptr1a(5), "1a5");
    EXPECT_EQ(ptr2b(10), "2b10");
}

TEST(InvokeToGet, MultiAxisWithSubspaceGenerator) {
    using Map = PermutationArrayMap<TwoAxisSpace, FunctionPtr<std::string, int>, nullptr>;

    Map map;
    SubspaceGenerator<ITGMultiAxisInstantiator, TwoAxisSpace>::apply(map);

    // Verify all 6 combinations are populated
    for (int i : {1, 2, 3}) {
        for (char c : {'a', 'b'}) {
            auto ptr = map.get(std::make_tuple(i, c));
            ASSERT_NE(ptr, nullptr) << "Null for (" << i << ", " << c << ")";
            EXPECT_EQ(ptr(0), std::to_string(i) + c + "0");
        }
    }
}

TEST(InvokeToGet, MultiAxisFullDispatcher) {
    // For multi-axis dispatch, axis values are template params, runtime args are separate.
    // Here we dispatch on (int, char) but the function only takes an int.
    // The encoder extracts axis values from somewhere (here we use the int arg as axis1,
    // and a fixed char as axis2 for demonstration).

    struct MultiAxisEncoder {
        // This encoder uses the int arg to select axis1, fixed 'a'/'b' based on value
        static std::tuple<int, char> encode(int x) {
            // Map: 1->'a', 2->'b', 3->'a' for testing
            char c = (x == 2) ? 'b' : 'a';
            return std::make_tuple(x, c);
        }
    };

    struct MultiAxisUnbound {
        [[noreturn]] static void unbound(int, char) {
            throw std::runtime_error("unbound");
        }
    };

    using Generators = SubspaceGenerator<ITGMultiAxisInstantiator, TwoAxisSpace>;

    auto dispatcher = build_dispatcher<
        TwoAxisSpace,
        Generators,
        MultiAxisEncoder,
        MultiAxisUnbound,
        std::string,
        int  // Only int arg at runtime, encoder derives both axes from it
    >();

    // ITGMultiAxisInvoker<1, 'a'>::invoke(1) = "1a1"
    EXPECT_EQ(dispatcher(1), "1a1");
    // ITGMultiAxisInvoker<2, 'b'>::invoke(2) = "2b2"
    EXPECT_EQ(dispatcher(2), "2b2");
    // ITGMultiAxisInvoker<3, 'a'>::invoke(3) = "3a3"
    EXPECT_EQ(dispatcher(3), "3a3");
}

// =============================================================================
// Function Pointer Type Matching Tests
// =============================================================================

// These tests verify that the function pointer types are correctly matched
// between the instantiator, the map, and the dispatcher.

namespace {
// Invokers with different signatures
template <int V> struct VoidReturnInvoker {
    static void invoke(int& out) { out = V; }
};

template <int V> struct VoidReturnInstantiator {
    static FunctionPtr<void, int&> get() {
        return &VoidReturnInvoker<V>::invoke;
    }
};

template <int V> struct ConstRefInvoker {
    static int invoke(const std::string& s) { return V + static_cast<int>(s.length()); }
};

template <int V> struct ConstRefInstantiator {
    static FunctionPtr<int, const std::string&> get() {
        return &ConstRefInvoker<V>::invoke;
    }
};
} // namespace

TEST(FunctionPtrTypeMatching, VoidReturnTypesWork) {
    struct VoidEncoder {
        static std::tuple<int> encode(int& /*out*/) { return std::make_tuple(10); }
    };

    struct VoidUnbound {
        [[noreturn]] static void unbound(int) { throw std::runtime_error("unbound"); }
    };

    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<void, int&>, nullptr>;

    Map map;
    PointGenerator<VoidReturnInstantiator, 10>::apply(map);

    int result = 0;
    auto ptr = map.get(std::make_tuple(10));
    ASSERT_NE(ptr, nullptr);
    ptr(result);
    EXPECT_EQ(result, 10);
}

TEST(FunctionPtrTypeMatching, ConstRefArgumentsWork) {
    using Map = PermutationArrayMap<SingleAxisSpace, FunctionPtr<int, const std::string&>, nullptr>;

    Map map;
    PointGenerator<ConstRefInstantiator, 10>::apply(map);
    PointGenerator<ConstRefInstantiator, 20>::apply(map);

    auto ptr10 = map.get(std::make_tuple(10));
    auto ptr20 = map.get(std::make_tuple(20));

    ASSERT_NE(ptr10, nullptr);
    ASSERT_NE(ptr20, nullptr);

    EXPECT_EQ(ptr10("hello"), 10 + 5);
    EXPECT_EQ(ptr20("hi"), 20 + 2);
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

TEST(DispatcherEdgeCases, InvalidEncodedValueReturnsNullptr) {
    // If the encoder returns a value not in the axis, the lookup should fail
    struct BadEncoder {
        static std::tuple<int> encode(int x) {
            // This encoder might return a value not in SingleAxis (10, 20, 30)
            return std::make_tuple(x);
        }
    };

    using DispatcherType = Dispatcher<SingleAxisSpace, BadEncoder, ThrowingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    dispatcher.permutation_map.set(std::make_tuple(10), &SimpleInvoker<10>::invoke);

    // Valid encoded value
    EXPECT_EQ(dispatcher(10), 110);

    // Invalid encoded value (15 is not in the axis)
    EXPECT_THROW(dispatcher(15), std::runtime_error);
}

TEST(DispatcherEdgeCases, EmptyDispatcherThrowsForAllCalls) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    DispatcherType dispatcher;
    // Nothing bound

    EXPECT_THROW(dispatcher(10), std::runtime_error);
    EXPECT_THROW(dispatcher(20), std::runtime_error);
    EXPECT_THROW(dispatcher(30), std::runtime_error);
}

TEST(DispatcherEdgeCases, PartiallyBoundDispatcher) {
    using Generators = GeneratorList<
        PointGenerator<SimpleInstantiator, 10>
        // 20 and 30 not bound
    >;

    auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    EXPECT_EQ(dispatcher(10), 110);
    EXPECT_THROW(dispatcher(20), std::runtime_error);
    EXPECT_THROW(dispatcher(30), std::runtime_error);
}

// =============================================================================
// Function Pointer Address Verification
// =============================================================================

// These tests verify that we're storing and retrieving the exact same
// function pointer addresses - no type-punning or reinterpret_cast issues.

TEST(FunctionPtrAddress, StoredAddressMatchesOriginal) {
    using FPtr = FunctionPtr<int, int>;
    using Map = PermutationArrayMap<SingleAxisSpace, FPtr, nullptr>;

    Map map;

    FPtr originalPtr10 = &SimpleInvoker<10>::invoke;
    FPtr originalPtr20 = &SimpleInvoker<20>::invoke;

    map.set(std::make_tuple(10), originalPtr10);
    map.set(std::make_tuple(20), originalPtr20);

    EXPECT_EQ(map.get(std::make_tuple(10)), originalPtr10);
    EXPECT_EQ(map.get(std::make_tuple(20)), originalPtr20);

    // Verify the pointers are actually different
    EXPECT_NE(originalPtr10, originalPtr20);
}

TEST(FunctionPtrAddress, GetFromInvokePreservesAddress) {
    // Get the function pointer through the GetFromInvoke wrapper
    auto wrappedPtr = TestGetFromInvoke<MyInvoker>::fromInvoke<10>::get();

    // Get the function pointer directly
    auto directPtr = &MyInvoker<10>::invoke;

    // They should be the exact same address
    EXPECT_EQ(wrappedPtr, directPtr);
}

// =============================================================================
// Multi-Axis GetFromInvoke Tests
// =============================================================================

namespace {
template <int V1, char V2> struct MultiAxisInvoker {
    static std::string invoke(int x) {
        return std::to_string(V1) + V2 + std::to_string(x);
    }
};

template <template <auto...> typename InvokerTemplate>
struct MultiAxisGetFromInvoke {
    template <auto V1, auto V2> struct fromInvoke {
        static constexpr auto get() {
            return &InvokerTemplate<V1, V2>::invoke;
        }
    };
};
} // namespace

TEST(GetFromInvokeMultiAxis, WorksWithTwoParameters) {
    auto ptr1a = MultiAxisGetFromInvoke<MultiAxisInvoker>::fromInvoke<1, 'a'>::get();
    auto ptr2b = MultiAxisGetFromInvoke<MultiAxisInvoker>::fromInvoke<2, 'b'>::get();

    EXPECT_EQ(ptr1a(5), "1a5");
    EXPECT_EQ(ptr2b(10), "2b10");
}

TEST(GetFromInvokeMultiAxis, CanBeUsedWithPointGenerator) {
    using Map = PermutationArrayMap<TwoAxisSpace, FunctionPtr<std::string, int>, nullptr>;

    Map map;
    PointGenerator<MultiAxisGetFromInvoke<MultiAxisInvoker>::fromInvoke, 1, 'a'>::apply(map);
    PointGenerator<MultiAxisGetFromInvoke<MultiAxisInvoker>::fromInvoke, 2, 'b'>::apply(map);

    auto ptr1a = map.get(std::make_tuple(1, 'a'));
    auto ptr2b = map.get(std::make_tuple(2, 'b'));

    ASSERT_NE(ptr1a, nullptr);
    ASSERT_NE(ptr2b, nullptr);

    EXPECT_EQ(ptr1a(7), "1a7");
    EXPECT_EQ(ptr2b(8), "2b8");
}

TEST(GetFromInvokeMultiAxis, CanBeUsedWithSubspaceGenerator) {
    using Map = PermutationArrayMap<TwoAxisSpace, FunctionPtr<std::string, int>, nullptr>;

    Map map;
    SubspaceGenerator<MultiAxisGetFromInvoke<MultiAxisInvoker>::fromInvoke, TwoAxisSpace>::apply(map);

    // Verify all 6 combinations are populated
    for (int i : {1, 2, 3}) {
        for (char c : {'a', 'b'}) {
            auto ptr = map.get(std::make_tuple(i, c));
            ASSERT_NE(ptr, nullptr) << "Null for (" << i << ", " << c << ")";
            EXPECT_EQ(ptr(0), std::to_string(i) + c + "0");
        }
    }
}

// =============================================================================
// Static Dispatch Table Tests (const storage pattern)
// =============================================================================

TEST(StaticDispatchTable, CanBeStoredAsStaticConst) {
    using Generators = SubspaceGenerator<SimpleInstantiator, SingleAxisSpace>;

    static const auto dispatcher = build_dispatcher<
        SingleAxisSpace,
        Generators,
        SingleAxisEncoder,
        ThrowingUnboundHandler,
        int,
        int
    >();

    // Should work the same as a non-static dispatcher
    EXPECT_EQ(dispatcher(10), 110);
    EXPECT_EQ(dispatcher(20), 220);
    EXPECT_EQ(dispatcher(30), 330);
}

// =============================================================================
// Type Deduction Verification Tests
// =============================================================================

TEST(TypeDeduction, InstantiatorTypeMatchesMapValueType) {
    using ExpectedFPtr = FunctionPtr<int, int>;

    // Verify SimpleInstantiator::get() returns the expected type
    static_assert(std::is_same_v<decltype(SimpleInstantiator<10>::get()), ExpectedFPtr>);

    // Verify the wrapped GetFromInvoke also returns the expected type
    static_assert(std::is_same_v<decltype(TestGetFromInvoke<MyInvoker>::fromInvoke<10>::get()), ExpectedFPtr>);

    // Verify the new InvokeToGet returns the expected type
    static_assert(std::is_same_v<decltype(InvokeToGet<ITGInvoker, 10>::get()), ExpectedFPtr>);

    // Verify the template alias also works
    static_assert(std::is_same_v<decltype(ITGInstantiator<10>::get()), ExpectedFPtr>);
}

TEST(TypeDeduction, DispatcherFunctionPtrTypeMatchesCallSignature) {
    using DispatcherType = Dispatcher<SingleAxisSpace, SingleAxisEncoder, ThrowingUnboundHandler, int, int>;

    // The dispatcher's function_ptr_type should match what we're storing
    static_assert(std::is_same_v<DispatcherType::function_ptr_type, int(*)(int)>);

    // And the stored functions should be callable with the same signature
    using StoredFPtr = decltype(std::declval<DispatcherType>().permutation_map.get(0));
    static_assert(std::is_same_v<StoredFPtr, DispatcherType::function_ptr_type>);
}

// =============================================================================
// Complex Signature Tests
// =============================================================================

namespace {
// Test with a more complex function signature
template <int V> struct ComplexInvoker {
    static std::tuple<int, std::string> invoke(int x, double d, const std::string& s) {
        return {V + x + static_cast<int>(d), s + std::to_string(V)};
    }
};

template <int V> struct ComplexInstantiator {
    static auto get() {
        return &ComplexInvoker<V>::invoke;
    }
};
} // namespace

TEST(ComplexSignature, WorksWithTupleReturnType) {
    using FPtr = FunctionPtr<std::tuple<int, std::string>, int, double, const std::string&>;
    using Map = PermutationArrayMap<SingleAxisSpace, FPtr, nullptr>;

    Map map;
    PointGenerator<ComplexInstantiator, 10>::apply(map);
    PointGenerator<ComplexInstantiator, 20>::apply(map);

    auto ptr10 = map.get(std::make_tuple(10));
    auto ptr20 = map.get(std::make_tuple(20));

    ASSERT_NE(ptr10, nullptr);
    ASSERT_NE(ptr20, nullptr);

    auto [num10, str10] = ptr10(5, 2.5, "test");
    EXPECT_EQ(num10, 10 + 5 + 2);
    EXPECT_EQ(str10, "test10");

    auto [num20, str20] = ptr20(3, 1.1, "hello");
    EXPECT_EQ(num20, 20 + 3 + 1);
    EXPECT_EQ(str20, "hello20");
}

} // namespace dispatch
} // namespace fvdb
