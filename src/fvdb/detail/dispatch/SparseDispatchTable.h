// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
#define FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H

#include "fvdb/detail/dispatch/PermutationMap.h"

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fvdb {
namespace dispatch {

// What are the key ideas behind the dispatch table? What does it add on top
// of the plain permutation map?
//
// it's a permutation map with a function pointer as the value type.
//
// it will usually be used in situations where the values of the arguments of
// the function are used to produce a the value tuple which is the index type
// of the permutation map.
//
// It is not an invalid usage to call with values that are not in the space
// of the axes, so long as they're the correct types as the axes value types.
//
// The generators, which permutation map wants to have with a static "get"
// function that returns a value, are often more easily expressed as a struct
// with a static "invoke" function which *is* the function pointer, so we want
// an adapter that turns the invoke struct type into the generator. We don't
// need to rebuild the permutation map machinery to support this variation,
// the wrapper invoke_to_get struct is fine.

template <typename ReturnType, typename... Args> using FunctionPtr = ReturnType (*)(Args...);

// -----------------------------------------------------------------------------
// Concepts for Dispatcher components
// -----------------------------------------------------------------------------

// Encoder must have a static encode() method that returns a tuple of axis values.
// Note: Implicit conversions in the arguments are allowed (standard C++ behavior).
// The encoder extracts axis values from runtime args; if those args undergo
// implicit conversion, that's fine and expected.
template <typename Encoder, typename AxesT, typename... Args>
concept EncoderConcept = requires(Args... args) {
    { Encoder::encode(args...) } -> std::convertible_to<typename AxesT::value_types_tuple_type>;
};

// UnboundHandler must have a static unbound() method (typically [[noreturn]])
template <typename Handler, typename AxesT>
concept UnboundHandlerConcept = requires {
    // We can't easily check the exact signature without unpacking the tuple,
    // so we just check the type exists and has a static unbound member
    &Handler::unbound;
};

// -----------------------------------------------------------------------------
// Dispatcher: A dispatch table with baked-in encoder and error handler
// -----------------------------------------------------------------------------
// The encoder maps runtime arguments to the axis value tuple for lookup.
// The error handler is invoked (with the encoded tuple) when no binding exists.
//
// Template parameters:
//   AxesT          - The AxisOuterProduct defining the dispatch space
//   Encoder        - Type with static encode(Args...) -> value_types_tuple_type
//   UnboundHandler - Type with static [[noreturn]] unbound(axis values...)
//   ReturnType     - Return type of the dispatched functions
//   Args...        - Argument types of the dispatched functions

template <typename AxesT,
          typename Encoder,
          typename UnboundHandler,
          typename ReturnType,
          typename... Args>
    requires AxisOuterProductConcept<AxesT> && EncoderConcept<Encoder, AxesT, Args...> &&
             UnboundHandlerConcept<UnboundHandler, AxesT>
struct Dispatcher {
    static_assert(AxisOuterProductConcept<AxesT>, "Dispatcher: AxesT must be an AxisOuterProduct");

    using axes_type         = AxesT;
    using return_type       = ReturnType;
    using function_ptr_type = FunctionPtr<ReturnType, Args...>;
    using map_type          = PermutationMap<AxesT, function_ptr_type, nullptr>;

    map_type permutation_map;

    ReturnType
    operator()(Args... args) const {
        auto const index_as_value_tuple = Encoder::encode(args...);
        auto const function_ptr         = permutation_map.get(index_as_value_tuple);
        if (function_ptr == nullptr) {
            std::apply([](auto... values) { UnboundHandler::unbound(values...); },
                       index_as_value_tuple);
            throw std::runtime_error(
                "Unbound dispatch handler did not throw. Dispatcher: unexpected code path reached.");
        }
        return function_ptr(args...);
    }
};

// -----------------------------------------------------------------------------
// build_dispatcher: Construct a Dispatcher from generator specifications
// -----------------------------------------------------------------------------
// Creates a dispatch table by populating a Dispatcher's permutation map using
// the provided generators. The generators determine which combinations of axis
// values are bound to function pointers.
//
// Template parameters:
//   AxesT          - The AxisOuterProduct defining the dispatch space
//   Generators     - A generator or GeneratorList that populates the map
//   Encoder        - Type with static encode(Args...) -> axes value tuple
//   UnboundHandler - Type with static [[noreturn]] unbound(axis values...)
//   ReturnType     - Return type of the dispatched functions
//   Args...        - Argument types of the dispatched functions
//
// Usage:
//   using MyBindings = GeneratorList<...>;
//   static const auto table = build_dispatcher<MyAxes, MyBindings,
//                                              MyEncoder, MyErrorHandler,
//                                              ReturnT, Arg1, Arg2>();

template <typename AxesT,
          typename Generators,
          typename Encoder,
          typename UnboundHandler,
          typename ReturnType,
          typename... Args>
    requires AxisOuterProductConcept<AxesT> && EncoderConcept<Encoder, AxesT, Args...> &&
             UnboundHandlerConcept<UnboundHandler, AxesT>
Dispatcher<AxesT, Encoder, UnboundHandler, ReturnType, Args...>
build_dispatcher() {
    using dispatcher_type = Dispatcher<AxesT, Encoder, UnboundHandler, ReturnType, Args...>;
    using map_type        = typename dispatcher_type::map_type;

    static_assert(GeneratorForConcept<Generators, map_type>,
                  "Generators must be valid for the dispatcher's map type");

    dispatcher_type dispatcher{};
    Generators::apply(dispatcher.permutation_map);
    return dispatcher;
}

// =============================================================================
// InvokeToGet: Adaptor from invoke()-style to get()-style instantiators
// =============================================================================
// Transforms a template with a static `invoke` function into an instantiator
// with a static `get` function returning the function pointer to `invoke`.
//
// The Invoker template must have:
//   template <auto... Values> struct Invoker {
//       static ReturnType invoke(Args...);  // The actual function to dispatch to
//   };
//
// This produces an Instantiator template usable with PermutationMap generators:
//   template <auto... Values> struct Instantiator {
//       static FunctionPtr<ReturnType, Args...> get();  // Returns &Invoker<Values...>::invoke
//   };
//
// Design: This is a single-level template taking both the invoker template AND
// the values. This allows users to create clean template aliases:
//
//   template <auto... Vs>
//   using MyInstantiator = InvokeToGet<MyInvoker, Vs...>;
//
//   using MyGen = SubspaceGenerator<MyInstantiator, MySubspace>;
//
// This avoids the awkward nested template access pattern that was previously
// required (GetFromInvoke<MyInvoker>::template fromInvoke).

template <template <auto...> typename InvokerTemplate, auto... Values>
struct InvokeToGet {
    static constexpr auto
    get() {
        return &InvokerTemplate<Values...>::invoke;
    }
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
