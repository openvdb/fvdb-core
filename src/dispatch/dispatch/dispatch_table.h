// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// dispatch_table: Sparse-subspace dispatch table with runtime lookup.
// Supports sparse instantiation via subspaces, with factory patterns
// (from_op<Op>() / from_visitor()) for populating entries.
//
#ifndef DISPATCH_DISPATCH_DISPATCH_TABLE_H
#define DISPATCH_DISPATCH_DISPATCH_TABLE_H

#include "dispatch/axes_map.h"
#include "dispatch/detail.h"
#include "dispatch/types.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Exception for dispatch lookup failures
//------------------------------------------------------------------------------

// Exception thrown when dispatch lookup fails (coordinate not in space or no handler).
// This allows callers to distinguish dispatch failures from other runtime errors.
class dispatch_lookup_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

//------------------------------------------------------------------------------
// Sparse dispatch table
//------------------------------------------------------------------------------

// Sparse dispatch table: only instantiates entries for declared subspaces.
// Runtime dispatch via operator() throws dispatch_lookup_error if coordinate not found.
template <typename Axes, typename FunctionSignature> class dispatch_table;

template <typename... Axes, typename ReturnType, typename... Args>
class dispatch_table<axes<Axes...>, ReturnType(Args...)> {
  public:
    using axes_type             = axes<Axes...>;
    using return_type           = ReturnType;
    using function_pointer_type = ReturnType (*)(Args...);
    using coord_tuple_type      = value_tuple_type_t<axes_type>;
    using map_type              = axes_map<axes_type, function_pointer_type>;

    // Default constructor - empty table
    dispatch_table() : _data(std::make_shared<map_type const>()) {}

    // Rule of zero - all default (shared_ptr handles everything)
    dispatch_table(dispatch_table const &)            = default;
    dispatch_table(dispatch_table &&)                 = default;
    dispatch_table &operator=(dispatch_table const &) = default;
    dispatch_table &operator=(dispatch_table &&)      = default;
    ~dispatch_table()                                 = default;

    // Initial construction with factory and coordinates or subspaces
    template <typename Factory, typename... Subs>
    explicit dispatch_table(Factory &&factory, Subs... subs) {
        static_assert((within<Subs, axes_type> && ... && true), "Subs must be within the axes");

        // Can't put it in _data just yet, because _data is const.
        auto mutable_data = std::make_unique<map_type>();
        create_and_store(*mutable_data, factory, subs...);

        // Then we make it shared to const
        _data = std::shared_ptr<map_type const>{std::move(mutable_data)};
    }

    // Functional update - returns new table with additional entries
    template <typename Factory, typename... Subs>
    dispatch_table
    with(Factory &&factory, Subs... subs) const {
        static_assert((within<Subs, axes_type> && ... && true), "Subs must be within the axes");

        // We make a mutable copy of the map data first, that we can modify.
        auto mutable_data = std::make_unique<map_type>(*_data);
        create_and_store(*mutable_data, factory, subs...);

        // Then we convert it to shared const
        std::shared_ptr<map_type const> const_data{std::move(mutable_data)};

        // And move it along.
        return dispatch_table(const_data);
    }

    // Runtime dispatch: looks up coordinate and invokes registered handler.
    // Throws dispatch_lookup_error if coordinate not found or handler is null.
    return_type
    operator()(coord_tuple_type const &coord_tuple, Args... args) const {
        auto const it = _data->find(coord_tuple);
        if (it == _data->end()) {
            throw dispatch_lookup_error("Dispatch failed: coordinate not in space");
        }
        if (it->second == nullptr) {
            throw dispatch_lookup_error("Dispatch failed: no handler registered for coordinate");
        }
        function_pointer_type fptr = it->second;
        return fptr(args...);
    }

    // Factory for struct-based dispatch: Op must have static op(tag<...>, args...)
    // overloads matching the dispatch space coordinates.
    template <typename Op>
    static auto
    from_op() {
        return [](auto coord) -> function_pointer_type {
            using C = decltype(coord);
            return &op_call<Op, C>;
        };
    }

    // Factory for visitor-based dispatch: creates dispatch entries from a visitor type.
    //
    // IMPORTANT: The visitor instance is used ONLY for type deduction. A fresh
    // Visitor{} is default-constructed for each dispatch call. This means:
    //   - Capturing lambdas are NOT supported (they aren't default-constructible)
    //   - Stateful visitors will lose their state
    //   - Only stateless, default-constructible visitors work correctly
    //
    // This design enables the std::visit "overloaded" idiom (C++20):
    //
    //   // Helper type for combining multiple lambdas into one visitor
    //   template<class... Ts>
    //   struct overloaded : Ts... { using Ts::operator()...; };
    //
    //   // Usage with dispatch_table (analogous to std::visit):
    //   auto table = dispatch_table<MyAxes, void(int)>(
    //       dispatch_table<MyAxes, void(int)>::from_visitor(overloaded{
    //           [](tag<A>, int x) { /* handle A */ },
    //           [](tag<B>, int x) { /* handle B */ },
    //           [](auto, int x)   { /* fallback */ }
    //       }),
    //       full_space<MyAxes>
    //   );
    //
    // In C++20, non-capturing lambdas are default-constructible, so the
    // overloaded type inheriting from them is also default-constructible.
    //
    // See: https://en.cppreference.com/w/cpp/utility/variant/visit
    //
    template <typename Visitor>
    static auto
    from_visitor(Visitor) {
        static_assert(std::is_default_constructible_v<Visitor>,
                      "Visitor must be default-constructible. The passed instance is used "
                      "only for type deduction; a fresh Visitor{} is constructed for each "
                      "dispatch call. Capturing lambdas are not supported.");

        return [](auto coord) -> function_pointer_type {
            using C = decltype(coord);
            return &visitor_call<Visitor, C>;
        };
    }

  private:
    std::shared_ptr<map_type const> _data;

    // Private constructor for internal use (from with())
    explicit dispatch_table(std::shared_ptr<map_type const> data) : _data(data) {}

    template <typename Op, typename Coord>
    static return_type
    op_call(Args... args) {
        return Op::op(Coord{}, args...);
    }

    template <typename Visitor, typename Coord>
    static return_type
    visitor_call(Args... args) {
        return Visitor{}(Coord{}, args...);
    }
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_DISPATCH_TABLE_H
