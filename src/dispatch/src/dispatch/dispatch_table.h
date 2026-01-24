// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_DISPATCH_TABLE_H
#define DISPATCH_DISPATCH_TABLE_H

#include "dispatch/axes_map.h"
#include "dispatch/detail.h"
#include "dispatch/types.h"

#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace dispatch {

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

    // Dispatch - throws if coordinate not found or handler is null
    return_type
    operator()(coord_tuple_type const &coord_tuple, Args... args) const {
        auto const it = _data->find(coord_tuple);
        if (it == _data->end()) {
            throw std::runtime_error("Dispatch failed: coordinate not in space");
        }
        if (it->second == nullptr) {
            throw std::runtime_error("Dispatch failed: no handler registered for coordinate");
        }
        function_pointer_type fptr = it->second;
        return fptr(args...);
    }

    // For struct with static op() overloads
    template <typename Op>
    static auto
    from_op() {
        return [](auto coord) -> function_pointer_type {
            using C = decltype(coord);
            return &op_call<Op, C>;
        };
    }

    // For overloaded/visitor pattern (default-constructible functor)
    template <typename Visitor>
    static auto
    from_visitor(Visitor) {
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

#endif // DISPATCH_DISPATCH_TABLE_H
