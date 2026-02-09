// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// dispatch_table: Sparse-subspace dispatch table with runtime lookup.
// Supports sparse instantiation via subspaces, with factory patterns
// (from_op<Op>() / from_visitor()) for populating entries.
//
// Dispatch separates "select" from "invoke":
//   auto fn = table.select(dispatch_set{dev, stype, contig});
//   fn(input, output);
//
#ifndef DISPATCH_DISPATCH_DISPATCH_TABLE_H
#define DISPATCH_DISPATCH_DISPATCH_TABLE_H

#include "dispatch/detail/axes_map.h"
#include "dispatch/detail/index_math.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// dispatch_set: curly-brace constructible runtime dispatch coordinates
//------------------------------------------------------------------------------

template <typename... Ts> struct dispatch_set {
    std::tuple<Ts...> values;

    constexpr dispatch_set(Ts... args) : values{args...} {}

    template <typename T>
    constexpr T const &
    get() const {
        return std::get<T>(values);
    }
};

// CTAD
template <typename... Ts> dispatch_set(Ts...) -> dispatch_set<Ts...>;

//------------------------------------------------------------------------------
// Exception for dispatch lookup failures
//------------------------------------------------------------------------------

class dispatch_lookup_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

//------------------------------------------------------------------------------
// Sparse dispatch table
//------------------------------------------------------------------------------

template <typename Axes, typename FunctionSignature> class dispatch_table;

template <typename... Axes, typename ReturnType, typename... Args>
class dispatch_table<axes_storage<Axes...>, ReturnType(Args...)> {
  public:
    using axes_type             = axes_storage<Axes...>;
    using return_type           = ReturnType;
    using function_pointer_type = ReturnType (*)(Args...);
    using coord_tuple_type      = value_tuple_type_t<axes_type>;
    using map_type              = axes_map<axes_type, function_pointer_type>;

    // Default constructor - empty table
    explicit dispatch_table(std::string_view name)
        : _name(name), _data(std::make_shared<map_type const>()) {}

    // Rule of zero - all default (shared_ptr handles everything)
    dispatch_table(dispatch_table const &)            = default;
    dispatch_table(dispatch_table &&)                 = default;
    dispatch_table &operator=(dispatch_table const &) = default;
    dispatch_table &operator=(dispatch_table &&)      = default;
    ~dispatch_table()                                 = default;

    // Construction with name, factory, and coordinates or subspaces
    template <typename Factory, typename... Subs>
    explicit dispatch_table(std::string_view name, Factory &&factory, Subs... subs) : _name(name) {
        static_assert((within<Subs, axes_type> && ... && true), "Subs must be within the axes");

        auto mutable_data = std::make_unique<map_type>();
        create_and_store(*mutable_data, factory, subs...);
        _data = std::shared_ptr<map_type const>{std::move(mutable_data)};
    }

    // Functional update - returns new table with additional entries
    template <typename Factory, typename... Subs>
    dispatch_table
    with(Factory &&factory, Subs... subs) const {
        static_assert((within<Subs, axes_type> && ... && true), "Subs must be within the axes");

        auto mutable_data = std::make_unique<map_type>(*_data);
        create_and_store(*mutable_data, factory, subs...);

        std::shared_ptr<map_type const> const_data{std::move(mutable_data)};
        return dispatch_table(_name, const_data);
    }

    // ---- select / try_select ----
    // Takes a dispatch_set, matches values to axes by type, and looks up
    // the corresponding function pointer.

    // select: returns function pointer, throws on failure.
    template <typename... Ts>
    function_pointer_type
    select(dispatch_set<Ts...> const &ds) const {
        auto const coord = reorder_dispatch_set(ds);
        return select_canonical(coord);
    }

    // try_select: returns function pointer or nullptr on failure.
    template <typename... Ts>
    function_pointer_type
    try_select(dispatch_set<Ts...> const &ds) const {
        auto const coord = reorder_dispatch_set(ds);
        return try_select_canonical(coord);
    }

    // ---- Name accessor ----

    std::string_view
    name() const {
        return _name;
    }

    // ---- Factories ----

    template <typename Op>
    static auto
    from_op() {
        return [](auto coord) -> function_pointer_type {
            using C = decltype(coord);
            return &op_call<Op, C>;
        };
    }

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
    std::string _name;
    std::shared_ptr<map_type const> _data;

    // Private constructor for internal use (from with())
    explicit dispatch_table(std::string_view name, std::shared_ptr<map_type const> data)
        : _name(name), _data(data) {}

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

    // ---- internal lookup by canonically ordered tuple ----

    function_pointer_type
    select_canonical(coord_tuple_type const &coord_tuple) const {
        auto const it = _data->find(coord_tuple);
        if (it == _data->end()) {
            throw dispatch_lookup_error(std::string(_name) +
                                        ": dispatch failed — coordinate not in space");
        }
        if (it->second == nullptr) {
            throw dispatch_lookup_error(std::string(_name) +
                                        ": dispatch failed — no handler registered");
        }
        return it->second;
    }

    function_pointer_type
    try_select_canonical(coord_tuple_type const &coord_tuple) const {
        auto const it = _data->find(coord_tuple);
        if (it == _data->end()) {
            return nullptr;
        }
        return it->second;
    }

    // Reorder dispatch_set values to match axes order.
    template <typename... Ts>
    static coord_tuple_type
    reorder_dispatch_set(dispatch_set<Ts...> const &ds) {
        return std::make_tuple(ds.template get<axis_value_type_t<Axes>>()...);
    }
};

//------------------------------------------------------------------------------
// coverage
//------------------------------------------------------------------------------

template <typename... Subs> struct coverage {};

//------------------------------------------------------------------------------
// dispatch_table_from_op
//------------------------------------------------------------------------------

namespace detail {

template <typename Op, typename Coverage> struct dispatch_table_from_op_helper;

template <typename Op, typename... Subs>
struct dispatch_table_from_op_helper<Op, coverage<Subs...>> {
    static typename Op::dispatcher
    create(std::string_view name) {
        return typename Op::dispatcher{name, Op::dispatcher::template from_op<Op>(), Subs{}...};
    }
};

} // namespace detail

template <typename Op>
typename Op::dispatcher
dispatch_table_from_op(std::string_view name) {
    return detail::dispatch_table_from_op_helper<Op, typename Op::subspaces>::create(name);
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_DISPATCH_TABLE_H
