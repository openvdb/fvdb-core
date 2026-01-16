// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
#define FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H

#include "fvdb/detail/dispatch/ValueSpace.h"
#include "fvdb/detail/dispatch/ValueSpaceMap.h"
#include "fvdb/detail/dispatch/Values.h"

#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

template <ValueSpace Space, typename FunctionSignature> class DispatchTable;

template <ValueSpace Space, typename ReturnType, typename... Args>
class DispatchTable<Space, ReturnType(Args...)> {
  public:
    using return_type           = ReturnType;
    using function_pointer_type = ReturnType (*)(Args...);
    using coord_tuple_type      = SpaceTupleType_t<Space>;
    using map_type              = ValueSpaceMap_t<Space, function_pointer_type>;

  private:
    std::shared_ptr<map_type const> _data;

    // Private constructor for internal use (from with())
    explicit DispatchTable(std::shared_ptr<map_type const> data) : _data(std::move(data)) {}

  public:
    // Default constructor - empty table
    DispatchTable() : _data(std::make_shared<map_type>()) {}

    // Rule of zero - all default (shared_ptr handles everything)
    DispatchTable(DispatchTable const &)            = default;
    DispatchTable(DispatchTable &&)                 = default;
    DispatchTable &operator=(DispatchTable const &) = default;
    DispatchTable &operator=(DispatchTable &&)      = default;
    ~DispatchTable()                                = default;

    // Initial construction with factory and coordinates or subspaces
    template <typename Factory, typename... Subs>
        requires(SpaceCovers<Space, Subs> && ...)
    explicit DispatchTable(Factory &&factory, Subs... subs) {
        auto newData = std::make_shared<map_type>();
        create_and_store(*newData, factory, subs...);
        _data = std::move(newData);
    }

    // Functional update - returns new table with additional entries
    template <typename Factory, typename... Subs>
        requires(SpaceCovers<Space, Subs> && ...)
    DispatchTable
    with(Factory &&factory, Subs... subs) const {
        auto newData = std::make_shared<map_type>(*_data); // copy existing entries
        create_and_store(*newData, factory, subs...);
        return DispatchTable(std::move(newData));
    }

    // Dispatch - throws if coordinate not found or handler is null
    ReturnType
    operator()(coord_tuple_type const &coordTuple, Args... args) const {
        auto const it = _data->find(coordTuple);
        if (it == _data->end()) {
            throw std::runtime_error("Dispatch failed: coordinate not in space");
        }
        if (it->second == nullptr) {
            throw std::runtime_error("Dispatch failed: no handler registered for coordinate");
        }
        function_pointer_type const fptr = it->second;
        return fptr(args...);
    }
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_SPARSEDISPATCHTABLE_H
