// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_VALUESPACEMAP_H
#define FVDB_DETAIL_DISPATCH_VALUESPACEMAP_H

#include "fvdb/detail/dispatch/ValueSpace.h"

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace fvdb {
namespace dispatch {

// ----------------------------------------------------------------------
// ValueSpaceMap: A map keyed by coordinates in a ValueSpace
// ----------------------------------------------------------------------
//
// This map enforces that only valid coordinates (tuples within the space) can
// be inserted, but allows querying with any tuple. Invalid queries gracefully
// return map.end() rather than throwing. This asymmetry makes it ideal as a
// foundation for dispatch tables:
//
//   - INSERT/EMPLACE: Throws if the coordinate is not in the space.
//   - FIND/LOOKUP:    Returns map.end() if the coordinate is not in the space.
//
// Usage:
//
//   using MySpace = ValueAxes<DeviceAxis, ScalarAxis>;
//   ValueSpaceMap_t<MySpace, std::function<void()>> dispatchTable;
//
//   auto validCoord   = std::make_tuple(Device::CUDA, Scalar::Float);
//   auto invalidCoord = std::make_tuple(Device::CUDA, static_cast<Scalar>(999));
//
//   // --- Lookup with invalid coordinate: graceful failure ---
//   auto it = dispatchTable.find(invalidCoord);
//   if (it == dispatchTable.end()) {
//       // Not found (no exception thrown)
//   }
//
//   // --- Insert with invalid coordinate: throws ---
//   try {
//       dispatchTable.emplace(invalidCoord, []{ /* ... */ });
//   } catch (const std::exception& e) {
//       // "Insert Error: Coordinate is not in space!"
//   }
//
//   // --- Insert with valid coordinate: succeeds ---
//   dispatchTable.emplace(validCoord, []{ std::cout << "dispatched!\n"; });
//
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// ValueSpaceMapKey
// ----------------------------------------------------------------------
// The Key is only instantiated when an insert or an emplace is called,
// it is not instantiated for a find. It is also instantiated with
// operator[] (which takes a key).

template <ValueSpace Space> struct ValueSpaceMapKey {
    using space_type = Space;
    using coord_type = SpaceTupleType_t<Space>;

    size_t linear_index;

    constexpr explicit ValueSpaceMapKey(coord_type const &in_coord) {
        auto const opt_linear_index = spaceLinearIndex(space_type{}, in_coord);
        if (!opt_linear_index) {
            throw std::runtime_error("Insert Error: Coordinate is not in space!");
        }
        linear_index = *opt_linear_index;
    }

    template <ValuePack Coord>
        requires SpaceContains<space_type, Coord>
    constexpr explicit ValueSpaceMapKey(Coord in_coord) {
        linear_index = LinearIndexFromCoord_v<space_type, Coord>();
    }

    constexpr bool
    operator==(ValueSpaceMapKey const &other) const {
        return linear_index == other.linear_index;
    }
};

// ----------------------------------------------------------------------
// ValueSpaceMapHash
// ----------------------------------------------------------------------
// By marking the hash as transparent, we can use tuples as lookup keys
// without instantiating the Key type.
template <ValueSpace Space> struct ValueSpaceMapHash {
    using space_type = Space;
    using coord_type = SpaceTupleType_t<Space>;
    using key_type   = ValueSpaceMapKey<Space>;

    // Enable C++20 lookup via tuples without creating keys directly.
    using is_transparent = void;

    // Hash for stored Keys
    constexpr std::size_t
    operator()(key_type const &k) const {
        return k.linear_index;
    }

    // Hash for Tuples (The "Get" path)
    constexpr std::size_t
    operator()(coord_type const &t) const {
        auto const opt_linear_index = spaceLinearIndex(space_type{}, t);
        if (!opt_linear_index) {
            // "Get" behavior: GRACEFUL FAILURE
            // Return a dummy hash. It doesn't matter what it is,
            // as long as the Equality operator handles the mismatch next.
            return 0;
        }
        return *opt_linear_index;
    }

    template <ValuePack Coord>
        requires SpaceContains<space_type, Coord>
    constexpr std::size_t
    operator()(Coord in_coord) const {
        return LinearIndexFromCoord_v<space_type, Coord>();
    }
};

// ----------------------------------------------------------------------
// ValueSpaceMapEqual
// ----------------------------------------------------------------------
// Transparent equality allows tuple lookups. An invalid tuple is never
// equal to any valid key, enabling graceful "not found" for bad coords.

template <ValueSpace Space> struct ValueSpaceMapEqual {
    using space_type = Space;
    using coord_type = SpaceTupleType_t<Space>;
    using key_type   = ValueSpaceMapKey<Space>;

    // Enable C++20 lookup via tuples without creating keys directly.
    using is_transparent = void;

    // Key vs Key
    constexpr bool
    operator()(key_type const &lhs, key_type const &rhs) const {
        return lhs == rhs;
    }

    // Key vs Tuple (The "Get" path)
    constexpr bool
    operator()(key_type const &k, coord_type const &t) const {
        auto const opt_linear_index = spaceLinearIndex(space_type{}, t);
        if (!opt_linear_index) {
            // "Get" behavior: GRACEFUL FAILURE
            // An invalid tuple is never equal to any valid key.
            return false;
        }
        return k.linear_index == *opt_linear_index;
    }

    // Tuple vs Key (The "Get" path - reverse argument order)
    // Required for transparent lookup: the library may call with either order.
    constexpr bool
    operator()(coord_type const &t, key_type const &k) const {
        return (*this)(k, t); // Delegate to the Key vs Tuple overload
    }

    template <ValuePack Coord>
        requires SpaceContains<space_type, Coord>
    constexpr bool
    operator()(Coord in_coord, key_type const &k) const {
        return ValueSpaceMapKey<space_type>(in_coord) == k;
    }

    template <ValuePack Coord>
        requires SpaceContains<space_type, Coord>
    constexpr bool
    operator()(key_type const &k, Coord in_coord) const {
        return k == ValueSpaceMapKey<space_type>(in_coord);
    }
};

// ----------------------------------------------------------------------
// ValueSpaceMap_t
// ----------------------------------------------------------------------

template <ValueSpace Space, typename T>
using ValueSpaceMap_t = std::
    unordered_map<ValueSpaceMapKey<Space>, T, ValueSpaceMapHash<Space>, ValueSpaceMapEqual<Space>>;

// ----------------------------------------------------------------------
// create_and_store utilities
// ----------------------------------------------------------------------
// These utilities populate a ValueSpaceMap with entries created by a factory.
// The factory is called with each coordinate and should return the value to store.

// Visitor that creates and stores entries for each coordinate in a space
template <ValueSpace Space, typename T, typename Factory> struct CreateAndStoreVisitor {
    std::reference_wrapper<ValueSpaceMap_t<Space, T>> map;
    std::reference_wrapper<Factory> factory;

    template <ValuePack Coord>
        requires SpaceContains<Space, Coord>
    void
    operator()(Coord coord) const {
        map.get().emplace(coord, factory.get()(coord));
    }
};

// Create and store a single coordinate
template <ValueSpace Space, typename T, typename Factory, ValuePack Coord>
    requires SpaceContains<Space, Coord>
void
create_and_store_coord(ValueSpaceMap_t<Space, T> &map, Factory &&factory, Coord coord) {
    map.emplace(coord, std::forward<Factory>(factory)(coord));
}

namespace detail {

// Implementation helper: handle a single sub (either a coord or a subspace)
template <ValueSpace Space, typename T, typename Factory, typename Sub>
    requires SpaceCovers<Space, Sub>
void
create_and_store_one(ValueSpaceMap_t<Space, T> &map, Factory &factory, Sub sub) {
    if constexpr (ValueSpace<Sub>) {
        CreateAndStoreVisitor<Space, T, Factory> visitor{map, factory};
        visit_value_space(visitor, sub);
    } else {
        static_assert(SpaceContains<Space, Sub>, "Coord must be in Space");
        map.emplace(sub, factory(sub));
    }
}

} // namespace detail

// Populate map entries for multiple subs (coords or subspaces)
template <ValueSpace Space, typename T, typename Factory, typename... Subs>
    requires(SpaceCovers<Space, Subs> && ...)
void
create_and_store(ValueSpaceMap_t<Space, T> &map, Factory &factory, Subs... subs) {
    (detail::create_and_store_one(map, factory, subs), ...);
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_VALUESPACEMAP_H
