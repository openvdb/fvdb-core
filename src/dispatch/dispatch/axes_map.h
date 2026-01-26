// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_DISPATCH_AXES_MAP_H
#define DISPATCH_DISPATCH_AXES_MAP_H

#include "dispatch/detail.h"
#include "dispatch/types.h"
#include "dispatch/visit_spaces.h"

#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace dispatch {

//------------------------------------------------------------------------------
// axes_map: A map keyed by coordinates in an Axes
//------------------------------------------------------------------------------
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
//   using MyAxes = axes<DeviceAxis, ScalarAxis>;
//   axes_map<MyAxes, std::function<void()>> dispatchTable;
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

//------------------------------------------------------------------------------
// axes_map_key
//------------------------------------------------------------------------------
// The Key is only instantiated when an insert or an emplace is called,
// it is not instantiated for a find. It is also instantiated with
// operator[] (which takes a key).

template <typename Axes> struct axes_map_key {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");

    using axes_type        = Axes;
    using value_tuple_type = value_tuple_type_t<Axes>;

    size_t linear_index;

    constexpr explicit axes_map_key(value_tuple_type const &in_coord) {
        auto const opt_linear_index = linear_index_from_value_tuple(axes_type{}, in_coord);
        if (!opt_linear_index) {
            throw std::runtime_error("Insert Error: Coordinate is not in space!");
        }
        linear_index = *opt_linear_index;
    }

    template <typename Tag> constexpr explicit axes_map_key(Tag tag) {
        static_assert(within<Tag, axes_type>, "Tag must be within the axes");
        linear_index = linear_index_from_tag_v<axes_type, Tag>();
    }

    constexpr bool
    operator==(axes_map_key const &other) const {
        return linear_index == other.linear_index;
    }
};

//------------------------------------------------------------------------------
// axes_map_hash
//------------------------------------------------------------------------------
// By marking the hash as transparent, we can use tuples as lookup keys
// without instantiating the Key type.
template <typename Axes> struct axes_map_hash {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");

    using axes_type        = Axes;
    using value_tuple_type = value_tuple_type_t<Axes>;
    using key_type         = axes_map_key<Axes>;

    // Enable C++20 lookup via tuples without creating keys directly.
    using is_transparent = void;

    // Hash for stored Keys
    constexpr std::size_t
    operator()(key_type const &k) const {
        return k.linear_index;
    }

    // Hash for Tuples (The "Get" path)
    constexpr std::size_t
    operator()(value_tuple_type const &t) const {
        auto const opt_linear_index = linear_index_from_value_tuple(axes_type{}, t);
        if (!opt_linear_index) {
            // "Get" behavior: GRACEFUL FAILURE
            // Return a dummy hash. It doesn't matter what it is,
            // as long as the Equality operator handles the mismatch next.
            return 0;
        }
        return *opt_linear_index;
    }

    template <typename Tag>
    constexpr std::size_t
    operator()(Tag tag) const {
        static_assert(within<Tag, axes_type>, "Tag must be within the axes");
        return linear_index_from_tag_v<axes_type, Tag>();
    }
};

//------------------------------------------------------------------------------
// axes_map_equal
//------------------------------------------------------------------------------
// Transparent equality allows tuple lookups. An invalid tuple is never
// equal to any valid key, enabling graceful "not found" for bad coords.

template <typename Axes> struct axes_map_equal {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");

    using axes_type        = Axes;
    using value_tuple_type = value_tuple_type_t<Axes>;
    using key_type         = axes_map_key<Axes>;

    // Enable C++20 lookup via tuples without creating keys directly.
    using is_transparent = void;

    // Key vs Key
    constexpr bool
    operator()(key_type const &lhs, key_type const &rhs) const {
        return lhs == rhs;
    }

    // Key vs Tuple (The "Get" path)
    constexpr bool
    operator()(key_type const &k, value_tuple_type const &t) const {
        auto const opt_linear_index = linear_index_from_value_tuple(axes_type{}, t);
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
    operator()(value_tuple_type const &t, key_type const &k) const {
        return (*this)(k, t); // Delegate to the Key vs Tuple overload
    }

    template <typename Tag>
    constexpr bool
    operator()(Tag tag, key_type const &k) const {
        static_assert(within<Tag, axes_type>, "Tag must be within the axes");
        return axes_map_key<axes_type>(tag) == k;
    }

    template <typename Tag>
    constexpr bool
    operator()(key_type const &k, Tag tag) const {
        static_assert(within<Tag, axes_type>, "Tag must be within the axes");
        return k == axes_map_key<axes_type>(tag);
    }
};

//------------------------------------------------------------------------------
// axes_map
//------------------------------------------------------------------------------

template <typename Axes, typename T>
using axes_map =
    std::unordered_map<axes_map_key<Axes>, T, axes_map_hash<Axes>, axes_map_equal<Axes>>;

//------------------------------------------------------------------------------
// insert_or_assign
//------------------------------------------------------------------------------
// C++23 introduces an insert_or_assign that respects transparency, allowing
// lookup with tag types or tuples without explicitly creating a key. Since
// C++20's try_emplace doesn't support transparent keys, we use find + emplace.
//
// Thread-safety: Same as std::unordered_map - not safe for concurrent
// modifications.
//
// The Key doesn't need to be a literal axes_map_key, because of the way we
// are using transparent maps in C++20, it can be anything which fulfills the
// hash and equal capabilities, so in our case - an instance of a tag will work,
// as will a tuple with the right types.

template <typename Axes, typename T, typename Key_like, typename V>
void
insert_or_assign(axes_map<Axes, T> &map, Key_like key, V &&value) {
    // Validate the key type. Has to be a tag or a tuple, and to be within
    // the axes.
    if constexpr (tag_like<Key_like>) {
        static_assert(within<Key_like, Axes>, "tag keys must be within axes");
    } else {
        static_assert(std::is_same_v<value_tuple_type_t<Axes>, Key_like>,
                      "tuple keys must be the right types for axes");
    }

    // The key type above isn't expensive to create. The whole goal of the
    // transparency is to let us test at runtime against tags or tuples that
    // might *not* be within the axes, and still compile. It will throw an
    // exception for a tag or a tuple that's outside the domain, but it won't
    // fail to compile. (a goal of this class). In this case, though, we're
    // actually trying to assign, so we require (and have just checked)
    // that the key given is valid. Therefore, we can just do this:
    map[axes_map_key<Axes>{key}] = std::forward<V>(value);
}

//------------------------------------------------------------------------------
// create_and_store utilities
//------------------------------------------------------------------------------
// These utilities populate a axes_map with entries created by a factory.
// The factory is called with each coordinate and should return the value to store.

namespace detail {

// Visitor that creates and stores entries for each coordinate in a space
// This is most likely to be used with a visitation over a subspace of the Axes space.
// That's not expressed here, we only see the tag operator() being called,
// which corresponds to a single tag. The "visit_value_space" function will
// take this visitor, which contains the map defined over the outer space and the
// visit function can take the subspace.
template <typename Axes, typename T, typename Factory> struct create_and_store_visitor {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");

    using map_type = axes_map<Axes, T>;

    std::reference_wrapper<map_type> map;
    std::reference_wrapper<Factory> factory;

    template <typename Tag>
    void
    operator()(Tag coord) const {
        static_assert(tag_like<Tag>, "Tag must be a tag type");
        static_assert(within<Tag, Axes>, "Tag must be within the axes");
        insert_or_assign(map.get(), coord, factory.get()(coord));
    }
};

// Implementation helper: handle a single sub (either a coord or a subspace)
template <typename Axes, typename T, typename Factory, typename... SubAxes>
void
create_and_store_helper(axes_map<Axes, T> &map, Factory &factory, axes<SubAxes...> sub) {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");
    static_assert(within<axes<SubAxes...>, Axes>, "SubAxes must be within the Axes");
    create_and_store_visitor<Axes, T, Factory> visitor{map, factory};
    visit_axes_space(visitor, sub);
}

// Create and store a single coordinate
template <typename Axes, typename T, typename Factory, auto... Vs>
void
create_and_store_helper(axes_map<Axes, T> &map, Factory &factory, tag<Vs...> t) {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");
    static_assert(within<tag<Vs...>, Axes>, "tag must be within the axes");
    insert_or_assign(map, t, factory(t));
}

template <typename Axes, typename T, typename Factory, typename Other>
void
create_and_store_helper(axes_map<Axes, T> &map, Factory &factory, Other other) {
    static_assert(within<Other, Axes>,
                  "create_and_store target must be either a tag or a subspace");
}

} // namespace detail

// Populate map entries for multiple subs (coords or subspaces)
template <typename Axes, typename T, typename Factory, typename... Subs>
void
create_and_store(axes_map<Axes, T> &map, Factory &factory, Subs... subs) {
    static_assert(axes_like<Axes>, "Axes must be an axes type");
    static_assert(non_empty<Axes>, "Axes must be non-empty space");
    static_assert((within<Subs, Axes> && ... && true), "Subs must be within the Axes");
    (detail::create_and_store_helper(map, factory, subs), ...);
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_AXES_MAP_H
