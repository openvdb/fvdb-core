// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/ValueSpaceMap.h>

#include <gtest/gtest.h>

#include <functional>
#include <string>

namespace fvdb {
namespace dispatch {

// Test enums (matching ValueSpaceTest.cpp)
enum class Device { CPU, CUDA, Metal };
enum class DType { Float32, Float64, Int32 };

// Axis type aliases
using DeviceAxis = Values<Device::CPU, Device::CUDA, Device::Metal>;
using DTypeAxis  = Values<DType::Float32, DType::Float64, DType::Int32>;
using IntAxis    = Values<1, 2, 4, 8>;

// ----------------------------------------------------------------------
// ValueSpaceMapKey Tests
// ----------------------------------------------------------------------

TEST(ValueSpaceMapKey, ConstructsWithValidCoord) {
    using Space = ValueAxes<IntAxis>;
    using Key   = ValueSpaceMapKey<Space>;

    // Valid coordinates construct without throwing
    EXPECT_NO_THROW({ Key key(std::make_tuple(1)); });
    EXPECT_NO_THROW({ Key key(std::make_tuple(2)); });
    EXPECT_NO_THROW({ Key key(std::make_tuple(4)); });
    EXPECT_NO_THROW({ Key key(std::make_tuple(8)); });
}

TEST(ValueSpaceMapKey, ThrowsOnInvalidCoord) {
    using Space = ValueAxes<IntAxis>;
    using Key   = ValueSpaceMapKey<Space>;

    // Invalid coordinates throw
    EXPECT_THROW({ Key key(std::make_tuple(3)); }, std::runtime_error);
    EXPECT_THROW({ Key key(std::make_tuple(0)); }, std::runtime_error);
    EXPECT_THROW({ Key key(std::make_tuple(16)); }, std::runtime_error);
    EXPECT_THROW({ Key key(std::make_tuple(-1)); }, std::runtime_error);
}

TEST(ValueSpaceMapKey, StoresCorrectLinearIndex) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}
    using Key   = ValueSpaceMapKey<Space>;

    Key k1(std::make_tuple(1));
    Key k2(std::make_tuple(2));
    Key k4(std::make_tuple(4));
    Key k8(std::make_tuple(8));

    EXPECT_EQ(k1.linear_index, 0u);
    EXPECT_EQ(k2.linear_index, 1u);
    EXPECT_EQ(k4.linear_index, 2u);
    EXPECT_EQ(k8.linear_index, 3u);
}

TEST(ValueSpaceMapKey, TwoAxisLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    using Key   = ValueSpaceMapKey<Space>;

    // Row-major: DType varies fastest
    Key k00(std::make_tuple(Device::CPU, DType::Float32));
    Key k01(std::make_tuple(Device::CPU, DType::Float64));
    Key k10(std::make_tuple(Device::CUDA, DType::Float32));
    Key k22(std::make_tuple(Device::Metal, DType::Int32));

    EXPECT_EQ(k00.linear_index, 0u);
    EXPECT_EQ(k01.linear_index, 1u);
    EXPECT_EQ(k10.linear_index, 3u);
    EXPECT_EQ(k22.linear_index, 8u);
}

TEST(ValueSpaceMapKey, EqualityOperator) {
    using Space = ValueAxes<IntAxis>;
    using Key   = ValueSpaceMapKey<Space>;

    Key k1a(std::make_tuple(4));
    Key k1b(std::make_tuple(4));
    Key k2(std::make_tuple(8));

    EXPECT_TRUE(k1a == k1b);
    EXPECT_FALSE(k1a == k2);
}

// ----------------------------------------------------------------------
// ValueSpaceMap_t Find Tests (Graceful Failure)
// ----------------------------------------------------------------------

TEST(ValueSpaceMap, FindWithValidCoordReturnsIterator) {
    using Space = ValueAxes<IntAxis>;
    ValueSpaceMap_t<Space, std::string> map;

    // Insert a value
    map.emplace(std::make_tuple(4), "four");

    // Find returns valid iterator
    auto it = map.find(std::make_tuple(4));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, "four");
}

TEST(ValueSpaceMap, FindWithMissingValidCoordReturnsEnd) {
    using Space = ValueAxes<IntAxis>;
    ValueSpaceMap_t<Space, std::string> map;

    // Insert a value
    map.emplace(std::make_tuple(4), "four");

    // Find different valid coord returns end (not in map, but valid coord)
    auto it = map.find(std::make_tuple(8));
    EXPECT_EQ(it, map.end());
}

TEST(ValueSpaceMap, FindWithInvalidCoordReturnsEndNoThrow) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}
    ValueSpaceMap_t<Space, std::string> map;

    // Insert a value
    map.emplace(std::make_tuple(4), "four");

    // Find with INVALID coord (3 is not in the space) returns end, does NOT throw
    auto it = map.find(std::make_tuple(3));
    EXPECT_EQ(it, map.end());

    // More invalid coords - all gracefully return end()
    EXPECT_EQ(map.find(std::make_tuple(0)), map.end());
    EXPECT_EQ(map.find(std::make_tuple(16)), map.end());
    EXPECT_EQ(map.find(std::make_tuple(-99)), map.end());
}

TEST(ValueSpaceMap, FindWithInvalidCoordTwoAxis) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    ValueSpaceMap_t<Space, int> map;

    map.emplace(std::make_tuple(Device::CUDA, 4), 42);

    // Valid lookup
    auto it = map.find(std::make_tuple(Device::CUDA, 4));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 42);

    // Invalid second axis - graceful failure
    EXPECT_EQ(map.find(std::make_tuple(Device::CUDA, 3)), map.end());
    EXPECT_EQ(map.find(std::make_tuple(Device::CPU, 16)), map.end());
}

// ----------------------------------------------------------------------
// ValueSpaceMap_t Emplace/Insert Tests (Throws on Invalid)
// ----------------------------------------------------------------------

TEST(ValueSpaceMap, EmplaceWithValidCoordSucceeds) {
    using Space = ValueAxes<IntAxis>;
    ValueSpaceMap_t<Space, std::string> map;

    // All valid coords succeed
    EXPECT_NO_THROW(map.emplace(std::make_tuple(1), "one"));
    EXPECT_NO_THROW(map.emplace(std::make_tuple(2), "two"));
    EXPECT_NO_THROW(map.emplace(std::make_tuple(4), "four"));
    EXPECT_NO_THROW(map.emplace(std::make_tuple(8), "eight"));

    EXPECT_EQ(map.size(), 4u);
}

TEST(ValueSpaceMap, EmplaceWithInvalidCoordThrows) {
    using Space = ValueAxes<IntAxis>; // {1, 2, 4, 8}
    ValueSpaceMap_t<Space, std::string> map;

    // Invalid coords throw on emplace
    EXPECT_THROW(map.emplace(std::make_tuple(3), "three"), std::runtime_error);
    EXPECT_THROW(map.emplace(std::make_tuple(0), "zero"), std::runtime_error);
    EXPECT_THROW(map.emplace(std::make_tuple(16), "sixteen"), std::runtime_error);

    // Map should still be empty (all emplaces failed)
    EXPECT_EQ(map.size(), 0u);
}

TEST(ValueSpaceMap, EmplaceWithInvalidCoordTwoAxis) {
    using Space = ValueAxes<DeviceAxis, IntAxis>;
    ValueSpaceMap_t<Space, int> map;

    // Valid emplace
    EXPECT_NO_THROW(map.emplace(std::make_tuple(Device::CPU, 1), 1));
    EXPECT_NO_THROW(map.emplace(std::make_tuple(Device::Metal, 8), 2));

    // Invalid second axis
    EXPECT_THROW(map.emplace(std::make_tuple(Device::CPU, 3), 99), std::runtime_error);
    EXPECT_THROW(map.emplace(std::make_tuple(Device::CUDA, 16), 99), std::runtime_error);

    EXPECT_EQ(map.size(), 2u);
}

// ----------------------------------------------------------------------
// ValueSpaceMap_t Integration Tests
// ----------------------------------------------------------------------

TEST(ValueSpaceMap, FullDispatchTablePattern) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    ValueSpaceMap_t<Space, std::function<int()>> dispatchTable;

    // Register handlers for some coords
    dispatchTable.emplace(std::make_tuple(Device::CPU, DType::Float32), [] { return 1; });
    dispatchTable.emplace(std::make_tuple(Device::CUDA, DType::Float32), [] { return 2; });
    dispatchTable.emplace(std::make_tuple(Device::CUDA, DType::Float64), [] { return 3; });

    // Dispatch to registered handlers
    auto it1 = dispatchTable.find(std::make_tuple(Device::CPU, DType::Float32));
    ASSERT_NE(it1, dispatchTable.end());
    EXPECT_EQ(it1->second(), 1);

    auto it2 = dispatchTable.find(std::make_tuple(Device::CUDA, DType::Float64));
    ASSERT_NE(it2, dispatchTable.end());
    EXPECT_EQ(it2->second(), 3);

    // Unregistered but valid coord
    auto it3 = dispatchTable.find(std::make_tuple(Device::Metal, DType::Int32));
    EXPECT_EQ(it3, dispatchTable.end());
}

TEST(ValueSpaceMap, PopulateEntireSpace) {
    using Space = ValueAxes<IntAxis>; // 4 elements
    ValueSpaceMap_t<Space, size_t> map;

    // Populate via visit_value_space
    visit_value_space(
        [&](auto coord) {
            auto tuple = coordToTuple(coord);
            auto idx   = spaceLinearIndex(Space{}, tuple);
            map.emplace(tuple, *idx);
        },
        Space{});

    EXPECT_EQ(map.size(), 4u);

    // Verify all entries
    EXPECT_EQ(map.find(std::make_tuple(1))->second, 0u);
    EXPECT_EQ(map.find(std::make_tuple(2))->second, 1u);
    EXPECT_EQ(map.find(std::make_tuple(4))->second, 2u);
    EXPECT_EQ(map.find(std::make_tuple(8))->second, 3u);
}

TEST(ValueSpaceMap, ThreeAxisSpace) {
    using Space = ValueAxes<Values<'a', 'b'>, Values<10, 20>, Values<true, false>>;
    ValueSpaceMap_t<Space, std::string> map;

    // Insert a few entries
    map.emplace(std::make_tuple('a', 10, true), "a-10-true");
    map.emplace(std::make_tuple('b', 20, false), "b-20-false");

    // Valid lookups
    EXPECT_EQ(map.find(std::make_tuple('a', 10, true))->second, "a-10-true");
    EXPECT_EQ(map.find(std::make_tuple('b', 20, false))->second, "b-20-false");

    // Invalid coord (30 not in axis) - graceful failure
    EXPECT_EQ(map.find(std::make_tuple('a', 30, true)), map.end());

    // Invalid coord throws on insert
    EXPECT_THROW(map.emplace(std::make_tuple('c', 10, true), "bad"), std::runtime_error);
}

TEST(ValueSpaceMap, HashUsesLinearIndex) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    ValueSpaceMapHash<Space> hash;

    // Hash of tuple should equal the linear index for valid coords
    EXPECT_EQ(hash(std::make_tuple(Device::CPU, DType::Float32)), 0u);
    EXPECT_EQ(hash(std::make_tuple(Device::CPU, DType::Float64)), 1u);
    EXPECT_EQ(hash(std::make_tuple(Device::CUDA, DType::Float32)), 3u);
    EXPECT_EQ(hash(std::make_tuple(Device::Metal, DType::Int32)), 8u);
}

TEST(ValueSpaceMap, EqualReturnsFalseForInvalidCoord) {
    using Space = ValueAxes<IntAxis>;
    using Key   = ValueSpaceMapKey<Space>;
    ValueSpaceMapEqual<Space> equal;

    Key validKey(std::make_tuple(4));

    // Comparing valid key with invalid tuple returns false (not throws)
    EXPECT_FALSE(equal(validKey, std::make_tuple(3)));
    EXPECT_FALSE(equal(validKey, std::make_tuple(99)));
}

} // namespace dispatch
} // namespace fvdb
