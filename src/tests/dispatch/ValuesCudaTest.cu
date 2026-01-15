// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// This test verifies nvcc can compile the template-heavy Values.h code.
// It does not test functionality (the C++ test does that), just compilation.

#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Instantiate templates to verify nvcc compilation
// =============================================================================

// Value pack types (all use unified Values<> type now, concepts distinguish them)
using TestMixedPack  = Values<1, 'x', true>;
using TestSamePack   = Values<10, 20, 30>;
using TestUniquePack = Values<10, 20, 30>;
using TestEmptyPack  = Values<>;

// Concept checks via consteval helper functions (avoid nvcc issues with direct concept use)
static_assert(is_value_pack<TestMixedPack>());
static_assert(is_value_pack<TestSamePack>());
static_assert(is_value_pack<TestEmptyPack>());

static_assert(is_non_empty_value_pack<TestMixedPack>());
static_assert(is_non_empty_value_pack<TestSamePack>());
static_assert(is_empty_value_pack<TestEmptyPack>());

static_assert(is_same_type_value_pack<TestSamePack>());
static_assert(!is_same_type_value_pack<TestMixedPack>());
static_assert(is_same_type_value_pack<TestEmptyPack>()); // vacuously true

static_assert(is_unique_value_pack<TestUniquePack>());
static_assert(is_unique_value_pack<TestMixedPack>());    // mixed types are unique
static_assert(is_unique_value_pack<TestEmptyPack>());    // vacuously true

static_assert(is_same_type_non_empty_value_pack<TestSamePack>());
static_assert(is_unique_non_empty_value_pack<TestUniquePack>());

// PackSize
static_assert(PackSize_v<TestMixedPack>() == 3);
static_assert(PackSize_v<TestSamePack>() == 3);
static_assert(PackSize_v<TestEmptyPack>() == 0);

// PackElement
static_assert(PackElement_v<TestMixedPack, 0>() == 1);
static_assert(PackElement_v<TestSamePack, 1>() == 20);
static_assert(PackElement_v<TestUniquePack, 2>() == 30);

// PackContains
static_assert(PackContains_v<TestSamePack, 20>());
static_assert(!PackContains_v<TestSamePack, 99>());
static_assert(value_pack_contains<TestSamePack, 10>());
static_assert(!value_pack_contains<TestSamePack, 99>());

// PackDefiniteFirstIndex and PackDefiniteIndex
static_assert(PackDefiniteFirstIndex_v<TestSamePack, 20>() == 1);
static_assert(PackDefiniteIndex_v<TestUniquePack, 30>() == 2);

// PackIsSubset
static_assert(PackIsSubset_v<TestSamePack, Values<10, 30>>());
static_assert(PackIsSubset_v<TestSamePack, Values<>>());
static_assert(!PackIsSubset_v<TestEmptyPack, TestSamePack>());

// Low-level Values* utilities
static_assert(ValuesElement_v<1, 10, 20, 30>() == 20);
static_assert(ValuesContain_v<20, 10, 20, 30>());
static_assert(ValuesUnique_v<10, 20, 30>());
static_assert(ValuesHead_v<10, 20, 30>() == 10);
static_assert(ValuesSameType_v<10, 20, 30>());
static_assert(!ValuesSameType_v<10, 'x', true>());

// PackPrepend
using PrependedMixed = PackPrepend_t<Values<'x', true>, 42>;
using PrependedSame  = PackPrepend_t<Values<20, 30>, 10>;
using PrependedEmpty = PackPrepend_t<Values<>, 10>;

static_assert(std::is_same_v<PrependedMixed, Values<42, 'x', true>>);
static_assert(std::is_same_v<PrependedSame, Values<10, 20, 30>>);
static_assert(std::is_same_v<PrependedEmpty, Values<10>>);

// Runtime packPrepend function
static_assert(std::is_same_v<decltype(packPrepend<10>(Values<20, 30>{})), Values<10, 20, 30>>);
static_assert(std::is_same_v<decltype(packPrepend<10>(Values<>{})), Values<10>>);

// IndexSequencePrepend
static_assert(std::is_same_v<IndexSequencePrepend_t<std::index_sequence<1, 2, 3>, 0>,
                             std::index_sequence<0, 1, 2, 3>>);
static_assert(
    std::is_same_v<IndexSequencePrepend_t<std::index_sequence<>, 42>, std::index_sequence<42>>);
static_assert(std::is_same_v<decltype(indexSequencePrepend<0>(std::index_sequence<1, 2>{})),
                             std::index_sequence<0, 1, 2>>);

// PackValueType (only valid for SameTypeNonEmptyValuePack)
static_assert(std::is_same_v<PackValueType_t<TestSamePack>, int>);

// =============================================================================
// Minimal runtime test to ensure linking works
// =============================================================================

TEST(ValuesCuda, TemplatesCompileWithNvcc) {
    // Runtime functions
    EXPECT_EQ(packSize(TestSamePack{}), 3u);
    EXPECT_EQ(packElement(TestSamePack{}, 1), 20);
    EXPECT_TRUE(packContains(TestSamePack{}, 20));
    EXPECT_FALSE(packContains(TestSamePack{}, 99));
    EXPECT_EQ(packFirstIndex(TestSamePack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packFirstIndex(TestSamePack{}, 99), std::nullopt);
    EXPECT_EQ(packDefiniteIndex(TestUniquePack{}, 30), 2u);
    EXPECT_EQ(packDefiniteFirstIndex(TestSamePack{}, 10), 0u);
    EXPECT_TRUE(packIsSubset(TestSamePack{}, Values<10, 30>{}));
    EXPECT_FALSE(packIsSubset(TestEmptyPack{}, TestSamePack{}));

    // packPrepend runtime
    auto prepended = packPrepend<5>(Values<10, 20>{});
    EXPECT_EQ(packSize(prepended), 3u);
    EXPECT_EQ(packElement(prepended, 0), 5);
}

} // namespace dispatch
} // namespace fvdb
