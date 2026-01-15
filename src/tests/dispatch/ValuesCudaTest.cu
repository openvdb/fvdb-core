// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// This test verifies nvcc can compile the template-heavy Values.h code.
// It does not test functionality (the C++ test does that), just compilation.

#if 0

#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Instantiate templates to verify nvcc compilation
// =============================================================================

// Value pack types
using TestAnyPack    = AnyTypeValuePack<1, 'x', true>;
using TestSamePack   = SameTypeValuePack<10, 20, 30>;
using TestUniquePack = SameTypeUniqueValuePack<10, 20, 30>;

// Static assertions to force template instantiation
static_assert(PackSize<TestAnyPack>::value() == 3);
static_assert(PackSize<TestSamePack>::value() == 3);
static_assert(PackSize<TestUniquePack>::value() == 3);

static_assert(PackElement<TestAnyPack, 0>::value() == 1);
static_assert(PackElement<TestSamePack, 1>::value() == 20);
static_assert(PackElement<TestUniquePack, 2>::value() == 30);

static_assert(PackContains<TestSamePack, 20>::value());
static_assert(!PackContains<TestSamePack, 99>::value());

static_assert(PackDefiniteFirstIndex<TestSamePack, 20>::value() == 1);
static_assert(PackDefiniteIndex<TestUniquePack, 30>::value() == 2);

static_assert(PackIsSubset<TestSamePack, AnyTypeValuePack<10, 30>>::value());

static_assert(ValuesElement<1, 10, 20, 30>::value() == 20);
static_assert(ValuesContain<20, 10, 20, 30>::value());
static_assert(ValuesUnique<10, 20, 30>::value());
static_assert(ValuesHead<10, 20, 30>::value() == 10);

// PackPrepend
using PrependedAny    = PackPrepend_t<AnyTypeValuePack<20, 30>, 10>;
using PrependedSame   = PackPrepend_t<SameTypeValuePack<20, 30>, 10>;
using PrependedUnique = PackPrepend_t<SameTypeUniqueValuePack<20, 30>, 10>;

static_assert(std::is_same_v<PrependedAny, AnyTypeValuePack<10, 20, 30>>);
static_assert(std::is_same_v<PrependedSame, SameTypeValuePack<10, 20, 30>>);
static_assert(std::is_same_v<PrependedUnique, SameTypeUniqueValuePack<10, 20, 30>>);

// PackPrepend to empty packs
static_assert(std::is_same_v<PackPrepend_t<AnyTypeValuePack<>, 10>, AnyTypeValuePack<10>>);
static_assert(std::is_same_v<PackPrepend_t<SameTypeValuePack<>, 10>, SameTypeValuePack<10>>);
static_assert(
    std::is_same_v<PackPrepend_t<SameTypeUniqueValuePack<>, 10>, SameTypeUniqueValuePack<10>>);

// Runtime packPrepend function
static_assert(std::is_same_v<decltype(packPrepend<10>(SameTypeValuePack<20, 30>{})),
                             SameTypeValuePack<10, 20, 30>>);
static_assert(
    std::is_same_v<decltype(packPrepend<10>(SameTypeValuePack<>{})), SameTypeValuePack<10>>);

// IndexSequencePrepend
static_assert(std::is_same_v<IndexSequencePrepend_t<std::index_sequence<1, 2, 3>, 0>,
                             std::index_sequence<0, 1, 2, 3>>);
static_assert(
    std::is_same_v<IndexSequencePrepend_t<std::index_sequence<>, 42>, std::index_sequence<42>>);
static_assert(std::is_same_v<decltype(indexSequencePrepend<0>(std::index_sequence<1, 2>{})),
                             std::index_sequence<0, 1, 2>>);

// =============================================================================
// Minimal runtime test to ensure linking works
// =============================================================================

TEST(ValuesCuda, TemplatesCompileWithNvcc) {
    // Runtime functions
    EXPECT_EQ(packSize(TestSamePack{}), 3u);
    EXPECT_EQ(packElement(TestSamePack{}, 1), 20);
    EXPECT_TRUE(packContains(TestSamePack{}, 20));
    EXPECT_EQ(packFirstIndex(TestSamePack{}, 20), std::optional<size_t>{1});
    EXPECT_EQ(packDefiniteIndex(TestUniquePack{}, 30), 2u);
    EXPECT_TRUE(packIsSubset(TestSamePack{}, AnyTypeValuePack<10, 30>{}));
}

} // namespace dispatch
} // namespace fvdb

#endif
