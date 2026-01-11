// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/AxisOuterProduct.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// is_dimensional_axis Tests
// =============================================================================

TEST(IsDimensionalAxis, SameTypeUniqueValuePackIsTrue) {
    static_assert(is_dimensional_axis_v<SameTypeUniqueValuePack<1, 2, 3>>);
    static_assert(is_dimensional_axis_v<SameTypeUniqueValuePack<'a', 'b'>>);
    static_assert(is_dimensional_axis_v<SameTypeUniqueValuePack<true>>);
    EXPECT_TRUE((is_dimensional_axis_v<SameTypeUniqueValuePack<1, 2, 3>>));
}

TEST(IsDimensionalAxis, EmptyPackIsTrue) {
    static_assert(is_dimensional_axis_v<SameTypeUniqueValuePack<>>);
    EXPECT_TRUE((is_dimensional_axis_v<SameTypeUniqueValuePack<>>));
}

TEST(IsDimensionalAxis, OtherTypesAreFalse) {
    static_assert(!is_dimensional_axis_v<int>);
    static_assert(!is_dimensional_axis_v<std::tuple<int, int>>);
    static_assert(!is_dimensional_axis_v<SameTypeValuePack<1, 2, 3>>);
    static_assert(!is_dimensional_axis_v<AnyTypeValuePack<1, 2, 3>>);
    EXPECT_FALSE(is_dimensional_axis_v<int>);
}

// =============================================================================
// AxisOuterProduct Basic Tests
// =============================================================================

TEST(AxisOuterProduct, SingleAxisHasCorrectProperties) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    static_assert(Space::num_axes == 1);
    static_assert(Space::size == 3);
    EXPECT_EQ(Space::num_axes, 1u);
    EXPECT_EQ(Space::size, 3u);
}

TEST(AxisOuterProduct, TwoAxesHaveCorrectSize) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    static_assert(Space::num_axes == 2);
    static_assert(Space::size == 6); // 2 * 3
    EXPECT_EQ(Space::size, 6u);
}

TEST(AxisOuterProduct, ThreeAxesHaveCorrectSize) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<'a', 'b', 'c'>;
    using Axis3 = SameTypeUniqueValuePack<true, false>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    static_assert(Space::num_axes == 3);
    static_assert(Space::size == 12); // 2 * 3 * 2
    EXPECT_EQ(Space::size, 12u);
}

TEST(AxisOuterProduct, EmptySpaceHasSizeOne) {
    using Space = AxisOuterProduct<>;
    static_assert(Space::num_axes == 0);
    static_assert(Space::size == 1); // Empty product is 1
    EXPECT_EQ(Space::size, 1u);
}

// Note: AxisOuterProduct with empty axes (e.g., SameTypeUniqueValuePack<>) is
// rejected at compile time via static_assert. This is intentional - empty axes
// have no valid values to dispatch on. The rejection cannot be tested at runtime.

// =============================================================================
// AxisOuterProduct Type Aliases Tests
// =============================================================================

TEST(AxisOuterProduct, AxisAtTypeReturnsCorrectAxis) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<'x', 'y'>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    static_assert(std::is_same_v<Space::axis_at_type<0>, Axis1>);
    static_assert(std::is_same_v<Space::axis_at_type<1>, Axis2>);
}

TEST(AxisOuterProduct, ValueAtTypeReturnsCorrectValueType) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<'x', 'y'>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    static_assert(std::is_same_v<Space::value_at_type<0>, int>);
    static_assert(std::is_same_v<Space::value_at_type<1>, char>);
}

// =============================================================================
// AxisOuterProduct::index_of_values Tests
// =============================================================================

TEST(AxisOuterProduct, IndexOfValuesSingleAxis) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    EXPECT_EQ(Space::index_of_values(10), 0u);
    EXPECT_EQ(Space::index_of_values(20), 1u);
    EXPECT_EQ(Space::index_of_values(30), 2u);
}

TEST(AxisOuterProduct, IndexOfValuesSingleAxisNotFound) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    EXPECT_FALSE(Space::index_of_values(15).has_value());
    EXPECT_FALSE(Space::index_of_values(0).has_value());
}

TEST(AxisOuterProduct, IndexOfValuesTwoAxesRowMajor) {
    // Row-major: last axis varies fastest
    // For axes [A, B, C] x [1, 2]:
    //   (A, 1) -> 0, (A, 2) -> 1
    //   (B, 1) -> 2, (B, 2) -> 3
    //   (C, 1) -> 4, (C, 2) -> 5
    using Axis1 = SameTypeUniqueValuePack<'A', 'B', 'C'>;
    using Axis2 = SameTypeUniqueValuePack<1, 2>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    EXPECT_EQ(Space::index_of_values('A', 1), 0u);
    EXPECT_EQ(Space::index_of_values('A', 2), 1u);
    EXPECT_EQ(Space::index_of_values('B', 1), 2u);
    EXPECT_EQ(Space::index_of_values('B', 2), 3u);
    EXPECT_EQ(Space::index_of_values('C', 1), 4u);
    EXPECT_EQ(Space::index_of_values('C', 2), 5u);
}

TEST(AxisOuterProduct, IndexOfValuesThreeAxes) {
    using Axis1 = SameTypeUniqueValuePack<0, 1>;
    using Axis2 = SameTypeUniqueValuePack<0, 1>;
    using Axis3 = SameTypeUniqueValuePack<0, 1>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    // Binary counting pattern with row-major order
    EXPECT_EQ(Space::index_of_values(0, 0, 0), 0u);
    EXPECT_EQ(Space::index_of_values(0, 0, 1), 1u);
    EXPECT_EQ(Space::index_of_values(0, 1, 0), 2u);
    EXPECT_EQ(Space::index_of_values(0, 1, 1), 3u);
    EXPECT_EQ(Space::index_of_values(1, 0, 0), 4u);
    EXPECT_EQ(Space::index_of_values(1, 0, 1), 5u);
    EXPECT_EQ(Space::index_of_values(1, 1, 0), 6u);
    EXPECT_EQ(Space::index_of_values(1, 1, 1), 7u);
}

TEST(AxisOuterProduct, IndexOfValuesReturnsNulloptIfAnyValueMissing) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    EXPECT_FALSE(Space::index_of_values(1, 30).has_value()); // Second axis invalid
    EXPECT_FALSE(Space::index_of_values(3, 10).has_value()); // First axis invalid
    EXPECT_FALSE(Space::index_of_values(3, 30).has_value()); // Both invalid
}

TEST(AxisOuterProduct, IndexOfValuesIsConstexpr) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    static_assert(Space::index_of_values(1, 10) == 0);
    static_assert(Space::index_of_values(2, 20) == 3);
    static_assert(!Space::index_of_values(3, 10).has_value());
}

// =============================================================================
// is_axis_outer_product Tests
// =============================================================================

TEST(IsAxisOuterProduct, AxisOuterProductIsTrue) {
    using Space = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    static_assert(is_axis_outer_product_v<Space>);
    EXPECT_TRUE(is_axis_outer_product_v<Space>);
}

TEST(IsAxisOuterProduct, EmptyAxisOuterProductIsTrue) {
    static_assert(is_axis_outer_product_v<AxisOuterProduct<>>);
    EXPECT_TRUE(is_axis_outer_product_v<AxisOuterProduct<>>);
}

TEST(IsAxisOuterProduct, OtherTypesAreFalse) {
    static_assert(!is_axis_outer_product_v<int>);
    static_assert(!is_axis_outer_product_v<SameTypeUniqueValuePack<1, 2>>);
    EXPECT_FALSE(is_axis_outer_product_v<int>);
}

TEST(IsAxisOuterProduct, ConceptWorks) {
    using Space = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    static_assert(AxisOuterProductConcept<Space>);
    static_assert(!AxisOuterProductConcept<int>);
}

// =============================================================================
// is_subspace_of Tests
// =============================================================================

TEST(IsSubspaceOf, SpaceIsSubspaceOfItself) {
    using Space = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>;
    static_assert(is_subspace_of_v<Space, Space>);
    EXPECT_TRUE((is_subspace_of_v<Space, Space>));
}

TEST(IsSubspaceOf, SmallerAxisIsSubspace) {
    using SubAxis  = SameTypeUniqueValuePack<1, 2>;
    using FullAxis = SameTypeUniqueValuePack<1, 2, 3, 4>;
    using Sub      = AxisOuterProduct<SubAxis>;
    using Full     = AxisOuterProduct<FullAxis>;

    static_assert(is_subspace_of_v<Sub, Full>);
    EXPECT_TRUE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, MultipleAxesSubspace) {
    using SubAxis1  = SameTypeUniqueValuePack<1>;
    using SubAxis2  = SameTypeUniqueValuePack<'a', 'b'>;
    using FullAxis1 = SameTypeUniqueValuePack<1, 2, 3>;
    using FullAxis2 = SameTypeUniqueValuePack<'a', 'b', 'c'>;

    using Sub  = AxisOuterProduct<SubAxis1, SubAxis2>;
    using Full = AxisOuterProduct<FullAxis1, FullAxis2>;

    static_assert(is_subspace_of_v<Sub, Full>);
    EXPECT_TRUE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, DifferentAxisCountIsNotSubspace) {
    using OneAxis = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    using TwoAxes = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>, SameTypeUniqueValuePack<3, 4>>;

    // Different axis counts cannot be subspaces of each other
    static_assert(!is_subspace_of_v<OneAxis, TwoAxes>);
    static_assert(!is_subspace_of_v<TwoAxes, OneAxis>);
    EXPECT_FALSE((is_subspace_of_v<OneAxis, TwoAxes>));
    EXPECT_FALSE((is_subspace_of_v<TwoAxes, OneAxis>));
}

TEST(IsSubspaceOf, DisjointValuesIsNotSubspace) {
    using Sub  = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    using Full = AxisOuterProduct<SameTypeUniqueValuePack<3, 4, 5>>;

    static_assert(!is_subspace_of_v<Sub, Full>);
    EXPECT_FALSE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, PartialOverlapIsNotSubspace) {
    // Only some values overlap - not a proper subspace
    using Sub  = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 99>>;
    using Full = AxisOuterProduct<SameTypeUniqueValuePack<1, 2, 3>>;

    static_assert(!is_subspace_of_v<Sub, Full>);
    EXPECT_FALSE((is_subspace_of_v<Sub, Full>));
}

// =============================================================================
// for_each_values Tests
// =============================================================================

namespace {
template <auto... Vs> struct CollectValues {
    static void
    apply(std::vector<std::tuple<decltype(Vs)...>> &out) {
        out.emplace_back(Vs...);
    }
};
} // namespace

TEST(ForEachValues, SingleAxisIteratesAllValues) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    std::vector<std::tuple<int>> result;
    for_each_values<Space, CollectValues>::apply(result);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(std::get<0>(result[0]), 10);
    EXPECT_EQ(std::get<0>(result[1]), 20);
    EXPECT_EQ(std::get<0>(result[2]), 30);
}

TEST(ForEachValues, TwoAxesIteratesCartesianProduct) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<'a', 'b'>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::vector<std::tuple<int, char>> result;
    for_each_values<Space, CollectValues>::apply(result);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], std::make_tuple(1, 'a'));
    EXPECT_EQ(result[1], std::make_tuple(1, 'b'));
    EXPECT_EQ(result[2], std::make_tuple(2, 'a'));
    EXPECT_EQ(result[3], std::make_tuple(2, 'b'));
}

TEST(ForEachValues, ThreeAxesIteratesAllCombinations) {
    using Axis1 = SameTypeUniqueValuePack<0, 1>;
    using Axis2 = SameTypeUniqueValuePack<0, 1>;
    using Axis3 = SameTypeUniqueValuePack<0, 1>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    std::vector<std::tuple<int, int, int>> result;
    for_each_values<Space, CollectValues>::apply(result);

    ASSERT_EQ(result.size(), 8u);
    // Verify linear index corresponds to iteration order
    for (size_t i = 0; i < result.size(); ++i) {
        auto [v0, v1, v2] = result[i];
        auto idx          = Space::index_of_values(v0, v1, v2);
        EXPECT_EQ(idx, i);
    }
}

// =============================================================================
// for_each_permutation Tests
// =============================================================================

namespace {
template <auto V1, auto V2> struct InstantiatedType {
    static constexpr auto first  = V1;
    static constexpr auto second = V2;
};

template <typename InstType, auto V1, auto V2> struct CollectInstantiated {
    static void
    apply(std::vector<std::pair<int, int>> &out) {
        // Verify the instantiated type has correct values
        static_assert(InstType::first == V1);
        static_assert(InstType::second == V2);
        out.emplace_back(V1, V2);
    }
};
} // namespace

TEST(ForEachPermutation, InstantiatesAndIterates) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::vector<std::pair<int, int>> result;
    for_each_permutation<Space, InstantiatedType, CollectInstantiated>::apply(result);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], std::make_pair(1, 10));
    EXPECT_EQ(result[1], std::make_pair(1, 20));
    EXPECT_EQ(result[2], std::make_pair(2, 10));
    EXPECT_EQ(result[3], std::make_pair(2, 20));
}

} // namespace dispatch
} // namespace fvdb
