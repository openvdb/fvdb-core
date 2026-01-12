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
// SameTypeUniqueValuePack::value_at Tests
// =============================================================================

TEST(SameTypeUniqueValuePack, ValueAtReturnsCorrectValues) {
    using Pack = SameTypeUniqueValuePack<10, 20, 30>;

    static_assert(Pack::value_at<0> == 10);
    static_assert(Pack::value_at<1> == 20);
    static_assert(Pack::value_at<2> == 30);

    EXPECT_EQ(Pack::value_at<0>, 10);
    EXPECT_EQ(Pack::value_at<1>, 20);
    EXPECT_EQ(Pack::value_at<2>, 30);
}

TEST(SameTypeUniqueValuePack, ValueAtWorksWithChars) {
    using Pack = SameTypeUniqueValuePack<'a', 'b', 'c'>;

    static_assert(Pack::value_at<0> == 'a');
    static_assert(Pack::value_at<1> == 'b');
    static_assert(Pack::value_at<2> == 'c');
}

TEST(SameTypeUniqueValuePack, ValueAtRoundtripsWithIndexOfValue) {
    using Pack = SameTypeUniqueValuePack<100, 200, 300>;

    // value_at<index_of_value(v)> == v
    static_assert(Pack::value_at<Pack::index_of_value(100).value()> == 100);
    static_assert(Pack::value_at<Pack::index_of_value(200).value()> == 200);
    static_assert(Pack::value_at<Pack::index_of_value(300).value()> == 300);

    // index_of_value(value_at<i>) == i
    static_assert(Pack::index_of_value(Pack::value_at<0>) == 0);
    static_assert(Pack::index_of_value(Pack::value_at<1>) == 1);
    static_assert(Pack::index_of_value(Pack::value_at<2>) == 2);
}

// =============================================================================
// AxisOuterProduct::axis_index_at Tests
// =============================================================================

TEST(AxisOuterProduct, AxisIndexAtSingleAxis) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    // For single axis, axis_index_at<0> should just return the linear index
    EXPECT_EQ(Space::axis_index_at<0>(0), 0u);
    EXPECT_EQ(Space::axis_index_at<0>(1), 1u);
    EXPECT_EQ(Space::axis_index_at<0>(2), 2u);
}

TEST(AxisOuterProduct, AxisIndexAtTwoAxes) {
    // Row-major: last axis varies fastest
    // For axes [A, B, C] x [1, 2]:
    //   linear 0 -> (0, 0), linear 1 -> (0, 1)
    //   linear 2 -> (1, 0), linear 3 -> (1, 1)
    //   linear 4 -> (2, 0), linear 5 -> (2, 1)
    using Axis1 = SameTypeUniqueValuePack<'A', 'B', 'C'>;
    using Axis2 = SameTypeUniqueValuePack<1, 2>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    // Check axis 0 indices
    EXPECT_EQ(Space::axis_index_at<0>(0), 0u); // A
    EXPECT_EQ(Space::axis_index_at<0>(1), 0u); // A
    EXPECT_EQ(Space::axis_index_at<0>(2), 1u); // B
    EXPECT_EQ(Space::axis_index_at<0>(3), 1u); // B
    EXPECT_EQ(Space::axis_index_at<0>(4), 2u); // C
    EXPECT_EQ(Space::axis_index_at<0>(5), 2u); // C

    // Check axis 1 indices
    EXPECT_EQ(Space::axis_index_at<1>(0), 0u); // 1
    EXPECT_EQ(Space::axis_index_at<1>(1), 1u); // 2
    EXPECT_EQ(Space::axis_index_at<1>(2), 0u); // 1
    EXPECT_EQ(Space::axis_index_at<1>(3), 1u); // 2
    EXPECT_EQ(Space::axis_index_at<1>(4), 0u); // 1
    EXPECT_EQ(Space::axis_index_at<1>(5), 1u); // 2
}

TEST(AxisOuterProduct, AxisIndexAtCompileTimeVersion) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    // Compile-time version: axis_index_at_v<AxisIdx, LinearIdx>
    static_assert(Space::axis_index_at_v<0, 0> == 0);
    static_assert(Space::axis_index_at_v<1, 0> == 0);
    static_assert(Space::axis_index_at_v<0, 1> == 0);
    static_assert(Space::axis_index_at_v<1, 1> == 1);
    static_assert(Space::axis_index_at_v<0, 2> == 1);
    static_assert(Space::axis_index_at_v<1, 2> == 0);
    static_assert(Space::axis_index_at_v<0, 3> == 1);
    static_assert(Space::axis_index_at_v<1, 3> == 1);
}

// =============================================================================
// AxisOuterProduct::values_at_index Tests
// =============================================================================

TEST(AxisOuterProduct, ValuesAtIndexSingleAxis) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    static_assert(Space::values_at_index<0> == std::make_tuple(10));
    static_assert(Space::values_at_index<1> == std::make_tuple(20));
    static_assert(Space::values_at_index<2> == std::make_tuple(30));
}

TEST(AxisOuterProduct, ValuesAtIndexTwoAxes) {
    using Axis1 = SameTypeUniqueValuePack<'A', 'B', 'C'>;
    using Axis2 = SameTypeUniqueValuePack<1, 2>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    static_assert(Space::values_at_index<0> == std::make_tuple('A', 1));
    static_assert(Space::values_at_index<1> == std::make_tuple('A', 2));
    static_assert(Space::values_at_index<2> == std::make_tuple('B', 1));
    static_assert(Space::values_at_index<3> == std::make_tuple('B', 2));
    static_assert(Space::values_at_index<4> == std::make_tuple('C', 1));
    static_assert(Space::values_at_index<5> == std::make_tuple('C', 2));
}

TEST(AxisOuterProduct, ValuesAtIndexThreeAxes) {
    using Axis1 = SameTypeUniqueValuePack<0, 1>;
    using Axis2 = SameTypeUniqueValuePack<0, 1>;
    using Axis3 = SameTypeUniqueValuePack<0, 1>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    // Binary counting pattern
    static_assert(Space::values_at_index<0> == std::make_tuple(0, 0, 0));
    static_assert(Space::values_at_index<1> == std::make_tuple(0, 0, 1));
    static_assert(Space::values_at_index<2> == std::make_tuple(0, 1, 0));
    static_assert(Space::values_at_index<3> == std::make_tuple(0, 1, 1));
    static_assert(Space::values_at_index<4> == std::make_tuple(1, 0, 0));
    static_assert(Space::values_at_index<5> == std::make_tuple(1, 0, 1));
    static_assert(Space::values_at_index<6> == std::make_tuple(1, 1, 0));
    static_assert(Space::values_at_index<7> == std::make_tuple(1, 1, 1));
}

TEST(AxisOuterProduct, ValuesAtIndexRoundtripsWithIndexOfValues) {
    using Axis1 = SameTypeUniqueValuePack<1, 2, 3>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    // For each linear index, decode to values, then encode back
    // values_at_index<L> gives tuple, index_of_values(tuple) gives L
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<0>) == 0);
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<1>) == 1);
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<2>) == 2);
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<3>) == 3);
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<4>) == 4);
    static_assert(std::apply([](auto... vs) { return Space::index_of_values(vs...); },
                             Space::values_at_index<5>) == 5);
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
// for_each_values_flat Tests
// =============================================================================

namespace {
template <auto... Vs> struct CollectValues {
    static void
    apply(std::vector<std::tuple<decltype(Vs)...>> &out) {
        out.emplace_back(Vs...);
    }
};
} // namespace

TEST(ForEachValuesFlat, SingleAxisMatchesRecursive) {
    using Axis  = SameTypeUniqueValuePack<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    std::vector<std::tuple<int>> recursive_result;
    for_each_values<Space, CollectValues>::apply(recursive_result);

    std::vector<std::tuple<int>> flat_result;
    for_each_values_flat<Space, CollectValues>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

TEST(ForEachValuesFlat, TwoAxesMatchesRecursive) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<'a', 'b'>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::vector<std::tuple<int, char>> recursive_result;
    for_each_values<Space, CollectValues>::apply(recursive_result);

    std::vector<std::tuple<int, char>> flat_result;
    for_each_values_flat<Space, CollectValues>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

TEST(ForEachValuesFlat, ThreeAxesMatchesRecursive) {
    using Axis1 = SameTypeUniqueValuePack<0, 1>;
    using Axis2 = SameTypeUniqueValuePack<0, 1>;
    using Axis3 = SameTypeUniqueValuePack<0, 1>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    std::vector<std::tuple<int, int, int>> recursive_result;
    for_each_values<Space, CollectValues>::apply(recursive_result);

    std::vector<std::tuple<int, int, int>> flat_result;
    for_each_values_flat<Space, CollectValues>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

TEST(ForEachValuesFlat, LargerSpaceMatchesRecursive) {
    // Test with a larger space to verify correctness
    using Axis1 = SameTypeUniqueValuePack<1, 2, 3>;
    using Axis2 = SameTypeUniqueValuePack<10, 20, 30, 40>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::vector<std::tuple<int, int>> recursive_result;
    for_each_values<Space, CollectValues>::apply(recursive_result);

    std::vector<std::tuple<int, int>> flat_result;
    for_each_values_flat<Space, CollectValues>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), 12u); // 3 * 4
    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

// =============================================================================
// for_each_permutation_flat Tests
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

TEST(ForEachPermutationFlat, MatchesRecursive) {
    using Axis1 = SameTypeUniqueValuePack<1, 2>;
    using Axis2 = SameTypeUniqueValuePack<10, 20>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::vector<std::pair<int, int>> recursive_result;
    for_each_permutation<Space, InstantiatedType, CollectInstantiated>::apply(recursive_result);

    std::vector<std::pair<int, int>> flat_result;
    for_each_permutation_flat<Space, InstantiatedType, CollectInstantiated>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

// Test with a three-axis space for for_each_permutation_flat
namespace {
template <auto V1, auto V2, auto V3> struct ThreeAxisInstType {
    static constexpr auto first  = V1;
    static constexpr auto second = V2;
    static constexpr auto third  = V3;
};

template <typename InstType, auto V1, auto V2, auto V3> struct CollectThreeAxis {
    static void
    apply(std::vector<std::tuple<int, int, int>> &out) {
        static_assert(InstType::first == V1);
        static_assert(InstType::second == V2);
        static_assert(InstType::third == V3);
        out.emplace_back(V1, V2, V3);
    }
};
} // namespace

TEST(ForEachPermutationFlat, ThreeAxesMatchesRecursive) {
    using Axis1 = SameTypeUniqueValuePack<0, 1>;
    using Axis2 = SameTypeUniqueValuePack<0, 1>;
    using Axis3 = SameTypeUniqueValuePack<0, 1>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    std::vector<std::tuple<int, int, int>> recursive_result;
    for_each_permutation<Space, ThreeAxisInstType, CollectThreeAxis>::apply(recursive_result);

    std::vector<std::tuple<int, int, int>> flat_result;
    for_each_permutation_flat<Space, ThreeAxisInstType, CollectThreeAxis>::apply(flat_result);

    ASSERT_EQ(flat_result.size(), 8u);
    ASSERT_EQ(flat_result.size(), recursive_result.size());
    for (size_t i = 0; i < flat_result.size(); ++i) {
        EXPECT_EQ(flat_result[i], recursive_result[i]) << "Mismatch at index " << i;
    }
}

} // namespace dispatch
} // namespace fvdb
