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

TEST(IsDimensionalAxis, DimensionalAxisIsTrue) {
    EXPECT_TRUE((is_dimensional_axis_v<DimensionalAxis<1, 2, 3>>));
    EXPECT_TRUE((is_dimensional_axis_v<DimensionalAxis<'a', 'b'>>));
    EXPECT_TRUE((is_dimensional_axis_v<DimensionalAxis<true>>));
}

TEST(IsDimensionalAxis, OtherTypesAreFalse) {
    EXPECT_FALSE(is_dimensional_axis_v<int>);
    EXPECT_FALSE((is_dimensional_axis_v<std::tuple<int, int>>));
    EXPECT_FALSE((is_dimensional_axis_v<SameTypeValuePack<1, 2, 3>>));
    EXPECT_FALSE((is_dimensional_axis_v<AnyTypeValuePack<1, 2, 3>>));
}

// =============================================================================
// AxisOuterProduct Basic Tests
// =============================================================================

TEST(AxisOuterProduct, SingleAxisHasCorrectProperties) {
    using Axis  = DimensionalAxis<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    EXPECT_EQ(Space::rank, 1u);
    EXPECT_EQ(Space::numel, 3u);
}

TEST(AxisOuterProduct, TwoAxesHaveCorrectSize) {
    using Axis1 = DimensionalAxis<1, 2>;
    using Axis2 = DimensionalAxis<10, 20, 30>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    EXPECT_EQ(Space::rank, 2u);
    EXPECT_EQ(Space::numel, 6u); // 2 * 3
}

TEST(AxisOuterProduct, ThreeAxesHaveCorrectSize) {
    using Axis1 = DimensionalAxis<1, 2>;
    using Axis2 = DimensionalAxis<'a', 'b', 'c'>;
    using Axis3 = DimensionalAxis<true, false>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    EXPECT_EQ(Space::rank, 3u);
    EXPECT_EQ(Space::numel, 12u); // 2 * 3 * 2
}

// =============================================================================
// is_axis_outer_product Tests
// =============================================================================

TEST(IsAxisOuterProduct, AxisOuterProductIsTrue) {
    using Space = AxisOuterProduct<DimensionalAxis<1, 2>>;
    EXPECT_TRUE(is_axis_outer_product_v<Space>);
}

TEST(IsAxisOuterProduct, OtherTypesAreFalse) {
    EXPECT_FALSE(is_axis_outer_product_v<int>);
    EXPECT_FALSE((is_axis_outer_product_v<SameTypeUniqueValuePack<1, 2>>));
}

TEST(IsAxisOuterProduct, ConceptWorks) {
    using Space = AxisOuterProduct<SameTypeUniqueValuePack<1, 2>>;
    EXPECT_TRUE(AxisOuterProductConcept<Space>);
    EXPECT_FALSE(AxisOuterProductConcept<int>);
}

// =============================================================================
// AxisOuterProduct CoordFromIndices Tests
// =============================================================================

TEST(AxisOuterProduct, CoordFromIndicesType) {
    using Axis1 = DimensionalAxis<1, 2, 3>;
    using Axis2 = DimensionalAxis<'a', 'b', 'c', 'd'>;
    using Axis3 = DimensionalAxis<true, false>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    using ExpectedCoordType1 = AnyTypeValuePack<1, 'a', true>;
    using IndicesType1       = std::index_sequence<0, 0, 0>;
    using CoordType1         = Space::coord_from_indices_type<IndicesType1>;
    EXPECT_TRUE((std::is_same_v<CoordType1, ExpectedCoordType1>));

    using ExpectedCoordType2 = AnyTypeValuePack<3, 'd', false>;
    using IndicesType2       = std::index_sequence<2, 3, 1>;
    using CoordType2         = Space::coord_from_indices_type<IndicesType2>;
    EXPECT_TRUE((std::is_same_v<CoordType2, ExpectedCoordType2>));
}

#if 0

TEST(AxisOuterProduct, IndicesFromCoordType) {
    using Axis1 = DimensionalAxis<1, 2, 3>;
    using Axis2 = DimensionalAxis<'a', 'b', 'c', 'd'>;
    using Axis3 = DimensionalAxis<true, false>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    // Direct conversion: coord values -> indices
    using CoordType1       = AnyTypeValuePack<2, 'c', false>;
    using ExpectedIndices1 = std::index_sequence<1, 2, 1>;
    using IndicesType1     = Space::indices_from_coord_type<CoordType1>;
    EXPECT_TRUE((std::is_same_v<IndicesType1, ExpectedIndices1>));

    // Round-trip: indices -> coord -> indices (should be identity)
    using OriginalIndices  = std::index_sequence<2, 3, 0>;
    using CoordFromIndices = Space::coord_from_indices_type<OriginalIndices>;
    using IndicesFromCoord = Space::indices_from_coord_type<CoordFromIndices>;
    EXPECT_TRUE((std::is_same_v<IndicesFromCoord, OriginalIndices>));
}

// =============================================================================
// is_subspace_of Tests
// =============================================================================

TEST(IsSubspaceOf, SpaceIsSubspaceOfItself) {
    using Space = AxisOuterProduct<DimensionalAxis<1, 2, 3>>;
    EXPECT_TRUE((is_subspace_of_v<Space, Space>));
}

TEST(IsSubspaceOf, SmallerAxisIsSubspace) {
    using SubAxis  = DimensionalAxis<1, 2>;
    using FullAxis = DimensionalAxis<1, 2, 3, 4>;
    using Sub      = AxisOuterProduct<SubAxis>;
    using Full     = AxisOuterProduct<FullAxis>;

    EXPECT_TRUE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, MultipleAxesSubspace) {
    using SubAxis1  = DimensionalAxis<1>;
    using SubAxis2  = DimensionalAxis<'a', 'b'>;
    using FullAxis1 = DimensionalAxis<1, 2, 3>;
    using FullAxis2 = DimensionalAxis<'a', 'b', 'c'>;

    using Sub  = AxisOuterProduct<SubAxis1, SubAxis2>;
    using Full = AxisOuterProduct<FullAxis1, FullAxis2>;

    EXPECT_TRUE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, DifferentAxisCountIsNotSubspace) {
    using OneAxis = AxisOuterProduct<DimensionalAxis<1, 2>>;
    using TwoAxes = AxisOuterProduct<DimensionalAxis<1, 2>, DimensionalAxis<3, 4>>;

    // Different axis counts cannot be subspaces of each other
    EXPECT_FALSE((is_subspace_of_v<OneAxis, TwoAxes>));
    EXPECT_FALSE((is_subspace_of_v<TwoAxes, OneAxis>));
}

TEST(IsSubspaceOf, DisjointValuesIsNotSubspace) {
    using Sub  = AxisOuterProduct<DimensionalAxis<1, 2>>;
    using Full = AxisOuterProduct<DimensionalAxis<3, 4, 5>>;

    EXPECT_FALSE((is_subspace_of_v<Sub, Full>));
}

TEST(IsSubspaceOf, PartialOverlapIsNotSubspace) {
    // Only some values overlap - not a proper subspace
    using Sub  = AxisOuterProduct<DimensionalAxis<1, 2, 99>>;
    using Full = AxisOuterProduct<DimensionalAxis<1, 2, 3>>;

    EXPECT_FALSE((is_subspace_of_v<Sub, Full>));
}

// =============================================================================
// AxisOuterProduct Visit Tests
// =============================================================================

// Helper to convert an AnyTypeValuePack coordinate to a string
struct ValueCoordToString {
    template <auto... Values>
    static std::string
    toString(AnyTypeValuePack<Values...>) {
        std::string result = "<";
        bool first         = true;
        // Use fold expression to iterate through values
        ((result += (first ? (first = false, "") : ", ") + valueToString(Values)), ...);
        result += ">";
        return result;
    }

  private:
    template <typename T>
    static std::string
    valueToString(T val) {
        if constexpr (std::is_same_v<T, bool>) {
            return val ? "true" : "false";
        } else if constexpr (std::is_same_v<T, char>) {
            return std::string("'") + val + "'";
        } else {
            return std::to_string(val);
        }
    }
};

TEST(AxisOuterProduct, VisitSingleAxis) {
    using Axis  = DimensionalAxis<10, 20, 30>;
    using Space = AxisOuterProduct<Axis>;

    std::string output;
    Space::visit([&output](auto coord) {
        if (!output.empty()) {
            output += ", ";
        }
        output += ValueCoordToString::toString(coord);
    });

    EXPECT_EQ(output, "<10>, <20>, <30>");
}

TEST(AxisOuterProduct, VisitTwoAxes) {
    using Axis1 = DimensionalAxis<1, 2>;
    using Axis2 = DimensionalAxis<'a', 'b', 'c'>;
    using Space = AxisOuterProduct<Axis1, Axis2>;

    std::string output;
    Space::visit([&output](auto coord) {
        if (!output.empty()) {
            output += ", ";
        }
        output += ValueCoordToString::toString(coord);
    });

    // Row-major order: last axis varies fastest
    EXPECT_EQ(output, "<1, 'a'>, <1, 'b'>, <1, 'c'>, <2, 'a'>, <2, 'b'>, <2, 'c'>");
}

TEST(AxisOuterProduct, VisitThreeAxesMixedTypes) {
    using Axis1 = DimensionalAxis<1, 2>;
    using Axis2 = DimensionalAxis<'x', 'y'>;
    using Axis3 = DimensionalAxis<true, false>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    std::vector<std::string> coords;
    Space::visit([&coords](auto coord) { coords.push_back(ValueCoordToString::toString(coord)); });

    EXPECT_EQ(coords.size(), 8u); // 2 * 2 * 2
    // First coordinate should be <1, 'x', true>
    EXPECT_EQ(coords[0], "<1, 'x', true>");
    // Last coordinate should be <2, 'y', false>
    EXPECT_EQ(coords[7], "<2, 'y', false>");
}

TEST(AxisOuterProduct, VisitCountsCorrectly) {
    using Axis1 = DimensionalAxis<1, 2, 3>;
    using Axis2 = DimensionalAxis<10, 20>;
    using Axis3 = DimensionalAxis<100, 200, 300, 400>;
    using Space = AxisOuterProduct<Axis1, Axis2, Axis3>;

    size_t count = 0;
    Space::visit([&count](auto) { ++count; });

    EXPECT_EQ(count, Space::numel);
    EXPECT_EQ(count, 24u); // 3 * 2 * 4
}

#endif

} // namespace dispatch
} // namespace fvdb
