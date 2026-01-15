// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/IndexSpace.h>
#include <fvdb/detail/dispatch/Traits.h>

#include <gtest/gtest.h>

#include <string>

namespace fvdb {
namespace dispatch {

// Helper to convert an index_sequence coordinate to a string
struct CoordToString {
    template <typename Seq>
    static std::string
    toString(Seq) {
        constexpr auto arr = array_from_indices<Seq>::value;
        std::string result = "<";
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0)
                result += ", ";
            result += std::to_string(arr[i]);
        }
        result += ">";
        return result;
    }
};

TEST(IndexSpace, ConceptTestHelpers) {
    using Space  = Sizes<2, 3, 4>;
    using Scalar = Sizes<>;
    using Pt     = Point<1, 2>;

    // is_index_space
    static_assert(is_index_space<Space>());
    static_assert(is_index_space<Scalar>());
    static_assert(is_index_space<Pt>());
    static_assert(!is_index_space<int>());

    // is_index_point
    static_assert(is_index_point<Pt>());
    static_assert(is_index_point<Point<0, 0, 0>>());

    // is_tensor_index_space
    static_assert(is_tensor_index_space<Space>());
    static_assert(!is_tensor_index_space<Scalar>());
    static_assert(is_tensor_index_space<Pt>());

    // is_scalar_index_space
    static_assert(!is_scalar_index_space<Space>());
    static_assert(is_scalar_index_space<Scalar>());

    // is_non_empty_index_space
    static_assert(is_non_empty_index_space<Space>());
    static_assert(!is_non_empty_index_space<Scalar>());
    static_assert(is_non_empty_index_space<Pt>());
    static_assert(!is_non_empty_index_space<Sizes<0>>());
    static_assert(!is_non_empty_index_space<Sizes<2, 0, 3>>());

    // is_same_rank
    static_assert(is_same_rank<Sizes<2, 3>, Sizes<4, 5>>());
    static_assert(is_same_rank<Sizes<2, 3>, Point<1, 2>>());
    static_assert(!is_same_rank<Sizes<2, 3>, Sizes<4, 5, 6>>());
}

TEST(IndexSpace, RankAndNumel) {
    using Space = Sizes<2, 3, 4>;

    // Using _v() helpers
    static_assert(Rank_v<Space>() == 3);
    static_assert(Numel_v<Space>() == 24); // 2*3*4

    // Scalar space
    static_assert(Rank_v<Sizes<>>() == 0);
    static_assert(Numel_v<Sizes<>>() == 0);

    // Runtime checks
    EXPECT_EQ(Rank_v<Space>(), size_t{3});
    EXPECT_EQ(Numel_v<Space>(), size_t{24});
}

TEST(IndexSpace, ShapeType) {
    using Space = Sizes<2, 3, 4>;

    // Shape_t typedef
    static_assert(std::is_same_v<Shape_t<Space>, std::index_sequence<2, 3, 4>>);

    // Shape via array
    constexpr auto shape = array_from_indices<Shape_t<Space>>::value;
    EXPECT_EQ(shape.size(), size_t{3});
    EXPECT_EQ(shape[0], size_t{2});
    EXPECT_EQ(shape[1], size_t{3});
    EXPECT_EQ(shape[2], size_t{4});
}

TEST(IndexSpace, PrependType) {
    static_assert(std::is_same_v<Prepend_t<Sizes<2, 3>, 1>, Sizes<1, 2, 3>>);
    static_assert(std::is_same_v<Prepend_t<Sizes<>, 5>, Sizes<5>>);
}

TEST(IndexSpace, PointInBounds) {
    using Space = Sizes<3, 2, 5>;

    // Valid points (in bounds)
    static_assert(is_point_in_bounds<Space, Point<0, 0, 0>>());
    static_assert(is_point_in_bounds<Space, Point<2, 1, 4>>()); // max valid
    static_assert(is_point_in_bounds<Space, Point<1, 0, 3>>());

    // Invalid points (out of bounds)
    static_assert(!is_point_in_bounds<Space, Point<3, 0, 0>>()); // first dim out
    static_assert(!is_point_in_bounds<Space, Point<0, 2, 0>>()); // second dim out
    static_assert(!is_point_in_bounds<Space, Point<0, 0, 5>>()); // third dim out
    static_assert(!is_point_in_bounds<Space, Point<3, 2, 5>>()); // all out

    // Rank mismatch
    static_assert(!is_point_in_bounds<Space, Point<0, 0>>());
    static_assert(!is_point_in_bounds<Space, Point<0, 0, 0, 0>>());

    // Empty space/point
    static_assert(is_point_in_bounds<Sizes<>, Point<>>());
}

TEST(IndexSpace, LinearIndexFromPoint) {
    using Space = Sizes<2, 3, 4>;

    // Using _v() helper
    static_assert(LinearIndexFromPoint_v<Space, Point<0, 0, 0>>() == 0);
    static_assert(LinearIndexFromPoint_v<Space, Point<0, 0, 1>>() == 1);
    static_assert(LinearIndexFromPoint_v<Space, Point<0, 1, 0>>() == 4);
    static_assert(LinearIndexFromPoint_v<Space, Point<1, 0, 0>>() == 12);
    static_assert(LinearIndexFromPoint_v<Space, Point<1, 2, 3>>() == 23); // last valid

    // Runtime checks
    EXPECT_EQ((LinearIndexFromPoint_v<Space, Point<0, 0, 0>>()), size_t{0});
    EXPECT_EQ((LinearIndexFromPoint_v<Space, Point<0, 0, 1>>()), size_t{1});
    EXPECT_EQ((LinearIndexFromPoint_v<Space, Point<0, 1, 0>>()), size_t{4});
    EXPECT_EQ((LinearIndexFromPoint_v<Space, Point<1, 0, 0>>()), size_t{12});
    EXPECT_EQ((LinearIndexFromPoint_v<Space, Point<1, 2, 3>>()), size_t{23});
}

TEST(IndexSpace, PointFromLinearIndex) {
    using Space = Sizes<2, 3>;

    // Using _t typedef
    static_assert(std::is_same_v<PointFromLinearIndex_t<Space, 0>, std::index_sequence<0, 0>>);
    static_assert(std::is_same_v<PointFromLinearIndex_t<Space, 1>, std::index_sequence<0, 1>>);
    static_assert(std::is_same_v<PointFromLinearIndex_t<Space, 2>, std::index_sequence<0, 2>>);
    static_assert(std::is_same_v<PointFromLinearIndex_t<Space, 3>, std::index_sequence<1, 0>>);
    static_assert(std::is_same_v<PointFromLinearIndex_t<Space, 5>, std::index_sequence<1, 2>>);
}

TEST(IndexSpace, RoundTripConversion) {
    using Space = Sizes<2, 3, 4>;

    // Linear -> Point -> Linear roundtrip
    static_assert(LinearIndexFromPoint_v<Space, PointFromLinearIndex_t<Space, 0>>() == 0);
    static_assert(LinearIndexFromPoint_v<Space, PointFromLinearIndex_t<Space, 7>>() == 7);
    static_assert(LinearIndexFromPoint_v<Space, PointFromLinearIndex_t<Space, 23>>() == 23);
}

TEST(IndexSpace, VisitSpace) {
    using Space = Sizes<2, 3>;
    std::string output;
    visit_index_space(
        [&output](auto &&coord) {
            if (!output.empty()) {
                output += ", ";
            }

            output += CoordToString::toString(coord);
        },
        Space{});

    EXPECT_EQ(output, "<0, 0>, <0, 1>, <0, 2>, <1, 0>, <1, 1>, <1, 2>");
}

TEST(IndexSpace, VisitIndexSpaces) {
    using Space1 = Sizes<2, 3>;
    using Space2 = Sizes<3, 2>;
    std::string output;
    visit_index_spaces(
        [&output](auto &&coord) {
            if (!output.empty()) {
                output += ", ";
            }
            output += CoordToString::toString(coord);
        },
        Space1{},
        Space2{});
    std::string expected_space_1 = "<0, 0>, <0, 1>, <0, 2>, <1, 0>, <1, 1>, <1, 2>";
    std::string expected_space_2 = "<0, 0>, <0, 1>, <1, 0>, <1, 1>, <2, 0>, <2, 1>";
    std::string expected_output  = expected_space_1 + ", " + expected_space_2;
    EXPECT_EQ(output, expected_output);
}

} // namespace dispatch
} // namespace fvdb
