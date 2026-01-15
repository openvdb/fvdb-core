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

TEST(IndexSpace, ConstantsAndTypes) {
    using Space = Sizes<2, 3, 4>;

    // Constants
    EXPECT_EQ(Rank<Space>::value(), size_t{3});
    EXPECT_EQ(Numel<Space>::value(), size_t{24}); // 2*3*4

    // Shape via get_shape()
    constexpr auto shape = array_from_indices<Shape<Space>::type>::value;
    EXPECT_EQ(shape.size(), size_t{3});
    EXPECT_EQ(shape[0], size_t{2});
    EXPECT_EQ(shape[1], size_t{3});
    EXPECT_EQ(shape[2], size_t{4});

    // Type aliases
    static_assert(std::is_same_v<Shape<Space>::type, std::index_sequence<2, 3, 4>>);
}

TEST(IndexSpace, LinearIndexFromIndices) {
    using Space = Sizes<2, 3, 4>;

    // coord → linear_index
    EXPECT_EQ((LinearIndexFromIndices<Space, Indices<0, 0, 0>>::value()), size_t{0});
    EXPECT_EQ((LinearIndexFromIndices<Space, Indices<0, 0, 1>>::value()), size_t{1});
    EXPECT_EQ((LinearIndexFromIndices<Space, Indices<0, 1, 0>>::value()), size_t{4});
    EXPECT_EQ((LinearIndexFromIndices<Space, Indices<1, 0, 0>>::value()), size_t{12});
    EXPECT_EQ((LinearIndexFromIndices<Space, Indices<1, 2, 3>>::value()),
              size_t{23}); // last valid element
}

TEST(IndexSpace, IndicesFromLinearIndex) {
    using Space = Sizes<2, 3>;

    // Verify compile-time coordinate types
    static_assert(
        std::is_same_v<IndicesFromLinearIndex<Space, 0>::type, std::index_sequence<0, 0>>);
    static_assert(
        std::is_same_v<IndicesFromLinearIndex<Space, 1>::type, std::index_sequence<0, 1>>);
    static_assert(
        std::is_same_v<IndicesFromLinearIndex<Space, 2>::type, std::index_sequence<0, 2>>);
    static_assert(
        std::is_same_v<IndicesFromLinearIndex<Space, 3>::type, std::index_sequence<1, 0>>);
    static_assert(
        std::is_same_v<IndicesFromLinearIndex<Space, 5>::type, std::index_sequence<1, 2>>);
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
