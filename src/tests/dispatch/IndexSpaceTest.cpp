// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/IndexSpace.h>

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
    using Space = IndexSpace<2, 3, 4>;

    // Constants
    EXPECT_EQ(Space::rank, size_t{3});
    EXPECT_EQ(Space::numel, size_t{24}); // 2*3*4

    // Shape array
    EXPECT_EQ(Space::shape.size(), size_t{3});
    EXPECT_EQ(Space::shape[0], size_t{2});
    EXPECT_EQ(Space::shape[1], size_t{3});
    EXPECT_EQ(Space::shape[2], size_t{4});

    // Strides array (row-major: 3*4=12, 4, 1)
    EXPECT_EQ(Space::strides.size(), size_t{3});
    EXPECT_EQ(Space::strides[0], size_t{12});
    EXPECT_EQ(Space::strides[1], size_t{4});
    EXPECT_EQ(Space::strides[2], size_t{1});

    // Type aliases
    static_assert(std::is_same_v<Space::shape_seq, std::index_sequence<2, 3, 4>>);
    static_assert(std::is_same_v<Space::strides_seq, std::index_sequence<12, 4, 1>>);
    static_assert(std::is_same_v<Space::Coord, std::array<size_t, 3>>);
}

TEST(IndexSpace, LinearIndexAndCoord) {
    using Space = IndexSpace<2, 3, 4>;

    // coord → linear_index
    EXPECT_EQ(Space::linear_index({0, 0, 0}), size_t{0});
    EXPECT_EQ(Space::linear_index({0, 0, 1}), size_t{1});
    EXPECT_EQ(Space::linear_index({0, 1, 0}), size_t{4});
    EXPECT_EQ(Space::linear_index({1, 0, 0}), size_t{12});
    EXPECT_EQ(Space::linear_index({1, 2, 3}), size_t{23}); // last valid element

    // linear_index → coord (round-trip)
    for (size_t i = 0; i < Space::numel; ++i) {
        EXPECT_EQ(Space::linear_index(Space::coord(i)), i);
    }
}

TEST(IndexSpace, LinearIndexCoordType) {
    using Space = IndexSpace<2, 3>;

    // Verify compile-time coordinate types
    static_assert(std::is_same_v<Space::LinearIndexCoord<0>::type, std::index_sequence<0, 0>>);
    static_assert(std::is_same_v<Space::LinearIndexCoord<1>::type, std::index_sequence<0, 1>>);
    static_assert(std::is_same_v<Space::LinearIndexCoord<2>::type, std::index_sequence<0, 2>>);
    static_assert(std::is_same_v<Space::LinearIndexCoord<3>::type, std::index_sequence<1, 0>>);
    static_assert(std::is_same_v<Space::LinearIndexCoord<5>::type, std::index_sequence<1, 2>>);
}

TEST(IndexSpace, VisitMethod) {
    using Space = IndexSpace<2, 3>;
    std::string output;
    Space::visit([&output](auto &&coord) {
        if (!output.empty()) {
            output += ", ";
        }

        output += CoordToString::toString(coord);
    });

    EXPECT_EQ(output, "<0, 0>, <0, 1>, <0, 2>, <1, 0>, <1, 1>, <1, 2>");
}

TEST(IndexSpace, VisitSpacesFunction) {
    using Space1 = IndexSpace<2, 3>;
    using Space2 = IndexSpace<3, 2>;
    std::string output;
    visit_spaces(
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
