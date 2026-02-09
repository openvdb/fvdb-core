// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for with_type and with_value concepts.
// These are the primary matching primitives for concept-constrained dispatch.
//
#include "dispatch/detail/core_types.h"
#include "dispatch/with_value.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace dispatch {

// ============================================================================
// Test enums with type_label specializations
// ============================================================================

enum class dev { cpu, gpu, pvt1 };
enum class stype { f16, f32, f64, i32, i64 };
enum class algo { clever, quick, lazy, stupid };

template <> struct type_label<dev> {
    static consteval auto
    value() {
        return fixed_label("test.dev");
    }
};

template <> struct type_label<stype> {
    static consteval auto
    value() {
        return fixed_label("test.stype");
    }
};

template <> struct type_label<algo> {
    static consteval auto
    value() {
        return fixed_label("test.algo");
    }
};

// ============================================================================
// with_type: does the tag contain any value of this type?
// ============================================================================

TEST(WithType, Present) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    static_assert(with_type<T, dev>);
    static_assert(with_type<T, stype>);
    static_assert(with_type<T, algo>);
}

TEST(WithType, Absent) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    static_assert(!with_type<T, placement>);
    static_assert(!with_type<T, determinism>);
    static_assert(!with_type<T, contiguity>);
}

TEST(WithType, SingleValueTag) {
    using T = tag<dev::gpu>;

    static_assert(with_type<T, dev>);
    static_assert(!with_type<T, stype>);
    static_assert(!with_type<T, algo>);
}

// ============================================================================
// with_value: does the tag contain this specific value?
// ============================================================================

TEST(WithValue, ExactMatch) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    static_assert(with_value<T, dev::cpu>);
    static_assert(with_value<T, stype::f32>);
    static_assert(with_value<T, algo::clever>);
}

TEST(WithValue, WrongValue) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    // Right type, wrong value
    static_assert(!with_value<T, dev::gpu>);
    static_assert(!with_value<T, stype::f64>);
    static_assert(!with_value<T, algo::lazy>);
}

TEST(WithValue, AbsentType) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    // Type not in tag at all
    static_assert(!with_value<T, placement::in_place>);
    static_assert(!with_value<T, contiguity::contiguous>);
}

TEST(WithValue, SingleValueTag) {
    using T = tag<dev::gpu>;

    static_assert(with_value<T, dev::gpu>);
    static_assert(!with_value<T, dev::cpu>);
}

// ============================================================================
// Order independence: tag normalization + with_value
// ============================================================================

TEST(WithValue, OrderIndependent) {
    using T1 = tag<dev::gpu, stype::f16, algo::quick>;
    using T2 = tag<algo::quick, dev::gpu, stype::f16>;
    using T3 = tag<stype::f16, algo::quick, dev::gpu>;

    // All resolve to the same type
    static_assert(std::is_same_v<T1, T2>);
    static_assert(std::is_same_v<T2, T3>);

    // with_value works on all
    static_assert(with_value<T1, dev::gpu>);
    static_assert(with_value<T2, dev::gpu>);
    static_assert(with_value<T3, dev::gpu>);
    static_assert(with_value<T1, stype::f16>);
    static_assert(with_value<T2, algo::quick>);
}

// ============================================================================
// Concept composition with &&
// ============================================================================

TEST(WithValue, ConceptComposition) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    // Two-way conjunction
    static_assert(with_value<T, dev::cpu> && with_value<T, stype::f32>);
    static_assert(with_value<T, dev::cpu> && with_value<T, algo::clever>);
    static_assert(with_value<T, stype::f32> && with_value<T, algo::clever>);

    // Three-way conjunction
    static_assert(with_value<T, dev::cpu> && with_value<T, stype::f32> &&
                  with_value<T, algo::clever>);

    // Partial mismatch
    static_assert(!(with_value<T, dev::gpu> && with_value<T, stype::f32>));
    static_assert(!(with_value<T, dev::cpu> && with_value<T, stype::f64>));
}

// ============================================================================
// with_value with dispatch enums
// ============================================================================

TEST(WithValue, DispatchEnums) {
    using T = tag<placement::in_place, determinism::required, contiguity::contiguous>;

    static_assert(with_value<T, placement::in_place>);
    static_assert(!with_value<T, placement::out_of_place>);
    static_assert(with_value<T, determinism::required>);
    static_assert(!with_value<T, determinism::not_required>);
    static_assert(with_value<T, contiguity::contiguous>);
    static_assert(!with_value<T, contiguity::strided>);
}

// ============================================================================
// with_value subsumes with_type
// ============================================================================
// This is tested via concept semantics: with_value<Tag, V> is defined as
// with_type<Tag, decltype(V)> && is_with_value_v<Tag, V>(), ensuring that
// a constraint using with_value is more specific than one using with_type.

TEST(WithValue, SubsumptionRelationship) {
    using T = tag<dev::cpu, stype::f32>;

    // with_value implies with_type
    static_assert(with_value<T, dev::cpu>);
    static_assert(with_type<T, dev>);

    // with_type does not imply with_value
    static_assert(with_type<T, dev>);        // true
    static_assert(with_value<T, dev::cpu>);  // true
    static_assert(!with_value<T, dev::gpu>); // false — type present, value wrong
}

// ============================================================================
// is_with_type / is_with_value traits (struct-based)
// ============================================================================

TEST(IsWithType, Trait) {
    using T = tag<dev::cpu, stype::f32>;

    static_assert(is_with_type_v<T, dev>());
    static_assert(is_with_type_v<T, stype>());
    static_assert(!is_with_type_v<T, algo>());
}

TEST(IsWithValue, Trait) {
    using T = tag<dev::cpu, stype::f32>;

    static_assert(is_with_value_v<T, dev::cpu>());
    static_assert(!is_with_value_v<T, dev::gpu>());
    static_assert(is_with_value_v<T, stype::f32>());
    static_assert(!is_with_value_v<T, stype::f64>());
}

// ============================================================================
// tag_get: extract value(s) from a tag by type
// ============================================================================

TEST(TagGet, SingleType) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    constexpr auto d = tag_get<dev>(T{});
    constexpr auto s = tag_get<stype>(T{});
    constexpr auto a = tag_get<algo>(T{});

    static_assert(d == dev::cpu);
    static_assert(s == stype::f32);
    static_assert(a == algo::clever);
}

TEST(TagGet, SingleTypeDefaultArg) {
    using T = tag<dev::gpu, stype::f64>;

    // Can call without an argument when Tag is explicit
    constexpr auto d = tag_get<dev, T>();
    constexpr auto s = tag_get<stype, T>();

    static_assert(d == dev::gpu);
    static_assert(s == stype::f64);
}

TEST(TagGet, MultipleTypes) {
    using T = tag<dev::cpu, stype::f32, algo::clever>;

    auto const [d, s]       = tag_get<dev, stype>(T{});
    auto const [d2, a]      = tag_get<dev, algo>(T{});
    auto const [s2, a2]     = tag_get<stype, algo>(T{});
    auto const [d3, s3, a3] = tag_get<dev, stype, algo>(T{});

    EXPECT_EQ(d, dev::cpu);
    EXPECT_EQ(s, stype::f32);
    EXPECT_EQ(d2, dev::cpu);
    EXPECT_EQ(a, algo::clever);
    EXPECT_EQ(s2, stype::f32);
    EXPECT_EQ(a2, algo::clever);
    EXPECT_EQ(d3, dev::cpu);
    EXPECT_EQ(s3, stype::f32);
    EXPECT_EQ(a3, algo::clever);

    // Verify multi-type result is constexpr-usable
    static_assert(std::get<0>(tag_get<dev, stype>(T{})) == dev::cpu);
    static_assert(std::get<1>(tag_get<dev, stype>(T{})) == stype::f32);
    static_assert(std::get<0>(tag_get<dev, stype, algo>(T{})) == dev::cpu);
    static_assert(std::get<1>(tag_get<dev, stype, algo>(T{})) == stype::f32);
    static_assert(std::get<2>(tag_get<dev, stype, algo>(T{})) == algo::clever);
}

TEST(TagGet, OrderIndependent) {
    using T1 = tag<dev::gpu, stype::f16>;
    using T2 = tag<stype::f16, dev::gpu>;

    // Same tag, same results
    static_assert(tag_get<dev>(T1{}) == tag_get<dev>(T2{}));
    static_assert(tag_get<stype>(T1{}) == tag_get<stype>(T2{}));

    // Multi-type extraction also order-independent
    static_assert(std::get<0>(tag_get<dev, stype>(T1{})) == std::get<0>(tag_get<dev, stype>(T2{})));
    static_assert(std::get<1>(tag_get<dev, stype>(T1{})) == std::get<1>(tag_get<dev, stype>(T2{})));
}

TEST(TagGet, WithDispatchEnums) {
    using T = tag<placement::in_place, determinism::required, contiguity::contiguous>;

    constexpr auto p = tag_get<placement>(T{});
    constexpr auto d = tag_get<determinism>(T{});
    constexpr auto c = tag_get<contiguity>(T{});

    static_assert(p == placement::in_place);
    static_assert(d == determinism::required);
    static_assert(c == contiguity::contiguous);

    static_assert(std::get<0>(tag_get<placement, determinism, contiguity>(T{})) ==
                  placement::in_place);
    static_assert(std::get<1>(tag_get<placement, determinism, contiguity>(T{})) ==
                  determinism::required);
    static_assert(std::get<2>(tag_get<placement, determinism, contiguity>(T{})) ==
                  contiguity::contiguous);
}

} // namespace dispatch
