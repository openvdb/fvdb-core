// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Morton curve implementation by Christopher Horvath
//
#ifndef FVDB_DETAIL_UTILS_MORTONCODE_H
#define FVDB_DETAIL_UTILS_MORTONCODE_H

#include <nanovdb/util/Util.h> // For __hostdev__ definition

#include <cstdint>

namespace fvdb {
namespace detail {
namespace utils {

// Expand the lower 21 bits of a uint32_t into a uint64_t,
// inserting two zero bits between each source bit.
// Result has set bits at positions 0,3,6,...,60.
__hostdev__ constexpr inline uint64_t
expand_lower_21_bits_by_3(uint32_t const x) {
    uint64_t v = static_cast<uint64_t>(x & 0x001FFFFFu); // 21 bits

    // 64-bit part1by2 (insert two 0s between bits), classic shift-and-mask cascade
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v << 8)) & 0x100F00F00F00F00Full;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v << 2)) & 0x1249249249249249ull; // target positions 0,3,6,...
    return v;
}

/// @brief Compute Morton code for 3D coordinates
/// @param i I coordinate (must be 0 <= i < 2^21)
/// @param j J coordinate (must be 0 <= j < 2^21)
/// @param k K coordinate (must be 0 <= k < 2^21)
/// @return 64-bit Morton code
__hostdev__ constexpr inline uint64_t
morton(uint32_t const i, uint32_t const j, uint32_t const k) {
    return (expand_lower_21_bits_by_3(i) << 0) | (expand_lower_21_bits_by_3(j) << 1) |
           (expand_lower_21_bits_by_3(k) << 2);
}

static_assert(morton(0, 0, 0) == 0b000, "morton(0, 0, 0) should be 0b000");
static_assert(morton(1, 0, 0) == 0b001, "morton(1, 0, 0) should be 0b001");
static_assert(morton(0, 1, 0) == 0b010, "morton(0, 1, 0) should be 0b010");
static_assert(morton(1, 1, 0) == 0b011, "morton(1, 1, 0) should be 0b011");
static_assert(morton(0, 0, 1) == 0b100, "morton(0, 0, 1) should be 0b100");
static_assert(morton(1, 0, 1) == 0b101, "morton(1, 0, 1) should be 0b101");
static_assert(morton(0, 1, 1) == 0b110, "morton(0, 1, 1) should be 0b110");
static_assert(morton(1, 1, 1) == 0b111, "morton(1, 1, 1) should be 0b111");

static_assert(morton(0, 0, 0) == 0b000000, "morton(0, 0, 0) should be 0b000000");
static_assert(morton(2, 0, 0) == 0b001000, "morton(2, 0, 0) should be 0b001000");
static_assert(morton(0, 2, 0) == 0b010000, "morton(0, 2, 0) should be 0b010000");
static_assert(morton(2, 2, 0) == 0b011000, "morton(2, 2, 0) should be 0b011000");
static_assert(morton(0, 0, 2) == 0b100000, "morton(0, 0, 2) should be 0b100000");
static_assert(morton(2, 0, 2) == 0b101000, "morton(2, 0, 2) should be 0b101000");
static_assert(morton(0, 2, 2) == 0b110000, "morton(0, 2, 2) should be 0b110000");
static_assert(morton(2, 2, 2) == 0b111000, "morton(2, 2, 2) should be 0b111000");

static_assert(morton(0, 0, 0) == 0b000000, "morton(0, 0, 0) should be 0b000000");
static_assert(morton(3, 0, 0) == 0b001001, "morton(3, 0, 0) should be 0b001001");
static_assert(morton(0, 3, 0) == 0b010010, "morton(0, 3, 0) should be 0b010010");
static_assert(morton(3, 3, 0) == 0b011011, "morton(3, 3, 0) should be 0b011011");
static_assert(morton(0, 0, 3) == 0b100100, "morton(0, 0, 3) should be 0b100100");
static_assert(morton(3, 0, 3) == 0b101101, "morton(3, 0, 3) should be 0b101101");
static_assert(morton(0, 3, 3) == 0b110110, "morton(0, 3, 3) should be 0b110110");
static_assert(morton(3, 3, 3) == 0b111111, "morton(3, 3, 3) should be 0b111111");

// ---- Static self-check ----
constexpr inline uint64_t
_expected_mask_21bits_spaced3() {
    uint64_t m = 0;
    for (int i = 0; i < 21; ++i)
        m |= (1ull << (3 * i));
    return m;
}

static_assert(expand_lower_21_bits_by_3(0x1FFFFFu) == _expected_mask_21bits_spaced3(),
              "21-bit expand failed");

} // namespace utils
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_MORTONCODE_H
