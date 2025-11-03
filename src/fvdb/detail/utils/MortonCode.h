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
/// @param x X coordinate (must be <= 2^21-1)
/// @param y Y coordinate (must be <= 2^21-1)
/// @param z Z coordinate (must be <= 2^21-1)
/// @return 64-bit Morton code
__hostdev__ constexpr inline uint64_t
morton(uint32_t const x, uint32_t const y, uint32_t const z) {
    return expand_lower_21_bits_by_3(x) | (expand_lower_21_bits_by_3(y) << 1) |
           (expand_lower_21_bits_by_3(z) << 2);
}

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

/// @brief Compute Morton code for 3D coordinates with offset handling
/// @param i I coordinate (can be negative)
/// @param j J coordinate (can be negative)
/// @param k K coordinate (can be negative)
/// @param offset_i Offset to make i non-negative
/// @param offset_j Offset to make j non-negative
/// @param offset_k Offset to make k non-negative
/// @return 64-bit Morton code
__hostdev__ inline uint64_t
morton_with_offset(int32_t const i,
                   int32_t const j,
                   int32_t const k,
                   int32_t const offset_i,
                   int32_t const offset_j,
                   int32_t const offset_k) {
    auto const x = static_cast<uint32_t>(i + offset_i);
    auto const y = static_cast<uint32_t>(j + offset_j);
    auto const z = static_cast<uint32_t>(k + offset_k);

    return morton(x, y, z);
}

} // namespace utils
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_MORTONCODE_H
