// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_MORTONCODE_H
#define FVDB_DETAIL_UTILS_MORTONCODE_H

#include <cstdint>
#include <nanovdb/util/Util.h>  // For __hostdev__ definition

namespace fvdb {
namespace detail {
namespace utils {

//-----------------------------------------------------------------------------
// This strange bit-shifting thing comes from here:
// http://programming.sirrida.de/calcperm.php
//-----------------------------------------------------------------------------

template <uint32_t mask, int shift>
__hostdev__ inline uint32_t
bit_permute_step(uint32_t x) {
    return ((x & mask) << shift) | ((x >> shift) & mask);
}

//-----------------------------------------------------------------------------
// 3D case
__hostdev__ inline uint32_t
expand_lower_10_bits_by_3(uint32_t x) {
    x = bit_permute_step<0x000007c0, 16>(x);  // Butterfly, stage 4
    x = bit_permute_step<0x00000038, 8>(x);   // Butterfly, stage 3
    x = bit_permute_step<0x04040004, 4>(x);   // Butterfly, stage 2
    x = bit_permute_step<0x02202202, 2>(x);   // Butterfly, stage 1
    return x & 0b01001001001001001001001001001u;
}

__hostdev__ inline uint64_t
expand_lower_21_bits_by_3(uint32_t const x) {
    static constexpr uint32_t bit_21 = 0x1 << 20;
    return static_cast<uint64_t>(expand_lower_10_bits_by_3(x)) |
           (static_cast<uint64_t>(expand_lower_10_bits_by_3(x >> 10)) << 30) |
           (static_cast<uint64_t>(x & bit_21) << 40);
}

/// @brief Compute Morton code for 3D coordinates
/// @param x X coordinate (must be <= 2^21-1)
/// @param y Y coordinate (must be <= 2^21-1)  
/// @param z Z coordinate (must be <= 2^21-1)
/// @return 64-bit Morton code
__hostdev__ inline uint64_t
morton(uint32_t const x, uint32_t const y, uint32_t const z) {
    return expand_lower_21_bits_by_3(x) |
           (expand_lower_21_bits_by_3(y) << 1) |
           (expand_lower_21_bits_by_3(z) << 2);
}

/// @brief Compute Morton code for 3D coordinates with offset handling
/// @param i I coordinate (can be negative)
/// @param j J coordinate (can be negative)
/// @param k K coordinate (can be negative)
/// @param offset_i Offset to make i non-negative
/// @param offset_j Offset to make j non-negative
/// @param offset_k Offset to make k non-negative
/// @return 64-bit Morton code
__hostdev__ inline uint64_t
morton_with_offset(int32_t const i, int32_t const j, int32_t const k,
                   uint32_t const offset_i, uint32_t const offset_j, uint32_t const offset_k) {
    // Add offsets to ensure non-negative coordinates
    uint32_t x = static_cast<uint32_t>(i + static_cast<int32_t>(offset_i));
    uint32_t y = static_cast<uint32_t>(j + static_cast<int32_t>(offset_j));
    uint32_t z = static_cast<uint32_t>(k + static_cast<int32_t>(offset_k));
    
    return morton(x, y, z);
}

/// @brief Calculates a 63-bit 3D Hilbert curve index for 21-bit coordinates.
/// This is a common, high-performance implementation that processes bits
/// from most-significant to least-significant.
/// @param x X coordinate (must be <= 2^21-1)
/// @param y Y coordinate (must be <= 2^21-1)  
/// @param z Z coordinate (must be <= 2^21-1)
/// @return 63-bit Hilbert index
__hostdev__ inline uint64_t
hilbert(uint32_t x, uint32_t y, uint32_t z) {
    // The number of bits per dimension.
    constexpr int bits = 21;
    uint64_t hilbert_index = 0;

    for (int s = bits - 1; s >= 0; --s) {
        uint32_t rx = (x >> s) & 1;
        uint32_t ry = (y >> s) & 1;
        uint32_t rz = (z >> s) & 1;

        // Determine the octant for the current level
        uint32_t octant = (rz << 2) | (ry << 1) | rx;

        // Append the 3-bit octant index to the Hilbert index
        hilbert_index <<= 3;
        hilbert_index |= octant;

        // Transform the coordinates to orient for the next sub-cube iteration.
        // This is the core of the Hilbert curve algorithm.
        switch (octant) {
            case 0: { // (0,0,0) -> (y, x, z)
                uint32_t temp = x; x = y; y = temp;
                break;
            }
            case 1: { // (0,0,1) -> (z, y, x)
                uint32_t temp = x; x = z; z = temp;
                break;
            }
            case 2: { // (0,1,0) -> (z, y, x)
                uint32_t temp = x; x = z; z = temp;
                break;
            }
            case 3: { // (0,1,1) -> (~x, z, y)
                uint32_t temp_x = x;
                uint32_t temp_y = y;
                uint32_t s_mask = (1u << s) - 1; // Mask for lower bits
                x = ~temp_x & s_mask;
                y = temp_y; // y remains y
                z = z;      // z remains z
                uint32_t temp_swap = z; z = y; y = temp_swap;
                break;
            }
            case 4: { // (1,0,0) -> (~x, z, y)
                uint32_t temp_x = x;
                uint32_t temp_y = y;
                uint32_t s_mask = (1u << s) - 1; // Mask for lower bits
                x = ~temp_x & s_mask;
                y = temp_y; // y remains y
                z = z;      // z remains z
                uint32_t temp_swap = z; z = y; y = temp_swap;
                break;
            }
            case 5: { // (1,0,1) -> (z, y, ~x)
                uint32_t s_mask = (1u << s) - 1;
                uint32_t temp = ~x & s_mask; x = z; z = temp;
                break;
            }
            case 6: { // (1,1,0) -> (y, x, ~z)
                uint32_t s_mask = (1u << s) - 1;
                uint32_t temp = ~z & s_mask; z = y; y = x; x = temp;
                break;
            }
            case 7: { // (1,1,1) -> (y, ~x, z)
                uint32_t s_mask = (1u << s) - 1;
                uint32_t temp = ~x & s_mask; x = y; y = temp;
                break;
            }
        }
    }
    return hilbert_index;
}

/// @brief Compute Hilbert curve index for 3D coordinates with offset handling
/// @param i I coordinate (can be negative)
/// @param j J coordinate (can be negative)
/// @param k K coordinate (can be negative)
/// @param offset_i Offset to make i non-negative
/// @param offset_j Offset to make j non-negative
/// @param offset_k Offset to make k non-negative
/// @return 63-bit Hilbert index
__hostdev__ inline uint64_t
hilbert_with_offset(int32_t const i, int32_t const j, int32_t const k,
                    uint32_t const offset_i, uint32_t const offset_j, uint32_t const offset_k) {
    // Add offsets to ensure non-negative coordinates
    uint32_t x = static_cast<uint32_t>(i + static_cast<int32_t>(offset_i));
    uint32_t y = static_cast<uint32_t>(j + static_cast<int32_t>(offset_j));
    uint32_t z = static_cast<uint32_t>(k + static_cast<int32_t>(offset_k));
    
    return hilbert(x, y, z);
}

} // namespace utils
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_MORTONCODE_H
