// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Hilbert curve implementation provided by Christopher Horvath
//
#ifndef FVDB_DETAIL_UTILS_HILBERTCODE_H
#define FVDB_DETAIL_UTILS_HILBERTCODE_H

#include <nanovdb/util/Util.h>

#include <cstdint>
#include <cstdlib>

namespace fvdb {
namespace detail {
namespace utils {

constexpr uint32_t _HILBERT_NBITS = 21;
constexpr uint32_t _HILBERT_MASK  = (1u << _HILBERT_NBITS) - 1;

// One full "round" of the nested loop for a given bit position,
// applied to the three per-dimension packed bitfields (g0, g1, g2).
// This performs exactly the operations:
//   - if gray[dim, bit] == 1: invert gray[0, bit+1 :]
//   - else: swap gray[dim, bit+1 :] with gray[0, bit+1 :]
//
// Here, bits are stored little-endian in the integers, but indexes
// follow the "bit" (0..20) which is MSB-first. We map
// index `bit` to integer bit position `p = NBITS-1-bit`.
// The suffix `bit+1:` becomes the lower `p` bits in the integer.
__hostdev__ inline void
hilbert_step_for_bit(uint32_t p, uint32_t &g0, uint32_t &g1, uint32_t &g2) {
    uint32_t const M = (p == 0) ? 0u : ((1u << p) - 1u);

    // dim = 0
    if ((g0 >> p) & 1u) {
        // invert gray[0, bit+1:]
        g0 ^= M;
    }
    // else {
    //  swapping with itself -> no-op (kept for faithful structure)
    // t = (g0 ^ g0) & M; g0 ^= t; g0 ^= t;
    //}

    // dim = 1
    if ((g1 >> p) & 1u) {
        // invert lower bits of dim 0
        g0 ^= M;
    } else {
        // swap lower bits between dim 1 and dim 0
        uint32_t const t = (g0 ^ g1) & M;
        g1 ^= t;
        g0 ^= t;
    }

    // dim = 2
    if ((g2 >> p) & 1u) {
        g0 ^= M;
    } else {
        uint32_t const t = (g0 ^ g2) & M;
        g2 ^= t;
        g0 ^= t;
    }
}

// Stream the post-loop per-dimension Gray bits into a single
// 63-bit sequence in the *same order*:
//   for bit in [0..20] (MSB to LSB):
//     emit g0[bit], g1[bit], g2[bit]
// Then convert Gray to binary via prefix XOR and accumulate MSB-first
// into a 64-bit integer. The top bit (bit 63) remains 0, matching the
// one-bit left padding before packing to uint64.
__hostdev__ inline uint64_t
gray_stream_to_uint64(uint32_t g0, uint32_t g1, uint32_t g2) {
    uint64_t value  = 0ull;
    uint32_t prefix = 0u; // running XOR accumulator (0/1)

    // `bit` index 0..20 corresponds to integer bit positions
    // p = 20..0 respectively.
    for (uint32_t bit = 0; bit < _HILBERT_NBITS; ++bit) {
        uint32_t const p = _HILBERT_NBITS - 1u - bit;

        // dim 0
        prefix ^= (g0 >> p) & 1u;
        value = (value << 1) | static_cast<uint64_t>(prefix);

        // dim 1
        prefix ^= (g1 >> p) & 1u;
        value = (value << 1) | static_cast<uint64_t>(prefix);

        // dim 2
        prefix ^= (g2 >> p) & 1u;
        value = (value << 1) | static_cast<uint64_t>(prefix);
    }

    // value is a 63-bit number; bit 63 is 0
    return value;
}

/// @brief Compute Hilbert curve index for 3D coordinates
/// @param i I coordinate (must be 0 <= i < 2^21)
/// @param j J coordinate (must be 0 <= j < 2^21)
/// @param k K coordinate (must be 0 <= k < 2^21)
/// @return 63-bit Hilbert index
__hostdev__ inline uint64_t
hilbert(uint32_t i, uint32_t j, uint32_t k) {
    // encoder: (i, j, k) -> uint64_t code.

    i = i & _HILBERT_MASK;
    j = j & _HILBERT_MASK;
    k = k & _HILBERT_MASK;

    // Iterates "bit = 0..20" (MSB to LSB), and for each bit runs
    // dim = 0..2 with the invert-or-swap rules on the suffix.
    for (uint32_t bit = 0; bit < _HILBERT_NBITS; ++bit) {
        uint32_t const p = _HILBERT_NBITS - 1u - bit; // integer bit position for this "bit"
        hilbert_step_for_bit(p, i, j, k);
    }

    // swapaxes+reshape to a 63-bit Gray stream, Gray to Binary (prefix XOR),
    // then pack to uint64 with a leading zero bit.
    return gray_stream_to_uint64(i, j, k);
}

} // namespace utils
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_HILBERTCODE_H
