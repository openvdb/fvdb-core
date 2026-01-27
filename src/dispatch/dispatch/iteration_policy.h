// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Iteration policies for N-dimensional traversal.
// Provides row_major/col_major policies and linear_to_nd index conversion.
// This header is torch-free and can be used with any tensor library.
//
#ifndef DISPATCH_DISPATCH_ITERATION_POLICY_H
#define DISPATCH_DISPATCH_ITERATION_POLICY_H

#include "dispatch/macros.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace dispatch {

//------------------------------------------------------------------------------
// Iteration policies
//------------------------------------------------------------------------------
// row_major: Last index varies fastest (C-style, layout_right)
//   For shape [D0, D1, D2], iteration order is:
//   (0,0,0), (0,0,1), (0,0,2), ..., (0,1,0), (0,1,1), ...
//
// col_major: First index varies fastest (Fortran-style, layout_left)
//   For shape [D0, D1, D2], iteration order is:
//   (0,0,0), (1,0,0), (2,0,0), ..., (0,1,0), (1,1,0), ...

struct row_major {};
struct col_major {};

//------------------------------------------------------------------------------
// linear_to_nd: Convert linear index to N-dimensional indices
//------------------------------------------------------------------------------
// Given a linear index and shape, compute the corresponding N-D indices
// according to the iteration policy.
//
// Template parameters:
//   Rank   - Number of dimensions
//   Policy - Iteration policy (row_major or col_major)
//
// Parameters:
//   linear_idx  - Linear index in [0, product(shape))
//   shape       - Array of dimension sizes
//   out_indices - Output array for N-D indices

template <int64_t Rank, typename Policy>
__hostdev__ constexpr void
linear_to_nd(int64_t linear_idx, int64_t const *shape, int64_t *out_indices) {
    if constexpr (std::is_same_v<Policy, row_major>) {
        // Last index varies fastest
        // idx[Rank-1] = linear % shape[Rank-1]
        // idx[Rank-2] = (linear / shape[Rank-1]) % shape[Rank-2]
        // etc.
        for (int64_t d = Rank - 1; d >= 0; --d) {
            out_indices[d] = linear_idx % shape[d];
            linear_idx /= shape[d];
        }
    } else {
        // First index varies fastest
        // idx[0] = linear % shape[0]
        // idx[1] = (linear / shape[0]) % shape[1]
        // etc.
        for (int64_t d = 0; d < Rank; ++d) {
            out_indices[d] = linear_idx % shape[d];
            linear_idx /= shape[d];
        }
    }
}

// Overload taking std::array for convenience
template <int64_t Rank, typename Policy>
__hostdev__ constexpr void
linear_to_nd(int64_t linear_idx,
             std::array<int64_t, Rank> const &shape,
             std::array<int64_t, Rank> &out_indices) {
    linear_to_nd<Rank, Policy>(linear_idx, shape.data(), out_indices.data());
}

// Return-by-value overload for convenience
template <int64_t Rank, typename Policy>
__hostdev__ constexpr std::array<int64_t, Rank>
linear_to_nd(int64_t linear_idx, std::array<int64_t, Rank> const &shape) {
    std::array<int64_t, Rank> indices;
    linear_to_nd<Rank, Policy>(linear_idx, shape.data(), indices.data());
    return indices;
}

//------------------------------------------------------------------------------
// nd_to_linear: Convert N-dimensional indices to linear index
//------------------------------------------------------------------------------
// Given N-D indices and shape, compute the corresponding linear index
// according to the iteration policy.

template <int64_t Rank, typename Policy>
__hostdev__ constexpr int64_t
nd_to_linear(int64_t const *indices, int64_t const *shape) {
    int64_t linear = 0;
    if constexpr (std::is_same_v<Policy, row_major>) {
        // linear = idx[0] * (shape[1] * shape[2] * ...) + idx[1] * (shape[2] * ...) + ... +
        // idx[N-1]
        for (int64_t d = 0; d < Rank; ++d) {
            linear *= shape[d];
            linear += indices[d];
        }
    } else {
        // linear = idx[N-1] * (shape[N-2] * ... * shape[0]) + ... + idx[0]
        for (int64_t d = Rank - 1; d >= 0; --d) {
            linear *= shape[d];
            linear += indices[d];
        }
    }
    return linear;
}

// Overload taking std::array
template <int64_t Rank, typename Policy>
__hostdev__ constexpr int64_t
nd_to_linear(std::array<int64_t, Rank> const &indices, std::array<int64_t, Rank> const &shape) {
    return nd_to_linear<Rank, Policy>(indices.data(), shape.data());
}

//------------------------------------------------------------------------------
// shape_volume: Compute total number of elements in a shape
//------------------------------------------------------------------------------
// Named shape_volume to avoid conflict with dispatch::volume<T> struct in detail.h

template <int64_t Rank>
__hostdev__ constexpr int64_t
shape_volume(int64_t const *shape) {
    int64_t v = 1;
    for (int64_t d = 0; d < Rank; ++d) {
        v *= shape[d];
    }
    return v;
}

template <int64_t Rank>
__hostdev__ constexpr int64_t
shape_volume(std::array<int64_t, Rank> const &shape) {
    return shape_volume<Rank>(shape.data());
}

} // namespace dispatch

#endif // DISPATCH_DISPATCH_ITERATION_POLICY_H
