// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// TensorChecks.h — Precondition validation for tensors before view construction.
//
// The dispatch framework's views (tensor_in, tensor_out, flat_in, flat_out)
// encode scalar type, rank, and contiguity as compile-time template parameters.
// Their constructors use assert() for these, which is stripped in release.
//
// This header provides check functions that produce clear TORCH_CHECK errors
// at runtime, suitable for op entry points.  The template parameters mirror
// the view parameters so the checks are always in sync with what the view
// requires.
//
// NOTE: These utilities were generalized from the first few ops ported to the
// new dispatch pattern (ActiveGridGoords, MortonHilbertFromIjk).  As more ops
// are migrated, we expect to discover additional common precondition patterns
// and may need to extend or refine the API here.
//
#ifndef FVDB_DETAIL_DISPATCH_TENSOR_CHECKS_H
#define FVDB_DETAIL_DISPATCH_TENSOR_CHECKS_H

#include "dispatch/torch/types.h"

#include <torch/types.h>

#include <cstdint>
#include <string>

namespace fvdb {
namespace detail {
namespace dispatch {

// ---- Individual checks ----
// Composable building blocks.  Ops combine them as needed.

/// @brief Check that a tensor's scalar type matches the expected type.
template <torch::ScalarType ExpectedStype>
void
check_dtype(torch::Tensor const &t, std::string_view name) {
    TORCH_CHECK_VALUE(t.scalar_type() == ExpectedStype,
                      name,
                      " must have dtype ",
                      c10::toString(ExpectedStype),
                      " but got ",
                      c10::toString(t.scalar_type()));
}

/// @brief Check that a tensor has exactly the expected rank (number of dimensions).
inline void
check_rank(torch::Tensor const &t, int64_t expected_rank, std::string_view name) {
    TORCH_CHECK_VALUE(
        t.dim() == expected_rank, name, " must be ", expected_rank, "D but got ", t.dim(), "D");
}

/// @brief Check that a specific dimension has the expected size.
inline void
check_dim_size(torch::Tensor const &t, int64_t dim, int64_t expected_size, std::string_view name) {
    TORCH_CHECK_VALUE(t.size(dim) == expected_size,
                      name,
                      ".size(",
                      dim,
                      ") must be ",
                      expected_size,
                      " but got ",
                      t.size(dim));
}

/// @brief Check that a tensor is contiguous.
inline void
check_contiguous(torch::Tensor const &t, std::string_view name) {
    TORCH_CHECK_VALUE(t.is_contiguous(), name, " must be contiguous");
}

/// @brief Check that a tensor is non-empty (numel > 0).
inline void
check_non_empty(torch::Tensor const &t, std::string_view name) {
    TORCH_CHECK_VALUE(t.numel() > 0, name, " must be non-empty");
}

// ---- Composite check ----
// Validates all properties a dispatch view requires in a single call.
// May gain additional overloads (e.g. accepting expected sizes) as more
// ops are ported and common patterns emerge.

/// @brief Validate a tensor against the properties required by a dispatch view.
///
/// Checks scalar type, minimum rank, and (if contiguous) contiguity.
/// Template parameters mirror those of dispatch::tensor_in / tensor_out.
///
/// Example:
///   check_for_view<torch::kInt32, 2, contiguity::contiguous>(ijk, "ijk");
///
template <torch::ScalarType ExpectedStype,
          int64_t MinRank,
          ::dispatch::contiguity Contig = ::dispatch::contiguity::strided>
void
check_for_view(torch::Tensor const &t, std::string_view name) {
    check_dtype<ExpectedStype>(t, name);
    TORCH_CHECK_VALUE(t.dim() >= MinRank,
                      name,
                      " must have at least ",
                      MinRank,
                      " dimensions but got ",
                      t.dim(),
                      "D");
    if constexpr (Contig == ::dispatch::contiguity::contiguous) {
        check_contiguous(t, name);
    }
}

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TENSOR_CHECKS_H
