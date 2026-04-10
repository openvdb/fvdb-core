// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_OPTYPE_CUH
#define FVDB_DETAIL_UTILS_CUDA_OPTYPE_CUH

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cuda_bf16.hpp>
#include <cuda_fp16.hpp>

namespace fvdb {
namespace detail {

/// Scalar type promotion trait: upcasts half-precision types to float for
/// numerically stable computation while leaving other types unchanged.
template <typename T> struct OpType {
    using type = T;
};

template <> struct OpType<__nv_bfloat16> {
    using type = float;
};

template <> struct OpType<__half> {
    using type = float;
};

template <> struct OpType<c10::Half> {
    using type = float;
};

template <> struct OpType<c10::BFloat16> {
    using type = float;
};

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_OPTYPE_CUH
