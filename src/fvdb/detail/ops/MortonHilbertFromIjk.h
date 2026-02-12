// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H
#define FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Compute Morton (Z-order) codes from ijk coordinates.
/// @param ijk  [N, 3] int32 tensor of coordinates.
/// @return     [N] int64 tensor of Morton codes.
torch::Tensor mortonFromIjk(torch::Tensor ijk);

/// @brief Compute Hilbert codes from ijk coordinates.
/// @param ijk  [N, 3] int32 tensor of coordinates.
/// @return     [N] int64 tensor of Hilbert codes.
torch::Tensor hilbertFromIjk(torch::Tensor ijk);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H
