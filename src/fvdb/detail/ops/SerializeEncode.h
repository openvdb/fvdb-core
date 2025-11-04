// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SERIALIZEENCODE_H
#define FVDB_DETAIL_OPS_SERIALIZEENCODE_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Get the space-filling curve codes for active voxels in a batch of grids
/// @tparam DeviceTag Which device to run on
/// @param gridBatch The batch of grids to get the space-filling curve codes for
/// @param order_type The type of space-filling curve to use for encoding
/// @param offset Offset to apply to voxel coordinates before encoding
/// @return A JaggedTensor of shape [B, -1, 1] of space-filling curve codes for active voxels
template <torch::DeviceType>
JaggedTensor dispatchSerializeEncode(GridBatchImpl const &gridBatch,
                                     SpaceFillingCurveType order_type,
                                     nanovdb::Coord const &offset);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SERIALIZEENCODE_H
