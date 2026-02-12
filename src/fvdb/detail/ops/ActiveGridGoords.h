// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H
#define FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {

class GridBatchImpl;

namespace ops {

/// @brief Return the ijk coordinates of every active voxel in the grid batch.
/// Device dispatch is handled internally — no template parameter needed.
JaggedTensor activeGridCoords(GridBatchImpl const &gridBatch);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_ACTIVEGRIDGOORDS_H
