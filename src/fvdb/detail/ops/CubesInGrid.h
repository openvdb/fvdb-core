// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CUBESINGRID_H
#define FVDB_DETAIL_OPS_CUBESINGRID_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor cubesInGrid(const GridBatchData &batchHdl,
                         const JaggedTensor &cubeCenters,
                         const nanovdb::Vec3d &padMin,
                         const nanovdb::Vec3d &padMax);

JaggedTensor cubesIntersectGrid(const GridBatchData &batchHdl,
                                const JaggedTensor &cubeCenters,
                                const nanovdb::Vec3d &padMin,
                                const nanovdb::Vec3d &padMax);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CUBESINGRID_H
