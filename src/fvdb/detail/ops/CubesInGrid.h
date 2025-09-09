// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CUBESINGRID_H
#define FVDB_DETAIL_OPS_CUBESINGRID_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
JaggedTensor dispatchCubesInGrid(const GridBatchImpl &batchHdl,
                                 const JaggedTensor &cubeCenters,
                                 const Vec3dOrScalar &padMin,
                                 const Vec3dOrScalar &padMax);

template <torch::DeviceType>
JaggedTensor dispatchCubesIntersectGrid(const GridBatchImpl &batchHdl,
                                        const JaggedTensor &cubeCenters,
                                        const Vec3dOrScalar &padMin,
                                        const Vec3dOrScalar &padMax);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CUBESINGRID_H
