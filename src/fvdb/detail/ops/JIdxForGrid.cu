// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/ops/JIdxForGrid.h>
#include <fvdb/detail/ops/JIdxForJOffsets.h>
#include <fvdb/detail/utils/Utils.h>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

torch::Tensor
jIdxForGrid(const GridBatchData &batchHdl) {
    if (batchHdl.batchSize() == 1 || batchHdl.totalVoxels() == 0) {
        return torch::empty(
            {0}, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device()));
    }
    return jIdxForJOffsets(batchHdl.voxelOffsets(), batchHdl.totalVoxels());
}

} // namespace ops
} // namespace detail
} // namespace fvdb
