// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDDENSEGRID_H
#define FVDB_DETAIL_OPS_BUILDDENSEGRID_H

#include <fvdb/detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchImpl>
createNanoGridFromDense(int64_t batchSize,
                        nanovdb::Coord origin,
                        nanovdb::Coord size,
                        torch::Device device,
                        const std::optional<torch::Tensor> &maybeMask,
                        const std::vector<nanovdb::Vec3d> &voxelSizes,
                        const std::vector<nanovdb::Vec3d> &origins);
} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDDENSEGRID_H
