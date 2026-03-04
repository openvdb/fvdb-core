// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/ActiveGridGoords.h>
#include <fvdb/detail/utils/SimpleOpHelper.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <torch::DeviceType DeviceTag>
struct Processor : public BasePerActiveVoxelProcessor<DeviceTag,
                                                      Processor<DeviceTag>,
                                                      FixedElementType<int32_t, 3>> {
    // active coords get saved directly to the output tensor
    __hostdev__ void
    perActiveVoxel(nanovdb::Coord const &ijk, int64_t const feature_idx, auto out_accessor) const {
        auto const i = static_cast<int32_t>(ijk[0]);
        auto const j = static_cast<int32_t>(ijk[1]);
        auto const k = static_cast<int32_t>(ijk[2]);

        auto &&out = out_accessor[feature_idx];
        out[0]     = i;
        out[1]     = j;
        out[2]     = k;
    }
};

} // End anonymous namespace

JaggedTensor
activeGridCoords(GridBatchImpl const &gridBatch) {
    return FVDB_DISPATCH_KERNEL(gridBatch.device(),
                                [&]() { return Processor<DeviceTag>{}.execute(gridBatch); });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
