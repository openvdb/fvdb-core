// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/ActiveGridGoords.h>
#include <fvdb/detail/utils/SimpleOpHelper.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <torch::DeviceType DeviceTag>
struct Processor : public BaseProcessor<DeviceTag, Processor<DeviceTag>, int32_t, 3> {
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

template <torch::DeviceType DeviceTag>
JaggedTensor
dispatchActiveGridCoords(GridBatchImpl const &gridBatch) {
    return Processor<DeviceTag>{}.execute(gridBatch);
}

template JaggedTensor dispatchActiveGridCoords<torch::kCUDA>(GridBatchImpl const &);
template JaggedTensor dispatchActiveGridCoords<torch::kCPU>(GridBatchImpl const &);
template JaggedTensor dispatchActiveGridCoords<torch::kPrivateUse1>(GridBatchImpl const &);

} // namespace ops
} // namespace detail
} // namespace fvdb
