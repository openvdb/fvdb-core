// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/SerializeEncode.h>
#include <fvdb/detail/utils/HilbertCode.h>
#include <fvdb/detail/utils/MortonCode.h>
#include <fvdb/detail/utils/SimpleOpHelper.h>

#include <cuda_runtime.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <torch::DeviceType DeviceTag>
struct Processor : public BasePerActiveVoxelProcessor<DeviceTag,
                                                      Processor<DeviceTag>,
                                                      ScalarElementType<int64_t>> {
    nanovdb::Coord offset            = nanovdb::Coord{0, 0, 0};
    SpaceFillingCurveType order_type = SpaceFillingCurveType::ZOrder;

    // Per-voxel callback which computes the space-filling
    // curve code (Morton or Hilbert) for
    // each active voxel in a batch of grids
    __hostdev__ void
    perActiveVoxel(nanovdb::Coord const &ijk, int64_t const feature_idx, auto out_accessor) const {
        // Apply offset to coordinates
        auto const i = static_cast<uint32_t>(ijk[0] + offset[0]);
        auto const j = static_cast<uint32_t>(ijk[1] + offset[1]);
        auto const k = static_cast<uint32_t>(ijk[2] + offset[2]);

        // Compute Morton or Hilbert code with offset to ensure non-negative coordinates
        uint64_t space_filling_code;
        switch (order_type) {
        case SpaceFillingCurveType::ZOrder:            // Regular z-order: xyz
            space_filling_code = utils::morton(i, j, k);
            break;
        case SpaceFillingCurveType::ZOrderTransposed:  // Transposed z-order: zyx
            space_filling_code = utils::morton(k, j, i);
            break;
        case SpaceFillingCurveType::Hilbert:           // Regular Hilbert curve: xyz
            space_filling_code = utils::hilbert(i, j, k);
            break;
        case SpaceFillingCurveType::HilbertTransposed: // Transposed Hilbert curve: zyx
            space_filling_code = utils::hilbert(k, j, i);
            break;
        default:
            // Invalid order type - use assert for device code
            space_filling_code = 0;
            break;
        }

        out_accessor[feature_idx] = static_cast<int64_t>(space_filling_code);
    }
};

} // End anonymous namespace

template <torch::DeviceType DeviceTag>
JaggedTensor
dispatchSerializeEncode(GridBatchImpl const &gridBatch,
                        SpaceFillingCurveType order_type,
                        nanovdb::Coord const &offset) {
    Processor<DeviceTag> processor{.offset = offset, .order_type = order_type};
    return processor.execute(gridBatch);
}

template JaggedTensor dispatchSerializeEncode<torch::kCUDA>(GridBatchImpl const &,
                                                            SpaceFillingCurveType,
                                                            nanovdb::Coord const &);
template JaggedTensor dispatchSerializeEncode<torch::kCPU>(GridBatchImpl const &,
                                                           SpaceFillingCurveType,
                                                           nanovdb::Coord const &);
template JaggedTensor dispatchSerializeEncode<torch::kPrivateUse1>(GridBatchImpl const &,
                                                                   SpaceFillingCurveType,
                                                                   nanovdb::Coord const &);

} // namespace ops
} // namespace detail
} // namespace fvdb
