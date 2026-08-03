// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H
#define FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H

#include <fvdb/GridBatchData.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Allocation accounting for the most recent generated forward topology.
///
/// The values are produced from the same count/prefix/fill inputs used to
/// allocate staging, so the accounting cannot describe a different algorithm.
struct BuildGridForConvResourceStats {
    int64_t inputVoxelCount{0};
    int64_t kernelVolume{0};
    int64_t validEmissionCount{0};
    uint64_t countRequestedBytes{0};
    uint64_t prefixRequestedBytes{0};
    uint64_t emissionRequestedBytes{0};
    uint64_t peakRequestedBytes{0};
    bool usedDirectProjection{false};
};

BuildGridForConvResourceStats lastBuildGridForConvResourceStats();

c10::intrusive_ptr<GridBatchData> buildGridForConv(const GridBatchData &baseBatchHdl,
                                                   const nanovdb::Coord &kernelSize,
                                                   const nanovdb::Coord &stride);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_BUILDGRIDFORCONV_H
