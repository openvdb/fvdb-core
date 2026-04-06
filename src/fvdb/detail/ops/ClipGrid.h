// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CLIPGRID_H
#define FVDB_DETAIL_OPS_CLIPGRID_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchData.h>

#include <nanovdb/NanoVDB.h>

#include <tuple>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

c10::intrusive_ptr<GridBatchData> clipGrid(const GridBatchData &grid,
                                           const std::vector<nanovdb::Coord> &ijkMin,
                                           const std::vector<nanovdb::Coord> &ijkMax);

std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor>
clipGridWithMask(const GridBatchData &grid,
                 const std::vector<nanovdb::Coord> &ijkMin,
                 const std::vector<nanovdb::Coord> &ijkMax);

std::pair<JaggedTensor, c10::intrusive_ptr<GridBatchData>>
clipGridFeaturesWithMask(const GridBatchData &grid,
                         const JaggedTensor &features,
                         const std::vector<nanovdb::Coord> &ijkMin,
                         const std::vector<nanovdb::Coord> &ijkMax);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CLIPGRID_H
