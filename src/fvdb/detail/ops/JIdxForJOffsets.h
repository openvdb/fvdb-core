// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JIDXFORJOFFSETS_H
#define FVDB_DETAIL_OPS_JIDXFORJOFFSETS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchJIdxForJOffsets(torch::Tensor joffsets, int64_t numElements);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JIDXFORJOFFSETS_H
