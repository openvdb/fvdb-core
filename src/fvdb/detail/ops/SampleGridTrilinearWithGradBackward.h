// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_SAMPLEGRIDTRILINEARWITHGRADBACKWARD_H
#define FVDB_DETAIL_OPS_SAMPLEGRIDTRILINEARWITHGRADBACKWARD_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchSampleGridTrilinearWithGradBackward(const GridBatchImpl &batchHdl,
                                                          const JaggedTensor &points,
                                                          const torch::Tensor &data,
                                                          const torch::Tensor &gradOutFeatures,
                                                          const torch::Tensor &gradOutGradFeatures);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_SAMPLEGRIDTRILINEARWITHGRADBACKWARD_H
