// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONLGGS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONLGGS_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchSparseConvolutionLggs(
    const torch::Tensor &inFeatures,
    const torch::Tensor &kernel,
    const torch::Tensor &spokeIndicesFlattenedOffset,
    const torch::Tensor &spokeInputGlobalIndicesFlattenedData,
    const torch::Tensor &spokeOutputLocalOffsetsRelativeToBlockFlattenedData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONLGGS_H
