// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONIMPLICITGEMMSORTED_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONIMPLICITGEMMSORTED_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
torch::Tensor dispatchSparseConvolutionImplicitGEMMSorted(torch::Tensor inFeat,
                                                          torch::Tensor kernel,
                                                          torch::Tensor outInMap,
                                                          torch::Tensor reducedMask,
                                                          torch::Tensor reorderLoc,
                                                          int numOutFeats,
                                                          int numOutChannels,
                                                          bool allow_tf32,
                                                          bool allow_fp16);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONIMPLICITGEMMSORTED_H
