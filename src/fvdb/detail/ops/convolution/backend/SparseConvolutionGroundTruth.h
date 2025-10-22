// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONGROUNDTRUTH_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONGROUNDTRUTH_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
void
dispatchSparseConvolutionGroundTruth(torch::Tensor input_features,  // (B, src_active_voxels, inC)
                                     torch::Tensor output_features, // (B, dst_active_voxels, outC)
                                     torch::Tensor weights,     // (outC, inC, Axis0, Axis1, Axis2)
                                     const GridBatchImpl &src_grid,
                                     const GridBatchImpl &dst_grid,
                                     torch::Tensor kernel_size, // (3,) - [Axis0, Axis1, Axis2]
                                     torch::Tensor stride,      // (3,) - [Axis0, Axis1, Axis2]
                                     bool transpose);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_SPARSECONVOLUTIONGROUNDTRUTH_H
