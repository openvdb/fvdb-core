// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H
#define FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType> torch::Tensor dispatchMortonFromIjk(torch::Tensor ijk);

torch::Tensor mortonFromIjk(torch::Tensor ijk);

template <torch::DeviceType> torch::Tensor dispatchHilbertFromIjk(torch::Tensor ijk);

torch::Tensor hilbertFromIjk(torch::Tensor ijk);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_MORTONHILBERTFROMIJK_H
