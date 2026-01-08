// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CPU overload
//-----------------------------------------------------------------------------------

template <typename ScalarT, size_t Rank, typename UnusedIndexT = int64_t>
auto
accessor(CpuTensor<ScalarT, Rank> ct) {
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<ScalarT, Rank>();
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
