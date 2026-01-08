// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CPU accessor
//-----------------------------------------------------------------------------------
// Extracts the C++ scalar type from the DtypeTag to call torch's accessor.
// Matches any ConcreteTensor with CPU device tag.

template <typename DtypeTag, size_t Rank>
auto
accessor(ConcreteTensor<TorchDeviceCpuTag, DtypeTag, Rank> ct) {
    using ScalarT = ScalarCppTypeT<DtypeTag>;
    // Note: Host TensorAccessor only takes <T, N>, no index type parameter
    // (unlike packed_accessor64/32 for CUDA)
    return ct.tensor.template accessor<ScalarT, Rank>();
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
