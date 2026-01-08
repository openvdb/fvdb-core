// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
#define FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H

#include "fvdb/detail/dispatch/ConcreteTensor.h"

#include <cstdint>

namespace fvdb {
namespace dispatch {

//-----------------------------------------------------------------------------------
// ConcreteTensor CPU overload
//-----------------------------------------------------------------------------------

template <typename ScalarT, size_t Rank, typename IndexT = int64_t>
auto
accessor(CpuTensor<ScalarT, Rank> ct) {
    return ct.tensor.template accessor<ScalarT, Rank, IndexT>();
}

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_CONCRETETENSORACCESSORHOST_H
