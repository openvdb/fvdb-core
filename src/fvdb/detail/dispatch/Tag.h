// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TAG_H
#define FVDB_DETAIL_DISPATCH_TAG_H

#include "fvdb/detail/dispatch/Values.h"

namespace fvdb {
namespace dispatch {

template <auto... values> using Tag = Values<values...>;

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TAG_H
