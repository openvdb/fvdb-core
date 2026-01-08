// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TORCHTAGS_H
#define FVDB_DETAIL_DISPATCH_TORCHTAGS_H

#include "fvdb/detail/dispatch/Tag.h"

#include <torch/types.h>

namespace fvdb {
namespace dispatch {

// Device tag types - each is a distinct type that carries its device enum value
using TorchDeviceCpuTag         = Tag<torch::kCPU>;
using TorchDeviceCudaTag        = Tag<torch::kCUDA>;
using TorchDevicePrivateUse1Tag = Tag<torch::kPrivateUse1>;

} // namespace dispatch
} // namespace fvdb
#endif // FVDB_DETAIL_DISPATCH_TORCHTAGS_H
