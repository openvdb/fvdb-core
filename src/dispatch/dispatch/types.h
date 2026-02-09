// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility shim — bundles the core dispatch type headers.
//
// This includes only the core dispatch types (tags, axes, enums, labels).
// Torch-specific headers (torch/dispatch.h, torch/types.h, torch/for_each.h,
// torch/views.h) and the thread pool (thread_pool.h) are separate includes.
//
#ifndef DISPATCH_DISPATCH_TYPES_H
#define DISPATCH_DISPATCH_TYPES_H

#include "dispatch/axes.h"
#include "dispatch/axis.h"
#include "dispatch/consteval_types.h"
#include "dispatch/enums.h"
#include "dispatch/extents.h"
#include "dispatch/indices.h"
#include "dispatch/label.h"
#include "dispatch/tag.h"

#endif // DISPATCH_DISPATCH_TYPES_H
