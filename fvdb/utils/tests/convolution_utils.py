# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
The goal of this module is to provide a set of baseline references for 3d convolution performed
by other libraries, but with the same frontend and outputs as fvdb, at least in terms of tensor
dimension meaning and order.

We want tests to compare apples-to-apples, rather than having the tests be performing lots of
permute, unsqueeze, and other similar things.

fVDB uses the following order for tensors in convolution:

[BATCH, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2, FEATURES]

SPATIAL_AXIS_0 is the major axis (slowest-changing spatial coord in contiguous tensor layout)
SPATIAL_AXIS_2 is the minor axis (fastest-changing spatial coord in contiguous tensor layout)

in fVDB voxel coordinates, x is the major axis, z is the minor axis.

It is important that when spatial axes are referred to, we avoid calling them
"width", "height", or "depth", and we ignore the application of those terms in the torch
documentation. Because the spatial axes don't always have the same physical meaning, for example
for Z-up interpretations of x, y, z, the concept of the "height" of the volume would be ambiguous.

When we interact with torch's convolution, we'll swap the order of the channels and the spatial
axes, but we'll otherwise keep the spatial axes in the same order as fVDB, so it would be:

[BATCH, FEATURES, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2]

That way, spatial function arguments like kernel_size, stride, bias - don't need to be reversed.
"""

import torch
import torch.nn.functional as tF
from fvdb.types import (
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    to_Vec3i,
    to_Vec3iBatch,
)
