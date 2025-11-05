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

from fvdb import GridBatch, JaggedTensor


def conv_ground_truth_stride_1(
    grid_batch: GridBatch,
    activation: JaggedTensor,
    weights: torch.Tensor,
    *,
    dense_dims: NumericMaxRank1 | None = None,
    ijk_min: NumericMaxRank1 | None = None,
    allow_tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ground truth 3D convolution (with stride 1) over a GridBatch using PyTorch.

    This function first densifies the sparse input activation to a dense tensor in
    channel-major ("C-major") order as required by PyTorch's `conv3d`. The dense region
    is determined by the optional `dense_dims`/`ijk_min` arguments or, if not provided,
    by the total bounding box of the grid batch.

    The function then performs a 3D convolution using `torch.nn.functional.conv3d`
    with "same" padding (which is supported only for stride 1 in PyTorch). The resulting
    dense tensor is mapped back into a sparse JaggedTensor, matching the original sparse layout.

    Args:
        grid_batch (GridBatch): The input spatial grid batch over which to convolve.
        activation (JaggedTensor): Voxel features or activations over the grid (sparse).
            Shape: (batch_size, total_voxels, channels)
        weights (torch.Tensor): Convolution kernel weights in
            PyTorch conv3d format. Shape:
            (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
        dense_dims (NumericMaxRank1 | None, optional): The spatial dimensions
            of the dense tensor region to extract. If None, uses the bounding box of `grid_batch`.
        ijk_min (NumericMaxRank1 | None, optional): The minimum IJK coordinate
            (origin) for the dense region. If None, uses the bbox origin of `grid_batch`.
        allow_tf32 (bool, optional): If True, enables TF32 on supported hardware for
            faster computation. Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - dense_activation (torch.Tensor): The densified input features in C-major order.
                Shape: (batch_size, in_channels, dim0, dim1, dim2)
            - convolved (torch.Tensor): The dense convolved features (same shape as dense_activation).
    """
    bbox = grid_batch.total_bbox
    if ijk_min is None:
        ijk_min = torch.tensor(bbox[0], device="cpu")
    else:
        ijk_min = to_Vec3i(ijk_min)

    if dense_dims is None:
        dense_dims = 1 + (torch.tensor(bbox[1], device="cpu") - ijk_min)
    else:
        dense_dims = to_Vec3i(dense_dims, value_constraint=ValueConstraint.POSITIVE)

    dense_activation = grid_batch.inject_to_dense_cmajor(
        sparse_data=activation, min_coord=ijk_min, grid_size=dense_dims
    )

    _backend_setting = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    convolved = torch.nn.functional.conv3d(input=dense_activation, weight=weights, padding="same")
    torch.backends.cudnn.allow_tf32 = _backend_setting
    if dense_activation.shape != convolved.shape:
        raise ValueError(
            f"Dense activation shape {dense_activation.shape} does not match convolved shape {convolved.shape}"
        )

    return dense_activation, convolved
