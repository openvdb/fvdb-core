# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Utilities for testing 3D convolution operations.

This module provides:
1. Baseline references for 3D convolution using PyTorch dense operations
2. Helper functions for comparing sparse and dense convolution results
3. Utilities for coordinate manipulation and comparison
4. Standard test configuration (device/dtype combos, tolerances)

fVDB uses the following order for tensors in convolution:

[BATCH, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2, FEATURES]

SPATIAL_AXIS_0 is the major axis (slowest-changing spatial coord in contiguous tensor layout)
SPATIAL_AXIS_2 is the minor axis (fastest-changing spatial coord in contiguous tensor layout)

In fVDB voxel coordinates, x is the major axis, z is the minor axis.

It is important that when spatial axes are referred to, we avoid calling them
"width", "height", or "depth", and we ignore the application of those terms in the torch
documentation. Because the spatial axes don't always have the same physical meaning, for example
for Z-up interpretations of x, y, z, the concept of the "height" of the volume would be ambiguous.

When we interact with torch's convolution, we swap the order of the channels and the spatial
axes, but we otherwise keep the spatial axes in the same order as fVDB:

[BATCH, FEATURES, SPATIAL_AXIS_0, SPATIAL_AXIS_1, SPATIAL_AXIS_2]

That way, spatial function arguments like kernel_size, stride, bias don't need to be reversed.
"""

import math
from contextlib import contextmanager
from typing import Sequence

import torch
from fvdb.types import NumericMaxRank1, ValueConstraint, to_Vec3i

from fvdb import GridBatch, JaggedTensor

# =============================================================================
# Test Configuration Constants
# =============================================================================

# Device and dtype combinations for parameterized tests
ALL_DEVICE_DTYPE_COMBOS = [
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

# Reduced coverage for tests where device/dtype doesn't affect the property being tested
# (e.g., topology tests, coordinate mapping tests)
REDUCED_DEVICE_DTYPE_COMBOS = [
    ["cuda", torch.float32],
]


# =============================================================================
# Standard Tolerances
# =============================================================================
#
# These tolerances are calibrated for comparing sparse convolution results
# against dense PyTorch ground truth. The values account for:
#   - Floating-point accumulation order differences between sparse and dense
#   - Different internal algorithms (cuDNN vs custom kernels)
#
# TOLERANCE RATIONALE:
#   float64: Near machine precision. Both sparse and dense use the same
#            accumulation precision, so results should match very closely.
#
#   float32: Looser tolerances needed because:
#     - cuDNN may use different accumulation strategies
#     - Sparse ops may accumulate in different order than dense
#     - TF32 is disabled but internal algorithms still differ
#
#   Kernel gradients: Accumulate contributions from ALL output voxels,
#     leading to more floating-point error. For small gradient values,
#     absolute tolerance dominates; for larger values, relative tolerance
#     applies. The 5e-4 tolerances handle both cases reasonably.


def get_tolerances(
    dtype: torch.dtype,
    kernel_size: tuple[int, int, int] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Get standard (rtol, atol) tolerances for a given dtype.

    Args:
        dtype: Data type for the computation
        kernel_size: Optional kernel size tuple. If provided and the kernel volume
            exceeds 27 (3x3x3), kernel gradient tolerances are scaled up to account
            for increased floating-point accumulation error.

    Returns a dict with keys:
        'forward': tolerances for forward pass comparison
        'input_grad': tolerances for input gradient comparison
        'kernel_grad': tolerances for kernel gradient comparison

    These tolerances are validated to pass on both CPU and CUDA for
    typical convolution sizes (kernel 3-7, feature counts 1-8).
    """
    if dtype == torch.float64:
        return {
            "forward": (1e-10, 1e-12),
            "input_grad": (1e-10, 1e-12),
            "kernel_grad": (1e-10, 1e-12),
        }
    else:  # float32
        # Base tolerances for small kernels
        input_grad_tol = (1e-5, 1e-6)
        kernel_grad_tol = (5e-4, 5e-4)

        # Scale gradient tolerances for large kernels.
        # Both input and kernel gradients accumulate over the kernel volume
        # (input grad via atomic scatter-add, kernel grad over all outputs),
        # and CUDA atomic ordering is non-deterministic, so FP error grows
        # with kernel size.
        if kernel_size is not None:
            kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
            base_volume = 27  # 3x3x3 baseline
            if kernel_volume > base_volume:
                scale = math.sqrt(kernel_volume / base_volume)
                input_grad_tol = (1e-5 * scale, 1e-6 * scale)
                kernel_grad_tol = (5e-4 * scale, 5e-4 * scale)

        return {
            "forward": (1e-5, 1e-6),
            "input_grad": input_grad_tol,
            "kernel_grad": kernel_grad_tol,
        }


# =============================================================================
# TF32 Control
# =============================================================================


@contextmanager
def disable_tf32():
    """
    Context manager to temporarily disable TF32 for consistent precision.

    TF32 (TensorFloat-32) can cause numerical differences between CPU and CUDA.
    Use this when comparing results across devices or when exact precision matters.

    This disables TF32 for both:
    - cuDNN operations (conv3d, etc.) via cudnn.allow_tf32
    - cuBLAS operations (mm, matmul, etc.) via cuda.matmul.allow_tf32

    Example:
        with disable_tf32():
            output = torch.nn.functional.conv3d(input, weight, padding="same")
    """
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


# =============================================================================
# Coordinate Utilities
# =============================================================================


def sort_coords_by_ijk(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort coordinates by a unique encoding and return sorted coords and permutation.

    This is useful for comparing coordinate sets that may be in different orders
    (e.g., due to different tiling strategies in sparse representations).

    Args:
        coords: Tensor of shape (N, 3) containing ijk coordinates.
                Supports negative coordinates.

    Returns:
        Tuple of (sorted_coords, permutation_indices) where:
        - sorted_coords: The input coordinates sorted by (i, j, k) order
        - permutation_indices: Indices such that coords[permutation_indices] == sorted_coords
    """
    # Use large multipliers to ensure unique encoding even with negative coords
    encoding = (
        coords[:, 0].to(torch.int64) * 1000000000
        + coords[:, 1].to(torch.int64) * 1000000
        + coords[:, 2].to(torch.int64)
    )
    perm = torch.argsort(encoding)
    return coords[perm], perm


def assert_coords_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    msg: str = "",
) -> None:
    """
    Assert that two coordinate tensors contain the same coordinates (order-independent).

    Args:
        actual: Tensor of shape (N, 3) with actual coordinates
        expected: Tensor of shape (M, 3) with expected coordinates
        msg: Optional message to include in assertion errors

    Raises:
        AssertionError: If the coordinate sets don't match
    """
    assert len(actual) == len(
        expected
    ), f"Coordinate count mismatch: got {len(actual)}, expected {len(expected)}. {msg}"

    actual_sorted, _ = sort_coords_by_ijk(actual)
    expected_sorted, _ = sort_coords_by_ijk(expected)

    assert torch.equal(actual_sorted, expected_sorted), f"Coordinates do not match. {msg}"


def normalize_stride(stride: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize stride to a 3-tuple."""
    if isinstance(stride, int):
        return (stride, stride, stride)
    return tuple(stride)  # type: ignore


# =============================================================================
# Topology Ground Truth
# =============================================================================


def compute_conv_grid_topology_ground_truth(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute the expected output coordinates for conv_grid using dense convolution.

    This function computes which output coordinates would be non-zero after
    convolving sparse input with an all-ones kernel. This is useful for validating
    the topology (coordinate set) produced by sparse convolution operations.

    For stride=1:
        Uses dense convolution with 'same' padding and finds non-zero locations.

    For stride>1:
        Computes output coordinates analytically based on receptive field overlap.
        The output at coordinate o receives contributions from input coordinates
        in the range [o*stride - kernel_half, o*stride + kernel_half].

    Args:
        input_coords: Tensor of shape (N, 3) with input ijk coordinates
        kernel_size: Tuple of kernel dimensions (k0, k1, k2)
        stride: Tuple of stride values (s0, s1, s2)
        device: Target device
        dtype: Data type for computation

    Returns:
        Tensor of shape (M, 3) with expected output ijk coordinates
    """
    kernel_half = tuple(k // 2 for k in kernel_size)

    # Compute coordinate ranges
    input_min = input_coords.min(dim=0).values.tolist()
    input_max = input_coords.max(dim=0).values.tolist()

    if stride == (1, 1, 1):
        # Stride 1: output extends by half-kernel beyond input
        output_min = tuple(input_min[i] - kernel_half[i] for i in range(3))
        output_max = tuple(input_max[i] + kernel_half[i] for i in range(3))
        coord_offset = output_min
        dense_shape = tuple(output_max[i] - output_min[i] + 1 for i in range(3))

        # Create dense input
        dense_input = torch.zeros((1, 1) + dense_shape, device=device, dtype=dtype)
        for coord in input_coords:
            idx = tuple(coord[i].item() - coord_offset[i] for i in range(3))
            dense_input[0, 0, idx[0], idx[1], idx[2]] = 1

        # All-ones kernel
        kernel = torch.ones((1, 1) + kernel_size, device=device, dtype=dtype)

        # Dense convolution with 'same' padding
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel, padding="same")

        # Find non-zero coordinates and convert back to grid coordinates
        nonzero_indices = torch.nonzero(dense_output[0, 0] != 0)
        offset_tensor = torch.tensor(coord_offset, device=device, dtype=torch.int32)
        expected_coords = nonzero_indices.to(torch.int32) + offset_tensor

    else:
        # For stride > 1, compute analytically based on receptive field overlap
        # For an input at coord c, it contributes to outputs at:
        #   o where o*stride - kernel_half <= c <= o*stride + kernel_half
        #   i.e., (c - kernel_half) / stride <= o <= (c + kernel_half) / stride
        # Using ceiling/floor for the bounds:
        #   o_min = ceil((c - kernel_half) / stride)
        #   o_max = floor((c + kernel_half) / stride)

        output_coords_set: set[tuple[int, int, int]] = set()

        for coord in input_coords:
            c = coord.tolist()

            # For each input coordinate, find all output coordinates it contributes to
            o_ranges = []
            for dim in range(3):
                c_val = c[dim]
                kh = kernel_half[dim]
                s = stride[dim]
                o_min = math.ceil((c_val - kh) / s)
                o_max = math.floor((c_val + kh) / s)
                o_ranges.append(range(o_min, o_max + 1))

            # Add all combinations
            for o0 in o_ranges[0]:
                for o1 in o_ranges[1]:
                    for o2 in o_ranges[2]:
                        output_coords_set.add((o0, o1, o2))

        expected_coords = torch.tensor(list(output_coords_set), device=device, dtype=torch.int32)

    return expected_coords


# =============================================================================
# Dense Convolution Ground Truth
# =============================================================================


def conv_ground_truth_strided(
    src_grid: GridBatch,
    dst_grid: GridBatch,
    activation: JaggedTensor,
    weights: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    allow_tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ground truth 3D convolution with arbitrary stride using dense PyTorch operations.

    This function:
    1. Densifies the sparse input activation based on the source grid coordinates
    2. Runs PyTorch's strided conv3d
    3. Returns the dense output and the values at the destination grid coordinates

    Args:
        src_grid: Source GridBatch containing input voxel coordinates
        dst_grid: Destination GridBatch containing expected output voxel coordinates
        activation: Sparse input features as JaggedTensor
        weights: Convolution kernel weights, shape (out_channels, in_channels, k0, k1, k2)
        stride: Stride tuple (s0, s1, s2)
        allow_tf32: If True, enables TF32 for faster but less precise computation

    Returns:
        Tuple of:
        - dense_output: Full dense output tensor from conv3d
        - sparse_output_values: Values at the destination grid coordinates,
          shape (num_dst_voxels, out_channels)
    """
    src_coords = src_grid.ijk.jdata
    dst_coords = dst_grid.ijk.jdata

    kernel_size = weights.shape[2:]
    kernel_half = tuple(k // 2 for k in kernel_size)

    device = activation.jdata.device
    dtype = activation.jdata.dtype
    in_channels = weights.shape[1]
    out_channels = weights.shape[0]

    # Compute the input coordinate ranges needed to cover all output coordinates
    # For output coord o, we need input coords in [o*stride - kernel_half, o*stride + kernel_half]
    dst_min = dst_coords.min(dim=0).values.tolist()
    dst_max = dst_coords.max(dim=0).values.tolist()

    # Input range needed for these outputs
    input_min_needed = tuple(dst_min[i] * stride[i] - kernel_half[i] for i in range(3))
    input_max_needed = tuple(dst_max[i] * stride[i] + kernel_half[i] for i in range(3))

    # Also include actual source coordinates in case they extend beyond
    src_min = src_coords.min(dim=0).values.tolist()
    src_max = src_coords.max(dim=0).values.tolist()

    dense_min_raw = tuple(min(input_min_needed[i], src_min[i]) for i in range(3))
    dense_max = tuple(max(input_max_needed[i], src_max[i]) for i in range(3))

    # CRITICAL: Align dense_min to the stride grid to ensure correct coordinate mapping.
    # The dense volume's origin must be at a coordinate that corresponds to output index 0
    # in the local coordinate system. This means dense_min must be aligned such that
    # local output index 0 corresponds to a known global output coordinate.
    #
    # With padding=kernel_half, local output[o] sees local input centered at o*stride.
    # For global output O to map to local output o, we need:
    #   O * stride = o * stride + dense_min
    #   O = o + dense_min / stride
    #
    # For this mapping to work with integer indices, dense_min must be divisible by stride.
    # We round DOWN to ensure we include all needed input coordinates.
    dense_min = tuple((dense_min_raw[i] // stride[i]) * stride[i] for i in range(3))
    dense_shape = tuple(dense_max[i] - dense_min[i] + 1 for i in range(3))

    # Create dense input
    dense_input = torch.zeros((1, in_channels) + dense_shape, device=device, dtype=dtype)
    for idx, coord in enumerate(src_coords):
        local_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
        dense_input[0, :, local_idx[0], local_idx[1], local_idx[2]] = activation.jdata[idx]

    # Compute padding to get outputs at the right coordinates
    # PyTorch strided conv: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
    # We want output coord o to correspond to input region starting at o*stride - kernel_half
    # Padding = kernel_half ensures the first output sees input starting at -kernel_half
    padding = kernel_half

    # Run convolution
    if allow_tf32:
        dense_output = torch.nn.functional.conv3d(input=dense_input, weight=weights, padding=padding, stride=stride)
    else:
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=weights, padding=padding, stride=stride)

    # Compute output coordinate offset
    # With padding=kernel_half and dense_min aligned to stride:
    # Local output[o] has center at local input o*stride, which is global input o*stride + dense_min
    # Global output O has center at global input O*stride
    # So: O*stride = o*stride + dense_min
    #     O = o + dense_min/stride
    # Since dense_min is now aligned, dense_min/stride is an integer.
    output_offset = tuple(dense_min[i] // stride[i] for i in range(3))

    # Extract values at destination coordinates
    sparse_output_values = torch.zeros((len(dst_coords), out_channels), device=device, dtype=dtype)
    for idx, coord in enumerate(dst_coords):
        out_idx = tuple(coord[i].item() - output_offset[i] for i in range(3))
        # Check bounds
        if all(0 <= out_idx[i] < dense_output.shape[i + 2] for i in range(3)):
            sparse_output_values[idx] = dense_output[0, :, out_idx[0], out_idx[1], out_idx[2]]

    return dense_output, sparse_output_values


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

    if allow_tf32:
        convolved = torch.nn.functional.conv3d(input=dense_activation, weight=weights, padding="same")
    else:
        with disable_tf32():
            convolved = torch.nn.functional.conv3d(input=dense_activation, weight=weights, padding="same")

    if dense_activation.shape != convolved.shape:
        raise ValueError(
            f"Dense activation shape {dense_activation.shape} does not match convolved shape {convolved.shape}"
        )

    return dense_activation, convolved


# =============================================================================
# Shared Test Helpers
# =============================================================================


class DisableTF32Mixin:
    """Mixin that disables TF32 for the entire test so sparse and dense paths use identical precision."""

    def setUp(self):
        torch.random.manual_seed(2024)
        self._saved_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self._saved_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cudnn.allow_tf32 = self._saved_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = self._saved_matmul_tf32


def diagnose_tensor_mismatch(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> str:
    """
    Generate diagnostic info for tensor mismatch (for debugging test failures).

    Args:
        name: Description of what's being compared
        actual: Actual tensor from sparse convolution
        expected: Expected tensor from dense ground truth
        rtol: Relative tolerance used
        atol: Absolute tolerance used

    Returns:
        Formatted string with error statistics and top mismatches
    """
    lines = [f"\n{'='*60}", f"TENSOR MISMATCH: {name}", f"{'='*60}"]

    if actual.shape != expected.shape:
        lines.append(f"SHAPE MISMATCH: actual={actual.shape}, expected={expected.shape}")
        return "\n".join(lines)

    lines.append(f"Shape: {actual.shape}, dtype: {actual.dtype}")

    abs_diff = (actual - expected).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    lines.append(f"\nError Statistics:")
    lines.append(f"  Max absolute error:  {max_abs:.6e}")
    lines.append(f"  Mean absolute error: {mean_abs:.6e}")
    lines.append(f"  Tolerance: rtol={rtol}, atol={atol}")

    exceeds = abs_diff > (atol + rtol * expected.abs())
    num_exceed = exceeds.sum().item()
    total = actual.numel()
    pct = 100 * num_exceed / total

    lines.append(f"\nMismatched elements: {num_exceed}/{total} ({pct:.1f}%)")

    if num_exceed > 0:
        flat_abs_diff = abs_diff.flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        _, top_indices = flat_abs_diff.topk(min(5, int(num_exceed)))

        lines.append(f"\nTop mismatches:")
        for idx in top_indices:
            i: int = int(idx.item())
            a_val: float = flat_actual[i].item()
            e_val: float = flat_expected[i].item()
            diff: float = flat_abs_diff[i].item()
            lines.append(f"  idx={i}: actual={a_val:.6f}, expected={e_val:.6f}, diff={diff:.6e}")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def create_grid_from_coords(
    coords: torch.Tensor,
    device: torch.device,
) -> GridBatch:
    """Create a GridBatch from coordinate tensor."""
    ijks = JaggedTensor(coords.to(device=device, dtype=torch.int32))
    return GridBatch.from_ijk(ijks, device=device)


def get_cluster_near_origin(device: torch.device) -> torch.Tensor:
    """
    Get a sparse cluster of coordinates near the origin.

    This cluster includes coordinates at (0,0,0) which will produce negative
    output coordinates when convolved with stride=1.
    """
    return torch.tensor(
        [
            # Group 1: At/near origin - produces negative output coords with stride=1
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 2],
            # Group 2: Slightly separated - some kernel overlap with group 1
            [3, 4, 5],
            [4, 4, 5],
            [3, 5, 6],
            # Group 3: Further out - tests larger coordinate range
            [7, 8, 9],
        ],
        device=device,
        dtype=torch.int32,
    )


def get_cluster_edge_aligned(
    kernel_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get a sparse cluster positioned so outputs stay non-negative.

    The minimum coordinate is offset by (half_kernel + 1) to ensure
    all output coordinates are non-negative.
    """
    kernel_half = tuple(k // 2 for k in kernel_size)
    base = tuple(k + 1 for k in kernel_half)  # Extra margin

    return torch.tensor(
        [
            [base[0] + 0, base[1] + 0, base[2] + 0],
            [base[0] + 2, base[1] + 0, base[2] + 1],
            [base[0] + 1, base[1] + 2, base[2] + 0],
            [base[0] + 3, base[1] + 3, base[2] + 3],
            [base[0] + 5, base[1] + 4, base[2] + 5],
        ],
        device=device,
        dtype=torch.int32,
    )


# =============================================================================
# Transpose Convolution Ground Truth
# =============================================================================


def compute_conv_transpose_topology_ground_truth(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the expected output coordinates for transposed convolution.

    For transposed convolution, each input at coordinate c contributes to
    outputs at coordinates: c*S + (k - K//2) for each kernel position k.

    This means the output range for input c is:
        [c*S - K//2, c*S + K//2]

    Args:
        input_coords: Tensor of shape (N, 3) with input ijk coordinates
        kernel_size: Tuple of kernel dimensions (k0, k1, k2)
        stride: Tuple of stride values (s0, s1, s2)
        device: Target device

    Returns:
        Tensor of shape (M, 3) with expected output ijk coordinates
    """
    kernel_half = tuple(k // 2 for k in kernel_size)

    output_coords_set: set[tuple[int, int, int]] = set()

    for coord in input_coords:
        c = coord.tolist()

        # For transposed convolution, input at c produces outputs at
        # c*S + (k - K//2) for each kernel position k
        for k0 in range(kernel_size[0]):
            for k1 in range(kernel_size[1]):
                for k2 in range(kernel_size[2]):
                    out_coord = (
                        c[0] * stride[0] + (k0 - kernel_half[0]),
                        c[1] * stride[1] + (k1 - kernel_half[1]),
                        c[2] * stride[2] + (k2 - kernel_half[2]),
                    )
                    output_coords_set.add(out_coord)

    return torch.tensor(list(output_coords_set), device=device, dtype=torch.int32)


def conv_transpose_ground_truth_stride_1(
    grid_batch: GridBatch,
    activation: JaggedTensor,
    weights: torch.Tensor,
    dense_dims: tuple[int, int, int],
    ijk_min: tuple[int, int, int],
    *,
    allow_tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ground truth transposed convolution (stride=1) using dense PyTorch.

    This function densifies the sparse input, runs PyTorch's conv_transpose3d,
    and returns both the dense activation and the convolved result.

    For transposed convolution with stride=1 and padding=K//2, the output
    has the same shape as the input (same as conv3d with padding="same").

    Args:
        grid_batch: Input GridBatch containing voxel coordinates
        activation: Sparse input features as JaggedTensor
        weights: Convolution kernel, shape (out_channels, in_channels, k0, k1, k2)
        dense_dims: Shape of the dense volume (d0, d1, d2)
        ijk_min: Minimum coordinate (origin) of the dense volume
        allow_tf32: If True, enables TF32 for faster but less precise computation

    Returns:
        Tuple of:
        - dense_activation: Densified input in channel-major format
        - convolved: Dense output from conv_transpose3d
    """
    device = activation.jdata.device
    dtype = activation.jdata.dtype
    kernel_size = weights.shape[2:]
    kernel_half = tuple(k // 2 for k in kernel_size)
    in_channels = weights.shape[1]

    # Create dense input
    dense_activation = torch.zeros((1, in_channels) + dense_dims, device=device, dtype=dtype)

    src_coords = grid_batch.ijk.jdata
    for idx, coord in enumerate(src_coords):
        local_idx = tuple(int(coord[i].item()) - ijk_min[i] for i in range(3))
        if all(0 <= local_idx[i] < dense_dims[i] for i in range(3)):
            dense_activation[0, :, local_idx[0], local_idx[1], local_idx[2]] = activation.jdata[idx]

    # Run transposed convolution.
    # PyTorch conv_transpose3d expects weight [in_channels, out_channels, K],
    # but our convention is [out_channels, in_channels, K].  Transpose channel dims.
    weights_torch = weights.transpose(0, 1).contiguous()

    # For stride=1, padding=K//2 gives same output size as input
    if allow_tf32:
        convolved = torch.nn.functional.conv_transpose3d(
            input=dense_activation,
            weight=weights_torch,
            padding=kernel_half,
            stride=1,
        )
    else:
        with disable_tf32():
            convolved = torch.nn.functional.conv_transpose3d(
                input=dense_activation,
                weight=weights_torch,
                padding=kernel_half,
                stride=1,
            )

    return dense_activation, convolved


def conv_transpose_ground_truth_strided(
    src_grid: GridBatch,
    dst_grid: GridBatch,
    activation: JaggedTensor,
    weights: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    allow_tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ground truth transposed 3D convolution with arbitrary stride using dense PyTorch.

    This function:
    1. Densifies the sparse input activation based on the source grid coordinates
    2. Runs PyTorch's strided conv_transpose3d
    3. Returns the dense output and the values at the destination grid coordinates

    PyTorch's ``conv_transpose3d`` expects weight shape ``[in_channels, out_channels, K]``,
    but our sparse convention is ``[out_channels, in_channels, K]``, so the weight is
    transposed internally before calling into PyTorch.

    Args:
        src_grid: Source GridBatch containing input voxel coordinates
        dst_grid: Destination GridBatch containing expected output voxel coordinates
        activation: Sparse input features as JaggedTensor
        weights: Convolution kernel weights, shape (out_channels, in_channels, k0, k1, k2)
        stride: Stride tuple (s0, s1, s2)
        allow_tf32: If True, enables TF32 for faster but less precise computation

    Returns:
        Tuple of:
        - dense_output: Full dense output tensor from conv_transpose3d
        - sparse_output_values: Values at the destination grid coordinates,
          shape (num_dst_voxels, out_channels)
    """
    src_coords = src_grid.ijk.jdata
    dst_coords = dst_grid.ijk.jdata

    kernel_size = weights.shape[2:]
    kernel_half = tuple(k // 2 for k in kernel_size)

    device = activation.jdata.device
    dtype = activation.jdata.dtype
    in_channels = weights.shape[1]
    out_channels = weights.shape[0]

    src_min = src_coords.min(dim=0).values.tolist()
    src_max = src_coords.max(dim=0).values.tolist()
    dense_shape = tuple(src_max[i] - src_min[i] + 1 for i in range(3))

    dense_input = torch.zeros((1, in_channels) + dense_shape, device=device, dtype=dtype)
    for idx, coord in enumerate(src_coords):
        local_idx = tuple(coord[i].item() - src_min[i] for i in range(3))
        dense_input[0, :, local_idx[0], local_idx[1], local_idx[2]] = activation.jdata[idx]

    weights_torch = weights.transpose(0, 1).contiguous()

    if allow_tf32:
        dense_output = torch.nn.functional.conv_transpose3d(
            input=dense_input, weight=weights_torch, padding=0, stride=stride
        )
    else:
        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input, weight=weights_torch, padding=0, stride=stride
            )

    # With padding=0 the full unpadded output is produced.  Local output o
    # for input c and kernel position k is o = c*S + k.  In global coords
    # (k centered): global = (src_min + c)*S + (k - K//2) = src_min*S - K//2 + o.
    output_origin = tuple(src_min[i] * stride[i] - kernel_half[i] for i in range(3))

    sparse_output_values = torch.zeros((len(dst_coords), out_channels), device=device, dtype=dtype)
    for idx, coord in enumerate(dst_coords):
        out_idx = tuple(coord[i].item() - output_origin[i] for i in range(3))
        if all(0 <= out_idx[i] < dense_output.shape[i + 2] for i in range(3)):
            sparse_output_values[idx] = dense_output[0, :, out_idx[0], out_idx[1], out_idx[2]]

    return dense_output, sparse_output_values


__all__ = [
    # Test configuration
    "ALL_DEVICE_DTYPE_COMBOS",
    "REDUCED_DEVICE_DTYPE_COMBOS",
    "get_tolerances",
    # TF32 control
    "disable_tf32",
    "DisableTF32Mixin",
    # Coordinate utilities
    "sort_coords_by_ijk",
    "assert_coords_equal",
    "normalize_stride",
    # Shared test helpers
    "diagnose_tensor_mismatch",
    "create_grid_from_coords",
    "get_cluster_near_origin",
    "get_cluster_edge_aligned",
    # Ground truth computation
    "compute_conv_grid_topology_ground_truth",
    "compute_conv_transpose_topology_ground_truth",
    "conv_ground_truth_strided",
    "conv_ground_truth_stride_1",
    "conv_transpose_ground_truth_stride_1",
    "conv_transpose_ground_truth_strided",
]
