# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the fVDB sparse transposed convolution implementation.

These tests compare sparse transposed convolution topology and results against
dense PyTorch ground truth to verify correctness of:
  - Topology computation (output coordinates for transposed convolution)
  - Forward pass values
  - Backward pass gradients

=============================================================================
TOPOLOGY FOR TRANSPOSED CONVOLUTION
=============================================================================

For REGULAR convolution with stride S:
    - Output o exists if any input c has: o*S - K//2 <= c <= o*S + K//2
    - Each output gathers from multiple inputs
    - Output coordinates are "compressed" (divided by stride)

For TRANSPOSED convolution with stride S:
    - Input c contributes to outputs at: c*S + (k - K//2) for each kernel position k
    - Each input scatters to multiple outputs
    - Output coordinates are "expanded" (multiplied by stride)

CRITICAL INSIGHT:
    For stride=1, both topologies are IDENTICAL!
    - Regular conv: output at c + offset for each input c
    - Transpose conv: output at c + offset for each input c

    For stride > 1, they DIFFER:
    - Regular conv: outputs at {o : exists c with o*S in [c-K//2, c+K//2]}
    - Transpose conv: outputs at {c*S + offset : for each input c}

=============================================================================
VALUES FOR TRANSPOSED CONVOLUTION
=============================================================================

For REGULAR convolution:
    - Output at o = sum over kernel positions k of: input[o + k - K//2] * kernel[k]
    - Each output GATHERS from its neighborhood

For TRANSPOSED convolution:
    - Each input at c SCATTERS to outputs: output[c + k - K//2] += input[c] * kernel[k]
    - Equivalently: output at o = sum over k of: input[o - k + K//2] * kernel[k]
    - This is convolution with a FLIPPED kernel

The mathematical relationship: conv_transpose(x, W) = conv(x, flip(W))
where flip reverses all spatial dimensions of the kernel.

For stride=1, PyTorch's conv_transpose3d with padding=K//2 gives the same
output shape as conv3d with padding="same", making comparison straightforward.
"""

import math
import unittest

import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import (
    fourier_anti_symmetric_kernel,
    generate_hermit_impulses_dense,
    has_any_symmetry,
)
from fvdb.utils.tests.convolution_utils import (
    ALL_DEVICE_DTYPE_COMBOS,
    REDUCED_DEVICE_DTYPE_COMBOS,
    assert_coords_equal,
    disable_tf32,
    get_tolerances,
    sort_coords_by_ijk,
)
from parameterized import parameterized

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

# =============================================================================
# Test-Specific Helpers
# =============================================================================


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
    output coordinates when convolved.
    """
    return torch.tensor(
        [
            # Group 1: At/near origin - produces negative output coords
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

    # Run transposed convolution
    # For stride=1, padding=K//2 gives same output size as input
    if allow_tf32:
        convolved = torch.nn.functional.conv_transpose3d(
            input=dense_activation,
            weight=weights,
            padding=kernel_half,
            stride=1,
        )
    else:
        with disable_tf32():
            convolved = torch.nn.functional.conv_transpose3d(
                input=dense_activation,
                weight=weights,
                padding=kernel_half,
                stride=1,
            )

    return dense_activation, convolved


def compute_conv_topology_ground_truth(
    input_coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the expected output coordinates for regular convolution.

    For regular convolution, output at coordinate o receives contributions
    from inputs where: o*S - K//2 <= c <= o*S + K//2

    Equivalently, input at c contributes to outputs in range:
        [ceil((c - K//2) / S), floor((c + K//2) / S)]

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

        # For regular convolution, input at c contributes to outputs in range
        # [ceil((c - K//2) / S), floor((c + K//2) / S)] for each dimension
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

    return torch.tensor(list(output_coords_set), device=device, dtype=torch.int32)


# =============================================================================
# Test Class
# =============================================================================


class TestConvTransposeTopology(unittest.TestCase):
    """
    Test topology computation for transposed convolution.

    For stride=1, the topology of transposed convolution is IDENTICAL to
    regular convolution. This is because:
        - Regular conv: output at c + offset for each input c
        - Transpose conv: output at c*1 + offset = c + offset for each input c

    We verify this by:
    1. Computing ground truth topology for both regular and transposed conv
    2. Verifying they match for stride=1
    3. Verifying conv_grid produces this topology
    """

    KERNEL_SIZE = (3, 5, 7)

    def setUp(self):
        torch.random.manual_seed(2024)

    # =========================================================================
    # Stride=1 Topology Tests (conv and conv_transpose have same topology)
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_stride1_topology_equivalence(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that for stride=1, conv and conv_transpose produce identical topologies.

        This is a fundamental property: when stride=1, both operations produce
        outputs at coordinates c + offset for each input c.
        """
        device = resolve_device(device)
        stride = (1, 1, 1)

        cluster_coords = get_cluster_near_origin(device)

        # Compute ground truth for both
        conv_topology = compute_conv_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        conv_t_topology = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        # They should be identical for stride=1
        assert_coords_equal(
            conv_topology,
            conv_t_topology,
            msg="Conv and conv_transpose topologies should match for stride=1",
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_matches_transpose_topology_stride1(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that conv_grid produces the correct topology for stride=1.

        Since conv and conv_transpose have the same topology for stride=1,
        we can use conv_grid for transposed convolution topology computation.
        """
        device = resolve_device(device)
        stride = (1, 1, 1)

        cluster_coords = get_cluster_near_origin(device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        # Get topology from conv_grid
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Compute ground truth for transposed convolution
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        # Verify they match
        assert_coords_equal(
            dst_ijks,
            expected_coords,
            msg="conv_grid should produce correct topology for stride=1 transposed conv",
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_topology_stride1(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test topology for a single input voxel with stride=1.

        A single input at coordinate c should produce outputs at all coordinates
        in the range [c - K//2, c + K//2] for each dimension.
        """
        device = resolve_device(device)
        stride = (1, 1, 1)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Single coordinate
        single_coord = torch.tensor([[5, 6, 7]], device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(single_coord, device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Expected: kernel_size^3 outputs
        expected_count = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), expected_count)

        # Verify bounds
        c = single_coord[0].tolist()
        expected_min = tuple(c[i] - kernel_half[i] for i in range(3))
        expected_max = tuple(c[i] + kernel_half[i] for i in range(3))

        actual_min = tuple(dst_ijks[:, i].min().item() for i in range(3))
        actual_max = tuple(dst_ijks[:, i].max().item() for i in range(3))

        self.assertEqual(actual_min, expected_min)
        self.assertEqual(actual_max, expected_max)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_topology_with_negative_output_coords(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that topology correctly handles negative output coordinates.

        When input is near origin, transposed convolution produces negative
        output coordinates (e.g., input at (0,0,0) with K//2=1 produces
        outputs at (-1,-1,-1) through (1,1,1)).
        """
        device = resolve_device(device)
        stride = (1, 1, 1)

        cluster_coords = get_cluster_near_origin(device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify some negative coordinates exist
        has_negative = (dst_ijks < 0).any()
        self.assertTrue(has_negative, "Expected some negative output coordinates")

        # Verify against ground truth
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        assert_coords_equal(dst_ijks, expected_coords)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_topology_non_negative_outputs(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test topology with edge-aligned inputs that produce non-negative outputs.
        """
        device = resolve_device(device)
        stride = (1, 1, 1)

        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify all non-negative
        all_non_negative = (dst_ijks >= 0).all()
        self.assertTrue(all_non_negative, "Expected all non-negative output coordinates")

        # Verify against ground truth
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        assert_coords_equal(dst_ijks, expected_coords)

    # =========================================================================
    # Documentation Tests: Stride > 1 Topology Difference
    # =========================================================================

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_stride2_topology_difference_documented(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Document that for stride > 1, conv and conv_transpose have DIFFERENT topologies.

        This test demonstrates the theoretical difference, even though fVDB
        doesn't yet support stride > 1 for transposed convolution.

        Regular conv (downsampling):
            - Input at c contributes to outputs in [ceil((c-K//2)/S), floor((c+K//2)/S)]
            - Output coordinates are "compressed"

        Transposed conv (upsampling):
            - Input at c produces outputs at c*S + (k - K//2) for each k
            - Output coordinates are "expanded"

        Example with K=3, S=2, input at c=3:
            - Regular conv: output at floor((3+1)/2) = 2 and ceil((3-1)/2) = 1
            - Transpose conv: outputs at 3*2 + {-1,0,1} = {5,6,7}
        """
        device = resolve_device(device)
        stride = (2, 2, 2)
        kernel_size = (3, 3, 3)

        # Simple test case: single input at coordinate (3, 3, 3)
        single_coord = torch.tensor([[3, 3, 3]], device=device, dtype=torch.int32)

        # Compute both topologies
        conv_topology = compute_conv_topology_ground_truth(
            input_coords=single_coord,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
        )

        conv_t_topology = compute_conv_transpose_topology_ground_truth(
            input_coords=single_coord,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
        )

        # They should be DIFFERENT for stride > 1
        # Regular conv: c=3, K//2=1, S=2 -> outputs at [ceil((3-1)/2), floor((3+1)/2)] = [1, 2]
        # That's outputs at (1,1,1), (1,1,2), (1,2,1), etc. (2^3 = 8 combinations)
        # Transpose conv: c=3, S=2, K//2=1 -> outputs at 3*2 + {-1,0,1} = {5,6,7}
        # That's outputs at (5,5,5) through (7,7,7) (3^3 = 27 outputs)

        # Verify counts are different
        self.assertNotEqual(
            len(conv_topology),
            len(conv_t_topology),
            "Conv and conv_transpose should have different output counts for stride > 1",
        )

        # Verify specific expected counts
        # Regular conv with K=3, S=2: each input contributes to 2^3 = 8 output cells
        # (because floor((c+1)/2) - ceil((c-1)/2) + 1 = 2 for each dimension)
        self.assertEqual(len(conv_topology), 8)

        # Transpose conv with K=3: each input contributes to 3^3 = 27 output cells
        self.assertEqual(len(conv_t_topology), 27)

        # Verify the coordinate ranges are different
        conv_min = tuple(conv_topology[:, i].min().item() for i in range(3))
        conv_max = tuple(conv_topology[:, i].max().item() for i in range(3))
        conv_t_min = tuple(conv_t_topology[:, i].min().item() for i in range(3))
        conv_t_max = tuple(conv_t_topology[:, i].max().item() for i in range(3))

        # Regular conv outputs near c/S = 3/2 ~ 1-2
        self.assertEqual(conv_min, (1, 1, 1))
        self.assertEqual(conv_max, (2, 2, 2))

        # Transpose conv outputs near c*S = 3*2 = 6, +/- K//2 = 1
        self.assertEqual(conv_t_min, (5, 5, 5))
        self.assertEqual(conv_t_max, (7, 7, 7))


class TestConvTransposeGridTopology(unittest.TestCase):
    """
    Test the conv_transpose_grid function topology computation.

    These tests verify that conv_transpose_grid produces the correct output
    coordinates for transposed convolution operations.
    """

    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_transpose_grid_single_impulse_bounds(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Validate that conv_transpose_grid output bounds match expected kernel footprint
        for a single input coordinate.

        For stride=1, a single input at coord c should produce outputs spanning
        [c - K//2, c + K//2] in each dimension.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)

        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Check count - should be kernel_size^3 outputs
        kernel_volume = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), kernel_volume)

        # Check bounds
        expected_start = tuple(self.SINGLE_COORD[i] - kernel_half[i] for i in range(3))
        expected_end = tuple(self.SINGLE_COORD[i] + kernel_half[i] + 1 for i in range(3))

        actual_start = tuple(dst_ijks[:, dim].min().item() for dim in range(3))
        actual_end = tuple(dst_ijks[:, dim].max().item() + 1 for dim in range(3))

        self.assertEqual(actual_start, expected_start)
        self.assertEqual(actual_end, expected_end)

    def _test_conv_transpose_grid_topology(
        self,
        device: DeviceIdentifier,
        dtype: torch.dtype,
        stride: tuple[int, int, int],
        cluster_coords: torch.Tensor,
        check_negative_outputs: bool = False,
        check_non_negative_outputs: bool = False,
    ):
        """
        Core topology test: verify conv_transpose_grid output matches ground truth.

        Args:
            device: Device identifier
            dtype: Data type
            stride: Stride tuple (s0, s1, s2)
            cluster_coords: Input coordinates
            check_negative_outputs: Assert that some outputs have negative coords
            check_non_negative_outputs: Assert that all outputs are non-negative
        """
        device = resolve_device(device)
        cluster_coords = cluster_coords.to(device=device)

        # Create grid and get conv_transpose_grid output
        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Compute ground truth
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )

        # Compare
        assert_coords_equal(
            dst_ijks,
            expected_coords,
            msg=f"stride={stride}",
        )

        # Additional checks
        if check_negative_outputs:
            has_negative = (dst_ijks < 0).any()
            self.assertTrue(has_negative, "Expected some negative output coordinates")

        if check_non_negative_outputs:
            all_non_negative = (dst_ijks >= 0).all()
            self.assertTrue(all_non_negative, "Expected all non-negative output coordinates")

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_transpose_grid_topology_stride1_near_origin(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with coordinates near origin, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_near_origin(device_resolved)
        self._test_conv_transpose_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_negative_outputs=True,
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_transpose_grid_topology_stride1_edge_aligned(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with edge-aligned coordinates, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_transpose_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_non_negative_outputs=True,
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_transpose_grid_matches_conv_grid_stride1(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that conv_transpose_grid produces the same topology as conv_grid for stride=1.

        This is a fundamental property: for stride=1, both operations produce the same
        output coordinate set (though with different values).
        """
        device = resolve_device(device)
        cluster_coords = get_cluster_near_origin(device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        conv_grid = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        conv_transpose_grid = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)

        assert_coords_equal(
            conv_grid.ijk.jdata,
            conv_transpose_grid.ijk.jdata,
            msg="conv_grid and conv_transpose_grid should match for stride=1",
        )


class TestConvTransposeValues(unittest.TestCase):
    """
    Test transposed convolution forward pass values.

    These tests verify that sparse transposed convolution produces the same
    results as dense PyTorch conv_transpose3d.
    """

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that each kernel weight position activates the correct output coordinate
        for transposed convolution.

        For transposed convolution with a single input impulse, each kernel position k
        produces an output at coordinate: input_coord + (k - K//2)

        This is the OPPOSITE direction from regular convolution, where kernel position k
        produces output at: input_coord - (k - K//2)
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))

        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )

        for k0 in range(self.KERNEL_SIZE[0]):
            for k1 in range(self.KERNEL_SIZE[1]):
                for k2 in range(self.KERNEL_SIZE[2]):
                    # Kernel with single impulse at (k0, k1, k2)
                    weights = torch.zeros((1, 1, *self.KERNEL_SIZE), device=device, dtype=dtype)
                    weights[0, 0, k0, k1, k2] = 1

                    output = conv_plan.execute(features, weights)
                    output_flat = output.jdata.flatten()

                    # Find the non-zero output location
                    nonzero_mask = output_flat != 0
                    self.assertEqual(nonzero_mask.sum().item(), 1)
                    got_coord = tuple(dst_ijks[nonzero_mask].flatten().tolist())

                    # Expected: for transposed conv, output at input_coord + (k - K//2)
                    # This is the OPPOSITE of regular conv which has - (k - K//2)
                    expected_coord = (
                        self.SINGLE_COORD[0] + (k0 - kernel_half[0]),
                        self.SINGLE_COORD[1] + (k1 - kernel_half[1]),
                        self.SINGLE_COORD[2] + (k2 - kernel_half[2]),
                    )
                    self.assertEqual(got_coord, expected_coord)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_forward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test forward transposed convolution of a single impulse against dense ground truth.

        Creates a single-voxel input, performs transposed convolution with an anti-symmetric
        kernel, and verifies the sparse output matches the dense conv_transpose3d result.
        """
        device = resolve_device(device)
        half_kernel = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Validate test configuration
        expected_volume = tuple(k + 2 for k in self.KERNEL_SIZE)
        expected_coord = tuple(1 + k for k in half_kernel)
        self.assertEqual(expected_volume, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_coord, self.SINGLE_COORD)

        # Create input
        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))

        # Anti-symmetric kernel (no symmetry that could hide bugs)
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        kernel_sum = kernel_5d.sum().item()

        # Dense ground truth using conv_transpose3d
        dense_input = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_input[0, 0, coord[0], coord[1], coord[2]] = 1

        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input, weight=kernel_5d, padding=half_kernel, stride=1
            )

        self.assertAlmostEqual(dense_output.sum().item(), kernel_sum, places=5)

        # Sparse transposed convolution
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )

        sparse_output = conv_plan.execute(features, kernel_5d)
        sparse_output_flat = sparse_output.jdata.flatten()

        # Verify sparse matches dense at output locations
        tols = get_tolerances(dtype)
        dense_at_dst = dense_output[0, 0, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]]
        torch.testing.assert_close(sparse_output_flat, dense_at_dst, rtol=tols["forward"][0], atol=tols["forward"][1])

        # Verify sum matches kernel sum
        self.assertAlmostEqual(sparse_output_flat.sum().item(), kernel_sum, places=5)

        # Verify utility function also produces correct ground truth
        gt_activation, gt_convolved = conv_transpose_ground_truth_stride_1(
            grid_batch=grid_batch,
            activation=features,
            weights=kernel_5d,
            dense_dims=self.SINGLE_VOLUME_SHAPE,
            ijk_min=(0, 0, 0),
            allow_tf32=False,
        )
        torch.testing.assert_close(gt_convolved, dense_output, rtol=tols["forward"][0], atol=tols["forward"][1])

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_multiple_impulses_forward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test forward transposed convolution of multiple isolated impulses.

        Uses hermit impulses (no kernel overlap) to ensure each impulse
        contributes independently to the output.
        """
        device = resolve_device(device)

        # Generate hermit impulses (no overlap)
        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )
        num_impulses = len(impulse_coords)

        # Anti-symmetric kernel
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        self.assertTrue(torch.all(kernel != 0))
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Dense ground truth using conv_transpose3d
        dense_input = impulse_field.reshape(1, 1, *self.VOLUME_SHAPE)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)
        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input, weight=kernel_5d, padding=kernel_half, stride=1
            )

        # Sparse transposed convolution
        grid_batch = create_grid_from_coords(impulse_coords, device)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify topology matches dense non-zeros
        dense_nonzero = torch.nonzero(dense_output[0, 0]).to(torch.int32)
        assert_coords_equal(dst_ijks, dense_nonzero)

        # Execute transposed convolution
        features = JaggedTensor(torch.ones((num_impulses, 1), device=device, dtype=dtype))
        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )

        sparse_output = conv_plan.execute(features, kernel_5d)
        sparse_flat = sparse_output.jdata.flatten()

        # Compare values (need to align ordering)
        dst_sorted, dst_perm = sort_coords_by_ijk(dst_ijks)
        dense_sorted, _ = sort_coords_by_ijk(dense_nonzero)

        dense_values_sorted = dense_output[0, 0, dense_sorted[:, 0], dense_sorted[:, 1], dense_sorted[:, 2]]
        sparse_values_sorted = sparse_flat[dst_perm]

        tols = get_tolerances(dtype)
        torch.testing.assert_close(
            sparse_values_sorted, dense_values_sorted, rtol=tols["forward"][0], atol=tols["forward"][1]
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_vs_conv_transpose_flipped_kernel(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that conv_transpose(x, W) == conv(x, flip(W)) for stride=1.

        This is a fundamental mathematical property of transposed convolution.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Create input
        cluster_coords = get_cluster_near_origin(device)
        grid_batch = create_grid_from_coords(cluster_coords, device)
        features_data = torch.randn((len(cluster_coords), 1), device=device, dtype=dtype)
        features = JaggedTensor(features_data)

        # Anti-symmetric kernel
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Flip the kernel (reverse all spatial dimensions)
        kernel_flipped = torch.flip(kernel_5d, dims=[2, 3, 4])

        # Get output grid (same for stride=1)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)

        # Transposed convolution with original kernel
        conv_t_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        conv_t_output = conv_t_plan.execute(features, kernel_5d)

        # Regular convolution with flipped kernel
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        conv_output = conv_plan.execute(features, kernel_flipped)

        # They should be equal
        tols = get_tolerances(dtype)
        torch.testing.assert_close(
            conv_t_output.jdata, conv_output.jdata, rtol=tols["forward"][0], atol=tols["forward"][1]
        )


class TestConvTransposeBackward(unittest.TestCase):
    """
    Test transposed convolution backward pass (gradient computation).

    These tests verify that gradients flow correctly through sparse transposed
    convolution, matching the dense PyTorch conv_transpose3d gradients.
    """

    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass of sparse transposed convolution against dense ground truth.

        Creates a single impulse, runs forward and backward passes on both
        sparse and dense implementations, and verifies gradients match.
        """
        device = resolve_device(device)
        half_kernel = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Validate config
        expected_coord = tuple(1 + k for k in half_kernel)
        self.assertEqual(expected_coord, self.SINGLE_COORD)

        # Create input with gradient tracking
        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)

        features_data = torch.ones((1, 1), device=device, dtype=dtype, requires_grad=True)
        features = JaggedTensor(features_data)

        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Kernel with gradient tracking
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone().requires_grad_(True)

        # === Dense path ===
        dense_input = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_input[0, 0, coord[0], coord[1], coord[2]] = 1
        dense_input = dense_input.clone().requires_grad_(True)
        dense_kernel = kernel_5d.detach().clone().requires_grad_(True)

        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input, weight=dense_kernel, padding=half_kernel, stride=1
            )

        # === Sparse path ===
        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        sparse_output = conv_plan.execute(features, kernel_5d)

        # Verify forward match
        tols = get_tolerances(dtype)
        dense_at_dst = dense_output[0, 0, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]]
        torch.testing.assert_close(
            sparse_output.jdata.flatten(), dense_at_dst, rtol=tols["forward"][0], atol=tols["forward"][1]
        )

        # === Backward ===
        # Create output gradient at center coordinate
        grad_coord = self.SINGLE_COORD

        # Dense gradient
        dense_grad = torch.zeros_like(dense_output)
        dense_grad[0, 0, grad_coord[0], grad_coord[1], grad_coord[2]] = 1
        dense_output.backward(dense_grad)

        # Sparse gradient
        grad_coord_tensor = torch.tensor(grad_coord, device=device, dtype=torch.int32)
        grad_idx = int(torch.nonzero((dst_ijks == grad_coord_tensor).all(dim=1)).squeeze().item())
        sparse_grad = torch.zeros_like(sparse_output.jdata)
        sparse_grad[grad_idx, 0] = 1
        sparse_output.jdata.backward(sparse_grad)

        # Compare input gradients
        dense_input_grad = dense_input.grad
        sparse_input_grad = features_data.grad
        assert dense_input_grad is not None and sparse_input_grad is not None

        dense_grad_at_coord = dense_input_grad[0, 0, coord[0], coord[1], coord[2]]
        torch.testing.assert_close(
            sparse_input_grad.flatten(),
            dense_grad_at_coord.unsqueeze(0),
            rtol=tols["input_grad"][0],
            atol=tols["input_grad"][1],
        )

        # Compare kernel gradients
        dense_kernel_grad = dense_kernel.grad
        sparse_kernel_grad = kernel_5d.grad
        assert dense_kernel_grad is not None and sparse_kernel_grad is not None

        torch.testing.assert_close(
            sparse_kernel_grad, dense_kernel_grad, rtol=tols["kernel_grad"][0], atol=tols["kernel_grad"][1]
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_multiple_inputs_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass with multiple input voxels and random gradients.

        This is a more thorough test that verifies gradients are computed correctly
        when multiple inputs contribute to the output.
        """
        device = resolve_device(device)
        tols = get_tolerances(dtype, kernel_size=self.KERNEL_SIZE)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Create sparse input with multiple voxels
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        num_voxels = len(cluster_coords)
        in_channels = 2
        out_channels = 3

        features_data = torch.randn((num_voxels, in_channels), device=device, dtype=dtype, requires_grad=True)
        features = JaggedTensor(features_data)

        # Create anti-symmetric kernel
        kernel_3d = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0).expand(out_channels, in_channels, -1, -1, -1)
        kernel_5d = kernel_5d.clone().requires_grad_(True)

        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # === Sparse forward and backward ===
        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        sparse_output = conv_plan.execute(features, kernel_5d)

        # Random output gradient for thorough testing
        output_grad = torch.randn_like(sparse_output.jdata)
        sparse_output.jdata.backward(output_grad)

        sparse_input_grad = features_data.grad
        sparse_kernel_grad = kernel_5d.grad

        # === Dense ground truth ===
        # Compute dense bounds
        src_min = cluster_coords.min(dim=0).values.tolist()
        src_max = cluster_coords.max(dim=0).values.tolist()
        dst_min = dst_ijks.min(dim=0).values.tolist()
        dst_max = dst_ijks.max(dim=0).values.tolist()

        # Dense volume needs to cover both input and output
        dense_min = tuple(min(src_min[i], dst_min[i]) for i in range(3))
        dense_max = tuple(max(src_max[i], dst_max[i]) for i in range(3))
        dense_shape = tuple(dense_max[i] - dense_min[i] + 1 for i in range(3))

        # Create dense input with gradient tracking
        dense_features = features_data.detach().clone().requires_grad_(True)
        dense_kernel = kernel_5d.detach().clone().requires_grad_(True)

        dense_input = torch.zeros((1, in_channels) + dense_shape, device=device, dtype=dtype)
        for idx, coord in enumerate(cluster_coords):
            local_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
            dense_input[0, :, local_idx[0], local_idx[1], local_idx[2]] = dense_features[idx]

        dense_input = dense_input.clone().requires_grad_(True)

        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input, weight=dense_kernel, padding=kernel_half, stride=1
            )

        # Apply gradient at output coordinates
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for idx, coord in enumerate(dst_ijks):
            out_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
            if all(0 <= out_idx[i] < dense_output.shape[i + 2] for i in range(3)):
                loss = loss + (dense_output[0, :, out_idx[0], out_idx[1], out_idx[2]] * output_grad[idx]).sum()

        loss.backward()

        # Extract dense input gradient at sparse locations
        dense_input_grad_at_sparse = torch.zeros_like(dense_features)
        assert dense_input.grad is not None, "Dense input grad is None"
        for idx, coord in enumerate(cluster_coords):
            local_idx = tuple(int(coord[i].item()) - dense_min[i] for i in range(3))
            dense_input_grad_at_sparse[idx] = dense_input.grad[0, :, local_idx[0], local_idx[1], local_idx[2]]

        # Compare input gradients
        assert sparse_input_grad is not None
        try:
            torch.testing.assert_close(
                sparse_input_grad, dense_input_grad_at_sparse, rtol=tols["input_grad"][0], atol=tols["input_grad"][1]
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Input gradient (dtype={dtype})",
                sparse_input_grad,
                dense_input_grad_at_sparse,
                rtol=tols["input_grad"][0],
                atol=tols["input_grad"][1],
            )
            raise AssertionError(diag) from None

        # Compare kernel gradients
        dense_kernel_grad = dense_kernel.grad
        assert sparse_kernel_grad is not None and dense_kernel_grad is not None

        try:
            torch.testing.assert_close(
                sparse_kernel_grad, dense_kernel_grad, rtol=tols["kernel_grad"][0], atol=tols["kernel_grad"][1]
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Kernel gradient (dtype={dtype})",
                sparse_kernel_grad,
                dense_kernel_grad,
                rtol=tols["kernel_grad"][0],
                atol=tols["kernel_grad"][1],
            )
            raise AssertionError(diag) from None


if __name__ == "__main__":
    unittest.main()
