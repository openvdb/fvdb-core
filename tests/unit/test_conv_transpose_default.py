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
    DisableTF32Mixin,
    assert_coords_equal,
    compute_conv_transpose_topology_ground_truth,
    conv_transpose_ground_truth_stride_1,
    conv_transpose_ground_truth_strided,
    create_grid_from_coords,
    diagnose_tensor_mismatch,
    disable_tf32,
    get_cluster_edge_aligned,
    get_cluster_near_origin,
    get_tolerances,
    sort_coords_by_ijk,
)
from parameterized import parameterized

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

# =============================================================================
# Test Classes
# =============================================================================


class TestConvTransposeTopology(DisableTF32Mixin, unittest.TestCase):
    """
    Test topology computation for transposed convolution.

    Verifies that conv_transpose_grid produces the correct output coordinate
    set, and that it matches conv_grid for stride=1 (a fundamental property).
    """

    KERNEL_SIZE = (3, 5, 7)
    SINGLE_COORD = (2, 3, 4)

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_bounds(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Validate that conv_transpose_grid output bounds match expected kernel footprint
        for a single input coordinate.

        For stride=1, a single input at coord c should produce outputs spanning
        [c - K//2, c + K//2] in each dimension, totalling prod(K) voxels.
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

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_topology_stride1_near_origin(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Topology test with coordinates near origin, stride=1.

        Verifies negative output coordinates are produced and that the
        coordinate set matches analytical ground truth.
        """
        device = resolve_device(device)
        cluster_coords = get_cluster_near_origin(device)

        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=(1, 1, 1))
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify some negative coordinates exist
        has_negative = (dst_ijks < 0).any()
        self.assertTrue(has_negative, "Expected some negative output coordinates")

        # Verify against ground truth
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=(1, 1, 1),
            device=device,
        )

        assert_coords_equal(dst_ijks, expected_coords)

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_topology_stride1_edge_aligned(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Topology test with edge-aligned coordinates that produce non-negative outputs.

        Verifies all output coordinates are non-negative and match analytical ground truth.
        """
        device = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device)

        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=(1, 1, 1))
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify all non-negative
        all_non_negative = (dst_ijks >= 0).all()
        self.assertTrue(all_non_negative, "Expected all non-negative output coordinates")

        # Verify against ground truth
        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=(1, 1, 1),
            device=device,
        )

        assert_coords_equal(dst_ijks, expected_coords)

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_topology_matches_conv_grid_stride1(self, device: DeviceIdentifier, dtype: torch.dtype):
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

    # --- Strided topology tests ---

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_topology_stride_uniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Transposed topology test with uniform stride (2,2,2)."""
        device = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        stride = (2, 2, 2)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )
        assert_coords_equal(dst_ijks, expected_coords, msg=f"stride={stride}")

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_topology_stride_nonuniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Transposed topology test with non-uniform stride (1,2,3)."""
        device = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device)
        grid_batch = create_grid_from_coords(cluster_coords, device)

        stride = (1, 2, 3)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        expected_coords = compute_conv_transpose_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
        )
        assert_coords_equal(dst_ijks, expected_coords, msg=f"stride={stride}")


class TestConvTransposeValues(DisableTF32Mixin, unittest.TestCase):
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

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
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

    # =========================================================================
    # Strided Transposed Convolution Tests (Forward + Backward)
    # =========================================================================

    def _test_strided_conv_transpose_forward_backward(
        self,
        device: DeviceIdentifier,
        dtype: torch.dtype,
        stride: tuple[int, int, int],
        cluster_coords: torch.Tensor,
        in_channels: int = 2,
        out_channels: int = 3,
    ):
        """
        Core test for strided transposed convolution forward and backward passes.

        Tests that:
        1. Forward pass matches dense PyTorch conv_transpose3d ground truth
        2. Backward pass produces correct input gradients
        3. Backward pass produces correct kernel gradients
        """
        tols = get_tolerances(dtype, kernel_size=self.KERNEL_SIZE)
        fwd_rtol, fwd_atol = tols["forward"]
        input_grad_rtol, input_grad_atol = tols["input_grad"]
        kernel_grad_rtol, kernel_grad_atol = tols["kernel_grad"]

        device = resolve_device(device)
        cluster_coords = cluster_coords.to(device=device)

        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_transpose_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_coords = dst_grid_batch.ijk.jdata

        num_voxels = len(cluster_coords)

        features_data = torch.randn((num_voxels, in_channels), device=device, dtype=dtype, requires_grad=True)
        features = JaggedTensor(features_data)

        kernel_3d = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel_3d))
        kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0).expand(out_channels, in_channels, -1, -1, -1)
        kernel_5d = kernel_5d.clone().requires_grad_(True)

        # === Sparse forward ===
        conv_plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        sparse_output = conv_plan.execute(features, kernel_5d)

        # === Dense ground truth forward ===
        dense_features = features_data.detach().clone().requires_grad_(True)
        dense_kernel = kernel_5d.detach().clone().requires_grad_(True)

        dense_output_full, dense_output_at_dst = conv_transpose_ground_truth_strided(
            src_grid=grid_batch,
            dst_grid=dst_grid_batch,
            activation=JaggedTensor(dense_features),
            weights=dense_kernel,
            stride=stride,
            allow_tf32=False,
        )

        # Compare forward values (sort by coordinate for stable comparison)
        sparse_values_sorted, sparse_perm = sort_coords_by_ijk(dst_coords)
        sparse_out_sorted = sparse_output.jdata[sparse_perm]
        dense_out_sorted = dense_output_at_dst[sparse_perm]

        try:
            torch.testing.assert_close(sparse_out_sorted, dense_out_sorted, rtol=fwd_rtol, atol=fwd_atol)
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Transpose forward (stride={stride}, dtype={dtype})",
                sparse_out_sorted,
                dense_out_sorted,
                rtol=fwd_rtol,
                atol=fwd_atol,
            )
            raise AssertionError(diag) from None

        # === Backward pass ===
        output_grad = torch.randn_like(sparse_output.jdata)

        sparse_output.jdata.backward(output_grad)
        sparse_input_grad = features_data.grad
        sparse_kernel_grad = kernel_5d.grad

        # Dense backward: re-run ground truth with grad tracking, apply same output grad
        dense_features2 = features_data.detach().clone().requires_grad_(True)
        dense_kernel2 = kernel_5d.detach().clone().requires_grad_(True)

        dense_output_full2, dense_output_at_dst2 = conv_transpose_ground_truth_strided(
            src_grid=grid_batch,
            dst_grid=dst_grid_batch,
            activation=JaggedTensor(dense_features2),
            weights=dense_kernel2,
            stride=stride,
            allow_tf32=False,
        )

        loss = (dense_output_at_dst2 * output_grad).sum()
        loss.backward()

        dense_input_grad = dense_features2.grad
        dense_kernel_grad = dense_kernel2.grad

        # Compare input gradients
        assert sparse_input_grad is not None, "Sparse input grad is None"
        assert dense_input_grad is not None, "Dense input grad is None"

        try:
            torch.testing.assert_close(
                sparse_input_grad, dense_input_grad, rtol=input_grad_rtol, atol=input_grad_atol
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Transpose input gradient (stride={stride}, dtype={dtype})",
                sparse_input_grad,
                dense_input_grad,
                rtol=input_grad_rtol,
                atol=input_grad_atol,
            )
            raise AssertionError(diag) from None

        # Compare kernel gradients
        assert sparse_kernel_grad is not None, "Sparse kernel grad is None"
        assert dense_kernel_grad is not None, "Dense kernel grad is None"

        try:
            torch.testing.assert_close(
                sparse_kernel_grad, dense_kernel_grad, rtol=kernel_grad_rtol, atol=kernel_grad_atol
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Transpose kernel gradient (stride={stride}, dtype={dtype})",
                sparse_kernel_grad,
                dense_kernel_grad,
                rtol=kernel_grad_rtol,
                atol=kernel_grad_atol,
            )
            raise AssertionError(diag) from None

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_transpose_uniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Strided transposed convolution forward+backward, stride (2,2,2)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_strided_conv_transpose_forward_backward(
            device, dtype, stride=(2, 2, 2), cluster_coords=cluster_coords
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_transpose_nonuniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Strided transposed convolution forward+backward, stride (1,2,3)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_strided_conv_transpose_forward_backward(
            device, dtype, stride=(1, 2, 3), cluster_coords=cluster_coords
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_transpose_large_stride(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Strided transposed convolution forward+backward, stride (3,3,3), full device/dtype coverage."""
        device_resolved = resolve_device(device)
        cluster_coords = torch.tensor(
            [
                [3, 3, 4],
                [6, 3, 4],
                [3, 6, 4],
                [3, 3, 7],
                [6, 6, 7],
                [9, 6, 10],
                [6, 9, 7],
                [9, 9, 10],
            ],
            device=device_resolved,
            dtype=torch.int32,
        )
        self._test_strided_conv_transpose_forward_backward(
            device, dtype, stride=(3, 3, 3), cluster_coords=cluster_coords
        )


class TestConvTransposeBackward(DisableTF32Mixin, unittest.TestCase):
    """
    Test transposed convolution backward pass (gradient computation).

    These tests verify that gradients flow correctly through sparse transposed
    convolution, matching the dense PyTorch conv_transpose3d gradients.
    """

    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)

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

        # Create leaf tensors with gradient tracking for dense ground truth
        dense_features = features_data.detach().clone().requires_grad_(True)
        # PyTorch conv_transpose3d expects weight [in_channels, out_channels, K],
        # but our sparse convention is [out_channels, in_channels, K].
        # Transpose channel dims for the dense ground truth.
        dense_kernel_transposed = kernel_5d.detach().transpose(0, 1).contiguous().clone().requires_grad_(True)

        # Build dense input via clone-then-scatter to maintain autograd graph.
        # Use grid_batch.ijk.jdata (the grid's actual storage order) rather than
        # cluster_coords, because GridBatch.from_ijk may reorder voxels internally.
        src_coords = grid_batch.ijk.jdata
        dense_input = torch.zeros((1, in_channels) + dense_shape, device=device, dtype=dtype, requires_grad=True)
        dense_input_data = dense_input.clone()
        for idx, coord in enumerate(src_coords):
            local_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
            dense_input_data[0, :, local_idx[0], local_idx[1], local_idx[2]] = dense_features[idx]

        with disable_tf32():
            dense_output = torch.nn.functional.conv_transpose3d(
                input=dense_input_data, weight=dense_kernel_transposed, padding=kernel_half, stride=1
            )

        # Apply gradient at output coordinates
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for idx, coord in enumerate(dst_ijks):
            out_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
            if all(0 <= out_idx[i] < dense_output.shape[i + 2] for i in range(3)):
                loss = loss + (dense_output[0, :, out_idx[0], out_idx[1], out_idx[2]] * output_grad[idx]).sum()

        loss.backward()

        # Read gradients directly from the leaf tensors
        dense_input_grad = dense_features.grad
        dense_kernel_grad_transposed = dense_kernel_transposed.grad

        # Compare input gradients
        assert sparse_input_grad is not None, "Sparse input grad is None"
        assert dense_input_grad is not None, "Dense input grad is None"

        try:
            torch.testing.assert_close(
                sparse_input_grad, dense_input_grad, rtol=tols["input_grad"][0], atol=tols["input_grad"][1]
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Input gradient (dtype={dtype})",
                sparse_input_grad,
                dense_input_grad,
                rtol=tols["input_grad"][0],
                atol=tols["input_grad"][1],
            )
            raise AssertionError(diag) from None

        # Compare kernel gradients
        # Dense gradient is in [in_ch, out_ch, K] layout; transpose back to [out_ch, in_ch, K]
        assert sparse_kernel_grad is not None and dense_kernel_grad_transposed is not None
        dense_kernel_grad = dense_kernel_grad_transposed.transpose(0, 1).contiguous()

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
