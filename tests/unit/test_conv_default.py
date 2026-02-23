# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the fVDB sparse convolution implementation.

These tests compare sparse convolution results against dense PyTorch ground truth
to verify correctness of:
  - Topology computation (conv_grid output coordinates)
  - Forward pass values
  - Backward pass gradients (input and kernel)
  - Strided convolution behavior

Ground truth computation uses utilities from convolution_utils.py.
"""

import math
import unittest

import torch
from fvdb.convolution_plan import _GatherScatterBackend, _GatherScatterConvFn
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
    compute_conv_grid_topology_ground_truth,
    conv_ground_truth_stride_1,
    conv_ground_truth_strided,
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
# Test Class
# =============================================================================


class TestConvDefault(DisableTF32Mixin, unittest.TestCase):

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    # =========================================================================
    # Topology Tests (conv_grid)
    # =========================================================================

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_single_impulse_bounds(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Validate that conv_grid output bounds match expected kernel footprint
        for a single input coordinate.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Check count
        kernel_volume = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), kernel_volume)

        # Check bounds
        expected_start = tuple(self.SINGLE_COORD[i] - kernel_half[i] for i in range(3))
        expected_end = tuple(self.SINGLE_COORD[i] + kernel_half[i] + 1 for i in range(3))

        actual_start = tuple(dst_ijks[:, dim].min().item() for dim in range(3))
        actual_end = tuple(dst_ijks[:, dim].max().item() + 1 for dim in range(3))

        self.assertEqual(actual_start, expected_start)
        self.assertEqual(actual_end, expected_end)

    def _test_conv_grid_topology(
        self,
        device: DeviceIdentifier,
        dtype: torch.dtype,
        stride: tuple[int, int, int],
        cluster_coords: torch.Tensor,
        check_negative_outputs: bool = False,
        check_non_negative_outputs: bool = False,
    ):
        """
        Core topology test: verify conv_grid output matches dense ground truth.

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

        # Create grid and get conv_grid output
        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Compute ground truth
        expected_coords = compute_conv_grid_topology_ground_truth(
            input_coords=cluster_coords,
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            device=device,
            dtype=dtype,
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

    # --- Stride 1 tests ---

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride1_near_origin(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with coordinates near origin, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_near_origin(device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_negative_outputs=True,
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride1_edge_aligned(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with edge-aligned coordinates, stride=1."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 1, 1),
            cluster_coords=cluster_coords,
            check_non_negative_outputs=True,
        )

    # --- Strided tests ---

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride_uniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with uniform stride (2,2,2)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(2, 2, 2),
            cluster_coords=cluster_coords,
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_conv_grid_topology_stride_nonuniform(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Topology test with non-uniform stride (1,2,3)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_conv_grid_topology(
            device,
            dtype,
            stride=(1, 2, 3),
            cluster_coords=cluster_coords,
        )

    # =========================================================================
    # Convolution Value Tests (forward pass)
    # =========================================================================

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that each kernel weight position activates the correct output coordinate.

        For a single input impulse, iterates over each kernel position and verifies
        that the output impulse appears at the expected coordinate.
        """
        device = resolve_device(device)
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        grid_batch = create_grid_from_coords(coord.unsqueeze(0), device)
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch(
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

                    # Expected: input_coord - (kernel_idx - kernel_half)
                    expected_coord = (
                        self.SINGLE_COORD[0] - (k0 - kernel_half[0]),
                        self.SINGLE_COORD[1] - (k1 - kernel_half[1]),
                        self.SINGLE_COORD[2] - (k2 - kernel_half[2]),
                    )
                    self.assertEqual(got_coord, expected_coord)

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_forward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test forward convolution of a single impulse against dense ground truth.

        Creates a single-voxel input, convolves with an anti-symmetric kernel,
        and verifies the sparse output matches the dense convolution result.
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

        # Dense ground truth
        dense_input = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_input[0, 0, coord[0], coord[1], coord[2]] = 1

        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        self.assertAlmostEqual(dense_output.sum().item(), kernel_sum, places=5)

        # Sparse convolution
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=1,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        self.assertIsInstance(conv_plan._backend, _GatherScatterBackend)

        sparse_output = conv_plan.execute(features, kernel_5d)
        sparse_output_flat = sparse_output.jdata.flatten()

        # Verify sparse matches dense at output locations
        tols = get_tolerances(dtype)
        dense_at_dst = dense_output[0, 0, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]]
        torch.testing.assert_close(sparse_output_flat, dense_at_dst, rtol=tols["forward"][0], atol=tols["forward"][1])

        # Verify sum matches kernel sum
        self.assertAlmostEqual(sparse_output_flat.sum().item(), kernel_sum, places=5)

        # Verify utility function also produces correct ground truth
        gt_activation, gt_convolved = conv_ground_truth_stride_1(
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
        Test forward convolution of multiple isolated impulses.

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

        # Dense ground truth
        dense_input = impulse_field.reshape(1, 1, *self.VOLUME_SHAPE)
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        # Sparse convolution
        grid_batch = create_grid_from_coords(impulse_coords, device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # Verify topology matches dense non-zeros
        dense_nonzero = torch.nonzero(dense_output[0, 0]).to(torch.int32)
        assert_coords_equal(dst_ijks, dense_nonzero)

        # Execute convolution
        features = JaggedTensor(torch.ones((num_impulses, 1), device=device, dtype=dtype))
        conv_plan = ConvolutionPlan.from_grid_batch(
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

    # =========================================================================
    # Backward Pass Tests
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass of sparse convolution against dense ground truth.

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

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
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
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=dense_kernel, padding="same")

        # === Sparse path ===
        conv_plan = ConvolutionPlan.from_grid_batch(
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

    # =========================================================================
    # Strided Convolution Tests (Forward + Backward)
    # =========================================================================

    def _test_strided_conv_forward_backward(
        self,
        device: DeviceIdentifier,
        dtype: torch.dtype,
        stride: tuple[int, int, int],
        cluster_coords: torch.Tensor,
        in_channels: int = 2,
        out_channels: int = 3,
    ):
        """
        Core test for strided convolution forward and backward passes.

        Tests that:
        1. Forward pass matches dense PyTorch ground truth
        2. Backward pass produces correct input gradients
        3. Backward pass produces correct kernel gradients

        Args:
            device: Device identifier
            dtype: Data type
            stride: Stride tuple (s0, s1, s2)
            cluster_coords: Input coordinates
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        tols = get_tolerances(dtype, kernel_size=self.KERNEL_SIZE)
        fwd_rtol, fwd_atol = tols["forward"]
        input_grad_rtol, input_grad_atol = tols["input_grad"]
        kernel_grad_rtol, kernel_grad_atol = tols["kernel_grad"]

        device = resolve_device(device)
        cluster_coords = cluster_coords.to(device=device)

        # Create grid and destination grid
        grid_batch = create_grid_from_coords(cluster_coords, device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=stride)
        dst_coords = dst_grid_batch.ijk.jdata

        num_voxels = len(cluster_coords)

        # Create features with gradient tracking
        features_data = torch.randn((num_voxels, in_channels), device=device, dtype=dtype, requires_grad=True)
        features = JaggedTensor(features_data)

        # Create anti-symmetric kernel with gradient tracking
        kernel_3d = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel_3d))

        # Expand to multi-channel
        kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0).expand(out_channels, in_channels, -1, -1, -1)
        kernel_5d = kernel_5d.clone().requires_grad_(True)

        # === Sparse Forward ===
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE,
            stride=stride,
            source_grid=grid_batch,
            target_grid=dst_grid_batch,
        )
        sparse_output = conv_plan.execute(features, kernel_5d)

        # === Dense Ground Truth Forward ===
        dense_features = features_data.detach().clone().requires_grad_(True)
        dense_kernel = kernel_5d.detach().clone().requires_grad_(True)

        dense_output_full, dense_output_at_dst = conv_ground_truth_strided(
            src_grid=grid_batch,
            dst_grid=dst_grid_batch,
            activation=JaggedTensor(dense_features),
            weights=dense_kernel,
            stride=stride,
            allow_tf32=False,
        )

        # Verify forward pass matches
        sparse_values_sorted, sparse_perm = sort_coords_by_ijk(dst_coords)

        sparse_out_sorted = sparse_output.jdata[sparse_perm]
        dense_out_sorted = dense_output_at_dst[sparse_perm]

        try:
            torch.testing.assert_close(sparse_out_sorted, dense_out_sorted, rtol=fwd_rtol, atol=fwd_atol)
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Forward output (stride={stride}, dtype={dtype})",
                sparse_out_sorted,
                dense_out_sorted,
                rtol=fwd_rtol,
                atol=fwd_atol,
            )
            raise AssertionError(diag) from None

        # === Backward Pass ===
        # Create output gradient (random for thorough gradient testing)
        output_grad = torch.randn_like(sparse_output.jdata)

        # Sparse backward
        sparse_output.jdata.backward(output_grad)
        sparse_input_grad = features_data.grad
        sparse_kernel_grad = kernel_5d.grad

        # Dense backward
        # Re-run the ground truth with gradient tracking
        dense_features2 = features_data.detach().clone().requires_grad_(True)
        dense_kernel2 = kernel_5d.detach().clone().requires_grad_(True)

        # Manually build dense input and run conv
        src_coords = grid_batch.ijk.jdata
        kernel_size = dense_kernel2.shape[2:]
        kernel_half = tuple(k // 2 for k in kernel_size)

        dst_min = dst_coords.min(dim=0).values.tolist()
        dst_max = dst_coords.max(dim=0).values.tolist()
        src_min = src_coords.min(dim=0).values.tolist()
        src_max = src_coords.max(dim=0).values.tolist()

        input_min_needed = tuple(dst_min[i] * stride[i] - kernel_half[i] for i in range(3))
        input_max_needed = tuple(dst_max[i] * stride[i] + kernel_half[i] for i in range(3))

        dense_min_raw = tuple(min(input_min_needed[i], src_min[i]) for i in range(3))
        dense_max = tuple(max(input_max_needed[i], src_max[i]) for i in range(3))

        # Align dense_min to stride grid for correct coordinate mapping
        dense_min = tuple((dense_min_raw[i] // stride[i]) * stride[i] for i in range(3))
        dense_shape = tuple(dense_max[i] - dense_min[i] + 1 for i in range(3))

        dense_input = torch.zeros((1, in_channels) + dense_shape, device=device, dtype=dtype, requires_grad=True)

        # Scatter features into dense (need differentiable path)
        dense_input_data = dense_input.clone()
        for idx, coord in enumerate(src_coords):
            local_idx = tuple(coord[i].item() - dense_min[i] for i in range(3))
            dense_input_data[0, :, local_idx[0], local_idx[1], local_idx[2]] = dense_features2[idx]

        with disable_tf32():
            dense_output2 = torch.nn.functional.conv3d(
                input=dense_input_data, weight=dense_kernel2, padding=kernel_half, stride=stride
            )

        # Compute output offset (dense_min is now aligned to stride)
        output_offset = tuple(dense_min[i] // stride[i] for i in range(3))

        # Apply gradient at dst coordinates
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for idx, coord in enumerate(dst_coords):
            out_idx = tuple(coord[i].item() - output_offset[i] for i in range(3))
            if all(0 <= out_idx[i] < dense_output2.shape[i + 2] for i in range(3)):
                loss = loss + (dense_output2[0, :, out_idx[0], out_idx[1], out_idx[2]] * output_grad[idx]).sum()

        loss.backward()

        dense_input_grad = dense_features2.grad
        dense_kernel_grad = dense_kernel2.grad

        # Compare input gradients
        assert sparse_input_grad is not None, "Sparse input grad is None"
        assert dense_input_grad is not None, "Dense input grad is None"

        try:
            torch.testing.assert_close(sparse_input_grad, dense_input_grad, rtol=input_grad_rtol, atol=input_grad_atol)
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Input gradient (stride={stride}, dtype={dtype})",
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
                f"Kernel gradient (stride={stride}, dtype={dtype})",
                sparse_kernel_grad,
                dense_kernel_grad,
                rtol=kernel_grad_rtol,
                atol=kernel_grad_atol,
            )
            raise AssertionError(diag) from None

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_uniform_forward_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Test forward and backward for uniform stride (2,2,2)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_strided_conv_forward_backward(
            device,
            dtype,
            stride=(2, 2, 2),
            cluster_coords=cluster_coords,
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_nonuniform_forward_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Test forward and backward for non-uniform stride (1,2,3)."""
        device_resolved = resolve_device(device)
        cluster_coords = get_cluster_edge_aligned(self.KERNEL_SIZE, device_resolved)
        self._test_strided_conv_forward_backward(
            device,
            dtype,
            stride=(1, 2, 3),
            cluster_coords=cluster_coords,
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_large_stride_forward_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Test forward and backward for larger stride (3,3,3) with full device/dtype coverage."""
        device_resolved = resolve_device(device)
        # Use a larger cluster to have meaningful output
        cluster_coords = torch.tensor(
            [
                [3, 3, 4],  # Base position (offset to ensure positive outputs)
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
        self._test_strided_conv_forward_backward(
            device,
            dtype,
            stride=(3, 3, 3),
            cluster_coords=cluster_coords,
        )

    # =========================================================================
    # Autograd gradcheck tests
    # =========================================================================
    #
    # torch.autograd.gradcheck verifies that the analytical backward produced by
    # _GatherScatterConvFn matches numerical finite differences.  This covers
    # the full Python autograd path (topology caching, JaggedTensor wrapping,
    # save_for_backward) that the C++ adjoint identity tests cannot reach.

    GRADCHECK_KERNEL_SIZE = (3, 3, 3)
    GRADCHECK_IN_CH = 2
    GRADCHECK_OUT_CH = 3
    GRADCHECK_DEVICE_COMBOS = [
        ["cpu"],
        ["cuda"],
    ]

    def _make_gradcheck_inputs(self, device: torch.device, in_ch: int, out_ch: int, kernel_size: tuple[int, int, int]):
        """Build a small sparse grid and random float64 features/weights for gradcheck."""
        cluster_coords = torch.tensor(
            [
                [3, 3, 4],
                [4, 3, 4],
                [3, 4, 4],
                [3, 3, 5],
                [4, 4, 5],
                [5, 4, 5],
                [4, 5, 4],
                [5, 5, 5],
            ],
            device=device,
            dtype=torch.int32,
        )
        grid = create_grid_from_coords(cluster_coords, device)
        num_voxels = len(cluster_coords)

        features = torch.randn(num_voxels, in_ch, device=device, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(out_ch, in_ch, *kernel_size, device=device, dtype=torch.float64, requires_grad=True)
        return grid, features, weights

    @parameterized.expand(GRADCHECK_DEVICE_COMBOS)
    def test_gradcheck_forward_stride1(self, device: DeviceIdentifier):
        """gradcheck: forward convolution, stride 1."""
        device = resolve_device(device)
        grid, features, weights = self._make_gradcheck_inputs(
            device, self.GRADCHECK_IN_CH, self.GRADCHECK_OUT_CH, self.GRADCHECK_KERNEL_SIZE
        )

        dst_grid = grid.conv_grid(kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=1)
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=1, source_grid=grid, target_grid=dst_grid
        )
        topo = plan._backend.topology

        def conv_fn(f, w):
            return _GatherScatterConvFn.apply(f, w, topo, False)

        torch.autograd.gradcheck(conv_fn, (features, weights), eps=1e-6, atol=1e-4, rtol=1e-3)

    @parameterized.expand(GRADCHECK_DEVICE_COMBOS)
    def test_gradcheck_forward_strided(self, device: DeviceIdentifier):
        """gradcheck: forward convolution, stride (2,2,2)."""
        device = resolve_device(device)
        grid, features, weights = self._make_gradcheck_inputs(
            device, self.GRADCHECK_IN_CH, self.GRADCHECK_OUT_CH, self.GRADCHECK_KERNEL_SIZE
        )

        stride = (2, 2, 2)
        dst_grid = grid.conv_grid(kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=stride)
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=stride, source_grid=grid, target_grid=dst_grid
        )
        topo = plan._backend.topology

        def conv_fn(f, w):
            return _GatherScatterConvFn.apply(f, w, topo, False)

        torch.autograd.gradcheck(conv_fn, (features, weights), eps=1e-6, atol=1e-4, rtol=1e-3)

    @parameterized.expand(GRADCHECK_DEVICE_COMBOS)
    def test_gradcheck_transpose_stride1(self, device: DeviceIdentifier):
        """gradcheck: transposed convolution, stride 1."""
        device = resolve_device(device)
        grid, features, weights = self._make_gradcheck_inputs(
            device, self.GRADCHECK_IN_CH, self.GRADCHECK_OUT_CH, self.GRADCHECK_KERNEL_SIZE
        )

        dst_grid = grid.conv_transpose_grid(kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=1)
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=1, source_grid=grid, target_grid=dst_grid
        )
        topo = plan._backend.topology

        def conv_fn(f, w):
            return _GatherScatterConvFn.apply(f, w, topo, True)

        torch.autograd.gradcheck(conv_fn, (features, weights), eps=1e-6, atol=1e-4, rtol=1e-3)

    @parameterized.expand(GRADCHECK_DEVICE_COMBOS)
    def test_gradcheck_transpose_strided(self, device: DeviceIdentifier):
        """gradcheck: transposed convolution, stride (2,2,2)."""
        device = resolve_device(device)
        grid, features, weights = self._make_gradcheck_inputs(
            device, self.GRADCHECK_IN_CH, self.GRADCHECK_OUT_CH, self.GRADCHECK_KERNEL_SIZE
        )

        stride = (2, 2, 2)
        dst_grid = grid.conv_transpose_grid(kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=stride)
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.GRADCHECK_KERNEL_SIZE, stride=stride, source_grid=grid, target_grid=dst_grid
        )
        topo = plan._backend.topology

        def conv_fn(f, w):
            return _GatherScatterConvFn.apply(f, w, topo, True)

        torch.autograd.gradcheck(conv_fn, (features, weights), eps=1e-6, atol=1e-4, rtol=1e-3)
