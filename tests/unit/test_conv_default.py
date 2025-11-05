# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the default sparse convolution.

"""

import math
import unittest

import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import fourier_anti_symmetric_kernel, has_any_symmetry
from fvdb.utils.tests.convolution_utils import conv_ground_truth_stride_1
from parameterized import parameterized

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

all_device_dtype_combos = [
    # ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


class TestConvDefault(unittest.TestCase):

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_conv_grid(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        This test validates that the topology of the output grid correctly
        matches the shape of the kernel centered at the single impulse coordinate,
        and is not transposed across spatial dimensions. This was broken before!
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        ijks = JaggedTensor(coord.unsqueeze(0))
        grid_batch = GridBatch.from_ijk(ijks, device=device)

        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        kernel_volume = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), kernel_volume)

        # Extract the region around the impulse coordinate where the kernel should appear
        # The kernel is centered at the impulse coordinate
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        # Define the slice boundaries for extracting the kernel region
        start_coords = tuple(self.SINGLE_COORD[i] - kernel_half[i] for i in range(3))
        end_coords = tuple(self.SINGLE_COORD[i] + kernel_half[i] + 1 for i in range(3))

        # Find the actual bounds of the non-zero region
        actual_start_coords = tuple(dst_ijks[:, dim].min().item() for dim in range(3))
        actual_end_coords = tuple(dst_ijks[:, dim].max().item() + 1 for dim in range(3))

        # The actual bounds should exactly match the expected bounds
        self.assertEqual(actual_start_coords, start_coords)
        self.assertEqual(actual_end_coords, end_coords)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        This test iterates over each single weight location in the kernel space,
        creates a kernel that has just an impulse at that location, and convolves a single impulse
        conv grid with it. For each location within the kernel, we should expect the output to have
        a single impulse at the activation impulse coord minus the centered kernel impulse coord
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        ijks = JaggedTensor(coord.unsqueeze(0))
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))
        grid_batch = GridBatch.from_ijk(ijks, device=device)
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        kernel_half_width = tuple(k // 2 for k in self.KERNEL_SIZE)

        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1, source_grid=grid_batch, target_grid=dst_grid_batch
        )

        kernel_volume = math.prod(self.KERNEL_SIZE)
        self.assertEqual(len(dst_ijks), kernel_volume)

        for k0 in range(self.KERNEL_SIZE[0]):
            for k1 in range(self.KERNEL_SIZE[1]):
                for k2 in range(self.KERNEL_SIZE[2]):
                    # Create a kernel that has just an impulse at the current location
                    weights = torch.zeros((1, 1, *self.KERNEL_SIZE), device=device, dtype=dtype)
                    weights[0, 0, k0, k1, k2] = 1
                    self.assertEqual(weights.sum().item(), 1)

                    convolved_features_jagged = conv_plan.execute(features, weights)
                    convolved_features_flat = convolved_features_jagged.jdata.flatten()
                    self.assertEqual(convolved_features_flat.sum().item(), 1)

                    nonzero_mask = convolved_features_flat != 0
                    ijks_of_nonzero = dst_ijks[nonzero_mask].contiguous()
                    self.assertEqual(len(ijks_of_nonzero), 1)
                    got_output_coord = tuple(ijks_of_nonzero.flatten().tolist())
                    print(f"{k0, k1, k2} -> ijks_of_nonzero: {got_output_coord}")

                    # Expected output coordinate
                    centered_kernel_coord = (
                        k0 - kernel_half_width[0],
                        k1 - kernel_half_width[1],
                        k2 - kernel_half_width[2],
                    )
                    expected_output_coord = (
                        self.SINGLE_COORD[0] - centered_kernel_coord[0],
                        self.SINGLE_COORD[1] - centered_kernel_coord[1],
                        self.SINGLE_COORD[2] - centered_kernel_coord[2],
                    )
                    self.assertEqual(got_output_coord, expected_output_coord)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        This test creates a src grid with a single nonzero voxel as an impulse.
        The weights kernel that is created is anti-symmetric. The dst grid topology
        is already tested elsewhere. Here we validate that the resulting convolution
        is correct.
        """
        device = resolve_device(device)

        # For single impulse, we just need to make sure it's far enough away from
        # the boundary of the volume.
        # This is just validating that our test parameters are set up correctly.
        expected_volume_shape = tuple(a + 2 for a in self.KERNEL_SIZE)
        half_kernel_size = tuple(a // 2 for a in self.KERNEL_SIZE)
        expected_impulse_coord = tuple(1 + a for a in half_kernel_size)
        self.assertEqual(expected_volume_shape, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_impulse_coord, self.SINGLE_COORD)
        print("Confirmed that the expected volume/coord matches the config.")

        # Create a src grid with batch size 1 and a single nonzero voxel as an impulse.
        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        ijks = JaggedTensor(coord.unsqueeze(0))
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))
        grid_batch = GridBatch.from_ijk(ijks)
        dense_field_from_grid_batch = grid_batch.inject_to_dense_cmajor(
            features, min_coord=(0, 0, 0), grid_size=self.SINGLE_VOLUME_SHAPE
        )

        # Anti-symmetric kernel
        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        print("Confirmed that the kernel is anti-symmetric.")
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        kernel_sum = kernel_with_channels.sum().item()

        # Dense impulse field.
        dense_impulse_field = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        dense_impulse_field[0, 0, coord[0], coord[1], coord[2]] = 1
        self.assertEqual(dense_impulse_field.sum().item(), 1)
        print("Confirmed that the dense impulse field has a single nonzero voxel.")

        # Test that our manually constructed impulse field matches what we get from the grid batch
        torch.testing.assert_close(dense_impulse_field, dense_field_from_grid_batch, atol=1e-5, rtol=1e-6)
        print("Confirmed that the dense impulse field matches the grid batch impulse field.")

        # Test the dense convolution ourselves.
        _backend_setting = torch.backends.cudnn.allow_tf32
        # Disable TF32 for consistent precision across CPU and CUDA
        torch.backends.cudnn.allow_tf32 = False
        dense_convolved = torch.nn.functional.conv3d(
            input=dense_impulse_field, weight=kernel_with_channels, padding="same"
        )
        torch.backends.cudnn.allow_tf32 = _backend_setting
        self.assertEqual(dense_impulse_field.shape, dense_convolved.shape)
        dense_convolved_flat = dense_convolved.flatten()
        print("Confirmed that the dense convolved has the same shape as the dense impulse field.")

        # This dense convolution is our ground truth. Since the input is just an impulse, the
        # convolution should have the same sum as the kernel, and in fact should be the kernel
        # flipped. We've tested this heavily elsewhere, so we'll just assert that the sums are
        # the same.
        self.assertAlmostEqual(dense_convolved.sum().item(), kernel_sum, places=5)
        print("Confirmed that the dense convolved has the same sum as the kernel.")

        # Get a convolution plan
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        dst_ijks = dst_grid_batch.ijk.jdata

        # The dst_ijks should correspond to all voxel locations in the footprint of the
        # kernel over the impulse. Since the kernel is centered at SINGLE_COORD in a volume of
        # SINGLE_VOLUME_SHAPE, with stride=1, the output ijks should cover all coordinates
        # reachable by the kernel centered at SINGLE_COORD, which is exactly all coordinates
        # in a (3,5,7) block centered at (2,3,4), i.e. ijk in (2+a, 3+b, 4+c)
        # for a in [-1,0,1], b in [-2,-1,0,1,2], c in [-3,-2,-1,0,1,2,3]
        #
        # Let's build the set of expected ijks and compare to dst_ijks.
        expected_ijks = []
        for di in range(self.KERNEL_SIZE[0]):
            for dj in range(self.KERNEL_SIZE[1]):
                for dk in range(self.KERNEL_SIZE[2]):
                    i = self.SINGLE_COORD[0] + di - half_kernel_size[0]
                    j = self.SINGLE_COORD[1] + dj - half_kernel_size[1]
                    k = self.SINGLE_COORD[2] + dk - half_kernel_size[2]
                    expected_ijks.append([i, j, k])
        expected_ijks = torch.tensor(expected_ijks, device=device, dtype=torch.int32)
        # dst_ijks is shape (N, 3), expected_ijks is (N, 3)
        self.assertEqual(dst_ijks.shape, expected_ijks.shape)
        self.assertEqual(3, dst_ijks.shape[1])
        self.assertEqual(3, expected_ijks.shape[1])
        print("Confirmed that the dst ijks matches the expected ijks.")

        # Test the expected kernel volume against the dst ijks. This is technically
        # just testing the dst grid, which we've done elsewhere.
        expected_kernel_volume = math.prod(self.KERNEL_SIZE)
        dst_ijks_volume = len(dst_ijks)
        print(f"Expected kernel volume: {expected_kernel_volume}")
        print(f"Number of output ijks: {dst_ijks_volume}")
        self.assertEqual(dst_ijks_volume, expected_kernel_volume)
        print("Confirmed that the dst ijks volume matches the expected kernel volume.")

        # Test the expected bounds.
        expected_bounds_min = expected_ijks.min(dim=0)[0].tolist()
        expected_bounds_max = expected_ijks.max(dim=0)[0].tolist()
        dst_bounds_min = dst_ijks.min(dim=0)[0].tolist()
        dst_bounds_max = dst_ijks.max(dim=0)[0].tolist()
        print(f"Expected ijks bounds: {expected_bounds_min} to {expected_bounds_max}")
        print(f"Output ijks bounds: {dst_bounds_min} to {dst_bounds_max}")
        self.assertEqual(dst_bounds_min, expected_bounds_min)
        self.assertEqual(dst_bounds_max, expected_bounds_max)
        print("Confirmed that the dst ijks bounds matches the expected ijks bounds.")

        # The expected ijks should be the same as the dst ijks.
        self.assertTrue(torch.equal(expected_ijks, dst_ijks))
        print("Confirmed that the expected ijks matches the dst ijks.")

        # Create a convolution plan!
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.KERNEL_SIZE, stride=1, source_grid=grid_batch, target_grid=dst_grid_batch
        )
        self.assertEqual(str(conv_plan._backend), "ConvPackBackend.GATHER_SCATTER")
        print(f"Confirmed that the conv plan backend is GATHER_SCATTER.")

        # Execute the convolution plan!
        sparse_convolved_jagged = conv_plan.execute(features, kernel_with_channels)
        sparse_convolved_flat = sparse_convolved_jagged.jdata.flatten()

        # The sum of the convolved sparse plan should be the same as the kernel sum.
        conv_plan_sum = sparse_convolved_flat.sum().item()
        self.assertAlmostEqual(conv_plan_sum, kernel_sum, places=5)
        print("Confirmed that the sparse convolved has the same sum as the kernel.")

        # Extract the convolved region around the impulse
        print(f"expected_bounds_min: {expected_bounds_min}")
        print(f"expected_bounds_max: {expected_bounds_max}")
        expected_size = tuple(mx - mn + 1 for mn, mx in zip(expected_bounds_min, expected_bounds_max))
        print(f"expected_size: {expected_size}")
        dense_convolved_region = dense_convolved[
            0,
            0,
            expected_bounds_min[0] : expected_bounds_max[0] + 1,
            expected_bounds_min[1] : expected_bounds_max[1] + 1,
            expected_bounds_min[2] : expected_bounds_max[2] + 1,
        ]
        dense_convolved_region_flat = dense_convolved_region.flatten()
        print(f"dense_convolved_region.shape: {dense_convolved_region.shape}")
        print(f"dense_convolved.shape: {dense_convolved.shape}")

        self.assertEqual(dense_convolved_region.shape, kernel.shape)
        print("Confirmed that the dense convolved region matches the kernel shape.")

        # Since PyTorch conv3d performs cross-correlation, we need to flip to get the result to
        # match the kernel.
        kernel_flipped = torch.flip(kernel, dims=[0, 1, 2])

        # The flipped convolved region should match the kernel
        torch.testing.assert_close(dense_convolved_region, kernel_flipped, rtol=1e-5, atol=1e-6)
        print("Confirmed that the dense convolved region matches the flipped kernel.")

        # The sparse convolved values should match the dense values, because we've
        # already tested that the ijks are the same.
        torch.testing.assert_close(sparse_convolved_flat, dense_convolved_region_flat, rtol=1e-5, atol=1e-6)
        print("Confirmed that the sparse convolved values match the dense convolved region.")

        # Even though this is transitive, we'll assert it explicitly.
        torch.testing.assert_close(sparse_convolved_flat, kernel_flipped.flatten(), rtol=1e-5, atol=1e-6)
        print("Confirmed that the sparse convolved values match the flipped kernel.")

        # Lastly, we'll get dense convolved from the sparse convolved
        dense_convolved_from_sparse = dst_grid_batch.inject_to_dense_cmajor(
            sparse_convolved_jagged, min_coord=(0, 0, 0), grid_size=self.SINGLE_VOLUME_SHAPE
        )
        torch.testing.assert_close(dense_convolved_from_sparse, dense_convolved, rtol=1e-5, atol=1e-6)
        print("Confirmed that the dense convolved from the sparse convolved matches the dense convolved.")

        # Let's also test the output of the convolution_utils ground truth.
        gt_dense_activation, gt_dense_convolved = conv_ground_truth_stride_1(
            grid_batch=grid_batch,
            activation=features,
            weights=kernel_with_channels,
            dense_dims=self.SINGLE_VOLUME_SHAPE,
            ijk_min=(0, 0, 0),
            allow_tf32=False,
        )
        torch.testing.assert_close(gt_dense_activation, dense_impulse_field, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gt_dense_convolved, dense_convolved, rtol=1e-5, atol=1e-6)
        print("Confirmed that the utils ground truth dense convolved matches the dense convolved.")
