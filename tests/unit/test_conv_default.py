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
    def test_single_impulse(self, device: DeviceIdentifier, dtype: torch.dtype):
        device = resolve_device(device)

        # For single impulse, we just need to make sure it's far enough away from
        # the boundary of the volume.

        expected_volume_shape = tuple(a + 2 for a in self.KERNEL_SIZE)
        expected_impulse_coord = tuple(1 + a // 2 for a in self.KERNEL_SIZE)

        self.assertEqual(expected_volume_shape, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_impulse_coord, self.SINGLE_COORD)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)

        ijks = JaggedTensor(coord.unsqueeze(0))
        features = JaggedTensor(torch.ones((1, 1), device=device, dtype=dtype))
        grid_batch = GridBatch.from_ijk(ijks, device=device)
        dense_field_from_grid_batch = grid_batch.write_to_dense_cmajor(
            features, min_coord=(0, 0, 0), grid_size=self.SINGLE_VOLUME_SHAPE
        )

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        kernel_sum = kernel_with_channels.sum().item()

        impulse_field = torch.zeros((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, 0, coord[0], coord[1], coord[2]] = 1
        self.assertEqual(impulse_field.sum().item(), 1)
        # Test that our manually constructed impulse field matches what we get from the grid batch
        torch.testing.assert_close(impulse_field, dense_field_from_grid_batch, atol=1e-5, rtol=1e-6)

        gt_dense_activation, gt_convolved = conv_ground_truth_stride_1(
            grid_batch, features, kernel_with_channels, dense_dims=self.SINGLE_VOLUME_SHAPE, ijk_min=(0, 0, 0)
        )

        # Test that the dense activation matches the ground truth
        torch.testing.assert_close(dense_field_from_grid_batch, gt_dense_activation, atol=1e-5, rtol=1e-6)

        # Do a single convolution
        _backend_setting = torch.backends.cudnn.allow_tf32
        # Disable TF32 for consistent precision across CPU and CUDA
        torch.backends.cudnn.allow_tf32 = False
        convolved = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")
        self.assertEqual(impulse_field.shape, convolved.shape)
        torch.backends.cudnn.allow_tf32 = _backend_setting

        # Check the sums
        conv_sum = convolved.sum().item()
        gt_conv_sum = gt_convolved.sum().item()
        self.assertAlmostEqual(conv_sum, gt_conv_sum, places=5)
        self.assertAlmostEqual(conv_sum, kernel_sum, places=5)

        # Test that the convolved field matches the ground truth
        torch.testing.assert_close(convolved, gt_convolved, atol=1e-5, rtol=1e-6)

        # Get a convolution plan
        dst_grid_batch = grid_batch.conv_grid(kernel_size=self.KERNEL_SIZE, stride=1)
        # dst_grid_batch = grid_batch.conv_grid(kernel_size=(7, 5, 3), stride=1)

        dst_ijks = dst_grid_batch.ijk.jdata
        # The dst_ijks should correspond to all voxel locations in the footprint of the kernel over the impulse.
        # Since the kernel is centered at SINGLE_COORD in a volume of SINGLE_VOLUME_SHAPE, with stride=1,
        # the output ijks should cover all coordinates reachable by the kernel centered at SINGLE_COORD, which is
        # exactly all coordinates in a (3,5,7) block centered at (2,3,4), i.e. ijk in (2+a, 3+b, 4+c) for
        # a in [-1,0,1], b in [-2,-1,0,1,2], c in [-3,-2,-1,0,1,2,3]
        #
        # Let's build the set of expected ijks and compare to dst_ijks.
        expected_ijks = []
        for di in range(-(self.KERNEL_SIZE[0] // 2), self.KERNEL_SIZE[0] // 2 + 1):
            for dj in range(-(self.KERNEL_SIZE[1] // 2), self.KERNEL_SIZE[1] // 2 + 1):
                for dk in range(-(self.KERNEL_SIZE[2] // 2), self.KERNEL_SIZE[2] // 2 + 1):
                    i = self.SINGLE_COORD[0] + di
                    j = self.SINGLE_COORD[1] + dj
                    k = self.SINGLE_COORD[2] + dk
                    expected_ijks.append([i, j, k])
        expected_ijks = torch.tensor(expected_ijks, device=device, dtype=torch.int32)
        # dst_ijks is shape (N, 3), expected_ijks is (N, 3)
        self.assertEqual(dst_ijks.shape, expected_ijks.shape)

        print(f"Kernel volume: {math.prod(self.KERNEL_SIZE)}")
        print(f"Number of output ijks: {len(dst_ijks)}")
        self.assertEqual(len(dst_ijks), math.prod(self.KERNEL_SIZE))

        expected_bounds_min = expected_ijks.min(dim=0)[0].tolist()
        expected_bounds_max = expected_ijks.max(dim=0)[0].tolist()
        dst_bounds_min = dst_ijks.min(dim=0)[0].tolist()
        dst_bounds_max = dst_ijks.max(dim=0)[0].tolist()
        print(f"Expected ijks bounds: {expected_bounds_min} to {expected_bounds_max}")
        print(f"Output ijks bounds: {dst_bounds_min} to {dst_bounds_max}")
        self.assertEqual(dst_bounds_min, expected_bounds_min)
        self.assertEqual(dst_bounds_max, expected_bounds_max)

        expected_ijks_fake_morton = (
            expected_ijks[:, 0] * 10000 + expected_ijks[:, 1] * 100 + expected_ijks[:, 2] * 1
        ).to(torch.int64)

        expected_ijks_perm = torch.argsort(expected_ijks_fake_morton)
        expected_ijks_sorted = expected_ijks[expected_ijks_perm]

        dst_ijks_fake_morton = (dst_ijks[:, 0] * 10000 + dst_ijks[:, 1] * 100 + dst_ijks[:, 2] * 1).to(torch.int64)
        dst_ijks_perm = torch.argsort(dst_ijks_fake_morton)
        dst_ijks_sorted = dst_ijks[dst_ijks_perm]
        torch.testing.assert_close(dst_ijks_sorted, expected_ijks_sorted)

        # conv_plan = ConvolutionPlan.from_grid_batch(
        #     kernel_size=self.KERNEL_SIZE, stride=1, source_grid=grid_batch, target_grid=dst_grid_batch
        # )
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=(7, 5, 3), stride=1, source_grid=grid_batch, target_grid=dst_grid_batch
        )
        print(f"\n\nconv plan backend: {conv_plan._backend}\n\n")

        convolved_sparse_plan = conv_plan.execute(features, kernel_with_channels)
        conv_plan_sum = convolved_sparse_plan.jdata.sum().item()
        self.assertAlmostEqual(conv_plan_sum, gt_conv_sum, places=5)
        self.assertAlmostEqual(conv_plan_sum, kernel_sum, places=5)

        out_ijks_list = dst_grid_batch.ijk.jdata.tolist()
        out_features_list = convolved_sparse_plan.jdata.tolist()
        kernel_list = kernel.tolist()
        kernel_flattened_list = kernel.contiguous().view(-1).tolist()
        print(f"out_ijks_list: {out_ijks_list}")
        print(f"out_features_list: {out_features_list}")
        print(f"kernel_list: {kernel_list}")
        print(f"kernel_flattened_list: {kernel_flattened_list}")

        assoc = {}
        kmajor_stride = self.KERNEL_SIZE[0] * self.KERNEL_SIZE[1]
        kinner_stride = self.KERNEL_SIZE[0]
        kminor_stride = 1
        for i in range(len(out_ijks_list)):
            ijk_theory = out_ijks_list[i]
            ijk_theory = tuple(k - 1 for k in ijk_theory)
            feature = out_features_list[i][0]
            for j in range(len(kernel_flattened_list)):
                if abs(kernel_flattened_list[j] - feature) < 1e-5:
                    kmajor = j // kmajor_stride
                    kinner = (j - kmajor * kmajor_stride) // kinner_stride
                    kminor = (j - (kmajor * kmajor_stride + kinner * kinner_stride)) // kminor_stride

                    print(f"{i} theory, {j} actual \t\tk: {(kmajor, kinner, kminor)}")

                    assoc[i] = j
            # for ii in range(self.KERNEL_SIZE[0]):
            #     for jj in range(self.KERNEL_SIZE[1]):
            #         for kk in range(self.KERNEL_SIZE[2]):
            #             feature_kernel = kernel_list[ii][jj][kk]
            #             if abs(feature_kernel - feature) < 1e-5:
            #                 print(f"{ijk_theory} theory, {ii, jj, kk} actual")
            #                 assoc[ijk_theory] = (ii, jj, kk)

        print(f"Length of assoc: {len(assoc)}")
        assoc_keys_set = set(assoc.keys())
        print(f"Length of assoc_keys_set: {len(assoc_keys_set)}")

        assoc_values_set = set(assoc.values())
        print(f"Length of assoc_values_set: {len(assoc_values_set)}")

        self.assertEqual(len(assoc_keys_set), len(assoc_values_set))

        convolved_dense_plan = dst_grid_batch.write_to_dense_cmajor(
            convolved_sparse_plan, min_coord=(0, 0, 0), grid_size=self.SINGLE_VOLUME_SHAPE
        )
        print(f"convolved_dense_plan.shape: {convolved_dense_plan.shape}")
        print(f"gt_convolved.shape: {gt_convolved.shape}")

        # Try all possible axis permutations and flips of the spatial dims (1,2,3) to see what matches gt_convolved.
        import itertools

        def find_best_match(convolved, gt, spatial_dims=(1, 2, 3), rtol=1e-6, atol=1e-5):
            """
            Brute-force try permutations and flips along the spatial dims of convolved to match gt.
            Returns (perm, flip_combo) if match is found, else None.
            """
            original_shape = convolved.shape
            spatial_axes = list(spatial_dims)
            perms = list(itertools.permutations(spatial_axes))
            flip_combos = list(itertools.product([False, True], repeat=3))
            for perm in perms:
                # Map original (d1,d2,d3) axes to permuted axes
                permute_map = [0] + list(perm) + [4] if convolved.ndim == 5 else [0] + list(perm)
                permuted = convolved.permute(permute_map)
                for flips in flip_combos:
                    mod = permuted
                    flip_dims = [ax for flip, ax in zip(flips, range(1, 4)) if flip]
                    if flip_dims:
                        mod = torch.flip(mod, dims=flip_dims)
                    try:
                        torch.testing.assert_close(mod, gt, atol=atol, rtol=rtol)
                        print(f"Match found! Permutation {perm}, Flips {flips}")
                        return perm, flips
                    except AssertionError as e:
                        # Print some measure of total error to help debugging
                        abs_error = (mod - gt).abs().sum().item()
                        max_error = (mod - gt).abs().max().item()
                        print(
                            f"Permutation {perm}, Flips {flips}: Abs error sum={abs_error:.7g}, Max error={max_error:.7g}"
                        )
                        continue
            print("No permutation/flip combination matched.")
            return None

        # # Use this detective function to see if spatial dims are just shuffled/flipped.
        # _match_result = find_best_match(convolved_dense_plan, gt_convolved)

        # if _match_result is not None:
        #     perm, flips = _match_result
        #     convolved_dense_plan = convolved_dense_plan.permute(perm).flip(dims=flips)
        # else:
        #     print("No permutation/flip combination matched.")
        #     self.fail("No permutation/flip combination matched.")
        self.fail("Boo")

        # torch.testing.assert_close(convolved_dense_plan, gt_convolved, atol=1e-5, rtol=1e-6)
