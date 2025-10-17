# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import torch
import numpy as np
from parameterized import parameterized

from fvdb import GridBatch, JaggedTensor
from fvdb.utils.tests import dtype_to_atol
from fvdb.utils.tests.grid_utils import create_test_grid_batch

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


class TestSerialization(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)

    @parameterized.expand(all_device_dtype_combos)
    def test_morton_permutation(self, device, dtype):
        """Test Morton order permutations (xyz and zyx variants)."""
        # Create a test grid batch with known active voxels
        batch_size = 2
        grid_size = 8

        # Create a grid batch with some active voxels
        grid_batch = create_test_grid_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            device=device,
            density=0.3  # 30% of voxels will be active
        )

        # Get permutation indices for both Morton orderings
        morton_perm = grid_batch.permutation_morton()
        morton_zyx_perm = grid_batch.permutation_morton_zyx()

        # Test that permutations are valid for each grid in batch
        offset = 0
        for grid_idx in range(grid_batch.grid_count):
            num_voxels = grid_batch.num_voxels_at(grid_idx)
            if num_voxels == 0:
                continue

            # Extract permutation indices for this grid
            grid_perm = morton_perm.jdata[offset:offset + num_voxels].squeeze(-1)
            grid_perm_zyx = morton_zyx_perm.jdata[offset:offset + num_voxels].squeeze(-1)

            # Verify permutations contain all indices
            expected_indices = torch.arange(offset, offset + num_voxels, device=device)
            self.assertTrue(torch.sort(grid_perm)[0].equal(expected_indices))
            self.assertTrue(torch.sort(grid_perm_zyx)[0].equal(expected_indices))

            # Get Morton codes
            morton_codes = grid_batch.encode_morton().jdata[offset:offset + num_voxels]
            morton_zyx_codes = grid_batch.encode_morton_zyx().jdata[offset:offset + num_voxels]

            # Verify codes are sorted after applying permutation
            sorted_codes = morton_codes[grid_perm - offset]
            sorted_codes_zyx = morton_zyx_codes[grid_perm_zyx - offset]

            self.assertTrue(torch.all(sorted_codes[1:] >= sorted_codes[:-1]))
            self.assertTrue(torch.all(sorted_codes_zyx[1:] >= sorted_codes_zyx[:-1]))

            offset += num_voxels


    @parameterized.expand(all_device_dtype_combos)
    def test_hilbert_permutation(self, device, dtype):
        """Test Hilbert curve permutations (xyz and zyx variants)."""
        # Create a test grid batch with known active voxels
        batch_size = 2
        grid_size = 8

        # Create a grid batch with some active voxels
        grid_batch = create_test_grid_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            device=device,
            density=0.3  # 30% of voxels will be active
        )

        # Get permutation indices for both Hilbert orderings
        hilbert_perm = grid_batch.permutation_hilbert()
        hilbert_zyx_perm = grid_batch.permutation_hilbert_zyx()

        # Test that permutations are valid for each grid in batch
        offset = 0
        for grid_idx in range(grid_batch.grid_count):
            num_voxels = grid_batch.num_voxels_at(grid_idx)
            if num_voxels == 0:
                continue

            # Extract permutation indices for this grid
            grid_perm = hilbert_perm.jdata[offset:offset + num_voxels].squeeze(-1)
            grid_perm_zyx = hilbert_zyx_perm.jdata[offset:offset + num_voxels].squeeze(-1)

            # Verify permutations contain all indices
            expected_indices = torch.arange(offset, offset + num_voxels, device=device)
            self.assertTrue(torch.sort(grid_perm)[0].equal(expected_indices))
            self.assertTrue(torch.sort(grid_perm_zyx)[0].equal(expected_indices))

            # Get Hilbert codes
            hilbert_codes = grid_batch.encode_hilbert().jdata[offset:offset + num_voxels]
            hilbert_zyx_codes = grid_batch.encode_hilbert_zyx().jdata[offset:offset + num_voxels]

            # Verify codes are sorted after applying permutation
            sorted_codes = hilbert_codes[grid_perm - offset]
            sorted_codes_zyx = hilbert_zyx_codes[grid_perm_zyx - offset]

            self.assertTrue(torch.all(sorted_codes[1:] >= sorted_codes[:-1]))
            self.assertTrue(torch.all(sorted_codes_zyx[1:] >= sorted_codes_zyx[:-1]))

            offset += num_voxels


    @parameterized.expand(all_device_dtype_combos)
    def test_permutation_validity(self, device, dtype):
        """Test that permutation indices are valid (complete and unique)."""
        batch_size = 3
        grid_size = 16

        # Create grid batch with varying densities
        densities = [0.1, 0.5, 0.8]  # Test different sparsity levels
        for density in densities:
            grid_batch = create_test_grid_batch(
                batch_size=batch_size,
                grid_size=grid_size,
                device=device,
                density=density
            )

            # Test all permutation types
            permutations = [
                grid_batch.permutation_morton(),
                grid_batch.permutation_morton_zyx(),
                grid_batch.permutation_hilbert(),
                grid_batch.permutation_hilbert_zyx()
            ]

            for perm in permutations:
                offset = 0
                for grid_idx in range(grid_batch.grid_count):
                    num_voxels = grid_batch.num_voxels_at(grid_idx)
                    if num_voxels == 0:
                        continue

                    # Extract permutation indices for this grid
                    grid_perm = perm.jdata[offset:offset + num_voxels].squeeze(-1)

                    # Verify indices are within valid range
                    self.assertTrue(torch.all(grid_perm >= offset))
                    self.assertTrue(torch.all(grid_perm < offset + num_voxels))

                    # Verify indices are unique
                    self.assertEqual(len(torch.unique(grid_perm)), num_voxels)

                    # Verify all indices are present
                    expected_indices = torch.arange(offset, offset + num_voxels, device=device)
                    self.assertTrue(torch.sort(grid_perm)[0].equal(expected_indices))

                    offset += num_voxels

if __name__ == "__main__":
    unittest.main()
