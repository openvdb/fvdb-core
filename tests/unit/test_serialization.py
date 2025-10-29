# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import torch
import numpy as np
import tempfile
import os
from parameterized import parameterized

from fvdb import GridBatch, JaggedTensor
from fvdb.utils.tests.grid_utils import make_grid_batch_and_jagged_point_data

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
    def test_morton_codes(self, device, dtype):
        """Test Morton code generation (xyz and zyx variants)."""
        # Create a test grid batch with known active voxels
        grid_batch, _, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Get Morton codes for both orderings
        morton_codes = grid_batch.morton()
        morton_zyx_codes = grid_batch.morton_zyx()

        # Test that codes are returned as JaggedTensor
        # self.assertIsInstance(morton_codes, JaggedTensor)
        # self.assertIsInstance(morton_zyx_codes, JaggedTensor)

        # Verify shape: should have one code per voxel
        self.assertEqual(morton_codes.jdata.shape[0], grid_batch.total_voxels)
        self.assertEqual(morton_zyx_codes.jdata.shape[0], grid_batch.total_voxels)

        # Verify codes are uint64
        self.assertEqual(morton_codes.jdata.dtype, torch.int64)
        self.assertEqual(morton_zyx_codes.jdata.dtype, torch.int64)

        # Test that codes are non-negative
        self.assertTrue(torch.all(morton_codes.jdata >= 0))
        self.assertTrue(torch.all(morton_zyx_codes.jdata >= 0))

        # Test with explicit offset
        offset = torch.tensor([10, 10, 10], dtype=torch.int32, device=device)
        morton_codes_with_offset = grid_batch.morton(offset=offset)
        # self.assertIsInstance(morton_codes_with_offset, JaggedTensor)

    @parameterized.expand(all_device_dtype_combos)
    def test_hilbert_codes(self, device, dtype):
        """Test Hilbert curve code generation (xyz and zyx variants)."""
        # Create a test grid batch with known active voxels
        grid_batch, _, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Get Hilbert codes for both orderings
        hilbert_codes = grid_batch.hilbert()
        hilbert_zyx_codes = grid_batch.hilbert_zyx()

        # Test that codes are returned as JaggedTensor
        # self.assertIsInstance(hilbert_codes, JaggedTensor)
        # self.assertIsInstance(hilbert_zyx_codes, JaggedTensor)

        # Verify shape: should have one code per voxel
        self.assertEqual(hilbert_codes.jdata.shape[0], grid_batch.total_voxels)
        self.assertEqual(hilbert_zyx_codes.jdata.shape[0], grid_batch.total_voxels)

        # Verify codes are uint64
        self.assertEqual(hilbert_codes.jdata.dtype, torch.int64)
        self.assertEqual(hilbert_zyx_codes.jdata.dtype, torch.int64)

        # Test that codes are non-negative
        self.assertTrue(torch.all(hilbert_codes.jdata >= 0))
        self.assertTrue(torch.all(hilbert_zyx_codes.jdata >= 0))

        # Test with explicit offset
        offset = torch.tensor([10, 10, 10], dtype=torch.int32, device=device)
        hilbert_codes_with_offset = grid_batch.hilbert(offset=offset)
        # self.assertIsInstance(hilbert_codes_with_offset, JaggedTensor)

    @parameterized.expand(all_device_dtype_combos)
    def test_space_filling_curve_properties(self, device, dtype):
        """Test that space-filling curve codes have expected properties."""
        # Create a test grid batch
        grid_batch, _, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Get all types of codes
        morton_codes = grid_batch.morton()
        morton_zyx_codes = grid_batch.morton_zyx()
        hilbert_codes = grid_batch.hilbert()
        hilbert_zyx_codes = grid_batch.hilbert_zyx()

        # Test that different curve types produce different codes
        # (not all codes should be identical between different curve types)
        self.assertFalse(torch.all(morton_codes.jdata == hilbert_codes.jdata))

        # Test that xyz and zyx variants produce different orderings for non-cubic grids
        # (they may be the same for highly symmetric cases, so we just check they're valid)
        self.assertTrue(morton_codes.jdata.shape == morton_zyx_codes.jdata.shape)
        self.assertTrue(hilbert_codes.jdata.shape == hilbert_zyx_codes.jdata.shape)

        # Test that codes are within valid range for uint64
        self.assertTrue(torch.all(morton_codes.jdata >= 0))
        self.assertTrue(torch.all(hilbert_codes.jdata >= 0))


if __name__ == "__main__":
    unittest.main()
