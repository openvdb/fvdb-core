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
from fvdb.grid_batch import save_gridbatch, load_gridbatch
from fvdb.utils.tests import dtype_to_atol
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
        self.assertIsInstance(morton_codes, JaggedTensor)
        self.assertIsInstance(morton_zyx_codes, JaggedTensor)

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
        self.assertIsInstance(morton_codes_with_offset, JaggedTensor)

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
        self.assertIsInstance(hilbert_codes, JaggedTensor)
        self.assertIsInstance(hilbert_zyx_codes, JaggedTensor)

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
        self.assertIsInstance(hilbert_codes_with_offset, JaggedTensor)

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

    @parameterized.expand(all_device_dtype_combos)
    def test_save_load_gridbatch(self, device, dtype):
        """Test saving and loading grid batches to/from files."""
        # Create a test grid batch with data
        grid_batch, jagged_data, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save the grid batch (without data to test structure only)
            save_gridbatch(tmp_path, grid_batch, data=None, name="test_grid")

            # Load it back (always load to CPU first, then move to device)
            loaded_grid_batch, loaded_data, loaded_names = load_gridbatch(tmp_path, device="cpu")

            # Move to target device if needed
            if device != "cpu":
                loaded_grid_batch = loaded_grid_batch.to(device)

            # Verify grid structure matches
            self.assertEqual(loaded_grid_batch.grid_count, grid_batch.grid_count)
            self.assertEqual(loaded_grid_batch.total_voxels, grid_batch.total_voxels)

            # Verify voxel counts match for each grid
            for i in range(grid_batch.grid_count):
                self.assertEqual(loaded_grid_batch.num_voxels_at(i), grid_batch.num_voxels_at(i))

            # Verify voxel coordinates match
            self.assertTrue(
                torch.allclose(loaded_grid_batch.ijk.jdata.float(), grid_batch.ijk.jdata.float(), atol=1e-5)
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @parameterized.expand(all_device_dtype_combos)
    def test_save_load_with_names(self, device, dtype):
        """Test saving and loading grid batches with named grids."""
        # Create a test grid batch
        grid_batch, jagged_data, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Create names for each grid
        grid_names = [f"grid_{i}" for i in range(grid_batch.grid_count)]

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save with names (without data)
            save_gridbatch(tmp_path, grid_batch, data=None, names=grid_names)

            # Load back
            loaded_grid_batch, loaded_data, loaded_names = load_gridbatch(tmp_path, device="cpu")

            # Move to target device if needed
            if device != "cpu":
                loaded_grid_batch = loaded_grid_batch.to(device)

            # Verify names match
            self.assertEqual(len(loaded_names), len(grid_names))
            for original_name, loaded_name in zip(grid_names, loaded_names):
                self.assertEqual(original_name, loaded_name)

            # Verify grid structure matches
            self.assertEqual(loaded_grid_batch.grid_count, grid_batch.grid_count)

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @parameterized.expand(all_device_dtype_combos)
    def test_save_load_compressed(self, device, dtype):
        """Test saving and loading with compression enabled."""
        # Create a test grid batch
        grid_batch, jagged_data, _ = make_grid_batch_and_jagged_point_data(
            device=device, dtype=dtype, include_boundary_points=True
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save with compression (without data)
            save_gridbatch(tmp_path, grid_batch, data=None, name="compressed_test", compressed=True)

            # Load back
            loaded_grid_batch, loaded_data, loaded_names = load_gridbatch(tmp_path, device="cpu")

            # Move to target device if needed
            if device != "cpu":
                loaded_grid_batch = loaded_grid_batch.to(device)

            # Verify grid structure matches
            self.assertEqual(loaded_grid_batch.total_voxels, grid_batch.total_voxels)
            self.assertEqual(loaded_grid_batch.grid_count, grid_batch.grid_count)

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
