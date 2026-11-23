# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import torch
from parameterized import parameterized

from fvdb import Grid, GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]

# 1292**3 = 2,156,689,088 elements, just over INT32_MAX. These tests guard
# against dense accessor/index calculations overflowing for tensors that cannot
# be described by a signed 32-bit element count.
RESOLUTION = 1292
# RESOLUTION = 64 # under int32_t max limit


class TestAccessors(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_read_from_dense_cminor(self, device):
        """Read sparse values from a dense tensor whose element count exceeds INT32_MAX.

        Nonzero values at the active voxels verify that the operation is not a
        no-op. Both GridBatch and Grid are covered because they expose separate
        public APIs over the dense accessor implementation.
        """
        dense_origin = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
        grid_size = (RESOLUTION, RESOLUTION, RESOLUTION)
        # FP16 keeps the threshold-crossing dense allocation to about 4.02 GiB.
        dense_grid = torch.zeros(
            (1, *grid_size, 1),
            dtype=torch.float16,
            device=device,
        )

        sparse_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device)
        grid_batch = GridBatch.from_points(JaggedTensor(sparse_points), voxel_sizes=0.1, origins=0.0)
        batch_ijk = grid_batch.ijk.jdata.to(torch.long)
        expected_batch_data = torch.arange(
            1, grid_batch.total_voxels + 1, dtype=torch.float16, device=device
        ).unsqueeze(1)
        dense_grid[0, batch_ijk[:, 0], batch_ijk[:, 1], batch_ijk[:, 2]] = expected_batch_data

        read_jagged_data = grid_batch.inject_from_dense_cminor(dense_grid, dense_origin)
        self.assertIsInstance(read_jagged_data, JaggedTensor)
        self.assertTrue(torch.equal(read_jagged_data.jdata, expected_batch_data))

        grid = Grid.from_points(sparse_points, voxel_size=0.1, origin=0.0)
        grid_ijk = grid.ijk.to(torch.long)
        expected_data = dense_grid[0, grid_ijk[:, 0], grid_ijk[:, 1], grid_ijk[:, 2]]
        read_data = grid.inject_from_dense_cminor(dense_grid, dense_origin)
        self.assertIsInstance(read_data, torch.Tensor)
        self.assertTrue(torch.equal(read_data, expected_data))

    @parameterized.expand(all_device_combos)
    def test_write_to_dense_cminor(self, device):
        """Write sparse values into a dense tensor whose element count exceeds INT32_MAX.

        Shape and active-voxel checks exercise the large tensor's indexing
        without scanning its entire 4 GiB allocation. The GridBatch and Grid
        paths are checked independently.
        """
        dense_origin = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
        grid_size = (RESOLUTION, RESOLUTION, RESOLUTION)
        expected_dense_shape = (*grid_size, 1)

        sparse_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device)
        grid_batch = GridBatch.from_points(JaggedTensor(sparse_points), voxel_sizes=0.1, origins=0.0)
        batch_ijk = grid_batch.ijk.jdata.to(torch.long)
        batch_sparse_data = torch.arange(1, grid_batch.total_voxels + 1, dtype=torch.float16, device=device).unsqueeze(
            1
        )

        dense_batch = grid_batch.inject_to_dense_cminor(
            JaggedTensor(batch_sparse_data), dense_origin, grid_size
        ).squeeze(0)
        self.assertEqual(dense_batch.shape, expected_dense_shape)
        self.assertTrue(
            torch.equal(
                dense_batch[batch_ijk[:, 0], batch_ijk[:, 1], batch_ijk[:, 2]],
                batch_sparse_data,
            )
        )
        # Release the first 4 GiB view before testing Grid so the CUDA allocator
        # can reuse its storage instead of requiring two simultaneous outputs.
        del dense_batch

        grid = Grid.from_points(sparse_points, voxel_size=0.1, origin=0.0)
        grid_ijk = grid.ijk.to(torch.long)
        sparse_data = torch.arange(1, grid.num_voxels + 1, dtype=torch.float16, device=device).unsqueeze(1)

        dense = grid.inject_to_dense_cminor(sparse_data, dense_origin, grid_size).squeeze(0)
        self.assertEqual(dense.shape, expected_dense_shape)
        self.assertTrue(torch.equal(dense[grid_ijk[:, 0], grid_ijk[:, 1], grid_ijk[:, 2]], sparse_data))


if __name__ == "__main__":
    unittest.main()
