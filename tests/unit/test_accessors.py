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

RESOLUTION = 1292  # over int32_t max limit
# RESOLUTION = 64 # under int32_t max limit


class TestAccessors(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_read_from_dense_cminor(self, device):
        dense_origin = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
        dense_grid = torch.zeros(
            (1, RESOLUTION, RESOLUTION, RESOLUTION, 1),
            dtype=torch.float16,
            device=device,
        )

        sparse_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device)
        grid_batch = GridBatch.from_points(JaggedTensor(sparse_points), voxel_sizes=0.1, origins=0.0)

        read_jagged_data = grid_batch.read_from_dense_cminor(dense_grid, dense_origin)
        self.assertIsInstance(read_jagged_data, JaggedTensor)

        grid = Grid.from_points(sparse_points, voxel_size=0.1, origin=0.0)
        read_tensor_data = grid.inject_from_dense_cminor(dense_grid, dense_origin)
        self.assertIsInstance(read_tensor_data, torch.Tensor)

    @parameterized.expand(all_device_combos)
    def test_write_to_dense_cminor(self, device):
        dense_origin = torch.tensor([0, 0, 0]).to(torch.long).to(device)

        zero_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device)
        grid_batch = GridBatch.from_points(JaggedTensor(zero_points), voxel_sizes=0.1, origins=0.0)

        sparse_data = torch.tensor([[0], [0]], dtype=torch.float16, device=device)
        grid_batch.write_to_dense_cminor(JaggedTensor(sparse_data), dense_origin, (RESOLUTION, RESOLUTION, RESOLUTION))

        grid = Grid.from_points(zero_points, voxel_size=0.1, origin=0.0)
        grid.write_to_dense_cminor(sparse_data, dense_origin, (RESOLUTION, RESOLUTION, RESOLUTION))


if __name__ == "__main__":
    unittest.main()
