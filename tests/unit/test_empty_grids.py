# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import os
import pickle
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


def dtype_to_atol(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 1e-1
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float64:
        return 1e-5
    raise ValueError("dtype must be a valid torch floating type")


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        # self.test_path = os.path.join(os.path.dirname(
        #     os.path.realpath(__file__)), "..", "data")
        pass

    @parameterized.expand(["cuda", "cpu"])
    def test_building_zero_grids_constructor(self, device):
        grids = GridBatch.from_zero_grids(device=device)
        self.assertEqual(grids.all_have_zero_voxels, True)
        self.assertEqual(grids.bboxes.shape, (0, 2, 3))
        self.assertEqual(grids.cum_voxels.shape, (0,))
        self.assertEqual(grids.dual_bboxes.shape, (0, 2, 3))
        self.assertEqual(grids.grid_count, 0)
        self.assertEqual(grids.voxel_to_world_matrices.shape, (0, 4, 4))
        self.assertTrue(grids.has_zero_grids)
        self.assertEqual(grids.ijk.jdata.shape, (0, 3))
        self.assertEqual(grids.jidx.shape, (0,))
        self.assertEqual(grids.joffsets.shape, (0,))
        self.assertEqual(grids.num_bytes.shape, (0,))
        self.assertEqual(grids.num_leaf_nodes.shape, (0,))
        self.assertEqual(grids.num_voxels.shape, (0,))
        self.assertEqual(grids.origins.shape, (0, 3))
        self.assertEqual(grids.total_bbox.shape, (2, 3))
        self.assertEqual(grids.total_bytes, 0)
        self.assertEqual(grids.total_leaf_nodes, 0)
        self.assertEqual(grids.total_voxels, 0)
        self.assertEqual(grids.voxel_sizes.shape, (0, 3))
        self.assertEqual(grids.world_to_voxel_matrices.shape, (0, 4, 4))

    @parameterized.expand(["cuda", "cpu"])
    def test_building_zero_voxels_constructor(self, device):
        expected_voxel_sizes = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        expected_origins = torch.tensor([[0.0, 0.0, 0.0]], device=device)

        grids = GridBatch.from_zero_voxels(device=device, voxel_sizes=1, origins=0)

        self.assertEqual(grids.all_have_zero_voxels, True)
        self.assertEqual(grids.bboxes.shape, (1, 2, 3))
        self.assertEqual(grids.cum_voxels.shape, (1,))
        self.assertEqual(grids.dual_bboxes.shape, (1, 2, 3))
        self.assertEqual(grids.grid_count, 1)
        self.assertEqual(grids.voxel_to_world_matrices.shape, (1, 4, 4))
        self.assertFalse(grids.has_zero_grids)
        self.assertEqual(grids.ijk.jdata.shape, (0, 3))
        self.assertEqual(grids.jidx.shape, (0,))
        self.assertEqual(grids.joffsets.shape, (2,))
        self.assertEqual(grids.num_bytes.shape, (1,))
        self.assertEqual(grids.num_leaf_nodes.shape, (1,))
        self.assertEqual(grids.num_voxels.shape, (1,))
        self.assertEqual(grids.origins.shape, (1, 3))
        self.assertEqual(grids.total_bbox.shape, (2, 3))
        self.assertEqual(grids.total_leaf_nodes, 0)
        self.assertEqual(grids.total_voxels, 0)
        self.assertEqual(grids.voxel_sizes.shape, (1, 3))
        self.assertEqual(grids.world_to_voxel_matrices.shape, (1, 4, 4))
        self.assertTrue(torch.allclose(grids.voxel_sizes, expected_voxel_sizes))
        self.assertTrue(torch.allclose(grids.origins, expected_origins))

    @parameterized.expand(["cuda", "cpu"])
    def test_zero_voxels_grid_bounding_boxes(self, device):
        origins = torch.tensor(
            [[0, 4, 3], [-1, 1, -100], [-7, 23, 14], [19, 18, -441]], device=device, dtype=torch.float32
        )
        voxel_sizes = torch.tensor(
            [[0.1, 0.1, 0.1], [0.2, 0.05, 0.14], [0.05, 0.17, 0.05], [1, 1.1, 1.11]], device=device, dtype=torch.float32
        )
        grid_batch = GridBatch.from_zero_voxels(device=device, voxel_sizes=voxel_sizes, origins=origins)

        self.assertEqual(grid_batch.grid_count, 4)
        self.assertEqual(grid_batch.total_voxels, 0)
        self.assertEqual(grid_batch.device.type, device)
        self.assertEqual(len(grid_batch.voxel_sizes), 4)
        self.assertEqual(len(grid_batch.origins), 4)

        expected_minmaxs = torch.zeros((4, 3), dtype=torch.int32, device=device)

        bboxes = grid_batch.bboxes
        self.assertEqual(bboxes.shape, (4, 2, 3))
        bbox_mins = bboxes[:, 0, :]
        bbox_maxs = bboxes[:, 1, :]
        print(f"bbox_mins: {bbox_mins}")
        print(f"bbox_maxs: {bbox_maxs}")
        self.assertTrue(torch.equal(bbox_mins, expected_minmaxs))
        self.assertTrue(torch.equal(bbox_maxs, expected_minmaxs))

        dual_bboxes = grid_batch.dual_bboxes
        self.assertEqual(dual_bboxes.shape, (4, 2, 3))
        dual_bbox_mins = dual_bboxes[:, 0, :]
        dual_bbox_maxs = dual_bboxes[:, 1, :]
        self.assertTrue(torch.equal(dual_bbox_mins, expected_minmaxs))
        self.assertTrue(torch.equal(dual_bbox_maxs, expected_minmaxs))

    @parameterized.expand(all_device_dtype_combos)
    def test_building_zero_voxels_grids_from_ijk(self, device, dtype):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor([torch.randint(-512, 512, (0, 3)) for i in range(batch_size)]).to(device)
        grid = GridBatch.from_ijk(grid_ijk)

        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0].item(), 0)
        self.assertEqual(grid.joffsets[1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor([torch.randint(-512, 512, (0, 2)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_ijk(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.randint(-512, 512, (0,)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_ijk(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.randint(-512, 512, (5, 0)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_ijk(grid_ijk_bad)

    @parameterized.expand(all_device_dtype_combos)
    def test_building_grid_with_one_zero_voxels_element_in_jagged_tensor(self, device, dtype):
        shapes = [512, 0, 128]
        batch_size = len(shapes)
        grid_ijk = fvdb.JaggedTensor([torch.randint(-512, 512, (shapes[i], 3)) for i in range(batch_size)]).to(device)
        grid = GridBatch.from_ijk(grid_ijk)
        self.assertEqual(len(grid), batch_size)

        off = 0
        for i in range(batch_size):
            self.assertEqual(grid.num_voxels[i], shapes[i])
            self.assertEqual(grid.joffsets[i].item(), off)
            self.assertEqual(grid.joffsets[i + 1].item(), off + shapes[i])
            off += shapes[i]

    @parameterized.expand(all_device_dtype_combos)
    def test_building_zero_voxels_grids_from_points(self, device, dtype):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor([torch.rand(0, 3) for i in range(batch_size)]).to(device)
        grid = GridBatch.from_points(grid_ijk)
        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0].item(), 0)
        self.assertEqual(grid.joffsets[1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(0, 2) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_points(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_points(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(5, 0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_points(grid_ijk_bad)

    @parameterized.expand(all_device_dtype_combos)
    def test_building_zero_voxels_grids_from_nearest_points(self, device, dtype):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor([torch.rand(0, 3) for i in range(batch_size)]).to(device)
        grid = GridBatch.from_nearest_voxels_to_points(grid_ijk)
        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0].item(), 0)
        self.assertEqual(grid.joffsets[1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(0, 2) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_nearest_voxels_to_points(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_nearest_voxels_to_points(grid_ijk_bad)

        grid_ijk_bad = fvdb.JaggedTensor([torch.rand(5, 0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            GridBatch.from_nearest_voxels_to_points(grid_ijk_bad)

    @parameterized.expand(all_device_dtype_combos)
    def test_fvdb_cat(self, device, dtype):
        def _make_random_grid(batch_size):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor([torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(
                device
            )
            return GridBatch.from_ijk(grid_ijk)

        # Test concat batches with 1 grid
        grid1, grid2 = _make_random_grid(1), _make_random_grid(1)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 2)

        # Test concat batches with 1 grid and multiple grids
        grid1, grid2 = _make_random_grid(2), _make_random_grid(1)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 3)

        # Test concat batches with multiple grids
        grid1, grid2 = _make_random_grid(2), _make_random_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 7)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_random_grid(1), _make_random_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2, grid3])
        self.assertTrue(
            torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata]))
        )
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) + len(grid3))
        self.assertEqual(len(grid_cat), 8)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_random_grid(4), _make_random_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2, grid3])
        self.assertTrue(
            torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata]))
        )
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) + len(grid3))
        self.assertEqual(len(grid_cat), 11)

        # Cat to the cat /ᐠ - ˕ -マ < Meow
        grid4, grid5 = _make_random_grid(2), _make_random_grid(5)
        grid_cat2 = fvdb.jcat([grid_cat, grid1, grid4, grid5, grid3])
        self.assertTrue(
            torch.allclose(
                grid_cat2.ijk.jdata,
                torch.cat(
                    [
                        grid1.ijk.jdata,
                        grid2.ijk.jdata,
                        grid3.ijk.jdata,
                        grid1.ijk.jdata,
                        grid4.ijk.jdata,
                        grid5.ijk.jdata,
                        grid3.ijk.jdata,
                    ]
                ),
            )
        )
        self.assertEqual(
            len(grid_cat2), len(grid1) + len(grid2) + len(grid3) + len(grid1) + len(grid4) + len(grid5) + len(grid3)
        )
        self.assertEqual(len(grid_cat2), 25)

    @parameterized.expand(all_device_dtype_combos)
    def test_fvdb_cat_zero_voxels_grid(self, device, dtype):

        def _make_random_grid(batch_size):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor([torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(
                device
            )
            return GridBatch.from_ijk(grid_ijk)

        def _make_zero_voxels_grid(batch_size):
            sizes = [0 for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor([torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(
                device
            )
            return GridBatch.from_ijk(grid_ijk)

        # Test concat batches with 1 grid
        grid1, grid2 = _make_random_grid(1), _make_zero_voxels_grid(1)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 2)

        # Test concat batches with 1 grid and multiple grids
        grid1, grid2 = _make_zero_voxels_grid(2), _make_random_grid(1)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 3)

        # Test concat batches with multiple grids
        grid1, grid2 = _make_random_grid(2), _make_zero_voxels_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 7)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_zero_voxels_grid(1), _make_random_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2, grid3])
        self.assertTrue(
            torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata]))
        )
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) + len(grid3))
        self.assertEqual(len(grid_cat), 8)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_zero_voxels_grid(4), _make_random_grid(5)
        grid_cat = fvdb.jcat([grid1, grid2, grid3])
        self.assertTrue(
            torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata]))
        )
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) + len(grid3))
        self.assertEqual(len(grid_cat), 11)

        # Cat to the cat /ᐠ - ˕ -マ < Meow
        grid4, grid5 = _make_zero_voxels_grid(2), _make_random_grid(5)
        grid_cat2 = fvdb.jcat([grid_cat, grid1, grid4, grid5, grid3])
        self.assertTrue(
            torch.allclose(
                grid_cat2.ijk.jdata,
                torch.cat(
                    [
                        grid1.ijk.jdata,
                        grid2.ijk.jdata,
                        grid3.ijk.jdata,
                        grid1.ijk.jdata,
                        grid4.ijk.jdata,
                        grid5.ijk.jdata,
                        grid3.ijk.jdata,
                    ]
                ),
            )
        )
        self.assertEqual(
            len(grid_cat2), len(grid1) + len(grid2) + len(grid3) + len(grid1) + len(grid4) + len(grid5) + len(grid3)
        )
        self.assertEqual(len(grid_cat2), 25)


if __name__ == "__main__":
    unittest.main()
