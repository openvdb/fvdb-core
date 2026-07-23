# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import pickle
import unittest

import fvdb.nn as fvnn
import numpy as np
import torch
from fvdb.utils.tests import (
    dtype_to_atol,
    expand_tests,
    make_dense_grid_batch_and_jagged_point_data,
    make_grid_batch_and_jagged_point_data,
)
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

bfloat16_combos = [["cuda", torch.bfloat16]]


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)

    @parameterized.expand(["cpu", "cuda"])
    def test_dilate_grid(self, device):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 2
        vxl_size = 0.4
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        d_amt = 2
        dilated_grid_batch = grid_batch.dilated_grid(d_amt)
        expected_ijk = []
        for i in range(batch_size):
            ijk_i = grid_batch.ijk[i].jdata
            if ijk_i.numel() == 0:
                expected_ijk.append(ijk_i)
            else:
                dilated_ijk_i = torch.cat(
                    [
                        ijk_i + torch.tensor([[a, b, c]]).to(ijk_i)
                        for (a, b, c) in itertools.product(range(-d_amt, d_amt + 1), repeat=3)
                    ],
                    dim=0,
                )
                expected_ijk.append(dilated_ijk_i)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)

        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch.voxel_sizes, origins=grid_batch.origins
        )

        self.assertTrue(torch.equal(dilated_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

    @parameterized.expand(["cpu", "cuda"])
    def test_dilate_grid_zero(self, device):
        pts = torch.randn(500, 3, device=device, dtype=torch.float32)
        grid = fvdb.GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=0.3)
        dilated_grid = grid.dilated_grid(0)
        self.assertTrue(torch.equal(grid.ijk.jdata, dilated_grid.ijk.jdata))

    @parameterized.expand(["cpu", "cuda"])
    def test_merge_grids(self, device):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 2
        vxl_size = 0.4
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch1 = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch2 = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        merged_grid_batch = grid_batch1.merged_grid(grid_batch2)
        expected_ijk = []
        for i in range(batch_size):
            ijk1_i = grid_batch1.ijk[i].jdata
            ijk2_i = grid_batch2.ijk[i].jdata
            ijk_union = torch.cat([ijk1_i, ijk2_i])
            expected_ijk.append(ijk_union)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)

        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch1.voxel_sizes, origins=grid_batch1.origins
        )

        self.assertTrue(torch.equal(merged_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

    @parameterized.expand(["cpu", "cuda"])
    def test_merge_empty_grids(self, device):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 2
        vxl_size = 0.4
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch1 = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)
        npc = torch.zeros(size=(batch_size,), dtype=torch.int32, device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch2 = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        # Merge non-empty gridbatch against an empty one
        merged_grid_batch = grid_batch1.merged_grid(grid_batch2)
        self.assertTrue(torch.equal(merged_grid_batch.ijk.jdata, grid_batch1.ijk.jdata))

        # Merge empty gridbatch against a non-empty one
        merged_grid_batch = grid_batch2.merged_grid(grid_batch1)
        self.assertTrue(torch.equal(merged_grid_batch.ijk.jdata, grid_batch1.ijk.jdata))

        # Merge empty gridbatch against a non-empty one
        merged_grid_batch = grid_batch2.merged_grid(grid_batch2)
        self.assertTrue(torch.equal(merged_grid_batch.ijk.jdata, grid_batch2.ijk.jdata))

    @parameterized.expand(["cpu", "cuda"])
    def test_prune_grids(self, device):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 2
        vxl_size = 0.4
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        mask = grid_batch.jagged_like(torch.rand(grid_batch.total_voxels, device=device) > 0.5)

        pruned_grid_batch = grid_batch.pruned_grid(mask)
        expected_ijk = []
        for i in range(batch_size):
            ijk_i = grid_batch.ijk[i].jdata
            ijk_pruned = ijk_i[mask[i].jdata]
            expected_ijk.append(ijk_pruned)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)

        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch.voxel_sizes, origins=grid_batch.origins
        )

        self.assertTrue(torch.equal(pruned_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

    @parameterized.expand(["cpu", "cuda"])
    def test_prune_grids_empty(self, device):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 2
        vxl_size = 0.4
        npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        # Case 1: one tensor is empty
        mask = grid_batch.jagged_like(torch.rand(grid_batch.total_voxels, device=device) > 0.5)
        mask[0].jdata = torch.zeros_like(mask[0].jdata, dtype=torch.bool)
        pruned_grid_batch = grid_batch.pruned_grid(mask)
        expected_ijk = []
        for i in range(batch_size):
            ijk_i = grid_batch.ijk[i].jdata
            ijk_pruned = ijk_i[mask[i].jdata]
            expected_ijk.append(ijk_pruned)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)
        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch.voxel_sizes, origins=grid_batch.origins
        )
        self.assertTrue(torch.equal(pruned_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

        # Case 2: the other tensor is empty
        mask = grid_batch.jagged_like(torch.rand(grid_batch.total_voxels, device=device) > 0.5)
        mask[1].jdata = torch.zeros_like(mask[1].jdata, dtype=torch.bool)
        pruned_grid_batch = grid_batch.pruned_grid(mask)
        expected_ijk = []
        for i in range(batch_size):
            ijk_i = grid_batch.ijk[i].jdata
            ijk_pruned = ijk_i[mask[i].jdata]
            expected_ijk.append(ijk_pruned)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)
        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch.voxel_sizes, origins=grid_batch.origins
        )
        self.assertTrue(torch.equal(pruned_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

        # Case 3: both tensors are empty
        mask = grid_batch.jagged_like(torch.zeros(grid_batch.total_voxels, device=device, dtype=torch.bool))
        pruned_grid_batch = grid_batch.pruned_grid(mask)
        expected_ijk = []
        for i in range(batch_size):
            ijk_i = grid_batch.ijk[i].jdata
            ijk_pruned = ijk_i[mask[i].jdata]
            expected_ijk.append(ijk_pruned)
        expected_ijk = fvdb.JaggedTensor(expected_ijk)
        expected_grid = fvdb.GridBatch.from_ijk(
            expected_ijk, voxel_sizes=grid_batch.voxel_sizes, origins=grid_batch.origins
        )
        self.assertTrue(torch.equal(pruned_grid_batch.ijk.jdata, expected_grid.ijk.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_refine_1x_with_mask(self, device, dtype):
        def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
            batch_size = len(npc)
            plist = []
            for i in range(batch_size):
                ni = npc[i]
                plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
            return plist

        batch_size = 5
        vxl_size = 0.4
        npc = torch.randint(low=0, high=100, size=(batch_size,), device=device).tolist()
        plist = get_point_list(npc, device)
        pc_jagged = fvdb.JaggedTensor(plist)
        grid_batch = fvdb.GridBatch.from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

        random_mask = (
            torch.randn(grid_batch.total_voxels, device=device)
        ) > 0.5  # random mask that selects voxels randomly from different grids
        random_mask = grid_batch.jagged_like(random_mask)
        filtered_grid_batch = grid_batch.refined_grid(1, random_mask)
        sum = 0
        for i in range(batch_size):
            si = grid_batch.joffsets[i]
            ei = grid_batch.joffsets[i + 1]
            ri = random_mask.jdata[si:ei]
            self.assertEqual(ri.sum().item(), filtered_grid_batch.num_voxels_at(i))
            sum += torch.sum(ri)

        self.assertEqual(sum, torch.sum(random_mask.jdata))
        self.assertEqual(torch.sum(random_mask.jdata), filtered_grid_batch.total_voxels)
        print(f"Random mask unbound: {random_mask.unbind()}")
        print(f"Random mask: {random_mask.int().jdata}")
        print(f"Filtered grid batch: {filtered_grid_batch.num_voxels.int()}")

        self.assertTrue(
            torch.all(random_mask.int().jsum().jdata == filtered_grid_batch.num_voxels.int()).item(),
        )

    @parameterized.expand(["cpu", "cuda"])
    def test_is_same(self, device):
        grid = GridBatch.from_dense(3, [16, 16, 16], [0, 0, 0], voxel_sizes=1.0 / 16, origins=[0, 0, 0], device=device)
        self.assertTrue(grid.total_voxels == 3 * 16**3)

        grid2 = GridBatch.from_dense(3, [16, 16, 16], [0, 0, 0], voxel_sizes=1.0 / 16, origins=[0, 0, 0], device=device)
        self.assertFalse(grid.is_same(grid2))
        self.assertNotEqual(grid.address, grid2.address)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxel_neighborhood(self, device, dtype):
        randvox = torch.randint(0, 256, size=(10_000, 3), dtype=torch.int32).to(device)
        randvox = torch.cat(
            [randvox, randvox + torch.ones(1, 3).to(randvox)], dim=0
        )  # Ensure there are always neighbors

        grid = GridBatch.from_ijk(fvdb.JaggedTensor(randvox))

        gt_nhood = torch.zeros((randvox.shape[0], 3, 3, 3), dtype=torch.int32).to(device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    off = torch.tensor([[i - 1, j - 1, k - 1]]).to(randvox)
                    nh_ijk = randvox + off
                    idx = grid.ijk_to_index(fvdb.JaggedTensor(nh_ijk)).jdata
                    mask = grid.coords_in_grid(fvdb.JaggedTensor(nh_ijk)).jdata
                    gt_nhood[:, i, j, k] = torch.where(mask, idx, -torch.ones_like(idx))

        nhood = grid.neighbor_indexes(fvdb.JaggedTensor(randvox), 1, 0).jdata

        self.assertTrue(torch.equal(nhood, gt_nhood))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid = grid.dual_grid()

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata

        self.assertTrue(
            torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)),
            f"max_diff = {torch.abs(pred_dual_coordinates - target_dual_coordinates).max()}",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_primal(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)

        target_primal_coordinates = (pts - vox_origin) / vox_size
        pred_primal_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata

        self.assertTrue(torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        pts.requires_grad = True

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid = grid.dual_grid()

        pred_dual_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        grad_out = torch.rand_like(pred_dual_coordinates)
        pred_dual_coordinates.backward(grad_out)

        assert pts.grad is not None  # Removes type errors with .grad
        pred_grad = pts.grad.clone()

        pts.grad.zero_()
        self.assertFalse(torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(pts.grad, torch.zeros_like(pts.grad)))

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        target_dual_coordinates.backward(grad_out)

        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_primal_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        pts.requires_grad = True

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)

        pred_primal_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        grad_out = torch.rand_like(pred_primal_coordinates)
        pred_primal_coordinates.backward(grad_out)

        assert pts.grad is not None  # Removes type errors with .grad
        pred_grad = pts.grad.clone()

        pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(pts.grad, torch.zeros_like(pts.grad)))

        target_primal_coordinates = (pts - vox_origin) / vox_size
        target_primal_coordinates.backward(grad_out)

        self.assertTrue(torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)))
        # diff_idxs = torch.where(~torch.isclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_primal_to_world(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)

        target_world_pts = (grid_pts * vox_size) + vox_origin
        pred_world_pts = grid.voxel_to_world(fvdb.JaggedTensor(grid_pts)).jdata

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_dual_to_world(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid = grid.dual_grid()

        target_world_pts = ((grid_pts - 0.5) * vox_size) + vox_origin
        pred_world_pts = grid.voxel_to_world(fvdb.JaggedTensor(grid_pts)).jdata

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_primal_to_world_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)
        grid_pts.requires_grad = True

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)

        pred_world_pts = grid.voxel_to_world(fvdb.JaggedTensor(grid_pts)).jdata
        grad_out = torch.rand_like(pred_world_pts)
        pred_world_pts.backward(grad_out)

        assert grid_pts.grad is not None  # Removes type errors with .grad
        pred_grad = grid_pts.grad.clone()

        grid_pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(grid_pts.grad, torch.zeros_like(grid_pts.grad)))

        target_world_pts = (grid_pts * vox_size) + vox_origin
        target_world_pts.backward(grad_out)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(grid_pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_dual_to_world_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)
        grid_pts.requires_grad = True

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid = grid.dual_grid()

        pred_world_pts = grid.voxel_to_world(fvdb.JaggedTensor(grid_pts)).jdata
        grad_out = torch.rand_like(pred_world_pts)
        pred_world_pts.backward(grad_out)

        assert grid_pts.grad is not None  # Removes type errors with .grad
        pred_grad = grid_pts.grad.clone()

        grid_pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(grid_pts.grad, torch.zeros_like(grid_pts.grad)))

        target_world_pts = ((grid_pts - 0.5) * vox_size) + vox_origin
        target_world_pts.backward(grad_out)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(grid_pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_dual_of_dual_is_primal(self, device, dtype):
        torch.random.manual_seed(0)
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(dtype).to(device)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid_d = grid.dual_grid()
        grid_dd = grid_d.dual_grid()

        primal_origin = grid.origins[0]
        dual_origin = grid_d.origins[0]

        self.assertFalse(torch.allclose(primal_origin, dual_origin))
        self.assertTrue(torch.all(primal_origin == grid_dd.origins[0]))
        self.assertTrue(torch.all(dual_origin == grid_dd.dual_grid().origins[0]))

        target_primal_coordinates = (pts - vox_origin) / vox_size
        pred_primal_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata

        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)),
            f"Max diff = {torch.max(torch.abs(target_primal_coordinates- pred_primal_coordinates)).item()}",
        )

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid_d.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))

        pred_primal_coordinates_dd = grid_dd.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates_dd, atol=dtype_to_atol(dtype))
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_index(self, device, dtype):
        gsize = 7

        grid_p, grid_d, _ = make_dense_grid_batch_and_jagged_point_data(gsize, device, dtype)

        pijk = grid_p.ijk.jdata
        dijk = grid_d.ijk.jdata

        for in_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            pijk, dijk = pijk.to(in_dtype), dijk.to(in_dtype)
            pidx = grid_p.ijk_to_index(fvdb.JaggedTensor(pijk)).jdata
            didx = grid_d.ijk_to_index(fvdb.JaggedTensor(dijk)).jdata

            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx))
            self.assertTrue(torch.all(didx == target_didx))

            ppmt = torch.randperm(pidx.shape[0])
            dpmt = torch.randperm(pidx.shape[0])

            pidx = grid_p.ijk_to_index(fvdb.JaggedTensor(pijk[ppmt])).jdata
            didx = grid_d.ijk_to_index(fvdb.JaggedTensor(dijk[dpmt])).jdata
            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx[ppmt]))
            self.assertTrue(torch.all(didx == target_didx[dpmt]))

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_index_batched(self, device, dtype):
        gsize = 7

        grid_p1, grid_d1, _ = make_dense_grid_batch_and_jagged_point_data(gsize, device, dtype)
        grid_p2, grid_d2, _ = make_dense_grid_batch_and_jagged_point_data(gsize - 2, device, dtype)

        grid_p, grid_d = fvdb.GridBatch.from_cat([grid_p1, grid_p2]), fvdb.GridBatch.from_cat([grid_d1, grid_d2])

        pijk = grid_p.ijk
        dijk = grid_d.ijk

        for in_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            pijk, dijk = pijk.to(in_dtype), dijk.to(in_dtype)
            pidx = grid_p.ijk_to_index(grid_p.ijk)
            didx = grid_d.ijk_to_index(grid_d.ijk)

            target_pidx = fvdb.JaggedTensor(
                [torch.arange(n.item()).to(device=pidx.device, dtype=pidx.dtype) for n in grid_p.num_voxels]
            )
            target_didx = fvdb.JaggedTensor(
                [torch.arange(n.item()).to(device=pidx.device, dtype=didx.dtype) for n in grid_d.num_voxels]
            )

            self.assertTrue(torch.all(pidx.jdata == target_pidx.jdata))
            self.assertTrue(torch.all(didx.jdata == target_didx.jdata))

            ppmt = fvdb.JaggedTensor([torch.randperm(pidx_i.rshape[0]).to(pidx.device) for pidx_i in pidx])
            dpmt = fvdb.JaggedTensor([torch.randperm(didx_i.rshape[0]).to(pidx.device) for didx_i in didx])

            pidx = grid_p.ijk_to_index(pijk[ppmt])
            didx = grid_d.ijk_to_index(dijk[dpmt])
            # target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            # target_didx = torch.arange(didx.shape[0]).to(didx)
            target_pidx = fvdb.JaggedTensor(
                [torch.arange(n.item()).to(device=pidx.device, dtype=pidx.dtype) for n in grid_p.num_voxels]
            )
            target_didx = fvdb.JaggedTensor(
                [torch.arange(n.item()).to(device=pidx.device, dtype=didx.dtype) for n in grid_d.num_voxels]
            )

            self.assertTrue(torch.all(pidx.jdata == target_pidx[ppmt].jdata))
            self.assertTrue(torch.all(didx.jdata == target_didx[dpmt].jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_coords_in_grid(self, device, _):
        num_inside = 1000 if device == "cpu" else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = GridBatch.from_ijk(fvdb.JaggedTensor(random_coords))

        enabled_coords = grid.ijk.jdata
        num_outside = 1000 if device == "cpu" else 10_000

        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        pred_mask = grid.coords_in_grid(fvdb.JaggedTensor(all_coords)).jdata
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_points_in_grid(self, device, dtype):
        num_inside = 1000 if device == "cpu" else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = GridBatch.from_ijk(fvdb.JaggedTensor(random_coords))

        enabled_coords = grid.ijk.jdata
        num_outside = 1000 if device == "cpu" else 10_000
        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        all_world_points = grid.voxel_to_world(fvdb.JaggedTensor(all_coords.to(dtype))).jdata

        pred_mask = grid.points_in_grid(fvdb.JaggedTensor(all_world_points)).jdata
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_cubes_intersect_grid(self, device, dtype):
        # TODO: (@Caenorst) tests are a bit too light, should test on more variety of range
        # import random
        torch.random.manual_seed(0)
        # random.seed(0)
        # np.random.seed(0)

        grid, grid_d, p = make_grid_batch_and_jagged_point_data(device, dtype, include_boundary_points=True)
        voxel_size = grid.voxel_size_at(0)

        primal_mask = grid.cubes_in_grid(p).jdata
        dual_mask = grid_d.cubes_in_grid(p, -voxel_size / 2, voxel_size / 2).jdata
        # # Here: Note that dual_mask != primal_mask because their connectivities could be different!
        # #   Instead, we can only ensure that dual_mask is always true where primal_mask is true!
        #
        # from pycg import vis
        # vis.show_3d([vis.wireframe_bbox(grid.grid_to_world(grid.ijk - 0.5),
        #                                 grid.grid_to_world(grid.ijk + 0.5), solid=True),
        #              vis.wireframe_bbox(grid_d.grid_to_world(grid_d.ijk - 0.5),
        #                                 grid_d.grid_to_world(grid_d.ijk + 0.5), ucid=1),
        #              vis.pointcloud(p[primal_mask != dual_mask])])
        #
        self.assertTrue(torch.all(dual_mask[primal_mask]))

        primal_mask = grid.cubes_intersect_grid(p, -voxel_size / 2, voxel_size / 2).jdata
        dual_mask = grid_d.cubes_intersect_grid(p).jdata
        # gt_dual_mask = grid_d.points_in_grid(p).jdata
        self.assertTrue(torch.all(primal_mask == dual_mask))

        # # TODO: (@Caenorst) not sure what we are testing here
        # # We should probably replace that by comparison to pytorch implementation

        # # This is to avoid points on voxel faces, that have ambiguous values
        # # cubes_intersect_grid uses ceil() while sample_trilinear use floor()
        # # TODO: (@Caenorst) this is a bit too strong modification as it doesn't test
        # # points on faces inside the volume (although previous tests are testing it)
        # dual_grid_p = grid_d.world_to_grid(p).jdata
        # p[dual_grid_p == dual_grid_p.int()] += 1e-3
        # # dummy_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        # _ = grid_d.cubes_intersect_grid(p, -0.5, 0.5).jdata

        # # This is to avoid points on voxel faces, that have ambiguous values
        # # cubes_intersect_grid uses ceil() while sample_bezier use floor()
        # # TODO: (@Caenorst) this is a bit too strong modification as it doesn't test
        # # points on faces inside the volume (although previous tests are testing it)
        # grid_p = grid.world_to_grid(p).jdata
        # p[grid_p == grid_p.int()] += 1e-3
        # # dummy_features = torch.rand(grid_d.total_voxels, 4).to(device).to(dtype)
        # _ = grid_d.cubes_intersect_grid(p, -1, 1).jdata

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_refined_grid(self, device, dtype):
        p = torch.randn(100, 3, device=device, dtype=torch.float)
        vox_size = 0.1
        grid = GridBatch.from_points(fvdb.JaggedTensor(p), voxel_sizes=vox_size, origins=(0.0, 0.0, 0.0))
        grid = grid.dilated_grid(1)

        grids = [grid]
        for i in range(2):
            subdiv_factor = i + 2
            mask = torch.rand(grids[i].total_voxels, device=device) > 0.5

            grids.append(grids[-1].refined_grid(subdiv_factor, fvdb.JaggedTensor(mask)))
            self.assertEqual(int(mask.sum().item()) * subdiv_factor**3, grids[-1].total_voxels)

        grids = [grid]
        for i, subdiv_factor in enumerate([(2, 2, 1), (3, 2, 2), (1, 1, 3)]):
            mask = torch.rand(grids[i].total_voxels, device=device) > 0.5

            nsubvox = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
            grids.append(grids[-1].refined_grid(subdiv_factor, fvdb.JaggedTensor(mask)))
            self.assertEqual(int(mask.sum().item()) * nsubvox, grids[-1].total_voxels)
        if device == "cuda":
            torch.cuda.synchronize()

    @parameterized.expand(all_device_dtype_combos)
    def test_build_from_pointcloud_nearest_voxels(self, device, dtype):
        p = torch.randn((100, 3), device=device, dtype=dtype)

        vox_size = 0.01
        grid = GridBatch.from_nearest_voxels_to_points(fvdb.JaggedTensor(p), voxel_sizes=vox_size)

        if p.dtype == torch.half:
            p = p.float()

        expected_ijk = torch.floor(grid.world_to_voxel(fvdb.JaggedTensor(p)).jdata)
        offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=device,
            dtype=torch.long,
        )
        expected_ijk = expected_ijk.unsqueeze(1) + offsets.unsqueeze(0)
        expected_ijk = expected_ijk.view(-1, 3).to(torch.int32)

        expected_ijk_set = set(
            {
                (expected_ijk[i, 0].item(), expected_ijk[i, 1].item(), expected_ijk[i, 2].item())
                for i in range(expected_ijk.shape[0])
            }
        )

        predicted_ijk = grid.ijk.jdata

        predicted_ijk_set = set(
            {
                (predicted_ijk[i, 0].item(), predicted_ijk[i, 1].item(), predicted_ijk[i, 2].item())
                for i in range(predicted_ijk.shape[0])
            }
        )

        self.assertEqual(predicted_ijk_set, expected_ijk_set)

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_refine(self, device, dtype):
        p = torch.randn(100, 3, device=device, dtype=torch.float)
        vox_size = 0.01

        for subdiv_factor in (4, (4, 3, 2)):
            if isinstance(subdiv_factor, tuple):
                nvoxsub = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
                fac_sub_one = torch.tensor([subdiv_factor]).to(device) - 1
                subvec = torch.tensor(subdiv_factor).to(device)
            else:
                nvoxsub = subdiv_factor**3
                fac_sub_one = subdiv_factor - 1
                subvec = subdiv_factor

            vox_size = 0.01
            grid = GridBatch.from_nearest_voxels_to_points(fvdb.JaggedTensor(p), voxel_sizes=vox_size, origins=0)

            feats = torch.randn(grid.total_voxels, 32).to(p)
            feats.requires_grad = True

            mask = torch.ones(grid.total_voxels, dtype=torch.bool).to(device)

            feats_fine, grid_fine = grid.refine(subdiv_factor, fvdb.JaggedTensor(feats), mask=fvdb.JaggedTensor(mask))
            self.assertTrue(torch.allclose(grid_fine.voxel_sizes[0], grid.voxel_sizes[0] / subvec))
            self.assertTrue(
                torch.allclose(grid_fine.origins[0], grid.origins[0] - 0.5 * grid_fine.voxel_sizes[0] * fac_sub_one)
            )

            fine_to_coarse_ijk = (grid_fine.ijk.jdata / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fvdb.JaggedTensor(fine_to_coarse_ijk.to(torch.int32))).jdata

            self.assertTrue(torch.all(feats_fine.jdata == feats[fine_to_coarse_idx]))

            loss = feats_fine.jdata.pow(3).sum()
            loss.backward()

            assert feats.grad is not None  # Removes type errors with .grad
            feats_grad_thru_subdiv = feats.grad.clone()

            feats.grad.zero_()
            self.assertTrue(torch.all(feats.grad == torch.zeros_like(feats.grad)))
            self.assertTrue(not torch.all(feats.grad == feats_grad_thru_subdiv))

            loss = (torch.cat([feats] * (nvoxsub)).pow(3)).sum()
            loss.backward()

            self.assertTrue(torch.all(feats_grad_thru_subdiv == feats.grad))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_refine_with_mask(self, device, dtype):
        p = torch.randn(100, 3, device=device, dtype=torch.float)
        vox_size = 0.01
        subdiv_factor = 4

        for subdiv_factor in (4, (4, 3, 2)):
            if isinstance(subdiv_factor, tuple):
                nvoxsub = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
                fac_sub_one = torch.tensor([subdiv_factor]).to(device) - 1
                subvec = torch.tensor(subdiv_factor).to(device)
            else:
                nvoxsub = subdiv_factor**3
                fac_sub_one = subdiv_factor - 1
                subvec = subdiv_factor

            grid = GridBatch.from_nearest_voxels_to_points(fvdb.JaggedTensor(p), voxel_sizes=vox_size, origins=0)

            feats = torch.randn(grid.total_voxels, 32).to(p)
            feats.requires_grad = True

            mask = torch.rand(grid.total_voxels).to(device) > 0.5

            feats_fine, grid_fine = grid.refine(subdiv_factor, fvdb.JaggedTensor(feats), mask=fvdb.JaggedTensor(mask))
            self.assertTrue(torch.allclose(grid_fine.voxel_sizes[0], grid.voxel_sizes[0] / subvec))
            self.assertTrue(
                torch.allclose(grid_fine.origins[0], grid.origins[0] - 0.5 * grid_fine.voxel_sizes[0] * fac_sub_one)
            )

            fine_to_coarse_ijk = (grid_fine.ijk.jdata / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fvdb.JaggedTensor(fine_to_coarse_ijk.to(torch.int32))).jdata

            self.assertTrue(torch.all(feats_fine.jdata == feats[fine_to_coarse_idx]))

            loss = feats_fine.jdata.pow(3).sum()
            loss.backward()

            assert feats.grad is not None  # Removes type errors with .grad

            feats_grad_thru_subdiv = feats.grad.clone()
            masked_gradients = feats_grad_thru_subdiv[~mask]
            self.assertTrue(torch.all(masked_gradients == torch.zeros_like(masked_gradients)))

            feats.grad.zero_()
            self.assertTrue(torch.all(feats.grad == torch.zeros_like(feats.grad)))
            self.assertTrue(not torch.all(feats.grad == feats_grad_thru_subdiv))

            loss = (torch.cat([feats[mask]] * nvoxsub).pow(3)).sum()
            loss.backward()

            self.assertTrue(torch.all(feats_grad_thru_subdiv == feats.grad))

            masked_gradients = feats.grad[~mask]
            self.assertTrue(torch.all(masked_gradients == torch.zeros_like(masked_gradients)))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_max_pool(self, device, dtype):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch.from_dense(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin, device=device)
        assert grid.total_voxels == 20**3
        grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

        for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
            grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, fvdb.JaggedTensor(grid_vals))
            grid_vals_coarse = grid_vals_coarse.jdata
            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)
                    )
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0],
                        grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device),
                    )
                )

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
            grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device="cpu", dtype=dtype)
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
            grid_vals_t = grid_vals_t.to(device)
            grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device="cpu")
            grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
            for i, coord in enumerate(grid_coarse_ijk_cpu):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
            grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_strided_max_pool(self, device, dtype):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch.from_dense(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin, device=device)
        assert grid.total_voxels == 20**3
        grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

        for pool_factor in ((2, 3, 4), 2, 4, 5, 10):
            # Our behavior differs slightly from PyTorch when pool_factor < stride, so only test this.
            if isinstance(pool_factor, int):
                pools = (pool_factor, pool_factor + 1, pool_factor + 2, pool_factor + 5)
            else:
                assert isinstance(pool_factor, tuple)

                def addit(pf, val_):
                    assert isinstance(pf, tuple)
                    return (pf[0] + val_, pf[1] + val_, pf[2] + val_)

                pools = (pool_factor, addit(pool_factor, 1), addit(pool_factor, 2), addit(pool_factor, 5))
            for stride in pools:
                grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, fvdb.JaggedTensor(grid_vals), stride=stride)
                grid_vals_coarse = grid_vals_coarse.jdata
                if isinstance(stride, int):
                    self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * stride))
                    self.assertTrue(
                        torch.allclose(
                            grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (stride - 1)
                        )
                    )
                else:
                    self.assertTrue(
                        torch.allclose(
                            grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(stride).to(device)
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            grid_coarse.origins[0],
                            grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(stride) - 1).to(device),
                        )
                    )

                # Pytorch pooling
                torch_pool_op = torch.nn.MaxPool3d(pool_factor, stride=stride, ceil_mode=True)
                # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
                grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device="cpu", dtype=dtype)
                grid_ijk_cpu = grid.ijk.jdata.cpu()
                grid_vals_cpu = grid_vals.cpu()
                for i, coord in enumerate(grid_ijk_cpu):
                    grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
                grid_vals_t = grid_vals_t.to(device)
                grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
                grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

                grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device="cpu")
                grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
                for i, coord in enumerate(grid_coarse_ijk_cpu):
                    grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
                grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
                self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_max_pool_grad(self, device, dtype):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch.from_dense(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin, device=device)
        assert grid.total_voxels == 20**3
        for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
            grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
            grid_vals.requires_grad = True

            grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, fvdb.JaggedTensor(grid_vals))
            grid_vals_coarse = grid_vals_coarse.jdata
            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)
                    )
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0],
                        grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device),
                    )
                )

            loss = (grid_vals_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals.grad is not None  # Removes type errors with .grad

            grid_vals_grad = grid_vals.grad.clone()
            self.assertEqual(
                (grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1],
            )

            mask = grid_vals_grad.abs() > 0
            a = torch.sort(torch.tensor([x.item() for x in grid_vals[mask[:, 0]][:, 0]]))[0]
            b = torch.sort(torch.tensor([x.item() for x in grid_vals_coarse[:, 0]]))[0]
            self.assertEqual(torch.max(a - b).max().item(), 0)

            grid_vals.grad.zero_()

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            dense_grid = torch.zeros((gsize, gsize, gsize, 3), dtype=dtype, device=device)
            ijk = grid.ijk.jdata
            dense_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = grid_vals.detach()
            grid_vals_t = dense_grid.permute(3, 0, 1, 2)

            grid_vals_t.requires_grad = True

            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
            for i, coord in enumerate(grid_coarse.ijk.jdata):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

            loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals_t.grad is not None  # Removes type errors with .grad

            grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device="cpu")
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
            grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

            expected_nnz = (
                grid_vals_t_coarse.shape[1]
                * grid_vals_t_coarse.shape[2]
                * grid_vals_t_coarse.shape[3]
                * grid_vals_t_coarse.shape[0]
            )
            self.assertEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(), expected_nnz)

            self.assertEqual(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item(), 0.0)

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_avg_pool_grad(self, device, dtype):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch.from_dense(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin, device=device)
        assert grid.total_voxels == 20**3
        for pool_factor in ((2, 4, 5), 1, 2, 4, 5, 10):
            grid_vals = torch.randn(grid.total_voxels, 3, device=device, dtype=dtype) + 0.5
            grid_vals.requires_grad = True

            grid_vals_coarse, grid_coarse = grid.avg_pool(pool_factor, fvdb.JaggedTensor(grid_vals))
            grid_vals_coarse = grid_vals_coarse.jdata

            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)
                    )
                )
                npool_vox = pool_factor**3
            else:
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origins[0],
                        grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device),
                    )
                )
                npool_vox = pool_factor[0] * pool_factor[1] * pool_factor[2]
            # self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
            # self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))

            loss = (grid_vals_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals.grad is not None  # Removes type errors with .grad

            grid_vals_grad = grid_vals.grad.clone()
            self.assertLessEqual(
                (grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1] * npool_vox,
            )

            grid_vals.grad.zero_()

            # Pytorch pooling
            torch_pool_op = torch.nn.AvgPool3d(pool_factor, pool_factor, ceil_mode=True)
            dense_grid = torch.zeros((gsize, gsize, gsize, 3), dtype=dtype, device=device)
            ijk = grid.ijk.jdata
            dense_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = grid_vals.detach()
            grid_vals_t = dense_grid.permute(3, 0, 1, 2)

            grid_vals_t.requires_grad = True

            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            x, y, z = torch.split(grid_coarse.ijk.jdata, 1, dim=-1)
            grid_vals_coarse_t_flat = grid_vals_t_coarse[:, x.squeeze(), y.squeeze(), z.squeeze()].permute(1, 0)

            diff_idxs = torch.where(
                ~torch.isclose(
                    grid_vals_coarse, grid_vals_coarse_t_flat, atol=dtype_to_atol(dtype), rtol=dtype_to_atol(dtype)
                )
            )
            self.assertTrue(
                torch.allclose(
                    grid_vals_coarse, grid_vals_coarse_t_flat, atol=dtype_to_atol(dtype), rtol=dtype_to_atol(dtype)
                ),
                f"({pool_factor}) Exceed at {diff_idxs}:\n{grid_vals_coarse[diff_idxs][:10]}\nvs\n{grid_vals_coarse_t_flat[diff_idxs][:10]}",
            )

            loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals_t.grad is not None  # Removes type errors with .grad

            grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device="cpu")
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
            grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

            expected_nnz_ub = (
                grid_vals_t_coarse.shape[1]
                * grid_vals_t_coarse.shape[2]
                * grid_vals_t_coarse.shape[3]
                * grid_vals_t_coarse.shape[0]
                * npool_vox
            )
            self.assertLessEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(), expected_nnz_ub)

            self.assertTrue(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item() < dtype_to_atol(dtype))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_strided_max_pool_grad(self, device, dtype):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch.from_dense(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin, device=device)
        assert grid.total_voxels == 20**3
        for pool_factor in (2, 4, 5, 10):
            for stride in (pool_factor, pool_factor + 1, pool_factor + 2, pool_factor + 5):
                grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
                grid_vals.requires_grad = True

                grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, fvdb.JaggedTensor(grid_vals), stride=stride)
                grid_vals_coarse = grid_vals_coarse.jdata
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * stride))
                self.assertTrue(
                    torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (stride - 1))
                )

                loss = (grid_vals_coarse.pow(3) * -1.111).sum()
                loss.backward()

                assert grid_vals.grad is not None  # Removes type errors with .grad

                grid_vals_grad = grid_vals.grad.clone()
                self.assertEqual(
                    (grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                    grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1],
                )

                mask = grid_vals_grad.abs() > 0
                a = torch.sort(torch.tensor([x.item() for x in grid_vals[mask[:, 0]][:, 0]]))[0]
                b = torch.sort(torch.tensor([x.item() for x in grid_vals_coarse[:, 0]]))[0]
                self.assertEqual(torch.max(a - b).max().item(), 0)

                grid_vals.grad.zero_()

                # Pytorch pooling
                torch_pool_op = torch.nn.MaxPool3d(pool_factor, stride=stride, ceil_mode=True)
                dense_grid = torch.zeros((gsize, gsize, gsize, 3), dtype=dtype, device=device)
                ijk = grid.ijk.jdata
                dense_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = grid_vals.detach()
                grid_vals_t = dense_grid.permute(3, 0, 1, 2)

                grid_vals_t.requires_grad = True

                grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

                grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
                for i, coord in enumerate(grid_coarse.ijk.jdata):
                    grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

                self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

                loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
                loss.backward()

                assert grid_vals_t.grad is not None  # Removes type errors with .grad

                grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device="cpu")
                grid_ijk_cpu = grid.ijk.jdata.cpu()
                grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
                for i, coord in enumerate(grid_ijk_cpu):
                    grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
                grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

                expected_nnz = (
                    grid_vals_t_coarse.shape[1]
                    * grid_vals_t_coarse.shape[2]
                    * grid_vals_t_coarse.shape[3]
                    * grid_vals_t_coarse.shape[0]
                )
                self.assertEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(), expected_nnz)

                self.assertEqual(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item(), 0.0)

    @parameterized.expand(all_device_dtype_combos)
    def test_pickle(self, device, dtype):
        grid, _, _ = make_grid_batch_and_jagged_point_data(device, dtype)
        pkl_str = pickle.dumps(grid)
        grid_2 = pickle.loads(pkl_str)
        self.assertTrue(torch.all(grid.ijk.jdata == grid_2.ijk.jdata))
        self.assertEqual(grid.device, grid_2.device)
        self.assertTrue(torch.all(grid.voxel_sizes[0] == grid_2.voxel_sizes[0]))
        self.assertTrue(torch.all(grid.origins[0] == grid_2.origins[0]))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_device(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)
        grid = grid.dual_grid()

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))
        self.assertEqual(grid.device.type, torch.device(device).type)

        to_device = torch.device("cpu")
        grid2 = grid.to(to_device)
        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        if torch.device(device).type != to_device.type:
            with self.assertRaises(Exception):
                pred_dual_coordinates = grid2.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        pred_dual_coordinates = grid2.world_to_voxel(fvdb.JaggedTensor(pts.to(to_device))).jdata
        self.assertTrue(
            torch.allclose(pred_dual_coordinates, target_dual_coordinates.to(to_device), atol=dtype_to_atol(dtype))
        )
        self.assertEqual(grid2.device, to_device)

        to_device = torch.device("cuda:0")
        grid2 = grid.to(to_device)
        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        if torch.device(device).type != to_device.type:
            with self.assertRaises(Exception):
                pred_dual_coordinates = grid2.world_to_voxel(fvdb.JaggedTensor(pts)).jdata
        pred_dual_coordinates = grid2.world_to_voxel(fvdb.JaggedTensor(pts.to(to_device))).jdata
        self.assertTrue(
            torch.allclose(pred_dual_coordinates, target_dual_coordinates.to(to_device), atol=dtype_to_atol(dtype))
        )
        self.assertEqual(grid2.device, to_device)

    @parameterized.expand(all_device_dtype_combos)
    def test_grid_construction(self, device, dtype):
        rand_ijk = torch.randint(-100, 100, (1000, 3), device=device)
        rand_pts = torch.randn(1000, 3, device=device, dtype=dtype)

        def build_from_ijk(vsize, vorigin):
            grid = GridBatch.from_ijk(fvdb.JaggedTensor(rand_ijk), vsize, vorigin)
            return grid

        def build_from_pts(vsize, vorigin):
            grid = GridBatch.from_points(fvdb.JaggedTensor(rand_pts), voxel_sizes=vsize, origins=vorigin)
            return grid

        def build_from_pts_nn(vsize, vorigin):
            grid = GridBatch.from_nearest_voxels_to_points(
                fvdb.JaggedTensor(rand_pts), voxel_sizes=vsize, origins=vorigin
            )
            return grid

        def build_from_dense(vsize, vorigin):
            grid = GridBatch.from_dense(1, [10, 10, 10], [0, 0, 0], vsize, vorigin, device=device)
            return grid

        vox_size = np.random.rand(3) * 0.2 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid = GridBatch.from_points(fvdb.JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        grid = grid.dilated_grid(1)

        for builder in [build_from_ijk, build_from_pts, build_from_pts_nn, build_from_dense]:
            with self.assertRaises(ValueError):
                grid = builder(-vox_size, [0.01] * 3)

            with self.assertRaises(ValueError):
                grid = builder(-1.0, [0.01] * 3)

            with self.assertRaises(ValueError):
                grid = builder(vox_size * 0.0, [0.01] * 3)

            with self.assertRaises(ValueError):
                grid = builder(0.0, [0.01] * 3)

            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01] * 4)

            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01] * 2)

    @parameterized.expand(all_device_dtype_combos)
    def test_inverting_grid_indices(self, device, dtype):
        vox_size = 0.1

        # Unique IJK since for duplicates the permutation is non-bijective
        ijk = list(set([tuple([a for a in (np.random.randn(3) / vox_size).astype(np.int32)]) for _ in range(10000)]))
        ijk = torch.from_numpy(np.array([list(a) for a in ijk])).to(torch.int32).to(device)

        grid = GridBatch.from_ijk(fvdb.JaggedTensor(ijk), voxel_sizes=vox_size, origins=[0.0] * 3)

        inv_index = grid.inject_from_ijk(
            grid.jagged_like(ijk), grid.jagged_like(torch.arange(len(ijk), device=grid.device)), default_value=-1
        ).jdata

        target_inv_index = torch.full_like(grid.ijk.jdata[:, 0], -1)
        idx = grid.ijk_to_index(fvdb.JaggedTensor(ijk)).jdata
        for i in range(ijk.shape[0]):
            target_inv_index[idx[i]] = i

        self.assertTrue(torch.all(inv_index == target_inv_index))

        # Test functionality where size of inverse index's argument != len(grid.ijk)
        # Pick random ijk subset
        rand_ijks = []
        for i in range(grid.grid_count):
            ijks = grid.ijk.jdata[grid.ijk.jidx == i]
            rand_ijks.append(torch.unique(ijks[torch.randint(len(ijks), (50,), device=ijks.device)], dim=0))

        rand_ijks = fvdb.JaggedTensor(rand_ijks)

        # valid ijk indices
        rand_ijk_inv_indices = grid.inject_from_ijk(
            rand_ijks, rand_ijks.jagged_like(torch.arange(len(rand_ijks.jdata))), default_value=-1
        )
        inv_rand_ijk = grid.ijk.jdata[rand_ijk_inv_indices.jdata != -1]
        assert len(inv_rand_ijk) == len(rand_ijks.jdata)
        inv_rand_ijk = rand_ijks.jagged_like(inv_rand_ijk)

        def check_order(t1: torch.Tensor, t2: torch.Tensor):
            t1_list = t1.tolist()
            t2_list = t2.tolist()

            last_index = -1
            for elem in t2_list:
                try:
                    current_index = t1_list.index(elem)
                    # Check if the current index is greater than the last index
                    if current_index > last_index:
                        last_index = current_index
                    else:
                        return False
                except Exception:
                    return False
            return True

        for i, (inv_ijks, ijks) in enumerate(zip(inv_rand_ijk, rand_ijks)):
            # ensure output of inverse_index is a permutation of the input
            inv_ijks_sorted, _ = torch.sort(inv_ijks.jdata, dim=0)
            ijks_sorted, _ = torch.sort(ijks.jdata, dim=0)
            assert torch.equal(inv_ijks_sorted, ijks_sorted)

            # ensure output of inverse_index appears in ascending order in ijks
            assert check_order(grid.ijk.jdata[grid.ijk.jidx == i], inv_ijks.jdata)

    @parameterized.expand(all_device_dtype_combos)
    def test_inverting_grid_indices_batched(self, device, dtype):
        vox_size = 0.1

        # Unique IJK since for duplicates the permutation is non-bijective
        ijk = [
            list(
                set(
                    [
                        tuple([a for a in (np.random.randn(3) / vox_size).astype(np.int32)])
                        for _ in range(100 + np.random.randint(10))
                    ]
                )
            )
            for _ in range(3)
        ]
        ijk = [torch.from_numpy(np.array([list(a) for a in ijk_i])).to(torch.int32).to(device) for ijk_i in ijk]
        ijk = fvdb.JaggedTensor(ijk)

        grid = GridBatch.from_ijk(ijk, voxel_sizes=vox_size, origins=[0.0] * 3)

        unsorted_indices = fvdb.JaggedTensor([torch.arange(x.shape[0], device=grid.device) for x in ijk.unbind()])  # type: ignore
        inv_index = grid.inject_from_ijk(ijk, unsorted_indices, default_value=-1).jdata

        target_inv_index = torch.full_like(grid.ijk.jdata[:, 0], -1)
        idx = grid.ijk_to_index(ijk, cumulative=True)

        for i, ijk_i in enumerate(ijk):
            for j in range(ijk_i.rshape[0]):
                target_inv_index[idx[i].jdata[j]] = j

        self.assertTrue(torch.all(inv_index == target_inv_index))

        # Test functionality where size of inv_index's argument != len(grid.ijk)
        # Pick random ijk subset
        rand_ijks = []
        for i in range(grid.grid_count):
            ijks = grid[i].ijk.jdata
            rand_ijks.append(torch.unique(ijks[torch.randint(len(ijks), (50,), device=ijks.device)], dim=0))

        rand_ijks = fvdb.JaggedTensor(rand_ijks)
        unsorted_indices = fvdb.JaggedTensor([torch.arange(x.shape[0], device=grid.device) for x in rand_ijks.unbind()])  # type: ignore
        rand_ijk_inv_indices = grid.inject_from_ijk(rand_ijks, unsorted_indices, default_value=-1).jdata

        # # valid ijk indices
        inv_rand_ijk = grid.ijk[rand_ijk_inv_indices != -1]
        assert len(inv_rand_ijk.jdata) == len(rand_ijks.jdata)

        def check_order(t1: torch.Tensor, t2: torch.Tensor):
            t1_list = t1.tolist()
            t2_list = t2.tolist()

            last_index = -1
            for elem in t2_list:
                try:
                    current_index = t1_list.index(elem)
                    # Check if the current index is greater than the last index
                    if current_index > last_index:
                        last_index = current_index
                    else:
                        return False
                except ValueError:
                    return False
            return True

        for i, (inv_ijks, ijks) in enumerate(zip(inv_rand_ijk, rand_ijks)):
            # ensure output of inverse_index is a permutation of the input
            inv_ijks_sorted, _ = torch.sort(inv_ijks.jdata, dim=0)
            ijks_sorted, _ = torch.sort(ijks.jdata, dim=0)
            assert torch.equal(inv_ijks_sorted, ijks_sorted)

            # ensure output of inverse_index appears in ascending order in ijks
            assert check_order(grid.ijk.jdata[grid.ijk.jidx == i], inv_ijks.jdata)

    @parameterized.expand(all_device_dtype_combos)
    def test_invert_grid_indices_batched_noncumulative(self, device, dtype):
        batch_size = 5

        ijks = [torch.randint(-10, 10, (int(torch.randint(1_000, 10_000, (1,))), 3)) for i in range(batch_size)]
        ijks = fvdb.JaggedTensor([i.to(device) for i in ijks])
        unsorted_idxs = fvdb.JaggedTensor([torch.arange(i.shape[0], device=device) for i in ijks.unbind()])  # type: ignore
        gridbatch = fvdb.GridBatch.from_ijk(ijks)

        inv_idx = gridbatch.inject_from_ijk(ijks, unsorted_idxs, default_value=-1)

        assert torch.equal(ijks[inv_idx].jdata, gridbatch.ijk.jdata)

        # Test whether when the ijks > grid.ijks, and duplicate entries appear in the ijks, result is still valid
        ijks_unbound = ijks.unbind()
        assert all(isinstance(i, torch.Tensor) for i in ijks_unbound)
        double_ijks = fvdb.JaggedTensor([torch.cat([i, i]) for i in ijks_unbound])  # type: ignore
        unsorted_idxs = fvdb.JaggedTensor([torch.arange(i.shape[0], device=device) for i in double_ijks.unbind()])  # type: ignore
        d_inv_idx = gridbatch.inject_from_ijk(double_ijks, unsorted_idxs, default_value=-1)
        assert torch.equal(double_ijks[d_inv_idx].jdata, gridbatch.ijk.jdata)

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_index_batched_noncumulative(self, device, dtype):
        batch_size = 5

        ijks = [torch.randint(-10, 10, (int(torch.randint(1_000, 10_000, (1,))), 3)) for i in range(batch_size)]
        ijks = fvdb.JaggedTensor([i.to(device) for i in ijks])

        gridbatch = fvdb.GridBatch.from_ijk(ijks)

        idx = gridbatch.ijk_to_index(ijks, cumulative=False)

        assert torch.equal(ijks.jdata, gridbatch.ijk[idx].jdata)

        # Test whether when the ijks > grid.ijks, and duplicate entries appear in the ijks, result is still valid
        ijks_unbound = ijks.unbind()
        assert all(isinstance(i, torch.Tensor) for i in ijks_unbound)
        double_ijks = fvdb.JaggedTensor([torch.cat([i, i]) for i in ijks_unbound])  # type: ignore
        d_idx = gridbatch.ijk_to_index(double_ijks, cumulative=False)
        assert torch.equal(double_ijks.jdata, gridbatch.ijk[d_idx].jdata)

    @parameterized.expand(all_device_dtype_combos)
    def test_no_use_after_free_on_backward(self, device, dtype):

        grid, grid_d, p = make_grid_batch_and_jagged_point_data(device, dtype)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_trilinear(p, fvdb.JaggedTensor(primal_features)).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        del grid, grid_d
        fv.backward(grad_out)

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection(self, device, dtype):
        # Generate the SDF for a sphere on a grid
        N = 32
        sphere_rad = 0.35
        (
            ii,
            jj,
            kk,
        ) = torch.meshgrid([torch.arange(N)] * 3, indexing="ij")
        xx, yy, zz = (
            ii.float() / (float(N) - 1) - 0.5,
            jj.float() / (float(N) - 1) - 0.5,
            kk.float() / (float(N) - 1) - 0.5,
        )
        sphere_sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad

        # Generate a bunch of points on the sphere which we'll send rays to
        cam_o = torch.tensor([0.0, 0.0, -2.0]).unsqueeze(0).repeat(100, 1)
        cam_targets = torch.randn(100, 3)
        cam_targets /= torch.norm(cam_targets, dim=-1, keepdim=True)
        cam_targets *= sphere_rad
        cam_targets += 0.5 - 0.5 / N
        cam_d = cam_targets - cam_o
        cam_d /= torch.norm(cam_d, dim=-1, keepdim=True)

        sphere_sdf, cam_o, cam_d = sphere_sdf.to(device), cam_o.to(device), cam_d.to(device)
        sphere_sdf, cam_o, cam_d = sphere_sdf.to(dtype), cam_o.to(dtype), cam_d.to(dtype)

        # Build a grid with the SDF
        grid = GridBatch.from_dense(
            1, [sphere_sdf.shape[i] for i in range(3)], [0] * 3, voxel_sizes=1.0 / N, origins=[0] * 3, device=device
        )
        sdf_p = grid.inject_from_dense_cminor(
            sphere_sdf.unsqueeze(-1).unsqueeze(0)
        ).jdata.squeeze()  # permuted sdf values

        # Intersect rays with the SDF
        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(cam_o), fvdb.JaggedTensor(cam_d), fvdb.JaggedTensor(sdf_p.squeeze())
        ).jdata
        hit_mask = isect >= 0.0
        self.assertTrue(hit_mask.sum().item() > 25)
        hit_pts = cam_o[hit_mask] + isect[hit_mask, None] * cam_d[hit_mask]

        # Sample intersected values and make sure they're within a half voxel from the true intersection
        sdf_samp = grid.sample_trilinear(fvdb.JaggedTensor(hit_pts), fvdb.JaggedTensor(sdf_p.unsqueeze(1))).jdata
        self.assertLess(sdf_samp.max().item(), 0.5 * torch.norm(grid.voxel_sizes[0]).item())

        # import polyscope as ps
        # ps.init()
        # xyz = grid.grid_to_world(grid.ijk.to(dtype))
        # pc = ps.register_point_cloud("sdf_p", xyz.cpu().numpy())
        # pc.add_scalar_quantity("sdf", sdf_p.cpu().numpy(), enabled=True)
        # pc.add_scalar_quantity("occ", sdf_p.cpu().numpy() <= 0.0, enabled=True)
        # ps.register_point_cloud("cam_target", cam_targets.numpy())
        # ps.register_point_cloud("cam_o", cam_o.cpu().numpy())
        # ps.register_point_cloud("hits", hit_pts.cpu().numpy())
        # ps.show()

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_starts_inside_surface(self, device, dtype):
        # Regression test: a ray whose origin is INSIDE the surface should
        # report a hit at the EXIT crossing (where SDF flips back from
        # negative to positive), not -1 and not the bracket-entry of the
        # very first active voxel. This matches nanovdb::ZeroCrossing
        # semantics, which seeds the sign reference from the first valid
        # voxel and reports the first subsequent sign flip.
        N = 32
        sphere_rad = 0.35
        ii, jj, kk = torch.meshgrid([torch.arange(N)] * 3, indexing="ij")
        xx, yy, zz = (
            ii.float() / (float(N) - 1) - 0.5,
            jj.float() / (float(N) - 1) - 0.5,
            kk.float() / (float(N) - 1) - 0.5,
        )
        sphere_sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad

        # Origin sits at the sphere center (in world coords this is
        # (0.5, 0.5, 0.5) since voxel_sizes=1/N and origins=[0,0,0]) so the
        # ray starts well inside the surface.
        origin_world = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        # Six axis-aligned rays leaving the center should each hit the
        # sphere surface at distance ~ sphere_rad.
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        ray_o = origin_world.unsqueeze(0).repeat(directions.shape[0], 1)
        ray_d = directions

        sphere_sdf = sphere_sdf.to(device).to(dtype)
        ray_o = ray_o.to(device).to(dtype)
        ray_d = ray_d.to(device).to(dtype)

        grid = GridBatch.from_dense(
            1, [sphere_sdf.shape[i] for i in range(3)], [0] * 3, voxel_sizes=1.0 / N, origins=[0] * 3, device=device
        )
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(ray_d), fvdb.JaggedTensor(sdf_p.squeeze())
        ).jdata

        # Every ray must report a hit (no -1).
        self.assertTrue(torch.all(isect >= 0).item(), f"unexpected misses, isect={isect}")

        # The hit point should sit on the sphere surface within a voxel of
        # tolerance — bracket-entry precision is the worst case here.
        hit_pts = ray_o + isect.unsqueeze(-1) * ray_d
        sdf_samp = grid.sample_trilinear(fvdb.JaggedTensor(hit_pts), fvdb.JaggedTensor(sdf_p.unsqueeze(1))).jdata
        voxel_diag = torch.norm(grid.voxel_sizes[0]).item()
        self.assertLess(abs(sdf_samp).max().item(), voxel_diag)

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_two_disjoint_regions(self, device, dtype):
        # Regression test: when a ray crosses two disjoint narrow-band
        # regions (here two separate spheres along +x), the reported hit
        # must lie on the surface of the FIRST region. Before band
        # continuity was tracked the algorithm could linearly interpolate
        # across the empty gap between bands and produce a hit time inside
        # that gap (i.e. nowhere near either surface).
        N = 64
        sphere_rad = 0.07
        ii, jj, kk = torch.meshgrid([torch.arange(N)] * 3, indexing="ij")
        xx, yy, zz = (
            ii.float() / (float(N) - 1) - 0.5,
            jj.float() / (float(N) - 1) - 0.5,
            kk.float() / (float(N) - 1) - 0.5,
        )
        # Two spheres, centered at (-0.25, 0, 0) and (+0.25, 0, 0) in
        # normalized [-0.5, 0.5] coords (so world coords (0.25, 0.5, 0.5)
        # and (0.75, 0.5, 0.5)). SDF is the min of the two per-sphere SDFs.
        sdf_a = torch.sqrt((xx + 0.25) ** 2 + yy**2 + zz**2) - sphere_rad
        sdf_b = torch.sqrt((xx - 0.25) ** 2 + yy**2 + zz**2) - sphere_rad
        scene_sdf = torch.minimum(sdf_a, sdf_b)

        # Ray starts at world x = -1 (well outside the bbox), aimed in +x.
        # World y, z = 0.5 to pass through both sphere centers.
        ray_o = torch.tensor([[-1.0, 0.5, 0.5]], dtype=torch.float32)
        ray_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        scene_sdf = scene_sdf.to(device).to(dtype)
        ray_o = ray_o.to(device).to(dtype)
        ray_d = ray_d.to(device).to(dtype)

        grid = GridBatch.from_dense(
            1, [scene_sdf.shape[i] for i in range(3)], [0] * 3, voxel_sizes=1.0 / N, origins=[0] * 3, device=device
        )
        sdf_p = grid.inject_from_dense_cminor(scene_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(ray_d), fvdb.JaggedTensor(sdf_p.squeeze())
        ).jdata

        self.assertTrue(torch.all(isect >= 0).item(), f"unexpected miss, isect={isect}")

        # Expected first-surface entry x-coord (world) ~ 0.25 - sphere_rad =
        # 0.18. Hit time along this ray equals the world-space distance
        # because |ray_d| == 1. The reported t must therefore be near
        # ray_o.x + 0.25 - sphere_rad - ray_o.x = 1 + 0.25 - 0.07 = 1.18,
        # i.e. far away from the gap region between the two spheres
        # (around x=0.5, t=1.5). One voxel of slack covers bracket-entry
        # precision.
        expected_t = (0.25 - sphere_rad) - (-1.0)
        gap_t = 0.5 - (-1.0)
        voxel_size = 1.0 / N
        self.assertLess(abs(isect.item() - expected_t), 2.0 * voxel_size)
        self.assertGreater(abs(isect.item() - gap_t), 5.0 * voxel_size)

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_sign_of_zero(self, device, dtype):
        # Regression test (cf. nanovdb HDDA's "+-0" issue exercised in
        # TestNanoVDB.cc:1520-1552 and the paired-direction cases in
        # OpenVDB's TestLevelSetRayIntersector.cc:71-215). A ray with
        # direction (1, +0, +0) must produce the exact same hit time as
        # one with (1, -0, -0) (and likewise for y, z and negative
        # directions). This guards against sign-of-zero divergence in
        # the HDDA initialisation, where an unhandled `-0` axis would
        # produce a different stepping schedule from `+0`.
        N = 32
        sphere_rad = 0.3
        voxel_size = 1.0 / N
        sphere_center = torch.tensor([0.5, 0.5, 0.5])

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii / N - sphere_center[0]
        yy = jj / N - sphere_center[1]
        zz = kk / N - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        # For each axis-aligned principal direction, the two listed
        # directions differ ONLY by the sign of the zeroed components.
        paired_dirs = [
            ([1.0, 0.0, 0.0], [1.0, -0.0, -0.0]),
            ([-1.0, 0.0, 0.0], [-1.0, -0.0, -0.0]),
            ([0.0, 1.0, 0.0], [-0.0, 1.0, -0.0]),
            ([0.0, -1.0, 0.0], [-0.0, -1.0, -0.0]),
            ([0.0, 0.0, 1.0], [-0.0, -0.0, 1.0]),
            ([0.0, 0.0, -1.0], [-0.0, -0.0, -1.0]),
        ]

        # Origin at the sphere centre so every axis-aligned ray must hit.
        ray_o = sphere_center.unsqueeze(0).to(device).to(dtype)

        for d_pos, d_neg in paired_dirs:
            d_pos_t = torch.tensor([d_pos], dtype=torch.float32).to(device).to(dtype)
            d_neg_t = torch.tensor([d_neg], dtype=torch.float32).to(device).to(dtype)

            isect_pos = grid.ray_implicit_intersection(
                fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(d_pos_t), fvdb.JaggedTensor(sdf_p)
            ).jdata
            isect_neg = grid.ray_implicit_intersection(
                fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(d_neg_t), fvdb.JaggedTensor(sdf_p)
            ).jdata

            self.assertTrue((isect_pos >= 0).all().item(), f"+0 dir miss: {d_pos}")
            self.assertTrue((isect_neg >= 0).all().item(), f"-0 dir miss: {d_neg}")
            self.assertTrue(
                torch.equal(isect_pos, isect_neg),
                f"sign-of-zero divergence: dir {d_pos} vs {d_neg}, "
                f"isect {isect_pos.tolist()} vs {isect_neg.tolist()}",
            )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_axis_aligned_analytic(self, device, dtype):
        # Adapted from TestLevelSetRayIntersector.cc:43-247: axis-aligned
        # rays through the sphere centre should report a hit time within
        # a voxel diagonal of the analytic ray-sphere intersection root.
        # All six principal directions are exercised, which also catches
        # negative-axis HDDA back-stepping bugs (the "Jan@GitHub" case
        # in TestVolumeRayIntersector.cc:213-232).
        N = 64 if device == "cuda" else 32
        sphere_rad = 0.3
        voxel_size = 1.0 / N
        sphere_center = torch.tensor([0.5, 0.5, 0.5])

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii / N - sphere_center[0]
        yy = jj / N - sphere_center[1]
        zz = kk / N - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        # Ray origin sits 0.5 world units outside the bbox along the
        # ±axis, aimed at the sphere centre. Analytic hit time is the
        # distance from origin to the nearest sphere surface point.
        cases = [
            (torch.tensor([-0.5, 0.5, 0.5]), torch.tensor([1.0, 0.0, 0.0])),
            (torch.tensor([1.5, 0.5, 0.5]), torch.tensor([-1.0, 0.0, 0.0])),
            (torch.tensor([0.5, -0.5, 0.5]), torch.tensor([0.0, 1.0, 0.0])),
            (torch.tensor([0.5, 1.5, 0.5]), torch.tensor([0.0, -1.0, 0.0])),
            (torch.tensor([0.5, 0.5, -0.5]), torch.tensor([0.0, 0.0, 1.0])),
            (torch.tensor([0.5, 0.5, 1.5]), torch.tensor([0.0, 0.0, -1.0])),
        ]
        expected_t = 1.0 - sphere_rad
        voxel_diag = (3.0**0.5) * voxel_size

        for ray_o, ray_d in cases:
            ray_o_t = ray_o.unsqueeze(0).to(device).to(dtype)
            ray_d_t = ray_d.unsqueeze(0).to(device).to(dtype)
            isect = grid.ray_implicit_intersection(
                fvdb.JaggedTensor(ray_o_t), fvdb.JaggedTensor(ray_d_t), fvdb.JaggedTensor(sdf_p)
            ).jdata
            self.assertTrue(
                (isect >= 0).all().item(),
                f"miss: o={ray_o.tolist()} d={ray_d.tolist()}",
            )
            err = abs(isect.item() - expected_t)
            self.assertLess(
                err,
                voxel_diag,
                f"hit-time {isect.item()} expected {expected_t} (err {err}), voxel_diag {voxel_diag} "
                f"for o={ray_o.tolist()} d={ray_d.tolist()}",
            )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_diagonal_analytic(self, device, dtype):
        # Adapted from TestLevelSetRayIntersector.cc:249-278: a diagonal
        # ray hitting the sphere surface should report a hit time within
        # a voxel diagonal of the analytic root. Unlike the axis-aligned
        # cases this exercises HDDA stepping along all three axes
        # simultaneously.
        N = 64 if device == "cuda" else 32
        sphere_rad = 0.3
        voxel_size = 1.0 / N
        sphere_center = torch.tensor([0.5, 0.5, 0.5])

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii / N - sphere_center[0]
        yy = jj / N - sphere_center[1]
        zz = kk / N - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        # Ray from a corner of the bbox aimed at the sphere centre.
        ray_o_world = torch.tensor([-0.2, -0.2, -0.2])
        diff = sphere_center - ray_o_world
        ray_d_world = diff / diff.norm()  # unit-length toward centre
        analytic_t0 = diff.norm().item() - sphere_rad

        ray_o_t = ray_o_world.unsqueeze(0).to(device).to(dtype)
        ray_d_t = ray_d_world.unsqueeze(0).to(device).to(dtype)
        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o_t), fvdb.JaggedTensor(ray_d_t), fvdb.JaggedTensor(sdf_p)
        ).jdata

        voxel_diag = (3.0**0.5) * voxel_size
        self.assertTrue((isect >= 0).all().item(), f"miss: o={ray_o_world.tolist()} d={ray_d_world.tolist()}")
        err = abs(isect.item() - analytic_t0)
        self.assertLess(
            err,
            voxel_diag,
            f"hit-time {isect.item()} expected {analytic_t0} (err {err}), voxel_diag {voxel_diag}",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_explicit_misses(self, device, dtype):
        # Adapted from TestLevelSetRayIntersector.cc:311-389
        # (testMissedIntersections): rays that miss the surface must
        # return -1, both for rays that completely miss the grid bbox
        # AND for rays that traverse the bbox but never cross the zero
        # level set. The existing fvdb ray_implicit_intersection tests
        # only assert positive hits; the miss path was unguarded.
        N = 32
        sphere_rad = 0.3
        voxel_size = 1.0 / N
        sphere_center = torch.tensor([0.5, 0.5, 0.5])

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii / N - sphere_center[0]
        yy = jj / N - sphere_center[1]
        zz = kk / N - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        miss_cases = [
            # Ray bypasses the bbox entirely (y stays at 5.0, well outside [0, 1]).
            (torch.tensor([-0.5, 5.0, 0.5]), torch.tensor([1.0, 0.0, 0.0])),
            # Ray clips a corner of the bbox at (y=0.05, z=0.05); along
            # this line the SDF stays well above zero (min distance to
            # sphere centre ~0.636, so SDF >= 0.336 throughout).
            (torch.tensor([-0.5, 0.05, 0.05]), torch.tensor([1.0, 0.0, 0.0])),
            # Ray points away from the bbox and never enters it.
            (torch.tensor([-0.5, -0.5, -0.5]), torch.tensor([0.0, 0.0, -1.0])),
            # Ray traverses the bbox along a face-aligned path that
            # grazes outside the sphere on +y.
            (torch.tensor([-0.5, 0.95, 0.5]), torch.tensor([1.0, 0.0, 0.0])),
        ]
        for ray_o, ray_d in miss_cases:
            ray_o_t = ray_o.unsqueeze(0).to(device).to(dtype)
            ray_d_t = ray_d.unsqueeze(0).to(device).to(dtype)
            isect = grid.ray_implicit_intersection(
                fvdb.JaggedTensor(ray_o_t), fvdb.JaggedTensor(ray_d_t), fvdb.JaggedTensor(sdf_p)
            ).jdata
            self.assertEqual(
                isect.item(),
                -1.0,
                f"expected miss (-1), got {isect.item()} for o={ray_o.tolist()} d={ray_d.tolist()}",
            )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_non_trivial_transform(self, device, dtype):
        # Adapted from TestLevelSetRayIntersector.cc:99-216 (which uses
        # `s = 0.5, 1.5, 1.0` voxel sizes and sphere centres at
        # (20, 0, 0), (0, 20, 0), (0, 0, 20)). Every existing fvdb
        # ray_implicit_intersection test runs with `voxel_sizes=1/N,
        # origins=[0,0,0]`, leaving `transform.applyToRay` (in
        # RayImplicitIntersection.cu) unexercised for non-identity
        # transforms.
        N = 32
        voxel_size = 0.25
        grid_origin = (10.0, 20.0, 30.0)
        sphere_rad = 2.0

        # Sphere centred in world space at the centre of the grid bbox.
        sphere_center = torch.tensor(
            [
                grid_origin[0] + N * voxel_size / 2,
                grid_origin[1] + N * voxel_size / 2,
                grid_origin[2] + N * voxel_size / 2,
            ]
        )

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii * voxel_size + grid_origin[0] - sphere_center[0]
        yy = jj * voxel_size + grid_origin[1] - sphere_center[1]
        zz = kk * voxel_size + grid_origin[2] - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(
            1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=list(grid_origin), device=device
        )
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        # Ray fired along +x toward the sphere centre, originating
        # 1 world unit before the bbox.
        ray_o_world = torch.tensor([grid_origin[0] - 1.0, sphere_center[1].item(), sphere_center[2].item()])
        ray_d_world = torch.tensor([1.0, 0.0, 0.0])
        ray_o_t = ray_o_world.unsqueeze(0).to(device).to(dtype)
        ray_d_t = ray_d_world.unsqueeze(0).to(device).to(dtype)

        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o_t), fvdb.JaggedTensor(ray_d_t), fvdb.JaggedTensor(sdf_p)
        ).jdata

        expected_t = sphere_center[0].item() - sphere_rad - ray_o_world[0].item()
        voxel_diag = (3.0**0.5) * voxel_size
        self.assertTrue((isect >= 0).all().item())
        self.assertLess(
            abs(isect.item() - expected_t),
            voxel_diag,
            f"hit-time {isect.item()} expected {expected_t}, voxel_diag {voxel_diag}",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_high_resolution_sweep(self, device, dtype):
        # Stress test adapted from TestLevelSetRayIntersector.cc:280-308:
        # fire a 2-D grid of axis-aligned rays at an SDF sphere and
        # verify every reported hit lands on the sphere surface within
        # one voxel diagonal. This exercises the linear-interpolation
        # branch (lines 121-136 of RayImplicitIntersection.cu) over
        # thousands of rays — a uniform half-voxel bias would slip
        # past the existing 3 tests but get caught here.
        if dtype == torch.float16:
            self.skipTest("fp16 sweep precision insufficient for voxel-diagonal tolerance")
        N = 64 if device == "cuda" else 32
        width = 64 if device == "cuda" else 32
        sphere_rad = 0.3
        voxel_size = 1.0 / N
        sphere_center = torch.tensor([0.5, 0.5, 0.5])

        ii, jj, kk = torch.meshgrid([torch.arange(N, dtype=torch.float)] * 3, indexing="ij")
        xx = ii / N - sphere_center[0]
        yy = jj / N - sphere_center[1]
        zz = kk / N - sphere_center[2]
        sphere_sdf = (torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad).to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        # Sweep ray origins on a 2-D grid in the xy plane at z = -0.5,
        # all firing along +z. Origins span the full bbox in xy plus a
        # 0.05 world-unit margin so the rays at the edges of the sweep
        # also cleanly miss.
        margin = 0.05
        coords = torch.linspace(-margin, 1.0 + margin, width)
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        rays_o = torch.stack([gx.flatten(), gy.flatten(), torch.full((width * width,), -0.5)], dim=-1)
        rays_d = torch.tensor([0.0, 0.0, 1.0]).expand(width * width, 3).contiguous()
        rays_o_t = rays_o.to(device).to(dtype)
        rays_d_t = rays_d.to(device).to(dtype)

        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(rays_o_t), fvdb.JaggedTensor(rays_d_t), fvdb.JaggedTensor(sdf_p)
        ).jdata
        hit_mask = isect >= 0.0

        # The sphere covers a known fraction of the swept square. The
        # area of the analytic shadow disk (radius 0.3) divided by the
        # window area (1.1 x 1.1) is ~0.234. Require at least 70% of
        # that as a sanity floor.
        sphere_xy_area_frac = 3.14159265 * sphere_rad**2 / ((1.0 + 2 * margin) ** 2)
        expected_hits_low = int(0.7 * sphere_xy_area_frac * width**2)
        self.assertGreater(
            hit_mask.sum().item(),
            expected_hits_low,
            f"only {hit_mask.sum().item()} hits, expected at least {expected_hits_low}",
        )

        # Each hit point must lie on the sphere surface within a voxel
        # diagonal of slack (matches the bracket-entry precision the
        # interp branch can guarantee).
        o_hits = rays_o_t[hit_mask]
        d_hits = rays_d_t[hit_mask]
        t_hits = isect[hit_mask]
        hit_pts = o_hits + t_hits.unsqueeze(-1) * d_hits
        # Promote to fp32 for the geometric distance calculation so
        # this check stays meaningful for fp16 ray data (skipped above
        # but kept robust here).
        sc_f32 = sphere_center.to(device).to(torch.float32)
        hit_pts_f32 = hit_pts.to(torch.float32)
        hit_dists = (hit_pts_f32 - sc_f32.unsqueeze(0)).norm(dim=-1)
        position_err = (hit_dists - sphere_rad).abs()
        voxel_diag = (3.0**0.5) * voxel_size
        max_err = position_err.max().item()
        self.assertLess(
            max_err,
            voxel_diag,
            f"max sphere-surface position err {max_err} exceeds voxel diag {voxel_diag} "
            f"over {hit_mask.sum().item()} hits",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_single_voxel_bracket_entry(self, device, dtype):
        # Pin down the linear-interpolation branch on
        # RayImplicitIntersection.cu:121-136.
        #
        # fvdb stores values at primal voxel positions (world coord
        # i * voxel_size + origin). The trilinear sampler treats those
        # as 8 corners of an interpolation cube around a query point
        # (uses primalTransform; see SampleTrilinear.cu:43 + the
        # `xyz.floor()` in TrilinearStencil.h). The ray-marching
        # kernels — including this one — use dualTransform (see
        # RayImplicitIntersection.cu:79), which shifts the ray by half
        # a voxel in voxel space so that the HDDA's voxel-space cells
        # [i, i+1] map back to world-space cells of width voxel_size
        # CENTERED at the primal voxel position. Equivalently: a value
        # stored at primal voxel `i` is the cell-centered sample for
        # the cell that spans world x in [(i-0.5)*w + origin,
        # (i+0.5)*w + origin]. These are the same world positions
        # described two different ways; the kernel's linear interp
        # between two adjacent cell centers equals the linear interp
        # between two adjacent corner samples spaced one voxel apart.
        #
        # Test setup: SDF = +1 for i < 4, -1 for i >= 4, voxel_size=1,
        # origin=0. The two bracketing samples sit at world x=3 (+1)
        # and x=4 (-1). A symmetric +1/-1 step puts the linearly
        # interpolated zero at world x=3.5 exactly, so a ray from
        # world x=-1 along +x reports hit time t=4.5.
        N = 8
        voxel_size = 1.0
        sdf_dense = torch.full((N, N, N), 1.0)
        sdf_dense[4:, :, :] = -1.0
        sdf_dense = sdf_dense.to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sdf_dense.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        ray_o = torch.tensor([[-1.0, 4.0, 4.0]]).to(device).to(dtype)
        ray_d = torch.tensor([[1.0, 0.0, 0.0]]).to(device).to(dtype)
        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(ray_d), fvdb.JaggedTensor(sdf_p)
        ).jdata
        self.assertTrue((isect >= 0).all().item())
        # Quarter-voxel tolerance covers numerical noise around the
        # exact analytic answer of t=4.5.
        self.assertLess(
            abs(isect.item() - 4.5),
            0.25 * voxel_size,
            f"hit-time {isect.item()} expected ~4.5 (linear-interp zero crossing) within quarter voxel",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection_no_data_zero_prefix(self, device, dtype):
        # Regression test for issue #692: a ray that starts far from the
        # surface band must report the FIRST genuine sign change, not a
        # spurious crossing seeded by the no-data prefix.
        #
        # Sparse fvdb SDFs (e.g. integrate_tsdf + reinitialize_sdf) leave
        # active-but-unobserved voxels outside the narrow band at exactly
        # 0. Mathematically sgn(0) == 0, so if the kernel treated 0 as a
        # real sign it would latch the very first 0 voxel as its reference
        # and flag a "sign change" the instant the ray reached the first
        # signed (+/-) voxel — collapsing the hit to ~a voxel from the
        # origin regardless of where the real surface is. The kernel must
        # instead treat 0 as a gap (like NaN): it neither seeds the sign
        # reference nor triggers a hit.
        #
        # Setup along +x (voxel_size=1, origin=0): i in [0,3] = 0 (no-data
        # prefix), i in [4,5] = +1 (outside band), i in [6,7] = -1 (inside).
        # The only true crossing is the +1 -> -1 step bracketed by the
        # samples at world x=5 and x=6, whose symmetric interpolated zero
        # sits at world x=5.5. A ray from world x=-1 along +x therefore hits
        # at t=6.5. The buggy kernel instead reported ~t=4.5 (a bogus 0->+1
        # "crossing" near the start of the signed band).
        N = 8
        voxel_size = 1.0
        sdf_dense = torch.zeros((N, N, N))
        sdf_dense[4:6, :, :] = 1.0
        sdf_dense[6:, :, :] = -1.0
        sdf_dense = sdf_dense.to(device).to(dtype)

        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=voxel_size, origins=[0, 0, 0], device=device)
        sdf_p = grid.inject_from_dense_cminor(sdf_dense.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()

        ray_o = torch.tensor([[-1.0, 4.0, 4.0]]).to(device).to(dtype)
        ray_d = torch.tensor([[1.0, 0.0, 0.0]]).to(device).to(dtype)
        isect = grid.ray_implicit_intersection(
            fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(ray_d), fvdb.JaggedTensor(sdf_p)
        ).jdata
        self.assertTrue((isect >= 0).all().item(), f"unexpected miss, isect={isect}")
        self.assertLess(
            abs(isect.item() - 6.5),
            0.25 * voxel_size,
            f"hit-time {isect.item()} expected ~6.5 (true +1->-1 crossing); a value near 4.5 means "
            f"the no-data (0) prefix spuriously seeded the sign reference (issue #692)",
        )

    @parameterized.expand(["cpu", "cuda"])
    def test_ray_implicit_intersection_wrong_scalar_count_errors(self, device):
        # ray_implicit_intersection iterates leaf voxels (HDDALeafVoxelIterator)
        # and indexes gridScalars by getValue(ijk)-1, which is only valid with
        # exactly one scalar per active voxel. A mismatched count must raise an
        # error that points at the iterator contract rather than a generic
        # shape complaint, so a caller with per-active-value data knows to use
        # HDDAActiveValueIterator instead.
        N = 8
        grid = GridBatch.from_dense(1, [N, N, N], [0, 0, 0], voxel_sizes=1.0 / N, origins=[0, 0, 0], device=device)
        num_voxels = int(grid.total_voxels)
        bad_scalars = torch.zeros(num_voxels - 1, device=device, dtype=torch.float32)  # one too few
        ray_o = torch.tensor([[-1.0, 0.5, 0.5]], device=device, dtype=torch.float32)
        ray_d = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "HDDALeafVoxelIterator"):
            grid.ray_implicit_intersection(
                fvdb.JaggedTensor(ray_o), fvdb.JaggedTensor(ray_d), fvdb.JaggedTensor(bad_scalars)
            )

    @expand_tests(list(itertools.product(["cpu", "cuda"], [torch.float32, torch.float64])))
    def test_marching_cubes(self, device, dtype):
        # Generate the SDF for a sphere on a grid
        N = 32 if device == "cpu" else 64
        sphere_rads = [0.5, 0.33, 0.3, 0.28, 0.25]
        for batch_size in [1, 3, 5]:
            # Build a dense tensor of SDF values
            (
                ii,
                jj,
                kk,
            ) = torch.meshgrid(
                [torch.arange(N, device=device)] * 3, indexing="ij"
            )  # index space [0, N-1]
            xx, yy, zz = (
                ii.float() / (float(N) - 1) - 0.5,
                jj.float() / (float(N) - 1) - 0.5,
                kk.float() / (float(N) - 1) - 0.5,
            )  # normalize to [-1, 1]
            sphere_sdf = torch.stack(
                [-torch.sqrt(xx**2 + yy**2 + zz**2) + sphere_rad for sphere_rad in sphere_rads[:batch_size]]
            ).unsqueeze(
                -1
            )  # [B, N, N, N, 1] sdf

            # Build a grid with the SDF
            grid = GridBatch.from_dense(
                batch_size,
                [sphere_sdf[0].shape[i] for i in range(3)],
                [0] * 3,
                voxel_sizes=1.0 / N,
                origins=[0] * 3,
                device=device,
            )
            sdf_p = grid.inject_from_dense_cminor(sphere_sdf)  # permuted sdf values

            for level in [0.0, 0.2, -0.2]:
                v, f, _ = grid.marching_cubes(sdf_p, level)

                for bi in range(batch_size):
                    mesh_radius = torch.linalg.norm(
                        v[bi].jdata - torch.tensor([[0.5] * 3], device=device, dtype=dtype), axis=1
                    )
                    vox_size = torch.norm(grid.voxel_sizes[bi])
                    self.assertTrue(torch.all(mesh_radius - sphere_rads[bi] < vox_size / 2.0 - level))
                    self.assertTrue(torch.all(torch.logical_and(f[bi].jdata >= 0, f[bi].jdata < v[bi].jdata.shape[0])))
                # import polyscope as ps
                # ps.init()
                # ps.register_surface_mesh("marching_cubes", v.cpu()[0].jdata.numpy(), f.cpu()[0].jdata.numpy())
                # ps.show()

    @expand_tests(list(itertools.product(["cuda"], [torch.float32, torch.float64])))
    def test_integrate_tsdf_pixel_weight_blending(self, device, dtype):
        """Verify that per-pixel weights are applied to *new* samples during TSDF integration."""
        N = 8
        voxel_size = 0.1
        trunc_dist = 0.4
        depth1 = 1.2
        depth2 = 1.0
        pw = 3.0

        origin = torch.tensor([-N * voxel_size / 2, -N * voxel_size / 2, 1.0], device=device, dtype=dtype)
        grid = GridBatch.from_dense(
            1,
            [N, N, N],
            [0, 0, 0],
            voxel_sizes=voxel_size,
            origins=origin.tolist(),
            device=device,
        )

        cam_to_world = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        img_h, img_w = 100, 100
        proj = torch.tensor(
            [
                [100.0, 0, 50.0],
                [0, 100.0, 50.0],
                [0, 0, 1],
            ],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)

        depth_img1 = torch.full((1, img_h, img_w), depth1, device=device, dtype=dtype)
        depth_img2 = torch.full((1, img_h, img_w), depth2, device=device, dtype=dtype)

        tsdf_init = fvdb.JaggedTensor(torch.zeros(grid.total_voxels, device=device, dtype=dtype))
        weights_init = fvdb.JaggedTensor(torch.zeros(grid.total_voxels, device=device, dtype=dtype))

        grid1, tsdf1, w1 = grid.integrate_tsdf(
            trunc_dist,
            proj,
            cam_to_world,
            tsdf_init,
            weights_init,
            depth_img1,
            weight_images=None,
        )

        weight_img = torch.full((1, img_h, img_w), pw, device=device, dtype=dtype)
        grid2_w, tsdf2_w, w2_w = grid1.integrate_tsdf(
            trunc_dist,
            proj,
            cam_to_world,
            tsdf1,
            w1,
            depth_img2,
            weight_images=weight_img,
        )
        grid2_u, tsdf2_u, w2_u = grid1.integrate_tsdf(
            trunc_dist,
            proj,
            cam_to_world,
            tsdf1,
            w1,
            depth_img2,
            weight_images=None,
        )

        atol = dtype_to_atol(dtype)

        self.assertEqual(grid2_w.total_voxels, grid2_u.total_voxels)

        m = w2_w.jdata.flatten() != w2_u.jdata.flatten()
        self.assertTrue(m.any(), "Some voxels should have been updated")

        # Weights must accumulate pw (not 1): w2_w - w2_u == pw - 1
        torch.testing.assert_close(
            w2_w.jdata.flatten()[m] - w2_u.jdata.flatten()[m],
            torch.full_like(w2_w.jdata.flatten()[m], pw - 1.0),
            atol=atol,
            rtol=0,
        )

        # Sample TSDF at known points and verify against the weighted-average formula:
        #   expected = (1 * s1 + pw * s2) / (1 + pw),  where s = clamp((depth - z) / trunc, max=1)
        query_world = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [0.0, 0.0, 1.2]],
            device=device,
            dtype=dtype,
        )
        query_z = query_world[:, 2]

        s1 = torch.clamp((depth1 - query_z) / trunc_dist, max=1.0)
        s2 = torch.clamp((depth2 - query_z) / trunc_dist, max=1.0)
        expected_tsdf = (1.0 * s1 + pw * s2) / (1.0 + pw)

        pts = fvdb.JaggedTensor(query_world)
        tsdf2_w_2d = fvdb.JaggedTensor(tsdf2_w.jdata.unsqueeze(-1))
        sampled = grid2_w.sample_trilinear(pts, tsdf2_w_2d).jdata.flatten()
        torch.testing.assert_close(sampled, expected_tsdf, atol=atol, rtol=0)

        # Same check for the uniform run (implicit weight_images=1)
        expected_tsdf_u = (1.0 * s1 + 1.0 * s2) / (1.0 + 1.0)
        tsdf2_u_2d = fvdb.JaggedTensor(tsdf2_u.jdata.unsqueeze(-1))
        sampled_u = grid2_u.sample_trilinear(pts, tsdf2_u_2d).jdata.flatten()
        torch.testing.assert_close(sampled_u, expected_tsdf_u, atol=atol, rtol=0)

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_refine_empty_grid(self, device, dtype):
        grid = GridBatch.from_dense(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0], device=device)
        values = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        values, subgrid = grid.refine(
            1,
            fvdb.JaggedTensor(values),
            mask=fvdb.JaggedTensor(torch.zeros(grid.total_voxels, dtype=torch.bool, device=device)),
        )
        self.assertTrue(subgrid.total_voxels == 0)
        self.assertTrue(values.rshape[0] == 0)
        self.assertTrue(values.rshape[1] == 17)

    @parameterized.expand(all_device_dtype_combos)
    def test_bbox_attrs(self, device, dtype):
        grid = GridBatch.from_zero_voxels(device=device)
        self.assertTrue(torch.equal(grid.bboxes, torch.tensor([[[0, 0, 0], [0, 0, 0]]], device=device)))
        self.assertTrue(torch.equal(grid.total_bbox, torch.tensor([[0, 0, 0], [0, 0, 0]], device=device)))
        grid = GridBatch.from_dense(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0], device=device)
        self.assertTrue(torch.equal(grid.bboxes, torch.tensor([[[0, 0, 0], [31, 31, 31]]], device=device)))
        self.assertTrue(torch.equal(grid.dual_bboxes, torch.tensor([[[0, 0, 0], [32, 32, 32]]], device=device)))
        self.assertTrue(torch.equal(grid.total_bbox, torch.tensor([[0, 0, 0], [31, 31, 31]], device=device)))

    @parameterized.expand(all_device_dtype_combos)
    def test_clip_grid(self, device, dtype):

        grid = GridBatch.from_dense(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0], device=device)
        values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(fvdb.JaggedTensor(values_in), [[0, 0, 0]], [[5, 5, 5]])
        self.assertTrue(clipped_grid.num_voxels == 6**3)
        self.assertTrue(clipped_data.jdata.shape[0] == 6**3)
        self.assertTrue(torch.equal(clipped_data.joffsets, clipped_grid.joffsets))

        grid = GridBatch.from_dense(
            1, [32, 32, 32], [-2, -2, -2], voxel_sizes=1.0 / 32, origins=[0, 0, 0], device=device
        )
        values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(fvdb.JaggedTensor(values_in), [[-2, -2, -2]], [[5, 5, 5]])
        self.assertTrue(clipped_grid.num_voxels == 8**3)
        self.assertTrue(clipped_data.jdata.shape[0] == 8**3)
        self.assertTrue(torch.equal(clipped_data.joffsets, clipped_grid.joffsets))

        # Test gradients through clip
        num_features = 17
        grid = GridBatch.from_dense(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0], device=device)
        features = torch.randn(grid.total_voxels, num_features, device=device, dtype=dtype, requires_grad=True)

        clipped_features, clipped_grid = grid.clip(fvdb.JaggedTensor(features), [[0, 0, 0]], [[5, 5, 5]])

        loss = clipped_features.jdata.pow(3).sum()
        loss.backward()

        assert features.grad is not None  # Removes type errors with .grad
        clipped_features_grad = features.grad.clone()

        features.grad.zero_()
        self.assertTrue(torch.all(features.grad == torch.zeros_like(features.grad)))
        self.assertTrue(not torch.all(features.grad == clipped_features_grad))

        ijk_clip_mask = torch.all(grid.ijk.jdata <= 5, 1)

        loss = (features[ijk_clip_mask.repeat(num_features, 1).swapaxes(0, 1)].pow(3)).sum()
        loss.backward()
        self.assertTrue(torch.equal(clipped_features_grad, features.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_dual_without_border(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(dtype).to(device)
        for b in [1, 3]:
            pts = JaggedTensor([torch.randn(np.random.randint(100_000, 300_000), 3).to(device=device, dtype=dtype)] * b)
            grid = fvdb.GridBatch.from_points(pts, vox_size, vox_origin)
            dual_grid = grid.dual_grid()

            neighbors = grid.neighbor_indexes(dual_grid.ijk, 1)
            inner_mask = torch.all(neighbors.jdata[:, 1:, 1:, 1:].reshape(-1, 8) != -1, dim=-1)
            inner_ijk = dual_grid.ijk.rmask(inner_mask)
            dual_inner = fvdb.GridBatch.from_ijk(inner_ijk, voxel_sizes=vox_size, origins=vox_origin)

            dual_outer_with_skip = grid.dual_grid(exclude_border=True)
            for i in range(b):
                ijk1 = dual_inner.ijk[i].jdata
                ijk2 = dual_outer_with_skip.ijk[i].jdata
                ijk1_i = set([tuple(ijk1[j].cpu().numpy().tolist()) for j in range(ijk1.shape[0])])
                ijk2_i = set([tuple(ijk2[j].cpu().numpy().tolist()) for j in range(ijk2.shape[0])])
                self.assertTrue(ijk1_i == ijk2_i)

    @parameterized.expand(["cuda", "cpu"])
    def test_max_grids(self, device):
        dtype = torch.float32
        VAL_BIG = fvdb.GridBatch.max_grids_per_batch + 1
        VAL_OKAY = fvdb.GridBatch.max_grids_per_batch
        pts_too_big = fvdb.JaggedTensor([torch.randn(2, 3).to(device=device, dtype=dtype)] * VAL_BIG)
        ijk_too_big = fvdb.JaggedTensor([(torch.randn(2, 3).to(device=device, dtype=dtype) * 100.0).int()] * VAL_BIG)
        faces_too_big = fvdb.JaggedTensor([torch.randint(2, (2, 3)).to(device=device)] * VAL_BIG)

        with self.assertRaises(ValueError):
            fvdb.GridBatch.from_points(pts_too_big, 1.0, [0] * 3)

        with self.assertRaises(ValueError):
            fvdb.GridBatch.from_ijk(ijk_too_big, voxel_sizes=1.0, origins=[0] * 3)

        with self.assertRaises(ValueError):
            fvdb.GridBatch.from_mesh(pts_too_big, faces_too_big, voxel_sizes=1.0, origins=[0] * 3)

        with self.assertRaises(ValueError):
            fvdb.GridBatch.from_nearest_voxels_to_points(pts_too_big, voxel_sizes=1.0, origins=[0] * 3)

        with self.assertRaises(ValueError):
            fvdb.GridBatch.from_dense(VAL_BIG, [4, 4, 4], [0, 0, 0], 1.0, [0] * 3)

        fvdb.GridBatch.from_points(pts_too_big[:-1], 1.0, [0] * 3)
        fvdb.GridBatch.from_ijk(ijk_too_big[:-1], voxel_sizes=1.0, origins=[0] * 3)
        fvdb.GridBatch.from_mesh(pts_too_big[:-1], faces_too_big[:-1], voxel_sizes=1.0, origins=[0] * 3)
        fvdb.GridBatch.from_nearest_voxels_to_points(pts_too_big[:-1], voxel_sizes=1.0, origins=[0] * 3)
        fvdb.GridBatch.from_dense(VAL_OKAY, [4, 4, 4], [0, 0, 0], 1.0, [0] * 3, device=device)


if __name__ == "__main__":
    unittest.main()
