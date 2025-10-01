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
    make_dense_grid_and_point_data,
    make_grid_and_point_data,
)
from parameterized import parameterized

import fvdb
from fvdb import Grid, JaggedTensor

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

bfloat16_combos = [["cuda", torch.bfloat16]]


class TestBasicOpsSingle(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)
        pass

    @parameterized.expand(["cpu", "cuda"])
    def test_dilate_grid(self, device):
        def get_points(npc: int, device: torch.device | str) -> torch.Tensor:
            return torch.randn((npc, 3), dtype=torch.float32, device=device, requires_grad=False)

        vxl_size = 0.4
        npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
        plist = get_points(npc, device)
        grid = Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        d_amt = 2
        dilated_grid = grid.dilated_grid(d_amt)

        ijk = grid.ijk
        if ijk.numel == 0:
            expected_ijk = ijk
        else:
            dilated_ijk = torch.cat(
                [
                    ijk + torch.tensor([[a, b, c]]).to(ijk)
                    for (a, b, c) in itertools.product(range(-d_amt, d_amt + 1), repeat=3)
                ],
                dim=0,
            )
            expected_ijk = dilated_ijk

        expected_grid = Grid.from_ijk(
            expected_ijk,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            device=device,
        )

        self.assertTrue(torch.equal(dilated_grid.ijk, expected_grid.ijk))

    @parameterized.expand(["cpu", "cuda"])
    def test_merge_grids(self, device):
        def get_points(npc: int, device: torch.device | str) -> torch.Tensor:
            return torch.randn((npc, 3), dtype=torch.float32, device=device, requires_grad=False)

        vxl_size = 0.4
        npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
        plist = get_points(npc, device)
        grid1 = Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
        plist = get_points(npc, device)
        grid2 = Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        merged_grid = grid1.merged_grid(grid2)
        ijk1 = grid1.ijk
        ijk2 = grid2.ijk
        expected_ijk = torch.cat([ijk1, ijk2])

        expected_grid = Grid.from_ijk(
            expected_ijk,
            voxel_size=grid1.voxel_size,
            origin=grid1.origin,
            device=device,
        )

        self.assertTrue(torch.equal(merged_grid.ijk, expected_grid.ijk))

    @parameterized.expand(["cpu", "cuda"])
    def test_prune_grids(self, device):
        def get_points(npc: int, device: torch.device | str) -> torch.Tensor:
            return torch.randn((npc, 3), dtype=torch.float32, device=device, requires_grad=False)

        vxl_size = 0.4
        npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
        plist = get_points(npc, device)
        grid = Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        mask = torch.rand(grid.num_voxels, device=device) > 0.5

        pruned_grid = grid.pruned_grid(mask)
        ijk = grid.ijk
        expected_ijk = ijk[mask]

        expected_grid = Grid.from_ijk(
            expected_ijk,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            device=device,
        )

        self.assertTrue(torch.equal(pruned_grid.ijk, expected_grid.ijk))

    @parameterized.expand(["cpu", "cuda"])
    def test_prune_grids_empty(self, device):
        def get_points(npc: int, device: torch.device | str) -> torch.Tensor:
            return torch.randn((npc, 3), dtype=torch.float32, device=device, requires_grad=False)

        vxl_size = 0.4
        npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
        plist = get_points(npc, device)
        grid = Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        # tensor is empty
        mask = torch.zeros(grid.num_voxels, dtype=torch.bool, device=device)
        pruned_grid = grid.pruned_grid(mask)

        ijk = grid.ijk
        expected_ijk = ijk[mask]
        expected_grid = Grid.from_ijk(
            expected_ijk,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            device=device,
        )
        self.assertTrue(torch.equal(pruned_grid.ijk, expected_grid.ijk))

    @parameterized.expand(["cpu", "cuda"])
    def test_inject_grids(self, device):
        def get_points(npc: int, device: torch.device | str) -> torch.Tensor:
            return torch.randn((npc, 3), dtype=torch.float32, device=device, requires_grad=False)

        def build_random_grid(voxel_size):
            npc = int(torch.randint(low=0, high=1000, size=(1,), device=device).cpu().item())
            plist = get_points(npc, device)
            return Grid.from_points(plist, voxel_size=[vxl_size] * 3, device=device)

        vxl_size = 0.4

        grid1 = build_random_grid(vxl_size)
        grid2 = build_random_grid(vxl_size)
        grid3 = build_random_grid(vxl_size)
        grid12 = grid1.merged_grid(grid2)
        grid23 = grid2.merged_grid(grid3)

        sidecar1 = torch.rand(grid1.num_voxels, device=device)
        sidecar2 = torch.rand(grid2.num_voxels, device=device)
        sidecar3 = torch.rand(grid3.num_voxels, device=device)

        sidecar12 = torch.zeros(grid12.num_voxels, device=device)
        sidecar23 = torch.zeros(grid23.num_voxels, device=device)
        sidecar23_ref = torch.zeros(grid23.num_voxels, device=device)

        grid2.inject_to(grid23, sidecar2, sidecar23_ref)
        grid23.inject_from(grid3, sidecar3, sidecar23_ref)

        grid1.inject_to(grid12, sidecar1, sidecar12)
        grid12.inject_from(grid2, sidecar2, sidecar12)
        grid12.inject_to(grid23, sidecar12, sidecar23)
        self.assertFalse(torch.equal(sidecar23, sidecar23_ref))
        grid23.inject_from(grid3, sidecar3, sidecar23)
        self.assertTrue(torch.equal(sidecar23, sidecar23_ref))

    # @parameterized.expand(["cpu", "cuda"])
    # def test_inject_grids_multidim(self, device):
    #     def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
    #         batch_size = len(npc)
    #         plist = []
    #         for i in range(batch_size):
    #             ni = npc[i]
    #             plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
    #         return plist

    #     def build_random_gridbatch(batch_size, voxel_size):
    #         npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
    #         plist = get_point_list(npc, device)
    #         pc_jagged = fvdb.JaggedTensor(plist)
    #         return fvdb.gridbatch_from_points(pc_jagged, voxel_sizes=[[voxel_size] * 3] * batch_size)

    #     batch_size = 2
    #     vxl_size = 0.4

    #     grid_batch1 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch2 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch3 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch12 = grid_batch1.merged_grid(grid_batch2)
    #     grid_batch23 = grid_batch2.merged_grid(grid_batch3)

    #     sidecar1 = grid_batch1.jagged_like(torch.rand(grid_batch1.total_voxels, 3, device=device))
    #     sidecar2 = grid_batch2.jagged_like(torch.rand(grid_batch2.total_voxels, 3, device=device))
    #     sidecar3 = grid_batch3.jagged_like(torch.rand(grid_batch3.total_voxels, 3, device=device))

    #     sidecar12 = grid_batch12.jagged_like(torch.zeros(grid_batch12.total_voxels, 3, device=device))
    #     sidecar23 = grid_batch23.jagged_like(torch.zeros(grid_batch23.total_voxels, 3, device=device))
    #     sidecar23_ref = grid_batch23.jagged_like(torch.zeros(grid_batch23.total_voxels, 3, device=device))

    #     # def inject_legacy(src_grid: GridBatch, dst_grid: GridBatch, src: JaggedTensor, dst: JaggedTensor):
    #     #     src_ijk = src_grid.ijk
    #     #     dst_idx = dst_grid.ijk_to_index(src_ijk)
    #     #     mask = dst_idx.jagged_like(dst_idx.jdata >= 0)
    #     #     dst_idx.jdata = dst_idx.jdata[mask.jdata]
    #     #     dst_ijk = src_ijk.jagged_like(src_ijk.jdata[mask.jdata])
    #     #     src_idx = src_grid.ijk_to_index(dst_ijk)
    #     #     dst.jdata[dst_idx.jdata] = src.jdata[src_idx.jdata]

    #     grid_batch2.inject_to(grid_batch23, sidecar2, sidecar23_ref)
    #     # inject_legacy(grid_batch2, grid_batch23, sidecar2, sidecar23_ref)
    #     grid_batch23.inject_from(grid_batch3, sidecar3, sidecar23_ref)
    #     # inject_legacy(grid_batch3, grid_batch23, sidecar3, sidecar23_ref)

    #     grid_batch1.inject_to(grid_batch12, sidecar1, sidecar12)
    #     grid_batch12.inject_from(grid_batch2, sidecar2, sidecar12)
    #     grid_batch12.inject_to(grid_batch23, sidecar12, sidecar23)
    #     self.assertFalse(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))
    #     grid_batch23.inject_from(grid_batch3, sidecar3, sidecar23)
    #     self.assertTrue(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))

    # @parameterized.expand(["cpu", "cuda"])
    # def test_inject_grids_multidim2(self, device):
    #     def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
    #         batch_size = len(npc)
    #         plist = []
    #         for i in range(batch_size):
    #             ni = npc[i]
    #             plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
    #         return plist

    #     def build_random_gridbatch(batch_size, voxel_size):
    #         npc = torch.randint(low=0, high=1000, size=(batch_size,), device=device).tolist()
    #         plist = get_point_list(npc, device)
    #         pc_jagged = fvdb.JaggedTensor(plist)
    #         return fvdb.gridbatch_from_points(pc_jagged, voxel_sizes=[[voxel_size] * 3] * batch_size)

    #     batch_size = 2
    #     vxl_size = 0.4

    #     grid_batch1 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch2 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch3 = build_random_gridbatch(batch_size, vxl_size)
    #     grid_batch12 = grid_batch1.merged_grid(grid_batch2)
    #     grid_batch23 = grid_batch2.merged_grid(grid_batch3)

    #     sidecar1 = grid_batch1.jagged_like(torch.rand(grid_batch1.total_voxels, 3, 2, device=device))
    #     sidecar2 = grid_batch2.jagged_like(torch.rand(grid_batch2.total_voxels, 3, 2, device=device))
    #     sidecar3 = grid_batch3.jagged_like(torch.rand(grid_batch3.total_voxels, 3, 2, device=device))

    #     sidecar12 = grid_batch12.jagged_like(torch.zeros(grid_batch12.total_voxels, 3, 2, device=device))
    #     sidecar23 = grid_batch23.jagged_like(torch.zeros(grid_batch23.total_voxels, 3, 2, device=device))
    #     sidecar23_ref = grid_batch23.jagged_like(torch.zeros(grid_batch23.total_voxels, 3, 2, device=device))

    #     # def inject_legacy(src_grid: GridBatch, dst_grid: GridBatch, src: JaggedTensor, dst: JaggedTensor):
    #     #     src_ijk = src_grid.ijk
    #     #     dst_idx = dst_grid.ijk_to_index(src_ijk)
    #     #     mask = dst_idx.jagged_like(dst_idx.jdata >= 0)
    #     #     dst_idx.jdata = dst_idx.jdata[mask.jdata]
    #     #     dst_ijk = src_ijk.jagged_like(src_ijk.jdata[mask.jdata])
    #     #     src_idx = src_grid.ijk_to_index(dst_ijk)
    #     #     dst.jdata[dst_idx.jdata] = src.jdata[src_idx.jdata]

    #     grid_batch2.inject_to(grid_batch23, sidecar2, sidecar23_ref)
    #     # inject_legacy(grid_batch2, grid_batch23, sidecar2, sidecar23_ref)
    #     grid_batch23.inject_from(grid_batch3, sidecar3, sidecar23_ref)
    #     # inject_legacy(grid_batch3, grid_batch23, sidecar3, sidecar23_ref)

    #     grid_batch1.inject_to(grid_batch12, sidecar1, sidecar12)
    #     grid_batch12.inject_from(grid_batch2, sidecar2, sidecar12)
    #     grid_batch12.inject_to(grid_batch23, sidecar12, sidecar23)
    #     self.assertFalse(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))
    #     grid_batch23.inject_from(grid_batch3, sidecar3, sidecar23)
    #     self.assertTrue(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))

    # @parameterized.expand(all_device_dtype_combos)
    # def test_subdivide_1x_with_mask(self, device, dtype):
    #     def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
    #         batch_size = len(npc)
    #         plist = []
    #         for i in range(batch_size):
    #             ni = npc[i]
    #             plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
    #         return plist

    #     batch_size = 5
    #     vxl_size = 0.4
    #     npc = torch.randint(low=0, high=100, size=(batch_size,), device=device).tolist()
    #     plist = get_point_list(npc, device)
    #     pc_jagged = fvdb.JaggedTensor(plist)
    #     grid_batch = fvdb.gridbatch_from_points(pc_jagged, voxel_sizes=[[vxl_size] * 3] * batch_size)

    #     random_mask = (
    #         torch.randn(grid_batch.total_voxels, device=device)
    #     ) > 0.5  # random mask that selects voxels randomly from different grids
    #     random_mask = grid_batch.jagged_like(random_mask)
    #     filtered_grid_batch = grid_batch.subdivided_grid(1, random_mask)
    #     sum = 0
    #     for i in range(batch_size):
    #         si = grid_batch.joffsets[i]
    #         ei = grid_batch.joffsets[i + 1]
    #         ri = random_mask.jdata[si:ei]
    #         self.assertEqual(ri.sum().item(), filtered_grid_batch.num_voxels_at(i))
    #         sum += torch.sum(ri)

    #     self.assertEqual(sum, torch.sum(random_mask.jdata))
    #     self.assertEqual(torch.sum(random_mask.jdata), filtered_grid_batch.total_voxels)
    #     self.assertTrue(
    #         torch.all(random_mask.int().jsum().jdata == filtered_grid_batch.num_voxels.int()).item(),
    #     )

    @parameterized.expand(["cpu", "cuda"])
    def test_is_same(self, device):
        grid = Grid.from_dense([16, 16, 16], [0, 0, 0], voxel_size=1.0 / 16, origin=[0, 0, 0])
        self.assertTrue(grid.num_voxels == 16**3)

        grid2 = Grid.from_dense([16, 16, 16], [0, 0, 0], voxel_size=1.0 / 16, origin=[0, 0, 0])
        self.assertFalse(grid.is_same(grid2))
        self.assertNotEqual(grid.address, grid2.address)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxel_neighborhood(self, device, dtype):
        randvox = torch.randint(0, 256, size=(10_000, 3), dtype=torch.int32).to(device)
        randvox = torch.cat(
            [randvox, randvox + torch.ones(1, 3).to(randvox)], dim=0
        )  # Ensure there are always neighbors

        grid = Grid.from_ijk(randvox)

        gt_nhood = torch.zeros((randvox.shape[0], 3, 3, 3), dtype=torch.int32).to(device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    off = torch.tensor([[i - 1, j - 1, k - 1]]).to(randvox)
                    nh_ijk = randvox + off
                    idx = grid.ijk_to_index(nh_ijk)
                    mask = grid.coords_in_grid(nh_ijk)
                    gt_nhood[:, i, j, k] = torch.where(mask, idx, -torch.ones_like(idx))

        nhood = grid.neighbor_indexes(randvox, 1, 0)

        self.assertTrue(torch.equal(nhood, gt_nhood))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1).dual_grid()

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid.world_to_grid(pts)

        self.assertTrue(
            torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)),
            f"max_diff = {torch.abs(pred_dual_coordinates - target_dual_coordinates).max()}",
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_primal(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)

        target_primal_coordinates = (pts - vox_origin) / vox_size
        pred_primal_coordinates = grid.world_to_grid(pts)

        self.assertTrue(torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        pts.requires_grad = True

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1).dual_grid()

        pred_dual_coordinates = grid.world_to_grid(pts)
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

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)

        pred_primal_coordinates = grid.world_to_grid(pts)
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

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)

        target_world_pts = (grid_pts * vox_size) + vox_origin
        pred_world_pts = grid.grid_to_world(grid_pts)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_dual_to_world(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1).dual_grid()

        target_world_pts = ((grid_pts - 0.5) * vox_size) + vox_origin
        pred_world_pts = grid.grid_to_world(grid_pts)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_primal_to_world_grad(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)
        grid_pts.requires_grad = True

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)

        pred_world_pts = grid.grid_to_world(grid_pts)
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

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1).dual_grid()

        pred_world_pts = grid.grid_to_world(grid_pts)
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

        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)
        grid_d = grid.dual_grid()
        grid_dd = grid_d.dual_grid()

        primal_origin = grid.origin
        dual_origin = grid_d.origin

        self.assertFalse(torch.allclose(primal_origin, dual_origin))
        self.assertTrue(torch.all(primal_origin == grid_dd.origin))
        self.assertTrue(torch.all(dual_origin == grid_dd.dual_grid().origin))

        target_primal_coordinates = (pts - vox_origin) / vox_size
        pred_primal_coordinates = grid.world_to_grid(pts)

        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)),
            f"Max diff = {torch.max(torch.abs(target_primal_coordinates- pred_primal_coordinates)).item()}",
        )

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid_d.world_to_grid(pts)
        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))

        pred_primal_coordinates_dd = grid_dd.world_to_grid(pts)
        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates_dd, atol=dtype_to_atol(dtype))
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_index(self, device, dtype):
        gsize = 7

        grid_p, grid_d, _ = make_dense_grid_and_point_data(gsize, device, dtype)

        pijk = grid_p.ijk
        dijk = grid_d.ijk

        for in_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            pijk, dijk = pijk.to(in_dtype), dijk.to(in_dtype)
            pidx = grid_p.ijk_to_index(pijk)
            didx = grid_d.ijk_to_index(dijk)

            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx))
            self.assertTrue(torch.all(didx == target_didx))

            ppmt = torch.randperm(pidx.shape[0])
            dpmt = torch.randperm(pidx.shape[0])

            pidx = grid_p.ijk_to_index(pijk[ppmt])
            didx = grid_d.ijk_to_index(dijk[dpmt])
            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx[ppmt]))
            self.assertTrue(torch.all(didx == target_didx[dpmt]))

    @parameterized.expand(all_device_dtype_combos)
    def test_coords_in_grid(self, device, _):
        num_inside = 1000 if device == "cpu" else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = Grid.from_ijk(random_coords, device=device)

        enabled_coords = grid.ijk
        num_outside = 1000 if device == "cpu" else 10_000

        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        pred_mask = grid.coords_in_grid(all_coords)
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_points_in_grid(self, device, dtype):
        num_inside = 1000 if device == "cpu" else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = Grid.from_ijk(random_coords, device=device)

        enabled_coords = grid.ijk
        num_outside = 1000 if device == "cpu" else 10_000
        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        all_world_points = grid.grid_to_world(all_coords.to(dtype))

        pred_mask = grid.points_in_grid(all_world_points)
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_cubes_intersect_grid(self, device, dtype):
        torch.random.manual_seed(0)

        grid, grid_d, p = make_grid_and_point_data(device, dtype, include_boundary_points=True)
        voxel_size = grid.voxel_size

        primal_mask = grid.cubes_in_grid(p)
        dual_mask = grid_d.cubes_in_grid(p, -voxel_size / 2, voxel_size / 2)
        self.assertTrue(torch.all(dual_mask[primal_mask]))

        primal_mask = grid.cubes_intersect_grid(p, -voxel_size / 2, voxel_size / 2)
        dual_mask = grid_d.cubes_intersect_grid(p)
        self.assertTrue(torch.all(primal_mask == dual_mask))

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_subdivided_grid(self, device, dtype):
        p = torch.randn(100, 3, device=device, dtype=torch.float)
        vox_size = 0.1
        grid = Grid.from_points(p, vox_size, (0.0, 0.0, 0.0), device=device).dilated_grid(1)

        grids = [grid]
        for i in range(2):
            subdiv_factor = i + 2
            mask = torch.rand(grids[i].num_voxels, device=device) > 0.5

            grids.append(grids[-1].refined_grid(subdiv_factor, mask))
            self.assertEqual(int(mask.sum().item()) * subdiv_factor**3, grids[-1].num_voxels)

        grids = [grid]
        for i, subdiv_factor in enumerate([(2, 2, 1), (3, 2, 2), (1, 1, 3)]):
            mask = torch.rand(grids[i].num_voxels, device=device) > 0.5

            nsubvox = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
            grids.append(grids[-1].refined_grid(subdiv_factor, mask))
            self.assertEqual(int(mask.sum().item()) * nsubvox, grids[-1].num_voxels)
        if device == "cuda":
            torch.cuda.synchronize()

    @parameterized.expand(all_device_dtype_combos)
    def test_build_from_pointcloud_nearest_voxels(self, device, dtype):
        p = torch.randn((100, 3), device=device, dtype=dtype)

        vox_size = 0.01
        grid = Grid.from_nearest_voxels_to_points(p, vox_size)

        if p.dtype == torch.half:
            p = p.float()

        expected_ijk = torch.floor(grid.world_to_grid(p))
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

        predicted_ijk = grid.ijk

        predicted_ijk_set = set(
            {
                (predicted_ijk[i, 0].item(), predicted_ijk[i, 1].item(), predicted_ijk[i, 2].item())
                for i in range(predicted_ijk.shape[0])
            }
        )

        self.assertEqual(predicted_ijk_set, expected_ijk_set)

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_subdivide(self, device, dtype):
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
            grid = Grid.from_nearest_voxels_to_points(p, vox_size, (0.0, 0.0, 0.0))

            feats = torch.randn(grid.num_voxels, 32).to(p)
            feats.requires_grad = True

            mask = torch.ones(grid.num_voxels, dtype=torch.bool).to(device)

            feats_fine, grid_fine = grid.refine(subdiv_factor, feats, mask=mask)
            self.assertTrue(torch.allclose(grid_fine.voxel_size, grid.voxel_size / subvec))
            self.assertTrue(torch.allclose(grid_fine.origin, grid.origin - 0.5 * grid_fine.voxel_size * fac_sub_one))

            fine_to_coarse_ijk = (grid_fine.ijk / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fine_to_coarse_ijk.to(torch.int32))

            self.assertTrue(torch.all(feats_fine == feats[fine_to_coarse_idx]))

            loss = feats_fine.pow(3).sum()
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
    def test_subdivide_with_mask(self, device, dtype):
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

            grid = Grid.from_nearest_voxels_to_points(p, vox_size, (0.0, 0.0, 0.0))

            feats = torch.randn(grid.num_voxels, 32).to(p)
            feats.requires_grad = True

            mask = torch.rand(grid.num_voxels).to(device) > 0.5

            feats_fine, grid_fine = grid.refine(subdiv_factor, feats, mask=mask)
            self.assertTrue(torch.allclose(grid_fine.voxel_size, grid.voxel_size / subvec))
            self.assertTrue(torch.allclose(grid_fine.origin, grid.origin - 0.5 * grid_fine.voxel_size * fac_sub_one))

            fine_to_coarse_ijk = (grid_fine.ijk / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fine_to_coarse_ijk.to(torch.int32))

            self.assertTrue(torch.all(feats_fine == feats[fine_to_coarse_idx]))

            loss = feats_fine.pow(3).sum()
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
        grid = Grid.from_dense([20, 20, 20], voxel_size=vox_size, origin=vox_origin, device=device)
        assert grid.num_voxels == 20**3
        grid_vals = torch.randn(grid.num_voxels, 3).to(device).to(dtype)

        for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
            grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals)
            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_size, grid.voxel_size * pool_factor))
                self.assertTrue(
                    torch.allclose(grid_coarse.origin, grid.origin + 0.5 * grid.voxel_size * (pool_factor - 1))
                )
            else:
                self.assertTrue(
                    torch.allclose(grid_coarse.voxel_size, grid.voxel_size * torch.tensor(pool_factor).to(device))
                )
                self.assertTrue(
                    torch.allclose(
                        grid_coarse.origin,
                        grid.origin + 0.5 * grid.voxel_size * (torch.tensor(pool_factor) - 1).to(device),
                    )
                )

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
            grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device="cpu", dtype=dtype)
            grid_ijk_cpu = grid.ijk.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
            grid_vals_t = grid_vals_t.to(device)
            grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device="cpu")
            grid_coarse_ijk_cpu = grid_coarse.ijk.cpu()
            for i, coord in enumerate(grid_coarse_ijk_cpu):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
            grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    # @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    # def test_strided_max_pool(self, device, dtype):
    #     vox_size = 0.05
    #     vox_origin = (0.0, 0.0, 0.0)
    #     gsize = int(1 / vox_size)
    #     grid = GridBatch(device=device)
    #     grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
    #     assert grid.total_voxels == 20**3
    #     grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

    #     for pool_factor in ((2, 3, 4), 2, 4, 5, 10):
    #         # Our behavior differs slightly from PyTorch when pool_factor < stride, so only test this.
    #         if isinstance(pool_factor, int):
    #             pools = (pool_factor, pool_factor + 1, pool_factor + 2, pool_factor + 5)
    #         else:
    #             assert isinstance(pool_factor, tuple)

    #             def addit(pf, val_):
    #                 assert isinstance(pf, tuple)
    #                 return (pf[0] + val_, pf[1] + val_, pf[2] + val_)

    #             pools = (pool_factor, addit(pool_factor, 1), addit(pool_factor, 2), addit(pool_factor, 5))
    #         for stride in pools:
    #             grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals, stride=stride)
    #             grid_vals_coarse = grid_vals_coarse.jdata
    #             if isinstance(stride, int):
    #                 self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * stride))
    #                 self.assertTrue(
    #                     torch.allclose(
    #                         grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (stride - 1)
    #                     )
    #                 )
    #             else:
    #                 self.assertTrue(
    #                     torch.allclose(
    #                         grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(stride).to(device)
    #                     )
    #                 )
    #                 self.assertTrue(
    #                     torch.allclose(
    #                         grid_coarse.origins[0],
    #                         grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(stride) - 1).to(device),
    #                     )
    #                 )

    #             # Pytorch pooling
    #             torch_pool_op = torch.nn.MaxPool3d(pool_factor, stride=stride, ceil_mode=True)
    #             # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
    #             grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device="cpu", dtype=dtype)
    #             grid_ijk_cpu = grid.ijk.jdata.cpu()
    #             grid_vals_cpu = grid_vals.cpu()
    #             for i, coord in enumerate(grid_ijk_cpu):
    #                 grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
    #             grid_vals_t = grid_vals_t.to(device)
    #             grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
    #             grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

    #             grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device="cpu")
    #             grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
    #             for i, coord in enumerate(grid_coarse_ijk_cpu):
    #                 grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
    #             grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
    #             self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    # @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    # def test_max_pool_grad(self, device, dtype):
    #     vox_size = 0.05
    #     vox_origin = (0.0, 0.0, 0.0)
    #     gsize = int(1 / vox_size)
    #     grid = GridBatch(device=device)
    #     grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
    #     assert grid.total_voxels == 20**3
    #     for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
    #         grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
    #         grid_vals.requires_grad = True

    #         grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals)
    #         grid_vals_coarse = grid_vals_coarse.jdata
    #         if isinstance(pool_factor, int):
    #             self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
    #             self.assertTrue(
    #                 torch.allclose(
    #                     grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)
    #                 )
    #             )
    #         else:
    #             self.assertTrue(
    #                 torch.allclose(
    #                     grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)
    #                 )
    #             )
    #             self.assertTrue(
    #                 torch.allclose(
    #                     grid_coarse.origins[0],
    #                     grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device),
    #                 )
    #             )

    #         loss = (grid_vals_coarse.pow(3) * -1.111).sum()
    #         loss.backward()

    #         assert grid_vals.grad is not None  # Removes type errors with .grad

    #         grid_vals_grad = grid_vals.grad.clone()
    #         self.assertEqual(
    #             (grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
    #             grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1],
    #         )

    #         mask = grid_vals_grad.abs() > 0
    #         a = torch.sort(torch.tensor([x.item() for x in grid_vals[mask[:, 0]][:, 0]]))[0]
    #         b = torch.sort(torch.tensor([x.item() for x in grid_vals_coarse[:, 0]]))[0]
    #         self.assertEqual(torch.max(a - b).max().item(), 0)

    #         grid_vals.grad.zero_()

    #         # Pytorch pooling
    #         torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
    #         dense_grid = torch.zeros((gsize, gsize, gsize, 3), dtype=dtype, device=device)
    #         ijk = grid.ijk.jdata
    #         dense_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = grid_vals.detach()
    #         grid_vals_t = dense_grid.permute(3, 0, 1, 2)

    #         grid_vals_t.requires_grad = True

    #         grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

    #         grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
    #         for i, coord in enumerate(grid_coarse.ijk.jdata):
    #             grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

    #         self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    #         loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
    #         loss.backward()

    #         assert grid_vals_t.grad is not None  # Removes type errors with .grad

    #         grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device="cpu")
    #         grid_ijk_cpu = grid.ijk.jdata.cpu()
    #         grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
    #         for i, coord in enumerate(grid_ijk_cpu):
    #             grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
    #         grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

    #         expected_nnz = (
    #             grid_vals_t_coarse.shape[1]
    #             * grid_vals_t_coarse.shape[2]
    #             * grid_vals_t_coarse.shape[3]
    #             * grid_vals_t_coarse.shape[0]
    #         )
    #         self.assertEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(), expected_nnz)

    #         self.assertEqual(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item(), 0.0)

    @parameterized.expand(all_device_dtype_combos)
    def test_pickle(self, device, dtype):
        grid, _, _ = make_grid_and_point_data(device, dtype)
        pkl_str = pickle.dumps(grid)
        grid_2 = pickle.loads(pkl_str)
        self.assertTrue(torch.all(grid.ijk == grid_2.ijk))
        self.assertEqual(grid.device, grid_2.device)
        self.assertTrue(torch.all(grid.voxel_size == grid_2.voxel_size))
        self.assertTrue(torch.all(grid.origin == grid_2.origin))

    @parameterized.expand(all_device_dtype_combos)
    def test_grid_construction(self, device, dtype):
        rand_ijk = torch.randint(-100, 100, (1000, 3), device=device)
        rand_pts = torch.randn(1000, 3, device=device, dtype=dtype)

        def build_from_ijk(vsize, vorigin):
            return Grid.from_ijk(rand_ijk, vsize, vorigin, device=device)

        def build_from_pts(vsize, vorigin):
            return Grid.from_points(rand_pts, vsize, vorigin, device=device)

        def build_from_pts_nn(vsize, vorigin):
            return Grid.from_nearest_voxels_to_points(rand_pts, vsize, vorigin, device=device)

        def build_from_dense(vsize, vorigin):
            return Grid.from_dense([10, 10, 10], [0, 0, 0], vsize, vorigin, device=device)

        vox_size = np.random.rand(3) * 0.2 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid = Grid.from_points(pts, vox_size, vox_origin, device=device).dilated_grid(1)

        for builder in [build_from_ijk, build_from_pts, build_from_pts_nn, build_from_dense]:

            # Value error because of negative voxel size
            with self.assertRaises(ValueError):
                grid = builder(-vox_size, [0.01] * 3)

            # Value error because of negative voxel size
            with self.assertRaises(ValueError):
                grid = builder(-1.0, [0.01] * 3)

            # Value error because of zero voxel size
            with self.assertRaises(ValueError):
                grid = builder(vox_size * 0.0, [0.01] * 3)

            # Value error because of zero voxel size
            with self.assertRaises(ValueError):
                grid = builder(0.0, [0.01] * 3)

            # Value error because origins is wrong shape
            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01] * 4)

            # Value error because origins is wrong shape
            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01] * 2)

            # Type error because origins is a list of lists, rather than a list of numbers,
            # and is not the right python type to be rank 1.
            with self.assertRaises(TypeError):
                grid = builder(vox_size, [[0.01, 0.01, 0.01]])

            # These should work just fine. It's no longer an error to have a scalar
            # origin - it broadcasts just fine, so we accept it.
            grid = builder(vox_size, [0.01] * 3)
            grid = builder(vox_size, [0.01] * 1)

    # @parameterized.expand(all_device_dtype_combos)
    # def test_ijk_to_inv_index(self, device, dtype):
    #     vox_size = 0.1

    #     # Unique IJK since for duplicates the permutation is non-bijective
    #     ijk = list(set([tuple([a for a in (np.random.randn(3) / vox_size).astype(np.int32)]) for _ in range(10000)]))
    #     ijk = torch.from_numpy(np.array([list(a) for a in ijk])).to(torch.int32).to(device)

    #     grid = Grid.from_ijk(ijk, voxel_size=vox_size, origin=[0.0] * 3, device=device)

    #     inv_index = grid.ijk_to_inv_index(ijk)

    #     target_inv_index = torch.full_like(grid.ijk[:, 0], -1)
    #     idx = grid.ijk_to_index(ijk)
    #     for i in range(ijk.shape[0]):
    #         target_inv_index[idx[i]] = i

    #     self.assertTrue(torch.all(inv_index == target_inv_index))

    #     # Test functionality where size of ijk_to_inv_index's argument != len(grid.ijk)
    #     # Pick random ijk subset
    #     ijks = grid.ijk
    #     rand_ijks = torch.unique(ijks[torch.randint(len(ijks), (50,), device=ijks.device)], dim=0)

    #     rand_ijk_inv_indices = grid.ijk_to_inv_index(rand_ijks)

    #     # valid ijk indices
    #     inv_rand_ijk = grid.ijk[rand_ijk_inv_indices != -1]
    #     assert len(inv_rand_ijk) == len(rand_ijks)

    #     def check_order(t1: torch.Tensor, t2: torch.Tensor):
    #         t1_list = t1.tolist()
    #         t2_list = t2.tolist()

    #         last_index = -1
    #         for elem in t2_list:
    #             try:
    #                 current_index = t1_list.index(elem)
    #                 # Check if the current index is greater than the last index
    #                 if current_index > last_index:
    #                     last_index = current_index
    #                 else:
    #                     return False
    #             except ValueError:
    #                 return False
    #         return True

    #     for i, (inv_ijks, ijks) in enumerate(zip(inv_rand_ijk, rand_ijks)):
    #         # ensure output of ijk_to_inv_index is a permutation of the input
    #         inv_ijks_sorted, _ = torch.sort(inv_ijks, dim=0)
    #         ijks_sorted, _ = torch.sort(ijks, dim=0)
    #         assert torch.equal(inv_ijks_sorted, ijks_sorted)

    #         # ensure output of ijk_to_inv_index appears in ascending order in ijks
    #         assert check_order(grid.ijk, inv_ijks)

    @parameterized.expand(all_device_dtype_combos)
    def test_no_use_after_free_on_backward(self, device, dtype):

        grid, grid_d, p = make_grid_and_point_data(device, dtype)

        # Primal
        primal_features = torch.rand((grid.num_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_trilinear(p, primal_features)
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        del grid, grid_d
        fv.backward(grad_out)

    # @parameterized.expand(all_device_dtype_combos)
    # def test_ray_implicit_intersection(self, device, dtype):
    #     if dtype == torch.float16:
    #         return  # TODO: Implement rayimplicit marching for half

    #     # Generate the SDF for a sphere on a grid
    #     N = 32
    #     sphere_rad = 0.35
    #     (
    #         ii,
    #         jj,
    #         kk,
    #     ) = torch.meshgrid([torch.arange(N)] * 3, indexing="ij")
    #     xx, yy, zz = (
    #         ii.float() / (float(N) - 1) - 0.5,
    #         jj.float() / (float(N) - 1) - 0.5,
    #         kk.float() / (float(N) - 1) - 0.5,
    #     )
    #     sphere_sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad

    #     # Generate a bunch of points on the sphere which we'll send rays to
    #     cam_o = torch.tensor([0.0, 0.0, -2.0]).unsqueeze(0).repeat(100, 1)
    #     cam_targets = torch.randn(100, 3)
    #     cam_targets /= torch.norm(cam_targets, dim=-1, keepdim=True)
    #     cam_targets *= sphere_rad
    #     cam_targets += 0.5 - 0.5 / N
    #     cam_d = cam_targets - cam_o
    #     cam_d /= torch.norm(cam_d, dim=-1, keepdim=True)

    #     sphere_sdf, cam_o, cam_d = sphere_sdf.to(device), cam_o.to(device), cam_d.to(device)
    #     sphere_sdf, cam_o, cam_d = sphere_sdf.to(dtype), cam_o.to(dtype), cam_d.to(dtype)

    #     # Build a grid with the SDF
    #     grid = GridBatch(device=device)
    #     grid.set_from_dense_grid(
    #         1, [sphere_sdf.shape[i] for i in range(3)], [0] * 3, voxel_sizes=1.0 / N, origins=[0] * 3
    #     )
    #     sdf_p = grid.read_from_dense_xyzc(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()  # permuted sdf values

    #     # Intersect rays with the SDF
    #     isect = grid.ray_implicit_intersection(cam_o, cam_d, sdf_p.squeeze()).jdata
    #     hit_mask = isect >= 0.0
    #     self.assertTrue(hit_mask.sum().item() > 25)
    #     hit_pts = cam_o[hit_mask] + isect[hit_mask, None] * cam_d[hit_mask]

    #     # Sample intersected values and make sure they're within a half voxel from the true intersection
    #     sdf_samp = grid.sample_trilinear(hit_pts, sdf_p.unsqueeze(1)).jdata
    #     self.assertLess(sdf_samp.max().item(), 0.5 * torch.norm(grid.voxel_sizes[0]).item())

    #     # import polyscope as ps
    #     # ps.init()
    #     # xyz = grid.grid_to_world(grid.ijk.to(dtype))
    #     # pc = ps.register_point_cloud("sdf_p", xyz.cpu().numpy())
    #     # pc.add_scalar_quantity("sdf", sdf_p.cpu().numpy(), enabled=True)
    #     # pc.add_scalar_quantity("occ", sdf_p.cpu().numpy() <= 0.0, enabled=True)
    #     # ps.register_point_cloud("cam_target", cam_targets.numpy())
    #     # ps.register_point_cloud("cam_o", cam_o.cpu().numpy())
    #     # ps.register_point_cloud("hits", hit_pts.cpu().numpy())
    #     # ps.show()

    # @expand_tests(list(itertools.product(["cpu", "cuda"], [torch.float32, torch.float64])))
    # def test_marching_cubes(self, device, dtype):
    #     # Generate the SDF for a sphere on a grid
    #     N = 32 if device == "cpu" else 64
    #     sphere_rads = [0.5, 0.33, 0.3, 0.28, 0.25]
    #     for batch_size in [1, 3, 5]:
    #         # Build a dense tensor of SDF values
    #         (
    #             ii,
    #             jj,
    #             kk,
    #         ) = torch.meshgrid(
    #             [torch.arange(N, device=device)] * 3, indexing="ij"
    #         )  # index space [0, N-1]
    #         xx, yy, zz = (
    #             ii.float() / (float(N) - 1) - 0.5,
    #             jj.float() / (float(N) - 1) - 0.5,
    #             kk.float() / (float(N) - 1) - 0.5,
    #         )  # normalize to [-1, 1]
    #         sphere_sdf = torch.stack(
    #             [-torch.sqrt(xx**2 + yy**2 + zz**2) + sphere_rad for sphere_rad in sphere_rads[:batch_size]]
    #         ).unsqueeze(
    #             -1
    #         )  # [B, N, N, N, 1] sdf

    #         # Build a grid with the SDF
    #         grid = GridBatch(device=device)
    #         grid.set_from_dense_grid(
    #             batch_size, [sphere_sdf[0].shape[i] for i in range(3)], [0] * 3, voxel_sizes=1.0 / N, origins=[0] * 3
    #         )
    #         sdf_p = grid.read_from_dense_xyzc(sphere_sdf)  # permuted sdf values

    #         for level in [0.0, 0.2, -0.2]:
    #             v, f, _ = grid.marching_cubes(sdf_p, level)

    #             for bi in range(batch_size):
    #                 mesh_radius = torch.linalg.norm(
    #                     v[bi].jdata - torch.tensor([[0.5] * 3], device=device, dtype=dtype), axis=1
    #                 )
    #                 vox_size = torch.norm(grid.voxel_sizes[bi])
    #                 self.assertTrue(torch.all(mesh_radius - sphere_rads[bi] < vox_size / 2.0 - level))
    #                 self.assertTrue(torch.all(torch.logical_and(f[bi].jdata >= 0, f[bi].jdata < v[bi].jdata.shape[0])))
    #             # import polyscope as ps
    #             # ps.init()
    #             # ps.register_surface_mesh("marching_cubes", v.cpu()[0].jdata.numpy(), f.cpu()[0].jdata.numpy())
    #             # ps.show()

    @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    def test_subdivide_empty_grid(self, device, dtype):
        grid = Grid.from_dense([32, 32, 32], [0, 0, 0], voxel_size=1.0 / 32, origin=[0, 0, 0], device=device)
        values = torch.randn(grid.num_voxels, 17, device=device, dtype=dtype)
        values, subgrid = grid.refine(1, values, mask=torch.zeros(grid.num_voxels, dtype=torch.bool, device=device))
        self.assertTrue(subgrid.num_voxels == 0)
        self.assertTrue(values.shape[0] == 0)
        self.assertTrue(values.shape[1] == 17)

    # @parameterized.expand(all_device_dtype_combos + bfloat16_combos)
    # def test_conv_empty_grid(self, device, dtype):
    #     grid = GridBatch(device=device)
    #     grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
    #     values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
    #     values, subgrid = grid.subdivide(
    #         1, values_in, mask=torch.zeros(grid.total_voxels, dtype=torch.bool, device=device)
    #     )
    #     self.assertTrue(subgrid.total_voxels == 0)
    #     self.assertTrue(values.rshape[0] == 0)
    #     self.assertTrue(values.rshape[1] == 17)

    #     kmap, _ = grid.sparse_conv_kernel_map(3, 1, target_grid=subgrid)
    #     kmap.build_gather_scatter()
    #     res = kmap.sparse_conv_3d(values_in, torch.randn(17, 17, 3, 3, 3))
    #     self.assertTrue(res.rshape[0] == 0)
    #     self.assertTrue(res.rshape[1] == 17)

    @parameterized.expand(all_device_dtype_combos)
    def test_zero_voxels_grid_construction(self, device, dtype):
        """Test Grid.from_empty() creates an empty grid with correct properties"""
        # Test with default device
        grid = Grid.from_zero_voxels()
        self.assertEqual(grid.device.type, "cpu")
        self.assertEqual(grid.num_voxels, 0)
        self.assertTrue(torch.equal(grid.bbox, torch.zeros(2, 3, dtype=torch.int32, device="cpu")))

        # Test with specified device
        grid = Grid.from_zero_voxels(device=device)
        self.assertEqual(grid.device.type, device)
        self.assertEqual(grid.num_voxels, 0)
        self.assertTrue(torch.equal(grid.bbox, torch.zeros(2, 3, dtype=torch.int32, device=device)))

        # Test with torch.device object
        torch_device = torch.device(device)
        grid = Grid.from_zero_voxels(device=torch_device)
        self.assertEqual(grid.device.type, device)
        self.assertEqual(grid.num_voxels, 0)

    @parameterized.expand(all_device_dtype_combos)
    def test_bbox_attrs(self, device, dtype):
        grid = Grid.from_zero_voxels(device=device)
        print(f"Empty grid bbox: {grid.bbox}")
        self.assertTrue(torch.equal(grid.bbox, torch.zeros(2, 3, dtype=torch.float32, device=device)))

        grid = Grid.from_dense([32, 32, 32], [0, 0, 0], voxel_size=1.0 / 32, origin=[0, 0, 0], device=device)
        self.assertTrue(torch.equal(grid.bbox, torch.tensor([[0, 0, 0], [31, 31, 31]], device=device)))
        self.assertTrue(torch.equal(grid.dual_bbox, torch.tensor([[0, 0, 0], [32, 32, 32]], device=device)))

    @parameterized.expand(all_device_dtype_combos)
    def test_clip_grid(self, device, dtype):

        grid = Grid.from_dense([32, 32, 32], [0, 0, 0], voxel_size=1.0 / 32, origin=[0, 0, 0], device=device)
        values_in = torch.randn(grid.num_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(values_in, [0, 0, 0], [5, 5, 5])
        self.assertTrue(clipped_grid.num_voxels == 6**3)
        self.assertTrue(clipped_data.shape[0] == 6**3)

        grid = Grid.from_dense([32, 32, 32], [-2, -2, -2], voxel_size=1.0 / 32, origin=[0, 0, 0], device=device)
        values_in = torch.randn(grid.num_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(values_in, [-2, -2, -2], [5, 5, 5])
        self.assertTrue(clipped_grid.num_voxels == 8**3)
        self.assertTrue(clipped_data.shape[0] == 8**3)

        # Test gradients through clip
        num_features = 17
        grid = Grid.from_dense([32, 32, 32], [0, 0, 0], voxel_size=1.0 / 32, origin=[0, 0, 0], device=device)
        features = torch.randn(grid.num_voxels, num_features, device=device, dtype=dtype, requires_grad=True)

        clipped_features, clipped_grid = grid.clip(features, [0, 0, 0], [5, 5, 5])

        loss = clipped_features.pow(3).sum()
        loss.backward()

        assert features.grad is not None  # Removes type errors with .grad
        clipped_features_grad = features.grad.clone()

        features.grad.zero_()
        self.assertTrue(torch.all(features.grad == torch.zeros_like(features.grad)))
        self.assertTrue(not torch.all(features.grad == clipped_features_grad))

        ijk_clip_mask = torch.all(grid.ijk <= 5, 1)

        loss = (features[ijk_clip_mask.repeat(num_features, 1).swapaxes(0, 1)].pow(3)).sum()
        loss.backward()
        self.assertTrue(torch.equal(clipped_features_grad, features.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_dual_without_border(self, device, dtype):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(dtype).to(device)

        pts = torch.randn(np.random.randint(100_000, 300_000), 3).to(device=device, dtype=dtype)
        grid = Grid.from_points(pts, vox_size, vox_origin, device=device)
        dual_grid = grid.dual_grid()

        neighbors = grid.neighbor_indexes(dual_grid.ijk, 1)
        inner_mask = torch.all(neighbors[:, 1:, 1:, 1:].reshape(-1, 8) != -1, dim=-1)
        inner_ijk = dual_grid.ijk[inner_mask]
        dual_inner = Grid.from_ijk(inner_ijk, voxel_size=vox_size, origin=vox_origin, device=device)

        dual_outer_with_skip = grid.dual_grid(exclude_border=True)

        ijk1 = dual_inner.ijk
        ijk2 = dual_outer_with_skip.ijk
        ijk1_i = set([tuple(ijk1[j].cpu().numpy().tolist()) for j in range(ijk1.shape[0])])
        ijk2_i = set([tuple(ijk2[j].cpu().numpy().tolist()) for j in range(ijk2.shape[0])])
        self.assertTrue(ijk1_i == ijk2_i)


if __name__ == "__main__":
    unittest.main()
