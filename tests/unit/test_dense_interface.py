# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import numpy as np
import torch
from parameterized import parameterized

from fvdb import Grid, GridBatch, JaggedTensor

all_device_dtype_combos = [
    ["cpu", torch.float16],
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

all_device_combos = [
    ["cpu"],
    ["cuda"],
]


class TestDenseInterfaceBatch(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_dense(self, device, dtype):
        dense_vdb = GridBatch.from_dense_axis_aligned_bounds(
            num_grids=1,
            dense_dims=[10, 11, 12],
            bounds_min=[-2.0, -2.0, -2.0],
            bounds_max=[1.0, 1.0, 1.0],
            voxel_center=False,
            device=device,
        )
        self.assertTrue(dense_vdb.total_voxels == 10 * 11 * 12)

        vdb_coords = dense_vdb.grid_to_world(dense_vdb.ijk.float()).jdata
        self.assertAlmostEqual(torch.min(vdb_coords).item(), -2.0 + 3 / 12 * 0.5, places=6)
        self.assertAlmostEqual(torch.max(vdb_coords).item(), 1.0 - 3 / 12 * 0.5, places=6)

        vdb_feature = torch.randn((dense_vdb.total_voxels, 4), device=device, dtype=dtype)
        dense_feature = dense_vdb.write_to_dense_cminor(JaggedTensor(vdb_feature)).squeeze(0)
        for i in range(10):
            for j in range(11):
                for k in range(12):
                    vdb_f = vdb_feature[
                        dense_vdb.ijk_to_index(JaggedTensor(torch.tensor([[i, j, k]], device=device))).jdata
                    ]
                    dense_f = dense_feature[i, j, k, :]
                    self.assertTrue(torch.allclose(vdb_f, dense_f))
        vdb_feature2 = dense_vdb.read_from_dense_cminor(dense_feature.unsqueeze(0)).jdata
        self.assertTrue(torch.allclose(vdb_feature, vdb_feature2))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, device=device, dtype=dtype)
        ijk = grid.ijk.jdata

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=-1)
            )

            grid_index = grid.ijk_to_index(JaggedTensor(ijk)).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * dense_size[2] * dense_size[1] + j * dense_size[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.read_from_dense_cminor(random_grid.unsqueeze(0), dense_origin).jdata

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_multidim(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        ijk = grid.ijk.jdata

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1)
            )

            grid_index = grid.ijk_to_index(JaggedTensor(ijk)).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid.shape[2] * random_grid.shape[1] + j * random_grid.shape[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.read_from_dense_cminor(random_grid.unsqueeze(0), dense_origin).jdata

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_multidim_grad(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        random_grid_copy = random_grid.clone()
        random_grid.requires_grad = True
        random_grid_copy.requires_grad = True

        ijk = grid.ijk.jdata

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1)
            )

            grid_index = grid.ijk_to_index(JaggedTensor(ijk)).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid_copy.shape[2] * random_grid_copy.shape[1] + j * random_grid_copy.shape[2] + k

            feat_shape = [c for c in random_grid_copy.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid_copy.view(-1, *feat_shape)[offset]
            loss_copy = target_sparse.sum()
            loss_copy.backward()

            pred_sparse = grid.read_from_dense_cminor(random_grid.unsqueeze(0), dense_origin).jdata
            loss = pred_sparse.sum()
            loss.backward()

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

            assert random_grid.grad is not None
            assert random_grid_copy.grad is not None
            self.assertTrue(torch.equal(random_grid.grad, random_grid_copy.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        ijk = grid.ijk.jdata
        sparse_data = torch.randn((grid.total_voxels, 4), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), sparse_data.shape[-1], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, sparse_data.shape[-1])[idx] = sparse_data[keep_mask]

            pred_crop = grid.write_to_dense_cminor(JaggedTensor(sparse_data), crop_min, crop_size).squeeze(0)

            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_multidim(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        ijk = grid.ijk.jdata
        sparse_data = torch.randn((grid.total_voxels, 4, 3, 2), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data[keep_mask]

            pred_crop = grid.write_to_dense_cminor(JaggedTensor(sparse_data), crop_min, crop_size).squeeze(0)

            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_multidim_grad(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)

        ijk = grid.ijk.jdata
        sparse_data = torch.randn((grid.total_voxels, 4, 3, 2), device=device, dtype=dtype)
        sparse_data_copy = sparse_data.clone()
        sparse_data.requires_grad = True
        sparse_data_copy.requires_grad = True

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data_copy[keep_mask]

            loss_copy = target_crop.sum()
            loss_copy.backward()

            pred_crop = grid.write_to_dense_cminor(JaggedTensor(sparse_data), crop_min, crop_size).squeeze(0)
            loss = pred_crop.sum()
            loss.backward()

            assert sparse_data.grad is not None
            assert sparse_data_copy.grad is not None
            self.assertEqual(torch.abs(sparse_data.grad - sparse_data_copy.grad).max().item(), 0.0)
            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_cmajor_dense_grid(self, device, dtype):

        dims = [11, 6, 8]
        grid = GridBatch.from_dense(1, dims, 0, 1, 0, device=device)
        total_voxels = grid.total_voxels

        min_coord = torch.tensor([0, 0, 0], device=device)
        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            sparse_data = torch.randn((total_voxels, *eshape), device=device, dtype=dtype)

            dense_default = grid.write_to_dense_cminor(JaggedTensor(sparse_data), min_coord, dense_size)
            dense_conv = grid.write_to_dense_cmajor(JaggedTensor(sparse_data), min_coord, dense_size)

            self.assertEqual(dense_default.shape, (1, dims[0], dims[1], dims[2], *eshape))
            self.assertEqual(dense_conv.shape, (1, *eshape, dims[0], dims[1], dims[2]))

            n = len(eshape)
            conv_to_default_permute_order = (0, n + 1, n + 2, n + 3, *range(1, 1 + n))
            if n == 1:
                assert conv_to_default_permute_order == (0, 2, 3, 4, 1)

            dense_conv_like_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            print(f"dense_default.shape: {dense_default.shape}")
            print(f"dense_conv_like_default.shape: {dense_conv_like_default.shape}")

            self.assertEqual(dense_default.shape, dense_conv_like_default.shape)

            self.assertTrue(torch.equal(dense_default, dense_conv_like_default))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_cmajor_rand_point_grid(self, device, dtype):

        # Build a random sparse grid
        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch.from_points(JaggedTensor(random_points), voxel_sizes=0.1, origins=[0.0] * 3)
        total_voxels = grid.total_voxels

        ijk = grid.ijk.jdata
        min_coord = ijk.min(0).values
        max_coord = ijk.max(0).values
        dims = max_coord - min_coord + 1

        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            sparse_data = torch.randn((total_voxels, *eshape), device=device, dtype=dtype)

            dense_default = grid.write_to_dense_cminor(JaggedTensor(sparse_data), min_coord, dense_size)
            dense_conv = grid.write_to_dense_cmajor(JaggedTensor(sparse_data), min_coord, dense_size)

            self.assertEqual(dense_default.shape, (1, dims[0], dims[1], dims[2], *eshape))
            self.assertEqual(dense_conv.shape, (1, *eshape, dims[0], dims[1], dims[2]))

            n = len(eshape)
            conv_to_default_permute_order = (0, n + 1, n + 2, n + 3, *range(1, 1 + n))
            if n == 1:
                assert conv_to_default_permute_order == (0, 2, 3, 4, 1)

            dense_conv_like_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            print(f"dense_default.shape: {dense_default.shape}")
            print(f"dense_conv_like_default.shape: {dense_conv_like_default.shape}")

            self.assertEqual(dense_default.shape, dense_conv_like_default.shape)

            self.assertTrue(torch.equal(dense_default, dense_conv_like_default))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_cmajor_dense_grid(self, device, dtype):

        dims = [11, 6, 8]
        grid = GridBatch.from_dense(1, dims, 0, 1, 0, device=device)
        total_voxels = grid.total_voxels

        min_coord = torch.tensor([0, 0, 0], device=device)
        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            dense_conv = torch.randn((1, *eshape, dims[2], dims[1], dims[0]), device=device, dtype=dtype)

            n = len(eshape)
            conv_to_default_permute_order = (0, n + 1, n + 2, n + 3, *range(1, 1 + n))
            if n == 1:
                assert conv_to_default_permute_order == (0, 2, 3, 4, 1)
            dense_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            sparse_conv = grid.read_from_dense_cmajor(dense_conv, min_coord)
            sparse_default = grid.read_from_dense_cminor(dense_default, min_coord)

            self.assertEqual(sparse_conv.jdata.shape, (total_voxels, *eshape))
            self.assertEqual(sparse_default.jdata.shape, (total_voxels, *eshape))

            self.assertTrue(torch.equal(sparse_conv.jdata, sparse_default.jdata))

    @parameterized.expand(all_device_combos)
    def test_build_from_dense(self, device):
        gorigin = tuple([int(a.item()) for a in torch.randint(-32, 32, (3,))])
        gsize = torch.randint(4, 8, (3,))
        gsize = (int(gsize[0].item()), int(gsize[1].item()), int(gsize[2].item()))
        num_dense_vox = gsize[0] * gsize[1] * gsize[2]

        grid = GridBatch.from_dense(1, gsize, gorigin, 0.1, [0.0] * 3, device=device)
        grid_ijk = grid.ijk.jdata
        target_min_ijk = torch.zeros(3).to(grid_ijk) + torch.tensor(gorigin).to(grid_ijk)
        target_max_ijk = torch.tensor(gsize).to(grid_ijk) - 1 + torch.tensor(gorigin).to(grid_ijk)

        self.assertEqual(grid.total_voxels, num_dense_vox)
        self.assertTrue(torch.all(grid_ijk.min(0)[0] == target_min_ijk))
        self.assertTrue(torch.all(grid_ijk.max(0)[0] == target_max_ijk))

        ijk_mask = torch.stack(
            [
                torch.randint(0, gsize[0], (100,), device=device),
                torch.randint(0, gsize[1], (100,), device=device),
                torch.randint(0, gsize[2], (100,), device=device),
            ],
            dim=-1,
        )
        dense_mask = torch.zeros(*gsize, dtype=torch.bool).to(device)
        mask_coord_set = set()
        for idx in range(ijk_mask.shape[0]):
            i, j, k = [int(a.item()) for a in ijk_mask[idx]]
            dense_mask[i, j, k] = True
            mask_coord_set.add((i, j, k))

        grid = GridBatch.from_dense(1, gsize, gorigin, 0.1, [0.0] * 3, mask=dense_mask, device=device)
        grid_ijk = grid.ijk.jdata

        self.assertEqual(len(mask_coord_set), grid_ijk.shape[0])

        grid_ijk -= torch.tensor(gorigin).unsqueeze(0).to(grid_ijk)
        pred_set = set()
        for idx in range(grid_ijk.shape[0]):
            i, j, k = [a.item() for a in grid_ijk[idx]]
            pred_set.add((i, j, k))

        self.assertEqual(pred_set, mask_coord_set)


class TestDenseInterfaceSingle(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_dense(self, device, dtype):
        dense_vdb = Grid.from_dense_axis_aligned_bounds(
            dense_dims=[10, 11, 12],
            bounds_min=[-2.0, -2.0, -2.0],
            bounds_max=[1.0, 1.0, 1.0],
            voxel_center=False,
            device=device,
        )
        self.assertTrue(dense_vdb.num_voxels == 10 * 11 * 12)

        vdb_coords = dense_vdb.voxel_to_world(dense_vdb.ijk.float())
        self.assertAlmostEqual(torch.min(vdb_coords).item(), -2.0 + 3 / 12 * 0.5, places=6)
        self.assertAlmostEqual(torch.max(vdb_coords).item(), 1.0 - 3 / 12 * 0.5, places=6)

        vdb_feature = torch.randn((dense_vdb.num_voxels, 4), device=device, dtype=dtype)
        dense_feature = dense_vdb.inject_to_dense_cminor(vdb_feature)
        for i in range(10):
            for j in range(11):
                for k in range(12):
                    vdb_f = vdb_feature[dense_vdb.ijk_to_index(torch.tensor([[i, j, k]], device=device))]
                    dense_f = dense_feature[i, j, k, :]
                    self.assertTrue(torch.allclose(vdb_f, dense_f))
        vdb_feature2 = dense_vdb.inject_from_dense_cminor(dense_feature)
        self.assertTrue(torch.allclose(vdb_feature, vdb_feature2))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, device=device, dtype=dtype)
        ijk = grid.ijk

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=-1)
            )

            grid_index = grid.ijk_to_index(ijk)[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * dense_size[2] * dense_size[1] + j * dense_size[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.num_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.inject_from_dense_cminor(random_grid, dense_origin)

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_multidim(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        ijk = grid.ijk

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1)
            )

            grid_index = grid.ijk_to_index(ijk)[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid.shape[2] * random_grid.shape[1] + j * random_grid.shape[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.num_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.inject_from_dense_cminor(random_grid, dense_origin)

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_multidim_grad(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        random_grid_copy = random_grid.clone()
        random_grid.requires_grad = True
        random_grid_copy.requires_grad = True

        ijk = grid.ijk

        for _ in range(10):
            dense_origin = torch.tensor(
                [
                    np.random.randint(low=int(ijk.min(0).values[i].item()), high=int(ijk.max(0).values[i].item()))
                    for i in range(3)
                ],
                dtype=torch.long,
                device=device,
            )

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1)
            )

            grid_index = grid.ijk_to_index(ijk)[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid_copy.shape[2] * random_grid_copy.shape[1] + j * random_grid_copy.shape[2] + k

            feat_shape = [c for c in random_grid_copy.shape[3:]]
            target_sparse = torch.zeros(grid.num_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid_copy.view(-1, *feat_shape)[offset]
            loss_copy = target_sparse.sum()
            loss_copy.backward()

            pred_sparse = grid.inject_from_dense_cminor(random_grid, dense_origin)
            loss = pred_sparse.sum()
            loss.backward()

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

            assert random_grid.grad is not None
            assert random_grid_copy.grad is not None
            self.assertTrue(torch.equal(random_grid.grad, random_grid_copy.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        ijk = grid.ijk
        sparse_data = torch.randn((grid.num_voxels, 4), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), sparse_data.shape[-1], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, sparse_data.shape[-1])[idx] = sparse_data[keep_mask]

            pred_crop = grid.inject_to_dense_cminor(sparse_data, crop_min, crop_size)

            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_multidim(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        ijk = grid.ijk
        sparse_data = torch.randn((grid.num_voxels, 4, 3, 2), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data[keep_mask]

            pred_crop = grid.inject_to_dense_cminor(sparse_data, crop_min, crop_size)

            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_multidim_grad(self, device, dtype):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)

        ijk = grid.ijk
        sparse_data = torch.randn((grid.num_voxels, 4, 3, 2), device=device, dtype=dtype)
        sparse_data_copy = sparse_data.clone()
        sparse_data.requires_grad = True
        sparse_data_copy.requires_grad = True

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor(
                [
                    np.random.randint(low=int(min_crop_coord[i].item()), high=int(max_crop_coord[i].item()))
                    for i in range(3)
                ],
                device=device,
            )
            crop_size = torch.tensor(
                [np.random.randint(low=1, high=int(max_crop_size[i].item())) for i in range(3)], device=device
            )

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                torch.all(ijk_offset >= 0, dim=-1), torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1)
            )
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data_copy[keep_mask]

            loss_copy = target_crop.sum()
            loss_copy.backward()

            pred_crop = grid.inject_to_dense_cminor(sparse_data, crop_min, crop_size)
            loss = pred_crop.sum()
            loss.backward()

            assert sparse_data.grad is not None
            assert sparse_data_copy.grad is not None
            self.assertEqual(torch.abs(sparse_data.grad - sparse_data_copy.grad).max().item(), 0.0)
            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_cmajor_dense_grid(self, device, dtype):

        dims = [11, 6, 8]
        grid = Grid.from_dense(dims, 0, 1, 0, device=device)
        total_voxels = grid.num_voxels

        min_coord = torch.tensor([0, 0, 0], device=device)
        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            sparse_data = torch.randn((total_voxels, *eshape), device=device, dtype=dtype)

            dense_default = grid.inject_to_dense_cminor(sparse_data, min_coord, dense_size)
            dense_conv = grid.inject_to_dense_cmajor(sparse_data, min_coord, dense_size)

            self.assertEqual(dense_default.shape, (dims[0], dims[1], dims[2], *eshape))
            self.assertEqual(dense_conv.shape, (*eshape, dims[0], dims[1], dims[2]))

            n = len(eshape)
            conv_to_default_permute_order = (n, n + 1, n + 2, *range(0, n))
            if n == 1:
                assert conv_to_default_permute_order == (1, 2, 3, 0)

            dense_conv_like_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            print(f"dense_default.shape: {dense_default.shape}")
            print(f"dense_conv_like_default.shape: {dense_conv_like_default.shape}")

            self.assertEqual(dense_default.shape, dense_conv_like_default.shape)

            self.assertTrue(torch.equal(dense_default, dense_conv_like_default))

    @parameterized.expand(all_device_dtype_combos)
    def test_write_to_dense_cminor_cmajor_rand_point_grid(self, device, dtype):

        # Build a random sparse grid
        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = Grid.from_points(random_points, voxel_size=0.1, origin=[0.0] * 3)
        total_voxels = grid.num_voxels

        ijk = grid.ijk
        min_coord = ijk.min(0).values
        max_coord = ijk.max(0).values
        dims = max_coord - min_coord + 1

        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            sparse_data = torch.randn((total_voxels, *eshape), device=device, dtype=dtype)

            dense_default = grid.inject_to_dense_cminor(sparse_data, min_coord, dense_size)
            dense_conv = grid.inject_to_dense_cmajor(sparse_data, min_coord, dense_size)

            self.assertEqual(dense_default.shape, (dims[0], dims[1], dims[2], *eshape))
            self.assertEqual(dense_conv.shape, (*eshape, dims[0], dims[1], dims[2]))

            n = len(eshape)
            conv_to_default_permute_order = (n, n + 1, n + 2, *range(0, n))
            if n == 1:
                assert conv_to_default_permute_order == (1, 2, 3, 0)

            dense_conv_like_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            print(f"dense_default.shape: {dense_default.shape}")
            print(f"dense_conv_like_default.shape: {dense_conv_like_default.shape}")

            self.assertEqual(dense_default.shape, dense_conv_like_default.shape)

            self.assertTrue(torch.equal(dense_default, dense_conv_like_default))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_cminor_cmajor_dense_grid(self, device, dtype):

        dims = [11, 6, 8]
        grid = Grid.from_dense(dims, 0, 1, 0, device=device)
        total_voxels = grid.num_voxels

        min_coord = torch.tensor([0, 0, 0], device=device)
        dense_size = torch.tensor(dims, device=device)

        # Single-channel and multi-dimensional channel shapes
        for eshape in [(3,), (5, 7)]:
            dense_conv = torch.randn((*eshape, dims[0], dims[1], dims[2]), device=device, dtype=dtype)

            n = len(eshape)
            conv_to_default_permute_order = (n, n + 1, n + 2, *range(0, n))
            if n == 1:
                assert conv_to_default_permute_order == (1, 2, 3, 0)
            dense_default = dense_conv.permute(*conv_to_default_permute_order).contiguous()

            sparse_conv = grid.inject_from_dense_cmajor(dense_conv, min_coord)
            sparse_default = grid.inject_from_dense_cminor(dense_default, min_coord)

            self.assertEqual(sparse_conv.shape, (total_voxels, *eshape))
            self.assertEqual(sparse_default.shape, (total_voxels, *eshape))

            self.assertTrue(torch.equal(sparse_conv, sparse_default))

    @parameterized.expand(all_device_combos)
    def test_build_from_dense(self, device):
        gorigin = tuple([int(a.item()) for a in torch.randint(-32, 32, (3,))])
        gsize = torch.randint(4, 8, (3,))
        gsize = (int(gsize[0].item()), int(gsize[1].item()), int(gsize[2].item()))
        num_dense_vox = gsize[0] * gsize[1] * gsize[2]

        grid = Grid.from_dense(gsize, gorigin, 0.1, [0.0] * 3, device=device)
        grid_ijk = grid.ijk
        target_min_ijk = torch.zeros(3).to(grid_ijk) + torch.tensor(gorigin).to(grid_ijk)
        target_max_ijk = torch.tensor(gsize).to(grid_ijk) - 1 + torch.tensor(gorigin).to(grid_ijk)

        self.assertEqual(grid.num_voxels, num_dense_vox)
        self.assertTrue(torch.all(grid_ijk.min(0)[0] == target_min_ijk))
        self.assertTrue(torch.all(grid_ijk.max(0)[0] == target_max_ijk))

        ijk_mask = torch.stack(
            [
                torch.randint(0, gsize[0], (100,), device=device),
                torch.randint(0, gsize[1], (100,), device=device),
                torch.randint(0, gsize[2], (100,), device=device),
            ],
            dim=-1,
        )
        dense_mask = torch.zeros(*gsize, dtype=torch.bool).to(device)
        mask_coord_set = set()
        for idx in range(ijk_mask.shape[0]):
            i, j, k = [int(a.item()) for a in ijk_mask[idx]]
            dense_mask[i, j, k] = True
            mask_coord_set.add((i, j, k))

        grid = Grid.from_dense(gsize, gorigin, 0.1, [0.0] * 3, mask=dense_mask, device=device)
        grid_ijk = grid.ijk

        self.assertEqual(len(mask_coord_set), grid_ijk.shape[0])

        grid_ijk -= torch.tensor(gorigin).unsqueeze(0).to(grid_ijk)
        pred_set = set()
        for idx in range(grid_ijk.shape[0]):
            i, j, k = [a.item() for a in grid_ijk[idx]]
            pred_set.add((i, j, k))

        self.assertEqual(pred_set, mask_coord_set)


if __name__ == "__main__":
    unittest.main()
