# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os
import unittest
from collections.abc import MutableMapping
from typing import NamedTuple

import fvdb.nn as fvnn
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn as nn
from parameterized import parameterized

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

all_device_dtype_combos = [
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


class TestNNModules(unittest.TestCase):

    def _make_dense_grid(self, device, batch_size=1, shape=(10, 10, 10)):
        return GridBatch.from_dense(batch_size, list(shape), voxel_sizes=0.1, origins=(0.0, 0.0, 0.0), device=device)

    def _make_features(self, grid, num_channels, device, dtype):
        data = torch.randn(grid.total_voxels, num_channels, device=device, dtype=dtype)
        return grid.jagged_like(data)

    # =========================================================================
    # AvgPool
    # =========================================================================

    def test_avg_pool_construction(self):
        pool = fvnn.AvgPool(kernel_size=2)
        self.assertEqual(pool.kernel_size.item(), 2)
        self.assertTrue(torch.equal(pool.stride, pool.kernel_size))

        pool = fvnn.AvgPool(kernel_size=(2, 3, 4), stride=1)
        self.assertTrue(torch.equal(pool.kernel_size, torch.tensor([2, 3, 4])))
        self.assertEqual(pool.stride.item(), 1)

        self.assertIn("kernel_size", pool.extra_repr())
        self.assertIn("stride", pool.extra_repr())

    @parameterized.expand(all_device_dtype_combos)
    def test_avg_pool_forward(self, device, dtype):
        grid = self._make_dense_grid(device)
        features = self._make_features(grid, 4, device, dtype)

        for pool_factor in (2, 5):
            pool = fvnn.AvgPool(kernel_size=pool_factor)
            pooled_data, coarse_grid = pool(features, grid)

            expected_data, expected_grid = grid.avg_pool(pool.kernel_size, features, stride=pool.stride)

            self.assertEqual(pooled_data.jdata.shape, expected_data.jdata.shape)
            self.assertTrue(torch.equal(pooled_data.jdata, expected_data.jdata))
            self.assertEqual(coarse_grid.total_voxels, expected_grid.total_voxels)

    @parameterized.expand(all_device_dtype_combos)
    def test_avg_pool_with_coarse_grid(self, device, dtype):
        grid = self._make_dense_grid(device)
        features = self._make_features(grid, 4, device, dtype)

        _, coarse_grid = grid.avg_pool(2, features, stride=2)

        pool = fvnn.AvgPool(kernel_size=2)
        pooled_data, result_grid = pool(features, grid, coarse_grid=coarse_grid)

        self.assertEqual(result_grid.total_voxels, coarse_grid.total_voxels)
        self.assertEqual(pooled_data.jdata.shape[1], 4)

    # =========================================================================
    # MaxPool
    # =========================================================================

    def test_max_pool_construction(self):
        pool = fvnn.MaxPool(kernel_size=3)
        self.assertEqual(pool.kernel_size.item(), 3)
        self.assertTrue(torch.equal(pool.stride, pool.kernel_size))

        pool = fvnn.MaxPool(kernel_size=(2, 4, 6), stride=2)
        self.assertTrue(torch.equal(pool.kernel_size, torch.tensor([2, 4, 6])))
        self.assertEqual(pool.stride.item(), 2)

        self.assertIn("kernel_size", pool.extra_repr())

    @parameterized.expand(all_device_dtype_combos)
    def test_max_pool_forward(self, device, dtype):
        grid = self._make_dense_grid(device)
        features = self._make_features(grid, 4, device, dtype)

        for pool_factor in (2, 5):
            pool = fvnn.MaxPool(kernel_size=pool_factor)
            pooled_data, coarse_grid = pool(features, grid)

            self.assertGreater(pooled_data.jdata.shape[0], 0)
            self.assertEqual(pooled_data.jdata.shape[1], 4)
            self.assertGreater(coarse_grid.total_voxels, 0)
            self.assertFalse(torch.any(torch.isinf(pooled_data.jdata)), "MaxPool output should not contain inf values")

    @parameterized.expand(all_device_dtype_combos)
    def test_max_pool_inf_zeroing(self, device, dtype):
        """Verify MaxPool zeros out inf values that grid.max_pool produces for uncovered voxels."""
        fine_ijk = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.int32, device=device)
        fine_grid = GridBatch.from_ijk(JaggedTensor(fine_ijk))
        features = fine_grid.jagged_like(torch.randn(fine_grid.total_voxels, 4, device=device, dtype=dtype))

        # Coarse grid with a voxel far from any fine voxel
        coarse_ijk = torch.tensor([[0, 0, 0], [100, 100, 100]], dtype=torch.int32, device=device)
        coarse_grid = GridBatch.from_ijk(JaggedTensor(coarse_ijk))

        raw_data, _ = fine_grid.max_pool(2, features, stride=2, coarse_grid=coarse_grid)

        pool = fvnn.MaxPool(kernel_size=2)
        pooled_data, _ = pool(features, fine_grid, coarse_grid=coarse_grid)

        if torch.any(torch.isinf(raw_data.jdata)):
            self.assertFalse(torch.any(torch.isinf(pooled_data.jdata)))
            inf_mask = torch.isinf(raw_data.jdata)
            self.assertTrue(torch.all(pooled_data.jdata[inf_mask] == 0.0))

    # =========================================================================
    # UpsamplingNearest
    # =========================================================================

    def test_upsampling_construction(self):
        up = fvnn.UpsamplingNearest(scale_factor=2)
        self.assertEqual(up.scale_factor.item(), 2)
        self.assertIn("scale_factor", up.extra_repr())

        up = fvnn.UpsamplingNearest(scale_factor=(2, 3, 4))
        self.assertTrue(torch.equal(up.scale_factor, torch.tensor([2, 3, 4])))

    @parameterized.expand(all_device_dtype_combos)
    def test_upsampling_forward(self, device, dtype):
        coarse_grid = self._make_dense_grid(device, shape=(5, 5, 5))
        coarse_features = self._make_features(coarse_grid, 4, device, dtype)

        up = fvnn.UpsamplingNearest(scale_factor=2)
        fine_data, fine_grid = up(coarse_features, coarse_grid)

        expected_data, expected_grid = coarse_grid.refine(up.scale_factor, coarse_features)

        self.assertEqual(fine_data.jdata.shape, expected_data.jdata.shape)
        self.assertTrue(torch.equal(fine_data.jdata, expected_data.jdata))
        self.assertEqual(fine_grid.total_voxels, expected_grid.total_voxels)

    # =========================================================================
    # SparseConv3d
    # =========================================================================

    def test_sparse_conv3d_construction(self):
        conv = fvnn.SparseConv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 32)
        self.assertTrue(torch.equal(conv.kernel_size, torch.tensor([3, 3, 3])))
        self.assertTrue(torch.equal(conv.stride, torch.tensor([1, 1, 1])))
        self.assertEqual(conv.weight.shape, (32, 16, 3, 3, 3))
        self.assertIsNotNone(conv.bias)
        self.assertEqual(conv.bias.shape, (32,))
        self.assertIn("16", conv.extra_repr())
        self.assertIn("32", conv.extra_repr())

    def test_sparse_conv3d_no_bias(self):
        conv = fvnn.SparseConv3d(in_channels=8, out_channels=16, bias=False)
        self.assertIsNone(conv.bias)

    def test_sparse_conv3d_kernel_size_1(self):
        conv = fvnn.SparseConv3d(in_channels=8, out_channels=16, kernel_size=1)
        self.assertEqual(conv.kernel_volume, 1)
        self.assertEqual(conv.weight.shape, (16, 8))

    @parameterized.expand(all_device_dtype_combos)
    def test_sparse_conv3d_forward(self, device, dtype):
        grid = self._make_dense_grid(device, shape=(8, 8, 8))
        features = self._make_features(grid, 16, device, dtype)

        conv = fvnn.SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).to(device=device, dtype=dtype)
        plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid)
        output = conv(features, plan)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape[1], 32)

    @parameterized.expand(all_device_dtype_combos)
    def test_sparse_conv3d_kernel_size_1_forward(self, device, dtype):
        grid = self._make_dense_grid(device, shape=(8, 8, 8))
        features = self._make_features(grid, 8, device, dtype)

        conv = fvnn.SparseConv3d(in_channels=8, out_channels=16, kernel_size=1).to(device=device, dtype=dtype)
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=grid)
        output = conv(features, plan)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape[1], 16)
        self.assertEqual(output.jdata.shape[0], features.jdata.shape[0])

    def test_sparse_conv3d_plan_mismatch(self):
        device, dtype = "cpu", torch.float32
        grid = self._make_dense_grid(device, shape=(8, 8, 8))
        features = self._make_features(grid, 16, device, dtype)

        conv = fvnn.SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).to(device=device, dtype=dtype)
        plan = ConvolutionPlan.from_grid_batch(kernel_size=5, stride=1, source_grid=grid)

        with self.assertRaises(ValueError):
            conv(features, plan)

    # =========================================================================
    # SparseConvTranspose3d
    # =========================================================================

    def test_sparse_conv_transpose3d_construction(self):
        conv = fvnn.SparseConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.assertEqual(conv.in_channels, 32)
        self.assertEqual(conv.out_channels, 16)
        self.assertEqual(conv.weight.shape, (16, 32, 3, 3, 3))
        self.assertIsNotNone(conv.bias)

    @parameterized.expand(all_device_dtype_combos)
    def test_sparse_conv_transpose3d_forward(self, device, dtype):
        grid = self._make_dense_grid(device, shape=(8, 8, 8))
        features = self._make_features(grid, 32, device, dtype)

        conv = fvnn.SparseConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3).to(device=device, dtype=dtype)
        target_grid = grid.conv_grid(kernel_size=3, stride=1)
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=3, stride=1, source_grid=grid, target_grid=target_grid
        )
        output = conv(features, plan)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape[1], 16)

    def test_sparse_conv_transpose3d_plan_mismatch(self):
        device, dtype = "cpu", torch.float32
        grid = self._make_dense_grid(device, shape=(8, 8, 8))
        features = self._make_features(grid, 32, device, dtype)

        conv = fvnn.SparseConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3).to(device=device, dtype=dtype)
        # Non-transposed plan for a transposed conv module
        plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid)

        with self.assertRaises(ValueError):
            conv(features, plan)

    # =========================================================================
    # GroupNorm
    # =========================================================================

    @parameterized.expand(all_device_dtype_combos)
    def test_group_norm_forward(self, device, dtype):
        num_channels, num_groups = 16, 4
        grid = self._make_dense_grid(device, batch_size=2, shape=(8, 8, 8))
        features = self._make_features(grid, num_channels, device, dtype)

        gn = fvnn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=dtype).to(device)
        output = gn(features, grid)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape, features.jdata.shape)

    @parameterized.expand(all_device_dtype_combos)
    def test_group_norm_against_pytorch(self, device, dtype):
        num_channels, num_groups = 16, 4
        grid = self._make_dense_grid(device, batch_size=1, shape=(8, 8, 8))

        for affine in (True, False):
            data = torch.randn(grid.total_voxels, num_channels, device=device, dtype=dtype, requires_grad=True)
            features = grid.jagged_like(data)

            our_gn = fvnn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=affine, dtype=dtype).to(
                device
            )
            our_output = our_gn(features, grid)
            our_output.jdata.sum().backward()
            self.assertIsNotNone(data.grad)
            our_grad = data.grad.clone()

            torch_gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=affine, dtype=dtype).to(
                device
            )
            if affine:
                with torch.no_grad():
                    torch_gn.weight.copy_(our_gn.weight)
                    torch_gn.bias.copy_(our_gn.bias)
            torch_input = data.detach().t().unsqueeze(0).requires_grad_(True)  # (N, C) -> (1, C, N)
            torch_output = torch_gn(torch_input).squeeze(0).t()  # -> (N, C)
            torch_output.sum().backward()
            self.assertIsNotNone(torch_input.grad)
            torch_grad = torch_input.grad.squeeze(0).t()  # -> (N, C)

            self.assertTrue(torch.allclose(our_output.jdata, torch_output, atol=1e-3))
            self.assertTrue(torch.allclose(our_grad, torch_grad, atol=1e-3))

    def test_group_norm_channel_mismatch(self):
        grid = self._make_dense_grid("cpu")
        features = self._make_features(grid, 8, "cpu", torch.float32)

        gn = fvnn.GroupNorm(num_groups=4, num_channels=16)
        with self.assertRaises(AssertionError):
            gn(features, grid)

    # =========================================================================
    # BatchNorm
    # =========================================================================

    @parameterized.expand(all_device_dtype_combos)
    def test_batch_norm_forward(self, device, dtype):
        num_features = 16
        grid = self._make_dense_grid(device)
        features = self._make_features(grid, num_features, device, dtype)

        bn = fvnn.BatchNorm(num_features, dtype=dtype).to(device)
        bn.train()
        output = bn(features, grid)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape, features.jdata.shape)

    @parameterized.expand(all_device_dtype_combos)
    def test_batch_norm_against_pytorch(self, device, dtype):
        num_features = 16
        grid = self._make_dense_grid(device)
        data = torch.randn(grid.total_voxels, num_features, device=device, dtype=dtype)
        features = grid.jagged_like(data)

        our_bn = fvnn.BatchNorm(num_features, affine=False, dtype=dtype).to(device)
        our_bn.train()
        our_output = our_bn(features, grid)

        torch_bn = nn.BatchNorm1d(num_features, affine=False, dtype=dtype).to(device)
        torch_bn.train()
        torch_output = torch_bn(data)

        self.assertTrue(torch.allclose(our_output.jdata, torch_output, atol=1e-5))

    def test_batch_norm_eval_deterministic(self):
        num_features = 8
        grid = self._make_dense_grid("cpu")
        features = self._make_features(grid, num_features, "cpu", torch.float32)

        bn = fvnn.BatchNorm(num_features)
        bn.train()
        _ = bn(features, grid)
        bn.eval()

        out1 = bn(features, grid)
        out2 = bn(features, grid)
        self.assertTrue(torch.equal(out1.jdata, out2.jdata))

    def test_batch_norm_channel_mismatch(self):
        grid = self._make_dense_grid("cpu")
        features = self._make_features(grid, 8, "cpu", torch.float32)

        bn = fvnn.BatchNorm(16)
        with self.assertRaises(AssertionError):
            bn(features, grid)

    # =========================================================================
    # SyncBatchNorm
    # =========================================================================

    @parameterized.expand(all_device_dtype_combos)
    def test_sync_batch_norm_forward(self, device, dtype):
        num_features = 16
        grid = self._make_dense_grid(device)
        features = self._make_features(grid, num_features, device, dtype)

        sbn = fvnn.SyncBatchNorm(num_features, dtype=dtype).to(device)
        sbn.train()
        output = sbn(features, grid)

        self.assertIsInstance(output, JaggedTensor)
        self.assertEqual(output.jdata.shape, features.jdata.shape)

    def test_convert_sync_batchnorm(self):
        channels = 16
        network = nn.Sequential(
            fvnn.BatchNorm(channels),
            nn.ReLU(),
            fvnn.BatchNorm(channels),
        )

        converted = fvnn.SyncBatchNorm.convert_sync_batchnorm(network)

        num_sync_bn = 0
        for module in converted.modules():
            self.assertNotIsInstance(module, fvnn.BatchNorm)
            if isinstance(module, fvnn.SyncBatchNorm):
                num_sync_bn += 1

        self.assertEqual(num_sync_bn, 2)

    def test_convert_sync_batchnorm_preserves_weights(self):
        channels = 8
        bn = fvnn.BatchNorm(channels)
        original_weight = bn.weight.data.clone()
        original_bias = bn.bias.data.clone()
        original_running_mean = bn.running_mean.clone()
        original_running_var = bn.running_var.clone()

        converted = fvnn.SyncBatchNorm.convert_sync_batchnorm(bn)

        self.assertIsInstance(converted, fvnn.SyncBatchNorm)
        self.assertTrue(torch.equal(converted.weight.data, original_weight))
        self.assertTrue(torch.equal(converted.bias.data, original_bias))
        self.assertTrue(torch.equal(converted.running_mean, original_running_mean))
        self.assertTrue(torch.equal(converted.running_var, original_running_var))

    # =========================================================================
    # _trace_fvdb_nn_forward decorator
    # =========================================================================

    def test_trace_fvdb_nn_forward_decorator(self):
        for cls in (
            fvnn.AvgPool,
            fvnn.MaxPool,
            fvnn.UpsamplingNearest,
            fvnn.SparseConv3d,
            fvnn.SparseConvTranspose3d,
            fvnn.GroupNorm,
            fvnn.BatchNorm,
            fvnn.SyncBatchNorm,
        ):
            self.assertTrue(issubclass(cls, nn.Module), f"{cls.__name__} should be an nn.Module subclass")


# =========================================================================
# Distributed SyncBatchNorm helpers
# =========================================================================


class RunningStatistics(NamedTuple):
    mean: torch.Tensor
    var: torch.Tensor


def _sync_bn_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _sync_bn_cleanup() -> None:
    torch.distributed.destroy_process_group()


def _run_syncbn_test(rank: int, world_size: int, return_dict: MutableMapping[int, RunningStatistics]) -> None:
    _sync_bn_setup(rank, world_size)

    # Each rank gets a different number of jagged tensors, making it extremely
    # unlikely the running stats would match without synchronization.
    num_features = 2
    points = fvdb.JaggedTensor([torch.randn(num_features, 3, device="cuda") for _ in range(rank + 1)])

    grid = GridBatch.from_points(points)
    channels = 8
    features = grid.jagged_like(torch.randn(grid.ijk.jdata.shape[0], channels, device="cuda"))

    layer = fvnn.SyncBatchNorm(channels, device="cuda")
    layer(features, grid)

    return_dict[rank] = RunningStatistics(
        mean=layer.running_mean.cpu().detach().clone(),
        var=layer.running_var.cpu().detach().clone(),
    )

    _sync_bn_cleanup()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available(),
    "SyncBatchNorm is only supported on CUDA backends with distributed enabled.",
)
class TestSyncBatchNormDistributed(unittest.TestCase):
    def setUp(self):
        self.original_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        torch.multiprocessing.set_start_method("spawn", force=True)

    def tearDown(self):
        torch.multiprocessing.set_start_method(self.original_start_method, force=True)

    def test_running_stats_are_synchronized(self) -> None:
        """Validates the running statistics are properly synchronized across ranks."""
        manager = torch.multiprocessing.Manager()
        return_dict = manager.dict()

        world_size = 2
        torch.multiprocessing.spawn(
            _run_syncbn_test,
            args=(world_size, return_dict),
            nprocs=world_size,
            join=True,
        )

        means = [stats.mean for stats in return_dict.values()]
        vars_ = [stats.var for stats in return_dict.values()]

        for mean, var in zip(means[1:], vars_[1:], strict=True):
            torch.testing.assert_close(mean, means[0], rtol=0, atol=0)
            torch.testing.assert_close(var, vars_[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
