# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the SimpleUNet.

Verifies that the full UNet and its sub-modules (Pad, Unpad, Down, Up,
Bottleneck, DownUp) produce finite outputs, preserve grid topology, and
propagate gradients without NaN.
"""

import unittest

import torch
from fvdb.nn import SimpleUNet
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests.convolution_utils import (
    REDUCED_DEVICE_DTYPE_COMBOS,
    create_grid_from_coords,
)
from parameterized import parameterized

from fvdb import GridBatch, JaggedTensor


def _make_dense_block_coords(extent: int, offset: int, device: torch.device) -> torch.Tensor:
    """Create a dense cube of coordinates with side length `extent`, shifted by `offset`."""
    r = torch.arange(offset, offset + extent, device=device, dtype=torch.int32)
    coords = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    return coords


class TestSimpleUNet(unittest.TestCase):
    """Smoke / integration tests for SimpleUNet and its sub-modules."""

    SEED = 42
    IN_CHANNELS = 3
    BASE_CHANNELS = 4
    OUT_CHANNELS = 5
    KERNEL_SIZE = 3
    CHANNEL_GROWTH_RATE = 2
    BLOCK_LAYER_COUNT = 1

    def setUp(self):
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

    def _make_grid_and_features(
        self,
        device: torch.device,
        dtype: torch.dtype,
        extent: int = 16,
        downup_layer_count: int = 2,
    ) -> tuple[GridBatch, JaggedTensor, SimpleUNet]:
        """Build a dense grid, random features, and a small UNet on `device`."""
        coords = _make_dense_block_coords(extent, offset=0, device=device)
        grid = create_grid_from_coords(coords, device)

        num_voxels = grid.total_voxels
        features = JaggedTensor(torch.randn(num_voxels, self.IN_CHANNELS, device=device, dtype=dtype))

        model = SimpleUNet(
            in_channels=self.IN_CHANNELS,
            base_channels=self.BASE_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            channel_growth_rate=self.CHANNEL_GROWTH_RATE,
            kernel_size=self.KERNEL_SIZE,
            downup_layer_count=downup_layer_count,
            block_layer_count=self.BLOCK_LAYER_COUNT,
        )
        model = model.to(device=device, dtype=dtype)
        return grid, features, model

    # --------------------------------------------------------------------- #
    #  Forward pass smoke tests
    # --------------------------------------------------------------------- #

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_forward_produces_finite_output(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Full forward pass returns finite values with correct shape."""
        device = resolve_device(device)
        grid, features, model = self._make_grid_and_features(device, dtype)

        with torch.no_grad():
            out = model(features, grid)

        self.assertEqual(out.jdata.shape[0], grid.total_voxels)
        self.assertEqual(out.jdata.shape[1], self.OUT_CHANNELS)
        self.assertTrue(torch.isfinite(out.jdata).all(), "Output contains NaN or Inf")

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_forward_single_downup_layer(self, device: DeviceIdentifier, dtype: torch.dtype):
        """UNet with downup_layer_count=1 (single down/up stage with bottleneck) runs cleanly."""
        device = resolve_device(device)
        grid, features, model = self._make_grid_and_features(device, dtype, extent=8, downup_layer_count=1)

        with torch.no_grad():
            out = model(features, grid)

        self.assertEqual(out.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(out.jdata).all(), "Output contains NaN or Inf")

    # --------------------------------------------------------------------- #
    #  Backward pass / gradient tests
    # --------------------------------------------------------------------- #

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_backward_produces_finite_gradients(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Loss.backward() through the full UNet produces finite parameter gradients."""
        device = resolve_device(device)
        grid, features, model = self._make_grid_and_features(device, dtype)

        out = model(features, grid)
        loss = out.jdata.sum()
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")

    # --------------------------------------------------------------------- #
    #  Batch support
    # --------------------------------------------------------------------- #

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_forward_batch(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Forward pass with a batched GridBatch (two grids of different sizes)."""
        device = resolve_device(device)

        coords_a = _make_dense_block_coords(extent=16, offset=0, device=device)
        coords_b = _make_dense_block_coords(extent=12, offset=4, device=device)
        ijks = JaggedTensor([coords_a, coords_b])
        grid = GridBatch.from_ijk(ijks)

        num_voxels = grid.total_voxels
        features = JaggedTensor(torch.randn(num_voxels, self.IN_CHANNELS, device=device, dtype=dtype))

        model = SimpleUNet(
            in_channels=self.IN_CHANNELS,
            base_channels=self.BASE_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            channel_growth_rate=self.CHANNEL_GROWTH_RATE,
            kernel_size=self.KERNEL_SIZE,
            downup_layer_count=2,
            block_layer_count=self.BLOCK_LAYER_COUNT,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            out = model(features, grid)

        self.assertEqual(out.jdata.shape[0], grid.total_voxels)
        self.assertEqual(out.jdata.shape[1], self.OUT_CHANNELS)
        self.assertTrue(torch.isfinite(out.jdata).all(), "Output contains NaN or Inf")

    # --------------------------------------------------------------------- #
    #  Reset parameters
    # --------------------------------------------------------------------- #

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_reset_parameters(self, device: DeviceIdentifier, dtype: torch.dtype):
        """reset_parameters() runs without error and produces different weights."""
        device = resolve_device(device)
        _, _, model = self._make_grid_and_features(device, dtype)

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        model.reset_parameters()
        any_changed = any(not torch.equal(params_before[n], p) for n, p in model.named_parameters())
        self.assertTrue(any_changed, "reset_parameters() did not change any weights")


if __name__ == "__main__":
    unittest.main()
