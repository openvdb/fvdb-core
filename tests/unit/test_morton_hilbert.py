# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import numpy as np
import torch
from fvdb.types import resolve_device
from parameterized import parameterized

from fvdb import GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]
import numpy as np


def _as_uint32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.uint32, copy=False)


def _as_int64(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.int64, copy=False)


def _expand_21_bits_hard(u: np.ndarray) -> np.ndarray:
    """
    'Hard way' expansion of a single axis:
      - iterate bit positions b = 0..20
      - mask (1<<b)
      - if set, place at position (3*b) in a uint64 accumulator.

    Inputs:
      u : array of shape (N, ...) dtype uint32
    Returns:
      uint64 array with bits placed at positions 0,3,6,...
    """
    u = _as_int64(u)
    out = np.zeros_like(u, dtype=np.int64)

    # Only lower 21 bits matter; mask anyway to make intent explicit.
    u_masked = u & np.int64(0x1FFFFF)

    for b in range(21):
        mask = np.int64(1 << b)
        bit_set = (u_masked & mask) != 0
        out |= bit_set.astype(np.int64) << np.int64(3 * b)

    return out


def _morton3D_encode_hard(ijk: np.ndarray) -> np.ndarray:
    """
    Unit-test–oriented, visibly-correct 3D Morton encoder.
    For each of i, j, k (uint32), take lower 21 bits and place them one by one:
      i bit b -> position 3*b + 0
      j bit b -> position 3*b + 1
      k bit b -> position 3*b + 2

    Inputs:
      ijk: array of shape (N, 3) dtype uint32
    Returns:
      uint64 array with Morton code.
    """
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError("Input must be of shape (N, 3)")

    i = _as_uint32(ijk[:, 0])
    j = _as_uint32(ijk[:, 1])
    k = _as_uint32(ijk[:, 2])

    code_i = _expand_21_bits_hard(i) << np.int64(0)
    code_j = _expand_21_bits_hard(j) << np.int64(1)
    code_k = _expand_21_bits_hard(k) << np.int64(2)
    return _as_int64(code_i | code_j | code_k)


def _morton3D_encode_hard_torch(ijk: torch.Tensor) -> torch.Tensor:
    ijk_np = ijk.cpu().numpy()
    code = _morton3D_encode_hard(ijk_np)
    return torch.from_numpy(code).to(ijk.device)


def _create_test_grid_batch(batch_size: int, device: torch.device) -> tuple[JaggedTensor, GridBatch]:
    voxels_per_batch = [np.random.randint(100, 1000) for _ in range(batch_size)]

    # Generate batch-size of random ijk coordinates
    ijk_batches = [torch.randint(0, 1000, (voxels_per_batch[i], 3), device=device) for i in range(batch_size)]
    jagged_ijk = JaggedTensor(ijk_batches)

    # Make grid batch from jagged ijk
    return jagged_ijk, GridBatch.from_ijk(jagged_ijk)


class TestMortonHilbert(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    @parameterized.expand(all_device_combos)
    def test_validate_morton3D_encode_hard(self, device):
        device = resolve_device(device)

        # 1. Simple 2x2x2 cube with single-bit coordinates
        ijk = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.uint32,
            device=device,
        )
        code = _morton3D_encode_hard_torch(ijk)
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=device)
        print(f"code: {code}, expected: {expected}")
        self.assertTrue(torch.equal(code, expected))

        # 2. Single-axis increments (only X changes)
        ijk = torch.tensor([[i, 0, 0] for i in range(8)], dtype=torch.uint32, device=device)
        code = _morton3D_encode_hard_torch(ijk)
        # 000 -> 000 000 000 -> 0
        # 001 -> 000 000 001 -> 1
        # 010 -> 000 001 000 -> 8
        # 011 -> 000 001 001 -> 9
        # 100 -> 001 000 000 -> 64
        # 101 -> 001 000 001 -> 65
        # 110 -> 001 001 000 -> 72
        # 111 -> 001 001 001 -> 73
        expected = torch.tensor([0, 1, 8, 9, 64, 65, 72, 73], dtype=torch.int64, device=device)
        self.assertTrue(torch.equal(code, expected))

        # 3. Mixed mid-range coordinate
        ijk = torch.tensor([[2, 3, 5]], dtype=torch.uint32, device=device)
        code = _morton3D_encode_hard_torch(ijk)

        # 2 -> 010 -> 000 001 000 -> 8
        # 3 -> 011 -> 000 010 010 -> 18
        # 5 -> 101 -> 100 000 100 -> 260
        # 260 + 18 + 8 = 286
        expected = torch.tensor([286], dtype=torch.int64, device=device)
        print(f"code: {code}, expected: {expected}")
        self.assertTrue(torch.equal(code, expected))

        # 4. Maximum 21-bit all ones
        max21 = (1 << 21) - 1
        ijk = torch.tensor(
            [
                [max21, 0, 0],
                [0, max21, 0],
                [0, 0, max21],
                [max21, max21, max21],
            ],
            dtype=torch.uint32,
            device=device,
        )
        code = _morton3D_encode_hard_torch(ijk)

        def mask(axis):
            return sum(1 << (3 * b + axis) for b in range(21))

        ex = torch.tensor(
            [mask(0), mask(1), mask(2), mask(0) | mask(1) | mask(2)],
            dtype=torch.int64,
            device=device,
        )
        self.assertTrue(torch.equal(code, ex))

    @parameterized.expand(all_device_combos)
    def test_morton_codes(self, device):
        device = resolve_device(device)
        jagged_ijk, grid_batch = _create_test_grid_batch(7, device)
        ijks_flat = grid_batch.ijk.jdata

        # Get Morton codes for both orderings, zero offset
        morton_codes = grid_batch.morton(offset=0)
        morton_zyx_codes = grid_batch.morton_zyx(offset=0)
        self.assertIsInstance(morton_codes, JaggedTensor)
        self.assertIsInstance(morton_zyx_codes, JaggedTensor)

        morton_codes_flat = morton_codes.jdata
        morton_zyx_codes_flat = morton_zyx_codes.jdata
        self.assertEqual(len(ijks_flat), len(morton_codes_flat))
        self.assertEqual(len(ijks_flat), len(morton_zyx_codes_flat))

        # Verify codes are int64
        self.assertEqual(morton_codes_flat.dtype, torch.int64)
        self.assertEqual(morton_zyx_codes_flat.dtype, torch.int64)

        # Test that codes are non-negative
        self.assertTrue(torch.all(morton_codes.jdata >= 0))
        self.assertTrue(torch.all(morton_zyx_codes.jdata >= 0))

        # Codes should match the hard-way implementation
        kjis_flat = ijks_flat.flip(dims=[1])
        morton_codes_hard = _morton3D_encode_hard_torch(ijks_flat)
        morton_zyx_codes_hard = _morton3D_encode_hard_torch(kjis_flat)
        self.assertTrue(torch.equal(morton_codes_hard, morton_codes_flat))
        self.assertTrue(torch.equal(morton_zyx_codes_hard, morton_zyx_codes_flat))

        # Test with explicit offset
        morton_codes_with_offset = grid_batch.morton(offset=10)
        morton_zyx_codes_with_offset = grid_batch.morton_zyx(offset=10)
        ijks_flat_with_offset = ijks_flat + 10
        kjis_flat_with_offset = kjis_flat + 10
        morton_codes_hard_with_offset = _morton3D_encode_hard_torch(ijks_flat_with_offset)
        morton_zyx_codes_hard_with_offset = _morton3D_encode_hard_torch(kjis_flat_with_offset)
        self.assertTrue(torch.equal(morton_codes_hard_with_offset, morton_codes_with_offset.jdata))
        self.assertTrue(torch.equal(morton_zyx_codes_hard_with_offset, morton_zyx_codes_with_offset.jdata))

    @parameterized.expand(all_device_combos)
    def test_hilbert_codes(self, device):
        device = resolve_device(device)
        jagged_ijk, grid_batch = _create_test_grid_batch(7, device)

        hilbert_codes = grid_batch.hilbert(offset=0)
        hilbert_zyx_codes = grid_batch.hilbert_zyx(offset=0)
        # Test that codes are non-negative
        self.assertTrue(torch.all(hilbert_codes.jdata >= 0))
        self.assertTrue(torch.all(hilbert_zyx_codes.jdata >= 0))

        # Test with explicit offset
        # We don't have full testing tools for this yet.
        hilbert_codes_with_offset = grid_batch.hilbert(offset=10)
        hilbert_zyx_codes_with_offset = grid_batch.hilbert_zyx(offset=10)
        self.assertEqual(len(hilbert_codes_with_offset), len(hilbert_codes))
        self.assertEqual(len(hilbert_zyx_codes_with_offset), len(hilbert_zyx_codes))

    @parameterized.expand(all_device_combos)
    def test_space_filling_curve_properties(self, device):
        device = resolve_device(device)
        batch_size = 7
        jagged_ijk, grid_batch = _create_test_grid_batch(batch_size, device)

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
