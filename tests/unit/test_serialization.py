# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for space-filling-curve serialization (fvdb.serialization)."""

import unittest

import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor
from fvdb.serialization import _packed_key_shift, _segmented_argsort

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
CURVE_ORDERS = ["z", "z-trans", "hilbert", "hilbert-trans"]
DEVICE_ORDER_COMBOS = [[d, o] for d in DEVICES for o in CURVE_ORDERS]


def _make_grid(sizes, device, lo=-20, hi=32, seed=0) -> GridBatch:
    gen = torch.Generator().manual_seed(seed)
    ijks = [torch.randint(lo, hi, (n, 3), generator=gen, dtype=torch.int32).to(device) for n in sizes]
    return GridBatch.from_ijk(JaggedTensor(ijks))


def _pointcept_z_order(xyz: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """Pointcept's z-order key: bit 3i+2 from x, 3i+1 from y, 3i from z."""
    x, y, z = (xyz[:, c].long() for c in range(3))
    key = torch.zeros_like(x)
    for i in range(depth):
        bit = 1 << i
        key |= ((x & bit) << (2 * i + 2)) | ((y & bit) << (2 * i + 1)) | ((z & bit) << (2 * i))
    return key


def _skilling_hilbert(xyz: torch.Tensor, nbits: int) -> torch.Tensor:
    """Skilling's Hilbert encoder with ``nbits`` bits per axis.

    For every depth this gives the same codes as Pointcept's ``hilbert_encode(xyz, depth=nbits)``.
    """
    x = [xyz[:, a].long().clone() for a in range(3)]
    q = 1 << (nbits - 1)
    while q > 1:
        p = q - 1
        for i in range(3):
            hi = (x[i] & q) != 0
            x[0] = torch.where(hi, x[0] ^ p, x[0])
            t = torch.where(hi, torch.zeros_like(x[0]), (x[0] ^ x[i]) & p)
            x[0] ^= t
            x[i] ^= t
        q >>= 1
    for i in range(1, 3):
        x[i] ^= x[i - 1]
    t = torch.zeros_like(x[0])
    q = 1 << (nbits - 1)
    while q > 1:
        t = torch.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    code = torch.zeros_like(x[0])
    for b in range(nbits - 1, -1, -1):
        for i in range(3):
            code = (code << 1) | (((x[i] ^ t) >> b) & 1)
    return code


def _make_origin_grid(sizes, device, extent, seed=0) -> GridBatch:
    """Grids whose coordinates start at 0 on every axis, like Pointcept's grid_coord."""
    gen = torch.Generator().manual_seed(seed)
    ijks = []
    for n in sizes:
        ijk = torch.randint(0, extent, (n, 3), generator=gen, dtype=torch.int32)
        if n > 0:
            ijk[0] = 0
        ijks.append(ijk.to(device))
    return GridBatch.from_ijk(JaggedTensor(ijks))


def _local_ijk(grid: GridBatch) -> torch.Tensor:
    """Voxel coordinates shifted by each grid's own minimum, computed with a host loop."""
    ijk = grid.ijk.jdata.long()
    out = torch.empty_like(ijk)
    offsets = grid.joffsets.tolist()
    for b in range(grid.grid_count):
        s, e = offsets[b], offsets[b + 1]
        if e > s:
            out[s:e] = ijk[s:e] - ijk[s:e].min(0).values
    return out


def _loop_perm(grid: GridBatch, code: torch.Tensor) -> torch.Tensor:
    """Reference permutation: per-grid stable argsort with a host loop."""
    offsets = grid.joffsets.tolist()
    parts = [torch.argsort(code[s:e], stable=True) + s for s, e in zip(offsets[:-1], offsets[1:])]
    return torch.cat(parts) if parts else code.new_empty(0)


class TestSerializationCodes(unittest.TestCase):
    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_codes_shape_dtype_nonnegative(self, device, order):
        grid = _make_grid([50, 0, 120], device)
        codes = fvdb.serialization_codes(grid, order)
        self.assertEqual(codes.shape, (grid.total_voxels,))
        self.assertEqual(codes.dtype, torch.int64)
        self.assertTrue(bool((codes >= 0).all()))

    @parameterized.expand(DEVICES)
    def test_z_matches_pointcept_key_order(self, device):
        grid = _make_grid([300, 200], device)
        local = _local_ijk(grid)
        for order, cols in (("z", [0, 1, 2]), ("z-trans", [1, 0, 2])):
            expected = _loop_perm(grid, _pointcept_z_order(local[:, cols]))
            perm, _ = fvdb.serialization_perm(grid, order)
            self.assertTrue(torch.equal(perm, expected), order)

    @parameterized.expand(DEVICES)
    def test_batch_offset_matches_cpp_kernels(self, device):
        # "z" encodes (k, j, i) like the zyx Morton kernel; at depth 21 "hilbert" is the plain kernel.
        grid = _make_grid([80, 60], device)
        z = fvdb.serialization_codes(grid, "z", offset_mode="batch")
        h = fvdb.serialization_codes(grid, "hilbert", offset_mode="batch", depth=21)
        self.assertTrue(torch.equal(z, grid.morton_zyx().jdata))
        self.assertTrue(torch.equal(h, grid.hilbert().jdata))

    @parameterized.expand(DEVICES)
    def test_hilbert_matches_reference_at_every_depth(self, device):
        for depth in range(1, 13):
            grid = _make_origin_grid([300], device, extent=2**depth, seed=depth)
            local = _local_ijk(grid)
            for order, cols in (("hilbert", [0, 1, 2]), ("hilbert-trans", [1, 0, 2])):
                codes = fvdb.serialization_codes(grid, order, depth=depth)
                expected = _skilling_hilbert(local[:, cols], depth)
                self.assertTrue(torch.equal(codes, expected), (order, depth))

    @parameterized.expand(DEVICES)
    def test_default_depth_follows_pointcept_rule(self, device):
        grid = _make_origin_grid([50, 0, 40], device, extent=37)
        expected = (int(_local_ijk(grid).max()) + 1).bit_length()
        self.assertEqual(fvdb.serialization_depth(grid), expected)
        ser = fvdb.serialize(grid, "hilbert", keep_codes=True)
        self.assertEqual(ser.depth, expected)
        assert ser.codes is not None
        self.assertTrue(torch.equal(ser.codes[0], _skilling_hilbert(_local_ijk(grid), expected)))

    @parameterized.expand(DEVICES)
    def test_pooled_codes_drop_three_bits_per_halving(self, device):
        fine = _make_origin_grid([400, 0, 250], device, extent=40)
        ser = fvdb.serialize(fine, CURVE_ORDERS, keep_codes=True)
        coarse = fine.coarsened_grid(2)
        coarser = coarse.coarsened_grid(2)
        ser_c = ser.pooled(coarse)
        ser_cc = ser_c.pooled(coarser)
        self.assertEqual((ser_c.depth, ser_c.code_shift, ser_cc.code_shift), (ser.depth, 1, 2))
        assert ser.codes is not None and ser_c.codes is not None and ser_cc.codes is not None

        fine_ijk = fine.ijk
        parent = coarse.ijk_to_index(fine_ijk.jagged_like(fine_ijk.jdata // 2), cumulative=True).jdata
        grandparent = coarser.ijk_to_index(fine_ijk.jagged_like(fine_ijk.jdata // 4), cumulative=True).jdata
        for i in range(len(CURVE_ORDERS)):
            self.assertTrue(torch.equal(ser_c.codes[i][parent], ser.codes[i] >> 3), ser.orders[i])
            self.assertTrue(torch.equal(ser_cc.codes[i][grandparent], ser.codes[i] >> 6), ser.orders[i])

    def test_depth_too_small_raises(self):
        grid = _make_origin_grid([30], "cpu", extent=64)
        with self.assertRaisesRegex(ValueError, "depth"):
            fvdb.serialization_codes(grid, "z", depth=4)
        with self.assertRaises(ValueError):
            fvdb.serialization_codes(grid, "z", depth=22)

    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_grid_offset_is_translation_invariant(self, device, order):
        base = torch.randint(0, 16, (100, 3), dtype=torch.int32, generator=torch.Generator().manual_seed(3))
        other = torch.randint(-50, -30, (40, 3), dtype=torch.int32, generator=torch.Generator().manual_seed(4))
        shift = torch.tensor([7, -3, 11], dtype=torch.int32)
        grid_a = GridBatch.from_ijk(JaggedTensor([base.to(device), other.to(device)]))
        grid_b = GridBatch.from_ijk(JaggedTensor([(base + shift).to(device), other.to(device)]))
        # Translation changes the native voxel order, so compare the sets of codes.
        n0 = int(grid_a.num_voxels[0].item())
        codes_a = fvdb.serialization_codes(grid_a, order)[:n0]
        codes_b = fvdb.serialization_codes(grid_b, order)[:n0]
        self.assertTrue(torch.equal(torch.sort(codes_a).values, torch.sort(codes_b).values))
        shifted_b = fvdb.serialization_codes(grid_b, order, offset_mode="batch")[:n0]
        self.assertFalse(torch.equal(torch.sort(codes_a).values, torch.sort(shifted_b).values))

    def test_invalid_order_names(self):
        grid = _make_grid([10], "cpu")
        for bad in ("morton_zyx", "hilbert_zyx", "foo"):
            with self.assertRaisesRegex(ValueError, "-trans"):
                fvdb.serialization_codes(grid, bad)
        with self.assertRaises(ValueError):
            fvdb.serialization_codes(grid, "z", offset_mode="bogus")  # type: ignore[arg-type]

    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_single_grid(self, device, order):
        grid = _make_grid([200], device)
        perm, _ = fvdb.serialization_perm(grid, order)
        self.assertTrue(torch.equal(perm, _loop_perm(grid, fvdb.serialization_codes(grid, order))))

    def test_aliases(self):
        grid = _make_grid([40], "cpu")
        self.assertTrue(torch.equal(fvdb.serialization_codes(grid, "morton"), fvdb.serialization_codes(grid, "z")))
        self.assertTrue(torch.equal(fvdb.serialization_codes(grid, "identity"), torch.arange(grid.total_voxels)))


class TestSerializationPerm(unittest.TestCase):
    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_perm_sorts_within_grid_and_keeps_grids_contiguous(self, device, order):
        grid = _make_grid([70, 0, 130, 1], device)
        perm, _ = fvdb.serialization_perm(grid, order)
        codes = fvdb.serialization_codes(grid, order)
        jidx = grid.jidx
        self.assertTrue(torch.equal(jidx[perm], jidx))
        sorted_codes = codes[perm]
        same_grid = jidx[1:] == jidx[:-1]
        self.assertTrue(bool((sorted_codes[1:][same_grid] >= sorted_codes[:-1][same_grid]).all()))

    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_perm_and_inverse_round_trip(self, device, order):
        grid = _make_grid([90, 33], device)
        perm, inv = fvdb.serialization_perm(grid, order)
        ar = torch.arange(grid.total_voxels, device=grid.device)
        self.assertTrue(torch.equal(torch.sort(perm).values, ar))
        self.assertTrue(torch.equal(perm[inv], ar))
        self.assertTrue(torch.equal(inv[perm], ar))

    @parameterized.expand(DEVICE_ORDER_COMBOS)
    def test_perm_matches_host_loop_reference(self, device, order):
        grid = _make_grid([64, 5, 0, 200], device)
        perm, _ = fvdb.serialization_perm(grid, order)
        self.assertTrue(torch.equal(perm, _loop_perm(grid, fvdb.serialization_codes(grid, order))))

    @parameterized.expand(DEVICES)
    def test_packed_and_two_pass_sorts_agree(self, device):
        grid = _make_grid([150, 0, 75], device)
        shift = _packed_key_shift(grid, fvdb.serialization_depth(grid), 0)
        self.assertIsNotNone(shift)
        for order in CURVE_ORDERS:
            code = fvdb.serialization_codes(grid, order)
            packed = _segmented_argsort(code, grid.jidx.long(), shift)
            two_pass = _segmented_argsort(code, grid.jidx.long(), None)
            self.assertTrue(torch.equal(packed, two_pass), order)

    @parameterized.expand(DEVICES)
    def test_wide_extent_uses_two_pass_fallback(self, device):
        far = torch.tensor([[0, 0, 0], [2**20, 1, 2], [5, 2**20, 9]], dtype=torch.int32, device=device)
        grid = GridBatch.from_ijk(JaggedTensor([far] * 3))
        self.assertIsNone(_packed_key_shift(grid, fvdb.serialization_depth(grid), 0))
        for order in CURVE_ORDERS:
            perm, _ = fvdb.serialization_perm(grid, order)
            self.assertTrue(torch.equal(perm, _loop_perm(grid, fvdb.serialization_codes(grid, order))), order)

    @parameterized.expand(DEVICES)
    def test_curve_codes_bounded_by_extent(self, device):
        # The packed sort key assumes coordinates below 2**k give codes below 2**(3k).
        gen = torch.Generator().manual_seed(5)
        for k in range(1, 11):
            c = torch.randint(0, 2**k, (4000, 3), generator=gen, dtype=torch.int32).to(device)
            self.assertLess(int(fvdb.morton(c).max()), 2 ** (3 * k))
            self.assertLess(int(fvdb.hilbert(c).max()), 2 ** (3 * k))

    @parameterized.expand(DEVICES)
    def test_vdb_order_is_identity(self, device):
        grid = _make_grid([30, 12], device)
        perm, inv = fvdb.serialization_perm(grid, "vdb")
        ar = torch.arange(grid.total_voxels, device=grid.device)
        self.assertTrue(torch.equal(perm, ar))
        self.assertTrue(torch.equal(inv, ar))

    @parameterized.expand(DEVICES)
    def test_empty_grids(self, device):
        all_empty = GridBatch.from_ijk(JaggedTensor([torch.zeros((0, 3), dtype=torch.int32, device=device)] * 2))
        ser = fvdb.serialize(all_empty, ("z", "hilbert"), shuffle=True)
        self.assertEqual(ser.perm(0).numel(), 0)
        self.assertEqual(ser.inv_perm(1).numel(), 0)

        no_grids = GridBatch.from_zero_grids(device=device)
        perm, inv = fvdb.serialization_perm(no_grids, "z")
        self.assertEqual(perm.numel(), 0)
        self.assertEqual(inv.numel(), 0)


class TestGridSerialization(unittest.TestCase):
    @parameterized.expand(DEVICES)
    def test_container_indexing_and_shuffle(self, device):
        grid = _make_grid([60, 40], device)
        ser = fvdb.serialize(grid, ("z", "z-trans", "hilbert"), keep_codes=True)
        self.assertEqual(len(ser), 3)
        self.assertEqual(ser.order(4), "z-trans")
        self.assertIs(ser.perm(3), ser.perm(0))
        assert ser.codes is not None

        shuffled = ser.shuffled(torch.Generator().manual_seed(0))
        self.assertEqual(sorted(shuffled.orders), sorted(ser.orders))
        assert shuffled.codes is not None
        for i, name in enumerate(shuffled.orders):
            j = ser.orders.index(name)
            self.assertIs(shuffled.perm(i), ser.perm(j))
            self.assertIs(shuffled.inv_perm(i), ser.inv_perm(j))
            self.assertIs(shuffled.codes[i], ser.codes[j])

        again = ser.shuffled(torch.Generator().manual_seed(0))
        self.assertEqual(again.orders, shuffled.orders)

    @parameterized.expand(DEVICES)
    def test_matches_and_permute_jagged(self, device):
        grid = _make_grid([25, 0, 35], device)
        other = _make_grid([10, 10], device, seed=7)
        ser = fvdb.serialize(grid, "hilbert")
        feats = grid.jagged_like(torch.randn(grid.total_voxels, 4, device=grid.device))
        self.assertTrue(ser.matches(feats))
        self.assertFalse(ser.matches(other.ijk))

        ordered = fvdb.permute_jagged(feats, ser.perm(0))
        self.assertTrue(torch.equal(ordered.joffsets, feats.joffsets))
        restored = fvdb.permute_jagged(ordered, ser.inv_perm(0))
        self.assertTrue(torch.equal(restored.jdata, feats.jdata))

    def test_requires_an_order(self):
        with self.assertRaises(ValueError):
            fvdb.serialize(_make_grid([5], "cpu"), ())


if __name__ == "__main__":
    unittest.main()
