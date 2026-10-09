# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Equivalence tests for the batched leaf-mask topology builder (issue #755).

On CUDA, ``refined_grid`` / ``coarsened_grid`` (and through them ``conv_grid`` /
``conv_transpose_grid`` for K == S) build all batch members in a single batched pass instead of
one NanoVDB build + merge per member. These tests pin the batched results against:

- ``from_ijk`` (PointsToGrid) grids built from independently computed expected coordinates,
  compared **elementwise** (``torch.equal`` on ``ijk.jdata``), which pins the canonical NanoVDB
  node ordering (root tiles by offset-shifted key, then x-major upper/lower child offsets), and
- the CPU implementations of the same ops, compared as per-member coordinate sets.

Emphasis is on the cases the batched pipeline must get right and a per-member loop got for free:
batches with empty members (first/middle/last/all), members of wildly unequal sizes, coordinates
straddling root-tile boundaries (+-4096) and negative octants (where the sort-key encoding differs
from the stored ``Tile::key`` encoding), and multi-pass factors (4 = two chained passes).
"""

import unittest

import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor


def _build(ijks, device):
    jt = JaggedTensor([t.to(device=device, dtype=torch.int32) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=1.0, origins=0.0)


def _expected_refine_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """All fine coordinates of ``ijk`` subdivided by ``factor`` (unique by construction)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    offsets = torch.stack(
        torch.meshgrid(torch.arange(factor), torch.arange(factor), torch.arange(factor), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    fine = ijk.to(torch.int64)[:, None, :] * factor + offsets[None, :, :].to(ijk.device)
    return fine.reshape(-1, 3).to(torch.int32)


def _expected_coarsen_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """Unique coarse coordinates floor(ijk / factor)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    coarse = torch.div(ijk.to(torch.int64), factor, rounding_mode="floor")
    return torch.unique(coarse, dim=0).to(torch.int32)


def _ijk_sets(grid: GridBatch):
    return [set(map(tuple, t.cpu().to(torch.int64).tolist())) for t in grid.ijk.unbind()]


# Coordinate batches exercising the tricky regimes of the batched builder.
def _tricky_batches():
    torch.manual_seed(42)
    return {
        "mixed_sizes": [
            torch.randint(-8, 8, (200, 3), dtype=torch.int32),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.randint(20, 60, (500, 3), dtype=torch.int32),
        ],
        "empty_members": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.randint(-10, 10, (100, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.tensor([[5, 5, 5], [5, 5, 6]], dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        "all_empty": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        # Straddles the +-4096 root-tile boundaries and negative octants: multiple root tiles per
        # grid and coordinates where the offset-shifted sort key and stored Tile::key encodings
        # order tiles differently.
        "tile_boundaries": [
            torch.tensor(
                [
                    [-4097, -4097, -4097],
                    [-4096, -4096, -4096],
                    [-2049, 0, 0],
                    [-1, -1, -1],
                    [0, 0, 0],
                    [4095, 4095, 4095],
                    [4096, 4096, 4096],
                    [4096, -4097, 0],
                ],
                dtype=torch.int32,
            ),
            torch.randint(-4200, -3900, (300, 3), dtype=torch.int32),
            torch.randint(3900, 4200, (300, 3), dtype=torch.int32),
        ],
        "single_grid": [
            torch.randint(-16, 16, (400, 3), dtype=torch.int32),
        ],
        "larger_batch": [torch.randint(-32, 32, (50 + 37 * i, 3), dtype=torch.int32) for i in range(16)],
    }


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for the batched builder")
class TestBatchedTopologyBuilder(unittest.TestCase):
    def _check_against_expected(self, result: GridBatch, expected_ijks, msg: str):
        """Elementwise-pin `result` against a from_ijk build of the expected coordinates."""
        expected = _build(expected_ijks, result.device)
        self.assertEqual(result.grid_count, expected.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels, expected.num_voxels), msg)
        self.assertTrue(
            torch.equal(result.ijk.jdata, expected.ijk.jdata),
            f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
        )
        for b in range(result.grid_count):
            self.assertTrue(
                torch.equal(result.bbox_at(b).cpu(), expected.bbox_at(b).cpu()),
                f"{msg}: bbox of member {b}",
            )

    def _check_against_cpu(self, result: GridBatch, cpu_result: GridBatch, msg: str):
        self.assertEqual(result.grid_count, cpu_result.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels.cpu(), cpu_result.num_voxels), msg)
        self.assertEqual(_ijk_sets(result), _ijk_sets(cpu_result), msg)

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_refined_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.refined_grid(factor)
            expected = [_expected_refine_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} refine x{factor}")
            cpu_result = _build(coords, "cpu").refined_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} refine x{factor} vs CPU")

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_coarsened_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.coarsened_grid(factor)
            expected = [_expected_coarsen_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} coarsen x{factor}")
            cpu_result = _build(coords, "cpu").coarsened_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} coarsen x{factor} vs CPU")

    def test_conv_grid_k2s2_multi_grid_matches_per_member(self):
        coords = _tricky_batches()["mixed_sizes"]
        full = _build(coords, "cuda")
        conv = full.conv_grid(kernel_size=2, stride=2)
        convt = full.conv_transpose_grid(kernel_size=2, stride=2)
        for b, c in enumerate(coords):
            single = _build([c], "cuda")
            conv_single = single.conv_grid(kernel_size=2, stride=2)
            convt_single = single.conv_transpose_grid(kernel_size=2, stride=2)
            self.assertTrue(
                torch.equal(conv.ijk.unbind()[b], conv_single.ijk.jdata),
                f"conv_grid k2s2 member {b}",
            )
            self.assertTrue(
                torch.equal(convt.ijk.unbind()[b], convt_single.ijk.jdata),
                f"conv_transpose_grid k2s2 member {b}",
            )

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_conv_stride1_and_k3s2_match_cpu(self, name):
        # Stride-1 uniform K routes through batched box-dilate passes (odd K: [-1,1]^3 per pass;
        # even K: one-sided {-1,0}^3 / {0,1}^3 passes), and k3s2 transpose through refine + one
        # negative pad pass. Pin against the CPU implementation (coordinate sets) and against a
        # from_ijk build of the CPU coordinates (elementwise: canonical node order).
        coords = _tricky_batches()[name]
        cuda_grid = _build(coords, "cuda")
        cpu_grid = _build(coords, "cpu")
        cases = (
            [("conv_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", 3, 2)]
        )
        for op, k, s in cases:
            msg = f"{name} {op} k{k}s{s}"
            result = getattr(cuda_grid, op)(kernel_size=k, stride=s)
            reference = getattr(cpu_grid, op)(kernel_size=k, stride=s)
            self._check_against_cpu(result, reference, msg)
            expected = _build(list(reference.ijk.unbind()), "cuda")
            self.assertTrue(
                torch.equal(result.ijk.jdata, expected.ijk.jdata),
                f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
            )

    def test_masked_refine_multi_grid(self):
        # Masked subdivision routes through pruneGrid then the batched refine.
        coords = _tricky_batches()["mixed_sizes"]
        grid = _build(coords, "cuda")
        mask = JaggedTensor([((t.sum(-1) % 2) == 0) for t in grid.ijk.unbind()])
        result = grid.refined_grid(2, mask=mask)
        cpu_grid = _build(coords, "cpu")
        cpu_mask = JaggedTensor([m.cpu() for m in mask.unbind()])
        cpu_result = cpu_grid.refined_grid(2, mask=cpu_mask)
        self._check_against_cpu(result, cpu_result, "masked refine x2 vs CPU")

    def test_refine_coarsen_roundtrip_grid_count(self):
        # Serialization-visible metadata: grid_count/address arithmetic must be consistent for a
        # multi-grid handle produced by the batched builder.
        coords = _tricky_batches()["empty_members"]
        grid = _build(coords, "cuda")
        fine = grid.refined_grid(2)
        back = fine.coarsened_grid(2)
        self.assertEqual(fine.grid_count, grid.grid_count)
        self.assertEqual(back.grid_count, grid.grid_count)
        self.assertEqual(_ijk_sets(back), _ijk_sets(grid))
        self.assertTrue(torch.equal(back.num_voxels, grid.num_voxels))

    # -- from_dense: build one member, replicate the buffer B times (issue #775 item 4) --------

    def _check_from_dense_matches_per_member(self, num_grids, dims, ijk_min, mask, msg):
        """Multi-member from_dense equals B independent single-member builds, elementwise."""
        batched = GridBatch.from_dense(num_grids, dims, ijk_min, 1.0, 0.0, mask=mask, device="cuda")
        single = GridBatch.from_dense(1, dims, ijk_min, 1.0, 0.0, mask=mask, device="cuda")
        self.assertEqual(batched.grid_count, num_grids, msg)
        self.assertEqual(single.grid_count, 1, msg)
        self.assertTrue(torch.equal(batched.num_voxels, single.num_voxels.repeat(num_grids)), msg)
        self.assertEqual(batched.total_voxels, num_grids * single.total_voxels, msg)
        self.assertTrue(
            torch.equal(batched.ijk.jdata, single.ijk.jdata.repeat(num_grids, 1)),
            f"{msg}: voxel enumeration of replicated members differs from a single-member build",
        )
        self.assertTrue(torch.equal(batched.ijk.joffsets.cpu(), torch.arange(num_grids + 1) * single.total_voxels), msg)
        for b in range(num_grids):
            self.assertTrue(
                torch.equal(batched.bbox_at(b).cpu(), single.bbox_at(0).cpu()), f"{msg}: bbox of member {b}"
            )
            self.assertTrue(torch.equal(batched.ijk[b].jdata, single.ijk.jdata), f"{msg}: ijk of member {b}")
        # Member b of the batch is addressable as its own grid (header mGridIndex / mGridCount and
        # the handle's per-grid offset table must agree).
        idx = torch.tensor([num_grids - 1, 0], device="cuda")
        sliced = batched[idx]
        self.assertEqual(sliced.grid_count, 2, msg)
        self.assertTrue(torch.equal(sliced.ijk.jdata, single.ijk.jdata.repeat(2, 1)), msg)
        # And the whole thing matches the CPU implementation as coordinate sets.
        cpu_mask = None if mask is None else mask.cpu()
        cpu = GridBatch.from_dense(num_grids, dims, ijk_min, 1.0, 0.0, mask=cpu_mask, device="cpu")
        self._check_against_cpu(batched, cpu, f"{msg}: vs CPU")
        return batched, single

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_unmasked_matches_per_member(self, num_grids):
        # Non-cubic box, negative / tile-straddling ijk_min (crosses the 8-voxel leaf boundary and
        # a 128-voxel upper-node boundary), so leaf origins are not aligned to the box corner.
        for dims, ijk_min in (([8, 8, 8], [0, 0, 0]), ([5, 12, 7], [-3, 125, 4]), ([17, 4, 33], [-4100, 0, 4090])):
            batched, single = self._check_from_dense_matches_per_member(
                num_grids, dims, ijk_min, None, f"from_dense B={num_grids} dims={dims} ijk_min={ijk_min}"
            )
            self.assertEqual(single.total_voxels, dims[0] * dims[1] * dims[2])
            lo = torch.tensor(ijk_min, dtype=torch.int32)
            hi = lo + torch.tensor(dims, dtype=torch.int32) - 1
            for b in range(num_grids):
                self.assertTrue(torch.equal(batched.bbox_at(b).cpu(), torch.stack([lo, hi])))

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_masked_matches_per_member(self, num_grids):
        torch.manual_seed(775)
        dims = [9, 6, 14]
        ijk_min = [-2, 7, -13]
        mask = torch.rand(*dims, device="cuda") < 0.3
        batched, single = self._check_from_dense_matches_per_member(
            num_grids, dims, ijk_min, mask, f"masked from_dense B={num_grids}"
        )
        self.assertEqual(single.total_voxels, int(mask.sum().item()))
        # Selected coordinates are exactly the mask's true cells, shifted by ijk_min.
        expected = torch.nonzero(mask).to(torch.int32) + torch.tensor(ijk_min, dtype=torch.int32, device="cuda")
        self.assertEqual(set(map(tuple, expected.tolist())), _ijk_sets(single)[0])

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_mask_selects_nothing(self, num_grids):
        dims = [4, 5, 6]
        mask = torch.zeros(*dims, dtype=torch.bool, device="cuda")
        batched, single = self._check_from_dense_matches_per_member(
            num_grids, dims, [1, 2, 3], mask, f"all-false mask from_dense B={num_grids}"
        )
        self.assertEqual(batched.total_voxels, 0)
        self.assertEqual(batched.ijk.jdata.shape, (0, 3))
        self.assertTrue(torch.equal(batched.num_voxels.cpu(), torch.zeros(num_grids, dtype=torch.int64)))
        # Empty members are still valid grids: topology ops on them work and stay empty.
        self.assertEqual(batched.dual_grid().total_voxels, 0)
        self.assertEqual(batched.dual_grid().grid_count, num_grids)

    def test_from_dense_replicated_members_are_independent_grids(self):
        # Ops that address each member through its own header (voxel lookup, coarsening, and
        # concatenation with another batch) must see B identical but separate grids.
        num_grids = 5
        grid = GridBatch.from_dense(num_grids, [10, 6, 3], [1, -9, 2], 1.0, 0.0, device="cuda")
        pts = torch.tensor([[1, -9, 2], [10, -4, 4], [11, -4, 4]], dtype=torch.int32, device="cuda")
        idx = grid.ijk_to_index(JaggedTensor([pts] * num_grids), cumulative=True)
        for b in range(num_grids):
            self.assertEqual(idx[b].jdata.tolist(), [b * 180, b * 180 + 179, -1])
        coarse = grid.coarsened_grid(2)
        coarse_single = GridBatch.from_dense(1, [10, 6, 3], [1, -9, 2], 1.0, 0.0, device="cuda").coarsened_grid(2)
        self.assertTrue(torch.equal(coarse.ijk.jdata, coarse_single.ijk.jdata.repeat(num_grids, 1)))
        other = _build(_tricky_batches()["mixed_sizes"], "cuda")
        cat = GridBatch.from_cat([grid, other])
        self.assertEqual(cat.grid_count, num_grids + other.grid_count)
        self.assertTrue(torch.equal(cat.ijk.jdata, torch.cat([grid.ijk.jdata, other.ijk.jdata])))


if __name__ == "__main__":
    unittest.main()
