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

    @parameterized.expand([("mixed_sizes",), ("empty_members",), ("tile_boundaries",)])
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


if __name__ == "__main__":
    unittest.main()
