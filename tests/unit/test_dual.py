# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]


def _ijk_sets(grid: GridBatch):
    """Return, per batch item, the set of (i, j, k) tuples of active voxels."""
    return [set(map(tuple, t.detach().cpu().to(torch.int64).tolist())) for t in grid.ijk.unbind()]


def _build_grid(ijks, device, voxel_sizes=1.0, origins=0.0):
    """Build a GridBatch from a list of per-item (Ni, 3) int coordinate tensors."""
    jt = JaggedTensor([t.to(device=device, dtype=torch.int32) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=voxel_sizes, origins=origins)


def _build_padded_grid(grid: GridBatch, bmin: int, bmax: int, exclude_border: bool = False) -> GridBatch:
    """Wrapper around the low-level ``build_padded_grid`` binding (generic [bmin, bmax] box)."""
    return GridBatch(data=fvdb._fvdb_cpp.build_padded_grid(grid.data, bmin, bmax, exclude_border))


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_combos)
    def test_world_to_dual(self, device):
        torch.manual_seed(42)
        np.random.seed(42)

        # Raw grid:
        # o o x x x
        # o o o x x
        # o x o x x
        # o x x x x
        # x x x x x

        # dual_grid():
        # o o o x x
        # o o o o x
        # o o o o x
        # o o o o x
        # o o x x x

        # dual_grid(exclude_border=True):
        # o x x x x
        # x x x x x
        # x x x x x
        # x x x x x
        # x x x x x

        # dual_grid().dual_grid(exclude_border=True):
        # o o x x x
        # o o o x x
        # o o o x x
        # o x x x x
        # x x x x x

        ij = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 2], [3, 0]], device=device)
        ijk = torch.cat(
            [
                torch.nn.functional.pad(ij, (0, 1), mode="constant", value=0),
                torch.nn.functional.pad(ij, (0, 1), mode="constant", value=1),
            ],
            dim=0,
        )
        grid = GridBatch.from_ijk(JaggedTensor([ijk]))

        _grid = grid.dual_grid(exclude_border=True)
        _target_ijk = torch.tensor([[0, 0, 0]], device=device)
        assert (_grid.ijk.jdata == _target_ijk).all(), _grid.ijk.jdata

        _grid = grid.dual_grid().dual_grid(exclude_border=True)
        _target_ijk = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 2, 0],
                [1, 2, 1],
                [2, 0, 0],
                [2, 0, 1],
                [2, 1, 0],
                [2, 1, 1],
                [2, 2, 0],
                [2, 2, 1],
                [3, 0, 0],
                [3, 0, 1],
            ],
            device=device,
        )
        assert (_grid.ijk.jdata == _target_ijk).all(), _grid.ijk.jdata

    @parameterized.expand(all_device_combos)
    def test_dual_grid_transform_is_dual_lattice(self, device):
        # dual_grid places its result on the corner (dual) lattice: the source's dual
        # (corner-aligned) transform becomes the result's primal transform, so the reported origin
        # shifts by exactly -0.5 voxel while the voxel size is unchanged.
        torch.manual_seed(7)
        ijks = [torch.randint(-30, 30, (500, 3), dtype=torch.int32)]
        voxel_sizes = [0.3, 0.5, 0.7]
        origins = [1.25, -2.0, 0.5]
        g = _build_grid(ijks, device, voxel_sizes, origins)
        d = g.dual_grid()
        self.assertTrue(torch.allclose(d.voxel_sizes, g.voxel_sizes), "dual_grid changed the voxel size")
        expected_origin = g.origins - 0.5 * g.voxel_sizes
        self.assertTrue(
            torch.allclose(d.origins, expected_origin),
            f"dual origin {d.origins.tolist()} != base - 0.5*voxel {expected_origin.tolist()}",
        )

    @parameterized.expand(all_device_combos)
    def test_build_padded_grid_preserves_transform(self, device):
        # A plain padded grid lives on the *same* lattice as the source, so build_padded_grid must
        # keep the source's primal origin and voxel size for every box -- it must NOT swap
        # primal/dual transforms the way dual_grid does. Regression test: buildPaddedGrid used to
        # unconditionally swap, which leaked dual semantics into this generic padding primitive
        # (openvdb/fvdb-core#710 review).
        torch.manual_seed(8)
        ijks = [torch.randint(-30, 30, (500, 3), dtype=torch.int32)]
        voxel_sizes = [0.3, 0.5, 0.7]
        origins = [1.25, -2.0, 0.5]
        g = _build_grid(ijks, device, voxel_sizes, origins)
        for bmin, bmax in [(0, 0), (0, 1), (-1, 0), (-1, 1), (0, 2), (-2, 1)]:
            for exclude_border in (False, True):
                r = _build_padded_grid(g, bmin, bmax, exclude_border)
                self.assertTrue(
                    torch.allclose(r.voxel_sizes, g.voxel_sizes),
                    f"box=({bmin},{bmax}) exclude_border={exclude_border}: voxel size changed",
                )
                self.assertTrue(
                    torch.allclose(r.origins, g.origins),
                    f"box=({bmin},{bmax}) exclude_border={exclude_border}: "
                    f"origin changed to {r.origins.tolist()} (base {g.origins.tolist()})",
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the mask-morphology padded-grid path")
class TestPadGridCuda(unittest.TestCase):
    """Tests for the leaf-mask-morphology implementation of dual_grid / build_padded_grid.

    The CPU path (unchanged proxy-grid builder) is used as the ground truth; the CUDA path
    must produce the same active-voxel sets.
    """

    def _assert_same_voxels(self, g1: GridBatch, g2: GridBatch, msg: str = ""):
        s1, s2 = _ijk_sets(g1), _ijk_sets(g2)
        self.assertEqual(len(s1), len(s2), f"{msg}: batch size mismatch")
        for b, (a, c) in enumerate(zip(s1, s2)):
            self.assertEqual(a, c, f"{msg}: voxel set mismatch in batch item {b}")

    def test_dual_grid_cuda_cpu_parity(self):
        torch.manual_seed(0)
        for batch_size in (1, 3, 5):
            ijks = [torch.randint(-40, 40, (2000, 3), dtype=torch.int32) for _ in range(batch_size)]
            voxel_sizes = [[0.1 * (b + 1)] * 3 for b in range(batch_size)]
            origins = [[float(b), -0.5 * b, 0.25 * b] for b in range(batch_size)]
            g_cpu = _build_grid(ijks, "cpu", voxel_sizes, origins)
            g_cuda = _build_grid(ijks, "cuda", voxel_sizes, origins)
            for exclude_border in (False, True):
                d_cpu = g_cpu.dual_grid(exclude_border=exclude_border)
                d_cuda = g_cuda.dual_grid(exclude_border=exclude_border)
                self._assert_same_voxels(d_cpu, d_cuda, f"dual_grid bs={batch_size} exclude_border={exclude_border}")
                # The transform swap (dual origin/voxel size) must be device independent.
                self.assertTrue(torch.allclose(d_cpu.voxel_sizes, d_cuda.voxel_sizes.cpu()))
                self.assertTrue(torch.allclose(d_cpu.origins, d_cuda.origins.cpu()))

    def test_build_padded_grid_generic_boxes(self):
        torch.manual_seed(1)
        ijks = [torch.randint(-20, 20, (3000, 3), dtype=torch.int32)]
        boxes = [(0, 1), (-1, 0), (-1, 1), (0, 2), (-2, 1), (0, 0)]
        for (bmin, bmax), exclude_border in itertools.product(boxes, (False, True)):
            g_cpu = _build_grid(ijks, "cpu")
            g_cuda = _build_grid(ijks, "cuda")
            r_cpu = _build_padded_grid(g_cpu, bmin, bmax, exclude_border)
            r_cuda = _build_padded_grid(g_cuda, bmin, bmax, exclude_border)
            self._assert_same_voxels(
                r_cpu, r_cuda, f"build_padded_grid box=({bmin},{bmax}) exclude_border={exclude_border}"
            )

    def test_padded_grid_symmetric_box_equals_dilation(self):
        # Padding by the full box [-1, 1]^3 (no border exclusion) is the symmetric 26-connected
        # dilation by one voxel.
        torch.manual_seed(2)
        ijks = [torch.randint(-15, 15, (1500, 3), dtype=torch.int32)]
        g = _build_grid(ijks, "cuda")
        padded = _build_padded_grid(g, -1, 1, False)
        dilated = g.dilated_grid(1)
        self._assert_same_voxels(padded, dilated, "pad[-1,1] vs dilate(1)")

    def test_dual_grid_leaf_boundary_crossing(self):
        # Coordinates on/around leaf-local boundaries (leaves are 8^3) in all axes.
        coords = torch.tensor(
            [[0, 0, 0], [7, 7, 7], [7, 3, 0], [15, 15, 15], [-1, -1, -1], [8, 8, 8], [7, 0, 7]],
            dtype=torch.int32,
        )
        g_cpu = _build_grid([coords], "cpu")
        g_cuda = _build_grid([coords], "cuda")
        for exclude_border in (False, True):
            self._assert_same_voxels(
                g_cpu.dual_grid(exclude_border=exclude_border),
                g_cuda.dual_grid(exclude_border=exclude_border),
                f"leaf-boundary exclude_border={exclude_border}",
            )

        # Explicit check: the dual of a single corner voxel (7,7,7) is the 2x2x2 block {7,8}^3,
        # which straddles the leaf boundary in every axis.
        g = _build_grid([torch.tensor([[7, 7, 7]], dtype=torch.int32)], "cuda")
        expected = set(itertools.product((7, 8), (7, 8), (7, 8)))
        self.assertEqual(_ijk_sets(g.dual_grid())[0], expected)

    def test_dual_grid_root_tile_crossing(self):
        # Upper/root nodes span 4096^3 index space; content at a tile face must speculate and
        # populate the neighboring tile.
        coords = torch.tensor([[4095, 4095, 4095], [4095, 0, 0], [0, 4095, 0], [0, 0, 4095]], dtype=torch.int32)
        g_cpu = _build_grid([coords], "cpu")
        g_cuda = _build_grid([coords], "cuda")
        d_cpu = g_cpu.dual_grid()
        d_cuda = g_cuda.dual_grid()
        self._assert_same_voxels(d_cpu, d_cuda, "root-tile crossing")
        # The dual of (4095,4095,4095) must include the voxel (4096,4096,4096) in the next tile.
        self.assertIn((4096, 4096, 4096), _ijk_sets(d_cuda)[0])

    def test_dual_grid_batch_with_empty_item(self):
        torch.manual_seed(3)
        ijks = [
            torch.randint(-10, 10, (500, 3), dtype=torch.int32),
            torch.zeros((0, 3), dtype=torch.int32),
            torch.randint(-10, 10, (700, 3), dtype=torch.int32),
        ]
        g_cpu = _build_grid(ijks, "cpu")
        g_cuda = _build_grid(ijks, "cuda")
        d_cpu = g_cpu.dual_grid()
        d_cuda = g_cuda.dual_grid()
        self._assert_same_voxels(d_cpu, d_cuda, "batch with empty item")
        counts = [t.shape[0] for t in d_cuda.ijk.unbind()]
        self.assertEqual(counts[1], 0, "empty batch item should stay empty")
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[2], 0)

    def test_dual_grid_erode_to_empty(self):
        # A single voxel has no coordinate whose full 2x2x2 forward neighborhood is active, so the
        # exclude-border dual is empty.
        for device in ("cpu", "cuda"):
            g = _build_grid([torch.tensor([[5, 5, 5]], dtype=torch.int32)], device)
            self.assertEqual(g.dual_grid(exclude_border=True).total_voxels, 0)

    def test_dual_grid_peak_memory(self):
        # The old coordinate-list path allocated ~160 B of torch tensors per input voxel (8N int32
        # coords + two 8N int32 batch-index arrays). The mask-morphology path allocates only the
        # output grid buffer through torch; its scratch is raw cudaMalloc, invisible to these stats.
        n = 128
        rng = torch.arange(n, dtype=torch.int32)
        ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
        coords = torch.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)], dim=1)  # n^3 voxels
        grid = _build_grid([coords], "cuda")

        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        dual = grid.dual_grid()
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - base

        # The old path would peak at > 300 MiB of torch allocations for a 128^3 grid; the new path
        # should be a small multiple of the (tens of MiB) output grid buffer.
        self.assertGreater(dual.total_voxels, grid.total_voxels)
        self.assertLess(
            peak_extra,
            150 * 1024 * 1024,
            f"dual_grid torch peak {peak_extra / 1024 / 1024:.1f} MiB is unexpectedly large",
        )


if __name__ == "__main__":
    unittest.main()
