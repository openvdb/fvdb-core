# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for grid-topology ops on sliced / non-contiguous GridBatch views.

A sliced or indexed GridBatch (``grid[idx]``, ``grid[a:b]``) is a *view*: it shares the
underlying NanoVDB handle (which keeps every grid) and only shrinks ``grid_count`` and the
per-grid metadata, so ``grid_count < nanoGridHandle().gridCount()`` and each item's grid is
located by a byte offset rather than by physical index. The topology builders must resolve
each item's grid via that byte offset (GridBatchData::deviceGridPtrAt/hostGridPtrAt), not
``deviceGrid(i)``. Before the fix these returned the wrong grids (or corrupted the heap) on
any indexed batch. Each op below is compared, on a sliced view, against the same grids built
as their own contiguous batch.
"""

import unittest

import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]


def _ijk_sets(grid: GridBatch):
    return [set(map(tuple, t.cpu().to(torch.int64).tolist())) for t in grid.ijk.unbind()]


def _build(ijks, device):
    jt = JaggedTensor([t.to(device=device, dtype=torch.int32) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=1.0, origins=0.0)


class TestSlicedBatchTopology(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_topology_ops_on_sliced_views(self, device):
        torch.manual_seed(17)
        # A 4-grid batch with a mix of tiny and larger grids in disjoint regions.
        coords = [
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.randint(-8, 8, (200, 3), dtype=torch.int32),
            torch.tensor([[16, 16, 16], [16, 16, 17], [16, 17, 16]], dtype=torch.int32),
            torch.randint(30, 50, (150, 3), dtype=torch.int32),
        ]
        full = _build(coords, device)

        # Each op: name -> callable(GridBatch) -> GridBatch. Covers every fast/fallback path
        # flagged in the sliced-batch audit (conv dilate/pad, coarsen incl. factor-1 identity
        # copy, refine, dilate incl. the dilation-0 byte-clone branch, and prune via clip).
        ops = [
            ("conv_grid k3s1", lambda g: g.conv_grid(kernel_size=3, stride=1)),
            ("conv_grid k2s1", lambda g: g.conv_grid(kernel_size=2, stride=1)),
            ("conv_transpose_grid k3s1", lambda g: g.conv_transpose_grid(kernel_size=3, stride=1)),
            ("coarsened_grid x2", lambda g: g.coarsened_grid(2)),
            ("coarsened_grid x1 (identity)", lambda g: g.coarsened_grid(1)),
            ("refined_grid x2", lambda g: g.refined_grid(2)),
            ("dilated_grid 1", lambda g: g.dilated_grid(1)),
            ("dilated_grid 0 (identity)", lambda g: g.dilated_grid(0)),
            # Masked refine exercises pruneGrid (BuildPrunedGrid) on the (sliced) grid: keep voxels
            # with even coordinate sum, then subdivide. The mask is derived from each grid's own
            # ijk, so it corresponds for the view and its contiguous reference.
            (
                "refined_grid x2 masked (prune)",
                lambda g: g.refined_grid(2, mask=JaggedTensor([((t.sum(-1) % 2) == 0) for t in g.ijk.unbind()])),
            ),
        ]

        # Non-contiguous selections: tail, gap, reversed pair, single middle item.
        for sel in ([1, 2, 3], [0, 2], [3, 1], [2]):
            view = full[sel]
            self.assertEqual(view.grid_count, len(sel), f"sel={sel}: view grid_count")
            ref = _build([coords[s] for s in sel], device)
            for name, op in ops:
                got = _ijk_sets(op(view))
                exp = _ijk_sets(op(ref))
                self.assertEqual(len(got), len(sel), f"{name} sel={sel}: result grid_count")
                self.assertEqual(got, exp, f"{name} sel={sel}: voxel sets differ")

    @parameterized.expand(all_device_combos)
    def test_merge_and_inject_on_sliced_views(self, device):
        coords_a = [
            torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
            torch.tensor([[10, 0, 0], [11, 0, 0]], dtype=torch.int32),
            torch.tensor([[20, 0, 0], [21, 0, 0]], dtype=torch.int32),
        ]
        coords_b = [
            torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.int32),
            torch.tensor([[11, 0, 0], [12, 0, 0]], dtype=torch.int32),
            torch.tensor([[21, 0, 0], [22, 0, 0]], dtype=torch.int32),
        ]
        full_a = _build(coords_a, device)
        full_b = _build(coords_b, device)

        # Reverse and skip items so logical batch indices differ from physical handle indices.
        sel = [2, 0]
        view_a = full_a[sel]
        view_b = full_b[sel]
        ref_a = _build([coords_a[i] for i in sel], device)
        ref_b = _build([coords_b[i] for i in sel], device)

        self.assertEqual(_ijk_sets(view_a.merged_grid(view_b)), _ijk_sets(ref_a.merged_grid(ref_b)))

        def coordinate_values(grid):
            ijk = grid.ijk
            values = (100 * ijk.jdata[:, 0] + 10 * ijk.jdata[:, 1] + ijk.jdata[:, 2]).float()
            return grid.jagged_like(values)

        got = view_b.inject_from(view_a, coordinate_values(view_a), default_value=-1)
        expected = ref_b.inject_from(ref_a, coordinate_values(ref_a), default_value=-1)
        self.assertTrue(torch.equal(got.jdata, expected.jdata))


if __name__ == "__main__":
    unittest.main()
