# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Regression test for pruning a batch that contains a single-voxel grid.

A grid with exactly one voxel yields a length-1 per-grid mask list. The CPU prune path used
`mask.index(bidx).jdata().squeeze()`, and `squeeze()` collapses a `[1]` tensor to a 0-D scalar,
so the subsequent `.accessor<bool, 1>()` threw "TensorAccessor expected 1 dims but tensor has 0".
"""

import unittest

import torch
from parameterized import parameterized

from fvdb import GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]


class TestPruneSingleVoxel(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_prune_batch_with_single_voxel_grid(self, device):
        coords = [
            torch.tensor([[0, 0, 0]], dtype=torch.int32),  # single-voxel grid (triggers the bug)
            torch.tensor([[5, 5, 5], [5, 5, 6], [5, 6, 5]], dtype=torch.int32),
        ]
        g = GridBatch.from_ijk(JaggedTensor([c.to(device) for c in coords]))

        # Keep-all mask (one bool per voxel, per grid): pruning leaves the topology unchanged.
        keep_all = JaggedTensor([torch.ones(t.shape[0], dtype=torch.bool, device=t.device) for t in g.ijk.unbind()])
        pruned = g.pruned_grid(keep_all)
        self.assertEqual(pruned.grid_count, 2)
        got = [set(map(tuple, t.cpu().to(torch.int64).tolist())) for t in pruned.ijk.unbind()]
        exp = [set(map(tuple, c.to(torch.int64).tolist())) for c in coords]
        self.assertEqual(got, exp, "pruned topology should match the input")

        # Dropping the single voxel must yield an empty first grid (still no crash).
        drop_first = JaggedTensor(
            [
                (
                    torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
                    if i == 0
                    else torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
                )
                for i, t in enumerate(g.ijk.unbind())
            ]
        )
        pruned2 = g.pruned_grid(drop_first)
        counts = [t.shape[0] for t in pruned2.ijk.unbind()]
        self.assertEqual(counts[0], 0, "single voxel should be pruned away")
        self.assertEqual(counts[1], 3)


if __name__ == "__main__":
    unittest.main()
