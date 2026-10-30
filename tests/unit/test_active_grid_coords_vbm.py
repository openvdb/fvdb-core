# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the VBM-backed CUDA per-active-voxel iteration path used by
# Grid.ijk / GridBatch.ijk (ActiveGridCoords) and morton/hilbert (SerializeEncode):
# the CUDA results must match the CPU leaf-scan path bitwise, including through
# sliced GridBatch views (which build their own VBMs) and for batches containing
# empty grids.
import unittest

import torch

import fvdb


def _random_ijk(num: int, box: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(box * box * box, generator=gen)[:num]
    return torch.stack([perm // (box * box), (perm // box) % box, perm % box], dim=1).to(torch.int32)


class ActiveGridCoordsVbmTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a CUDA device")
        self.device = torch.device("cuda:0")

    def _make_batches(self, ijks):
        cpu = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([i for i in ijks]))
        cuda = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([i.to(self.device) for i in ijks]))
        return cpu, cuda

    def test_ijk_matches_cpu(self):
        ijks = [_random_ijk(100, 16, 1), _random_ijk(512, 32, 2), _random_ijk(513, 32, 3), _random_ijk(9000, 48, 4)]
        cpu, cuda = self._make_batches(ijks)
        self.assertTrue(torch.equal(cuda.ijk.jdata.cpu(), cpu.ijk.jdata))

    def test_ijk_on_sliced_batch(self):
        ijks = [_random_ijk(600, 16, 5), _random_ijk(700, 16, 6), _random_ijk(800, 16, 7)]
        cpu, cuda = self._make_batches(ijks)
        # The parent's VBMs are built first; the view builds its own.
        _ = cuda.ijk
        self.assertTrue(torch.equal(cuda[1:3].ijk.jdata.cpu(), cpu[1:3].ijk.jdata))
        self.assertTrue(torch.equal(cuda[1:3].ijk.jdata.cpu(), cpu.ijk[1:3].jdata))

    def test_ijk_with_empty_grid(self):
        ijks = [torch.empty(0, 3, dtype=torch.int32), _random_ijk(100, 16, 8)]
        cpu, cuda = self._make_batches(ijks)
        self.assertEqual(cuda.ijk[0].jdata.shape[0], 0)
        self.assertTrue(torch.equal(cuda.ijk.jdata.cpu(), cpu.ijk.jdata))

    def test_ijk_large_batch_with_empty_grids(self):
        # Empty grids at the start, middle and end own no VBM blocks in the batched build and launch.
        empty = torch.empty(0, 3, dtype=torch.int32)
        ijks = [empty] + [_random_ijk(50 + 97 * k, 24, 20 + k) for k in range(12)] + [empty, empty]
        ijks.insert(6, empty)
        cpu, cuda = self._make_batches(ijks)
        self.assertTrue(torch.equal(cuda.ijk.jdata.cpu(), cpu.ijk.jdata))
        self.assertTrue(torch.equal(cuda[3:11].ijk.jdata.cpu(), cpu[3:11].ijk.jdata))

    def test_ijk_inside_cuda_graph_capture(self):
        # The VBM build cannot be captured, so the first use must come before capture
        _, cuda = self._make_batches([_random_ijk(600, 16, 40), _random_ijk(700, 16, 41)])
        with self.assertRaisesRegex(RuntimeError, "Run the op once before capturing"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                cuda.ijk

        expected = cuda.ijk.jdata.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = cuda.ijk.jdata
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, expected))

    def test_morton_matches_cpu(self):
        ijks = [_random_ijk(1000, 24, 9), _random_ijk(1500, 24, 10)]
        cpu, cuda = self._make_batches(ijks)
        self.assertTrue(torch.equal(cuda.morton().jdata.cpu(), cpu.morton().jdata))


if __name__ == "__main__":
    unittest.main()
