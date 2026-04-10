# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for find_unique_pixels, ported from DeduplicatePixelsTest.cpp."""

import unittest

import torch

from fvdb import JaggedTensor
from fvdb.functional._gaussian_tile_intersection import _find_unique_pixels as find_unique_pixels

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


class TestFindUniquePixels(unittest.TestCase):

    def _run(self, pixels_jt, w=IMAGE_WIDTH, h=IMAGE_HEIGHT):
        return find_unique_pixels(pixels_jt, image_width=w, image_height=h)

    def test_empty(self):
        pixels = JaggedTensor(torch.empty(0, 2, dtype=torch.int32, device="cuda"))
        unique, inv, has_dups = self._run(pixels)
        self.assertFalse(has_dups)
        self.assertEqual(inv.size(0), 0)

    def test_single_pixel(self):
        coords = torch.tensor([[5, 10]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([coords])
        unique, inv, has_dups = self._run(pixels)
        self.assertFalse(has_dups)
        self.assertEqual(len(unique.jdata), 1)

    def test_all_unique(self):
        coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 3]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([coords])
        unique, inv, has_dups = self._run(pixels)
        self.assertFalse(has_dups)
        self.assertEqual(len(unique.jdata), 5)
        self.assertEqual(inv.size(0), 5)

    def test_some_duplicates(self):
        coords = torch.tensor([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([coords])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique.jdata), 3)
        self.assertEqual(inv.size(0), 4)
        inv_cpu = inv.cpu()
        self.assertEqual(inv_cpu[0].item(), inv_cpu[2].item())
        self.assertNotEqual(inv_cpu[1].item(), inv_cpu[3].item())

    def test_all_same_pixel(self):
        coords = torch.tensor([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([coords])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique.jdata), 1)
        self.assertEqual(inv.size(0), 4)
        inv_cpu = inv.cpu()
        for i in range(4):
            self.assertEqual(inv_cpu[i].item(), 0)

    def test_multi_batch_no_duplicates(self):
        batch0 = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32, device="cuda")
        batch1 = torch.tensor([[0, 0], [2, 2]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([batch0, batch1])
        unique, inv, has_dups = self._run(pixels)
        self.assertFalse(has_dups)
        self.assertEqual(len(unique.jdata), 4)
        self.assertEqual(len(unique), 2)

    def test_multi_batch_with_duplicates(self):
        batch0 = torch.tensor([[0, 0], [1, 1], [0, 0]], dtype=torch.int32, device="cuda")
        batch1 = torch.tensor([[0, 0], [3, 3]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([batch0, batch1])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique), 2)
        self.assertEqual(len(unique.jdata), 4)
        self.assertEqual(inv.size(0), 5)
        inv_cpu = inv.cpu()
        self.assertEqual(inv_cpu[0].item(), inv_cpu[2].item())

    def test_multi_batch_all_same_pixel(self):
        batch0 = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.int32, device="cuda")
        batch1 = torch.tensor([[2, 2], [2, 2]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([batch0, batch1])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique), 2)
        self.assertEqual(len(unique.jdata), 2)
        offsets = unique.joffsets.cpu()
        self.assertEqual(offsets[0].item(), 0)
        self.assertEqual(offsets[1].item(), 1)
        self.assertEqual(offsets[2].item(), 2)
        inv_cpu = inv.cpu()
        self.assertEqual(inv_cpu[0].item(), inv_cpu[1].item())
        self.assertEqual(inv_cpu[0].item(), inv_cpu[2].item())
        self.assertEqual(inv_cpu[3].item(), inv_cpu[4].item())
        self.assertNotEqual(inv_cpu[0].item(), inv_cpu[3].item())

    def test_round_trip_some_duplicates(self):
        coords = torch.tensor([[3, 7], [1, 2], [3, 7], [5, 5], [1, 2], [9, 0]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([coords])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique.jdata), 4)
        reconstructed = unique.jdata.index_select(0, inv)
        self.assertTrue(torch.equal(reconstructed.cpu(), coords.cpu().to(reconstructed.dtype)))

    def test_round_trip_multi_batch(self):
        batch0 = torch.tensor([[2, 3], [4, 5], [2, 3]], dtype=torch.int32, device="cuda")
        batch1 = torch.tensor([[6, 7], [6, 7], [8, 9]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([batch0, batch1])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        original_jdata = pixels.jdata
        reconstructed = unique.jdata.index_select(0, inv)
        self.assertTrue(torch.equal(reconstructed.cpu(), original_jdata.cpu().to(reconstructed.dtype)))

    def test_jagged_tensor_offsets(self):
        batch0 = torch.tensor([[0, 0], [0, 0], [1, 1]], dtype=torch.int32, device="cuda")
        batch1 = torch.tensor([[2, 2]], dtype=torch.int32, device="cuda")
        batch2 = torch.tensor([[3, 3], [4, 4], [3, 3], [4, 4]], dtype=torch.int32, device="cuda")
        pixels = JaggedTensor([batch0, batch1, batch2])
        unique, inv, has_dups = self._run(pixels)
        self.assertTrue(has_dups)
        self.assertEqual(len(unique), 3)
        self.assertEqual(len(unique.jdata), 5)
        offsets = unique.joffsets.cpu()
        self.assertEqual(offsets[0].item(), 0)
        self.assertEqual(offsets[1].item(), 2)
        self.assertEqual(offsets[2].item(), 3)
        self.assertEqual(offsets[3].item(), 5)


if __name__ == "__main__":
    unittest.main()
