# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import unittest
from typing import Callable

import numpy as np
import torch
from parameterized import parameterized_class

import fvdb


@parameterized_class(
    ("device", "element_shape"), list(itertools.product(["cuda", "cpu"], [(), (3,), (3, 2), (3, 2, 1)]))
)
class InjectionTests(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)

        # This is a workaround for the fact that the parameterized_class decorator
        # creates red squiggles with mypy, which does not understand the
        # parameterized_class decorator.
        self.device: torch.device = torch.device(self.device)
        self.element_shape: list[int] = self.element_shape

        self.batch_size: int = 2
        self.voxel_size: float = 0.4

        self.grid_batch1: fvdb.GridBatch = self.build_random_gridbatch()
        self.grid_batch2: fvdb.GridBatch = self.build_random_gridbatch()
        self.grid_batch3: fvdb.GridBatch = self.build_random_gridbatch()
        self.grid_batch12: fvdb.GridBatch = self.grid_batch1.merged_grid(self.grid_batch2)
        self.grid_batch23: fvdb.GridBatch = self.grid_batch2.merged_grid(self.grid_batch3)
        self.grid_batch123: fvdb.GridBatch = self.grid_batch12.merged_grid(self.grid_batch3)

    @staticmethod
    def get_point_list(npc: list, device: torch.device | str) -> list[torch.Tensor]:
        batch_size = len(npc)
        plist = []
        for i in range(batch_size):
            ni = npc[i]
            plist.append(torch.randn((ni, 3), dtype=torch.float32, device=device, requires_grad=False))
        return plist

    def build_random_gridbatch(self) -> fvdb.GridBatch:
        npc = torch.randint(low=10, high=1000, size=(self.batch_size,), device=self.device).tolist()
        plist = self.get_point_list(npc, self.device)
        pc_jagged = fvdb.JaggedTensor(plist)
        return fvdb.gridbatch_from_points(pc_jagged, voxel_sizes=[[self.voxel_size] * 3] * self.batch_size)

    def build_sidecar(self, grid_batch: fvdb.GridBatch, build_func: Callable) -> fvdb.JaggedTensor:
        sizes = [grid_batch.total_voxels] + list(self.element_shape)
        return grid_batch.jagged_like(build_func(*sizes, device=self.device))

    @staticmethod
    def inject_bruteforce(
        src_grid: fvdb.GridBatch,
        dst_grid: fvdb.GridBatch,
        src_features: fvdb.JaggedTensor,
        dst_features: fvdb.JaggedTensor,
    ) -> fvdb.JaggedTensor:
        src_ijk = src_grid.ijk
        src_idx_in_dst = dst_grid.ijk_to_index(src_ijk)
        src_idx_mask = src_idx_in_dst >= 0
        src_idx_in_dst = src_idx_in_dst[src_idx_mask]
        for i in range(len(src_features)):
            dst_features[i].jdata[src_idx_in_dst[i].jdata] = src_features[i].jdata[src_idx_mask[i].jdata]
        return dst_features

    def test_inject_in_place_subset_into_superset(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)

        # We construct two grids grid12 = union(grid1, grid2) and grid23 = union(grid2, grid3).
        # grid12 and grid23 have a common grid2 but are not strictly overlapping.
        sidecar12: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch12, torch.zeros)
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.zeros)

        sidecar23_ref = self.build_sidecar(self.grid_batch23, torch.zeros)

        # Build a reference sidecar for grid23 by injecting data from grid2 and grid3 into it
        self.grid_batch2.inject_to(self.grid_batch23, sidecar2, sidecar23_ref)  # sidecar2 -> sidecar23_ref
        self.grid_batch23.inject_from(self.grid_batch3, sidecar3, sidecar23_ref)  # sidecar3 -> sidecar23_ref

        # Now build a new sidecar for grid12 by injecting data from sidecar1 and sidecar2 into it
        self.grid_batch1.inject_to(self.grid_batch12, sidecar1, sidecar12)  # sidecar1 -> sidecar12
        self.grid_batch12.inject_from(self.grid_batch2, sidecar2, sidecar12)  # sidecar2 -> sidecar12

        # Now Inject data from grid_12 into grid_23, which should only update the voxels corresponding to grid2
        # and not affect the voxels corresponding to grid3.
        self.grid_batch12.inject_to(self.grid_batch23, sidecar12, sidecar23)  # sidecar12 -> sidecar23

        # Sidecar23 should not equal sidecar23_ref yet. But they should be equal at voxels corresponding to grid2.
        self.assertFalse(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))

        # Now inject data from sidecar3 into sidecar23, which should update the voxels corresponding to grid3.
        # making sidecar23 equal to sidecar23_ref.
        self.grid_batch23.inject_from(self.grid_batch3, sidecar3, sidecar23)
        self.assertTrue(torch.equal(sidecar23.jdata, sidecar23_ref.jdata))

    def test_inject_in_place_superset_into_subset(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)

        # Sidecar 23 holds the union of grid2 and grid3
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.zeros)

        # Brute force copies
        sidecar1_bf = sidecar1.clone().detach()
        sidecar2_bf = sidecar2.clone().detach()
        sidecar3_bf = sidecar3.clone().detach()
        sidecar23_bf = self.build_sidecar(self.grid_batch23, torch.zeros)

        # Run ours
        self.grid_batch2.inject_to(self.grid_batch23, sidecar2, sidecar23)  # sidecar2 -> sidecar23
        self.grid_batch1.inject_to(
            self.grid_batch23, sidecar1, sidecar23
        )  # sidecar1 -> sidecar23 (non overlapping, no effect)
        self.grid_batch23.inject_from(self.grid_batch3, sidecar3, sidecar23)  # sidecar3 -> sidecar23

        # Now inject superset sidecar23 into subset sidecar2
        self.grid_batch2.inject_from(self.grid_batch23, sidecar23, sidecar2)  # sidecar23 -> sidecar2

        # Run bruteforce
        self.inject_bruteforce(
            self.grid_batch2, self.grid_batch23, sidecar2_bf, sidecar23_bf
        )  # sidecar2 -> sidecar23_bf
        self.inject_bruteforce(
            self.grid_batch1, self.grid_batch23, sidecar1_bf, sidecar23_bf
        )  # sidecar1_bf -> sidecar23_bf (non overlapping, no effect)
        self.inject_bruteforce(
            self.grid_batch3, self.grid_batch23, sidecar3_bf, sidecar23_bf
        )  # sidecar3_bf -> sidecar23_bf

        # Now inject superset sidecar23_bf into subset sidecar2
        self.inject_bruteforce(
            self.grid_batch23, self.grid_batch2, sidecar23_bf, sidecar2_bf
        )  # sidecar23_bf -> sidecar2_bf

        self.assertTrue(torch.equal(sidecar23.jdata, sidecar23_bf.jdata), "sidecar23 and sidecar23_bf should be equal")
        self.assertTrue(torch.equal(sidecar2.jdata, sidecar2_bf.jdata), "sidecar2 and sidecar2_bf should be equal")
        self.assertTrue(torch.equal(sidecar1.jdata, sidecar1_bf.jdata), "sidecar1 and sidecar1_bf should be equal")
        self.assertTrue(torch.equal(sidecar3.jdata, sidecar3_bf.jdata), "sidecar3 and sidecar3_bf should be equal")

    def test_inject_in_place_subset_into_superset_backprop(self):
        # There are three grids, grid1, grid2, and grid3.
        # We construct two grids grid12 = union(grid1, grid2) and grid23 = union(grid2, grid3).
        # grid12 and grid23 have a common grid2 but are not strictly overlapping.

        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar12: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch12, torch.zeros)

        sidecar1.requires_grad = True
        sidecar2.requires_grad = True

        # Inject sidecar1 and sidecar2 into sidecar12
        # self.grid_batch12.inject_from(self.grid_batch1, sidecar1, sidecar12)  # sidecar1 -> sidecar12
        self.grid_batch1.inject_to(self.grid_batch12, sidecar1, sidecar12)  # sidecar1 -> sidecar12
        self.grid_batch12.inject_from(self.grid_batch2, sidecar2, sidecar12)  # sidecar2 -> sidecar12

        # Compute a loss on sidecar12 and compute gradients
        loss = sidecar12.jdata.sum()
        loss.backward()

        self.assertTrue(sidecar12.requires_grad, "sidecar12 should require gradients")
        assert sidecar1.jdata.grad is not None, "sidecar1 should have gradients"
        assert sidecar2.jdata.grad is not None, "sidecar2 should have gradients"
        sidecar1_grad = sidecar1.jdata.grad.clone().detach()
        sidecar2_grad = sidecar2.jdata.grad.clone().detach()

        # Now do the same thing with bruteforce injection
        sidecar1_bf = sidecar1.clone().detach()
        sidecar2_bf = sidecar2.clone().detach()
        sidecar12_bf: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch12, torch.zeros)

        sidecar1_bf.requires_grad = True
        sidecar2_bf.requires_grad = True

        # Inject sidecar1 and sidecar2 into sidecar12 using bruteforce
        sidecar12_bf = InjectionTests.inject_bruteforce(self.grid_batch1, self.grid_batch12, sidecar1_bf, sidecar12_bf)
        sidecar12_bf = InjectionTests.inject_bruteforce(self.grid_batch2, self.grid_batch12, sidecar2_bf, sidecar12_bf)

        self.assertTrue(torch.equal(sidecar12.jdata, sidecar12_bf.jdata), "sidecar12 and sidecar12_bf should be equal")

        # Compute a loss on sidecar12_bf and compute gradients
        loss_bf = sidecar12_bf.jdata.sum()
        loss_bf.backward()

        self.assertTrue(sidecar12_bf.requires_grad, "sidecar12_bf should require gradients")
        assert sidecar1_bf.jdata.grad is not None, "sidecar1_bf should have gradients"
        assert sidecar2_bf.jdata.grad is not None, "sidecar2_bf should have gradients"
        sidecar1_bf_grad = sidecar1_bf.jdata.grad.clone().detach()
        sidecar2_bf_grad = sidecar2_bf.jdata.grad.clone().detach()

        self.assertTrue(torch.equal(sidecar1_grad, sidecar1_bf_grad), "sidecar1 gradients should be equal")
        self.assertTrue(torch.equal(sidecar2_grad, sidecar2_bf_grad), "sidecar2 gradients should be equal")

    def test_inject_in_place_superset_into_subset_backprop(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)

        # Sidecar 23 holds the union of grid2 and grid3
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.zeros)

        sidecar1.requires_grad = True
        sidecar2.requires_grad = False
        sidecar3.requires_grad = True

        # Brute force copies
        sidecar1_bf = sidecar1.clone().detach()
        sidecar2_bf = sidecar2.clone().detach()
        sidecar3_bf = sidecar3.clone().detach()
        sidecar23_bf = self.build_sidecar(self.grid_batch23, torch.zeros)

        # Run ours
        self.grid_batch2.inject_to(self.grid_batch23, sidecar2, sidecar23)  # sidecar2 -> sidecar23
        self.grid_batch1.inject_to(
            self.grid_batch23, sidecar1, sidecar23
        )  # sidecar1 -> sidecar23 (non overlapping, no effect)
        self.grid_batch23.inject_from(self.grid_batch3, sidecar3, sidecar23)  # sidecar3 -> sidecar23

        # Now inject superset sidecar23 into subset sidecar2
        self.grid_batch2.inject_from(self.grid_batch23, sidecar23, sidecar2)  # sidecar23 -> sidecar2

        loss = sidecar23.jdata.sum()
        loss.backward()
        self.assertTrue(sidecar23.requires_grad, "sidecar23 should require gradients")
        assert sidecar2.jdata.grad is None, "sidecar2 should have gradients"
        assert sidecar1.jdata.grad is not None, "sidecar1 should not have gradients"
        assert sidecar3.jdata.grad is not None, "sidecar3 should have gradients"
        sidecar1_grad = sidecar1.jdata.grad.clone().detach()
        sidecar3_grad = sidecar3.jdata.grad.clone().detach()

        sidecar1_bf.requires_grad = True
        sidecar2_bf.requires_grad = False
        sidecar3_bf.requires_grad = True

        # Run bruteforce
        self.inject_bruteforce(
            self.grid_batch2, self.grid_batch23, sidecar2_bf, sidecar23_bf
        )  # sidecar2 -> sidecar23_bf
        self.inject_bruteforce(
            self.grid_batch1, self.grid_batch23, sidecar1_bf, sidecar23_bf
        )  # sidecar1_bf -> sidecar23_bf (non overlapping, no effect)
        self.inject_bruteforce(
            self.grid_batch3, self.grid_batch23, sidecar3_bf, sidecar23_bf
        )  # sidecar3_bf -> sidecar23_bf

        # Now inject superset sidecar23_bf into subset sidecar2
        self.inject_bruteforce(
            self.grid_batch23, self.grid_batch2, sidecar23_bf, sidecar2_bf
        )  # sidecar23_bf -> sidecar2_bf

        loss = sidecar23_bf.jdata.sum()
        loss.backward()
        self.assertTrue(sidecar23_bf.requires_grad, "sidecar23_bf should require gradients")
        assert sidecar2_bf.jdata.grad is None, "sidecar2_bf should have gradients"
        assert sidecar1_bf.jdata.grad is not None, "sidecar1_bf should not have gradients"
        assert sidecar3_bf.jdata.grad is not None, "sidecar3_bf should have gradients"
        sidecar1_bf_grad = sidecar1_bf.jdata.grad.clone().detach()
        sidecar3_bf_grad = sidecar3_bf.jdata.grad.clone().detach()

        self.assertTrue(torch.equal(sidecar23.jdata, sidecar23_bf.jdata), "sidecar23 and sidecar23_bf should be equal")
        self.assertTrue(torch.equal(sidecar2.jdata, sidecar2_bf.jdata), "sidecar2 and sidecar2_bf should be equal")
        self.assertTrue(torch.equal(sidecar1.jdata, sidecar1_bf.jdata), "sidecar1 and sidecar1_bf should be equal")
        self.assertTrue(torch.equal(sidecar3.jdata, sidecar3_bf.jdata), "sidecar3 and sidecar3_bf should be equal")
        self.assertTrue(
            torch.equal(sidecar3_grad, sidecar3_bf_grad), "sidecar3 and sidecar3_bf should have equal gradients"
        )
        self.assertTrue(
            torch.equal(sidecar1_grad, sidecar1_bf_grad), "sidecar1 and sidecar1_bf should have equal gradients"
        )

    def test_inject_in_place_backprop_mix_of_requires_grad_and_not(self):
        # There are three grids, grid1, grid2, and grid3.
        # We construct two grids grid12 = union(grid1, grid2) and grid23 = union(grid2, grid3).
        # grid12 and grid23 have a common grid2 but are not strictly overlapping.

        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)
        sidecar123: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch123, torch.zeros)

        # Only sidecar 1 and sidecar3 will require gradients
        sidecar1.requires_grad = True
        sidecar3.requires_grad = True

        # Inject sidecar1 and sidecar2 into sidecar12
        # self.grid_batch12.inject_from(self.grid_batch1, sidecar1, sidecar12)  # sidecar1 -> sidecar12
        self.grid_batch1.inject_to(self.grid_batch123, sidecar1, sidecar123)  # sidecar1 -> sidecar123
        self.grid_batch123.inject_from(self.grid_batch2, sidecar2, sidecar123)  # sidecar2 -> sidecar123
        self.grid_batch123.inject_from(self.grid_batch3, sidecar3, sidecar123)  # sidecar3 -> sidecar123

        # Compute a loss on sidecar123 and compute gradients
        loss = sidecar123.jdata.sum()
        loss.backward()

        self.assertTrue(sidecar123.requires_grad, "sidecar123 should require gradients")
        assert sidecar1.jdata.grad is not None, "sidecar1 should have gradients"
        assert sidecar2.jdata.grad is None, "sidecar2 should not have gradients"
        assert sidecar3.jdata.grad is not None, "sidecar3 should have gradients"
        sidecar1_grad = sidecar1.jdata.grad.clone().detach()
        sidecar3_grad = sidecar3.jdata.grad.clone().detach()

        # Now do the same thing with bruteforce injection
        sidecar1_bf = sidecar1.clone().detach()
        sidecar2_bf = sidecar2.clone().detach()
        sidecar3_bf = sidecar3.clone().detach()
        sidecar123_bf: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch123, torch.zeros)

        sidecar1_bf.requires_grad = True
        sidecar3_bf.requires_grad = True

        # Inject sidecar1 and sidecar2 into sidecar12 using bruteforce
        sidecar123_bf = InjectionTests.inject_bruteforce(
            self.grid_batch1, self.grid_batch123, sidecar1_bf, sidecar123_bf
        )
        sidecar123_bf = InjectionTests.inject_bruteforce(
            self.grid_batch2, self.grid_batch123, sidecar2_bf, sidecar123_bf
        )
        sidecar123_bf = InjectionTests.inject_bruteforce(
            self.grid_batch3, self.grid_batch123, sidecar3_bf, sidecar123_bf
        )

        self.assertTrue(
            torch.equal(sidecar123.jdata, sidecar123_bf.jdata), "sidecar123 and sidecar123_bf should be equal"
        )

        # Compute a loss on sidecar123_bf and compute gradients
        loss_bf = sidecar123_bf.jdata.sum()
        loss_bf.backward()

        self.assertTrue(sidecar123_bf.requires_grad, "sidecar123_bf should require gradients")
        assert sidecar1_bf.jdata.grad is not None, "sidecar1_bf should have gradients"
        assert sidecar2_bf.jdata.grad is None, "sidecar2_bf should have gradients"
        assert sidecar3_bf.jdata.grad is not None, "sidecar3_bf should have gradients"
        sidecar1_bf_grad = sidecar1_bf.jdata.grad.clone().detach()
        sidecar3_bf_grad = sidecar3_bf.jdata.grad.clone().detach()

        self.assertTrue(torch.equal(sidecar1_grad, sidecar1_bf_grad), "sidecar1 gradients should be equal")
        self.assertTrue(torch.equal(sidecar3_grad, sidecar3_bf_grad), "sidecar3 gradients should be equal")

    def test_inject_in_place_backprop_dst_sidecar_leaf_fails(self):
        # There are three grids, grid1, grid2, and grid3.
        # We construct two grids grid12 = union(grid1, grid2) and grid23 = union(grid2, grid3).
        # grid12 and grid23 have a common grid2 but are not strictly overlapping.

        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)
        sidecar123: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch123, torch.zeros)

        # Only sidecar 1 and sidecar3 will require gradients
        sidecar1.requires_grad = True
        sidecar3.requires_grad = True
        sidecar123.requires_grad = True  # This is the leaf tensor, so it should not require gradients

        # This should fail because sidecar123 is a leaf tensor
        with self.assertRaises(RuntimeError):
            self.grid_batch1.inject_to(self.grid_batch123, sidecar1, sidecar123)  # sidecar1 -> sidecar123
        with self.assertRaises(RuntimeError):
            self.grid_batch123.inject_from(self.grid_batch2, sidecar2, sidecar123)  # sidecar2 -> sidecar123

    def test_inject_in_place_backprop_dst_sidecar_requires_grad(self):
        # There are three grids, grid1, grid2, and grid3.
        # We construct two grids grid12 = union(grid1, grid2) and grid23 = union(grid2, grid3).
        # grid12 and grid23 have a common grid2 but are not strictly overlapping.

        sidecar1: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch1, torch.rand)
        sidecar3: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch3, torch.rand)
        sidecar123_base: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch123, torch.rand)

        # Sidecar 1, 3, and 123 will require gradients
        sidecar1.requires_grad = True
        sidecar3.requires_grad = True
        sidecar123_base.requires_grad = True

        sidecar123 = sidecar123_base * 10.0

        # Inject sidecar1 and sidecar2 into sidecar12
        self.grid_batch1.inject_to(self.grid_batch123, sidecar1, sidecar123)  # sidecar1 -> sidecar123
        self.grid_batch123.inject_from(self.grid_batch3, sidecar3, sidecar123)  # sidecar3 -> sidecar123

        # Compute a loss on sidecar123 and compute gradients
        loss = sidecar123.jdata.sum()
        loss.backward()

        self.assertTrue(sidecar123.requires_grad, "sidecar123 should require gradients")
        assert sidecar1.jdata.grad is not None, "sidecar1 should have gradients"
        assert sidecar3.jdata.grad is not None, "sidecar3 should have gradients"
        assert sidecar123.jdata.grad is None, "sidecar123 should not have gradients"
        assert sidecar123_base.jdata.grad is not None, "sidecar123_base should have gradients"
        sidecar1_grad = sidecar1.jdata.grad.clone().detach()
        sidecar3_grad = sidecar3.jdata.grad.clone().detach()
        sidecar123_base_grad = sidecar123_base.jdata.grad.clone().detach()

        # Now do the same thing with bruteforce injection
        sidecar1_bf = sidecar1.clone().detach()
        sidecar3_bf = sidecar3.clone().detach()
        sidecar123_base_bf = sidecar123_base.clone().detach()

        # Sidecar 1, 3, and 123 will require gradients
        sidecar1_bf.requires_grad = True
        sidecar3_bf.requires_grad = True
        sidecar123_base_bf.requires_grad = True

        sidecar123_bf = sidecar123_base_bf * 10.0

        # Inject sidecar1 and sidecar2 into sidecar12 using bruteforce
        sidecar123_bf = InjectionTests.inject_bruteforce(
            self.grid_batch1, self.grid_batch123, sidecar1_bf, sidecar123_bf
        )
        sidecar123_bf = InjectionTests.inject_bruteforce(
            self.grid_batch3, self.grid_batch123, sidecar3_bf, sidecar123_bf
        )

        self.assertTrue(
            torch.equal(sidecar123.jdata, sidecar123_bf.jdata), "sidecar123 and sidecar123_bf should be equal"
        )

        # Compute a loss on sidecar123_bf and compute gradients
        loss_bf = sidecar123_bf.jdata.sum()
        loss_bf.backward()

        self.assertTrue(sidecar123_bf.requires_grad, "sidecar123_bf should require gradients")
        assert sidecar1_bf.jdata.grad is not None, "sidecar1_bf should have gradients"
        assert sidecar3_bf.jdata.grad is not None, "sidecar3_bf should have gradients"
        assert sidecar123_bf.jdata.grad is None, "sidecar123_bf should not have gradients"
        assert sidecar123_base_bf.jdata.grad is not None, "sidecar123_base_bf should have gradients"
        sidecar1_bf_grad = sidecar1_bf.jdata.grad.clone().detach()
        sidecar3_bf_grad = sidecar3_bf.jdata.grad.clone().detach()
        sidecar123_base_bf_grad = sidecar123_base_bf.jdata.grad.clone().detach()

        self.assertTrue(torch.equal(sidecar1_grad, sidecar1_bf_grad), "sidecar1 gradients should be equal")
        self.assertTrue(torch.equal(sidecar3_grad, sidecar3_bf_grad), "sidecar3 gradients should be equal")
        self.assertTrue(
            torch.equal(sidecar123_base_grad, sidecar123_base_bf_grad), "sidecar123_base gradients should be equal"
        )


if __name__ == "__main__":
    unittest.main()
