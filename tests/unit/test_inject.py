# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import unittest
from typing import Callable

import numpy as np
import torch
from parameterized import parameterized, parameterized_class

import fvdb


@parameterized_class(
    ("device", "element_shape"),
    list(itertools.product(["cuda:0", "cpu"], [(), (3,), (3, 2), (3, 2, 1), (512,), (512, 3)])),
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

    @staticmethod
    def filter_every_n(
        gridbatch: fvdb.GridBatch, sidecar: fvdb.JaggedTensor, step: int
    ) -> tuple[fvdb.GridBatch, fvdb.JaggedTensor, fvdb.JaggedTensor]:
        mask = sidecar.jagged_like(torch.zeros(sidecar.jdata.shape[0], dtype=torch.bool))
        mask.jdata[::step] = True
        gridbatch_pruned = gridbatch.pruned_grid(mask)
        sidecar_pruned = gridbatch_pruned.jagged_like(sidecar.jdata[::step])
        return gridbatch_pruned, sidecar_pruned, mask

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

    def test_inject_in_place_non_contiguous_subset_into_contiguous_superset(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)

        gridbatch2_pruned, sidecar2_pruned, mask = self.filter_every_n(self.grid_batch2, sidecar2, 2)

        # Sanity checks to ensure the masks and pruned sidecars are correct
        self.assertTrue(
            torch.equal(sidecar2.jdata[mask.jdata], sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar2[mask].jdata, sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )

        # Check that the pruned sidecar is not contiguous
        self.assertFalse(sidecar2_pruned.jdata.is_contiguous(), "sidecar2_pruned should not be contiguous")

        # Inject the pruned sidecar into the superset gridbatch
        sidecar23_non_contig: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.zeros)
        gridbatch2_pruned.inject_to(
            self.grid_batch23, sidecar2_pruned, sidecar23_non_contig
        )  # sidecar2_pruned -> sidecar23_non_contig

        # Make a contiguous version of the pruned sidecar and inject it into the superset gridbatch
        sidecar23_contig: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.zeros)
        sidecar2_pruned_contig = sidecar2_pruned.jagged_like(sidecar2_pruned.jdata.contiguous())

        # Sanity check that the pruned sidecar is contiguous
        self.assertTrue(sidecar2_pruned_contig.jdata.is_contiguous(), "sidecar2_pruned_contig should be contiguous")
        self.assertTrue(
            torch.equal(sidecar2_pruned.jdata, sidecar2_pruned_contig.jdata),
            "sidecar2_pruned should equal sidecar2_pruned_contig",
        )

        # Inject the contiguous pruned sidecar into the superset gridbatch
        gridbatch2_pruned.inject_to(
            self.grid_batch23, sidecar2_pruned_contig, sidecar23_contig
        )  # sidecar2_pruned -> sidecar23_contig

        # Check that the non-contiguous and contiguous sidecars are equal
        self.assertTrue(
            torch.equal(sidecar23_non_contig.jdata, sidecar23_contig.jdata),
            "sidecar23_non_contig and sidecar23_contig should be equal",
        )

    def test_inject_in_place_non_contiguous_subset_into_non_contiguous_superset(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.rand)

        gridbatch2_pruned, sidecar2_pruned, mask2 = self.filter_every_n(self.grid_batch2, sidecar2, 3)
        gridbatch23_pruned, sidecar23_pruned, mask23 = self.filter_every_n(self.grid_batch23, sidecar23, 2)

        # Make a contiguous copy of the pruned sidecar and ensure it is contiguous
        sidecar2_pruned_contig = sidecar2_pruned.jagged_like(sidecar2_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar2_pruned_contig.jdata.is_contiguous(), "sidecar2_pruned_contig should be contiguous")
        sidecar23_pruned_contig = sidecar23_pruned.jagged_like(sidecar23_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar23_pruned_contig.jdata.is_contiguous(), "sidecar23_pruned_contig should be contiguous")

        # Sanity checks to ensure the masks and pruned sidecars are correct
        self.assertTrue(
            torch.equal(sidecar23.jdata[mask23.jdata], sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar23[mask23].jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar2.jdata[mask2.jdata], sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar2[mask2].jdata, sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )
        self.assertTrue(torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata))

        # Check that inputs are not contiguous
        self.assertFalse(sidecar23_pruned.jdata.is_contiguous(), "sidecar23_pruned should not be contiguous")
        self.assertFalse(sidecar2_pruned.jdata.is_contiguous(), "sidecar2_pruned should not be contiguous")

        # Inject the pruned sidecar into the pruned superset gridbatch (both are non contiguous)
        gridbatch23_pruned.inject_from(
            gridbatch2_pruned, sidecar2_pruned, sidecar23_pruned
        )  # sidecar2_pruned -> sidecar23_pruned

        # Assert the non contiguous target sidecar is now not equal to the contiguous one
        # after injection
        self.assertFalse(
            torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata),
            "sidecar23_pruned should equal sidecar23_pruned_contig",
        )

        # Inject the contiguous pruned sidecar into the contiguous pruned superset gridbatch
        gridbatch23_pruned.inject_from(
            gridbatch2_pruned, sidecar2_pruned_contig, sidecar23_pruned_contig
        )  # sidecar2 -> sidecar23_pruned_contig

        self.assertTrue(
            torch.equal(sidecar23_pruned_contig.jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned_contig and sidecar23_contig should be equal",
        )
        self.assertTrue(
            torch.equal(sidecar2_pruned_contig.jdata, sidecar2_pruned.jdata),
            "sidecar2_pruned_contig and sidecar2_pruned should be equal",
        )

    def test_inject_in_place_non_contiguous_subset_into_non_contiguous_superset_bakcprop(self):
        # There are three grids, grid1, grid2, and grid3 with random values

        # Create a source and target grid sidecars
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.rand)

        # We'll record gradients on the source grid
        sidecar2.requires_grad = True

        # Prune the source and target grids so they are not contiguous
        gridbatch2_pruned, sidecar2_pruned, mask2 = self.filter_every_n(self.grid_batch2, sidecar2, 3)
        gridbatch23_pruned, sidecar23_pruned, mask23 = self.filter_every_n(self.grid_batch23, sidecar23, 2)

        # Make a contiguous copy of the source and target grids that we'll use later
        sidecar2_pruned_contig = sidecar2_pruned.jagged_like(sidecar2_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar2_pruned_contig.jdata.is_contiguous(), "sidecar2_pruned_contig should be contiguous")
        sidecar23_pruned_contig = sidecar23_pruned.jagged_like(sidecar23_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar23_pruned_contig.jdata.is_contiguous(), "sidecar23_pruned_contig should be contiguous")

        # Sanity checks to ensure the masks and pruned sidecars are correct
        self.assertTrue(
            torch.equal(sidecar23.jdata[mask23.jdata], sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar23[mask23].jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar2.jdata[mask2.jdata], sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar2[mask2].jdata, sidecar2_pruned.jdata),
            "sidecar2_pruned should equal sidecar2 with mask applied",
        )
        self.assertTrue(torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata))

        # Check that non-contiguous inputs are indeed not contiguous
        self.assertFalse(sidecar23_pruned.jdata.is_contiguous(), "sidecar23_pruned should not be contiguous")
        self.assertFalse(sidecar2_pruned.jdata.is_contiguous(), "sidecar2_pruned should not be contiguous")

        # Inject the non contiguous pruned source into the non contiguous destination
        gridbatch23_pruned.inject_from(gridbatch2_pruned, sidecar2_pruned, sidecar23_pruned)

        # Assert the non contiguous target sidecar is now not equal to the contiguous one
        # after injection
        self.assertFalse(
            torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata),
            "sidecar23_pruned should equal sidecar23_pruned_contig",
        )

        # Compute a loss on the pruned sidecar and backpropagate
        loss = sidecar23_pruned.jdata.sum()
        loss.backward()

        # Check that we have gradients on the source sidecar
        assert sidecar2.jdata.grad is not None, "sidecar2 should have a gradient"
        sidecar2_grad = sidecar2.jdata.grad.clone().detach()

        # Make a detached and contiguous copy of the source sidecar and record gradients on it
        sidecar2_copy = sidecar2.clone().detach()
        sidecar2_copy.requires_grad = True

        # Prune the source sidecar again and make a contiguous copy
        gridbatch2_pruned, sidecar2_pruned, mask2 = self.filter_every_n(self.grid_batch2, sidecar2_copy, 3)
        sidecar_2_pruned_contig = sidecar2_pruned.jagged_like(sidecar2_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar_2_pruned_contig.jdata.is_contiguous(), "sidecar2_pruned_contig should be contiguous")
        self.assertTrue(sidecar23_pruned_contig.jdata.is_contiguous(), "sidecar23_pruned_contig should be contiguous")

        # Inject the contiguous pruned sidecar into the contiguous pruned superset gridbatch
        gridbatch23_pruned.inject_from(
            gridbatch2_pruned, sidecar_2_pruned_contig, sidecar23_pruned_contig
        )  # sidecar2_pruned -> sidecar23_pruned_contig

        # Compute a loss on the contiguous pruned sidecar and backpropagate
        loss = sidecar23_pruned_contig.jdata.sum()
        loss.backward()

        # Check that we have gradients on the source sidecar copy
        assert sidecar2_copy.jdata.grad is not None, "sidecar2 should have a gradient"
        sidecar2_copy_grad = sidecar2_copy.jdata.grad.clone().detach()

        # Check that the outputs and gradients match
        self.assertTrue(
            torch.equal(sidecar23_pruned_contig.jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned_contig and sidecar23_contig should be equal",
        )
        self.assertTrue(
            torch.equal(sidecar2_pruned_contig.jdata, sidecar2_pruned.jdata),
            "sidecar2_pruned_contig and sidecar2_pruned should be equal",
        )
        self.assertTrue(
            torch.equal(sidecar2_copy_grad, sidecar2_grad),
            "sidecar2_copy_grad and sidecar2_grad should be equal",
        )

    def test_inject_in_place_contiguous_subset_into_non_contiguous_superset(self):
        # There are three grids, grid1, grid2, and grid3 with random values
        sidecar2: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch2, torch.rand)
        sidecar23: fvdb.JaggedTensor = self.build_sidecar(self.grid_batch23, torch.rand)

        # Prune the target grid sidecar so it's non contiguous
        gridbatch23_pruned, sidecar23_pruned, mask = self.filter_every_n(self.grid_batch23, sidecar23, 2)

        # Make a contiguouos copy of the pruned sidecar and ensure it is contiguous
        sidecar23_pruned_contig = sidecar23_pruned.jagged_like(sidecar23_pruned.jdata.contiguous().clone())
        self.assertTrue(sidecar23_pruned_contig.jdata.is_contiguous(), "sidecar23_pruned_contig should be contiguous")

        # Sanity checks on the masks and pruned sidecars
        self.assertTrue(
            torch.equal(sidecar23.jdata[mask.jdata], sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(
            torch.equal(sidecar23[mask].jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned should equal sidecar23 with mask applied",
        )
        self.assertTrue(torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata))
        self.assertFalse(sidecar23_pruned.jdata.is_contiguous(), "sidecar23_pruned should not be contiguous")

        # Inject from the contiguous source into the non contiguous target
        gridbatch23_pruned.inject_from(self.grid_batch2, sidecar2, sidecar23_pruned)  # sidecar2 -> sidecar23_pruned

        # Check the non-contiguous target sidecar is not equal to the contiguous one after injection
        self.assertFalse(
            torch.equal(sidecar23_pruned.jdata, sidecar23_pruned_contig.jdata),
            "sidecar23_pruned should equal sidecar23_pruned_contig",
        )

        # Inject the contiguous source into the contiguous target
        gridbatch23_pruned.inject_from(
            self.grid_batch2, sidecar2, sidecar23_pruned_contig
        )  # sidecar2 -> sidecar23_pruned_contig

        # Check that the non-contiguous and contiguous sidecars are equal
        self.assertTrue(
            torch.equal(sidecar23_pruned_contig.jdata, sidecar23_pruned.jdata),
            "sidecar23_pruned_contig and sidecar23_pruned should be equal",
        )

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

    @parameterized.expand([(torch.float32,), (torch.float16,), (torch.float64,)])
    def test_inject_out_of_place(self, dtype: torch.dtype):
        grid1 = fvdb.GridBatch(self.device)
        grid2 = fvdb.GridBatch(self.device)

        random_points_b1 = torch.randn(100, 3, device=self.device, dtype=dtype)
        random_points_b2 = torch.randn(100, 3, device=self.device, dtype=dtype)

        grid1.set_from_points(
            fvdb.JaggedTensor([random_points_b1[:70], random_points_b2[:70]]), voxel_sizes=0.01, origins=[0, 0, 0]
        )
        grid2.set_from_points(
            fvdb.JaggedTensor([random_points_b1[30:], random_points_b2[30:]]), voxel_sizes=0.01, origins=[0, 0, 0]
        )

        random_features_b1 = torch.randn(grid1[0].total_voxels, 32, device=self.device, dtype=dtype)
        random_features_b2 = torch.randn(grid1[1].total_voxels, 32, device=self.device, dtype=dtype)

        ret = grid1.inject_to(grid2, fvdb.JaggedTensor([random_features_b1, random_features_b2]))
        # ret = grid2.inject_from(grid1, fvdb.JaggedTensor([random_features_b1, random_features_b2]))

        # Perform an all pairs comparison between grid1 and grid2 points.
        # All points that match up should have the same features.

        # Test independently for both batches.
        b1_comparison = torch.all(grid1[0].ijk.jdata.unsqueeze(0) == grid2[0].ijk.jdata.unsqueeze(1), dim=-1)
        b2_comparison = torch.all(grid1[1].ijk.jdata.unsqueeze(0) == grid2[1].ijk.jdata.unsqueeze(1), dim=-1)

        toinds, frominds = torch.where(b1_comparison)
        self.assertTrue(torch.all(ret[0].jdata[toinds] == random_features_b1[frominds]))

        toinds, frominds = torch.where(b2_comparison)
        self.assertTrue(torch.all(ret[1].jdata[toinds] == random_features_b2[frominds]))

        # All the rest should be zero.
        self.assertTrue(torch.all(ret[0].jdata[~torch.any(b1_comparison, dim=1)] == 0.0))
        self.assertTrue(torch.all(ret[1].jdata[~torch.any(b2_comparison, dim=1)] == 0.0))

        # Test the gradients
        grid1 = grid1[0]
        grid2 = grid2[0]

        random_features = torch.randn(grid1.total_voxels, 32, device=self.device, dtype=dtype, requires_grad=True)

        out = grid2.inject_from(grid1, random_features).jdata.sum()
        out.backward()

        one_indices = torch.where(torch.all(random_features.grad == 1.0, dim=1))[0]

        toinds, frominds = torch.where(b1_comparison)
        self.assertTrue(torch.all(one_indices == frominds))


if __name__ == "__main__":
    unittest.main()
