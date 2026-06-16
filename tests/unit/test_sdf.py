# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import torch

import fvdb


def _dense_cube_grid(vx: float, half: int, device: torch.device) -> "fvdb.Grid":
    """A dense (2*half+1)^3 cube grid built by placing one point at each voxel centre."""
    rng = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
    ijk = torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3)
    pts = ijk * vx
    return fvdb.Grid.from_points(pts, voxel_size=vx)


class ReinitializeSdfTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("reinitialize_sdf requires a CUDA device")
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.vx = 0.05
        self.R = 0.3
        self.band = 3
        self.bw = self.band * self.vx
        self.grid = _dense_cube_grid(self.vx, half=12, device=self.device)
        centers = self.grid.ijk.float() * self.vx  # voxel centres (origin 0)
        self.analytic = centers.norm(dim=1) - self.R  # exact sphere SDF at each voxel centre

    def _band_mask(self, width_voxels: float) -> torch.Tensor:
        return self.analytic.abs() < width_voxels * self.vx

    # ------------------------------------------------------------------ reinit
    def test_reinitialize_preserves_good_sdf(self):
        """Redistancing an already-correct SDF (|grad phi| = 1) should preserve it in the band."""
        field = self.analytic.clamp(-self.bw, self.bw)
        phi = self.grid.reinitialize_sdf(field, band=self.band, smooth=0, order=3)
        self.assertEqual(phi.shape[0], self.grid.num_voxels)
        m = self._band_mask(self.band - 1)
        err = (phi[m] - self.analytic[m]).abs()
        self.assertLess(err.mean().item(), 0.25 * self.vx)
        self.assertLess(err.max().item(), 1.0 * self.vx)

    def test_reinitialize_recovers_from_step(self):
        """Redistancing a crude +/-band sign step should recover the sphere SDF near the surface."""
        field = torch.where(
            self.analytic < 0,
            torch.full_like(self.analytic, -self.bw),
            torch.full_like(self.analytic, self.bw),
        )
        phi = self.grid.reinitialize_sdf(field, band=self.band, smooth=0, order=3)
        m = self._band_mask(self.band - 1)
        err = (phi[m] - self.analytic[m]).abs()
        self.assertLess(err.mean().item(), 0.6 * self.vx)

    def test_band_clamp(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        phi = self.grid.reinitialize_sdf(field, band=self.band)
        self.assertLessEqual(phi.abs().max().item(), self.bw + 1e-4)

    def test_order_and_smoothing_run(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        for order in (1, 2, 3):
            phi = self.grid.reinitialize_sdf(field, band=self.band, order=order)
            self.assertEqual(phi.shape[0], self.grid.num_voxels)
        # smoothing mode accepts the SmoothingMode enum or its integer value
        for mode in (fvdb.SmoothingMode.MEAN_CURVATURE, fvdb.SmoothingMode.TAUBIN, 1):
            phi = self.grid.reinitialize_sdf(field, band=self.band, smooth=4, smoothing=mode)
            self.assertEqual(phi.shape[0], self.grid.num_voxels)

    def test_float64(self):
        field = self.analytic.clamp(-self.bw, self.bw).double()
        phi = self.grid.reinitialize_sdf(field, band=self.band, order=3)
        self.assertEqual(phi.dtype, torch.float64)
        m = self._band_mask(self.band - 1)
        err = (phi[m] - self.analytic[m].double()).abs()
        self.assertLess(err.mean().item(), 0.25 * self.vx)

    # ------------------------------------------------------------------ retopo
    def test_retopologize_prune(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        pruned, phi = self.grid.retopologize_sdf(field, band=self.band, prune=True)
        self.assertEqual(phi.shape[0], pruned.num_voxels)
        self.assertLessEqual(pruned.num_voxels, self.grid.num_voxels)
        self.assertLess(phi.abs().max().item(), self.bw)  # strictly inside the band
        v, f, n = pruned.marching_cubes(phi, level=0.0)
        self.assertGreater(v.shape[0], 0)

    def test_prune_ordering_guard(self):
        """The rmask-based prune must align with the pruned grid's canonical voxel order."""
        field = self.analytic.clamp(-self.bw, self.bw)
        phi_full = self.grid.reinitialize_sdf(field, band=self.band)
        mask = phi_full.abs() < self.bw * 0.999
        pruned, phi = self.grid.retopologize_sdf(field, band=self.band, prune=True)
        self.assertTrue(torch.equal(self.grid.ijk[mask], pruned.ijk))
        self.assertTrue(torch.allclose(phi, phi_full[mask]))

    def test_no_prune_no_pad_returns_same_grid(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        grid_out, phi = self.grid.retopologize_sdf(field, band=self.band, pad=False, prune=False)
        self.assertEqual(grid_out.num_voxels, self.grid.num_voxels)
        self.assertEqual(phi.shape[0], self.grid.num_voxels)

    def test_pad_widens_thin_band(self):
        """A filled ball with only a ~1-voxel exterior shell should gain a full band with pad=True."""
        rng = torch.arange(-12, 13, device=self.device, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
        ijk = torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3)
        r = (ijk * self.vx).norm(dim=1)
        # solid interior + ~1 exterior layer (filled interior, thin exterior band)
        pts = (ijk * self.vx)[r < self.R + 0.5 * self.vx]
        g = fvdb.Grid.from_points(pts, voxel_size=self.vx)
        analytic = (g.ijk.float() * self.vx).norm(dim=1) - self.R
        field = analytic.clamp(-self.bw, self.bw)

        g0, phi0 = g.retopologize_sdf(field, band=self.band, pad=False, prune=True)
        g1, phi1 = g.retopologize_sdf(field, band=self.band, pad=True, prune=True)

        # padding produces a genuine multi-voxel exterior band; without it the band is truncated
        self.assertGreater((phi1 > 0.5 * self.vx).sum().item(), (phi0 > 0.5 * self.vx).sum().item())
        self.assertGreater(g1.num_voxels, g0.num_voxels)
        self.assertGreater(phi1.max().item(), 2.0 * self.vx)  # ~full band*vx exterior reach
        self.assertLess(phi0.max().item(), 1.5 * self.vx)  # truncated to the input topology
        v, f, n = g1.marching_cubes(phi1, level=0.0)
        self.assertGreater(v.shape[0], 0)

    # ------------------------------------------------------------------ batch
    def test_batch_matches_single(self):
        vx = self.vx
        gb = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(vx, 12), self._cube_points(vx, 10)]),
            voxel_sizes=vx,
        )
        analytic = (gb.ijk.jdata.float() * vx).norm(dim=1) - self.R
        field = gb.jagged_like(analytic.clamp(-self.bw, self.bw))
        phi = gb.reinitialize_sdf(field, band=self.band, order=3)
        self.assertEqual(phi.jdata.shape[0], gb.total_voxels)
        for i in range(gb.grid_count):
            single = fvdb.Grid.from_points(self._cube_points(vx, [12, 10][i]), voxel_size=vx)
            a = ((single.ijk.float() * vx).norm(dim=1) - self.R).clamp(-self.bw, self.bw)
            phi_single = single.reinitialize_sdf(a, band=self.band, order=3)
            self.assertTrue(torch.allclose(phi[i].jdata, phi_single, atol=1e-5))

    def _cube_points(self, vx: float, half: int) -> torch.Tensor:
        rng = torch.arange(-half, half + 1, device=self.device, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
        return torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3) * vx


if __name__ == "__main__":
    unittest.main()
