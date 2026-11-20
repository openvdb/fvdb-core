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
        # smoothing mode is selected with the SmoothingMode enum
        for mode in (fvdb.SmoothingMode.MEAN_CURVATURE, fvdb.SmoothingMode.TAUBIN):
            phi = self.grid.reinitialize_sdf(field, band=self.band, smooth=4, smoothing=mode)
            self.assertEqual(phi.shape[0], self.grid.num_voxels)

    def test_float64(self):
        field = self.analytic.clamp(-self.bw, self.bw).double()
        phi = self.grid.reinitialize_sdf(field, band=self.band, order=3)
        self.assertEqual(phi.dtype, torch.float64)
        m = self._band_mask(self.band - 1)
        err = (phi[m] - self.analytic[m].double()).abs()
        self.assertLess(err.mean().item(), 0.25 * self.vx)

    # ------------------------------------------------------------------ rebuild_narrow_band
    def test_rebuild_prune(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        pruned, phi = self.grid.rebuild_narrow_band(field, band=self.band, prune=True)
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
        pruned, phi = self.grid.rebuild_narrow_band(field, band=self.band, prune=True)
        self.assertTrue(torch.equal(self.grid.ijk[mask], pruned.ijk))
        self.assertTrue(torch.allclose(phi, phi_full[mask]))

    def test_no_prune_no_pad_returns_same_grid(self):
        field = self.analytic.clamp(-self.bw, self.bw)
        grid_out, phi = self.grid.rebuild_narrow_band(field, band=self.band, pad=False, prune=False)
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

        g0, phi0 = g.rebuild_narrow_band(field, band=self.band, pad=False, prune=True)
        g1, phi1 = g.rebuild_narrow_band(field, band=self.band, pad=True, prune=True)

        # padding produces a genuine multi-voxel exterior band; without it the band is truncated
        self.assertGreater((phi1 > 0.5 * self.vx).sum().item(), (phi0 > 0.5 * self.vx).sum().item())
        self.assertGreater(g1.num_voxels, g0.num_voxels)
        self.assertGreater(phi1.max().item(), 2.0 * self.vx)  # ~full band*vx exterior reach
        self.assertLess(phi0.max().item(), 1.5 * self.vx)  # truncated to the input topology
        v, f, n = g1.marching_cubes(phi1, level=0.0)
        self.assertGreater(v.shape[0], 0)

    # ------------------------------------------------------------------ inactive interior
    def _narrow_band_grid(self):
        """The sphere restricted to |phi| < band*vx: a narrow band whose INTERIOR is inactive."""
        keep = self.analytic.abs() < self.bw
        g = fvdb.Grid.from_ijk(self.grid.ijk[keep], voxel_size=self.vx, origin=0.0)
        analytic = (g.ijk.float() * self.vx).norm(dim=1) - self.R
        return g, analytic

    def test_reinitialize_narrow_band_with_inactive_interior(self):
        """A narrow band with an inactive interior must stay solid: the inner half of the band must
        match the analytic SDF as closely as the outer half, not grow a phantom inner surface.

        An IndexGrid has a single background slot, so inactive neighbours are resolved per read with
        the sign of the adjacent voxel (-band*vx inside, +band*vx outside). Before that fix the inner
        half-band error was ~3.6 vx and the innermost voxels flipped positive."""
        g, analytic = self._narrow_band_grid()
        field = analytic.clamp(-self.bw, self.bw)
        phi = g.reinitialize_sdf(field, band=self.band, order=3)
        err = (phi - analytic).abs()
        inner = (analytic <= -0.5 * self.bw) & (analytic > -self.bw)
        outer = (analytic >= 0.5 * self.bw) & (analytic < self.bw)
        self.assertLess(err[inner].mean().item(), 0.25 * self.vx)
        self.assertLess(err[inner].max().item(), 1.0 * self.vx)
        # inner and outer halves should be comparably accurate
        self.assertLess(err[inner].mean().item(), 3.0 * err[outer].mean().item() + 0.05 * self.vx)
        # no sign flips away from the surface
        away = analytic.abs() > 0.5 * self.vx
        self.assertEqual(((phi.sign() != analytic.sign()) & away).sum().item(), 0)
        # the deepest interior voxels still reach the band clamp
        self.assertLess(phi.min().item(), -(self.band - 1.25) * self.vx)

    def test_rk_boundary_uses_frozen_sign(self):
        """A stage sign flip must not turn an inactive face into an upwind contribution."""
        ijk = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            device=self.device,
            dtype=torch.int32,
        )
        g = fvdb.Grid.from_ijk(ijk, voxel_size=1.0, origin=0.0)
        center = (g.ijk == 0).all(dim=1)
        plus_x = (g.ijk == ijk[1]).all(dim=1)
        # All inputs are within [-B, B]. Opposing faces cancel in the central gradient,
        # giving the centre a frozen sign near +1 despite its small value of +0.1.
        field = torch.full((g.num_voxels,), -3.0, device=self.device, dtype=torch.float64)
        field[center] = 0.1
        field[plus_x] = 3.0
        for polarity in (1.0, -1.0):
            with self.subTest(polarity=polarity):
                phi = g.reinitialize_sdf(polarity * field, band=3, order=3, redistance_iters=1)
                # Evaluating RK3 with the missing faces excluded from the upwind gradient
                # gives centre stages -1.253625, -0.279841, -0.842034 for positive polarity.
                # The live-sign boundary instead flips to -B after stage 1 and ends at -1.103265.
                self.assertAlmostEqual(phi[center].item(), polarity * -0.842034, delta=1e-6)

    def test_rebuild_idempotent(self):
        """rebuild_narrow_band applied to its own (interior-pruned) output must reproduce that output."""
        field = self.analytic.clamp(-self.bw, self.bw)
        g1, phi1 = self.grid.rebuild_narrow_band(field, band=self.band)
        for pad in (False, True):
            g2, phi2 = g1.rebuild_narrow_band(phi1, band=self.band, pad=pad)
            a2 = (g2.ijk.float() * self.vx).norm(dim=1) - self.R
            err = (phi2 - a2).abs()
            self.assertLess(err.mean().item(), 0.25 * self.vx, f"pad={pad}")
            self.assertLess(phi2.min().item(), -(self.band - 1.25) * self.vx, f"pad={pad}")
            # same band on both passes (within a layer of voxels)
            self.assertLess(abs(g2.num_voxels - g1.num_voxels) / g1.num_voxels, 0.1, f"pad={pad}")

    def test_pad_seeds_interior_of_hollow_band(self):
        """Padding a narrow band with an inactive interior must seed the inward layers negative."""
        g, analytic = self._narrow_band_grid()
        field = analytic.clamp(-self.bw, self.bw)
        padded, phi = g.rebuild_narrow_band(field, band=self.band, pad=True, prune=False)
        a = (padded.ijk.float() * self.vx).norm(dim=1) - self.R
        interior_new = a < -self.bw  # voxels the padding added on the inside
        self.assertGreater(interior_new.sum().item(), 0)
        self.assertTrue((phi[interior_new] < 0).all().item())
        exterior_new = a > self.bw
        self.assertTrue((phi[exterior_new] > 0).all().item())

    def test_batch_pad_matches_single(self):
        """Batched sign-aware padding must agree with the single-grid path for each grid."""
        g, analytic = self._narrow_band_grid()
        field = analytic.clamp(-self.bw, self.bw)
        gb = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([g.ijk, g.ijk]), voxel_sizes=self.vx, origins=0.0)
        fb = gb.jagged_like(torch.cat([field, field]))
        gb2, phib = gb.rebuild_narrow_band(fb, band=self.band, pad=True)
        g2, phi = g.rebuild_narrow_band(field, band=self.band, pad=True)
        for i in range(2):
            self.assertTrue(torch.equal(gb2[i].ijk.jdata, g2.ijk))
            self.assertTrue(torch.allclose(phib[i].jdata, phi, atol=1e-5))

    def test_rebuild_accepts_column_field(self):
        """(N, 1) scalar fields (the usual TSDF layout) must work with the default pad=True, single and
        batched, and match the flat (N,) result."""
        field = self.analytic.clamp(-self.bw, self.bw)
        g_flat, phi_flat = self.grid.rebuild_narrow_band(field, band=self.band)
        g_col, phi_col = self.grid.rebuild_narrow_band(field[:, None], band=self.band)
        self.assertEqual(phi_col.dim(), 1)
        self.assertTrue(torch.equal(g_col.ijk, g_flat.ijk))
        self.assertTrue(torch.allclose(phi_col, phi_flat))

        gb = fvdb.GridBatch.from_ijk(
            fvdb.JaggedTensor([self.grid.ijk, self.grid.ijk]), voxel_sizes=self.vx, origins=0.0
        )
        fb_col = gb.jagged_like(torch.cat([field, field])[:, None])
        gb_out, phib = gb.rebuild_narrow_band(fb_col, band=self.band)
        self.assertEqual(phib.jdata.dim(), 1)
        for i in range(2):
            self.assertTrue(torch.equal(gb_out[i].ijk.jdata, g_flat.ijk))
            self.assertTrue(torch.allclose(phib[i].jdata, phi_flat, atol=1e-5))

    def test_sdf_rejects_non_scalar_shapes(self):
        """Validate scalar layout before flattening, including shapes whose numel happens to fit."""
        n = self.grid.num_voxels
        gb = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([self.grid.ijk]), voxel_sizes=self.vx, origins=0.0)
        for shape in ((n, 3), (1, n), (n, 1, 1)):
            field = torch.ones(shape, device=self.device)
            # Construct directly so malformed leading dimensions also reach the op's validation.
            batched = fvdb.JaggedTensor(field)
            for grid, values in ((self.grid, field), (gb, batched)):
                with self.subTest(shape=shape, batch=isinstance(grid, fvdb.GridBatch)):
                    with self.assertRaisesRegex(ValueError, "scalar field with shape"):
                        grid.reinitialize_sdf(values)
                    for pad in (False, True):
                        with self.assertRaisesRegex(ValueError, "scalar field with shape"):
                            grid.rebuild_narrow_band(values, pad=pad)

    def test_rebuild_rejects_wrong_voxel_count(self):
        n = self.grid.num_voxels
        gb = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([self.grid.ijk]), voxel_sizes=self.vx, origins=0.0)
        for count in (n - 1, n + 1):
            field = torch.ones(count, device=self.device)
            for grid, values in ((self.grid, field), (gb, fvdb.JaggedTensor(field))):
                with self.subTest(count=count, batch=isinstance(grid, fvdb.GridBatch)):
                    with self.assertRaisesRegex(ValueError, "one value per voxel"):
                        grid.rebuild_narrow_band(values)

    def test_sdf_rejects_nonfinite_values(self):
        """NaN must not be treated as new padding; neither solver accepts NaN or infinity."""
        gb = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([self.grid.ijk]), voxel_sizes=self.vx, origins=0.0)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            field = self.analytic.clamp(-self.bw, self.bw).clone()
            field[0] = invalid
            for grid, values in ((self.grid, field), (gb, gb.jagged_like(field))):
                with self.subTest(invalid=invalid, batch=isinstance(grid, fvdb.GridBatch)):
                    with self.assertRaisesRegex(ValueError, "finite values"):
                        grid.reinitialize_sdf(values)
                    for pad in (False, True):
                        with self.assertRaisesRegex(ValueError, "finite values"):
                            grid.rebuild_narrow_band(values, pad=pad)

    def test_single_sign_field_has_no_surface(self):
        """The surface is a sign change between ACTIVE voxels. An all-negative field (a raw occupancy
        mask) has none: reinitialize_sdf returns the constant -band*vx and rebuild_narrow_band an empty
        band. The active-region boundary is deliberately NOT a surface -- a tile lying entirely inside
        an object must come back empty. Adding one positive exterior layer restores the surface."""
        solid = fvdb.Grid.from_ijk(self.grid.ijk[self.analytic < 0], voxel_size=self.vx, origin=0.0)
        occupancy = -torch.ones(solid.num_voxels, device=self.device)
        phi = solid.reinitialize_sdf(occupancy, band=self.band)
        self.assertTrue(torch.allclose(phi, torch.full_like(phi, -self.bw)))
        empty, phi_empty = solid.rebuild_narrow_band(occupancy, band=self.band)
        self.assertEqual(empty.num_voxels, 0)
        self.assertEqual(phi_empty.shape[0], 0)

        # occupancy recipe: one exterior layer seeded positive -> the boundary becomes the surface
        shell = solid.dilated_grid(1)
        signed = shell.inject_from(solid, occupancy, default_value=1.0)
        g, sdf = shell.rebuild_narrow_band(signed, band=self.band)
        self.assertGreater(g.num_voxels, 0)
        self.assertGreater((sdf > 0).sum().item(), 0)
        self.assertGreater((sdf < 0).sum().item(), 0)
        # boundary sits ~half a voxel outside the outermost occupied voxel centre; a +/-1 step on a
        # voxelised boundary carries staircase error, so only check it lands within a voxel
        analytic = (g.ijk.float() * self.vx).norm(dim=1) - (self.R + 0.5 * self.vx)
        near = analytic.abs() < 1.5 * self.vx
        self.assertLess((sdf[near] - analytic[near]).abs().mean().item(), 1.0 * self.vx)

    def test_anisotropic_voxels_rejected(self):
        """The eikonal solve uses a single voxel size, so anisotropic grids must raise, not
        silently return distances scaled along y/z."""
        g = fvdb.Grid.from_ijk(self.grid.ijk, voxel_size=[self.vx, self.vx, 2 * self.vx], origin=0.0)
        field = self.analytic.clamp(-self.bw, self.bw)
        with self.assertRaisesRegex(ValueError, "isotropic"):
            g.reinitialize_sdf(field, band=self.band)
        # float32 round-off in an isotropic size must NOT trip the check
        g_iso = fvdb.Grid.from_ijk(self.grid.ijk, voxel_size=torch.tensor([0.1, 0.1, 0.1]), origin=0.0)
        phi = g_iso.reinitialize_sdf(field, band=self.band)
        self.assertEqual(phi.shape[0], g_iso.num_voxels)

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

    def test_sliced_batch_matches_standalone(self):
        """reinitialize_sdf on a sliced GridBatch view must read the view's grids, not the
        physical grids of the underlying batch (regression test for a physical-vs-logical
        grid indexing bug)."""
        vx = self.vx
        halves = [12, 10, 8]
        gb = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(vx, h) for h in halves]),
            voxel_sizes=vx,
        )
        view = gb[1:3]
        analytic = (view.ijk.jdata.float() * vx).norm(dim=1) - self.R
        field = view.jagged_like(analytic.clamp(-self.bw, self.bw))
        phi_view = view.reinitialize_sdf(field, band=self.band, order=3)

        standalone = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(vx, h) for h in halves[1:]]),
            voxel_sizes=vx,
        )
        analytic_sa = (standalone.ijk.jdata.float() * vx).norm(dim=1) - self.R
        field_sa = standalone.jagged_like(analytic_sa.clamp(-self.bw, self.bw))
        phi_sa = standalone.reinitialize_sdf(field_sa, band=self.band, order=3)

        self.assertTrue(torch.equal(view.ijk.jdata, standalone.ijk.jdata))
        self.assertTrue(torch.allclose(phi_view.jdata, phi_sa.jdata, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
