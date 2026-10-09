# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
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


_FACE_OFFSETS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def reference_redistance(grid: "fvdb.Grid", field: torch.Tensor, band: int, iters: int, order: int) -> torch.Tensor:
    """Float64 torch transcription of the CUDA redistance, the generator for the pinned values below.

    Mirrors ReinitializeSdf.cu step for step: inactive faces read as +/-band*vx with the sign of the
    voxel doing the reading; interface cells (a strict sign change across an active face in the input)
    relax to the Russo-Smereka distance D with denominator max(central gradient norm, largest
    one-sided slope, eps), shared with the steepest crossing neighbour; all other cells take the Godunov update with the frozen Peng sign; RK1,
    SSP Heun, and Shu-Osher RK3 with dt = 0.4 vx and a band clamp after every stage."""
    vx = float(grid.voxel_size[0])
    band_width = band * vx
    dt = 0.4 * vx
    ijk = grid.ijk
    face_index = torch.stack(
        [grid.ijk_to_index(ijk + torch.tensor(o, device=ijk.device, dtype=ijk.dtype)) for o in _FACE_OFFSETS], dim=1
    )
    active = face_index >= 0

    def faces(phi: torch.Tensor, sign_source: torch.Tensor) -> torch.Tensor:
        # Build the inactive value in phi's dtype; torch.where on two Python floats would yield float32.
        inactive = torch.where(sign_source < 0, -1.0, 1.0).to(phi.dtype) * band_width
        out = inactive[:, None].expand(-1, 6).clone()
        out[active] = phi[face_index[active]]
        return out

    phi0 = field.reshape(-1).double()
    phi0_faces = faces(phi0, phi0)
    center = phi0[:, None]
    interface = (phi0_faces * center < 0).any(dim=1)

    forward = torch.where(active[:, 1::2], (phi0_faces[:, 1::2] - center).abs(), torch.zeros_like(center))
    backward = torch.where(active[:, 0::2], (center - phi0_faces[:, 0::2]).abs(), torch.zeros_like(center))
    both = active[:, 1::2] & active[:, 0::2]
    axis_gradient = torch.where(
        both, (phi0_faces[:, 1::2] - phi0_faces[:, 0::2]).abs() / 2, torch.maximum(forward, backward)
    )
    denominator = torch.maximum(torch.maximum(forward, backward).max(dim=1).values, axis_gradient.norm(dim=1))
    denominator = denominator.clamp(min=1e-6 * vx)
    # Each interface cell shares the larger denominator with its steepest crossing neighbour so both
    # ends of a crossing edge scale alike and the interpolated crossing stays put.
    crossing = phi0_faces * center < 0
    edge_slope = torch.where(crossing, (phi0_faces - center).abs(), torch.full_like(phi0_faces, -1.0))
    steepest = face_index.gather(1, edge_slope.argmax(dim=1, keepdim=True)).squeeze(1).clamp(min=0)
    denominator = torch.where(interface, torch.maximum(denominator, denominator[steepest]), denominator)
    distance = vx * phi0 / denominator

    central = (phi0_faces[:, 1::2] - phi0_faces[:, 0::2]) / (2 * vx)
    frozen_sign = phi0 / torch.sqrt(phi0 * phi0 + central.pow(2).sum(dim=1) * vx * vx + 1e-10 * vx * vx)

    def rhs(phi: torch.Tensor) -> torch.Tensor:
        f = faces(phi, frozen_sign)
        back = (phi[:, None] - f[:, 0::2]) / vx
        fwd = (f[:, 1::2] - phi[:, None]) / vx
        positive = torch.maximum(back.clamp(min=0) ** 2, fwd.clamp(max=0) ** 2)
        negative = torch.maximum(back.clamp(max=0) ** 2, fwd.clamp(min=0) ** 2)
        gradient = torch.where(frozen_sign[:, None] > 0, positive, negative).sum(dim=1).sqrt()
        return torch.where(interface, (distance - phi) / vx, frozen_sign * (1 - gradient))

    def clamp(phi: torch.Tensor) -> torch.Tensor:
        return phi.clamp(-band_width, band_width)

    phi = phi0.clone()
    for _ in range(iters):
        stage1 = clamp(phi + dt * rhs(phi))
        if order == 1:
            phi = stage1
        elif order == 2:
            phi = clamp(0.5 * phi + 0.5 * stage1 + 0.5 * dt * rhs(stage1))
        else:
            stage2 = clamp(0.75 * phi + 0.25 * stage1 + 0.25 * dt * rhs(stage1))
            phi = clamp(phi / 3 + 2 * stage2 / 3 + 2 * dt * rhs(stage2) / 3)
    return phi.to(field.dtype)


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

    def test_subcell_update_pins_interface_cross(self):
        """Subcell update on a 7-voxel cross whose centre (+0.1) has five negative neighbours.

        The centre is an interface cell and relaxes toward its initial distance D = 0.1 / 3.1; the
        +x arm (+3.0) is not, and takes a Godunov step. Values are checked against the float64
        reference transcription in this file, in both polarities."""
        ijk = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            device=self.device,
            dtype=torch.int32,
        )
        g = fvdb.Grid.from_ijk(ijk, voxel_size=1.0, origin=0.0)
        center = (g.ijk == 0).all(dim=1)
        field = torch.full((g.num_voxels,), -3.0, device=self.device, dtype=torch.float64)
        field[center] = 0.1
        field[(g.ijk == ijk[1]).all(dim=1)] = 3.0
        for polarity in (1.0, -1.0):
            with self.subTest(polarity=polarity):
                phi = g.reinitialize_sdf(polarity * field, band=3, order=3, redistance_iters=1)
                expected = reference_redistance(g, polarity * field, band=3, iters=1, order=3)
                self.assertLess((phi - expected).abs().max().item(), 1e-9)
                # the centre moves toward D = 0.1 / 3.1 and keeps its sign
                self.assertLess(abs(phi[center].item()), 0.1)
                self.assertGreater(polarity * phi[center].item(), 0.1 / 3.1)

    def test_godunov_inactive_faces_continue_frozen_sign(self):
        """A non-interface cell's Godunov update must read inactive faces with its frozen sign.

        Two active voxels on a line, A = -0.5 and B = -1.5, everything else inactive. Neither is an
        interface cell, so both take the Godunov update, and every face but the one between them is
        inactive. Read with the frozen sign those faces are deep interior (-band) and downwind, so A
        relaxes toward -band. Read as +band (the PR #762 phantom-boundary bug) they would be upwind
        with a slope of 3.5 and A would rise toward a surface that does not exist. Values are checked
        against the float64 reference transcription in this file; the mirrored field checks the
        positive branch of the same rule."""
        ijk = torch.tensor([[0, 0, 0], [-1, 0, 0]], device=self.device, dtype=torch.int32)
        g = fvdb.Grid.from_ijk(ijk, voxel_size=1.0, origin=0.0)
        cell_a = (g.ijk == 0).all(dim=1)
        field = torch.empty(2, device=self.device, dtype=torch.float64)
        field[cell_a] = -0.5
        field[~cell_a] = -1.5
        for polarity in (1.0, -1.0):
            with self.subTest(polarity=polarity):
                for iters in (1, 5):
                    phi = g.reinitialize_sdf(polarity * field, band=3, order=1, redistance_iters=iters)
                    expected = reference_redistance(g, polarity * field, band=3, iters=iters, order=1)
                    self.assertLess((phi - expected).abs().max().item(), 1e-9, f"iters={iters}")
                # both keep their sign and move deeper, never toward a phantom surface
                self.assertTrue(((phi * polarity) < (field * polarity)).all().item())

    def test_matches_reference_transcription(self):
        """CUDA solve equals the float64 reference on the sphere for all three RK orders and inputs."""
        step = torch.where(self.analytic < 0, -self.bw, self.bw).expand_as(self.analytic).clone()
        slab_layer = (self.grid.ijk.float()).sum(dim=1)
        slab = ((slab_layer.abs() - 0.5) / math.sqrt(3.0) * self.vx).clamp(-self.bw, self.bw)
        cases = (
            ("exact sdf", self.analytic.clamp(-self.bw, self.bw)),
            ("sign step", step),
            ("oblique thin slab", slab),
        )
        for name, field in cases:
            for order in (1, 2, 3):
                with self.subTest(field=name, order=order):
                    field64 = field.double()
                    phi = self.grid.reinitialize_sdf(field64, band=self.band, order=order, redistance_iters=6)
                    expected = reference_redistance(self.grid, field64, band=self.band, iters=6, order=order)
                    self.assertLess((phi - expected).abs().max().item() / self.vx, 1e-9)

    # ------------------------------------------------------- thin features
    @staticmethod
    def _rod(width_vox: int, device: torch.device, length: int = 16, pad: int = 4):
        """Dense grid holding an infinite square rod (axis z) of `width_vox` voxels with its exact SDF.

        Voxel size 1. Returns (grid, field, is_interior)."""
        n = width_vox + 2 * pad
        ax = torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2
        half = width_vox / 2
        X, Y, _ = torch.meshgrid(ax, ax, torch.arange(length, device=device, dtype=torch.float32), indexing="ij")
        dx, dy = X.abs() - half, Y.abs() - half
        sdf = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2) + torch.maximum(dx, dy).clamp(max=0)
        grid = fvdb.Grid.from_dense_axis_aligned_bounds([n, n, length], [0, 0, 0], [n, n, length], device=device)
        ijk = grid.ijk
        field = sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]].contiguous().clamp(-3.0, 3.0)
        return grid, field, field < 0

    @staticmethod
    def _interface_cells(grid: "fvdb.Grid", field: torch.Tensor) -> torch.Tensor:
        """Voxels whose value changes sign across at least one active face neighbour."""
        ijk = grid.ijk
        neg = field < 0
        out = torch.zeros_like(neg)
        for o in ([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]):
            idx = grid.ijk_to_index(ijk + torch.tensor(o, device=ijk.device, dtype=ijk.dtype))
            act = idx >= 0
            out[act] |= neg[idx[act]] != neg[act]
        return out

    def test_thin_rod_zero_crossing_is_anchored(self):
        """Rods 1-3 voxels wide must keep their zero crossing under many redistance iterations.

        Interior voxels of a rod are equidistant from two faces, so the Godunov Hamiltonian reads
        sqrt(2) there and the pure scheme has no steady state: it eroded a 2-voxel rod to nothing in
        ~20 iterations. The subcell fix pins interface cells to their initial distance, which is exact
        here. Non-interface cells (e.g. the diagonal exterior corners) keep the usual first-order
        upwind error and are not checked."""
        for width in (1, 2, 3):
            grid, field, interior = self._rod(width, self.device)
            interface_cells = self._interface_cells(grid, field)
            for iters in (3, 12, 40):
                with self.subTest(width=width, iters=iters):
                    phi = grid.reinitialize_sdf(field, band=3, redistance_iters=iters)
                    self.assertEqual(((phi < 0) != interior).sum().item(), 0)
                    self.assertLess((phi[interface_cells] - field[interface_cells]).abs().max().item(), 0.05)
            with self.subTest(width=width, repeated_calls=3):
                phi = field
                for _ in range(3):
                    phi = grid.reinitialize_sdf(phi, band=3)
                self.assertEqual(((phi < 0) != interior).sum().item(), 0)
                self.assertLess((phi[interface_cells] - field[interface_cells]).abs().max().item(), 0.05)

    def test_oblique_plane_interface_distance(self):
        """Interface cells of an oblique plane SDF must keep their exact Euclidean distance.

        A single face difference only measures one gradient component, so a distance estimate built
        from it alone pins the (1,1,1) plane at +/-0.5 instead of +/-0.2887 and leaves the gradient
        across the interface at sqrt(3). The estimate uses the central-difference gradient norm as
        well, which is exact for a plane."""
        size = 12
        grid = fvdb.Grid.from_dense_axis_aligned_bounds(
            [size, size, size], [0, 0, 0], [size, size, size], device=self.device
        )
        centers = grid.ijk.float()
        interior = (centers > 1).all(dim=1) & (centers < size - 2).all(dim=1)
        for normal in ((1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (3.0, 1.0, 0.0)):
            unit_normal = torch.tensor(normal, device=self.device)
            unit_normal = unit_normal / unit_normal.norm()
            analytic = ((centers - (size - 1) / 2) @ unit_normal - 0.37).clamp(-3.0, 3.0)
            self.assertTrue((analytic.abs() > 1e-6).all().item(), "offset must not put a voxel centre on the plane")
            interface_cells = self._interface_cells(grid, analytic) & interior
            self.assertGreater(int(interface_cells.sum()), 0)
            for order in (1, 3):
                with self.subTest(normal=normal, order=order):
                    phi = grid.reinitialize_sdf(analytic, band=3, order=order, redistance_iters=40)
                    self.assertLess((phi[interface_cells] - analytic[interface_cells]).abs().max().item(), 1e-3)
                    self.assertEqual(((phi < 0) != (analytic < 0)).sum().item(), 0)

    @staticmethod
    def _axis_crossings(grid: "fvdb.Grid", phi: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        """Interpolated zero-crossing offsets along the +x/+y/+z edges leaving `cells`; NaN where none."""
        ijk = grid.ijk
        out = []
        for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            idx = grid.ijk_to_index(ijk + torch.tensor(o, device=ijk.device, dtype=ijk.dtype))
            neighbour = torch.where(idx >= 0, phi[idx.clamp(min=0)], phi)
            has = (idx >= 0) & ((neighbour < 0) != (phi < 0)) & cells
            out.append(torch.where(has, phi / (phi - neighbour), torch.full_like(phi, float("nan"))))
        return torch.stack(out, dim=1)

    def test_oblique_thin_slab_is_anchored(self):
        """A one-voxel slab oblique to the axes must keep its zero crossings, also under repeated calls.

        The exact SDF phi = (|i+j+k| - 0.5)/sqrt(3) has a medial layer at -0.2887 whose six
        neighbours all sit at +0.2887, so its central differences cancel. A per-cell distance
        estimate falls back to a single one-sided slope there and anchors the layer at -0.5 while
        the next layer stays at +0.2887, moving the crossing 0.13 voxels per call and compounding.
        Sharing the denominator with the steepest crossing neighbour makes the slab a fixed point.
        Interior is kept 3 voxels off the grid boundary, where inactive faces perturb the field."""
        size = 15
        grid = fvdb.Grid.from_dense_axis_aligned_bounds(
            [size, size, size], [0, 0, 0], [size, size, size], device=self.device
        )
        centers = grid.ijk.float() - (size - 1) / 2
        interior = (centers.abs() <= (size - 1) / 2 - 3).all(dim=1)
        for normal, half_width in (((1.0, 1.0, 1.0), 0.5), ((1.0, 1.0, 0.0), 0.5)):
            with self.subTest(normal=normal):
                nv = torch.tensor(normal, device=self.device)
                layer = centers @ nv  # integer layer index along the normal
                analytic = ((layer.abs() - half_width) / nv.norm()).clamp(-3.0, 3.0)
                self.assertTrue((analytic.abs() > 1e-6).all().item())
                interface_cells = self._interface_cells(grid, analytic) & interior
                before = self._axis_crossings(grid, analytic, interior)

                phi = grid.reinitialize_sdf(analytic, band=3)
                self.assertLess((phi[interface_cells] - analytic[interface_cells]).abs().max().item(), 1e-3)
                for _ in range(2):
                    phi = grid.reinitialize_sdf(phi, band=3)
                after = self._axis_crossings(grid, phi, interior)
                both = ~torch.isnan(before) & ~torch.isnan(after)
                self.assertLess((after[both] - before[both]).abs().max().item(), 5e-3)
                self.assertEqual(((phi < 0) != (analytic < 0)).sum().item(), 0)

    def test_thin_slab_is_fixed_point(self):
        """A 1- or 2-voxel slab with an exact SDF is unchanged by redistancing (1D-thin is stable)."""
        n = 12
        ext = 8
        for width in (1, 2):
            ax = torch.arange(n, device=self.device, dtype=torch.float32) - (n - 1) / 2
            X, _, _ = torch.meshgrid(
                ax,
                torch.arange(ext, device=self.device, dtype=torch.float32),
                torch.arange(ext, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            sdf = (X.abs() - width / 2).clamp(-3.0, 3.0)
            grid = fvdb.Grid.from_dense_axis_aligned_bounds([n, ext, ext], [0, 0, 0], [n, ext, ext], device=self.device)
            ijk = grid.ijk
            field = sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]].contiguous()
            phi = grid.reinitialize_sdf(field, band=3, redistance_iters=40)
            self.assertLess((phi - field).abs().max().item(), 1e-5, f"width={width}")

    def test_more_iterations_converge(self):
        """Redistancing must converge with iterations rather than drift: 20 -> 40 iterations changes
        the sphere field far less than 0 -> 20 does, and no voxel changes sign."""
        field = self.analytic.clamp(-self.bw, self.bw)
        phi20 = self.grid.reinitialize_sdf(field, band=self.band, redistance_iters=20)
        phi40 = self.grid.reinitialize_sdf(field, band=self.band, redistance_iters=40)
        self.assertEqual(((phi40 < 0) != (field < 0)).sum().item(), 0)
        self.assertEqual(((phi20 < 0) != (field < 0)).sum().item(), 0)
        step = (phi40 - phi20).abs().max().item()
        self.assertLess(step, 0.05 * self.vx)

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


if __name__ == "__main__":
    unittest.main()
