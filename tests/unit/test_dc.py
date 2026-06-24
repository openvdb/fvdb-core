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
    return fvdb.Grid.from_points(ijk * vx, voxel_size=vx)


def _edge_stats(faces: torch.Tensor, num_verts: int):
    """(num_edges, boundary_edges, nonmanifold_edges) from a triangle list, on-device."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    key = torch.minimum(e[:, 0], e[:, 1]) * num_verts + torch.maximum(e[:, 0], e[:, 1])
    counts = key.unique(return_counts=True)[1]
    return int(counts.shape[0]), int((counts == 1).sum()), int((counts > 2).sum())


class DualContourTests(unittest.TestCase):
    def setUp(self):
        """Build the shared fixture: a dense cube grid sampling an analytic sphere SDF, then a clean
        narrow-band SDF on a pruned band grid (``mesh_grid`` / ``phi``) -- the kind of input dual
        contouring is meant to consume."""
        if not torch.cuda.is_available():
            self.skipTest("dual_contour requires a CUDA device")
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.vx = 0.05
        self.R = 0.3
        self.band = 3
        self.bw = self.band * self.vx
        self.grid = _dense_cube_grid(self.vx, half=12, device=self.device)
        self.analytic = (self.grid.ijk.float() * self.vx).norm(dim=1) - self.R
        # a clean narrow-band SDF on a pruned band grid (what DC is meant to consume)
        self.mesh_grid, self.phi = self.grid.retopologize_sdf(self.analytic.clamp(-self.bw, self.bw), band=self.band)

    def test_dc_composes_with_reinitialize_and_retopologize(self):
        """Dual contouring composes with the SDF ops that produce its input: meshing the output of
        ``reinitialize_sdf`` (topology preserved) and of ``retopologize_sdf(pad=True)`` (re-banded /
        pruned grid) both produce non-empty meshes."""
        # reinitialize_sdf keeps topology; the dense cube has a thick band -> DC meshes it
        phi_re = self.grid.reinitialize_sdf(self.analytic.clamp(-self.bw, self.bw), band=self.band)
        v, f, _ = self.grid.dual_contour(phi_re, iso=0.0)
        self.assertGreater(v.shape[0], 0)
        self.assertGreater(f.shape[0], 0)
        # retopologize with padding then DC
        g2, phi2 = self.grid.retopologize_sdf(self.analytic.clamp(-self.bw, self.bw), band=self.band, pad=True)
        v2, f2, _ = g2.dual_contour(phi2, iso=0.0)
        self.assertGreater(v2.shape[0], 0)
        self.assertGreater(f2.shape[0], 0)

    def test_dc_extent_matches_marching_cubes(self):
        """Dual contouring traces the same isosurface as marching cubes: the two meshes' bounding
        boxes agree to within a voxel or two. A cross-implementation sanity check -- the precise
        surface location is pinned analytically by the planar and sphere-radius tests."""
        v_dc, _, _ = self.mesh_grid.dual_contour(self.phi, iso=0.0)
        v_mc, _, _ = self.mesh_grid.marching_cubes(self.phi, level=0.0)
        # different algorithms, but both trace the same isosurface -> same bounding box (within a voxel)
        self.assertTrue(torch.allclose(v_dc.amin(0), v_mc.amin(0), atol=2 * self.vx))
        self.assertTrue(torch.allclose(v_dc.amax(0), v_mc.amax(0), atol=2 * self.vx))

    def test_dc_batch_matches_single(self):
        """A batched dual_contour matches running each grid on its own: for every grid in the batch
        the per-grid vertex/face counts and vertex positions match the single-grid result, and face
        indices stay grid-local (max index < that grid's vertex count). Exercises the non-decimated
        path, whose vertex order is deterministic (stable surface-cell compaction)."""
        vx = self.vx
        gb = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(vx, 12), self._cube_points(vx, 10)]),
            voxel_sizes=vx,
        )
        analytic = (gb.ijk.jdata.float() * vx).norm(dim=1) - self.R
        gb2, phi_b = gb.retopologize_sdf(gb.jagged_like(analytic.clamp(-self.bw, self.bw)), band=self.band)
        vb, fb, nb = gb2.dual_contour(phi_b, iso=0.0)
        for i, half in enumerate([12, 10]):
            single = fvdb.Grid.from_points(self._cube_points(vx, half), voxel_size=vx)
            a = (single.ijk.float() * vx).norm(dim=1) - self.R
            g2, phi_s = single.retopologize_sdf(a.clamp(-self.bw, self.bw), band=self.band)
            vs, fs, _ = g2.dual_contour(phi_s, iso=0.0)
            # vertex order is deterministic (stable surface-cell compaction); positions match
            self.assertEqual(vb[i].jdata.shape[0], vs.shape[0])
            self.assertEqual(fb[i].jdata.shape[0], fs.shape[0])
            self.assertTrue(torch.allclose(vb[i].jdata, vs, atol=1e-4))
            self.assertLess(int(fb[i].jdata.max()), vb[i].jdata.shape[0])  # faces grid-local

    def test_dc_decimation_reduces(self):
        """Decimation reduces the mesh: uniform ``reduce=4`` yields strictly fewer (still valid,
        in-range) vertices than full detail, and curvature-adaptive ``adaptivity=0.5`` yields a
        non-empty mesh with no more vertices than full detail. The decimated paths use double
        atomics, so these are count/validity bounds rather than exact-equality checks."""
        v0, _, _ = self.mesh_grid.dual_contour(self.phi, iso=0.0)
        # uniform reduce collapses ~F^2 fewer vertices
        vr, fr, _ = self.mesh_grid.dual_contour(self.phi, iso=0.0, reduce=4)
        self.assertGreater(vr.shape[0], 0)
        self.assertLess(vr.shape[0], v0.shape[0])
        self.assertLess(int(fr.max()), vr.shape[0])
        # curvature-adaptive: non-empty, no more verts than full detail
        va, fa, _ = self.mesh_grid.dual_contour(self.phi, iso=0.0, adaptivity=0.5)
        self.assertGreater(va.shape[0], 0)
        self.assertLessEqual(va.shape[0], v0.shape[0])
        self.assertLess(int(fa.max()), va.shape[0])

    def test_dc_no_surface_is_empty(self):
        """A field with no sign crossing (all-positive) has no surface cells, so dual_contour returns
        empty but correctly shaped ``(0, 3)`` vertices, faces, and normals rather than erroring."""
        # an all-positive field has no sign crossing -> no surface cells -> empty mesh
        field = torch.full((self.grid.num_voxels,), self.bw, device=self.device)
        v, f, n = self.grid.dual_contour(field, iso=0.0)
        self.assertEqual(v.shape[0], 0)
        self.assertEqual(f.shape[0], 0)
        self.assertEqual(tuple(v.shape), (0, 3))
        self.assertEqual(tuple(f.shape), (0, 3))
        self.assertEqual(tuple(n.shape), (0, 3))

    def test_dc_planar_sdf_is_exact(self):
        """A planar (linear) SDF is the one input with an exact analytic output: QEF reproduces a
        plane exactly, so each interior vertex must lie on the plane ``n·x = d + iso`` and carry the
        plane normal ``n``, to float precision. The field is the unclamped exact plane SDF (so
        gradients are exact in the interior); vertices near the open mesh boundary are excluded since
        there the central-difference gradient falls back to the centre value (the surface SDF has no
        neighbour outside the finite grid). Covers axis-aligned and tilted planes plus a nonzero iso
        (which shifts the plane)."""
        grid = _dense_cube_grid(self.vx, half=12, device=self.device)
        world = grid.ijk.float() * self.vx

        def unit(*components: float) -> torch.Tensor:
            t = torch.tensor(components, device=self.device, dtype=torch.float32)
            return t / t.norm()

        # (plane normal n, offset d, iso level) -> surface is the plane  n·x = d + iso
        for n, d, iso in [
            (unit(1.0, 0.0, 0.0), 0.0, 0.0),
            (unit(0.0, 1.0, 0.0), 0.1, 0.0),
            (unit(1.0, 1.0, 1.0), 0.0, 0.0),
            (unit(2.0, -1.0, 0.0), -0.05, 0.05),
        ]:
            phi = world @ n - d  # exact SDF of the plane (|grad| = 1); unclamped -> no band artifacts
            v, _, nrm = grid.dual_contour(phi, iso=iso)
            interior = v.abs().amax(dim=1) < 0.45  # exclude the open boundary (block spans +/-0.6)
            self.assertGreater(int(interior.sum()), 0)
            v, nrm = v[interior], nrm[interior]
            # each interior vertex lies exactly on the plane n·x = d + iso
            self.assertLess(float((v @ n - (d + iso)).abs().max()), 1e-4)
            # each interior normal equals the constant plane normal
            self.assertTrue(torch.allclose(nrm, n.expand_as(nrm), atol=1e-4))

    def test_dc_sphere_radius_and_normals_accurate(self):
        """For a sphere SDF the surface is analytically ``‖x‖ = R``: DC must place every vertex within
        a sub-voxel band of that radius, and (the SDF gradient being exactly radial) every normal must
        point along ``x/‖x‖``."""
        v, _, n = self.mesh_grid.dual_contour(self.phi, iso=0.0)
        radius = v.norm(dim=1)
        self.assertLess(float((radius - self.R).abs().max()), 1.0 * self.vx)
        self.assertLess(float((radius - self.R).abs().mean()), 0.25 * self.vx)
        radial = v / radius.clamp_min(1e-9).unsqueeze(1)
        cos = (n * radial).sum(dim=1)  # +1 when the normal is the outward radial direction
        self.assertGreater(float(cos.min()), 0.8)
        self.assertGreater(float(cos.mean()), 0.99)

    def test_dc_sphere_is_valid_closed_genus0(self):
        """A closed sphere mesh is a well-formed genus-0 manifold: non-empty, face indices in range,
        every triangle non-degenerate, normals aligned with the vertices, no boundary or non-manifold
        edges, and Euler characteristic ``V - E + F == 2`` -- an exact topological invariant DC should
        satisfy for a well-resolved closed SDF. (Subsumes the generic mesh-validity checks.)"""
        v, f, n = self.mesh_grid.dual_contour(self.phi, iso=0.0)
        self.assertGreater(v.shape[0], 0)
        self.assertGreater(f.shape[0], 0)
        self.assertEqual(n.shape, v.shape)
        # face indices in range, no degenerate triangles
        self.assertGreaterEqual(int(f.min()), 0)
        self.assertLess(int(f.max()), v.shape[0])
        self.assertTrue(bool(((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])).all()))
        # closed genus-0 manifold: no boundary / non-manifold edges, Euler characteristic == 2
        num_edges, boundary, nonmanifold = _edge_stats(f, v.shape[0])
        self.assertEqual(boundary, 0)
        self.assertEqual(nonmanifold, 0)
        self.assertEqual(v.shape[0] - num_edges + f.shape[0], 2)

    def _cube_points(self, vx: float, half: int) -> torch.Tensor:
        rng = torch.arange(-half, half + 1, device=self.device, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
        return torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3) * vx


if __name__ == "__main__":
    unittest.main()
