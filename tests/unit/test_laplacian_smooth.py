# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import torch

import fvdb

from .test_sdf import _FACE_OFFSETS, _dense_cube_grid, reference_redistance


def reference_smooth(
    grid: "fvdb.Grid",
    field: torch.Tensor,
    iterations: int,
    mode: fvdb.SmoothingMode,
    boundary_value: float | None,
) -> torch.Tensor:
    """Float64 torch transcription of LaplacianSmooth.cu.

    Each pass is out = in + w * (mean of the 6 face neighbours - in), per channel, with an inactive
    face read as the centre value (boundary_value None) or as -/+boundary_value by the sign of the
    centre value. MEAN_CURVATURE is one w = 1 pass per iteration, TAUBIN a w = 0.5 then w = -0.53 pass.
    """
    ijk = grid.ijk
    face_index = torch.stack(
        [grid.ijk_to_index(ijk + torch.tensor(o, device=ijk.device, dtype=ijk.dtype)) for o in _FACE_OFFSETS], dim=1
    )
    active = face_index >= 0
    x = field.double()
    squeeze = x.dim() == 1
    if squeeze:
        x = x[:, None]
    weights = [1.0] * iterations if mode == fvdb.SmoothingMode.MEAN_CURVATURE else [0.5, -0.53] * iterations
    for w in weights:
        if boundary_value is None:
            inactive = x
        else:
            b = torch.tensor(boundary_value, dtype=x.dtype, device=x.device)
            inactive = torch.where(x < 0, -b, b)
        faces = inactive[:, None, :].expand(-1, 6, -1).clone()
        faces[active] = x[face_index[active]]
        x = x + w * (faces.sum(dim=1) / 6 - x)
    if squeeze:
        x = x[:, 0]
    return x.to(field.dtype)


class LaplacianSmoothTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("laplacian_smooth requires a CUDA device")
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.vx = 0.05
        self.band = 3
        self.bw = self.band * self.vx
        dense = _dense_cube_grid(self.vx, half=12, device=self.device)
        analytic = (dense.ijk.float() * self.vx).norm(dim=1) - 0.3
        # A narrow-band shell, so many voxels have inactive faces on both sides of the band.
        in_band = analytic.abs() < self.bw
        self.grid = dense.pruned_grid(in_band)
        self.sdf = analytic[in_band].clamp(-self.bw, self.bw)

    def _cube_points(self, half: int) -> torch.Tensor:
        rng = torch.arange(-half, half + 1, device=self.device, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
        return torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3) * self.vx

    def test_matches_reference(self):
        n = self.grid.num_voxels
        fields = {
            "scalar sdf": self.sdf.double(),
            "column": torch.randn(n, 1, device=self.device, dtype=torch.float64),
            "channels": torch.randn(n, 3, device=self.device, dtype=torch.float64),
        }
        for name, field in fields.items():
            for mode in (fvdb.SmoothingMode.MEAN_CURVATURE, fvdb.SmoothingMode.TAUBIN):
                for boundary in (None, self.bw):
                    with self.subTest(field=name, mode=mode.name, boundary=boundary):
                        out = self.grid.laplacian_smooth(field, iterations=3, mode=mode, boundary_value=boundary)
                        expected = reference_smooth(self.grid, field, 3, mode, boundary)
                        self.assertEqual(out.shape, field.shape)
                        self.assertLess((out - expected).abs().max().item(), 1e-12)

    def test_float32_matches_reference(self):
        out = self.grid.laplacian_smooth(self.sdf, iterations=4, boundary_value=self.bw)
        self.assertEqual(out.dtype, torch.float32)
        expected = reference_smooth(self.grid, self.sdf, 4, fvdb.SmoothingMode.MEAN_CURVATURE, self.bw)
        self.assertLess((out - expected).abs().max().item(), 1e-6)

    def test_zero_flux_keeps_constant_field(self):
        field = torch.full((self.grid.num_voxels, 2), 0.7, device=self.device)
        for mode in (fvdb.SmoothingMode.MEAN_CURVATURE, fvdb.SmoothingMode.TAUBIN):
            out = self.grid.laplacian_smooth(field, iterations=5, mode=mode)
            self.assertTrue(torch.allclose(out, field, atol=1e-6))

    def test_zero_iterations_returns_copy(self):
        out = self.grid.laplacian_smooth(self.sdf, iterations=0)
        self.assertTrue(torch.equal(out, self.sdf))
        self.assertNotEqual(out.data_ptr(), self.sdf.data_ptr())

    def test_smoothed_sdf_recipe_matches_reference(self):
        """The documented SDF recipe (redistance, smooth with sign continuation at band*vx, redistance)
        matches the float64 references composed the same way."""
        field = self.sdf.double()
        band_width = self.band * float(self.grid.voxel_size[0])
        for mode in (fvdb.SmoothingMode.MEAN_CURVATURE, fvdb.SmoothingMode.TAUBIN):
            with self.subTest(mode=mode.name):
                phi = self.grid.reinitialize_sdf(field, band=self.band, redistance_iters=6)
                phi = self.grid.laplacian_smooth(phi, iterations=2, mode=mode, boundary_value=band_width)
                phi = self.grid.reinitialize_sdf(phi, band=self.band, redistance_iters=6)
                expected = reference_redistance(self.grid, field, band=self.band, iters=6, order=1)
                expected = reference_smooth(self.grid, expected, 2, mode, band_width)
                expected = reference_redistance(self.grid, expected, band=self.band, iters=6, order=1)
                self.assertLess((phi - expected).abs().max().item() / self.vx, 1e-9)

    def test_batch_matches_single_with_per_grid_boundary(self):
        halves = [12, 10, 8]
        gb = fvdb.GridBatch.from_points(fvdb.JaggedTensor([self._cube_points(h) for h in halves]), voxel_sizes=self.vx)
        field = gb.jagged_like(torch.randn(gb.total_voxels, 2, device=self.device))
        boundary = torch.tensor([0.1, 0.2, 0.3], device=self.device)
        out = gb.laplacian_smooth(field, iterations=2, mode=fvdb.SmoothingMode.TAUBIN, boundary_value=boundary)
        for i, h in enumerate(halves):
            single = fvdb.Grid.from_points(self._cube_points(h), voxel_size=self.vx)
            expected = single.laplacian_smooth(
                field[i].jdata, iterations=2, mode=fvdb.SmoothingMode.TAUBIN, boundary_value=float(boundary[i])
            )
            self.assertTrue(torch.allclose(out[i].jdata, expected, atol=1e-6))

    def test_sliced_batch_matches_standalone(self):
        halves = [12, 10, 8]
        gb = fvdb.GridBatch.from_points(fvdb.JaggedTensor([self._cube_points(h) for h in halves]), voxel_sizes=self.vx)
        view = gb[1:3]
        standalone = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(h) for h in halves[1:]]), voxel_sizes=self.vx
        )
        values = torch.randn(view.total_voxels, device=self.device)
        out_view = view.laplacian_smooth(view.jagged_like(values), iterations=3, boundary_value=[0.1, 0.2])
        out_sa = standalone.laplacian_smooth(standalone.jagged_like(values), iterations=3, boundary_value=[0.1, 0.2])
        self.assertTrue(torch.equal(out_view.jdata, out_sa.jdata))

    def test_boundary_tensor_on_host_or_device(self):
        halves = [6, 5]
        gb = fvdb.GridBatch.from_points(fvdb.JaggedTensor([self._cube_points(h) for h in halves]), voxel_sizes=self.vx)
        field = gb.jagged_like(torch.randn(gb.total_voxels, device=self.device))
        host = torch.tensor([0.1, 0.25])
        out_host = gb.laplacian_smooth(field, iterations=2, boundary_value=host)
        out_device = gb.laplacian_smooth(field, iterations=2, boundary_value=host.to(self.device))
        out_list = gb.laplacian_smooth(field, iterations=2, boundary_value=[0.1, 0.25])
        self.assertTrue(torch.equal(out_host.jdata, out_device.jdata))
        self.assertTrue(torch.equal(out_host.jdata, out_list.jdata))

    def test_functional_matches_methods(self):
        out = fvdb.functional.laplacian_smooth_single(self.grid, self.sdf, 2, fvdb.SmoothingMode.TAUBIN, self.bw)
        expected = self.grid.laplacian_smooth(self.sdf, 2, fvdb.SmoothingMode.TAUBIN, self.bw)
        self.assertTrue(torch.equal(out, expected))

    def test_rejects_invalid_inputs(self):
        n = self.grid.num_voxels
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(torch.zeros(n + 1, device=self.device))
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(torch.zeros(n, 2, 2, device=self.device))
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(self.sdf, iterations=-1)
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(self.sdf, boundary_value=-0.1)
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(self.sdf, boundary_value=float("nan"))
        with self.assertRaises(ValueError):
            self.grid.laplacian_smooth(self.sdf, boundary_value=torch.tensor([-1.0], device=self.device))
        with self.assertRaises(TypeError):
            self.grid.laplacian_smooth(torch.zeros(n, dtype=torch.int32, device=self.device))
        with self.assertRaises(TypeError):
            self.grid.laplacian_smooth(self.sdf, mode="taubin")  # type: ignore[arg-type]
        gb = fvdb.GridBatch.from_points(
            fvdb.JaggedTensor([self._cube_points(4), self._cube_points(3)]), voxel_sizes=self.vx
        )
        with self.assertRaises(ValueError):
            gb.laplacian_smooth(gb.jagged_like(torch.zeros(gb.total_voxels, device=self.device)), boundary_value=[0.1])


if __name__ == "__main__":
    unittest.main()
