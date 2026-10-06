# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shape, dtype and structure contract tests for the flat Gaussian splatting surface in ``fvdb.functional``.

These tests pin the public contract that downstream autograd wrappers rely on. They do not check
rendering correctness or gradients; the C++ gtests cover kernel numerics and gradient checks live
with the differentiable pipeline downstream.
"""

import functools
import math
import os
import tempfile
import unittest

import torch

import fvdb
import fvdb.functional as F
from fvdb import CameraModel, JaggedTensor, RollingShutterType, _fvdb_cpp
from fvdb.enums import _to_cpp_enum

_GAUSSIAN_EXPORTS = [
    "project_gaussians_analytic_fwd",
    "project_gaussians_analytic_bwd",
    "project_gaussians_analytic_jagged_fwd",
    "project_gaussians_analytic_jagged_bwd",
    "project_gaussians_ut_fwd",
    "evaluate_spherical_harmonics_fwd",
    "evaluate_spherical_harmonics_bwd",
    "intersect_gaussian_tiles",
    "intersect_gaussian_tiles_sparse",
    "build_sparse_gaussian_tile_layout",
    "as_pixel_jagged",
    "rasterize_screen_space_gaussians_fwd",
    "rasterize_screen_space_gaussians_bwd",
    "rasterize_screen_space_gaussians_sparse_fwd",
    "rasterize_screen_space_gaussians_sparse_bwd",
    "rasterize_world_space_gaussians_fwd",
    "rasterize_world_space_gaussians_bwd",
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
    "rasterize_top_contributing_gaussian_ids",
    "rasterize_top_contributing_gaussian_ids_sparse",
    "mcmc_relocate_gaussians",
    "mcmc_add_noise_to_means",
    "load_gaussian_ply",
    "save_gaussian_ply",
]

_REMOVED_BINDINGS = [
    "project_gaussians_unscented_fwd",
    "sparse_rasterize_num_contributing_gaussians",
    "sparse_rasterize_contributing_gaussian_ids",
]


class PublicSurfaceTests(unittest.TestCase):
    """The exported names, and the binding renames, are what issue #797 specifies."""

    def test_functional_exports_resolve(self):
        for name in _GAUSSIAN_EXPORTS:
            self.assertIn(name, F.__all__)
            self.assertTrue(callable(getattr(F, name)), name)

    def test_renamed_bindings_have_no_aliases(self):
        for name in _REMOVED_BINDINGS:
            self.assertFalse(hasattr(_fvdb_cpp, name), name)

    def test_enums_mirror_cpp(self):
        for py_enum, cpp_enum in [
            (CameraModel, _fvdb_cpp.CameraModel),
            (RollingShutterType, _fvdb_cpp.RollingShutterType),
        ]:
            cpp_members = {name: int(member) for name, member in cpp_enum.__members__.items()}
            py_members = {member.name: int(member) for member in py_enum}
            self.assertEqual(py_members, cpp_members, py_enum.__name__)

    def test_enum_conversion_accepts_and_rejects(self):
        cpp = _fvdb_cpp.CameraModel
        for value in (CameraModel.ORTHOGRAPHIC, int(CameraModel.ORTHOGRAPHIC), cpp.ORTHOGRAPHIC):
            self.assertEqual(_to_cpp_enum(CameraModel, cpp, value), cpp.ORTHOGRAPHIC)
        for bad in (True, 0.0, "PINHOLE", RollingShutterType.NONE, _fvdb_cpp.RollingShutterType.NONE):
            with self.assertRaises(TypeError, msg=repr(bad)):
                _to_cpp_enum(CameraModel, cpp, bad)
        with self.assertRaises(ValueError):
            _to_cpp_enum(CameraModel, cpp, 99)

    def test_enums_exported_from_fvdb(self):
        self.assertIs(fvdb.CameraModel, CameraModel)
        self.assertIs(fvdb.RollingShutterType, RollingShutterType)
        for name in ("CameraModel", "RollingShutterType"):
            self.assertIn(name, fvdb.__all__)

    def test_as_pixel_jagged_normalizes_and_validates(self):
        dense = torch.zeros(2, 4, 2, dtype=torch.int32)
        pixels = F.as_pixel_jagged(dense)
        self.assertIsInstance(pixels, JaggedTensor)
        self.assertEqual(pixels.num_tensors, 2)
        self.assertEqual(tuple(pixels.jdata.shape), (8, 2))
        # A JaggedTensor is returned as is.
        self.assertIs(F.as_pixel_jagged(pixels), pixels)
        for bad, error in (
            (torch.zeros(2, 4, 2), TypeError),  # float coordinates
            (torch.zeros(0, 4, 2, dtype=torch.int64), ValueError),  # no cameras
            (torch.zeros(2, 4, 3, dtype=torch.int64), ValueError),  # not (row, col)
            (JaggedTensor([torch.zeros(4, 3, dtype=torch.int64)]), ValueError),
            ([[0, 0]], TypeError),
        ):
            with self.assertRaises(error, msg=repr(bad)):
                F.as_pixel_jagged(bad)


class GaussianSplatFunctionalTests(unittest.TestCase):
    """Run every kernel wrapper once on a small scene and check the shapes and dtypes it returns."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Gaussian splatting kernels require a CUDA device")
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.C, self.N, self.D = 2, 96, 3
        self.W, self.H = 64, 48  # non-square so (h, w) ordering mistakes fail
        self.tile_size = 16
        self.tiles_h = math.ceil(self.H / self.tile_size)
        self.tiles_w = math.ceil(self.W / self.tile_size)
        self.image_size = {"image_width": self.W, "image_height": self.H}
        self.sh_degree = 2
        dev, N = self.device, self.N

        # Gaussians in a slab in front of the cameras.
        xy = torch.rand(N, 2, device=dev) * 2.0 - 1.0
        z = torch.rand(N, 1, device=dev) * 3.0 + 2.5
        self.means = torch.cat([xy, z], dim=1)
        self.quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=1)
        self.log_scales = torch.log(torch.rand(N, 3, device=dev) * 0.2 + 0.1)
        self.logit_opacities = torch.randn(N, device=dev)
        self.sh0 = torch.rand(N, 1, self.D, device=dev)
        self.shN = torch.rand(N, (self.sh_degree + 1) ** 2 - 1, self.D, device=dev) * 0.1

        # Two nearly coincident pinhole cameras looking down +z.
        w2c = torch.eye(4, device=dev).repeat(self.C, 1, 1)
        w2c[1, 0, 3] = 0.2
        self.w2c = w2c
        K = torch.tensor([[64.0, 0.0, 32.0], [0.0, 64.0, 24.0], [0.0, 0.0, 1.0]], device=dev)
        self.K = K.repeat(self.C, 1, 1)
        self.distortion = torch.empty(self.C, 0, device=dev)

        self.projected = F.project_gaussians_analytic_fwd(
            self.means,
            self.quats,
            self.log_scales,
            self.w2c,
            self.K,
            self.W,
            self.H,
            0.3,
            0.01,
            1e10,
            0.0,
            False,
            False,
        )
        self.radii, self.means2d, self.depths, self.conics, _ = self.projected
        self.opacities = torch.sigmoid(self.logit_opacities).unsqueeze(0).expand(self.C, -1).contiguous()
        self.tile_offsets, self.tile_gaussian_ids = F.intersect_gaussian_tiles(
            self.means2d, self.radii, self.depths, self.C, self.tile_size, self.tiles_h, self.tiles_w
        )
        empty_ids = torch.empty(0, dtype=torch.int32, device=dev)
        self.features = F.evaluate_spherical_harmonics_fwd(
            self.sh_degree, self.C, self.means, self.w2c, empty_ids, empty_ids, self.sh0, self.shN, self.radii
        )

        # Distinct, duplicate-free pixel sets per camera, as (row, col).
        rows, cols = torch.meshgrid(
            torch.arange(0, self.H, 3, device=dev), torch.arange(0, self.W, 5, device=dev), indexing="ij"
        )
        px0 = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=1)
        px1 = px0[::2] + 1
        px1 = px1[(px1[:, 0] < self.H) & (px1[:, 1] < self.W)]
        self.pixels = JaggedTensor([px0, px1])
        self.num_pixels = px0.shape[0] + px1.shape[0]

    # ------------------------------------------------------------------ helpers
    def _assert_shape(self, t: torch.Tensor, shape: tuple, dtype: torch.dtype | None = None):
        self.assertEqual(tuple(t.shape), tuple(shape))
        if dtype is not None:
            self.assertEqual(t.dtype, dtype)
        self.assertEqual(t.device.type, "cuda")

    def _assert_jagged_like_pixels(self, jt: JaggedTensor, element_shape: tuple, dtype: torch.dtype | None = None):
        self.assertIsInstance(jt, JaggedTensor)
        self.assertEqual(len(jt), self.C)
        self.assertEqual(tuple(jt.jdata.shape), (self.num_pixels, *element_shape))
        self.assertTrue(torch.equal(jt.joffsets.cpu(), self.pixels.joffsets.cpu()))
        if dtype is not None:
            self.assertEqual(jt.jdata.dtype, dtype)

    def _sparse_layout(self):
        active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = (
            F.build_sparse_gaussian_tile_layout(
                self.tile_size, self.tiles_h, self.tiles_w, self.pixels, **self.image_size
            )
        )
        tile_offsets, tile_gaussian_ids = F.intersect_gaussian_tiles_sparse(
            self.means2d,
            self.radii,
            self.depths,
            active_tile_mask,
            active_tiles,
            self.C,
            self.tile_size,
            self.tiles_h,
            self.tiles_w,
            conics=self.conics,
            opacities=self.opacities,
        )
        return (
            active_tiles,
            active_tile_mask,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            tile_offsets,
            tile_gaussian_ids,
        )

    # --------------------------------------------------------------- projection
    def test_project_analytic_fwd(self):
        radii, means2d, depths, conics, comps = self.projected
        C, N = self.C, self.N
        self._assert_shape(radii, (C, N, 2), torch.int32)
        self._assert_shape(means2d, (C, N, 2), torch.float32)
        self._assert_shape(depths, (C, N), torch.float32)
        self._assert_shape(conics, (C, N, 3), torch.float32)
        self.assertIsNone(comps)
        self.assertGreater(int((radii > 0).all(dim=-1).sum()), 0)

        with_comps = F.project_gaussians_analytic_fwd(
            self.means, self.quats, self.log_scales, self.w2c, self.K, self.W, self.H, 0.3, 0.01, 1e10, 0.0, True, False
        )
        self._assert_shape(with_comps[4], (C, N), torch.float32)

    def test_project_analytic_bwd(self):
        C, N = self.C, self.N
        accum_norms = torch.zeros(N, device=self.device)
        accum_radii = torch.zeros(N, dtype=torch.int32, device=self.device)
        accum_steps = torch.zeros(N, dtype=torch.int32, device=self.device)
        grads = F.project_gaussians_analytic_bwd(
            self.means,
            self.quats,
            self.log_scales,
            self.w2c,
            self.K,
            None,
            self.W,
            self.H,
            0.3,
            self.radii,
            self.conics,
            torch.ones_like(self.means2d),
            torch.ones_like(self.depths),
            torch.ones_like(self.conics),
            None,
            True,
            False,
            accum_norms,
            accum_radii,
            accum_steps,
        )
        d_means, d_covars, d_quats, d_log_scales, d_w2c = grads
        self._assert_shape(d_means, (N, 3), torch.float32)
        self.assertIsNone(d_covars)
        self._assert_shape(d_quats, (N, 4), torch.float32)
        self._assert_shape(d_log_scales, (N, 3), torch.float32)
        self._assert_shape(d_w2c, (C, 4, 4), torch.float32)
        self.assertGreater(int(accum_steps.sum()), 0)

        no_cam_grad = F.project_gaussians_analytic_bwd(
            self.means,
            self.quats,
            self.log_scales,
            self.w2c,
            self.K,
            None,
            self.W,
            self.H,
            0.3,
            self.radii,
            self.conics,
            torch.ones_like(self.means2d),
            torch.ones_like(self.depths),
            torch.ones_like(self.conics),
            None,
            False,
            False,
        )
        self.assertIsNone(no_cam_grad[4])

    def test_project_analytic_jagged_fwd_bwd(self):
        N = self.N
        g_sizes = torch.tensor([N // 2, N - N // 2], dtype=torch.int64, device=self.device)
        c_sizes = torch.tensor([1, 1], dtype=torch.int64, device=self.device)
        scales = torch.exp(self.log_scales)
        radii, means2d, depths, conics, comps = F.project_gaussians_analytic_jagged_fwd(
            g_sizes,
            self.means,
            self.quats,
            scales,
            c_sizes,
            self.w2c,
            self.K,
            self.W,
            self.H,
            0.3,
            0.01,
            1e10,
            0.0,
            False,
        )
        M = N  # one camera per scene
        self._assert_shape(radii, (M, 2), torch.int32)
        self._assert_shape(means2d, (M, 2), torch.float32)
        self._assert_shape(depths, (M,), torch.float32)
        self._assert_shape(conics, (M, 3), torch.float32)

        d_means, d_covars, d_quats, d_scales, d_w2c = F.project_gaussians_analytic_jagged_bwd(
            g_sizes,
            self.means,
            self.quats,
            scales,
            c_sizes,
            self.w2c,
            self.K,
            self.W,
            self.H,
            0.3,
            radii,
            conics,
            torch.ones_like(means2d),
            torch.ones_like(depths),
            torch.ones_like(conics),
            True,
            False,
        )
        self._assert_shape(d_means, (N, 3), torch.float32)
        self._assert_shape(d_quats, (N, 4), torch.float32)
        self._assert_shape(d_scales, (N, 3), torch.float32)
        self._assert_shape(d_w2c, (self.C, 4, 4), torch.float32)

    def test_project_ut_fwd_accepts_enum_and_int(self):
        C, N = self.C, self.N
        for camera_model in (CameraModel.PINHOLE, int(CameraModel.PINHOLE), _fvdb_cpp.CameraModel.PINHOLE):
            radii, means2d, depths, conics, comps = F.project_gaussians_ut_fwd(
                self.means,
                self.quats,
                self.log_scales,
                self.w2c,
                self.w2c,
                self.K,
                self.distortion,
                camera_model,
                self.W,
                self.H,
                0.3,
                0.01,
                1e10,
                0.0,
                True,
                rolling_shutter_type=RollingShutterType.NONE,
            )
            self._assert_shape(radii, (C, N, 2), torch.int32)
            self._assert_shape(means2d, (C, N, 2), torch.float32)
            self._assert_shape(depths, (C, N), torch.float32)
            self._assert_shape(conics, (C, N, 3), torch.float32)
            self._assert_shape(comps, (C, N), torch.float32)
            self.assertGreater(int((radii > 0).all(dim=-1).sum()), 0)

    # ------------------------------------------------------ spherical harmonics
    def test_evaluate_spherical_harmonics_fwd_bwd(self):
        C, N, D = self.C, self.N, self.D
        self._assert_shape(self.features, (C, N, D), torch.float32)
        empty_ids = torch.empty(0, dtype=torch.int32, device=self.device)
        d_sh0, d_shN, d_means, d_w2c = F.evaluate_spherical_harmonics_bwd(
            self.sh_degree,
            C,
            N,
            self.means,
            self.w2c,
            empty_ids,
            empty_ids,
            self.shN,
            torch.ones_like(self.features),
            self.radii,
            True,
            True,
        )
        self._assert_shape(d_sh0, (N, 1, D), torch.float32)
        self._assert_shape(d_shN, tuple(self.shN.shape), torch.float32)
        self._assert_shape(d_means, (N, 3), torch.float32)
        self._assert_shape(d_w2c, (C, 4, 4), torch.float32)

    def test_evaluate_spherical_harmonics_packed(self):
        N, D = self.N, self.D
        ids = torch.tensor([1, 0, 0, 5, 3, 5, N - 1], dtype=torch.int32, device=self.device)
        M = ids.shape[0]
        cam = torch.zeros(M, dtype=torch.int32, device=self.device)
        # Coefficients and radii are read by work item, so gather them into work-item order.
        packed = F.evaluate_spherical_harmonics_fwd(
            self.sh_degree,
            1,
            self.means,
            self.w2c[:1],
            cam,
            ids,
            self.sh0[ids.long()],
            self.shN[ids.long()],
            self.radii[:1, ids.long()].contiguous(),
        )
        self._assert_shape(packed, (1, M, D), torch.float32)
        expected = self.features[0, ids.long()]
        self.assertTrue(torch.allclose(packed[0], expected, atol=1e-6))

        d_sh0, d_shN, d_means, d_w2c = F.evaluate_spherical_harmonics_bwd(
            self.sh_degree,
            1,
            M,
            self.means,
            self.w2c[:1],
            cam,
            ids,
            self.shN[ids.long()],
            torch.ones_like(packed),
            self.radii[:1, ids.long()].contiguous(),
            True,
            True,
        )
        self._assert_shape(d_sh0, (M, 1, D), torch.float32)
        self._assert_shape(d_shN, (M, self.shN.shape[1], D), torch.float32)
        self._assert_shape(d_means, (N, 3), torch.float32)
        self._assert_shape(d_w2c, (1, 4, 4), torch.float32)
        # Gaussians never referenced receive no mean gradient; repeated ones accumulate.
        untouched = torch.ones(N, dtype=torch.bool, device=self.device)
        untouched[ids.long()] = False
        self.assertEqual(float(d_means[untouched].abs().max()), 0.0)

    # ------------------------------------------------------- tile intersection
    def test_intersect_gaussian_tiles(self):
        self._assert_shape(self.tile_offsets, (self.C, self.tiles_h, self.tiles_w), torch.int64)
        self.assertEqual(self.tile_gaussian_ids.dtype, torch.int32)
        self.assertEqual(self.tile_gaussian_ids.dim(), 1)
        self.assertGreater(self.tile_gaussian_ids.numel(), 0)

    def test_build_sparse_gaussian_tile_layout(self):
        active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = (
            self._sparse_layout()
        )
        AT = active_tiles.shape[0]
        self.assertGreater(AT, 0)
        self.assertEqual(active_tiles.dtype, torch.int32)
        self._assert_shape(active_tile_mask, (self.C, self.tiles_h, self.tiles_w), torch.bool)
        self.assertEqual(int(active_tile_mask.sum()), AT)
        self.assertEqual(tile_pixel_mask.dtype, torch.uint64)
        self.assertEqual(tile_pixel_mask.shape[0], AT)
        self._assert_shape(tile_pixel_cumsum, (AT,), torch.int64)
        self.assertEqual(int(tile_pixel_cumsum[-1]), self.num_pixels)
        self._assert_shape(pixel_map, (self.num_pixels,), torch.int64)
        self._assert_shape(tile_offsets, (AT + 1,), torch.int64)
        self.assertEqual(ids.dtype, torch.int32)

    def test_sparse_layout_accepts_plain_tensor(self):
        px0 = self.pixels[0].jdata
        for pixels, num_cameras in (
            (px0.unsqueeze(0), 1),
            (torch.stack([px0, px0 + torch.tensor([0, 1], device=px0.device)]), 2),
        ):
            active_tiles, mask, *_ = F.build_sparse_gaussian_tile_layout(
                self.tile_size, self.tiles_h, self.tiles_w, pixels, **self.image_size
            )
            self.assertEqual(tuple(mask.shape), (num_cameras, self.tiles_h, self.tiles_w))
            self.assertGreater(active_tiles.numel(), 0)
        # 2-D tensors are ambiguous ([P, 2] vs [C, 2]) and refused; other malformed shapes too.
        P = px0.shape[0]
        for bad in (px0, px0.reshape(-1), px0.reshape(1, P * 2, 1), px0.new_empty(0, P, 2)):
            with self.assertRaises(ValueError):
                F.build_sparse_gaussian_tile_layout(self.tile_size, self.tiles_h, self.tiles_w, bad, **self.image_size)

    def test_sparse_layout_rejects_bad_pixels(self):
        px0 = self.pixels[0].jdata
        layout = functools.partial(F.build_sparse_gaussian_tile_layout, **self.image_size)
        with self.assertRaises(TypeError):
            layout(self.tile_size, self.tiles_h, self.tiles_w, [1, 2, 3])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([px0.float()]))
        with self.assertRaises(ValueError):  # wrong element shape
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([px0[:, :1]]))
        with self.assertRaises(ValueError):  # row past the last tile row
            bad = px0.clone()
            bad[0, 0] = self.tiles_h * self.tile_size
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([bad]))
        with self.assertRaises(ValueError):  # negative column
            bad = px0.clone()
            bad[0, 1] = -1
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([bad]))
        with self.assertRaises(ValueError):  # duplicate pixel within one camera
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([torch.cat([px0, px0[:1]])]))
        # The same pixel in two different cameras is fine.
        layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([px0, px0]))
        with self.assertRaises(ValueError):  # non-positive tile grid
            layout(self.tile_size, 0, self.tiles_w, JaggedTensor([px0]))
        with self.assertRaises(ValueError):  # nested lists would index past the tile mask
            layout(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([[px0, px0], [px0]]))

        # A pixel inside the last tile row but below the image is rejected, not rendered.
        exact = functools.partial(F.build_sparse_gaussian_tile_layout, image_width=self.W, image_height=self.H - 5)
        inside = px0[px0[:, 0] < self.H - 5]
        exact(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([inside]))
        with self.assertRaises(ValueError):
            bad = inside.clone()
            bad[0, 0] = self.H - 5
            exact(self.tile_size, self.tiles_h, self.tiles_w, JaggedTensor([bad]))
        for width, height in ((self.W, self.H + 1), (self.W - self.tile_size, self.H), (0, self.H)):
            with self.assertRaises(ValueError):  # image size does not fit the tile grid
                F.build_sparse_gaussian_tile_layout(
                    self.tile_size,
                    self.tiles_h,
                    self.tiles_w,
                    JaggedTensor([inside]),
                    image_width=width,
                    image_height=height,
                )

    def test_sparse_wrappers_reject_layout_selection_mismatch(self):
        empty = JaggedTensor([torch.empty(0, 2, dtype=torch.int64, device=self.device) for _ in range(self.C)])
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map = F.build_sparse_gaussian_tile_layout(
            self.tile_size, self.tiles_h, self.tiles_w, empty, **self.image_size
        )
        tile_offsets = torch.zeros(1, dtype=torch.int64, device=self.device)
        ids = torch.empty(0, dtype=torch.int32, device=self.device)
        # Empty layout with a non-empty selection must not silently return zeros.
        with self.assertRaises(ValueError):
            F.rasterize_screen_space_gaussians_sparse_fwd(
                self.pixels,
                self.means2d,
                self.conics,
                self.features,
                self.opacities,
                self.W,
                self.H,
                0,
                0,
                self.tile_size,
                tile_offsets,
                ids,
                active_tiles,
                tile_pixel_mask,
                tile_pixel_cumsum,
                pixel_map,
            )
        with self.assertRaises(ValueError):
            F.rasterize_screen_space_gaussians_sparse_bwd(
                self.pixels,
                self.means2d,
                self.conics,
                self.features,
                self.opacities,
                self.W,
                self.H,
                0,
                0,
                self.tile_size,
                tile_offsets,
                ids,
                empty,
                empty,
                empty,
                empty,
                active_tiles,
                tile_pixel_mask,
                tile_pixel_cumsum,
                pixel_map,
                False,
            )

    def test_sparse_wrappers_reject_non_16_tile_size(self):
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = self._sparse_layout()
        for bad_tile_size in (8, 32):
            with self.assertRaises(ValueError):
                F.build_sparse_gaussian_tile_layout(
                    bad_tile_size, self.tiles_h, self.tiles_w, self.pixels, **self.image_size
                )
            with self.assertRaises(ValueError):
                F.rasterize_screen_space_gaussians_sparse_fwd(
                    self.pixels,
                    self.means2d,
                    self.conics,
                    self.features,
                    self.opacities,
                    self.W,
                    self.H,
                    0,
                    0,
                    bad_tile_size,
                    tile_offsets,
                    ids,
                    active_tiles,
                    tile_pixel_mask,
                    tile_pixel_cumsum,
                    pixel_map,
                )
            with self.assertRaises(ValueError):
                F.rasterize_num_contributing_gaussians_sparse(
                    self.means2d,
                    self.conics,
                    self.opacities,
                    tile_offsets,
                    ids,
                    self.pixels,
                    active_tiles,
                    tile_pixel_mask,
                    tile_pixel_cumsum,
                    pixel_map,
                    self.W,
                    self.H,
                    0,
                    0,
                    bad_tile_size,
                )

    def test_empty_sparse_selection_flows_through(self):
        C, N, D, K = self.C, self.N, self.D, 4
        empty = JaggedTensor([torch.empty(0, 2, dtype=torch.int64, device=self.device) for _ in range(C)])
        active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = (
            F.build_sparse_gaussian_tile_layout(self.tile_size, self.tiles_h, self.tiles_w, empty, **self.image_size)
        )
        self._assert_shape(active_tiles, (0,), torch.int32)
        self._assert_shape(active_tile_mask, (C, self.tiles_h, self.tiles_w), torch.bool)
        self.assertFalse(bool(active_tile_mask.any()))
        self.assertEqual(tile_pixel_mask.dtype, torch.uint64)
        self.assertEqual(tile_pixel_mask.shape[0], 0)
        self._assert_shape(tile_pixel_cumsum, (0,), torch.int64)
        self._assert_shape(pixel_map, (0,), torch.int64)
        tile_offsets, ids = F.intersect_gaussian_tiles_sparse(
            self.means2d,
            self.radii,
            self.depths,
            active_tile_mask,
            active_tiles,
            C,
            self.tile_size,
            self.tiles_h,
            self.tiles_w,
        )
        self._assert_shape(tile_offsets, (1,), torch.int64)
        self._assert_shape(ids, (0,), torch.int32)

        layout = (active_tiles, tile_pixel_mask, tile_pixel_cumsum, pixel_map)
        rendered, alphas, last_ids = F.rasterize_screen_space_gaussians_sparse_fwd(
            empty,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            *layout,
        )
        for jt, element_shape, dtype in (
            (rendered, (D,), torch.float32),
            (alphas, (1,), torch.float32),
            (last_ids, (), torch.int32),
        ):
            self.assertEqual(len(jt), C)
            self.assertEqual(tuple(jt.jdata.shape), (0, *element_shape))
            self.assertEqual(jt.jdata.dtype, dtype)
        grads = F.rasterize_screen_space_gaussians_sparse_bwd(
            empty,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            alphas,
            last_ids,
            rendered,
            alphas,
            *layout,
            True,
        )
        for g, ref in zip(grads, (self.means2d, self.means2d, self.conics, self.features, self.opacities)):
            self.assertEqual(tuple(g.shape), tuple(ref.shape))
            self.assertEqual(float(g.abs().max()), 0.0)
        # Argument types are checked even when there is nothing to rasterize.
        with self.assertRaises(TypeError):
            F.rasterize_screen_space_gaussians_sparse_bwd(
                empty,
                self.means2d,
                self.conics,
                self.features,
                self.opacities,
                self.W,
                self.H,
                0,
                0,
                self.tile_size,
                tile_offsets,
                ids,
                alphas.jdata,
                last_ids,
                rendered,
                alphas,
                *layout,
                True,
            )

        common = (
            self.means2d,
            self.conics,
            self.opacities,
            tile_offsets,
            ids,
            empty,
            *layout,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
        )
        counts, count_alphas = F.rasterize_num_contributing_gaussians_sparse(*common)
        self.assertEqual(tuple(counts.jdata.shape), (0,))
        self.assertEqual(tuple(count_alphas.jdata.shape), (0,))
        for mode_args in ((K, None), (0, counts)):
            g_ids, weights = F.rasterize_contributing_gaussian_ids_sparse(*common, *mode_args)
            self.assertEqual(tuple(g_ids.jdata.shape), (0,))
            self.assertEqual(tuple(weights.jdata.shape), (0,))
        with self.assertRaises(TypeError):
            F.rasterize_contributing_gaussian_ids_sparse(*common, 0, counts.jdata)
        top_ids, top_weights = F.rasterize_top_contributing_gaussian_ids_sparse(*common, K)
        self.assertEqual(tuple(top_ids.jdata.shape), (0, K))
        self.assertEqual(tuple(top_weights.jdata.shape), (0, K))

    def test_sparse_bwd_requires_jagged_tensors(self):
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = self._sparse_layout()
        rendered, alphas, last_ids = F.rasterize_screen_space_gaussians_sparse_fwd(
            self.pixels,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
        )
        with self.assertRaises(TypeError):
            F.rasterize_screen_space_gaussians_sparse_bwd(
                self.pixels,
                self.means2d,
                self.conics,
                self.features,
                self.opacities,
                self.W,
                self.H,
                0,
                0,
                self.tile_size,
                tile_offsets,
                ids,
                alphas.jdata,  # plain tensor where a JaggedTensor is required
                last_ids,
                rendered.jagged_like(torch.ones_like(rendered.jdata)),
                alphas.jagged_like(torch.ones_like(alphas.jdata)),
                active_tiles,
                tile_pixel_mask,
                tile_pixel_cumsum,
                pixel_map,
                False,
            )

    # ------------------------------------------------------------ rasterization
    def test_rasterize_screen_space_fwd_bwd(self):
        C, N, D, H, W = self.C, self.N, self.D, self.H, self.W
        rendered, alphas, last_ids = F.rasterize_screen_space_gaussians_fwd(
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            W,
            H,
            0,
            0,
            self.tile_size,
            self.tile_offsets,
            self.tile_gaussian_ids,
        )
        self._assert_shape(rendered, (C, H, W, D), torch.float32)
        self._assert_shape(alphas, (C, H, W, 1), torch.float32)
        self._assert_shape(last_ids, (C, H, W), torch.int32)
        self.assertGreater(float(alphas.max()), 0.0)

        for abs_grad in (False, True):
            d_abs, d_means2d, d_conics, d_features, d_opacities = F.rasterize_screen_space_gaussians_bwd(
                self.means2d,
                self.conics,
                self.features,
                self.opacities,
                W,
                H,
                0,
                0,
                self.tile_size,
                self.tile_offsets,
                self.tile_gaussian_ids,
                alphas,
                last_ids,
                torch.ones_like(rendered),
                torch.ones_like(alphas),
                abs_grad,
            )
            if abs_grad:
                self._assert_shape(d_abs, (C, N, 2), torch.float32)
            else:
                self.assertIsNone(d_abs)
            self._assert_shape(d_means2d, (C, N, 2), torch.float32)
            self._assert_shape(d_conics, (C, N, 3), torch.float32)
            self._assert_shape(d_features, (C, N, D), torch.float32)
            self._assert_shape(d_opacities, (C, N), torch.float32)

    def test_rasterize_screen_space_fwd_with_backgrounds_and_masks(self):
        backgrounds = torch.rand(self.C, self.D, device=self.device)
        masks = torch.ones(self.C, self.tiles_h, self.tiles_w, dtype=torch.bool, device=self.device)
        masks[:, 0, :] = False
        rendered, alphas, _ = F.rasterize_screen_space_gaussians_fwd(
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            self.tile_offsets,
            self.tile_gaussian_ids,
            backgrounds=backgrounds,
            masks=masks,
        )
        # Masked tiles receive the background with zero alpha.
        self.assertEqual(float(alphas[:, : self.tile_size].abs().max()), 0.0)
        self.assertTrue(torch.allclose(rendered[0, 0, 0], backgrounds[0]))

    def test_rasterize_screen_space_sparse_fwd_bwd(self):
        C, N, D = self.C, self.N, self.D
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = self._sparse_layout()
        rendered, alphas, last_ids = F.rasterize_screen_space_gaussians_sparse_fwd(
            self.pixels,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
        )
        self._assert_jagged_like_pixels(rendered, (D,), torch.float32)
        self._assert_jagged_like_pixels(alphas, (1,), torch.float32)
        self._assert_jagged_like_pixels(last_ids, (), torch.int32)

        d_abs, d_means2d, d_conics, d_features, d_opacities = F.rasterize_screen_space_gaussians_sparse_bwd(
            self.pixels,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            alphas,
            last_ids,
            rendered.jagged_like(torch.ones_like(rendered.jdata)),
            alphas.jagged_like(torch.ones_like(alphas.jdata)),
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            True,
        )
        self._assert_shape(d_abs, (C, N, 2), torch.float32)
        self._assert_shape(d_means2d, (C, N, 2), torch.float32)
        self._assert_shape(d_conics, (C, N, 3), torch.float32)
        self._assert_shape(d_features, (C, N, D), torch.float32)
        self._assert_shape(d_opacities, (C, N), torch.float32)

    def test_sparse_matches_dense_at_requested_pixels(self):
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = self._sparse_layout()
        sparse, _, _ = F.rasterize_screen_space_gaussians_sparse_fwd(
            self.pixels,
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            tile_offsets,
            ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
        )
        dense, _, _ = F.rasterize_screen_space_gaussians_fwd(
            self.means2d,
            self.conics,
            self.features,
            self.opacities,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
            self.tile_offsets,
            self.tile_gaussian_ids,
        )
        for c in range(self.C):
            px = self.pixels[c].jdata
            expected = dense[c, px[:, 0], px[:, 1]]
            self.assertTrue(torch.allclose(sparse[c].jdata, expected, atol=1e-5), f"camera {c}")

    def test_rasterize_world_space_fwd_bwd(self):
        C, N, D, H, W = self.C, self.N, self.D, self.H, self.W
        for shutter, model in ((RollingShutterType.NONE, CameraModel.PINHOLE), (0, 0)):
            rendered, alphas, last_ids = F.rasterize_world_space_gaussians_fwd(
                self.means,
                self.quats,
                self.log_scales,
                self.features,
                self.opacities,
                self.w2c,
                self.w2c,
                self.K,
                self.distortion,
                shutter,
                model,
                W,
                H,
                0,
                0,
                self.tile_size,
                self.tile_offsets,
                self.tile_gaussian_ids,
            )
            self._assert_shape(rendered, (C, H, W, D), torch.float32)
            self._assert_shape(alphas, (C, H, W, 1), torch.float32)
            self._assert_shape(last_ids, (C, H, W), torch.int32)

        d_means, d_quats, d_log_scales, d_features, d_opacities = F.rasterize_world_space_gaussians_bwd(
            self.means,
            self.quats,
            self.log_scales,
            self.features,
            self.opacities,
            self.w2c,
            self.w2c,
            self.K,
            self.distortion,
            RollingShutterType.NONE,
            CameraModel.PINHOLE,
            W,
            H,
            0,
            0,
            self.tile_size,
            self.tile_offsets,
            self.tile_gaussian_ids,
            alphas,
            last_ids,
            torch.ones_like(rendered),
            torch.ones_like(alphas),
        )
        self._assert_shape(d_means, (N, 3), torch.float32)
        self._assert_shape(d_quats, (N, 4), torch.float32)
        self._assert_shape(d_log_scales, (N, 3), torch.float32)
        self._assert_shape(d_features, (C, N, D), torch.float32)
        self._assert_shape(d_opacities, (C, N), torch.float32)

    # ----------------------------------------------------------------- analysis
    def test_analysis_dense(self):
        C, H, W, K = self.C, self.H, self.W, 8
        counts, alphas = F.rasterize_num_contributing_gaussians(
            self.means2d,
            self.conics,
            self.opacities,
            self.tile_offsets,
            self.tile_gaussian_ids,
            W,
            H,
            0,
            0,
            self.tile_size,
        )
        self._assert_shape(counts, (C, H, W), torch.int32)
        self._assert_shape(alphas, (C, H, W), torch.float32)
        total = int(counts.sum())
        self.assertGreater(total, 0)

        dense_args = (self.means2d, self.conics, self.opacities, self.tile_offsets, self.tile_gaussian_ids)
        # Top-K mode: positive num_depth_samples, counts are optional and ignored.
        for precomputed in (None, counts):
            ids, weights = F.rasterize_contributing_gaussian_ids(
                *dense_args, W, H, 0, 0, self.tile_size, K, precomputed
            )
            self.assertIsInstance(ids, JaggedTensor)
            self.assertIsInstance(weights, JaggedTensor)
            self.assertEqual(ids.jdata.dtype, torch.int32)
            self.assertEqual(ids.jdata.shape[0], weights.jdata.shape[0])
            self.assertEqual(ids.jdata.shape[0], int(counts.clamp(max=K).sum()))
        # All-contributors mode: num_depth_samples <= 0, counts required and honoured exactly.
        ids, weights = F.rasterize_contributing_gaussian_ids(*dense_args, W, H, 0, 0, self.tile_size, 0, counts)
        self.assertEqual(ids.jdata.shape[0], total)
        self.assertEqual(weights.jdata.shape[0], total)
        with self.assertRaises(ValueError):
            F.rasterize_contributing_gaussian_ids(*dense_args, W, H, 0, 0, self.tile_size, 0, None)

        top_ids, top_weights = F.rasterize_top_contributing_gaussian_ids(
            self.means2d,
            self.conics,
            self.opacities,
            self.tile_offsets,
            self.tile_gaussian_ids,
            W,
            H,
            0,
            0,
            self.tile_size,
            K,
        )
        self._assert_shape(top_ids, (C, H, W, K), torch.int32)
        self._assert_shape(top_weights, (C, H, W, K), torch.float32)
        self.assertTrue(bool((top_ids >= -1).all()))
        self.assertTrue(bool((top_ids < self.N).all()))

    def test_analysis_sparse(self):
        K = 8
        active_tiles, _, tile_pixel_mask, tile_pixel_cumsum, pixel_map, tile_offsets, ids = self._sparse_layout()
        common = (
            self.means2d,
            self.conics,
            self.opacities,
            tile_offsets,
            ids,
            self.pixels,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            self.W,
            self.H,
            0,
            0,
            self.tile_size,
        )
        counts, alphas = F.rasterize_num_contributing_gaussians_sparse(*common)
        self._assert_jagged_like_pixels(counts, (), torch.int32)
        self._assert_jagged_like_pixels(alphas, (), torch.float32)

        for precomputed in (None, counts):
            g_ids, weights = F.rasterize_contributing_gaussian_ids_sparse(*common, K, precomputed)
            self.assertIsInstance(g_ids, JaggedTensor)
            self.assertEqual(g_ids.jdata.dtype, torch.int32)
            self.assertEqual(g_ids.jdata.shape[0], weights.jdata.shape[0])
            self.assertEqual(g_ids.jdata.shape[0], int(counts.jdata.clamp(max=K).sum()))
        g_ids, weights = F.rasterize_contributing_gaussian_ids_sparse(*common, 0, counts)
        self.assertEqual(g_ids.jdata.shape[0], int(counts.jdata.sum()))
        self.assertEqual(weights.jdata.shape[0], int(counts.jdata.sum()))
        with self.assertRaises(ValueError):
            F.rasterize_contributing_gaussian_ids_sparse(*common, 0, None)

        top_ids, top_weights = F.rasterize_top_contributing_gaussian_ids_sparse(*common, K)
        self._assert_jagged_like_pixels(top_ids, (K,), torch.int32)
        self._assert_jagged_like_pixels(top_weights, (K,), torch.float32)

    # --------------------------------------------------------------------- MCMC
    def test_mcmc_relocate_and_add_noise(self):
        N, n_max = self.N, 5
        ratios = torch.randint(1, n_max + 1, (N,), dtype=torch.int32, device=self.device)
        binoms = torch.zeros(n_max, n_max, device=self.device)
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        new_logit_opacities, new_log_scales = F.mcmc_relocate_gaussians(
            self.log_scales, self.logit_opacities, ratios, binoms, n_max, 0.005
        )
        self._assert_shape(new_logit_opacities, (N,), torch.float32)
        self._assert_shape(new_log_scales, (N, 3), torch.float32)

        means = self.means.clone()
        F.mcmc_add_noise_to_means(means, self.log_scales, self.logit_opacities, self.quats, 1.0, 0.005, 100.0)
        self.assertEqual(tuple(means.shape), (N, 3))
        self.assertFalse(torch.equal(means, self.means))

    # ---------------------------------------------------------------------- PLY
    def test_ply_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "splats.ply")
            F.save_gaussian_ply(
                path,
                self.means,
                self.quats,
                self.log_scales,
                self.logit_opacities,
                self.sh0,
                self.shN,
                {"iteration": 7, "note": "contract"},
            )
            means, quats, log_scales, logit_opacities, sh0, shN, metadata = F.load_gaussian_ply(path, self.device)
        for loaded, original in (
            (means, self.means),
            (quats, self.quats),
            (log_scales, self.log_scales),
            (logit_opacities, self.logit_opacities),
            (sh0, self.sh0),
            (shN, self.shN),
        ):
            self.assertEqual(loaded.device.type, "cuda")
            self.assertTrue(torch.allclose(loaded, original, atol=1e-6))
        self.assertEqual(metadata["iteration"], 7)
        self.assertEqual(metadata["note"], "contract")


if __name__ == "__main__":
    unittest.main()
