# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the decomposed sparse rendering pipeline: intersect_gaussian_tiles_sparse
and rasterize_screen_space_gaussians_sparse through the functional API.
"""
import math
import unittest

import numpy as np
import torch

from fvdb import GaussianSplat3d, JaggedTensor, _fvdb_cpp as _C
from fvdb.enums import CameraModel, GaussianRenderMode, ProjectionMethod
from fvdb.utils.tests import get_fvdb_test_data_path

import fvdb.functional as F


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class TestDecomposedSparse(unittest.TestCase):
    """Validate the decomposed sparse rendering bindings and pipeline."""

    def setUp(self):
        torch.random.manual_seed(42)
        np.random.seed(42)
        self.device = "cuda:0"

        data_path = get_fvdb_test_data_path() / "gsplat" / "test_garden_cropped.npz"
        data = np.load(data_path)

        means = torch.from_numpy(data["means3d"]).float().to(self.device)
        quats = torch.from_numpy(data["quats"]).float().to(self.device)
        scales = torch.from_numpy(data["scales"]).float().to(self.device)
        opacities = torch.from_numpy(data["opacities"]).float().to(self.device)
        colors = torch.from_numpy(data["colors"]).float().to(self.device)

        all_w2c = torch.from_numpy(data["viewmats"]).float().to(self.device)
        all_proj = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.W = data["width"].item()
        self.H = data["height"].item()
        self.tile_size = 16

        self.world_to_cam = all_w2c[0:1].contiguous()
        self.projection_matrices = all_proj[0:1].contiguous()

        self.means = means
        self.quats = quats
        self.log_scales = torch.log(scales)
        self.logit_opacities = torch.logit(opacities)

        N = means.shape[0]
        sh_degree = 3
        sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        self.sh0 = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        self.shN = sh_coeffs[:, 1:, :].clone()
        self.sh_degree = sh_degree

    def _project(self):
        return F.project_gaussians(
            means=self.means, quats=self.quats, log_scales=self.log_scales,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W, image_height=self.H,
        )

    def _make_pixel_grid(self, step=8):
        """Create a grid of pixel coordinates covering the image."""
        ys = torch.arange(0, self.H, step, device=self.device)
        xs = torch.arange(0, self.W, step, device=self.device)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
        return grid.int()

    def test_build_sparse_tile_layout_shapes(self):
        """build_sparse_gaussian_tile_layout produces valid tile metadata tensors."""
        pixels = self._make_pixel_grid()
        pixels_jt = JaggedTensor([pixels])

        num_tiles_w = math.ceil(self.W / self.tile_size)
        num_tiles_h = math.ceil(self.H / self.tile_size)

        active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = (
            _C.build_sparse_gaussian_tile_layout(self.tile_size, num_tiles_w, num_tiles_h, pixels_jt._impl)
        )

        self.assertGreater(active_tiles.numel(), 0)
        self.assertEqual(active_tile_mask.shape[0], 1)  # C=1

    def test_intersect_gaussian_tiles_sparse_shapes(self):
        """intersect_gaussian_tiles_sparse produces valid tile offsets and IDs."""
        projected = self._project()
        pixels = self._make_pixel_grid()
        pixels_jt = JaggedTensor([pixels])

        num_tiles_w = math.ceil(self.W / self.tile_size)
        num_tiles_h = math.ceil(self.H / self.tile_size)

        active_tiles, active_tile_mask, _, _, _ = (
            _C.build_sparse_gaussian_tile_layout(self.tile_size, num_tiles_w, num_tiles_h, pixels_jt._impl)
        )

        C = self.world_to_cam.size(0)
        tile_offsets, tile_gaussian_ids = _C.intersect_gaussian_tiles_sparse(
            projected.means2d, projected.radii, projected.depths,
            active_tile_mask, active_tiles,
            C, self.tile_size, num_tiles_h, num_tiles_w,
        )

        self.assertEqual(tile_offsets.dim(), 1)
        self.assertEqual(tile_gaussian_ids.dim(), 1)

    def test_4stage_sparse_pipeline_produces_output(self):
        """The 4-stage sparse pipeline produces non-trivial, finite rendered features."""
        pixels = self._make_pixel_grid(step=4)
        pixels_jt = JaggedTensor([pixels])

        projected = self._project()
        features = F.evaluate_gaussian_sh(
            self.means, self.sh0, self.shN, self.world_to_cam, projected,
            sh_degree_to_use=self.sh_degree, render_mode=GaussianRenderMode.FEATURES,
        )
        sparse_tiles = F.intersect_gaussian_tiles_sparse(
            pixels_jt, projected, tile_size=self.tile_size,
        )
        features_jt, alphas_jt = F.rasterize_screen_space_gaussians_sparse(
            projected, features, self.logit_opacities, sparse_tiles,
        )

        self.assertTrue(torch.isfinite(features_jt.jdata).all())
        self.assertTrue(torch.isfinite(alphas_jt.jdata).all())
        self.assertGreater(features_jt.jdata.abs().sum().item(), 0)

    def test_4stage_sparse_matches_oo_sparse(self):
        """4-stage sparse pipeline matches GaussianSplat3d.sparse_render_images."""
        pixels = self._make_pixel_grid(step=8)
        pixels_jt = JaggedTensor([pixels])

        gs3d = GaussianSplat3d.from_tensors(
            means=self.means, quats=self.quats, log_scales=self.log_scales,
            logit_opacities=self.logit_opacities, sh0=self.sh0, shN=self.shN,
        )

        oo_features, oo_alphas = gs3d.sparse_render_images(
            pixels_to_render=pixels_jt,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W, image_height=self.H,
            near=0.01, far=1e10,
            sh_degree_to_use=self.sh_degree,
        )

        projected = self._project()
        features = F.evaluate_gaussian_sh(
            self.means, self.sh0, self.shN, self.world_to_cam, projected,
            sh_degree_to_use=self.sh_degree, render_mode=GaussianRenderMode.FEATURES,
        )
        sparse_tiles = F.intersect_gaussian_tiles_sparse(
            pixels_jt, projected, tile_size=self.tile_size,
        )
        fn_features_jt, fn_alphas_jt = F.rasterize_screen_space_gaussians_sparse(
            projected, features, self.logit_opacities, sparse_tiles,
        )

        if sparse_tiles.has_duplicates:
            from fvdb import JaggedTensor as JT
            fn_features_jt = JT(impl=fn_features_jt._impl[sparse_tiles.inverse_indices])
            fn_alphas_jt = JT(impl=fn_alphas_jt._impl[sparse_tiles.inverse_indices])

        torch.testing.assert_close(
            fn_features_jt.jdata, oo_features.jdata,
            atol=1e-4, rtol=1e-4,
            msg="Functional sparse features don't match OO sparse features",
        )
        torch.testing.assert_close(
            fn_alphas_jt.jdata, oo_alphas.jdata,
            atol=1e-4, rtol=1e-4,
            msg="Functional sparse alphas don't match OO sparse alphas",
        )


if __name__ == "__main__":
    unittest.main()
