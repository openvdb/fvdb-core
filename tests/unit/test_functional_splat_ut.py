# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the Unscented Transform (UT) projection path through the
decomposed functional API.
"""

import unittest

import numpy as np
import torch
from fvdb.utils.tests import get_fvdb_test_data_path

import fvdb.functional as F
from fvdb import GaussianSplat3d
from fvdb.enums import CameraModel, GaussianRenderMode, ProjectionMethod
from fvdb import _fvdb_cpp as _C


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class TestUTProjection(unittest.TestCase):
    """Validate the UT projection path through the functional API."""

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

        self.world_to_cam = all_w2c[0:1].contiguous()
        self.projection_matrices = all_proj[0:1].contiguous()

        self.log_scales = torch.log(scales)
        self.logit_opacities = torch.logit(opacities)
        self.means = means
        self.quats = quats

        N = means.shape[0]
        sh_degree = 3
        sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        self.sh0 = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        self.shN = sh_coeffs[:, 1:, :].clone()
        self.sh_degree = sh_degree

    def test_ut_fwd_binding_produces_valid_shapes(self):
        """project_gaussians_ut_fwd produces correctly shaped outputs."""
        C = self.world_to_cam.size(0)
        N = self.means.size(0)
        dc = torch.empty(C, 0, device=self.device, dtype=self.means.dtype)

        radii, means2d, depths, conics, compensations = _C.project_gaussians_ut_fwd(
            self.means,
            self.quats,
            self.log_scales,
            self.world_to_cam,
            self.projection_matrices,
            dc,
            _C.CameraModel.PINHOLE,
            self.W,
            self.H,
            0.3,
            0.01,
            1e10,
            0.0,
            False,
        )

        self.assertEqual(radii.shape, (C, N))
        self.assertEqual(means2d.shape, (C, N, 2))
        self.assertEqual(depths.shape, (C, N))
        self.assertEqual(conics.shape, (C, N, 3))

    def test_project_gaussians_ut_returns_valid_projected(self):
        """project_gaussians with UNSCENTED produces a valid ProjectedGaussians."""
        projected = F.project_gaussians(
            means=self.means,
            quats=self.quats,
            log_scales=self.log_scales,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W,
            image_height=self.H,
            projection_method=ProjectionMethod.UNSCENTED,
        )

        self.assertIsNotNone(projected.means2d)
        self.assertIsNotNone(projected.conics)
        self.assertIsNotNone(projected.radii)
        self.assertIsNotNone(projected.depths)

    def test_ut_is_not_differentiable_through_projection(self):
        """UT projection does not produce gradients on means/quats/scales."""
        means = self.means.detach().requires_grad_(True)
        quats = self.quats.detach().requires_grad_(True)
        log_scales = self.log_scales.detach().requires_grad_(True)

        projected = F.project_gaussians(
            means=means,
            quats=quats,
            log_scales=log_scales,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W,
            image_height=self.H,
            projection_method=ProjectionMethod.UNSCENTED,
        )

        self.assertIsNone(projected.means2d.grad_fn)

    def test_ut_and_analytic_produce_comparable_images(self):
        """Both projection methods produce non-trivial, finite rendered images."""
        projected_analytic = F.project_gaussians(
            means=self.means,
            quats=self.quats,
            log_scales=self.log_scales,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W,
            image_height=self.H,
            projection_method=ProjectionMethod.ANALYTIC,
        )
        projected_ut = F.project_gaussians(
            means=self.means,
            quats=self.quats,
            log_scales=self.log_scales,
            world_to_camera_matrices=self.world_to_cam,
            projection_matrices=self.projection_matrices,
            image_width=self.W,
            image_height=self.H,
            projection_method=ProjectionMethod.UNSCENTED,
        )

        features_a = F.evaluate_gaussian_sh(
            self.means,
            self.sh0,
            self.shN,
            self.world_to_cam,
            projected_analytic,
            sh_degree_to_use=self.sh_degree,
            render_mode=GaussianRenderMode.FEATURES,
        )
        features_u = F.evaluate_gaussian_sh(
            self.means,
            self.sh0,
            self.shN,
            self.world_to_cam,
            projected_ut,
            sh_degree_to_use=self.sh_degree,
            render_mode=GaussianRenderMode.FEATURES,
        )

        tiles_a = F.intersect_gaussian_tiles(projected_analytic)
        tiles_u = F.intersect_gaussian_tiles(projected_ut)

        images_a, _ = F.rasterize_screen_space_gaussians(
            projected_analytic,
            features_a,
            self.logit_opacities,
            tiles_a,
        )
        images_u, _ = F.rasterize_screen_space_gaussians(
            projected_ut,
            features_u,
            self.logit_opacities,
            tiles_u,
        )

        self.assertTrue(torch.isfinite(images_a).all(), "Analytic image has non-finite values")
        self.assertTrue(torch.isfinite(images_u).all(), "UT image has non-finite values")
        self.assertGreater(images_a.abs().sum().item(), 0, "Analytic image is all zeros")
        self.assertGreater(images_u.abs().sum().item(), 0, "UT image is all zeros")


if __name__ == "__main__":
    unittest.main()
