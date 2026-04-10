# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests validating that the decomposed functional Gaussian splatting API can be
used to build a complete training loop WITHOUT GaussianSplat3d, and that
results match the OO API numerically.
"""

import unittest

import numpy as np
import torch
from fvdb.utils.tests import get_fvdb_test_data_path

import fvdb.functional as F
from fvdb import GaussianSplat3d
from fvdb.enums import CameraModel, GaussianRenderMode


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _functional_render_4stage(
    means,
    quats,
    log_scales,
    logit_opacities,
    sh0,
    shN,
    world_to_cam,
    projection_matrices,
    image_width,
    image_height,
    near=0.01,
    far=1e10,
    sh_degree_to_use=3,
    tile_size=16,
):
    """Full functional forward pass through the 4-stage pipeline."""
    projected = F.project_gaussians(
        means,
        quats,
        log_scales,
        world_to_cam,
        projection_matrices,
        image_width,
        image_height,
        eps_2d=0.3,
        near=near,
        far=far,
        radius_clip=0.0,
        antialias=False,
        camera_model=CameraModel.PINHOLE,
    )
    features = F.evaluate_gaussian_sh(
        means,
        sh0,
        shN,
        world_to_cam,
        projected,
        sh_degree_to_use=sh_degree_to_use,
        render_mode=GaussianRenderMode.FEATURES,
    )
    tiles = F.intersect_gaussian_tiles(projected, tile_size=tile_size)
    images, alphas = F.rasterize_screen_space_gaussians(
        projected,
        features,
        logit_opacities,
        tiles,
    )
    return images, alphas


class TestFunctionalSplatTraining(unittest.TestCase):
    """Validate functional API forward, backward, and training loop."""

    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)
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

        self.log_scales_data = torch.log(scales)
        self.logit_opacities_data = torch.logit(opacities)
        self.means_data = means

        N = means.shape[0]
        sh_degree = 3
        sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        self.sh0_data = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        self.shN_data = sh_coeffs[:, 1:, :].clone()

        self.quats_data = quats
        self.sh_degree_to_use = sh_degree

    # ------------------------------------------------------------------
    # Test 1: Forward pass numerical equivalence (4-stage vs OO)
    # ------------------------------------------------------------------
    def test_functional_forward_matches_oo(self):
        """Render one frame through both paths, assert images match."""
        means = self.means_data.detach().requires_grad_(True)
        quats = self.quats_data.detach().requires_grad_(True)
        log_scales = self.log_scales_data.detach().requires_grad_(True)
        logit_opacities = self.logit_opacities_data.detach().requires_grad_(True)
        sh0 = self.sh0_data.detach().requires_grad_(True)
        shN = self.shN_data.detach().requires_grad_(True)

        images_fn, alphas_fn = _functional_render_4stage(
            means,
            quats,
            log_scales,
            logit_opacities,
            sh0,
            shN,
            self.world_to_cam,
            self.projection_matrices,
            self.W,
            self.H,
            sh_degree_to_use=self.sh_degree_to_use,
        )

        gs3d = GaussianSplat3d.from_tensors(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
        )
        images_oo, alphas_oo = gs3d.render_images(
            self.world_to_cam,
            self.projection_matrices,
            self.W,
            self.H,
            near=0.01,
            far=1e10,
            sh_degree_to_use=self.sh_degree_to_use,
        )

        torch.testing.assert_close(images_fn, images_oo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas_fn, alphas_oo, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Test 2: Backward pass gradient equivalence (4-stage vs OO)
    # ------------------------------------------------------------------
    def test_functional_backward_matches_oo(self):
        """Forward + backward through both paths, compare gradients."""
        means_fn = self.means_data.detach().clone().requires_grad_(True)
        means_oo = self.means_data.detach().clone().requires_grad_(True)
        quats_fn = self.quats_data.detach().clone().requires_grad_(True)
        quats_oo = self.quats_data.detach().clone().requires_grad_(True)
        log_scales_fn = self.log_scales_data.detach().clone().requires_grad_(True)
        log_scales_oo = self.log_scales_data.detach().clone().requires_grad_(True)
        logit_opacities_fn = self.logit_opacities_data.detach().clone().requires_grad_(True)
        logit_opacities_oo = self.logit_opacities_data.detach().clone().requires_grad_(True)
        sh0_fn = self.sh0_data.detach().clone().requires_grad_(True)
        sh0_oo = self.sh0_data.detach().clone().requires_grad_(True)
        shN_fn = self.shN_data.detach().clone().requires_grad_(True)
        shN_oo = self.shN_data.detach().clone().requires_grad_(True)

        images_fn, _ = _functional_render_4stage(
            means_fn,
            quats_fn,
            log_scales_fn,
            logit_opacities_fn,
            sh0_fn,
            shN_fn,
            self.world_to_cam,
            self.projection_matrices,
            self.W,
            self.H,
            sh_degree_to_use=self.sh_degree_to_use,
        )
        images_fn.sum().backward()

        gs3d = GaussianSplat3d.from_tensors(
            means=means_oo,
            quats=quats_oo,
            log_scales=log_scales_oo,
            logit_opacities=logit_opacities_oo,
            sh0=sh0_oo,
            shN=shN_oo,
        )
        gs3d.requires_grad = True
        images_oo, _ = gs3d.render_images(
            self.world_to_cam,
            self.projection_matrices,
            self.W,
            self.H,
            near=0.01,
            far=1e10,
            sh_degree_to_use=self.sh_degree_to_use,
        )
        images_oo.sum().backward()

        fn_grads = {
            "means": means_fn.grad,
            "quats": quats_fn.grad,
            "log_scales": log_scales_fn.grad,
            "logit_opacities": logit_opacities_fn.grad,
            "sh0": sh0_fn.grad,
            "shN": shN_fn.grad,
        }
        for name, grad_fn in fn_grads.items():
            grad_oo = getattr(gs3d, name).grad
            self.assertIsNotNone(grad_fn, f"Functional gradient for {name} is None")
            self.assertIsNotNone(grad_oo, f"OO gradient for {name} is None")
            torch.testing.assert_close(
                grad_fn,
                grad_oo,
                atol=5e-3,
                rtol=1e-4,
                msg=f"Gradient mismatch for {name}",
            )

    # ------------------------------------------------------------------
    # Test 3: Training loop with Adam optimizer
    # ------------------------------------------------------------------
    def test_functional_training_loop(self):
        """5 steps of Adam on perturbed params, verify loss decreases."""
        means = self.means_data.detach().clone().requires_grad_(True)
        quats = self.quats_data.detach().clone().requires_grad_(True)
        log_scales = self.log_scales_data.detach().clone().requires_grad_(True)
        logit_opacities = self.logit_opacities_data.detach().clone().requires_grad_(True)
        sh0 = self.sh0_data.detach().clone().requires_grad_(True)
        shN = self.shN_data.detach().clone().requires_grad_(True)

        params = [means, quats, log_scales, logit_opacities, sh0, shN]
        param_names = ["means", "quats", "log_scales", "logit_opacities", "sh0", "shN"]
        optimizer = torch.optim.Adam(params, lr=0.01)

        with torch.no_grad():
            target_images, _ = _functional_render_4stage(
                means,
                quats,
                log_scales,
                logit_opacities,
                sh0,
                shN,
                self.world_to_cam,
                self.projection_matrices,
                self.W,
                self.H,
                sh_degree_to_use=self.sh_degree_to_use,
            )

        with torch.no_grad():
            means.add_(torch.randn_like(means) * 0.01)

        losses = []
        num_steps = 5
        for step in range(num_steps):
            optimizer.zero_grad()
            images, alphas = _functional_render_4stage(
                means,
                quats,
                log_scales,
                logit_opacities,
                sh0,
                shN,
                self.world_to_cam,
                self.projection_matrices,
                self.W,
                self.H,
                sh_degree_to_use=self.sh_degree_to_use,
            )
            loss = torch.nn.functional.l1_loss(images, target_images)
            loss.backward()

            for param, name in zip(params, param_names):
                grad = param.grad
                assert grad is not None, f"Gradient for {name} is None at step {step}"
                assert torch.isfinite(grad).all(), f"Non-finite gradient for {name} at step {step}"
                assert grad.abs().sum() > 0, f"Zero gradient for {name} at step {step}"

            optimizer.step()
            losses.append(loss.item())

        self.assertLess(
            losses[-1],
            losses[0],
            f"Loss did not decrease: {losses}",
        )


class TestFunctionalCrop(unittest.TestCase):
    """Validate that the crop parameter produces correct sub-regions and raises on invalid inputs."""

    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)
        self.device = "cuda:0"

        data_path = get_fvdb_test_data_path() / "gsplat" / "test_garden_cropped.npz"
        data = np.load(data_path)

        self.means = torch.from_numpy(data["means3d"]).float().to(self.device)
        self.quats = torch.from_numpy(data["quats"]).float().to(self.device)
        self.log_scales = torch.log(torch.from_numpy(data["scales"]).float().to(self.device))
        self.logit_opacities = torch.logit(torch.from_numpy(data["opacities"]).float().to(self.device))
        colors = torch.from_numpy(data["colors"]).float().to(self.device)

        sh_coeffs = torch.zeros((self.means.shape[0], 16, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        self.sh0 = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        self.shN = sh_coeffs[:, 1:, :].clone()

        self.world_to_cam = torch.from_numpy(data["viewmats"]).float().to(self.device)[0:1].contiguous()
        self.projection_matrices = torch.from_numpy(data["Ks"]).float().to(self.device)[0:1].contiguous()
        self.W = data["width"].item()
        self.H = data["height"].item()

        self.projected = F.project_gaussians(
            self.means,
            self.quats,
            self.log_scales,
            self.world_to_cam,
            self.projection_matrices,
            self.W,
            self.H,
            eps_2d=0.3,
            near=0.01,
            far=1e10,
            radius_clip=0.0,
            antialias=False,
            camera_model=CameraModel.PINHOLE,
        )
        self.features = F.evaluate_gaussian_sh(
            self.means,
            self.sh0,
            self.shN,
            self.world_to_cam,
            self.projected,
            sh_degree_to_use=3,
            render_mode=GaussianRenderMode.FEATURES,
        )
        self.tiles = F.intersect_gaussian_tiles(self.projected, tile_size=16)

    def test_crop_none_matches_full_image(self):
        """crop=None should be identical to not passing crop."""
        full, full_a = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
        )
        crop_none, crop_none_a = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
            crop=None,
        )
        torch.testing.assert_close(full, crop_none)
        torch.testing.assert_close(full_a, crop_none_a)

    def test_crop_full_image_matches_no_crop(self):
        """crop=(0, 0, W, H) should match no-crop rendering."""
        full, full_a = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
        )
        cropped, cropped_a = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
            crop=(0, 0, self.W, self.H),
        )
        self.assertEqual(cropped.shape, full.shape)
        torch.testing.assert_close(full, cropped)
        torch.testing.assert_close(full_a, cropped_a)

    def test_crop_sub_region_shape_and_content(self):
        """A sub-region crop should have the right shape and match the corresponding slice of the full render."""
        full, _ = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
        )
        ox, oy, cw, ch = 16, 16, 64, 48
        cropped, _ = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
            crop=(ox, oy, cw, ch),
        )
        self.assertEqual(cropped.shape[1], ch)
        self.assertEqual(cropped.shape[2], cw)
        expected = full[:, oy : oy + ch, ox : ox + cw, :]
        torch.testing.assert_close(cropped, expected, atol=1e-5, rtol=1e-5)

    def test_crop_is_differentiable(self):
        """Gradients should flow through the crop path."""
        logit_ops = self.logit_opacities.detach().clone().requires_grad_(True)
        cropped, _ = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            logit_ops,
            self.tiles,
            crop=(0, 0, 64, 48),
        )
        cropped.sum().backward()
        self.assertIsNotNone(logit_ops.grad)
        self.assertTrue(logit_ops.grad.abs().sum() > 0)

    def test_crop_clamps_to_image_bounds(self):
        """A crop extending beyond the image should be clamped, not error."""
        cropped, _ = F.rasterize_screen_space_gaussians(
            self.projected,
            self.features,
            self.logit_opacities,
            self.tiles,
            crop=(self.W - 32, self.H - 24, 128, 128),
        )
        self.assertEqual(cropped.shape[2], 32)
        self.assertEqual(cropped.shape[1], 24)

    def test_crop_rejects_negative_origin(self):
        with self.assertRaises(ValueError, msg="negative origin_x"):
            F.rasterize_screen_space_gaussians(
                self.projected,
                self.features,
                self.logit_opacities,
                self.tiles,
                crop=(-1, 0, 64, 48),
            )

    def test_crop_rejects_zero_size(self):
        with self.assertRaises(ValueError, msg="zero width"):
            F.rasterize_screen_space_gaussians(
                self.projected,
                self.features,
                self.logit_opacities,
                self.tiles,
                crop=(0, 0, 0, 48),
            )

    def test_crop_rejects_no_overlap(self):
        with self.assertRaises(ValueError, msg="origin beyond image"):
            F.rasterize_screen_space_gaussians(
                self.projected,
                self.features,
                self.logit_opacities,
                self.tiles,
                crop=(self.W + 10, 0, 64, 48),
            )

    def test_count_contributing_crop_matches_full(self):
        """count_contributing_gaussians with crop=(0,0,W,H) matches no-crop."""
        full_num, full_w = F.count_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
        )
        cropped_num, cropped_w = F.count_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            crop=(0, 0, self.W, self.H),
        )
        torch.testing.assert_close(full_num, cropped_num)
        torch.testing.assert_close(full_w, cropped_w)

    def test_count_contributing_crop_sub_region(self):
        """count_contributing_gaussians with a sub-region crop matches the slice of the full result."""
        full_num, _ = F.count_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
        )
        ox, oy, cw, ch = 16, 16, 64, 48
        cropped_num, _ = F.count_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            crop=(ox, oy, cw, ch),
        )
        self.assertEqual(cropped_num.shape[1], ch)
        self.assertEqual(cropped_num.shape[2], cw)
        expected = full_num[:, oy : oy + ch, ox : ox + cw]
        torch.testing.assert_close(cropped_num, expected)

    def test_world_space_crop_full_matches_no_crop(self):
        """rasterize_world_space_gaussians with crop=(0,0,W,H) matches no-crop."""
        distortion_coeffs = torch.empty(
            self.world_to_cam.shape[0],
            0,
            device=self.device,
            dtype=torch.float32,
        )
        full, full_a = F.rasterize_world_space_gaussians(
            self.means,
            self.quats,
            self.log_scales,
            self.projected,
            self.features,
            self.logit_opacities,
            self.world_to_cam,
            self.projection_matrices,
            distortion_coeffs,
            CameraModel.PINHOLE,
            self.tiles,
        )
        cropped, cropped_a = F.rasterize_world_space_gaussians(
            self.means,
            self.quats,
            self.log_scales,
            self.projected,
            self.features,
            self.logit_opacities,
            self.world_to_cam,
            self.projection_matrices,
            distortion_coeffs,
            CameraModel.PINHOLE,
            self.tiles,
            crop=(0, 0, self.W, self.H),
        )
        torch.testing.assert_close(full, cropped)
        torch.testing.assert_close(full_a, cropped_a)

    def test_world_space_crop_sub_region(self):
        """rasterize_world_space_gaussians with a sub-region crop matches the slice of the full result."""
        distortion_coeffs = torch.empty(
            self.world_to_cam.shape[0],
            0,
            device=self.device,
            dtype=torch.float32,
        )
        full, _ = F.rasterize_world_space_gaussians(
            self.means,
            self.quats,
            self.log_scales,
            self.projected,
            self.features,
            self.logit_opacities,
            self.world_to_cam,
            self.projection_matrices,
            distortion_coeffs,
            CameraModel.PINHOLE,
            self.tiles,
        )
        ox, oy, cw, ch = 16, 16, 64, 48
        cropped, _ = F.rasterize_world_space_gaussians(
            self.means,
            self.quats,
            self.log_scales,
            self.projected,
            self.features,
            self.logit_opacities,
            self.world_to_cam,
            self.projection_matrices,
            distortion_coeffs,
            CameraModel.PINHOLE,
            self.tiles,
            crop=(ox, oy, cw, ch),
        )
        self.assertEqual(cropped.shape[1], ch)
        self.assertEqual(cropped.shape[2], cw)
        expected = full[:, oy : oy + ch, ox : ox + cw, :]
        torch.testing.assert_close(cropped, expected, atol=1e-5, rtol=1e-5)

    def test_identify_crop_full_matches_no_crop(self):
        """identify_contributing_gaussians with crop=(0,0,W,H) matches no-crop."""
        full_ids, full_w = F.identify_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            top_k_contributors=5,
        )
        cropped_ids, cropped_w = F.identify_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            top_k_contributors=5,
            crop=(0, 0, self.W, self.H),
        )
        torch.testing.assert_close(full_ids.jdata, cropped_ids.jdata)
        torch.testing.assert_close(full_w.jdata, cropped_w.jdata)
        torch.testing.assert_close(full_ids.joffsets, cropped_ids.joffsets)

    def test_identify_crop_sub_region(self):
        """identify_contributing_gaussians with a sub-region crop selects the correct pixels."""
        full_ids, full_w = F.identify_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            top_k_contributors=5,
        )
        ox, oy, cw, ch = 16, 16, 64, 48
        cropped_ids, cropped_w = F.identify_contributing_gaussians(
            self.projected,
            self.logit_opacities,
            self.tiles,
            top_k_contributors=5,
            crop=(ox, oy, cw, ch),
        )

        C = len(full_ids)
        self.assertEqual(len(cropped_ids), C)

        for c in range(C):
            full_cam = full_ids[c]
            crop_cam = cropped_ids[c]
            self.assertEqual(len(crop_cam), cw * ch)

            for dy in range(ch):
                for dx in range(cw):
                    crop_pixel_idx = dy * cw + dx
                    full_pixel_idx = (oy + dy) * self.W + (ox + dx)
                    crop_pixel_data = crop_cam[crop_pixel_idx].jdata
                    full_pixel_data = full_cam[full_pixel_idx].jdata
                    torch.testing.assert_close(crop_pixel_data, full_pixel_data)

    def test_identify_crop_rejects_invalid(self):
        """identify_contributing_gaussians raises on invalid crop inputs."""
        with self.assertRaises(ValueError):
            F.identify_contributing_gaussians(
                self.projected,
                self.logit_opacities,
                self.tiles,
                top_k_contributors=5,
                crop=(-1, 0, 64, 48),
            )
        with self.assertRaises(ValueError):
            F.identify_contributing_gaussians(
                self.projected,
                self.logit_opacities,
                self.tiles,
                top_k_contributors=5,
                crop=(0, 0, 0, 48),
            )


if __name__ == "__main__":
    unittest.main()
