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

import fvdb.functional.splat as splat
from fvdb import GaussianSplat3d
from fvdb._fvdb_cpp import CameraModel


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _functional_render(
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
    """Full functional forward pass through the decomposed pipeline."""
    raw = splat.project_to_2d(
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
    C = world_to_cam.size(0)
    opacities = splat.compute_opacities(logit_opacities, C, raw.compensations)
    features = splat.prepare_render_features(
        means,
        sh0,
        shN,
        world_to_cam,
        raw.radii,
        raw.depths,
        sh_degree_to_use,
        "rgb",
    )
    tiles = splat.intersect_tiles(
        raw.means2d,
        raw.radii,
        raw.depths,
        C,
        tile_size,
        image_width,
        image_height,
    )
    images, alphas = splat.rasterize_dense(
        raw.means2d,
        raw.conics,
        features,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        image_width,
        image_height,
        tile_size,
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

        # viewmats in the data file are world-to-camera transforms
        all_w2c = torch.from_numpy(data["viewmats"]).float().to(self.device)
        all_proj = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.W = data["width"].item()
        self.H = data["height"].item()

        # Use only 1 camera view for speed
        self.world_to_cam = all_w2c[0:1].contiguous()
        self.projection_matrices = all_proj[0:1].contiguous()

        # Convert to log-space / logit-space (matching BaseGaussianTestCase)
        self.log_scales_data = torch.log(scales)
        self.logit_opacities_data = torch.logit(opacities)
        self.means_data = means

        # SH conversion
        N = means.shape[0]
        sh_degree = 3
        sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        self.sh0_data = sh_coeffs[:, 0, :].unsqueeze(1).clone()  # [N, 1, 3]
        self.shN_data = sh_coeffs[:, 1:, :].clone()  # [N, 15, 3]

        self.quats_data = quats
        self.sh_degree_to_use = sh_degree

    # ------------------------------------------------------------------
    # Test 1: Forward pass numerical equivalence
    # ------------------------------------------------------------------
    def test_functional_forward_matches_oo(self):
        """Render one frame through both paths, assert images match."""
        # Prepare shared tensors (same objects for both paths)
        means = self.means_data.detach().requires_grad_(True)
        quats = self.quats_data.detach().requires_grad_(True)
        log_scales = self.log_scales_data.detach().requires_grad_(True)
        logit_opacities = self.logit_opacities_data.detach().requires_grad_(True)
        sh0 = self.sh0_data.detach().requires_grad_(True)
        shN = self.shN_data.detach().requires_grad_(True)

        # Functional path
        images_fn, alphas_fn = _functional_render(
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

        # OO path (same tensors)
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
    # Test 2: Backward pass gradient equivalence
    # ------------------------------------------------------------------
    def test_functional_backward_matches_oo(self):
        """Forward + backward through both paths, compare gradients."""
        # Create separate but identically-valued tensors for each path
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

        # Functional path
        images_fn, _ = _functional_render(
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
        loss_fn = images_fn.sum()
        loss_fn.backward()

        # OO path
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
        loss_oo = images_oo.sum()
        loss_oo.backward()

        # Compare gradients
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
                atol=1e-3,
                rtol=1e-5,
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

        # Render a "target" from the initial state (no grad needed)
        with torch.no_grad():
            target_images, _ = _functional_render(
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

        # Perturb parameters slightly to create optimization gap
        with torch.no_grad():
            means.add_(torch.randn_like(means) * 0.01)

        losses = []
        num_steps = 5
        for step in range(num_steps):
            optimizer.zero_grad()
            images, alphas = _functional_render(
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

            # Verify all gradients are non-zero and finite
            for param, name in zip(params, param_names):
                grad = param.grad
                assert grad is not None, f"Gradient for {name} is None at step {step}"
                assert torch.isfinite(grad).all(), f"Non-finite gradient for {name} at step {step}"
                assert grad.abs().sum() > 0, f"Zero gradient for {name} at step {step}"

            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over training
        self.assertLess(
            losses[-1],
            losses[0],
            f"Loss did not decrease: {losses}",
        )


if __name__ == "__main__":
    unittest.main()
