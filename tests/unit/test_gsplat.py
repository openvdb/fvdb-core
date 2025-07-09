# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import tempfile
import unittest
from pathlib import Path

import imageio
import numpy as np
import OpenImageIO as oiio
import point_cloud_utils as pcu
import torch
from fvdb.utils.tests import (
    create_uniform_grid_points_at_depth,
    generate_center_frame_point_at_depth,
    generate_random_4x4_xform,
    get_fvdb_test_data_path,
)

from fvdb import GaussianSplat3d, JaggedTensor, gaussian_render_jagged


def compare_images(pixels_or_path_a, pixels_or_path_b):
    """Return true, if the two images perceptually differ

    Unlike what the documentation says here
    https://openimageio.readthedocs.io/en/master/imagebufalgo.html#_CPPv4N4OIIO12ImageBufAlgo11compare_YeeERK8ImageBufRK8ImageBufR14CompareResultsff3ROIi
    `compare_Yee` returns `False` if the images are the **same**.

    Populated entries of the `CompareResults` objects are `maxerror`, `maxx`, `maxy`, `maxz`, and `nfail`,
    """
    img_a = oiio.ImageBuf(pixels_or_path_a)  # type: ignore
    img_b = oiio.ImageBuf(pixels_or_path_b)  # type: ignore
    cmp = oiio.CompareResults()  # type: ignore
    differ = oiio.ImageBufAlgo.compare_Yee(img_a, img_b, cmp)  # type: ignore
    return differ, cmp


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class TestGaussianRender(unittest.TestCase):

    data_path = get_fvdb_test_data_path() / "gsplat"
    save_image_data = False
    # NB: The files for regression data are saved at pwd to prevent accidental overwrites
    save_regression_data = False

    def setUp(self):
        self.device = "cuda:0"

        data_path = self.data_path / "test_garden_cropped.npz"

        data = np.load(data_path)
        means = torch.from_numpy(data["means3d"]).float().to(self.device)
        quats = torch.from_numpy(data["quats"]).float().to(self.device)
        scales = torch.from_numpy(data["scales"]).float().to(self.device)
        opacities = torch.from_numpy(data["opacities"]).float().to(self.device)
        colors = torch.from_numpy(data["colors"]).float().to(self.device)
        self.cam_to_world_mats = torch.from_numpy(data["viewmats"]).float().to(self.device)
        self.projection_mats = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.width = data["width"].item()
        self.height = data["height"].item()

        self.sh_degree = 3
        sh_coeffs = torch.zeros((means.shape[0], (self.sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        sh_0 = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        sh_n = sh_coeffs[:, 1:, :].clone()

        self.gs3d = GaussianSplat3d(
            means=means,
            quats=quats,
            log_scales=torch.log(scales),
            logit_opacities=torch.logit(opacities),
            sh0=sh_0,
            shN=sh_n,
            requires_grad=True,
        )

        nan_mean = means.clone()
        nan_mean[0] = torch.tensor([float("nan"), float("nan"), float("nan")], device=self.device)
        self.nan_gs3d = GaussianSplat3d(
            means=nan_mean,
            quats=quats,
            log_scales=torch.log(scales),
            logit_opacities=torch.logit(opacities),
            sh0=sh_0,
            shN=sh_n,
            requires_grad=True,
        )

        self.num_cameras = self.cam_to_world_mats.shape[0]
        self.near_plane = 0.01
        self.far_plane = 1e10

    def test_fully_fused_projection(self):
        proj_res = self.gs3d.project_gaussians_for_images_and_depths(
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            self.near_plane,
            self.far_plane,
        )
        radii = proj_res.radii
        means2d = proj_res.means2d
        depths = proj_res.render_quantities[..., -1]
        conics = proj_res.conics

        if self.save_regression_data:
            torch.save(radii, "regression_radii.pt")
            torch.save(means2d, "regression_means2d.pt")
            torch.save(depths, "regression_depths.pt")
            torch.save(conics, "regression_conics.pt")

        # Regression test
        test_radii = torch.load(self.data_path / "regression_radii.pt", weights_only=True)
        test_means2d = torch.load(self.data_path / "regression_means2d.pt", weights_only=True)
        test_depths = torch.load(self.data_path / "regression_depths.pt", weights_only=True)
        test_conics = torch.load(self.data_path / "regression_conics.pt", weights_only=True)

        torch.testing.assert_close(radii, test_radii)
        torch.testing.assert_close(means2d[radii > 0], test_means2d[radii > 0])
        torch.testing.assert_close(depths[radii > 0], test_depths[radii > 0])
        torch.testing.assert_close(conics[radii > 0], test_conics[radii > 0], atol=1e-5, rtol=1e-4)

    def _tensors_to_pixel(self, colors, alphas):
        canvas = (
            torch.cat(
                [
                    colors.reshape(self.num_cameras * self.height, self.width, 3),
                    alphas.reshape(self.num_cameras * self.height, self.width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return (canvas * 255).astype(np.uint8)

    def _create_gs3d_without_first_gaussian(self, gs3d):
        """Helper to create a new GS3D instance with the first gaussian removed."""
        return GaussianSplat3d(
            means=gs3d.means[1:],
            quats=gs3d.quats[1:],
            log_scales=gs3d.log_scales[1:],
            logit_opacities=gs3d.logit_opacities[1:],
            sh0=gs3d.sh0[1:, :, :],
            shN=gs3d.shN[1:, :, :],
            requires_grad=True,
        )

    def test_save_ply_handles_nan(self):
        tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

        self.nan_gs3d.save_ply(tf.name)

        # Remove the first element from all tensors to compare with expected loaded ply
        gs3d_without_nan = self._create_gs3d_without_first_gaussian(self.nan_gs3d)

        loaded = pcu.load_triangle_mesh(tf.name)
        attribs = loaded.vertex_data.custom_attributes
        means_loaded = torch.from_numpy(loaded.vertex_data.positions).to(self.device)
        self.assertTrue(torch.allclose(means_loaded, gs3d_without_nan.means))

        scales_loaded = torch.from_numpy(
            np.stack([attribs["scale_0"], attribs["scale_1"], attribs["scale_2"]], axis=-1)
        ).to(self.device)
        self.assertTrue(torch.allclose(scales_loaded, gs3d_without_nan.log_scales))

        quats_loaded = torch.from_numpy(
            np.stack(
                [
                    attribs["rot_0"],
                    attribs["rot_1"],
                    attribs["rot_2"],
                    attribs["rot_3"],
                ],
                axis=-1,
            )
        ).to(self.device)
        self.assertTrue(torch.allclose(quats_loaded, gs3d_without_nan.quats))

        opacities_loaded = torch.from_numpy(attribs["opacity"]).to(self.device)
        self.assertTrue(torch.allclose(opacities_loaded, gs3d_without_nan.logit_opacities))

        sh0_loaded = (
            torch.from_numpy(np.stack([attribs[f"f_dc_{i}"] for i in range(3)], axis=1)).to(self.device).unsqueeze(1)
        )
        self.assertTrue(torch.allclose(sh0_loaded, gs3d_without_nan.sh0))
        shN_loaded = torch.from_numpy(np.stack([attribs[f"f_rest_{i}"] for i in range(45)], axis=1)).to(self.device)
        shN_loaded = shN_loaded.view(gs3d_without_nan.num_gaussians, 15, 3)
        self.assertTrue(torch.allclose(shN_loaded, gs3d_without_nan.shN))

    def test_save_ply(self):
        tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

        self.gs3d.save_ply(tf.name)

        loaded = pcu.load_triangle_mesh(tf.name)
        attribs = loaded.vertex_data.custom_attributes
        means_loaded = torch.from_numpy(loaded.vertex_data.positions).to(self.device)
        self.assertTrue(torch.allclose(means_loaded, self.gs3d.means))

        scales_loaded = torch.from_numpy(
            np.stack([attribs["scale_0"], attribs["scale_1"], attribs["scale_2"]], axis=-1)
        ).to(self.device)
        self.assertTrue(torch.allclose(scales_loaded, self.gs3d.log_scales))

        quats_loaded = torch.from_numpy(
            np.stack(
                [
                    attribs["rot_0"],
                    attribs["rot_1"],
                    attribs["rot_2"],
                    attribs["rot_3"],
                ],
                axis=-1,
            )
        ).to(self.device)
        self.assertTrue(torch.allclose(quats_loaded, self.gs3d.quats))

        opacities_loaded = torch.from_numpy(attribs["opacity"]).to(self.device)
        self.assertTrue(torch.allclose(opacities_loaded, self.gs3d.logit_opacities))

        sh0_loaded = (
            torch.from_numpy(np.stack([attribs[f"f_dc_{i}"] for i in range(3)], axis=1)).to(self.device).unsqueeze(1)
        )
        self.assertTrue(torch.allclose(sh0_loaded, self.gs3d.sh0))

        shN_loaded = torch.from_numpy(np.stack([attribs[f"f_rest_{i}"] for i in range(45)], axis=1)).to(self.device)
        shN_loaded = shN_loaded.view(self.gs3d.num_gaussians, 15, 3)
        self.assertTrue(torch.allclose(shN_loaded, self.gs3d.shN))

    def test_gaussian_render(self):
        render_colors, render_alphas = self.gs3d.render_images(
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            self.near_plane,
            self.far_plane,
        )

        pixels = self._tensors_to_pixel(render_colors, render_alphas)
        differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_result.png"))

        if self.save_image_data:
            imageio.imsave(self.data_path / "output_gaussian_render.png", pixels)

        if self.save_regression_data:
            imageio.imsave("regression_gaussian_render_result.png", pixels)

        self.assertFalse(
            differ,
            f"Gaussian renders for Torch tensors differ from reference image at {cmp.nfail} pixels",
        )

    def test_gaussian_render_jagged(self):
        # There are two scenes
        jt_means = JaggedTensor([self.gs3d.means, self.gs3d.means]).to(self.device)
        jt_quats = JaggedTensor([self.gs3d.quats, self.gs3d.quats]).to(self.device)
        jt_scales = JaggedTensor([self.gs3d.scales, self.gs3d.scales]).to(self.device)
        jt_opacities = JaggedTensor([self.gs3d.opacities, self.gs3d.opacities]).to(self.device)

        sh_coeffs = torch.cat([self.gs3d.sh0, self.gs3d.shN], dim=1)  # [N, K, 3]
        jt_sh_coeffs = JaggedTensor([sh_coeffs, sh_coeffs]).to(self.device)

        # The first scene renders to 2 views and the second scene renders to a single view
        jt_viewmats = JaggedTensor([self.cam_to_world_mats[:2], self.cam_to_world_mats[2:]]).to(self.device)
        jt_Ks = JaggedTensor([self.projection_mats[:2], self.projection_mats[2:]]).to(self.device)

        # g_sizes = means.joffsets[1:] - means.joffsets[:-1]
        # c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]
        # tt = g_sizes.repeat_interleave(c_sizes)
        # camera_ids = torch.arange(viewmats.rshape[0], device=device).repeat_interleave(tt, dim=0)

        # dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
        # dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
        # shifts = dd0[1:] - dd1[:-1]
        # shifts = torch.cat([torch.tensor([0], device=device), shifts])  # [0, -1000, 0]
        # shifts_cumsum = shifts.cumsum(0)  # [0, -1000, -1000]
        # gaussian_ids = torch.arange(len(camera_ids), device=device)  # [0, 1, 2, ..., 2999]
        # gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, dim=0)

        render_colors, render_alphas, _ = gaussian_render_jagged(
            jt_means,
            jt_quats,
            jt_scales,
            jt_opacities,
            jt_sh_coeffs,
            jt_viewmats,
            jt_Ks,
            self.width,
            self.height,
            self.near_plane,  # near_plane
            self.far_plane,  # far_plane
            self.sh_degree,  # sh_degree_to_use
            16,  # tile_size
            0.0,  # radius_clip
            0.3,  # eps2d
            False,  # antialias
            False,  # return depth
            False,  # return debug info
            False,  # ortho
        )
        torch.cuda.synchronize()

        pixels = self._tensors_to_pixel(render_colors, render_alphas)
        differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_jagged_result.png"))

        if self.save_image_data:
            imageio.imsave(self.data_path / "output_gaussian_render_jagged.png", pixels)

        if self.save_regression_data:
            imageio.imsave("regression_gaussian_render_jagged_result.png", pixels)

        self.assertFalse(
            differ,
            f"Gaussian renders for jagged tensors differ from reference image at {cmp.nfail} pixels",
        )


class TestTopGaussianContributionsRender(unittest.TestCase):
    data_path = get_fvdb_test_data_path() / "gsplat"
    save_image_data = False
    # NB: The files for regression data are saved at pwd to prevent accidental overwrites
    save_regression_data = False

    def setUp(self):
        self.device = "cuda:0"

        data_path = self.data_path / "test_garden_cropped.npz"

        data = np.load(data_path)
        means = torch.from_numpy(data["means3d"]).float().to(self.device)
        quats = torch.from_numpy(data["quats"]).float().to(self.device)
        scales = torch.from_numpy(data["scales"]).float().to(self.device)
        opacities = torch.from_numpy(data["opacities"]).float().to(self.device)
        colors = torch.from_numpy(data["colors"]).float().to(self.device)
        self.cam_to_world_mats = torch.from_numpy(data["viewmats"]).float().to(self.device)
        self.projection_mats = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.width = data["width"].item()
        self.height = data["height"].item()

        self.sh_degree = 3
        sh_coeffs = torch.zeros((means.shape[0], (self.sh_degree + 1) ** 2, 3), device=self.device)
        sh_coeffs[:, 0, :] = rgb_to_sh(colors)
        sh_0 = sh_coeffs[:, 0, :].unsqueeze(1).clone()
        sh_n = sh_coeffs[:, 1:, :].clone()

        self.gs3d = GaussianSplat3d(
            means=means,
            quats=quats,
            log_scales=torch.log(scales),
            logit_opacities=torch.logit(opacities),
            sh0=sh_0,
            shN=sh_n,
            requires_grad=True,
        )

    def test_gaussians_center_render(self):
        h = 1024
        w = 512

        num_gaussian_layers = 10
        num_samples = 16

        cam_to_world_xform = torch.from_numpy(generate_random_4x4_xform()).to(self.device)
        world_to_cam_xform = torch.linalg.inv(cam_to_world_xform).float()

        # Fix intrinsics to match the actual image size
        # For image size 1024x512, principal point should be around (512, 256)
        focal_length = 18.0  # Reasonable focal length for this image size
        intrinsics = torch.tensor(
            [[focal_length, 0.0, w / 2.0], [0.0, focal_length, h / 2.0], [0.0, 0.0, 1.0]], device=self.device
        )

        means3d = torch.cat(
            [
                generate_center_frame_point_at_depth(h, w, (i + 1) * 8, intrinsics, cam_to_world_xform).reshape(-1, 3)
                for i in range(num_gaussian_layers)
            ],
            dim=0,
        )

        opacities = torch.cat(
            [
                torch.full((means3d.shape[0] // num_gaussian_layers,), 0.4, device=means3d.device)
                for _ in range(num_gaussian_layers)
            ],
            dim=0,
        )
        logit_opacities = torch.logit(opacities)

        # Generate identity quaternions (no rotation)
        # Identity quaternion is [x=0, y=0, z=0, w=1] representing no rotation
        quats = torch.zeros(means3d.shape[0], 4, device=means3d.device)
        quats[:, 3] = 1.0  # Set w component to 1, others remain 0

        scales = torch.full((means3d.shape[0], 3), 1e-30, device=means3d.device)
        log_scales = torch.log(scales)

        sh0 = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)
        shN = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)

        gs3d = GaussianSplat3d(means3d, quats, log_scales, logit_opacities, sh0, shN)

        ids, weights = gs3d.render_top_contributing_gaussian_ids(
            num_samples,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        self.assertTrue(
            torch.equal(
                ids[0, h // 2 - 1, w // 2 - 1, :num_gaussian_layers],
                torch.arange(num_gaussian_layers, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                ids[0, h // 2 - 1, w // 2 - 1, num_gaussian_layers:],
                torch.full((ids.shape[3] - num_gaussian_layers,), -1, device=self.device, dtype=torch.int32),
            )
        )

        expected_weights = torch.zeros(num_samples, device=self.device)
        accumulated_transparency = 1.0
        for i in range(num_gaussian_layers):
            expected_weights[i] = accumulated_transparency * opacities[0]
            accumulated_transparency *= 1.0 - opacities[0]

        self.assertTrue(torch.allclose(weights[0][h // 2 - 1][w // 2 - 1], expected_weights))

        # sparse rendering
        pixels_to_render = JaggedTensor([torch.tensor([[h // 2 - 1, w // 2 - 1]])]).to(self.device)

        sparse_ids, sparse_weights = gs3d.sparse_render_top_contributing_gaussian_ids(
            num_samples,
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        self.assertTrue(torch.equal(sparse_ids.unbind()[0][0], ids[0][h // 2 - 1][w // 2 - 1]))

        self.assertTrue(torch.equal(sparse_weights.unbind()[0][0], weights[0][h // 2 - 1][w // 2 - 1]))

    def test_gaussians_grid_render(self):
        h = 1024
        w = 512

        num_gaussian_layers = 12
        num_samples = 16

        cam_to_world_xform = torch.from_numpy(generate_random_4x4_xform()).to(self.device)
        world_to_cam_xform = torch.linalg.inv(cam_to_world_xform).float()

        # Fix intrinsics to match the actual image size
        # For image size 1024x512, principal point should be around (512, 256)
        focal_length = 18.0  # Reasonable focal length for this image size
        intrinsics = torch.tensor(
            [[focal_length, 0.0, w / 2.0], [0.0, focal_length, h / 2.0], [0.0, 0.0, 1.0]], device=self.device
        )

        means3d = torch.cat(
            [
                create_uniform_grid_points_at_depth(
                    h, w, (i + 1) * 8, intrinsics, cam_to_world_xform, spacing=2
                ).reshape(-1, 3)
                for i in range(num_gaussian_layers)
            ],
            dim=0,
        )

        opacities = torch.cat(
            [
                torch.full((means3d.shape[0] // num_gaussian_layers,), 0.4, device=means3d.device)
                for _ in range(num_gaussian_layers)
            ],
            dim=0,
        )
        logit_opacities = torch.logit(opacities)

        # Generate identity quaternions (no rotation)
        # Identity quaternion is [x=0, y=0, z=0, w=1] representing no rotation
        quats = torch.zeros(means3d.shape[0], 4, device=means3d.device)
        quats[:, 3] = 1.0  # Set w component to 1, others remain 0

        scales = torch.full((means3d.shape[0], 3), 1e-30, device=means3d.device)
        log_scales = torch.log(scales)

        sh0 = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)
        shN = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)

        gs3d = GaussianSplat3d(means3d, quats, log_scales, logit_opacities, sh0, shN)

        ids, weights = gs3d.render_top_contributing_gaussian_ids(
            num_samples,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        id_centers = ids[0][::2, ::2]

        # assert that the ids are -1 for samples beyond the number of gaussian layers
        self.assertTrue(
            torch.equal(
                id_centers[:, :, num_gaussian_layers:],
                torch.full(
                    (id_centers.shape[0], id_centers.shape[1], num_samples - num_gaussian_layers),
                    -1,
                    device=self.device,
                    dtype=torch.int32,
                ),
            )
        )

        # assert that the ids are the correct gaussian layer for each sample
        for i in range(num_gaussian_layers):
            self.assertTrue(
                torch.equal(
                    id_centers[:, :, i].flatten(),
                    torch.arange(
                        i * means3d.shape[0] // num_gaussian_layers,
                        (i + 1) * means3d.shape[0] // num_gaussian_layers,
                        device=self.device,
                        dtype=torch.int32,
                    ),
                )
            )

        # assert that the weights are the correct for each sample
        expected_weights = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
        accumulated_transparency = torch.ones(1, device=self.device, dtype=torch.float32)
        for i in range(num_gaussian_layers):
            expected_weights[i] = accumulated_transparency * opacities[0]
            accumulated_transparency *= 1.0 - opacities[0]

        weight_centers = weights[0][::2, ::2]
        expected_weight_centers = torch.broadcast_to(expected_weights, weight_centers.shape)

        self.assertTrue(torch.allclose(weight_centers, expected_weight_centers, atol=1e-5, rtol=1e-8))

        # sparse rendering - use pixels that we know have gaussians (center pixels)
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, w, (num_pixels_to_render,))
        yCoords = torch.randint(0, h, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        test_pixels = torch.stack([yCoords, xCoords], 1)

        pixels_to_render = JaggedTensor([test_pixels]).to(self.device)

        sparse_ids, sparse_weights = gs3d.sparse_render_top_contributing_gaussian_ids(
            num_samples,
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        for pixels, sparse_ids, reference_ids in zip(pixels_to_render.unbind(), sparse_ids.unbind(), ids):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_ids using the coordinates
            selected_reference_ids = reference_ids[y_coords, x_coords]  # [num_pixels_to_render, num_samples]

            self.assertTrue(torch.equal(sparse_ids, selected_reference_ids))

        # check weights
        for pixels, sparse_weights, reference_weights in zip(
            pixels_to_render.unbind(), sparse_weights.unbind(), weights
        ):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_weights using the coordinates
            selected_reference_weights = reference_weights[y_coords, x_coords]  # [num_pixels_to_render, num_samples]
            self.assertTrue(torch.equal(sparse_weights, selected_reference_weights))

    def test_gaussians_scene_render(self):
        ids, weights = self.gs3d.render_top_contributing_gaussian_ids(
            6,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        if self.save_regression_data:
            torch.save(ids, "regression_top_contributors_ids.pt")
            torch.save(weights, "regression_top_contributors_weights.pt")

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=True)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=True)

        self.assertTrue(torch.equal(ids, ids_regression))
        self.assertTrue(torch.equal(weights, weights_regression))

    def test_gaussians_scene_sparse_render(self):

        # sparse rendering - use pixels that we know have gaussians (center pixels)
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, self.width, (num_pixels_to_render,))
        yCoords = torch.randint(0, self.height, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        test_pixels = torch.stack([yCoords, xCoords], 1)

        pixels_to_render = JaggedTensor([test_pixels]).to(self.device)

        sparse_ids, sparse_weights = self.gs3d.sparse_render_top_contributing_gaussian_ids(
            6,
            pixels_to_render,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=True)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=True)

        for pixels, sparse_ids, reference_ids in zip(pixels_to_render.unbind(), sparse_ids.unbind(), ids_regression):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_ids using the coordinates
            selected_reference_ids = reference_ids[y_coords, x_coords]  # [num_pixels_to_render, num_samples]

            self.assertTrue(torch.equal(sparse_ids, selected_reference_ids))

        # check weights
        for pixels, sparse_weights, reference_weights in zip(
            pixels_to_render.unbind(), sparse_weights.unbind(), weights_regression
        ):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_weights using the coordinates
            selected_reference_weights = reference_weights[y_coords, x_coords]  # [num_pixels_to_render, num_samples]
            self.assertTrue(torch.equal(sparse_weights, selected_reference_weights))

    def test_gaussians_scene_dense_pixels_sparse_render(self):
        # Test that the sparse render works with dense pixel specification
        # Taking a [C, R, 2] tensor as pixels_to_render and returning Tensors [C, R, num_samples]

        # sparse rendering - use pixels that we know have gaussians (center pixels)
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, self.width, (num_pixels_to_render,))
        yCoords = torch.randint(0, self.height, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        pixels_to_render = torch.stack([yCoords, xCoords], 1).unsqueeze(0).to(self.device)

        sparse_ids, sparse_weights = self.gs3d.sparse_render_top_contributing_gaussian_ids(
            6,
            pixels_to_render,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=True)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=True)

        for pixels, sparse_ids, reference_ids in zip(pixels_to_render, sparse_ids, ids_regression):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_ids using the coordinates
            selected_reference_ids = reference_ids[y_coords, x_coords]  # [num_pixels_to_render, num_samples]

            self.assertTrue(torch.equal(sparse_ids, selected_reference_ids))

        # check weights
        for pixels, sparse_weights, reference_weights in zip(pixels_to_render, sparse_weights, weights_regression):
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_weights using the coordinates
            selected_reference_weights = reference_weights[y_coords, x_coords]  # [num_pixels_to_render, num_samples]
            self.assertTrue(torch.equal(sparse_weights, selected_reference_weights))


if __name__ == "__main__":
    unittest.main()
