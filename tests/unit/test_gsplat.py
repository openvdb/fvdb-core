# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import sys
import tempfile
import unittest

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
from parameterized import parameterized, parameterized_class

from fvdb import GaussianSplat3d, Grid, JaggedTensor, gaussian_render_jagged


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


class BaseGaussianTestCase(unittest.TestCase):
    data_path = get_fvdb_test_data_path() / "gsplat"
    save_image_data = False
    # NB: The files for regression data are saved at pwd to prevent accidental overwrites
    save_regression_data = True

    def setUp(self):
        torch.random.manual_seed(0)
        np.random.seed(0)
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

        self.gs3d = GaussianSplat3d.from_tensors(
            means=means,
            quats=quats,
            log_scales=torch.log(scales),
            logit_opacities=torch.logit(opacities),
            sh0=sh_0,
            shN=sh_n,
        )
        self.gs3d.requires_grad = True

        nan_mean = means.clone()
        nan_mean[0] = torch.tensor([float("nan"), float("nan"), float("nan")], device=self.device)
        self.nan_gs3d = GaussianSplat3d.from_tensors(
            means=nan_mean,
            quats=quats,
            log_scales=torch.log(scales),
            logit_opacities=torch.logit(opacities),
            sh0=sh_0,
            shN=sh_n,
        ).detach()
        self.nan_gs3d.requires_grad = True

        self.num_cameras = self.cam_to_world_mats.shape[0]
        self.near_plane = 0.01
        self.far_plane = 1e10


# @parameterized_class(("run_backward"), [(True,), (False,)])
# class TestGaussianSplatCat(BaseGaussianTestCase):
#     def setUp(self):
#         super().setUp()

#         self.run_backward = self.run_backward

#         self.gs3d1 = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means.clone(),
#             quats=self.gs3d.quats.clone(),
#             log_scales=self.gs3d.log_scales.clone(),
#             logit_opacities=self.gs3d.logit_opacities.clone(),
#             sh0=self.gs3d.sh0.clone(),
#             shN=self.gs3d.shN.clone(),
#             accumulate_max_2d_radii=self.gs3d.accumulate_max_2d_radii,
#             accumulate_mean_2d_gradients=self.gs3d.accumulate_mean_2d_gradients,
#             detach=True,  # Detach to avoid gradients from the original Gaussian Splat
#         )
#         self.gs3d2 = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means.clone() + 0.1,
#             quats=self.gs3d.quats.clone() + 0.01,
#             log_scales=self.gs3d.log_scales.clone() + 0.01,
#             logit_opacities=self.gs3d.logit_opacities.clone() + 0.01,
#             sh0=self.gs3d.sh0.clone() + 0.01,
#             shN=self.gs3d.shN.clone() + 0.01,
#             accumulate_max_2d_radii=self.gs3d.accumulate_max_2d_radii,
#             accumulate_mean_2d_gradients=self.gs3d.accumulate_mean_2d_gradients,
#         )
#         self.gs3d3 = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means.clone() + 0.2,
#             quats=self.gs3d.quats.clone() + 0.02,
#             log_scales=self.gs3d.log_scales.clone() + 0.02,
#             logit_opacities=self.gs3d.logit_opacities.clone() + 0.02,
#             sh0=self.gs3d.sh0.clone() + 0.02,
#             shN=self.gs3d.shN.clone() + 0.02,
#             accumulate_max_2d_radii=self.gs3d.accumulate_max_2d_radii,
#             accumulate_mean_2d_gradients=self.gs3d.accumulate_mean_2d_gradients,
#         )

#         self.device = torch.device(self.device)
#         if self.run_backward:
#             self.run_backward_on_gs3d(self.gs3d1)

#     def run_backward_on_gs3d(self, gs3d):
#         gs3d.requires_grad = True
#         gs3d.accumulate_max_2d_radii = True
#         gs3d.accumulate_mean_2d_gradients = True
#         rgb, alpha = gs3d.render_images(
#             self.cam_to_world_mats,
#             self.projection_mats,
#             self.width,
#             self.height,
#             self.near_plane,
#             self.far_plane,
#         )
#         loss = rgb.sum()
#         loss.backward()

#     def check_grad(self):
#         if self.run_backward:
#             self.assertTrue(self.gs3d1.means.grad is not None)
#             self.assertTrue(self.gs3d1.quats.grad is not None)
#             self.assertTrue(self.gs3d1.log_scales.grad is not None)
#             self.assertTrue(self.gs3d1.logit_opacities.grad is not None)
#             self.assertTrue(self.gs3d1.sh0.grad is not None)
#             self.assertTrue(self.gs3d1.shN.grad is not None)
#             self.assertTrue(self.gs3d1.accumulated_gradient_step_counts is not None)
#             self.assertTrue(self.gs3d1.accumulated_mean_2d_gradient_norms is not None)
#             if self.gs3d1.accumulate_max_2d_radii:
#                 self.assertTrue(self.gs3d1.accumulated_max_2d_radii is not None)
#             else:
#                 self.assertEqual(self.gs3d1.accumulated_max_2d_radii, None)

#             self.assertTrue(self.gs3d1.accumulated_gradient_step_counts.shape == (self.gs3d1.num_gaussians,))
#             self.assertTrue(self.gs3d1.accumulated_mean_2d_gradient_norms.shape == (self.gs3d1.num_gaussians,))

#     def check_basic(self, gs3d_cat, gs3d_list, acc2d_rad, acc_m2dgrad):

#         self.assertTrue(gs3d_cat.num_gaussians == len(gs3d_list) * self.gs3d.num_gaussians)
#         self.assertTrue(torch.equal(gs3d_cat.means, torch.cat([gs.means for gs in gs3d_list], dim=0)))
#         self.assertTrue(torch.equal(gs3d_cat.quats, torch.cat([gs.quats for gs in gs3d_list], dim=0)))
#         self.assertTrue(torch.equal(gs3d_cat.log_scales, torch.cat([gs.log_scales for gs in gs3d_list], dim=0)))
#         self.assertTrue(
#             torch.equal(gs3d_cat.logit_opacities, torch.cat([gs.logit_opacities for gs in gs3d_list], dim=0))
#         )
#         self.assertTrue(torch.equal(gs3d_cat.sh0, torch.cat([gs.sh0 for gs in gs3d_list], dim=0)))
#         self.assertTrue(torch.equal(gs3d_cat.shN, torch.cat([gs.shN for gs in gs3d_list], dim=0)))
#         self.assertEqual(gs3d_cat.accumulate_max_2d_radii, acc2d_rad)
#         self.assertEqual(gs3d_cat.accumulate_mean_2d_gradients, acc_m2dgrad)

#         if gs3d_cat.accumulate_max_2d_radii:
#             self.assertTrue(gs3d_cat.accumulated_max_2d_radii.shape == (gs3d_cat.num_gaussians,))
#             self.assertTrue(gs3d_cat.accumulated_max_2d_radii.dtype == torch.int32)
#             self.assertTrue(gs3d_cat.accumulated_max_2d_radii.device == self.device)
#         else:
#             self.assertEqual(gs3d_cat.accumulated_max_2d_radii, None)

#         if gs3d_cat.accumulate_max_2d_radii:
#             self.assertTrue(gs3d_cat.accumulated_gradient_step_counts.shape == (gs3d_cat.num_gaussians,))
#             self.assertTrue(gs3d_cat.accumulated_gradient_step_counts.dtype == torch.int32)
#             self.assertTrue(gs3d_cat.accumulated_gradient_step_counts.device == self.device)

#             self.assertTrue(gs3d_cat.accumulated_mean_2d_gradient_norms.shape == (gs3d_cat.num_gaussians,))
#             self.assertTrue(gs3d_cat.accumulated_mean_2d_gradient_norms.dtype == gs3d_cat.dtype)
#             self.assertTrue(gs3d_cat.accumulated_mean_2d_gradient_norms.device == self.device)
#         else:
#             self.assertEqual(gs3d_cat.accumulated_gradient_step_counts, None)
#             self.assertEqual(gs3d_cat.accumulated_mean_2d_gradient_norms, None)

#     def test_cat_basic(self):
#         gs3d_cat = GaussianSplat3d.cat(
#             [self.gs3d1, self.gs3d2, self.gs3d3], accumulate_max_2d_radii=False, accumulate_mean_2d_gradients=False
#         )
#         self.check_grad()
#         self.check_basic(gs3d_cat, [self.gs3d1, self.gs3d2, self.gs3d3], False, False)

#     def test_cat_track_state_no_backward_on_two_and_three(self):
#         gs3d_cat = GaussianSplat3d.cat(
#             [self.gs3d1, self.gs3d2, self.gs3d3], accumulate_max_2d_radii=True, accumulate_mean_2d_gradients=True
#         )

#         self.check_grad()
#         self.check_basic(gs3d_cat, [self.gs3d1, self.gs3d2, self.gs3d3], True, True)

#         self.assertTrue(gs3d_cat.accumulated_gradient_step_counts.shape, (gs3d_cat.num_gaussians,))
#         self.assertTrue(gs3d_cat.accumulated_mean_2d_gradient_norms.shape, (gs3d_cat.num_gaussians,))
#         if self.run_backward:
#             step_counts = torch.cat(
#                 [
#                     self.gs3d1.accumulated_gradient_step_counts,
#                     torch.zeros_like(self.gs3d1.accumulated_gradient_step_counts),
#                     torch.zeros_like(self.gs3d1.accumulated_gradient_step_counts),
#                 ],
#                 dim=0,
#             )
#             means2dnorms = torch.cat(
#                 [
#                     self.gs3d1.accumulated_mean_2d_gradient_norms,
#                     torch.zeros_like(self.gs3d1.accumulated_mean_2d_gradient_norms),
#                     torch.zeros_like(self.gs3d1.accumulated_mean_2d_gradient_norms),
#                 ],
#                 dim=0,
#             )
#             max_radii = torch.cat(
#                 [
#                     self.gs3d1.accumulated_max_2d_radii,
#                     torch.zeros_like(self.gs3d1.accumulated_max_2d_radii),
#                     torch.zeros_like(self.gs3d1.accumulated_max_2d_radii),
#                 ],
#                 dim=0,
#             )
#         else:
#             step_counts = torch.zeros(gs3d_cat.num_gaussians, dtype=torch.int32, device=self.device)
#             means2dnorms = torch.zeros(gs3d_cat.num_gaussians, dtype=self.gs3d1.dtype, device=self.device)
#             max_radii = torch.zeros(gs3d_cat.num_gaussians, dtype=torch.int32, device=self.device)
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_gradient_step_counts, step_counts))
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_mean_2d_gradient_norms, means2dnorms))
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_max_2d_radii, max_radii))

#     def test_cat_track_state_all_backward(self):
#         gs3d1_d, gs3d2_d, gs3d3_d = self.gs3d1.detach(), self.gs3d2.detach(), self.gs3d3.detach()
#         if self.run_backward:
#             self.run_backward_on_gs3d(gs3d1_d)
#             self.run_backward_on_gs3d(gs3d2_d)
#             self.run_backward_on_gs3d(gs3d3_d)

#         gs3d_cat = GaussianSplat3d.cat(
#             [gs3d1_d, gs3d2_d, gs3d3_d], accumulate_max_2d_radii=True, accumulate_mean_2d_gradients=True
#         )

#         self.check_grad()
#         self.check_basic(gs3d_cat, [gs3d1_d, gs3d2_d, gs3d3_d], True, True)

#         self.assertTrue(gs3d_cat.accumulate_max_2d_radii)
#         self.assertTrue(gs3d_cat.accumulate_mean_2d_gradients)

#         self.assertTrue(gs3d_cat.accumulated_gradient_step_counts.shape, (gs3d_cat.num_gaussians,))
#         self.assertTrue(gs3d_cat.accumulated_mean_2d_gradient_norms.shape, (gs3d_cat.num_gaussians,))

#         if self.run_backward:
#             step_counts = torch.cat(
#                 [
#                     gs3d1_d.accumulated_gradient_step_counts,
#                     gs3d2_d.accumulated_gradient_step_counts,
#                     gs3d3_d.accumulated_gradient_step_counts,
#                 ],
#                 dim=0,
#             )
#             means2dnorms = torch.cat(
#                 [
#                     gs3d1_d.accumulated_mean_2d_gradient_norms,
#                     gs3d2_d.accumulated_mean_2d_gradient_norms,
#                     gs3d3_d.accumulated_mean_2d_gradient_norms,
#                 ],
#                 dim=0,
#             )
#             max_radii = torch.cat(
#                 [
#                     gs3d1_d.accumulated_max_2d_radii,
#                     gs3d2_d.accumulated_max_2d_radii,
#                     gs3d3_d.accumulated_max_2d_radii,
#                 ],
#                 dim=0,
#             )
#         else:
#             step_counts = torch.cat(
#                 [
#                     torch.zeros(gs3d1_d.num_gaussians, device=self.device, dtype=torch.int32),
#                 ]
#                 * 3,
#                 dim=0,
#             )
#             means2dnorms = torch.cat(
#                 [
#                     torch.zeros(gs3d1_d.num_gaussians, device=self.device, dtype=gs3d1_d.dtype),
#                 ]
#                 * 3,
#                 dim=0,
#             )
#             max_radii = torch.cat(
#                 [
#                     torch.zeros(gs3d1_d.num_gaussians, device=self.device, dtype=torch.int32),
#                 ]
#                 * 3,
#                 dim=0,
#             )
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_gradient_step_counts, step_counts))
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_mean_2d_gradient_norms, means2dnorms))
#         self.assertTrue(torch.equal(gs3d_cat.accumulated_max_2d_radii, max_radii))


# @parameterized_class(("run_backward"), [(True,), (False,)])
# class TestGaussianSplatTo(BaseGaussianTestCase):
#     def setUp(self):
#         super().setUp()

#         self.run_backward = self.run_backward

#         if self.run_backward:
#             self.gs3d.accumulate_max_2d_radii = True
#             self.gs3d.accumulate_mean_2d_gradients = True
#             rgb1, alpha1 = self.gs3d.render_images(
#                 self.cam_to_world_mats,
#                 self.projection_mats,
#                 self.width,
#                 self.height,
#                 self.near_plane,
#                 self.far_plane,
#             )
#             loss1 = rgb1.sum()
#             loss1.backward()

#     def check_device_and_dtype(self, gs3d, device, dtype):
#         self.assertTrue(gs3d.device == device)
#         self.assertTrue(gs3d.dtype == dtype)
#         if gs3d.accumulated_gradient_step_counts is not None:
#             assert self.run_backward, "accumulated_gradient_step_counts should only be set when run_backward is True"
#             self.assertTrue(
#                 gs3d.accumulated_gradient_step_counts.shape == self.gs3d.accumulated_gradient_step_counts.shape
#             )
#             self.assertTrue(gs3d.accumulated_gradient_step_counts.device == device)
#             self.assertTrue(gs3d.accumulated_gradient_step_counts.dtype == torch.int32)
#         else:
#             assert not self.run_backward, "accumulated_gradient_step_counts should be None when run_backward is False"
#             self.assertEqual(self.gs3d.accumulated_gradient_step_counts, None)
#         if gs3d.accumulated_mean_2d_gradient_norms is not None:
#             assert self.run_backward, "accumulated_mean_2d_gradient_norms should only be set when run_backward is True"
#             self.assertTrue(
#                 gs3d.accumulated_mean_2d_gradient_norms.shape == self.gs3d.accumulated_mean_2d_gradient_norms.shape
#             )
#             self.assertTrue(gs3d.accumulated_mean_2d_gradient_norms.device == device)
#             self.assertTrue(gs3d.accumulated_mean_2d_gradient_norms.dtype == dtype)
#         else:
#             assert not self.run_backward, "accumulated_mean_2d_gradient_norms should be None when run_backward is False"
#             self.assertEqual(self.gs3d.accumulated_mean_2d_gradient_norms, None)

#         if gs3d.accumulated_max_2d_radii is not None:
#             assert self.run_backward, "accumulated_max_2d_radii should only be set when run_backward is True"
#             self.assertTrue(gs3d.accumulated_max_2d_radii.shape == self.gs3d.accumulated_max_2d_radii.shape)
#             self.assertTrue(gs3d.accumulated_max_2d_radii.device == device)
#             self.assertTrue(gs3d.accumulated_max_2d_radii.dtype == torch.int32)
#         else:
#             assert not self.run_backward, "accumulated_max_2d_radii should be None when run_backward is False"
#             self.assertEqual(self.gs3d.accumulated_max_2d_radii, None)

#     def test_to_device(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))

#         gs3d = self.gs3d.to("cpu")
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), self.gs3d.dtype)

#         gs3d = self.gs3d.to(self.device)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), self.gs3d.dtype)

#     def test_to_device_and_dtype(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))
#         self.assertTrue(self.gs3d.dtype == torch.float32)

#         gs3d = self.gs3d.to("cpu", torch.float16)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(self.device, torch.float32)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#     def test_to_device_and_dtype_kwargs(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))
#         self.assertTrue(self.gs3d.dtype == torch.float32)

#         gs3d = self.gs3d.to(device="cpu", dtype=torch.float16)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(device=self.device, dtype=torch.float32)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#     def test_to_dtype_and_device_kwargs(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))
#         self.assertTrue(self.gs3d.dtype == torch.float32)

#         gs3d = self.gs3d.to(dtype=torch.float16, device="cpu")
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(dtype=torch.float32, device=self.device)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#     def test_to_other(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))
#         self.assertTrue(self.gs3d.dtype == torch.float32)

#         cpu_f16_tensor = self.gs3d.means.to(device="cpu", dtype=torch.float16)

#         gs3d = self.gs3d.to(cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(self.gs3d.means)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#         gs3d = self.gs3d.to(cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(self.gs3d)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#         gs3d = self.gs3d.to(cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         grid = Grid.from_dense(1, 0, device="cuda")
#         gs3d = gs3d.to(grid)
#         self.check_device_and_dtype(gs3d, torch.device(grid.device), torch.float16)

#         jagged_cpu_f32 = JaggedTensor([cpu_f16_tensor.to(torch.float32)])
#         gs3d = gs3d.to(jagged_cpu_f32)
#         self.check_device_and_dtype(gs3d, jagged_cpu_f32.device, torch.float32)

#     def test_to_other_kwargs(self):
#         self.assertTrue(self.gs3d.device == torch.device(self.device))
#         self.assertTrue(self.gs3d.dtype == torch.float32)

#         cpu_f16_tensor = self.gs3d.means.to(device="cpu", dtype=torch.float16)

#         gs3d = self.gs3d.to(other=cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(other=self.gs3d.means)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#         gs3d = self.gs3d.to(other=cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         gs3d = self.gs3d.to(other=self.gs3d)
#         self.check_device_and_dtype(gs3d, torch.device(self.device), torch.float32)

#         gs3d = self.gs3d.to(other=cpu_f16_tensor)
#         self.check_device_and_dtype(gs3d, torch.device("cpu"), torch.float16)

#         grid = Grid.from_dense(1, 0, device="cuda")
#         gs3d = gs3d.to(other=grid)
#         self.check_device_and_dtype(gs3d, torch.device(grid.device), torch.float16)

#         jagged_cpu_f32 = JaggedTensor([cpu_f16_tensor.to(torch.float32)])
#         gs3d = gs3d.to(other=jagged_cpu_f32)
#         self.check_device_and_dtype(gs3d, jagged_cpu_f32.device, torch.float32)


# class TestGaussianSplatIndexSet(BaseGaussianTestCase):
#     def setUp(self):
#         super().setUp()

#     def make_src_and_dst(self, indices, src_acc_grad_mean_2d, dst_acc_grad_mean_2d, acc_max_2d_radii):
#         # Create a destination Gaussian Splat (matching self.gs3d) that requires gradients
#         dst = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means,
#             quats=self.gs3d.quats,
#             log_scales=self.gs3d.log_scales,
#             logit_opacities=self.gs3d.logit_opacities,
#             sh0=self.gs3d.sh0,
#             shN=self.gs3d.shN,
#             accumulate_max_2d_radii=acc_max_2d_radii,
#             accumulate_mean_2d_gradients=dst_acc_grad_mean_2d,
#         ).detach()
#         dst.requires_grad = True

#         # Create a source Gaussian Splat with half the Gaussians of the destination
#         # and make sure it requires gradients
#         num_src_gs = int(indices.sum().item()) if indices.dtype == torch.bool else int(indices.numel())
#         src = GaussianSplat3d.from_tensors(
#             means=torch.randn(num_src_gs, 3, device=self.device),
#             quats=torch.randn(num_src_gs, 4, device=self.device),
#             log_scales=torch.randn(num_src_gs, 3, device=self.device),
#             logit_opacities=torch.randn(num_src_gs, device=self.device),
#             sh0=torch.randn(num_src_gs, 1, 3, device=self.device),
#             shN=torch.randn(num_src_gs, 15, 3, device=self.device),
#             accumulate_mean_2d_gradients=src_acc_grad_mean_2d,
#             accumulate_max_2d_radii=acc_max_2d_radii,
#         )
#         src.requires_grad = True

#         # Render and compute losses on the source and destination Gaussian Splats
#         # to make sure they have gradients but have a seperate autograd graph
#         if dst_acc_grad_mean_2d or acc_max_2d_radii:
#             rgb1, alpha1 = dst.render_images(
#                 self.cam_to_world_mats,
#                 self.projection_mats,
#                 self.width,
#                 self.height,
#                 self.near_plane,
#                 self.far_plane,
#             )
#             loss1 = rgb1.sum()
#             loss1.backward()
#         if src_acc_grad_mean_2d or acc_max_2d_radii:
#             rgb2, alpha2 = src.render_images(
#                 self.cam_to_world_mats,
#                 self.projection_mats,
#                 self.width,
#                 self.height,
#                 self.near_plane,
#                 self.far_plane,
#             )
#             loss2 = rgb2.sum()
#             loss2.backward()

#         return src, dst

#     def compare_src_and_dst(
#         self,
#         src: GaussianSplat3d,
#         dst: GaussianSplat3d,
#         src_acc_m2d_grads: bool,
#         dst_track_m2d_grads: bool,
#         track_max_2d_radii: bool,
#         assertfun,
#         selfun,
#     ):
#         # Check that the source and destination Gaussians values
#         assertfun(torch.equal(src.means, selfun(dst.means)))
#         assertfun(torch.equal(src.quats, selfun(dst.quats)))
#         assertfun(torch.equal(src.log_scales, selfun(dst.log_scales)))
#         assertfun(torch.equal(src.logit_opacities, selfun(dst.logit_opacities)))
#         assertfun(torch.equal(src.sh0, selfun(dst.sh0)))
#         assertfun(torch.equal(src.shN, selfun(dst.shN)))

#         # Check that both the source and destination Gaussian Splat get their accumulate
#         # gradient state correctly set
#         if src_acc_m2d_grads and dst_track_m2d_grads:
#             assertfun(
#                 torch.equal(
#                     src.accumulated_gradient_step_counts,
#                     selfun(dst.accumulated_gradient_step_counts),
#                 )
#             )
#             assertfun(
#                 torch.equal(
#                     src.accumulated_mean_2d_gradient_norms,
#                     selfun(dst.accumulated_mean_2d_gradient_norms),
#                 )
#             )
#             if track_max_2d_radii:
#                 assertfun(
#                     torch.equal(
#                         src.accumulated_max_2d_radii,
#                         selfun(dst.accumulated_max_2d_radii),
#                     )
#                 )
#         elif dst_track_m2d_grads and not src_acc_m2d_grads:
#             assertfun(
#                 torch.equal(
#                     torch.zeros(src.num_gaussians).to(dst.accumulated_gradient_step_counts),
#                     selfun(dst.accumulated_gradient_step_counts),
#                 )
#             )
#             assertfun(
#                 torch.equal(
#                     torch.zeros(src.num_gaussians).to(dst.accumulated_mean_2d_gradient_norms),
#                     selfun(dst.accumulated_mean_2d_gradient_norms),
#                 )
#             )
#             if track_max_2d_radii:
#                 assertfun(
#                     torch.equal(
#                         torch.zeros(src.num_gaussians).to(dst.accumulated_max_2d_radii),
#                         selfun(dst.accumulated_max_2d_radii),
#                     )
#                 )
#         elif src_acc_m2d_grads and not dst_track_m2d_grads:

#             self.assertEqual(dst.accumulated_mean_2d_gradient_norms, None)
#             self.assertEqual(dst.accumulated_gradient_step_counts, None)
#             # Check that the destination Gaussian Splat has the same gradient shapes as before
#             self.assertTrue(src.accumulated_gradient_step_counts.shape == (src.num_gaussians,))
#             self.assertTrue(src.accumulated_mean_2d_gradient_norms.shape == (src.num_gaussians,))
#             if track_max_2d_radii:
#                 self.assertTrue(src.accumulated_max_2d_radii.shape == (src.num_gaussians,))

#     def _run_test(self, indices, src_requires_grad, dst_requires_grad, track_max_2d_radii, slicefun=None):
#         # Create the source and destination Gaussian Splats
#         src, dst = self.make_src_and_dst(
#             indices,
#             src_acc_grad_mean_2d=src_requires_grad,
#             dst_acc_grad_mean_2d=dst_requires_grad,
#             acc_max_2d_radii=track_max_2d_radii,
#         )

#         # We're testing slicing, so we can't write to the destiation tensor if it
#         # has requires_grad = True (since it's a leaf tensor)
#         if slicefun:
#             dst.requires_grad = False

#         # Check that the source and destination Gaussian Splat do not match before the assignment
#         self.compare_src_and_dst(
#             src=src,
#             dst=dst,
#             track_max_2d_radii=track_max_2d_radii,
#             src_acc_m2d_grads=src_requires_grad,
#             dst_track_m2d_grads=dst_requires_grad,
#             assertfun=self.assertFalse,
#             selfun=lambda x: x[indices],
#         )

#         # Do the assignment
#         if slicefun:
#             slicefun(src, dst, indices)
#         else:
#             dst[indices] = src

#         # Check that the source and destination Gaussian Splat match after the assignment
#         self.compare_src_and_dst(
#             src=src,
#             dst=dst,
#             track_max_2d_radii=track_max_2d_radii,
#             src_acc_m2d_grads=src_requires_grad,
#             dst_track_m2d_grads=dst_requires_grad,
#             assertfun=self.assertTrue,
#             selfun=lambda x: x[indices],
#         )

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def def_test_int_tensor_index(self, src_acc_m2d_grads, dst_acc_m2d_grads, track_max_2d_radii):
#         # Create indices that select half the Gaussians
#         half_indices = torch.arange(self.gs3d.num_gaussians // 2, device=self.device, dtype=torch.long)
#         self._run_test(
#             indices=half_indices,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#         )

#         # Create indices that select every other Gaussian
#         every_other_indices = torch.arange(0, self.gs3d.num_gaussians, 2, device=self.device, dtype=torch.long)
#         self._run_test(
#             indices=every_other_indices,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#         )

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def test_mask_set(self, src_acc_m2d_grads, dst_acc_m2d_grads, track_max_2d_radii):
#         mask = torch.zeros(self.gs3d.num_gaussians, dtype=torch.bool, device=self.device)
#         mask[: len(mask) // 2] = True  # Select first half of the Gaussians
#         self._run_test(
#             indices=mask,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#         )

#         mask = torch.zeros(self.gs3d.num_gaussians, dtype=torch.bool, device=self.device)
#         mask[::2] = True  # Select every other Gaussian
#         self._run_test(
#             indices=mask,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#         )

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def test_slice_set(self, src_acc_m2d_grads, dst_acc_m2d_grads, track_max_2d_radii):
#         # Create indices that select half the Gaussians
#         gt_idx = torch.arange(self.gs3d.num_gaussians // 2, device=self.device, dtype=torch.long)

#         def assignfun(src, dst, _):
#             dst[: self.gs3d.num_gaussians // 2] = src

#         self._run_test(
#             indices=gt_idx,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#             slicefun=assignfun,  # Use slice assignment
#         )

#         # Create indices that select every other Gaussian
#         gt_idx = torch.arange(0, self.gs3d.num_gaussians // 2, 2, device=self.device, dtype=torch.long)

#         def assignfun2(src, dst, _):
#             dst[: self.gs3d.num_gaussians // 2 : 2] = src

#         self._run_test(
#             indices=gt_idx,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#             slicefun=assignfun2,  # Use slice assignment
#         )

#         # Create indices that select every other Gaussian from 10 up to half
#         gt_idx = torch.arange(10, self.gs3d.num_gaussians // 2, 2, device=self.device, dtype=torch.long)

#         def assignfun3(src, dst, _):
#             dst[10 : self.gs3d.num_gaussians // 2 : 2] = src

#         self._run_test(
#             indices=gt_idx,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#             slicefun=assignfun3,  # Use slice assignment
#         )

#         # Create indices that select every other Gaussian up to -7
#         gt_idx = torch.arange(self.gs3d.num_gaussians, device=self.device, dtype=torch.long)[:-7]

#         def assignfun4(src, dst, _):
#             dst[:-7] = src

#         self._run_test(
#             indices=gt_idx,
#             src_requires_grad=src_acc_m2d_grads,
#             dst_requires_grad=dst_acc_m2d_grads,
#             track_max_2d_radii=track_max_2d_radii,
#             slicefun=assignfun4,  # Use slice assignment
#         )


# class TestGaussianSplatIndex(BaseGaussianTestCase):
#     def setUp(self):
#         super().setUp()

#     def _check(
#         self,
#         indices_or_mask: torch.Tensor,
#         selected: GaussianSplat3d,
#         dst: GaussianSplat3d,
#         accumulate_max_2d_radii: bool,
#         accumulate_mean_2d_gradients: bool,
#     ):
#         num_gs = (
#             int(indices_or_mask.sum().item()) if indices_or_mask.dtype == torch.bool else int(indices_or_mask.numel())
#         )
#         self.assertEqual(selected.num_gaussians, num_gs)
#         self.assertTrue(torch.equal(selected.means, dst.means[indices_or_mask]))
#         self.assertTrue(torch.equal(selected.quats, dst.quats[indices_or_mask]))
#         self.assertTrue(torch.equal(selected.log_scales, dst.log_scales[indices_or_mask]))
#         self.assertTrue(torch.equal(selected.logit_opacities, dst.logit_opacities[indices_or_mask]))
#         self.assertTrue(torch.equal(selected.sh0, dst.sh0[indices_or_mask]))
#         self.assertTrue(torch.equal(selected.shN, dst.shN[indices_or_mask]))

#         # Ensure the selected Gaussian Splat is empty
#         self.assertEqual(selected.num_gaussians, num_gs)
#         self.assertTrue(selected.means.shape == (num_gs, 3))
#         self.assertTrue(selected.quats.shape == (num_gs, 4))
#         self.assertTrue(selected.log_scales.shape == (num_gs, 3))
#         self.assertTrue(selected.logit_opacities.shape == (num_gs,))
#         self.assertTrue(selected.sh0.shape == (num_gs, 1, 3))
#         self.assertTrue(selected.shN.shape == (num_gs, dst.shN.shape[1], 3))

#         if accumulate_mean_2d_gradients:
#             # Ensure the gradients and accumulated gradient state match at every other Gaussian
#             self.assertTrue(
#                 torch.equal(
#                     selected.accumulated_gradient_step_counts,
#                     dst.accumulated_gradient_step_counts[indices_or_mask],
#                 )
#             )
#             self.assertTrue(
#                 torch.equal(
#                     selected.accumulated_mean_2d_gradient_norms,
#                     dst.accumulated_mean_2d_gradient_norms[indices_or_mask],
#                 )
#             )
#         if accumulate_max_2d_radii:
#             self.assertTrue(
#                 torch.equal(
#                     selected.accumulated_max_2d_radii,
#                     dst.accumulated_max_2d_radii[indices_or_mask],
#                 )
#             )

#     def _make_gs3d(
#         self, accumulate_mean_2d_gradients: bool, accumulate_max_2d_radii: bool, empty_shN: bool
#     ) -> GaussianSplat3d:
#         # Create a GaussianSplat3d instance with gradients that matches self.gs3d
#         shN = torch.empty((self.gs3d.num_gaussians, 0, 3), device=self.device) if empty_shN else self.gs3d.shN
#         gs3d = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means,
#             quats=self.gs3d.quats,
#             log_scales=self.gs3d.log_scales,
#             logit_opacities=self.gs3d.logit_opacities,
#             sh0=self.gs3d.sh0,
#             shN=shN,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=accumulate_max_2d_radii,
#         )
#         gs3d.requires_grad = False
#         if accumulate_mean_2d_gradients or accumulate_max_2d_radii:
#             gs3d.requires_grad = True
#             # Render images and compute a loss so we get gradients
#             rgb, alpha = gs3d.render_images(
#                 self.cam_to_world_mats,
#                 self.projection_mats,
#                 self.width,
#                 self.height,
#                 self.near_plane,
#                 self.far_plane,
#             )
#             loss = rgb.sum()
#             loss.backward()

#             # Check that we tracked accumulated gradient state properly
#             if accumulate_mean_2d_gradients:
#                 self.assertTrue(gs3d.accumulated_gradient_step_counts.shape == (gs3d.num_gaussians,))
#                 self.assertTrue(gs3d.accumulated_mean_2d_gradient_norms.shape == (gs3d.num_gaussians,))
#             if accumulate_max_2d_radii:
#                 self.assertTrue(gs3d.accumulated_max_2d_radii.shape == (gs3d.num_gaussians,))
#         return gs3d

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def test_gaussian_mask_selection(self, accumulate_mean_2d_gradients, track_max_2d_radii, empty_shN):

#         # Create a mask that selects every other Gaussian and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         every_other_mask = torch.zeros(gs3d.num_gaussians, dtype=torch.bool, device=self.device)
#         every_other_mask[::2] = True
#         gs3d_every_other = gs3d[every_other_mask]

#         self._check(
#             indices_or_mask=every_other_mask,
#             selected=gs3d_every_other,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#         # Create a mask that selects half
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         half_mask = torch.zeros(gs3d.num_gaussians, dtype=torch.bool, device=self.device)
#         half_mask[: gs3d.num_gaussians // 2] = True
#         gs3d_half = gs3d[half_mask]

#         self._check(
#             indices_or_mask=half_mask,
#             selected=gs3d_half,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#         # Create a mask that selects none
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         empty_mask = torch.zeros(gs3d.num_gaussians, dtype=torch.bool, device=self.device)
#         gs3d_empty = gs3d[empty_mask]

#         self._check(
#             indices_or_mask=empty_mask,
#             selected=gs3d_empty,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def test_gaussian_index_selection(self, accumulate_mean_2d_gradients, track_max_2d_radii, empty_shN):

#         # Create indices that select every other Gaussian and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         every_other_idx = torch.arange(0, gs3d.num_gaussians, 2, device=self.device, dtype=torch.long)
#         gs3d_every_other = gs3d[every_other_idx]

#         self._check(
#             indices_or_mask=every_other_idx,
#             selected=gs3d_every_other,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#         # Create indices that select half
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         half_idx = torch.arange(gs3d.num_gaussians, device=self.device, dtype=torch.long)[: gs3d.num_gaussians // 2]
#         gs3d_half = gs3d[half_idx]

#         self._check(
#             indices_or_mask=half_idx,
#             selected=gs3d_half,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#         # Create indices that permutes
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         pmt_idx = torch.randperm(gs3d.num_gaussians, device=self.device)
#         gs3d_pmt = gs3d[pmt_idx]

#         self._check(
#             indices_or_mask=pmt_idx,
#             selected=gs3d_pmt,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#         # Create indices that duplicate the first half of the Gaussians three times
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         half_idx = torch.arange(gs3d.num_gaussians, device=self.device, dtype=torch.long)[: gs3d.num_gaussians // 2]
#         dup_idx = torch.cat([half_idx, half_idx, half_idx], dim=0)
#         gs3d_dup = gs3d[dup_idx]

#         self._check(
#             indices_or_mask=dup_idx,
#             selected=gs3d_dup,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )

#     @parameterized.expand(
#         [
#             [True, True, True],
#             [True, True, False],
#             [True, False, True],
#             [True, False, False],
#             [False, True, True],
#             [False, True, False],
#             [False, False, True],
#             [False, False, False],
#         ]
#     )
#     def test_gaussian_slice_selection(self, accumulate_mean_2d_gradients, track_max_2d_radii, empty_shN):

#         def check_is_view(selected, gtidx):
#             if not accumulate_mean_2d_gradients and not track_max_2d_radii:
#                 selected.means += 10.0
#                 self._check(
#                     indices_or_mask=gtidx,
#                     selected=selected,
#                     dst=gs3d,
#                     accumulate_max_2d_radii=track_max_2d_radii,
#                     accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#                 )

#         # Create indices that select every other Gaussian and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         gt_idx = torch.arange(0, gs3d.num_gaussians, 2, device=self.device, dtype=torch.long)
#         gs_sel = gs3d[::2]

#         self._check(
#             indices_or_mask=gt_idx,
#             selected=gs_sel,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )
#         check_is_view(gs_sel, gt_idx)

#         # Create indices that select every other Gaussian up to half and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         gt_idx = torch.arange(0, gs3d.num_gaussians // 2, 2, device=self.device, dtype=torch.long)
#         gs_sel = gs3d[: gs3d.num_gaussians // 2 : 2]

#         self._check(
#             indices_or_mask=gt_idx,
#             selected=gs_sel,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )
#         check_is_view(gs_sel, gt_idx)

#         # Create indices that select every other Gaussian from 10 up to half and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         gt_idx = torch.arange(10, gs3d.num_gaussians // 2, 2, device=self.device, dtype=torch.long)
#         gs_sel = gs3d[10 : gs3d.num_gaussians // 2 : 2]

#         self._check(
#             indices_or_mask=gt_idx,
#             selected=gs_sel,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )
#         check_is_view(gs_sel, gt_idx)

#         # Create indices that select every other Gaussian up to -7 and use it to select from the Gaussian Splat
#         gs3d = self._make_gs3d(
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             empty_shN=empty_shN,
#         )
#         gt_idx = torch.arange(gs3d.num_gaussians, device=self.device, dtype=torch.long)[:-7]
#         gs_sel = gs3d[:-7]

#         self._check(
#             indices_or_mask=gt_idx,
#             selected=gs_sel,
#             dst=gs3d,
#             accumulate_max_2d_radii=track_max_2d_radii,
#             accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
#         )
#         check_is_view(gs_sel, gt_idx)


# class TestLoadAndSavePly(BaseGaussianTestCase):
#     def setUp(self):
#         super().setUp()

#     def test_load_ply_with_no_shN(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")
#         shN_empty = torch.empty((self.gs3d.num_gaussians, 0, 3), device=self.device)
#         gs3d_no_shN = GaussianSplat3d.from_tensors(
#             means=self.gs3d.means,
#             quats=self.gs3d.quats,
#             log_scales=self.gs3d.log_scales,
#             logit_opacities=self.gs3d.logit_opacities,
#             sh0=self.gs3d.sh0,
#             shN=shN_empty,
#         )
#         gs3d_no_shN.save_ply(tf.name)

#         gs3d_loaded, metadata = GaussianSplat3d.from_ply(tf.name)

#         self.assertTrue(torch.allclose(gs3d_loaded.means, gs3d_no_shN.means))
#         self.assertTrue(torch.allclose(gs3d_loaded.quats, gs3d_no_shN.quats))
#         self.assertTrue(torch.allclose(gs3d_loaded.log_scales, gs3d_no_shN.log_scales))
#         self.assertTrue(torch.allclose(gs3d_loaded.logit_opacities, gs3d_no_shN.logit_opacities))
#         self.assertTrue(torch.allclose(gs3d_loaded.sh0, gs3d_no_shN.sh0))
#         self.assertTrue(gs3d_loaded.shN.shape == (gs3d_no_shN.num_gaussians, 0, 3))

#     def test_save_ply_handles_nan(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         self.nan_gs3d.save_ply(tf.name)

#         # Remove the first element from all tensors to compare with expected loaded ply
#         # since we set it to NaN
#         gs3d_without_nan = self.nan_gs3d[1:]

#         loaded = pcu.load_triangle_mesh(tf.name)
#         attribs = loaded.vertex_data.custom_attributes
#         means_loaded = torch.from_numpy(loaded.vertex_data.positions).to(self.device)
#         self.assertTrue(torch.allclose(means_loaded, gs3d_without_nan.means))

#         scales_loaded = torch.from_numpy(
#             np.stack([attribs["scale_0"], attribs["scale_1"], attribs["scale_2"]], axis=-1)
#         ).to(self.device)
#         self.assertTrue(torch.allclose(scales_loaded, gs3d_without_nan.log_scales))

#         quats_loaded = torch.from_numpy(
#             np.stack(
#                 [
#                     attribs["rot_0"],
#                     attribs["rot_1"],
#                     attribs["rot_2"],
#                     attribs["rot_3"],
#                 ],
#                 axis=-1,
#             )
#         ).to(self.device)
#         self.assertTrue(torch.allclose(quats_loaded, gs3d_without_nan.quats))

#         opacities_loaded = torch.from_numpy(attribs["opacity"]).to(self.device)
#         self.assertTrue(torch.allclose(opacities_loaded, gs3d_without_nan.logit_opacities))

#         sh0_loaded = (
#             torch.from_numpy(np.stack([attribs[f"f_dc_{i}"] for i in range(3)], axis=1)).to(self.device).unsqueeze(1)
#         )
#         self.assertTrue(torch.allclose(sh0_loaded, gs3d_without_nan.sh0))
#         shN_loaded = torch.from_numpy(np.stack([attribs[f"f_rest_{i}"] for i in range(45)], axis=1)).to(self.device)
#         shN_loaded = shN_loaded.view(gs3d_without_nan.num_gaussians, 15, 3)
#         self.assertTrue(torch.allclose(shN_loaded, gs3d_without_nan.shN))

#     def test_save_ply(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         self.gs3d.save_ply(tf.name)

#         loaded = pcu.load_triangle_mesh(tf.name)
#         attribs = loaded.vertex_data.custom_attributes
#         means_loaded = torch.from_numpy(loaded.vertex_data.positions).to(self.device)
#         self.assertTrue(torch.allclose(means_loaded, self.gs3d.means))

#         scales_loaded = torch.from_numpy(
#             np.stack([attribs["scale_0"], attribs["scale_1"], attribs["scale_2"]], axis=-1)
#         ).to(self.device)
#         self.assertTrue(torch.allclose(scales_loaded, self.gs3d.log_scales))

#         quats_loaded = torch.from_numpy(
#             np.stack(
#                 [
#                     attribs["rot_0"],
#                     attribs["rot_1"],
#                     attribs["rot_2"],
#                     attribs["rot_3"],
#                 ],
#                 axis=-1,
#             )
#         ).to(self.device)
#         self.assertTrue(torch.allclose(quats_loaded, self.gs3d.quats))

#         opacities_loaded = torch.from_numpy(attribs["opacity"]).to(self.device)
#         self.assertTrue(torch.allclose(opacities_loaded, self.gs3d.logit_opacities))

#         sh0_loaded = (
#             torch.from_numpy(np.stack([attribs[f"f_dc_{i}"] for i in range(3)], axis=1)).to(self.device).unsqueeze(1)
#         )
#         self.assertTrue(torch.allclose(sh0_loaded, self.gs3d.sh0))

#         shN_loaded = torch.from_numpy(np.stack([attribs[f"f_rest_{i}"] for i in range(45)], axis=1)).to(self.device)
#         shN_loaded = shN_loaded.view(self.gs3d.num_gaussians, 15, 3)
#         self.assertTrue(torch.allclose(shN_loaded, self.gs3d.shN))

#     def test_save_and_load_ply(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         self.gs3d.save_ply(tf.name)

#         gs3d_loaded, metadata = GaussianSplat3d.from_ply(tf.name)

#         self.assertTrue(torch.allclose(gs3d_loaded.means, self.gs3d.means))
#         self.assertTrue(torch.allclose(gs3d_loaded.quats, self.gs3d.quats))
#         self.assertTrue(torch.allclose(gs3d_loaded.log_scales, self.gs3d.log_scales))
#         self.assertTrue(torch.allclose(gs3d_loaded.logit_opacities, self.gs3d.logit_opacities))
#         self.assertTrue(torch.allclose(gs3d_loaded.sh0, self.gs3d.sh0))
#         self.assertTrue(torch.allclose(gs3d_loaded.shN, self.gs3d.shN))

#         self.assertTrue(len(metadata) == 0)

#     def test_save_and_load_ply_with_training_info(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         num_cams = 88
#         normalization_tx = torch.randn(4, 4)
#         cam_to_worlds = torch.randn(num_cams, 4, 4)
#         cam_types = torch.full((num_cams,), 8).to(torch.int32)
#         proj_params = torch.randn(num_cams, 4, 5, 7)

#         metadata_dict = {
#             "normalization_transform": normalization_tx,
#             "camera_to_world_matrices": cam_to_worlds,
#             "projection_types": cam_types,
#             "projection_parameters": proj_params,
#             "string_parameter": "The Quick brown fox jumps over the lazy dog",
#             "int_param": 8198767135,
#             "float_param": 0.121243243523524650345740953,
#         }
#         self.gs3d.save_ply(tf.name, metadata=metadata_dict)

#         gs3d_loaded, training_info = GaussianSplat3d.from_ply(tf.name)

#         self.assertTrue(torch.allclose(gs3d_loaded.means, self.gs3d.means))
#         self.assertTrue(torch.allclose(gs3d_loaded.quats, self.gs3d.quats))
#         self.assertTrue(torch.allclose(gs3d_loaded.log_scales, self.gs3d.log_scales))
#         self.assertTrue(torch.allclose(gs3d_loaded.logit_opacities, self.gs3d.logit_opacities))
#         self.assertTrue(torch.allclose(gs3d_loaded.sh0, self.gs3d.sh0))
#         self.assertTrue(torch.allclose(gs3d_loaded.shN, self.gs3d.shN))

#         assert isinstance(training_info["normalization_transform"], torch.Tensor)
#         assert isinstance(training_info["camera_to_world_matrices"], torch.Tensor)
#         assert isinstance(training_info["projection_types"], torch.Tensor)
#         assert isinstance(training_info["projection_parameters"], torch.Tensor)
#         self.assertTrue(torch.allclose(training_info["normalization_transform"], normalization_tx.to(self.device)))
#         self.assertTrue(torch.allclose(training_info["camera_to_world_matrices"], cam_to_worlds.to(self.device)))
#         self.assertTrue(torch.equal(training_info["projection_types"], cam_types.to(self.device)))
#         self.assertTrue(torch.allclose(training_info["projection_parameters"], proj_params.to(self.device)))
#         self.assertEqual(training_info["float_param"], 0.121243243523524650345740953)
#         self.assertEqual(training_info["int_param"], 8198767135)
#         self.assertEqual(training_info["string_parameter"], "The Quick brown fox jumps over the lazy dog")

#     def test_save_ply_only_string_keys(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         metadata_dict = {"_a_key_key": "foo bar baz", "anotherkey": "qux quux corge"}
#         self.gs3d.save_ply(tf.name, metadata=metadata_dict)

#         gs, meta = GaussianSplat3d.from_ply(tf.name)
#         self.assertEqual(meta["_a_key_key"], "foo bar baz")
#         self.assertEqual(meta["anotherkey"], "qux quux corge")

#     def test_save_ply_only_int_keys(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         metadata_dict = {"_a_key_key": 42, "anotherkey": sys.maxsize}
#         self.gs3d.save_ply(tf.name, metadata=metadata_dict)

#         gs, meta = GaussianSplat3d.from_ply(tf.name)
#         self.assertEqual(meta["_a_key_key"], 42)
#         self.assertEqual(meta["anotherkey"], sys.maxsize)

#     def test_save_ply_invalid_metadata_keys(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         metadata_dict = {
#             "invalid key": torch.randn(4, 4),
#         }
#         with self.assertRaises(ValueError):
#             self.gs3d.save_ply(tf.name, metadata=metadata_dict)

#         metadata_dict = {
#             "invalid@key": torch.randn(4, 4),
#         }
#         with self.assertRaises(ValueError):
#             self.gs3d.save_ply(tf.name, metadata=metadata_dict)

#     def test_save_and_load_ply_with_training_info_non_contiguous(self):
#         tf = tempfile.NamedTemporaryFile(delete=True, suffix=".ply")

#         num_cams = 88
#         normalization_tx = torch.randn(4, 4).T
#         cam_to_worlds = torch.randn(num_cams, 4, 4)[::2]
#         cam_types = torch.full((num_cams,), 8).to(torch.int32)[::2]
#         proj_params = torch.randn(num_cams, 4, 5, 7)[::2]

#         metadata_dict = {
#             "num_cams": 88,
#             "normalization_tx": normalization_tx,
#             "camera_to_world_matrices123": cam_to_worlds,
#             "projection_types": cam_types,
#             "projection_parameters": proj_params,
#             "version_string": "my version",
#         }
#         self.gs3d.save_ply(tf.name, metadata_dict)

#         gs3d_loaded, training_info = GaussianSplat3d.from_ply(tf.name)

#         self.assertTrue(torch.allclose(gs3d_loaded.means, self.gs3d.means))
#         self.assertTrue(torch.allclose(gs3d_loaded.quats, self.gs3d.quats))
#         self.assertTrue(torch.allclose(gs3d_loaded.log_scales, self.gs3d.log_scales))
#         self.assertTrue(torch.allclose(gs3d_loaded.logit_opacities, self.gs3d.logit_opacities))
#         self.assertTrue(torch.allclose(gs3d_loaded.sh0, self.gs3d.sh0))
#         self.assertTrue(torch.allclose(gs3d_loaded.shN, self.gs3d.shN))

#         assert isinstance(training_info["normalization_tx"], torch.Tensor)
#         assert isinstance(training_info["camera_to_world_matrices123"], torch.Tensor)
#         assert isinstance(training_info["projection_types"], torch.Tensor)
#         assert isinstance(training_info["projection_parameters"], torch.Tensor)
#         self.assertTrue(torch.allclose(training_info["normalization_tx"], normalization_tx.to(self.device)))
#         self.assertTrue(torch.allclose(training_info["camera_to_world_matrices123"], cam_to_worlds.to(self.device)))
#         self.assertTrue(torch.equal(training_info["projection_types"], cam_types.to(self.device)))
#         self.assertTrue(torch.allclose(training_info["projection_parameters"], proj_params.to(self.device)))
#         self.assertEqual(training_info["version_string"], "my version")
#         self.assertEqual(training_info["num_cams"], 88)


# class TestGaussianRender(BaseGaussianTestCase):

#     def setUp(self):
#         super().setUp()

#     def test_gaussian_projection(self):
#         proj_res = self.gs3d.project_gaussians_for_images_and_depths(
#             self.cam_to_world_mats,
#             self.projection_mats,
#             self.width,
#             self.height,
#             self.near_plane,
#             self.far_plane,
#         )
#         radii = proj_res.radii
#         means2d = proj_res.means2d
#         depths = proj_res.render_quantities[..., -1]
#         conics = proj_res.inv_covar_2d

#         if self.save_regression_data:
#             torch.save(radii, "regression_radii.pt")
#             torch.save(means2d, "regression_means2d.pt")
#             torch.save(depths, "regression_depths.pt")
#             torch.save(conics, "regression_conics.pt")

#         # Regression test
#         test_radii = torch.load(self.data_path / "regression_radii.pt", weights_only=True)
#         test_means2d = torch.load(self.data_path / "regression_means2d.pt", weights_only=True)
#         test_depths = torch.load(self.data_path / "regression_depths.pt", weights_only=True)
#         test_conics = torch.load(self.data_path / "regression_conics.pt", weights_only=True)

#         torch.testing.assert_close(radii, test_radii)
#         torch.testing.assert_close(means2d[radii > 0], test_means2d[radii > 0])
#         torch.testing.assert_close(depths[radii > 0], test_depths[radii > 0])
#         torch.testing.assert_close(conics[radii > 0], test_conics[radii > 0], atol=1e-5, rtol=1e-4)

#     def _tensors_to_pixel(self, colors, alphas):
#         canvas = (
#             torch.cat(
#                 [
#                     colors.reshape(self.num_cameras * self.height, self.width, 3),
#                     alphas.reshape(self.num_cameras * self.height, self.width, 1).expand(-1, -1, 3),
#                 ],
#                 dim=1,
#             )
#             .detach()
#             .cpu()
#             .numpy()
#         )
#         return (canvas * 255).astype(np.uint8)

#     def test_gaussian_render(self):
#         render_colors, render_alphas = self.gs3d.render_images(
#             self.cam_to_world_mats,
#             self.projection_mats,
#             self.width,
#             self.height,
#             self.near_plane,
#             self.far_plane,
#         )

#         pixels = self._tensors_to_pixel(render_colors, render_alphas)
#         differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_result.png"))

#         if self.save_image_data:
#             import cv2

#             cv2.imwrite(str(self.data_path / "output_gaussian_render.png"), pixels)

#         if self.save_regression_data:
#             import cv2

#             cv2.imwrite("regression_gaussian_render_result.png", pixels)

#         self.assertFalse(
#             differ,
#             f"Gaussian renders for Torch tensors differ from reference image at {cmp.nfail} pixels",
#         )

#     def test_gaussian_render_jagged(self):
#         # There are two scenes
#         jt_means = JaggedTensor([self.gs3d.means, self.gs3d.means]).to(self.device)
#         jt_quats = JaggedTensor([self.gs3d.quats, self.gs3d.quats]).to(self.device)
#         jt_scales = JaggedTensor([self.gs3d.scales, self.gs3d.scales]).to(self.device)
#         jt_opacities = JaggedTensor([self.gs3d.opacities, self.gs3d.opacities]).to(self.device)

#         sh_coeffs = torch.cat([self.gs3d.sh0, self.gs3d.shN], dim=1)  # [N, K, 3]
#         jt_sh_coeffs = JaggedTensor([sh_coeffs, sh_coeffs]).to(self.device)

#         # The first scene renders to 2 views and the second scene renders to a single view
#         jt_viewmats = JaggedTensor([self.cam_to_world_mats[:2], self.cam_to_world_mats[2:]]).to(self.device)
#         jt_Ks = JaggedTensor([self.projection_mats[:2], self.projection_mats[2:]]).to(self.device)

#         # g_sizes = means.joffsets[1:] - means.joffsets[:-1]
#         # c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]
#         # tt = g_sizes.repeat_interleave(c_sizes)
#         # camera_ids = torch.arange(viewmats.rshape[0], device=device).repeat_interleave(tt, dim=0)

#         # dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
#         # dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
#         # shifts = dd0[1:] - dd1[:-1]
#         # shifts = torch.cat([torch.tensor([0], device=device), shifts])  # [0, -1000, 0]
#         # shifts_cumsum = shifts.cumsum(0)  # [0, -1000, -1000]
#         # gaussian_ids = torch.arange(len(camera_ids), device=device)  # [0, 1, 2, ..., 2999]
#         # gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, dim=0)

#         render_colors, render_alphas, _ = gaussian_render_jagged(
#             jt_means,
#             jt_quats,
#             jt_scales,
#             jt_opacities,
#             jt_sh_coeffs,
#             jt_viewmats,
#             jt_Ks,
#             self.width,
#             self.height,
#             self.near_plane,  # near_plane
#             self.far_plane,  # far_plane
#             self.sh_degree,  # sh_degree_to_use
#             16,  # tile_size
#             0.0,  # radius_clip
#             0.3,  # eps2d
#             False,  # antialias
#             False,  # return depth
#             False,  # return debug info
#             False,  # ortho
#         )
#         torch.cuda.synchronize()

#         pixels = self._tensors_to_pixel(render_colors, render_alphas)
#         differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_jagged_result.png"))

#         if self.save_image_data:
#             import cv2

#             cv2.imwrite(str(self.data_path / "output_gaussian_render_jagged.png"), pixels)

#         if self.save_regression_data:
#             import cv2

#             cv2.imwrite("regression_gaussian_render_jagged_result.png", pixels)

#         self.assertFalse(
#             differ,
#             f"Gaussian renders for jagged tensors differ from reference image at {cmp.nfail} pixels",
#         )


class TestTopGaussianContributionsRender(BaseGaussianTestCase):

    def setUp(self):
        super().setUp()

    @staticmethod
    def calculate_expected_alpha(opacity: float, num_layers: int) -> float:
        accumulated_transparency = 1.0
        expected_alpha = 0.0
        for i in range(num_layers):
            expected_alpha += accumulated_transparency * opacity
            accumulated_transparency *= 1.0 - opacity
        return expected_alpha

    def test_gaussians_center_render(self):
        h = 1024
        w = 512

        num_gaussian_layers = 10

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

        gs3d = GaussianSplat3d.from_tensors(means3d, quats, log_scales, logit_opacities, sh0, shN)

        # Test render num contributing gaussians
        num_contributing_gaussians, alphas = gs3d.render_num_contributing_gaussians(
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )
        # test the center pixel should have num_gaussian_layers contributing gaussians
        self.assertTrue(num_contributing_gaussians[0][h // 2 - 1][w // 2 - 1] == num_gaussian_layers)

        expected_alpha = self.calculate_expected_alpha(opacities[0].item(), num_gaussian_layers)
        # test the center pixel should have the expected alpha
        self.assertTrue(alphas[0][h // 2 - 1][w // 2 - 1] == expected_alpha)

        # Test render top contributing gaussian ids
        ids, weights = gs3d.render_contributing_gaussian_ids(
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        middle_pixel_index = (h // 2 - 1) * w + (w // 2 - 1)
        middle_pixel_ids = ids.unbind()[0][middle_pixel_index]

        self.assertTrue(
            torch.equal(
                middle_pixel_ids,
                torch.arange(num_gaussian_layers, device=self.device),
            )
        )

        # Test render weights
        expected_weights = torch.zeros(num_gaussian_layers, device=self.device)
        accumulated_transparency = 1.0
        for i in range(num_gaussian_layers):
            expected_weights[i] = accumulated_transparency * opacities[0]
            accumulated_transparency *= 1.0 - opacities[0]

        middle_pixel_weights = weights.unbind()[0][middle_pixel_index]

        self.assertTrue(torch.allclose(middle_pixel_weights, expected_weights))

        # sparse rendering
        # render the center pixel
        pixels_to_render = JaggedTensor([torch.tensor([[h // 2 - 1, w // 2 - 1]])]).to(self.device)

        # test sparse render num contributing gaussians
        sparse_num_contributing_gaussians, sparse_alphas = gs3d.sparse_render_num_contributing_gaussians(
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        # test the center pixel should have the correct number of contributing gaussians
        self.assertTrue(
            torch.equal(
                sparse_num_contributing_gaussians.unbind()[0][0], num_contributing_gaussians[0][h // 2 - 1][w // 2 - 1]
            )
        )

        # test the center pixel should have the correct alpha
        self.assertTrue(torch.equal(sparse_alphas.unbind()[0][0], alphas[0][h // 2 - 1][w // 2 - 1]))

        # test sparse render top contributing gaussian ids
        sparse_ids, sparse_weights = gs3d.sparse_render_contributing_gaussian_ids(
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        self.assertTrue(torch.equal(sparse_ids.unbind()[0][0], middle_pixel_ids))

        self.assertTrue(torch.equal(sparse_weights.unbind()[0][0], middle_pixel_weights))

    def test_gaussians_grid_render(self):
        h = 1024
        w = 512

        num_gaussian_layers = 12

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

        gs3d = GaussianSplat3d.from_tensors(means3d, quats, log_scales, logit_opacities, sh0, shN)

        # Test render num contributing gaussians
        num_contributing_gaussians, alphas = gs3d.render_num_contributing_gaussians(
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        # pixelsdirectly under the centers of the gaussians should have the correct number of contributing gaussians
        num_gaussians_centers = num_contributing_gaussians[0][::2, ::2]
        alphas_centers = alphas[0][::2, ::2]

        expected_num_gaussians_centers = torch.full(
            (num_gaussians_centers.shape[0], num_gaussians_centers.shape[1]), num_gaussian_layers, device=self.device
        )
        self.assertTrue(torch.equal(num_gaussians_centers, expected_num_gaussians_centers))

        # pixels directly under the centers of the gaussians should have the correct alpha
        expected_alpha = self.calculate_expected_alpha(opacities[0].item(), num_gaussian_layers)
        expected_alphas_centers = torch.full(
            (alphas_centers.shape[0], alphas_centers.shape[1]), expected_alpha, device=self.device
        )
        self.assertTrue(torch.allclose(alphas_centers, expected_alphas_centers, atol=1e-5, rtol=1e-8))

        # Test render contributing gaussian ids
        ids, weights = gs3d.render_contributing_gaussian_ids(
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        # Calculate expected weights for all layers
        expected_weights = torch.zeros(num_gaussian_layers, device=self.device, dtype=torch.float32)
        accumulated_transparency = torch.ones(1, device=self.device, dtype=torch.float32)
        for i in range(num_gaussian_layers):
            expected_weights[i] = accumulated_transparency * opacities[0]
            accumulated_transparency *= 1.0 - opacities[0]

        # Test center pixels (every other pixel starting from 0)
        # Get the ids and weights from the JaggedTensor for camera 0
        camera_0_ids = ids.unbind()[0]
        camera_0_weights = weights.unbind()[0]

        # Check pixels at the center of each gaussian (every 2 pixels)

        # Generate all y and x coordinates
        y_coords = torch.arange(0, h, 2, device=self.device)  # [h//2]
        x_coords = torch.arange(0, w, 2, device=self.device)  # [w//2]

        # Create meshgrid to get all (y, x) combinations
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")  # Both [h//2, w//2]

        # Flatten to get all pixel coordinates
        y_flat = y_grid.flatten()  # [h//2 * w//2]
        x_flat = x_grid.flatten()  # [h//2 * w//2]

        # Calculate all pixel indices
        pixel_indices = y_flat * w + x_flat  # [h//2 * w//2]

        # Stack all pixel IDs and weights
        pixel_indices_int = pixel_indices.cpu().numpy().astype(int)
        all_pixel_ids = torch.stack(
            [camera_0_ids[idx] for idx in pixel_indices_int]
        )  # [h//2 * w//2, num_gaussian_layers]
        all_pixel_weights = torch.stack(
            [camera_0_weights[idx] for idx in pixel_indices_int]
        )  # [h//2 * w//2, num_gaussian_layers]

        # Calculate all expected IDs
        # Shape: [h//2 * w//2, num_gaussian_layers]
        gaussians_per_layer = means3d.shape[0] // num_gaussian_layers
        layer_indices = torch.arange(num_gaussian_layers, device=self.device)  # [num_gaussian_layers]
        layer_offsets = layer_indices * gaussians_per_layer  # [num_gaussian_layers]

        grid_y = y_flat // 2  # [h//2 * w//2]
        grid_x = x_flat // 2  # [h//2 * w//2]
        grid_indices = grid_y * (w // 2) + grid_x  # [h//2 * w//2]

        # expected IDs for all pixels and all layers
        expected_ids = layer_offsets.unsqueeze(0) + grid_indices.unsqueeze(1)  # [h//2 * w//2, num_gaussian_layers]

        # Check all IDs
        self.assertTrue(torch.equal(all_pixel_ids, expected_ids))

        # Check all weights - compare against broadcasted expected_weights
        expected_weights_broadcasted = expected_weights.unsqueeze(0).expand(all_pixel_weights.shape[0], -1)
        self.assertTrue(torch.allclose(all_pixel_weights, expected_weights_broadcasted, atol=1e-5, rtol=1e-8))

        ##########################################################
        ## Sparse Rendering- use pixels that we know have gaussians (center pixels)
        ##########################################################
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, w, (num_pixels_to_render,))
        yCoords = torch.randint(0, h, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        test_pixels = torch.stack([yCoords, xCoords], 1)

        pixels_to_render = JaggedTensor([test_pixels]).to(self.device)

        # test sparse render num contributing gaussians
        sparse_num_contributing_gaussians, sparse_alphas = gs3d.sparse_render_num_contributing_gaussians(
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        for pixels, sparse_num_contributing_gaussians, reference_num_contributing_gaussians in zip(
            pixels_to_render.unbind(), sparse_num_contributing_gaussians.unbind(), num_contributing_gaussians
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_num_contributing_gaussians, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_num_contributing_gaussians using the coordinates
            selected_reference_num_contributing_gaussians = reference_num_contributing_gaussians[y_coords, x_coords]
            self.assertTrue(
                torch.equal(sparse_num_contributing_gaussians, selected_reference_num_contributing_gaussians)
            )

        for pixels, sparse_alphas, reference_alphas in zip(pixels_to_render.unbind(), sparse_alphas.unbind(), alphas):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_alphas, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_alphas using the coordinates
            selected_reference_alphas = reference_alphas[y_coords, x_coords]
            self.assertTrue(torch.equal(sparse_alphas, selected_reference_alphas))

        # test sparse render contributing gaussian ids
        sparse_ids, sparse_weights = gs3d.sparse_render_contributing_gaussian_ids(
            pixels_to_render,
            world_to_cam_xform.unsqueeze(0).contiguous(),
            intrinsics.unsqueeze(0).contiguous(),
            w,
            h,
            0.1,
            10000.0,
        )

        # Compare sparse results with dense results
        for pixels, sparse_camera_ids, sparse_camera_weights, reference_camera_ids, reference_camera_weights in zip(
            pixels_to_render.unbind(), sparse_ids.unbind(), sparse_weights.unbind(), ids.unbind(), weights.unbind()
        ):
            assert isinstance(pixels, torch.Tensor)

            # For each pixel in the sparse render
            for i, pixel_coord in enumerate(pixels):
                y = int(pixel_coord[0].item())
                x = int(pixel_coord[1].item())
                pixel_index = y * w + x

                # Get the ids and weights for this pixel from both sparse and dense renders
                sparse_pixel_ids = sparse_camera_ids[i]
                sparse_pixel_weights = sparse_camera_weights[i]
                reference_pixel_ids = reference_camera_ids[pixel_index]
                reference_pixel_weights = reference_camera_weights[pixel_index]

                # Compare
                self.assertTrue(torch.equal(sparse_pixel_ids, reference_pixel_ids))
                self.assertTrue(torch.equal(sparse_pixel_weights, reference_pixel_weights))

    def test_gaussian_contributors_scene_render(self):
        # Test render num contributing gaussians
        num_contributing_gaussians, alphas = self.gs3d.render_num_contributing_gaussians(
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )
        prev_num_contributing_gaussians = num_contributing_gaussians
        for _ in range(50):
            num_contributing_gaussians, alphas = self.gs3d.render_num_contributing_gaussians(
                self.cam_to_world_mats,
                self.projection_mats,
                self.width,
                self.height,
                0.01,
                10000.0,
            )
            self.assertTrue(torch.equal(num_contributing_gaussians, prev_num_contributing_gaussians))
            prev_num_contributing_gaussians = num_contributing_gaussians

        if self.save_regression_data:
            torch.save(num_contributing_gaussians, self.data_path / "regression_num_contributing_gaussians.pt")
            torch.save(alphas, self.data_path / "regression_num_contributing_gaussians_alphas.pt")

        # load the regression data
        num_contributing_gaussians_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians.pt", weights_only=True
        )
        alphas_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians_alphas.pt", weights_only=True
        )

        self.assertTrue(torch.equal(num_contributing_gaussians, num_contributing_gaussians_regression))
        self.assertTrue(torch.equal(alphas, alphas_regression))

        # Test render top contributing gaussian ids
        ids, weights = self.gs3d.render_contributing_gaussian_ids(
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        if self.save_regression_data:
            torch.save(ids, self.data_path / "regression_top_contributors_ids.pt")
            torch.save(weights, self.data_path / "regression_top_contributors_weights.pt")

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=False)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=False)

        self.assertTrue(ids == ids_regression)
        self.assertTrue(weights == weights_regression)

    def test_gaussian_contributors_scene_sparse_render(self):
        # sparse rendering
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, self.width, (num_pixels_to_render,))
        yCoords = torch.randint(0, self.height, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        test_pixels = torch.stack([yCoords, xCoords], 1)

        pixels_to_render = JaggedTensor([test_pixels]).to(self.device)

        # Test render num contributing gaussians
        num_contributing_gaussians, num_contributing_gaussians_alphas = (
            self.gs3d.sparse_render_num_contributing_gaussians(
                pixels_to_render,
                self.cam_to_world_mats,
                self.projection_mats,
                self.width,
                self.height,
                0.01,
                10000.0,
            )
        )
        prev_num_contributing_gaussians = num_contributing_gaussians
        for _ in range(50):
            num_contributing_gaussians, num_contributing_gaussians_alphas = (
                self.gs3d.sparse_render_num_contributing_gaussians(
                    pixels_to_render,
                    self.cam_to_world_mats,
                    self.projection_mats,
                    self.width,
                    self.height,
                    0.01,
                    10000.0,
                )
            )
            self.assertTrue(torch.equal(num_contributing_gaussians.jdata, prev_num_contributing_gaussians.jdata))
            prev_num_contributing_gaussians = num_contributing_gaussians

        # load the regression data
        num_contributing_gaussians_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians.pt", weights_only=True
        )
        alphas_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians_alphas.pt", weights_only=True
        )

        for pixels, sparse_num_contributing_gaussians, reference_num_contributing_gaussians in zip(
            pixels_to_render.unbind(), num_contributing_gaussians.unbind(), num_contributing_gaussians_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_num_contributing_gaussians, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_num_contributing_gaussians using the coordinates
            selected_reference_num_contributing_gaussians = reference_num_contributing_gaussians[y_coords, x_coords]
            self.assertTrue(
                torch.equal(sparse_num_contributing_gaussians, selected_reference_num_contributing_gaussians)
            )

        for pixels, sparse_alphas, reference_alphas in zip(
            pixels_to_render.unbind(), num_contributing_gaussians_alphas.unbind(), alphas_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_alphas, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_alphas using the coordinates
            selected_reference_alphas = reference_alphas[y_coords, x_coords]
            self.assertTrue(torch.equal(sparse_alphas, selected_reference_alphas))

        # Test render top contributing gaussian ids
        sparse_ids, sparse_weights = self.gs3d.sparse_render_contributing_gaussian_ids(
            pixels_to_render,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=False)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=False)

        for pixels, image_sparse_ids, image_reference_ids in zip(pixels_to_render.unbind(), sparse_ids, ids_regression):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(image_sparse_ids, JaggedTensor)
            assert isinstance(image_reference_ids, JaggedTensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            pixel_indices = (y_coords * self.width + x_coords).long()  # [num_pixels_to_render]

            # Select tensors from reference_ids at the specified pixel positions
            reference_ids_list = image_reference_ids.unbind()
            selected_tensors = [reference_ids_list[idx.item()] for idx in pixel_indices]
            selected_reference_ids = JaggedTensor(selected_tensors)

            self.assertTrue(image_sparse_ids == selected_reference_ids)

        # check weights
        for pixels, image_sparse_weights, image_reference_weights in zip(
            pixels_to_render.unbind(), sparse_weights, weights_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(image_sparse_weights, JaggedTensor)
            assert isinstance(image_reference_weights, JaggedTensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            pixel_indices = (y_coords * self.width + x_coords).long()  # [num_pixels_to_render]

            # Select tensors from reference_weights at the specified pixel positions
            reference_weights_list = image_reference_weights.unbind()
            selected_tensors = [reference_weights_list[idx.item()] for idx in pixel_indices]
            selected_reference_weights = JaggedTensor(selected_tensors)

            self.assertTrue(image_sparse_weights == selected_reference_weights)

    def test_gaussian_contributors_scene_dense_pixels_sparse_render(self):
        # Test that the sparse render works with dense pixel specification
        # Taking a [C, R, 2] tensor as pixels_to_render and returning Tensors [C, R, num_samples]

        # sparse rendering - use pixels that we know have gaussians (center pixels)
        num_pixels_to_render = 100

        # Generate random pixel coordinates within image bounds
        xCoords = torch.randint(0, self.width, (num_pixels_to_render,))
        yCoords = torch.randint(0, self.height, (num_pixels_to_render,))

        # Stack x and y coordinates to form 2D pixel coordinates
        pixels_to_render = torch.stack([yCoords, xCoords], 1).unsqueeze(0).to(self.device)

        # test sparse render num contributing gaussians
        sparse_num_contributing_gaussians, sparse_alphas = self.gs3d.sparse_render_num_contributing_gaussians(
            pixels_to_render,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        # load the regression data
        num_contributing_gaussians_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians.pt", weights_only=True
        )
        alphas_regression = torch.load(
            self.data_path / "regression_num_contributing_gaussians_alphas.pt", weights_only=True
        )

        for pixels, sparse_num_contributing_gaussians, reference_num_contributing_gaussians in zip(
            pixels_to_render.unbind(), sparse_num_contributing_gaussians.unbind(), num_contributing_gaussians_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_num_contributing_gaussians, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_num_contributing_gaussians using the coordinates
            selected_reference_num_contributing_gaussians = reference_num_contributing_gaussians[y_coords, x_coords]
            self.assertTrue(
                torch.equal(sparse_num_contributing_gaussians, selected_reference_num_contributing_gaussians)
            )

        for pixels, sparse_alphas, reference_alphas in zip(
            pixels_to_render.unbind(), sparse_alphas.unbind(), alphas_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(sparse_alphas, torch.Tensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            # Index reference_alphas using the coordinates
            selected_reference_alphas = reference_alphas[y_coords, x_coords]
            self.assertTrue(torch.equal(sparse_alphas, selected_reference_alphas))

        # test sparse render top contributing gaussian ids
        sparse_ids, sparse_weights = self.gs3d.sparse_render_contributing_gaussian_ids(
            pixels_to_render,
            self.cam_to_world_mats,
            self.projection_mats,
            self.width,
            self.height,
            0.01,
            10000.0,
        )

        # load the regression data
        ids_regression = torch.load(self.data_path / "regression_top_contributors_ids.pt", weights_only=False)
        weights_regression = torch.load(self.data_path / "regression_top_contributors_weights.pt", weights_only=False)

        for pixels, image_sparse_ids, image_reference_ids in zip(pixels_to_render, sparse_ids, ids_regression):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(image_sparse_ids, JaggedTensor)
            assert isinstance(image_reference_ids, JaggedTensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            pixel_indices = (y_coords * self.width + x_coords).long()  # [num_pixels_to_render]

            # Select tensors from reference_ids at the specified pixel positions
            reference_ids_list = image_reference_ids.unbind()
            selected_tensors = [reference_ids_list[idx.item()] for idx in pixel_indices]
            selected_reference_ids = JaggedTensor(selected_tensors)

            self.assertTrue(image_sparse_ids == selected_reference_ids)

        # check weights
        for pixels, image_sparse_weights, image_reference_weights in zip(
            pixels_to_render, sparse_weights, weights_regression
        ):
            assert isinstance(pixels, torch.Tensor)
            assert isinstance(image_sparse_weights, JaggedTensor)
            assert isinstance(image_reference_weights, JaggedTensor)
            y_coords = pixels[:, 0]  # [num_pixels_to_render]
            x_coords = pixels[:, 1]  # [num_pixels_to_render]
            pixel_indices = (y_coords * self.width + x_coords).long()  # [num_pixels_to_render]

            # Select tensors from reference_weights at the specified pixel positions
            reference_weights_list = image_reference_weights.unbind()
            selected_tensors = [reference_weights_list[idx.item()] for idx in pixel_indices]
            selected_reference_weights = JaggedTensor(selected_tensors)

            self.assertTrue(image_sparse_weights == selected_reference_weights)


if __name__ == "__main__":
    unittest.main()
