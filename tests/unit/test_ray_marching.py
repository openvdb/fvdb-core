# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import unittest

import numpy as np
import torch
from fvdb.utils.tests import (
    dtype_to_atol,
    get_fvdb_test_data_path,
    make_dense_grid_batch_and_jagged_point_data,
)
from parameterized import parameterized, parameterized_class

import fvdb
from fvdb import GridBatch, JaggedTensor, volume_render

all_device_combos = [
    ["cpu", True],
    ["cuda", True],
    ["cpu", False],
    ["cuda", False],
]

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


@parameterized_class(("device", "dtype"), all_device_dtype_combos)
class TestBatchedRayMarchingWithMisses(unittest.TestCase):
    def setUp(self):
        # Typesystem hack to work with parameterized so we don't get red squiggles.
        # These are defined in the @parameterized_class decorator
        self.device = self.device
        self.dtype = self.dtype
        self.grid = GridBatch.from_dense(
            num_grids=2, dense_dims=[32, 32, 32], device=self.device, voxel_sizes=[0.1, 0.1, 0.1], origins=[0, 0, 0]
        )

        ray_o = torch.tensor([[100, 0, 0]]).to(self.device).to(self.dtype)
        ray_d_hit = torch.tensor([[-1, 0, 0]]).to(self.device).to(self.dtype)  # towards the grid
        ray_d_nohit = torch.tensor([[1, 0, 0]]).to(self.device).to(self.dtype)  # away from the grid

        ray_o = ray_o.repeat(10, 1)  # shape [10, 3]
        ray_d_interleave = torch.cat([ray_d_hit, ray_d_nohit], dim=0).repeat(5, 1)  # shape [10, 3]
        ray_d_alternate = torch.cat([ray_d_hit.repeat(5, 1), ray_d_nohit.repeat(5, 1)], dim=0)  # shape [10, 3]

        self.ray_o = fvdb.JaggedTensor([ray_o] * self.grid.grid_count)
        self.ray_d_interleave = fvdb.JaggedTensor([ray_d_interleave] * self.grid.grid_count)
        self.ray_d_alternate = fvdb.JaggedTensor([ray_d_alternate] * self.grid.grid_count)

    def test_rays_intersect_voxels(self):
        did_hit = self.grid.rays_intersect_voxels(self.ray_o, self.ray_d_interleave, eps=1e-3)
        expected_hits = JaggedTensor(
            [torch.Tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).to(torch.bool).to(self.device)] * self.grid.grid_count
        )
        self.assertTrue(torch.all(expected_hits.jdata == did_hit.jdata))
        self.assertTrue(torch.all(expected_hits.jidx == did_hit.jidx))
        self.assertTrue(torch.all(expected_hits.joffsets == did_hit.joffsets))
        self.assertTrue(torch.all(expected_hits.jlidx == did_hit.jlidx))

        did_hit = self.grid.rays_intersect_voxels(self.ray_o, self.ray_d_alternate, eps=1e-3)
        expected_hits = JaggedTensor(
            [torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).to(torch.bool).to(self.device)] * self.grid.grid_count
        )
        self.assertTrue(torch.all(expected_hits.jdata == did_hit.jdata))
        self.assertTrue(torch.all(expected_hits.jidx == did_hit.jidx))
        self.assertTrue(torch.all(expected_hits.joffsets == did_hit.joffsets))
        self.assertTrue(torch.all(expected_hits.jlidx == did_hit.jlidx))

    def test_voxels_along_rays(self):
        voxels, times = self.grid.voxels_along_rays(self.ray_o, self.ray_d_interleave, 1, eps=1e-3)
        target_lshape = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]] * self.grid.grid_count
        self.assertEqual(len(voxels), self.grid.grid_count)
        self.assertEqual(len(times), self.grid.grid_count)
        self.assertEqual(len(voxels.lshape), len(target_lshape))
        self.assertEqual(len(times.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = voxels.lshape[i]
            vls = times.lshape[i]
            assert isinstance(sls, list)
            assert isinstance(vls, list)
            self.assertEqual(len(sls), len(tls))
            self.assertEqual(len(sls), len(vls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)
                self.assertEqual(vls[j], tlsj)

        voxels, times = self.grid.voxels_along_rays(self.ray_o, self.ray_d_alternate, 1, eps=1e-3)
        target_lshape = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * self.grid.grid_count
        self.assertEqual(len(voxels), self.grid.grid_count)
        self.assertEqual(len(times), self.grid.grid_count)
        self.assertEqual(len(voxels.lshape), len(target_lshape))
        self.assertEqual(len(times.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = voxels.lshape[i]
            vls = times.lshape[i]
            assert isinstance(sls, list)
            assert isinstance(vls, list)
            self.assertEqual(len(sls), len(tls))
            self.assertEqual(len(sls), len(vls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)
                self.assertEqual(vls[j], tlsj)

    def test_segments_along_rays(self):
        segment = self.grid.segments_along_rays(self.ray_o, self.ray_d_interleave, 1, eps=1e-3)
        target_lshape = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]] * self.grid.grid_count
        self.assertEqual(len(segment), self.grid.grid_count)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

        segment = self.grid.segments_along_rays(self.ray_o, self.ray_d_alternate, 1, eps=1e-3)
        target_lshape = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * self.grid.grid_count
        self.assertEqual(len(segment), self.grid.grid_count)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

    def test_uniform_ray_samples(self):
        t_min = fvdb.JaggedTensor(
            [torch.zeros(self.ray_o[i].rshape[0]).to(self.ray_o.jdata) for i in range(len(self.ray_o))]
        )
        t_max = fvdb.JaggedTensor(
            [torch.ones(self.ray_o[i].rshape[0]).to(self.ray_o.jdata) * 1e10 for i in range(len(self.ray_o))]
        )

        segment = self.grid.uniform_ray_samples(
            self.ray_o,
            self.ray_d_interleave,
            t_min,
            t_max,
            0.5,
            eps=1e-3,
        )
        target_lshape = (
            [[8, 0, 8, 0, 8, 0, 8, 0, 8, 0]] * self.grid.grid_count
            if self.dtype != torch.float16
            else [[7, 0, 7, 0, 7, 0, 7, 0, 7, 0]] * self.grid.grid_count
        )
        self.assertEqual(len(segment), self.grid.grid_count)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

        segment = self.grid.uniform_ray_samples(
            self.ray_o,
            self.ray_d_alternate,
            t_min,
            t_max,
            0.5,
            eps=1e-3,
        )
        target_lshape = (
            [[8, 8, 8, 8, 8, 0, 0, 0, 0, 0]] * self.grid.grid_count
            if self.dtype != torch.float16
            else [[7, 7, 7, 7, 7, 0, 0, 0, 0, 0]] * self.grid.grid_count
        )
        self.assertEqual(len(segment), self.grid.grid_count)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)


@parameterized_class(("device", "dtype"), all_device_dtype_combos)
class TestSingleGridRayMarchingWithMisses(unittest.TestCase):
    def setUp(self):
        # Typesystem hack to work with parameterized so we don't get red squiggles.
        # These are defined in the @parameterized_class decorator
        self.device = self.device
        self.dtype = self.dtype
        self.grid = GridBatch.from_dense(
            num_grids=1, dense_dims=[32, 32, 32], device=self.device, voxel_sizes=[0.1, 0.1, 0.1], origins=[0, 0, 0]
        )

        ray_o = torch.tensor([[100, 0, 0]]).to(self.device).to(self.dtype)
        ray_d_hit = torch.tensor([[-1, 0, 0]]).to(self.device).to(self.dtype)  # towards the grid
        ray_d_nohit = torch.tensor([[1, 0, 0]]).to(self.device).to(self.dtype)  # away from the grid

        ray_o_rep = ray_o.repeat(10, 1)  # shape [10, 3]
        ray_d_interleave = torch.cat([ray_d_hit, ray_d_nohit], dim=0).repeat(5, 1)  # shape [10, 3]
        ray_d_alternate = torch.cat([ray_d_hit.repeat(5, 1), ray_d_nohit.repeat(5, 1)], dim=0)  # shape [10, 3]

        self.ray_o = fvdb.JaggedTensor([ray_o_rep])
        self.ray_d_interleave = fvdb.JaggedTensor([ray_d_interleave])
        self.ray_d_alternate = fvdb.JaggedTensor([ray_d_alternate])

    def test_rays_intersect_voxels(self):
        did_hit = self.grid.rays_intersect_voxels(self.ray_o, self.ray_d_interleave, eps=1e-3)
        expected_hits = torch.Tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).to(torch.bool).to(self.device)
        self.assertTrue(torch.all(expected_hits == did_hit.jdata))

        did_hit = self.grid.rays_intersect_voxels(self.ray_o, self.ray_d_alternate, eps=1e-3)
        expected_hits = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).to(torch.bool).to(self.device)
        self.assertTrue(torch.all(expected_hits == did_hit.jdata))

    def test_voxels_along_rays(self):
        voxels, times = self.grid.voxels_along_rays(self.ray_o, self.ray_d_interleave, 1, eps=1e-3)
        target_lshape = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sls = voxels.lshape[0]
        vls = times.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)
            self.assertEqual(vls[i], expected_isect)

        voxels, times = self.grid.voxels_along_rays(self.ray_o, self.ray_d_alternate, 1, eps=1e-3)
        target_lshape = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        sls = voxels.lshape[0]
        vls = times.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)
            self.assertEqual(vls[i], expected_isect)

    def test_segments_along_rays(self):
        segments = self.grid.segments_along_rays(self.ray_o, self.ray_d_interleave, 1, eps=1e-3)
        target_lshape = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sls = segments.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)

        segments = self.grid.segments_along_rays(self.ray_o, self.ray_d_alternate, 1, eps=1e-3)
        target_lshape = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        sls = segments.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)

    def test_uniform_ray_samples(self):
        t_min = fvdb.JaggedTensor([torch.zeros(self.ray_o[0].rshape[0]).to(self.ray_o.jdata)])
        t_max = fvdb.JaggedTensor([torch.ones(self.ray_o[0].rshape[0]).to(self.ray_o.jdata) * 1e10])

        samples = self.grid.uniform_ray_samples(
            self.ray_o,
            self.ray_d_interleave,
            t_min,
            t_max,
            0.5,
            eps=1e-3,
        )
        target_lshape = (
            [8, 0, 8, 0, 8, 0, 8, 0, 8, 0] if self.dtype != torch.float16 else [7, 0, 7, 0, 7, 0, 7, 0, 7, 0]
        )

        sls = samples.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)

        samples = self.grid.uniform_ray_samples(
            self.ray_o,
            self.ray_d_alternate,
            t_min,
            t_max,
            0.5,
            eps=1e-3,
        )
        target_lshape = (
            [8, 8, 8, 8, 8, 0, 0, 0, 0, 0] if self.dtype != torch.float16 else [7, 7, 7, 7, 7, 0, 0, 0, 0, 0]
        )

        sls = samples.lshape[0]
        for i, expected_isect in enumerate(target_lshape):
            self.assertEqual(sls[i], expected_isect)


@parameterized_class(("device", "dtype"), all_device_dtype_combos)
class TestVolumeRender(unittest.TestCase):
    def setUp(self):
        self.device = self.device
        self.dtype = self.dtype

        vox_size = np.random.rand(3) * 0.2 + 0.05
        self.step_size = 0.1 * float(np.linalg.norm(vox_size))
        vox_origin = torch.rand(3).to(self.device).to(self.dtype)

        pts = torch.rand(10000, 3).to(device=self.device, dtype=self.dtype) - 0.5
        grid = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=vox_size, origins=vox_origin)
        self.grid = grid.dilated_grid(1)
        self.grid_dual = grid.dual_grid()

    def make_ray_grid(self, origin, nrays, minb=(-0.45, -0.45), maxb=(0.45, 0.45)):
        """
        Make a raster grid of size nrays*nrays of rays starting from origin
        """
        ray_o = torch.tensor([origin] * nrays**2)  # + p.mean(0, keepdim=True)
        ray_d = torch.from_numpy(
            np.stack(
                [a.ravel() for a in np.mgrid[minb[0] : maxb[0] : nrays * 1j, minb[1] : maxb[1] : nrays * 1j]]
                + [np.ones(nrays**2)],
                axis=-1,
            ).astype(np.float32)
        )
        ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

        ray_o, ray_d = ray_o.to(self.device).to(self.dtype), ray_d.to(self.device).to(self.dtype)

        return ray_o, ray_d

    def volume_render_pytorch(self, sigma, color, dt, t, pack_info, t_threshold):
        """
        Pure pytorch volume rendering to use as ground truth
        """
        res_c = []
        res_d = []
        for ray_i in range(pack_info.shape[0] - 1):
            start_idx, end_idx = pack_info[ray_i], pack_info[ray_i + 1]
            sigma_i = sigma[start_idx:end_idx]
            color_i = color[start_idx:end_idx]
            dt_i = dt[start_idx:end_idx]
            t_i = t[start_idx:end_idx]

            alpha = -sigma_i.squeeze() * dt_i
            transmittance = torch.exp(torch.cumsum(alpha.squeeze(), dim=0))
            tmask = (transmittance > t_threshold).to(transmittance)

            summand = transmittance * (1.0 - torch.exp(alpha)) * tmask
            res_c.append((summand[:, None] * color_i).sum(0))
            res_d.append((summand * t_i).sum(0))
        return torch.stack(res_c), torch.stack(res_d)

    def test_uniform_ray_samples_return_mid_returns_midpoint(self):
        ray_o, ray_d = self.make_ray_grid((0.0, 0.0, -1.0), 8)
        tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
        tmax = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10
        ray_intervals = self.grid.uniform_ray_samples(
            JaggedTensor(ray_o), JaggedTensor(ray_d), JaggedTensor(tmin), JaggedTensor(tmax), self.step_size
        )

        ray_t = ray_intervals.jdata.mean(1)  # Midpoint between each pair of samples
        ray_mids = self.grid.uniform_ray_samples(
            JaggedTensor(ray_o),
            JaggedTensor(ray_d),
            JaggedTensor(tmin),
            JaggedTensor(tmax),
            self.step_size,
            return_midpoints=True,
        ).jdata
        self.assertTrue(torch.allclose(ray_mids, ray_t, atol=dtype_to_atol(self.dtype)))

    def test_volume_render(self):
        t_threshold = 0.001

        # Generate colors and opacities at the corners of voxels of the grid. We'll sample these when we volume render
        grid_data_rgb = torch.rand(self.grid_dual.total_voxels, 3).to(device=self.device, dtype=self.dtype) * 0.5
        grid_data_sigma = torch.rand(self.grid_dual.total_voxels, 1).to(device=self.device, dtype=self.dtype) * 0.5

        grid_data_rgb.requires_grad = True
        grid_data_sigma.requires_grad = True

        ray_o, ray_d = self.make_ray_grid((0.0, 0.0, -1.0), 8)
        tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
        tmax = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10
        ray_intervals = self.grid.uniform_ray_samples(
            JaggedTensor(ray_o), JaggedTensor(ray_d), JaggedTensor(tmin), JaggedTensor(tmax), self.step_size
        )  # [B, -1, 2]

        # FIXME: Francis -- The volume render API should take JaggedTensors as input
        # but for now we have to pass Joffsets explicitly.
        ray_idx = ray_intervals.jidx.int()
        ray_joffsets = ray_intervals.joffsets
        ray_intervals = ray_intervals.jdata

        ray_interval_midpoints = ray_intervals.mean(1)  # Midpoint between each pair of samples shape

        # Generate actual 3D points for the ray samples which we'll sample the grid with
        ray_delta_t = ray_intervals[:, 1] - ray_intervals[:, 0]
        ray_pts = ray_o[ray_idx] + ray_interval_midpoints[:, None] * ray_d[ray_idx]

        # Sample RGB and opacity from the grid corners
        rgb_samples = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_rgb)).jdata
        sigma_samples = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_sigma)).jdata

        assert isinstance(sigma_samples, torch.Tensor)  # Fix type errors

        # Volume render the sampled values and backprop
        rgb1, depth1, opacity, ws, tot_samples = volume_render(
            sigma_samples.squeeze(), rgb_samples, ray_delta_t, ray_interval_midpoints, ray_joffsets, t_threshold
        )
        loss = rgb1.sum() + depth1.sum()
        loss.backward()

        assert grid_data_rgb.grad is not None  # Removes type errors with .grad
        assert grid_data_sigma.grad is not None  # Removes type errors with .grad

        rgb_1_grad = grid_data_rgb.grad.detach().clone()
        sigma_1_grad = grid_data_sigma.grad.detach().clone()

        grid_data_rgb.grad.zero_()
        grid_data_sigma.grad.zero_()

        rgb_samples = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_rgb)).jdata
        sigma_samples = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_sigma)).jdata
        rgb2, depth2 = self.volume_render_pytorch(
            sigma_samples, rgb_samples, ray_delta_t, ray_interval_midpoints, ray_joffsets, t_threshold
        )
        loss = rgb2.sum() + depth2.sum()
        loss.backward()
        rgb_2_grad = grid_data_rgb.grad.detach().clone()
        sigma_2_grad = grid_data_sigma.grad.detach().clone()

        # The tolerances here are high because I think the algo has a large condition number
        # due to sums
        self.assertLess(torch.abs(rgb1 - rgb2).max().item(), 1e-2)
        self.assertLess(torch.abs(depth1 - depth2).max().item(), 2e-1)
        self.assertLess(torch.abs(rgb_1_grad - rgb_2_grad).max().item(), 4e-2)
        self.assertLess(torch.abs(sigma_1_grad - sigma_2_grad).max().item(), 1e-1)

        # import polyscope as ps
        # ps.init()
        # rv = torch.cat([ray_o, ray_o + ray_d])
        # re = torch.tensor([[i, i + ray_o.shape[0]] for i in range(ray_o.shape[0])]).to(ray_o)
        # v, e = grid.grid_edge_network()
        # ps.register_curve_network("grid", v.cpu(), e.cpu())
        # rg = ps.register_curve_network("rays", rv.cpu(), re.cpu())
        # rg.add_color_quantity("colors", rgb1.detach().cpu(), defined_on='edges', enabled=True)
        # pc = ps.register_point_cloud("raypts", ray_pts.cpu())
        # pc.add_scalar_quantity("sigma samples", sigma_samples.detach().squeeze().cpu(), enabled=True)
        # ps.show()

    def test_volume_render_nchannel(self):
        """
        volume_render must handle an arbitrary number of channels, not just 3.
        Run a 4-channel composite and cross-check it against two independent
        2-channel runs on the same sample topology. Also verifies the backward
        pass produces consistent gradients across widths.
        """
        if self.dtype == torch.float16:
            self.skipTest("half precision accumulates too much error for the N-channel round trip")
        t_threshold = 0.001
        num_channels = 4

        grid_data_rgb = (
            torch.rand(self.grid_dual.total_voxels, num_channels).to(device=self.device, dtype=self.dtype) * 0.5
        )
        grid_data_sigma = torch.rand(self.grid_dual.total_voxels, 1).to(device=self.device, dtype=self.dtype) * 0.5
        grid_data_rgb.requires_grad = True
        grid_data_sigma.requires_grad = True

        ray_o, ray_d = self.make_ray_grid((0.0, 0.0, -1.0), 8)
        tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
        tmax = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10
        ray_intervals = self.grid.uniform_ray_samples(
            JaggedTensor(ray_o), JaggedTensor(ray_d), JaggedTensor(tmin), JaggedTensor(tmax), self.step_size
        )
        ray_idx = ray_intervals.jidx.int()
        ray_joffsets = ray_intervals.joffsets
        ray_intervals = ray_intervals.jdata

        ray_mid = ray_intervals.mean(1)
        ray_dt = ray_intervals[:, 1] - ray_intervals[:, 0]
        ray_pts = ray_o[ray_idx] + ray_mid[:, None] * ray_d[ray_idx]

        rgb_samples_4 = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_rgb)).jdata
        sigma_samples = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_sigma)).jdata
        assert isinstance(rgb_samples_4, torch.Tensor)
        assert isinstance(sigma_samples, torch.Tensor)

        # 4-channel forward + backward
        rgb4, depth4, opacity4, ws4, _ = volume_render(
            sigma_samples.squeeze(), rgb_samples_4, ray_dt, ray_mid, ray_joffsets, t_threshold
        )
        self.assertEqual(rgb4.shape, (ray_o.shape[0], num_channels))
        loss_4 = rgb4.sum() + depth4.sum()
        loss_4.backward()
        assert grid_data_rgb.grad is not None
        assert grid_data_sigma.grad is not None
        rgb_grad_4 = grid_data_rgb.grad.detach().clone()
        sigma_grad_4 = grid_data_sigma.grad.detach().clone()
        grid_data_rgb.grad.zero_()
        grid_data_sigma.grad.zero_()

        # Two 2-channel forwards + backwards on the same sample topology
        grid_data_rgb_a = grid_data_rgb[:, 0:2].detach().clone().requires_grad_(True)
        grid_data_rgb_b = grid_data_rgb[:, 2:4].detach().clone().requires_grad_(True)
        rgb_samples_a = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_rgb_a)).jdata
        rgb_samples_b = self.grid_dual.sample_trilinear(JaggedTensor(ray_pts), JaggedTensor(grid_data_rgb_b)).jdata

        rgb_a, depth_a, opacity_a, ws_a, _ = volume_render(
            sigma_samples.squeeze(), rgb_samples_a, ray_dt, ray_mid, ray_joffsets, t_threshold
        )
        rgb_b, depth_b, opacity_b, ws_b, _ = volume_render(
            sigma_samples.squeeze(), rgb_samples_b, ray_dt, ray_mid, ray_joffsets, t_threshold
        )

        rgb_cat = torch.cat([rgb_a, rgb_b], dim=1)
        self.assertLess(torch.abs(rgb4 - rgb_cat).max().item(), 1e-4)
        # Depth/opacity/ws must be channel-agnostic -- same across runs.
        self.assertLess(torch.abs(depth4 - depth_a).max().item(), 1e-5)
        self.assertLess(torch.abs(depth_a - depth_b).max().item(), 1e-5)
        self.assertLess(torch.abs(opacity4 - opacity_a).max().item(), 1e-5)
        self.assertLess(torch.abs(ws4 - ws_a).max().item(), 1e-5)

        # Backward: loss = rgb.sum() + depth.sum(). For the 4-ch run, depth is counted once.
        # For the split runs, each contributes a depth term, so add them but subtract one depth.sum()
        # so the chain rule matches the 4-ch case.
        (rgb_a.sum() + rgb_b.sum() + depth_a.sum()).backward()
        assert grid_data_rgb_a.grad is not None
        assert grid_data_rgb_b.grad is not None
        assert grid_data_sigma.grad is not None
        rgb_grad_a = grid_data_rgb_a.grad.detach().clone()
        rgb_grad_b = grid_data_rgb_b.grad.detach().clone()
        sigma_grad_split = grid_data_sigma.grad.detach().clone()

        # Reconstructed 4-ch grad from two 2-ch grads
        rgb_grad_split = torch.cat([rgb_grad_a, rgb_grad_b], dim=1)
        self.assertLess(torch.abs(rgb_grad_4 - rgb_grad_split).max().item(), 1e-3)
        self.assertLess(torch.abs(sigma_grad_4 - sigma_grad_split).max().item(), 1e-3)

    def test_volume_render_rejects_too_many_channels(self):
        """Channels above MAX_VOLUME_RENDER_CHANNELS must be rejected with a clear error."""
        ray_o, ray_d = self.make_ray_grid((0.0, 0.0, -1.0), 4)
        tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
        tmax = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10
        ray_intervals = self.grid.uniform_ray_samples(
            JaggedTensor(ray_o), JaggedTensor(ray_d), JaggedTensor(tmin), JaggedTensor(tmax), self.step_size
        )
        ray_joffsets = ray_intervals.joffsets
        ray_intervals_jdata = ray_intervals.jdata
        N = ray_intervals_jdata.shape[0]
        if N == 0:
            self.skipTest("No samples along rays; cannot exercise channel guard")
        sigma = torch.zeros(N, device=self.device, dtype=self.dtype)
        dt = (ray_intervals_jdata[:, 1] - ray_intervals_jdata[:, 0]).contiguous()
        tmid = ray_intervals_jdata.mean(1).contiguous()
        # 17 > MAX_VOLUME_RENDER_CHANNELS (16)
        rgbs_too_wide = torch.zeros(N, 17, device=self.device, dtype=self.dtype)
        with self.assertRaisesRegex(RuntimeError, r"between 1 and .*channels.*17|17.*channels.*between 1 and"):
            volume_render(sigma, rgbs_too_wide, dt, tmid, ray_joffsets, 0.001)

    def test_volume_render_total_samples_counts_terminating_sample(self):
        """Regression test for the off-by-one in ``volumeRenderFwdCallback``.

        When the running transmittance ``T`` drops to or below ``tsmtThreshold`` the
        ray terminates early. The sample that triggered the termination has already
        been composited into ``outRGB``, ``outDepth``, ``outOpacity`` and ``outWs``,
        so ``outTotalSamples[rayIdx]`` must include it. A previous version of the
        kernel incremented ``numSamples`` only *after* the break check, which left
        the reported total one short of the samples actually written.

        The test constructs a deterministic per-ray sample layout with known
        sigmas and verifies ``outTotalSamples`` equals the number of entries
        actually written into ``outWs`` (i.e. the non-zero prefix).
        """
        if self.dtype == torch.float16:
            self.skipTest("half precision is noisy near the transmittance threshold")

        # sigma=20, dt=0.1 -> exp(-sigma*dt) = exp(-2) ~= 0.1353 per sample.
        # Starting from T=1, after 3 samples T ~= 0.00247 which is below
        # tsmtThreshold=0.01, so the loop breaks right after compositing sample 2.
        # Expected outTotalSamples for a high-sigma ray: 3.
        tsmt_threshold = 0.01

        # Per-ray sample layouts (each ray may have a different number of slots).
        ray_sizes = [10, 4, 0, 6]
        sigma_per_ray = [20.0, 0.1, 0.0, 20.0]
        # Expected tot_samples per ray after the fix:
        #   ray 0: sigma=20 over 10 slots -> breaks after composite 3 -> tot=3
        #   ray 1: sigma=0.1 over 4 slots -> T stays near 1, no break -> tot=4
        #   ray 2: no slots -> tot=0
        #   ray 3: sigma=20 over 6 slots -> breaks after composite 3 -> tot=3
        expected_tot = [3, 4, 0, 3]

        joffsets_py = [0]
        for s in ray_sizes:
            joffsets_py.append(joffsets_py[-1] + s)
        joffsets = torch.tensor(joffsets_py, dtype=torch.int64, device=self.device)
        N = int(joffsets[-1].item())

        sigmas = torch.zeros(N, dtype=self.dtype, device=self.device)
        for i, s in enumerate(ray_sizes):
            start = joffsets_py[i]
            sigmas[start : start + s] = sigma_per_ray[i]
        dt = torch.full((N,), 0.1, dtype=self.dtype, device=self.device)
        t = torch.arange(N, dtype=self.dtype, device=self.device) * 0.1
        # Single-channel rgbs keep the comparisons simple and are enough to
        # exercise the accumulation path. Mark sigmas as requires_grad so the
        # Python wrapper takes the backward-aware path and actually materializes
        # outWs / outTotalSamples (the inference fast path returns size-0
        # placeholders for those tensors).
        sigmas.requires_grad_(True)
        rgbs = torch.ones((N, 1), dtype=self.dtype, device=self.device)

        _, _, opacity, ws, tot_samples = volume_render(sigmas, rgbs, dt, t, joffsets, tsmt_threshold)

        for ray_i in range(len(ray_sizes)):
            start = joffsets_py[ray_i]
            end = joffsets_py[ray_i + 1]
            tot = int(tot_samples[ray_i].item())
            ws_ray = ws[start:end]

            # The reported total must be in-range and match the analytic expectation.
            self.assertGreaterEqual(tot, 0)
            self.assertLessEqual(tot, end - start)
            self.assertEqual(
                tot,
                expected_tot[ray_i],
                f"Ray {ray_i}: outTotalSamples={tot} but expected {expected_tot[ray_i]}",
            )

            # Every composited sample has strictly positive weight when sigma > 0
            # and T > 0, so the reported total must match the number of non-zero
            # entries in ws (which is zero-initialized at allocation time).
            nz = int((ws_ray != 0).sum().item())
            self.assertEqual(
                tot,
                nz,
                f"Ray {ray_i}: outTotalSamples={tot} but ws has {nz} non-zero entries; "
                "this indicates an off-by-one between outTotalSamples and outWs.",
            )

            # ws must be a contiguous non-zero prefix of length tot followed by
            # untouched zeros.
            if tot > 0:
                self.assertTrue(
                    torch.all(ws_ray[:tot] != 0).item(),
                    f"Ray {ray_i}: ws[:tot] contains a zero entry",
                )
            if tot < end - start:
                self.assertTrue(
                    torch.all(ws_ray[tot:] == 0).item(),
                    f"Ray {ray_i}: ws[tot:] contains a non-zero entry (write past tot)",
                )

            # outOpacity must equal the sum of the composited ws for the ray.
            # This cross-checks that tot_samples and the accumulators agree.
            self.assertTrue(
                torch.isclose(
                    opacity[ray_i],
                    ws_ray.sum(),
                    atol=1e-4 if self.dtype == torch.float32 else 1e-6,
                ).item(),
                f"Ray {ray_i}: outOpacity={opacity[ray_i].item()} disagrees with sum(ws)={ws_ray.sum().item()}",
            )


class TestRayMarching(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_combos)
    def test_uniform_ray_samples(self, device, include_end_segments: bool):
        grid = GridBatch.from_dense(num_grids=1, dense_dims=[2, 2, 2], device=device)

        rays_o = torch.tensor([[-0.6, 0.0, 0.0]], device=device)
        rays_d = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        nears = torch.tensor([0.0], device=device)
        fars = torch.tensor([5.0], device=device)
        step_size = 0.4
        cone_angle = 0.0
        if include_end_segments:
            t_targets = torch.tensor([0.1, 0.4, 0.8, 1.2, 1.6, 2.0, 2.1], device=device)
        else:
            t_targets = torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], device=device)

        intervals = grid.uniform_ray_samples(
            JaggedTensor(rays_o),
            JaggedTensor(rays_d),
            JaggedTensor(nears),
            JaggedTensor(fars),
            step_size,
            cone_angle,
            include_end_segments,
        )
        middles = grid.uniform_ray_samples(
            JaggedTensor(rays_o),
            JaggedTensor(rays_d),
            JaggedTensor(nears),
            JaggedTensor(fars),
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints=True,
        ).jdata
        t_starts, t_ends = torch.unbind(intervals.jdata, dim=-1)
        self.assertTrue(torch.allclose(middles, (t_starts + t_ends) / 2.0, atol=dtype_to_atol(t_starts.dtype)))

        assert torch.allclose(t_starts, t_targets[:-1])
        assert torch.allclose(t_ends, t_targets[1:])
        assert torch.allclose(intervals.jidx, torch.zeros_like(intervals.jidx))

    @parameterized.expand(all_device_dtype_combos)
    def test_uniform_step_size_first_step_is_multiple_of_step_size(self, device, dtype):
        gsize = 8
        grid, _, _ = make_dense_grid_batch_and_jagged_point_data(gsize, device, dtype)

        grid_centers = grid.voxel_to_world(grid.ijk.float()).jdata
        camera_origin_inside = torch.mean(grid_centers, dim=0)
        camera_origin_outside = camera_origin_inside - torch.tensor([0.0, 0.0, 4.0]).to(grid.device)

        ray_d_inside = grid_centers[torch.randperm(grid_centers.shape[0])[:24]] - camera_origin_inside[None, :]
        ray_d_inside /= torch.norm(ray_d_inside, dim=-1, keepdim=True)
        ray_d_outside = grid_centers[torch.randperm(grid_centers.shape[0])[:24]] - camera_origin_outside[None, :]
        ray_d_outside /= torch.norm(ray_d_outside, dim=-1, keepdim=True)

        ray_o_inside = torch.ones_like(ray_d_outside) * camera_origin_inside[None, :]
        ray_o_outside = torch.ones_like(ray_d_outside) * camera_origin_outside[None, :]

        tmin = torch.zeros(ray_d_inside.shape[0]).to(ray_d_inside)
        tmin += torch.rand_like(tmin) * 0.01
        tmax = torch.ones_like(tmin) * 1e10

        step_size = 0.01
        ray_times_inside = grid.uniform_ray_samples(
            JaggedTensor(ray_o_inside),
            JaggedTensor(ray_d_inside),
            JaggedTensor(tmin),
            JaggedTensor(tmax),
            step_size,
            include_end_segments=False,
        )
        ray_idx, ray_times_inside = ray_times_inside.jidx.long(), ray_times_inside.jdata
        nsteps_inside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_inside, torch.round(nsteps_inside), atol=dtype_to_atol(dtype)))

        ray_times_inside = grid.uniform_ray_samples(
            JaggedTensor(ray_o_outside),
            JaggedTensor(ray_d_outside),
            JaggedTensor(tmin),
            JaggedTensor(tmax),
            step_size,
            include_end_segments=False,
        )
        ray_idx, ray_times_inside = ray_times_inside.jidx.long(), ray_times_inside.jdata
        nsteps_outside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_outside, torch.round(nsteps_outside), atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_uniform_ray_samples_cone_tracing_count_matches_generate(self, device, dtype):
        """Regression test for the floor/ceil mismatch in ``SampleRaysUniform.cu``.

        With ``cone_angle > 0`` and ``include_end_segments=True`` the count and
        generate callbacks must agree on how ``t0`` is snapped to the first
        step boundary inside each HDDA segment. Historically the count path used
        ``floor(distToVox / stepSize)`` while the generate path used ``ceil``,
        which silently under-allocates the output buffer under cone tracing:
        since ``stepSize`` depends on ``t0`` when ``cone_angle > 0``, the two
        callbacks then walk different ``t0`` trajectories and disagree on the
        number of samples they emit. The generate pass ends up writing one
        sample past the buffer the count pass allocated, corrupting adjacent
        rays' slots.

        The test uses a ray that traverses a long 1D contiguous run of voxels
        with an aggressive cone angle so the per-step ``stepSize`` actually
        grows inside the segment. The sample count and per-sample boundaries
        are compared against a Python reference that mirrors the fixed
        (``ceil``-rounded) algorithm. A pre-fix CUDA/CPU kernel would return
        one fewer sample than the reference and mismatched boundaries in the
        tail of the ray.
        """
        if dtype == torch.float16:
            # The half-precision specialisation of ``nanovdb::math::Delta`` is 1e-3,
            # which reshuffles the boundary-gap decisions. That code path is
            # orthogonal to the floor/ceil fix we want to cover here, so we
            # only exercise float32 / float64.
            self.skipTest("half-precision Delta rearranges the boundary-gap checks")

        import math as _math

        # Long, thin dense grid so the ray sees a single merged HDDA segment
        # spanning many voxels. Voxel size 1 and origin 0 give a grid whose
        # axis-aligned bounding box is x in [-0.5, 49.5], y, z in [-0.5, 0.5].
        grid = GridBatch.from_dense(num_grids=1, dense_dims=[50, 1, 1], device=device)

        # Ray starts outside the grid at x = -5 and travels along +x through
        # y = z = 0, so it hits every voxel (i, 0, 0) exactly once. The ray
        # enters the grid at t = 4.5 and exits at t = 54.5.
        rays_o = torch.tensor([[-5.0, 0.0, 0.0]], dtype=dtype, device=device)
        rays_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        nears = torch.tensor([0.0], dtype=dtype, device=device)
        fars = torch.tensor([60.0], dtype=dtype, device=device)

        # Cone=0.5 is aggressive enough that ``stepSize = t0 * cone_angle``
        # overtakes the 1.0 floor after t0 >= 2, so the per-iteration step
        # really does grow across the while loop. This is the regime where
        # floor and ceil paths diverge.
        step_size = 1.0
        cone_angle = 0.5

        intervals = grid.uniform_ray_samples(
            JaggedTensor(rays_o),
            JaggedTensor(rays_d),
            JaggedTensor(nears),
            JaggedTensor(fars),
            step_size,
            cone_angle=cone_angle,
            include_end_segments=True,
        )
        actual = intervals.jdata.detach().cpu().double().numpy()

        # Reference implementation mirroring ``countSamplesPerRayCallback`` and
        # ``generateRaySamplesCallback`` for a single HDDA segment
        # [t_enter, t_exit] with the post-fix ``ceil`` rounding applied in both.
        t_enter, t_exit = 4.5, 54.5
        # ``Delta`` for float is small enough that the exact test boundaries
        # (0.5 and 8.9375 gap widths) are always considered gaps.
        delta = 1e-6

        def _calc_dt(t: float, cone: float, min_step: float, max_step: float = 1e10) -> float:
            return max(min_step, min(max_step, t * cone))

        t0 = 0.0
        step = _calc_dt(t0, cone_angle, step_size)
        expected: list[tuple[float, float]] = []
        dist_to_vox = t_enter - t0
        t0 += _math.ceil(dist_to_vox / step) * step
        t1 = t0 + step
        if t0 > t_exit:
            expected.append((t_enter, t_exit))
        else:
            if t0 - t_enter > delta:
                expected.append((t_enter, t0))
            while t1 < t_exit:
                expected.append((t0, t1))
                t0 = t1
                step = _calc_dt(t0, cone_angle, step_size)
                t1 += step
            if t_exit - t0 > delta:
                expected.append((t0, t_exit))

        self.assertEqual(
            actual.shape[0],
            len(expected),
            f"Sample count mismatch (device={device}, dtype={dtype}): got {actual.shape[0]}, "
            f"expected {len(expected)}. Pre-fix code used floor() in the count path and "
            f"typically returned one fewer sample than the generate path.",
        )

        # Use a fp32-appropriate tolerance for the cumulative step arithmetic.
        tol = 1e-3 if dtype == torch.float32 else 1e-6
        for i, (ref_s, ref_e) in enumerate(expected):
            self.assertAlmostEqual(
                float(actual[i, 0]),
                ref_s,
                delta=tol,
                msg=f"t_start mismatch at sample {i} (device={device}, dtype={dtype})",
            )
            self.assertAlmostEqual(
                float(actual[i, 1]),
                ref_e,
                delta=tol,
                msg=f"t_end mismatch at sample {i} (device={device}, dtype={dtype})",
            )

        # Sanity invariants that must hold regardless of the specific algorithm:
        # strict monotonicity of t_start, non-degenerate intervals, and that
        # all samples lie within the grid's HDDA-clipped [t_enter, t_exit].
        t_starts = intervals.jdata[:, 0].double()
        t_ends = intervals.jdata[:, 1].double()
        self.assertTrue(torch.all(t_ends > t_starts).item(), "Found a degenerate interval (t_end <= t_start)")
        self.assertTrue(
            torch.all(t_starts[1:] >= t_starts[:-1]).item(),
            "Sample starts are not monotonically non-decreasing along the ray",
        )
        self.assertGreaterEqual(float(t_starts.min().item()), t_enter - tol)
        self.assertLessEqual(float(t_ends.max().item()), t_exit + tol)

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_bug(self, device, dtype):
        data_path = get_fvdb_test_data_path()
        data = torch.load(str(data_path / "ray_marching" / "repro_bug.pth"))
        grid = GridBatch.from_ijk(
            JaggedTensor(data["ijk"].to(device)), voxel_sizes=data["vox_size"], origins=data["vox_origin"]
        )
        ray_o: torch.Tensor = torch.load(str(data_path / "ray_marching" / "ray_o.pth")).to(device=device, dtype=dtype)
        ray_d: torch.Tensor = torch.load(str(data_path / "ray_marching" / "ray_d.pth")).to(device=device, dtype=dtype)

        segments = grid.segments_along_rays(JaggedTensor(ray_o.to(dtype)), JaggedTensor(ray_d.to(dtype)), 100, 0.0)

        self.assertEqual(segments[0][0].jdata.shape[0], 52)

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_always_sorted(self, device, dtype):
        for eps in [0.0, 1e-5]:
            pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            grid = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=0.0001, origins=torch.zeros(3))
            grid = grid.dilated_grid(1)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

            segments = grid.segments_along_rays(JaggedTensor(rays_o), JaggedTensor(rays_d), 100, eps=eps)

            for segments_i in segments[0]:
                if segments_i.rshape[0] == 0:
                    continue
                segments_i = segments_i.jdata
                self.assertTrue(torch.all(segments_i[:, 1] - segments_i[:, 0] >= eps))
                self.assertTrue(
                    torch.all(segments_i[1:, 0] - segments_i[:-1, 0] >= eps),
                    f"mismatch eps = {eps}, diff = {segments_i[1:, 0] >= segments_i[:-1, 0]}, vals = {segments_i}, (1) = {segments_i[1:, 0]}, (2) = {segments_i[:-1, 0]}",
                )
                self.assertTrue(torch.all(segments_i[1:, 1] - segments_i[:-1, 1] >= eps))

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_always_sorted_batched(self, device, dtype):
        for eps in [0.0, 1e-5]:
            pts = JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)] * 2)
            grid = GridBatch.from_points(pts, voxel_sizes=0.0001, origins=torch.zeros(3))
            grid = grid.dilated_grid(1)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[0].jdata[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
            rays_o = fvdb.JaggedTensor([rays_o] * 2)
            rays_d = fvdb.JaggedTensor([rays_d] * 2)

            segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=eps)

            for b_i in range(len(pts)):
                for segments_i in segments[b_i]:
                    if segments_i.rshape[0] == 0:
                        continue
                    segments_i = segments_i.jdata
                    self.assertTrue(torch.all(segments_i[:, 1] - segments_i[:, 0] >= eps))
                    self.assertTrue(
                        torch.all(segments_i[1:, 0] - segments_i[:-1, 0] >= eps),
                        f"mismatch eps = {eps}, diff = {segments_i[1:, 0] >= segments_i[:-1, 0]}, vals = {segments_i}, (1) = {segments_i[1:, 0]}, (2) = {segments_i[:-1, 0]}",
                    )
                    self.assertTrue(torch.all(segments_i[1:, 1] - segments_i[:-1, 1] >= eps))

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_batch_size_mismatch_throws(self, device, dtype):
        pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
        # pts = fvdb.JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)]*2)
        grid = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=0.0001, origins=torch.zeros(3))
        grid = grid.dilated_grid(1)

        rays_o = -torch.ones(100, 3).to(device).to(dtype)
        rays_d = pts[:100] - rays_o
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = fvdb.JaggedTensor([rays_o] * 2)
        rays_d = fvdb.JaggedTensor([rays_d] * 2)

        with self.assertRaises(Exception):
            segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=1e-4)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_always_sorted(self, device, dtype):
        for i in range(3):
            pts = torch.rand(10000, 3, device=device, dtype=dtype)
            grid = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=0.01, origins=torch.zeros(3))
            grid = grid.dilated_grid(1)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

            out_voxels, out_times = grid.voxels_along_rays(JaggedTensor(rays_o), JaggedTensor(rays_d), 100, 1.0e-5)

            out_idx, out_times2 = grid.voxels_along_rays(
                JaggedTensor(rays_o), JaggedTensor(rays_d), 100, 1.0e-5, return_ijk=False
            )
            out_idx2 = grid.ijk_to_index(out_voxels.jflatten(dim=1)).jreshape_as(out_idx)
            self.assertTrue(torch.all(out_idx.jdata == out_idx2.jdata))
            self.assertTrue(torch.allclose(out_times.jdata, out_times2.jdata))

            for times_i, voxels_i in zip(out_times[0], out_voxels[0]):
                if times_i.rshape[0] == 0:
                    continue
                times_i, voxels_i = times_i.jdata, voxels_i.jdata
                self.assertTrue(
                    torch.all(times_i[:, 0] < times_i[:, 1]),
                    f"Max diff = {(times_i[:, 1] - times_i[:, 0]).max().item()}",
                )
                self.assertTrue(torch.all(times_i[1:, 0] > times_i[:-1, 0]))
                self.assertTrue(torch.all(times_i[1:, 1] > times_i[:-1, 1]))
                # Should always march
                max_diff = torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values.cpu().detach().numpy()
                self.assertTrue(
                    torch.all(torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values >= 1),
                    f"Max diff = {max_diff, voxels_i.cpu().numpy(), times_i.cpu().numpy()}",
                )

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_batch_size_mismatch_throws(self, device, dtype):
        pts = torch.rand(10000, 3, device=device, dtype=dtype)
        # pts = fvdb.JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)]*2)
        grid = GridBatch.from_points(JaggedTensor(pts), voxel_sizes=0.0001, origins=torch.zeros(3))
        grid = grid.dilated_grid(1)

        rays_o = -torch.ones(100, 3, device=device, dtype=dtype)
        rays_d = pts[:100] - rays_o
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = fvdb.JaggedTensor([rays_o] * 2)
        rays_d = fvdb.JaggedTensor([rays_d] * 2)

        with self.assertRaises(Exception):
            out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_always_sorted_batched(self, device, dtype):
        for i in range(3):
            # pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            pts = fvdb.JaggedTensor([torch.rand(100, 3, device=device, dtype=dtype)] * 2)
            grid = GridBatch.from_points(pts, 0.01, torch.zeros(3))
            grid = grid.dilated_grid(1)

            rays_o = [-torch.ones(100, 3).to(device).to(dtype)] * 2
            rays_d = [pts[i].jdata[:100] - rays_o[i] for i in range(2)]
            rays_d = [r / torch.norm(r, dim=-1, keepdim=True) for r in rays_d]
            rays_o = fvdb.JaggedTensor(rays_o)
            rays_d = fvdb.JaggedTensor(rays_d)

            out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)

            out_idx, out_times2 = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5, return_ijk=False)
            out_idx2 = grid.ijk_to_index(out_voxels.jflatten(dim=1)).jreshape_as(out_idx)
            self.assertTrue(torch.all(out_idx.jdata == out_idx2.jdata))
            self.assertTrue(torch.allclose(out_times.jdata, out_times2.jdata))

            for i, _ in enumerate(zip(out_voxels, out_times)):
                for times_i, voxels_i in zip(out_times[i], out_voxels[i]):
                    if times_i.rshape[0] == 0:
                        continue
                    times_i, voxels_i = times_i.jdata, voxels_i.jdata
                    self.assertTrue(
                        torch.all(times_i[:, 0] < times_i[:, 1]),
                        f"Max diff = {(times_i[:, 1] - times_i[:, 0]).max().item()}",
                    )
                    if times_i[1:, 0].numel() > 0:
                        self.assertTrue(
                            torch.all(times_i[1:, 0] > times_i[:-1, 0]),
                            f"Max diff = {(times_i[1:, 0] - times_i[:-1, 0]).max().item()}",
                        )
                    # Should always march
                    max_diff = torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values.cpu().detach().numpy()
                    self.assertTrue(
                        torch.all(torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values >= 1),
                        f"Max diff = {max_diff, voxels_i.cpu().numpy(), times_i.cpu().numpy()}",
                    )


if __name__ == "__main__":
    unittest.main()
