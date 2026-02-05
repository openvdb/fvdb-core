#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_3dgs_grads_nonzero():
    import fvdb._fvdb_cpp as _C

    device = torch.device("cuda")
    C, N, D = 1, 1, 3

    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=torch.float32, requires_grad=True)
    quats = torch.tensor([[0.97, 0.05, 0.20, -0.10]], device=device, dtype=torch.float32, requires_grad=True)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=torch.float32, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=torch.float32, requires_grad=True)

    sh0 = torch.randn((N, 1, D), device=device, dtype=torch.float32, requires_grad=True)
    shN = torch.empty((N, 0, D), device=device, dtype=torch.float32, requires_grad=True)

    gs = _C.GaussianSplat3d(
        means=means,
        quats=quats,
        log_scales=log_scales,
        logit_opacities=logit_opacities,
        sh0=sh0,
        shN=shN,
        accumulate_mean_2d_gradients=False,
        accumulate_max_2d_radii=False,
        detach=False,
    )

    image_width = 16
    image_height = 16
    world_to_cam = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor(
        [[[18.0, 0.0, 7.5], [0.0, 18.0, 7.5], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    rendered, alphas = gs.render_images_from_world_3dgs(
        world_to_camera_matrices=world_to_cam,
        projection_matrices=K,
        image_width=image_width,
        image_height=image_height,
        near=0.01,
        far=1e10,
        camera_model=_C.CameraModel.PINHOLE,
        distortion_coeffs=None,
        sh_degree_to_use=0,
        tile_size=16,
        min_radius_2d=0.0,
        eps_2d=0.3,
        antialias=False,
        backgrounds=None,
    )

    loss = (rendered * rendered).sum() + alphas.sum()
    loss.backward()

    assert means.grad is not None and torch.isfinite(means.grad).all() and means.grad.abs().sum().item() > 0.0
    assert quats.grad is not None and torch.isfinite(quats.grad).all() and quats.grad.abs().sum().item() > 0.0
    assert (
        log_scales.grad is not None
        and torch.isfinite(log_scales.grad).all()
        and log_scales.grad.abs().sum().item() > 0.0
    )
