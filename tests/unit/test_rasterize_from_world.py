#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_grads_nonzero():
    import fvdb

    device = torch.device("cuda")
    C, N, D = 1, 1, 3

    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=torch.float32, requires_grad=True)
    quats = torch.tensor([[0.97, 0.05, 0.20, -0.10]], device=device, dtype=torch.float32, requires_grad=True)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=torch.float32, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=torch.float32, requires_grad=True)

    sh0 = torch.randn((N, 1, D), device=device, dtype=torch.float32, requires_grad=True)
    shN = torch.empty((N, 0, D), device=device, dtype=torch.float32, requires_grad=True)

    gs = fvdb.GaussianSplat3d.from_tensors(
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

    rendered, alphas = gs.render_images_from_world(
        world_to_camera_matrices=world_to_cam,
        projection_matrices=K,
        image_width=image_width,
        image_height=image_height,
        near=0.01,
        far=1e10,
        camera_model=fvdb.CameraModel.PINHOLE,
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
    assert (
        logit_opacities.grad is not None
        and torch.isfinite(logit_opacities.grad).all()
        and logit_opacities.grad.abs().sum().item() > 0.0
    )
    assert sh0.grad is not None and torch.isfinite(sh0.grad).all() and sh0.grad.abs().sum().item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_grads_match_finite_differences():
    """
    Finite-difference check for the 3DGS dense rasterizer backward pass.

    This is intentionally small (C=N=1, single tile) to keep runtime down and to avoid
    discontinuities from tile assignment changes.
    """
    import fvdb

    device = torch.device("cuda")
    dtype = torch.float32

    C, N, D = 1, 1, 3
    image_width = 16
    image_height = 16

    world_to_cam = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)  # [1,4,4]
    K = torch.tensor(
        [[[18.0, 0.0, 7.5], [0.0, 18.0, 7.5], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )

    # Fixed SH (degree 0 only), so the only learnable feature is the SH0 coefficient itself.
    # We keep SH fixed for this test; we are checking geometry + opacity grads.
    sh0 = torch.tensor([[[0.8, -0.2, 0.3]]], device=device, dtype=dtype)  # [N,1,D]
    shN = torch.empty((N, 0, D), device=device, dtype=dtype)

    def loss_from_params(
        means_t: torch.Tensor, quats_t: torch.Tensor, log_scales_t: torch.Tensor, logit_opacities_t: torch.Tensor
    ) -> torch.Tensor:
        gs = fvdb.GaussianSplat3d.from_tensors(
            means=means_t,
            quats=quats_t,
            log_scales=log_scales_t,
            logit_opacities=logit_opacities_t,
            sh0=sh0,
            shN=shN,
            accumulate_mean_2d_gradients=False,
            accumulate_max_2d_radii=False,
            detach=False,
        )

        rendered, alphas = gs.render_images_from_world(
            world_to_camera_matrices=world_to_cam,
            projection_matrices=K,
            image_width=image_width,
            image_height=image_height,
            near=0.01,
            far=1e10,
            camera_model=fvdb.CameraModel.PINHOLE,
            distortion_coeffs=None,
            sh_degree_to_use=0,
            tile_size=16,  # single tile
            min_radius_2d=0.0,
            eps_2d=0.3,
            antialias=False,
            backgrounds=None,
        )

        # Smooth scalar objective; includes both color/features and opacity terms.
        return (rendered * rendered).sum() + 0.5 * alphas.sum()

    # Baseline parameters.
    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=dtype, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)  # fixed
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=dtype, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=dtype, requires_grad=True)

    # Autograd gradients.
    loss = loss_from_params(means, quats, log_scales, logit_opacities)
    loss.backward()

    assert means.grad is not None
    assert log_scales.grad is not None
    assert logit_opacities.grad is not None

    # Centered finite differences for a small subset of scalars.
    eps = 1.0e-3
    rtol = 2.5e-2
    atol = 2.5e-2

    def fd_scalar(param_name: str, i0: int, i1: int | None = None) -> float:
        with torch.no_grad():
            means_p = means.detach().clone()
            means_m = means.detach().clone()
            log_scales_p = log_scales.detach().clone()
            log_scales_m = log_scales.detach().clone()
            logit_p = logit_opacities.detach().clone()
            logit_m = logit_opacities.detach().clone()

            if param_name == "means":
                assert i1 is not None
                means_p[i0, i1] += eps
                means_m[i0, i1] -= eps
            elif param_name == "log_scales":
                assert i1 is not None
                log_scales_p[i0, i1] += eps
                log_scales_m[i0, i1] -= eps
            elif param_name == "logit_opacities":
                assert i1 is None
                logit_p[i0] += eps
                logit_m[i0] -= eps
            else:
                raise AssertionError(f"Unknown param_name: {param_name}")

            lp = loss_from_params(means_p, quats, log_scales_p, logit_p).item()
            lm = loss_from_params(means_m, quats, log_scales_m, logit_m).item()
            return (lp - lm) / (2.0 * eps)

    checks: list[tuple[str, float, float]] = []
    checks.append(("means[0,0]", float(means.grad[0, 0].item()), fd_scalar("means", 0, 0)))
    checks.append(("means[0,2]", float(means.grad[0, 2].item()), fd_scalar("means", 0, 2)))
    checks.append(("log_scales[0,0]", float(log_scales.grad[0, 0].item()), fd_scalar("log_scales", 0, 0)))
    checks.append(("log_scales[0,2]", float(log_scales.grad[0, 2].item()), fd_scalar("log_scales", 0, 2)))
    checks.append(("logit_opacities[0]", float(logit_opacities.grad[0].item()), fd_scalar("logit_opacities", 0)))

    for name, grad_autograd, grad_fd in checks:
        assert torch.isfinite(torch.tensor(grad_autograd))
        assert torch.isfinite(torch.tensor(grad_fd))
        assert grad_autograd == pytest.approx(
            grad_fd, rel=rtol, abs=atol
        ), f"{name}: autograd={grad_autograd} fd={grad_fd}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_masks_write_background_and_zero_grads():
    import fvdb

    device = torch.device("cuda")
    dtype = torch.float32

    C, N, D = 1, 1, 3
    image_width = 16
    image_height = 16
    tile_size = 16  # single tile -> mask is [C,1,1]

    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=dtype, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=dtype, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=dtype, requires_grad=True)
    sh0 = torch.randn((N, 1, D), device=device, dtype=dtype, requires_grad=True)
    shN = torch.empty((N, 0, D), device=device, dtype=dtype, requires_grad=True)

    gs = fvdb.GaussianSplat3d.from_tensors(
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

    world_to_cam = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    K = torch.tensor(
        [[[18.0, 0.0, 7.5], [0.0, 18.0, 7.5], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )

    backgrounds = torch.tensor([[0.10, -0.20, 0.30]], device=device, dtype=dtype)  # [C,D]
    masks = torch.zeros((C, 1, 1), device=device, dtype=torch.bool)  # mask out the only tile

    rendered, alphas = gs.render_images_from_world(
        world_to_camera_matrices=world_to_cam,
        projection_matrices=K,
        image_width=image_width,
        image_height=image_height,
        near=0.01,
        far=1e10,
        camera_model=fvdb.CameraModel.PINHOLE,
        distortion_coeffs=None,
        sh_degree_to_use=0,
        tile_size=tile_size,
        min_radius_2d=0.0,
        eps_2d=0.3,
        antialias=False,
        backgrounds=backgrounds,
        masks=masks,
    )

    expected = backgrounds.view(C, 1, 1, D).expand(C, image_height, image_width, D)
    assert torch.equal(alphas, torch.zeros_like(alphas))
    assert torch.equal(rendered, expected)

    loss = rendered.sum() + alphas.sum()
    loss.backward()

    # Masked pixels contribute nothing: grads should be exactly zero.
    assert means.grad is not None and torch.equal(means.grad, torch.zeros_like(means.grad))
    assert quats.grad is not None and torch.equal(quats.grad, torch.zeros_like(quats.grad))
    assert log_scales.grad is not None and torch.equal(log_scales.grad, torch.zeros_like(log_scales.grad))
    assert logit_opacities.grad is not None and torch.equal(
        logit_opacities.grad, torch.zeros_like(logit_opacities.grad)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_backgrounds_used_when_no_intersections():
    """
    If no Gaussians intersect the image, rasterization should return the background (if provided)
    with alpha=0 everywhere.
    """
    import fvdb

    device = torch.device("cuda")
    dtype = torch.float32

    C, N, D = 1, 1, 3
    image_width = 16
    image_height = 16

    # Put the Gaussian in front of the camera but closer than near plane so it should be clipped.
    means = torch.tensor([[0.0, 0.0, 1.0e-4]], device=device, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=dtype)
    logit_opacities = torch.tensor([2.0], device=device, dtype=dtype)
    sh0 = torch.randn((N, 1, D), device=device, dtype=dtype)
    shN = torch.empty((N, 0, D), device=device, dtype=dtype)

    gs = fvdb.GaussianSplat3d.from_tensors(
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

    world_to_cam = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    K = torch.tensor(
        [[[18.0, 0.0, 7.5], [0.0, 18.0, 7.5], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )

    backgrounds = torch.tensor([[0.25, 0.50, -0.75]], device=device, dtype=dtype)
    rendered, alphas = gs.render_images_from_world(
        world_to_camera_matrices=world_to_cam,
        projection_matrices=K,
        image_width=image_width,
        image_height=image_height,
        near=0.01,
        far=1e10,
        camera_model=fvdb.CameraModel.PINHOLE,
        distortion_coeffs=None,
        sh_degree_to_use=0,
        tile_size=16,
        min_radius_2d=0.0,
        eps_2d=0.3,
        antialias=False,
        backgrounds=backgrounds,
        masks=None,
    )

    expected = backgrounds.view(C, 1, 1, D).expand(C, image_height, image_width, D)
    assert torch.equal(alphas, torch.zeros_like(alphas))
    assert torch.equal(rendered, expected)
