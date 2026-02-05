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

    shN = torch.empty((N, 0, D), device=device, dtype=dtype)

    def loss_from_params(
        means_t: torch.Tensor,
        quats_t: torch.Tensor,
        log_scales_t: torch.Tensor,
        logit_opacities_t: torch.Tensor,
        sh0_t: torch.Tensor,
    ) -> torch.Tensor:
        gs = fvdb.GaussianSplat3d.from_tensors(
            means=means_t,
            quats=quats_t,
            log_scales=log_scales_t,
            logit_opacities=logit_opacities_t,
            sh0=sh0_t,
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
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=dtype, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=dtype, requires_grad=True)
    sh0 = torch.tensor([[[0.8, -0.2, 0.3]]], device=device, dtype=dtype, requires_grad=True)  # [N,1,D]

    # Autograd gradients.
    loss = loss_from_params(means, quats, log_scales, logit_opacities, sh0)
    loss.backward()

    assert means.grad is not None
    assert quats.grad is not None
    assert log_scales.grad is not None
    assert logit_opacities.grad is not None
    assert sh0.grad is not None

    # Centered finite differences for a small subset of scalars.
    eps = 1.0e-3
    rtol = 2.5e-2
    atol = 2.5e-2

    def fd_scalar(param_name: str, i0: int, i1: int | None = None, i2: int | None = None) -> float:
        with torch.no_grad():
            means_p = means.detach().clone()
            means_m = means.detach().clone()
            quats_p = quats.detach().clone()
            quats_m = quats.detach().clone()
            log_scales_p = log_scales.detach().clone()
            log_scales_m = log_scales.detach().clone()
            logit_p = logit_opacities.detach().clone()
            logit_m = logit_opacities.detach().clone()
            sh0_p = sh0.detach().clone()
            sh0_m = sh0.detach().clone()

            if param_name == "means":
                assert i1 is not None
                means_p[i0, i1] += eps
                means_m[i0, i1] -= eps
            elif param_name == "quats":
                assert i1 is not None
                quats_p[i0, i1] += eps
                quats_m[i0, i1] -= eps
            elif param_name == "log_scales":
                assert i1 is not None
                log_scales_p[i0, i1] += eps
                log_scales_m[i0, i1] -= eps
            elif param_name == "logit_opacities":
                assert i1 is None
                logit_p[i0] += eps
                logit_m[i0] -= eps
            elif param_name == "sh0":
                assert i1 is not None and i2 is not None
                sh0_p[i0, i1, i2] += eps
                sh0_m[i0, i1, i2] -= eps
            else:
                raise AssertionError(f"Unknown param_name: {param_name}")

            lp = loss_from_params(means_p, quats_p, log_scales_p, logit_p, sh0_p).item()
            lm = loss_from_params(means_m, quats_m, log_scales_m, logit_m, sh0_m).item()
            return (lp - lm) / (2.0 * eps)

    checks: list[tuple[str, float, float]] = []
    checks.append(("means[0,0]", float(means.grad[0, 0].item()), fd_scalar("means", 0, 0)))
    checks.append(("means[0,2]", float(means.grad[0, 2].item()), fd_scalar("means", 0, 2)))
    checks.append(("log_scales[0,0]", float(log_scales.grad[0, 0].item()), fd_scalar("log_scales", 0, 0)))
    checks.append(("log_scales[0,2]", float(log_scales.grad[0, 2].item()), fd_scalar("log_scales", 0, 2)))
    checks.append(("logit_opacities[0]", float(logit_opacities.grad[0].item()), fd_scalar("logit_opacities", 0)))
    checks.append(("sh0[0,0,0]", float(sh0.grad[0, 0, 0].item()), fd_scalar("sh0", 0, 0, 0)))
    checks.append(("sh0[0,0,2]", float(sh0.grad[0, 0, 2].item()), fd_scalar("sh0", 0, 0, 2)))

    for name, grad_autograd, grad_fd in checks:
        assert torch.isfinite(torch.tensor(grad_autograd))
        assert torch.isfinite(torch.tensor(grad_fd))
        assert grad_autograd == pytest.approx(
            grad_fd, rel=rtol, abs=atol
        ), f"{name}: autograd={grad_autograd} fd={grad_fd}"

    # Quaternion tangent-space finite-difference check.
    # We compare the FD directional derivative along a small axis-angle perturbation to the
    # autograd gradient dotted with dq/dt (estimated numerically).
    def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    def quat_exp_axis_angle_wxyz(v: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(v)
        half = 0.5 * theta
        if float(theta.item()) < 1.0e-12:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=v.device, dtype=v.dtype)
        axis = v / theta
        return torch.cat((torch.cos(half).view(1), torch.sin(half) * axis), dim=0)

    with torch.no_grad():
        q0 = quats.detach()[0]  # [4] wxyz
        # Two independent tangent directions.
        dirs = [
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.2], device=device, dtype=dtype),
        ]
        for j, d in enumerate(dirs):
            d = d / torch.linalg.norm(d)
            dq_plus = quat_exp_axis_angle_wxyz(eps * d)
            dq_minus = quat_exp_axis_angle_wxyz(-eps * d)
            q_plus = quat_mul_wxyz(dq_plus, q0).view(1, 4)
            q_minus = quat_mul_wxyz(dq_minus, q0).view(1, 4)

            lp = loss_from_params(
                means.detach(), q_plus, log_scales.detach(), logit_opacities.detach(), sh0.detach()
            ).item()
            lm = loss_from_params(
                means.detach(), q_minus, log_scales.detach(), logit_opacities.detach(), sh0.detach()
            ).item()
            fd_dir = (lp - lm) / (2.0 * eps)

            dqdt = (q_plus - q_minus) / (2.0 * eps)  # [1,4]
            dir_autograd = float((quats.grad.detach() * dqdt).sum().item())

            assert torch.isfinite(torch.tensor(fd_dir))
            assert torch.isfinite(torch.tensor(dir_autograd))
            assert dir_autograd == pytest.approx(
                fd_dir, rel=rtol, abs=atol
            ), f"quats tangent dir {j}: autograd={dir_autograd} fd={fd_dir}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_grads_nonzero_with_shN():
    import fvdb

    device = torch.device("cuda")
    C, N, D = 1, 1, 3

    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=torch.float32, requires_grad=True)
    quats = torch.tensor([[0.97, 0.05, 0.20, -0.10]], device=device, dtype=torch.float32, requires_grad=True)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=torch.float32, requires_grad=True)
    logit_opacities = torch.tensor([2.0], device=device, dtype=torch.float32, requires_grad=True)

    # Degree 1 uses 4 SH bases total; sh0 is [N,1,D] and shN holds the remaining K-1=3 bases.
    sh0 = torch.randn((N, 1, D), device=device, dtype=torch.float32, requires_grad=True)
    shN = torch.randn((N, 3, D), device=device, dtype=torch.float32, requires_grad=True)

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
        sh_degree_to_use=1,
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
    assert shN.grad is not None and torch.isfinite(shN.grad).all() and shN.grad.abs().sum().item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussiansplat3d_render_images_from_world_shN_grads_match_finite_differences():
    """
    Finite-difference check for SHN coefficients (non-constant SH terms).
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

    means = torch.tensor([[0.05, 0.05, 2.5]], device=device, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    log_scales = torch.tensor([[-0.6, -0.9, -0.4]], device=device, dtype=dtype)
    logit_opacities = torch.tensor([2.0], device=device, dtype=dtype)

    sh0 = torch.tensor([[[0.8, -0.2, 0.3]]], device=device, dtype=dtype)  # [N,1,D]
    shN0 = torch.tensor([[[0.05, -0.01, 0.02], [0.00, 0.03, -0.04], [0.01, 0.02, 0.00]]], device=device, dtype=dtype)

    def loss_from_shN(shN_t: torch.Tensor) -> torch.Tensor:
        gs = fvdb.GaussianSplat3d.from_tensors(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN_t,
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
            sh_degree_to_use=1,
            tile_size=16,
            min_radius_2d=0.0,
            eps_2d=0.3,
            antialias=False,
            backgrounds=None,
        )

        return (rendered * rendered).sum() + 0.5 * alphas.sum()

    shN = shN0.detach().clone().requires_grad_(True)
    loss = loss_from_shN(shN)
    loss.backward()
    assert shN.grad is not None

    eps = 1.0e-3
    rtol = 2.5e-2
    atol = 2.5e-2

    def fd_shN(i1: int, i2: int) -> float:
        with torch.no_grad():
            shN_p = shN0.detach().clone()
            shN_m = shN0.detach().clone()
            shN_p[0, i1, i2] += eps
            shN_m[0, i1, i2] -= eps
            lp = loss_from_shN(shN_p).item()
            lm = loss_from_shN(shN_m).item()
            return (lp - lm) / (2.0 * eps)

    checks = [
        ("shN[0,0,0]", float(shN.grad[0, 0, 0].item()), fd_shN(0, 0)),
        ("shN[0,2,1]", float(shN.grad[0, 2, 1].item()), fd_shN(2, 1)),
    ]

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
