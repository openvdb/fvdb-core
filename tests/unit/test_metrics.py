# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from fvdb.utils.metrics import psnr, ssim


@pytest.mark.parametrize("padding", ["same", "valid"])  # fused-ssim supports these paddings
def test_ssim_identical_images_is_one_and_has_grad(padding):
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Use a modest image size for speed; CHW must be >= 11 for valid padding
    img = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float32)
    if padding == "valid":
        # Ensure valid windowing works with a slightly larger image
        img = torch.rand(1, 1, 64, 64, device=device, dtype=torch.float32)

    img1 = img.clone().requires_grad_(True)
    img2 = img.clone()

    val = ssim(img1, img2, padding=padding, train=True)
    assert torch.is_tensor(val)
    assert val.ndim == 0
    assert torch.isfinite(val)
    assert pytest.approx(1.0, rel=0, abs=1e-4) == val.item()

    val.backward()
    assert img1.grad is not None
    assert img1.grad.shape == img1.shape


def test_ssim_identical_greater_than_random():
    torch.manual_seed(42)
    device = torch.device("cuda")

    img1 = torch.rand(1, 1, 64, 64, device=device, dtype=torch.float32)
    img2_same = img1.clone()
    img2_rand = torch.rand_like(img1)

    val_same = ssim(img1, img2_same, padding="same", train=False).item()
    val_rand = ssim(img1, img2_rand, padding="same", train=False).item()

    assert val_same > val_rand


def test_psnr_identical_images_is_inf_none_reduction():
    gt = torch.zeros((3, 4, 4))  # CHW per sample
    noisy = gt.clone()
    val = psnr(noisy.unsqueeze(0), gt.unsqueeze(0), max_value=1.0, reduction="none")  # add batch dim

    assert val.shape == (1,)
    assert torch.isinf(val).all()


def test_psnr_known_values_and_reductions():
    # Construct a batch of two images with known MSEs
    # Case 1: noisy = 0.1, gt = 0.0 → MSE = 0.01 → PSNR = 20 dB
    # Case 2: noisy = 0.01, gt = 0.0 → MSE = 0.0001 → PSNR = 40 dB
    gt = torch.zeros((2, 1, 8, 8))
    noisy = torch.stack(
        [
            torch.full((1, 8, 8), 0.1),
            torch.full((1, 8, 8), 0.01),
        ]
    )

    vals = psnr(noisy, gt, max_value=1.0, reduction="none")
    assert vals.shape == (2,)
    assert math.isclose(vals[0].item(), 20.0, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(vals[1].item(), 40.0, rel_tol=0, abs_tol=1e-4)

    val_mean = psnr(noisy, gt, max_value=1.0, reduction="mean")
    assert val_mean.ndim == 0
    assert math.isclose(val_mean.item(), 30.0, rel_tol=0, abs_tol=1e-4)

    val_sum = psnr(noisy, gt, max_value=1.0, reduction="sum")
    assert val_sum.ndim == 0
    assert math.isclose(val_sum.item(), 60.0, rel_tol=0, abs_tol=1e-4)


def test_psnr_input_validation():
    a = torch.zeros((1, 1, 8, 8))
    b = torch.zeros((1, 1, 8, 8))
    with pytest.raises(ValueError):
        _ = psnr(a, b, max_value=0.0)
    with pytest.raises(ValueError):
        _ = psnr(a, b, max_value=1.0, reduction="avg")  # type: ignore[arg-type]

    # Mismatched shapes
    a = torch.zeros((1, 1, 8, 8))
    b = torch.zeros((1, 1, 8, 7))
    with pytest.raises(ValueError):
        _ = psnr(a, b)
    # Wrong dimensionality (3D)
    c = torch.zeros((1, 8, 8))
    with pytest.raises(ValueError):
        _ = psnr(c, c)
