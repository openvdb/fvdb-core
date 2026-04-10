# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# This file contains source code from the fused-ssim library obtained from
# https://github.com/rahul-goel/fused-ssim. The fused-ssim library is licensed under the MIT
# License. Refer to ORSB 5512107 for more. Original license text follows.

# Copyright (c) 2024 Rahul Goel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Image-quality metrics: PSNR and SSIM."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .. import _fvdb_cpp as _fvdb_cpp  # noqa: F401  -- loads the custom C++ ops used by torch.ops.fvdb below

_ALLOWED_PADDING = ("same", "valid")


# ---------------------------------------------------------------------------
#  SSIM (fused CUDA implementation)
# ---------------------------------------------------------------------------


class _FusedSSIMFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):  # type: ignore[override]
        (
            ssim_map,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        ) = torch.ops.fvdb._fused_ssim.default(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):  # type: ignore[override]
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = torch.ops.fvdb._fused_ssim_backward.default(
            C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
        )
        return None, None, grad, None, None, None


def _fused_ssim(img1: torch.Tensor, img2: torch.Tensor, padding: str = "same", train: bool = True) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2

    if padding not in _ALLOWED_PADDING:
        raise ValueError(f"padding must be one of {_ALLOWED_PADDING}, got {padding!r}")

    img1 = img1.contiguous()
    ssim_map = _FusedSSIMFn.apply(C1, C2, img1, img2, padding, train)
    return ssim_map.mean()  # type: ignore[union-attr]


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: Literal["same", "valid"] = "same",
    train: bool = True,
) -> torch.Tensor:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): A batch of images of shape ``(B, C, H, W)``
        img2 (torch.Tensor): A batch of images of shape ``(B, C, H, W)``
        padding (str): The padding to use for the images (``"same"`` or ``"valid"``). Default is ``"same"``.
        train (bool): Whether or not to compute the gradients through the SSIM loss. Default is ``True``.

    Returns:
        ssim (torch.Tensor): The average SSIM between each image over the batch.
    """
    return _fused_ssim(img1, img2, padding, train)


# ---------------------------------------------------------------------------
#  PSNR
# ---------------------------------------------------------------------------


def psnr(
    noisy_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
    max_value: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Compute the Peak-Signal-to-Noise-Ratio (PSNR) ratio between two batches of images.

    Args:
        noisy_images (torch.Tensor): A batch of noisy images of shape ``(B, C, H, W)``
        ground_truth_images (torch.Tensor): A batch of ground truth images of shape ``(B, C, H, W)``
        max_value (float): The maximum possible value images computed with this loss can have.
            Default is 1.0.
        reduction (Literal["none", "mean", "sum"]): How to reduce over the batch dimension. ``"sum"``
            and ``"mean"`` will add-up and average the losses across the batch respectively. ``"none"`` will
            return each loss as a separate entry in the tensor. Default is ``"mean"``.

    Returns:
        psnr (torch.Tensor): The PSNR between the two images. If reduction is not "none", the result
            will be reduced over the batch dimension (*i.e.*  will be a single scalar), otherwise it will
            be a tensor of shape ``(B,)``.
    """
    if max_value <= 0:
        raise ValueError("max_value must be a positive number")

    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be one of ('none', 'mean', 'sum')")

    if (noisy_images.shape != ground_truth_images.shape) or (noisy_images.dim() != 4):
        raise ValueError("Input images must have the same shape and be 4-dimensional with shape (B, C, H, W)")

    mse = torch.mean((noisy_images - ground_truth_images) ** 2, dim=(1, 2, 3))  # [B]

    psnr_val = 10.0 * (2.0 * math.log10(max_value) - torch.log10(mse))
    if reduction == "none":
        return psnr_val
    elif reduction == "mean":
        return torch.mean(psnr_val)
    elif reduction == "sum":
        return torch.sum(psnr_val)
    raise ValueError("reduction must be one of ('none', 'mean', 'sum')")
