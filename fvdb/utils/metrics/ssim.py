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
#


from typing import NamedTuple

import torch

import fvdb

allowed_padding = ["same", "valid"]


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
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
    def backward(ctx, opt_grad):
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


def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01**2
    C2 = 0.03**2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()  # type: ignore


from typing import Literal


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
    return fused_ssim(img1, img2, padding, train)
