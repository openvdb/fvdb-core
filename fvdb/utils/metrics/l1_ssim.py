# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from numbers import Real

import torch
from torch.autograd.function import once_differentiable


class _FusedL1SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1: torch.Tensor, img2: torch.Tensor, ssim_weight: float, train: bool):
        loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = torch.ops.fvdb._fused_l1_ssim.default(
            img1, img2, ssim_weight, 0.01**2, 0.03**2, train
        )

        if train:
            ctx.save_for_backward(img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.ssim_weight = ssim_weight
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor | None):
        if grad_loss is None:
            return None, None, None, None

        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        grad_img1 = torch.ops.fvdb._fused_l1_ssim_backward.default(
            img1, img2, grad_loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, ctx.ssim_weight, 0.01**2, 0.03**2
        )
        return grad_img1, None, None, None


def fused_l1_ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    """Compute mean L1 and SSIM losses with fused forward and backward kernels.

    Computes ``torch.lerp(fused_l1_loss(img1, img2), 1 - ssim(img1, img2),
    ssim_weight)`` using SSIM's default ``same`` padding. Both inputs must be
    nonempty float32 NCHW images with identical shapes on the same CUDA or DGX
    device. Noncontiguous inputs are copied to contiguous NCHW storage.

    Forward computes L1 and SSIM together and reduces each 16x16 tile over all
    channels using SSIM's spatial row ownership, without allocating loss maps.
    Tile contributions use the full image element count for normalization.
    Reductions exchange only scalar sums between GPUs. The three SSIM derivative
    tensors are saved for backward, which applies the upstream scalar, computes
    both gradient contributions, and writes their weighted sum in a single kernel.
    Only first-order gradients for ``img1`` are supported; ``img2`` must not
    require gradients.

    Args:
        img1: Predicted images of shape ``(B, C, H, W)``.
        img2: Target images with the same shape, dtype, and device.
        ssim_weight: Scalar interpolation weight for the SSIM loss.

    Returns:
        A scalar tensor containing the aggregate L1 and SSIM loss.
    """
    if img2.requires_grad:
        raise ValueError("fused_l1_ssim_loss does not support target gradients")
    if not img1.numel() or not img2.numel():
        raise ValueError("fused_l1_ssim_loss requires nonempty image batches")
    if not isinstance(ssim_weight, Real):
        raise TypeError("ssim_weight must be a real scalar")

    train = torch.is_grad_enabled() and img1.requires_grad
    return _FusedL1SSIM.apply(img1.contiguous(), img2.contiguous(), float(ssim_weight), train)
