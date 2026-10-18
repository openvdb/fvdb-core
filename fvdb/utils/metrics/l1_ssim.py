# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from numbers import Real

import torch
from torch.autograd.function import once_differentiable


class _FusedL1SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1: torch.Tensor, img2: torch.Tensor, ssim_weight: float, train: bool, normalize_target: bool):
        loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = torch.ops.fvdb._fused_l1_ssim.default(
            img1, img2, ssim_weight, 0.01**2, 0.03**2, train, normalize_target
        )

        if train:
            ctx.save_for_backward(img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.ssim_weight = ssim_weight
        ctx.normalize_target = normalize_target
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor | None):
        if grad_loss is None:
            return None, None, None, None, None

        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        grad_img1 = torch.ops.fvdb._fused_l1_ssim_backward.default(
            img1,
            img2,
            grad_loss,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
            ctx.ssim_weight,
            0.01**2,
            0.03**2,
            ctx.normalize_target,
        )
        return grad_img1, None, None, None, None


def fused_l1_ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    ssim_weight: float,
    normalize_target: bool = False,
) -> torch.Tensor:
    """Compute mean L1 and SSIM losses with fused forward and backward kernels.

    Computes ``torch.lerp(fused_l1_loss(img1, img2), 1 - ssim(img1, img2),
    ssim_weight)`` using SSIM's default ``same`` padding. Both inputs must be
    nonempty NCHW images with identical shapes. Predictions must be float32 on
    CUDA or DGX; float32 or uint8 targets must be on the prediction device. When
    ``normalize_target=True``, target values are divided by 255 inside the
    forward and backward kernels without allocating a normalized target tensor.
    Noncontiguous inputs are copied to contiguous NCHW storage.

    Transfer CPU targets explicitly with
    ``torch.ops.fvdb._copy_image_to_device(img2, img1.device)`` before calling this
    function. The device target is saved for backward. The transfer operator uses
    ``to(img1.device)`` for CUDA; DGX uses asynchronous copies of each GPU's
    assigned tile rows into a DGX buffer, preserving NCHW storage. CPU targets for
    DGX must already be contiguous and pinned, and the caller must keep their
    storage alive and unchanged until the uploads finish.

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
        img2: Float32 or uint8 target images with the same shape and device as ``img1``.
        ssim_weight: Scalar interpolation weight for the SSIM loss.
        normalize_target: Divide target values by 255 before computing the loss.
            Defaults to ``False``, which uses target values unchanged.

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
    return _FusedL1SSIM.apply(img1.contiguous(), img2.contiguous(), float(ssim_weight), train, normalize_target)
