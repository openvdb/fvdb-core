# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd.function import once_differentiable


class _FusedL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fvdb._fused_l1_loss.default(img1, img2)
        ctx.save_for_backward(img1, img2)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor):
        img1, img2 = ctx.saved_tensors
        return torch.ops.fvdb._fused_l1_loss_backward.default(
            img1, img2, grad_loss, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )


def fused_l1_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute the mean absolute error between two float32 NCHW image batches.

    The inputs must have the same shape and be on the same CUDA or DGX device.
    Noncontiguous inputs are copied to contiguous NCHW storage. Broadcasting is
    not supported. An empty image batch returns NaN, matching a mean reduction.

    On DGX, the forward and backward image kernels use the same spatial tile-row
    ownership as :func:`ssim`. Forward writes a per-pixel absolute-difference map,
    then reduces tiles over all channels on their owning GPU to compute the mean.
    Backward writes gradients directly without materializing residual or sign
    images. First-order gradients are supported for either or both inputs.

    Args:
        img1: Predicted images of shape ``(B, C, H, W)`` and dtype ``torch.float32``.
        img2: Target images with the same shape, dtype, and device as ``img1``.

    Returns:
        A scalar tensor containing the mean absolute error over all elements.
    """
    return _FusedL1.apply(img1.contiguous(), img2.contiguous())
