# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import torch


def psnr(
    noisy_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
    max_value: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Compute the Peak-Signal-to-Noise-Ratio (PSNR) ratio between two batches of images.

    Args:
        noisy_images (torch.Tensor): A batch of noisy images of shape (B, C, H, W)
        ground_truth_images (torch.Tensor): A batch of ground truth images of shape (B, C, H, W)
        max_value (float): The maximum possible value images computed with this loss can have.
            Default is 1.0.
        reduction (Literal["none", "mean", "sum"]): How to reduce over the batch dimension. "sum"
            and "mean" will add-up and average the losses across the batch respectively. "none" will
            return each loss as a separate entry in the tensor. Default is "mean".

    Returns:
        torch.Tensor: The PSNR between the two images (optionally reduced over the batch
            if reduction is not "none")
    """
    if max_value <= 0:
        raise ValueError("max_value must be a positive number")

    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be one of ('none', 'mean', 'sum')")

    if (noisy_images.shape != ground_truth_images.shape) or (noisy_images.dim() != 4):
        raise ValueError("Input images must have the same shape and be 4-dimensional with shape (B, C, H, W)")

    mse = torch.mean((noisy_images - ground_truth_images) ** 2, dim=(1, 2, 3))  # [B]

    # Expand log of ratio to difference of logs for better stability
    psnr = 10.0 * (2.0 * math.log10(max_value) - torch.log10(mse))
    if reduction == "none":
        return psnr
    elif reduction == "mean":
        return torch.mean(psnr)
    elif reduction == "sum":
        return torch.sum(psnr)
