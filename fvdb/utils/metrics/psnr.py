# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]


class PSNR(torch.nn.Module):
    """
    A torch Module for computing the Peak-Signal-to-Noise-Ratio (PSNR) betweeen two minibatches of images.

    Given a noiseless image, I, and noisy image K, with shape (C, H, W) the PSNR (in dB) is defined as
    .. math::
        10 \\log_{10}(\\text{max}(I)^2 / \\text{mse}(I, K))
    where:
        - :math:`\\text{max}(I)` is the maximum possible value the image can take on (e.g. 1.0 if pixel values are in [0, 1])
        - :math:`\\text{mse}(I, K) = \\sum_{c, h, w} (I_{c,h,w} - K_{c,h,w})^2 / (C * H * W)` is the mean-squared-error between the two images.
    """

    def __init__(self, max_value: float, reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Create a new PSNR module.

        Args:
            max_value (float): The maximum possible value images computed with this loss can have.
            reduction (Literal["none", "mean", "sum]): How to reduce over the batch dimension. "sum" and "mean" will
                add-up and average the losses across the batch respectively. "none" will return each loss as a seperate
                entry in the tensor. Default is "mean".
        """
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be one of ('none', 'mean', 'sum')")

        if max_value <= 0:
            raise ValueError("max_value must be a positive number")

        self._max_value = max_value
        self._log_max_value = 2.0 * np.log10(self._max_value)
        self._reduction = reduction

    def forward(self, noisy_images: torch.Tensor, ground_truth_images: torch.Tensor):
        """
        Compute the Peak-Signal-to-Noise-Ratio (PSNR) ratio between two batches of images.

        Args:
            noisy_images (torch.Tensor): A batch of noisy images of shape (B, C, H, W)
            ground_truth_images (torch.Tensor): A batch of ground truth images of shape (B, C, H, W)

        Returns:
            torch.Tensor: The PSNR between the two images (optionally reduced over the batch
                if reduction is not "none")
        """
        if (noisy_images.shape != ground_truth_images.shape) or (noisy_images.dim() != 4):
            raise ValueError("Input images must have the same shape and be 4-dimensional with shape (B, C, H, W)")

        mse = torch.mean((noisy_images - ground_truth_images) ** 2, dim=(1, 2, 3))  # [B]

        # Expand log of ratio to difference of logs for better stability
        psnr = 10.0 * (self._log_max_value - torch.log10(mse))
        if self._reduction == "none":
            return psnr
        elif self._reduction == "mean":
            return torch.mean(psnr)
        elif self._reduction == "sum":
            return torch.sum(psnr)
        else:
            raise RuntimeError(
                "Unreachable. PSNR reduction should always be one of 'sum', 'mean' or 'none'"
            )  # should never happen


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
            return each loss as a seperate entry in the tensor. Default is "mean".

    Returns:
        torch.Tensor: The PSNR between the two images (optionally reduced over the batch
            if reduction is not "none")
    """
    return PSNR(max_value, reduction)(noisy_images, ground_truth_images)
