# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass
from typing import Literal

import torch


_RGB_RGB_RGB_SH_ORDERING = "rgb_rgb_rgb"
_RRR_GGG_BBB_SH_ORDERING = "rrr_ggg_bbb"
_VALID_SH_ORDERINGS = (_RGB_RGB_RGB_SH_ORDERING, _RRR_GGG_BBB_SH_ORDERING)
_ShOrdering = Literal["rgb_rgb_rgb", "rrr_ggg_bbb"]


@dataclass(frozen=True, slots=True, eq=False)
class GaussianSplatViewData:
    """Renderer-ready tensor data for a 3D Gaussian splat view.

    This is an immutable container, but it does not clone or make its tensors immutable. All tensors
    must be floating-point tensors on the same device with the same dtype and the same leading
    Gaussian dimension ``N``.

    ``sh_ordering`` describes the layout of the last two dimensions of ``sh0`` and ``shN``:

    - ``"rgb_rgb_rgb"`` uses shapes ``(N, 1, C)`` and ``(N, K - 1, C)``.
    - ``"rrr_ggg_bbb"`` uses shapes ``(N, C, 1)`` and ``(N, C, K - 1)``.

    Args:
        means: Gaussian means with shape ``(N, 3)``.
        quats: Gaussian quaternions with shape ``(N, 4)`` and component order ``(w, x, y, z)``.
        log_scales: Gaussian logarithmic scales with shape ``(N, 3)``.
        logit_opacities: Gaussian opacity logits with shape ``(N,)``.
        sh0: Zeroth-order spherical harmonics coefficients.
        shN: Higher-order spherical harmonics coefficients.
        sh_ordering: Spherical harmonics tensor layout.
    """

    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    logit_opacities: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor
    sh_ordering: _ShOrdering = _RGB_RGB_RGB_SH_ORDERING

    def __post_init__(self) -> None:
        tensors = {
            "means": self.means,
            "quats": self.quats,
            "log_scales": self.log_scales,
            "logit_opacities": self.logit_opacities,
            "sh0": self.sh0,
            "shN": self.shN,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if not tensor.dtype.is_floating_point:
                raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")

        if not isinstance(self.sh_ordering, str) or self.sh_ordering not in _VALID_SH_ORDERINGS:
            valid = ", ".join(repr(value) for value in _VALID_SH_ORDERINGS)
            raise ValueError(f"sh_ordering must be one of {valid}, got {self.sh_ordering!r}")
        canonical_sh_ordering = next(value for value in _VALID_SH_ORDERINGS if value == self.sh_ordering)
        object.__setattr__(self, "sh_ordering", canonical_sh_ordering)

        dtype = self.means.dtype
        device = self.means.device
        for name, tensor in tensors.items():
            if tensor.dtype != dtype:
                raise TypeError(f"All Gaussian splat tensors must have dtype {dtype}; {name} has dtype {tensor.dtype}")
            if tensor.device != device:
                raise ValueError(
                    f"All Gaussian splat tensors must be on device {device}; {name} is on device {tensor.device}"
                )

        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError(f"means must have shape (N, 3), got {tuple(self.means.shape)}")
        num_gaussians = self.means.shape[0]
        expected_shapes = {
            "quats": (num_gaussians, 4),
            "log_scales": (num_gaussians, 3),
            "logit_opacities": (num_gaussians,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(tensors[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")

        if self.sh0.ndim != 3:
            raise ValueError(f"sh0 must be a rank-3 tensor, got shape {tuple(self.sh0.shape)}")
        if self.shN.ndim != 3:
            raise ValueError(f"shN must be a rank-3 tensor, got shape {tuple(self.shN.shape)}")
        if self.sh0.shape[0] != num_gaussians:
            raise ValueError(
                f"sh0 must contain {num_gaussians} Gaussians in its leading dimension, got {self.sh0.shape[0]}"
            )
        if self.shN.shape[0] != num_gaussians:
            raise ValueError(
                f"shN must contain {num_gaussians} Gaussians in its leading dimension, got {self.shN.shape[0]}"
            )

        if self.sh_ordering == _RGB_RGB_RGB_SH_ORDERING:
            if self.sh0.shape[1] != 1:
                raise ValueError(
                    'sh0 must have shape (N, 1, C) for sh_ordering="rgb_rgb_rgb", '
                    f"got {tuple(self.sh0.shape)}"
                )
            if self.sh0.shape[2] == 0:
                raise ValueError("sh0 must contain at least one channel")
            if self.shN.shape[2] != self.sh0.shape[2]:
                raise ValueError(
                    f"sh0 and shN must have the same channel dimension, got {self.sh0.shape[2]} and {self.shN.shape[2]}"
                )
        else:
            if self.sh0.shape[2] != 1:
                raise ValueError(
                    'sh0 must have shape (N, C, 1) for sh_ordering="rrr_ggg_bbb", '
                    f"got {tuple(self.sh0.shape)}"
                )
            if self.sh0.shape[1] == 0:
                raise ValueError("sh0 must contain at least one channel")
            if self.shN.shape[1] != self.sh0.shape[1]:
                raise ValueError(
                    f"sh0 and shN must have the same channel dimension, got {self.sh0.shape[1]} and {self.shN.shape[1]}"
                )
