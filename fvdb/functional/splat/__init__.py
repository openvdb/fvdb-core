# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
``fvdb.functional.splat`` -- Pure-functional API for Gaussian splatting operations.

All functions accept raw tensors (means, quats, log_scales, etc.) rather than
a GaussianSplat3d object, enabling use without the OO wrapper.

Unlike the grid functional API, there is no batch/single split since
GaussianSplat3d scenes are always single.
"""

from ._projection import (
    project_gaussians,
    project_gaussians_for_camera,
)

from ._sh import (
    evaluate_spherical_harmonics,
)

from ._tile_intersection import (
    build_render_settings,
)

__all__ = [
    "project_gaussians",
    "project_gaussians_for_camera",
    "evaluate_spherical_harmonics",
    "build_render_settings",
]
