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

# Projection
from ._projection import (
    project_gaussians,
    project_gaussians_for_camera,
)

# SH evaluation
from ._sh import (
    evaluate_spherical_harmonics,
)

# Rasterization
from ._rasterize import (
    rasterize_from_projected,
)

from ._rasterize_from_world import (
    rasterize_from_world,
)

from ._rasterize_sparse import (
    sparse_render,
)

# Composite rendering pipelines
from ._render import (
    render_images,
    render_depths,
    render_images_and_depths,
    render_images_from_world,
    render_depths_from_world,
    render_images_and_depths_from_world,
)

# Query operations
from ._queries import (
    render_num_contributing_gaussians,
    render_contributing_gaussian_ids,
    sparse_render_num_contributing_gaussians,
    sparse_render_contributing_gaussian_ids,
)

# Helpers
from ._tile_intersection import (
    build_render_settings,
)

__all__ = [
    # Projection
    "project_gaussians",
    "project_gaussians_for_camera",
    # SH evaluation
    "evaluate_spherical_harmonics",
    # Rasterization
    "rasterize_from_projected",
    "rasterize_from_world",
    "sparse_render",
    # Composite pipelines
    "render_images",
    "render_depths",
    "render_images_and_depths",
    "render_images_from_world",
    "render_depths_from_world",
    "render_images_and_depths_from_world",
    # Queries
    "render_num_contributing_gaussians",
    "render_contributing_gaussian_ids",
    "sparse_render_num_contributing_gaussians",
    "sparse_render_contributing_gaussian_ids",
    # Helpers
    "build_render_settings",
]
