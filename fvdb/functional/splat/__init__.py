# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
``fvdb.functional.splat`` -- Pure-functional Gaussian splatting pipeline.

This module decomposes Gaussian splatting rendering into individually callable,
composable stages.  Each stage is a pure function: it takes tensors in, returns
tensors (or a frozen dataclass) out, and has no hidden state or side effects.

**Design philosophy.**  The object-oriented :class:`~fvdb.GaussianSplat3d` class
orchestrates projection, SH evaluation, opacity computation, tile intersection,
and rasterization internally, managing mutable accumulator state along the way.
This module extracts that same logic into five independent stages so that:

- Custom pipelines can insert logic between stages (e.g. custom opacity
  schedules, per-Gaussian masking, or alternative feature representations).
- Training loops can be built from raw tensors and a standard
  ``torch.optim`` optimizer, with no wrapper class required.
- Each stage is independently testable and type-checked.
- Intermediate results have explicit, frozen types (``RawProjection``,
  ``TileIntersection``) rather than opaque C++ objects.

**Two layers.**  The module provides:

1. *Decomposed stages* -- ``project_to_2d``, ``compute_opacities``,
   ``prepare_render_features``, ``intersect_tiles``, ``rasterize_dense``.
2. *Convenience functions* -- ``render_images``, ``render_depths``, etc.,
   which compose the stages internally and match the
   :class:`~fvdb.GaussianSplat3d` method signatures.

Both layers are fully differentiable via Python autograd.
"""

# Projection result types
from ._projected_gaussians import (
    ProjectedGaussians,
    SparseProjectedGaussians,
)

# Projection
from ._projection import (
    RawProjection,
    project_gaussians,
    project_gaussians_for_camera,
    project_to_2d,
)

# Opacities
from ._opacities import (
    compute_opacities,
)

# SH evaluation
from ._sh import (
    evaluate_spherical_harmonics,
    prepare_render_features,
)

# Rasterization
from ._rasterize import (
    rasterize_dense,
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
    TileIntersection,
    build_render_settings,
    intersect_tiles,
)

__all__ = [
    # Projection result types
    "ProjectedGaussians",
    "SparseProjectedGaussians",
    # Projection (monolith convenience)
    "project_gaussians",
    "project_gaussians_for_camera",
    # Projection (decomposed stage)
    "RawProjection",
    "project_to_2d",
    # Opacities (decomposed stage)
    "compute_opacities",
    # SH evaluation
    "evaluate_spherical_harmonics",
    "prepare_render_features",
    # Rasterization (decomposed stage)
    "rasterize_dense",
    # Rasterization (monolith convenience)
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
    # Tile intersection (decomposed stage)
    "TileIntersection",
    "intersect_tiles",
    # Helpers
    "build_render_settings",
]
