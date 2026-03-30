# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
``fvdb.functional`` -- Functional API for fVDB sparse grid operations.

Every function in this module accepts a :class:`~fvdb.GridBatch` as its first
argument.  Data arguments (voxel features, query points, etc.) can be passed as
either ``torch.Tensor`` (single-grid convenience) or :class:`~fvdb.JaggedTensor`
(batched), and the return type matches the input type.  This mirrors
``torch.nn.functional`` and provides a pure-function alternative to the
equivalent methods on :class:`~fvdb.GridBatch`.
"""

# Interpolation
from ._interpolation import (
    sample_bezier,
    sample_bezier_with_grad,
    sample_trilinear,
    sample_trilinear_with_grad,
    splat_bezier,
    splat_trilinear,
)

# Coordinate transforms
from ._transforms import voxel_to_world, world_to_voxel

# Pooling / refinement
from ._pooling import avg_pool, max_pool, refine

# Dense <-> sparse I/O and grid-to-grid injection
from ._dense import (
    inject,
    inject_from_dense_cmajor,
    inject_from_dense_cminor,
    inject_from_ijk,
    inject_to_dense_cmajor,
    inject_to_dense_cminor,
)

# Spatial queries
from ._query import (
    active_grid_coords,
    coords_in_grid,
    cubes_in_grid,
    cubes_intersect_grid,
    ijk_to_index,
    ijk_to_inv_index,
    neighbor_indexes,
    points_in_grid,
)

# Ray operations
from ._ray import (
    ray_implicit_intersection,
    rays_intersect_voxels,
    segments_along_rays,
    uniform_ray_samples,
    voxels_along_rays,
)

# Meshing / TSDF
from ._meshing import integrate_tsdf, integrate_tsdf_with_features, marching_cubes

# Grid topology
from ._topology import (
    clip,
    clipped_grid,
    clone_grid,
    coarsened_grid,
    contiguous,
    conv_grid,
    conv_transpose_grid,
    dilated_grid,
    dual_grid,
    edge_network,
    hilbert,
    hilbert_zyx,
    merged_grid,
    morton,
    morton_zyx,
    pruned_grid,
    refined_grid,
)

# Grid indexing
from ._indexing import index_grid

__all__ = [
    # Interpolation
    "sample_trilinear",
    "sample_trilinear_with_grad",
    "sample_bezier",
    "sample_bezier_with_grad",
    "splat_trilinear",
    "splat_bezier",
    # Transforms
    "voxel_to_world",
    "world_to_voxel",
    # Pooling
    "max_pool",
    "avg_pool",
    "refine",
    # Dense I/O
    "inject_from_dense_cminor",
    "inject_from_dense_cmajor",
    "inject_from_ijk",
    "inject_to_dense_cminor",
    "inject_to_dense_cmajor",
    "inject",
    # Queries
    "points_in_grid",
    "coords_in_grid",
    "cubes_in_grid",
    "cubes_intersect_grid",
    "ijk_to_index",
    "ijk_to_inv_index",
    "neighbor_indexes",
    "active_grid_coords",
    # Rays
    "voxels_along_rays",
    "rays_intersect_voxels",
    "segments_along_rays",
    "uniform_ray_samples",
    "ray_implicit_intersection",
    # Meshing
    "marching_cubes",
    "integrate_tsdf",
    "integrate_tsdf_with_features",
    # Topology
    "clip",
    "clipped_grid",
    "clone_grid",
    "contiguous",
    "coarsened_grid",
    "refined_grid",
    "dual_grid",
    "dilated_grid",
    "merged_grid",
    "pruned_grid",
    "conv_grid",
    "conv_transpose_grid",
    "morton",
    "morton_zyx",
    "hilbert",
    "hilbert_zyx",
    "edge_network",
    # Indexing
    "index_grid",
]
