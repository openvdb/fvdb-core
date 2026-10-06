# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
``fvdb.functional`` -- Pure-functional API for sparse grid operations.

Every grid operation has two variants:

- ``*_batch`` -- operates on :class:`~fvdb.GridBatch` with :class:`~fvdb.JaggedTensor`.
- ``*_single`` -- operates on :class:`~fvdb.Grid` with plain ``torch.Tensor``.

The Gaussian splatting functions are the exception. They are flat wrappers over the CUDA kernels
with a single variant each, no autograd, and separate forward and backward entry points.
"""

# Grid constructors (batch)
from ._constructors import (
    concatenate_grids,
    gridbatch_from_dense,
    gridbatch_from_dense_axis_aligned_bounds,
    gridbatch_from_ijk,
    gridbatch_from_mesh,
    gridbatch_from_nearest_voxels_to_points,
    gridbatch_from_points,
    gridbatch_from_zero_grids,
    gridbatch_from_zero_voxels,
)

# isort: split

# Grid constructors (single)
from ._constructors import (
    grid_from_dense,
    grid_from_dense_axis_aligned_bounds,
    grid_from_ijk,
    grid_from_mesh,
    grid_from_nearest_voxels_to_points,
    grid_from_points,
    grid_from_zero_voxels,
)

# Dense <-> sparse I/O and grid-to-grid injection
from ._dense import (
    inject_batch,
    inject_from_dense_cmajor_batch,
    inject_from_dense_cmajor_single,
    inject_from_dense_cminor_batch,
    inject_from_dense_cminor_single,
    inject_from_ijk_batch,
    inject_from_ijk_single,
    inject_single,
    inject_to_dense_cmajor_batch,
    inject_to_dense_cmajor_single,
    inject_to_dense_cminor_batch,
    inject_to_dense_cminor_single,
)

# Grid indexing
from ._indexing import index_grid_batch

# Interpolation / splatting
from ._interpolation import (
    sample_bezier_batch,
    sample_bezier_single,
    sample_bezier_with_grad_batch,
    sample_bezier_with_grad_single,
    sample_nearest_batch,
    sample_nearest_single,
    sample_trilinear_batch,
    sample_trilinear_single,
    sample_trilinear_with_grad_batch,
    sample_trilinear_with_grad_single,
    splat_bezier_batch,
    splat_bezier_single,
    splat_trilinear_batch,
    splat_trilinear_single,
)

# I/O
from ._io import (
    grid_names_in_nanovdb,
    load_nanovdb,
    load_nanovdb_single,
    read_nanovdb_metadata,
    save_nanovdb,
    save_nanovdb_single,
)

# Meshing / TSDF
from ._meshing import (
    integrate_tsdf_batch,
    integrate_tsdf_single,
    integrate_tsdf_with_features_batch,
    integrate_tsdf_with_features_single,
    marching_cubes_batch,
    marching_cubes_single,
)

# Signed distance fields
from ._sdf import (
    reinitialize_sdf_batch,
    reinitialize_sdf_single,
    rebuild_narrow_band_batch,
    rebuild_narrow_band_single,
)

# Pooling / refinement
from ._pooling import (
    avg_pool_batch,
    avg_pool_single,
    max_pool_batch,
    max_pool_single,
    refine_batch,
    refine_single,
)

# Spatial queries
from ._query import (
    active_grid_coords_batch,
    active_grid_coords_single,
    coords_in_grid_batch,
    coords_in_grid_single,
    cubes_in_grid_batch,
    cubes_in_grid_single,
    cubes_intersect_grid_batch,
    cubes_intersect_grid_single,
    ijk_to_index_batch,
    ijk_to_index_single,
    ijk_to_inv_index_batch,
    ijk_to_inv_index_single,
    neighbor_indexes_batch,
    neighbor_indexes_single,
    points_in_grid_batch,
    points_in_grid_single,
)

# Ray operations
from ._ray import (
    ray_implicit_intersection_batch,
    ray_implicit_intersection_single,
    rays_intersect_voxels_batch,
    rays_intersect_voxels_single,
    segments_along_rays_batch,
    segments_along_rays_single,
    uniform_ray_samples_batch,
    uniform_ray_samples_single,
    voxels_along_rays_batch,
    voxels_along_rays_single,
)

# Grid topology
from ._topology import (
    clip_batch,
    clip_single,
    clipped_grid_batch,
    clipped_grid_single,
    clone_grid_batch,
    clone_grid_single,
    coarsened_grid_batch,
    coarsened_grid_single,
    contiguous_batch,
    contiguous_single,
    conv_grid_batch,
    conv_grid_single,
    conv_transpose_grid_batch,
    conv_transpose_grid_single,
    dilated_grid_batch,
    dilated_grid_single,
    dual_grid_batch,
    dual_grid_single,
    edge_network_batch,
    edge_network_single,
    hilbert_batch,
    hilbert_single,
    hilbert_zyx_batch,
    hilbert_zyx_single,
    merged_grid_batch,
    merged_grid_single,
    morton_batch,
    morton_single,
    morton_zyx_batch,
    morton_zyx_single,
    pruned_grid_batch,
    pruned_grid_single,
    refined_grid_batch,
    refined_grid_single,
)

# Coordinate transforms
from ._transforms import (
    voxel_to_world_batch,
    voxel_to_world_single,
    world_to_voxel_batch,
    world_to_voxel_single,
)

# Gaussian splatting kernels (flat, non-differentiable wrappers)
from ._gaussian_splatting import (
    as_pixel_jagged,
    build_sparse_gaussian_tile_layout,
    evaluate_spherical_harmonics_bwd,
    evaluate_spherical_harmonics_fwd,
    intersect_gaussian_tiles,
    intersect_gaussian_tiles_sparse,
    load_gaussian_ply,
    mcmc_add_noise_to_means,
    mcmc_relocate_gaussians,
    project_gaussians_analytic_bwd,
    project_gaussians_analytic_fwd,
    project_gaussians_analytic_jagged_bwd,
    project_gaussians_analytic_jagged_fwd,
    project_gaussians_ut_fwd,
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
    rasterize_screen_space_gaussians_bwd,
    rasterize_screen_space_gaussians_fwd,
    rasterize_screen_space_gaussians_sparse_bwd,
    rasterize_screen_space_gaussians_sparse_fwd,
    rasterize_top_contributing_gaussian_ids,
    rasterize_top_contributing_gaussian_ids_sparse,
    rasterize_world_space_gaussians_bwd,
    rasterize_world_space_gaussians_fwd,
    save_gaussian_ply,
)

__all__ = [
    # Interpolation (batch)
    "sample_nearest_batch",
    "sample_trilinear_batch",
    "sample_trilinear_with_grad_batch",
    "sample_bezier_batch",
    "sample_bezier_with_grad_batch",
    "splat_trilinear_batch",
    "splat_bezier_batch",
    # Interpolation (single)
    "sample_nearest_single",
    "sample_trilinear_single",
    "sample_trilinear_with_grad_single",
    "sample_bezier_single",
    "sample_bezier_with_grad_single",
    "splat_trilinear_single",
    "splat_bezier_single",
    # Transforms
    "voxel_to_world_batch",
    "voxel_to_world_single",
    "world_to_voxel_batch",
    "world_to_voxel_single",
    # Pooling
    "max_pool_batch",
    "max_pool_single",
    "avg_pool_batch",
    "avg_pool_single",
    "refine_batch",
    "refine_single",
    # Dense I/O
    "inject_from_dense_cminor_batch",
    "inject_from_dense_cminor_single",
    "inject_from_dense_cmajor_batch",
    "inject_from_dense_cmajor_single",
    "inject_from_ijk_batch",
    "inject_from_ijk_single",
    "inject_to_dense_cminor_batch",
    "inject_to_dense_cminor_single",
    "inject_to_dense_cmajor_batch",
    "inject_to_dense_cmajor_single",
    "inject_batch",
    "inject_single",
    # Queries
    "points_in_grid_batch",
    "points_in_grid_single",
    "coords_in_grid_batch",
    "coords_in_grid_single",
    "cubes_in_grid_batch",
    "cubes_in_grid_single",
    "cubes_intersect_grid_batch",
    "cubes_intersect_grid_single",
    "ijk_to_index_batch",
    "ijk_to_index_single",
    "ijk_to_inv_index_batch",
    "ijk_to_inv_index_single",
    "neighbor_indexes_batch",
    "neighbor_indexes_single",
    "active_grid_coords_batch",
    "active_grid_coords_single",
    # Rays
    "voxels_along_rays_batch",
    "voxels_along_rays_single",
    "rays_intersect_voxels_batch",
    "rays_intersect_voxels_single",
    "segments_along_rays_batch",
    "segments_along_rays_single",
    "uniform_ray_samples_batch",
    "uniform_ray_samples_single",
    "ray_implicit_intersection_batch",
    "ray_implicit_intersection_single",
    # Meshing
    "marching_cubes_batch",
    "marching_cubes_single",
    "integrate_tsdf_batch",
    "integrate_tsdf_single",
    "integrate_tsdf_with_features_batch",
    "integrate_tsdf_with_features_single",
    # Signed distance fields
    "rebuild_narrow_band_batch",
    "rebuild_narrow_band_single",
    "reinitialize_sdf_batch",
    "reinitialize_sdf_single",
    # Topology
    "clip_batch",
    "clip_single",
    "clipped_grid_batch",
    "clipped_grid_single",
    "clone_grid_batch",
    "clone_grid_single",
    "contiguous_batch",
    "contiguous_single",
    "coarsened_grid_batch",
    "coarsened_grid_single",
    "refined_grid_batch",
    "refined_grid_single",
    "dual_grid_batch",
    "dual_grid_single",
    "dilated_grid_batch",
    "dilated_grid_single",
    "merged_grid_batch",
    "merged_grid_single",
    "pruned_grid_batch",
    "pruned_grid_single",
    "conv_grid_batch",
    "conv_grid_single",
    "conv_transpose_grid_batch",
    "conv_transpose_grid_single",
    "morton_batch",
    "morton_single",
    "morton_zyx_batch",
    "morton_zyx_single",
    "hilbert_batch",
    "hilbert_single",
    "hilbert_zyx_batch",
    "hilbert_zyx_single",
    "edge_network_batch",
    "edge_network_single",
    # Indexing
    "index_grid_batch",
    # Constructors (batch)
    "gridbatch_from_dense",
    "gridbatch_from_dense_axis_aligned_bounds",
    "gridbatch_from_ijk",
    "gridbatch_from_mesh",
    "gridbatch_from_nearest_voxels_to_points",
    "gridbatch_from_points",
    "gridbatch_from_zero_grids",
    "gridbatch_from_zero_voxels",
    "concatenate_grids",
    # Constructors (single)
    "grid_from_dense",
    "grid_from_dense_axis_aligned_bounds",
    "grid_from_ijk",
    "grid_from_mesh",
    "grid_from_nearest_voxels_to_points",
    "grid_from_points",
    "grid_from_zero_voxels",
    # I/O
    "load_nanovdb",
    "load_nanovdb_single",
    "save_nanovdb",
    "save_nanovdb_single",
    "read_nanovdb_metadata",
    "grid_names_in_nanovdb",
    # Gaussian splatting: projection
    "project_gaussians_analytic_fwd",
    "project_gaussians_analytic_bwd",
    "project_gaussians_analytic_jagged_fwd",
    "project_gaussians_analytic_jagged_bwd",
    "project_gaussians_ut_fwd",
    # Gaussian splatting: spherical harmonics
    "evaluate_spherical_harmonics_fwd",
    "evaluate_spherical_harmonics_bwd",
    # Gaussian splatting: tile intersection
    "intersect_gaussian_tiles",
    "intersect_gaussian_tiles_sparse",
    "build_sparse_gaussian_tile_layout",
    "as_pixel_jagged",
    # Gaussian splatting: rasterization
    "rasterize_screen_space_gaussians_fwd",
    "rasterize_screen_space_gaussians_bwd",
    "rasterize_screen_space_gaussians_sparse_fwd",
    "rasterize_screen_space_gaussians_sparse_bwd",
    "rasterize_world_space_gaussians_fwd",
    "rasterize_world_space_gaussians_bwd",
    # Gaussian splatting: analysis
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
    "rasterize_top_contributing_gaussian_ids",
    "rasterize_top_contributing_gaussian_ids_sparse",
    # Gaussian splatting: MCMC
    "mcmc_relocate_gaussians",
    "mcmc_add_noise_to_means",
    # Gaussian splatting: PLY I/O
    "load_gaussian_ply",
    "save_gaussian_ply",
]
