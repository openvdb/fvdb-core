Functional API
==============

.. module:: fvdb.functional

The :mod:`fvdb.functional` module provides a pure-function alternative to
the equivalent methods on :class:`~fvdb.Grid` and :class:`~fvdb.GridBatch`.
Every operation is available as a standalone function that takes the grid as
its first argument, mirroring the design of :mod:`torch.nn.functional`.

Each grid operation has two variants:

- ``*_batch`` -- operates on a :class:`~fvdb.GridBatch` with :class:`~fvdb.JaggedTensor` data.
- ``*_single`` -- operates on a :class:`~fvdb.Grid` with plain :class:`torch.Tensor` data.

The :ref:`Gaussian splatting kernels <functional-gaussian-splatting>` at the end of this page are
the exception: each is a single flat wrapper over a CUDA kernel.

.. tip::

   Most users should prefer the methods on :class:`~fvdb.Grid` and
   :class:`~fvdb.GridBatch` directly.  The functional API is useful when
   building custom operations or when you need explicit control over the
   single-grid vs. batched code path.


Coordinate Transforms
---------------------

.. autofunction:: voxel_to_world_batch
.. autofunction:: voxel_to_world_single
.. autofunction:: world_to_voxel_batch
.. autofunction:: world_to_voxel_single


Interpolation and Splatting
---------------------------

.. autofunction:: sample_nearest_batch
.. autofunction:: sample_nearest_single
.. autofunction:: sample_trilinear_batch
.. autofunction:: sample_trilinear_single
.. autofunction:: sample_trilinear_with_grad_batch
.. autofunction:: sample_trilinear_with_grad_single
.. autofunction:: sample_bezier_batch
.. autofunction:: sample_bezier_single
.. autofunction:: sample_bezier_with_grad_batch
.. autofunction:: sample_bezier_with_grad_single
.. autofunction:: splat_trilinear_batch
.. autofunction:: splat_trilinear_single
.. autofunction:: splat_bezier_batch
.. autofunction:: splat_bezier_single


Pooling and Refinement
----------------------

.. autofunction:: max_pool_batch
.. autofunction:: max_pool_single
.. autofunction:: avg_pool_batch
.. autofunction:: avg_pool_single
.. autofunction:: refine_batch
.. autofunction:: refine_single


Spatial Queries
---------------

.. autofunction:: points_in_grid_batch
.. autofunction:: points_in_grid_single
.. autofunction:: coords_in_grid_batch
.. autofunction:: coords_in_grid_single
.. autofunction:: cubes_in_grid_batch
.. autofunction:: cubes_in_grid_single
.. autofunction:: cubes_intersect_grid_batch
.. autofunction:: cubes_intersect_grid_single
.. autofunction:: ijk_to_index_batch
.. autofunction:: ijk_to_index_single
.. autofunction:: ijk_to_inv_index_batch
.. autofunction:: ijk_to_inv_index_single
.. autofunction:: neighbor_indexes_batch
.. autofunction:: neighbor_indexes_single
.. autofunction:: active_grid_coords_batch
.. autofunction:: active_grid_coords_single


Dense--Sparse Conversion
------------------------

.. autofunction:: inject_from_dense_cminor_batch
.. autofunction:: inject_from_dense_cminor_single
.. autofunction:: inject_from_dense_cmajor_batch
.. autofunction:: inject_from_dense_cmajor_single
.. autofunction:: inject_to_dense_cminor_batch
.. autofunction:: inject_to_dense_cminor_single
.. autofunction:: inject_to_dense_cmajor_batch
.. autofunction:: inject_to_dense_cmajor_single


Grid-to-Grid Injection
----------------------

.. autofunction:: inject_batch
.. autofunction:: inject_single
.. autofunction:: inject_from_ijk_batch
.. autofunction:: inject_from_ijk_single


Ray Operations
--------------

.. autofunction:: voxels_along_rays_batch
.. autofunction:: voxels_along_rays_single
.. autofunction:: segments_along_rays_batch
.. autofunction:: segments_along_rays_single
.. autofunction:: uniform_ray_samples_batch
.. autofunction:: uniform_ray_samples_single
.. autofunction:: ray_implicit_intersection_batch
.. autofunction:: ray_implicit_intersection_single
.. autofunction:: rays_intersect_voxels_batch
.. autofunction:: rays_intersect_voxels_single


Meshing and TSDF Integration
----------------------------

.. autofunction:: marching_cubes_batch
.. autofunction:: marching_cubes_single
.. autofunction:: integrate_tsdf_batch
.. autofunction:: integrate_tsdf_single
.. autofunction:: integrate_tsdf_with_features_batch
.. autofunction:: integrate_tsdf_with_features_single


Signed Distance Fields
----------------------

.. autofunction:: reinitialize_sdf_batch
.. autofunction:: reinitialize_sdf_single
.. autofunction:: rebuild_narrow_band_batch
.. autofunction:: rebuild_narrow_band_single


Grid Topology
-------------

.. autofunction:: coarsened_grid_batch
.. autofunction:: coarsened_grid_single
.. autofunction:: refined_grid_batch
.. autofunction:: refined_grid_single
.. autofunction:: dual_grid_batch
.. autofunction:: dual_grid_single
.. autofunction:: dilated_grid_batch
.. autofunction:: dilated_grid_single
.. autofunction:: merged_grid_batch
.. autofunction:: merged_grid_single
.. autofunction:: pruned_grid_batch
.. autofunction:: pruned_grid_single
.. autofunction:: clipped_grid_batch
.. autofunction:: clipped_grid_single
.. autofunction:: clip_batch
.. autofunction:: clip_single
.. autofunction:: contiguous_batch
.. autofunction:: contiguous_single
.. autofunction:: clone_grid_batch
.. autofunction:: clone_grid_single
.. autofunction:: conv_grid_batch
.. autofunction:: conv_grid_single
.. autofunction:: conv_transpose_grid_batch
.. autofunction:: conv_transpose_grid_single


Space-Filling Curves
--------------------

.. autofunction:: morton_batch
.. autofunction:: morton_single
.. autofunction:: morton_zyx_batch
.. autofunction:: morton_zyx_single
.. autofunction:: hilbert_batch
.. autofunction:: hilbert_single
.. autofunction:: hilbert_zyx_batch
.. autofunction:: hilbert_zyx_single


Edge Network
------------

.. autofunction:: edge_network_batch
.. autofunction:: edge_network_single


Grid Indexing
-------------

.. autofunction:: index_grid_batch


Grid Constructors (Batch)
-------------------------

.. autofunction:: gridbatch_from_dense
.. autofunction:: gridbatch_from_dense_axis_aligned_bounds
.. autofunction:: gridbatch_from_ijk
.. autofunction:: gridbatch_from_mesh
.. autofunction:: gridbatch_from_nearest_voxels_to_points
.. autofunction:: gridbatch_from_points
.. autofunction:: gridbatch_from_zero_grids
.. autofunction:: gridbatch_from_zero_voxels
.. autofunction:: concatenate_grids


Grid Constructors (Single)
---------------------------

.. autofunction:: grid_from_dense
.. autofunction:: grid_from_dense_axis_aligned_bounds
.. autofunction:: grid_from_ijk
.. autofunction:: grid_from_mesh
.. autofunction:: grid_from_nearest_voxels_to_points
.. autofunction:: grid_from_points
.. autofunction:: grid_from_zero_voxels


I/O
---

.. autofunction:: load_nanovdb
.. autofunction:: load_nanovdb_single
.. autofunction:: save_nanovdb
.. autofunction:: save_nanovdb_single
.. autofunction:: read_nanovdb_metadata
.. autofunction:: grid_names_in_nanovdb


.. _functional-gaussian-splatting:

Gaussian Splatting
------------------

Flat wrappers over the Gaussian splatting CUDA kernels, one function per kernel entry point. These
functions do not build an autograd graph. Forward and backward kernels are exposed separately so
that a downstream package can attach its own :class:`torch.autograd.Function` classes and compose
the stages into a differentiable pipeline.

Conventions: ``C`` is the number of cameras, ``N`` the number of Gaussians and ``D`` the feature
channel count. Pixel coordinates have their origin at the top-left corner of the image. Sparse
variants take ``pixels_to_render`` as a :class:`~fvdb.JaggedTensor` with one ``[P_c, 2]`` list of
integer ``(row, col)`` coordinates per camera. Camera enums are :class:`~fvdb.CameraModel` and
:class:`~fvdb.RollingShutterType`.

Projection
~~~~~~~~~~

.. autofunction:: project_gaussians_analytic_fwd
.. autofunction:: project_gaussians_analytic_bwd
.. autofunction:: project_gaussians_analytic_jagged_fwd
.. autofunction:: project_gaussians_analytic_jagged_bwd
.. autofunction:: project_gaussians_ut_fwd

Spherical Harmonics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: evaluate_spherical_harmonics_fwd
.. autofunction:: evaluate_spherical_harmonics_bwd

Tile Intersection
~~~~~~~~~~~~~~~~~

.. autofunction:: intersect_gaussian_tiles
.. autofunction:: intersect_gaussian_tiles_sparse
.. autofunction:: build_sparse_gaussian_tile_layout
.. autofunction:: as_pixel_jagged

Rasterization
~~~~~~~~~~~~~

.. autofunction:: rasterize_screen_space_gaussians_fwd
.. autofunction:: rasterize_screen_space_gaussians_bwd
.. autofunction:: rasterize_screen_space_gaussians_sparse_fwd
.. autofunction:: rasterize_screen_space_gaussians_sparse_bwd
.. autofunction:: rasterize_world_space_gaussians_fwd
.. autofunction:: rasterize_world_space_gaussians_bwd

Analysis
~~~~~~~~

.. autofunction:: rasterize_num_contributing_gaussians
.. autofunction:: rasterize_num_contributing_gaussians_sparse
.. autofunction:: rasterize_contributing_gaussian_ids
.. autofunction:: rasterize_contributing_gaussian_ids_sparse
.. autofunction:: rasterize_top_contributing_gaussian_ids
.. autofunction:: rasterize_top_contributing_gaussian_ids_sparse

MCMC Densification
~~~~~~~~~~~~~~~~~~

.. autofunction:: mcmc_relocate_gaussians
.. autofunction:: mcmc_add_noise_to_means

PLY I/O
~~~~~~~

.. autofunction:: save_gaussian_ply
.. autofunction:: load_gaussian_ply
