Functional Gaussian Splatting API
=================================

.. module:: fvdb.functional.splat

The :mod:`fvdb.functional.splat` module provides a pure-function interface for
Gaussian splatting rendering.  Every operation is a standalone function that
takes raw tensors as input, following the same design philosophy as
:mod:`fvdb.functional` for sparse-grid operations.

The API is organized into two layers:

- **Decomposed pipeline stages** -- individual pure functions that can be
  composed into custom rendering pipelines.  Each stage takes raw tensors
  and returns either a frozen dataclass or a tensor.
- **Convenience functions** -- higher-level functions that compose the stages
  internally, matching the methods on :class:`~fvdb.GaussianSplat3d`.

.. tip::

   For standard rendering, the convenience functions (``render_images``,
   ``render_depths``, etc.) or the methods on :class:`~fvdb.GaussianSplat3d`
   are the simplest entry points.

   The decomposed stages are for users who need fine-grained control over the
   rendering pipeline -- for example, to insert custom logic between projection
   and rasterization, or to build training loops without the
   :class:`~fvdb.GaussianSplat3d` wrapper.


Decomposed Pipeline Stages
---------------------------

The full differentiable render pipeline decomposes into five stages.
Each is a pure function with no side effects:

.. code-block:: text

   means, quats, log_scales        logit_opacities    means, sh0, shN
         │                               │                   │
         ▼                               ▼                   ▼
   ┌─────────────┐              ┌────────────────┐   ┌─────────────────┐
   │project_to_2d│              │compute_opacities│  │prepare_features │
   └──────┬──────┘              └───────┬────────┘   └───────┬─────────┘
          │ RawProjection               │ [C,N]              │ [C,N,D]
          │                             │                    │
          ├─────────────┐               │                    │
          ▼             ▼               ▼                    ▼
   ┌──────────────┐  ┌──────────────────────────────────────────┐
   │intersect_tiles│ │          rasterize_dense                  │
   └──────┬───────┘  └───────────────────┬──────────────────────┘
          │ TileIntersection             │
          │                              ▼
          └─────────────────────► (images, alphas)


**Example: building a custom render pipeline**

.. code-block:: python

   import torch
   import fvdb.functional.splat as splat
   from fvdb._fvdb_cpp import CameraModel

   # Raw tensors (no GaussianSplat3d needed)
   means = ...        # [N, 3]
   quats = ...        # [N, 4]
   log_scales = ...   # [N, 3]
   logit_opacities = ...  # [N]
   sh0 = ...          # [N, 1, 3]
   shN = ...          # [N, K-1, 3]
   world_to_cam = ... # [C, 4, 4]
   K = ...            # [C, 3, 3]

   # Stage 1: Geometric projection
   raw = splat.project_to_2d(
       means, quats, log_scales, world_to_cam, K,
       image_width=640, image_height=480,
       eps_2d=0.3, near=0.01, far=1e10,
       radius_clip=0.0, antialias=False,
       camera_model=CameraModel.PINHOLE,
   )

   # Stage 2: Opacities
   C = world_to_cam.size(0)
   opacities = splat.compute_opacities(logit_opacities, C, raw.compensations)

   # Stage 3: View-dependent features (SH evaluation)
   features = splat.prepare_render_features(
       means, sh0, shN, world_to_cam,
       raw.radii, raw.depths,
       sh_degree_to_use=3, render_mode="rgb",
   )

   # Stage 4: Tile intersection
   tiles = splat.intersect_tiles(
       raw.means2d, raw.radii, raw.depths, C,
       tile_size=16, image_width=640, image_height=480,
   )

   # Stage 5: Rasterize
   images, alphas = splat.rasterize_dense(
       raw.means2d, raw.conics, features, opacities,
       tiles.tile_offsets, tiles.tile_gaussian_ids,
       image_width=640, image_height=480,
   )

   # Compute loss and backpropagate -- gradients flow through all stages
   loss = torch.nn.functional.l1_loss(images, target_images)
   loss.backward()


Stage 1: Projection
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RawProjection
   :members:

.. autofunction:: project_to_2d


Stage 2: Opacities
^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_opacities


Stage 3: Render Features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: prepare_render_features


Stage 4: Tile Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TileIntersection
   :members:

.. autofunction:: intersect_tiles


Stage 5: Rasterization
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rasterize_dense


Convenience Functions
---------------------

These higher-level functions compose the decomposed stages internally.
They match the interface of the corresponding methods on
:class:`~fvdb.GaussianSplat3d`.


Monolith Projection
^^^^^^^^^^^^^^^^^^^^

These project, evaluate SH, compute opacities, and intersect tiles in a
single call, returning a :class:`~fvdb.ProjectedGaussianSplats` object.

.. autofunction:: project_gaussians
.. autofunction:: project_gaussians_for_camera


Spherical Harmonics
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: evaluate_spherical_harmonics


Monolith Rasterization
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rasterize_from_projected
.. autofunction:: rasterize_from_world
.. autofunction:: sparse_render


Composite Rendering Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: render_images
.. autofunction:: render_depths
.. autofunction:: render_images_and_depths
.. autofunction:: render_images_from_world
.. autofunction:: render_depths_from_world
.. autofunction:: render_images_and_depths_from_world


Query Operations
-----------------

.. autofunction:: render_num_contributing_gaussians
.. autofunction:: render_contributing_gaussian_ids
.. autofunction:: sparse_render_num_contributing_gaussians
.. autofunction:: sparse_render_contributing_gaussian_ids


Helpers
--------

.. autofunction:: build_render_settings
