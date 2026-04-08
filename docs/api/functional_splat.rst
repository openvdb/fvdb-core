Functional Gaussian Splatting API
=================================

.. module:: fvdb.functional

The Gaussian splatting functions in :mod:`fvdb.functional` provide a
pure-function interface for Gaussian splatting rendering.  Every operation is a
standalone function that takes raw tensors as input, following the same design
philosophy as the rest of :mod:`fvdb.functional` for sparse-grid operations.

The API is organized as a **4-stage composable pipeline**:

1. ``project_gaussians`` -- geometric projection (Stage 1)
2. ``evaluate_gaussian_sh`` -- SH / feature evaluation (Stage 2)
3. ``intersect_gaussian_tiles`` / ``intersect_gaussian_tiles_sparse`` -- tile intersection (Stage 3)
4. ``rasterize_screen_space_gaussians`` / ``rasterize_world_space_gaussians``
   / ``rasterize_screen_space_gaussians_sparse`` -- rasterization (Stage 4)

All stages are fully differentiable via Python autograd (except tile intersection).

.. tip::

   For standard rendering, the methods on :class:`~fvdb.GaussianSplat3d`
   are the simplest entry points.

   The decomposed stages are for users who need fine-grained control over the
   rendering pipeline -- for example, to insert custom logic between projection
   and rasterization, or to build training loops without the
   :class:`~fvdb.GaussianSplat3d` wrapper.


**Example: building a custom render pipeline**

.. code-block:: python

   import torch
   import fvdb.functional as F
   from fvdb.enums import CameraModel, GaussianRenderMode

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
   projected = F.project_gaussians(
       means, quats, log_scales, world_to_cam, K,
       image_width=640, image_height=480,
   )

   # Stage 2: View-dependent features (SH evaluation)
   features = F.evaluate_gaussian_sh(
       means, sh0, shN, world_to_cam, projected,
       sh_degree_to_use=3,
       render_mode=GaussianRenderMode.FEATURES,
   )

   # Stage 3: Tile intersection
   tiles = F.intersect_gaussian_tiles(projected)

   # Stage 4: Rasterize
   images, alphas = F.rasterize_screen_space_gaussians(
       projected, features, logit_opacities, tiles,
   )

   # Compute loss and backpropagate -- gradients flow through all stages
   loss = torch.nn.functional.l1_loss(images, target_images)
   loss.backward()


Types
------

.. autoclass:: ProjectedGaussians
   :members:

.. autoclass:: GaussianTileIntersection
   :members:

.. autoclass:: SparseGaussianTileIntersection
   :members:


Stage 1: Projection
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: project_gaussians


Stage 2: SH / Feature Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: evaluate_gaussian_sh


Stage 3: Tile Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: intersect_gaussian_tiles

.. autofunction:: intersect_gaussian_tiles_sparse


Stage 4: Rasterization
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rasterize_screen_space_gaussians

.. autofunction:: rasterize_world_space_gaussians

.. autofunction:: rasterize_screen_space_gaussians_sparse


Analysis
---------

.. autofunction:: count_contributing_gaussians

.. autofunction:: identify_contributing_gaussians

.. autofunction:: count_contributing_gaussians_sparse

.. autofunction:: identify_contributing_gaussians_sparse
