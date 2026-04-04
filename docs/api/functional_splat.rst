Functional Gaussian Splatting API
=================================

.. module:: fvdb.functional.splat

The :mod:`fvdb.functional.splat` module provides a pure-function interface for
Gaussian splatting rendering.  Every operation is a standalone function that
takes raw tensors as input, following the same design philosophy as
:mod:`fvdb.functional` for sparse-grid operations.

.. tip::

   Most users should prefer the methods on :class:`~fvdb.GaussianSplat3d`
   directly.  The functional API is useful when building custom rendering
   pipelines or when you need explicit control over the projection,
   rasterization, and compositing stages.


Projection
----------

.. autofunction:: project_gaussians
.. autofunction:: project_gaussians_for_camera


Spherical Harmonics
-------------------

.. autofunction:: evaluate_spherical_harmonics


Rasterization
-------------

.. autofunction:: rasterize_from_projected
.. autofunction:: rasterize_from_world
.. autofunction:: sparse_render


Rendering Pipelines
-------------------

.. autofunction:: render_images
.. autofunction:: render_depths
.. autofunction:: render_images_and_depths
.. autofunction:: render_images_from_world
.. autofunction:: render_depths_from_world
.. autofunction:: render_images_and_depths_from_world


Query Operations
----------------

.. autofunction:: render_num_contributing_gaussians
.. autofunction:: render_contributing_gaussian_ids
.. autofunction:: sparse_render_num_contributing_gaussians
.. autofunction:: sparse_render_contributing_gaussian_ids


Helpers
-------

.. autofunction:: build_render_settings
