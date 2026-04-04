Gaussian Splatting
==========================

The :class:`~fvdb.GaussianSplat3d` class provides an object-oriented interface
for Gaussian splatting rendering.  It manages Gaussian parameters (means,
quaternions, scales, opacities, SH coefficients) and provides methods for
projection, rasterization, and training-related operations like gradient
accumulation and MCMC densification.

.. autoclass:: fvdb.ProjectedGaussianSplats
   :members:
   :special-members: __getitem__, __setitem__

.. autoclass:: fvdb.GaussianSplat3d
   :members:
   :special-members: __getitem__, __setitem__

.. seealso::

   :mod:`fvdb.functional.splat` provides a pure-function alternative for
   building custom rendering pipelines without the
   :class:`~fvdb.GaussianSplat3d` wrapper.  The functional API decomposes
   the rendering pipeline into individually composable stages
   (:func:`~fvdb.functional.splat.project_to_2d`,
   :func:`~fvdb.functional.splat.compute_opacities`,
   :func:`~fvdb.functional.splat.prepare_render_features`,
   :func:`~fvdb.functional.splat.intersect_tiles`,
   :func:`~fvdb.functional.splat.rasterize_dense`), enabling custom
   training loops and pipeline composition without mutable state.
