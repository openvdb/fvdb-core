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

   :mod:`fvdb.functional` provides a pure-function alternative for
   building custom rendering pipelines without the
   :class:`~fvdb.GaussianSplat3d` wrapper.  The functional API decomposes
   the rendering pipeline into individually composable stages
   (:func:`~fvdb.functional.project_gaussians`,
   :func:`~fvdb.functional.evaluate_gaussian_sh`,
   :func:`~fvdb.functional.intersect_gaussian_tiles`,
   :func:`~fvdb.functional.rasterize_screen_space_gaussians`), enabling
   custom training loops and pipeline composition without mutable state.
