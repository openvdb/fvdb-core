Visualization
===============================

.. automodule:: fvdb.viz
    :members:

Gaussian splats
---------------

.. autoclass:: fvdb.viz.ShOrderingMode
    :members:

The viewer accepts Gaussian splats through the core-owned
:class:`fvdb.viz.GaussianSplatViewData` tensor contract. Libraries that own a
Gaussian representation can expose an adapter that creates this data object
without copying its tensors.

For callers that already have renderer-ready tensors,
:meth:`fvdb.viz.Scene.add_gaussian_splat_tensors` is the lower-level entry
point used by :meth:`fvdb.viz.Scene.add_gaussian_splat_3d`.
