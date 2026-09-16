Visualization
===============================

.. automodule:: fvdb.viz
    :members:
    :exclude-members: CheckboxView, NumberView, SliderView, TextView

Widgets
-------

Widget handles share ``name``, ``scene_name``, ``value``, ``on_update`` and
``remove_on_update``.

.. autoclass:: fvdb.viz.CheckboxView
    :members:
    :inherited-members:

.. autoclass:: fvdb.viz.NumberView
    :members:
    :inherited-members:

.. autoclass:: fvdb.viz.SliderView
    :members:
    :inherited-members:

.. autoclass:: fvdb.viz.TextView
    :members:
    :inherited-members:

Gaussian splats
---------------

.. autoclass:: fvdb.viz.ShOrderingMode
    :members:
    :no-index:

The viewer accepts Gaussian splats through the core-owned
:class:`fvdb.viz.GaussianSplatViewData` tensor contract. Libraries that own a
Gaussian representation can expose an adapter that creates this data object
without copying its tensors.

For callers that already have renderer-ready tensors,
:meth:`fvdb.viz.Scene.add_gaussian_splat_tensors` is the lower-level entry
point used by :meth:`fvdb.viz.Scene.add_gaussian_splat_3d`.
