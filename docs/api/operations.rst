Top-level Operations
====================

These operations are available directly from ``fvdb``. Tensor operations accept
both :class:`torch.Tensor` and :class:`fvdb.JaggedTensor`; see
:doc:`../tutorials/jagged_tensor` for examples of jagged tensor operations.

Concatenation
-------------

.. autofunction:: fvdb.jcat
.. autofunction:: fvdb.gcat

Attention and volume rendering
------------------------------

.. autofunction:: fvdb.scaled_dot_product_attention
.. autofunction:: fvdb.volume_render

Unary tensor operations
-----------------------

.. autofunction:: fvdb.relu
.. autofunction:: fvdb.relu_
.. autofunction:: fvdb.sigmoid
.. autofunction:: fvdb.tanh
.. autofunction:: fvdb.exp
.. autofunction:: fvdb.log
.. autofunction:: fvdb.sqrt
.. autofunction:: fvdb.floor
.. autofunction:: fvdb.ceil
.. autofunction:: fvdb.round
.. autofunction:: fvdb.nan_to_num
.. autofunction:: fvdb.clamp

Binary tensor operations
------------------------

.. autofunction:: fvdb.add
.. autofunction:: fvdb.sub
.. autofunction:: fvdb.mul
.. autofunction:: fvdb.true_divide
.. autofunction:: fvdb.floor_divide
.. autofunction:: fvdb.remainder
.. autofunction:: fvdb.pow
.. autofunction:: fvdb.maximum
.. autofunction:: fvdb.minimum

Comparisons
-----------

.. autofunction:: fvdb.eq
.. autofunction:: fvdb.ne
.. autofunction:: fvdb.lt
.. autofunction:: fvdb.le
.. autofunction:: fvdb.gt
.. autofunction:: fvdb.ge
.. autofunction:: fvdb.where

Reductions
----------

.. autofunction:: fvdb.sum
.. autofunction:: fvdb.mean
.. autofunction:: fvdb.amax
.. autofunction:: fvdb.amin
.. autofunction:: fvdb.argmax
.. autofunction:: fvdb.argmin
.. autofunction:: fvdb.all
.. autofunction:: fvdb.any
.. autofunction:: fvdb.norm
.. autofunction:: fvdb.var
.. autofunction:: fvdb.std

Space-filling curves
--------------------

.. py:function:: fvdb.morton(ijk)

   Encode integer coordinates in xyz Morton (Z-order) order.

   :param ijk: Coordinates of shape ``(N, 3)`` and dtype ``torch.int32`` on CPU or CUDA.
   :type ijk: torch.Tensor
   :returns: Codes of shape ``(N,)`` and dtype ``torch.int64`` on the input device.
   :rtype: torch.Tensor

.. py:function:: fvdb.hilbert(ijk)

   Encode integer coordinates in xyz Hilbert order.

   :param ijk: Coordinates of shape ``(N, 3)`` and dtype ``torch.int32`` on CPU or CUDA.
   :type ijk: torch.Tensor
   :returns: Codes of shape ``(N,)`` and dtype ``torch.int64`` on the input device.
   :rtype: torch.Tensor

NanoVDB metadata
----------------

.. py:class:: fvdb.NanoVDBGridMetadata

   Read-only metadata for a grid stored in a NanoVDB file. Obtain instances with
   :func:`fvdb.functional.read_nanovdb_metadata` without loading voxel data.

   .. py:attribute:: name

      Grid name stored in the file, as a string.

   .. py:attribute:: type

      NanoVDB grid type as a string, for example ``float``, ``Vec3f``, or ``OnIndex``.

   .. py:attribute:: grid_class

      NanoVDB grid class as a string, for example ``SDF`` or ``FOG``.

   .. py:attribute:: voxel_count

      Number of active voxels, as an integer.

   .. py:attribute:: voxel_size

      Voxel size in world units, as a tuple of three floats.

   .. py:attribute:: index_bbox_min

      Inclusive minimum of the index-space bounding box, as a tuple of three integers.

   .. py:attribute:: index_bbox_max

      Inclusive maximum of the index-space bounding box, as a tuple of three integers.

