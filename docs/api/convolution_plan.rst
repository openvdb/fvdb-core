Sparse Convolution
==========================

Convolution plans use the canonical Torch-phase relation
``p = stride * q + u - floor((kernel_size - 1) / 2)`` componentwise. A plan
without a target grid generates full structural support; an explicit target
restricts the same relation. Transposed convolution is adjoint connectivity,
not a value inverse. Use :meth:`fvdb.ConvolutionPlan.from_plan_transposed` for
the exact finite transpose of an existing plan.

.. autoclass:: fvdb.ConvolutionPlan
   :members:
