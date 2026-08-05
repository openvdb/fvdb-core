Sparse Convolution
==========================

Convolution plans use the canonical componentwise Torch-phase relation
``fine_ijk = stride * coarse_ijk + tap_ijk - padding_before``. Here
``fine_ijk`` and ``coarse_ijk`` are integer coordinates on the fine and strided
lattices, ``padding_before = floor((kernel_size - 1) / 2)``, and each zero-based
kernel tap satisfies ``0 <= tap_ijk[axis] < kernel_size[axis]``. A plan without
a target grid generates full structural support; an explicit target restricts
the same relation. Transposed convolution is adjoint connectivity, not a value
inverse. Use :meth:`fvdb.ConvolutionPlan.from_plan_transposed` for the exact
finite transpose of an existing plan.

.. autoclass:: fvdb.ConvolutionPlan
   :members:
