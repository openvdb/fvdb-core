# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Jagged Tensor data structure and operations for FVDB.

This module provides the JaggedTensor class, a specialized data structure for representing
sequences of tensors with varying lengths (jagged or ragged arrays) with efficient GPU support.

Classes:
- JaggedTensor: A jagged tensor data structure with support for efficient operations

Constructors:
- JaggedTensor(): Create from tensors, sequences, or sequences of sequences
- JaggedTensor.from_tensor(): Create from a single tensor
- JaggedTensor.from_list_of_tensors(): Create from a list of tensors
- JaggedTensor.from_list_of_lists_of_tensors(): Create from nested lists of tensors
- JaggedTensor.from_data_and_indices(): Create from flat data and indices
- JaggedTensor.from_data_and_offsets(): Create from flat data and offsets
- JaggedTensor.from_data_indices_and_list_ids(): Create with nested structure
- JaggedTensor.from_data_offsets_and_list_ids(): Create with nested structure using offsets

Module-level factory functions:
- jempty(): Create empty jagged tensor
- jrand(): Create jagged tensor with random values
- jrandn(): Create jagged tensor with normal distribution
- jones(): Create jagged tensor filled with ones
- jzeros(): Create jagged tensor filled with zeros

JaggedTensor supports PyTorch interoperability through __torch_function__, allowing
many torch operations to work seamlessly with jagged data structures.
"""

import typing
from typing import TYPE_CHECKING, Any, Sequence, cast, overload

import numpy as np
import torch

from . import _parse_device_string
from ._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from ._fvdb_cpp import jcat as jcat_cpp
from ._fvdb_cpp import jempty as jempty_cpp
from ._fvdb_cpp import jones as jones_cpp
from ._fvdb_cpp import jrand as jrand_cpp
from ._fvdb_cpp import jrandn as jrandn_cpp
from ._fvdb_cpp import jzeros as jzeros_cpp
from .types import DeviceIdentifier, LShapeSpec, RShapeSpec

if TYPE_CHECKING:
    from .grid import Grid

# --- JaggedTensor.__torch_function__ whitelist ---
# Whitelist of torch.<fn> names supported by JaggedTensor.__torch_function__.
# Only include ops that are elementwise or that *preserve* the primary (leading)
# dimension (i.e., the flattened jagged axis).
_JT_TORCH_WHITELIST: set[str] = {
    # Unary, elementwise (and their in-place variants where applicable)
    "abs",
    "abs_",
    "neg",
    "relu",
    "relu_",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sqrt",
    "rsqrt",
    "ceil",
    "floor",
    "round",
    "trunc",
    "nan_to_num",
    "clamp",
    # Binary / ternary, elementwise
    "add",
    "sub",
    "mul",
    "div",
    "true_divide",
    "floor_divide",
    "remainder",
    "fmod",
    "pow",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "lerp",
    # Reductions over *non-primary* dims (must keep the leading dim intact)
    "sum",
    "mean",
    "prod",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "all",
    "any",
    "norm",
    "var",
    "std",
}


@overload
def _convert_to_list(seq: Sequence[int]) -> list[int]: ...
@overload
def _convert_to_list(seq: Sequence[Sequence[int]]) -> list[list[int]]: ...


def _convert_to_list(seq: Sequence[int] | Sequence[Sequence[int]]) -> list[int] | list[list[int]]:
    """Helper to convert Sequence types to list types for C++ binding compatibility."""
    if isinstance(seq, (list, tuple)):
        if seq and isinstance(seq[0], (list, tuple)):
            # Nested sequence - convert inner sequences to lists
            converted: list[list[int]] = [
                list(inner) if isinstance(inner, tuple) else cast(list[int], inner) for inner in seq
            ]
            return list(converted) if isinstance(seq, tuple) else converted
        else:
            # Simple sequence of ints
            return list(seq) if isinstance(seq, tuple) else cast(list[int], seq)  # type: ignore
    else:
        return cast(list[int], seq)


class JaggedTensor:
    """
    A jagged (ragged) tensor data structure with support for efficient operations.

    :class:`JaggedTensor` represents sequences of tensors with varying lengths, stored efficiently
    in a flat contiguous format with associated index/offset structures. This is useful
    for batch processing of variable-length sequences on the GPU while maintaining memory
    efficiency and enabling vectorized operations.


    A :class:`JaggedTensor` can represent:

        1. A sequence of tensors with varying shapes along the first dimension.
        These are usually written as ``[tensor_1, tensor_2, ..., tensor_N]`` where each
        ``tensor_i`` can have a different shape along the first dimension.

        2. Nested sequences (list of lists) with varying lengths at multiple levels.
        These are usually written as ``[[tensor_11, tensor_12, ...], [tensor_21, tensor_22, ...], ...]``
        where both the outer and inner sequences can have varying lengths, and each ``tensor_ij``
        can have a different shape along the first dimension.

    The :class:`JaggedTensor` data structure consists of the following components:

        - :attr:`jdata`: The flattened data tensor containing all elements
        - Indexing structures (:attr:`jidx`, :attr:`joffsets`, :attr:`jlidx`) to track element boundaries
        - Shape information (:attr:`lshape`, :attr:`eshape`, :attr:`rshape`) describing the structure


    JaggedTensor integrates with PyTorch through __torch_function__, allowing many
    torch operations to work directly on jagged tensors while preserving the jagged
    structure. Operations that preserve the leading (flattened) dimension work
    seamlessly, while shape-changing operations require specialized j* methods.

    Example usage:

        # Create a JaggedTensor from a list of tensors
        jt = JaggedTensor.from_list_of_tensors([torch.randn(3, 4), torch.randn(2, 4), torch.randn(5, 4)])

        # Perform element-wise operations
        jt2 = jt + 1.0
        jt3 = torch.relu(jt2)

        # Access jagged data and structure
        data = jt3.jdata
        offsets = jt3.joffsets

        # Get the first tensor in the jagged sequence
        first_tensor = jt3[0]

        # Get the last tensor in the jagged sequence
        last_tensor = jt3[-1]


    .. note::

        The :class:`JaggedTensor` should be constructed using the explicit classmethods:
        - :meth:`from_tensor()` for a single tensor
        - :meth:`from_list_of_tensors()` for a list of tensors
        - :meth:`from_list_of_lists_of_tensors()` for nested lists of tensors
        - :meth:`from_data_and_indices()` for pre-computed flat format
        - :meth:`from_data_and_offsets()` for pre-computed flat format with offsets
        - :meth:`from_empty()` for uninitialized data
        - :meth:`from_ones()` for data filled with ones
        - :meth:`from_rand()` for data filled with random values from uniform distribution [0, 1)
        - :meth:`from_randn()` for data filled with random values from normal distribution (mean 0, variance 1)
        - :meth:`from_zeros()` for data filled with zeros
        - :meth:`from_cat()` for concatenation of jagged tensors

    """

    def __init__(
        self,
        tensors: torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None = None,
        *,
        impl: JaggedTensorCpp | None = None,
    ) -> None:
        """
        Create a JaggedTensor from various input formats.

        This constructor accepts multiple input formats for flexibility. For clearer
        code, prefer using the explicit from_* classmethods instead.

        Args:
            tensors (torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None):
                Input data in one of several formats:
                - torch.Tensor: A single tensor (creates jagged tensor with one element)
                - Sequence[torch.Tensor]: List/tuple of tensors with varying first dimension
                - Sequence[Sequence[torch.Tensor]]: Nested sequences for multi-level jagging
                Defaults to None when impl is provided.
            impl (JaggedTensorCpp | None): Internal C++ implementation object.
                Used internally, should not be provided by users. Defaults to None.
        """
        if impl is not None:
            if tensors is not None:
                raise ValueError("Cannot provide both tensors and impl")
            self._impl = impl
        else:
            if tensors is None:
                raise ValueError("Must provide either tensors or impl")

            if not isinstance(tensors, (torch.Tensor, list, tuple)):
                raise TypeError(
                    "tensors must be a torch.Tensor or a sequence (or sequence of sequences) of torch.Tensor"
                )

            # Convert sequences to lists for C++ binding compatibility
            if isinstance(tensors, torch.Tensor):
                self._impl = JaggedTensorCpp(tensors)
            elif isinstance(tensors, (list, tuple)):
                # Check if it's a sequence of sequences
                if tensors and isinstance(tensors[0], (list, tuple)):
                    # Convert nested sequences to lists
                    converted: list[list[torch.Tensor]] = [
                        list(inner) if isinstance(inner, tuple) else cast(list[torch.Tensor], inner)
                        for inner in tensors
                    ]
                    if isinstance(tensors, tuple):
                        converted = list(converted)
                    self._impl = JaggedTensorCpp(converted)
                else:
                    # Simple sequence of tensors
                    converted_flat: list[torch.Tensor] = (
                        list(tensors) if isinstance(tensors, tuple) else cast(list[torch.Tensor], tensors)  # type: ignore
                    )
                    self._impl = JaggedTensorCpp(converted_flat)
            else:
                self._impl = JaggedTensorCpp(tensors)

    # ============================================================
    #                  JaggedTensor from_* constructors
    # ============================================================

    @classmethod
    def from_empty(
        cls,
        lsizes: Sequence[int] | Sequence[Sequence[int]],
        rsizes: Sequence[int] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` with uninitialized data.

        Similar to :func:`torch.empty()`, creates a :class:`JaggedTensor` with allocated but uninitialized
        memory, which is faster than initializing values when they will be immediately
        overwritten.

        Example:

        ... code-block:: python

            jt = jempty([2, 3, 4], rsizes=[5])
            print(jt)  # Output: A JaggedTensor containing tensors [of shapes (2, 5), (3, 5), (4, 5)] with uninitialized values.

            jt = jempty([[2, 3], [4]], rsizes=[5, 6])
            print(jt)  # Output: A JaggedTensor containing tensors [of shapes (2, 5, 6), (3, 5, 6), (4, 5, 6)] with uninitialized values.

        Args:
            lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
                Can be a sequence of integers for simple jagged structure, or nested sequences
                for multi-level jagged structure.
            rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
                Defaults to ``None`` *i.e.* scalar elements.
            device (torch.device | str | None): Device to create the tensor on.
                Defaults to ``None`` *i.e.* ``"cpu"``.
            dtype (torch.dtype | None): Data type for the tensor elements.
                Defaults to ``None`` *i.e.* ``torch.float32``.
            requires_grad (bool): Whether to track gradients. Defaults to ``False``.
            pin_memory (bool): Whether to use pinned memory. Defaults to ``False``.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with uninitialized data.
        """
        lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
        rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
        return cls(impl=jempty_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))

    @classmethod
    def from_ones(
        cls,
        lsizes: Sequence[int] | Sequence[Sequence[int]],
        rsizes: Sequence[int] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` filled with ones.

        Similar to :func:`torch.ones()`, creates a :class:`JaggedTensor` where all elements are initialized
        to the value 1.


        Example:

        ... code-block:: python

            jt = jones([2, 3, 4], rsizes=[5])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5), (3, 5), (4, 5)] filled with ones.

            jt = jones([[2, 3], [4]], rsizes=[5, 6])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5, 6), (3, 5, 6), (4, 5, 6)] filled with ones.


        Args:
            lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
                Can be a sequence of integers for simple jagged structure, or nested sequences
                for multi-level jagged structure.
            rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
                Defaults to ``None`` *i.e.* (scalar elements).
            device (torch.device | str | None): Device to create the tensor on.
                Defaults to ``None`` *i.e.* (CPU).
            dtype (torch.dtype | None): Data type for the tensor elements.
                Defaults to ``None`` *i.e.* (torch.float32).
            requires_grad (bool): Whether to track gradients. Defaults to ``False``.
            pin_memory (bool): Whether to use pinned memory. Defaults to ``False``.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` filled with ones.
        """
        lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
        rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
        return cls(impl=jones_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))

    @classmethod
    def from_rand(
        cls,
        lsizes: Sequence[int] | Sequence[Sequence[int]],
        rsizes: Sequence[int] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` with random values from uniform distribution [0, 1).

        Similar to :func:`torch.rand()`, creates a :class:`JaggedTensor` filled with random values sampled
        from a uniform distribution on the interval [0, 1).

        Example:

        ... code-block:: python

            jt = jrand([2, 3, 4], rsizes=[5])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5), (3, 5), (4, 5)] with random values.

            jt = jrand([[2, 3], [4]], rsizes=[5, 6])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5, 6), (3, 5, 6), (4, 5, 6)] with random values.

        Args:
            lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
                Can be a sequence of integers for simple jagged structure, or nested sequences
                for multi-level jagged structure.
            rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
                Defaults to ``None`` *i.e.* (scalar elements).
            device (torch.device | str | None): Device to create the tensor on.
                Defaults to ``None`` *i.e.* ``"cpu"``.
            dtype (torch.dtype | None): Data type for the tensor elements.
                Defaults to ``None`` *i.e.* ``torch.float32``.
            requires_grad (bool): Whether to track gradients. Defaults to ``False``.
            pin_memory (bool): Whether to use pinned memory. Defaults to ``False``.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with random values in [0, 1).
        """
        lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
        rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
        return cls(impl=jrand_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))

    @classmethod
    def from_randn(
        cls,
        lsizes: Sequence[int] | Sequence[Sequence[int]],
        rsizes: Sequence[int] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` with random values from standard normal distribution.

        Similar to :func:`torch.randn()`, creates a :class:`JaggedTensor` filled with random values sampled
        from a standard normal distribution (mean=0, std=1).

        Example:

        ... code-block:: python

            jt = jrandn([2, 3, 4], rsizes=[5])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5), (3, 5), (4, 5)] with normal random values.

            jt = jrandn([[2, 3], [4]], rsizes=[5, 6])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5, 6), (3, 5, 6), (4, 5, 6)] with normal random values.

        Args:
            lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
                Can be a sequence of integers for simple jagged structure, or nested sequences
                for multi-level jagged structure.
            rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
                Defaults to ``None`` *i.e.* (scalar elements).
            device (torch.device | str | None): Device to create the tensor on.
                Defaults to ``None`` *i.e.* ``"cpu"``.
            dtype (torch.dtype | None): Data type for the tensor elements.
                Defaults to ``None`` *i.e.* ``torch.float32``.
            requires_grad (bool): Whether to track gradients. Defaults to ``False``.
            pin_memory (bool): Whether to use pinned memory. Defaults to ``False``.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with normal random values.
        """
        lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
        rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
        return cls(impl=jrandn_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))

    @classmethod
    def from_zeros(
        cls,
        lsizes: Sequence[int] | Sequence[Sequence[int]],
        rsizes: Sequence[int] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` filled with zeros.

        Similar to :func:`torch.zeros()`, creates a :class:`JaggedTensor` where all elements are initialized
        to the value 0.

        Example:

            jt = jzeros([2, 3, 4], rsizes=[5])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5), (3, 5), (4, 5)] filled with zeros

            jt = jzeros([[2, 3], [4]], rsizes=[5, 6])
            print(jt)  # Output: A JaggedTensor containing tensors
                [of shapes (2, 5, 6), (3, 5, 6), (4, 5, 6)] filled with zeros


        Args:
            lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
                Can be a sequence of integers for simple jagged structure, or nested sequences
                for multi-level jagged structure.
            rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
                Defaults to ``None`` *i.e.* scalar elements.
            device (torch.device | str | None): Device to create the tensor on.
                Defaults to ``None`` *i.e.* ``"cpu"``.
            dtype (torch.dtype | None): Data type for the tensor elements.
                Defaults to ``None`` *i.e.* ``torch.float32``.
            requires_grad (bool): Whether to track gradients. Defaults to ``False``.
            pin_memory (bool): Whether to use pinned memory. Defaults to ``False``.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` filled with zeros.
        """
        lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
        rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
        return cls(impl=jzeros_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))

    @classmethod
    def from_cat(
        cls,
        things_to_cat: "Sequence[JaggedTensor]",
        dim: int | None = None,
    ) -> "JaggedTensor":
        """
        Create a JaggedTensor by concatenating a sequence of JaggedTensors.

        Args:
            things_to_cat: The sequence of JaggedTensors to concatenate.
            dim: The dimension to concatenate along, optional.
        """
        if len(things_to_cat) == 0:
            raise ValueError("Cannot concatenate empty list")
        elif isinstance(things_to_cat[0], JaggedTensor):
            cpp_jags: list[JaggedTensorCpp] = [thing._impl for thing in things_to_cat]
            cpp_result = jcat_cpp(cpp_jags, dim)  # type: ignore
            return cls(impl=cpp_result)
        else:
            raise TypeError("JaggedTensor.from_cat() can only cat JaggedTensors")

    @classmethod
    def from_tensor(cls, data: torch.Tensor) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` from a single :class:`torch.Tensor`.

        Args:
            data (torch.Tensor): The input tensor.

        Returns:
            jagged_tensor (JaggedTensor): A new JaggedTensor wrapping the input tensor.
        """
        return cls(tensors=data)

    @classmethod
    def from_list_of_tensors(cls, tensors: Sequence[torch.Tensor]) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` from a sequence of tensors with varying first dimensions.

        All tensors must have the same shape except for the first dimension, which can vary.
        *e.g.* ``[tensor_1, tensor_2, ..., tensor_N]`` where each ``tensor_i`` has shape
        ``(L_i, D_1, D_2, ...)`` with varying ``L_i``.

        Args:
            tensors (Sequence[torch.Tensor]): List or tuple of :class:`torch.Tensor` with compatible shapes.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` containing the sequence of tensors.
        """
        return cls(tensors=tensors)

    @classmethod
    def from_list_of_lists_of_tensors(cls, tensors: Sequence[Sequence[torch.Tensor]]) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` from a nested sequences of :class:`torch.Tensor` s.

        Creates a multi-level jagged structure where both outer and inner sequences can
        have varying lengths.

        Args:
            tensors (Sequence[Sequence[torch.Tensor]]): Nested list/tuple of :class:`torch.Tensor` s.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with nested jagged structure.
        """
        return cls(tensors=tensors)

    @classmethod
    def from_data_and_indices(cls, data: torch.Tensor, indices: torch.Tensor, num_tensors: int) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` from flattened data and per-element indices.

        Example:

            data = torch.tensor([1, 2, 3, 4, 5, 6])
            indices = torch.tensor([0, 0, 1, 1, 1, 2])

            jt = JaggedTensor.from_data_and_indices(data, indices, num_tensors=3)

            # jt represents:
            #  - tensor 0: [1, 2]
            #  - tensor 1: [3, 4, 5]
            #  - tensor 2: [6]
        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: ``(total_elements, ...)``.
            indices (torch.Tensor): Index tensor mapping each element to its parent tensor.
                Shape: ``(total_elements,)``. Values in range ``[0, num_tensors)``.
            num_tensors (int): Total number of tensors in the sequence.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` constructed from the data and indices.
        """
        return cls(impl=JaggedTensorCpp.from_data_and_indices(data, indices, num_tensors))

    @classmethod
    def from_data_and_offsets(cls, data: torch.Tensor, offsets: torch.Tensor) -> "JaggedTensor":
        """
        Create a :class:`JaggedTensor` from flattened data and offset array.

        Offsets define boundaries between tensors in the flattened data array.
        Tensor ``i`` contains elements ``data[offsets[i]:offsets[i+1]]``.

        Example:

        .. code-block:: python

            data = torch.tensor([1, 2, 3, 4, 5, 6])

            offsets = torch.tensor([0, 2, 5, 6])  # 3 tensors: [0:2], [2:5], [5:6]

            jt = JaggedTensor.from_data_and_offsets(data, offsets)

            # jt represents:
            #  - tensor 0: [1, 2]
            #  - tensor 1: [3, 4, 5]
            #  - tensor 2: [6]

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: ``(total_elements, ...)``.
            offsets (torch.Tensor): Offset tensor marking tensor boundaries.
                Shape: ``(num_tensors + 1,)``. Must be monotonically increasing.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` constructed from the ``data`` and ``offsets``.
        """
        return cls(impl=JaggedTensorCpp.from_data_and_offsets(data, offsets))

    @classmethod
    def from_data_indices_and_list_ids(
        cls, data: torch.Tensor, indices: torch.Tensor, list_ids: torch.Tensor, num_tensors: int
    ) -> "JaggedTensor":
        """
        Create a nested JaggedTensor from data, indices, and list IDs.

        Creates a multi-level jagged structure where list_ids provide an additional
        level of grouping beyond the basic indices.

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: (total_elements, ...).
            indices (torch.Tensor): Index tensor mapping each element to its tensor.
                Shape: (total_elements,).
            list_ids (torch.Tensor): List ID tensor for nested structure.
                Shape: (total_elements,).
            num_tensors (int): Total number of tensors.

        Returns:
            jagged_tensor (JaggedTensor): A new JaggedTensor with nested jagged structure.
        """
        return cls(impl=JaggedTensorCpp.from_data_indices_and_list_ids(data, indices, list_ids, num_tensors))

    @classmethod
    def from_data_offsets_and_list_ids(
        cls, data: torch.Tensor, offsets: torch.Tensor, list_ids: torch.Tensor
    ) -> "JaggedTensor":
        """
        Create a nested :class:`JaggedTensor` from data, offsets, and list IDs.

        The offsets are used to define boundaries between tensors in the flattened array,
        and the list ids provide an additional level of grouping.

        Example:

        .. code-block:: python

            data = torch.tensor([1, 2, 3, 4, 5, 6])
            offsets = torch.tensor([0, 2, 5, 6])  # 3 tensors: [0:2], [2:5], [5:6]
            list_ids = torch.tensor([[0, 0], [0, 1], [1, 0]]) # First two tensors in list 0, last in list 1

            jt = JaggedTensor.from_data_offsets_and_list_ids(data, offsets, list_ids)

            # jt represents the structure [[t_00, t_01], [t_10]]
            # where t_00 = [1, 2], t_01 = [3, 4, 5], t_10 = [6]

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: ``(total_elements, ...)``.
            offsets (torch.Tensor): Offset tensor marking tensor boundaries.
                Shape: ``(num_tensors + 1,)``.
            list_ids (torch.Tensor): List ID tensor for nested structure.
                Shape: ``(num_tensors, 2)``.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with nested jagged structure.
        """
        return cls(impl=JaggedTensorCpp.from_data_offsets_and_list_ids(data, offsets, list_ids))

    # ============================================================
    #                Regular Instance Methods Begin
    # ============================================================

    def abs(self) -> "JaggedTensor":
        """
        Compute the absolute value element-wise.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with absolute values.
        """
        return JaggedTensor(impl=self._impl.abs())

    def abs_(self) -> "JaggedTensor":
        """
        Compute the absolute value element-wise in-place.

        Returns:
            jagged_tensor (JaggedTensor): The modified :class:`JaggedTensor` (self).
        """
        return JaggedTensor(impl=self._impl.abs_())

    def ceil(self) -> "JaggedTensor":
        """
        Round elements up to the nearest integer.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with ceiling applied.
        """
        return JaggedTensor(impl=self._impl.ceil())

    def ceil_(self) -> "JaggedTensor":
        """
        Round elements up to the nearest integer in-place.

        Returns:
            jagged_tensor (JaggedTensor): The modified :class:`JaggedTensor` (self).
        """
        return JaggedTensor(impl=self._impl.ceil_())

    def clone(self) -> "JaggedTensor":
        """
        Create a deep copy of the JaggedTensor.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with copied data and structure.
        """
        return JaggedTensor(impl=self._impl.clone())

    def cpu(self) -> "JaggedTensor":
        """
        Move the JaggedTensor to CPU memory.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` on CPU device.
        """
        return JaggedTensor(impl=self._impl.cpu())

    def cuda(self) -> "JaggedTensor":
        """
        Move the JaggedTensor to CUDA (GPU) memory.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` on CUDA device.
        """
        return JaggedTensor(impl=self._impl.cuda())

    def detach(self) -> "JaggedTensor":
        """
        Detach the JaggedTensor from the autograd graph.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` detached from the computation graph.
        """
        return JaggedTensor(impl=self._impl.detach())

    def double(self) -> "JaggedTensor":
        """
        Convert elements to double (float64) dtype.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with double precision.
        """
        return JaggedTensor(impl=self._impl.double())

    def float(self) -> "JaggedTensor":
        """
        Convert elements to float (float32) dtype.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with float32 precision.
        """
        return JaggedTensor(impl=self._impl.float())

    def floor(self) -> "JaggedTensor":
        """
        Round elements down to the nearest integer.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with floor applied.
        """
        return JaggedTensor(impl=self._impl.floor())

    def floor_(self) -> "JaggedTensor":
        """
        Round elements down to the nearest integer in-place.

        Returns:
            jagged_tensor (JaggedTensor): The modified :class:`JaggedTensor` ``(self)``.
        """
        return JaggedTensor(impl=self._impl.floor_())

    def jagged_like(self, data: torch.Tensor) -> "JaggedTensor":
        """
        Create a new JaggedTensor with the same structure but different data.

        The new JaggedTensor will have the same jagged structure ``(joffsets, jidx, etc.)``
        as the current one, but with new ``jdata`` values.

        Args:
            data (torch.Tensor): New data tensor with compatible shape.
                Must have the same leading dimension as self.jdata.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with the same structure but new data.
        """
        return JaggedTensor(impl=self._impl.jagged_like(data))

    def jflatten(self, dim: int = 0) -> "JaggedTensor":
        """
        Flatten the jagged dimensions starting from the specified dimension.

        Example:

            # Original jagged tensor with 2 jagged dimensions
            # representing a tensor of shape [ [ t_00, t_01, ... ], [ t_b0, t_b1, ... ] ]
            jt = JaggedTensor.from_list_of_lists_of_tensors(...)

            # Flatten starting from dim=0
            jt_flat = jt.jflatten(dim=0)

            # jt_flat is now a jagged tensor with 1 jagged dimension and represents
            # [ t_00, t_01, ..., t_b0, t_b1, ... ]


        Args:
            dim (int): The dimension from which to start flattening. Defaults to 0.

        Returns:
            jagged_tensor (JaggedTensor): A new :class:`JaggedTensor` with flattened jagged structure.
        """
        return JaggedTensor(impl=self._impl.jflatten(dim))

    def jmax(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        """
        Compute the maximum along a dimension of each tensor in the jagged structure.

        Returns both the maximum values and the indices where they occur.

        Example:

            # Create a jagged tensor from a list of tensors of each of shape (L_i, D)
            jt = JaggedTensor.from_list_of_lists_of_tensors([t1, t2, t3])

            # Compute the maximum along the jagged dimension (dim=0)
            values, indices = jt.jmax(dim=0)

            # values is now a jagged tensor containing the maximum values from each tensor
            # along dim=0
            # this is equivalent to (but faster than):
            # values = JaggedTensor.from_list_of_lists_of_tensors([torch.max(t, dim=0).values for t in [t1, t2, t3]])
            # indices = JaggedTensor.from_list_of_lists_of_tensors([torch.max(t, dim=0).indices for t in [t1, t2, t3]])

        Args:
            dim (int): The dimension along which to compute max for each tensor. Defaults to 0.
            keepdim (bool): Whether to keep the reduced dimension. Defaults to False.

        Returns:
            values (JaggedTensor): A :class:`JaggedTensor` containing the maximum values.
            indices (JaggedTensor): A :class:`JaggedTensor` containing the indices of the maximum values.
        """
        return [JaggedTensor(impl=impl) for impl in self._impl.jmax(dim, keepdim)]

    def jmin(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        """
        Compute the minimum along a dimension of each tensor in the jagged structure.

        Returns both the minimum values and the indices where they occur.

        Example:

        .. code-block:: python

            # Create a jagged tensor from a list of tensors of each of shape (L_i, D)
            jt = JaggedTensor.from_list_of_lists_of_tensors([t1, t2, t3])

            # Compute the minimum along the jagged dimension (dim=0)
            values, indices = jt.jmin(dim=0)

            # values is now a jagged tensor containing the minimum values from each tensor
            # along dim=0
            # this is equivalent to (but faster than):
            # values = JaggedTensor.from_list_of_lists_of_tensors([torch.min(t, dim=0).values for t in [t1, t2, t3]])
            # indices = JaggedTensor.from_list_of_lists_of_tensors([torch.min(t, dim=0).indices for t in [t1, t2, t3]])

        Args:
            values (JaggedTensor): A :class:`JaggedTensor` containing the minimum values.
            indices (JaggedTensor): A :class:`JaggedTensor` containing the indices of the minimum values.

        Returns:
            list[JaggedTensor]: A list containing [values, indices] as JaggedTensors.
        """
        return [JaggedTensor(impl=impl) for impl in self._impl.jmin(dim, keepdim)]

    def jreshape(self, lshape: Sequence[int] | Sequence[Sequence[int]]) -> "JaggedTensor":
        """
        Reshape the jagged dimensions to new sizes.

        Args:
            lshape (Sequence[int] | Sequence[Sequence[int]]): New shape(s) for jagged dimensions.
                Can be a single sequence of sizes or nested sequences for multi-level structure.

        Returns:
            JaggedTensor: A new JaggedTensor with reshaped jagged structure.
        """
        lshape_cpp = _convert_to_list(lshape)
        return JaggedTensor(impl=self._impl.jreshape(lshape_cpp))

    def jreshape_as(self, other: "JaggedTensor | torch.Tensor") -> "JaggedTensor":
        """
        Reshape the jagged structure to match another JaggedTensor or Tensor.

        Args:
            other (JaggedTensor | torch.Tensor): The target structure to match.

        Returns:
            JaggedTensor: A new JaggedTensor with structure matching other.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl.jreshape_as(other._impl))
        else:
            if not isinstance(other, torch.Tensor):
                raise TypeError("other must be a JaggedTensor or a torch.Tensor")
            return JaggedTensor(impl=self._impl.jreshape_as(other))

    def jsqueeze(self, dim: int | None = None) -> "JaggedTensor":
        """
        Remove singleton dimensions from the jagged structure.

        Args:
            dim (int | None): Specific dimension to squeeze, or None to squeeze all
                singleton dimensions. Defaults to None.

        Returns:
            JaggedTensor: A new JaggedTensor with singleton dimensions removed.
        """
        return JaggedTensor(impl=self._impl.jsqueeze(dim))

    def jsum(self, dim: int = 0, keepdim: bool = False) -> "JaggedTensor":
        """
        Sum along a jagged dimension.

        Args:
            dim (int): The jagged dimension along which to sum. Defaults to 0.
            keepdim (bool): Whether to keep the reduced dimension. Defaults to False.

        Returns:
            JaggedTensor: A new JaggedTensor with values summed along the specified dimension.
        """
        return JaggedTensor(impl=self._impl.jsum(dim, keepdim))

    def long(self) -> "JaggedTensor":
        """
        Convert elements to long (int64) dtype.

        Returns:
            JaggedTensor: A new JaggedTensor with int64 dtype.
        """
        return JaggedTensor(impl=self._impl.long())

    # FIXME(@chorvath, @fwilliams) Why is this here?
    def requires_grad_(self, requires_grad: bool) -> "JaggedTensor":
        """
        Set the requires_grad attribute in-place.

        Args:
            requires_grad (bool): Whether to track gradients for this tensor.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.requires_grad_(requires_grad))

    def rmask(self, mask: torch.Tensor) -> "JaggedTensor":
        """
        Apply a mask to filter elements along the regular (non-jagged) dimension.

        Args:
            mask (torch.Tensor): Boolean mask tensor to apply.
                Shape must be compatible with the regular dimensions.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with masked elements.
        """
        return JaggedTensor(impl=self._impl.rmask(mask))

    def round(self, decimals: int = 0) -> "JaggedTensor":
        """
        Round elements to the specified number of decimals.

        Args:
            decimals (int): Number of decimal places to round to. Defaults to 0.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with rounded values.
        """
        return JaggedTensor(impl=self._impl.round(decimals))

    def round_(self, decimals: int = 0) -> "JaggedTensor":
        """
        Round elements to the specified number of decimals in-place.

        Args:
            decimals (int): Number of decimal places to round to. Defaults to 0.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` ``(self)``.
        """
        return JaggedTensor(impl=self._impl.round_(decimals))

    def sqrt(self) -> "JaggedTensor":
        """
        Compute the square root element-wise.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with square root applied.
        """
        return JaggedTensor(impl=self._impl.sqrt())

    def sqrt_(self) -> "JaggedTensor":
        """
        Compute the square root element-wise in-place.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` ``(self)``.
        """
        return JaggedTensor(impl=self._impl.sqrt_())

    def to(self, device_or_dtype: torch.device | str | torch.dtype) -> "JaggedTensor":
        """
        Move the JaggedTensor to a device or convert to a dtype.

        Args:
            device_or_dtype (torch.device | str | torch.dtype): Target :class:`torch.device` or :class:`torch.dtype`.
                Can be a device ("cpu", "cuda"), or a dtype (torch.float32, etc.).

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` on the specified device or with specified dtype.
        """
        return JaggedTensor(impl=self._impl.to(device_or_dtype))

    def type(self, dtype: torch.dtype) -> "JaggedTensor":
        """
        Convert the :class:`JaggedTensor` to a specific dtype.

        Args:
            dtype (torch.dtype): Target data type (*e.g.* ``torch.float32``, ``torch.int64``).

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with the specified dtype.
        """
        return JaggedTensor(impl=self._impl.type(dtype))

    def type_as(self, other: "JaggedTensor | torch.Tensor") -> "JaggedTensor":
        """
        Convert the :class:`JaggedTensor` to match the dtype of another tensor.

        Args:
            other (JaggedTensor | torch.Tensor): Reference :class:`torch.Tensor` or :class:`JaggedTensor` whose dtype to match.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with dtype matching other.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl.type_as(other._impl))
        else:
            if not isinstance(other, torch.Tensor):
                raise TypeError("other must be a JaggedTensor or a torch.Tensor")
            return JaggedTensor(impl=self._impl.type_as(other))

    def unbind(self) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        """
        Unbind the :class:`JaggedTensor` into its constituent tensors.

        Returns:
            list[torch.Tensor] | list[list[torch.Tensor]]: A list of :class:`torch.Tensor` (for simple
                jagged structure) or a list of lists of :class:`torch.Tensor` (for nested structure).
        """
        return self._impl.unbind()

    def __add__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Add another tensor or scalar element-wise to this :class:`JaggedTensor`.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to add.

        Returns:
            JaggedTensor: Result of element-wise addition between ``self`` and ``other``.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl + other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl + other)

    def __eq__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise equality comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with element-wise comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl == other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl == other)

    def __floordiv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Floor division element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor.

        Returns:
            JaggedTensor: Result of floor division.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl // other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl // other)

    def __ge__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise greater-than-or-equal comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl >= other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl >= other)

    def __getitem__(self, index: Any) -> "JaggedTensor":
        """
        Index or slice the JaggedTensor. This slices along the outer jagged dimension.

        Example:

        .. code-block:: python

            jt = JaggedTensor.from_list_of_tensors([torch.randn(3, 4), torch.randn(2, 4), torch.randn(5, 4)])
            jt0 = jt[0]  # First tensor (shape: (3, 4))
            jt1_2 = jt[1:3]  # Second and third tensors (shape: (2, 4) and (5, 4))

            # Equivalent to JaggedTensor([jt[i].jdata[jt[i].jdata > 0]])
            jt_masked = jt[jt > 0]  # Masked selection

            jt_ll = JaggedTensor.from_list_of_lists_of_tensors([[torch.randn(2, 3), torch.randn(1, 3)], [torch.randn(4, 3)]])
            jt_ll0 = jt_ll[0]  # First list of tensors [torch.randn(2, 3), torch.randn(1, 3)]
            jt_ll1 = jt_ll[1]  # Second list of tensors [torch.randn(4, 3)]
            jt_ll0_0 = jt_ll0[0]  # First tensor in the first list (shape: (2, 3))
            jt_ll0_1 = jt_ll0[1]  # Second tensor in the first list (shape: (1, 3))


        Args:
            index (Any): Index, slice, or mask to apply. Can be a JaggedTensor for jagged indexing.

        Returns:
            JaggedTensor: The indexed/sliced :class:`JaggedTensor`.
        """
        if isinstance(index, JaggedTensor):
            return JaggedTensor(impl=self._impl[index._impl])
        else:
            return JaggedTensor(impl=self._impl[index])

    def __gt__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise greater-than comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl > other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl > other)

    def __iadd__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place addition element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to add.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl += other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl += other
        return self

    def __ifloordiv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place floor division element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl //= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl //= other
        return self

    def __imod__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place modulo operation element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor for modulo.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl %= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl %= other
        return self

    def __imul__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place multiplication element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to multiply.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl *= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl *= other
        return self

    def __ipow__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place exponentiation element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Exponent.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl **= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl **= other
        return self

    def __isub__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place subtraction element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to subtract.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl -= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl -= other
        return self

    def __itruediv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        In-place true division element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor.

        Returns:
            JaggedTensor: The modified :class:`JaggedTensor` (self).
        """
        if isinstance(other, JaggedTensor):
            self._impl /= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl /= other
        return self

    def __le__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise less-than-or-equal comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl <= other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl <= other)

    def __len__(self) -> int:
        """
        Return the number of tensors in the jagged sequence.

        Returns:
            int: Number of tensors in the :class:`JaggedTensor`.
        """
        return len(self._impl)

    def __lt__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise less-than comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl < other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl < other)

    def __mod__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Modulo operation element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor for modulo.

        Returns:
            JaggedTensor: Result of modulo operation.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl % other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl % other)

    def __mul__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Multiply element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to multiply.

        Returns:
            JaggedTensor: Result of element-wise multiplication.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl * other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl * other)

    def __ne__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise inequality comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean :class:`JaggedTensor` with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl != other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl != other)

    def __neg__(self) -> "JaggedTensor":
        """
        Negate all elements.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with all elements negated.
        """
        return JaggedTensor(impl=-self._impl)

    def __pow__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Raise elements to a power element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Exponent.

        Returns:
            JaggedTensor: Result of exponentiation.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl**other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl**other)

    def __sub__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Subtract element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to subtract.

        Returns:
            JaggedTensor: Result of element-wise subtraction.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl - other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl - other)

    def __truediv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        True division element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Divisor.

        Returns:
            JaggedTensor: Result of element-wise division.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl / other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl / other)

    def __iter__(self) -> typing.Iterator["JaggedTensor"]:
        """
        Iterate over the JaggedTensor, yielding each tensor in the sequence.

        .. note::

            This iterates over the outer jagged dimension, yielding individual
            :class:`JaggedTensor` elements. If this :class:`JaggedTensor` represents a single list of
            tensors, each yielded element will be a :class:`JaggedTensor` containing one tensor.
            You can access the underlying tensor via the ``.jdata`` attribute of the yielded
            :class:`JaggedTensor`.

        Returns:
            typing.Iterator[JaggedTensor]: Iterator yielding :class:`JaggedTensor` elements.
        """
        for i in range(len(self)):
            yield self[i]

    # ============================================================
    #                  PyTorch interop (__torch_function__)
    # ============================================================
    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """
        Intercept selected torch.<fn>(...) calls and forward them to the underlying
        contiguous storage (`jdata`). The operation is allowed only if the result
        preserves the JaggedTensor's primary (leading) dimension. The jagged
        layout (offsets/indices) is *not* changed.

        Examples:
            torch.relu(jt)              -> applies relu to jt.jdata (returns JaggedTensor)
            torch.add(jt, 1.0)         -> elementwise add on jt.jdata (returns JaggedTensor)
            torch.sum(jt, dim=-1)      -> reduces trailing dim(s) but preserves leading dim
            torch.relu_(jt)            -> in-place on jt.jdata, returns the mutated JaggedTensor

        Unsupported:
            - Any op that would change or reduce the leading dimension (e.g., torch.sum(jt) with dim=None)
            - Shape-rearranging ops like reshape/permute/transpose/cat/stack, etc. (use the provided j* APIs)
        """
        if kwargs is None:
            kwargs = {}

        # Only participate in dispatch when a JaggedTensor is present.
        if not any(issubclass(t, JaggedTensor) for t in types):
            return NotImplemented

        name = getattr(func, "__name__", None)
        if name is None or name not in _JT_TORCH_WHITELIST:
            return NotImplemented

        # Find a prototype JaggedTensor carrying the jagged structure.
        def _find_proto(obj: Any) -> "JaggedTensor | None":
            if isinstance(obj, JaggedTensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for x in obj:
                    jt = _find_proto(x)
                    if jt is not None:
                        return jt
            return None

        proto: "JaggedTensor | None" = None
        for o in args:
            proto = _find_proto(o)
            if proto is not None:
                break
        if proto is None:
            for o in kwargs.values():
                proto = _find_proto(o)
                if proto is not None:
                    break
        if proto is None:
            return NotImplemented

        # Unwrap JaggedTensors -> their underlying torch.Tensor (jdata)
        def _unwrap(obj: Any) -> Any:
            if isinstance(obj, JaggedTensor):
                return obj.jdata
            if isinstance(obj, (list, tuple)):
                typ = type(obj)
                return typ(_unwrap(x) for x in obj)
            return obj

        conv_args = tuple(_unwrap(a) for a in args)
        conv_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

        # Handle out= if provided as a JaggedTensor
        out_jt: "JaggedTensor | None" = None
        if "out" in kwargs:
            orig_out = kwargs["out"]
            if isinstance(orig_out, JaggedTensor):
                out_jt = orig_out
                conv_kwargs["out"] = orig_out.jdata
            elif isinstance(orig_out, (list, tuple)):
                raise TypeError("JaggedTensor: tuple/list form of 'out=' is not supported.")

        # Execute the torch operation on raw tensors.
        result = func(*conv_args, **conv_kwargs)

        N0 = int(proto.jdata.shape[0])

        # Wrap torch.Tensor result(s) back into JaggedTensor, verifying the primary dim.
        def _wrap(o: Any) -> Any:
            if isinstance(o, torch.Tensor):
                if o.ndim == 0 or int(o.shape[0]) != N0:
                    raise RuntimeError(
                        f"torch.{name} would change the primary jagged dimension "
                        f"(expected leading dim {N0}, got {tuple(o.shape)})."
                    )
                return proto.jagged_like(o)
            if isinstance(o, (list, tuple)):
                items = [_wrap(x) for x in o]
                if isinstance(o, tuple) and hasattr(o, "_fields"):
                    # namedtuple (e.g., values/indices from some reductions)
                    return type(o)(*items)
                return type(o)(items)
            return o

        # In-place variant: mutate proto/out and return the mutated JaggedTensor.
        if name.endswith("_"):
            if out_jt is not None:
                return out_jt
            return proto

        # If out= was a JaggedTensor, return it after the write.
        if out_jt is not None:
            return out_jt

        return _wrap(result)

    # ============================================================
    #                        Properties
    # ============================================================

    @property
    def jdata(self) -> torch.Tensor:
        """
        Flattened data tensor containing all elements in this JaggedTensor.

        For example, if this :class:`JaggedTensor` represents three tensors of shapes
        ``(2, 4)``, ``(3, 4)``, and ``(1, 4)``, then ``jdata`` will have shape ``(6, 4)``.

        Returns:
            torch.Tensor: The data tensor.
        """
        return self._impl.jdata

    @jdata.setter
    def jdata(self, value: torch.Tensor) -> None:
        """
        Set the flattened data tensor. The shape must be compatible with the jagged structure.
        This operation does not modify the jagged layout (offsets/indices).

        Example:

            jt = JaggedTensor.from_list_of_tensors([torch.randn(2, 4), torch.randn(3, 4)])
            print(jt.jdata.shape)  # Output: torch.Size([5, 4])

            # Update with data of the same shape
            new_data = torch.randn(5, 4)
            jt.jdata = new_data  # Update the data tensor
            print(jt.jdata)  # Output: new_data tensor

            # Update with the same outer shape but different inner shape is okay
            new_data_2 = torch.randn(5, 2, 3)
            jt.jdata = new_data_2  # Update the data tensor
            print(jt.jdata)  # Output: new_data_2 tensor

            # Update with a completely different shape is not allowed
            new_data_3 = torch.randn(4, 4)
            jt.jdata = new_data_3  # This will raise an error

        Args:
            value (torch.Tensor): New ``jdata`` tensor to set.
        """
        self._impl.jdata = value

    @property
    def requires_grad(self) -> bool:
        """
        Whether this :class:`JaggedTensor` requires gradient computation.

        Returns:
            bool: ``True`` if gradients are tracked, False otherwise.
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Set whether this :class:`JaggedTensor` requires gradient computation.

        Args:
            value (bool): ``True`` to require gradients, ``False`` otherwise.
        """
        # self._impl.set_requires_grad(value)
        self._impl.requires_grad = value

    @property
    def device(self) -> torch.device:
        """
        Device where this :class:`JaggedTensor` is stored.

        Returns:
            torch.device: The device of this :class:`JaggedTensor`.
        """
        return self._impl.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of the elements in this :class:`JaggedTensor`.

        Returns:
            torch.dtype: The data type of this :class:`JaggedTensor`.
        """
        return self._impl.dtype

    @property
    def edim(self) -> int:
        """
        Dimensionality of the element (regular) structure.

        For example, if each tensor in the jagged sequence has shape ``(?, 4, 5)``,
        then ``edim`` will be ``2`` since there are two regular dimensions per element.

        Returns:
            int: The dimensionality of the element structure.
        """
        return self._impl.edim

    @property
    def eshape(self) -> list[int]:
        """
        Shape of the element dimensions.

        For example, if each tensor in the jagged sequence has shape ``(?, 4, 5)``,
        then ``eshape`` will be ``[4, 5]``.

        Returns:
            list[int]: The shape of the element dimensions.
        """
        return self._impl.eshape

    @property
    def is_cpu(self) -> bool:
        """
        Whether this :class:`JaggedTensor` is stored on the CPU.

        Returns:
            bool: ``True`` if on CPU, ``False`` otherwise.
        """
        return self._impl.is_cpu

    @property
    def is_cuda(self) -> bool:
        """
        Whether this :class:`JaggedTensor` is stored on a CUDA device.

        Returns:
            bool: ``True`` if on CUDA, ``False`` otherwise.
        """
        return self._impl.is_cuda

    @property
    def jidx(self) -> torch.Tensor:
        """
        Indices for each element in the jagged structure. This maps each element
        in the ``jdata`` tensor to its corresponding position in the jagged layout.

        Example:

        .. code-block:: python

            # For a JaggedTensor representing three tensors of shapes (2, 4), (3, 4), and (1, 4),
            # the ``jidx`` tensor would be: ``tensor([0, 1, 0, 1, 2, 0])``.

            jt = JaggedTensor.from_list_of_tensors([torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)])
            print(jt.jidx)  # Output: tensor([0, 1, 0, 1, 2, 0])

        Returns:
            torch.Tensor: The jagged indices tensor.
        """
        return self._impl.jidx

    @property
    def jlidx(self) -> torch.Tensor:
        """
        List indices for nested jagged structures. This is a :class:`torch.Tensor` that
        maps each element in the ``jdata`` tensor to its corresponding list in the
        jagged layout.

        Example:

        .. code-block:: python

            # For a JaggedTensor representing two lists of tensors:
            # List 0: tensors of shapes (2, 3) and (1, 3)
            # List 1: tensor of shape (4, 3)
            # the jlidx tensor would be: tensor([0, 0], [0, 1], [1, 0]).

            jt = JaggedTensor.from_list_of_lists_of_tensors([[torch.randn(2, 3), torch.randn(1, 3)], [torch.randn(4, 3)]])
            print(jt.jlidx)  # Output: tensor([[0, 0], [0, 1], [1, 0]])

        Returns:
            torch.Tensor: The jagged list indices tensor.
        """
        return self._impl.jlidx

    @property
    def joffsets(self) -> torch.Tensor:
        """
        Offsets marking boundaries between tensors.

        Example:

        .. code-block:: python

            # For a JaggedTensor representing three tensors of shapes (2, 4), (3, 4), and (1, 4),
            # the ``joffsets`` tensor would be: ``tensor([0, 2, 5, 6])``.
            jt = JaggedTensor.from_list_of_tensors([torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)])
            print(jt.joffsets)  # Output: tensor([0, 2, 5, 6])

            # For a JaggedTensor representing two lists of tensors:
            # List 0: tensors of shapes (2, 3) and (1, 3)
            # List 1: tensor of shape (4, 3)
            # the joffsets tensor would be: tensor([0, 2, 3, 7]).
            jt_ll = JaggedTensor.from_list_of_lists_of_tensors([[torch.randn(2, 3), torch.randn(1, 3)], [torch.randn(4, 3)]])
            print(jt_ll.joffsets)  # Output: tensor([0, 2, 3, 7])

        Returns:
            torch.Tensor: The jagged offsets tensor.
        """
        return self._impl.joffsets

    @property
    def ldim(self) -> int:
        """
        Dimensionality of the jagged (leading) structure. *i.e.* the number of jagged levels.

        If the :class:`JaggedTensor` represents a simple jagged structure (a single list of tensors),
        then ``ldim`` will be ``1``. For nested jagged structures (lists of lists of tensors), ``ldim``
        will be greater than ``1``.

        Returns:
            int: The dimensionality of the jagged structure.
        """
        return self._impl.ldim

    @property
    def lshape(self) -> list[int] | list[list[int]]:
        """
        List structure shape(s) of the jagged dimensions.

        Example:

        .. code-block:: python

            # For a JaggedTensor representing three tensors of shapes (2, 4), (3, 4), and (1, 4),
            # the ``lshape`` will be: ``[2, 3, 4]`` (three tensors in the jagged structure).
            jt = JaggedTensor.from_list_of_tensors([torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)])
            print(jt.lshape)  # Output: [2, 3, 1]

            # For a JaggedTensor representing two lists of tensors:
            # List 0: tensors of shapes (2, 3) and (1, 3)
            # List 1: tensor of shape (4, 3)
            # the ``lshape`` will be: ``[[2, 1], [4]]``.
            jt_ll = JaggedTensor.from_list_of_lists_of_tensors([[torch.randn(2, 3), torch.randn(1, 3)], [torch.randn(4, 3)]])
            print(jt_ll.lshape)  # Output: [[2, 1], [4]]

        Returns:
            list[int] | list[list[int]]: The jagged structure shapes.
        """
        return self._impl.lshape

    @property
    def num_tensors(self) -> int:
        """
        Return the total number of tensors in the jagged sequence.

        Returns:
            int: Number of tensors in this :class:`JaggedTensor`.
        """
        return self._impl.num_tensors

    @property
    def rshape(self) -> tuple[int, ...]:
        """
        Return the shape of the ``jdata`` tensor.

        .. note::

            ``rshape`` stands for "raw shape" and represents the full shape of the
            underlying data tensor, including both jagged and regular dimensions.

        Returns:
            tuple[int, ...]: Shape of the underlying data tensor.
        """
        return self._impl.rshape

    # Weirdly, unless we put this last, it messes up static type checking.
    def int(self) -> "JaggedTensor":
        """
        Convert elements to int (int32) dtype.

        Returns:
            JaggedTensor: A new :class:`JaggedTensor` with int32 dtype.
        """
        return JaggedTensor(impl=self._impl.int())


def jcat(jagged_tensors: Sequence[JaggedTensor], dim: int | None = None) -> JaggedTensor:
    """
    Concatenate a sequence of JaggedTensors along a given dimension.

    Args:
        jagged_tensors: The sequence of JaggedTensors to concatenate.
        dim: The dimension to concatenate along, optional.
    """
    return JaggedTensor.from_cat(jagged_tensors, dim)
