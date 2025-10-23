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
from ._Cpp import JaggedTensor as JaggedTensorCpp
from ._Cpp import jempty as jempty_cpp
from ._Cpp import jones as jones_cpp
from ._Cpp import jrand as jrand_cpp
from ._Cpp import jrandn as jrandn_cpp
from ._Cpp import jzeros as jzeros_cpp
from .types import (
    DeviceIdentifier,
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    resolve_device,
    to_Vec3f,
    to_Vec3fBatch,
    to_Vec3fBatchBroadcastable,
    to_Vec3fBroadcastable,
    to_Vec3i,
    to_Vec3iBatch,
    to_Vec3iBatchBroadcastable,
    to_Vec3iBroadcastable,
)

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


class JaggedTensor:
    """
    A jagged (ragged) tensor data structure with support for efficient operations.

    JaggedTensor represents sequences of tensors with varying lengths, stored efficiently
    in a flat contiguous format with associated index/offset structures. This is useful
    for batch processing of variable-length sequences on GPU while maintaining memory
    efficiency and enabling vectorized operations.

    The jagged tensor consists of:
    - jdata: The flattened data tensor containing all elements
    - Indexing structures (jidx, joffsets, jlidx) to track element boundaries
    - Shape information (lshape, eshape, rshape) describing the structure

    JaggedTensor can represent:
    - A sequence of tensors with varying shapes along the first dimension
    - Nested sequences (list of lists) with varying lengths at multiple levels

    JaggedTensor integrates with PyTorch through __torch_function__, allowing many
    torch operations to work directly on jagged tensors while preserving the jagged
    structure. Operations that preserve the leading (flattened) dimension work
    seamlessly, while shape-changing operations require specialized j* methods.

    Note:
        The JaggedTensor constructor accepts various input formats. For clarity,
        prefer using the explicit classmethods:
        - JaggedTensor.from_tensor() for a single tensor
        - JaggedTensor.from_list_of_tensors() for a list of tensors
        - JaggedTensor.from_list_of_lists_of_tensors() for nested lists
        - JaggedTensor.from_data_and_indices() for pre-computed flat format
        - JaggedTensor.from_data_and_offsets() for pre-computed flat format with offsets

    Attributes:
        jdata (torch.Tensor): Flattened data tensor containing all elements
        jidx (torch.Tensor): Indices mapping each element to its parent tensor
        joffsets (torch.Tensor): Offsets marking boundaries between tensors
        jlidx (torch.Tensor): List indices for nested jagged structures
        requires_grad (bool): Whether the tensor requires gradient computation
        device (torch.device): Device where the tensor is stored
        dtype (torch.dtype): Data type of the tensor elements
        num_tensors (int): Number of tensors in the sequence
        ldim (int): Dimensionality of the jagged (leading) structure
        edim (int): Dimensionality of the element (regular) structure
        lshape (list[int] | list[list[int]]): Shape(s) of the jagged dimension(s)
        eshape (list[int]): Shape of the element dimensions
        rshape (tuple[int, ...]): Shape of the regular (trailing) dimensions
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
    def from_tensor(cls, data: torch.Tensor) -> "JaggedTensor":
        """
        Create a JaggedTensor from a single torch.Tensor.

        Args:
            data (torch.Tensor): The input tensor.

        Returns:
            JaggedTensor: A new JaggedTensor wrapping the input tensor.
        """
        return cls(tensors=data)

    @classmethod
    def from_list_of_tensors(cls, tensors: Sequence[torch.Tensor]) -> "JaggedTensor":
        """
        Create a JaggedTensor from a sequence of tensors with varying first dimensions.

        All tensors must have the same shape except for the first dimension, which can vary.

        Args:
            tensors (Sequence[torch.Tensor]): List or tuple of tensors with compatible shapes.

        Returns:
            JaggedTensor: A new JaggedTensor containing the sequence of tensors.
        """
        return cls(tensors=tensors)

    @classmethod
    def from_list_of_lists_of_tensors(cls, tensors: Sequence[Sequence[torch.Tensor]]) -> "JaggedTensor":
        """
        Create a JaggedTensor from nested sequences of tensors.

        Creates a multi-level jagged structure where both outer and inner sequences can
        have varying lengths.

        Args:
            tensors (Sequence[Sequence[torch.Tensor]]): Nested list/tuple of tensors.

        Returns:
            JaggedTensor: A new JaggedTensor with nested jagged structure.
        """
        return cls(tensors=tensors)

    @classmethod
    def from_data_and_indices(cls, data: torch.Tensor, indices: torch.Tensor, num_tensors: int) -> "JaggedTensor":
        """
        Create a JaggedTensor from flattened data and per-element indices.

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: (total_elements, ...).
            indices (torch.Tensor): Index tensor mapping each element to its parent tensor.
                Shape: (total_elements,). Values in range [0, num_tensors).
            num_tensors (int): Total number of tensors in the sequence.

        Returns:
            JaggedTensor: A new JaggedTensor constructed from the data and indices.
        """
        return cls(impl=JaggedTensorCpp.from_data_and_indices(data, indices, num_tensors))

    @classmethod
    def from_data_and_offsets(cls, data: torch.Tensor, offsets: torch.Tensor) -> "JaggedTensor":
        """
        Create a JaggedTensor from flattened data and offset array.

        Offsets define boundaries between tensors in the flattened data array.
        Tensor i contains elements data[offsets[i]:offsets[i+1]].

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: (total_elements, ...).
            offsets (torch.Tensor): Offset tensor marking tensor boundaries.
                Shape: (num_tensors + 1,). Must be monotonically increasing.

        Returns:
            JaggedTensor: A new JaggedTensor constructed from the data and offsets.
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
            JaggedTensor: A new JaggedTensor with nested jagged structure.
        """
        return cls(impl=JaggedTensorCpp.from_data_indices_and_list_ids(data, indices, list_ids, num_tensors))

    @classmethod
    def from_data_offsets_and_list_ids(
        cls, data: torch.Tensor, offsets: torch.Tensor, list_ids: torch.Tensor
    ) -> "JaggedTensor":
        """
        Create a nested JaggedTensor from data, offsets, and list IDs.

        Creates a multi-level jagged structure using offsets for boundaries and
        list_ids for nested grouping.

        Args:
            data (torch.Tensor): Flattened data tensor containing all elements.
                Shape: (total_elements, ...).
            offsets (torch.Tensor): Offset tensor marking tensor boundaries.
                Shape: (num_tensors + 1,).
            list_ids (torch.Tensor): List ID tensor for nested structure.
                Shape: (num_tensors,).

        Returns:
            JaggedTensor: A new JaggedTensor with nested jagged structure.
        """
        return cls(impl=JaggedTensorCpp.from_data_offsets_and_list_ids(data, offsets, list_ids))

    # ============================================================
    #                Regular Instance Methods Begin
    # ============================================================

    def abs(self) -> "JaggedTensor":
        """
        Compute the absolute value element-wise.

        Returns:
            JaggedTensor: A new JaggedTensor with absolute values.
        """
        return JaggedTensor(impl=self._impl.abs())

    def abs_(self) -> "JaggedTensor":
        """
        Compute the absolute value element-wise in-place.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.abs_())

    def ceil(self) -> "JaggedTensor":
        """
        Round elements up to the nearest integer.

        Returns:
            JaggedTensor: A new JaggedTensor with ceiling applied.
        """
        return JaggedTensor(impl=self._impl.ceil())

    def ceil_(self) -> "JaggedTensor":
        """
        Round elements up to the nearest integer in-place.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.ceil_())

    def clone(self) -> "JaggedTensor":
        """
        Create a deep copy of the JaggedTensor.

        Returns:
            JaggedTensor: A new JaggedTensor with copied data and structure.
        """
        return JaggedTensor(impl=self._impl.clone())

    def cpu(self) -> "JaggedTensor":
        """
        Move the JaggedTensor to CPU memory.

        Returns:
            JaggedTensor: A new JaggedTensor on CPU device.
        """
        return JaggedTensor(impl=self._impl.cpu())

    def cuda(self) -> "JaggedTensor":
        """
        Move the JaggedTensor to CUDA (GPU) memory.

        Returns:
            JaggedTensor: A new JaggedTensor on CUDA device.
        """
        return JaggedTensor(impl=self._impl.cuda())

    def detach(self) -> "JaggedTensor":
        """
        Detach the JaggedTensor from the autograd graph.

        Returns:
            JaggedTensor: A new JaggedTensor detached from the computation graph.
        """
        return JaggedTensor(impl=self._impl.detach())

    def double(self) -> "JaggedTensor":
        """
        Convert elements to double (float64) dtype.

        Returns:
            JaggedTensor: A new JaggedTensor with double precision.
        """
        return JaggedTensor(impl=self._impl.double())

    def float(self) -> "JaggedTensor":
        """
        Convert elements to float (float32) dtype.

        Returns:
            JaggedTensor: A new JaggedTensor with float32 precision.
        """
        return JaggedTensor(impl=self._impl.float())

    def floor(self) -> "JaggedTensor":
        """
        Round elements down to the nearest integer.

        Returns:
            JaggedTensor: A new JaggedTensor with floor applied.
        """
        return JaggedTensor(impl=self._impl.floor())

    def floor_(self) -> "JaggedTensor":
        """
        Round elements down to the nearest integer in-place.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.floor_())

    def jagged_like(self, data: torch.Tensor) -> "JaggedTensor":
        """
        Create a new JaggedTensor with the same structure but different data.

        The new JaggedTensor will have the same jagged structure (offsets, indices)
        as the current one, but with new data values.

        Args:
            data (torch.Tensor): New data tensor with compatible shape.
                Must have the same leading dimension as self.jdata.

        Returns:
            JaggedTensor: A new JaggedTensor with the same structure but new data.
        """
        return JaggedTensor(impl=self._impl.jagged_like(data))

    def jflatten(self, dim: int = 0) -> "JaggedTensor":
        """
        Flatten the jagged dimensions starting from the specified dimension.

        Args:
            dim (int): The dimension from which to start flattening. Defaults to 0.

        Returns:
            JaggedTensor: A new JaggedTensor with flattened jagged structure.
        """
        return JaggedTensor(impl=self._impl.jflatten(dim))

    def jmax(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        """
        Compute the maximum along a jagged dimension.

        Returns both the maximum values and the indices where they occur.

        Args:
            dim (int): The jagged dimension along which to compute max. Defaults to 0.
            keepdim (bool): Whether to keep the reduced dimension. Defaults to False.

        Returns:
            list[JaggedTensor]: A list containing [values, indices] as JaggedTensors.
        """
        return [JaggedTensor(impl=impl) for impl in self._impl.jmax(dim, keepdim)]

    def jmin(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        """
        Compute the minimum along a jagged dimension.

        Returns both the minimum values and the indices where they occur.

        Args:
            dim (int): The jagged dimension along which to compute min. Defaults to 0.
            keepdim (bool): Whether to keep the reduced dimension. Defaults to False.

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
            JaggedTensor: A new JaggedTensor with masked elements.
        """
        return JaggedTensor(impl=self._impl.rmask(mask))

    def round(self, decimals: int = 0) -> "JaggedTensor":
        """
        Round elements to the specified number of decimals.

        Args:
            decimals (int): Number of decimal places to round to. Defaults to 0.

        Returns:
            JaggedTensor: A new JaggedTensor with rounded values.
        """
        return JaggedTensor(impl=self._impl.round(decimals))

    def round_(self, decimals: int = 0) -> "JaggedTensor":
        """
        Round elements to the specified number of decimals in-place.

        Args:
            decimals (int): Number of decimal places to round to. Defaults to 0.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.round_(decimals))

    def sqrt(self) -> "JaggedTensor":
        """
        Compute the square root element-wise.

        Returns:
            JaggedTensor: A new JaggedTensor with square root applied.
        """
        return JaggedTensor(impl=self._impl.sqrt())

    def sqrt_(self) -> "JaggedTensor":
        """
        Compute the square root element-wise in-place.

        Returns:
            JaggedTensor: The modified JaggedTensor (self).
        """
        return JaggedTensor(impl=self._impl.sqrt_())

    def to(self, device_or_dtype: torch.device | str | torch.dtype) -> "JaggedTensor":
        """
        Move the JaggedTensor to a device or convert to a dtype.

        Args:
            device_or_dtype (torch.device | str | torch.dtype): Target device or dtype.
                Can be a device ("cpu", "cuda"), or a dtype (torch.float32, etc.).

        Returns:
            JaggedTensor: A new JaggedTensor on the specified device or with specified dtype.
        """
        return JaggedTensor(impl=self._impl.to(device_or_dtype))

    def type(self, dtype: torch.dtype) -> "JaggedTensor":
        """
        Convert the JaggedTensor to a specific dtype.

        Args:
            dtype (torch.dtype): Target data type (e.g., torch.float32, torch.int64).

        Returns:
            JaggedTensor: A new JaggedTensor with the specified dtype.
        """
        return JaggedTensor(impl=self._impl.type(dtype))

    def type_as(self, other: "JaggedTensor | torch.Tensor") -> "JaggedTensor":
        """
        Convert the JaggedTensor to match the dtype of another tensor.

        Args:
            other (JaggedTensor | torch.Tensor): Reference tensor whose dtype to match.

        Returns:
            JaggedTensor: A new JaggedTensor with dtype matching other.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl.type_as(other._impl))
        else:
            if not isinstance(other, torch.Tensor):
                raise TypeError("other must be a JaggedTensor or a torch.Tensor")
            return JaggedTensor(impl=self._impl.type_as(other))

    def unbind(self) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        """
        Unbind the JaggedTensor into its constituent tensors.

        Returns:
            list[torch.Tensor] | list[list[torch.Tensor]]: A list of tensors (for simple
                jagged structure) or a list of lists of tensors (for nested structure).
        """
        return self._impl.unbind()

    def __add__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Add another tensor or scalar element-wise.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to add.

        Returns:
            JaggedTensor: Result of element-wise addition.
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
            JaggedTensor: Boolean tensor with element-wise comparison results.
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
            JaggedTensor: Boolean tensor with comparison results.
        """
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl >= other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl >= other)

    def __getitem__(self, index: Any) -> "JaggedTensor":
        """
        Index or slice the JaggedTensor.

        Args:
            index (Any): Index, slice, or mask to apply. Can be a JaggedTensor for
                jagged indexing.

        Returns:
            JaggedTensor: The indexed/sliced JaggedTensor.
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
            JaggedTensor: Boolean tensor with comparison results.
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: The modified JaggedTensor (self).
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
            JaggedTensor: Boolean tensor with comparison results.
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
            int: Number of tensors in the JaggedTensor.
        """
        return len(self._impl)

    def __lt__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        """
        Element-wise less-than comparison.

        Args:
            other (torch.Tensor | JaggedTensor | int | float): Value to compare.

        Returns:
            JaggedTensor: Boolean tensor with comparison results.
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
            JaggedTensor: Boolean tensor with comparison results.
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
            JaggedTensor: A new JaggedTensor with all elements negated.
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

        Returns:
            typing.Iterator[JaggedTensor]: Iterator yielding JaggedTensors.
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
        return self._impl.jdata

    @jdata.setter
    def jdata(self, value: torch.Tensor) -> None:
        self._impl.jdata = value

    @property
    def requires_grad(self) -> bool:
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        # self._impl.set_requires_grad(value)
        self._impl.requires_grad = value

    @property
    def device(self) -> torch.device:
        return self._impl.device

    @property
    def dtype(self) -> torch.dtype:
        return self._impl.dtype

    @property
    def edim(self) -> int:
        return self._impl.edim

    @property
    def eshape(self) -> list[int]:
        return self._impl.eshape

    @property
    def is_cpu(self) -> bool:
        return self._impl.is_cpu

    @property
    def is_cuda(self) -> bool:
        return self._impl.is_cuda

    @property
    def jidx(self) -> torch.Tensor:
        return self._impl.jidx

    @property
    def jlidx(self) -> torch.Tensor:
        return self._impl.jlidx

    @property
    def joffsets(self) -> torch.Tensor:
        return self._impl.joffsets

    @property
    def ldim(self) -> int:
        return self._impl.ldim

    @property
    def lshape(self) -> list[int] | list[list[int]]:
        return self._impl.lshape

    @property
    def num_tensors(self) -> int:
        return self._impl.num_tensors

    @property
    def rshape(self) -> tuple[int, ...]:
        return self._impl.rshape

    # Weirdly, unless we put this last, it messes up static type checking.
    def int(self) -> "JaggedTensor":
        """
        Convert elements to int (int32) dtype.

        Returns:
            JaggedTensor: A new JaggedTensor with int32 dtype.
        """
        return JaggedTensor(impl=self._impl.int())


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


def jempty(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> JaggedTensor:
    """
    Create a JaggedTensor with uninitialized data.

    Similar to torch.empty(), creates a JaggedTensor with allocated but uninitialized
    memory, which is faster than initializing values when they will be immediately
    overwritten.

    Args:
        lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
            Can be a sequence of integers for simple jagged structure, or nested sequences
            for multi-level jagged structure.
        rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
            Defaults to None (scalar elements).
        device (torch.device | str | None): Device to create the tensor on.
            Defaults to None (CPU).
        dtype (torch.dtype | None): Data type for the tensor elements.
            Defaults to None (torch.float32).
        requires_grad (bool): Whether to track gradients. Defaults to False.
        pin_memory (bool): Whether to use pinned memory. Defaults to False.

    Returns:
        JaggedTensor: A new JaggedTensor with uninitialized data.
    """
    lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
    return JaggedTensor(impl=jempty_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))


def jrand(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> JaggedTensor:
    """
    Create a JaggedTensor with random values from uniform distribution [0, 1).

    Similar to torch.rand(), creates a JaggedTensor filled with random values sampled
    from a uniform distribution on the interval [0, 1).

    Args:
        lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
            Can be a sequence of integers for simple jagged structure, or nested sequences
            for multi-level jagged structure.
        rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
            Defaults to None (scalar elements).
        device (torch.device | str | None): Device to create the tensor on.
            Defaults to None (CPU).
        dtype (torch.dtype | None): Data type for the tensor elements.
            Defaults to None (torch.float32).
        requires_grad (bool): Whether to track gradients. Defaults to False.
        pin_memory (bool): Whether to use pinned memory. Defaults to False.

    Returns:
        JaggedTensor: A new JaggedTensor with random values in [0, 1).
    """
    lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
    return JaggedTensor(impl=jrand_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))


def jrandn(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> JaggedTensor:
    """
    Create a JaggedTensor with random values from standard normal distribution.

    Similar to torch.randn(), creates a JaggedTensor filled with random values sampled
    from a standard normal distribution (mean=0, std=1).

    Args:
        lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
            Can be a sequence of integers for simple jagged structure, or nested sequences
            for multi-level jagged structure.
        rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
            Defaults to None (scalar elements).
        device (torch.device | str | None): Device to create the tensor on.
            Defaults to None (CPU).
        dtype (torch.dtype | None): Data type for the tensor elements.
            Defaults to None (torch.float32).
        requires_grad (bool): Whether to track gradients. Defaults to False.
        pin_memory (bool): Whether to use pinned memory. Defaults to False.

    Returns:
        JaggedTensor: A new JaggedTensor with normal random values.
    """
    lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
    return JaggedTensor(impl=jrandn_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))


def jones(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> JaggedTensor:
    """
    Create a JaggedTensor filled with ones.

    Similar to torch.ones(), creates a JaggedTensor where all elements are initialized
    to the value 1.

    Args:
        lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
            Can be a sequence of integers for simple jagged structure, or nested sequences
            for multi-level jagged structure.
        rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
            Defaults to None (scalar elements).
        device (torch.device | str | None): Device to create the tensor on.
            Defaults to None (CPU).
        dtype (torch.dtype | None): Data type for the tensor elements.
            Defaults to None (torch.float32).
        requires_grad (bool): Whether to track gradients. Defaults to False.
        pin_memory (bool): Whether to use pinned memory. Defaults to False.

    Returns:
        JaggedTensor: A new JaggedTensor filled with ones.
    """
    lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
    return JaggedTensor(impl=jones_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))


def jzeros(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> JaggedTensor:
    """
    Create a JaggedTensor filled with zeros.

    Similar to torch.zeros(), creates a JaggedTensor where all elements are initialized
    to the value 0.

    Args:
        lsizes (Sequence[int] | Sequence[Sequence[int]]): Sizes for the jagged dimensions.
            Can be a sequence of integers for simple jagged structure, or nested sequences
            for multi-level jagged structure.
        rsizes (Sequence[int] | None): Sizes for the regular (trailing) dimensions.
            Defaults to None (scalar elements).
        device (torch.device | str | None): Device to create the tensor on.
            Defaults to None (CPU).
        dtype (torch.dtype | None): Data type for the tensor elements.
            Defaults to None (torch.float32).
        requires_grad (bool): Whether to track gradients. Defaults to False.
        pin_memory (bool): Whether to use pinned memory. Defaults to False.

    Returns:
        JaggedTensor: A new JaggedTensor filled with zeros.
    """
    lsizes_cpp: list[int] | list[list[int]] = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = _convert_to_list(rsizes) if rsizes is not None else None
    return JaggedTensor(impl=jzeros_cpp(lsizes_cpp, rsizes_cpp, dtype, device, requires_grad, pin_memory))
