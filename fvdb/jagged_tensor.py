# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Jagged Tensor data structure and operations for FVDB.

Classes:
- JaggedTensor: A jagged tensor data structure with support for efficient operations

TODO(): Add more documentation.
"""

import typing
from typing import TYPE_CHECKING, Any, Sequence, cast

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


class JaggedTensor:
    """
    JaggedTensor data structure with support for efficient operations.

    Members:
        jdata: torch.Tensor
        requires_grad: bool

    TODO(): Add more documentation.
    """

    def __init__(
        self,
        tensors: torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None = None,
        *,
        impl: JaggedTensorCpp | None = None,
    ) -> None:
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
        return cls(tensors=data)

    @classmethod
    def from_list_of_tensors(cls, tensors: Sequence[torch.Tensor]) -> "JaggedTensor":
        return cls(tensors=tensors)

    @classmethod
    def from_list_of_lists_of_tensors(cls, tensors: Sequence[Sequence[torch.Tensor]]) -> "JaggedTensor":
        return cls(tensors=tensors)

    @classmethod
    def from_data_and_indices(cls, data: torch.Tensor, indices: torch.Tensor, num_tensors: int) -> "JaggedTensor":
        return cls(impl=JaggedTensorCpp.from_data_and_indices(data, indices, num_tensors))

    @classmethod
    def from_data_and_offsets(cls, data: torch.Tensor, offsets: torch.Tensor) -> "JaggedTensor":
        return cls(impl=JaggedTensorCpp.from_data_and_offsets(data, offsets))

    @classmethod
    def from_data_indices_and_list_ids(
        cls, data: torch.Tensor, indices: torch.Tensor, list_ids: torch.Tensor, num_tensors: int
    ) -> "JaggedTensor":
        return cls(impl=JaggedTensorCpp.from_data_indices_and_list_ids(data, indices, list_ids, num_tensors))

    @classmethod
    def from_data_offsets_and_list_ids(
        cls, data: torch.Tensor, offsets: torch.Tensor, list_ids: torch.Tensor
    ) -> "JaggedTensor":
        return cls(impl=JaggedTensorCpp.from_data_offsets_and_list_ids(data, offsets, list_ids))

    # ============================================================
    #                Regular Instance Methods Begin
    # ============================================================

    def abs(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.abs())

    def abs_(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.abs_())

    def ceil(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.ceil())

    def ceil_(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.ceil_())

    def clone(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.clone())

    def cpu(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.cpu())

    def cuda(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.cuda())

    def detach(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.detach())

    def double(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.double())

    def float(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.float())

    def floor(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.floor())

    def floor_(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.floor_())

    def jagged_like(self, data: torch.Tensor) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.jagged_like(data))

    def jflatten(self, dim: int = 0) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.jflatten(dim))

    def jmax(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        return [JaggedTensor(impl=impl) for impl in self._impl.jmax(dim, keepdim)]

    def jmin(self, dim: int = 0, keepdim: bool = False) -> list["JaggedTensor"]:
        return [JaggedTensor(impl=impl) for impl in self._impl.jmin(dim, keepdim)]

    def jreshape(self, lshape: Sequence[int] | Sequence[Sequence[int]]) -> "JaggedTensor":
        lshape_cpp = _convert_to_list(lshape)
        return JaggedTensor(impl=self._impl.jreshape(lshape_cpp))

    def jreshape_as(self, other: "JaggedTensor | torch.Tensor") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl.jreshape_as(other._impl))
        else:
            if not isinstance(other, torch.Tensor):
                raise TypeError("other must be a JaggedTensor or a torch.Tensor")
            return JaggedTensor(impl=self._impl.jreshape_as(other))

    def jsqueeze(self, dim: int | None = None) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.jsqueeze(dim))

    def jsum(self, dim: int = 0, keepdim: bool = False) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.jsum(dim, keepdim))

    def long(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.long())

    # FIXME(@chorvath, @fwilliams) Why is this here?
    def requires_grad_(self, requires_grad: bool) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.requires_grad_(requires_grad))

    def rmask(self, mask: torch.Tensor) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.rmask(mask))

    def round(self, decimals: int = 0) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.round(decimals))

    def round_(self, decimals: int = 0) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.round_(decimals))

    def sqrt(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.sqrt())

    def sqrt_(self) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.sqrt_())

    def to(self, device_or_dtype: torch.device | str | torch.dtype) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.to(device_or_dtype))

    def type(self, dtype: torch.dtype) -> "JaggedTensor":
        return JaggedTensor(impl=self._impl.type(dtype))

    def type_as(self, other: "JaggedTensor | torch.Tensor") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl.type_as(other._impl))
        else:
            if not isinstance(other, torch.Tensor):
                raise TypeError("other must be a JaggedTensor or a torch.Tensor")
            return JaggedTensor(impl=self._impl.type_as(other))

    def unbind(self) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        return self._impl.unbind()

    def __add__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl + other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl + other)

    def __eq__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl == other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl == other)

    def __floordiv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl // other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl // other)

    def __ge__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl >= other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl >= other)

    def __getitem__(self, index: Any) -> "JaggedTensor":
        if isinstance(index, JaggedTensor):
            return JaggedTensor(impl=self._impl[index._impl])
        else:
            return JaggedTensor(impl=self._impl[index])

    def __gt__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl > other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl > other)

    def __iadd__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl += other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl += other
        return self

    def __ifloordiv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl //= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl //= other
        return self

    def __imod__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl %= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl %= other
        return self

    def __imul__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl *= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl *= other
        return self

    def __ipow__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl **= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl **= other
        return self

    def __isub__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl -= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl -= other
        return self

    def __itruediv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            self._impl /= other._impl
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            self._impl /= other
        return self

    def __le__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl <= other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl <= other)

    def __len__(self) -> int:
        return len(self._impl)

    def __lt__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl < other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl < other)

    def __mod__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl % other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl % other)

    def __mul__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl * other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl * other)

    def __ne__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl != other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl != other)

    def __neg__(self) -> "JaggedTensor":
        return JaggedTensor(impl=-self._impl)

    def __pow__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl**other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl**other)

    def __sub__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl - other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl - other)

    def __truediv__(self, other: "torch.Tensor | JaggedTensor | int | float") -> "JaggedTensor":
        if isinstance(other, JaggedTensor):
            return JaggedTensor(impl=self._impl / other._impl)
        else:
            if not isinstance(other, (torch.Tensor, int, float)):
                raise TypeError("other must be a torch.Tensor, int, or float")
            return JaggedTensor(impl=self._impl / other)

    def __iter__(self) -> typing.Iterator["JaggedTensor"]:
        for i in range(len(self)):
            yield self[i]

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
        return JaggedTensor(impl=self._impl.int())


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
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> JaggedTensor:
    lsizes_cpp = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = list(rsizes) if rsizes is not None and isinstance(rsizes, tuple) else rsizes  # type: ignore
    return JaggedTensor(impl=jempty_cpp(lsizes_cpp, rsizes_cpp, dtype, device))  # type: ignore


def jrand(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> JaggedTensor:
    lsizes_cpp = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = list(rsizes) if rsizes is not None and isinstance(rsizes, tuple) else rsizes  # type: ignore
    return JaggedTensor(impl=jrand_cpp(lsizes_cpp, rsizes_cpp, dtype, device))  # type: ignore


def jrandn(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> JaggedTensor:
    lsizes_cpp = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = list(rsizes) if rsizes is not None and isinstance(rsizes, tuple) else rsizes  # type: ignore
    return JaggedTensor(impl=jrandn_cpp(lsizes_cpp, rsizes_cpp, dtype, device))  # type: ignore


def jones(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> JaggedTensor:
    lsizes_cpp = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = list(rsizes) if rsizes is not None and isinstance(rsizes, tuple) else rsizes  # type: ignore
    return JaggedTensor(impl=jones_cpp(lsizes_cpp, rsizes_cpp, dtype, device))  # type: ignore


def jzeros(
    lsizes: Sequence[int] | Sequence[Sequence[int]],
    rsizes: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> JaggedTensor:
    lsizes_cpp = _convert_to_list(lsizes)
    rsizes_cpp: list[int] | None = list(rsizes) if rsizes is not None and isinstance(rsizes, tuple) else rsizes  # type: ignore
    return JaggedTensor(impl=jzeros_cpp(lsizes_cpp, rsizes_cpp, dtype, device))  # type: ignore
