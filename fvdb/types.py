# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy
import numpy as np
import torch

from ._Cpp import JaggedTensor

Numeric = int | float

Vec3i = torch.Tensor | numpy.ndarray | list[int] | tuple[int, int, int] | torch.Size
Vec3d = torch.Tensor | numpy.ndarray | list[int | float] | tuple[int | float, int | float, int | float] | torch.Size
Vec3dOrScalar = Vec3d | float | int
Vec3iOrScalar = Vec3i | int
Vec4i = torch.Tensor | numpy.ndarray | list[int] | tuple[int, int, int, int]

Vec3iBatch = (
    Vec3i
    | torch.Tensor
    | numpy.ndarray
    | list[int]
    | list[list[int]]
    | tuple[int, int, int]
    | list[tuple[int, int, int]]
)
Vec3dBatch = (
    torch.Tensor
    | numpy.ndarray
    | list[int | float]
    | list[list[int | float]]
    | tuple[int | float, int | float, int | float]
    | list[tuple[int | float, int | float, int | float]]
    | Vec3iBatch
    | Vec3d
)
Vec3dBatchOrScalar = Vec3dBatch | float | int

Index = int | slice | type(Ellipsis) | None

GridIdentifier = str | int | list[str] | list[int] | tuple[str, ...] | tuple[int, ...]

LShapeSpec = Iterable[int] | Iterable[Iterable[int]]
RShapeSpec = Iterable[int]

JaggedTensorOrTensor = JaggedTensor | torch.Tensor

# New type for GridBatch indexing
GridBatchIndex = int | np.integer | slice | list[bool] | list[int] | torch.Tensor


def is_Numeric(x: Any) -> bool:
    return isinstance(x, (int, float))


def is_Vec3i(x: Any) -> bool:
    if isinstance(x, torch.Size):
        return len(x) == 3
    if isinstance(x, (torch.Tensor, numpy.ndarray)):
        return x.shape == (3,) and x.dtype in (torch.int32, torch.int64, numpy.int32, numpy.int64)
    if isinstance(x, list):
        return len(x) == 3 and all(isinstance(i, int) for i in x)
    if isinstance(x, tuple):
        return len(x) == 3 and all(isinstance(i, int) for i in x)
    return False


def is_Vec3d(x: Any) -> bool:
    if isinstance(x, (torch.Tensor, numpy.ndarray)):
        return x.shape == (3,) and x.dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            numpy.float32,
            numpy.float64,
        )
    if isinstance(x, list):
        return len(x) == 3 and all(isinstance(i, (int, float)) for i in x)
    if isinstance(x, tuple):
        return len(x) == 3 and all(isinstance(i, (int, float)) for i in x)
    if isinstance(x, torch.Size):
        return len(x) == 3
    return False


def is_Vec3dOrScalar(x: Any) -> bool:
    return is_Vec3d(x) or isinstance(x, (float, int))


def is_Vec3iOrScalar(x: Any) -> bool:
    return is_Vec3i(x) or isinstance(x, int)


def is_Vec4i(x: Any) -> bool:
    if isinstance(x, (torch.Tensor, numpy.ndarray)):
        return x.shape == (4,) and x.dtype in (torch.int32, torch.int64, numpy.int32, numpy.int64)
    if isinstance(x, list):
        return len(x) == 4 and all(isinstance(i, int) for i in x)
    if isinstance(x, tuple):
        return len(x) == 4 and all(isinstance(i, int) for i in x)
    return False


def is_Vec3iBatch(x: Any) -> bool:
    if is_Vec3i(x):
        return True
    if isinstance(x, (torch.Tensor, numpy.ndarray)):
        return (
            len(x.shape) >= 1 and x.shape[-1] == 3 and x.dtype in (torch.int32, torch.int64, numpy.int32, numpy.int64)
        )
    if isinstance(x, list):
        if len(x) == 0:
            return True
        if isinstance(x[0], int):
            return True  # list[int]
        if isinstance(x[0], list):
            return all(len(item) == 3 and all(isinstance(i, int) for i in item) for item in x)  # list[list[int]]
        if isinstance(x[0], tuple):
            return all(
                len(item) == 3 and all(isinstance(i, int) for i in item) for item in x
            )  # list[tuple[int, int, int]]
    return False


def is_Vec3dBatch(x: Any) -> bool:
    if is_Vec3iBatch(x) or is_Vec3d(x):
        return True
    if isinstance(x, (torch.Tensor, numpy.ndarray)):
        return (
            len(x.shape) >= 1
            and x.shape[-1] == 3
            and x.dtype in (torch.float16, torch.float32, torch.float64, numpy.float32, numpy.float64)
        )
    if isinstance(x, list):
        if len(x) == 0:
            return True
        if isinstance(x[0], (int, float)):
            return True  # list[int|float]
        if isinstance(x[0], list):
            return all(
                len(item) == 3 and all(isinstance(i, (int, float)) for i in item) for item in x
            )  # list[list[int|float]]
        if isinstance(x[0], tuple):
            return all(
                len(item) == 3 and all(isinstance(i, (int, float)) for i in item) for item in x
            )  # list[tuple[int|float, int|float, int|float]]
    return False


def is_Vec3dBatchOrScalar(x: Any) -> bool:
    return is_Vec3dBatch(x) or isinstance(x, (float, int))


def is_Index(x: Any) -> bool:
    return isinstance(x, (int, slice, type(Ellipsis), type(None)))


def is_GridIdentifier(x: Any) -> bool:
    if isinstance(x, (str, int)):
        return True
    if isinstance(x, list):
        return all(isinstance(item, (str, int)) for item in x)
    if isinstance(x, tuple):
        return all(isinstance(item, (str, int)) for item in x)
    return False


def is_LShapeSpec(x: Any) -> bool:
    try:
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            # Check if it's Iterable[int]
            if all(isinstance(item, int) for item in x):
                return True
            # Check if it's Iterable[Iterable[int]]
            if all(
                hasattr(item, "__iter__")
                and not isinstance(item, (str, bytes))
                and all(isinstance(i, int) for i in item)
                for item in x
            ):
                return True
    except (TypeError, ValueError):
        pass
    return False


def is_RShapeSpec(x: Any) -> bool:
    try:
        return hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) and all(isinstance(item, int) for item in x)
    except (TypeError, ValueError):
        return False


def is_JaggedTensorOrTensor(x: Any) -> bool:
    return isinstance(x, (JaggedTensor, torch.Tensor))


# Corresponding validation function
def is_GridBatchIndex(x: Any) -> bool:
    if isinstance(x, (int, np.integer, slice)):
        return True
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, list):
        if len(x) == 0:
            return True  # Empty list is valid
        # Check if it's list[bool] or list[int]
        return all(isinstance(item, (bool, int)) for item in x)
    return False
