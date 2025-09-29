# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto
from typing import Any, Sequence, TypeGuard, TypeVar, cast

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


# ----------------------------------------------------------------------------------------------------------------------
# REDUX
# ----------------------------------------------------------------------------------------------------------------------


T = TypeVar("T")


def cast_check(x: object, expected_type: type[T], name: str) -> T:
    """
    Checks if x is of type expected_type, raises TypeError if not.
    Returns True if x is of type T (for use as a TypeGuard).
    """
    if not isinstance(x, expected_type):
        raise TypeError(f"Expected {name} to be a {expected_type}, got {type(x)}")
    return cast(T, x)


DeviceIdentifier = str | torch.device
NumericScalarNative = int | float | np.integer | np.floating
NumericScalar = torch.Tensor | numpy.ndarray | NumericScalarNative
NumericMaxRank1 = NumericScalar | Sequence[NumericScalarNative] | torch.Size
NumericMaxRank2 = NumericMaxRank1 | Sequence[Sequence[NumericScalarNative]]
NumericMaxRank3 = NumericMaxRank2 | Sequence[Sequence[Sequence[NumericScalarNative]]]


def is_DeviceIdentifier(x: Any) -> TypeGuard[DeviceIdentifier]:
    return isinstance(x, (str, torch.device))


def is_NumericScalarNative(x: Any) -> TypeGuard[NumericScalarNative]:
    return isinstance(x, (int, float, np.integer, np.floating))


def is_NumericScalar(x: Any) -> TypeGuard[NumericScalar]:
    return is_NumericScalarNative(x) or (isinstance(x, (torch.Tensor, numpy.ndarray)) and x.ndim == 0)


def is_SequenceOfNumericScalarNative(x: Any) -> TypeGuard[Sequence[NumericScalarNative]]:
    return isinstance(x, Sequence) and all(is_NumericScalarNative(item) for item in x)


def is_SequenceOfSequenceOfNumericScalarNative(x: Any) -> TypeGuard[Sequence[Sequence[NumericScalarNative]]]:
    return isinstance(x, Sequence) and all(is_SequenceOfNumericScalarNative(item) for item in x)


def is_SequenceOfSequenceOfSequenceOfNumericScalarNative(
    x: Any,
) -> TypeGuard[Sequence[Sequence[Sequence[NumericScalarNative]]]]:
    return isinstance(x, Sequence) and all(is_SequenceOfSequenceOfNumericScalarNative(item) for item in x)


def is_NumericMaxRank1(x: Any) -> TypeGuard[NumericMaxRank1]:
    return (
        is_NumericScalar(x)
        or is_SequenceOfNumericScalarNative(x)
        or isinstance(x, torch.Size)
        or (isinstance(x, (torch.Tensor, numpy.ndarray)) and x.ndim == 1)
    )


def is_NumericMaxRank2(x: Any) -> TypeGuard[NumericMaxRank2]:
    return (
        is_NumericMaxRank1(x)
        or is_SequenceOfSequenceOfNumericScalarNative(x)
        or (isinstance(x, (torch.Tensor, numpy.ndarray)) and x.ndim == 2)
    )


def is_NumericMaxRank3(x: Any) -> TypeGuard[NumericMaxRank3]:
    return (
        is_NumericMaxRank2(x)
        or is_SequenceOfSequenceOfSequenceOfNumericScalarNative(x)
        or (isinstance(x, (torch.Tensor, numpy.ndarray)) and x.ndim == 3)
    )


def resolve_device(device_id: DeviceIdentifier | None, inherit_from: Any = None) -> torch.device:
    """
    Resolve the target device for a tensor operation.

    The device_id argument always takes precedence over the inherit_from argument.
    If device_id is specified, normalize it (with explicit indices for CUDA).
    If device_id is None, inherit the device from inherit_from:
    - Python objects: use "cpu"
    - NumPy objects: use "cpu"
    - Torch objects: use inherit_from.device (preserved as-is, no normalization)

    Args:
        device_id: Device specification or None to inherit from inherit_from.
                   This argument always takes precedence over inherit_from when provided.
        inherit_from: Object to potentially inherit device from when device_id is None

    Returns:
        torch.device: The resolved target device with explicit indices when normalized

    Examples:
        >>> resolve_device("cuda")  # -> torch.device("cuda", 0)
        >>> resolve_device("cpu")  # -> torch.device("cpu")
        >>> resolve_device(torch.device("cuda"))  # -> torch.device("cuda", 0) (normalized)
        >>> resolve_device(None, torch.tensor([1, 2, 3]))  # -> inherits from tensor
        >>> resolve_device(None, [1, 2, 3])  # -> torch.device("cpu")
        >>> resolve_device(None)  # -> torch.device("cpu")
    """
    if device_id is not None:
        # Normalize the provided device
        if not isinstance(device_id, (str, torch.device)):
            raise TypeError(f"Expected DeviceIdentifier, got {type(device_id)}")

        device = torch.device(device_id)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    # device_id is None - inherit from inherit_from
    if inherit_from is not None and isinstance(inherit_from, (torch.Tensor, JaggedTensor)):
        return inherit_from.device  # Preserve original device without normalization
    elif hasattr(inherit_from, "device") and isinstance(inherit_from.device, torch.device):
        return inherit_from.device
    else:
        # Python objects, NumPy objects, None, etc. -> use CPU
        return torch.device("cpu")


# ============================================================
# ===           Begin "to_***" validation convertors        ===
# ============================================================


class ValueConstraint(Enum):
    NONE = auto()
    NON_NEGATIVE = auto()
    POSITIVE = auto()


def to_GenericScalar(
    x: NumericScalar,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Generic function to convert a NumericScalar to a scalar tensor.

    Args:
        x: The input scalar value.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages (e.g., "int", "float").
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A scalar torch.Tensor of the specified dtype on the specified device.
    """
    if not is_NumericScalar(x):
        raise TypeError(f"Expected NumericScalar, got {type(x)}")

    if dtype not in allowed_torch_dtypes:
        raise ValueError(f"Expected {dtype_category} dtype, got {dtype}")

    if isinstance(x, torch.Tensor):
        if x.ndim != 0:
            raise ValueError(f"Expected scalar tensor, got {x.shape}")
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected scalar tensor with {dtype_category} dtype, got {x.dtype}")
        result = x.to(dtype)

    elif isinstance(x, numpy.ndarray):
        if x.ndim != 0:
            raise ValueError(f"Expected scalar array, got {x.shape}")
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected scalar array with {dtype_category} dtype, got {x.dtype}")
        result = torch.from_numpy(x).to(dtype)

    else:
        # Validate native Python scalars against allowed types
        if dtype_category == "int":
            if not isinstance(x, (int, np.integer)):
                raise TypeError(f"Expected integer scalar, got {type(x)} with value {x}")
        elif dtype_category == "float":
            if not isinstance(x, (int, float, np.integer, np.floating)):
                raise TypeError(f"Expected numeric scalar, got {type(x)} with value {x}")
        elif dtype_category == "int or float":
            if not isinstance(x, (int, float, np.integer, np.floating)):
                raise TypeError(f"Expected numeric scalar, got {type(x)} with value {x}")

        result = torch.tensor(x, device="cpu", dtype=dtype)

    if value_constraint == ValueConstraint.NON_NEGATIVE:
        if torch.any(result < 0):
            raise ValueError(f"Expected non-negative values, got {result}")
    elif value_constraint == ValueConstraint.POSITIVE:
        if torch.any(result <= 0):
            raise ValueError(f"Expected positive values, got {result}")

    return result


def to_GenericTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Generic function to convert a NumericMaxRank1 to a tensor broadcastable against test_shape.

    Args:
        x: The input tensor.
        test_shape: The shape to test broadcastability against.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False

    Returns:
        A torch.Tensor of the specified dtype and device.
    """
    if not is_NumericMaxRank1(x):
        raise TypeError(f"Expected NumericMaxRank1, got {type(x)}")

    if len(test_shape) != 1:
        raise ValueError(f"Expected test_shape of rank 1, got {test_shape}")

    if dtype not in allowed_torch_dtypes:
        raise ValueError(f"Expected {dtype_category} dtype, got {dtype}")

    if is_NumericScalar(x):
        result = to_GenericScalar(
            x, dtype, allowed_torch_dtypes, allowed_numpy_dtypes, dtype_category, value_constraint=ValueConstraint.NONE
        )
        try:
            result_shape = torch.broadcast_shapes(result.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Scalar with shape {result.shape} cannot broadcast to {test_shape}: {e}")

    elif is_SequenceOfNumericScalarNative(x):
        x_shape = (len(x),)
        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"Sequence with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        result = torch.tensor(x, device="cpu", dtype=dtype)
    elif isinstance(x, torch.Size):
        x_shape = (len(x),)
        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"torch.Size with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        result = torch.tensor(x, device="cpu", dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected tensor with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 1
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = x.to(dtype)
    elif isinstance(x, numpy.ndarray):
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected array with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 1
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Array with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = torch.from_numpy(x).to(dtype)

    else:
        raise TypeError(f"Expected NumericMaxRank1, got {type(x)}")

    if value_constraint == ValueConstraint.NON_NEGATIVE:
        if torch.any(result < 0):
            raise ValueError(f"Expected non-negative values, got {result}")
    elif value_constraint == ValueConstraint.POSITIVE:
        if torch.any(result <= 0):
            raise ValueError(f"Expected positive values, got {result}")

    if do_broadcast_to:
        result = result.broadcast_to(result_shape)

    return result


def to_GenericTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Generic function to convert a NumericMaxRank2 to a tensor broadcastable against test_shape.

    Args:
        x: The input tensor.
        test_shape: The shape to test broadcastability against.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False

    Returns:
        A torch.Tensor of the specified dtype and device.
    """
    if not is_NumericMaxRank2(x):
        raise TypeError(f"Expected NumericMaxRank2, got {type(x)}")

    if len(test_shape) != 2:
        raise ValueError(f"Expected test_shape of rank 2, got {test_shape}")

    if dtype not in allowed_torch_dtypes:
        raise ValueError(f"Expected {dtype_category} dtype, got {dtype}")

    if is_NumericMaxRank1(x):
        result = to_GenericTensorBroadcastableRank1(
            x,
            (test_shape[1],),
            dtype,
            allowed_torch_dtypes,
            allowed_numpy_dtypes,
            dtype_category,
            value_constraint=ValueConstraint.NONE,
            do_broadcast_to=False,
        )

        try:
            result_shape = torch.broadcast_shapes(result.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {result.shape} cannot broadcast to {test_shape}: {e}")

    elif is_SequenceOfSequenceOfNumericScalarNative(x):
        rank_2_size = len(x)

        # test that all the rank 1 sizes are the same
        rank_1_sizes = [len(sub_sequence) for sub_sequence in x]
        if not all(size == rank_1_sizes[0] for size in rank_1_sizes):
            raise ValueError(f"All rank 1 sizes must be the same, got {rank_1_sizes}")

        x_shape = (rank_2_size, rank_1_sizes[0])

        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"Sequence with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        result = torch.tensor(x, device="cpu", dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected tensor with {dtype_category} dtype, got {x.dtype}")

        # This assertion is true because we already checked numeric rank 1 above.
        assert x.ndim == 2
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = x.to(dtype)
    elif isinstance(x, numpy.ndarray):
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected array with {dtype_category} dtype, got {x.dtype}")

        # This assertion is true because we already checked numeric rank 1 above.
        assert x.ndim == 2
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Array with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = torch.from_numpy(x).to(dtype)

    else:
        raise TypeError(f"Expected NumericMaxRank2, got {type(x)}")

    if value_constraint == ValueConstraint.NON_NEGATIVE:
        if torch.any(result < 0):
            raise ValueError(f"Expected non-negative scalar, got {result}")
    elif value_constraint == ValueConstraint.POSITIVE:
        if torch.any(result <= 0):
            raise ValueError(f"Expected positive scalar, got {result}")

    if do_broadcast_to:
        result = result.broadcast_to(result_shape)

    return result


def to_GenericTensorBroadcastableRank3(
    x: NumericMaxRank3,
    test_shape: tuple[int, int, int] | torch.Size,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Generic function to convert a NumericMaxRank3 to a tensor broadcastable against test_shape.

    Args:
        x: The input tensor.
        test_shape: The shape to test broadcastability against.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False

    Returns:
        A torch.Tensor of the specified dtype and device.
    """
    if not is_NumericMaxRank3(x):
        raise TypeError(f"Expected NumericMaxRank3, got {type(x)}")

    if len(test_shape) != 3:
        raise ValueError(f"Expected test_shape of rank 3, got {test_shape}")

    if dtype not in allowed_torch_dtypes:
        raise ValueError(f"Expected {dtype_category} dtype, got {dtype}")

    if is_NumericMaxRank2(x):
        result = to_GenericTensorBroadcastableRank2(
            x,
            (test_shape[1], test_shape[2]),
            dtype,
            allowed_torch_dtypes,
            allowed_numpy_dtypes,
            dtype_category,
            value_constraint=value_constraint,
            do_broadcast_to=do_broadcast_to,
        )
        try:
            result_shape = torch.broadcast_shapes(result.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {result.shape} cannot broadcast to {test_shape}: {e}")

    elif is_SequenceOfSequenceOfSequenceOfNumericScalarNative(x):
        rank_3_size = len(x)

        # test that all the rank 1 sizes are the same
        rank_2_sizes = [len(sub_sequence) for sub_sequence in x]
        rank_1_sizes = [len(sub_sub_sequence) for sub_sequence in x for sub_sub_sequence in sub_sequence]
        if not all(size == rank_2_sizes[0] for size in rank_2_sizes):
            raise ValueError(f"All rank 2 sizes must be the same, got {rank_2_sizes}")
        if not all(size == rank_1_sizes[0] for size in rank_1_sizes):
            raise ValueError(f"All rank 1 sizes must be the same, got {rank_1_sizes}")
        x_shape = (rank_3_size, rank_2_sizes[0], rank_1_sizes[0])

        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"Sequence with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        result = torch.tensor(x, device="cpu", dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected tensor with {dtype_category} dtype, got {x.dtype}")

        # This assertion is true because we already checked numeric rank 1 above.
        assert x.ndim == 3
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = x.to(dtype)
    elif isinstance(x, numpy.ndarray):
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected array with {dtype_category} dtype, got {x.dtype}")

        # This assertion is true because we already checked numeric rank 1 above.
        assert x.ndim == 3
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Array with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        result = torch.from_numpy(x).to(dtype)

    else:
        raise TypeError(f"Expected NumericMaxRank2, got {type(x)}")

    if value_constraint == ValueConstraint.NON_NEGATIVE:
        if torch.any(result < 0):
            raise ValueError(f"Expected non-negative scalar, got {result}")
    elif value_constraint == ValueConstraint.POSITIVE:
        if torch.any(result <= 0):
            raise ValueError(f"Expected positive scalar, got {result}")

    if do_broadcast_to:
        result = result.broadcast_to(result_shape)

    return result


def to_IntegerScalar(
    x: NumericScalar, dtype: torch.dtype = torch.int64, value_constraint: ValueConstraint = ValueConstraint.NONE
) -> torch.Tensor:
    """
    Converts a NumericScalar to an integer scalar tensor.

    Args:
        x (NumericScalar): The input scalar value.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A scalar, integer torch.Tensor of dtype on the inherited, requested, or default device.
    """
    return to_GenericScalar(
        x,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64),
        dtype_category="int",
        value_constraint=value_constraint,
    )


def to_FloatingScalar(
    x: NumericScalar, dtype: torch.dtype = torch.float32, value_constraint: ValueConstraint = ValueConstraint.NONE
) -> torch.Tensor:
    """
    Converts a NumericScalar to a floating scalar tensor.

    Args:
        x (NumericScalar): The input scalar value.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A scalar, floating torch.Tensor of dtype on the inherited, requested, or default device.
    """
    return to_GenericScalar(
        x,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
        value_constraint=value_constraint,
    )


def to_IntegerTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to an integer tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 1 shape.

    Args:
        x (NumericMaxRank1): The input tensor.
        test_shape (tuple[int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank1(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64),
        dtype_category="int",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_FloatingTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a floating tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 1 shape.

    Args:
        x (NumericMaxRank1): The input tensor.
        test_shape (tuple[int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank1(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_IntegerTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to an integer tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 2 shape.

    Args:
        x (NumericMaxRank2): The input tensor.
        test_shape (tuple[int, int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank2(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64),
        dtype_category="int",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_IntegerTensorBroadcastableRank3(
    x: NumericMaxRank3,
    test_shape: tuple[int, int, int] | torch.Size,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank3 to an integer tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 3 shape.

    Args:
        x (NumericMaxRank3): The input tensor.
        test_shape (tuple[int, int, int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank3(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64),
        dtype_category="int",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_FloatingTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a floating tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 2 shape.

    Args:
        x (NumericMaxRank2): The input tensor.
        test_shape (tuple[int, int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank2(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_FloatingTensorBroadcastableRank3(
    x: NumericMaxRank3,
    test_shape: tuple[int, int, int] | torch.Size,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
    do_broadcast_to: bool = False,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank3 to a floating tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 3 shape.

    Args:
        x (NumericMaxRank3): The input tensor.
        test_shape (tuple[int, int, int]|torch.Size): The shape to test the broadcastability against.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
        do_broadcast_to: If True, the tensor will be broadcast to the broadcast of the test_shape and the tensor's shape
            default: False
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_GenericTensorBroadcastableRank3(
        x,
        test_shape,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
        value_constraint=value_constraint,
        do_broadcast_to=do_broadcast_to,
    )


def to_Vec3iBroadcastable(
    x: NumericMaxRank1,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3i tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (3,).

    Args:
        x (NumericMaxRank1): The input tensor.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_IntegerTensorBroadcastableRank1(x, (3,), dtype, value_constraint=value_constraint, do_broadcast_to=False)


def to_Vec3fBroadcastable(
    x: NumericMaxRank1,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3f tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (3,).

    Args:
        x (NumericMaxRank1): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_FloatingTensorBroadcastableRank1(x, (3,), dtype, value_constraint=value_constraint, do_broadcast_to=False)


def to_Vec3iBatchBroadcastable(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3iBatch tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (1, 3).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_IntegerTensorBroadcastableRank2(
        x, (1, 3), dtype, value_constraint=value_constraint, do_broadcast_to=False
    )


def to_Vec3fBatchBroadcastable(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3fBatch tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (1, 3).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_FloatingTensorBroadcastableRank2(
        x, (1, 3), dtype, value_constraint=value_constraint, do_broadcast_to=False
    )


def to_Vec3i(
    x: NumericMaxRank1,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3i tensor, reshaped to (3,).

    Args:
        x (NumericMaxRank1): The input tensor.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_IntegerTensorBroadcastableRank1(x, (3,), dtype, value_constraint=value_constraint, do_broadcast_to=True)


def to_Vec3f(
    x: NumericMaxRank1,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3f tensor, reshaped to (3,).

    Args:
        x (NumericMaxRank1): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive
    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_FloatingTensorBroadcastableRank1(x, (3,), dtype, value_constraint=value_constraint, do_broadcast_to=True)


def to_Vec3iBatch(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.int64,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3iBatch tensor, reshaped to the broadcast of the input shape and (1, 3).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_IntegerTensorBroadcastableRank2(x, (1, 3), dtype, value_constraint=value_constraint, do_broadcast_to=True)


def to_Vec3fBatch(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3fBatch tensor, reshaped to the broadcast of the input shape and (1, 3).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype.
    """
    return to_FloatingTensorBroadcastableRank2(
        x, (1, 3), dtype, value_constraint=value_constraint, do_broadcast_to=True
    )


def to_Mat44f(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Mat44f tensor, reshaped to (4, 4).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype and shape (4, 4).
    """
    return to_FloatingTensorBroadcastableRank2(
        x, (4, 4), dtype, value_constraint=value_constraint, do_broadcast_to=True
    )


def to_Mat44fBroadcastable(
    x: NumericMaxRank2,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Mat44f tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (4, 4).

    Args:
        x (NumericMaxRank2): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype and shape (4, 4).
    """
    return to_FloatingTensorBroadcastableRank2(
        x, (4, 4), dtype, value_constraint=value_constraint, do_broadcast_to=False
    )


def to_Mat44fBatch(
    x: NumericMaxRank3,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank3 to a Mat44fBatch tensor, reshaped to the broadcast of the input shape and (1, 4, 4).

    Args:
        x (NumericMaxRank3): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype and shape reshaped to the broadcast of the input shape and (1, 4, 4).
    """
    return to_FloatingTensorBroadcastableRank3(
        x, (1, 4, 4), dtype, value_constraint=value_constraint, do_broadcast_to=True
    )


def to_Mat44fBatchBroadcastable(
    x: NumericMaxRank3,
    dtype: torch.dtype = torch.float32,
    value_constraint: ValueConstraint = ValueConstraint.NONE,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Mat44f tensor that respects the shape of the input without reshaping,
    but verifies that it can be broadcasted to the shape (1, 4, 4).

    Args:
        x (NumericMaxRank3): The input tensor.
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.
        value_constraint: Constraint on the value of the scalar.
            default: ValueConstraint.NONE
            if ValueConstraint.NON_NEGATIVE, the scalar must be non-negative
            if ValueConstraint.POSITIVE, the scalar must be positive

    Returns:
        A torch.Tensor of dtype dtype and shape broadcastable to (1, 4, 4).
    """
    return to_FloatingTensorBroadcastableRank3(
        x, (1, 4, 4), dtype, value_constraint=value_constraint, do_broadcast_to=False
    )
