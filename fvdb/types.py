# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections.abc import Iterable
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


def to_GenericScalar(
    x: NumericScalar,
    device: DeviceIdentifier | None,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
) -> torch.Tensor:
    """
    Generic function to convert a NumericScalar to a scalar tensor.

    Args:
        x: The input scalar value.
        device: The optional device to place the output tensor on.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages (e.g., "int", "float").

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
        device = resolve_device(device, inherit_from=x)
        return x.to(device).to(dtype)

    elif isinstance(x, numpy.ndarray):
        if x.ndim != 0:
            raise ValueError(f"Expected scalar array, got {x.shape}")
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected scalar array with {dtype_category} dtype, got {x.dtype}")
        device = resolve_device(device, inherit_from=x)
        return torch.from_numpy(x).to(device).to(dtype)

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

        device = resolve_device(device, inherit_from=x)
        return torch.tensor(x, device=device, dtype=dtype)


def to_GenericTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    device: DeviceIdentifier | None,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
) -> torch.Tensor:
    """
    Generic function to convert a NumericMaxRank1 to a tensor broadcastable against test_shape.

    Args:
        x: The input tensor.
        test_shape: The shape to test broadcastability against.
        device: The optional device to place the output tensor on.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages.

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
        return to_GenericScalar(x, device, dtype, allowed_torch_dtypes, allowed_numpy_dtypes, dtype_category)
    elif is_SequenceOfNumericScalarNative(x):
        x_shape = (len(x),)
        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"Sequence with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, torch.Size):
        x_shape = (len(x),)
        try:
            result_shape = torch.broadcast_shapes(x_shape, test_shape)
        except Exception as e:
            raise ValueError(f"torch.Size with shape {x_shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected tensor with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 1
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return x.to(device).to(dtype)
    elif isinstance(x, numpy.ndarray):
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected array with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 1
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Array with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return torch.from_numpy(x).to(device).to(dtype)

    else:
        raise TypeError(f"Expected NumericMaxRank1, got {type(x)}")


def to_GenericTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    device: DeviceIdentifier | None,
    dtype: torch.dtype,
    allowed_torch_dtypes: tuple[torch.dtype, ...],
    allowed_numpy_dtypes: tuple[np.dtype | type, ...],
    dtype_category: str,
) -> torch.Tensor:
    """
    Generic function to convert a NumericMaxRank2 to a tensor broadcastable against test_shape.

    Args:
        x: The input tensor.
        test_shape: The shape to test broadcastability against.
        device: The optional device to place the output tensor on.
        dtype: The dtype of the output tensor.
        allowed_torch_dtypes: Allowed torch dtypes for validation.
        allowed_numpy_dtypes: Allowed numpy dtypes for validation.
        dtype_category: String describing dtype category for error messages.

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
        return to_GenericTensorBroadcastableRank1(
            x, (test_shape[1],), device, dtype, allowed_torch_dtypes, allowed_numpy_dtypes, dtype_category
        )
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

        device = resolve_device(device, inherit_from=x)
        return torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype not in allowed_torch_dtypes:
            raise ValueError(f"Expected tensor with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 2
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Tensor with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return x.to(device).to(dtype)
    elif isinstance(x, numpy.ndarray):
        if x.dtype not in allowed_numpy_dtypes:
            raise ValueError(f"Expected array with {dtype_category} dtype, got {x.dtype}")

        assert x.ndim == 2
        try:
            result_shape = torch.broadcast_shapes(x.shape, test_shape)
        except Exception as e:
            raise ValueError(f"Array with shape {x.shape} cannot broadcast to {test_shape}: {e}")

        device = resolve_device(device, inherit_from=x)
        return torch.from_numpy(x).to(device).to(dtype)

    else:
        raise TypeError(f"Expected NumericMaxRank2, got {type(x)}")


def to_IntegerScalar(
    x: NumericScalar, device: DeviceIdentifier | None = None, dtype: torch.dtype = torch.int64
) -> torch.Tensor:
    """
    Converts a NumericScalar to an integer scalar tensor.

    Args:
        x (NumericScalar): The input scalar value.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A scalar, integer torch.Tensor of dtype on the inherited, requested, or default device.
    """
    return to_GenericScalar(
        x,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64),
        dtype_category="int",
    )


def to_FloatingScalar(
    x: NumericScalar, device: DeviceIdentifier | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Converts a NumericScalar to a floating scalar tensor.

    Args:
        x (NumericScalar): The input scalar value.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A scalar, floating torch.Tensor of dtype on the inherited, requested, or default device.
    """
    return to_GenericScalar(
        x,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
    )


def to_IntegerTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to an integer tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 1 shape.

    Args:
        x (NumericMaxRank1): The input tensor.
        test_shape (tuple[int]|torch.Size): The shape to test the broadcastability against.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_GenericTensorBroadcastableRank1(
        x,
        test_shape,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64),
        dtype_category="int",
    )


def to_FloatingTensorBroadcastableRank1(
    x: NumericMaxRank1,
    test_shape: tuple[int] | torch.Size,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a floating tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 1 shape.

    Args:
        x (NumericMaxRank1): The input tensor.
        test_shape (tuple[int]|torch.Size): The shape to test the broadcastability against.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_GenericTensorBroadcastableRank1(
        x,
        test_shape,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
    )


def to_IntegerTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to an integer tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 2 shape.

    Args:
        x (NumericMaxRank2): The input tensor.
        test_shape (tuple[int, int]|torch.Size): The shape to test the broadcastability against.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_GenericTensorBroadcastableRank2(
        x,
        test_shape,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64),
        allowed_numpy_dtypes=(np.int32, np.int64),
        dtype_category="int",
    )


def to_FloatingTensorBroadcastableRank2(
    x: NumericMaxRank2,
    test_shape: tuple[int, int] | torch.Size,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a floating tensor that is broadcastable against the given test_shape.
    The test_shape must be a valid rank 2 shape.

    Args:
        x (NumericMaxRank2): The input tensor.
        test_shape (tuple[int, int]|torch.Size): The shape to test the broadcastability against.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_GenericTensorBroadcastableRank2(
        x,
        test_shape,
        device,
        dtype,
        allowed_torch_dtypes=(torch.int32, torch.int64, torch.float16, torch.float32, torch.float64),
        allowed_numpy_dtypes=(np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64),
        dtype_category="int or float",
    )


def to_Vec3iLike(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3i.

    Args:
        x (NumericMaxRank1): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_IntegerTensorBroadcastableRank1(x, (3,), device, dtype)


def to_Vec3fLike(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3f.

    Args:
        x (NumericMaxRank1): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_FloatingTensorBroadcastableRank1(x, (3,), device, dtype)


def to_Vec3iBatchLike(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3iBatch.

    Args:
        x (NumericMaxRank2): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_IntegerTensorBroadcastableRank2(x, (1, 3), device, dtype)


def to_Vec3fBatchLike(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3fBatch.

    Args:
        x (NumericMaxRank2): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    return to_FloatingTensorBroadcastableRank2(x, (1, 3), device, dtype)


def to_Vec3i(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3i.

    Args:
        x (NumericMaxRank1): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    try:
        return to_Vec3iLike(x, device, dtype).broadcast_to((3,))
    except Exception as e:
        raise ValueError(f"Failed to broadcast {x} to Vec3i: {e}")


def to_Vec3f(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank1 to a Vec3f.

    Args:
        x (NumericMaxRank1): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    try:
        return to_Vec3fLike(x, device, dtype).broadcast_to((3,))
    except Exception as e:
        raise ValueError(f"Failed to broadcast {x} to Vec3f: {e}")


def to_Vec3iBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3iBatch.

    Args:
        x (NumericMaxRank2): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The integer dtype of the output tensor. Defaults to torch.int64.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    tensor = to_Vec3iBatchLike(x, device, dtype)

    # Determine the batch size 'N' from the resulting tensor's shape.
    if tensor.ndim == 2:
        batch_size = tensor.shape[0]
    else:
        # If the tensor is rank 1 or 0, it represents a single item for the batch.
        batch_size = 1

    # Define the final, non-negotiable target shape.
    target_shape = (batch_size, 3)

    # Explicitly broadcast the tensor to the target shape.
    try:
        return tensor.broadcast_to(target_shape)
    except Exception as e:
        raise ValueError(f"Failed to broadcast {x} to Vec3iBatch: {e}")


def to_Vec3fBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts a NumericMaxRank2 to a Vec3fBatch.

    Args:
        x (NumericMaxRank2): The input tensor.
        device (DeviceIdentifier | None): The optional device to place the output tensor on.
          If None, defaults to device of the torch input if it can, otherwise "cpu"
        dtype (torch.dtype): The floating dtype of the output tensor. Defaults to torch.float32.

    Returns:
        A torch.Tensor of dtype dtype and device device.
    """
    tensor = to_Vec3fBatchLike(x, device, dtype)

    # Determine the batch size 'N' from the resulting tensor's shape.
    if tensor.ndim == 2:
        batch_size = tensor.shape[0]
    else:
        # If the tensor is rank 1 or 0, it represents a single item for the batch.
        batch_size = 1

    # Define the final, non-negotiable target shape.
    target_shape = (batch_size, 3)

    # Explicitly broadcast the tensor to the target shape.
    try:
        return tensor.broadcast_to(target_shape)
    except Exception as e:
        raise ValueError(f"Failed to broadcast {x} to Vec3fBatch: {e}")


def to_PositiveVec3i(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts to Vec3i with all values > 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Integer dtype. Defaults to torch.int64.

    Returns:
        Vec3i tensor with all values > 0.
    """
    tensor = to_Vec3i(x, device, dtype)
    if torch.any(tensor <= 0):
        raise ValueError(f"All values must be greater than zero, got {tensor}")
    return tensor


def to_PositiveVec3f(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts to Vec3f with all values > 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Float dtype. Defaults to torch.float32.

    Returns:
        Vec3f tensor with all values > 0.
    """
    tensor = to_Vec3f(x, device, dtype)
    if torch.any(tensor <= 0):
        raise ValueError(f"All values must be greater than zero, got {tensor}")
    return tensor


def to_PositiveVec3iBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts to Vec3iBatch with all values > 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Integer dtype. Defaults to torch.int64.

    Returns:
        Vec3iBatch tensor with all values > 0.
    """
    tensor = to_Vec3iBatch(x, device, dtype)
    if torch.any(tensor <= 0):
        raise ValueError(f"All values must be greater than zero, got {tensor}")
    return tensor


def to_PositiveVec3fBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts to Vec3fBatch with all values > 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Float dtype. Defaults to torch.float32.

    Returns:
        Vec3fBatch tensor with all values > 0.
    """
    tensor = to_Vec3fBatch(x, device, dtype)
    if torch.any(tensor <= 0):
        raise ValueError(f"All values must be greater than zero, got {tensor}")
    return tensor


def to_NonNegativeVec3i(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts to Vec3i with all values >= 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Integer dtype. Defaults to torch.int64.

    Returns:
        Vec3i tensor with all values >= 0.
    """
    tensor = to_Vec3i(x, device, dtype)
    if torch.any(tensor < 0):
        raise ValueError(f"All values must be non-negative, got {tensor}")
    return tensor


def to_NonNegativeVec3f(
    x: NumericMaxRank1,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts to Vec3f with all values >= 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Float dtype. Defaults to torch.float32.

    Returns:
        Vec3f tensor with all values >= 0.
    """
    tensor = to_Vec3f(x, device, dtype)
    if torch.any(tensor < 0):
        raise ValueError(f"All values must be non-negative, got {tensor}")
    return tensor


def to_NonNegativeVec3iBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    Converts to Vec3iBatch with all values >= 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Integer dtype. Defaults to torch.int64.

    Returns:
        Vec3iBatch tensor with all values >= 0.
    """
    tensor = to_Vec3iBatch(x, device, dtype)
    if torch.any(tensor < 0):
        raise ValueError(f"All values must be non-negative, got {tensor}")
    return tensor


def to_NonNegativeVec3fBatch(
    x: NumericMaxRank2,
    device: DeviceIdentifier | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts to Vec3fBatch with all values >= 0.

    Args:
        x: Input tensor.
        device: Target device. Defaults to input device or "cpu".
        dtype: Float dtype. Defaults to torch.float32.

    Returns:
        Vec3fBatch tensor with all values >= 0.
    """
    tensor = to_Vec3fBatch(x, device, dtype)
    if torch.any(tensor < 0):
        raise ValueError(f"All values must be non-negative, got {tensor}")
    return tensor
