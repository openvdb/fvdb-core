# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Type-safe torch-compatible functions that work with both Tensor and JaggedTensor.

This module provides properly typed wrappers around torch functions that delegate
to the underlying torch operations. These functions work seamlessly with both
regular torch.Tensor and fvdb.JaggedTensor, with proper type inference.

While JaggedTensor implements __torch_function__ to enable torch.fn(jt) syntax at runtime,
static type checkers don't understand this dispatch mechanism. These functions provide
the type-safe interface.

Usage:
    import fvdb

    # Works with JaggedTensor (typed as JaggedTensor)
    jt = fvdb.JaggedTensor([...])
    out = fvdb.relu(jt)  # Correctly typed as JaggedTensor

    # Also works with regular Tensor (typed as Tensor)
    t = torch.randn(10, 3)
    out = fvdb.add(t, 1.0)  # Correctly typed as Tensor

These functions are exposed in the main fvdb namespace and can be used as drop-in
replacements for torch functions with the benefit of proper type inference.
"""

from typing import overload

import torch
from torch import Tensor

from .jagged_tensor import JaggedTensor

# ============================================================================
# Unary pointwise operations
# ============================================================================


@overload
def relu(input: JaggedTensor) -> JaggedTensor: ...
@overload
def relu(input: Tensor) -> Tensor: ...


def relu(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Apply ReLU (Rectified Linear Unit) activation function element-wise.

    Computes ReLU(x) = max(0, x) for each element in the input tensor.

    Args:
        input (JaggedTensor | Tensor): Input tensor to apply ReLU to.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with ReLU applied element-wise.
            Returns the same type as the input.
    """
    return torch.relu(input)  # type: ignore


@overload
def relu_(input: JaggedTensor) -> JaggedTensor: ...
@overload
def relu_(input: Tensor) -> Tensor: ...


def relu_(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Apply ReLU (Rectified Linear Unit) activation function element-wise in-place.

    Computes ReLU(x) = max(0, x) for each element in the input tensor, modifying
    the input tensor directly.

    Args:
        input (JaggedTensor | Tensor): Input tensor to apply ReLU to in-place.
            Can be a regular torch.Tensor or a JaggedTensor. This tensor will be modified.

    Returns:
        JaggedTensor | Tensor: The modified input tensor with ReLU applied element-wise.
            Returns the same type as the input.
    """
    return torch.relu_(input)  # type: ignore


@overload
def sigmoid(input: JaggedTensor) -> JaggedTensor: ...
@overload
def sigmoid(input: Tensor) -> Tensor: ...


def sigmoid(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Apply sigmoid activation function element-wise.

    Computes sigmoid(x) = 1 / (1 + exp(-x)) for each element in the input tensor.

    Args:
        input (JaggedTensor | Tensor): Input tensor to apply sigmoid to.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with sigmoid applied element-wise.
            Returns the same type as the input. Values are in the range (0, 1).
    """
    return torch.sigmoid(input)  # type: ignore


@overload
def tanh(input: JaggedTensor) -> JaggedTensor: ...
@overload
def tanh(input: Tensor) -> Tensor: ...


def tanh(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Apply hyperbolic tangent (tanh) activation function element-wise.

    Computes tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) for each element.

    Args:
        input (JaggedTensor | Tensor): Input tensor to apply tanh to.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with tanh applied element-wise.
            Returns the same type as the input. Values are in the range (-1, 1).
    """
    return torch.tanh(input)  # type: ignore


@overload
def exp(input: JaggedTensor) -> JaggedTensor: ...
@overload
def exp(input: Tensor) -> Tensor: ...


def exp(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Compute the exponential of each element.

    Computes exp(x) = e^x for each element in the input tensor.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with exponential applied element-wise.
            Returns the same type as the input.
    """
    return torch.exp(input)  # type: ignore


@overload
def log(input: JaggedTensor) -> JaggedTensor: ...
@overload
def log(input: Tensor) -> Tensor: ...


def log(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Compute the natural logarithm of each element.

    Computes log(x) = ln(x) for each element in the input tensor.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with natural logarithm applied element-wise.
            Returns the same type as the input.
    """
    return torch.log(input)  # type: ignore


@overload
def sqrt(input: JaggedTensor) -> JaggedTensor: ...
@overload
def sqrt(input: Tensor) -> Tensor: ...


def sqrt(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Compute the square root of each element.

    Computes sqrt(x) = x^(1/2) for each element in the input tensor.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with square root applied element-wise.
            Returns the same type as the input.
    """
    return torch.sqrt(input)  # type: ignore


@overload
def floor(input: JaggedTensor) -> JaggedTensor: ...
@overload
def floor(input: Tensor) -> Tensor: ...


def floor(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Round each element down to the nearest integer.

    Computes floor(x), returning the largest integer less than or equal to x
    for each element.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with floor applied element-wise.
            Returns the same type as the input.
    """
    return torch.floor(input)  # type: ignore


@overload
def ceil(input: JaggedTensor) -> JaggedTensor: ...
@overload
def ceil(input: Tensor) -> Tensor: ...


def ceil(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Round each element up to the nearest integer.

    Computes ceil(x), returning the smallest integer greater than or equal to x
    for each element.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with ceiling applied element-wise.
            Returns the same type as the input.
    """
    return torch.ceil(input)  # type: ignore


@overload
def round(input: JaggedTensor) -> JaggedTensor: ...
@overload
def round(input: Tensor) -> Tensor: ...


def round(input: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Round each element to the nearest integer.

    Computes round(x), returning the nearest integer to x. For values exactly
    halfway between two integers, rounds to the nearest even integer.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Output with rounding applied element-wise.
            Returns the same type as the input.
    """
    return torch.round(input)  # type: ignore


@overload
def nan_to_num(
    input: JaggedTensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None
) -> JaggedTensor: ...
@overload
def nan_to_num(input: Tensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None) -> Tensor: ...


def nan_to_num(
    input: JaggedTensor | Tensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None
) -> JaggedTensor | Tensor:
    """
    Replace NaN, positive infinity, and negative infinity values with specified numbers.

    Replaces NaN values with the `nan` argument, positive infinity with `posinf`,
    and negative infinity with `neginf`.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        nan (float): The value to replace NaN with. Default is 0.0.
        posinf (float | None): The value to replace positive infinity with.
            If None, uses the maximum representable value for the dtype.
        neginf (float | None): The value to replace negative infinity with.
            If None, uses the minimum representable value for the dtype.

    Returns:
        JaggedTensor | Tensor: Output with NaN and infinity values replaced.
            Returns the same type as the input.
    """
    return torch.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)  # type: ignore


@overload
def clamp(input: JaggedTensor, min: float | None = None, max: float | None = None) -> JaggedTensor: ...
@overload
def clamp(input: Tensor, min: float | None = None, max: float | None = None) -> Tensor: ...


def clamp(input: JaggedTensor | Tensor, min: float | None = None, max: float | None = None) -> JaggedTensor | Tensor:
    """
    Clamp all elements to be within the range [min, max].

    Values smaller than min are set to min. Values larger than max are set to max.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        min (float | None): Lower bound. If None, no lower bound is applied.
        max (float | None): Upper bound. If None, no upper bound is applied.

    Returns:
        JaggedTensor | Tensor: Output with values clamped to [min, max].
            Returns the same type as the input.
    """
    return torch.clamp(input, min=min, max=max)  # type: ignore


# ============================================================================
# Binary/ternary elementwise operations
# ============================================================================


@overload
def add(
    input: JaggedTensor, other: JaggedTensor | Tensor | float | int, *, out: JaggedTensor | None = None
) -> JaggedTensor: ...
@overload
def add(input: Tensor, other: Tensor | float | int, *, out: Tensor | None = None) -> Tensor: ...


def add(
    input: JaggedTensor | Tensor,
    other: JaggedTensor | Tensor | float | int,
    *,
    out: JaggedTensor | Tensor | None = None,
) -> JaggedTensor | Tensor:
    """
    Add two tensors element-wise or add a scalar to each element.

    Computes input + other element-wise. Supports broadcasting when both inputs
    are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to add.
            Can be a tensor (with broadcasting) or a scalar.
        out (JaggedTensor | Tensor | None): Optional output tensor.
            If provided, the result is written to this tensor.

    Returns:
        JaggedTensor | Tensor: Sum of input and other.
            Returns the same type as the input tensor.
    """
    if out is not None:
        return torch.add(input, other, out=out)  # type: ignore
    return torch.add(input, other)  # type: ignore


@overload
def sub(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def sub(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def sub(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Subtract two tensors element-wise or subtract a scalar from each element.

    Computes input - other element-wise. Supports broadcasting when both inputs
    are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to subtract.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Difference of input and other.
            Returns the same type as the input tensor.
    """
    return torch.sub(input, other)  # type: ignore


@overload
def mul(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def mul(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def mul(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Multiply two tensors element-wise or multiply each element by a scalar.

    Computes input * other element-wise. Supports broadcasting when both inputs
    are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to multiply by.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Product of input and other.
            Returns the same type as the input tensor.
    """
    return torch.mul(input, other)  # type: ignore


@overload
def true_divide(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def true_divide(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def true_divide(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Divide two tensors element-wise or divide each element by a scalar.

    Computes input / other element-wise using true (floating-point) division.
    Always returns a floating-point result, even for integer inputs.
    Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor (numerator).
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Divisor.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Quotient of input and other.
            Returns the same type as the input tensor.
    """
    return torch.true_divide(input, other)  # type: ignore


@overload
def floor_divide(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def floor_divide(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def floor_divide(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Divide two tensors element-wise and round down to the nearest integer.

    Computes floor(input / other) element-wise. Supports broadcasting when both
    inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor (numerator).
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Divisor.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Floor division result.
            Returns the same type as the input tensor.
    """
    return torch.floor_divide(input, other)  # type: ignore


@overload
def remainder(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def remainder(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def remainder(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute the element-wise remainder of division.

    Computes the remainder of input / other element-wise. The result has the
    same sign as the divisor (other). Supports broadcasting when both inputs
    are tensors.

    Args:
        input (JaggedTensor | Tensor): Input tensor (numerator).
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Divisor.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Remainder of input divided by other.
            Returns the same type as the input tensor.
    """
    return torch.remainder(input, other)  # type: ignore


@overload
def pow(input: JaggedTensor, exponent: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def pow(input: Tensor, exponent: Tensor | float | int) -> Tensor: ...


def pow(input: JaggedTensor | Tensor, exponent: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Raise each element to the power of an exponent.

    Computes input^exponent element-wise. Supports broadcasting when both inputs
    are tensors.

    Args:
        input (JaggedTensor | Tensor): Base tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        exponent (JaggedTensor | Tensor | float | int): Exponent.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Result of raising input to the given exponent.
            Returns the same type as the input tensor.
    """
    return torch.pow(input, exponent)  # type: ignore


@overload
def maximum(input: JaggedTensor, other: JaggedTensor) -> JaggedTensor: ...
@overload
def maximum(input: Tensor, other: Tensor) -> Tensor: ...


def maximum(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Compute the element-wise maximum of two tensors.

    Compares corresponding elements and returns the larger value for each position.
    Supports broadcasting.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor): Second input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Element-wise maximum of input and other.
            Returns the same type as the input tensor.
    """
    return torch.maximum(input, other)  # type: ignore


@overload
def minimum(input: JaggedTensor, other: JaggedTensor) -> JaggedTensor: ...
@overload
def minimum(input: Tensor, other: Tensor) -> Tensor: ...


def minimum(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor) -> JaggedTensor | Tensor:
    """
    Compute the element-wise minimum of two tensors.

    Compares corresponding elements and returns the smaller value for each position.
    Supports broadcasting.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor): Second input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Element-wise minimum of input and other.
            Returns the same type as the input tensor.
    """
    return torch.minimum(input, other)  # type: ignore


# ============================================================================
# Comparison operations
# ============================================================================


@overload
def eq(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def eq(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def eq(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise equality comparison.

    Compares corresponding elements and returns True where input equals other,
    False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise equality results.
            Returns the same type as the input tensor.
    """
    return torch.eq(input, other)  # type: ignore


@overload
def ne(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def ne(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def ne(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise inequality comparison.

    Compares corresponding elements and returns True where input does not equal other,
    False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise inequality results.
            Returns the same type as the input tensor.
    """
    return torch.ne(input, other)  # type: ignore


@overload
def lt(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def lt(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def lt(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise less than comparison.

    Compares corresponding elements and returns True where input is less than other,
    False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise less than results.
            Returns the same type as the input tensor.
    """
    return torch.lt(input, other)  # type: ignore


@overload
def le(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def le(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def le(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise less than or equal comparison.

    Compares corresponding elements and returns True where input is less than or
    equal to other, False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise less than or equal results.
            Returns the same type as the input tensor.
    """
    return torch.le(input, other)  # type: ignore


@overload
def gt(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def gt(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def gt(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise greater than comparison.

    Compares corresponding elements and returns True where input is greater than other,
    False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise greater than results.
            Returns the same type as the input tensor.
    """
    return torch.gt(input, other)  # type: ignore


@overload
def ge(input: JaggedTensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor: ...
@overload
def ge(input: Tensor, other: Tensor | float | int) -> Tensor: ...


def ge(input: JaggedTensor | Tensor, other: JaggedTensor | Tensor | float | int) -> JaggedTensor | Tensor:
    """
    Compute element-wise greater than or equal comparison.

    Compares corresponding elements and returns True where input is greater than or
    equal to other, False otherwise. Supports broadcasting when both inputs are tensors.

    Args:
        input (JaggedTensor | Tensor): First input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor | float | int): Value to compare against.
            Can be a tensor (with broadcasting) or a scalar.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with element-wise greater than or equal results.
            Returns the same type as the input tensor.
    """
    return torch.ge(input, other)  # type: ignore


@overload
def where(condition: JaggedTensor, input: JaggedTensor, other: JaggedTensor) -> JaggedTensor: ...
@overload
def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor: ...


def where(
    condition: JaggedTensor | Tensor, input: JaggedTensor | Tensor, other: JaggedTensor | Tensor
) -> JaggedTensor | Tensor:
    """
    Select elements from input or other based on condition.

    Returns a tensor where each element is selected from input if the corresponding
    element in condition is True, otherwise from other. Supports broadcasting.

    Args:
        condition (JaggedTensor | Tensor): Boolean tensor determining which values to select.
            Can be a regular torch.Tensor or a JaggedTensor.
        input (JaggedTensor | Tensor): Values selected where condition is True.
            Can be a regular torch.Tensor or a JaggedTensor.
        other (JaggedTensor | Tensor): Values selected where condition is False.
            Can be a regular torch.Tensor or a JaggedTensor.

    Returns:
        JaggedTensor | Tensor: Tensor with elements selected based on condition.
            Returns the same type as the condition tensor.
    """
    return torch.where(condition, input, other)  # type: ignore


# ============================================================================
# Reductions (that preserve primary dimension)
# ============================================================================


@overload
def sum(
    input: JaggedTensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor: ...
@overload
def sum(
    input: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False, *, dtype: torch.dtype | None = None
) -> Tensor: ...


def sum(
    input: JaggedTensor | Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor | Tensor:
    """
    Compute the sum of elements.

    Returns the sum of all elements, or sums along specified dimensions.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.
        dtype (torch.dtype | None): Desired output dtype.
            If None, uses the dtype of the input.

    Returns:
        JaggedTensor | Tensor: Sum of elements.
            Returns the same type as the input tensor.
    """
    if dtype is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)  # type: ignore
    if dim is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.sum(input)  # type: ignore


@overload
def mean(
    input: JaggedTensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor: ...
@overload
def mean(
    input: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False, *, dtype: torch.dtype | None = None
) -> Tensor: ...


def mean(
    input: JaggedTensor | Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor | Tensor:
    """
    Compute the mean (average) of elements.

    Returns the mean of all elements, or means along specified dimensions.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.
        dtype (torch.dtype | None): Desired output dtype.
            If None, uses the dtype of the input.

    Returns:
        JaggedTensor | Tensor: Mean of elements.
            Returns the same type as the input tensor.
    """
    if dtype is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype)  # type: ignore
    if dim is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.mean(input)  # type: ignore


@overload
def amax(input: JaggedTensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def amax(input: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> Tensor: ...


def amax(
    input: JaggedTensor | Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
) -> JaggedTensor | Tensor:
    """
    Compute the maximum value of elements.

    Returns the maximum value of all elements, or maximum values along specified dimensions.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Maximum value(s).
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.amax(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.amax(input)  # type: ignore


@overload
def amin(input: JaggedTensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def amin(input: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> Tensor: ...


def amin(
    input: JaggedTensor | Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
) -> JaggedTensor | Tensor:
    """
    Compute the minimum value of elements.

    Returns the minimum value of all elements, or minimum values along specified dimensions.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Minimum value(s).
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.amin(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.amin(input)  # type: ignore


@overload
def argmax(input: JaggedTensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def argmax(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: ...


def argmax(input: JaggedTensor | Tensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor | Tensor:
    """
    Return the indices of the maximum value(s).

    Returns the index of the maximum value, or indices along a specified dimension.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | None): Dimension along which to find the maximum.
            If None, returns the index of the maximum value in the flattened tensor.
        keepdim (bool): Whether to keep the reduced dimension with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Indices of maximum value(s).
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.argmax(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.argmax(input)  # type: ignore


@overload
def argmin(input: JaggedTensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def argmin(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: ...


def argmin(input: JaggedTensor | Tensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor | Tensor:
    """
    Return the indices of the minimum value(s).

    Returns the index of the minimum value, or indices along a specified dimension.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | None): Dimension along which to find the minimum.
            If None, returns the index of the minimum value in the flattened tensor.
        keepdim (bool): Whether to keep the reduced dimension with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Indices of minimum value(s).
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.argmin(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.argmin(input)  # type: ignore


@overload
def all(input: JaggedTensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def all(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: ...


def all(input: JaggedTensor | Tensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor | Tensor:
    """
    Test if all elements evaluate to True.

    Returns True if all elements are True (or non-zero), either over the entire tensor
    or along a specified dimension.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor. Typically boolean,
            but numeric types are also supported (0 is False, non-zero is True).
        dim (int | None): Dimension along which to reduce.
            If None, reduces all dimensions.
        keepdim (bool): Whether to keep the reduced dimension with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with result of all() reduction.
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.all(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.all(input)  # type: ignore


@overload
def any(input: JaggedTensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor: ...
@overload
def any(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: ...


def any(input: JaggedTensor | Tensor, dim: int | None = None, keepdim: bool = False) -> JaggedTensor | Tensor:
    """
    Test if any element evaluates to True.

    Returns True if any element is True (or non-zero), either over the entire tensor
    or along a specified dimension.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor. Typically boolean,
            but numeric types are also supported (0 is False, non-zero is True).
        dim (int | None): Dimension along which to reduce.
            If None, reduces all dimensions.
        keepdim (bool): Whether to keep the reduced dimension with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Boolean tensor with result of any() reduction.
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.any(input, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.any(input)  # type: ignore


@overload
def norm(
    input: JaggedTensor,
    p: float | str = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor: ...
@overload
def norm(
    input: Tensor,
    p: float | str = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor: ...


def norm(
    input: JaggedTensor | Tensor,
    p: float | str = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> JaggedTensor | Tensor:
    """
    Compute the p-norm of elements.

    Returns the p-norm of all elements, or p-norms along specified dimensions.
    Common values include p=2 (Euclidean norm), p=1 (Manhattan norm), p='fro' (Frobenius norm).

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        p (float | str): The order of the norm. Can be a float or 'fro' (Frobenius),
            'nuc' (nuclear). Default is 2.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.
        dtype (torch.dtype | None): Desired output dtype.
            If None, uses the dtype of the input.

    Returns:
        JaggedTensor | Tensor: p-norm of elements.
            Returns the same type as the input tensor.
    """
    if dtype is not None:
        return torch.norm(input, p=p, dim=dim, keepdim=keepdim, dtype=dtype)  # type: ignore
    if dim is not None:
        return torch.norm(input, p=p, dim=dim, keepdim=keepdim)  # type: ignore
    return torch.norm(input, p=p)  # type: ignore


@overload
def var(
    input: JaggedTensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> JaggedTensor: ...
@overload
def var(
    input: Tensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> Tensor: ...


def var(
    input: JaggedTensor | Tensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> JaggedTensor | Tensor:
    """
    Compute the variance of elements.

    Returns the variance of all elements, or variances along specified dimensions.
    Uses Bessel's correction (dividing by N-1) when unbiased=True.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        unbiased (bool): Whether to use Bessel's correction (divide by N-1 instead of N).
            Default is True.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Variance of elements.
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.var(input, dim=dim, unbiased=unbiased, keepdim=keepdim)  # type: ignore
    return torch.var(input, unbiased=unbiased)  # type: ignore


@overload
def std(
    input: JaggedTensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> JaggedTensor: ...
@overload
def std(
    input: Tensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> Tensor: ...


def std(
    input: JaggedTensor | Tensor, dim: int | tuple[int, ...] | None = None, unbiased: bool = True, keepdim: bool = False
) -> JaggedTensor | Tensor:
    """
    Compute the standard deviation of elements.

    Returns the standard deviation of all elements, or standard deviations along specified
    dimensions. Uses Bessel's correction (dividing by N-1) when unbiased=True.

    Args:
        input (JaggedTensor | Tensor): Input tensor.
            Can be a regular torch.Tensor or a JaggedTensor.
        dim (int | tuple[int, ...] | None): Dimension(s) along which to reduce.
            If None, reduces all dimensions. Can be a single dimension or tuple of dimensions.
        unbiased (bool): Whether to use Bessel's correction (divide by N-1 instead of N).
            Default is True.
        keepdim (bool): Whether to keep the reduced dimension(s) with size 1.
            Default is False.

    Returns:
        JaggedTensor | Tensor: Standard deviation of elements.
            Returns the same type as the input tensor.
    """
    if dim is not None:
        return torch.std(input, dim=dim, unbiased=unbiased, keepdim=keepdim)  # type: ignore
    return torch.std(input, unbiased=unbiased)  # type: ignore
