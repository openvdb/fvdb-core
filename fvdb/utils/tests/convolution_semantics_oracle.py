# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Independent scalar and dense references for sparse-convolution semantics.

This module deliberately does not import fVDB grids, plans, or topology helpers.
It is test-only authority for the componentwise integer relation
``fine_ijk = stride * coarse_ijk + dilation * tap_ijk - padding_before`` at the canonical registration
``a = 0``, where ``padding_before = floor(dilation * (kernel_size - 1) / 2)`` and each tap coordinate is
zero-based: ``0 <= tap_ijk[axis] < kernel_size[axis]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterable, Sequence

import torch

Coord = tuple[int, int, int]
Kernel = tuple[int, int, int]

MAX_DENSE_ORACLE_SPATIAL_SITES = 2**20
MAX_DENSE_ORACLE_TENSOR_BYTES = 64 * 2**20


class DenseOraclePreflightError(ValueError):
    """Raised before a bounded dense oracle allocation would exceed its budget."""


@dataclass(frozen=True)
class ConvolutionRelation:
    """Scalar, Torch-phase geometry with no dependency on production geometry."""

    kernel_size: Kernel
    stride: Kernel
    dilation: Kernel = (1, 1, 1)

    def __post_init__(self) -> None:
        for name, values in (("kernel_size", self.kernel_size), ("stride", self.stride), ("dilation", self.dilation)):
            if len(values) != 3 or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain three positive integers, got {values}")

    @property
    def p_before(self) -> Kernel:
        return tuple((self.dilation[axis] * (self.kernel_size[axis] - 1)) // 2 for axis in range(3))  # type: ignore[return-value]

    @property
    def p_after(self) -> Kernel:
        return tuple(
            self.dilation[axis] * (self.kernel_size[axis] - 1) - self.p_before[axis] for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def r_min(self) -> Kernel:
        return tuple(-value for value in self.p_before)  # type: ignore[return-value]

    @property
    def r_max(self) -> Kernel:
        return tuple(
            self.dilation[axis] * (self.kernel_size[axis] - 1) - self.p_before[axis] for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def kernel_volume(self) -> int:
        return prod(self.kernel_size)

    def taps(self) -> Iterable[Coord]:
        return product(*(range(size) for size in self.kernel_size))

    def offset(self, tap: Coord) -> Coord:
        return tuple(self.dilation[axis] * tap[axis] - self.p_before[axis] for axis in range(3))  # type: ignore[return-value]

    def fine_from_coarse(self, coarse: Coord, tap: Coord) -> Coord:
        offset = self.offset(tap)
        return tuple(self.stride[axis] * coarse[axis] + offset[axis] for axis in range(3))  # type: ignore[return-value]

    def coarse_from_fine(self, fine: Coord, tap: Coord) -> Coord | None:
        offset = self.offset(tap)
        numerators = tuple(fine[axis] - offset[axis] for axis in range(3))
        if any(numerators[axis] % self.stride[axis] != 0 for axis in range(3)):
            return None
        return tuple(numerators[axis] // self.stride[axis] for axis in range(3))  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class RelationEdge:
    """One normalized fine/coarse/tap edge in the scalar reference."""

    fine: Coord
    coarse: Coord
    tap: Coord


@dataclass(frozen=True)
class DenseOracleResult:
    """Dense result plus global coordinate origin for its spatial tensor."""

    values: torch.Tensor
    origin: Coord

    def value_at(self, coordinate: Coord) -> torch.Tensor:
        local = tuple(coordinate[axis] - self.origin[axis] for axis in range(3))
        if any(local[axis] < 0 or local[axis] >= self.values.shape[axis + 2] for axis in range(3)):
            return torch.zeros(self.values.shape[1], dtype=self.values.dtype, device=self.values.device)
        return self.values[(0, slice(None), *local)]


def normalize_3d(value: int | Sequence[int]) -> Kernel:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"expected three values, got {value}")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def floor_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


def relation_edges(
    fine_coordinates: Iterable[Coord], relation: ConvolutionRelation, coarse_coordinates: Iterable[Coord] | None = None
) -> tuple[RelationEdge, ...]:
    """Enumerate the canonical finite relation, optionally restricted to coarse rows."""
    fine_set = {tuple(int(component) for component in coordinate) for coordinate in fine_coordinates}
    coarse_set = (
        None
        if coarse_coordinates is None
        else {tuple(int(component) for component in coordinate) for coordinate in coarse_coordinates}
    )
    edges = set()
    for fine in fine_set:
        for tap in relation.taps():
            coarse = relation.coarse_from_fine(fine, tap)
            if coarse is not None and (coarse_set is None or coarse in coarse_set):
                edges.add(RelationEdge(fine=fine, coarse=coarse, tap=tap))
    return tuple(sorted(edges))


def forward_degrees(fine_coordinates: Iterable[Coord], relation: ConvolutionRelation) -> dict[Coord, int]:
    degrees: dict[Coord, int] = {}
    for edge in relation_edges(fine_coordinates, relation):
        degrees[edge.coarse] = degrees.get(edge.coarse, 0) + 1
    return degrees


def forward_support(fine_coordinates: Iterable[Coord], relation: ConvolutionRelation) -> set[Coord]:
    return set(forward_degrees(fine_coordinates, relation))


def transpose_degrees(coarse_coordinates: Iterable[Coord], relation: ConvolutionRelation) -> dict[Coord, int]:
    degrees: dict[Coord, int] = {}
    for coarse in {tuple(int(component) for component in coordinate) for coordinate in coarse_coordinates}:
        for tap in relation.taps():
            fine = relation.fine_from_coarse(coarse, tap)
            degrees[fine] = degrees.get(fine, 0) + 1
    return degrees


def transpose_support(coarse_coordinates: Iterable[Coord], relation: ConvolutionRelation) -> set[Coord]:
    return set(transpose_degrees(coarse_coordinates, relation))


def _bounds(coordinates: Iterable[Coord]) -> tuple[Coord, Coord]:
    materialized = list(coordinates)
    if not materialized:
        raise ValueError("dense oracle requires a nonempty coordinate set")
    return (
        tuple(min(coordinate[axis] for coordinate in materialized) for axis in range(3)),
        tuple(max(coordinate[axis] for coordinate in materialized) for axis in range(3)),
    )  # type: ignore[return-value]


def _preflight(dtype: torch.dtype, shapes: Iterable[Sequence[int]]) -> None:
    element_size = torch.empty((), dtype=dtype).element_size()
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes)
    spatial_sites = [prod(shape[-3:]) for shape in shapes if len(shape) >= 3]
    total_bytes = sum(prod(shape) * element_size for shape in shapes)
    if (
        any(sites > MAX_DENSE_ORACLE_SPATIAL_SITES for sites in spatial_sites)
        or total_bytes > MAX_DENSE_ORACLE_TENSOR_BYTES
    ):
        raise DenseOraclePreflightError(
            f"dense convolution oracle preflight rejected shapes={shapes}, spatial_sites={spatial_sites}, "
            f"estimated_bytes={total_bytes}, limits=({MAX_DENSE_ORACLE_SPATIAL_SITES}, {MAX_DENSE_ORACLE_TENSOR_BYTES})"
        )


def dense_forward_oracle(
    fine_coordinates: Sequence[Coord], features: torch.Tensor, weights: torch.Tensor, relation: ConvolutionRelation
) -> DenseOracleResult:
    """Run complete-topology Torch cross-correlation on a global-coordinate canvas."""
    if features.ndim != 2 or weights.ndim != 5 or len(fine_coordinates) != features.shape[0]:
        raise ValueError("expected coordinates [N,3], features [N,C], and fVDB weights [Cout,Cin,K0,K1,K2]")
    if tuple(weights.shape[2:]) != relation.kernel_size or weights.shape[1] != features.shape[1]:
        raise ValueError("feature/weight shape does not match relation")
    if weights.dtype != features.dtype or weights.device != features.device:
        raise ValueError("features and weights must share dtype and device")
    fine_min, fine_max = _bounds(fine_coordinates)
    coarse_min = tuple(floor_div(fine_min[axis] - relation.r_max[axis], relation.stride[axis]) for axis in range(3))
    coarse_max = tuple(ceil_div(fine_max[axis] - relation.r_min[axis], relation.stride[axis]) for axis in range(3))
    input_min = relation.fine_from_coarse(coarse_min, (0, 0, 0))
    input_max = relation.fine_from_coarse(coarse_max, tuple(size - 1 for size in relation.kernel_size))
    input_shape = tuple(input_max[axis] - input_min[axis] + 1 for axis in range(3))
    output_shape = tuple(coarse_max[axis] - coarse_min[axis] + 1 for axis in range(3))
    payload_shapes = [
        tuple(features.shape),
        (1, features.shape[1], *input_shape),
        (1, weights.shape[0], *output_shape),
        tuple(weights.shape),
    ]
    if features.requires_grad or weights.requires_grad:
        payload_shapes.extend((tuple(features.shape), tuple(weights.shape), (1, weights.shape[0], *output_shape)))
    _preflight(
        features.dtype,
        payload_shapes,
    )
    dense_input = torch.zeros((1, features.shape[1], *input_shape), dtype=features.dtype, device=features.device)
    for index, coordinate in enumerate(fine_coordinates):
        local = tuple(coordinate[axis] - input_min[axis] for axis in range(3))
        dense_input[(0, slice(None), *local)] = features[index]
    values = torch.nn.functional.conv3d(
        dense_input, weights, stride=relation.stride, padding=0, dilation=relation.dilation
    )
    return DenseOracleResult(values=values, origin=coarse_min)  # type: ignore[arg-type]


def dense_transpose_oracle(
    coarse_coordinates: Sequence[Coord], features: torch.Tensor, weights: torch.Tensor, relation: ConvolutionRelation
) -> DenseOracleResult:
    """Run complete-topology Torch transposed convolution on a global-coordinate canvas."""
    if features.ndim != 2 or weights.ndim != 5 or len(coarse_coordinates) != features.shape[0]:
        raise ValueError("expected coordinates [N,3], features [N,C], and fVDB weights [Cout,Cin,K0,K1,K2]")
    if tuple(weights.shape[2:]) != relation.kernel_size or weights.shape[1] != features.shape[1]:
        raise ValueError("feature/weight shape does not match relation")
    if weights.dtype != features.dtype or weights.device != features.device:
        raise ValueError("features and weights must share dtype and device")
    coarse_min, coarse_max = _bounds(coarse_coordinates)
    input_shape = tuple(coarse_max[axis] - coarse_min[axis] + 1 for axis in range(3))
    output_shape = tuple(
        (input_shape[axis] - 1) * relation.stride[axis] + relation.dilation[axis] * (relation.kernel_size[axis] - 1) + 1
        for axis in range(3)
    )
    payload_shapes = [
        tuple(features.shape),
        (1, features.shape[1], *input_shape),
        (1, weights.shape[0], *output_shape),
        tuple(weights.shape),
    ]
    if features.requires_grad or weights.requires_grad:
        payload_shapes.extend((tuple(features.shape), tuple(weights.shape), (1, weights.shape[0], *output_shape)))
    _preflight(
        features.dtype,
        payload_shapes,
    )
    dense_input = torch.zeros((1, features.shape[1], *input_shape), dtype=features.dtype, device=features.device)
    for index, coordinate in enumerate(coarse_coordinates):
        local = tuple(coordinate[axis] - coarse_min[axis] for axis in range(3))
        dense_input[(0, slice(None), *local)] = features[index]
    values = torch.nn.functional.conv_transpose3d(
        dense_input, weights.transpose(0, 1).contiguous(), stride=relation.stride, padding=0, dilation=relation.dilation
    )
    origin = tuple(relation.stride[axis] * coarse_min[axis] + relation.r_min[axis] for axis in range(3))
    return DenseOracleResult(values=values, origin=origin)  # type: ignore[arg-type]
