# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Black-box encapsulation of configuration structures for sparse convolution using
fVDB GridBatch. Design is intended to be reminiscent of the "plan" concept from FFT
libraries. Like FFT plans, the convolution plan encapsulates a single direction - regular
convolution, or transposed convolution, but can represent either.
"""

import warnings
from dataclasses import dataclass
from threading import Lock
from typing import Any, overload

import torch
from fvdb.types import NumericMaxRank1, ValueConstraint, to_Vec3i

from fvdb import GridBatch, JaggedTensor
from fvdb.functional._dispatch import _get_grid_data

from . import _fvdb_cpp
from .enums import ConvolutionPhasePolicy, ConvolutionTopologyPolicy, ConvolutionTopologyProvenance

_DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "default",
}

_ANY_CHANNEL_PAIRS: tuple[tuple[int, int], ...] = ()

_TRANSFORM_COMPATIBILITY_ATOL = 1.0e-6
_TRANSFORM_COMPATIBILITY_RTOL = 1.0e-6
_WARNED_INCOMPLETE_COVERAGE_GEOMETRIES: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
_DENSE_BACKEND_DISABLED_MESSAGE = (
    "The dense convolution backend is disabled because it does not yet implement the canonical sparse geometry "
    "and public transposed-weight layout; use backend='default' or backend='gather_scatter'."
)


class ConvolutionCoverageWarning(UserWarning):
    """A convolution geometry leaves some stride residues structurally uncovered.

    The warning is emitted once per ``(kernel_size, stride)`` geometry for the
    lifetime of the process. Pass ``acknowledge_incomplete_coverage=True`` when
    constructing a plan to acknowledge and suppress it explicitly.
    """


@dataclass(frozen=True)
class ConvolutionCoverageReport:
    """Exact rulebook degree diagnostics for a finite convolution plan."""

    input_row_count: int
    output_row_count: int
    input_zero_count: int
    input_zero_fraction: float
    input_degree_min: int
    input_degree_max: int
    input_degree_histogram: tuple[tuple[int, int], ...]
    output_zero_count: int
    output_zero_fraction: float
    output_degree_min: int
    output_degree_max: int
    output_degree_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConvolutionTransformCompatibility:
    """Compatibility of a plan's normalized fine and coarse lattices.

    Compatibility requires matching batch/device metadata,
    ``h_coarse == stride * h_fine``, and the canonical uniform registration
    ``a == 0``. Comparisons use ``atol=rtol=1e-6``.
    """

    fine_grid_count: int
    coarse_grid_count: int
    same_batch_size: bool
    same_device: bool
    scale_compatible: bool
    registration_integer: bool
    registration_zero: bool
    compatible: bool
    registration_offset: torch.Tensor | None


def _transform_compatibility(
    fine_grid: GridBatch, coarse_grid: GridBatch, geometry: _fvdb_cpp.ConvolutionGeometry
) -> ConvolutionTransformCompatibility:
    """Compute the canonical lattice-registration diagnostic."""
    same_batch_size = fine_grid.grid_count == coarse_grid.grid_count
    same_device = fine_grid.device == coarse_grid.device
    if not same_batch_size or not same_device:
        return ConvolutionTransformCompatibility(
            fine_grid_count=fine_grid.grid_count,
            coarse_grid_count=coarse_grid.grid_count,
            same_batch_size=same_batch_size,
            same_device=same_device,
            scale_compatible=False,
            registration_integer=False,
            registration_zero=False,
            compatible=False,
            registration_offset=None,
        )

    # GridBatchData exposes the authoritative double-precision metadata on the
    # host. Keep this construction-time diagnostic off the execution device and
    # avoid losing registration precision through GridBatch's float32 view.
    fine_data = _get_grid_data(fine_grid)
    coarse_data = _get_grid_data(coarse_grid)
    fine_voxel_sizes = fine_data.voxel_sizes
    coarse_voxel_sizes = coarse_data.voxel_sizes
    fine_origins = fine_data.origins
    coarse_origins = coarse_data.origins
    expected_coarse_voxel_sizes = fine_voxel_sizes * torch.tensor(
        geometry.stride, dtype=fine_voxel_sizes.dtype, device=fine_voxel_sizes.device
    )
    scale_compatible = bool(
        torch.allclose(
            coarse_voxel_sizes,
            expected_coarse_voxel_sizes,
            atol=_TRANSFORM_COMPATIBILITY_ATOL,
            rtol=_TRANSFORM_COMPATIBILITY_RTOL,
        )
    )
    registration_offset = (coarse_origins - fine_origins) / fine_voxel_sizes
    rounded_registration = torch.round(registration_offset)
    registration_integer = bool(
        torch.allclose(
            registration_offset,
            rounded_registration,
            atol=_TRANSFORM_COMPATIBILITY_ATOL,
            rtol=_TRANSFORM_COMPATIBILITY_RTOL,
        )
    )
    registration_zero = bool(
        torch.allclose(
            registration_offset,
            torch.zeros_like(registration_offset),
            atol=_TRANSFORM_COMPATIBILITY_ATOL,
            rtol=_TRANSFORM_COMPATIBILITY_RTOL,
        )
    )
    return ConvolutionTransformCompatibility(
        fine_grid_count=fine_grid.grid_count,
        coarse_grid_count=coarse_grid.grid_count,
        same_batch_size=True,
        same_device=True,
        scale_compatible=scale_compatible,
        registration_integer=registration_integer,
        registration_zero=registration_zero,
        compatible=scale_compatible and registration_integer and registration_zero,
        registration_offset=registration_offset,
    )


def _validate_transform_compatibility(compatibility: ConvolutionTransformCompatibility) -> None:
    migration_hint = (
        "Rebuild the explicit target with stride-scaled voxel sizes and identical origins. "
        "GridBatch.coarsened_grid uses a different block-centroid transform contract."
    )
    if not compatibility.same_batch_size:
        raise ValueError(
            "Convolution fine and coarse grids must have the same batch size; "
            f"got {compatibility.fine_grid_count} and {compatibility.coarse_grid_count}. {migration_hint}"
        )
    if not compatibility.same_device:
        raise ValueError(f"Convolution fine and coarse grids must be on the same device. {migration_hint}")
    if not compatibility.scale_compatible:
        raise ValueError(
            "Convolution voxel size mismatch: expected h_coarse = stride * h_fine in every batch and axis. "
            f"{migration_hint}"
        )
    registration_offset = compatibility.registration_offset
    if registration_offset is None or not bool(torch.isfinite(registration_offset).all().item()):
        raise ValueError(f"Convolution registration offset must be finite. {migration_hint}")
    if not compatibility.registration_integer:
        raise ValueError(
            "Convolution grids have a fractional lattice registration offset. "
            f"Convolution currently supports only a=0. {migration_hint}"
        )
    if not compatibility.registration_zero:
        raise ValueError(
            "Convolution grids have a nonzero integer lattice registration offset. "
            f"Convolution currently supports only a=0. {migration_hint}"
        )
    if not compatibility.compatible:
        raise RuntimeError("Convolution transform compatibility validation reached an inconsistent state")


def _resolve_topology_policy(
    target_grid: GridBatch | None, topology_policy: ConvolutionTopologyPolicy | None
) -> ConvolutionTopologyPolicy:
    if topology_policy is None:
        return ConvolutionTopologyPolicy.COMPLETE if target_grid is None else ConvolutionTopologyPolicy.RESTRICTED
    if not isinstance(topology_policy, ConvolutionTopologyPolicy):
        raise TypeError("topology_policy must be a ConvolutionTopologyPolicy value")
    if topology_policy is ConvolutionTopologyPolicy.COMPLETE and target_grid is not None:
        raise ValueError("topology_policy=ConvolutionTopologyPolicy.COMPLETE requires target_grid=None")
    if topology_policy is ConvolutionTopologyPolicy.RESTRICTED and target_grid is None:
        raise ValueError("topology_policy=ConvolutionTopologyPolicy.RESTRICTED requires an explicit target_grid")
    return topology_policy


def _warn_if_incomplete_residue_coverage(
    geometry: _fvdb_cpp.ConvolutionGeometry, acknowledge_incomplete_coverage: bool
) -> None:
    if acknowledge_incomplete_coverage:
        return
    kernel_size = tuple(geometry.kernel_size)
    stride = tuple(geometry.stride)
    uncovered_axes = []
    for axis, (kernel, step, padding_before) in enumerate(
        zip(kernel_size, stride, geometry.padding_before, strict=True)
    ):
        residues = {(tap - padding_before) % step for tap in range(kernel)}
        if len(residues) != step:
            uncovered_axes.append(axis)
    if not uncovered_axes:
        return
    geometry_key = (kernel_size, stride)
    if geometry_key in _WARNED_INCOMPLETE_COVERAGE_GEOMETRIES:
        return
    _WARNED_INCOMPLETE_COVERAGE_GEOMETRIES.add(geometry_key)
    warnings.warn(
        "This convolution geometry leaves uncovered stride residues on axes "
        f"{uncovered_axes}; some active fine coordinates can have zero rulebook degree. "
        "This matches dense Torch sampling. Pass acknowledge_incomplete_coverage=True to suppress this warning.",
        ConvolutionCoverageWarning,
        stacklevel=3,
    )


def _degree_summary(degrees: torch.Tensor) -> tuple[int, float, int, int, tuple[tuple[int, int], ...]]:
    row_count = int(degrees.numel())
    if row_count == 0:
        return 0, 0.0, 0, 0, ()
    degrees = degrees.cpu()
    zero_count = int((degrees == 0).sum().item())
    unique_degrees, counts = torch.unique(degrees, sorted=True, return_counts=True)
    histogram = tuple(
        (int(degree.item()), int(count.item())) for degree, count in zip(unique_degrees, counts, strict=True)
    )
    return (
        zero_count,
        zero_count / row_count,
        int(degrees.min().item()),
        int(degrees.max().item()),
        histogram,
    )


def _coverage_report(
    backend: "_Backend", source_grid: GridBatch, target_grid: GridBatch
) -> ConvolutionCoverageReport | None:
    if isinstance(backend, _MatmulBackend):
        input_row_count = source_grid.total_voxels
        output_row_count = target_grid.total_voxels
        return ConvolutionCoverageReport(
            input_row_count=input_row_count,
            output_row_count=output_row_count,
            input_zero_count=0,
            input_zero_fraction=0.0,
            input_degree_min=1 if input_row_count else 0,
            input_degree_max=1 if input_row_count else 0,
            input_degree_histogram=((1, input_row_count),) if input_row_count else (),
            output_zero_count=0,
            output_zero_fraction=0.0,
            output_degree_min=1 if output_row_count else 0,
            output_degree_max=1 if output_row_count else 0,
            output_degree_histogram=((1, output_row_count),) if output_row_count else (),
        )
    elif isinstance(backend, _GatherScatterBackend):
        input_degrees = torch.bincount(
            backend.topology.gather_indices,
            minlength=backend.topology.feature_total_voxels,
        )
        output_degrees = torch.bincount(
            backend.topology.scatter_indices,
            minlength=backend.topology.output_total_voxels,
        )
    elif isinstance(backend, _PredGatherIGemmBackend):
        input_degrees = torch.bincount(
            backend.gs_topology.gather_indices,
            minlength=backend.gs_topology.feature_total_voxels,
        )
        output_degrees = torch.bincount(
            backend.gs_topology.scatter_indices,
            minlength=backend.gs_topology.output_total_voxels,
        )
    else:
        return None

    input_summary = _degree_summary(input_degrees)
    output_summary = _degree_summary(output_degrees)
    return ConvolutionCoverageReport(
        input_row_count=int(input_degrees.numel()),
        output_row_count=int(output_degrees.numel()),
        input_zero_count=input_summary[0],
        input_zero_fraction=input_summary[1],
        input_degree_min=input_summary[2],
        input_degree_max=input_summary[3],
        input_degree_histogram=input_summary[4],
        output_zero_count=output_summary[0],
        output_zero_fraction=output_summary[1],
        output_degree_min=output_summary[2],
        output_degree_max=output_summary[3],
        output_degree_histogram=output_summary[4],
    )


def _swap_coverage_report(report: ConvolutionCoverageReport) -> ConvolutionCoverageReport:
    """Reverse input/output degree diagnostics without revisiting the rulebook."""
    return ConvolutionCoverageReport(
        input_row_count=report.output_row_count,
        output_row_count=report.input_row_count,
        input_zero_count=report.output_zero_count,
        input_zero_fraction=report.output_zero_fraction,
        input_degree_min=report.output_degree_min,
        input_degree_max=report.output_degree_max,
        input_degree_histogram=report.output_degree_histogram,
        output_zero_count=report.input_zero_count,
        output_zero_fraction=report.input_zero_fraction,
        output_degree_min=report.input_degree_min,
        output_degree_max=report.input_degree_max,
        output_degree_histogram=report.input_degree_histogram,
    )


def _output_zero_count(backend: "_Backend") -> int | None:
    """Return the output zero-degree count without constructing full diagnostics."""
    if isinstance(backend, _MatmulBackend):
        return 0
    if isinstance(backend, _GatherScatterBackend):
        topology = backend.topology
    elif isinstance(backend, _PredGatherIGemmBackend):
        topology = backend.gs_topology
    else:
        return None

    output_degrees = torch.bincount(
        topology.scatter_indices,
        minlength=topology.output_total_voxels,
    )
    return int((output_degrees == 0).sum().item())


def _validate_coverage_policy(
    backend: "_Backend", topology_policy: ConvolutionTopologyPolicy, strict_output_coverage: bool
) -> None:
    if topology_policy is not ConvolutionTopologyPolicy.COMPLETE and not strict_output_coverage:
        return
    output_zero_count = _output_zero_count(backend)
    if output_zero_count is None:
        return
    if topology_policy is ConvolutionTopologyPolicy.COMPLETE and output_zero_count:
        raise RuntimeError("Generated complete topology contains " f"{output_zero_count} zero-degree output rows.")
    if topology_policy is ConvolutionTopologyPolicy.RESTRICTED and strict_output_coverage and output_zero_count:
        raise ValueError("Restricted topology contains " f"{output_zero_count} zero-degree output rows.")


def _vec_is_all(v: torch.Tensor, i: int | float) -> bool:
    return bool(torch.all(torch.eq(v, i)).item())


def _channel_pair_supported(in_channels: int, out_channels: int, channel_pairs: tuple[tuple[int, int], ...]) -> bool:
    if len(channel_pairs) == 0:
        return True
    return (in_channels, out_channels) in channel_pairs


def _pred_gather_igemm_channel_pair_supported(in_channels: int, out_channels: int) -> bool:
    return in_channels > 0 and out_channels > 0 and in_channels % 32 == 0 and out_channels % 32 == 0


def _validate_pred_gather_igemm_admission(
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    channel_pairs: tuple[tuple[int, int], ...],
    *,
    transposed: bool,
) -> tuple[int, int]:
    """Pin the legacy PredGatherIGemm phase-safe admission boundary."""
    if transposed:
        raise ValueError("PredGatherIGemm backend does not support transposed convolution.")

    kernel_values = [int(value) for value in kernel_size.tolist()]
    if len(set(kernel_values)) != 1 or kernel_values[0] not in (3, 5, 7):
        raise ValueError(f"PredGatherIGemm supports only uniform kernel sizes 3, 5, 7; got {kernel_values}.")

    stride_values = [int(value) for value in stride.tolist()]
    if len(set(stride_values)) != 1 or stride_values[0] not in (1, 2):
        raise ValueError(f"PredGatherIGemm supports only uniform strides 1, 2; got {stride_values}.")

    for channel_pair in channel_pairs:
        if len(channel_pair) != 2 or channel_pair[0] <= 0 or channel_pair[1] <= 0:
            raise ValueError("channel_pair must be a tuple of two positive integers")
        in_channels, out_channels = channel_pair
        if not _pred_gather_igemm_channel_pair_supported(in_channels, out_channels):
            raise ValueError(
                f"PredGatherIGemm requires channel counts divisible by 32; got ({in_channels}, {out_channels})."
            )

    return kernel_values[0], stride_values[0]


def _validate_pred_gather_igemm_grid_admission(source_grid: GridBatch, target_grid: GridBatch | None = None) -> None:
    """Reject grid configurations that the native PredGatherIGemm kernel cannot execute."""
    if source_grid.device.type != "cuda":
        raise ValueError("PredGatherIGemm requires source and target grids on CUDA.")
    if source_grid.grid_count != 1:
        raise ValueError(f"PredGatherIGemm supports only batch size 1; got {source_grid.grid_count} source grids.")
    if target_grid is None:
        return
    if target_grid.device.type != "cuda":
        raise ValueError("PredGatherIGemm requires source and target grids on CUDA.")
    if target_grid.grid_count != 1:
        raise ValueError(f"PredGatherIGemm supports only batch size 1; got {target_grid.grid_count} target grids.")


def _matmul_weight_matrix(weights: torch.Tensor) -> torch.Tensor:
    """Normalize supported identity-convolution weights to ``[C_out, C_in]``."""
    if weights.ndim == 2:
        return weights
    if weights.ndim == 5 and tuple(weights.shape[2:]) == (1, 1, 1):
        return weights[:, :, 0, 0, 0]
    raise ValueError("The K=1, S=1 matmul backend requires weights shaped [C_out, C_in] or [C_out, C_in, 1, 1, 1].")


# ============================================================
#  Autograd functions for gather-scatter convolution
# ============================================================


class _GatherScatterConvFn(torch.autograd.Function):
    """Autograd wrapper for the default gather-scatter convolution (forward + transposed)."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, weights: torch.Tensor, topo: _fvdb_cpp.GatherScatterDefaultTopology, transposed: bool) -> torch.Tensor:  # type: ignore[override]
        if transposed:
            output = _fvdb_cpp.gs_conv_transpose(features, weights, topo)
        else:
            output = _fvdb_cpp.gs_conv(features, weights, topo)
        ctx.save_for_backward(features, weights)
        ctx.topo = topo
        ctx.transposed = transposed
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:  # type: ignore[override]
        features, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.transposed:
            grad_feat, grad_w = _fvdb_cpp.gs_conv_transpose_backward(grad_output, features, weights, ctx.topo)
        else:
            grad_feat, grad_w = _fvdb_cpp.gs_conv_backward(grad_output, features, weights, ctx.topo)
        return grad_feat, grad_w, None, None


class _PredGatherIGemmConvFn(torch.autograd.Function):
    """Autograd wrapper for the PredGatherIGemm CUTLASS IGEMM convolution.

    Forward uses the IGEMM kernel; backward falls back to GatherScatterDefault.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        features: torch.Tensor,
        weights: torch.Tensor,
        feature_grid,
        output_grid,
        gs_topo: _fvdb_cpp.GatherScatterDefaultTopology,
        kernel_size: int,
        stride: int,
    ) -> torch.Tensor:
        output = _fvdb_cpp.pred_gather_igemm_conv(features, weights, feature_grid, output_grid, kernel_size, stride)
        ctx.save_for_backward(features, weights)
        ctx.gs_topo = gs_topo
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:  # type: ignore[override]
        features, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_feat, grad_w = _fvdb_cpp.gs_conv_backward(grad_output, features, weights, ctx.gs_topo)
        return grad_feat, grad_w, None, None, None, None, None


# ============================================================
#  Backend data classes — cached precomputed data per method
# ============================================================


@dataclass(frozen=True)
class _MatmulBackend:
    """1x1x1 convolution with stride 1 — pure matmul, no precomputed data."""

    pass


@dataclass(frozen=True)
class _GatherScatterBackend:
    """Default gather-scatter convolution with precomputed compacted topology (Python autograd)."""

    topology: _fvdb_cpp.GatherScatterDefaultTopology


@dataclass(frozen=True)
class _PredGatherIGemmBackend:
    """CUTLASS IGEMM convolution with predicated gather (SM80+, forward only).

    The GatherScatterDefault topology is precomputed for backward pass fallback.
    """

    gs_topology: _fvdb_cpp.GatherScatterDefaultTopology
    kernel_size: int
    stride: int


_Backend = _MatmulBackend | _GatherScatterBackend | _PredGatherIGemmBackend


class _CoverageReportCache:
    """Thread-safe lazy coverage diagnostics shared by exact transposes."""

    def __init__(self, backend: _Backend, source_grid: GridBatch, target_grid: GridBatch):
        self._backend = backend
        self._source_grid = source_grid
        self._target_grid = target_grid
        self._report: ConvolutionCoverageReport | None = None
        self._swapped_report: ConvolutionCoverageReport | None = None
        self._lock = Lock()

    def get(self, swapped: bool) -> ConvolutionCoverageReport | None:
        report = self._report
        if report is None:
            with self._lock:
                report = self._report
                if report is None:
                    report = _coverage_report(self._backend, self._source_grid, self._target_grid)
                    self._report = report

        if not swapped or report is None:
            return report

        swapped_report = self._swapped_report
        if swapped_report is None:
            with self._lock:
                swapped_report = self._swapped_report
                if swapped_report is None:
                    swapped_report = _swap_coverage_report(report)
                    self._swapped_report = swapped_report
        return swapped_report


@dataclass(frozen=True)
class ConvolutionPlan:
    """
    A pre-configured plan for efficient sparse 3D convolution operations on :class:`fvdb.GridBatch`.

    :class:`ConvolutionPlan` encapsulates all the configuration and optimization structures needed
    to perform sparse convolution operations efficiently. Like `FFT plans in signal processing libraries <https://www.fftw.org/fftw3_doc/Using-Plans.html>`_,
    a :class:`ConvolutionPlan` represents a single direction of computation - either
    regular convolution or transposed convolution.

    The plan handles the complex sparse data structures and backend optimizations internally,
    allowing users to focus on the core convolution parameters: input/output channels,
    kernel size, stride, and the grid structure.

    A plan stores one finite convolution relation. Componentwise, that relation is
    ``fine_ijk = stride * coarse_ijk + tap_ijk - padding_before``, where
    ``padding_before = floor((kernel_size - 1) / 2)`` and each zero-based tap satisfies
    ``0 <= tap_ijk[axis] < kernel_size[axis]``. Generated plans use complete structural
    support; an explicit target restricts that relation. A transposed plan evaluates the
    same fine/coarse connectivity in the opposite direction and is not a value inverse.
    For an exact finite adjoint, use :meth:`from_plan_transposed` rather than reconstructing
    a plan from grids.

    For framework portability, this index relation matches spconv and TorchSparse when
    they use ``padding=(kernel_size - 1) // 2``. MinkowskiEngine corner-anchors even
    kernels on ``[0, kernel_size)``, so porting an even-kernel topology from
    MinkowskiEngine shifts it by ``floor((kernel_size - 1) / 2)``. fVDB's complete policy
    also materializes uncropped boundary support: a full ``16^3`` input with
    ``kernel_size=stride=4`` produces a ``5^3`` coarse topology rather than ``4^3``.

    Usage Pattern:

    1. Create a plan using one of the ``from_*`` class methods (see :meth:`from_grid_batch()`).
    2. Use the :meth:`execute()` method to perform convolutions with different weights and data on
       the same grid structures.
    3. Reuse the same plan for multiple convolutions with the same configuration

    Example Usage:

    .. code-block:: python

        from fvdb import GridBatch, ConvolutionPlan

        # Create a grid batch
        my_grid_batch = GridBatch.from_ijk(...)

        # Create a plan for 3x3x3 convolution with stride 1
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=my_grid_batch
        )

        # execute convolution with different weights
        features = torch.randn(num_voxels, 32, device="cuda")
        weights = torch.randn(64, 32, 3, 3, 3, device="cuda")
        output = plan.execute(features, weights)

    .. note::

        - Always create plans using the ``from_*`` class methods, never call ``__init__`` directly
        - Plans are immutable once created
        - The same plan can be reused for multiple :meth:`execute()` calls with different data/weights
        - Channel pairs can be specified at plan creation time for optimal backend selection
    """

    _source_grid: GridBatch
    _target_grid: GridBatch
    _geometry: _fvdb_cpp.ConvolutionGeometry
    _channel_pairs: tuple[tuple[int, int], ...]
    _transposed: bool
    _backend: _Backend
    _transform_compatibility: ConvolutionTransformCompatibility
    _topology_policy: ConvolutionTopologyPolicy
    _topology_provenance: ConvolutionTopologyProvenance
    _coverage_report_cache: _CoverageReportCache
    _coverage_report_swapped: bool

    # ============================================================
    #                 Factory methods
    # ============================================================

    @classmethod
    def from_grid_batch(
        cls,
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: GridBatch,
        target_grid: GridBatch | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
        channel_pairs: tuple[tuple[int, int], ...] = _ANY_CHANNEL_PAIRS,
        topology_policy: ConvolutionTopologyPolicy | None = None,
        strict_output_coverage: bool = False,
        acknowledge_incomplete_coverage: bool = False,
    ) -> "ConvolutionPlan":
        """
        Create a :class:`ConvolutionPlan` for convolution on batches of grids. *i.e.* convolution where the input
        and output domains are both of type :class:`fvdb.GridBatch`.

        The plan returned by this method is optimized for running convolution on a batch of grids simultaneously and in parallel,
        which is more efficient than processing individual grids separately when you have a batch of data.

        Args:
            kernel_size (NumericRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for (x, y, z) dimensions.
            stride (NumericRank1): Convolution stride. Can be a single int or 3-element sequence.
            source_grid (GridBatch): :class:`fvdb.GridBatch` encoding the structure of the input domain.
            target_grid (GridBatch | None): :class:`fvdb.GridBatch` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
            expert_config (dict[str, Any]): Advanced configuration options *(rarely needed by typical users)*.
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.
            topology_policy (ConvolutionTopologyPolicy | None): ``COMPLETE`` generates the complete structural
                support; ``RESTRICTED`` requires ``target_grid`` and restricts the relation to it. When omitted,
                the policy is inferred from ``target_grid``.
            strict_output_coverage (bool): Reject explicit targets with degree-zero output rows.
            acknowledge_incomplete_coverage (bool): Suppress the once-per-geometry warning for
                stride residues that are intentionally not sampled.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for :meth:`execute()` operations.

        Example:

        .. code-block:: python

            # Create a batched grid
            grid_batch = GridBatch.from_points(...)

            # Create plan for 3x3x3 convolution on batched grids
            plan = ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=1,
                source_grid=grid_batch
            )

            # execute to batched data
            batch_data = JaggedTensor(torch.randn(5, 1000, 8, device="cuda"))
            weights = torch.randn(16, 8, 3, 3, 3, device="cuda")
            output = plan.execute(batch_data, weights)

        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        resolved_policy = _resolve_topology_policy(target_grid, topology_policy)
        topology_provenance = (
            ConvolutionTopologyProvenance.GENERATED
            if resolved_policy is ConvolutionTopologyPolicy.COMPLETE
            else ConvolutionTopologyProvenance.EXPLICIT_TARGET
        )
        backend_name = expert_config.get("backend", "default")

        if backend_name == "dense":
            raise ValueError(_DENSE_BACKEND_DISABLED_MESSAGE)
        if backend_name == "pred_gather_igemm":
            _validate_pred_gather_igemm_admission(kernel_size, stride, channel_pairs, transposed=False)
            _validate_pred_gather_igemm_grid_admission(source_grid, target_grid)
        if target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

        geometry = _fvdb_cpp.ConvolutionGeometry(kernel_size, stride)
        compatibility = _transform_compatibility(source_grid, target_grid, geometry)
        _validate_transform_compatibility(compatibility)
        _warn_if_incomplete_residue_coverage(geometry, acknowledge_incomplete_coverage)
        backend = cls._build_backend(source_grid, target_grid, kernel_size, stride, channel_pairs, expert_config)
        _validate_coverage_policy(backend, resolved_policy, strict_output_coverage)
        return cls(
            source_grid,
            target_grid,
            geometry,
            channel_pairs,
            False,
            backend,
            compatibility,
            resolved_policy,
            topology_provenance,
            _CoverageReportCache(backend, source_grid, target_grid),
            False,
        )

    @classmethod
    def from_grid_batch_transposed(
        cls,
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: GridBatch,
        target_grid: GridBatch | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
        channel_pairs: tuple[tuple[int, int], ...] = _ANY_CHANNEL_PAIRS,
        topology_policy: ConvolutionTopologyPolicy | None = None,
        strict_output_coverage: bool = False,
        acknowledge_incomplete_coverage: bool = False,
    ) -> "ConvolutionPlan":
        """
        Create a :class:`ConvolutionPlan` for *transposed* convolution on batches of grids.
        *i.e.* transposed convolution where the input
        and output domains are both of type :class:`fvdb.GridBatch`.

        Transposed convolution is commonly used for decoder and generative operations. It evaluates
        the same fine/coarse graph in the opposite direction; it is not an inverse and need not
        recover an input or its topology.

        .. note::

            ``target_grid=None`` selects :attr:`~fvdb.ConvolutionTopologyPolicy.COMPLETE` and generates the
            complete uncropped transposed support. An explicit target selects
            :attr:`~fvdb.ConvolutionTopologyPolicy.RESTRICTED` and may contain
            zero-degree rows. This factory supports independently learned transpose weights; for
            the exact weighted adjoint of a particular plan, use :meth:`from_plan_transposed` and
            pass ``weight.transpose(0, 1).contiguous()`` at execution.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid (GridBatch): :class:`fvdb.GridBatch` encoding the structure of the input domain.
            target_grid (GridBatch | None): :class:`fvdb.GridBatch` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.
            topology_policy (ConvolutionTopologyPolicy | None): ``COMPLETE`` generates the complete structural
                support; ``RESTRICTED`` requires ``target_grid``. When omitted, the policy is inferred from
                ``target_grid``.
            strict_output_coverage (bool): Reject explicit targets with degree-zero output rows.
            acknowledge_incomplete_coverage (bool): Suppress the once-per-geometry warning for
                stride residues that are intentionally not sampled.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for transposed convolution operations via :meth:`execute()`.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        resolved_policy = _resolve_topology_policy(target_grid, topology_policy)
        topology_provenance = (
            ConvolutionTopologyProvenance.GENERATED
            if resolved_policy is ConvolutionTopologyPolicy.COMPLETE
            else ConvolutionTopologyProvenance.EXPLICIT_TARGET
        )
        backend_name = expert_config.get("backend", "default")

        if backend_name == "dense":
            raise ValueError(_DENSE_BACKEND_DISABLED_MESSAGE)
        if backend_name == "pred_gather_igemm":
            _validate_pred_gather_igemm_admission(kernel_size, stride, channel_pairs, transposed=True)
            _validate_pred_gather_igemm_grid_admission(source_grid, target_grid)
        if target_grid is None:
            target_grid = source_grid.conv_transpose_grid(kernel_size, stride)

        geometry = _fvdb_cpp.ConvolutionGeometry(kernel_size, stride)
        compatibility = _transform_compatibility(target_grid, source_grid, geometry)
        _validate_transform_compatibility(compatibility)
        _warn_if_incomplete_residue_coverage(geometry, acknowledge_incomplete_coverage)
        backend = cls._build_backend(
            source_grid, target_grid, kernel_size, stride, channel_pairs, expert_config, transposed=True
        )
        _validate_coverage_policy(backend, resolved_policy, strict_output_coverage)
        return cls(
            source_grid,
            target_grid,
            geometry,
            channel_pairs,
            True,
            backend,
            compatibility,
            resolved_policy,
            topology_provenance,
            _CoverageReportCache(backend, source_grid, target_grid),
            False,
        )

    @classmethod
    def from_plan_transposed(cls, plan: "ConvolutionPlan") -> "ConvolutionPlan":
        """
        Create a transposed version of an existing :class:`ConvolutionPlan`.

        This method creates a new plan that performs the exact transpose operation of
        the given plan (*i.e* convolution becomes transposed convolution and vice versa).
        It swaps the source and target grids, reverses the stored finite edge set and
        channel pairs, and flips the transposed flag. It does not reconstruct topology
        from the swapped grids.

        .. note::

            This is useful when a paired layer must apply the exact finite
            adjoint connectivity of an existing plan. It is not an inverse and
            need not recover the original input.

        Args:
            plan (ConvolutionPlan): An existing :class:`ConvolutionPlan` to transpose.

        Returns:
            convolution_plan (ConvolutionPlan): A new plan that performs the transpose of the input plan.

        Example:

        .. code-block:: python

            # Create forward plan
            forward_plan = ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=1,
                source_grid=input_grid_batch
            )

            # Create the corresponding backward/transpose plan
            transposed_plan = ConvolutionPlan.from_plan_transposed(forward_plan)
        """
        # Swap source/target grids, flip transposed flag, reverse channel pairs.
        source_grid = plan._target_grid
        target_grid = plan._source_grid
        transposed = not plan._transposed
        channel_pairs = tuple((dst, src) for src, dst in plan._channel_pairs)

        if transposed:
            compatibility = _transform_compatibility(target_grid, source_grid, plan._geometry)
        else:
            compatibility = _transform_compatibility(source_grid, target_grid, plan._geometry)
        _validate_transform_compatibility(compatibility)

        if isinstance(plan._backend, _GatherScatterBackend):
            backend: _Backend = _GatherScatterBackend(topology=_fvdb_cpp.gs_reverse_topology(plan._backend.topology))
        elif isinstance(plan._backend, _PredGatherIGemmBackend):
            # PredGatherIGemm is forward-only. Its exact transpose uses the
            # already-stored default gather/scatter fallback rulebook.
            backend = _GatherScatterBackend(topology=_fvdb_cpp.gs_reverse_topology(plan._backend.gs_topology))
        elif isinstance(plan._backend, _MatmulBackend):
            backend = plan._backend
        else:
            raise TypeError(f"Cannot transpose unknown convolution backend: {type(plan._backend)}")

        return cls(
            source_grid,
            target_grid,
            plan._geometry,
            channel_pairs,
            transposed,
            backend,
            compatibility,
            ConvolutionTopologyPolicy.RESTRICTED,
            ConvolutionTopologyProvenance.EXACT_TRANSPOSE,
            plan._coverage_report_cache,
            not plan._coverage_report_swapped,
        )

    # ============================================================
    #                 Validation
    # ============================================================

    def valid_usage(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        transposed: bool,
    ) -> bool:
        """
        Check if this :class:`ConvolutionPlan` is valid for the given usage. This method
        returns ``True`` if the plan can apply a (transposed) convolution with the given ``kernel_size`` and ``stride``
        from ``in_channels`` to ``out_channels``.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (NumericMaxRank1): Kernel size. Can be a single int or 3-element sequence.
            stride (NumericMaxRank1): Stride. Can be a single int or 3-element sequence.
            transposed (bool): Whether the plan is transposed.

        Returns:
            is_valid (bool): ``True`` if the plan is valid for the given configuration, ``False`` otherwise.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend_channels_supported = not isinstance(
            self._backend, _PredGatherIGemmBackend
        ) or _pred_gather_igemm_channel_pair_supported(in_channels, out_channels)
        return (
            _channel_pair_supported(in_channels, out_channels, self._channel_pairs)
            and backend_channels_supported
            and tuple(kernel_size.tolist()) == tuple(self._geometry.kernel_size)
            and tuple(stride.tolist()) == tuple(self._geometry.stride)
            and transposed == self._transposed
        )

    # ============================================================
    #                 Execute
    # ============================================================

    @overload
    def execute(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor: ...

    @overload
    def execute(self, data: JaggedTensor, weights: torch.Tensor) -> JaggedTensor: ...

    def execute(self, data: JaggedTensor | torch.Tensor, weights: torch.Tensor) -> JaggedTensor | torch.Tensor:
        """
        Execute this :class:`ConvolutionPlan` with the input data and weights.

        This is the main method for performing convolution operations. It applies
        the convolution kernel to the sparse voxel data according to the plan's
        pre-configured structure and optimizations.

        If the source grid batch has size 1,
        then ``data`` can be a :class:`torch.Tensor` with shape ``(total_voxels, in_channels)``.

        If the source grid batch has size > 1,
        then ``data`` should be a :class:`~fvdb.JaggedTensor` with shape ``(batch_size, num_voxels_in_grid_b, in_channels)``.

        .. note::

            - The same plan can be reused with different weights and data
            - Channel pairs must match those specified during plan creation
            - The plan automatically handles the sparse structure and backend optimizations
            - For transposed convolution plans, this performs the transpose operation

        Args:
            data (torch.Tensor | JaggedTensor): Input voxel features. Can be either:
                 *(i)* :class:`torch.Tensor` for single grids: shape ``(total_voxels, in_channels)`` **or**
                 *(ii)* :class:`~fvdb.JaggedTensor` for batches of grids: shape ``(batch_size, num_voxels_in_grid_b, in_channels)``
            weights (torch.Tensor): Convolution kernel weights with shape:
                    ``(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])``.
                    Identity ``K=1, S=1`` plans also accept the compact shape
                    ``(out_channels, in_channels)``.

        Returns:
            output_features (torch.Tensor | JaggedTensor): Convolved features with the same type as input:
                *(i)* :class:`torch.Tensor` with shape ``(total_output_voxels, out_channels)`` for single grids **or**
                *(ii)* :class:`~fvdb.JaggedTensor` with shape ``(batch_size, output_voxels_per_grid, out_channels)`` for batches

        Raises:
            ValueError: If the channel pair ``(in_channels, out_channels)`` from the weights
                       is not supported by this plan's channel_pairs configuration.

        Example:

        .. code-block:: python

            # Single grid example
            features = torch.randn(1000, 32, device="cuda")  # 1000 voxels, 32 channels
            weights = torch.randn(64, 32, 3, 3, 3, device="cuda")  # 32->64 channels, 3x3x3 kernel
            output = plan.execute(features, weights)  # Shape: (output_voxels, 64)

            # Batched example
            batch_features = JaggedTensor(torch.randn(5, 1000, 32, device="cuda"))
            output = plan.execute(batch_features, weights)  # Shape: (5, output_voxels, 64)
        """
        assert isinstance(data, (torch.Tensor, JaggedTensor)), "data must be a torch.Tensor or JaggedTensor"
        assert isinstance(weights, torch.Tensor), "weights must be a torch.Tensor"

        backend = self._backend
        if isinstance(backend, _MatmulBackend):
            weight_matrix = _matmul_weight_matrix(weights)
            out_c, in_c = weight_matrix.shape
        else:
            if weights.ndim < 2:
                raise ValueError("Convolution weights must have output and input channel dimensions")
            out_c = weights.shape[0]
            in_c = weights.shape[1]

        if not _channel_pair_supported(in_c, out_c, self._channel_pairs):
            raise ValueError(f"Channel pair {in_c, out_c} is not supported")
        if isinstance(backend, _PredGatherIGemmBackend) and not _pred_gather_igemm_channel_pair_supported(in_c, out_c):
            raise ValueError(
                f"PredGatherIGemm requires input and output channel counts divisible by 32; got ({in_c}, {out_c})."
            )

        is_flat: bool = isinstance(data, torch.Tensor)
        if is_flat:
            if self._source_grid.grid_count != 1:
                raise ValueError("Source grid must have batch size of 1 for flat data")

        # Matmul: 1x1x1 kernel, stride 1 — no kernel map needed
        if isinstance(backend, _MatmulBackend):
            if is_flat:
                return data.matmul(weight_matrix.transpose(0, 1))
            else:
                out_data = data.jdata.matmul(weight_matrix.transpose(0, 1))
                return data.jagged_like(out_data)

        if is_flat:
            data = JaggedTensor(data)

        # Gather-scatter: precomputed normalized topology with Python autograd
        if isinstance(backend, _GatherScatterBackend):
            out_tensor = _GatherScatterConvFn.apply(data.jdata, weights, backend.topology, self._transposed)
            if out_tensor is None:
                raise ValueError("Gather-scatter convolution returned None")
            if not isinstance(out_tensor, torch.Tensor):
                raise ValueError("Gather-scatter convolution returned non-tensor")
            result = self._target_grid.jagged_like(out_tensor)

        elif isinstance(backend, _PredGatherIGemmBackend):
            out_tensor = _PredGatherIGemmConvFn.apply(
                data.jdata,
                weights,
                _get_grid_data(self._source_grid),
                _get_grid_data(self._target_grid),
                backend.gs_topology,
                backend.kernel_size,
                backend.stride,
            )
            if out_tensor is None:
                raise ValueError("PredGatherIGemm convolution returned None")
            if not isinstance(out_tensor, torch.Tensor):
                raise ValueError("PredGatherIGemm convolution returned non-tensor")
            result = self._target_grid.jagged_like(out_tensor)

        else:
            raise TypeError(f"Unknown backend type: {type(backend)}")

        if is_flat:
            return result.jdata
        else:
            return result

    # ============================================================
    #                 Properties
    # ============================================================

    @property
    def source_grid_batch(self) -> GridBatch:
        """
        Return the :class:`fvdb.GridBatch` representing the source domain of the convolution.
        If the plan was created for a single grid, it is returned as a batch of size 1.

        Returns:
            source_grid_batch (GridBatch): The source :class:`fvdb.GridBatch` of the convolution plan.
        """
        return self._source_grid

    @property
    def target_grid_batch(self) -> GridBatch:
        """
        Return the :class:`fvdb.GridBatch` representing the target domain of the convolution.
        If the plan was created for a single grid, it is returned as a batch of size 1.

        Returns:
            target_grid_batch (GridBatch): The target :class:`fvdb.GridBatch` of the convolution plan.
        """
        return self._target_grid

    @property
    def geometry(self) -> _fvdb_cpp.ConvolutionGeometry:
        """Canonical immutable geometry shared by this plan's topology and executors."""
        return self._geometry

    @property
    def kernel_size(self) -> torch.Tensor:
        """Kernel dimensions in the canonical ``torch_same_phase`` geometry."""
        return torch.tensor(self._geometry.kernel_size, dtype=torch.int32)

    @property
    def stride(self) -> torch.Tensor:
        """Stride dimensions in the canonical ``torch_same_phase`` geometry."""
        return torch.tensor(self._geometry.stride, dtype=torch.int32)

    @property
    def transform_compatibility(self) -> ConvolutionTransformCompatibility:
        """Validated fine/coarse transform compatibility diagnostic."""
        return self._transform_compatibility

    @property
    def phase_policy(self) -> ConvolutionPhasePolicy:
        """Kernel phase convention used by this plan."""
        return ConvolutionPhasePolicy(self._geometry.phase_policy)

    @property
    def topology_policy(self) -> ConvolutionTopologyPolicy:
        """Resolved complete or restricted topology policy."""
        return self._topology_policy

    @property
    def topology_provenance(self) -> ConvolutionTopologyProvenance:
        """How this plan's finite topology was obtained.

        Exact transposes always use the stored finite edge set of their source
        plan and therefore have restricted policy.
        """
        return self._topology_provenance

    @property
    def coverage_report(self) -> ConvolutionCoverageReport | None:
        """Exact input/output rulebook degree diagnostics, when the backend has a rulebook."""
        return self._coverage_report_cache.get(self._coverage_report_swapped)

    @property
    def has_fixed_topology(self) -> bool:
        """
        Returns ``True`` if the source and target grids have the same topology,
        meaning the same voxel structure.

        Returns:
            has_fixed_topology (bool): ``True`` if source and target grids are the same topology, ``False`` otherwise.
        """
        return _get_grid_data(self._source_grid).is_same(_get_grid_data(self._target_grid))

    # ============================================================
    #                 Private methods
    # ============================================================

    @staticmethod
    def _build_backend(
        source_grid: GridBatch,
        target_grid: GridBatch,
        kernel_size: torch.Tensor,
        stride: torch.Tensor,
        channel_pairs: tuple[tuple[int, int], ...],
        expert_config: dict[str, Any],
        transposed: bool = False,
    ) -> _Backend:
        """
        Determine the convolution method and build the appropriate backend.
        """
        backend_name = expert_config.get("backend", "default")
        if backend_name == "dense":
            raise ValueError(_DENSE_BACKEND_DISABLED_MESSAGE)

        for channel_pair in channel_pairs:
            if len(channel_pair) != 2 or channel_pair[0] <= 0 or channel_pair[1] <= 0:
                raise ValueError("channel_pair must be a tuple of two positive integers")

        if backend_name == "pred_gather_igemm":
            kernel, step = _validate_pred_gather_igemm_admission(
                kernel_size, stride, channel_pairs, transposed=transposed
            )
            _validate_pred_gather_igemm_grid_admission(source_grid, target_grid)
            gs_topo = _fvdb_cpp.gs_build_topology(
                _get_grid_data(source_grid), _get_grid_data(target_grid), kernel_size, stride
            )
            return _PredGatherIGemmBackend(gs_topology=gs_topo, kernel_size=kernel, stride=step)

        if backend_name not in ("gather_scatter", "default"):
            raise ValueError(f"Unknown backend: {backend_name!r}")

        # Identity geometry is matmul only when row ordering is guaranteed by
        # shared GridBatchData identity. Distinct equal-looking grids use the
        # ordinary rulebook until an ordered-topology equality predicate exists.
        if (
            _vec_is_all(stride, 1)
            and _vec_is_all(kernel_size, 1)
            and _get_grid_data(source_grid).is_same(_get_grid_data(target_grid))
        ):
            return _MatmulBackend()

        if transposed:
            topology = _fvdb_cpp.gs_build_transpose_topology(
                _get_grid_data(source_grid), _get_grid_data(target_grid), kernel_size, stride
            )
        else:
            topology = _fvdb_cpp.gs_build_topology(
                _get_grid_data(source_grid), _get_grid_data(target_grid), kernel_size, stride
            )
        return _GatherScatterBackend(topology=topology)


# These tests are to validate that the type-checking is happy. They won't actually run because
# the grid generation is nonsense.


def _grid_batch_test_for_typing():
    batch_size = 5
    voxel_sizes = [0.1] * batch_size
    origins = [0] * batch_size

    grid_batch = GridBatch.from_zero_voxels(device="cuda", voxel_sizes=voxel_sizes, origins=origins)

    plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid_batch)
    plan_t = ConvolutionPlan.from_plan_transposed(plan)

    weights_1 = torch.randn(16, 8, 3, 3, 3, device="cuda")
    weights_2 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_3 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_4 = torch.randn(8, 16, 3, 3, 3, device="cuda")

    data_1 = torch.randn(batch_size, 100, 8, device="cuda")

    out_1: torch.Tensor = plan.execute(data_1, weights_1)
    out_2: torch.Tensor = plan.execute(out_1, weights_2)

    out_3: torch.Tensor = plan_t.execute(out_2, weights_3)
    out_4: torch.Tensor = plan_t.execute(out_3, weights_4)
