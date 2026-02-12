# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Black-box encapsulation of configuration structures for sparse convolution using
fVDB Grid and GridBatch. Design is intended to be reminiscent of the "plan" concept from FFT
libraries. Like FFT plans, the convolution plan encapsulates a single direction - regular
convolution, or transposed convolution, but can represent either.
"""

from dataclasses import dataclass
from typing import Any, overload

import torch
from fvdb.types import NumericMaxRank1, ValueConstraint, to_Vec3i

from fvdb import Grid, GridBatch, JaggedTensor

from . import _fvdb_cpp

_DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "default",
}

_ANY_CHANNEL_PAIRS: tuple[tuple[int, int], ...] = ()


def _vec_is_all(v: torch.Tensor, i: int | float) -> bool:
    return bool(torch.all(torch.eq(v, i)).item())


def _channel_pair_supported(in_channels: int, out_channels: int, channel_pairs: tuple[tuple[int, int], ...]) -> bool:
    if len(channel_pairs) == 0:
        return True
    return (in_channels, out_channels) in channel_pairs


# ============================================================
#  Autograd functions for gather-scatter convolution
# ============================================================


class _GatherScatterConvFn(torch.autograd.Function):
    """Autograd wrapper for the GEMM-based gather-scatter convolution with precomputed topology."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, weights: torch.Tensor, topo: _fvdb_cpp.GatherScatterTopology) -> torch.Tensor:  # type: ignore[override]
        output = _fvdb_cpp.gs_conv(features, weights, topo)
        ctx.save_for_backward(features, weights)
        ctx.topo = topo
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:  # type: ignore[override]
        features, weights = ctx.saved_tensors
        grad_feat, grad_w = _fvdb_cpp.gs_conv_backward(grad_output, features, weights, ctx.topo)
        return grad_feat, grad_w, None


class _GatherScatterFusedConvFn(torch.autograd.Function):
    """Autograd wrapper for the fused gather-scatter convolution (small-C optimized)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        features: torch.Tensor,
        weights: torch.Tensor,
        src_grid: _fvdb_cpp.GridBatch,
        dst_grid: _fvdb_cpp.GridBatch,
        kernel_size: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        output = _fvdb_cpp.gs_conv_fused(features, weights, src_grid, dst_grid, kernel_size, stride)
        ctx.save_for_backward(features, weights)
        ctx.src_grid = src_grid
        ctx.dst_grid = dst_grid
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:  # type: ignore[override]
        features, weights = ctx.saved_tensors
        grad_feat, grad_w = _fvdb_cpp.gs_conv_fused_backward(
            grad_output, features, weights, ctx.src_grid, ctx.dst_grid, ctx.kernel_size, ctx.stride
        )
        return grad_feat, grad_w, None, None, None, None


# ============================================================
#  Backend data classes — cached precomputed data per method
# ============================================================


@dataclass(frozen=True)
class _MatmulBackend:
    """1x1x1 convolution with stride 1 — pure matmul, no precomputed data."""

    pass


@dataclass(frozen=True)
class _DenseBackend:
    """Dense convolution via torch.nn.functional — no precomputed data."""

    pass


@dataclass(frozen=True)
class _GatherScatterBackend:
    """Gather-scatter convolution with precomputed GatherScatterTopology (Python autograd)."""

    topology: _fvdb_cpp.GatherScatterTopology


@dataclass(frozen=True)
class _GatherScatterFusedBackend:
    """Fused gather-scatter convolution -- no precomputed topology (Python autograd)."""

    pass


@dataclass(frozen=True)
class _GatherScatterOldBackend:
    """Legacy gather-scatter using SparseConvolutionKernelMap C++ autograd."""

    neighbor_map: torch.Tensor  # [#IO, 2] int32
    neighbor_sizes: torch.Tensor  # [K] int32


_Backend = (
    _MatmulBackend | _DenseBackend | _GatherScatterBackend | _GatherScatterFusedBackend | _GatherScatterOldBackend
)


@dataclass(frozen=True)
class ConvolutionPlan:
    """
    A pre-configured plan for efficient sparse 3D convolution operations on :class:`fvdb.Grid`
    and :class:`fvdb.GridBatch`.

    :class:`ConvolutionPlan` encapsulates all the configuration and optimization structures needed
    to perform sparse convolution operations efficiently. Like `FFT plans in signal processing libraries <https://www.fftw.org/fftw3_doc/Using-Plans.html>`_,
    a :class:`ConvolutionPlan` represents a single direction of computation - either
    regular convolution or transposed convolution.

    The plan handles the complex sparse data structures and backend optimizations internally,
    allowing users to focus on the core convolution parameters: input/output channels,
    kernel size, stride, and the grid structure.

    Transposition is treated as just a different kind of kernel, so the inputs and outputs and
    weights are treated the same as if it were a regular convolution. For the default padded case,
    transposed outputs can't automatically infer the target_grid, so it must be provided.

    Usage Pattern:

    1. Create a plan using one of the ``from_*`` class methods (see :meth:`from_grid_batch()`, and :meth:`from_grid()`).
    2. Use the :meth:`execute()` method to perform convolutions with different weights and data on
       the same grid structures.
    3. Reuse the same plan for multiple convolutions with the same configuration

    Example Usage:

    .. code-block:: python

        from fvdb import Grid, ConvolutionPlan

        # Create a grid
        my_grid = Grid.from_ijk(...)

        # Create a plan for 3x3x3 convolution with stride 1
        plan = ConvolutionPlan.from_grid(
            kernel_size=3,
            stride=1,
            source_grid=my_grid
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
    _kernel_size: torch.Tensor
    _stride: torch.Tensor
    _channel_pairs: tuple[tuple[int, int], ...]
    _transposed: bool
    _backend: _Backend

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
                *(For the dense backend, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options *(rarely needed by typical users)*.
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

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

        backend_name = expert_config.get("backend", "default")

        if backend_name == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend.")
            target_grid = source_grid
        elif target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

        backend = cls._build_backend(source_grid, target_grid, kernel_size, stride, channel_pairs, expert_config)
        return cls(source_grid, target_grid, kernel_size, stride, channel_pairs, False, backend)

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
    ) -> "ConvolutionPlan":
        """
        Create a :class:`ConvolutionPlan` for *transposed* convolution on batches of grids.
        *i.e.* transposed convolution where the input
        and output domains are both of type :class:`fvdb.GridBatch`.

        Transposed convolution (also known as deconvolution) is commonly used for
        upsampling operations, such as in decoder networks or generative models.
        It performs the mathematical transpose of the convolution operation.

        .. note::

            Though deconvolution is the "reverse" of convolution in some sense, this configuration
            still treats input and output channels as inputs and outputs, it doesn't swap them.
            The source and target grids are not swapped, it is best to think of deconvolution as
            convolution with a different kernel than deconvolution, but it is otherwise the same kind
            of abstract operation.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid (GridBatch): :class:`fvdb.GridBatch` encoding the structure of the input domain.
            target_grid (GridBatch | None): :class:`fvdb.GridBatch` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
                *(For the dense backend, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for transposed convolution operations via :meth:`execute()`.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend_name = expert_config.get("backend", "default")

        if backend_name == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend, transposed.")
            target_grid = source_grid
        elif target_grid is None:
            raise ValueError("Target grid must be provided for transposed convolution, except for dense backend.")

        backend = cls._build_backend(
            source_grid, target_grid, kernel_size, stride, channel_pairs, expert_config, transposed=True
        )
        return cls(source_grid, target_grid, kernel_size, stride, channel_pairs, True, backend)

    @classmethod
    def from_grid(
        cls,
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: Grid,
        target_grid: Grid | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
        channel_pairs: tuple[tuple[int, int], ...] = _ANY_CHANNEL_PAIRS,
    ) -> "ConvolutionPlan":
        """
        Create a :class:`ConvolutionPlan` for convolution on a single grid. *i.e.* convolution where the input
        and output domains are both of type :class:`fvdb.Grid`.

        This method creates a plan for processing a single grid, which is suitable
        when you have individual grids rather than batched data (for that case, use :meth:`from_grid_batch`).

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride (NumericMaxRank1): Convolution stride. Can be a single int or 3-element sequence.
            source_grid (Grid): :class:`fvdb.Grid` encoding the structure of the input domain.
            target_grid (Grid | None): :class:`fvdb.Grid` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
                *(For the dense backend, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for :meth:`execute()` operations.

        Example:

        .. code-block:: python

            # Create a single grid
            grid = Grid.from_zero_voxels(device="cuda", voxel_size=0.1, origin=0)

            # Create plan for 3x3x3 convolution
            plan = ConvolutionPlan.from_grid(
                kernel_size=3,
                stride=1,
                source_grid=grid
            )

            # execute to single grid data
            features = torch.randn(100, 8, device="cuda")
            weights = torch.randn(16, 8, 3, 3, 3, device="cuda")
            output = plan.execute(features, weights)

        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend_name = expert_config.get("backend", "default")

        source_grid_batch = GridBatch(impl=source_grid._impl)
        if backend_name == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend.")
            target_grid_batch = source_grid_batch
        elif target_grid is None:
            target_grid_batch = source_grid_batch.conv_grid(kernel_size, stride)
        else:
            target_grid_batch = GridBatch(impl=target_grid._impl)

        backend = cls._build_backend(
            source_grid_batch, target_grid_batch, kernel_size, stride, channel_pairs, expert_config
        )
        return cls(source_grid_batch, target_grid_batch, kernel_size, stride, channel_pairs, False, backend)

    @classmethod
    def from_grid_transposed(
        cls,
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: Grid,
        target_grid: Grid | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
        channel_pairs: tuple[tuple[int, int], ...] = _ANY_CHANNEL_PAIRS,
    ) -> "ConvolutionPlan":
        """
        Create a :class:`ConvolutionPlan` for *transposed* convolution on a single grid.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride (NumericMaxRank1): Convolution stride. Can be a single int or 3-element sequence.
            source_grid (Grid): :class:`fvdb.Grid` encoding the structure of the input domain.
            target_grid (Grid | None): :class:`fvdb.Grid` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
                *(For the dense backend, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for transposed convolution operations.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend_name = expert_config.get("backend", "default")

        source_grid_batch = GridBatch(impl=source_grid._impl)
        if backend_name == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend, transposed.")
            target_grid_batch = source_grid_batch
        elif target_grid is None:
            raise ValueError("Target grid must be provided for transposed convolution, except for dense backend.")
        else:
            target_grid_batch = GridBatch(impl=target_grid._impl)

        backend = cls._build_backend(
            source_grid_batch, target_grid_batch, kernel_size, stride, channel_pairs, expert_config, transposed=True
        )
        return cls(source_grid_batch, target_grid_batch, kernel_size, stride, channel_pairs, True, backend)

    @classmethod
    def from_plan_transposed(cls, plan: "ConvolutionPlan") -> "ConvolutionPlan":
        """
        Create a transposed version of an existing :class:`ConvolutionPlan`.

        This method creates a new plan that performs the transpose operation of the
        given plan (*i.e* convolution becomes transposed convolution and vice versa).
        It automatically swaps the source and target grids, reverses the channel pairs, and flips the transposed flag.

        .. note::

            This is particularly useful for creating encoder-decoder pairs where
            the decoder needs to undo the operations of the encoder.

        Args:
            plan (ConvolutionPlan): An existing :class:`ConvolutionPlan` to transpose.

        Returns:
            convolution_plan (ConvolutionPlan): A new plan that performs the transpose of the input plan.

        Example:

        .. code-block:: python

            # Create forward plan
            forward_plan = ConvolutionPlan.from_grid(
                kernel_size=3,
                stride=1,
                source_grid=input_grid
            )

            # Create the corresponding backward/transpose plan
            transposed_plan = ConvolutionPlan.from_plan_transposed(forward_plan)
        """
        # Swap source/target grids, flip transposed flag, reverse channel pairs
        source_grid = plan._target_grid
        target_grid = plan._source_grid
        transposed = not plan._transposed
        channel_pairs = tuple((dst, src) for src, dst in plan._channel_pairs)

        backend = cls._build_backend(
            source_grid,
            target_grid,
            plan._kernel_size,
            plan._stride,
            channel_pairs,
            _DEFAULT_CONFIG,
            transposed=transposed,
        )
        return cls(source_grid, target_grid, plan._kernel_size, plan._stride, channel_pairs, transposed, backend)

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

        return (
            _channel_pair_supported(in_channels, out_channels, self._channel_pairs)
            and torch.equal(kernel_size, self._kernel_size)
            and torch.equal(stride, self._stride)
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

        If this plan was created for a single grid (*e.g.* using :meth:`from_grid()` or :meth:`from_grid_transposed()`),
        then ``data`` should be a :class:`torch.Tensor` with shape ``(total_voxels, in_channels)``.

        If this plan was created for a batch of grids (*e.g.* using :meth:`from_grid_batch()` or :meth:`from_grid_batch_transposed()`),
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
                    ``(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])``

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
        out_c = weights.shape[0]
        in_c = weights.shape[1]
        if not _channel_pair_supported(in_c, out_c, self._channel_pairs):
            raise ValueError(f"Channel pair {in_c, out_c} is not supported")

        assert isinstance(data, (torch.Tensor, JaggedTensor)), "data must be a torch.Tensor or JaggedTensor"
        assert isinstance(weights, torch.Tensor), "weights must be a torch.Tensor"

        is_flat: bool = isinstance(data, torch.Tensor)
        if is_flat:
            if self._source_grid.grid_count != 1:
                raise ValueError("Source grid must have batch size of 1 for flat data")

        backend = self._backend

        # Matmul: 1x1x1 kernel, stride 1 — no kernel map needed
        if isinstance(backend, _MatmulBackend):
            if is_flat:
                return data.matmul(weights.transpose(0, 1))
            else:
                out_data = data.jdata.matmul(weights.transpose(0, 1))
                return data.jagged_like(out_data)

        if is_flat:
            data = JaggedTensor(data)

        # Dense: pure-Python path via torch.nn.functional
        if isinstance(backend, _DenseBackend):
            result = self._execute_dense(data, weights)

        # Gather-scatter (new): precomputed topology with Python autograd
        elif isinstance(backend, _GatherScatterBackend):
            out_tensor = _GatherScatterConvFn.apply(data.jdata, weights, backend.topology)
            if out_tensor is None:
                raise ValueError("Gather-scatter convolution returned None")
            if not isinstance(out_tensor, torch.Tensor):
                raise ValueError("Gather-scatter convolution returned non-tensor")
            result = self._target_grid.jagged_like(out_tensor)

        # Gather-scatter fused: no precomputed topology, Python autograd
        elif isinstance(backend, _GatherScatterFusedBackend):
            out_tensor = _GatherScatterFusedConvFn.apply(
                data.jdata,
                weights,
                self._source_grid._impl,
                self._target_grid._impl,
                self._kernel_size,
                self._stride,
            )
            if out_tensor is None:
                raise ValueError("Gather-scatter fused convolution returned None")
            if not isinstance(out_tensor, torch.Tensor):
                raise ValueError("Gather-scatter fused convolution returned non-tensor")
            result = self._target_grid.jagged_like(out_tensor)

        # Legacy gather-scatter: old C++ autograd path
        elif isinstance(backend, _GatherScatterOldBackend):
            src_voxels = int(self._source_grid.total_voxels)
            dst_voxels = int(self._target_grid.total_voxels)
            middle_accel = _vec_is_all(self._stride, 1)
            out_tensor = _fvdb_cpp.sparse_conv_kernel_map(
                data.jdata,
                weights,
                backend.neighbor_map,
                backend.neighbor_sizes,
                src_voxels,
                dst_voxels,
                middle_accel,
                self._transposed,
            )
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
    def source_grid(self) -> Grid:
        """
        Return the :class:`fvdb.Grid` representing the source domain of the convolution, or
        raise an error if the plan was created for a batch of grids.

        Returns:
            source_grid (Grid): The source :class:`fvdb.Grid` of the convolution plan.

        Raises:
            ValueError: If the plan was created for a batch of grids.
        """
        if self._source_grid.grid_count != 1:
            raise ValueError("Source grid must have batch size of 1 for Grid")
        return Grid(impl=self._source_grid._impl)

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
    def target_grid(self) -> Grid:
        """
        Return the :class:`fvdb.Grid` representing the target domain of the convolution, or
        raise an error if the plan was created for a batch of grids.

        Returns:
            target_grid (Grid): The target :class:`fvdb.Grid` of the convolution plan.

        Raises:
            ValueError: If the plan was created for a batch of grids.
        """
        if self._target_grid.grid_count != 1:
            raise ValueError("Target grid must have batch size of 1 for Grid")
        return Grid(impl=self._target_grid._impl)

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
    def has_fixed_topology(self) -> bool:
        """
        Returns ``True`` if the source and target grids have the same topology,
        meaning the same voxel structure.

        Returns:
            has_fixed_topology (bool): ``True`` if source and target grids are the same topology, ``False`` otherwise.
        """
        return self._source_grid._impl.is_same(self._target_grid._impl)

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
        for channel_pair in channel_pairs:
            if len(channel_pair) != 2 or channel_pair[0] <= 0 or channel_pair[1] <= 0:
                raise ValueError("channel_pair must be a tuple of two positive integers")

        backend_name = expert_config.get("backend", "default")

        # 1x1x1 conv with stride 1 is just a matmul — no kernel map needed
        if _vec_is_all(stride, 1) and _vec_is_all(kernel_size, 1):
            return _MatmulBackend()

        # Dense backend — pure Python, no kernel map
        if backend_name == "dense":
            if not _vec_is_all(stride, 1):
                raise ValueError("Dense backend requires stride 1.")
            if not source_grid._impl.is_same(target_grid._impl):
                raise ValueError("Dense backend requires source_grid and target_grid to be the same.")
            return _DenseBackend()

        # Gather-scatter fused — no precomputed topology
        if backend_name == "gather_scatter_fused":
            return _GatherScatterFusedBackend()

        # Gather-scatter (new) — precomputed topology with Python autograd
        if backend_name == "gather_scatter":
            topo = _fvdb_cpp.gs_build_topology(
                source_grid._impl,
                target_grid._impl,
                kernel_size,
                stride,
            )
            return _GatherScatterBackend(topology=topo)

        # Legacy gather-scatter — old C++ autograd path
        if backend_name == "gather_scatter_old":
            neighbor_map, neighbor_sizes = _fvdb_cpp.build_kernel_map(
                source_grid._impl,
                target_grid._impl,
                kernel_size,
                stride,
            )
            return _GatherScatterOldBackend(neighbor_map=neighbor_map, neighbor_sizes=neighbor_sizes)

        # Default: use old backend for transposed (not yet supported by new path),
        # new gather-scatter otherwise.
        if backend_name == "default":
            if transposed:
                neighbor_map, neighbor_sizes = _fvdb_cpp.build_kernel_map(
                    source_grid._impl,
                    target_grid._impl,
                    kernel_size,
                    stride,
                )
                return _GatherScatterOldBackend(neighbor_map=neighbor_map, neighbor_sizes=neighbor_sizes)
            else:
                topo = _fvdb_cpp.gs_build_topology(
                    source_grid._impl,
                    target_grid._impl,
                    kernel_size,
                    stride,
                )
                return _GatherScatterBackend(topology=topo)

        raise ValueError(f"Unknown backend: {backend_name!r}")

    def _execute_dense(self, data: JaggedTensor, weights: torch.Tensor) -> JaggedTensor:
        source_grid = self._source_grid
        assert source_grid._impl.is_same(self._target_grid._impl)

        min_coord = source_grid.ijk.jdata.min(dim=0).values
        # BXYZC -> BCXYZ
        dense_feature = source_grid.inject_to_dense_cmajor(data, min_coord=min_coord)
        if self._transposed:
            dense_feature = torch.nn.functional.conv_transpose3d(dense_feature, weights, padding=1, stride=1)
        else:
            dense_feature = torch.nn.functional.conv3d(dense_feature, weights, padding=1, stride=1)
        # BCXYZ -> BXYZC
        dense_feature = dense_feature.contiguous()
        return source_grid.inject_from_dense_cmajor(dense_feature, dense_origins=min_coord)


# These tests are to validate that the type-checking is happy. They won't actually run because
# the grid generation is nonsense.


def _grid_test_for_typing():
    voxel_size = 0.1
    origin = 0

    grid = Grid.from_zero_voxels(device="cuda", voxel_size=voxel_size, origin=origin)

    plan = ConvolutionPlan.from_grid(kernel_size=3, stride=1, source_grid=grid)
    plan_t = ConvolutionPlan.from_plan_transposed(plan)

    weights_1 = torch.randn(16, 8, 3, 3, 3, device="cuda")
    weights_2 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_3 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_4 = torch.randn(8, 16, 3, 3, 3, device="cuda")

    data_1 = torch.randn(100, 8, device="cuda")

    out_1: torch.Tensor = plan.execute(data_1, weights_1)
    out_2: torch.Tensor = plan.execute(out_1, weights_2)

    out_3: torch.Tensor = plan_t.execute(out_2, weights_3)
    out_4: torch.Tensor = plan_t.execute(out_3, weights_4)


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
