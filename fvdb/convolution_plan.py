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

from ._fvdb_cpp import ConvPackBackend
from ._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from ._fvdb_cpp import SparseConvPackInfo as SparseConvPackInfoCpp

_CUTLASS_SUPPORTED_CHANNELS: tuple[tuple[int, int], ...] = (
    (32, 64),
    (64, 128),
    (128, 256),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (128, 64),
    (64, 32),
    (256, 128),
    (384, 256),
    (192, 128),
    (256, 512),
    (512, 256),
    (512, 512),
)

_DEFAULT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

_DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "default",
    "allow_tf32": True,
    "weight_dtypes": _DEFAULT_DTYPES,
    "feature_dtypes": _DEFAULT_DTYPES,
}

_ANY_CHANNEL_PAIRS: tuple[tuple[int, int], ...] = ()


def _vec_is_all(v: torch.Tensor, i: int | float) -> bool:
    return bool(torch.all(torch.eq(v, i)).item())


def _channel_pair_supported(in_channels: int, out_channels: int, channel_pairs: tuple[tuple[int, int], ...]) -> bool:
    if len(channel_pairs) == 0:
        return True
    return (in_channels, out_channels) in channel_pairs


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
    transposed outputs can't automatically infer the target_grid, so it must be provided, unless
    the dense, halo, and lggs backends are used.

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

    _pack_info: SparseConvPackInfoCpp
    _channel_pairs: tuple[tuple[int, int], ...]
    _transposed: bool
    _expert_config: dict[str, Any]
    _backend: ConvPackBackend

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
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``. *(except for the case of the dense, halo, and
                lggs backends where it uses ``target_grid = source_grid``.
                For those backends, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options *(rarely needed by typical users)*.
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                This parameter can be useful to select a more performant backend when the
                channel configurations are known in advance.
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

        if not _vec_is_all(stride, 1):
            raise NotImplementedError("Strides not equal to 1 are not currently supported")

        if not _vec_is_all(stride, stride[0].item()):
            raise NotImplementedError("Non-uniform strides are not currently supported")

        backend = expert_config.get("backend", "default")
        if backend != "default":
            raise NotImplementedError("Non-default backends are not currently supported")

        if backend in ["dense", "halo", "lggs"]:
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense, halo, and lggs backends.")
            target_grid = source_grid
        elif target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

        pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid._impl, target_grid._impl)

        transposed = False
        backend = cls._configure_backend(pack_info, channel_pairs, transposed, expert_config)
        return cls(pack_info, channel_pairs, transposed, expert_config, backend)

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

        .. note::

            For most backends, ``target_grid`` can be automatically computed. Only certain
            expert backends require specific ``target_grid`` configurations.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid (GridBatch): :class:`fvdb.GridBatch` encoding the structure of the input domain.
            target_grid (GridBatch | None): :class:`fvdb.GridBatch` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``.
                *(except for the case of the dense, halo, and lggs backends where it uses ``target_grid = source_grid``.
                For those backends, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                This parameter can be useful to select a more performant backend when the
                channel configurations are known in advance.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for transposed convolution operations via :meth:`execute()`.


        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if not _vec_is_all(stride, 1):
            raise NotImplementedError("Strides not equal to 1 are not currently supported")

        if not _vec_is_all(stride, stride[0].item()):
            raise NotImplementedError("Non-uniform strides are not currently supported")

        backend = expert_config.get("backend", "default")
        if backend != "default":
            raise NotImplementedError("Non-default backends are not currently supported")

        if backend == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend, transposed.")
            target_grid = source_grid
        elif target_grid is None:
            raise ValueError("Target grid must be provided for transposed convolution, except for dense backend.")

        pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid._impl, target_grid._impl)

        transposed = True
        backend = cls._configure_backend(pack_info, channel_pairs, transposed, expert_config)
        return cls(pack_info, channel_pairs, transposed, expert_config, backend)

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
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``. *(except for the case of the dense, halo, and
                lggs backends where it uses ``target_grid = source_grid``.
                For those backends, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                This parameter can be useful to select a more performant backend when the
                channel configurations are known in advance.
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

        if not _vec_is_all(stride, 1):
            raise NotImplementedError("Strides not equal to 1 are not currently supported")

        if not _vec_is_all(stride, stride[0].item()):
            raise NotImplementedError("Non-uniform strides are not currently supported")

        backend = expert_config.get("backend", "default")
        if backend != "default":
            raise NotImplementedError("Non-default backends are not currently supported")

        if backend in ["dense", "halo", "lggs"]:
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense, halo, and lggs backends.")
            target_grid = source_grid
        elif target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

        pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid._impl, target_grid._impl)

        transposed = False
        backend = cls._configure_backend(pack_info, channel_pairs, transposed, expert_config)
        return cls(pack_info, channel_pairs, transposed, expert_config, backend)

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
        Create a :class:`ConvolutionPlan` for *transposed* convolution on a single grid. *i.e.* transposed convolution where the input
        and output domains are both of type :class:`fvdb.Grid`.

        Transposed convolution (also known as deconvolution) is commonly used for
        upsampling operations, such as in decoder networks or generative models.
        It performs the mathematical transpose of the convolution operation.

        .. note::

            Though deconvolution is the "reverse" of convolution in some sense, this configuration
            still treats input and output channels as inputs and outputs, it doesn't swap them.
            The source and target grids are not swapped, it is best to think of deconvolution as
            convolution with a different kernel than deconvolution, but it is otherwise the same kind
            of abstract operation.

        .. note::

            For most backends, ``target_grid`` can be automatically computed. Only certain
            expert backends require specific ``target_grid`` configurations.

        Args:
            kernel_size (NumericMaxRank1): Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for ``(x, y, z)`` dimensions.
            stride (NumericMaxRank1): Convolution stride. Can be a single int or 3-element sequence.
            source_grid (Grid): :class:`fvdb.Grid` encoding the structure of the input domain.
            target_grid (Grid | None): :class:`fvdb.Grid` encoding the structure of the output domain.
                If ``None``, the ``target_grid`` is automatically computed
                based on ``kernel_size`` and ``stride`` applied to ``source_grid``. *(except for the case of the dense, halo, and
                lggs backends where it uses ``target_grid = source_grid``.
                For those backends, ``target_grid`` must be ``None``.)*
            expert_config (dict[str, Any]): Advanced configuration options (rarely needed by typical users).
            channel_pairs (tuple[tuple[int, int], ...]): Supported input/output channel combinations as tuples.
                Each tuple represents (input_channels, output_channels).
                *e.g*: ``((32, 64), (64, 128))`` supports 32->64 and 64->128 convolutions.
                This parameter can be useful to select a more performant backend when the
                channel configurations are known in advance.
                Defaults to ``_ANY_CHANNEL_PAIRS``, which means any channel pairs are supported.

        Returns:
            convolution_plan (ConvolutionPlan): Configured plan ready for transposed convolution operations.

        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if not _vec_is_all(stride, 1):
            raise NotImplementedError("Strides not equal to 1 are not currently supported")

        if not _vec_is_all(stride, stride[0].item()):
            raise NotImplementedError("Non-uniform strides are not currently supported")

        backend = expert_config.get("backend", "default")
        if backend != "default":
            raise NotImplementedError("Non-default backends are not currently supported")

        if backend == "dense":
            if target_grid is not None:
                raise ValueError("Target grid must be None for dense backend, transposed.")
            target_grid = source_grid
        elif target_grid is None:
            raise ValueError("Target grid must be provided for transposed convolution, except for dense backend.")

        pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid._impl, target_grid._impl)

        transposed = True
        backend = cls._configure_backend(pack_info, channel_pairs, transposed, expert_config)
        return cls(pack_info, channel_pairs, transposed, expert_config, backend)

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
        kernel_size = to_Vec3i(plan._pack_info.kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(plan._pack_info.stride, value_constraint=ValueConstraint.POSITIVE)

        # Transposing!
        source_grid = plan._pack_info.target_grid
        target_grid = plan._pack_info.source_grid
        transposed = not plan._transposed
        channel_pairs = tuple((dst, src) for src, dst in plan._channel_pairs)
        expert_config = plan._expert_config

        t_pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid, target_grid)
        t_backend = cls._configure_backend(t_pack_info, channel_pairs, transposed, expert_config)
        return cls(t_pack_info, channel_pairs, transposed, expert_config, t_backend)

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

        self_kernel_size = to_Vec3i(self._pack_info.kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self_stride = to_Vec3i(self._pack_info.stride, value_constraint=ValueConstraint.POSITIVE)

        return (
            _channel_pair_supported(in_channels, out_channels, self._channel_pairs)
            and torch.equal(kernel_size, self_kernel_size)
            and torch.equal(stride, self_stride)
            and transposed == self._transposed
        )

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
            if self._pack_info.source_grid.grid_count != 1:
                raise ValueError("Source grid must have batch size of 1 for flat data")

        # Handle the simple matmul case before the more complex cases
        if self._backend == ConvPackBackend.MATMUL:
            if is_flat:
                return data.matmul(weights.transpose(0, 1))
            else:
                out_data = data.jdata.matmul(weights.transpose(0, 1))
                return data.jagged_like(out_data)

        if is_flat:
            data = JaggedTensor(data)

        if self._backend == ConvPackBackend.HALO:
            result = self._execute_halo(data, weights)
        elif self._backend == ConvPackBackend.DENSE:
            result = self._execute_dense(data, weights)
        else:
            if self._transposed:
                result = JaggedTensor(impl=self._pack_info.sparse_transpose_conv_3d(data._impl, weights, self._backend))
            else:
                result = JaggedTensor(impl=self._pack_info.sparse_conv_3d(data._impl, weights, self._backend))

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
        source_grid_impl = self._pack_info.source_grid
        if source_grid_impl.grid_count != 1:
            raise ValueError("Source grid impl must have batch size of 1 for Grid")
        return Grid(impl=source_grid_impl)

    @property
    def source_grid_batch(self) -> GridBatch:
        """
        Return the :class:`fvdb.GridBatch` representing the source domain of the convolution.
        If the plan was created for a single grid, it is returned as a batch of size 1.

        Returns:
            source_grid_batch (GridBatch): The source :class:`fvdb.GridBatch` of the convolution plan.
        """

        return GridBatch(impl=self._pack_info.source_grid)

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
        target_grid_impl = self._pack_info.target_grid
        if target_grid_impl.grid_count != 1:
            raise ValueError("Target grid impl must have batch size of 1 for Grid")
        return Grid(impl=target_grid_impl)

    @property
    def target_grid_batch(self) -> GridBatch:
        """
        Return the :class:`fvdb.GridBatch` representing the target domain of the convolution.
        If the plan was created for a single grid, it is returned as a batch of size 1.

        Returns:
            target_grid_batch (GridBatch): The target :class:`fvdb.GridBatch` of the convolution plan.
        """
        return GridBatch(impl=self._pack_info.target_grid)

    @property
    def has_fixed_topology(self) -> bool:
        """
        Returns ``True`` if the source and target grids have the same topology,
        meaning the same voxel structure. This is often required by certain backends.

        Returns:
            has_fixed_topology (bool): ``True`` if source and target grids are the same topology, ``False`` otherwise.
        """
        return self._pack_info.source_grid.is_same(self._pack_info.target_grid)

    # ============================================================
    #                 Private methods
    # ============================================================

    @staticmethod
    def _configure_backend(
        pack_info: SparseConvPackInfoCpp,
        channel_pairs: tuple[tuple[int, int], ...],
        transposed: bool,
        expert_config: dict[str, Any],
    ) -> ConvPackBackend:
        """
        Configures the pack_info in place, building whatever backend structure was asked for. Returns the backend to
        """

        for channel_pair in channel_pairs:
            if len(channel_pair) != 2 or channel_pair[0] <= 0 or channel_pair[1] <= 0:
                raise ValueError("channel_pair must be a tuple of two positive integers")

        backend = expert_config.get("backend", "default")
        allow_tf32 = expert_config.get("allow_tf32", True)
        weight_dtypes = expert_config.get("weight_dtypes", _DEFAULT_DTYPES)
        feature_dtypes = expert_config.get("feature_dtypes", _DEFAULT_DTYPES)
        is_cuda = pack_info.source_grid.device.type == "cuda"

        kernel_size = to_Vec3i(pack_info.kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(pack_info.stride, value_constraint=ValueConstraint.POSITIVE)

        all_dtypes = set(weight_dtypes + feature_dtypes)

        sm_arch = (
            0.0 if not is_cuda else torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10
        )

        # tf32 requires compute capability >= 8.0 (Ampere)
        if allow_tf32 and is_cuda:
            if sm_arch < 8:
                raise ValueError("TF32 requires GPU with compute capability >= 8.0")

        # bf16 requires compute capability >= 8.0 (Ampere)
        if is_cuda and torch.bfloat16 in all_dtypes:
            if sm_arch < 8:
                raise ValueError("BF16 requires GPU with compute capability >= 8.0")

        # float16 requires compute capability >= 7.5 (Turing)
        if is_cuda and torch.float16 in all_dtypes:
            if sm_arch < 7.5:
                raise ValueError("FP16 requires GPU with compute capability >= 7.5")

        if backend == "default":
            if (not is_cuda) or torch.float64 in all_dtypes:
                backend = "legacy"
            else:
                backend = "igemm_mode1"

        # -------------------------------------------------------------------------------------------
        # Choose the actual backend
        # -------------------------------------------------------------------------------------------
        if _vec_is_all(stride, 1) and _vec_is_all(kernel_size, 1):
            return ConvPackBackend.MATMUL

        elif backend == "halo":
            if not is_cuda:
                raise ValueError("Halo backend requires GPU")

            if sm_arch < 8:
                raise ValueError("Halo backend requires GPU with compute capability >= 8.0")

            if not _vec_is_all(stride, 1) or not _vec_is_all(kernel_size, 3):
                raise ValueError("Halo backend requires stride 1 and kernel_size 3.")

            if not pack_info.source_grid.is_same(pack_info.target_grid):
                raise ValueError("Halo backend requires source_grid and target_grid to be the same.")

            if transposed:
                raise ValueError("Halo backend does not support transposed convolution.")

            return ConvPackBackend.HALO

        elif backend == "dense":
            if not _vec_is_all(stride, 1):
                raise ValueError("Dense backend requires stride 1.")

            if not pack_info.source_grid.is_same(pack_info.target_grid):
                raise ValueError("Dense backend requires source_grid and target_grid to be the same.")

            return ConvPackBackend.DENSE

        elif backend == "cutlass":
            if not is_cuda:
                raise ValueError("Cutlass backend requires GPU")

            if sm_arch < 8:
                raise ValueError("Cutlass backend requires GPU with compute capability >= 8.0")

            if transposed:
                raise ValueError("Cutlass backend does not support transposed convolution.")

            if len(channel_pairs) == 0:
                raise ValueError("Cutlass backend requires channel_pairs to be non-empty")

            for channel_pair in channel_pairs:
                if channel_pair not in _CUTLASS_SUPPORTED_CHANNELS:
                    raise ValueError(f"Cutlass backend does not support {channel_pair} convolution.")

            pack_info.build_cutlass(benchmark=False)
            return ConvPackBackend.CUTLASS

        elif backend == "lggs":
            if not is_cuda:
                raise ValueError("LGGS backend requires GPU")

            if sm_arch < 8:
                raise ValueError("LGGS backend requires GPU with compute capability >= 8.0")

            if channel_pairs != [(128, 128)]:
                raise ValueError("LGGS backend only supports 128 to 128 convolution.")

            if transposed:
                raise ValueError("LGGS backend does not support transposed convolution.")

            if not _vec_is_all(kernel_size, 3):
                raise ValueError("LGGS backend requires kernel_size 3.")

            pack_info.build_lggs()
            return ConvPackBackend.LGGS

        elif backend == "legacy":
            pack_info.build_gather_scatter(False)
            return ConvPackBackend.GATHER_SCATTER

        elif backend == "me":
            pack_info.build_gather_scatter(True)
            return ConvPackBackend.GATHER_SCATTER

        elif backend == "igemm_mode0":
            if torch.float64 in all_dtypes:
                raise ValueError("IGEMM backend does not support float64.")
            # TODO(chorvath): training has to be set to True because we can't change it later.
            # This is a bug, issue #9.
            pack_info.build_implicit_gemm(
                sorted=False,
                split_mask_num=1,
                training=True,
                split_mask_num_bwd=3,
                use_tf32=allow_tf32,
            )
            return ConvPackBackend.IGEMM

        elif backend == "igemm_mode1":
            if torch.float64 in all_dtypes:
                raise ValueError("IGEMM backend does not support float64.")
            # TODO(chorvath): training has to be set to True because we can't change it later.
            # This is a bug, issue #9.
            pack_info.build_implicit_gemm(
                sorted=True,
                split_mask_num=1,
                training=True,
                split_mask_num_bwd=3,
                use_tf32=allow_tf32,
            )
            return ConvPackBackend.IGEMM

        elif backend == "igemm_mode2":
            if torch.float64 in all_dtypes:
                raise ValueError("IGEMM backend does not support float64.")
            # TODO(chorvath): training has to be set to True because we can't change it later.
            # This is a bug, issue #9.
            pack_info.build_implicit_gemm(
                sorted=True,
                split_mask_num=3,
                training=True,
                split_mask_num_bwd=3,
                use_tf32=allow_tf32,
            )
            return ConvPackBackend.IGEMM

        else:
            raise NotImplementedError(f"Backend {backend} is not supported")

    def _execute_halo(self, data: JaggedTensor | torch.Tensor, weights: torch.Tensor) -> JaggedTensor:
        assert not self._transposed, "Halo backend does not support transposed convolution."

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        return JaggedTensor(impl=self._pack_info.source_grid.sparse_conv_halo(data._impl, weights, 8))

    def _execute_dense(self, data: JaggedTensor | torch.Tensor, weights: torch.Tensor) -> JaggedTensor:
        source_grid = self._pack_info.source_grid
        target_grid = self._pack_info.target_grid
        assert source_grid.is_same(target_grid)

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        min_coord = source_grid.ijk.jdata.min(dim=0).values
        # BXYZC -> BCXYZ
        dense_feature = source_grid.write_to_dense_cmajor(data._impl, min_coord=min_coord)
        if self._transposed:
            dense_feature = torch.nn.functional.conv_transpose3d(dense_feature, weights, padding=1, stride=1)
        else:
            dense_feature = torch.nn.functional.conv3d(dense_feature, weights, padding=1, stride=1)
        # BCXYZ -> BXYZC
        dense_feature = dense_feature.contiguous()
        return JaggedTensor(impl=source_grid.read_from_dense_cmajor(dense_feature, dense_origins=min_coord))


# These tests are to validate that the type-checking is happy. They won't actually run because
# the grid genenration is nonsense.


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
