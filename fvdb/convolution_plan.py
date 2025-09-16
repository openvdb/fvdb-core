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
from fvdb.types import JaggedTensorOrTensor, NumericMaxRank1, ValueConstraint, to_Vec3i

from fvdb import Grid, GridBatch, JaggedTensor

from ._Cpp import ConvPackBackend
from ._Cpp import SparseConvPackInfo as SparseConvPackInfoCpp

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


def _vec_is_all(v: torch.Tensor, i: int | float) -> bool:
    return bool(torch.all(torch.eq(v, i)).item())


@dataclass(frozen=True)
class ConvolutionPlan:
    """
    A pre-configured plan for efficient sparse 3D convolution operations on fVDB grids.

    ConvolutionPlan encapsulates all the configuration and optimization structures needed
    to perform sparse convolution operations efficiently. Like FFT plans in signal processing
    libraries, a ConvolutionPlan represents a single direction of computation - either
    regular convolution or transposed convolution.

    The plan handles the complex sparse data structures and backend optimizations internally,
    allowing users to focus on the core convolution parameters: input/output channels,
    kernel size, stride, and the grid structure.

    Transposition is treated as just a different kind of kernel, so the inputs and outputs and
    weights are treated the same as if it were a regular convolution. For the default padded case,
    transposed outputs can't automatically infer the target_grid, so it must be provided, unless
    the dense, halo, and lggs backends are used.

    Usage Pattern:
        1. Create a plan using one of the `from_*` class methods
        2. Use the `apply()` method to perform convolutions with different weights
        3. Reuse the same plan for multiple convolutions with the same configuration

    Key Benefits:
        - Pre-computes sparse data structures for optimal performance
        - Automatically selects the best backend (CUTLASS, IGEMM, etc.) for your hardware
        - Supports both single grids and batched operations
        - Handles both regular and transposed convolutions

    Example:
        ```python
        # Create a plan for 3x3x3 convolution with stride 1
        plan = ConvolutionPlan.from_grid(
            channel_pairs=((32, 64), (64, 128)),
            kernel_size=3,
            stride=1,
            source_grid=my_grid
        )

        # Apply convolution with different weights
        features = torch.randn(num_voxels, 32, device="cuda")
        weights = torch.randn(64, 32, 3, 3, 3, device="cuda")
        output = plan.apply(features, weights)
        ```

    Note:
        - Always create plans using the `from_*` class methods, never call `__init__` directly
        - Plans are immutable once created
        - The same plan can be reused for multiple `apply()` calls with different data/weights
        - Channel pairs must be specified at plan creation time for optimal backend selection
    """

    _pack_info: SparseConvPackInfoCpp
    _channel_pairs: tuple[tuple[int, int], ...]
    _transposed: bool
    _expert_config: dict[str, Any]
    _backend: ConvPackBackend

    @classmethod
    def from_grid_batch(
        cls,
        channel_pairs: tuple[tuple[int, int], ...],
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: GridBatch,
        target_grid: GridBatch | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Create a convolution plan for batched grid operations (regular convolution).

        This method creates a plan optimized for processing multiple grids simultaneously,
        which is more efficient than processing individual grids separately when you have
        a batch of data.

        Args:
            channel_pairs: Supported input/output channel combinations as tuples.
                          Each tuple represents (input_channels, output_channels).
                          Example: ((32, 64), (64, 128)) supports 32→64 and 64→128 convolutions.
            kernel_size: Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for (x, y, z) dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid: Input GridBatch containing the sparse voxel structure.
            target_grid: Output GridBatch structure. If None, automatically computed
                based on kernel_size and stride applied to source_grid, except for dense, halo, and
                lggs backends where it uses source_grid. For those backends, target_grid must be None.
            expert_config: Advanced configuration options (rarely needed by typical users).

        Returns:
            ConvolutionPlan: Configured plan ready for apply() operations.

        Example:
            ```python
            # Create plan for 3x3x3 convolution
            plan = ConvolutionPlan.from_grid_batch(
                channel_pairs=((8, 16), (16, 32)),
                kernel_size=3,
                stride=1,
                source_grid=grid_batch
            )

            # Apply to batched data
            batch_data = JaggedTensor(torch.randn(5, 1000, 8, device="cuda"))
            weights = torch.randn(16, 8, 3, 3, 3, device="cuda")
            output = plan.apply(batch_data, weights)
            ```
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend = expert_config.get("backend", "default")
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
        channel_pairs: tuple[tuple[int, int], ...],
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: GridBatch,
        target_grid: GridBatch | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Create a transposed convolution plan for batched grid operations.

        Transposed convolution (also known as deconvolution) is commonly used for
        upsampling operations, such as in decoder networks or generative models.
        It performs the mathematical transpose of the convolution operation.

        Though deconvolution is the "reverse" of convolution in some sense, this configuration
        still treats input and output channels as inputs and outputs, it doesn't swap them.
        The source and target grids are not swapped, it is best to think of deconvolution as
        convolution with a different kernel than convolution, but it is otherwise the same kind
        of abstract operation.

        Args:
            channel_pairs: Supported input/output channel combinations as tuples.
                          Each tuple represents (input_channels, output_channels).
            kernel_size: Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for (x, y, z) dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid: Input GridBatch containing the sparse voxel structure.
            target_grid: Output GridBatch structure. If None, automatically computed
                except for dense backend where it uses source_grid.
            expert_config: Advanced configuration options (rarely needed by typical users).

        Returns:
            ConvolutionPlan: Configured plan ready for transposed convolution operations.

        Note:
            For most backends, target_grid can be automatically computed. Only certain
            expert backends require specific target_grid configurations.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend = expert_config.get("backend", "default")
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
        channel_pairs: tuple[tuple[int, int], ...],
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: Grid,
        target_grid: Grid | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Create a convolution plan for single grid operations (regular convolution).

        This method creates a plan for processing a single grid, which is suitable
        when you have individual grids rather than batched data.

        Args:
            channel_pairs: Supported input/output channel combinations as tuples.
                          Each tuple represents (input_channels, output_channels).
                          Example: ((32, 64), (64, 128)) supports 32→64 and 64→128 convolutions.
            kernel_size: Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for (x, y, z) dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid: Input Grid containing the sparse voxel structure.
            target_grid: Output Grid structure. If None, automatically computed
                based on kernel_size and stride applied to source_grid, except for dense, halo, and
                lggs backends where it uses source_grid. For those backends, target_grid must be None..
            expert_config: Advanced configuration options (rarely needed by typical users).

        Returns:
            ConvolutionPlan: Configured plan ready for apply() operations.

        Example:
            ```python
            # Create a single grid
            grid = Grid.from_zero_voxels(device="cuda", voxel_size=0.1, origin=0)

            # Create plan for 3x3x3 convolution
            plan = ConvolutionPlan.from_grid(
                channel_pairs=((8, 16), (16, 16)),
                kernel_size=3,
                stride=1,
                source_grid=grid
            )

            # Apply to single grid data
            features = torch.randn(100, 8, device="cuda")
            weights = torch.randn(16, 8, 3, 3, 3, device="cuda")
            output = plan.apply(features, weights)
            ```
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend = expert_config.get("backend", "default")
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
        channel_pairs: tuple[tuple[int, int], ...],
        kernel_size: NumericMaxRank1,
        stride: NumericMaxRank1,
        source_grid: Grid,
        target_grid: Grid | None = None,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Create a transposed convolution plan for single grid operations.

        Transposed convolution (also known as deconvolution) is commonly used for
        upsampling operations, such as in decoder networks or generative models.
        It performs the mathematical transpose of the convolution operation.

        Though deconvolution is the "reverse" of convolution in some sense, this configuration
        still treats input and output channels as inputs and outputs, it doesn't swap them.
        The source and target grids are not swapped, it is best to think of deconvolution as
        convolution with a different kernel than convolution, but it is otherwise the same kind
        of abstract operation.

        Args:
            channel_pairs: Supported input/output channel combinations as tuples.
                          Each tuple represents (input_channels, output_channels).
            kernel_size: Size of the convolution kernel. Can be a single int (cubic kernel)
                        or a 3-element sequence for (x, y, z) dimensions.
            stride: Convolution stride. Can be a single int or 3-element sequence.
            source_grid: Input Grid containing the sparse voxel structure.
            target_grid: Output Grid structure. If None, automatically computed
                        except for dense backend where it uses source_grid.
            expert_config: Advanced configuration options (rarely needed by typical users).

        Returns:
            ConvolutionPlan: Configured plan ready for transposed convolution operations.

        Note:
            For most backends, target_grid can be automatically computed. Only certain
            expert backends require specific target_grid configurations.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        backend = expert_config.get("backend", "default")
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
        Create a transposed version of an existing convolution plan.

        This method creates a new plan that performs the transpose operation of the
        given plan. It automatically swaps the source and target grids, reverses
        the channel pairs, and flips the transposed flag.

        Args:
            plan: An existing ConvolutionPlan to transpose.

        Returns:
            ConvolutionPlan: A new plan that performs the transpose of the input plan.

        Example:
            ```python
            # Create forward plan
            forward_plan = ConvolutionPlan.from_grid(
                channel_pairs=((32, 64),),
                kernel_size=3,
                stride=1,
                source_grid=input_grid
            )

            # Create the corresponding backward/transpose plan
            backward_plan = ConvolutionPlan.from_plan_transposed(forward_plan)

            # Now backward_plan has channel_pairs=((64, 32),) and swapped grids
            ```

        Note:
            This is particularly useful for creating encoder-decoder pairs where
            the decoder needs to undo the operations of the encoder.
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

    @overload
    def apply(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolution plan to single grid data.

        Args:
            data: Input features for each voxel in the source grid.
                 Shape: (total_input_voxels, in_channels)
            weights: Convolution kernel weights.
                    Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])

        Returns:
            Output features after convolution.
            Shape: (total_output_voxels, out_channels)
        """

    @overload
    def apply(self, data: JaggedTensor, weights: torch.Tensor) -> JaggedTensor:
        """
        Apply the convolution plan to batched grid data.

        Args:
            data: Input features for each voxel across multiple grids in the batch.
                 Shape: (batch_size, total_input_voxels_per_grid, in_channels)
            weights: Convolution kernel weights.
                    Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])

        Returns:
            Output features after convolution for each grid in the batch.
            Shape: (batch_size, total_output_voxels_per_grid, out_channels)
        """

    def apply(self, data: JaggedTensorOrTensor, weights: torch.Tensor) -> JaggedTensorOrTensor:
        """
        Execute the sparse convolution using this plan's configuration.

        This is the main method for performing convolution operations. It applies
        the convolution kernel to the sparse voxel data according to the plan's
        pre-configured structure and optimizations.

        Args:
            data: Input voxel features. Can be either:
                 - torch.Tensor for single grids: (total_voxels, in_channels)
                 - JaggedTensor for batched grids: (batch_size, voxels_per_grid, in_channels)
            weights: Convolution kernel weights with shape:
                    (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])

        Returns:
            Convolved features with the same type as input:
            - torch.Tensor: (total_output_voxels, out_channels) for single grids
            - JaggedTensor: (batch_size, output_voxels_per_grid, out_channels) for batches

        Raises:
            ValueError: If the channel pair (in_channels, out_channels) from the weights
                       is not supported by this plan's channel_pairs configuration.

        Example:
            ```python
            # Single grid example
            features = torch.randn(1000, 32, device="cuda")  # 1000 voxels, 32 channels
            weights = torch.randn(64, 32, 3, 3, 3, device="cuda")  # 32->64 channels, 3x3x3 kernel
            output = plan.apply(features, weights)  # Shape: (output_voxels, 64)

            # Batched example
            batch_features = JaggedTensor(torch.randn(5, 1000, 32, device="cuda"))
            output = plan.apply(batch_features, weights)  # Shape: (5, output_voxels, 64)
            ```

        Note:
            - The same plan can be reused with different weights and data
            - Channel pairs must match those specified during plan creation
            - The plan automatically handles the sparse structure and backend optimizations
            - For transposed convolution plans, this performs the transpose operation
        """
        out_c = weights.shape[0]
        in_c = weights.shape[1]
        if (in_c, out_c) not in self._channel_pairs:
            raise ValueError(f"Channel pair {in_c, out_c} is not supported")

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
            result = self._apply_halo(data, weights)
        elif self._backend == ConvPackBackend.DENSE:
            result = self._apply_dense(data, weights)
        else:
            if self._transposed:
                result = self._pack_info.sparse_transpose_conv_3d(data, weights, self._backend)
            else:
                result = self._pack_info.sparse_conv_3d(data, weights, self._backend)

        if is_flat:
            return result.jdata
        else:
            return result

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

        if len(channel_pairs) == 0:
            raise ValueError("channel_pairs must be non-empty")

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
                sorted=False, split_mask_num=1, training=True, split_mask_num_bwd=3, use_tf32=allow_tf32
            )
            return ConvPackBackend.IGEMM

        elif backend == "igemm_mode1":
            if torch.float64 in all_dtypes:
                raise ValueError("IGEMM backend does not support float64.")
            # TODO(chorvath): training has to be set to True because we can't change it later.
            # This is a bug, issue #9.
            pack_info.build_implicit_gemm(
                sorted=True, split_mask_num=1, training=True, split_mask_num_bwd=3, use_tf32=allow_tf32
            )
            return ConvPackBackend.IGEMM

        elif backend == "igemm_mode2":
            if torch.float64 in all_dtypes:
                raise ValueError("IGEMM backend does not support float64.")
            # TODO(chorvath): training has to be set to True because we can't change it later.
            # This is a bug, issue #9.
            pack_info.build_implicit_gemm(
                sorted=True, split_mask_num=3, training=True, split_mask_num_bwd=3, use_tf32=allow_tf32
            )
            return ConvPackBackend.IGEMM

        else:
            raise NotImplementedError(f"Backend {backend} is not supported")

    def _apply_halo(self, data: JaggedTensorOrTensor, weights: torch.Tensor) -> JaggedTensor:
        assert not self._transposed, "Halo backend does not support transposed convolution."

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        return self._pack_info.source_grid.sparse_conv_halo(data, weights, 8)

    def _apply_dense(self, data: JaggedTensorOrTensor, weights: torch.Tensor) -> JaggedTensor:
        source_grid = self._pack_info.source_grid
        target_grid = self._pack_info.target_grid
        assert source_grid.is_same(target_grid)

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        min_coord = source_grid.ijk.jdata.min(dim=0).values
        # BWHDC -> BCDHW
        dense_feature = source_grid.write_to_dense_czyx(data, min_coord=min_coord)
        if self._transposed:
            dense_feature = torch.nn.functional.conv_transpose3d(dense_feature, weights, padding=1, stride=1)
        else:
            dense_feature = torch.nn.functional.conv3d(dense_feature, weights, padding=1, stride=1)
        # BCDHW -> BWHDC
        dense_feature = dense_feature.contiguous()
        return source_grid.read_from_dense_czyx(dense_feature, dense_origins=min_coord)


# These tests are to validate that the type-checking is happy. They won't actually run because
# the grid genenration is nonsense.


def _grid_test_for_typing():
    voxel_size = 0.1
    origin = 0

    grid = Grid.from_zero_voxels(device="cuda", voxel_size=voxel_size, origin=origin)

    plan = ConvolutionPlan.from_grid(channel_pairs=((8, 16), (16, 16)), kernel_size=3, stride=1, source_grid=grid)
    plan_t = ConvolutionPlan.from_plan_transposed(plan)

    weights_1 = torch.randn(16, 8, 3, 3, 3, device="cuda")
    weights_2 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_3 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_4 = torch.randn(8, 16, 3, 3, 3, device="cuda")

    data_1 = torch.randn(100, 8, device="cuda")

    out_1: torch.Tensor = plan.apply(data_1, weights_1)
    out_2: torch.Tensor = plan.apply(out_1, weights_2)

    out_3: torch.Tensor = plan_t.apply(out_2, weights_3)
    out_4: torch.Tensor = plan_t.apply(out_3, weights_4)


def _grid_batch_test_for_typing():
    batch_size = 5
    voxel_sizes = [0.1] * batch_size
    origins = [0] * batch_size

    grid_batch = GridBatch.from_zero_voxels(device="cuda", voxel_sizes=voxel_sizes, origins=origins)

    plan = ConvolutionPlan.from_grid_batch(
        channel_pairs=((8, 16), (16, 16)), kernel_size=3, stride=1, source_grid=grid_batch
    )
    plan_t = ConvolutionPlan.from_plan_transposed(plan)

    weights_1 = torch.randn(16, 8, 3, 3, 3, device="cuda")
    weights_2 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_3 = torch.randn(16, 16, 3, 3, 3, device="cuda")
    weights_4 = torch.randn(8, 16, 3, 3, 3, device="cuda")

    data_1 = torch.randn(batch_size, 100, 8, device="cuda")

    out_1: torch.Tensor = plan.apply(data_1, weights_1)
    out_2: torch.Tensor = plan.apply(out_1, weights_2)

    out_3: torch.Tensor = plan_t.apply(out_2, weights_3)
    out_4: torch.Tensor = plan_t.apply(out_3, weights_4)
