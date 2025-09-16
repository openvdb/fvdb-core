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
    Encapsulation of configuration structures for sparse convolution using
    fVDB Grid and GridBatch.
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
        Convolution plan over grid batch for source and target grids, non-transposed.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if target_grid is None:
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
        target_grid: GridBatch,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Convolution plan over grid batch for source and target grids, transposed.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

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
        Convolution plan over grid for source and target grids, non-transposed.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if target_grid is None:
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
        target_grid: Grid,
        *,
        expert_config: dict[str, Any] = _DEFAULT_CONFIG,
    ) -> "ConvolutionPlan":
        """
        Convolution plan over grid for source and target grids, transposed.
        """
        kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        if target_grid is None:
            target_grid = source_grid.conv_grid(kernel_size, stride)

        pack_info = SparseConvPackInfoCpp(kernel_size, stride, source_grid._impl, target_grid._impl)

        transposed = True
        backend = cls._configure_backend(pack_info, channel_pairs, transposed, expert_config)
        return cls(pack_info, channel_pairs, transposed, expert_config, backend)

    @classmethod
    def from_plan_transposed(cls, plan: "ConvolutionPlan") -> "ConvolutionPlan":
        """
        Returns the plan which is the transpose of the given plan.
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

        if backend == "halo":
            if not is_cuda:
                raise ValueError("Halo backend requires GPU")

            if sm_arch < 8:
                raise ValueError("Halo backend requires GPU with compute capability >= 8.0")

            if not _vec_is_all(stride, 1) or not _vec_is_all(kernel_size, 3):
                raise ValueError("Halo backend requires stride 1 and kernel_size 3.")

            if not pack_info.source_grid.is_same(pack_info.target_grid):
                raise ValueError("Halo backend requires source_grid and target_grid to be the same.")

            return ConvPackBackend.HALO

        elif backend == "dense":
            if not _vec_is_all(stride, 1):
                raise ValueError("Dense backend requires stride 1.")

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

    @overload
    def apply(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolution plan to the data, assuming a batch size of 1.
        Args:
            data (torch.Tensor): Input features for each voxel.
                Shape: (total_input_voxels, in_channels).
            weights (torch.Tensor): Convolution weights.
                Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).

        Returns:
            torch.Tensor: Output features after convolution.
                Shape: (total_output_voxels, out_channels).
        """

    @overload
    def apply(self, data: JaggedTensor, weights: torch.Tensor) -> JaggedTensor:
        """
        Apply the convolution plan to the data, batched.
        Args:
            data (JaggedTensor): Input features for each voxel.
                Shape: (batch_size, total_input_voxels, in_channels).
            weights (torch.Tensor): Convolution weights.
                Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).

        Returns:
            JaggedTensor: Output features after convolution.
                Shape: (batch_size, total_output_voxels, out_channels).
        """

    def apply(self, data: JaggedTensorOrTensor, weights: torch.Tensor) -> JaggedTensorOrTensor:
        out_c = weights.shape[0]
        in_c = weights.shape[1]
        if (in_c, out_c) not in self._channel_pairs:
            raise ValueError(f"Channel pair {in_c, out_c} is not supported")

        is_flat: bool = isinstance(data, torch.Tensor)

        if is_flat:
            if self._pack_info.source_grid.grid_count != 1:
                raise ValueError("Source grid must have batch size of 1 for flat data")
            data = JaggedTensor(data)

        if self._transposed:
            result = self._pack_info.sparse_transpose_conv_3d(data, weights, self._backend)
        else:
            result = self._pack_info.sparse_conv_3d(data, weights, self._backend)

        if is_flat:
            return result.jdata
        else:
            return result


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
