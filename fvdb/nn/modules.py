# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
from fvdb.types import (
    NumericMaxRank1,
    NumericMaxRank2,
    ValueConstraint,
    to_Vec3i,
    to_Vec3iBroadcastable,
)
from torch.profiler import record_function

import fvdb
from fvdb import ConvolutionPlan, Grid, GridBatch, JaggedTensor


def fvnn_module(module):
    # Register class as a module in fvdb.nn
    old_forward = module.forward

    def _forward(self, *args, **kwargs):
        with record_function(repr(self)):
            return old_forward(self, *args, **kwargs)

    module.forward = _forward
    return module


@fvnn_module
class AvgPool(nn.Module):
    """
    Applies a 3D average pooling over an input :class:`JaggedTensor` of features
    associated with a :class:`fvdb.GridBatch`.

    Args:
        kernel_size (NumericMaxRank1): the size of the window to take the average over, broadcastable to (3,)
        stride (NumericMaxRank1, optional): the stride of the window. Default value is :attr:`kernel_size`

    .. note::

        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self._kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self._stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    @property
    def kernel_size(self) -> torch.Tensor:
        """
        The size of the window (in voxels) to take the average over.

        Returns:
            kernel_size (torch.Tensor): The kernel size as a ``(3,)``-shaped :class:`torch.Tensor`.
        """
        return self._kernel_size

    @property
    def stride(self) -> torch.Tensor:
        """
        The stride of the window (in voxels) to take the average over.

        Returns:
            stride (torch.Tensor): The stride as a ``(3,)``-shaped :class:`torch.Tensor`.
        """
        return self._stride

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        """
        Apply 3D average pooling to the input :class:`JaggedTensor` of ``fine_data`` associated with
        the :class:`fvdb.GridBatch` ``fine_grid``. If ``coarse_grid`` is provided, it will be used
        to define the output grid structure; otherwise, a new coarsened grid will be created.

        Args:
            fine_data (JaggedTensor): Input features associated with ``fine_grid``.
            fine_grid (GridBatch): The fine grid batch corresponding to ``fine_data``.
            coarse_grid (GridBatch, optional): An optional coarse grid batch to define the output structure.

        Returns:
            pooled_data (JaggedTensor): The pooled features associated with the coarse grid.
            coarse_grid (GridBatch): The coarse :class:`fvdb.GridBatch` after pooling.
        """
        return fine_grid.avg_pool(self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid)


@fvnn_module
class MaxPool(nn.Module):
    """
    Applies a 3D max pooling over an input :class:`JaggedTensor` of features
    associated with a :class:`fvdb.GridBatch`.

    Args:
        kernel_size (NumericMaxRank1): the size of the window to take the max over, broadcastable to (3,)
        stride (NumericMaxRank1, optional): the stride of the window. Default value is :attr:`kernel_size`

    .. note::

        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self._kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self._stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    @property
    def kernel_size(self) -> torch.Tensor:
        """
        The size of the window (in voxels) to take the max over.

        Returns:
            kernel_size (torch.Tensor): The kernel size as a ``(3,)``-shaped :class:`torch.Tensor`.
        """
        return self._kernel_size

    @property
    def stride(self) -> torch.Tensor:
        """
        The stride of the window (in voxels) to take the max over.

        Returns:
            stride (torch.Tensor): The stride as a ``(3,)``-shaped :class:`torch.Tensor`.
        """
        return self._stride

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        """
        Apply 3D max pooling to the input :class:`JaggedTensor` of ``fine_data`` associated with
        the :class:`fvdb.GridBatch` ``fine_grid``. If ``coarse_grid`` is provided, it will be used
        to define the output grid structure; otherwise, a new coarsened grid will be created.

        Args:
            fine_data (JaggedTensor): Input features associated with ``fine_grid``.
            fine_grid (GridBatch): The fine grid batch corresponding to ``fine_data``.
            coarse_grid (GridBatch, optional): An optional coarse grid batch to define the output structure.

        Returns:
            pooled_data (JaggedTensor): The pooled features associated with the coarse grid.
            coarse_grid (GridBatch): The coarse :class:`fvdb.GridBatch` after pooling.
        """
        new_coarse_data, new_coarse_grid = fine_grid.max_pool(
            self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid
        )

        # TODO(chorvath): If this is desired behavior, build into GridBatch directly.
        new_coarse_data.jdata[torch.isinf(new_coarse_data.jdata)] = 0.0

        return new_coarse_data, new_coarse_grid


@fvnn_module
class UpsamplingNearest(nn.Module):
    """
    Refines a :class:`JaggedTensor` of features associated with a coarse :class:`fvdb.GridBatch`
    to a fine :class:`GridBatch` using nearest-neighbor upsampling.
    *i.e.* each voxel in the coarse grid expands to a cube of voxels in the fine grid.

    Args:
        scale_factor (NumericMaxRank1): the upsampling factor, broadcastable to (3,)
    """

    def __init__(self, scale_factor: NumericMaxRank1):
        super().__init__()
        self._scale_factor = to_Vec3iBroadcastable(scale_factor, value_constraint=ValueConstraint.POSITIVE)

    @property
    def scale_factor(self) -> torch.Tensor:
        """
        The upsampling factor (in voxels) for each dimension.

        Returns:
            scale_factor (torch.Tensor): The scale factor as a ``(3,)``-shaped :class:`torch.Tensor`.
        """
        return self._scale_factor

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"

    def forward(
        self,
        coarse_data: JaggedTensor,
        coarse_grid: GridBatch,
        mask: JaggedTensor | None = None,
        fine_grid: GridBatch | None = None,
    ) -> tuple[JaggedTensor, GridBatch]:
        """
        Apply nearest-neighbor upsampling to the input :class:`JaggedTensor` of ``coarse_data`` associated with
        the :class:`fvdb.GridBatch` ``coarse_grid``. If ``fine_grid`` is provided, it will be used
        to define the output grid structure; otherwise, a new refined grid will be created.

        Args:
            coarse_data (JaggedTensor): Input features associated with ``coarse_grid``.
            coarse_grid (GridBatch): The coarse grid batch corresponding to ``coarse_data``.
            mask (JaggedTensor, optional): An optional mask :class:`JaggedTensor` associated with ``fine_grid``.
                If provided, only voxels where the mask is non-zero will be populated in the output.
            fine_grid (GridBatch, optional): An optional fine grid batch to define the output structure.
        Returns:
            refined_data (JaggedTensor): The refined features associated with the fine grid.
            fine_grid (GridBatch): The fine :class:`fvdb.GridBatch` after upsampling.
        """
        return coarse_grid.refine(self.scale_factor, coarse_data, mask, fine_grid=fine_grid)


class _SparseConv3dBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: NumericMaxRank1 = 3,
        stride: NumericMaxRank1 = 1,
        bias: bool = True,
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = to_Vec3i(stride, value_constraint=ValueConstraint.POSITIVE)

        self.kernel_volume: int = int(torch.prod(self.kernel_size).item())
        if self.kernel_volume > 1:
            # Weight tensor is of shape (Do, Di, K0, K1, K2), but the underlying data is (K2, K1, K0, Di, Do)
            #   so we don't need to make a copy of the permuted tensor within the conv kernel.
            weight_shape = [out_channels, in_channels] + self.kernel_size.tolist()
            weight = torch.zeros(*weight_shape[::-1]).permute(4, 3, 2, 1, 0)
            self.weight = nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"
        if self.bias is None:
            s += ", bias=False"
        return s

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(self.in_channels * self.kernel_volume)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)


@fvnn_module
class SparseConv3d(_SparseConv3dBase):
    """
    A sparse 3D convolution module that operates on :class:`JaggedTensor` inputs
    according to a provided :class:`ConvolutionPlan`.

    A :class:`ConvolutionPlan` defines the mapping of a sparse convolution operation
    between data associated with an input :class:`fvdb.GridBatch` and an output
    :class:`fvdb.GridBatch`. This allows for efficient sparse convolution operations
    without explicitly constructing dense tensors.

    Args:
        in_channels (int): Number of channels in the input :class:`JaggedTensor`.
        out_channels (int): Number of channels in the output :class:`JaggedTensor`.
        kernel_size (NumericMaxRank1, optional): Size of the convolution kernel, broadcastable to ``(3,)``. Default: 3
        stride (NumericMaxRank1, optional): Stride of the convolution, broadcastable to ``(3,)``. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: ``True``
    """

    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
        """
        Apply the sparse convolution to the input :class:`JaggedTensor` using the provided :class:`ConvolutionPlan`.

        Args:
            data (JaggedTensor): Input features to be convolved.
            plan (ConvolutionPlan): The convolution plan defining the mapping between input and output grids.

        Returns:
            JaggedTensor: The result of the sparse convolution.
        """
        if not plan.valid_usage(self.in_channels, self.out_channels, self.kernel_size, self.stride, transposed=False):
            raise ValueError(
                "Convolution plan used with a SparseConv3d module that had "
                "mismatched input/output channels, kernel size, or stride, or transposition"
            )

        out_data = plan.execute(data, self.weight)

        if self.bias is not None:
            out_data.jdata = out_data.jdata + self.bias

        return out_data


@fvnn_module
class SparseConvTranspose3d(_SparseConv3dBase):
    """
    A sparse 3D transposed convolution module that operates on :class:`JaggedTensor` inputs
    according to a provided :class:`ConvolutionPlan`.

    A :class:`ConvolutionPlan` defines the mapping of a sparse convolution operation
    between data associated with an input :class:`fvdb.GridBatch` and an output
    :class:`fvdb.GridBatch`. This allows for efficient sparse convolution operations
    without explicitly constructing dense tensors.

    Args:
        in_channels (int): Number of channels in the input :class:`JaggedTensor`.
        out_channels (int): Number of channels in the output :class:`JaggedTensor`.
        kernel_size (NumericMaxRank1, optional): Size of the convolution kernel, broadcastable to ``(3,)``. Default: 3
        stride (NumericMaxRank1, optional): Stride of the convolution, broadcastable to ``(3,)``. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: ``True``
    """

    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
        """
        Apply the sparse transposed convolution to the input :class:`JaggedTensor` using the provided :class:`ConvolutionPlan`.

        Args:
            data (JaggedTensor): Input features to be convolved.
            plan (ConvolutionPlan): The convolution plan defining the mapping between input and output grids.

        Returns:
            JaggedTensor: The result of the sparse transposed convolution.
        """
        if not plan.valid_usage(self.in_channels, self.out_channels, self.kernel_size, self.stride, transposed=True):
            raise ValueError(
                "Convolution plan used with a SparseConvTranspose3d module that had "
                "mismatched input/output channels, kernel size, or stride, or transposition"
            )

        out_data = plan.execute(data, self.weight)

        if self.bias is not None:
            out_data.jdata = out_data.jdata + self.bias

        return out_data


@fvnn_module
class GroupNorm(nn.GroupNorm):
    """
    Applies Group Normalization over a :class:`JaggedTensor` batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.GroupNorm` for detailed information on Group Normalization.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels in the input :class:`JaggedTensor`
        eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool, optional): a boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``
        device (torch.device, optional): device on which the module is allocated. Default: ``None``
        dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        """
        Apply Group Normalization to the input :class:`JaggedTensor` using the provided :class:`GridBatch`.

        Args:
            data (JaggedTensor): Input features to be normalized.
            grid (GridBatch): The grid batch corresponding to ``data``.

        Returns:
            JaggedTensor: The result of the group normalization.
        """
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"
        num_batches = grid.grid_count

        flat_data, flat_offsets = data.jdata, data.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[b] : flat_offsets[b + 1]]
            if feat.size(0) != 0:
                feat = feat.transpose(0, 1).contiguous().reshape(1, num_channels, -1)
                feat = super().forward(feat)
                feat = feat.reshape(num_channels, -1).transpose(0, 1)

                result_data[flat_offsets[b] : flat_offsets[b + 1]] = feat

        return grid.jagged_like(result_data)


@fvnn_module
class BatchNorm(nn.BatchNorm1d):
    """
    Applies Batch Normalization over a :class:`JaggedTensor` batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.BatchNorm1d` for detailed information on Batch Normalization.

    Args:
        num_features (int): number of features in the input :class:`JaggedTensor`
        eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float, optional): the value used for the running_mean and running_var computation. Default: 0.1
        affine (bool, optional): a boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this module
            tracks the running mean and variance, and when set to ``False``, this module does not
            track such statistics and always uses batch statistics in both training and eval modes.
            Default: ``True``
        device (torch.device, optional): device on which the module is allocated. Default: ``None``
        dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        """
        Apply Batch Normalization to the input :class:`JaggedTensor` using the provided :class:`GridBatch`.

        Args:
            data (JaggedTensor): Input features to be normalized.
            grid (GridBatch): The grid batch corresponding to ``data``.

        Returns:
            JaggedTensor: The result of the batch normalization.
        """
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)


@fvnn_module
class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies distributed Batch Normalization over a :class:`JaggedTensor` batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.SyncBatchNorm` for detailed information on distributed batch normalization.

    .. note::
        Only supports :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
        :meth:`fvdb.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
        :attr:`BatchNorm` layer to :class:`SyncBatchNorm` before wrapping
        Network with DDP.

    Args:
        num_features (int): number of features in the input :class:`JaggedTensor`
        eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float, optional): the value used for the running_mean and running_var computation. Default: 0.1
        affine (bool, optional): a boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this module
            tracks the running mean and variance, and when set to ``False``, this module does not
            track such statistics and always uses batch statistics in both training and eval modes.
            Default: ``True``
        process_group (Any, optional): the process group to scope synchronization. Default: ``None``
        device (torch.device, optional): device on which the module is allocated. Default: ``None``
        dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        """
        Apply Synchronized Batch Normalization to the input :class:`JaggedTensor` using the provided :class:`GridBatch`.

        Args:
            data (JaggedTensor): Input features to be normalized.
            grid (GridBatch): The grid batch corresponding to ``data``.

        Returns:
            JaggedTensor: The result of the synchronized batch normalization.
        """
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)

    @classmethod
    def convert_sync_batchnorm(cls, module: nn.Module, process_group: Any = None) -> nn.Module:
        """
        Helper function to convert :attr:`fvdb.nn.BatchNorm` layer in the model to :attr:`fvdb.nn.SyncBatchNorm` layer.

        Args:
            module: Module for which all :attr:`fvdb.nn.BatchNorm` layers will be converted to
                :attr:`fvdb.nn.SyncBatchNorm` layers.
            process_group: process group to scope synchronization, default is the whole world.

        Returns:
            torch.nn.Module:  The original module with the converted :attr:`fvdb.nn.SyncBatchNorm` layers.

        Example::

            >>> # Network with fvdb.nn.SyncBatchNorm layer
            >>> module = fvdb.nn.Sequential(
            >>>            fvdb.nn.Linear(20, 100),
            >>>            fvdb.nn.BatchNorm(100)
            >>>          )
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = fvdb.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, BatchNorm):
            module_output = cls(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
