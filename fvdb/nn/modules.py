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


# ------------------------------------------------------------------------------------------------


@fvnn_module
class AvgPool(nn.Module):
    r"""Applies a 3D average pooling over an input signal.

    Args:
        kernel_size: the size of the window to take average over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self.kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        return fine_grid.avg_pool(self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid)


# ------------------------------------------------------------------------------------------------


@fvnn_module
class MaxPool(nn.Module):
    r"""Applies a 3D max pooling over an input signal.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: NumericMaxRank1, stride: NumericMaxRank1 | None = None):
        super().__init__()
        self.kernel_size = to_Vec3iBroadcastable(kernel_size, value_constraint=ValueConstraint.POSITIVE)
        self.stride = (
            to_Vec3iBroadcastable(stride, value_constraint=ValueConstraint.POSITIVE) if stride else self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"

    def forward(
        self, fine_data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch | None = None
    ) -> tuple[JaggedTensor, GridBatch]:
        new_coarse_data, new_coarse_grid = fine_grid.max_pool(
            self.kernel_size, fine_data, stride=self.stride, coarse_grid=coarse_grid
        )

        # TODO(chorvath): If this is desired behavior, build into GridBatch directly.
        new_coarse_data.jdata[torch.isinf(new_coarse_data.jdata)] = 0.0

        return new_coarse_data, new_coarse_grid


# ------------------------------------------------------------------------------------------------


@fvnn_module
class UpsamplingNearest(nn.Module):
    r"""Upsamples the input by a given scale factor using nearest upsampling.

    Args:
        scale_factor: the upsampling factor
    """

    def __init__(self, scale_factor: NumericMaxRank1):
        super().__init__()
        self.scale_factor = to_Vec3iBroadcastable(scale_factor, value_constraint=ValueConstraint.POSITIVE)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"

    def forward(
        self,
        coarse_data: JaggedTensor,
        coarse_grid: GridBatch,
        mask: JaggedTensor | None = None,
        fine_grid: GridBatch | None = None,
    ) -> tuple[JaggedTensor, GridBatch]:
        return coarse_grid.refine(self.scale_factor, coarse_data, mask, fine_grid=fine_grid)


# ------------------------------------------------------------------------------------------------


@fvnn_module
class InjectFromGrid(nn.Module):
    r"""
    Inject the content of input vdb-tensor to another grid.

    Args:
        default_value: the default value to inject in the new grid.
    """

    def __init__(self, default_value: float = 0.0) -> None:
        super().__init__()
        self.default_value = default_value

    def extra_repr(self) -> str:
        return f"default_value={self.default_value}"

    def forward(
        self,
        data: JaggedTensor,
        grid: GridBatch,
        other_grid: GridBatch | None = None,
    ) -> tuple[JaggedTensor, GridBatch]:
        return (
            (other_grid.inject_from(grid, data, default_value=self.default_value), other_grid)
            if other_grid
            else (data, grid)
        )


# ------------------------------------------------------------------------------------------------
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
    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
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
    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
        if not plan.valid_usage(self.in_channels, self.out_channels, self.kernel_size, self.stride, transposed=True):
            raise ValueError("Invalid usage of the convolution plan")

        out_data = plan.execute(data, self.weight)

        if self.bias is not None:
            out_data.jdata = out_data.jdata + self.bias

        return out_data


# ------------------------------------------------------------------------------------------------


@fvnn_module
class GroupNorm(nn.GroupNorm):
    r"""Applies Group Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.GroupNorm` for detailed information.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
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


# ------------------------------------------------------------------------------------------------


@fvnn_module
class BatchNorm(nn.BatchNorm1d):
    r"""Applies Batch Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.BatchNorm1d` for detailed information.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)


# ------------------------------------------------------------------------------------------------


@fvnn_module
class SyncBatchNorm(nn.SyncBatchNorm):
    r"""Applies distributed Batch Normalization over a JaggedTensor/GridBatch.
    See :class:`~torch.nn.SyncBatchNorm` for detailed information.

    Only supports :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`fvdb.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.
    """

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        """Layer forward pass.

        Args:
            input: input JaggedTensor/GridBatch.

        Returns:
            Output JaggedTensor/GridBatch with batch norm applied to the feature dimension, across all ranks.
        """
        num_channels = data.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(data.jdata)
        return grid.jagged_like(result_data)

    @classmethod
    def convert_sync_batchnorm(cls, module: nn.Module, process_group: Any = None) -> nn.Module:
        r"""Helper function to convert
        :attr:`fvdb.nn.BatchNorm` layer in the model to
        :attr:`fvdb.nn.SyncBatchNorm` layer.

        Args:
            module: Module for which all :attr:`fvdb.nn.BatchNorm` layers will be converted to
                :attr:`fvdb.nn.SyncBatchNorm` layers.
            process_group: process group to scope synchronization, default is the whole world.

        Returns:
            The original module with the converted :attr:`fvdb.nn.SyncBatchNorm` layers.

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


# ------------------------------------------------------------------------------------------------

# Non-linear Activations


@fvnn_module
class ElementwiseMixin:
    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        res = super().forward(data.jdata)  # type: ignore
        return grid.jagged_like(res)


class ELU(ElementwiseMixin, nn.ELU):
    r"""
    Applies the Exponential Linear Unit function element-wise:
    .. math::
    \text{ELU}(x) = \begin{cases}
    x, & \text{ if } x > 0\\
    \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
    \end{cases}
    """


class CELU(ElementwiseMixin, nn.CELU):
    r"""
    Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    """


class GELU(ElementwiseMixin, nn.GELU):
    r"""
    Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    """


class Linear(ElementwiseMixin, nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    """


class ReLU(ElementwiseMixin, nn.ReLU):
    r"""
    Applies the rectified linear unit function element-wise: :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """


class LeakyReLU(ElementwiseMixin, nn.LeakyReLU):
    r"""
    Applies the element-wise function: :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    """


class SELU(ElementwiseMixin, nn.SELU):
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = \lambda \left\{
    \begin{array}{lr}
    x, & \text{if } x > 0 \\
    \text{negative\_slope} \times e^x - \text{negative\_slope}, & \text{otherwise }
    \end{array}
    \right.`
    """


class SiLU(ElementwiseMixin, nn.SiLU):
    r"""
    Applies element-wise, :math:`\text{SiLU}(x) = x * \sigma(x)`, where :math:`\sigma(x)` is the sigmoid function.
    """


class Tanh(ElementwiseMixin, nn.Tanh):
    r"""
    Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}`
    """


class Sigmoid(ElementwiseMixin, nn.Sigmoid):
    r"""
    Applies element-wise, :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """


# Dropout Layers


class Dropout(ElementwiseMixin, nn.Dropout):
    r"""
    During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`
    using samples from a Bernoulli distribution. The elements to zero are randomized on every forward call.
    """


# ------------------------------------------------------------------------------------------------
# Sequential
# torch.nn.Sequential is designed around layers which take a single input and produce a single output.
# fvdb layers always take a JaggedTensor as the first argument, but sometimes take a GridBatch
# as the second argument, and sometimes a GridBatch as the third argument.
# Sometimes they return just a JaggedTensor, sometimes a tuple of (JaggedTensor, GridBatch),
# and sometimes a tuple of (JaggedTensor, GridBatch, SparseConvPackInfo).


@fvnn_module
class Sequential(nn.Module):
    r"""A sequential container for fvdb neural network modules.

    This container properly handles the different input/output signatures of fvdb modules:
    - Modules returning JaggedTensor only implicitly use the same GridBatch as input
    - Modules returning (JaggedTensor, GridBatch) update the GridBatch for subsequent layers
    - SparseConvPackInfo is ignored as this Sequential is for convenience when structural changes aren't needed

    Args:
        *args: Variable length argument list of modules to be added to the sequential container.
               Can be modules or an OrderedDict of modules.

    Example::
        >>> # Simple sequential with activation
        >>> seq = fvdb.nn.Sequential(
        ...     fvdb.nn.SparseConv3d(64, 128, 3),
        ...     fvdb.nn.BatchNorm(128),
        ...     fvdb.nn.ReLU()
        ... )
        >>> out_data, out_grid = seq(data, grid)

        >>> # Sequential with pooling (changes grid structure)
        >>> seq = fvdb.nn.Sequential(
        ...     fvdb.nn.SparseConv3d(32, 64, 3),
        ...     fvdb.nn.ReLU(),
        ...     fvdb.nn.MaxPool(2)
        ... )
        >>> out_data, out_grid = seq(data, grid)
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            # Handle OrderedDict or regular dict
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # Handle individual modules passed as arguments
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(dict(list(self.named_children())[idx]))
        else:
            return list(self.children())[idx]

    def __len__(self):
        return len(self._modules)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        """Forward pass through all modules in sequence.

        Args:
            data: Input jagged tensor
            grid: Input grid batch

        Returns:
            Tuple of (output_data, output_grid)
        """
        current_data = data
        current_grid = grid

        for module in self:
            result = module(current_data, current_grid)

            if isinstance(result, tuple):
                if len(result) == 2:
                    # (JaggedTensor, GridBatch)
                    current_data, current_grid = result
                elif len(result) == 3:
                    # (JaggedTensor, GridBatch, SparseConvPackInfo | None)
                    # Ignore SparseConvPackInfo as mentioned in requirements
                    current_data, current_grid, _ = result
                else:
                    raise ValueError(f"Unexpected return tuple length {len(result)} from module {module}")
            else:
                # JaggedTensor only - implicitly uses same GridBatch
                current_data = result
                # current_grid remains unchanged

        return current_data, current_grid
