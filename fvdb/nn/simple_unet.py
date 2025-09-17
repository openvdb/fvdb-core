# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Simple Residual U-Net Architecture for Sparse Voxel Data.

This module implements a U-Net architecture specifically designed for sparse voxel grids
using the fVDB framework. The architecture is a traditional
U-Net design with an encoder-decoder structure, skip connections, and residual blocks.

The network operates on sparse voxel data represented as JaggedTensors and GridBatch
objects, enabling efficient processing of 3D volumetric data with varying sparsity
patterns. The architecture includes:

- Encoder path: Progressive downsampling with channel expansion
- Decoder path: Progressive upsampling with channel reduction
- Skip connections: Feature concatenation between encoder and decoder
- Residual connections: Within convolutional blocks for improved gradient flow
- Adaptive padding: Handles convolution boundary conditions for sparse grids

Components:
- SimpleUNetBasicBlock: Basic convolution-batchnorm-activation block
- SimpleUNetConvBlock: Multi-layer residual block
- SimpleUNetDown/Up: Resolution changing operations with channel adjustment
- SimpleUNetSkipCat: Skip connection concatenation and channel fusion
- SimpleUNetPad/Depad: Boundary handling for sparse convolutions
- SimpleUNetDownUp: Recursive encoder-decoder structure
- SimpleUNet: Main network combining all components

The architecture is designed for tasks requiring dense predictions on sparse 3D data,
such as 3D semantic segmentation, shape completion, or volumetric reconstruction.
"""

import math
from typing import Any, Sequence

import fvdb.nn as fvnn
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

from .modules import fvnn_module


@fvnn_module
class SimpleUNetBasicBlock(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation.

    This is the fundamental building block of the U-Net architecture, consisting
    of a 3D sparse convolution followed by batch normalization and ReLU activation.
    The block maintains the same number of input and output channels and is designed
    to preserve the spatial structure of sparse voxel data.

    Args:
        channels (int): Number of input and output channels.
        kernel_size (NumericMaxRank1): Size of the convolution kernel. Defaults to 3.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: NumericMaxRank1 = 3,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.momentum = momentum

        self.conv = fvnn.SparseConv3d(channels, channels, kernel_size=kernel_size, stride=1, bias=False)
        self.batch_norm = fvnn.BatchNorm(channels, momentum=momentum)
        self.relu = fvnn.ReLU(inplace=True)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}, momentum={self.momentum}"

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
        x = self.conv(data, plan)
        out_grid = plan.target_grid_batch
        x = self.batch_norm(x, out_grid)
        x = self.relu(x, out_grid)
        return x


@fvnn_module
class SimpleUNetConvBlock(nn.Module):
    """
    Multi-layer residual convolutional block.

    Combines multiple SimpleUNetBasicBlocks with a residual connection around
    the entire sequence. The input is added to the output of the block sequence,
    enabling improved gradient flow and training stability. Requires that the
    convolution plan maintains fixed topology to ensure compatible tensor shapes
    for the residual connection.

    Args:
        channels (int): Number of input and output channels.
        kernel_size (NumericMaxRank1): Size of the convolution kernel. Defaults to 3.
        layer_count (int): Number of basic blocks to stack. Defaults to 2.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: NumericMaxRank1 = 3,
        layer_count: int = 2,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.layer_count = layer_count
        self.momentum = momentum

        self.blocks = nn.ModuleList([SimpleUNetBasicBlock(channels, kernel_size, momentum) for _ in range(layer_count)])

        self.final_relu = fvnn.ReLU(inplace=True)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"layer_count={self.layer_count}, momentum={self.momentum}"
        )

    def reset_parameters(self) -> None:
        for block in self.blocks:
            assert isinstance(block, SimpleUNetBasicBlock)
            block.reset_parameters()

    def forward(
        self,
        data: JaggedTensor,
        plan: ConvolutionPlan,
    ) -> JaggedTensor:
        # In order for this to work, the plan's source and target grids must be the same.
        if not plan.has_fixed_topology:
            raise ValueError("Convolution plan must have fixed topology for repeated conv blocks.")

        residual = data

        for block in self.blocks:
            data = block(data, plan)

        data = data + residual
        data = self.final_relu(data, plan.target_grid_batch)

        return data


@fvnn_module
class SimpleUNetDown(nn.Module):
    """
    Downsampling module for the encoder path of the U-Net.

    Reduces spatial resolution by a factor of 2 using max pooling while simultaneously
    increasing the number of channels through a 1x1 convolution. This design follows
    the typical U-Net encoder pattern where spatial information is traded for feature
    depth as the network progresses toward the bottleneck.

    The module performs two operations in sequence:
    1. Max pooling with factor 2 to reduce spatial resolution
    2. 1x1 convolution to adjust channel count

    Args:
        in_channels (int): Number of input channels from the fine grid.
        out_channels (int): Number of output channels for the coarse grid.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.momentum = momentum

        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm = fvnn.BatchNorm(out_channels, momentum=momentum)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, momentum={self.momentum}"

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(self, data: JaggedTensor, fine_grid: GridBatch, coarse_grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=fine_grid, target_grid=coarse_grid)

        # Decrease the resolution by a factor of 2, same channel count
        data = fine_grid.max_pool(pool_factor=2, data=data, coarse_grid=coarse_grid)[0]

        # Increase the channel count at the lower resolution
        data = self.conv(data, plan)
        return self.batch_norm(data, coarse_grid)


@fvnn_module
class SimpleUNetUp(nn.Module):
    """
    Upsampling module for the decoder path of the U-Net.

    Increases spatial resolution by a factor of 2 using subdivision while simultaneously
    decreasing the number of channels through a 1x1 convolution. This design follows
    the typical U-Net decoder pattern where feature depth is traded for spatial
    information as the network progresses toward the output.

    The module performs two operations in sequence:
    1. 1x1 convolution to adjust channel count
    2. Grid subdivision with factor 2 to increase spatial resolution

    Args:
        in_channels (int): Number of input channels from the coarse grid.
        out_channels (int): Number of output channels for the fine grid.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.momentum = momentum

        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm = fvnn.BatchNorm(out_channels, momentum=momentum)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, momentum={self.momentum}"

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(self, data: JaggedTensor, coarse_grid: GridBatch, fine_grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=coarse_grid, target_grid=fine_grid)

        # Decrease the channel count at the lower resolution
        data = self.conv(data, plan)
        data = self.batch_norm(data, coarse_grid)

        # Increase the resolution by a factor of 2
        return coarse_grid.subdivide(subdiv_factor=2, data=data, fine_grid=fine_grid)[0]


@fvnn_module
class SimpleUNetSkipCat(nn.Module):
    """
    Skip connection module that combines encoder and decoder features.

    Implements the characteristic skip connections of U-Net architecture by concatenating
    features from the encoder path (skip_data) with corresponding features from the
    decoder path (lower_data). After concatenation, a 1x1 convolution reduces the
    doubled channel count back to the original number, enabling efficient feature
    fusion while maintaining computational efficiency.

    This module is crucial for preserving fine-grained spatial details that might
    be lost during the downsampling operations in the encoder path.

    Args:
        channels (int): Number of channels for both input tensors and the output.
                       The concatenated tensor will have 2*channels before reduction.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, channels: int, momentum: float = 0.1):
        super().__init__()
        self.channels = channels
        self.momentum = momentum

        self.conv = fvnn.SparseConv3d(channels * 2, channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm = fvnn.BatchNorm(channels, momentum=momentum)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, momentum={self.momentum}"

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(self, skip_data: JaggedTensor, lower_data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        data = fvdb.jcat([skip_data, lower_data], dim=1)
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=grid, target_grid=grid)

        data = self.conv(data, plan)
        data = self.batch_norm(data, grid)
        return data


@fvnn_module
class SimpleUNetBottleneck(nn.Module):
    """
    Bottleneck module at the deepest level of the U-Net architecture.

    Represents the coarsest spatial resolution in the network where the receptive
    field is maximized and the most abstract features are learned. The bottleneck
    consists of a residual convolutional block that processes features at the
    lowest resolution before they begin the upsampling path.

    This module operates at a fixed spatial resolution and channel count,
    serving as the bridge between the encoder and decoder paths.

    Args:
        channels (int): Number of input and output channels.
        kernel_size (NumericMaxRank1): Size of the convolution kernel. Defaults to 3.
        layer_count (int): Number of basic blocks in the residual block. Defaults to 2.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, channels: int, kernel_size: NumericMaxRank1 = 3, layer_count: int = 2, momentum: float = 0.1):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.layer_count = layer_count
        self.momentum = momentum

        self.block = SimpleUNetConvBlock(channels, kernel_size, layer_count, momentum)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"layer_count={self.layer_count}, momentum={self.momentum}"
        )

    def reset_parameters(self) -> None:
        self.block.reset_parameters()

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.block.kernel_size, stride=1, source_grid=grid, target_grid=grid
        )

        return self.block(data, plan)


@fvnn_module
class SimpleUNetDownUp(nn.Module):
    """
    Recursive encoder-decoder module forming the core U-Net structure.

    Implements the complete encoder-decoder architecture with skip connections.
    The module recursively creates nested U-Net structures, where each level
    performs downsampling, processes features at a coarser resolution, upsamples,
    and combines results via skip connections.

    The recursive structure enables the network to capture multi-scale features
    effectively. At each level:
    1. Input convolution processes features at current resolution
    2. Downsampling reduces resolution and increases channels
    3. Inner module (either bottleneck or another DownUp) processes coarser features
    4. Upsampling increases resolution and reduces channels
    5. Skip connection combines encoder and decoder features
    6. Output convolution refines the combined features

    Args:
        in_channels (int): Number of input and output channels.
        channel_growth_rate (int): Factor by which channels increase at each level.
        kernel_size (NumericMaxRank1): Size of convolution kernels. Defaults to 3.
        downup_layer_count (int): Number of recursive levels. Defaults to 4.
        block_layer_count (int): Number of layers per convolutional block. Defaults to 2.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        channel_growth_rate: int,
        kernel_size: NumericMaxRank1 = 3,
        downup_layer_count: int = 4,
        block_layer_count: int = 2,
        momentum: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channel_growth_rate = channel_growth_rate
        self.kernel_size = kernel_size
        self.downup_layer_count = downup_layer_count
        self.block_layer_count = block_layer_count
        self.momentum = momentum

        coarse_channels = in_channels * channel_growth_rate

        self.conv_in = SimpleUNetConvBlock(in_channels, kernel_size, block_layer_count, momentum)
        self.down = SimpleUNetDown(in_channels, coarse_channels, momentum)
        self.inner = (
            SimpleUNetBottleneck(coarse_channels, kernel_size, block_layer_count, momentum)
            if downup_layer_count < 1
            else SimpleUNetDownUp(
                coarse_channels, channel_growth_rate, kernel_size, downup_layer_count - 1, block_layer_count, momentum
            )
        )
        self.up = SimpleUNetUp(coarse_channels, in_channels, momentum)
        self.skip_cat = SimpleUNetSkipCat(in_channels, momentum)
        self.conv_out = SimpleUNetConvBlock(in_channels, kernel_size, block_layer_count, momentum)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, channel_growth_rate={self.channel_growth_rate}, "
            f"kernel_size={self.kernel_size}, downup_layer_count={self.downup_layer_count}, "
            f"block_layer_count={self.block_layer_count}, momentum={self.momentum}"
        )

    def reset_parameters(self) -> None:
        self.conv_in.reset_parameters()
        self.down.reset_parameters()
        self.inner.reset_parameters()
        self.up.reset_parameters()
        self.skip_cat.reset_parameters()
        self.conv_out.reset_parameters()

    def forward(self, data: JaggedTensor, fine_grid: GridBatch) -> JaggedTensor:
        coarse_grid = fine_grid.conv_grid(kernel_size=self.kernel_size, stride=1)
        conv_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.kernel_size, stride=1, source_grid=fine_grid, target_grid=fine_grid
        )

        data = self.conv_in(data, conv_plan)
        skip_data = data
        data = self.down(data, fine_grid, coarse_grid)
        assert isinstance(self.inner, (SimpleUNetBottleneck, SimpleUNetDownUp))
        data = self.inner(data, coarse_grid)
        data = self.up(data, coarse_grid, fine_grid)
        data = self.skip_cat(skip_data, data, fine_grid)
        return self.conv_out(data, conv_plan)


@fvnn_module
class SimpleUNetPad(nn.Module):
    """
    Input padding module for handling convolution boundary conditions.

    Transforms the input grid to accommodate the convolution operations throughout
    the network. The module simultaneously handles two transformations:
    1. Spatial padding: Expands the grid to ensure valid convolutions at boundaries
    2. Channel adjustment: Converts input channels to the base channel count

    The spatial padding is determined by the kernel size and ensures that the
    network can process boundary voxels correctly. This is particularly important
    for sparse voxel data where boundary handling affects the final output quality.

    Args:
        in_channels (int): Number of input channels from the original data.
        out_channels (int): Number of output channels (base channels for the network).
        kernel_size (NumericMaxRank1): Size of convolution kernels. Defaults to 3.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: NumericMaxRank1 = 3, momentum: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.momentum = momentum

        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False)
        self.batch_norm = fvnn.BatchNorm(out_channels, momentum=momentum)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, momentum={self.momentum}"
        )

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.batch_norm.reset_parameters()

    def create_padded_grid(self, grid: GridBatch) -> GridBatch:
        return grid.conv_grid(kernel_size=self.kernel_size, stride=1)

    def forward(self, data: JaggedTensor, grid: GridBatch, padded_grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.kernel_size, stride=1, source_grid=grid, target_grid=padded_grid
        )
        data = self.conv(data, plan)
        data = self.batch_norm(data, padded_grid)
        return data


@fvnn_module
class SimpleUNetDepad(nn.Module):
    """
    Output depadding module for producing final predictions.

    Transforms the network output back to the original grid dimensions and target
    channel count. The module simultaneously handles two transformations:
    1. Spatial depadding: Removes padding to match original grid dimensions
    2. Channel adjustment: Converts base channels to final output channels

    Uses transposed convolution to ensure proper gradient flow during training
    while accurately mapping from the padded feature space back to the original
    input space. This is the final step that produces the network's predictions.

    Since this is the last module in the network, we don't need a final batchnorm.

    Builds its plan inline, since not needed elsewhere.

    Args:
        in_channels (int): Number of input channels (base channels from the network).
        out_channels (int): Number of output channels for final predictions.
        kernel_size (NumericMaxRank1): Size of convolution kernels. Defaults to 3.
        momentum (float): Momentum parameter for batch normalization. Defaults to 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: NumericMaxRank1 = 3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.deconv = fvnn.SparseConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False
        )

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " f"kernel_size={self.kernel_size}"

    def reset_parameters(self) -> None:
        self.deconv.reset_parameters()

    def forward(self, data: JaggedTensor, padded_grid: GridBatch, grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.kernel_size, stride=1, source_grid=padded_grid, target_grid=grid
        )
        return self.deconv(data, plan)


@fvnn_module
class SimpleUNet(nn.Module):
    """
    Complete U-Net architecture for sparse voxel data processing.

    Implements a residual U-Net designed specifically for sparse 3D voxel grids.
    The network follows the classic U-Net structure with encoder-decoder paths,
    skip connections, and residual blocks, adapted for efficient processing of
    sparse volumetric data using the FVDB framework.

    The architecture consists of three main stages:
    1. Padding: Prepares input data with appropriate spatial and channel dimensions
    2. Encoder-Decoder: Recursive downsampling and upsampling with skip connections
    3. Depadding: Produces final output in original grid dimensions

    The network is designed for dense prediction tasks on sparse 3D data, such as
    semantic segmentation, shape completion, or volumetric reconstruction. The
    residual connections improve gradient flow and training stability, while the
    U-Net structure preserves both local and global spatial information.

    Args:
        in_channels (int): Number of input channels from the original data.
        base_channels (int): Number of base channels used throughout the network.
        out_channels (int): Number of output channels for final predictions.
        channel_growth_rate (int): Factor by which channels grow at each encoder level.
        kernel_size (NumericMaxRank1): Size of convolution kernels used throughout.
                                      Defaults to 3.
        downup_layer_count (int): Number of encoder-decoder levels. Defaults to 4.
        block_layer_count (int): Number of basic blocks per convolutional block.
                                Defaults to 2.
        momentum (float): Momentum parameter for all batch normalization layers.
                         Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        channel_growth_rate: int,
        kernel_size: NumericMaxRank1 = 3,
        downup_layer_count: int = 4,
        block_layer_count: int = 2,
        momentum: float = 0.1,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.channel_growth_rate = channel_growth_rate
        self.kernel_size = kernel_size
        self.downup_layer_count = downup_layer_count
        self.block_layer_count = block_layer_count
        self.momentum = momentum

        self.pad = SimpleUNetPad(in_channels, base_channels, kernel_size, momentum)
        self.downup = SimpleUNetDownUp(
            base_channels, channel_growth_rate, kernel_size, downup_layer_count, block_layer_count, momentum
        )
        self.depad = SimpleUNetDepad(base_channels, out_channels, kernel_size)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, base_channels={self.base_channels}, out_channels={self.out_channels}, "
            f"channel_growth_rate={self.channel_growth_rate}, kernel_size={self.kernel_size}, "
            f"downup_layer_count={self.downup_layer_count}, "
            f"block_layer_count={self.block_layer_count}, momentum={self.momentum}"
        )

    def reset_parameters(self) -> None:
        self.pad.reset_parameters()
        self.downup.reset_parameters()
        self.depad.reset_parameters()

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:

        padded_grid = self.pad.create_padded_grid(grid)

        data = self.pad(data, grid, padded_grid)
        data = self.downup(data, padded_grid)
        return self.depad(data, padded_grid, grid)
