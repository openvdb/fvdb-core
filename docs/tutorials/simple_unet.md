# A Simple Convolutional U-Net

In this tutorial, you will be guided on how to build a simple sparse convolutional neural network using fVDB.
If you were using MinkowskiEngine to tackle sparse 3D data previously, we will also guide you step-by-step to help you smoothly transfer from it and enjoy speed-ups and memory-savings.

In our simplistic U-Net case, we want to build a Res-UNet with four layers, and each layer contains several blocks.
First, we import basic `fvdb` libraries:

```python
import fvdb
import fvdb.nn as fvnn
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor
import torch
```

Here `fvdb.nn` is a namespace similar to `torch.nn`, containing a broad definition of different neural layers.
Every `fvdb.nn` layer takes explicit `(data: JaggedTensor, plan_or_grid)` arguments — topology and features are always passed separately.  A `ConvolutionPlan` pre-computes the kernel map for a given grid and kernel configuration, and is passed alongside the data to convolution layers.

We could then build a basic block as follows:

```python continuation
class Downsample1x1(torch.nn.Module):
    """1x1 conv + BN for channel projection in residual connections."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = fvnn.SparseConv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.bn = fvnn.BatchNorm(out_ch)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch(1, 1, source_grid=grid, target_grid=grid)
        return self.bn(self.conv(data, plan), grid)

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, downsample=None, bn_momentum: float = 0.1):
        super().__init__()
        self.conv1 = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm1 = fvnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.conv2 = fvnn.SparseConv3d(out_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = fvnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, data: JaggedTensor, plan: ConvolutionPlan) -> JaggedTensor:
        grid = plan.target_grid_batch
        residual = data

        out = self.relu(self.norm1(self.conv1(data, plan), grid))
        out = self.norm2(self.conv2(out, plan), grid)

        if self.downsample is not None:
            residual = self.downsample(data, grid)

        out = fvdb.relu(out + residual)

        return out
```

This defines a similar block as `MinkowskiEngine`:

```python notest
import MinkowskiEngine as ME


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, downsample=None, bn_momentum: float = 0.1):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=3)
        self.norm1 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=3)
        self.norm2 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

All the network layers are fully compatible with `torch.nn`. The key difference is that `fvdb.nn` layers take explicit `(JaggedTensor, ConvolutionPlan)` or `(JaggedTensor, GridBatch)` arguments instead of wrapping them in a carrier object.
A full network definition could then be built as:

```python continuation
class FVDBUNetBase(torch.nn.Module):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    CHANNELS = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = fvnn.SparseConv3d(in_channels, self.inplanes, kernel_size=5, stride=1, bias=False)
        self.bn0 = fvnn.BatchNorm(self.inplanes)

        self.conv1p1s2 = fvnn.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn1 = fvnn.BatchNorm(self.inplanes)

        self.block1 = self._make_layer(BasicBlock, self.CHANNELS[0], self.LAYERS[0])

        self.conv2p2s2 = fvnn.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn2 = fvnn.BatchNorm(self.inplanes)

        self.block2 = self._make_layer(BasicBlock, self.CHANNELS[1], self.LAYERS[1])

        self.conv3p4s2 = fvnn.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn3 = fvnn.BatchNorm(self.inplanes)
        self.block3 = self._make_layer(BasicBlock, self.CHANNELS[2], self.LAYERS[2])

        self.conv4p8s2 = fvnn.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn4 = fvnn.BatchNorm(self.inplanes)
        self.block4 = self._make_layer(BasicBlock, self.CHANNELS[3], self.LAYERS[3])

        # Decoder uses SparseConvTranspose3d (separate class)
        self.convtr4p16s2 = fvnn.SparseConvTranspose3d(
            self.inplanes, self.CHANNELS[4], kernel_size=2, stride=2, bias=False)
        self.bntr4 = fvnn.BatchNorm(self.CHANNELS[4])

        self.inplanes = self.CHANNELS[4] + self.CHANNELS[2]
        self.block5 = self._make_layer(BasicBlock, self.CHANNELS[4], self.LAYERS[4])
        self.convtr5p8s2 = fvnn.SparseConvTranspose3d(
            self.inplanes, self.CHANNELS[5], kernel_size=2, stride=2, bias=False)
        self.bntr5 = fvnn.BatchNorm(self.CHANNELS[5])

        self.inplanes = self.CHANNELS[5] + self.CHANNELS[1]
        self.block6 = self._make_layer(BasicBlock, self.CHANNELS[5], self.LAYERS[5])
        self.convtr6p4s2 = fvnn.SparseConvTranspose3d(
            self.inplanes, self.CHANNELS[6], kernel_size=2, stride=2, bias=False)
        self.bntr6 = fvnn.BatchNorm(self.CHANNELS[6])

        self.inplanes = self.CHANNELS[6] + self.CHANNELS[0]
        self.block7 = self._make_layer(BasicBlock, self.CHANNELS[6], self.LAYERS[6])
        self.convtr7p2s2 = fvnn.SparseConvTranspose3d(
            self.inplanes, self.CHANNELS[7], kernel_size=2, stride=2, bias=False)
        self.bntr7 = fvnn.BatchNorm(self.CHANNELS[7])

        self.inplanes = self.CHANNELS[7] + self.INIT_DIM
        self.block8 = self._make_layer(BasicBlock, self.CHANNELS[7], self.LAYERS[7])

        self.final = fvnn.SparseConv3d(self.CHANNELS[7], out_channels, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = Downsample1x1(self.inplanes, planes * block.expansion)
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return torch.nn.ModuleList(layers)

    def _run_block(self, block, data, plan):
        for layer in block:
            data = layer(data, plan)
        return data

    def forward(self, features: JaggedTensor, grid: GridBatch) -> JaggedTensor:
        # --- Encoder ---
        # stride=1: initial convolution
        plan0 = ConvolutionPlan.from_grid_batch(5, 1, source_grid=grid, target_grid=grid)
        out = self.relu(self.bn0(self.conv0p1s1(features, plan0), grid))
        out_p1 = out
        grid1 = grid

        # stride=2 downsample
        plan_d1 = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid1)
        out = self.relu(self.bn1(self.conv1p1s2(out, plan_d1), plan_d1.target_grid_batch))
        grid2 = plan_d1.target_grid_batch
        plan_s1 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid2, target_grid=grid2)
        out_b1p2 = self._run_block(self.block1, out, plan_s1)

        plan_d2 = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid2)
        out = self.relu(self.bn2(self.conv2p2s2(out_b1p2, plan_d2), plan_d2.target_grid_batch))
        grid4 = plan_d2.target_grid_batch
        plan_s2 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid4, target_grid=grid4)
        out_b2p4 = self._run_block(self.block2, out, plan_s2)

        plan_d3 = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid4)
        out = self.relu(self.bn3(self.conv3p4s2(out_b2p4, plan_d3), plan_d3.target_grid_batch))
        grid8 = plan_d3.target_grid_batch
        plan_s3 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid8, target_grid=grid8)
        out_b3p8 = self._run_block(self.block3, out, plan_s3)

        plan_d4 = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid8)
        out = self.relu(self.bn4(self.conv4p8s2(out_b3p8, plan_d4), plan_d4.target_grid_batch))
        grid16 = plan_d4.target_grid_batch
        plan_s4 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid16, target_grid=grid16)
        out = self._run_block(self.block4, out, plan_s4)

        # --- Decoder ---
        # Transposed convolutions use from_grid_batch_transposed with target grids from encoder
        plan_u4 = ConvolutionPlan.from_grid_batch_transposed(2, 2, source_grid=grid16, target_grid=grid8)
        out = self.relu(self.bntr4(self.convtr4p16s2(out, plan_u4), grid8))
        out = fvdb.jcat([out, out_b3p8], dim=1)
        plan_s5 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid8, target_grid=grid8)
        out = self._run_block(self.block5, out, plan_s5)

        plan_u5 = ConvolutionPlan.from_grid_batch_transposed(2, 2, source_grid=grid8, target_grid=grid4)
        out = self.relu(self.bntr5(self.convtr5p8s2(out, plan_u5), grid4))
        out = fvdb.jcat([out, out_b2p4], dim=1)
        plan_s6 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid4, target_grid=grid4)
        out = self._run_block(self.block6, out, plan_s6)

        plan_u6 = ConvolutionPlan.from_grid_batch_transposed(2, 2, source_grid=grid4, target_grid=grid2)
        out = self.relu(self.bntr6(self.convtr6p4s2(out, plan_u6), grid2))
        out = fvdb.jcat([out, out_b1p2], dim=1)
        plan_s7 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid2, target_grid=grid2)
        out = self._run_block(self.block7, out, plan_s7)

        plan_u7 = ConvolutionPlan.from_grid_batch_transposed(2, 2, source_grid=grid2, target_grid=grid1)
        out = self.relu(self.bntr7(self.convtr7p2s2(out, plan_u7), grid1))
        out = fvdb.jcat([out, out_p1], dim=1)
        plan_s8 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid1, target_grid=grid1)
        out = self._run_block(self.block8, out, plan_s8)

        plan_final = ConvolutionPlan.from_grid_batch(1, 1, source_grid=grid1, target_grid=grid1)
        return self.final(out, plan_final)
```

Please note that here, when we apply transposed convolution layers, we build a `ConvolutionPlan` using `from_grid_batch_transposed` with the encoder-side grid as the `target_grid`.
This is needed to guide the output domain of the network, because for perception networks, the output grid topology should align with the input topology.
`ConvolutionPlan`s should be built once and reused across forward passes for efficiency.

> **Note:** fVDB also provides a built-in `fvdb.nn.SimpleUNet` that implements this pattern with plan caching and other optimizations. See `fvdb/nn/simple_unet.py` for the reference implementation.

To perform inference with the network, pass the features and grid explicitly:

```python continuation
coords = fvdb.JaggedTensor([
    (torch.randn(10_000, 3, device='cuda')),
    (torch.randn(11_000, 3, device='cuda')),
])

grid = fvdb.GridBatch.from_points(coords)
features = grid.jagged_like(torch.randn(grid.total_voxels, 32, device='cuda'))

model = FVDBUNetBase(32, 1).to('cuda')
output = model(features, grid)
```

The output will carry gradients during training, and you could train the sparse network accordingly.
Please find a fully working example at `examples/perception_example.py`. The same network is implemented using `MinkowskiEngine` for reference.
