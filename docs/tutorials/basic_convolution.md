# Grid and GridBatch Convolution Operations


Convolving the features of a `GridBatch` can be accomplished with either a high-level `torch.nn.Module` derived class provided by `fvdb.nn` or with more low-level methods available with `GridBatch`, we will illustrate both techniques.

## High-level Usage with `fvdb.nn`

`fvdb.nn.SparseConv3d` provides a high-level `torch.nn.Module` class for convolution on `fvdb` classes that is an analogue to the use of `torch.nn.Conv3d`.  Using this module is the recommended functionality for performing convolution with `fvdb` because it not only manages functionality such as initializing the weights of the convolution and calling appropriate backend implementation functions but it also provides certain backend optimizations which will be illustrated in the [Low-level usage](#low-level-usage-with-gridbatch) section.

`fvdb.nn.SparseConv3d` takes explicit `(data: JaggedTensor, plan: ConvolutionPlan)` arguments — topology and features are always passed separately.  A `ConvolutionPlan` precomputes the necessary acceleration structures for a given grid and kernel configuration.

A simple example of using `fvdb.nn.SparseConv3d` is as follows:

```python
import fvdb
import fvdb.nn as fvdbnn
from fvdb import ConvolutionPlan
from fvdb.utils.examples import load_car_1_mesh
import torch
import numpy as np
import point_cloud_utils as pcu

num_pts = 10_000
vox_size = 0.02

mesh_load_funcs = [load_car_1_mesh]

points = []
normals = []

for func in mesh_load_funcs:
    pts, nms = func(mode="vn")
    pmt = torch.randperm(pts.shape[0])[:num_pts]
    pts, nms = pts[pmt], nms[pmt]
    points.append(pts)
    normals.append(nms)

# JaggedTensors of points and normals
points = fvdb.JaggedTensor(points)
normals = fvdb.JaggedTensor(normals)

# Create a grid
grid = fvdb.GridBatch.from_points(points, voxel_sizes=vox_size)

# Splat the normals into the grid with trilinear interpolation
vox_normals = grid.splat_trilinear(points, normals)

# Build a ConvolutionPlan for stride=1 same-topology convolution
plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid, target_grid=grid)

# fvdb.nn.SparseConv3d is a convenient torch.nn.Module implementing the fVDB convolution
conv = fvdbnn.SparseConv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, bias=False).to(grid.device)

output = conv(vox_normals, plan)
```
Let's visualize the original grid with normals visualized as colours alongside the result of these features after a convolution initialized with random weights:
![](../imgs/fig/simple_conv.png)

For stride values greater than 1, the output is a sampled convolution lattice: its voxel size is multiplied by the
stride while its origin is preserved. It is not a pooling or block-coarsening grid. For the default registration, an
integer fine-lattice coordinate `fine_ijk` and an integer coarse-lattice coordinate `coarse_ijk` are connected by the
zero-based kernel tap `tap_ijk` when `fine_ijk = stride * coarse_ijk + tap_ijk - padding_before` (componentwise),
where `padding_before = floor((kernel_size - 1) / 2)` and
`0 <= tap_ijk[axis] < kernel_size[axis]`. Generated output topology is the positive structural support of that
relation.

```python continuation
# Stride=2: output grid has half the resolution (twice the world-space voxel size)
plan_down = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=2, source_grid=grid, target_grid=None)
conv_down = fvdbnn.SparseConv3d(in_channels=3, out_channels=3, kernel_size=3, stride=2, bias=False).to(grid.device)

output = conv_down(vox_normals, plan_down)
coarse_grid = plan_down.target_grid_batch
```

![](../imgs/fig/stride_conv.png)

The following animations illustrate strided sparse convolution and transposed convolution. A forward operation evaluates the fine-to-coarse edges of this graph; a transposed operation evaluates the same connectivity in the opposite direction. Neither operation is a value inverse of the other:

![Strided sparse convolution animation](../imgs/fig/strided_sparse_conv.gif)

![Strided transposed sparse convolution animation](../imgs/fig/strided_transposed_sparse_conv.gif)

Strided transposed convolution can be performed with `fvdb.nn.SparseConvTranspose3d` (a separate class). With `target_grid=None` it generates the complete, uncropped structural topology (`ConvolutionTopologyPolicy.COMPLETE`). Supplying a target chooses the symmetric `ConvolutionTopologyPolicy.RESTRICTED` policy: the same relation is evaluated only on those requested rows, which may include unreachable zero-degree rows. An encoder-decoder commonly uses a saved fine grid as a restricted target, but this does not make the operation an inverse or guarantee that every saved coordinate is reachable.

```python continuation
# Strided transposed convolution operator, stride=2
transposed_conv = fvdbnn.SparseConvTranspose3d(in_channels=3, out_channels=3, kernel_size=3, stride=2, bias=False).to(grid.device)

# Build a restricted decoder plan: source is coarse and the saved fine grid limits output rows.
plan_up = ConvolutionPlan.from_grid_batch_transposed(kernel_size=3, stride=2, source_grid=coarse_grid, target_grid=grid)
transposed_output = transposed_conv(output, plan_up)

# Or generate the complete transposed topology without a saved target.
plan_up_complete = ConvolutionPlan.from_grid_batch_transposed(
    kernel_size=3,
    stride=2,
    source_grid=coarse_grid,
    topology_policy=fvdb.ConvolutionTopologyPolicy.COMPLETE,
)
```

Here we visualize the original grid, the grid after strided convolution, and a transposed convolution restricted to the original grid. The saved target selects the displayed output topology; it does not restore values or define the generated transposed support:

![](../imgs/fig/transposed_stride_conv.png)

For the exact finite adjoint of an existing plan, use `ConvolutionPlan.from_plan_transposed(plan_down)` rather than rebuilding from grids. It reverses that plan's stored edges exactly. With tied weights, pass `weight.transpose(0, 1).contiguous()` to the transposed execution. An independently learned `SparseConvTranspose3d` instead has its own weights and uses transposed connectivity without an adjoint claim.

`grid.coarsened_grid(stride)` serves pooling and cell aggregation: it has a block-centroid transform and is not interchangeable with `conv_grid(kernel_size, stride)`. In particular, it is not a valid shortcut for convolution targets. The special `kernel_size=1, stride=1` generated forward grid is the source object itself; `kernel_size=1, stride>1` samples only the stride-aligned residues rather than coarsening every block. Issue #668 is the motivating even-kernel example: a complete `kernel_size=stride=4` convolution of a `16^3` cube produces a `5^3` coarse support, not `4^3`.


## Low-level Usage with `GridBatch`

The [high-level `fvdb.nn.SparseConv3d` class](#high-level-usage-with-fvdbnn) wraps several pieces of `GridBatch` functionality to provide a convenient `torch.nn.Module` for convolution.  However, for a more low-level approach that accomplishes the same outcome, the `GridBatch` class itself can be the starting point for performing convolution on the grid and its features.  We will illustrate this approach for completeness, though we do recommend the use of the `fvdb.nn.SparseConv3d` Module for most use-cases.

Using the `GridBatch` convolution functions directly requires a little more knowledge about what happens under the hood.  Due to the nature of a sparse grid, in order to make convolution performant, fVDB precomputes the necessary acceleration structures for a given sparse grid, kernel size, and stride.

The `fvdb.ConvolutionPlan` class encapsulates these acceleration structures and uses them to perform the convolution.  Here is an example of how to construct a `ConvolutionPlan` and use it to perform a convolution:

```python
import fvdb
from fvdb import ConvolutionPlan
from fvdb.utils.examples import load_car_1_mesh
import torch
import numpy as np
import point_cloud_utils as pcu

num_pts = 10_000
vox_size = 0.02

mesh_load_funcs = [load_car_1_mesh]

points = []
normals = []

for func in mesh_load_funcs:
    pts, nms = func(mode="vn")
    pmt = torch.randperm(pts.shape[0])[:num_pts]
    pts, nms = pts[pmt], nms[pmt]
    points.append(pts)
    normals.append(nms)

# JaggedTensors of points and normals
points = fvdb.JaggedTensor(points)
normals = fvdb.JaggedTensor(normals)

# Create a grid
grid = fvdb.GridBatch.from_points(points, voxel_sizes=vox_size)

# Splat the normals into the grid with trilinear interpolation
vox_normals = grid.splat_trilinear(points, normals)

# Create a convolution plan — this precomputes the acceleration structures for the given grid and kernel
plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid)

# Create random weights for our convolution kernel of size 3x3x3 that takes 3 input channels and produces 3 output channels
kernel_weights = torch.randn(3, 3, 3, 3, 3, device=grid.device)

# Execute the convolution
conv_vox_normals = plan.execute(vox_normals, kernel_weights)
```
Here we visualize the output of our convolution alongside the original grid with normals visualized as colours:
![](../imgs/fig/gridbatch_conv.png)

These acceleration structures can potentially be expensive to compute, so it is often useful to re-use the `ConvolutionPlan` in the same network to perform a convolution on other features or with different weights.  This optimization is something `fvdb.nn.SparseConv3d` attempts to do where appropriate and is one reason we recommend using `fvdb.nn.SparseConv3d` over this low-level approach.
