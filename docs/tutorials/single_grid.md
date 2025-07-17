# Grid: A Simplified API for Single Sparse Grids

The `Grid` class provides a cleaner, more ergonomic interface for working with individual
sparse grids, eliminating the complexity of batch operations when you only need a single grid.
This tutorial demonstrates the key differences and simplifications compared to `GridBatch`
through practical examples.

## Simple Example: Point Cloud Grid Creation and Sampling

The most common use case for sparse grids is creating them from point clouds and sampling
interpolated values. Here we show how `Grid` simplifies this workflow compared to `GridBatch`.

### Using GridBatch (Batched Approach)

```python
import torch
import fvdb
from fvdb import GridBatch, JaggedTensor

device = torch.device("cuda:0")

# Load your point cloud data
points = torch.randn(1000, 3, device=device)

# Creating a grid from points requires instantiation then population
grid_batch = GridBatch(device=device)
grid_batch.set_from_points(
    JaggedTensor(points),  # Must wrap in JaggedTensor even for single grid
    voxel_sizes=0.1,
    origins=[0, 0, 0]
)

features = torch.randn(grid_batch.total_voxels, 32, device=device)
query_points = torch.randn(500, 3, device=device)

# Sampling requires handling JaggedTensor inputs/outputs
sample_points = JaggedTensor(query_points)
voxel_features = JaggedTensor(features)
interpolated = grid_batch.sample_trilinear(sample_points, voxel_features)
result = interpolated.jdata  # Extract actual data from JaggedTensor
```

### Using Grid (Simplified Approach)

```python
import torch
import fvdb
from fvdb import Grid

device = torch.device("cuda:0")

# Load your point cloud data
points = torch.randn(1000, 3, device=device)

# Clean creation using classmethod
grid = Grid.from_points(points, voxel_size=0.1, origin=[0, 0, 0], device=device)

# Assume features tensor matches grid voxel count
features = torch.randn(grid.num_voxels, 32, device=device)

query_points = torch.randn(500, 3, device=device)

# Direct tensor operations - no JaggedTensor wrapping needed
interpolated = grid.sample_trilinear(query_points, features)
# result is already a regular tensor!
```

The `Grid` approach eliminates the need for `JaggedTensor` wrapping and provides direct tensor
operations, making the code cleaner and more intuitive.

## Ergonomic Simplifications

The `Grid` class simplifies the `GridBatch` API for single-grid operations:

1. **Cleaner Construction**: Named constructors (`Grid.from_points()`, `Grid.from_mesh()`, etc.) make intent clear and reduce boilerplate code.

2. **No JaggedTensor Wrapping**: Work directly with PyTorch tensors without the overhead of wrapping single tensors in `JaggedTensor` containers.

3. **Automatic Batch Handling**: Grid manages batch dimensions internally for operations that require them, eliminating manual tensor reshaping.

4. **Simplified Return Values**: Get tensors directly instead of unwrapping results from `JaggedTensors`.

5. **More Pythonic**: Properties like `grid.num_voxels` instead of `grid_batch.num_voxels_at(0)` provide cleaner access to grid attributes.

6. **Consistent API**: All methods operate on regular tensors, providing a consistent interface that integrates naturally with PyTorch workflows.

The `Grid` class maintains full compatibility with GPU operations and sparse grid functionality while providing a much more intuitive interface for the common case of working with single sparse grids. For batch operations with multiple grids, `GridBatch` remains the appropriate choice.

For more examples on grid construction patterns, refer to the [Building Sparse Grids](building_grids.md) tutorial. Grid operations and neural network integration are covered in the corresponding sections of this documentation.
