# JaggedTensor: Efficient Batching for Variable-Length Data

This tutorial introduces `JaggedTensor`, fVDB's data structure for efficiently handling batches of tensors with varying sizes. We'll explore how to create, manipulate, and perform operations on jagged tensors, with special emphasis on the type-safe PyTorch function overloads.

## What is a JaggedTensor?

In many machine learning applications, we need to process batches of data where each item has a different number of elements. For example:

- Point clouds with different numbers of points per scene
- Graphs with different numbers of nodes
- Sequences with different lengths
- Sparse voxel grids with different numbers of active voxels

A `JaggedTensor` is conceptually a list of tensors with varying first dimensions: `[N_0, *], [N_1, *], ..., [N_{B-1}, *]` where `B` is the batch size, `N_i` is the number of elements in the i-th batch item, and `*` represents additional dimensions that match across all tensors.

![jaggedtensor1.png](../imgs/fig/jaggedtensor1.png)

Internally, `JaggedTensor` concatenates these tensors into a single flat `jdata` tensor of shape `[N_0 + N_1 + ... + N_{B-1}, *]` for efficient GPU processing. It maintains indexing structures (`jidx`, `joffsets`) to track batch boundaries.

![jaggedtensor4.png](../imgs/fig/jaggedtensor4.png)

## Creating JaggedTensors

### From a List of Tensors

The most common way to create a `JaggedTensor` is from a list of PyTorch tensors:

```python
import torch
import fvdb

# Create three tensors with different first dimensions
t0 = torch.randn(100, 3)  # 100 points, 3D
t1 = torch.randn(150, 3)  # 150 points, 3D
t2 = torch.randn(120, 3)  # 120 points, 3D

# Create a JaggedTensor containing all three
jt = fvdb.JaggedTensor([t0, t1, t2])

print(f"Number of tensors: {jt.num_tensors}")  # 3
print(f"Total elements: {jt.jdata.shape[0]}")  # 370 (100 + 150 + 120)
print(f"Element shape: {jt.rshape}")  # (3,)
```

### Using Factory Functions

fVDB provides factory functions similar to PyTorch's tensor creation functions:

```python continuation
# Create jagged tensor filled with random values
jt_rand = fvdb.jrand(lsizes=[100, 150, 120], rsizes=[3])
# lsizes: list of sizes for each tensor in the batch
# rsizes: shape of the regular (non-jagged) dimensions

# Create jagged tensor filled with zeros
jt_zeros = fvdb.jzeros(lsizes=[100, 150, 120], rsizes=[3], device="cuda")

# Create jagged tensor filled with ones
jt_ones = fvdb.jones(lsizes=[100, 150, 120], rsizes=[3])

# Create jagged tensor filled with normal random values
jt_randn = fvdb.jrandn(lsizes=[100, 150, 120], rsizes=[3])
```

### From Flat Data and Indices

If you already have flattened data and indexing information:

```python continuation
# Flattened data: 370 total elements
data = torch.randn(370, 3)

# Indices indicating which batch each element belongs to
indices = torch.tensor([0]*100 + [1]*150 + [2]*120)

# Create JaggedTensor
jt = fvdb.JaggedTensor.from_data_and_indices(data, indices, num_tensors=3)
```

Or using offsets:

```python continuation
# Offsets marking boundaries: [0, 100, 250, 370]
offsets = torch.tensor([0, 100, 250, 370])

jt = fvdb.JaggedTensor.from_data_and_offsets(data, offsets)
```

## Accessing JaggedTensor Data

### The Underlying Data: jdata

The `jdata` property provides access to the flattened concatenated tensor:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Access the underlying flattened data
print(jt.jdata.shape)  # torch.Size([250, 3])

# Modify the data directly
jt.jdata *= 2.0
```

### Indexing and Iteration

You can index into a `JaggedTensor` to extract individual tensors:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3), torch.randn(120, 3)])

# Access the first tensor (returns a JaggedTensor with one element)
first = jt[0]
print(first.jdata.shape)  # torch.Size([100, 3])
print(first.num_tensors)  # 1

# Slice to get a subset
subset = jt[1:3]  # Get tensors 1 and 2
print(subset.num_tensors)  # 2

# Iterate over tensors
for i, tensor in enumerate(jt):
    print(f"Tensor {i}: {tensor.jdata.shape}")
```

### Unbinding to Regular Tensors

To convert a `JaggedTensor` back to a list of regular PyTorch tensors:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Get list of individual tensors
tensors = jt.unbind()
print(len(tensors))  # 2
print(tensors[0].shape)  # torch.Size([100, 3])
print(tensors[1].shape)  # torch.Size([150, 3])
```

## PyTorch Function Overloads: torch.* vs fvdb.*

A key feature of `JaggedTensor` is its integration with PyTorch's function dispatch system through `__torch_function__`. This allows many PyTorch functions to work directly with `JaggedTensor` at runtime. However, static type checkers cannot understand this dynamic dispatch mechanism.

**The solution**: fVDB provides type-safe wrappers in the `fvdb` namespace that work with both `JaggedTensor` and regular `Tensor`, with proper type inference.

### The Type Safety Problem

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# This works at RUNTIME due to __torch_function__
result1 = torch.relu(jt)  # type: ignore
# BUT: Type checker sees result1 as Tensor, not JaggedTensor
# You need type: ignore to suppress the warning

# This is type-safe and works at runtime AND compile-time
result2 = fvdb.relu(jt)
# Type checker correctly infers result2 as JaggedTensor
```

Both approaches produce identical results at runtime, but `fvdb.relu()` provides proper type safety for static analysis tools like mypy and pyright.

### Why Use fvdb.* Functions?

1. **Type Safety**: Static type checkers correctly infer return types
2. **IDE Support**: Better autocomplete and inline documentation
3. **Code Clarity**: Explicit about working with both Tensor types
4. **Zero Overhead**: Direct delegation to PyTorch functions

### Elementwise Operations

Elementwise operations apply to each element independently, preserving the jagged structure:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Activation functions
activated = fvdb.relu(jt)      # ReLU activation
sigmoid_out = fvdb.sigmoid(jt)  # Sigmoid activation
tanh_out = fvdb.tanh(jt)        # Tanh activation

# Mathematical operations
sqrt_out = fvdb.sqrt(jt.abs())  # Square root (on absolute values)
exp_out = fvdb.exp(jt)          # Exponential
log_out = fvdb.log(jt.abs())    # Natural logarithm

# Rounding operations
floor_out = fvdb.floor(jt)      # Round down
ceil_out = fvdb.ceil(jt)        # Round up
round_out = fvdb.round(jt)      # Round to nearest

# Utility operations
clamped = fvdb.clamp(jt, min=-1.0, max=1.0)  # Clamp values
safe = fvdb.nan_to_num(jt)      # Replace NaN/inf with numbers
```

### In-Place Operations

Some operations have in-place variants (ending with `_`):

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# In-place ReLU (modifies jt directly)
fvdb.relu_(jt)

# Compare with torch.* (works but needs type: ignore)
torch.relu_(jt)  # type: ignore
```

### Binary Operations

Binary operations work between two `JaggedTensor`s, or between a `JaggedTensor` and a scalar:

```python continuation
jt1 = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])
jt2 = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Arithmetic operations with scalars
scaled = fvdb.mul(jt1, 2.0)       # Multiply by scalar
shifted = fvdb.add(jt1, 1.0)      # Add scalar
divided = fvdb.true_divide(jt1, 3.0)  # Divide by scalar

# Arithmetic operations between JaggedTensors
sum_jt = fvdb.add(jt1, jt2)       # Elementwise addition
diff_jt = fvdb.sub(jt1, jt2)      # Elementwise subtraction
prod_jt = fvdb.mul(jt1, jt2)      # Elementwise multiplication
quot_jt = fvdb.true_divide(jt1, jt2)  # Elementwise division

# Power and other operations
powered = fvdb.pow(jt1, 2.0)      # Square each element
remainder = fvdb.remainder(jt1, 2.0)  # Modulo operation

# Element-wise min/max
max_jt = fvdb.maximum(jt1, jt2)   # Elementwise maximum
min_jt = fvdb.minimum(jt1, jt2)   # Elementwise minimum
```

### Comparison Operations

Comparison operations return boolean `JaggedTensor`s:

```python continuation
jt1 = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])
jt2 = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Comparisons with scalars
gt_zero = fvdb.gt(jt1, 0.0)       # Greater than
ge_zero = fvdb.ge(jt1, 0.0)       # Greater than or equal
lt_zero = fvdb.lt(jt1, 0.0)       # Less than
le_zero = fvdb.le(jt1, 0.0)       # Less than or equal
eq_zero = fvdb.eq(jt1, 0.0)       # Equal to
ne_zero = fvdb.ne(jt1, 0.0)       # Not equal to

# Comparisons between JaggedTensors
gt_mask = fvdb.gt(jt1, jt2)       # Elementwise comparison
eq_mask = fvdb.eq(jt1, jt2)       # Elementwise equality

# Conditional selection with where
# Select from jt1 where mask is True, otherwise from jt2
selected = fvdb.where(gt_mask, jt1, jt2)
```

### Reduction Operations

Reductions that preserve the leading (flattened) dimension work seamlessly:

```python continuation
# Create JaggedTensor with shape [100, 10, 3] and [150, 10, 3]
jt = fvdb.JaggedTensor([torch.randn(100, 10, 3), torch.randn(150, 10, 3)])
print(jt.jdata.shape)  # torch.Size([250, 10, 3])

# Reduce over non-primary dimensions
# These preserve the jagged structure (batch boundaries)

summed = fvdb.sum(jt, dim=-1)     # Sum over last dimension
print(summed.jdata.shape)         # torch.Size([250, 10])
print(summed.num_tensors)         # 2 (structure preserved!)

mean_val = fvdb.mean(jt, dim=1)   # Mean over dimension 1
print(mean_val.jdata.shape)       # torch.Size([250, 3])

max_val = fvdb.amax(jt, dim=-1)   # Max over last dimension
min_val = fvdb.amin(jt, dim=-1)   # Min over last dimension

# Standard deviation and variance
std_val = fvdb.std(jt, dim=1)     # Standard deviation
var_val = fvdb.var(jt, dim=1)     # Variance

# Norms
l2_norm = fvdb.norm(jt, p=2, dim=-1)  # L2 norm over last dimension
l1_norm = fvdb.norm(jt, p=1, dim=-1)  # L1 norm

# Logical reductions
all_positive = fvdb.all(fvdb.gt(jt, 0.0), dim=-1)  # All elements > 0?
any_positive = fvdb.any(fvdb.gt(jt, 0.0), dim=-1)  # Any element > 0?

# Argmax and argmin
max_indices = fvdb.argmax(jt, dim=-1)  # Indices of max values
min_indices = fvdb.argmin(jt, dim=-1)  # Indices of min values
```

**Important**: These reductions only work on non-primary dimensions. To reduce over the jagged dimension itself (collapsing different-length tensors), use the specialized `j*` methods described below.

### Chaining Operations

All fvdb functions return the same type as their input, enabling seamless chaining:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Chain multiple operations (all type-safe!)
result = fvdb.sigmoid(
    fvdb.add(
        fvdb.relu(jt),
        1.0
    )
)

# Or more readable step-by-step:
activated = fvdb.relu(jt)          # Apply ReLU
shifted = fvdb.add(activated, 1.0)  # Add 1.0
normalized = fvdb.sigmoid(shifted)  # Apply sigmoid

# Type checker knows the type at each step
print(type(activated))   # JaggedTensor
print(type(shifted))     # JaggedTensor
print(type(normalized))  # JaggedTensor
```

## Jagged-Specific Operations: j* Methods

While PyTorch functions preserve the jagged structure, sometimes you need to operate directly on the jagged dimensions. fVDB provides specialized `j*` methods for this:

### Jagged Sum: jsum()

Sum along the jagged dimension to reduce varying-length tensors:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3), torch.randn(120, 3)])
print(jt.jdata.shape)  # torch.Size([370, 3])

# Sum each tensor in the batch (reduces jagged dimension)
summed = jt.jsum(dim=0)
print(summed.jdata.shape)  # torch.Size([3, 3])
print(summed.num_tensors)  # 3

# Each row is the sum over one tensor in the batch
```

### Jagged Max/Min: jmax(), jmin()

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Max along jagged dimension (returns values and indices)
max_values, max_indices = jt.jmax(dim=0)
print(max_values.jdata.shape)  # torch.Size([2, 3])
print(max_indices.jdata.shape)  # torch.Size([2, 3])
```

### Jagged Reshape: jreshape()

Reshape the jagged structure:

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# Reshape to different jagged sizes
# Note: Total number of elements must match
reshaped = jt.jreshape(lshape=[50, 50, 75, 75])  # Split into 4 tensors
print(reshaped.num_tensors)  # 4
```

### Jagged Flatten: jflatten()

Flatten nested jagged structures:

```python continuation
# Create nested jagged structure
nested = fvdb.JaggedTensor([[torch.randn(10, 3), torch.randn(20, 3)],
                             [torch.randn(15, 3)]])
print(nested.ldim)  # 2 (two levels of jagging)

# Flatten to single level
flattened = nested.jflatten()
print(flattened.ldim)  # 1
print(flattened.num_tensors)  # 3
```

## Complete Example: Processing Variable-Length Point Clouds

Here's a complete example demonstrating JaggedTensor usage for batch processing point clouds:

```python continuation
# Load or generate point clouds with different numbers of points
# (simulating 3 different scenes)
points_scene1 = torch.randn(1523, 3, device="cuda")  # Scene 1: 1523 points
points_scene2 = torch.randn(2847, 3, device="cuda")  # Scene 2: 2847 points
points_scene3 = torch.randn(1102, 3, device="cuda")  # Scene 3: 1102 points

# Create JaggedTensor for batch processing
jt_points = fvdb.JaggedTensor([points_scene1, points_scene2, points_scene3])
print(f"Batched {jt_points.num_tensors} point clouds")
print(f"Total points: {jt_points.jdata.shape[0]}")

# Generate per-point features (simulating a neural network output)
features = fvdb.jrandn(lsizes=jt_points.lshape, rsizes=[64], device="cuda")

# Apply a series of transformations
# Step 1: Apply ReLU activation
features = fvdb.relu(features)

# Step 2: L2 normalize each feature vector
feature_norms = fvdb.norm(features, p=2, dim=-1, keepdim=True)
features = fvdb.true_divide(features, fvdb.add(feature_norms, 1e-8))

# Step 3: Apply attention-like scaling
# Compute attention scores per scene (jagged sum)
attention_scores = fvdb.sigmoid(fvdb.sum(features, dim=-1, keepdim=True))
features = fvdb.mul(features, attention_scores)

# Step 4: Aggregate features per scene
# Sum features across all points in each scene
scene_features = features.jsum(dim=0)
print(f"Scene features: {scene_features.jdata.shape}")  # [3, 64]

# Step 5: Apply final transformation
scene_features = fvdb.relu(scene_features)

# Convert back to list if needed
feature_list = scene_features.unbind()
print(f"Scene 1 features: {feature_list[0].shape}")  # [64]
print(f"Scene 2 features: {feature_list[1].shape}")  # [64]
print(f"Scene 3 features: {feature_list[2].shape}")  # [64]
```

## When to Use torch.* vs fvdb.*

Here's a decision guide:

**Use `fvdb.*` functions when:**
- Working with `JaggedTensor` and you want type safety
- Building typed APIs or libraries
- Using static type checkers (mypy, pyright)
- You want better IDE autocomplete and documentation

**Use `torch.*` functions when:**
- Working only with regular `torch.Tensor`
- Type safety is not a concern
- You're okay with adding `type: ignore` comments

**Remember**: Both approaches are identical at runtime for supported operations. The difference is purely in static type checking.

## Common Patterns and Best Practices

### Pattern 1: Batch Processing with Variable Sizes

```python continuation
def process_batch(data_list):
    """Process a batch of variable-size data efficiently."""
    # Bundle into JaggedTensor for efficient GPU processing
    jt = fvdb.JaggedTensor(data_list)

    # Apply transformations (single GPU kernel launch!)
    jt = fvdb.relu(jt)
    jt = fvdb.mul(jt, 2.0)

    # Unbind if you need individual results
    return jt.unbind()
```

### Pattern 2: Safe Operations with Type Checking

```python continuation
def safe_activation(jt: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    """Apply activation with proper type hints."""
    # Type checker knows return type is JaggedTensor
    activated = fvdb.relu(jt)
    normalized = fvdb.sigmoid(activated)
    return normalized  # Type-safe!
```

### Pattern 3: Mixing Regular and Jagged Dimensions

```python continuation
def process_with_reductions(jt: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    """Reduce over regular dimensions, preserve jagged structure."""
    # Input: [N_0, D, C] and [N_1, D, C] and ...
    # where N_i varies (jagged), D and C are fixed (regular)

    # Mean over D dimension (preserves jagged structure)
    reduced = fvdb.mean(jt, dim=1)  # Now shape: [N_0, C] and [N_1, C] and ...

    # Apply activation
    activated = fvdb.relu(reduced)

    return activated
```

### Pattern 4: Avoiding Invalid Operations

```python continuation
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# WRONG: Cannot reduce over primary dimension with torch/fvdb functions
# result = fvdb.sum(jt)  # This would fail!
# result = fvdb.sum(jt, dim=0)  # This would also fail!

# RIGHT: Use j* methods for jagged dimension operations
result = jt.jsum(dim=0)  # Correctly reduces jagged dimension
```

## Integration with GridBatch

`JaggedTensor` is the foundation for `GridBatch` operations in fVDB. When working with sparse voxel grids, you'll frequently encounter jagged tensors as inputs and outputs:

```python continuation
# Create point clouds (using JaggedTensor)
points = fvdb.JaggedTensor([torch.randn(1000, 3), torch.randn(1500, 3)])

# Build grids from points
grids = fvdb.GridBatch.from_points(points, voxel_sizes=0.1)

# Grid operations return JaggedTensors
voxel_coords = grids.ijk  # JaggedTensor of voxel coordinates
print(type(voxel_coords))  # JaggedTensor

# You can use fvdb.* functions on the results
# Example: Normalize coordinates
normalized = fvdb.true_divide(voxel_coords.float(), 10.0)
```

## Summary

- **JaggedTensor** efficiently represents batches of variable-length data
- **fvdb.* functions** provide type-safe operations that work with both `JaggedTensor` and regular `Tensor`
- **torch.* functions** work at runtime via `__torch_function__` but lack static type safety
- **j* methods** (jsum, jmax, jreshape, etc.) operate on the jagged dimension itself
- Use **fvdb.*** for better type checking and IDE support
- Use **j*** methods when you need to manipulate the jagged structure

For more examples, see:
- [Basic Concepts](basic_concepts.md) - Overview of JaggedTensor in fVDB
- [Single Grid](single_grid.md) - Using Grid (non-batched) vs GridBatch with JaggedTensor
- [Basic Grid Operations](basic_grid_ops.md) - How GridBatch operations use JaggedTensor

