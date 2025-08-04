# Ray Tracing

# voxels_along_rays

Computes the intersection of rays with voxels in sparse volume grids. This function performs ray marching through sparse voxel grids to find all voxels that each ray passes through, along with the entry and exit times.

## Overview

This function traces rays through a batch of sparse volume grids, returning the voxels intersected by each ray and the time parameters where the ray enters and exits each voxel. It's particularly useful for volume rendering, ray casting, and other applications requiring ray-voxel intersection queries.

## Function Signature

```python
voxels, times = grid_batch.voxels_along_rays(
    ray_origins,
    ray_directions,
    max_vox,
    eps=0.0,
    return_ijk=True,
    cumulative=False
)
```

## Parameters

### `ray_origins` (JaggedTensor)
- **Shape**: `[B, M_i, 3]` where `B` is the batch size and `M_i` is the number of rays for batch element `i`
- **Description**: Starting points of rays in world coordinates. Each batch element can have a different number of rays.
- **Data type**: Floating point (float32 or float64)

### `ray_directions` (JaggedTensor)
- **Shape**: `[B, M_i, 3]` - must match the structure of `ray_origins`
- **Description**: Direction vectors of rays in world coordinates. These vectors define the direction of ray propagation.
- **Data type**: Must match `ray_origins`

### `max_vox` (int)
- **Description**: Maximum number of voxels to return per ray
- **Values**:
  - `-1`: No limit, return all intersected voxels
  - `> 0`: Return at most this many voxels per ray (useful for limiting memory usage)

### `eps` (float, default=0.0)
- **Description**: Minimum intersection length threshold. Voxel intersections with length less than `eps` are filtered out.
- **Purpose**: Helps eliminate numerical artifacts from grazing intersections
- **Range**: Must be ≥ 0.0

### `return_ijk` (bool, default=True)
- **Description**: Controls the format of returned voxel information
- **Values**:
  - `True`: Return voxel coordinates as `[i, j, k]` indices
  - `False`: Return linear voxel indices (useful for indexing into voxel data arrays)

### `cumulative` (bool, default=False)
- **Description**: Controls the indexing scheme for linear indices (only relevant when `return_ijk=False`)
- **Values**:
  - `False`: Indices are relative to each grid in the batch
  - `True`: Indices are cumulative across the entire batch

## Return Values

The function returns a tuple of two JaggedTensors:

### `voxels` (JaggedTensor)
- **Shape**:
  - If `return_ijk=True`: `[B, R_i, V_{i,j}, 3]` - voxel coordinates
  - If `return_ijk=False`: `[B, R_i, V_{i,j}]` - linear indices
- Where `V_{i,j}` is the number of voxels intersected by ray `j` in batch `i`

### `times` (JaggedTensor)
- **Shape**: `[B, R_i, V_{i,j}, 2]`
- **Description**: Entry and exit times for each voxel intersection
  - `times[..., 0]`: Ray parameter `t` at voxel entry
  - `times[..., 1]`: Ray parameter `t` at voxel exit
- **Note**: The intersection point is computed as `ray_origin + t * ray_direction`

## Usage Examples

### Basic Ray Tracing
```python
# Trace rays through a sparse volume
ray_origins = JaggedTensor(...)     # Starting points
ray_directions = JaggedTensor(...)  # Ray directions

# Get all voxel intersections
voxels, times = grid_batch.voxels_along_rays(
    ray_origins,
    ray_directions,
    max_vox=-1  # unlimited max voxels
)

# Process each ray's intersections
for batch_idx in range(grid_batch.batch_size()):
    for ray_idx in range(voxels.size(batch_idx, 0)):
        ray_voxels = voxels[batch_idx, ray_idx]  # [N, 3] array of voxel coords
        ray_times = times[batch_idx, ray_idx]    # [N, 2] array of entry/exit times
```

### Limited Ray Marching
```python
# Limit to first 10 voxels per ray (useful for early termination)
voxels, times = grid_batch.voxels_along_rays(
    ray_origins,
    ray_directions,
    max_vox=10
)
```

### Using Linear Indices
```python
# Get linear indices instead of coordinates
voxels, times = grid_batch.voxels_along_rays(
    ray_origins,
    ray_directions,
    return_ijk=False
)

# Use indices to look up voxel data
voxel_data = grid_data[voxels]  # Index into your voxel data array
```

### Filtering Grazing Intersections
```python
# Filter out very small intersections
voxels, times = grid_batch.voxels_along_rays(
    ray_origins,
    ray_directions,
    eps=1e-6  # Ignore intersections smaller than this
)
```

## Implementation Notes

1. **Ray Transformation**: Rays are automatically transformed from world space to voxel space using each grid's transform.

2. **Bounding Box Clipping**: Rays are clipped against each grid's bounding box before traversal begins.

3. **Duplicate Handling**: The algorithm handles numerical edge cases where the same voxel might be visited twice, merging such duplicates automatically.

4. **Performance**: The function uses optimized GPU kernels (HDDA - Hierarchical Digital Differential Analyzer) for efficient traversal of sparse grids.

5. **Memory Efficiency**: The function returns results in a jagged format, allowing different rays to intersect different numbers of voxels without wasting memory.

6. **Thread Configuration**: The implementation automatically adjusts thread counts based on data type (384 threads for float32, 256 for float64).

## Common Use Cases

- **Volume Rendering**: Accumulate density/color along rays
- **Ray Casting**: Find first intersection for surface rendering
- **Collision Detection**: Check if rays intersect with voxelized objects
- **Light Transport**: Trace light paths through volumetric media
- **Medical Imaging**: Ray-based reconstruction and visualization
