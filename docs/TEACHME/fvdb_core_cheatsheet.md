# fVDB API Cheat Sheet

Quick reference for the core fVDB APIs. For full signatures see `fvdb/grid_batch.py`; for narrative explanations see `docs/wip/`.

---

## Imports

```python
import fvdb
import fvdb.nn as fvnn
from fvdb import ConvolutionPlan
import torch
```

---

## JaggedTensor

A batch of tensors with variable first dimension. Stored flat on GPU.

```python continuation
# --- Creation ---
t0, t1, t2 = torch.randn(100, 3), torch.randn(150, 3), torch.randn(120, 3)
jt = fvdb.JaggedTensor([t0, t1, t2])                          # from list of Tensors
jt = fvdb.JaggedTensor.from_zeros(lsizes=[100,150], rsizes=[3]) # zeros;  lsizes=per-item counts, rsizes=feature shape
jt = fvdb.JaggedTensor.from_ones(lsizes=[100,150],  rsizes=[3]) # ones
jt = fvdb.JaggedTensor.from_randn(lsizes=[100,150], rsizes=[3]) # normal random
jt = fvdb.JaggedTensor.from_rand(lsizes=[100,150],  rsizes=[3]) # uniform random

data = torch.randn(370, 3)
idx  = torch.tensor([0]*100 + [1]*150 + [2]*120)
jt = fvdb.JaggedTensor.from_data_and_indices(data, idx, num_tensors=3)
offsets = torch.tensor([0, 100, 250, 370])
jt = fvdb.JaggedTensor.from_data_and_offsets(data, offsets)  # offsets: [B+1] cumulative

# --- Key properties ---
jt.jdata       # Tensor[N_total, *]  — flat concatenated data
jt.jidx        # Tensor[N_total]     — batch index of each element (int)
jt.joffsets    # Tensor[B+1]         — cumulative offsets; item i is jdata[joffsets[i]:joffsets[i+1]]
jt.num_tensors # int — number of items in the batch
jt.rshape      # tuple — shape of the non-jagged dimensions

# --- Indexing ---
jt[0]          # JaggedTensor with 1 item
jt[1:3]        # JaggedTensor with 2 items
jt.unbind()    # List[Tensor]

# --- Operations (type-safe fvdb.* wrappers) ---
fvdb.relu(jt)
fvdb.sum(jt, dim=-1)
fvdb.mean(jt, dim=-1)
jt_a = fvdb.JaggedTensor([torch.randn(100, 8),  torch.randn(150, 8)])
jt_b = fvdb.JaggedTensor([torch.randn(100, 16), torch.randn(150, 16)])
fvdb.jcat([jt_a, jt_b], dim=1)   # concat along feature dim (same batch structure required)
```

---

## GridBatch

Sparse voxel grid batch. Stores topology only — features live in a separate `JaggedTensor`.

### Building Grids

```python continuation
# From point cloud — points snap to nearest voxel, duplicates deduplicated
pts_JT = fvdb.JaggedTensor([torch.randn(500, 3).cuda(), torch.randn(800, 3).cuda()])
grid = fvdb.GridBatch.from_points(
    pts_JT,              # JaggedTensor[B, N_i, 3] float
    voxel_sizes=0.1,     # scalar | [sx,sy,sz] | [[sx,sy,sz], ...] one per grid
    origins=[0., 0., 0.] # world coords of center of voxel (0,0,0); default zeros
)

# From integer ijk coordinates directly
ijk_JT = fvdb.JaggedTensor([torch.randint(-10, 10, (200, 3)).long().cuda()])
grid = fvdb.GridBatch.from_ijk(
    ijk_JT,              # JaggedTensor[B, N_i, 3] int64
    voxel_sizes=0.1,
    origins=[0., 0., 0.]
)

# From triangle mesh surface (triangle soup — no watertightness needed)
from fvdb.utils.examples import load_car_1_mesh
v, f = load_car_1_mesh(mode="vf")
vertices_JT = fvdb.JaggedTensor([v.float().cuda()])
faces_JT    = fvdb.JaggedTensor([f.long().cuda()])
grid = fvdb.GridBatch.from_mesh(
    vertices_JT,         # JaggedTensor[B, V_i, 3] float
    faces_JT,            # JaggedTensor[B, F_i, 3] int64
    voxel_sizes=0.1,
    origins=[0., 0., 0.]
)

# From nearest voxels — activates up to eight nearby voxels (2×2×2) per input point
grid = fvdb.GridBatch.from_nearest_voxels_to_points(pts_JT, voxel_sizes=0.1)

# Fully dense W×H×D box (dense_dims = [W, H, D])
grid = fvdb.GridBatch.from_dense(num_grids=2, dense_dims=[32,32,32], device='cuda')
```

### Key Properties

```python continuation
# Rebuild a point-based grid for the remaining examples
grid = fvdb.GridBatch.from_points(pts_JT, voxel_sizes=0.1)

grid.grid_count          # int — number of grids in the batch
grid.total_voxels        # int — total active voxels across all grids
grid.num_voxels_at(0)    # int — active voxels in grid 0
grid.voxel_sizes         # Tensor[B, 3]
grid.origins             # Tensor[B, 3]
grid.bboxes              # Tensor[B, 2, 3] — min/max ijk corners per grid
grid.ijk                 # JaggedTensor (batched, per-grid voxel counts); .jdata is int32 Tensor[total_voxels, 3]
```

### Attaching Features

```python continuation
# Wrap a flat Tensor with the grid's batch index structure — no data copy
C = 8
features = grid.jagged_like(torch.zeros(grid.total_voxels, C, device='cuda'))
# → JaggedTensor with jidx/joffsets matching grid.ijk
```

### Coordinate Transforms

```python continuation
# World-space xyz → index-space ijk (float, for interpolation)
ijk_float = grid.world_to_voxel(pts_JT)      # JaggedTensor[N, 3] float

# Index-space ijk (float) → world-space xyz
xyz = grid.voxel_to_world(grid.ijk.float())  # JaggedTensor[N, 3] float

# Integer ijk → per-grid linear index (-1 if voxel not active)
idx = grid.ijk_to_index(grid.ijk)                        # JaggedTensor[N] int64, per-grid
# For a single flat index space across the whole batch (to index grid.ijk.jdata):
flat_idx = grid.ijk_to_index(grid.ijk, cumulative=True)  # JaggedTensor[N] int64, batch-global

# Flat index → ijk: index into grid.ijk.jdata with cumulative indices
valid = flat_idx.jdata != -1
grid.ijk.jdata[flat_idx.jdata[valid]]       # Tensor[K, 3], K = valid.sum()
```

### Sampling and Splatting

```python continuation
# Interpolate per-voxel features at arbitrary world-space points
# Differentiable w.r.t. vox_feat
vox_feat_JT = grid.jagged_like(torch.randn(grid.total_voxels, 3, device='cuda'))
sampled = grid.sample_trilinear(pts_JT, vox_feat_JT)  # → JaggedTensor[N, C]
sampled = grid.sample_bezier(pts_JT, vox_feat_JT)     # smoother, higher-order

# Scatter point features onto voxels (adjoint of sample)
# Differentiable w.r.t. point_feat
point_feat_JT = fvdb.JaggedTensor([torch.randn(500, 3).cuda(), torch.randn(800, 3).cuda()])
vox_feat = grid.splat_trilinear(pts_JT, point_feat_JT)      # → JaggedTensor; .jdata shape [grid.total_voxels, C]
vox_feat = grid.splat_bezier(pts_JT, point_feat_JT)
```

### Spatial Queries

```python continuation
mask = grid.points_in_grid(pts_JT)          # JaggedTensor of bool — is each point in an active voxel?
mask = grid.coords_in_grid(grid.ijk)        # JaggedTensor of bool — is each ijk coord active?
```

### Topology Operations

```python continuation
grid2 = grid.coarsened_grid(2)         # merge 2×2×2 blocks; voxel_sizes × factor
grid2 = grid.refined_grid(2)          # split each voxel; voxel_sizes / factor
grid2 = grid.dilated_grid(1)          # add all voxels within Chebyshev distance (26-neighbors per shell)
grid2 = grid.dual_grid()              # voxels at corners of primal voxels — useful for SDF
```

---

## fvdb.nn Layers

Every `fvdb.nn` layer takes explicit `(data: JaggedTensor, grid_or_plan)` arguments. Topology and features are always passed separately — there is no wrapper object.

### ConvolutionPlan

Build a plan before calling any conv layer. Plans pre-compute the kernel map and are expensive; cache and reuse them.

```python continuation
fine_grid = grid  # use our existing grid as the fine-resolution grid

# Same-topology stride=1: pass target_grid=source_grid
plan_same = ConvolutionPlan.from_grid_batch(
    kernel_size=3, stride=1,
    source_grid=fine_grid, target_grid=fine_grid
)

# Stride=2 downsampling: target_grid=None auto-computes coarser grid
plan_down = ConvolutionPlan.from_grid_batch(
    kernel_size=2, stride=2,
    source_grid=fine_grid, target_grid=None
)
coarse_grid = plan_down.target_grid_batch   # retrieve auto-computed topology

# Transposed conv (upsampling): separate factory, supply fine target grid
plan_up = ConvolutionPlan.from_grid_batch_transposed(
    kernel_size=2, stride=2,
    source_grid=coarse_grid, target_grid=fine_grid
)

# Useful properties
plan_same.source_grid_batch   # GridBatch — input topology
plan_same.target_grid_batch   # GridBatch — output topology
```

### Convolution

```python continuation
in_channels, out_channels = 8, 16
feat_JT = fine_grid.jagged_like(torch.randn(fine_grid.total_voxels, in_channels, device='cuda'))

# Stride=1: same topology in and out
conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=3, stride=1, bias=True).cuda()
feat_out = conv(feat_JT, plan_same)          # JaggedTensor

# Stride=2 downsampling: output grid is coarser (~1/8 voxels)
down = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=2, stride=2).cuda()
feat_coarse = down(feat_JT, plan_down)       # JaggedTensor on coarser grid

# Transposed (upsampling): use SparseConvTranspose3d — separate class
up = fvnn.SparseConvTranspose3d(out_channels, in_channels, kernel_size=2, stride=2).cuda()
feat_fine = up(feat_coarse, plan_up)         # JaggedTensor on fine_grid

# 1×1×1 conv: per-voxel feature projection, no spatial context
head = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=1).cuda()
```

### Normalization and Activation

```python continuation
bn = fvnn.BatchNorm(out_channels, momentum=0.1)   # signature: forward(data, grid) -> JaggedTensor
gn = fvnn.GroupNorm(num_groups=4, num_channels=out_channels)  # signature: forward(data, grid) -> JaggedTensor
relu = torch.nn.ReLU(inplace=False)                # fvdb.nn has no ReLU — use torch.nn directly
```

### Pooling

```python continuation
# MaxPool / AvgPool: forward(fine_data, fine_grid, coarse_grid=None) -> (JaggedTensor, GridBatch)
pooled_feat, pooled_grid = fvnn.MaxPool(kernel_size=2)(feat_JT, fine_grid)
pooled_feat, pooled_grid = fvnn.AvgPool(kernel_size=2)(feat_JT, fine_grid)

# UpsamplingNearest: forward(coarse_data, coarse_grid, mask=None, fine_grid=None)
#                    -> (JaggedTensor, GridBatch)
up_feat, up_grid = fvnn.UpsamplingNearest(scale_factor=2)(pooled_feat, pooled_grid, fine_grid=fine_grid)
```

---

## Low-Level: ConvolutionPlan.execute

Use when you need direct control over the kernel map (e.g. custom conv loops).

```python continuation
# weights shape: [out_channels, in_channels, kx, ky, kz]
weights = torch.randn(out_channels, in_channels, 3, 3, 3, device=fine_grid.device)

out_feat = plan_same.execute(feat_JT, weights)   # differentiable
```

The kernel map is expensive to compute — construct and reuse `plan` across forward passes. `fvnn.SparseConv3d` does not automatically build or cache plans; you must pass one explicitly.

---

## Example Data Loaders

```python continuation
from fvdb.utils.examples import (
    load_car_1_mesh,   # → (vertices, faces_or_normals) depending on mode
    load_car_2_mesh,
    load_happy_mesh,   # Stanford Happy Buddha
)

# mode="vf" → (vertices Tensor[V,3], faces Tensor[F,3] int)
# mode="vn" → (vertices Tensor[V,3], normals Tensor[V,3])
v, f = load_car_1_mesh(mode="vf")
v, n = load_car_1_mesh(mode="vn")
```

---

## Common Patterns

```python continuation
# Build grid from point cloud, attach splatted normals as input features
pts1, nrm1 = torch.randn(500, 3), torch.randn(500, 3)
pts2, nrm2 = torch.randn(800, 3), torch.randn(800, 3)
pts_JT  = fvdb.JaggedTensor([pts1.cuda(), pts2.cuda()])
nrm_JT  = fvdb.JaggedTensor([nrm1.cuda(), nrm2.cuda()])
grid0   = fvdb.GridBatch.from_points(pts_JT, voxel_sizes=0.05)
feat_JT = grid0.splat_trilinear(pts_JT, nrm_JT)   # JaggedTensor [total_voxels, 3]

# conv → bn → relu  (the standard building block)
conv = fvnn.SparseConv3d(3, 16, kernel_size=3, stride=1).cuda()
bn   = fvnn.BatchNorm(16).cuda()
relu = torch.nn.ReLU(inplace=True)
plan = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid0, target_grid=grid0)
feat0 = relu(bn(conv(feat_JT, plan), grid0))       # all JaggedTensors

# Loss computation — access .jdata directly from the output JaggedTensor
target_JT = feat0  # for demonstration
loss = torch.nn.functional.mse_loss(feat0.jdata, target_JT.jdata)

# Typical U-Net encoder/decoder with explicit plans and grids
#   Encoder
enc1  = fvnn.SparseConv3d(16, 32, kernel_size=2, stride=2).cuda()
bn1   = fvnn.BatchNorm(32).cuda()
plan_e1  = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid0, target_grid=None)
feat1    = relu(bn1(enc1(feat0, plan_e1), plan_e1.target_grid_batch))
grid1    = plan_e1.target_grid_batch          # save for decoder

#   Decoder
dec0  = fvnn.SparseConvTranspose3d(32, 16, kernel_size=2, stride=2).cuda()
bn_d  = fvnn.BatchNorm(16).cuda()
plan_d0  = ConvolutionPlan.from_grid_batch_transposed(2, 2,
               source_grid=grid1, target_grid=grid0)
feat_up  = relu(bn_d(dec0(feat1, plan_d0), grid0))
feat_up  = fvdb.jcat([feat_up, feat0], dim=1) # skip connection (feature dim)

# Random subsample of a point cloud
n_pts = 1000
perm = torch.randperm(v.shape[0])[:n_pts]
pts  = v[perm].cuda()
```
