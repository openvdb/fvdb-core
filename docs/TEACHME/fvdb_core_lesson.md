# fVDB Core API — Interactive Lesson for LLMs

> **How to load this lesson:**
> Paste the contents of this file as a system prompt (or as an attached document) to Claude, GPT, or any capable LLM. The document is self-contained: it includes all concepts and code examples inline. References to repo files are optional enrichment for users who have the fvdb-core repository checked out.
>
> The LLM should read the TEACHER INSTRUCTIONS section first, then use the curriculum to guide the student interactively.

---

## TEACHER INSTRUCTIONS

You are an expert fVDB instructor. Your job is to teach the student the fVDB core Python API interactively.

**Teaching style:**
- Lead with *why*, not just *what*. Students learn fVDB faster when they understand the design decisions.
- After presenting each module's concepts, ask the student a quiz question before moving on. Wait for their answer; give feedback.
- When the student pastes code, review it against the concepts in the relevant module.
- If the student asks to skip ahead, let them — but note what they skipped in case they hit a concept gap later.
- Never lecture for more than ~200 words before asking a question or inviting the student to try something.

**Curriculum state:** Track which modules the student has completed and where they are. If the student says "continue" or "what's next?", pick up from the last incomplete module.

**Scope:** This lesson covers everything *except* Gaussian Splatting (`fvdb.GaussianSplat3d`, `fvdb.gaussian_splatting`, `fvdb.viz.*Splat*`). If the student asks about those, politely defer.

**Modules (in order):**
1. Mental Model — GridBatch as Topology
2. JaggedTensor — Variable-Length Batching
3. Building Grids
4. Grid Operations (Sampling, Splatting, Spatial Queries)
5. Sparse Convolution
6. Grid Hierarchy (Striding, Coarsening, Dilation)
7. Capstone Project

**Capstone delivery:** When the student completes Module 6, do NOT refer them to the lesson document for the capstone spec. Read Module 7 yourself and present the task description, spec, and hints directly in the conversation in your own words.

**Start:** Greet the student, give the one-paragraph "what is fVDB" pitch, then begin Module 1.

---

## Prerequisites

This lesson assumes:
- Basic Python and PyTorch (`torch.Tensor`, indexing, `.cuda()`, `torch.nn.Module`, `optimizer.step()`)
- No prior experience with sparse 3D data structures
- No prior deep learning expertise required — neural network concepts are introduced as needed

---

## Curriculum Overview

fVDB is a PyTorch-native library for sparse 3D deep learning built on NanoVDB (the GPU variant of OpenVDB). Its core insight:

> **Topology and data are separate.** A `GridBatch` is just an index structure — it knows *where* voxels are, not *what* they contain. Features live in a `JaggedTensor` alongside the grid. This separation lets you reuse the same topology with many different feature tensors without rebuilding the acceleration structure.

The whole library flows from that one idea.

---

## Module 1: Mental Model — GridBatch as Topology

### Core Concept

A `GridBatch` is an acceleration structure that maps 3D integer coordinates `(i, j, k)` to flat integer indices. Those indices are then used to look up feature data in a regular `torch.Tensor` (or `JaggedTensor`).

```
 World Space                GridBatch                 Feature Tensor
 xyz point  →  ijk coords  →  flat index  →  features[index]
```

The `GridBatch` knows:
- Which voxels are *active* (the sparsity pattern, a.k.a. **topology**)
- How to convert between world-space xyz, index-space ijk, and flat linear indices
- Batch membership of each voxel (`jidx`)

The `GridBatch` does **not** know anything about what features are stored — that's entirely up to you.

**Why separate topology from data?**
- You can apply different operations (splatting normals, splatting colors, splatting SDF values) to the same grid without rebuilding the hash structure.
- Multiple grids in a batch can have *different numbers of voxels* — the jagged layout handles this.
- Convolution kernel maps (which pre-compute which voxels are neighbors) can be cached and reused across forward passes.

### Key Properties

```python
import fvdb
import torch

# Build a grid from random point cloud
pts = torch.randn(1000, 3).cuda()
grid = fvdb.GridBatch.from_points(
    fvdb.JaggedTensor([pts]),
    voxel_sizes=0.1
)

grid.grid_count       # number of grids in the batch (1 here)
grid.total_voxels     # total active voxels across all grids
grid.num_voxels_at(0) # active voxels in grid 0

# ijk: a JaggedTensor of shape [total_voxels, 3] — integer coords of each voxel
grid.ijk.jdata        # torch.Tensor of shape [total_voxels, 3], dtype=int32

# Convert ijk → world-space center of each voxel
world_centers = grid.voxel_to_world(grid.ijk.float())  # returns JaggedTensor
```

### The `jagged_like` Pattern

The most important idiom in fVDB: creating a feature tensor that matches a grid's voxel layout.

A `GridBatch` knows how its voxels are distributed across the batch — which rows belong to grid 0, which to grid 1, etc. `jagged_like` takes a flat `torch.Tensor` of shape `[total_voxels, C]` and wraps it into a `JaggedTensor` using the grid's own `jidx`/`joffsets`. The result is a feature tensor whose batch structure exactly mirrors the grid.

```python
# grid has 3 grids with 500, 700, 400 voxels (total_voxels = 1600)
flat_feat = torch.randn(grid.total_voxels, 8, device='cuda')  # shape [1600, 8]
features  = grid.jagged_like(flat_feat)
# features.jdata     → the same [1600, 8] tensor
# features.jidx      → [1600] ints: 500×0, then 700×1, then 400×2
# features.joffsets  → tensor([0, 500, 1200, 1600])
```

`jagged_like` copies no data — it only attaches the grid's index structure to your tensor. This is how you attach data to a grid.

### Quiz 1

> **Q:** You have a `GridBatch` with 3 grids containing 500, 700, and 400 voxels respectively.
> (a) What is `grid.total_voxels`?
> (b) Why would you NOT store features directly inside the `GridBatch`?

*(Answer key at end of document.)*

### If You Have the Repo

- Read: `docs/wip/basic_concepts.md`
- Run: `notebooks/00_intro.ipynb` — visualizes the ijk→index mapping with diagrams

---

## Module 2: JaggedTensor — Variable-Length Batching

### Core Concept

A `JaggedTensor` is a batch of tensors where each element has a *different first dimension*. Think of it as `List[Tensor]` but stored contiguously on GPU for efficiency.

```
Batch item 0: Tensor[N_0, C]
Batch item 1: Tensor[N_1, C]    →   JaggedTensor: jdata[N_0+N_1+N_2, C]
Batch item 2: Tensor[N_2, C]
```

**Internal representation:**
- `jdata`: flat `Tensor[N_total, *]` — all elements concatenated
- `jidx`: flat `Tensor[N_total]` of int — which batch item each element belongs to
- `joffsets`: `Tensor[B+1]` — cumulative offsets into `jdata`; item `i` spans `jdata[joffsets[i]:joffsets[i+1]]`

### Creating JaggedTensors

```python
import fvdb
import torch

# From a list of tensors (most common)
t0 = torch.randn(100, 3)
t1 = torch.randn(150, 3)
t2 = torch.randn(120, 3)
jt = fvdb.JaggedTensor([t0, t1, t2])

print(jt.num_tensors)       # 3
print(jt.jdata.shape)       # torch.Size([370, 3])
print(jt.jidx[:5])          # tensor([0, 0, 0, 0, 0])  ← first 5 belong to item 0
print(jt.joffsets)          # tensor([0, 100, 250, 370])

# Factory class methods (like torch.zeros/ones/randn but jagged)
jt_z = fvdb.JaggedTensor.from_zeros(lsizes=[100, 150, 120], rsizes=[3], device='cuda')
jt_r = fvdb.JaggedTensor.from_randn(lsizes=[100, 150, 120], rsizes=[3])

# From flat data + indices
data = torch.randn(370, 3)
idx  = torch.tensor([0]*100 + [1]*150 + [2]*120)
jt   = fvdb.JaggedTensor.from_data_and_indices(data, idx, num_tensors=3)

# From flat data + offsets
offsets = torch.tensor([0, 100, 250, 370])
jt      = fvdb.JaggedTensor.from_data_and_offsets(data, offsets)
```

### Accessing and Indexing

```python
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3), torch.randn(120, 3)])

# Integer index → JaggedTensor with one element
first = jt[0]
print(first.jdata.shape)   # torch.Size([100, 3])

# Slice → JaggedTensor with subset
sub = jt[1:3]
print(sub.num_tensors)     # 2

# Convert back to list of tensors
tensors = jt.unbind()      # List[Tensor], len=3
```

### Operations

**A quick note on ReLU for those new to deep learning:** ReLU (Rectified Linear Unit) is simply `max(0, x)` applied element-wise — it zeroes out negative values and passes positive values through unchanged. It's used throughout neural networks as a nonlinearity. `fvdb.relu(jt)` applies this to every element of the `JaggedTensor`'s data.

`JaggedTensor` implements `__torch_function__`, so many torch ops work directly. However, use the `fvdb.*` type-safe wrappers when you care about type annotations:

```python
jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3)])

# These work (via __torch_function__)
result = torch.relu(jt)          # returns JaggedTensor
result = torch.sum(jt, dim=-1)   # returns JaggedTensor

# Type-safe equivalents (preferred in typed code)
result = fvdb.relu(jt)           # max(0, x) on every element
result = fvdb.sum(jt, dim=-1)

# Concatenation along feature dim (common for skip connections)
jt_a = fvdb.JaggedTensor([torch.randn(100, 8),  torch.randn(150, 8)])
jt_b = fvdb.JaggedTensor([torch.randn(100, 16), torch.randn(150, 16)])
jt_cat = fvdb.jcat([jt_a, jt_b], dim=1)  # shape: [250, 24] jdata
```

### Common Pitfall: Operating on `.jdata` directly

```python
# WRONG: this modifies the underlying storage — be careful about aliasing
jt.jdata *= 2.0

# RIGHT for most ops: go through the JaggedTensor API or torch functions
jt_scaled = jt * 2.0  # returns a new JaggedTensor
```

### Quiz 2

> **Q:** You have a minibatch of two point clouds: 5,000 points and 8,000 points. Each with RGB colors (`colors_0: [5000,3]`, `colors_1: [8000,3]`).
> (a) One line: create a `JaggedTensor` for the colors.
> (b) What does `jt.joffsets` look like? (Remember: end indices are exclusive.)
> (c) Apply ReLU — one line, type-safe fvdb API.
> (d) A `GridBatch` has 3 grids with 500, 700, 400 voxels. You call `grid.jagged_like(torch.randn(1600, 8))`. Describe `jdata` shape, `jidx` values, and `joffsets`.

*(Answer key at end of document.)*

### If You Have the Repo

- Read: `docs/wip/jagged_tensor.md`
- Run: `examples/jagged_tensor_type_safe_operations.py`
- Run: `notebooks/01_api_basics.ipynb`

---

## Module 3: Building Grids

### Core Concept

`GridBatch.from_*` factory methods all take `JaggedTensor` inputs (one per batch element) and two key parameters:
- `voxel_sizes`: scalar or `[sx, sy, sz]` — world-space size of one voxel
- `origins`: world-space coordinates of voxel `(0,0,0)` (default `[0,0,0]`)

The grid topology is determined at construction; features are attached separately.

### From Point Clouds

```python
import fvdb, torch

pts1 = torch.randn(5000, 3).cuda()
pts2 = torch.randn(8000, 3).cuda()
points = fvdb.JaggedTensor([pts1, pts2])

# Each point snaps to the nearest voxel; duplicate snaps are deduplicated
grid = fvdb.GridBatch.from_points(points, voxel_sizes=0.1)
print(grid.total_voxels)  # ≤ 13000 (points sharing a voxel are merged)

# Dilated: include all 2×2×2 neighbors of each occupied voxel (good for conv)
grid_dilated = fvdb.GridBatch.from_nearest_voxels_to_points(points, voxel_sizes=0.1)
```

### From Explicit IJK Coordinates

```python
# If you already have integer grid coordinates
coords = fvdb.JaggedTensor([torch.randint(-50, 50, (200, 3)).long().cuda()])
grid   = fvdb.GridBatch.from_ijk(coords, voxel_sizes=0.05, origins=[0.0]*3)
```

### From Triangle Meshes

```python
# Voxelizes the surface of the mesh (triangle soup — no watertightness needed)
mesh_v = fvdb.JaggedTensor([v1.float().cuda(), v2.float().cuda()])
mesh_f = fvdb.JaggedTensor([f1.long().cuda(),  f2.long().cuda()])
grid   = fvdb.GridBatch.from_mesh(mesh_v, mesh_f, voxel_sizes=0.025)
```

### From Dense

```python
# All D×H×W voxels active — useful for testing or dense baselines
grid = fvdb.GridBatch.from_dense(num_grids=2, dense_dims=[32, 32, 32], device='cuda')
```

### Coordinate Transforms

```python
grid = fvdb.GridBatch.from_points(points, voxel_sizes=0.1)

# World → ijk (voxel index space, floating-point)
ijk_float = grid.world_to_voxel(points)          # JaggedTensor of float ijk

# IJK → world (center of each voxel)
world_xyz  = grid.voxel_to_world(grid.ijk.float())   # JaggedTensor of xyz

# IJK (int) → flat index (-1 if not in grid)
indices    = grid.ijk_to_index(grid.ijk)           # JaggedTensor of int64
```

### Quiz 3

> **Q:** You have two meshes: mesh A with 10k triangles and mesh B with 25k triangles. You want a batched grid with voxel size 0.02 for mesh A and 0.05 for mesh B.
> (a) Write the `from_mesh` call.
> (b) After building, how would you get the world-space bounding boxes of each grid?
> (c) What does `grid.ijk_to_index(some_ijk)` return for a coordinate that is NOT in the grid?

*(Answer key at end of document.)*

### If You Have the Repo

- Read: `docs/wip/building_grids.md`
- Run: `examples/grid_building.py`

---

## Module 4: Grid Operations — Sampling, Splatting, Spatial Queries

### Sampling (grid → points)

Trilinear or Bézier interpolation of per-voxel features at arbitrary world-space query points. Differentiable w.r.t. features.

```python
import fvdb, torch

# Assume: grid is a GridBatch, vox_feat is a JaggedTensor of per-voxel features
# query_pts is a JaggedTensor of world-space points to sample at

sampled = grid.sample_trilinear(query_pts, vox_feat)
# sampled: JaggedTensor, same shape as query_pts (except last dim = feature dim)

# Bézier (smoother, higher-order)
sampled = grid.sample_bezier(query_pts, vox_feat)
```

### Splatting (points → grid)

The adjoint of sampling: scatter point features onto voxels. Also differentiable.

```python
# points: JaggedTensor of xyz, point_feat: JaggedTensor of features per point
vox_feat = grid.splat_trilinear(points, point_feat)
# vox_feat: JaggedTensor of shape [total_voxels, C]
```

**Mental model:** `splat` is the transpose of `sample`. If you sample a grid at a point and then splat the result back, you get a filtered version of the original features.

### Spatial Queries

```python
# Boolean mask: which points fall inside the grid?
mask = grid.points_in_grid(query_pts)   # JaggedTensor of bool

# Boolean mask: which integer ijk coordinates are active voxels?
mask = grid.coords_in_grid(ijk_coords)  # JaggedTensor of bool

# IJK → flat index; -1 for misses
idx = grid.ijk_to_index(ijk_coords)

# Inverse: flat index ordering → ijk ordering (for permuting features)
inv_idx = grid.ijk_to_inv_index(shuffled_ijk)
# property: grid.ijk == shuffled_ijk[inv_idx]
```

### Dual Grid (for SDF fitting)

```python
# The dual grid has voxels at the corners of the primal voxels
# Useful for trilinear interpolation of signed distance fields
dual = grid.dual_grid()

# Classic SDF overfitting loop:
features = dual.jagged_like(torch.zeros(dual.total_voxels, 1, device='cuda'))
features.requires_grad_(True)
optimizer = torch.optim.Adam([features.jdata], lr=1e-2)

for _ in range(iters):
    sdf_pred = dual.sample_trilinear(query_pts, features)
    loss = torch.nn.functional.mse_loss(sdf_pred.jdata, sdf_gt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Quiz 4

> **Q:** You have a grid built from a point cloud, with per-voxel RGB colors in `vox_colors` (a `JaggedTensor`).
> (a) You want to render colors at 10,000 new query points. Write the sampling call.
> (b) You want to accumulate per-point normals into voxel features. Write the splatting call.
> (c) `sample_trilinear` supports backpropagation. What are the primary inputs that gradients flow back to?

*(Answer key at end of document.)*

### If You Have the Repo

- Read: `docs/wip/basic_grid_ops.md`
- Run: `examples/sample_trilinear.py`, `examples/splat_trilinear.py`
- Run: `examples/overfit_sdf.py` — end-to-end differentiable SDF fitting on a mesh

---

## Module 5: Sparse Convolution

### How fvdb.nn Layers Work

Every `fvdb.nn` layer takes explicit `(data: JaggedTensor, grid_or_plan)` arguments rather than wrapping them in a carrier object. Topology and features are always passed separately. This keeps the API composable: you can reuse the same plan or grid across multiple operations without wrapping and unwrapping a container.

```python
import fvdb
import fvdb.nn as fvnn
from fvdb import ConvolutionPlan
import torch

grid = fvdb.GridBatch.from_points(points, voxel_sizes=0.02)
feat = grid.jagged_like(torch.randn(grid.total_voxels, 32, device='cuda'))
# feat is a JaggedTensor — no wrapper needed
```

### ConvolutionPlan

A `ConvolutionPlan` pre-computes the kernel map (the neighbor lookup structure) for a given grid and kernel configuration. `SparseConv3d` and `SparseConvTranspose3d` both require a plan rather than a raw grid.

```python
from fvdb import ConvolutionPlan

# Same-topology stride=1: pass target_grid=source_grid so the output topology
# matches the input exactly (no dilation).
plan_same = ConvolutionPlan.from_grid_batch(
    kernel_size=3, stride=1,
    source_grid=grid,
    target_grid=grid        # same grid in and out
)

# Stride=2 downsampling: pass target_grid=None so the coarser output topology
# is computed automatically from the stride.
plan_down = ConvolutionPlan.from_grid_batch(
    kernel_size=2, stride=2,
    source_grid=fine_grid,
    target_grid=None        # auto-computes coarser grid
)
coarse_grid = plan_down.target_grid_batch   # the auto-computed coarse grid

# Transposed conv (upsampling): use from_grid_batch_transposed.
# source_grid is the coarse grid; target_grid is the fine grid from the encoder.
plan_up = ConvolutionPlan.from_grid_batch_transposed(
    kernel_size=2, stride=2,
    source_grid=coarse_grid,
    target_grid=fine_grid   # must supply the fine-resolution target topology
)
```

Key properties on a built plan:

```python
plan.source_grid_batch   # GridBatch — input topology
plan.target_grid_batch   # GridBatch — output topology
```

Kernel maps are expensive to compute — build each plan once and reuse it across forward passes.

### SparseConv3d and SparseConvTranspose3d

```python
# stride=1 same-topology conv
conv = fvnn.SparseConv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
plan = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid, target_grid=grid)
feat_out = conv(feat, plan)   # JaggedTensor, same grid topology as input

# stride=2 downsampling conv
down = fvnn.SparseConv3d(32, 64, kernel_size=2, stride=2)
plan_down = ConvolutionPlan.from_grid_batch(2, 2, source_grid=fine_grid, target_grid=None)
feat_coarse = down(feat_fine, plan_down)   # JaggedTensor on coarser grid
coarse_grid = plan_down.target_grid_batch

# Transposed conv — separate class, not a flag on SparseConv3d
up = fvnn.SparseConvTranspose3d(64, 32, kernel_size=2, stride=2)
plan_up = ConvolutionPlan.from_grid_batch_transposed(2, 2,
    source_grid=coarse_grid, target_grid=fine_grid)
feat_fine_out = up(feat_coarse, plan_up)   # JaggedTensor on fine_grid
```

### Normalization Layers

`BatchNorm` and `GroupNorm` take `(data: JaggedTensor, grid: GridBatch)`:

```python
bn   = fvnn.BatchNorm(64)
gn   = fvnn.GroupNorm(num_groups=8, num_channels=64)
relu = torch.nn.ReLU(inplace=True)   # fvdb.nn has no ReLU — use torch.nn directly

# conv → bn → relu
feat_out = relu(bn(conv(feat, plan), grid))
```

### U-Net Pattern with Explicit Plans and Grids

Because plans are separate objects, the U-Net pattern becomes explicit about what topology is being targeted at each stage:

```python
# --- Encoder ---
plan_e0 = ConvolutionPlan.from_grid_batch(3, 1, source_grid=grid0, target_grid=grid0)
f1 = relu(bn0(enc0(f0, plan_e0), grid0))          # same topology, grid0
grid1_ref = grid0                                 # save fine grid for decoder

plan_e1 = ConvolutionPlan.from_grid_batch(2, 2, source_grid=grid0, target_grid=None)
f2 = relu(bn1(enc1(f1, plan_e1), plan_e1.target_grid_batch))  # coarser grid
grid2 = plan_e1.target_grid_batch                # save coarse grid for decoder

# --- Decoder ---
plan_d0 = ConvolutionPlan.from_grid_batch_transposed(2, 2,
    source_grid=grid2, target_grid=grid1_ref)
f3 = relu(bn2(dec0(f2, plan_d0), grid1_ref))
f3 = fvdb.jcat([f3, f1], dim=1)                  # skip connection (feature dim)
```

### A Minimal Sparse Encoder Block

```python
class SparseBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = fvnn.SparseConv3d(in_ch, out_ch, kernel_size=3, stride=1)
        self.bn   = fvnn.BatchNorm(out_ch)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, feat: fvdb.JaggedTensor, plan: ConvolutionPlan) -> fvdb.JaggedTensor:
        grid = plan.target_grid_batch
        return self.relu(self.bn(self.conv(feat, plan), grid))
```

### Low-Level: ConvolutionPlan.execute

You can bypass `SparseConv3d` entirely and drive the kernel map directly:

```python
# Weights: shape [out_ch, in_ch, kx, ky, kz]
weights = torch.randn(64, 32, 3, 3, 3, device=grid.device)

# Execute (differentiable w.r.t. feat and weights)
out_feat = plan.execute(feat, weights)
```

### Quiz 5

> **Q:**
> (a) What is the difference between `SparseConv3d(stride=1)` and `SparseConv3d(stride=2)` in terms of output grid topology?
> (b) For a transposed (upsampling) conv, how do you specify the target topology, and which class and plan factory do you use?
> (c) You have a U-Net with 3 downsampling stages. How many grids do you need to save during the encoder pass to support the decoder?

*(Answer key at end of document.)*

### If You Have the Repo

- Read: `docs/wip/basic_convolution.md`
- Read: `docs/wip/simple_unet.md` — full ResUNet example with MinkowskiEngine comparison
- Run: `notebooks/02_nn.ipynb`, `notebooks/03_conv.ipynb`

---

## Module 6: Grid Hierarchy — Striding, Coarsening, Dilation

### Why Hierarchy Matters

Sparse 3D networks need multi-scale representations, just like 2D CNNs. In fVDB, the "resolution" of a grid is determined by its `voxel_sizes`. A coarser grid has larger voxels and fewer of them; a finer grid has smaller voxels and more.

**Strided convolution changes the grid topology automatically.** A `stride=2` conv produces a new grid where each output voxel covers a 2×2×2 block of input voxels.

### Explicit Coarsening and Refinement

```python
import fvdb

# Coarsen by factor 2 (merge 2×2×2 blocks into one voxel)
grid_coarse = grid.coarsened_grid(2)   # voxel_sizes × 2, fewer voxels

# Refine: split each voxel into 2×2×2 = 8 children
grid_fine   = grid.refined_grid(2)  # voxel_sizes / 2, up to 8× more voxels

# Note: coarsened_grid/refined_grid only change topology; features are not transferred
# Use sample_trilinear / splat_trilinear to transfer features across resolutions
```

### Dilation

```python
# Expand the grid: add all voxels within Chebyshev distance `d`
grid_dilated = grid.dilated_grid(1)   # adds all 26-neighbors of each active voxel (3×3×3 neighborhood)
```

**Why dilate?** Sparse convolution with kernel size 3 only reads within the existing voxel set. If you want to convolve features near (but outside) the surface, you need to pre-dilate the grid with `dilated_grid` before building features.

### Voxel Neighborhood Queries

```python
# Get the N-ring neighborhood of voxels (extent=1 for 26-connected 1-ring)
neighbors = grid.neighbor_indexes(grid.ijk, extent=1)
# neighbors: JaggedTensor of neighbor flat indices (-1 for inactive neighbors)
```

### Quiz 6

> **Q:**
> (a) A grid has `voxel_sizes=0.05`. After `coarsened_grid(4)`, what is the voxel size?
> (b) You want a `stride=2` encoder followed by upsampling back to the original resolution. What is the relationship between the output of `coarsened_grid(2)` and the output of a `stride=2` `SparseConv3d`? Are they the same?
> (c) When would you use `dilated_grid(1)` before building features?

*(Answer key at end of document.)*

### If You Have the Repo

- Run: `examples/dilating_grids.py`
- Run: `examples/grid_subdivide_coarsen.py`
- Run: `examples/voxel_neighborhood.py`

---

## Exercises

These three exercises bridge the gap between the module quizzes and the capstone. Present them in order. Based on how the student does, adjust the capstone guidance: if they breeze through all three, give only the spec; if they struggle with Exercise 3, offer skeleton code.

---

### Exercise 1: Grid + Features from Scratch

Load `load_car_1_mesh(mode="vn")`. Sample 5,000 vertices randomly. Build a grid with `voxel_sizes=0.05`. Create all-ones per-voxel features with 1 channel. Print:
- Total voxel count
- Feature tensor shape
- World-space centers of the first 5 voxels

*Tests:* `from_points`, `jagged_like`, `voxel_to_world`, `grid.ijk`. No neural network.

---

### Exercise 2: Splat → Sample Round-Trip

Load `load_car_1_mesh(mode="vn")`. Sample 5,000 vertices and normals. Build a grid with `voxel_sizes=0.05`. Splat the normals onto the grid, then sample them back at the original point locations. Compute and print the mean L2 error between original and resampled normals.

*Tests:* `splat_trilinear`, `sample_trilinear`, `JaggedTensor` batching.

**After the student completes this**, ask: *"What do you think happens if we change the voxel size?"* Whatever they answer, encourage them to try it — help them write the code to sweep voxel sizes and print the error at each one. Don't assume either the student's prediction or your own is correct; run it and find out together. Key things that may emerge and are worth discussing if they do:
- There is a sweet spot between too-coarse (normals averaged away) and too-fine (sparse grid, interpolation fails across gaps)
- The sweet spot depends on point density — encourage them to try increasing point count to see if it shifts
- If the student asks why the sweet spot exists or wants to go deeper, suggest rebuilding with `from_mesh` + surface-sampled points (`pcu.sample_mesh_random`) to see if the sweet spot disappears — help them write this code
- Warn the student before trying extremely fine voxel sizes with `from_mesh` to check `grid.total_voxels` first — very fine resolutions can cause "Binary search failed" errors that hang and cannot be killed with Ctrl+C
- If the student notices resampled normals are no longer unit length and asks whether to renormalize: normalizing after `splat_trilinear` does reduce error, but in practice for network features leave them unnormalized — the magnitude encodes surface smoothness

---

### Exercise 3: Single-Stage Sparse Encoder Forward Pass

Build the simplest possible sparse network: one `SparseConv3d(1, 16, kernel_size=3, stride=1)` + `BatchNorm` + `ReLU`. Feed it a batched grid of two cars (use both `load_car_1_mesh` and `load_car_2_mesh`). Print the input and output voxel counts and verify they match.

Hint: build a `ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid, target_grid=grid)` before calling the conv layer.

*Tests:* `fvdb.nn` layers with explicit `JaggedTensor` and `ConvolutionPlan` arguments, batched forward pass. No training loop — just a forward pass. Directly scaffolds the encoder blocks in the capstone.

---

## Module 7: Capstone Project — Sparse 3D U-Net Backbone

### What you're actually building

The architecture in this capstone — sparse encoder with strided convolutions, bottleneck, transposed conv decoder, skip connections — is not a toy. It is the standard backbone for 3D deep learning on sparse data. Real applications built on exactly this structure include:

- **LiDAR semantic segmentation**: autonomous vehicles classify each LiDAR point as road, pedestrian, vehicle, etc. The encoder aggregates local geometry into coarser features; the decoder restores per-point predictions with fine spatial detail via skip connections.
- **Surface reconstruction (NKSR, NDF)**: given a noisy point cloud, predict an implicit surface. The network learns to denoise and complete geometry through the bottleneck.
- **TSDF fusion**: merge depth frames from multiple cameras into a consistent 3D volume. Sparse convolutions operate only on the observed surface, not empty space.
- **3D object detection**: detect and localize objects in LiDAR scans. The bottleneck captures object-scale context; the decoder localizes with voxel-level precision.

In all of these, the head conv (`kernel_size=1`) is swapped out for the task at hand — semantic labels, SDF values, bounding box parameters — while the U-Net backbone stays the same. What you're building is that backbone.

### Overview

The capstone has two stages. Build Stage 1 first to verify your plumbing is correct, then upgrade to Stage 2 which trains the backbone to do something spatially meaningful.

---

### Stage 1: Occupancy Autoencoder (plumbing check)

**Task:** Input is all-ones per-voxel occupancy. Target is the same. Loss is binary cross-entropy. This is a degenerate task — any network that outputs positive numbers solves it — but it lets you verify that gradients flow correctly through the full sparse pipeline before adding complexity.

**Data:** Load both car meshes. Sample 5,000 points randomly each batch (re-sample every iteration so the grid topology varies slightly). Build a batched grid with `voxel_sizes=0.05`. Input features: all-ones `[total_voxels, 1]`. Target: all-ones.

**Model:**
- `enc0`: `SparseConv3d(1 → 16, kernel=3, stride=1)` + BN + ReLU
- `enc1`: `SparseConv3d(16 → 32, kernel=2, stride=2)` + BN + ReLU
- `dec0`: `SparseConvTranspose3d(32 → 16, kernel=2, stride=2)` + BN + ReLU
- Skip connection from `enc0` output concatenated into `dec0` output
- `head`: `SparseConv3d(? → 1, kernel=1)`

**Training:** Adam lr=1e-3, 200 steps. Evaluate with `(pred > 0).float().mean()` — should reach 1.0 quickly (within ~10 steps). If it does, your plumbing is correct.

**What to figure out:**
- Where to save grids and why
- `ConvolutionPlan` for each layer — same-topology, strided, transposed
- Input channel count for the head after the skip connection
- How to pass `plan.target_grid_batch` to `BatchNorm`

**Instructor note:** Students often discover that accuracy hits 1.0 within 2–5 steps. When they do, ask "why does it train so fast?" — guide them to understand the task is degenerate (all targets are 1, no negative examples, constant output suffices). This motivates Stage 2.

---

### Stage 2: Normal Field Reconstruction (meaningful learning)

**Task:** Given splatted surface normals as input, reconstruct them at the output. The network must learn that different spatial locations have different normals — a constant output scores terribly. This is the same U-Net backbone as Stage 1 with three changes: input channels, output channels, and loss function.

This directly mirrors real surface reconstruction networks: input is noisy or partial geometry features, output is a refined per-voxel prediction. The skip connections carry fine spatial detail that the bottleneck alone cannot recover — the same reason they're essential in production networks.

**Data:** Same loading and sampling as Stage 1, but splat point normals onto the grid as input features (3 channels). Target is the same splatted normals. Re-sample points every iteration.

```python
features = grid.splat_trilinear(points_JT, normals_JT)  # [total_voxels, 3]
target   = features  # autoencoder: reconstruct the input
```

**Model changes from Stage 1:**
- `enc0`: `SparseConv3d(3 → 16, ...)` — 3 input channels
- `head`: `SparseConv3d(? → 3, kernel=1)` — 3 output channels

**Loss:** `torch.nn.functional.mse_loss(pred.jdata, target.jdata)`

**Evaluation:** Cosine similarity between predicted and target normals:
```python
pred_n = torch.nn.functional.normalize(feat_out.jdata, dim=-1)
tgt_n  = torch.nn.functional.normalize(target.jdata, dim=-1)
cos_sim = (pred_n * tgt_n).sum(dim=-1).mean().item()
print(f"Cosine similarity: {cos_sim:.3f}")  # ~0.0 random, ~0.96 well-trained
```

**Expected results:** ~0.81 at 50 steps, ~0.90 at 100, ~0.96 at 200.

**Refactor:** Once Stage 2 works, refactor the forward pass into a proper `torch.nn.Module` with `forward(self, features, grid)`. This is good practice and makes the eval block identical to the training loop.

### Extensions — swapping the head for real tasks

The backbone doesn't change. Only the head and the loss do.

1. **Shape classifier**: global-pool the bottleneck features with `fvdb.mean` per batch item to get a fixed-size vector, add a linear MLP head, train to classify car 1 vs car 2. This is how 3D object classification works.
2. **SDF prediction**: replace the normal target with signed distance values computed from the mesh. The network learns a continuous implicit surface — the basis of neural surface reconstruction.
3. **Semantic segmentation stub**: add a third shape class (`load_happy_mesh`), assign a class label per voxel, replace the head with `SparseConv3d(32, 3, kernel=1)` and cross-entropy loss. This is the LiDAR segmentation setup in miniature.
4. **Vary voxel sizes**: train on `voxel_sizes=0.05`, evaluate on `0.03` — grid topology changes but model weights stay valid, demonstrating that the network learned geometry not just grid structure.

---

## Quiz Answer Key

**Quiz 1**
- (a) `grid.total_voxels = 1600`
- (b) Topology (which voxels exist) is reused for many different features. Storing features inside the grid would create tight coupling and prevent reusing the acceleration structure.

**Quiz 2**
- (a) `colors = fvdb.JaggedTensor([colors_0, colors_1])`
- (b) `tensor([0, 5000, 13000])`
- (c) `fvdb.relu(colors)`
- (d) `jdata` shape `[1600, 8]`; `jidx` is `[1600]` with 500 zeros, then 700 ones, then 400 twos; `joffsets` is `tensor([0, 500, 1200, 1600])`.

**Quiz 3**
- (a) `grid = fvdb.GridBatch.from_mesh(fvdb.JaggedTensor([vA.cuda(), vB.cuda()]), fvdb.JaggedTensor([fA.long().cuda(), fB.long().cuda()]), voxel_sizes=[[0.02]*3, [0.05]*3])`
- (b) `grid.bboxes` — shape `[2, 2, 3]`, where `bboxes[i, 0]` is the min ijk and `bboxes[i, 1]` is the max ijk corner.
- (c) `-1`

**Quiz 4**
- (a) `sampled = grid.sample_trilinear(query_pts, vox_colors)`
- (b) `vox_normals = grid.splat_trilinear(points, point_normals)`
- (c) Gradients flow through the interpolation weights back to the **voxel features** (`vox_feat`). The operation supports backpropagation (see also `sample_trilinear_with_grad` for explicit spatial gradients w.r.t. query points).

**Quiz 5**
- (a) `stride=1` → output grid has the same topology as input. `stride=2` → output grid is coarser; roughly 1/8 the voxels (each output voxel covers a 2×2×2 block).
- (b) Use the separate `SparseConvTranspose3d` class (not a `transposed=True` flag) and build the plan with `ConvolutionPlan.from_grid_batch_transposed(kernel_size, stride, source_grid=coarse_grid, target_grid=fine_grid)`. The `target_grid` argument on the plan pins the output topology to the encoder-side fine grid, because without it there is no unambiguous definition of which voxels should receive output.
- (c) 3 grids (one per downsampling stage), retrieved from `plan.target_grid_batch` after each stride-2 `ConvolutionPlan.from_grid_batch` call, before building the transposed plans in the decoder.

**Quiz 6**
- (a) `0.05 × 4 = 0.20`
- (b) They are **not** the same in general. `coarsened_grid(2)` creates a topology derived from the original voxels by grouping. `SparseConv3d(stride=2)` creates a topology determined by the kernel map computation in the convolution. The resulting voxel sets may differ depending on the grid structure.
- (c) When you want convolutions to "see" the neighborhood outside the immediate surface — e.g., encoding free-space voxels adjacent to the surface, or when your network's first layer needs context from the empty voxels surrounding the shape.

**Capstone — head input channels**
- `dec0` outputs 16 channels. The skip connection concatenates `enc0`'s 16-channel output via `fvdb.jcat([dec0_out, enc0_out], dim=1)`, giving 32 channels. So the head is `SparseConv3d(32, 1, kernel_size=1)` in Stage 1 and `SparseConv3d(32, 3, kernel_size=1)` in Stage 2.

---

## Reference: Key APIs at a Glance

| Task | API |
|---|---|
| Build grid from points | `GridBatch.from_points(JT, voxel_sizes=)` |
| Build grid from mesh | `GridBatch.from_mesh(v_JT, f_JT, voxel_sizes=)` |
| Build grid from ijk | `GridBatch.from_ijk(ijk_JT, voxel_sizes=, origins=)` |
| Attach features to grid | `grid.jagged_like(flat_tensor)` |
| Sample grid at points | `grid.sample_trilinear(pts_JT, feat_JT)` |
| Splat points onto grid | `grid.splat_trilinear(pts_JT, feat_JT)` |
| World → ijk | `grid.world_to_voxel(pts_JT)` |
| IJK → world | `grid.voxel_to_world(ijk_JT)` |
| IJK → flat index | `grid.ijk_to_index(ijk_JT)` → -1 if absent |
| Point in grid? | `grid.points_in_grid(pts_JT)` → bool JT |
| Coarsen topology | `grid.coarsened_grid(factor)` |
| Refine topology | `grid.refined_grid(factor)` |
| Dilate topology | `grid.dilated_grid(radius)` |
| Conv plan (same/down) | `ConvolutionPlan.from_grid_batch(kernel_size, stride, source_grid, target_grid=)` |
| Conv plan (transposed) | `ConvolutionPlan.from_grid_batch_transposed(kernel_size, stride, source_grid, target_grid)` |
| Output grid from plan | `plan.target_grid_batch` |
| Sparse conv | `fvnn.SparseConv3d(in, out, kernel_size, stride)(feat_JT, plan)` |
| Transposed sparse conv | `fvnn.SparseConvTranspose3d(in, out, kernel_size, stride)(feat_JT, plan)` |
| Batch norm | `fvnn.BatchNorm(channels)(feat_JT, grid)` |
| Skip connection | `fvdb.jcat([a, b], dim=1)` |

---

*This lesson was generated from the fvdb-core repository documentation and is versioned alongside the code at `docs/TEACHME/fvdb_core_lesson.md`.*
