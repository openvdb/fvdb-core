# Accumulator In-Place Optimization

## Problem

At 500M+ Gaussians, the projection backward pass allocates ~6 GB of temporary
tensors per call (3 × `[N]` tensors for grad norms, step counts, max radii).
These are zeroed, written to by the CUDA kernel via `atomicAdd`/`atomicMax`,
then accumulated into persistent accumulators in Python. The temporaries are
immediately discarded.

## Solution

The CUDA kernel already uses `atomicAdd` and `atomicMax`, which are inherently
accumulative — they work correctly on non-zero tensors. Pass the persistent
accumulators directly to the kernel, eliminating the temporary allocations
entirely. This is a Python-only change; no C++ or CUDA modifications needed.

## Milestones

### Milestone 1: Eliminate fresh-zeros pattern in projection backward
- [ ] Remove temporary tensor creation in `_ProjectGaussiansFn.backward`
- [ ] Pass `ctx.accum_*` directly to `_C.gsplat_projection_bwd`
- [ ] Delete Python-side `add_()` / `torch.maximum()` accumulation step
- [ ] Update `project_to_2d` docstring to document in-place backward mutation
- [ ] All 135 tests pass

### Milestone 2 (Future): Reference-count-based copy-on-write
- [ ] Add `storage().use_count()` check in C++ dispatch for shared accumulators
- [ ] Clone accumulator if storage is shared before passing to kernel
- [ ] Benchmark to confirm no regression for the common (non-shared) case

## Key Files

| File | Role |
|------|------|
| `fvdb/functional/splat/_projection.py` | Python autograd backward (lines 127-173) |
| `src/python/GaussianSplatOps.cpp` | Pybind binding (lines 700-765) — no changes needed |
| `src/fvdb/detail/ops/gsplat/GaussianProjectionBackward.cu` | CUDA kernel (lines 191-200) — no changes needed |
