# Sparse Convolution Refactor — Next Steps

This document tracks what remains to be done for the sparse convolution refactor,
building on the dispatch framework work completed in Phase 4.

---

## What's Done (Phase 4)

### Dispatch Core
- **Tags** (`tag.h`): Self-normalizing, order-independent value sets
- **Constraints** (`with_value.h`): `with_value<Tag, V>` and `with_type<Tag, T>` concepts with proper subsumption
- **Dispatch tables** (`dispatch_table.h`): Runtime-to-compile-time dispatch via `dispatch_set{...}`
- **Enums** (`enums.h`): `placement`, `determinism`, `contiguity`, `scheduling`

### Thread Pool (`thread_pool.h`)
- `thread_pool<scheduling::uniform>` (broadcast_pool): static partitioning
- `thread_pool<scheduling::adaptive>` (work_stealing_pool): Chase-Lev work-stealing
- `default_thread_pool` = work_stealing_pool (benchmarked as best general-purpose choice)

### for_each (`torch/for_each.h`)
- Scalar index generation over `[0, count)` on CPU, CUDA, or PrivateUse1
- CPU uses `default_thread_pool`, per-element loop
- CUDA uses one-element-per-thread grid-stride kernel (optimal coalescing)
- GPU block size configurable via `tag_add<Tag, block_dim::bN>`, defaults to 256
- Functor signature: `func(Tag, int64_t idx)`

### Views (`torch/views.h`)
- `tensor_in<Dev, Stype, Rank, Contig>` (read-only) and `tensor_out<Dev, Stype, Rank, Contig>` (writable)
- Contiguity-specialized: contiguous uses row-major offset, strided uses full stride computation
- Handles broadcast (stride-0) dimensions correctly
- Trivially copyable, safe for CUDA kernel capture

---

## Next Step: gather_scatter

Re-introduce `dispatch/torch/gather_scatter.h` on top of the new for_each + views.

### Algorithm (unchanged from original design)

```
gather:      dst[i, :] = src[idx[i], :]    for i in [0, n)
scatter_add: dst[idx[i], :] += src[i, :]   for i in [0, n)
```

### Implementation using for_each + views

```cpp
template <typename Tag>
    requires with_type<Tag, torch::DeviceType>
          && with_type<Tag, torch::ScalarType>
          && with_type<Tag, contiguity>
void gather(Tag tag,
            tensor_in<...>  src,     // [N_src, C]
            tensor_out<...> dst,     // [n_active, C]
            tensor_in<...>  indices) // [n_active] or [n_active] with stride 2 (interleaved kmap)
{
    int64_t const n = dst.size(0);
    int64_t const c = dst.size(1);

    for_each(tag, n * c, [=] __hostdev__ (Tag, int64_t idx) {
        int64_t const i = idx / c;
        int64_t const j = idx % c;
        int64_t const src_row = indices(i);
        if (src_row >= 0)
            dst(i, j) = src(src_row, j);
    });
}
```

### scatter_add — device-specific atomics

The only device-specialized code is `atomic_add_helper`:
- **CPU:** `std::atomic_ref<scalar_t>(*dst).fetch_add(src, std::memory_order_relaxed)`
- **CUDA:** `atomicAdd(dst, src)`

This can be a small helper struct specialized on DeviceType, same pattern as the
original design.

### Interleaved kmap format

The current `neighbor_map` is interleaved `(src0, dst0, src1, dst1, ...)`.
A Rank-1 `tensor_in` with stride 2 handles this naturally:
```cpp
auto src_indices = tensor_in<dev, torch::kInt32, 1, contig>(neighbor_map_slice);
// src_indices(i) reads neighbor_map[i * stride] — stride=2 gives every other element
```

---

## After gather_scatter: Sparse Convolution Rewrite

### Target: collapse SparseConvolutionKernelMap.cu

The current 620-line file has 4 functions (forward CUDA, forward CPU, backward CUDA,
backward CPU) that are ~80% identical. The rewrite collapses them into one template:

```cpp
template <typename Tag>
void sparse_conv_forward(Tag tag,
                         torch::Tensor in_feat,
                         torch::Tensor out_feat,
                         torch::Tensor kernel,
                         torch::Tensor neighbor_map,
                         torch::Tensor neighbor_offset,
                         bool transpose,
                         bool middle_accel)
{
    // ... extract coordinates from tag ...
    // ... device guard, buffer allocation ...

    for (int64_t k = 0; k < kernel_volume; ++k) {
        // gather -> mm_out -> scatter_add
        gather(tag, in_feat_view, buf_in_view, kmap_view);
        torch::mm_out(buf_out, buf_in, kernel[k]);
        scatter_add(tag, buf_out_view, out_feat_view, kmap_view);
    }
}
```

### Fix autograd layer overhead

- Cache `maxBufferSize` in `SparseConvPackInfo` (avoids `.cpu().contiguous()` per call)
- Cache permuted weights at plan creation (avoids `.permute().reshape().contiguous()` per call)
- Single dtype dispatch outside the kernel-weight loop

---

## Future: Vectorization Layer (`for_each_vectorized`)

A separate `for_each_vectorized` for operations that benefit from vector loads.
This is intentionally NOT built into `for_each` because vectorization and
coalescing require fundamentally different thread-to-element layouts:

- `for_each` uses **interleaved** layout (one element per thread per stride step),
  which gives optimal coalescing for scalar loads — adjacent threads access
  adjacent memory locations.
- Vectorized loads (`float4`, etc.) require **contiguous-per-thread** layout so
  each thread can issue a single wide load. With scalar loads this layout causes
  stride-N coalescing (bad), but with vector loads each thread's wide transaction
  packs perfectly into cache lines.

`for_each_vectorized` would:

- Use contiguous-per-thread layout with vector-width batches
- Provide chunk-based functor or automatic vector-type wrapping
- Handle masked loads for tail elements (count not divisible by vector width)
- Use vector types (float4/double2 on CUDA, ATen Vec on CPU)
- Close the ~1.6x gap to torch's native CUDA softplus (measured in benchmarks)
