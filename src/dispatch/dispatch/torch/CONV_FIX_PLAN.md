# Sparse Convolution Gather/Scatter Refactor Plan

This document describes the refactoring plan for the default sparse convolution path
(`GATHER_SCATTER` backend) to eliminate unnecessary complexity, allocations, and
device-specific duplication.

---

## Problem Summary

The default sparse convolution path has **four layers of indirection**, each adding
unnecessary overhead:

### Layer 1: Python (`fvdb/convolution_plan.py`)

```python
def execute(self, data, weights):
    # Wrap tensor -> JaggedTensor if flat
    if is_flat:
        data = JaggedTensor(data)

    # For GATHER_SCATTER backend:
    result = JaggedTensor(impl=self._pack_info.sparse_conv_3d(data._impl, weights, self._backend))
```

**Issues:**
- JaggedTensor wrap/unwrap on every call
- Multiple backend dispatch paths (HALO, DENSE, CUTLASS, LGGS, etc.) all in one function

### Layer 2: C++ Pack Info (`fvdb/SparseConvPackInfo.cpp`)

```cpp
JaggedTensor SparseConvPackInfo::sparseConv3d(const JaggedTensor &input,
                                              const torch::Tensor &weights,
                                              ConvPackBackend backend) const {
    // For GATHER_SCATTER:
    auto ret = detail::autograd::SparseConvolutionKernelMap::apply(
        input.jdata(), weights, *this, false);
    return mTargetGrid.jagged_like(ret);
}
```

**Issues:**
- Another JaggedTensor wrap/unwrap
- Backend dispatch via if-else chain

### Layer 3: Autograd Wrapper (`fvdb/detail/autograd/SparseConvolutionKernelMap.cpp`)

```cpp
static variable_list forward(AutogradContext *ctx, Variable inFeatures,
                             Variable kernels, const SparseConvPackInfo &packInfo,
                             bool transposed) {
    // WEIGHT PERMUTATION: (out_ch, in_ch, D, H, W) → (K^3, in_ch, out_ch)
    // THIS ALLOCATES A NEW TENSOR EVERY FORWARD CALL!
    kernels = kernels.permute({2, 3, 4, 1, 0}).reshape({-1, inC, outC}).contiguous();

    // CPU COPY EVERY CALL! (for std::max_element in backend)
    nbsizes.cpu().contiguous();

    // Finally dispatch to backend
    ops::dispatchSparseConvolutionKernelMap<DeviceTag>(...);
}
```

**Issues:**
- **Allocates new weight tensor every forward!** `.permute().reshape().contiguous()`
- **Copies neighbor_sizes to CPU every call!** For a single `std::max_element`
- Saves tensors for backward (necessary, but adds overhead)

### Layer 4: Backend (`fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.cu`)

The 620-line mess with:

1. **Separate CUDA/CPU implementations** that differ only in kernel launch mechanism
2. **AT_DISPATCH called inside the per-kernel-weight loop** (12 calls total, wasteful)
3. **Manual dtype-specific buffer slicing** (`is_half`, `is_bfloat16` branches for `from_blob`)
4. **Forward and backward sharing 80% structure** but fully duplicated as 4 separate functions

### The Actual Algorithm

The algorithm is ~10 lines of pseudocode buried under 620 lines of boilerplate:

```
for each kernel_weight_position i in [0, kernel_volume):
    if n_active[i] == 0: skip
    if i == mid_kernel and precompute_mid: skip (already done via mm)

    1. GATHER:  in_buffer[j] = in_feat[kmap[j].src]   for j in [0, n_active[i])
    2. GEMM:    out_buffer = in_buffer @ kernel[i]
    3. SCATTER: out_feat[kmap[j].dst] += out_buffer[j]
```

### Full Call Chain (Current)

```
Python: plan.execute(data, weights)
  │
  ├─► JaggedTensor(data) if flat                    # wrap
  │
  └─► C++: SparseConvPackInfo::sparseConv3d()
        │
        └─► autograd::SparseConvolutionKernelMap::apply()
              │
              ├─► weights.permute({2,3,4,1,0})      # ALLOCATION!
              │         .reshape({-1, inC, outC})
              │         .contiguous()
              │
              ├─► nbsizes.cpu().contiguous()        # CPU COPY!
              │
              └─► ops::dispatchSparseConvolutionKernelMap<DeviceTag>()
                    │
                    └─► for k in kernel_positions:
                          │
                          ├─► AT_DISPATCH_V2(...)   # DISPATCH IN LOOP!
                          │     └─► gatherKernel<scalar_t><<<...>>>
                          │
                          ├─► torch::mm_out(...)
                          │
                          └─► AT_DISPATCH_V2(...)   # DISPATCH IN LOOP!
                                └─► scatterKernel<scalar_t><<<...>>>
```

---

## Target State

A cleaner implementation where:

- **Weights stored in internal format** - `(K^3, C_in, C_out)` at plan creation, not permuted per call
- **neighbor_sizes max computed once** - at plan build time, not every forward
- Device dispatch is a template parameter
- Dtype dispatch happens **once**, outside all loops
- `gather` and `scatter_add` are reusable primitives using unified `for_each`
- Forward/backward share the core loop structure
- Buffer views use `narrow()` instead of dtype-specific `from_blob()`

### Target Call Chain

```
Python: plan.execute(data, weights)
  │
  └─► C++: SparseConvPackInfo::sparseConv3d()
        │
        ├─► weights already in (K^3, C_in, C_out)   # NO ALLOCATION
        ├─► max_buffer_size from plan               # NO CPU COPY
        │
        └─► AT_DISPATCH once, outside loop
              │
              └─► sparse_conv_forward_impl<Dev, Stype>()
                    │
                    └─► for k in kernel_positions:
                          ├─► gather(tag<Dev, Stype>{}, src_view, dst_view, idx_view)
                          ├─► torch::mm_out(...)
                          └─► scatter_add(tag<Dev, Stype>{}, src_view, dst_view, idx_view)
```

---

## Implementation Steps

### Step 1: for_each Wrapper (DONE)

**File:** `dispatch/torch/for_each.h` - **Already exists**

The dispatch framework already provides a device-templated `for_each`:

```cpp
// Current interface (tag-based):
template <typename Tag, typename Func>
    requires tag_match<Tag, torch::kCUDA>  // or kCPU or kPrivateUse1
void for_each(Tag t, int64_t count, Func &&func);

// Functor receives: func(tag, index)
```

For gather/scatter, we'll use this directly with a minimal tag:

```cpp
// Usage in gather_scatter.h:
dispatch::for_each(tag<Dev>{}, count, [=] __hostdev__ (auto, int64_t idx) {
    // idx is the linear index
});
```

The existing implementation handles:
- CUDA: grid-stride loop with configurable grain size
- CPU: ATen `parallel_for` with PyTorch's thread pool
- PrivateUse1: Multi-GPU work distribution

### Step 2: Create gather/scatter_add Primitives (DONE)

**File:** `dispatch/torch/gather_scatter.h`

The implementation uses tag-based dispatch with lightweight view types that carry device
and scalar type as template parameters:

```cpp
namespace dispatch {

// View types - carry device + scalar type, enabling type-safe dispatch
template <torch::DeviceType Dev, torch::ScalarType Stype>
struct matrix_const_view : device_scalar_pair<Dev, Stype> {
    value_type const *data;
    int64_t rows;
    int64_t cols;
    __hostdev__ value_type operator()(int64_t row, int64_t col) const;
};

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct matrix_mutable_view : device_scalar_pair<Dev, Stype> { /* similar */ };

template <torch::DeviceType Dev, torch::ScalarType Stype>
struct vector_const_view : device_scalar_pair<Dev, Stype> {
    value_type const *data;
    int64_t count;
    int64_t stride;  // For interleaved kmap format
    __hostdev__ value_type operator[](int64_t i) const;
};

// Unified gather - works on CPU and GPU via __hostdev__
template <typename Tag, typename SrcView, typename DstView, typename IdxView>
void gather(Tag t, SrcView src, DstView dst, IdxView idx);

// Unified scatter_add - delegates to atomic_add_helper for device-specific atomics
template <typename Tag, typename SrcView, typename DstView, typename IdxView>
void scatter_add(Tag t, SrcView src, DstView dst, IdxView idx);

} // namespace dispatch
```

**Key design choices:**
- **Tag-based dispatch**: `tag<Dev, Stype>` flows through, enabling `tag_match` constraints
- **View types**: Encapsulate pointer + dimensions, carry device/scalar type for specialization
- **Minimal specialization**: Only `atomic_add_helper` is device-specialized (CPU: `std::atomic_ref`, GPU: `atomicAdd`)
- **`for_each_nd<2>`**: 2D iteration space, views handle indexing
- **Stride support**: `vector_const_view.stride` handles interleaved kmap format

### Step 3: Rewrite SparseConvolutionKernelMap Core

**File:** `fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.cu`

Replace the 4 dispatch functions with a single template:

```cpp
template <torch::DeviceType Dev, typename ScalarT>
void sparse_conv_kernel_map_forward_impl(
    torch::Tensor in_feat,      // (N_in, C_in)
    torch::Tensor out_feat,     // (N_out, C_out)
    torch::Tensor kernel,       // (K^3, C_in, C_out)
    torch::Tensor neighbor_map, // (total_pairs, 2) or flat with stride 2
    torch::Tensor neighbor_offset, // (K^3,) - on CPU
    bool transpose,
    bool middle_accel)
{
    auto guard = c10::cuda::CUDAGuard(in_feat.device());  // no-op for CPU

    const int64_t kernel_volume = kernel.size(0);
    const int64_t C_in = in_feat.size(1);
    const int64_t C_out = kernel.size(2);
    const int64_t N_in = in_feat.size(0);
    const int64_t N_out = out_feat.size(0);

    // Compute max buffer size needed
    int64_t max_active = *std::max_element(
        neighbor_offset.data_ptr<int>(),
        neighbor_offset.data_ptr<int>() + kernel_volume);
    max_active = std::max(max_active, int64_t(1));

    // Allocate buffers once
    auto opts = in_feat.options();
    auto in_buffer = torch::empty({max_active, C_in}, opts);
    auto out_buffer = torch::empty({max_active, C_out}, opts);

    // Middle kernel acceleration
    int64_t mid = kernel_volume / 2;
    bool do_mid_accel = middle_accel &&
                        (kernel_volume % 2 == 1) &&
                        (N_in == N_out);
    if (do_mid_accel) {
        torch::mm_out(out_feat, in_feat, kernel[mid]);
    }

    // Get raw pointers
    auto* in_ptr = in_feat.data_ptr<ScalarT>();
    auto* out_ptr = out_feat.data_ptr<ScalarT>();
    auto* kmap_ptr = neighbor_map.data_ptr<int32_t>();
    auto* noff_ptr = neighbor_offset.data_ptr<int>();

    int64_t kmap_offset = 0;
    for (int64_t k = 0; k < kernel_volume; ++k) {
        int64_t n_active = noff_ptr[k];

        if (n_active == 0) continue;

        if (do_mid_accel && k == mid) {
            kmap_offset += 2 * n_active;  // skip pairs
            continue;
        }

        // Buffer views (no allocation, just narrowing)
        auto in_buf = in_buffer.narrow(0, 0, n_active);
        auto out_buf = out_buffer.narrow(0, 0, n_active);

        // GATHER: in_buf[i] = in_feat[kmap[i].src]
        int src_col = transpose ? 1 : 0;
        dispatch::gather(
            tag<Dev, Stype>{},
            matrix_const_view<Dev, Stype>{in_ptr, N_in, C_in},
            matrix_mutable_view<Dev, Stype>{in_buf.data_ptr<ScalarT>(), n_active, C_in},
            vector_const_view<Dev, torch::kInt32>{kmap_ptr + kmap_offset + src_col, n_active, 2});

        // GEMM: out_buf = in_buf @ kernel[k]
        torch::mm_out(out_buf, in_buf, kernel[k]);

        // SCATTER: out_feat[kmap[i].dst] += out_buf[i]
        int dst_col = transpose ? 0 : 1;
        dispatch::scatter_add(
            tag<Dev, Stype>{},
            matrix_const_view<Dev, Stype>{out_buf.data_ptr<ScalarT>(), n_active, C_out},
            matrix_mutable_view<Dev, Stype>{out_ptr, N_out, C_out},
            vector_const_view<Dev, torch::kInt32>{kmap_ptr + kmap_offset + dst_col, n_active, 2});

        kmap_offset += 2 * n_active;
    }
}
```

**Note on kmap stride:** The current `neighbor_map` is interleaved `(src0, dst0, src1, dst1, ...)`.
The gather/scatter functions will need a `stride` parameter to handle this, or we preprocess into
separate arrays. The cleaner solution is to add `int64_t index_stride = 1` parameter.

### Step 4: Single Entry Point with Outside Dispatch

```cpp
void dispatchSparseConvolutionKernelMap(
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    torch::Tensor kernel,
    torch::Tensor neighbor_map,
    torch::Tensor neighbor_offset,
    bool transpose,
    bool middle_accel)
{
    // Validation
    TORCH_CHECK(in_feat.device() == out_feat.device());
    TORCH_CHECK(in_feat.device() == kernel.device());
    TORCH_CHECK(in_feat.device() == neighbor_map.device());
    TORCH_CHECK(neighbor_offset.device().is_cpu(), "neighbor_offset must be on CPU");

    auto dev = in_feat.device().type();

    // Dispatch dtype ONCE, outside all loops
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        in_feat.scalar_type(),
        "SparseConvKernelMap",
        [&] {
            if (dev == torch::kCUDA) {
                c10::cuda::CUDAGuard guard(in_feat.device());
                sparse_conv_kernel_map_forward_impl<torch::kCUDA, scalar_t>(
                    in_feat, out_feat, kernel, neighbor_map, neighbor_offset,
                    transpose, middle_accel);
            } else {
                sparse_conv_kernel_map_forward_impl<torch::kCPU, scalar_t>(
                    in_feat, out_feat, kernel, neighbor_map, neighbor_offset,
                    transpose, middle_accel);
            }
        });
}
```

### Step 5: Fix Autograd Layer Overhead

**File:** `fvdb/detail/autograd/SparseConvolutionKernelMap.cpp`

The autograd layer currently does unnecessary work every forward call:

```cpp
// CURRENT: Allocates new tensor every forward!
kernels = kernels.permute({2, 3, 4, 1, 0}).reshape({-1, inC, outC}).contiguous();

// CURRENT: Copies to CPU every call for std::max_element!
nbsizes.cpu().contiguous()
```

**Fixes:**

1. **Cache max buffer size in SparseConvPackInfo:**
   ```cpp
   // In SparseConvPackInfo.h
   int64_t maxBufferSize() const { return mMaxBufferSize; }

   // Computed once in buildGatherScatter():
   mMaxBufferSize = *std::max_element(nbsizes.begin(), nbsizes.end());
   ```

2. **Accept pre-permuted weights or permute at plan creation:**

   Option A (minimal change): Accept weights in internal format from caller
   ```cpp
   // Caller ensures weights are (K^3, C_in, C_out)
   // No permute needed in forward
   ```

   Option B (better UX): Store permuted weights in ConvolutionPlan at creation
   ```python
   # In ConvolutionPlan.__init__:
   self._kernel_internal = None  # Lazily permuted on first execute
   ```

3. **Remove nbsizes.cpu() call:**
   ```cpp
   // Use cached maxBufferSize from packInfo instead
   int64_t max_active = packInfo.maxBufferSize();
   ```

### Step 6: Apply Same Pattern to Backward

The backward follows the same structure:

```cpp
template <torch::DeviceType Dev, typename ScalarT>
void sparse_conv_kernel_map_backward_impl(
    torch::Tensor in_feat,       // saved from forward
    torch::Tensor grad_in_feat,  // output: gradient w.r.t. input
    torch::Tensor grad_out_feat, // input: gradient from downstream
    torch::Tensor kernel,
    torch::Tensor grad_kernel,   // output: gradient w.r.t. kernel
    torch::Tensor neighbor_map,
    torch::Tensor neighbor_offset,
    bool transpose)
{
    // Similar structure:
    // For each kernel position k:
    //   1. GATHER grad_out into buffer
    //   2. GATHER in_feat into buffer
    //   3. GEMM: grad_in_buffer = grad_out_buffer @ kernel[k].T
    //   4. GEMM: grad_kernel[k] = in_buffer.T @ grad_out_buffer
    //   5. SCATTER-ADD grad_in_buffer into grad_in_feat
}
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `dispatch/torch/for_each.h` | **Exists** - no changes needed |
| `dispatch/torch/gather_scatter.h` | **DONE** - gather/scatter_add primitives |
| `fvdb/detail/autograd/SparseConvolutionKernelMap.cpp` | **Fix** - remove weight permute, cache max_buffer_size |
| `fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.cu` | **Rewrite** - use new primitives |
| `fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.cpp` | **Delete** - merged into .cu via templates |
| `fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.h` | **Simplify** - remove device template, single signature |
| `fvdb/SparseConvPackInfo.h` | **Add** - `maxBufferSize()` accessor |
| `fvdb/convolution_plan.py` | **Update** - store weights in internal format (optional, phase 2) |

---

## Open Questions / Refinements

### 1. kmap Index Stride (RESOLVED)

The current `neighbor_map` is interleaved `(src0, dst0, src1, dst1, ...)`.

**Solution:** Use `vector_const_view.stride = 2`. The gather call becomes:
```cpp
gather(tag<Dev, Stype>{},
       matrix_const_view<Dev, Stype>{in_ptr, N_in, C_in},
       matrix_mutable_view<Dev, Stype>{buf_ptr, n_active, C_in},
       vector_const_view<Dev, torch::kInt32>{kmap_ptr + src_col, n_active, 2});  // stride=2
```

### 2. Weight Format Convention

Where should the weight permutation happen?

- **Option A (API breaking):** Require weights in internal format `(K^3, C_in, C_out)` everywhere
- **Option B (Cache at plan):** `ConvolutionPlan` caches permuted weights on first execute
- **Option C (Do nothing):** Keep current behavior (permute every call)
- **Recommendation:** Option B - best UX with no API break. Can be phase 2.

### 3. PrivateUse1 Support (NEEDS ATOMIC_ADD_HELPER SPECIALIZATION)

The existing `for_each.h` already supports PrivateUse1 (multi-GPU) via `tag_match<Tag, torch::kPrivateUse1>`.
The gather primitive works out of the box (unified via `__hostdev__`).

For scatter_add, need to add a PrivateUse1 specialization of `atomic_add_helper` that uses
`atomicAdd` (same as CUDA). Currently only CPU and CUDA specializations exist.

### 4. Atomics for CPU scatter_add (RESOLVED)

**Solution:** Use C++20 `std::atomic_ref` for thread-safe atomic add on CPU:
```cpp
std::atomic_ref<scalar_t>(dst[...]).fetch_add(src[...], std::memory_order_relaxed);
```

This is implemented via `atomic_add_helper` specialization - the only device-specific
code in gather_scatter.h. The helper is specialized per device:
- CPU: `std::atomic_ref<T>::fetch_add`
- GPU: `atomicAdd`

### 5. ME (MinkowskiEngine) Backward Path

The backward has a special `use_me` branch that calls `dispatchMESparseConvolutionKernelMapGrad`.
This needs to be preserved or unified into the same pattern.

---

## Validation

1. **Existing tests:** `tests/unit/test_conv_default.py`, `tests/unit/test_conv_ground_truth.py`
2. **Numerical equivalence:** Run before/after and diff outputs
3. **Performance:** Should be equivalent or slightly better (fewer dispatch calls)

---

## Not In Scope

- Element accessors (deferred to future work)
- Other convolution backends (ImplicitGEMM, Halo, CUTLASS)
- Kernel map building (`ConvolutionKernelMap.cu` in pack_info/)
- PrivateUse1 support (can add later with same pattern)

---

## Implementation Phases

### Phase 1: Backend Cleanup (Core Goal)

Fix the worst offender - the 620-line backend file.

1. ~~Create `for_each.h` wrapper~~ **DONE** - already exists in dispatch framework
2. ~~Create `gather_scatter.h` primitives~~ **DONE** - `dispatch/torch/gather_scatter.h`
3. Rewrite `SparseConvolutionKernelMap.cu` to use new primitives
4. Delete `SparseConvolutionKernelMap.cpp` (CPU code now in .cu via templates)
5. Add `maxBufferSize()` to `SparseConvPackInfo` (avoids CPU copy per call)

**Result:** Backend reduced from 620 lines to ~150 lines, no more AT_DISPATCH in loops.

### Phase 2: Autograd Layer Cleanup (Optional)

Eliminate per-call allocations.

1. Cache permuted weights in `ConvolutionPlan` on first execute
2. Remove `nbsizes.cpu().contiguous()` call (use cached max)
3. Simplify autograd forward to just call the backend

**Result:** Zero allocations per forward call (beyond output tensor).

### Phase 3: Python Layer Cleanup (Optional)

Simplify `convolution_plan.py`.

1. Separate backend implementations into strategy classes
2. Remove JaggedTensor wrap/unwrap for single grids

**Result:** Cleaner code, but no performance impact.

---

## Estimated Effort

### Phase 1 (Backend Cleanup)

- ~~**for_each.h:** 1-2 hours~~ **DONE** - already exists
- ~~**gather_scatter.h:** 1 hour~~ **DONE** - primitives created
- **SparseConvolutionKernelMap rewrite:** 2-3 hours (main work)
- **maxBufferSize cache:** 30 min
- **Testing and debugging:** 1-2 hours

**Phase 1 Total:** ~4-6 hours

### Phase 2 (Autograd Cleanup)

- **Weight caching in plan:** 1-2 hours
- **Remove cpu() copy:** 30 min
- **Testing:** 1 hour

**Phase 2 Total:** ~3-4 hours

### Phase 3 (Python Cleanup)

- **Backend strategy classes:** 2-3 hours
- **Testing:** 1 hour

**Phase 3 Total:** ~3-4 hours

---

**Recommended:** Complete Phase 1 first. It delivers most of the value.
