# Dispatch Framework

A C++20 dispatch system for multi-dimensional type and value dispatch, parallel
iteration, and stride-correct tensor access. Designed for high-performance GPU/CPU
code that needs to dispatch tensors to different kernel implementations based on
device type, scalar type, memory layout, and other runtime properties.

## Architecture

The framework is built in layers. Read bottom to top.

```
┌─────────────────────────────────────────────────────────────────┐
│  for_each (torch/for_each.h)                                    │
│    Scalar index generation: CPU thread pool, CUDA grid-stride   │
├─────────────────────────────────────────────────────────────────┤
│  views (torch/views.h)                                          │
│    flat_in/flat_out (rank-free) + tensor_in/tensor_out (ranked) │
├─────────────────────────────────────────────────────────────────┤
│  torch utilities (torch/dispatch.h, torch/types.h)              │
│    Device guards, type mappings, contiguity helpers             │
├─────────────────────────────────────────────────────────────────┤
│  thread_pool (thread_pool.h)                                    │
│    broadcast_pool (static) and work_stealing_pool (adaptive)    │
├─────────────────────────────────────────────────────────────────┤
│  dispatch_table (dispatch_table.h)                              │
│    Sparse subspace instantiation, runtime select + invoke       │
├─────────────────────────────────────────────────────────────────┤
│  core types (tag.h, with_value.h, axis.h, axes.h, enums.h)     │
│    Tags, concepts, axes, dispatch coordinates                   │
└─────────────────────────────────────────────────────────────────┘
```

## Design Philosophy

- **No macros.** The entire framework uses C++20 templates, concepts, and consteval.
- **Tags encode semantics.** A `tag<torch::kCUDA, torch::kFloat32, contiguity::contiguous>`
  carries all dispatch coordinates as compile-time values. Order doesn't matter.
- **Call-site readability over implementation elegance.** The user's code should parse
  at a glance by someone who doesn't know the framework.
- **Separation of concerns.** Dispatch coordinates (tag) are resolved before invocation.
  Select and invoke are separate steps.

## Core Dispatch System

### Tags and Constraints

A **tag** is an unordered set of compile-time values. `tag<A, B>` and `tag<B, A>`
resolve to the same type.

```cpp
using my_tag = tag<torch::kCUDA, torch::kFloat32, contiguity::contiguous>;
```

**Concepts** constrain template parameters on tags:

```cpp
template <typename Tag>
    requires with_value<Tag, torch::kCUDA>     // specific value
          && with_type<Tag, torch::ScalarType>  // any value of this type
void my_impl(Tag tag, ...) {
    constexpr auto stype = tag_get<torch::ScalarType>(Tag{});
    using scalar_t = torch_scalar_cpp_type_t<stype>;
}
```

`with_value` subsumes `with_type` — the compiler selects the most specific overload.

### Tag Composition

`tag_add` and `tag_subtract` produce new tags by adding or removing values:

```cpp
using T = tag<torch::kCUDA, torch::kFloat32>;
using U = tag_add<T, block_dim::b128>;       // add block_dim
// U = tag<torch::kCUDA, torch::kFloat32, block_dim::b128>

using V = tag_subtract<U, block_dim>;        // remove block_dim
// V = tag<torch::kCUDA, torch::kFloat32>  (same as T)
```

`tag_add` replaces the existing value if the type is already present. This is
how op authors inject operational parameters (like GPU block size) into a tag
that came from a dispatch table — without polluting the dispatch space.

### Dispatch Tables

A **dispatch table** maps runtime values to compile-time tag instantiations:

```cpp
struct my_op {
    template <typename Tag>
    static void op(Tag tag, torch::Tensor input, torch::Tensor output) { ... }

    using space      = axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

// Build table (instantiates op for every point in the space)
static auto const table = dispatch_table_from_op<my_op>("my_op");

// Runtime dispatch
table.select(dispatch_set{dev, stype, contig})(input, output);
```

Only declared subspaces are instantiated. Unsupported combinations produce clear
runtime errors.

### Enums

Pre-defined dispatch coordinates with `type_label` specializations:

- `placement { in_place, out_of_place }`
- `determinism { not_required, required }`
- `contiguity { strided, contiguous }`
- `scheduling { uniform, adaptive }`

## Thread Pool (`thread_pool.h`)

Two scheduling strategies as template specializations:

- `thread_pool<scheduling::uniform>` (alias: `broadcast_pool`) — static partitioning,
  lowest dispatch overhead, optimal for uniform workloads
- `thread_pool<scheduling::adaptive>` (alias: `work_stealing_pool`) — Chase-Lev
  work-stealing, optimal for imbalanced workloads
- `default_thread_pool` — aliases `work_stealing_pool` (best general-purpose choice)

Interface: `pool.parallel_for(start, end, grain, func)` where func receives
`(begin, end)` range.

## for_each (`torch/for_each.h`)

Scalar index generation over `[0, count)`, dispatched by device tag:

```cpp
for_each(tag, count, [=] __hostdev__ (Tag, int64_t idx) {
    out[idx] = some_computation(in[idx]);
});
```

- **CPU:** Uses `default_thread_pool` (work-stealing), per-element loop
- **CUDA:** One-element-per-thread grid-stride kernel, optimal coalescing
- **PrivateUse1:** Multi-GPU distribution, grid-stride per device

The functor receives `(Tag, int64_t idx)`. The tag is passed through so the functor
can be concept-constrained.

### GPU block size

Defaults to 256 (optimal for most elementwise workloads). Op authors who need
a different block size inject it inline via `tag_add`:

```cpp
for_each(tag_add<Tag, block_dim::b128>{}, count, func);
```

### Why no vectorization

`for_each` is a scalar index generator — one element per functor call, one
thread per element (interleaved), optimal coalescing for scalar loads. It does
not provide vectorization infrastructure.

Vectorized loads (`float4`, etc.) require contiguous-per-thread layout, which
conflicts with the coalescing-optimal interleaved layout that `for_each` uses.
They also require masked load/store for tail handling, vector type selection,
and alignment management. This infrastructure belongs in a future
`for_each_vectorized`. `for_each` does not prevent manual vectorization, but it
does not facilitate it.

## Views (`torch/views.h`)

Two view families, both specialized on contiguity and trivially copyable (safe for
CUDA kernel capture). Contiguity is a dispatch axis — resolved once at dispatch time.

### Flat views: `flat_in` / `flat_out`

Rank-free elementwise access via `operator[]`. Designed for `for_each` patterns where
the tensor's shape is irrelevant — we just need every element visited once.

```cpp
auto in  = flat_in<dev, stype, contig>(input);    // read-only, any rank
auto out = flat_out<dev, stype, contig>(output);   // writable, any rank

out[idx] = f(in[idx]);   // flat linear index, handles any tensor rank
```

- **`contiguity::contiguous`:** Just `data[flat_idx]`. Zero overhead regardless of rank.
- **`contiguity::strided`:** Unravels `flat_idx` via div/mod chain using runtime
  sizes/strides. Handles transposed, sliced, and broadcast (stride-0) tensors.

### Ranked views: `tensor_in` / `tensor_out`

Multi-index access via `operator()(i, j, ...)` with compile-time `Rank`. For structured
patterns where you need explicit dimensional indices (gather-scatter, morton encoding).

```cpp
auto ijk = tensor_in<dev, torch::kInt32, 2, contig>(coords);   // [N, 3]
auto out = tensor_out<dev, torch::kInt64, 1, contig>(codes);    // [N]

auto i = ijk(idx, 0);   // explicit multi-index access
```

### Broadcasting

Broadcast dimensions (stride 0) work naturally in both families. For binary ops,
the entry-point function handles `expand_as()` internally (following PyTorch convention),
so the caller never has to pre-expand.

## Examples

### `softplus.cu` — for_each + Flat Views (Recommended Pattern)

The showcase example. Demonstrates how `for_each` + flat views eliminate all manual work
and handle tensors of any rank:

```cpp
// ONE __hostdev__ scalar function. T = storage type, C = compute type.
// torch_compute_type_t promotes half types to float; identity for float/double.
template <typename T, typename C>
__hostdev__ T softplus_scalar(T x, T beta, T threshold) {
    C const bx = static_cast<C>(beta) * static_cast<C>(x);
    if (bx > static_cast<C>(threshold)) return x;
    return static_cast<T>(log1p(exp(bx)) / static_cast<C>(beta));
}

// ONE impl — all devices, all scalar types, both contiguities, ANY tensor rank
template <typename Tag>
    requires with_type<Tag, torch::DeviceType>
          && with_type<Tag, torch::ScalarType>
          && with_type<Tag, contiguity>
void softplus_impl(Tag tag, torch::Tensor const& input, torch::Tensor& output,
                     double beta_d, double threshold_d) {
    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});
    using scalar_t  = torch_scalar_cpp_type_t<stype>;
    using compute_t = torch_compute_type_t<stype>;  // float for halves, identity otherwise

    auto guard = make_device_guard(tag, input);
    auto in    = flat_in<dev, stype, contig>(input);    // rank-free flat access
    auto out   = flat_out<dev, stype, contig>(output);

    auto const beta      = static_cast<scalar_t>(beta_d);
    auto const threshold = static_cast<scalar_t>(threshold_d);

    for_each(tag, input.numel(), [=] __hostdev__ (Tag, int64_t idx) {
        out[idx] = softplus_scalar<scalar_t, compute_t>(in[idx], beta, threshold);
    });
}
```

**Contrast with `relu.cu`** (the "before"):
- relu needs 2 scalar function overloads (half vs builtin float comparison issues)
- relu needs a separate `__global__` CUDA kernel
- relu needs 2 `op()` overloads (CPU serial loop, CUDA manual launch)
- relu only handles 1D contiguous tensors

softplus needs none of that. Zero device-specific, contiguity-specific, or rank-specific code.

### `relu.cu` — Manual Dispatch (Baseline)

Shows the traditional pattern without `for_each`: separate CPU and CUDA overloads,
manual device guard, manual kernel launch. Useful for understanding what the framework
automates.

### `functional.cu` / `op.cu` — Multi-Axis Sparse Dispatch

5-dimensional dispatch space with partial coverage. Demonstrates overload resolution
with `requires` clauses and sparse subspace instantiation. `functional.cu` uses free
functions; `op.cu` uses a struct with static methods.

## File Reference

| File | Purpose |
|------|---------|
| `tag.h` | Self-normalizing compile-time value tags, `tag_add`, `tag_subtract` |
| `with_value.h` | `with_value`, `with_type` concepts, `tag_get` extraction |
| `axis.h`, `axes.h` | Value sets and Cartesian products |
| `enums.h` | `placement`, `determinism`, `contiguity`, `scheduling`, `block_dim` |
| `label.h`, `label_sorted.h` | Compile-time label infrastructure for tag ordering |
| `consteval_types.h` | Compile-time type traits |
| `dispatch_set.h` | Runtime dispatch coordinates |
| `dispatch_table.h` | Sparse dispatch table with select/invoke |
| `detail.h` | Internal utilities and helper traits |
| `visit_spaces.h` | Compile-time space visitation |
| `axes_map.h` | Hash map keyed by axes space coordinates |
| `indices.h`, `extents.h` | Index/extent utilities |
| `types.h` | Compatibility shim bundling core headers |
| `macros.h` | `__hostdev__`, `DISPATCH_SPIN_PAUSE`, `DISPATCH_UNROLL` |
| `thread_pool.h` | `broadcast_pool`, `work_stealing_pool`, `default_thread_pool` |
| `torch/types.h` | PyTorch type labels, axes, scalar type mappings, `torch_compute_type_t` |
| `torch/dispatch.h` | Device guards, contiguity helpers, device concepts |
| `torch/for_each.h` | Scalar index generation (CPU, CUDA, PrivateUse1) |
| `torch/views.h` | `flat_in`/`flat_out` (rank-free), `tensor_in`/`tensor_out` (ranked) |
| `examples/` | `relu.cu`, `softplus.cu`, `functional.cu`, `op.cu` |

## Technical Note: `consteval` Functions

Throughout this codebase, compile-time values use `consteval` functions instead of
`constexpr` variables:

```cpp
template <typename T>
consteval size_t extent_v() { return extent<T>::value(); }
```

This avoids symbol explosion, link-time bloat, and ODR complications that arise with
`static constexpr` variables in deeply nested template instantiations — especially
with nvcc.
