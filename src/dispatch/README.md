# Multi-Axis Dispatch Framework

A modern C++20 compile-time dispatch system for multi-dimensional type and value dispatch.
Designed for high-performance GPU/CPU code where you need to dispatch tensors to different
kernel implementations based on device type, scalar type, memory layout, and other runtime
properties.

## Why This Exists

When writing CUDA kernels that work with PyTorch tensors, you often need code like this:

```cpp
if (tensor.device().is_cuda()) {
    if (tensor.scalar_type() == torch::kFloat32) {
        my_kernel<float><<<...>>>(tensor.data_ptr<float>(), ...);
    } else if (tensor.scalar_type() == torch::kFloat64) {
        my_kernel<double><<<...>>>(tensor.data_ptr<double>(), ...);
    } // ... more types
} else {
    // CPU implementations...
}
```

This gets unwieldy fast. Add memory contiguity, in-place vs out-of-place, determinism
requirements, and suddenly you're looking at a 5-dimensional dispatch space with potentially
64+ combinations. Traditional approaches use:

- **Macro soup** (`AT_DISPATCH_FLOATING_TYPES`, `DISPATCH_DEVICE`, etc.) - hard to compose,
  harder to debug
- **Deeply nested lambdas** - poor error messages, unclear instantiation control
- **Manual switch statements** - verbose, error-prone, no compile-time safety

This framework provides a different approach: **composable, type-safe multi-axis dispatch
with explicit instantiation control and zero macros**.

## Design Philosophy

### No Macros

The entire framework is built using C++20 templates, concepts, and consteval functions,
without any macros. This means:

- Full IDE support (autocomplete, go-to-definition, refactoring)
- Readable error messages that point to actual code
- Debuggable—you can step through dispatch logic
- Composable—combine axes naturally with type aliases

### Minimal Lambda Nesting

Traditional dispatch approaches often look like:

```cpp
AT_DISPATCH_FLOATING_TYPES(dtype, "my_op", [&] {
    AT_DISPATCH_INDEX_TYPES(itype, "my_op", [&] {
        DISPATCH_DEVICE(device, [&] {
            // finally, your actual code, 3 lambdas deep
        });
    });
});
```

This framework uses **tag dispatch** instead. This is similar to the NVidia thrust library.
Your implementation is a normal function (or struct with static methods) that takes a
`tag<...>` as its first parameter:

```cpp
template <torch::ScalarType stype>
void my_op(tag<torch::kCUDA, stype>, torch::Tensor input, torch::Tensor output) {
    using T = torch_scalar_cpp_type_t<stype>;
    my_kernel<T><<<...>>>(input.data_ptr<T>(), output.data_ptr<T>(), ...);
}
```

The dispatch table calls your function with the appropriate `tag<...>` instantiation. No
nested lambdas, no captured state complications, no closure overhead.

### Sparse Instantiation

A 5-dimensional dispatch space (device × dtype × contiguity × placement × determinism) has
2×4×2×2×2 = 64 points. But you probably don't support all combinations:

- GPU might only support contiguous tensors
- Integer types might only support deterministic operations
- Some dtypes might not be implemented yet

This framework lets you declare **subspaces**—hyper-rectangles within the full space that you
actually support:

```cpp
// GPU: only contiguous, out-of-place, float types with non-deterministic
using gpu_float_subspace = axes<axis<torch::kCUDA>,
                                axis<torch::kFloat32, torch::kFloat64>,
                                axis<contiguity::contiguous>,
                                axis<placement::out_of_place>,
                                axis<determinism::not_required>>;
```

Only the points in your declared subspaces get instantiated. Unsupported combinations produce
clear runtime errors, not missing symbol linker errors or silent failures.

## Architecture

The framework is built in layers, each adding capabilities on top of the previous. The
stack below is high-level on top, foundation on the bottom. (Read bottom to top).

```
┌─────────────────────────────────────────────────────────────────┐
│  torch.h - PyTorch-specific dispatch launcher with TORCH_CHECK  │
|             Heavyweight, included in .cpp/.cu                   |
├─────────────────────────────────────────────────────────────────┤
│  torch_types.h  - PyTorch-specific typedefs                     │
|             Lightweight for forward declaration in headers      |
├─────────────────────────────────────────────────────────────────┤
│  dispatch_table.h  - dispatch_table with sparse subspaces       │
|             Heavyweight, included in .cpp/.cu                   |
├─────────────────────────────────────────────────────────────────┤
│  axes_map.h  - Map keyed by coordinates in an axes space        │
|             Heavyweight, included in .cpp/.cu                   |
├─────────────────────────────────────────────────────────────────┤
│  visit_spaces.h  - Visiting and iterating over axes spaces      │
|             Heavyweight, included in .cpp/.cu                   |
├─────────────────────────────────────────────────────────────────┤
│  detail.h  - Internal utilities and helper traits               │
|             Heavyweight, indirectly included in .cpp/.cu        |
├─────────────────────────────────────────────────────────────────┤
│  types.h  - axis, axes, tag, and basic type traits              │
|             Lightweight for forward declaration in headers      |
└─────────────────────────────────────────────────────────────────┘
```


Most of the time, you'd only need the types files in header files, because they
are lightweight.

```cpp
// header file
#include "dispatch/torch_types.h"
// OR
#include "dispatch/types.h"
```

And in the .cpp or .cu files for instantiating the dispatch, you'd need only:

```cpp
// .cpp or .cu source file
#include "dispatch/torch.h"
// OR
#include "dispatch/dispatch_table.h"
```


### Layer 1: Types (`types.h`)

The foundation. Defines the core types:

```cpp
// An axis is a collection of unique values of the same type
using device_axis = axis<torch::kCPU, torch::kCUDA>;
using dtype_axis  = axis<torch::kFloat32, torch::kFloat64, torch::kInt32>;

// Axes combine to form a dispatch space (Cartesian product)
using my_space = axes<device_axis, dtype_axis>;  // 2×3 = 6 combinations

// A tag represents a specific coordinate in the space
using coord = tag<torch::kCUDA, torch::kFloat32>;
```

Provides:
- `axis<V...>` - compile-time sequences of unique, same-type values
- `axes<Axis1, Axis2, ...>` - Cartesian products of axes
- `tag<V...>` - compile-time coordinate tags
- Basic type traits and enum definitions
- Pre-defined axis typedefs: `full_placement_axis`, `full_determinism_axis`,
  `full_contiguity_axis` (from `types.h`)
- Pre-defined PyTorch axis typedefs: `torch_cpu_cuda_device_axis`,
  `torch_full_device_axis`, `torch_full_float_stype_axis`,
  `torch_builtin_float_stype_axis`, `torch_full_signed_int_stype_axis`,
  `torch_full_numeric_stype_axis` (from `torch_types.h`)

### Layer 2: Space Visiting (`visit_spaces.h`)

Compile-time visiting utilities that instantiate a visitor separately for each permutation in
a defined space:

```cpp
visit_axes_space(visitor, my_space{});
```

This is **compile-time visiting**, not runtime iteration. The function uses fold expressions
to expand all tag types in the space at compile time. Each call to the visitor receives a
different `tag<...>` type (e.g., `tag<torch::kCPU, torch::kFloat32>`,
`tag<torch::kCPU, torch::kFloat64>`, etc.), and if the visitor is a generic lambda or
template, it gets instantiated separately for each tag type. This is used internally by
`dispatch_table` construction to instantiate function pointers for each coordinate.

Provides:
- `visit_axes_space()` - compile-time visit all tag coordinates in an axes space (separate
  instantiation per permutation)
- `visit_extents_space()` - compile-time iterate over extents (size-based spaces)
- Coordinate ↔ index conversions

### Layer 3: Axes Maps (`axes_map.h`)

A regular `std::unordered_map` from C++20, using **transparent hash and equality
comparators** to enable flexible key lookups:

```cpp
axes_map<my_space, FunctionPtr> table;

// Both tags and tuples work with emplace and find (thanks to transparency)
auto coord_tuple = std::make_tuple(torch::kCUDA, torch::kFloat32);
table.emplace(coord_tuple, &my_implementation);

auto it1 = table.find(coord_tuple);           // O(1) lookup with tuple
auto it2 = table.find(tag<torch::kCUDA, torch::kFloat32>{});  // O(1) lookup with tag

// Helper function that works with both tags and tuples
insert_or_assign(table, coord_tuple, &my_implementation);
insert_or_assign(table, tag<torch::kCUDA, torch::kFloat32>{}, &my_implementation);
```

The transparency feature (via `is_transparent` in the hash and equality comparators) allows
both `tag<...>` types and tuples to be used directly as keys in `emplace()` and `find()`
operations without needing to construct an explicit key object. The free function
`insert_or_assign()` provides the same convenience for insert/update operations.

Features:
- Regular `std::unordered_map` with transparent comparators (C++20 feature)
- Both `tag<...>` types and tuples work as keys in `emplace()` and `find()`
- Compile-time coordinate validation on insert
- Graceful "not found" for invalid runtime lookups (returns `end()`)
- No key object construction needed for lookups (transparent hashing)

### Layer 4: Dispatch Tables (`dispatch_table.h`)

Combines everything into a dispatch table with sparse instantiation:

```cpp
using dispatcher = dispatch_table<full_space, ReturnType(Args...)>;

// Instantiate only specific subspaces
static dispatcher const table{
    dispatcher::from_op<my_op>(),
    cpu_subspace{},
    gpu_subspace{}
};

// Runtime dispatch
return table(coord_tuple, args...);
```

Two instantiation patterns:
- `from_op<Op>()` — Op is a struct with `static op(tag<...>, args...)` overloads
- `from_visitor(lambda)` — lambda takes `(auto coord, args...)` and calls overloaded
  functions

### Layer 5: Torch Utilities (`torch.h` / `torch_types.h`)

PyTorch-specific conveniences:
- Pre-defined axes: `torch_cpu_cuda_device_axis`, `torch_full_float_stype_axis`, etc.
- Concepts: `torch_integer_stype`, `torch_float_stype`
- `torch_dispatch()` wrapper with friendly error messages
- Coordinate stringification for diagnostics
- `torch_scalar_cpp_type_t<Stype>` - map ScalarType enum to C++ type

## Examples

The `examples/` directory contains three demonstrations:

### `relu.cu` — Simple 2D Dispatch

Shows the minimal pattern: dispatch across device (CPU/CUDA) and dtype (float types). The full
space is just 2×4 = 8 points, all instantiated.

Key pattern: separate `op` overloads for CPU and CUDA, each templated on scalar type.

### `functional.cu` — Free Function Overloads

A 5-dimensional dispatch space (device × dtype × contiguity × placement × determinism). Uses
**overloaded free functions** with `requires` clauses to select implementations:

```cpp
template <torch::ScalarType stype, contiguity cont, determinism det>
tensor_with_notes iscan_impl(tag<torch::kCPU, stype, cont, placement::out_of_place, det>,
                              torch::Tensor input) { ... }

template <torch::ScalarType stype, contiguity cont>
    requires torch_integer_stype<stype>
tensor_with_notes iscan_impl(tag<torch::kCPU, stype, cont, placement::out_of_place, determinism::required>,
                              torch::Tensor input) { ... }
```

The compiler's overload resolution picks the most specific match. **Sparse instantiation**: only
4 subspaces are declared, not the full 64-point space.

### `op.cu` — Single Struct with if-constexpr

Same 5D space, same sparse subspaces, but uses a **single templated `op` function** with `if
constexpr` branching:

```cpp
struct iscan_op {
    template <torch::DeviceType device, torch::ScalarType stype, contiguity cont,
              placement plc, determinism det>
    static tensor_with_notes op(tag<device, stype, cont, plc, det>,
                                 torch::Tensor input) {
        if constexpr (device == torch::kCPU) {
            if constexpr (plc == placement::in_place) {
                // CPU in-place implementation
            } else {
                // CPU out-of-place implementation
            }
        } else {
            // CUDA implementation
        }
    }
};
```

Both `functional.cu` and `op.cu` wrap the same stand-in "external library" (`scan_lib.h`),
demonstrating how to integrate with libraries like CUB, Thrust, or CUTLASS.

## Usage Pattern Summary

1. **Define your space** as an `axes<Axis1, Axis2, ...>`
2. **Define subspaces** that cover only the combinations you support
3. **Write your implementation** as either:
   - A struct with `static op(tag<...>, args...)` overloads, or
   - Free functions that take `tag<...>` as the first parameter
4. **Create a static dispatch table** using `from_op<>()` or `from_visitor()`
5. **Call `torch_dispatch()`** with the runtime coordinate tuple

## Technical Appendix: Why `consteval` Functions Instead of `constexpr` Variables?

Throughout this codebase, you'll notice patterns like:

```cpp
template <typename T>
consteval size_t extent_v() {
    return extent<T>::value();
}
```

Instead of the more traditional:

```cpp
template <typename T>
constexpr size_t extent_v = extent<T>::value;
```

This is intentional. When a `static constexpr` variable is passed to a function that takes it
by const reference, the compiler may generate an address for that variable. With deeply nested
templates (common in this dispatch machinery), this can cause:

1. **Symbol explosion**: Each unique template instantiation gets its own static storage
2. **Link-time bloat**: Especially problematic with nvcc, which can generate enormous object
   files
3. **ODR complications**: Multiple translation units may each define the same symbol

`consteval` functions are guaranteed to be evaluated at compile-time with **zero instantiation
footprint**. There's no storage, no symbol, no address—just a compile-time computation that
produces a value. This is critical when you're instantiating across a 5-dimensional dispatch
space.

The trade-off is slightly more verbose syntax (`extent_v<T>()` vs `extent_v<T>`), but the
compile-time and binary-size benefits are substantial for template-heavy code targeting nvcc.

---

## File Reference

| File | Purpose |
|------|---------|
| `types.h` | Core types: `axis<>`, `axes<>`, `tag<>`, enums (`placement`, `determinism`, `contiguity`) |
| `detail.h` | Internal utilities, helper traits, and type concepts |
| `visit_spaces.h` | Space visitation utilities for iterating over axes and extents |
| `axes_map.h` | Hash map keyed by axes space coordinates |
| `dispatch_table.h` | `dispatch_table` class with sparse subspace instantiation |
| `torch_types.h` | PyTorch-specific axes, scalar type mappings, concepts |
| `torch.h` | PyTorch-specific utilities, error handling, coordinate stringification |
| `examples/` | Working examples (`relu.cu`, `functional.cu`, `op.cu`) with detailed headers |
