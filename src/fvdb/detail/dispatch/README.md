# Multi-Axis Dispatch Framework

A modern C++20 compile-time dispatch system for multi-dimensional type and value dispatch. Designed for high-performance GPU/CPU code where you need to dispatch tensors to different kernel implementations based on device type, scalar type, memory layout, and other runtime properties.

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

This gets unwieldy fast. Add memory contiguity, in-place vs out-of-place, determinism requirements, and suddenly you're looking at a 5-dimensional dispatch space with potentially 64+ combinations. Traditional approaches use:

- **Macro soup** (`AT_DISPATCH_FLOATING_TYPES`, `DISPATCH_DEVICE`, etc.) - hard to compose, harder to debug
- **Deeply nested lambdas** - poor error messages, unclear instantiation control
- **Manual switch statements** - verbose, error-prone, no compile-time safety

This framework provides a different approach: **composable, type-safe multi-axis dispatch with explicit instantiation control and zero macros**.

## Design Philosophy

### No Macros

The entire framework is built using C++20 templates, concepts, and consteval functions. Not a single macro. This means:

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

This framework uses **tag dispatch** instead. Your implementation is a normal function (or struct with static methods) that takes a `Tag<...>` as its first parameter:

```cpp
template <torch::ScalarType stype>
void my_op(Tag<torch::kCUDA, stype>, torch::Tensor input, torch::Tensor output) {
    using T = typename c10::impl::ScalarTypeToCPPType<stype>::type;
    my_kernel<T><<<...>>>(input.data_ptr<T>(), output.data_ptr<T>(), ...);
}
```

The dispatch table calls your function with the appropriate `Tag<...>` instantiation. No nested lambdas, no captured state complications, no closure overhead.

### Sparse Instantiation

A 5-dimensional dispatch space (device × dtype × contiguity × placement × determinism) has 2×4×2×2×2 = 64 points. But you probably don't support all combinations:

- GPU might only support contiguous tensors
- Integer types might only support deterministic operations
- Some dtypes might not be implemented yet

This framework lets you declare **subspaces**—hyper-rectangles within the full space that you actually support:

```cpp
// GPU: only contiguous, out-of-place, float types with non-deterministic
using GPUFloatSubspace = ValueAxes<Values<torch::kCUDA>,
                                   Values<torch::kFloat32, torch::kFloat64>,
                                   Values<Contiguity::Contiguous>,
                                   Values<Placement::OutOfPlace>,
                                   Values<Determinism::NonDeterministic>>;
```

Only the points in your declared subspaces get instantiated. Unsupported combinations produce clear runtime errors, not missing symbol linker errors or silent failures.

## Architecture

The framework is built in layers, each adding capabilities on top of the previous:

```
┌─────────────────────────────────────────────────────────────────┐
│  TorchDispatch.h  - PyTorch-specific utilities and error msgs   │
├─────────────────────────────────────────────────────────────────┤
│  DispatchTable.h  - DispatchTable with sparse subspaces   │
├─────────────────────────────────────────────────────────────────┤
│  ValueSpaceMap.h  - Map keyed by coordinates in a ValueSpace    │
├─────────────────────────────────────────────────────────────────┤
│  ValueSpace.h  - Cartesian products of value axes               │
├─────────────────────────────────────────────────────────────────┤
│  Values.h  - Compile-time value packs and operations            │
├─────────────────────────────────────────────────────────────────┤
│  IndexSpace.h  - N-dimensional index spaces and iteration       │
├─────────────────────────────────────────────────────────────────┤
│  Traits.h / TypesFwd.h  - Basic type traits and enums           │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Index Spaces (`IndexSpace.h`)

The foundation. An index space is an N-dimensional grid of size_t indices:

```cpp
using MySpace = Sizes<3, 4, 2>;  // 3×4×2 = 24 points
```

Provides:
- Rank, Numel (total elements), Shape queries
- Point type (`Point<i, j, k>`) for coordinates
- Linear index ↔ Point conversions
- `visit_index_space(visitor, space)` to iterate all points

### Layer 2: Value Packs (`Values.h`)

Compile-time sequences of arbitrary values (not just size_t):

```cpp
using Devices = Values<torch::kCPU, torch::kCUDA>;
using Dtypes  = Values<torch::kFloat32, torch::kFloat64, torch::kInt32>;
```

Provides:
- Size, element access, contains checks
- Subset relationships
- `ValuePack` concept and refinements (`UniqueValuePack`, `SameTypeValuePack`)

### Layer 3: Value Spaces (`ValueSpace.h`)

Cartesian products of value axes (which are unique, same-type, non-empty value packs):

```cpp
using MyDispatchSpace = ValueAxes<Devices, Dtypes>;  // 2×3 = 6 value combinations
```

Provides:
- Coordinate type (`Values<torch::kCUDA, torch::kFloat32>`)
- Space containment checks (`SpaceContains<Space, Coord>`)
- Subspace relationships (`SpaceHasSubspace<Space, SubSpace>`)
- Bidirectional conversion between value coordinates and index points
- `visit_value_space(visitor, space)` to iterate all coordinates

### Layer 4: Value Space Maps (`ValueSpaceMap.h`)

A `std::unordered_map` variant where keys are coordinates in a value space:

```cpp
ValueSpaceMap_t<MyDispatchSpace, FunctionPtr> table;
table.emplace(coord, &my_implementation);
auto it = table.find(runtime_coord);  // O(1) lookup
```

Features:
- Compile-time coordinate validation on insert
- Graceful "not found" for invalid runtime lookups
- Transparent hashing (no key object construction for lookups)

### Layer 5: Dispatch Tables (`DispatchTable.h`)

Combines everything into a dispatch table with sparse instantiation:

```cpp
using Dispatcher = DispatchTable<FullSpace, ReturnType(Args...)>;

// Instantiate only specific subspaces
static Dispatcher const table{
    Dispatcher::from_op<MyOp>(),
    CPUSubspace{},
    GPUSubspace{}
};

// Runtime dispatch
return table(coord_tuple, args...);
```

Two instantiation patterns:
- `from_op<Op>()` — Op is a struct with `static op(Tag<...>, args...)` overloads
- `from_visitor(lambda)` — lambda takes `(auto coord, args...)` and calls overloaded functions

### Layer 6: Torch Utilities (`TorchDispatch.h`)

PyTorch-specific conveniences:
- Pre-defined axes: `CpuCudaDeviceAxis`, `FullFloatDtypeAxis`, etc.
- Concepts: `IntegerTorchScalarType`, `FloatTorchScalarType`
- `torchDispatch()` wrapper with friendly error messages
- Coordinate stringification for diagnostics

## Examples

The `example/` directory contains three demonstrations:

### `ReLU.cu` — Simple 2D Dispatch

Shows the minimal pattern: dispatch across device (CPU/CUDA) and dtype (float types). The full space is just 2×4 = 8 points, all instantiated.

Key pattern: separate `op` overloads for CPU and CUDA, each templated on scalar type.

### `Functional.cu` — Free Function Overloads

A 5-dimensional dispatch space (device × dtype × contiguity × placement × determinism). Uses **overloaded free functions** with `requires` clauses to select implementations:

```cpp
template <torch::ScalarType stype, Contiguity contiguity, Determinism determinism>
TensorWithNotes iscan_impl(Tag<torch::kCPU, stype, contiguity, Placement::OutOfPlace, determinism>,
                           torch::Tensor input) { ... }

template <torch::ScalarType stype, Contiguity contiguity>
    requires IntegerTorchScalarType<stype>
TensorWithNotes iscan_impl(Tag<torch::kCPU, stype, contiguity, Placement::OutOfPlace, Determinism::Deterministic>,
                           torch::Tensor input) { ... }
```

The compiler's overload resolution picks the most specific match. **Sparse instantiation**: only 4 subspaces are declared, not the full 64-point space.

### `Op.cu` — Single Struct with if-constexpr

Same 5D space, same sparse subspaces, but uses a **single templated `op` function** with `if constexpr` branching:

```cpp
struct InclusiveScanOp {
    template <torch::DeviceType device, torch::ScalarType stype, Contiguity contiguity,
              Placement placement, Determinism determinism>
    static TensorWithNotes op(Tag<device, stype, contiguity, placement, determinism>,
                              torch::Tensor input) {
        if constexpr (device == torch::kCPU) {
            if constexpr (placement == Placement::InPlace) {
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

Both `Functional.cu` and `Op.cu` wrap the same stand-in "external library" (`ScanLib.h`), demonstrating how to integrate with libraries like CUB, Thrust, or CUTLASS.

## Usage Pattern Summary

1. **Define your space** as a `ValueAxes<Axis1, Axis2, ...>`
2. **Define subspaces** that cover only the combinations you support
3. **Write your implementation** as either:
   - A struct with `static op(Tag<...>, args...)` overloads, or
   - Free functions that take `Tag<...>` as the first parameter
4. **Create a static dispatch table** using `from_op<>()` or `from_visitor()`
5. **Call `torchDispatch()`** with the runtime coordinate tuple

## Technical Appendix: Why `consteval` Functions Instead of `constexpr` Variables?

Throughout this codebase, you'll notice patterns like:

```cpp
template <ValuePack Pack>
consteval size_t PackSize_v() {
    return PackSize<Pack>::value();
}
```

Instead of the more traditional:

```cpp
template <ValuePack Pack>
constexpr size_t PackSize_v = PackSize<Pack>::value;
```

This is intentional. When a `static constexpr` variable is passed to a function that takes it by const reference, the compiler may generate an address for that variable. With deeply nested templates (common in this dispatch machinery), this can cause:

1. **Symbol explosion**: Each unique template instantiation gets its own static storage
2. **Link-time bloat**: Especially problematic with nvcc, which can generate enormous object files
3. **ODR complications**: Multiple translation units may each define the same symbol

`consteval` functions are guaranteed to be evaluated at compile-time with **zero instantiation footprint**. There's no storage, no symbol, no address—just a compile-time computation that produces a value. This is critical when you're instantiating across a 5-dimensional dispatch space.

The trade-off is slightly more verbose syntax (`PackSize_v<Pack>()` vs `PackSize_v<Pack>`), but the compile-time and binary-size benefits are substantial for template-heavy code targeting nvcc.

---

## File Reference

| File | Purpose |
|------|---------|
| `Traits.h` | Basic type traits (tuple utilities, index_sequence detection) |
| `TypesFwd.h` | Forward declarations, `Values<>`, `Tag<>`, enums (Placement, Determinism, Contiguity) |
| `IndexSpace.h` | N-dimensional index spaces, Points, visitation |
| `Values.h` | Value packs, compile-time operations, concepts |
| `ValueSpace.h` | Value axes, value spaces (Cartesian products), coordinate conversion |
| `ValueSpaceMap.h` | Hash map keyed by value space coordinates |
| `DispatchTable.h` | `DispatchTable` class with sparse subspace instantiation |
| `TorchDispatch.h` | PyTorch-specific axes, concepts, error handling |
| `example/` | Working examples (ReLU, Functional, Op) with detailed headers |
