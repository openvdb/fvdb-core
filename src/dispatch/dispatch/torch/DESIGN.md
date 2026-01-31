# Accessor & Tile Framework Design

This document captures the design conclusions for the elementwise dispatch framework,
with an eye toward future generalization to matmul-esque tile-based operations.

## Core Philosophy

**Users should not have to understand GPU mechanics.** They provide:
- Input/output element shapes (e.g., `mat3x3`, `vec3`)
- Scalar types (often inferred)
- The tensors themselves (framework infers device, contiguity, iteration count)
- A pure function: `inputs → outputs`

**The framework decides everything else:**
- Device dispatch (GPU/CPU)
- Contiguous vs strided path
- Block size (typically 256)
- Elements per thread (`16 / max_elem_bytes`, clamped to [1, 16])
- Load/store strategy per tensor
- Tile shape

---

## Vocabulary

| Term | Definition |
|------|------------|
| **Element** | The semantic unit of data (`vec3f`, `mat3x3f`, or tuple of these) |
| **Tile** | What the accessor returns when indexed — a chunk of elements |
| **Accessor** | Converts a type-erased tensor into a typed, shaped, indexable thing |
| **Fragment** | Per-thread portion of a tile (in cooperative/matmul case) |

**Tiles all the way down:** `accessor.load(idx) → tile → element → scalar`

Same abstraction at every level, different shapes.

---

## The Four Backend Algorithms

For elementwise operations, there are fundamentally **4 algorithms** (2 devices × 2 contiguity):

| Device | Contiguity | Load | Store |
|--------|------------|------|-------|
| GPU | Contiguous | Direct/vectorized | Direct/vectorized |
| GPU | Strided | Gather | Scatter |
| CPU | Contiguous | memcpy/loop | memcpy/loop |
| CPU | Strided | Gather loop | Scatter loop |

### Contiguity Decisions

- **Combined contiguity**: One decision for all inputs, one for all outputs
- Avoids 2^N kernel explosion
- Input/output contiguity can be separate (3-4 variants per device)
- Output strided case can be omitted (outputs are almost always contiguous)

---

## Per-Thread Local Storage

Each thread has local storage organized **per-tensor**, not as an array of tuples:

```cpp
// Per-tensor storage
std::array<mat3x3, elems_per_thread> R_local;
std::array<vec3, elems_per_thread>   T_local;
std::array<vec3, elems_per_thread>   x_local;

// Load each tensor (each with optimal strategy)
load_elements(R_base, idx, R_local);
load_elements(T_base, idx, T_local);
load_elements(x_base, idx, x_local);

// Call function composing the tuple on the fly
for (size_t i = 0; i < elems_per_thread; ++i) {
    y_local[i] = fn(R_local[i], T_local[i], x_local[i]);
}
```

This matches memory layout (tensors are separate in memory).

---

## Elements Per Thread

Determined by the **largest element** in the tuple:

```cpp
elems_per_thread = 16 / max_elem_bytes;  // clamped to [1, 16]
```

| elem_bytes | elems_per_thread | Load type |
|------------|------------------|-----------|
| 1 (int8) | 16 | char16 |
| 2 (half) | 8 | half8 via float4 |
| 4 (float) | 4 | float4 |
| 8 (double) | 2 | double2 |
| 16+ | 1 | multiple loads |

For tuples, the largest element limits vectorization (all tensors iterate together).

---

## Dispatch Axes

The policy (compile-time specialization) includes:

| Axis | Values | Runtime/Compile-time |
|------|--------|----------------------|
| Device | GPU, CPU | Runtime dispatch |
| Scalar type | float, half, double, ... | Runtime dispatch |
| Input contiguity | contiguous, strided | Runtime dispatch |
| Output contiguity | contiguous, (strided) | Runtime dispatch |
| **Storage** | registers, shared_memory | Compile-time (future) |
| **Tile shape** | `tile_shape<N>`, `tile_shape<M, N>` | Compile-time |

---

## Future: Matmul-Esque Operations

The same framework extends to cooperative/matmul operations by adding:

| Aspect | Elementwise | Matmul |
|--------|-------------|--------|
| Storage axis | `registers` | `shared_memory` |
| Tile shape | 1D (`tile_shape<4>`) | 2D+ (`tile_shape<128, 128>`) |
| Thread mapping | 1 thread = 1 tile | Many threads = 1 tile (cooperative) |
| Sync | None | `__syncthreads()` |
| Data reuse | None | Massive (shared memory) |

### Accessor Specialization

```cpp
template <typename Policy, typename Element, typename TileShape, storage_policy Storage>
struct accessor;

// Elementwise: register storage, no sync
template <...>
struct accessor<Policy, Element, TileShape, storage_policy::registers> {
    std::array<Element, TileShape::size> local;
    void load(index_t idx);   // independent, no sync
    void store(index_t idx);  // independent, no sync
};

// Matmul: shared memory, cooperative with sync
template <...>
struct accessor<Policy, Element, TileShape, storage_policy::shared_memory> {
    __shared__ Element tile[...];
    void load(tile_coord idx);   // cooperative, then __syncthreads()
    void store(tile_coord idx);  // __syncthreads(), then cooperative
};
```

---

## User-Facing API

```cpp
for_each_elements<inputs<mat3x3, vec3, vec3>, outputs<vec3>>(
    {R, T, x}, {y},
    [](mat3x3 R, vec3 T, vec3 x) -> vec3 {
        // Just math. No policy, no tiles, no load/store.
        vec3 result;
        for (int i = 0; i < 3; ++i) {
            result[i] = T[i];
            for (int j = 0; j < 3; ++j)
                result[i] += R[i*3+j] * x[j];
        }
        return result;
    }
);
```

User writes math. Framework handles everything else.

---

## C-Level Mental Model

At the lowest level, it's all bytes:

```c
// Per thread:
// 1. Calculate: src_addr = base + idx * elem_bytes
// 2. Load N bytes from src_addr → registers
// 3. Do compute (user's function on registers)
// 4. Calculate: dst_addr = out_base + idx * out_elem_bytes
// 5. Store M bytes from registers → dst_addr
```

The C++ layer adds type safety and template-based inlining.

---

## Key Insights

1. **Block size**: Just use 256. Almost always fine.

2. **Coalescing**: Happens automatically when consecutive threads access consecutive addresses. Don't break it, don't overthink it.

3. **Vectorization**: Use widest load possible (float4 = 16 bytes). Framework picks based on element size.

4. **Pass-by-value = local storage**: When the user's lambda receives `mat3x3 R`, those 9 floats are in registers. No escaping it at the element level.

5. **Strided = slow**: If any tensor is strided, we gather/scatter. Unavoidable, but rare in practice.

6. **Tiles all the way down**: Same abstraction at every level (accessor → tile → element). Same vocabulary for elementwise and matmul, just different shapes.

---

## Summary

| Layer | Who writes it | What they think about |
|-------|---------------|----------------------|
| Element function | User | Math only |
| `for_each_elements` | Framework | Dispatch, tile size, load/store strategy |
| Device kernels | Framework | Thread mapping, memory access patterns |
| Load/store helpers | Framework | Vectorization, coalescing, local buffers |

The user is at the top. They see only element shapes, scalar types, and math.
Everything below is the framework's job.
