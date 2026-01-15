# General ideas about compile-time permutation spaces

Start with the simplest version - some number of axes, each one is characterized by a single
size_t Ni representing extent. So the total permutation space, linearized is the product of
all the Ni's.

Given N0, N1, ... Ni,

You can go from a set of selections on each axis v0, v1, ... vi to a linear index idx
easily. You can go from linear index to v0, v1, ... vi easily just using strides.

how to subspace?

subspace linear index (size_t) space::-> subspace coord (size_t... Is)
subspace coord (size_t... Is) subspace::-> point value pack (auto... Vs)
point value pack (auto... Vs) space::-> space coord (size_t... Is)
space coord (size_t... Is) space::-> space linear index (size_t)

................
Jan 15, 2026

The conceptual design of this dispatch framework has evolved a little as I've gotten it refined. The original idea was to be able to do controlled instantiation over select individual permutations and blocks of permutations over multiple dimensions of value-based variations. The motivation for this is to dispatch torch tensors into different kernels based on device and scalar type and sometimes channel order, so the "permutation space" might be something like this:

{ kCUDA, kCPU, kPrivateUse1 } x { kFloat32, kFloat64 } x { 32, 64, 128, 256, 512 }

However, after many design iterations, it became clear that the smartest way to approach this was first to project it down into just an index space where each axis had a size. In the example above, the index space is 3 x 2 x 5. The index space provides controls and useful typedefs for creating instantiated calls with a zero-storage tag-type that's essentially a std::index_sequence that represents a single point in the index space as a unique type.

It's then relatively trivial to convert the index point into a tag type which represents a specific permutation:

using one_specific_permutation = Values<kCUDA, kFloat32, 128>;

The thing which was previously called an AxisOuterProduct feels misnamed now. Its job is basically to do bidirectional conversion between the space portion which are axes of variation across specific types:

Axis0: { kCUDA, kCPU, kPrivateUse1 } type torch::DeviceType
Axis1: { kFloat32, kFloat64 } type torch::ScalarType
Axis2: { 32, 64, 128, 256, 512 } type int

these get mapped to an index space of Sizes<3, 2, 5>. The visit_index_space tool can then be used, and instead of the user providing a visitor that takes index coords like <0, 1, 2>, it will take value coords like <kCPU, kFloat64, 256>.

----------------------

Downstream from the AxisOuterProduct "concept" is a thing called a permutation map, where the value coord can be used as a key to create an index into either a std::array or a std::unordered_map, the permutation map is really nothing more than this - a container wrapper that uses ValueCoords as keys, either as types or as tuples.

-------------------------

Finally, Downstream from PermutationMap is the DispatchTable, which combines the map with the AxisOuterProduct to create a streamlined utility for creation of a dispatch table in which specific value permutations, either individually or in hyper-rectangles, are instantiated and stored as function pointers in the map.


I'm clear on what the job of Traits, Values, IndexSpace is, and PermutationMap and DispatchTable afterwards. I no longer thing AxisOuterProduct is the most descriptive name for this middle thing - and I'm also not sure that DimensionalAxis is the right name for a specific set of values of permutation on a single axis.

Given the style of the code in this dispatch directory and the overall design sensibility, and the explanataion above, help me brainstorm better names.

================================================================================
Op Implementation Patterns
================================================================================

The dispatch machinery handles the runtime lookup: a type-erased input (torch::Tensor, DLPack) 
produces a ValueTuple at runtime, which becomes a key into the dispatch table. The function 
pointer stored there was created during instantiation.

The design question: **what does the op implementer's code look like?**

Key insight: Device determines the *algorithm shape* (loops vs kernels), while dtype determines 
the *working type* within that algorithm. Value specialization (explicit specialization for 
every Device×DType combination) leads to combinatorial explosion and code duplication. Better 
to separate concerns.

The recommended pattern uses Tag<V> for device dispatch via overloading, with the scalar type 
as a template parameter. This works identically for free functions and functors.

### Free Function Style

```cpp
// Device-specific implementations, templated on scalar type
template <typename T>
void saxpy_impl(Tag<torch::kCPU>, T* y, T a, const T* x, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <typename T>
void saxpy_impl(Tag<torch::kCUDA>, T* y, T a, const T* x, size_t n) {
    saxpy_kernel<<<...>>>(y, a, x, n);
}

// Entry point from dispatch table - type erased
template <typename Coord>
void saxpy_dispatch(Coord, void* y, double a, const void* x, size_t n) {
    constexpr auto device = get<0>(Coord{});
    constexpr auto dtype  = get<1>(Coord{});
    using T = ScalarCppTypeT<dtype>;
    
    saxpy_impl(Tag<device>{},
               static_cast<T*>(y), T(a), static_cast<const T*>(x), n);
}
```

### Functor Style

```cpp
struct SaxpyOp {
    // Entry point from dispatch table
    template <typename Coord>
    static void call(Coord, void* y, double a, const void* x, size_t n) {
        constexpr auto device = get<0>(Coord{});
        constexpr auto dtype  = get<1>(Coord{});
        using T = ScalarCppTypeT<dtype>;
        
        run(Tag<device>{},
            static_cast<T*>(y), T(a), static_cast<const T*>(x), n);
    }
    
private:
    template <typename T>
    static void run(Tag<torch::kCPU>, T* y, T a, const T* x, size_t n) {
        for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
    }
    
    template <typename T>
    static void run(Tag<torch::kCUDA>, T* y, T a, const T* x, size_t n) {
        saxpy_kernel<<<...>>>(y, a, x, n);
    }
};
```

### Summary

The `call(Coord, ...)` signature is the type-erasure boundary. Everything below that is fully 
typed (`T*` not `void*`). Device selection is via tag dispatch (overload resolution), not 
if-constexpr chains. The actual algorithm is just normal C++ code.

The Coord's job is to inform type mapping and device selection, then it disappears. The inner 
code doesn't care about `Values<...>` or enums—it just works with real types.

The `Values<...>` / `ValueSpace` machinery defines the dispatch space, and `Tag<V>` is the 
overload selection mechanism at the implementation level. They compose nicely.
