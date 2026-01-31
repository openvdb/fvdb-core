// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// DESIGN EXPLORATION: Composable Iteration Space Abstractions
//
// This file is a design sketch - it won't compile. The goal is to explore
// how to minimize problem-specific code for structured elementwise operations
// by building composable, reusable abstractions inspired by APL/J/K adverbs.
//
// Compare to affine_xform_ternary.cu which is ~200 lines. With these
// abstractions, the problem-specific code should be ~10-20 lines.
//
//==============================================================================
// PART 1: SHAPE TYPES
//==============================================================================
//
// Compile-time shape descriptors. These are the foundation for describing
// iteration spaces and element shapes.

namespace dispatch {
namespace design {

// A compile-time shape with known dimensions
template <int64_t... Dims> struct shape {
    static constexpr int64_t rank       = sizeof...(Dims);
    static constexpr int64_t dims[rank] = {Dims...};

    static constexpr int64_t
    size() {
        return (Dims * ... * 1);
    }

    static constexpr int64_t
    dim(int64_t i) {
        return dims[i];
    }
};

// Common shape aliases
using scalar    = shape<>;
using vector3   = shape<3>;
using matrix3x3 = shape<3, 3>;

// A dynamic shape (runtime-known iteration dimension)
template <int64_t... StaticDims> struct dynamic_shape {
    int64_t dynamic_dim; // The iteration dimension (N)
    // Static element dimensions follow

    static constexpr int64_t static_rank = sizeof...(StaticDims);
    static constexpr int64_t total_rank  = 1 + static_rank;
};

//==============================================================================
// PART 2: ITERATION SPACE TYPES
//==============================================================================
//
// These types describe HOW we iterate over data, separate from what the
// data contains.

// indexed<Shape>: An N-D space with explicit indices
// - Used for the "outer" iteration dimension
// - Runtime-sized (we don't know N at compile time)
template <typename Shape = shape<>> struct indexed {
    using element_shape = Shape;
    int64_t count; // Runtime iteration count
};

// elemental<Shape>: A contiguous chunk treated as a unit
// - Used for the "inner" element shape
// - Compile-time sized
template <typename Shape> struct elemental {
    using shape_type                    = Shape;
    static constexpr int64_t rank       = Shape::rank;
    static constexpr int64_t total_size = Shape::size();
};

// Convenience: elemental from dimension pack
template <int64_t... Dims> using elem = elemental<shape<Dims...>>;

// flattened<Space>: Linearize an N-D space to 1-D
// - Converts multi-index iteration to linear iteration
template <typename Space> struct flattened {
    using inner_space = Space;
};

// zipped<Spaces...>: Synchronous iteration over multiple spaces
// - All spaces must have compatible iteration dimensions
// - Used to describe multi-input operations
template <typename... Spaces> struct zipped {
    static constexpr size_t arity = sizeof...(Spaces);
};

//==============================================================================
// PART 3: TENSOR VIEW - SEPARATING ITERATION FROM ELEMENT SHAPE
//==============================================================================
//
// A tensor view describes:
//   - The iteration shape (how many elements to process)
//   - The element shape (what each element looks like)
//   - The tag (device, scalar type, contiguity)
//
// This separation is key: we iterate over N things, each thing is a 3x3 matrix.

template <typename ElemShape, typename Tag> struct tensor_view {
    using element_shape = ElemShape;
    using tag_type      = Tag;

    // The actual tensor data
    torch::Tensor tensor;

    // Iteration count (first dimension)
    int64_t
    iteration_count() const {
        return tensor.size(0);
    }

    // Element size (product of remaining dimensions)
    static constexpr int64_t
    element_size() {
        return ElemShape::size();
    }
};

// Factory: wrap a tensor with its element shape declaration
template <typename ElemShape, typename Tag>
tensor_view<ElemShape, Tag>
make_view(Tag, torch::Tensor t) {
    return tensor_view<ElemShape, Tag>{t};
}

//==============================================================================
// PART 4: ELEMENT ACCESS PROTOCOLS
//==============================================================================
//
// For contiguous tensors: return pointer (writes go directly to memory)
// For strided tensors: return value type (must be scattered after modification)
//
// This abstraction hides the gather/scatter logic inside the accessor.

// element_ref: Pointer-like access for contiguous data
template <typename T, typename Shape> struct element_ref {
    T *data;

    __hostdev__ T &
    operator[](int64_t i) const {
        return data[i];
    }

    // For 2D access
    __hostdev__ T &
    operator()(int64_t i, int64_t j) const {
        static_assert(Shape::rank == 2);
        return data[i * Shape::dim(1) + j];
    }

    __hostdev__ T *
    ptr() const {
        return data;
    }
};

// element_val: Value-like access for strided data (gathered into registers)
template <typename T, typename Shape> struct element_val {
    T data[Shape::size()];

    __hostdev__ T &
    operator[](int64_t i) {
        return data[i];
    }

    __hostdev__ T const &
    operator[](int64_t i) const {
        return data[i];
    }

    // For 2D access
    __hostdev__ T &
    operator()(int64_t i, int64_t j) {
        static_assert(Shape::rank == 2);
        return data[i * Shape::dim(1) + j];
    }

    __hostdev__ T const &
    operator()(int64_t i, int64_t j) const {
        static_assert(Shape::rank == 2);
        return data[i * Shape::dim(1) + j];
    }

    __hostdev__ T *
    ptr() {
        return data;
    }

    __hostdev__ T const *
    ptr() const {
        return data;
    }
};

// The accessor returns the appropriate type based on contiguity
template <typename T, typename Shape, contiguity Contig>
using element_access_t = std::
    conditional_t<Contig == contiguity::contiguous, element_ref<T, Shape>, element_val<T, Shape>>;

//==============================================================================
// PART 5: SINGLE-TENSOR ACCESSOR
//==============================================================================
//
// An accessor wraps a tensor and provides element access at iteration indices.
// The contiguity determines whether we return pointers or gathered values.

template <typename ElemShape, torch::ScalarType Stype, contiguity Contig> struct single_accessor;

// Contiguous specialization: direct pointer access
template <typename ElemShape, torch::ScalarType Stype>
struct single_accessor<ElemShape, Stype, contiguity::contiguous> {
    using T            = torch_scalar_cpp_type_t<Stype>;
    using element_type = element_ref<T, ElemShape>;

    T *data;
    int64_t element_size;
    int64_t iter_size; // for broadcast

    __hostdev__ element_type
    get(int64_t n) const {
        int64_t idx = (iter_size == 1) ? 0 : n;
        return element_type{data + idx * element_size};
    }

    static single_accessor
    from_tensor(torch::Tensor t) {
        single_accessor acc;
        acc.data         = t.data_ptr<T>();
        acc.element_size = ElemShape::size();
        acc.iter_size    = t.size(0);
        return acc;
    }
};

// Strided specialization: gather into registers
template <typename ElemShape, torch::ScalarType Stype>
struct single_accessor<ElemShape, Stype, contiguity::strided> {
    using T            = torch_scalar_cpp_type_t<Stype>;
    using element_type = element_val<T, ElemShape>;

    T *data;
    int64_t strides[1 + ElemShape::rank]; // iteration stride + element strides
    int64_t sizes[1 + ElemShape::rank];

    __hostdev__ element_type
    get(int64_t n) const {
        element_type elem;
        int64_t idx = (sizes[0] == 1) ? 0 : n;
        T *base     = data + idx * strides[0];

        // Gather based on element shape rank
        if constexpr (ElemShape::rank == 1) {
            DISPATCH_UNROLL
            for (int64_t i = 0; i < ElemShape::dim(0); ++i) {
                elem[i] = base[i * strides[1]];
            }
        } else if constexpr (ElemShape::rank == 2) {
            DISPATCH_UNROLL
            for (int64_t i = 0; i < ElemShape::dim(0); ++i) {
                DISPATCH_UNROLL
                for (int64_t j = 0; j < ElemShape::dim(1); ++j) {
                    elem(i, j) = base[i * strides[1] + j * strides[2]];
                }
            }
        }
        // ... extend for higher ranks as needed

        return elem;
    }

    static single_accessor
    from_tensor(torch::Tensor t) {
        single_accessor acc;
        acc.data = t.data_ptr<T>();
        for (int64_t d = 0; d < 1 + ElemShape::rank; ++d) {
            acc.strides[d] = t.stride(d);
            acc.sizes[d]   = t.size(d);
        }
        return acc;
    }
};

//==============================================================================
// PART 6: ZIPPED ACCESSOR (TUPLE-BASED COMPOSITE)
//==============================================================================
//
// A zipped accessor bundles multiple single accessors and provides
// synchronized access. At each iteration index n, it returns a tuple
// of elements from all inputs.

template <typename... Accessors> struct zipped_accessor {
    cuda::std::tuple<Accessors...> accessors;

    // Get element tuple at iteration index n
    template <size_t... Is>
    __hostdev__ auto
    get_impl(int64_t n, std::index_sequence<Is...>) const {
        return cuda::std::make_tuple(cuda::std::get<Is>(accessors).get(n)...);
    }

    __hostdev__ auto
    get(int64_t n) const {
        return get_impl(n, std::index_sequence_for<Accessors...>{});
    }

    // Access individual accessor by index
    template <size_t I>
    __hostdev__ auto &
    accessor() {
        return cuda::std::get<I>(accessors);
    }
};

// Factory: create zipped accessor from element shapes and tensors
template <typename Tag, typename... ElemShapes, size_t... Is>
auto
make_zipped_accessor_impl(Tag tag,
                          std::index_sequence<Is...>,
                          types<ElemShapes...>,
                          torch::Tensor const &...tensors) {
    using Stype                 = decltype(Tag::stype)::value;
    constexpr contiguity Contig = Tag::contig;

    return zipped_accessor<single_accessor<ElemShapes, Stype, Contig>...>{
        cuda::std::make_tuple(single_accessor<ElemShapes, Stype, Contig>::from_tensor(tensors)...)};
}

template <typename... ElemShapes, typename Tag, typename... Tensors>
auto
make_zipped_accessor(Tag tag, types<ElemShapes...> shapes, Tensors const &...tensors) {
    static_assert(sizeof...(ElemShapes) == sizeof...(Tensors));
    return make_zipped_accessor_impl(
        tag, std::index_sequence_for<ElemShapes...>{}, shapes, tensors...);
}

//==============================================================================
// PART 7: APL-STYLE ADVERBS
//==============================================================================
//
// Adverbs modify how operations apply over structure.

//------------------------------------------------------------------------------
// Each<F>: Map F over the iteration dimension
//------------------------------------------------------------------------------
// This is essentially for_each but expressed as a composable type.

template <typename F> struct Each {
    // Apply F to each element in the iteration space
    template <typename Tag, typename Accessor>
    static void
    apply(Tag t, int64_t count, Accessor acc, F f) {
        for_each(t, count, [=] __hostdev__(Tag, int64_t n) mutable { f(t, acc.get(n)); });
    }
};

//------------------------------------------------------------------------------
// Over<Rank>: Apply operation at a specific rank
//------------------------------------------------------------------------------
// Like J's rank conjunction. f Over<0> applies to scalars, Over<1> to vectors.

template <int64_t Rank> struct Over {
    // Rank 0: scalar operation (current unary_elementwise)
    // Rank 1: vector operation (apply to each vector)
    // Rank 2: matrix operation (apply to each matrix)
};

//------------------------------------------------------------------------------
// Zip<F, InputShapes..., OutputShape>: Zipped map with multiple inputs
//------------------------------------------------------------------------------
// The core abstraction for multi-input elementwise operations.

template <typename F, typename InputShapes, typename OutputShape> struct ZipMap;

// Specialization for zipped inputs
template <typename F, typename... InShapes, typename OutShape>
struct ZipMap<F, zipped<elemental<InShapes>...>, elemental<OutShape>> {
    using input_shapes                  = types<InShapes...>;
    using output_shape                  = OutShape;
    static constexpr size_t input_arity = sizeof...(InShapes);

    //--------------------------------------------------------------------------
    // The core operation template
    //--------------------------------------------------------------------------
    template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig>
    static void
    op(tag<Dev, Stype, Contig> t,
       std::array<torch::Tensor, input_arity> const &inputs,
       torch::Tensor output) {
        auto guard = make_device_guard(t, output);

        // Create zipped input accessor
        auto in_acc = make_zipped_accessor_from_array<input_shapes>(t, inputs);

        // Create output accessor
        auto out_acc = single_accessor<OutShape, Stype, Contig>::from_tensor(output);

        int64_t const N = output.size(0);

        for_each(t, N, [=] __hostdev__(tag<Dev, Stype, Contig>, int64_t n) mutable {
            auto in_elements = in_acc.get(n);
            auto out_element = out_acc.get(n);
            F::apply(t, in_elements, out_element);
        });
    }

    //--------------------------------------------------------------------------
    // Dispatch space definition
    //--------------------------------------------------------------------------
    using space =
        axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;

    // ... dispatcher and map() similar to unary_elementwise
};

//==============================================================================
// PART 8: AFFINE TRANSFORM - THE GOAL
//==============================================================================
//
// With all the above machinery, the affine transform reduces to:

// THE ONLY PROBLEM-SPECIFIC CODE:
struct affine_element_op {
    template <typename T, typename Shape3x3, typename Shape3>
    __hostdev__ static void
    apply(auto tag,
          cuda::std::tuple<element_access_t<T, Shape3x3, contiguity::contiguous>,
                           element_access_t<T, Shape3, contiguity::contiguous>,
                           element_access_t<T, Shape3, contiguity::contiguous>> const &in,
          element_access_t<T, Shape3, contiguity::contiguous> &out) {
        auto const &[R, t, x] = in;

        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            T sum = t[i];
            DISPATCH_UNROLL
            for (int j = 0; j < 3; ++j) {
                sum += R(i, j) * x[j];
            }
            out[i] = sum;
        }
    }
};

// GENERIC WIRING (could be even more concise with macros/helpers):
using affine_op = ZipMap<affine_element_op,
                         zipped<elem<3, 3>, elem<3>, elem<3>>, // inputs: R, t, x
                         elem<3>>;                             // output: y

// PUBLIC API:
torch::Tensor
example_affine_xform_v2(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    // Validation (could be generated from shape descriptors)
    TORCH_CHECK_VALUE(R.dim() == 3 && R.size(1) == 3 && R.size(2) == 3, "R must be (N, 3, 3)");
    TORCH_CHECK_VALUE(T.dim() == 2 && T.size(1) == 3, "T must be (N, 3)");
    TORCH_CHECK_VALUE(x.dim() == 2 && x.size(1) == 3, "x must be (N, 3)");

    int64_t N = x.size(0);
    if (N == 0)
        return torch::empty({0, 3}, x.options());

    auto output = torch::empty({N, 3}, x.options());

    // Dispatch (the generic machinery handles this)
    affine_op::map("affine_xform_v2", {R, T, x}, output);

    return output;
}

//==============================================================================
// PART 9: ALTERNATIVE SYNTAX IDEAS
//==============================================================================
//
// Even more concise approaches to consider:

// Idea 1: Infer element shapes from tensor dimensions
//
// Instead of declaring shapes, infer them:
//   auto op = zip_over([](R, t, x, y) { /* math */ });
//   op(tensors...); // shapes inferred from tensor.sizes()

// Idea 2: DSL-like declaration
//
//   DEFINE_ZIPPED_OP(affine_xform,
//       INPUTS((R, matrix3x3), (t, vector3), (x, vector3)),
//       OUTPUT(vector3),
//       COMPUTE(R, t, x, y) {
//           // math
//       }
//   );

// Idea 3: Constexpr lambda for element op
//
//   constexpr auto affine_elem = []<typename T>(auto R, auto t, auto x, auto y) {
//       // math
//   };
//   using affine_op = ZipMap<decltype(affine_elem), ...>;

//==============================================================================
// PART 10: SCATTER PROTOCOL FOR STRIDED OUTPUTS
//==============================================================================
//
// For strided outputs, we need to scatter the result back to memory.
// The accessor should handle this transparently.

template <typename ElemShape, torch::ScalarType Stype> struct output_accessor_strided {
    using T            = torch_scalar_cpp_type_t<Stype>;
    using element_type = element_val<T, ElemShape>;

    T *data;
    int64_t strides[1 + ElemShape::rank];

    // Get a value to write into (caller fills it, then calls set)
    __hostdev__ element_type
    get(int64_t n) const {
        return element_type{}; // Zero-initialized for output
    }

    // Scatter the value back to strided memory
    __hostdev__ void
    set(int64_t n, element_type const &elem) const {
        T *base = data + n * strides[0];

        if constexpr (ElemShape::rank == 1) {
            DISPATCH_UNROLL
            for (int64_t i = 0; i < ElemShape::dim(0); ++i) {
                base[i * strides[1]] = elem[i];
            }
        } else if constexpr (ElemShape::rank == 2) {
            DISPATCH_UNROLL
            for (int64_t i = 0; i < ElemShape::dim(0); ++i) {
                DISPATCH_UNROLL
                for (int64_t j = 0; j < ElemShape::dim(1); ++j) {
                    base[i * strides[1] + j * strides[2]] = elem(i, j);
                }
            }
        }
    }
};

// Alternative: RAII-style scatterer
template <typename Accessor, typename Element> struct scoped_scatter {
    Accessor &acc;
    int64_t n;
    Element elem;

    __hostdev__
    scoped_scatter(Accessor &a, int64_t idx)
        : acc(a), n(idx), elem(a.get(idx)) {}

    __hostdev__ ~scoped_scatter() { acc.set(n, elem); }

    __hostdev__ Element &
    operator*() {
        return elem;
    }
};

//==============================================================================
// PART 11: CHUNKED EVALUATION (FUTURE DIRECTION)
//==============================================================================
//
// The for_each machinery has GrainSize for ILP. We could extend accessors
// to work with chunks for better memory access patterns.

template <int64_t ChunkSize, typename ElemShape, torch::ScalarType Stype> struct chunked_accessor {
    using T              = torch_scalar_cpp_type_t<Stype>;
    using single_element = element_ref<T, ElemShape>;
    using chunk_type     = std::array<single_element, ChunkSize>;

    T *data;
    int64_t element_size;

    // Get a chunk of ChunkSize elements starting at base_n
    __hostdev__ chunk_type
    get_chunk(int64_t base_n) const {
        chunk_type chunk;
        DISPATCH_UNROLL
        for (int64_t g = 0; g < ChunkSize; ++g) {
            chunk[g] = single_element{data + (base_n + g) * element_size};
        }
        return chunk;
    }
};

// This would integrate with for_each's grain loop:
//
// for_each<GrainSize>(t, N, [=](tag, int64_t base_n) {
//     auto in_chunk = in_acc.get_chunk<GrainSize>(base_n);
//     auto out_chunk = out_acc.get_chunk<GrainSize>(base_n);
//     for (int g = 0; g < GrainSize; ++g) {
//         if (base_n + g < N) {
//             F::apply(t, in_chunk[g], out_chunk[g]);
//         }
//     }
// });

//==============================================================================
// PART 12: BROADCAST HANDLING (SKETCH)
//==============================================================================
//
// The current element_accessor handles broadcast on the iteration dimension
// (size(0) == 1 means broadcast). We can generalize this.

template <typename ElemShape, torch::ScalarType Stype, contiguity Contig>
struct broadcast_aware_accessor {
    using base_accessor = single_accessor<ElemShape, Stype, Contig>;
    base_accessor inner;
    bool is_broadcast; // true if iter_size == 1

    __hostdev__ auto
    get(int64_t n) const {
        return inner.get(is_broadcast ? 0 : n);
    }
};

//==============================================================================
// PART 13: PUTTING IT ALL TOGETHER - COMPLETE PATTERN
//==============================================================================
//
// Here's how a complete structured elementwise op would look:

/*
// 1. Define the element operation (PROBLEM-SPECIFIC)
struct my_element_op {
    template <typename T>
    __hostdev__ static void
    apply(auto tag, auto inputs, auto& output) {
        auto [a, b, c] = inputs;
        // ... computation using a, b, c, writing to output ...
    }
};

// 2. Declare the op with shapes (PROBLEM-SPECIFIC)
using my_op = ZipMap<
    my_element_op,
    zipped<elem<...>, elem<...>, elem<...>>,  // input element shapes
    elem<...>                                   // output element shape
>;

// 3. Optionally customize dispatch axes
// using my_op = ZipMap<..., CustomDeviceAxis, CustomStypeAxis>;

// 4. Call it
torch::Tensor result = my_op::map("my_op", input1, input2, input3);
*/

//==============================================================================
// SUMMARY
//==============================================================================
//
// Key insights from this exploration:
//
// 1. SEPARATION OF CONCERNS
//    - Iteration shape vs element shape vs scalar type
//    - Each concern is a separate template parameter
//
// 2. COMPOSABLE PRIMITIVES
//    - shape<Dims...> for compile-time shapes
//    - indexed<>, elemental<>, zipped<> for iteration spaces
//    - single_accessor, zipped_accessor for data access
//
// 3. APL-STYLE ADVERBS
//    - Each<F>, Over<Rank>, ZipMap<F, Inputs, Output>
//    - Operations compose over iteration structure
//
// 4. CONTIGUITY ABSTRACTION
//    - element_ref (pointer) vs element_val (gathered value)
//    - Accessor hides gather/scatter logic
//
// 5. MINIMAL PROBLEM-SPECIFIC CODE
//    - User writes: element shapes + inner computation
//    - Library provides: accessor creation, dispatch, for_each, gather/scatter
//
// The goal: affine_xform_ternary.cu's 200+ lines become ~20 lines of
// problem-specific code plus a type alias for the operation.

//==============================================================================
//==============================================================================
//
// PART II: THE MAPPED CALCULATION
//
// The previous sections focused on iteration structure and accessors.
// Now we explore the "soul" of an op: the inner computation itself.
//
//==============================================================================
//==============================================================================

//==============================================================================
// PART 14: WHAT IS THE MAPPED CALCULATION?
//==============================================================================
//
// The "mapped calculation" is what happens at each iteration point, stripped
// of all the machinery that gets us there. It's the function f in:
//
//     for each n: output[n] = f(input[n])
//
// Or more generally for zipped inputs:
//
//     for each n: output[n] = f(input1[n], input2[n], ...)
//
// Key insight: The calculation is SEPARABLE from the iteration structure.
// The same f can be applied:
//   - To each element (map)
//   - Across elements to produce one result (reduce)
//   - Cumulatively across elements (scan)
//   - At different "ranks" (APL's rank operator)
//
// This is the APL/J/K philosophy: operations are nouns, adverbs modify them.
//
//   +      is addition (a verb)
//   +/     is sum (verb modified by reduce adverb)
//   +\     is cumulative sum (verb modified by scan adverb)
//   +/"1   is sum along axis 1 (reduce at rank 1)
//
// Our framework should capture this: the calculation is one thing,
// how it's applied is another.

//==============================================================================
// PART 15: COMPUTATION EXPRESSION STYLES
//==============================================================================
//
// Different ways to express the same inner calculation:

//------------------------------------------------------------------------------
// Style A: Pure Value Function
//------------------------------------------------------------------------------
// Takes values, returns value. No mutation, no pointers.
// Easiest to reason about, compose, and test.

template <typename T> struct affine_pure {
    // Input: R (3x3), t (3), x (3) as value arrays
    // Output: y (3) as returned value array
    __hostdev__ static std::array<T, 3>
    apply(std::array<std::array<T, 3>, 3> const &R,
          std::array<T, 3> const &t,
          std::array<T, 3> const &x) {
        std::array<T, 3> y;
        for (int i = 0; i < 3; ++i) {
            y[i] = t[i];
            for (int j = 0; j < 3; ++j) {
                y[i] += R[i][j] * x[j];
            }
        }
        return y;
    }
};

//------------------------------------------------------------------------------
// Style B: Output Reference (In-Place Mutation)
//------------------------------------------------------------------------------
// Takes input values, writes to output reference.
// More efficient (no copy on return), but less composable.

template <typename T> struct affine_mutate {
    __hostdev__ static void
    apply(T const (&R)[3][3], T const (&t)[3], T const (&x)[3], T (&y)[3]) {
        for (int i = 0; i < 3; ++i) {
            y[i] = t[i];
            for (int j = 0; j < 3; ++j) {
                y[i] += R[i][j] * x[j];
            }
        }
    }
};

//------------------------------------------------------------------------------
// Style C: Pointer-Based (Current Style)
//------------------------------------------------------------------------------
// Takes pointers. Compatible with both contiguous and gathered data.
// Matches what accessors naturally provide.

template <typename T> struct affine_pointer {
    __hostdev__ static void
    apply(T const *R, T const *t, T const *x, T *y) {
        for (int i = 0; i < 3; ++i) {
            y[i] = t[i];
            for (int j = 0; j < 3; ++j) {
                y[i] += R[i * 3 + j] * x[j];
            }
        }
    }
};

//------------------------------------------------------------------------------
// Style D: Element Wrapper (Our Proposed Style)
//------------------------------------------------------------------------------
// Takes element wrappers that abstract contiguous vs strided.
// The wrapper provides operator[] regardless of underlying storage.

template <typename T, typename RWrap, typename VWrap> struct affine_wrapped {
    __hostdev__ static void
    apply(RWrap const &R, VWrap const &t, VWrap const &x, VWrap &y) {
        for (int i = 0; i < 3; ++i) {
            y[i] = t[i];
            for (int j = 0; j < 3; ++j) {
                y[i] += R(i, j) * x[j];
            }
        }
    }
};

//------------------------------------------------------------------------------
// Style E: Accumulator/Fold Style
//------------------------------------------------------------------------------
// For reductions, the calculation takes an accumulator.
// This shows how the same operation morphs for reduce vs map.

template <typename T> struct dot_accumulator {
    // This is the "kernel" of a dot product
    // Applied via reduce: result = fold(dot_accumulator, 0, zip(a, b))
    __hostdev__ static T
    combine(T acc, T a, T b) {
        return acc + a * b;
    }

    static constexpr T identity = T(0);
};

//------------------------------------------------------------------------------
// Style F: Decomposed into Primitives
//------------------------------------------------------------------------------
// Express affine as composition of smaller operations.
// This is the most "APL-like" approach.

template <typename T> struct primitives {
    // Scalar operations
    __hostdev__ static T
    add(T a, T b) {
        return a + b;
    }
    __hostdev__ static T
    mul(T a, T b) {
        return a * b;
    }
    __hostdev__ static T
    fma(T a, T b, T c) {
        return a + b * c;
    } // a + b*c

    // Vector operations (operate on pointers with known size)
    template <int N>
    __hostdev__ static T
    dot(T const *a, T const *b) {
        T sum = T(0);
        for (int i = 0; i < N; ++i)
            sum += a[i] * b[i];
        return sum;
    }

    template <int N>
    __hostdev__ static void
    axpy(T a, T const *x, T const *y, T *out) {
        // out = a*x + y
        for (int i = 0; i < N; ++i)
            out[i] = a * x[i] + y[i];
    }

    template <int N>
    __hostdev__ static void
    add_vec(T const *a, T const *b, T *out) {
        for (int i = 0; i < N; ++i)
            out[i] = a[i] + b[i];
    }

    // Matrix-vector: y = R @ x
    template <int M, int N>
    __hostdev__ static void
    matvec(T const *R, T const *x, T *y) {
        for (int i = 0; i < M; ++i) {
            y[i] = dot<N>(R + i * N, x);
        }
    }

    // Affine: y = R @ x + t (composed from primitives)
    template <int N>
    __hostdev__ static void
    affine(T const *R, T const *t, T const *x, T *y) {
        matvec<N, N>(R, x, y); // y = R @ x
        add_vec<N>(y, t, y);   // y = y + t (in-place)
    }
};

//==============================================================================
// PART 16: PRIMITIVE OPERATIONS AND COMPOSITION
//==============================================================================
//
// The primitives above suggest a compositional structure:
//
//   Level 0: Scalar ops     - add, mul, fma, max, min, etc.
//   Level 1: Vector ops     - dot, axpy, scale, norm, etc.
//   Level 2: Matrix ops     - matvec, matmul, outer, etc.
//   Level 3: Composed ops   - affine, quaternion_rotate, etc.
//
// Each level can be expressed in terms of lower levels:
//
//   dot(a, b)     = reduce(add, map(mul, zip(a, b)))
//   matvec(R, x)  = map(row => dot(row, x), rows(R))
//   affine(R,t,x) = add(matvec(R, x), t)
//
// This is exactly the APL/J decomposition:
//
//   dot     ←  +/ ∘ ×           (reduce-add after multiply)
//   matvec  ←  dot"1            (dot at rank 1, i.e., per row)
//   affine  ←  +                (add translation after matvec)
//
// The question for our framework: which level do we operate at?

//------------------------------------------------------------------------------
// Option 1: User writes at Level 3 (monolithic inner op)
//------------------------------------------------------------------------------
// Current approach. User writes the full affine loop.
// Pros: Maximum control, no abstraction overhead
// Cons: Boilerplate, easy to make mistakes, no reuse

//------------------------------------------------------------------------------
// Option 2: User composes from Level 1-2 primitives
//------------------------------------------------------------------------------
// Library provides dot, matvec, etc. User composes.

/*
struct affine_composed_op {
    template <typename T>
    __hostdev__ static void
    apply(auto R, auto t, auto x, auto y) {
        matvec<3,3>(R.ptr(), x.ptr(), y.ptr());
        add_vec<3>(y.ptr(), t.ptr(), y.ptr());
    }
};
*/

// Pros: Reusable primitives, clearer intent
// Cons: Intermediate writes (y written twice), less fusion

//------------------------------------------------------------------------------
// Option 3: Expression templates / lazy evaluation
//------------------------------------------------------------------------------
// Build an expression tree, evaluate at the end.

/*
auto expr = t + R * x;  // builds expression tree
eval(expr, y);          // evaluates into y
*/

// Pros: Optimal fusion, very clean syntax
// Cons: Complex to implement, compile time cost, debugging difficulty

//------------------------------------------------------------------------------
// Option 4: Fused primitives
//------------------------------------------------------------------------------
// Provide common fused operations directly.

/*
template <int N>
__hostdev__ void affine_fused(T const* R, T const* t, T const* x, T* y) {
    // Single pass: y[i] = t[i] + sum_j(R[i,j] * x[j])
    for (int i = 0; i < N; ++i) {
        T sum = t[i];
        for (int j = 0; j < N; ++j) sum += R[i*N+j] * x[j];
        y[i] = sum;
    }
}
*/

// Pros: Optimal performance, still expressive
// Cons: Need to anticipate common fusions

//==============================================================================
// PART 17: AFFINE TRANSFORM - MULTIPLE EXPRESSIONS
//==============================================================================
//
// Here's the affine transform expressed in different styles.
// All are semantically equivalent: y = R @ x + t

//------------------------------------------------------------------------------
// Expression 1: Nested loops (imperative)
//------------------------------------------------------------------------------

template <typename T>
__hostdev__ void
affine_v1(T const *R, T const *t, T const *x, T *y) {
    for (int i = 0; i < 3; ++i) {
        T sum = t[i];
        for (int j = 0; j < 3; ++j) {
            sum += R[i * 3 + j] * x[j];
        }
        y[i] = sum;
    }
}

//------------------------------------------------------------------------------
// Expression 2: Dot product per row (functional decomposition)
//------------------------------------------------------------------------------

template <typename T>
__hostdev__ T
dot3(T const *a, T const *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
__hostdev__ void
affine_v2(T const *R, T const *t, T const *x, T *y) {
    y[0] = t[0] + dot3(R + 0, x);
    y[1] = t[1] + dot3(R + 3, x);
    y[2] = t[2] + dot3(R + 6, x);
}

//------------------------------------------------------------------------------
// Expression 3: FMA chain (maximizing FMA usage)
//------------------------------------------------------------------------------

template <typename T>
__hostdev__ void
affine_v3(T const *R, T const *t, T const *x, T *y) {
    // y[i] = fma(R[i,2], x[2], fma(R[i,1], x[1], fma(R[i,0], x[0], t[i])))
    for (int i = 0; i < 3; ++i) {
        T acc = t[i];
        acc   = fma(R[i * 3 + 0], x[0], acc);
        acc   = fma(R[i * 3 + 1], x[1], acc);
        acc   = fma(R[i * 3 + 2], x[2], acc);
        y[i]  = acc;
    }
}

//------------------------------------------------------------------------------
// Expression 4: Transposed loop order (better for some memory layouts)
//------------------------------------------------------------------------------

template <typename T>
__hostdev__ void
affine_v4(T const *R, T const *t, T const *x, T *y) {
    // Initialize with t
    y[0] = t[0];
    y[1] = t[1];
    y[2] = t[2];

    // Accumulate R @ x column-by-column
    for (int j = 0; j < 3; ++j) {
        T xj = x[j];
        y[0] += R[0 * 3 + j] * xj;
        y[1] += R[1 * 3 + j] * xj;
        y[2] += R[2 * 3 + j] * xj;
    }
}

//------------------------------------------------------------------------------
// Expression 5: Fully unrolled (what the compiler hopefully produces)
//------------------------------------------------------------------------------

template <typename T>
__hostdev__ void
affine_v5(T const *R, T const *t, T const *x, T *y) {
    y[0] = t[0] + R[0] * x[0] + R[1] * x[1] + R[2] * x[2];
    y[1] = t[1] + R[3] * x[0] + R[4] * x[1] + R[5] * x[2];
    y[2] = t[2] + R[6] * x[0] + R[7] * x[1] + R[8] * x[2];
}

//------------------------------------------------------------------------------
// Which is "canonical"?
//------------------------------------------------------------------------------
// For a framework, Style 1 (nested loops) is probably best:
//   - Most explicit about the computation
//   - Compiler can unroll and optimize
//   - Generalizes to any dimension N
//   - DISPATCH_UNROLL hints available
//
// But the framework should allow any of these - it just receives a callable.

//==============================================================================
// PART 18: CHUNKS AND ILP (INSTRUCTION-LEVEL PARALLELISM)
//==============================================================================
//
// Even for "elementwise" operations, processing multiple elements per thread
// iteration improves performance via ILP. This is what GrainSize does in
// for_each.
//
// The question: should the inner calculation be aware of chunks?

//------------------------------------------------------------------------------
// Approach A: Opaque to the calculation
//------------------------------------------------------------------------------
// The calculation sees one element at a time. The framework handles chunking.
// This is our current approach with for_each's grain loop.

/*
for_each<GrainSize>(t, N, [=](tag, int64_t base) {
    #pragma unroll
    for (int g = 0; g < GrainSize; ++g) {
        if (base + g < N) {
            auto in = gather.get(base + g);
            auto out = scatter.at(base + g);
            f(in, out);  // f is unaware of chunking
        }
    }
});
*/

// Pros: Simple calculation, chunking is automatic
// Cons: May miss optimization opportunities (register reuse, vectorization)

//------------------------------------------------------------------------------
// Approach B: Chunk-aware calculation
//------------------------------------------------------------------------------
// The calculation receives a chunk of inputs, produces a chunk of outputs.

/*
template <int ChunkSize>
struct affine_chunked {
    template <typename T>
    __hostdev__ static void
    apply(T const* R[ChunkSize],   // ChunkSize rotation matrices
          T const* t[ChunkSize],   // ChunkSize translations
          T const* x[ChunkSize],   // ChunkSize input points
          T* y[ChunkSize]) {       // ChunkSize output points

        // Process all ChunkSize elements with interleaved loads
        T x_reg[ChunkSize][3];

        // Load all x vectors
        for (int g = 0; g < ChunkSize; ++g) {
            for (int j = 0; j < 3; ++j) {
                x_reg[g][j] = x[g][j];
            }
        }

        // Compute all outputs
        for (int g = 0; g < ChunkSize; ++g) {
            for (int i = 0; i < 3; ++i) {
                T sum = t[g][i];
                for (int j = 0; j < 3; ++j) {
                    sum += R[g][i*3+j] * x_reg[g][j];
                }
                y[g][i] = sum;
            }
        }
    }
};
*/

// Pros: Can optimize across chunk (shared loads, register blocking)
// Cons: More complex calculation, ties calculation to chunk size

//------------------------------------------------------------------------------
// Approach C: Vectorized calculation (SIMD within element)
//------------------------------------------------------------------------------
// For elements with regular structure (like 3-vectors), use SIMD.

/*
// Using CUDA's float4 for 4-wide SIMD (3 elements + padding)
__device__ void
affine_simd(float4 const& R0, float4 const& R1, float4 const& R2,
            float4 const& t, float4 const& x, float4& y) {
    y.x = t.x + R0.x*x.x + R0.y*x.y + R0.z*x.z;
    y.y = t.y + R1.x*x.x + R1.y*x.y + R1.z*x.z;
    y.z = t.z + R2.x*x.x + R2.y*x.y + R2.z*x.z;
}
*/

// This is a separate axis of optimization from chunking.
// Could combine: process ChunkSize elements, each with SIMD.

//------------------------------------------------------------------------------
// Approach D: SoA (Structure of Arrays) for the element
//------------------------------------------------------------------------------
// Instead of AoS (array of R,t,x structs), use SoA layout.
// Then the chunk becomes naturally SIMD-friendly.

/*
// ChunkSize x 3 matrices laid out as 3 x ChunkSize vectors
struct affine_soa_input {
    T R_row0[ChunkSize][3];  // Row 0 of R for each element
    T R_row1[ChunkSize][3];  // Row 1 of R for each element
    T R_row2[ChunkSize][3];  // Row 2 of R for each element
    T t[ChunkSize][3];
    T x[ChunkSize][3];
};
*/

// This is a memory layout concern that affects how gather/scatter work,
// not the calculation itself. But it shows calculation and layout interact.

//==============================================================================
// PART 19: THE REDUCTION QUESTION
//==============================================================================
//
// When does an "elementwise" operation become a reduction?
//
// In APL terms:
//   +      dyadic plus (a + b)
//   +/     reduce (sum): a + b + c + ...
//   +\     scan (prefix sum): a, a+b, a+b+c, ...
//   +⌿     reduce-first (sum along first axis)
//
// The SAME operation (+) is applied differently by the adverb.

//------------------------------------------------------------------------------
// Map: f applied independently to each element
//------------------------------------------------------------------------------
// output[n] = f(input[n])
// No interaction between elements.

template <typename F, typename In, typename Out>
void
map_pattern(int64_t N, In in, Out out, F f) {
    for (int64_t n = 0; n < N; ++n) {
        out[n] = f(in[n]);
    }
}

//------------------------------------------------------------------------------
// Reduce: f applied cumulatively to produce single result
//------------------------------------------------------------------------------
// result = f(f(f(init, in[0]), in[1]), in[2]) ...
// All elements contribute to one output.

template <typename F, typename In, typename T>
T
reduce_pattern(int64_t N, In in, T init, F f) {
    T acc = init;
    for (int64_t n = 0; n < N; ++n) {
        acc = f(acc, in[n]);
    }
    return acc;
}

//------------------------------------------------------------------------------
// Scan: f applied cumulatively, keeping all intermediates
//------------------------------------------------------------------------------
// out[0] = in[0]
// out[1] = f(in[0], in[1])
// out[2] = f(f(in[0], in[1]), in[2])
// ...

template <typename F, typename In, typename Out>
void
scan_pattern(int64_t N, In in, Out out, F f) {
    if (N == 0)
        return;
    out[0] = in[0];
    for (int64_t n = 1; n < N; ++n) {
        out[n] = f(out[n - 1], in[n]);
    }
}

//------------------------------------------------------------------------------
// Inner Product: reduce after zip-map
//------------------------------------------------------------------------------
// dot(a, b) = +/ (a * b) = reduce(+, map(*, zip(a, b)))
// This is APL's +.× (plus dot times)

template <typename T, typename Combine, typename Multiply>
T
inner_product_pattern(
    int64_t N, T const *a, T const *b, T init, Combine combine, Multiply multiply) {
    T acc = init;
    for (int64_t n = 0; n < N; ++n) {
        acc = combine(acc, multiply(a[n], b[n]));
    }
    return acc;
}

//------------------------------------------------------------------------------
// Outer Product: map-map (all pairs)
//------------------------------------------------------------------------------
// out[i,j] = f(a[i], b[j])
// This is APL's ∘.f (jot dot f)

template <typename F, typename A, typename B, typename Out>
void
outer_product_pattern(int64_t M, int64_t N, A a, B b, Out out, F f) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            out[i * N + j] = f(a[i], b[j]);
        }
    }
}

//------------------------------------------------------------------------------
// Affine as composition of patterns
//------------------------------------------------------------------------------
// y = R @ x + t
//   = (inner_product per row) + t
//   = map(i => inner_product(R[i,:], x, 0, +, *) + t[i], range(3))

// The affine transform contains a REDUCTION (the dot product per row)
// wrapped in a MAP (over the output components).
//
// This is why it's "structured elementwise" - at the outer level it's
// elementwise (each point independently), but internally there's reduction.

//------------------------------------------------------------------------------
// How our adverbs might express this
//------------------------------------------------------------------------------

/*
// Hypothetical syntax:

// Pure elementwise (no reduction)
using relu_op = Map<relu_scalar>;

// Reduction
using sum_op = Reduce<add_scalar>;

// Scan
using cumsum_op = Scan<add_scalar>;

// Inner product (reduce after zip-map)
using dot_op = Reduce<add_scalar> ∘ Map<mul_scalar> ∘ Zip;

// Matrix-vector (map of inner products)
using matvec_op = Map<dot_op, Over<Rows>>;

// Affine (map of fused dot+add)
using affine_op = ZipMap<affine_element, ...>;

// Or decomposed:
using affine_op = Map<add_scalar> ∘ Zip<matvec_op, Identity>;
*/

//==============================================================================
// PART 20: SUMMARY - THE CALCULATION DESIGN SPACE
//==============================================================================
//
// The "mapped calculation" can be viewed along several axes:
//
// 1. EXPRESSION STYLE
//    - Pure value function (returns result)
//    - Mutating function (writes to output ref)
//    - Pointer-based (most flexible)
//    - Wrapper-based (abstracts storage)
//
// 2. COMPOSITION LEVEL
//    - Monolithic (user writes full loop)
//    - Composed (user combines primitives)
//    - Expression template (lazy evaluation)
//
// 3. REDUCTION STRUCTURE
//    - Pure map (no reduction)
//    - Pure reduce (single output)
//    - Mixed (reduction inside map, like matvec)
//
// 4. CHUNKING AWARENESS
//    - Opaque (sees one element)
//    - Chunk-aware (sees GrainSize elements)
//    - Vectorized (uses SIMD intrinsics)
//
// For our framework, the recommended approach:
//
// - User writes pointer-based or wrapper-based calculation
// - Calculation is "opaque to chunking" (framework handles grain)
// - Calculation can use library primitives (dot, matvec, etc.)
// - Adverbs (Map, Reduce, ZipMap) control how calculation applies
//
// The affine transform then becomes:
//
//   1. Define element op: affine_element_op (the inner calculation)
//   2. Apply via ZipMap adverb with element shapes
//   3. Framework generates accessor, dispatch, for_each, gather/scatter
//
// Problem-specific code: just the calculation and the shape declarations.

//==============================================================================
//==============================================================================
//
// PART III: THE HARDENED DESIGN
//
// Based on MLIR-style thinking: separate algorithm from schedule.
// The inner function should be pure and abstract, knowing nothing about
// contiguity, chunking, or memory layout. This enables future backends
// (cuTile, block-cooperative, etc.) without changing user code.
//
//==============================================================================
//==============================================================================

//==============================================================================
// PART 21: ABSTRACT ELEMENT TYPES
//==============================================================================
//
// The inner function receives ABSTRACT element types, not pointers or
// contiguity-aware wrappers. These types behave like values and provide
// mathematical operations.

// Vec<N, T>: An N-dimensional vector of type T
// This is a VALUE type. The user treats it like a mathematical vector.
// How it's stored (pointer, gathered values, shared memory) is invisible.

template <int N, typename T> struct Vec {
    T data[N];

    __hostdev__ T &
    operator[](int i) {
        return data[i];
    }
    __hostdev__ T const &
    operator[](int i) const {
        return data[i];
    }

    // Vector operations
    __hostdev__ Vec
    operator+(Vec const &other) const {
        Vec result;
        DISPATCH_UNROLL for (int i = 0; i < N; ++i) result[i] = data[i] + other[i];
        return result;
    }

    __hostdev__ Vec
    operator-(Vec const &other) const {
        Vec result;
        DISPATCH_UNROLL for (int i = 0; i < N; ++i) result[i] = data[i] - other[i];
        return result;
    }

    __hostdev__ Vec
    operator*(T scalar) const {
        Vec result;
        DISPATCH_UNROLL for (int i = 0; i < N; ++i) result[i] = data[i] * scalar;
        return result;
    }

    __hostdev__ T
    dot(Vec const &other) const {
        T sum = T(0);
        DISPATCH_UNROLL for (int i = 0; i < N; ++i) sum += data[i] * other[i];
        return sum;
    }
};

// Mat<M, N, T>: An MxN matrix of type T (row-major)
// Also a VALUE type. Mathematical matrix operations.

template <int M, int N, typename T> struct Mat {
    T data[M * N];

    __hostdev__ T &
    operator()(int i, int j) {
        return data[i * N + j];
    }
    __hostdev__ T const &
    operator()(int i, int j) const {
        return data[i * N + j];
    }

    // Row access
    __hostdev__ Vec<N, T>
    row(int i) const {
        Vec<N, T> r;
        DISPATCH_UNROLL for (int j = 0; j < N; ++j) r[j] = (*this)(i, j);
        return r;
    }

    // Matrix-vector multiply
    __hostdev__ Vec<M, T>
    operator*(Vec<N, T> const &v) const {
        Vec<M, T> result;
        DISPATCH_UNROLL for (int i = 0; i < M; ++i) {
            result[i] = row(i).dot(v);
        }
        return result;
    }
};

// Convenience aliases
template <typename T> using Vec3 = Vec<3, T>;
template <typename T> using Vec4 = Vec<4, T>;
template <typename T> using Mat3 = Mat<3, 3, T>;
template <typename T> using Mat4 = Mat<4, 4, T>;

//==============================================================================
// PART 22: THE PURE INNER FUNCTION
//==============================================================================
//
// The inner function is PURE: it takes abstract value types and returns
// an abstract value type. No pointers. No contiguity. No side effects.
//
// This is what the op-writer writes. Everything else is framework.

//------------------------------------------------------------------------------
// Example 1: Affine Transform (y = R @ x + t)
//------------------------------------------------------------------------------

struct affine_transform {
    template <typename T>
    __hostdev__ static Vec3<T>
    apply(Mat3<T> const &R, Vec3<T> const &t, Vec3<T> const &x) {
        return R * x + t;
    }
};

// That's it. That's the entire op-specific code for affine transform.
// Compare to the 200+ lines in affine_xform_ternary.cu.

//------------------------------------------------------------------------------
// Example 2: Quaternion Rotation
//------------------------------------------------------------------------------

struct quaternion_rotate {
    template <typename T>
    __hostdev__ static Vec3<T>
    apply(Vec4<T> const &q, Vec3<T> const &v) {
        // q = (w, x, y, z) quaternion
        // v' = q * v * q^(-1) expanded
        T w = q[0], qx = q[1], qy = q[2], qz = q[3];

        // t = 2 * cross(q.xyz, v)
        Vec3<T> t;
        t[0] = T(2) * (qy * v[2] - qz * v[1]);
        t[1] = T(2) * (qz * v[0] - qx * v[2]);
        t[2] = T(2) * (qx * v[1] - qy * v[0]);

        // v' = v + w * t + cross(q.xyz, t)
        Vec3<T> result;
        result[0] = v[0] + w * t[0] + (qy * t[2] - qz * t[1]);
        result[1] = v[1] + w * t[1] + (qz * t[0] - qx * t[2]);
        result[2] = v[2] + w * t[2] + (qx * t[1] - qy * t[0]);

        return result;
    }
};

//------------------------------------------------------------------------------
// Example 3: SDF Sphere Query
//------------------------------------------------------------------------------

struct sdf_sphere {
    template <typename T>
    __hostdev__ static T
    apply(Vec3<T> const &center, T radius, Vec3<T> const &query) {
        Vec3<T> diff = query - center;
        T dist_sq    = diff.dot(diff);
        return sqrt(dist_sq) - radius;
    }
};

//------------------------------------------------------------------------------
// Example 4: Bilinear Interpolation (4 corners -> 1 value)
//------------------------------------------------------------------------------

struct bilinear_interp {
    template <typename T>
    __hostdev__ static T
    apply(T v00, T v01, T v10, T v11, T u, T v) {
        T a = v00 * (T(1) - u) + v01 * u;
        T b = v10 * (T(1) - u) + v11 * u;
        return a * (T(1) - v) + b * v;
    }
};

//==============================================================================
// PART 23: PATTERN DECLARATION (What the Framework Needs)
//==============================================================================
//
// The user declares the access pattern: what element shapes go in, what comes out.
// This is the "algorithm specification" that enables the framework to:
//   - Generate appropriate accessors
//   - Decide on tiling/blocking strategy
//   - Handle contiguity transparently

// Element shape descriptor
template <int... Dims> struct elem_shape {
    static constexpr int rank   = sizeof...(Dims);
    static constexpr int dims[] = {Dims...};
    static constexpr int size   = (Dims * ... * 1);
};

// Input/output pattern declaration
template <typename... InputShapes> struct inputs {};

template <typename OutputShape> struct output {};

// The complete pattern for an operation
template <typename Func, typename Inputs, typename Output> struct op_pattern;

// Example: affine transform pattern
// Input: Mat3 (3x3), Vec3 (3), Vec3 (3)
// Output: Vec3 (3)
using affine_pattern = op_pattern<affine_transform,
                                  inputs<elem_shape<3, 3>, elem_shape<3>, elem_shape<3>>,
                                  output<elem_shape<3>>>;

// Example: quaternion rotate pattern
using quat_rotate_pattern =
    op_pattern<quaternion_rotate, inputs<elem_shape<4>, elem_shape<3>>, output<elem_shape<3>>>;

// Example: SDF sphere pattern
using sdf_sphere_pattern =
    op_pattern<sdf_sphere,
               inputs<elem_shape<3>, elem_shape<>, elem_shape<3>>, // center, radius (scalar), query
               output<elem_shape<>>                                // scalar output
               >;

//==============================================================================
// PART 24: HOW THE FRAMEWORK USES THIS
//==============================================================================
//
// The framework takes the pattern and generates everything else.
// Here's a sketch of what the framework does internally.

template <typename Pattern> struct elementwise_executor;

template <typename Func, typename... InShapes, typename OutShape>
struct elementwise_executor<op_pattern<Func, inputs<InShapes...>, output<OutShape>>> {
    // The element types, derived from shapes
    // (In real code, also parameterized by scalar type T)
    // using input_element_types = std::tuple<shape_to_element_t<InShapes>...>;
    // using output_element_type = shape_to_element_t<OutShape>;

    //--------------------------------------------------------------------------
    // Current backend: element-wise for_each
    //--------------------------------------------------------------------------
    template <torch::DeviceType Dev, torch::ScalarType Stype, contiguity Contig>
    static void
    execute_elementwise(tag<Dev, Stype, Contig> t,
                        int64_t N,
                        /* accessors for inputs */
                        /* accessor for output */) {
        // Pseudocode:
        // for_each(t, N, [=](tag, int64_t n) {
        //     auto in_elements = gather_inputs(n);  // Tuple of Vec/Mat values
        //     auto out_element = std::apply(Func::apply, in_elements);
        //     scatter_output(n, out_element);
        // });
    }

    //--------------------------------------------------------------------------
    // Future backend: block-cooperative (cuTile style)
    //--------------------------------------------------------------------------
    /*
    template <int TileSize>
    static void
    execute_tiled(/* ... */) {
        // Pseudocode:
        // __shared__ input_storage[TileSize];
        // __shared__ output_storage[TileSize];
        //
        // for (tile in tiles) {
        //     cooperative_load(input_storage, tile);
        //     __syncthreads();
        //
        //     for (local_n in my_portion_of_tile) {
        //         auto in = load_from_shared(input_storage, local_n);
        //         auto out = Func::apply(in...);
        //         store_to_shared(output_storage, local_n, out);
        //     }
        //
        //     __syncthreads();
        //     cooperative_store(output_storage, tile);
        // }
    }
    */

    //--------------------------------------------------------------------------
    // Future backend: vectorized (process 4 elements with SIMD)
    //--------------------------------------------------------------------------
    /*
    static void
    execute_vectorized(/* ... */) {
        // The same Func::apply, but called with vectorized Vec/Mat types
        // Vec<3, float4> instead of Vec<3, float>
        // Framework handles packing/unpacking
    }
    */
};

//==============================================================================
// PART 25: THE COMPLETE USER EXPERIENCE
//==============================================================================
//
// Here's what the op-writer actually writes, start to finish:

// STEP 1: Define the pure function (the "soul" of the op)
struct my_affine_op {
    template <typename T>
    __hostdev__ static Vec3<T>
    apply(Mat3<T> const &R, Vec3<T> const &t, Vec3<T> const &x) {
        return R * x + t;
    }
};

// STEP 2: Declare the pattern
using my_affine_pattern = op_pattern<my_affine_op,
                                     inputs<elem_shape<3, 3>, elem_shape<3>, elem_shape<3>>,
                                     output<elem_shape<3>>>;

// STEP 3: Instantiate the operation with dispatch axes
/*
using my_affine = make_op<
    my_affine_pattern,
    torch_cpu_cuda_device_axis,
    torch_full_float_stype_axis
>;
*/

// STEP 4: Public API (could be macro-generated)
/*
torch::Tensor
affine_xform(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    return my_affine::map("affine_xform", R, T, x);
}
*/

// Total user code: ~15 lines
// Framework handles: accessors, dispatch, for_each, contiguity, validation

//==============================================================================
// PART 26: WHY THIS WORKS WITH FUTURE BACKENDS
//==============================================================================
//
// The key insight: the pure function and pattern declaration contain ALL
// the information a backend needs:
//
// 1. ELEMENT SHAPES: Tells the backend memory footprint per element
//    - Can compute shared memory requirements
//    - Can compute register pressure
//    - Can decide tile sizes
//
// 2. PURE FUNCTION: No side effects, no memory access patterns baked in
//    - Can be called with different element representations
//    - Can be called once per element OR with batched element types
//    - Can be inlined and optimized by the compiler
//
// 3. NO CONTIGUITY IN FUNCTION: Accessor layer handles this
//    - Contiguous: Vec3<T> wraps a pointer, operator[] is direct access
//    - Strided: Vec3<T> contains gathered values, already in registers
//    - Shared memory: Vec3<T> wraps a shared memory pointer
//
// The same Func::apply works in all cases because it only sees Vec3<T>.

//==============================================================================
// PART 27: SKETCH OF THE ACCESSOR BRIDGE
//==============================================================================
//
// How do we go from torch::Tensor to Vec3<T>? The accessor layer.

// Contiguous accessor: Vec3 wraps pointer
template <typename T> struct vec3_contiguous_accessor {
    T *data;
    int64_t stride; // = 3 for contiguous

    __hostdev__ Vec3<T>
    get(int64_t n) const {
        T *p = data + n * stride;
        return Vec3<T>{{p[0], p[1], p[2]}};
    }

    // OR: return a view that wraps the pointer (zero-copy for contiguous)
    // This is an optimization the framework can choose to apply
};

// Strided accessor: Vec3 contains gathered values
template <typename T> struct vec3_strided_accessor {
    T *data;
    int64_t outer_stride;
    int64_t inner_stride;

    __hostdev__ Vec3<T>
    get(int64_t n) const {
        T *p = data + n * outer_stride;
        return Vec3<T>{{p[0 * inner_stride], p[1 * inner_stride], p[2 * inner_stride]}};
    }
};

// The framework chooses which accessor to use based on contiguity.
// The inner function receives Vec3<T> either way.

//==============================================================================
// PART 28: COMPARISON - BEFORE AND AFTER
//==============================================================================
//
// BEFORE (current affine_xform_ternary.cu):
//   - Custom affine_gatherer with contiguous/strided specializations
//   - Custom affine_scatterer with contiguous/strided specializations
//   - Manual accessor creation with element_accessor<2>::template contiguous<stype>
//   - Explicit gather loops for strided case
//   - ~200 lines of code
//
// AFTER (this design):
//   - Pure function: Vec3<T> apply(Mat3<T>, Vec3<T>, Vec3<T>)
//   - Pattern declaration: inputs<elem_shape<3,3>, elem_shape<3>, elem_shape<3>>
//   - ~15 lines of code
//   - Framework generates all the accessor/gather/scatter machinery
//
// The 185 lines we eliminated were all derivable from the pattern declaration.

//==============================================================================
// SUMMARY: THE HARDENED DESIGN PRINCIPLES
//==============================================================================
//
// 1. INNER FUNCTION IS PURE
//    - Takes abstract value types (Vec, Mat)
//    - Returns abstract value type
//    - No pointers, no contiguity, no side effects
//
// 2. PATTERN DECLARATION IS COMPLETE
//    - Element shapes fully specify memory layout requirements
//    - Framework derives accessor, gather, scatter from this
//
// 3. CONTIGUITY IS INVISIBLE TO USER
//    - Accessor layer handles it before function is called
//    - Same function works for contiguous, strided, shared memory
//
// 4. CHUNKING IS INVISIBLE TO USER
//    - Framework applies grain/tile/block strategy
//    - Function is called once per element (or with batched types)
//
// 5. BACKEND IS PLUGGABLE
//    - Same pattern can execute via:
//      - for_each (current)
//      - tiled/block-cooperative (future cuTile)
//      - vectorized (future)
//    - User code unchanged
//
// This is the MLIR-style separation of algorithm from schedule,
// expressed in static C++ for a PyTorch extension context.

// A dimension requires an extent. This can be known at run-time or compile-time.
// An extent can be the same at every index, or it can vary over indices.

// when iterating over leaf node indices, what are we iterating over?
// the "elements" are int64_t x 8x8x8

} // namespace design
} // namespace dispatch
