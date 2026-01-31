#pragma once
//======================================================================================
// dispatch/element_accessor.hpp
//
// Single-header listing of:
//   - consteval-driven meta utilities (avoids static constexpr data members)
//   - extents<...>, indices<...>, linear_block, tile_block
//   - element_accessor<device, stype, contiguity, IndexRank, ElementShape>
//       * contiguity::contiguous specialization
//       * contiguity::strided specialization (accepts contiguous at runtime too)
//       * load/prepare/store for:
//           - linear_block<Vec>
//           - tile_block<extents<...>, indices<...>>
//   - helper for_each_linear_blocks<Vec>
//   - example affine_xform2 implemented using the accessor/block interface
//
// Notes:
//   * This header assumes you have an existing dispatch framework providing:
//       - for_each(tag, N, lambda)
//       - dispatch_table / torch_dispatch
//       - make_device_guard
//       - combined_contiguity
//     Minimal forward declarations are provided for integration.
//   * This header does NOT attempt to replicate your dispatcher implementation.
//
//======================================================================================

#include "dispatch/macros.h"
#include "dispatch/torch/types.h"
#include "dispatch/detail.h"
#include "dispatch/types.h"
#include "dispatch/torch/dispatch.h"

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>


namespace dispatch {

//======================================================================================
// element_accessor implementation
//======================================================================================
namespace element_accessor_detail {

//------------------------------------------------------------------------------
// consteval device traits
//------------------------------------------------------------------------------
template <torch::DeviceType D>
consteval bool
is_cpu() {
    return D == torch::kCPU;
}

template <torch::DeviceType D>
consteval bool
is_cuda_like() {
    return (D == torch::kCUDA) || (D == torch::kPrivateUse1);
}

// For CUDA-like devices prefer 32-bit indexing (PackedTensorAccessor32 style).
template <torch::DeviceType D>
using index_t = std::conditional_t<is_cuda_like<D>(), int32_t, int64_t>;

template <typename To>
inline To
checked_narrow_int(int64_t v, const char *what) {
    TORCH_CHECK(v >= static_cast<int64_t>(std::numeric_limits<To>::min()) &&
                    v <= static_cast<int64_t>(std::numeric_limits<To>::max()),
                "dispatch::element_accessor: ",
                what,
                " out of range for target index type. Value=",
                v);
    return static_cast<To>(v);
}

//------------------------------------------------------------------------------
// Compile-time extents helpers (consteval; no static constexpr objects)
//------------------------------------------------------------------------------
template <typename Shape> struct shape_rank;

template <size_t... E> struct shape_rank<extents<E...>> {
    static consteval size_t
    value() {
        return sizeof...(E);
    }
};

template <typename Shape>
consteval size_t
shape_rank_v() {
    return shape_rank<Shape>::value();
}

template <typename Shape> struct shape_numel;

template <size_t... E> struct shape_numel<extents<E...>> {
    static consteval size_t
    value() {
        return (E * ... * size_t{1});
    } // extents<> -> 1
};

template <typename Shape>
consteval size_t
shape_numel_v() {
    return shape_numel<Shape>::value();
}

template <typename Shape> struct shape_dims_i64;

template <size_t... E> struct shape_dims_i64<extents<E...>> {
    static consteval auto
    value() {
        return std::array<int64_t, sizeof...(E)>{{static_cast<int64_t>(E)...}};
    }
};

template <typename Shape> struct shape_expected_contig_strides_i64;

template <size_t... E> struct shape_expected_contig_strides_i64<extents<E...>> {
    static consteval auto
    value() {
        constexpr size_t R = sizeof...(E);
        auto dims          = shape_dims_i64<extents<E...>>::value();
        std::array<int64_t, R> s{};
        int64_t expected = 1;
        for (int i = static_cast<int>(R) - 1; i >= 0; --i) {
            s[static_cast<size_t>(i)] = expected;
            expected *= dims[static_cast<size_t>(i)];
        }
        return s;
    }
};

//------------------------------------------------------------------------------
// indices<> type (axis permutations, etc.)
//------------------------------------------------------------------------------
template <size_t... I> struct indices {};

template <typename Seq> struct index_sequence_to_indices;

template <size_t... I> struct index_sequence_to_indices<std::index_sequence<I...>> {
    using type = indices<I...>;
};

template <size_t N>
using make_indices_t = typename index_sequence_to_indices<std::make_index_sequence<N>>::type;

// split_last for extents and indices
template <typename Shape> struct split_last_extents;

template <size_t... Outer, size_t Inner> struct split_last_extents<extents<Outer..., Inner>> {
    using outer = extents<Outer...>;
    static consteval size_t
    inner() {
        return Inner;
    }
};

template <typename Ind> struct split_last_indices;

template <size_t... Outer, size_t Inner> struct split_last_indices<indices<Outer..., Inner>> {
    using outer = indices<Outer...>;
    static consteval size_t
    inner() {
        return Inner;
    }
};

template <size_t N, size_t... AX>
consteval bool
is_permutation_of_0_to_n_minus_1() {
    std::array<bool, N> seen{};
    for (bool &b: seen)
        b = false;

    bool ok   = true;
    auto mark = [&](size_t a) consteval {
        if (a >= N) {
            ok = false;
            return;
        }
        if (seen[a]) {
            ok = false;
            return;
        }
        seen[a] = true;
    };

    (mark(AX), ...);
    if (!ok)
        return false;

    for (size_t i = 0; i < N; ++i) {
        if (!seen[i])
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Packed element for shaped elements
//------------------------------------------------------------------------------
template <typename T, size_t N> struct packed_element {
    static_assert(N > 0, "packed_element N must be > 0");
    T v[N];

    __hostdev__ T &
    operator[](size_t i) {
        return v[i];
    }
    __hostdev__ const T &
    operator[](size_t i) const {
        return v[i];
    }
};

template <typename T, size_t N>
consteval bool
packed_is_tight() {
    return sizeof(packed_element<T, N>) == sizeof(T) * N;
}

// value_type selection: extents<> -> scalar; otherwise packed_element
template <typename T, typename Shape> struct value_type_for_shape;

template <typename T> struct value_type_for_shape<T, extents<>> {
    using type = T;
};

template <typename T, size_t... E>
    requires(sizeof...(E) > 0)
struct value_type_for_shape<T, extents<E...>> {
    using type = packed_element<T, (E * ... * size_t{1})>;
};

template <typename T, typename Shape>
using value_type_for_shape_t = typename value_type_for_shape<T, Shape>::type;

//------------------------------------------------------------------------------
// Block storage (register/local friendly)
//------------------------------------------------------------------------------
template <typename V>
consteval size_t
block_alignment() {
    return (alignof(V) > 16) ? alignof(V) : 16;
}

template <typename V, size_t N> struct alignas(block_alignment<V>()) block_storage {
    using value_type = V;
    V values[N];

    __hostdev__ V &
    operator[](size_t i) {
        return values[i];
    }
    __hostdev__ const V &
    operator[](size_t i) const {
        return values[i];
    }

    __hostdev__ V *
    data() {
        return values;
    }
    __hostdev__ const V *
    data() const {
        return values;
    }

    static consteval size_t
    size() {
        return N;
    }
};

//------------------------------------------------------------------------------
// Blocks
//------------------------------------------------------------------------------
template <size_t BlockSize> struct linear_block {
    using linear_block_tag = void;
    int64_t base           = 0;
    static consteval size_t
    size() {
        return BlockSize;
    }
};

template <typename Shape, typename AxisMap = make_indices_t<shape_rank_v<Shape>()>>
struct tile_block;

template <size_t... BE, size_t... AX> struct tile_block<extents<BE...>, indices<AX...>> {
    using tile_block_tag = void;
    using shape_type     = extents<BE...>;
    using axis_map       = indices<AX...>;

    static_assert(sizeof...(BE) == sizeof...(AX),
                  "tile_block: shape rank and axis_map rank must match");
    static_assert(is_permutation_of_0_to_n_minus_1<sizeof...(BE), AX...>(),
                  "tile_block: axis_map must be a permutation of [0..rank-1]");

    std::array<int64_t, sizeof...(BE)> origin{}; // origin in tensor index dims

    static consteval size_t
    rank() {
        return sizeof...(BE);
    }
    static consteval size_t
    size() {
        return (BE * ... * size_t{1});
    }
};

//------------------------------------------------------------------------------
// Strided element gather/scatter recursion across element dims (ES...)
//------------------------------------------------------------------------------
template <size_t Dim, size_t... Es> struct element_loop;

template <size_t Dim> struct element_loop<Dim> {
    template <typename T, typename IndexT, typename Vec>
    __hostdev__ C10_ALWAYS_INLINE static void
    load(const T *base, const IndexT *, Vec &out, size_t &linear) {
        out[linear++] = *base;
    }

    template <typename T, typename IndexT, typename Vec>
    __hostdev__ C10_ALWAYS_INLINE static void
    store(T *base, const IndexT *, const Vec &in, size_t &linear) {
        *base = in[linear++];
    }
};

template <size_t Dim, size_t E0, size_t... Rest> struct element_loop<Dim, E0, Rest...> {
    template <typename T, typename IndexT, typename Vec>
    __hostdev__ C10_ALWAYS_INLINE static void
    load(const T *base, const IndexT *strides, Vec &out, size_t &linear) {
        DISPATCH_UNROLL
        for (size_t i = 0; i < E0; ++i) {
            element_loop<Dim + 1, Rest...>::load(base + static_cast<ptrdiff_t>(i) *
                                                            static_cast<ptrdiff_t>(strides[Dim]),
                                                 strides,
                                                 out,
                                                 linear);
        }
    }

    template <typename T, typename IndexT, typename Vec>
    __hostdev__ C10_ALWAYS_INLINE static void
    store(T *base, const IndexT *strides, const Vec &in, size_t &linear) {
        DISPATCH_UNROLL
        for (size_t i = 0; i < E0; ++i) {
            element_loop<Dim + 1, Rest...>::store(base + static_cast<ptrdiff_t>(i) *
                                                             static_cast<ptrdiff_t>(strides[Dim]),
                                                  strides,
                                                  in,
                                                  linear);
        }
    }
};

// Safe contiguous pack/unpack for shaped elements (no aliasing UB).
template <typename T, size_t N>
__hostdev__ C10_ALWAYS_INLINE packed_element<T, N>
load_packed_contig(const T *base) {
    static_assert(packed_is_tight<T, N>(), "packed_element must be tight");
    packed_element<T, N> out{};
    DISPATCH_UNROLL
    for (size_t i = 0; i < N; ++i)
        out.v[i] = base[i];
    return out;
}

template <typename T, size_t N>
__hostdev__ C10_ALWAYS_INLINE void
store_packed_contig(T *base, const packed_element<T, N> &in) {
    static_assert(packed_is_tight<T, N>(), "packed_element must be tight");
    DISPATCH_UNROLL
    for (size_t i = 0; i < N; ++i)
        base[i] = in.v[i];
}

//------------------------------------------------------------------------------
// Tile span walker (outer loops + inner span)
//------------------------------------------------------------------------------
template <size_t InnerExtent, typename OuterShape, typename OuterAxes> struct outer_span_loop;

// recursive
template <size_t InnerExtent, size_t E0, size_t... Rest, size_t A0, size_t... ARest>
struct outer_span_loop<InnerExtent, extents<E0, Rest...>, indices<A0, ARest...>> {
    template <typename IndexT, typename F>
    __hostdev__ C10_ALWAYS_INLINE static void
    run(const IndexT *strides, IndexT offset, size_t &linear, F &&f) {
        IndexT off = offset;
        DISPATCH_UNROLL
        for (size_t i = 0; i < E0; ++i) {
            outer_span_loop<InnerExtent, extents<Rest...>, indices<ARest...>>::run(
                strides, off, linear, f);
            off += strides[A0];
        }
    }
};

// leaf
template <size_t InnerExtent> struct outer_span_loop<InnerExtent, extents<>, indices<>> {
    template <typename IndexT, typename F>
    __hostdev__ C10_ALWAYS_INLINE static void
    run(const IndexT *, IndexT offset, size_t &linear, F &&f) {
        f(offset, linear);
        linear += InnerExtent;
    }
};

} // namespace element_accessor_detail

//------------------------------------------------------------------------------
// Primary template declaration
//------------------------------------------------------------------------------
template <torch::DeviceType D,
          torch::ScalarType S,
          contiguity C,
          size_t IndexRank,
          typename ElementShape>
struct element_accessor;

//==============================================================================
// CONTIGUOUS specialization
//==============================================================================

template <torch::DeviceType D, torch::ScalarType S, size_t IndexRank, size_t... ES>
struct element_accessor<D, S, contiguity::contiguous, IndexRank, extents<ES...>> {
    using index_t     = element_accessor_detail::index_t<D>;
    using scalar_type = torch_scalar_cpp_type_t<S>;
    using shape_type  = extents<ES...>;

    static consteval size_t
    elem_rank() {
        return sizeof...(ES);
    }
    static consteval size_t
    elem_numel() {
        return (ES * ... * size_t{1});
    }
    static consteval bool
    is_scalar() {
        return elem_rank() == 0;
    }

    using value_type = element_accessor_detail::value_type_for_shape_t<scalar_type, shape_type>;

    static_assert(std::is_trivially_copyable_v<value_type>,
                  "value_type must be trivially copyable");
    static_assert(is_scalar() ||
                      element_accessor_detail::packed_is_tight<scalar_type, elem_numel()>(),
                  "packed_element must be tight");

    scalar_type *data_ = nullptr;

    std::array<index_t, IndexRank> index_sizes_{};
    std::array<index_t, IndexRank> index_strides_{}; // scalar strides for index dims

    element_accessor() = default;

    // Host-only construction/validation
    static element_accessor
    from_tensor(const torch::Tensor &t) {
        validate(t);

        element_accessor a;
        a.data_ = static_cast<scalar_type *>(t.data_ptr<scalar_type>());

        for (size_t d = 0; d < IndexRank; ++d) {
            a.index_sizes_[d] = element_accessor_detail::checked_narrow_int<index_t>(
                t.size((int64_t)d), "index size");
            a.index_strides_[d] = element_accessor_detail::checked_narrow_int<index_t>(
                t.stride((int64_t)d), "index stride");
        }

        if constexpr (element_accessor_detail::is_cuda_like<D>()) {
            for (int64_t d = 0; d < t.dim(); ++d) {
                (void)element_accessor_detail::checked_narrow_int<index_t>(t.size(d), "size");
                (void)element_accessor_detail::checked_narrow_int<index_t>(t.stride(d), "stride");
            }
        }

        return a;
    }

    static void
    validate(const torch::Tensor &t) {
        TORCH_CHECK(t.defined(), "dispatch::element_accessor(contiguous): tensor is undefined");
        TORCH_CHECK(t.layout() == torch::kStrided,
                    "dispatch::element_accessor(contiguous): expected strided layout");
        TORCH_CHECK(t.device().type() == D,
                    "dispatch::element_accessor(contiguous): device mismatch");
        TORCH_CHECK(t.scalar_type() == S, "dispatch::element_accessor(contiguous): dtype mismatch");

        constexpr int64_t want_dim = static_cast<int64_t>(IndexRank + elem_rank());
        TORCH_CHECK(t.dim() == want_dim,
                    "dispatch::element_accessor(contiguous): rank mismatch. Got dim=",
                    t.dim(),
                    " expected ",
                    want_dim);

        if constexpr (elem_rank() > 0) {
            auto dims = element_accessor_detail::shape_dims_i64<shape_type>::value();
            for (size_t e = 0; e < elem_rank(); ++e) {
                TORCH_CHECK(
                    t.size(static_cast<int64_t>(IndexRank + e)) == dims[e],
                    "dispatch::element_accessor(contiguous): element shape mismatch at trailing dim ",
                    e,
                    ". Got ",
                    t.size(static_cast<int64_t>(IndexRank + e)),
                    " expected ",
                    dims[e]);
            }
        }

        TORCH_CHECK(t.is_contiguous(),
                    "dispatch::element_accessor(contiguous): tensor must be contiguous");
    }

    // ----------------------------
    // linear_block<Vec>
    // ----------------------------
    template <typename Block>
        requires { typename Block::linear_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE auto
    load(const Block &block) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        constexpr size_t BS = Block::size();
        constexpr size_t EN = elem_numel();

        element_accessor_detail::block_storage<value_type, BS> out;

        const index_t start = static_cast<index_t>(block.base);
        const scalar_type *C10_RESTRICT p =
            data_ + static_cast<ptrdiff_t>(start) * static_cast<ptrdiff_t>(EN);

#if !defined(__CUDA_ARCH__)
        // Host fast path: memcpy bytes when possible.
        std::memcpy(
            static_cast<void *>(out.data()), static_cast<const void *>(p), sizeof(value_type) * BS);
#else
        if constexpr (is_scalar()) {
            DISPATCH_UNROLL
            for (size_t i = 0; i < BS; ++i)
                out[i] = p[i];
        } else {
            DISPATCH_UNROLL
            for (size_t i = 0; i < BS; ++i) {
                out[i] = element_accessor_detail::load_packed_contig<scalar_type, EN>(
                    p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(EN));
            }
        }
#endif
        return out;
    }

    template <typename Block>
        requires { typename Block::linear_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE void
    store(const Block &block,
          const element_accessor_detail::block_storage<value_type, Block::size()> &buf) const {
        constexpr size_t BS = Block::size();
        constexpr size_t EN = elem_numel();

        const index_t start = static_cast<index_t>(block.base);
        scalar_type *C10_RESTRICT p =
            data_ + static_cast<ptrdiff_t>(start) * static_cast<ptrdiff_t>(EN);

#if !defined(__CUDA_ARCH__)
        std::memcpy(
            static_cast<void *>(p), static_cast<const void *>(buf.data()), sizeof(value_type) * BS);
#else
        if constexpr (is_scalar()) {
            DISPATCH_UNROLL
            for (size_t i = 0; i < BS; ++i)
                p[i] = buf[i];
        } else {
            DISPATCH_UNROLL
            for (size_t i = 0; i < BS; ++i) {
                element_accessor_detail::store_packed_contig<scalar_type, EN>(
                    p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(EN), buf[i]);
            }
        }
#endif
    }

    // ----------------------------
    // tile_block<...>
    // ----------------------------
    template <typename Block>
        requires { typename Block::tile_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE auto
    load(const Block &block) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        static_assert(Block::rank() == IndexRank, "tile_block rank must match IndexRank");

        using BShape = typename Block::shape_type;
        using BAxes  = typename Block::axis_map;

        using OuterShape = typename element_accessor_detail::split_last_extents<BShape>::outer;
        using OuterAxes  = typename element_accessor_detail::split_last_indices<BAxes>::outer;

        constexpr size_t InnerExtent = element_accessor_detail::split_last_extents<BShape>::inner();
        constexpr size_t InnerAxis   = element_accessor_detail::split_last_indices<BAxes>::inner();
        constexpr size_t EN          = elem_numel();

        element_accessor_detail::block_storage<value_type, Block::size()> out;

        index_t base_offset = 0;
        DISPATCH_UNROLL
        for (size_t d = 0; d < IndexRank; ++d) {
            base_offset += static_cast<index_t>(block.origin[d]) * index_strides_[d];
        }

        const index_t inner_stride = index_strides_[InnerAxis];

        size_t linear = 0;
        element_accessor_detail::outer_span_loop<InnerExtent, OuterShape, OuterAxes>::run(
            index_strides_.data(),
            base_offset,
            linear,
            [&](index_t span_offset, size_t linear_base) __hostdev__ {
                const scalar_type *C10_RESTRICT p = data_ + static_cast<ptrdiff_t>(span_offset);

#if !defined(__CUDA_ARCH__)
                // Host fast span copy if inner walks are contiguous in element units.
                if (inner_stride == static_cast<index_t>(EN)) {
                    std::memcpy(static_cast<void *>(out.data() + linear_base),
                                static_cast<const void *>(p),
                                sizeof(value_type) * InnerExtent);
                    return;
                }
#endif

                if constexpr (is_scalar()) {
                    DISPATCH_UNROLL
                    for (size_t i = 0; i < InnerExtent; ++i) {
                        out[linear_base + i] =
                            p[static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride)];
                    }
                } else {
                    DISPATCH_UNROLL
                    for (size_t i = 0; i < InnerExtent; ++i) {
                        const scalar_type *pe =
                            p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride);
                        out[linear_base + i] =
                            element_accessor_detail::load_packed_contig<scalar_type, EN>(pe);
                    }
                }
            });

        return out;
    }

    template <typename Block>
        requires { typename Block::tile_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE void
    store(const Block &block,
          const element_accessor_detail::block_storage<value_type, Block::size()> &buf) const {
        static_assert(Block::rank() == IndexRank, "tile_block rank must match IndexRank");

        using BShape = typename Block::shape_type;
        using BAxes  = typename Block::axis_map;

        using OuterShape = typename element_accessor_detail::split_last_extents<BShape>::outer;
        using OuterAxes  = typename element_accessor_detail::split_last_indices<BAxes>::outer;

        constexpr size_t InnerExtent = element_accessor_detail::split_last_extents<BShape>::inner();
        constexpr size_t InnerAxis   = element_accessor_detail::split_last_indices<BAxes>::inner();
        constexpr size_t EN          = elem_numel();

        index_t base_offset = 0;
        DISPATCH_UNROLL
        for (size_t d = 0; d < IndexRank; ++d) {
            base_offset += static_cast<index_t>(block.origin[d]) * index_strides_[d];
        }

        const index_t inner_stride = index_strides_[InnerAxis];

        size_t linear = 0;
        element_accessor_detail::outer_span_loop<InnerExtent, OuterShape, OuterAxes>::run(
            index_strides_.data(),
            base_offset,
            linear,
            [&](index_t span_offset, size_t linear_base) __hostdev__ {
                scalar_type *C10_RESTRICT p = data_ + static_cast<ptrdiff_t>(span_offset);

#if !defined(__CUDA_ARCH__)
                if (inner_stride == static_cast<index_t>(EN)) {
                    std::memcpy(static_cast<void *>(p),
                                static_cast<const void *>(buf.data() + linear_base),
                                sizeof(value_type) * InnerExtent);
                    return;
                }
#endif

                if constexpr (is_scalar()) {
                    DISPATCH_UNROLL
                    for (size_t i = 0; i < InnerExtent; ++i) {
                        p[static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride)] =
                            buf[linear_base + i];
                    }
                } else {
                    DISPATCH_UNROLL
                    for (size_t i = 0; i < InnerExtent; ++i) {
                        scalar_type *pe =
                            p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride);
                        element_accessor_detail::store_packed_contig<scalar_type, EN>(
                            pe, buf[linear_base + i]);
                    }
                }
            });
    }

    template <typename Block>
    __hostdev__ C10_ALWAYS_INLINE auto
    prepare(const Block &) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        element_accessor_detail::block_storage<value_type, Block::size()> out;
        return out;
    }
};

//==============================================================================
// STRIDED specialization (accepts contiguous tensors too; runtime fast paths)
//==============================================================================

template <torch::DeviceType D, torch::ScalarType S, size_t IndexRank, size_t... ES>
struct element_accessor<D, S, contiguity::strided, IndexRank, extents<ES...>> {
    using index_t     = element_accessor_detail::index_t<D>;
    using scalar_type = torch_scalar_cpp_type_t<S>;
    using shape_type  = extents<ES...>;

    static consteval size_t
    elem_rank() {
        return sizeof...(ES);
    }
    static consteval size_t
    elem_numel() {
        return (ES * ... * size_t{1});
    }
    static consteval bool
    is_scalar() {
        return elem_rank() == 0;
    }

    using value_type = element_accessor_detail::value_type_for_shape_t<scalar_type, shape_type>;

    static_assert(std::is_trivially_copyable_v<value_type>,
                  "value_type must be trivially copyable");
    static_assert(is_scalar() ||
                      element_accessor_detail::packed_is_tight<scalar_type, elem_numel()>(),
                  "packed_element must be tight");

    scalar_type *data_ = nullptr;

    std::array<index_t, IndexRank> index_sizes_{};
    std::array<index_t, IndexRank> index_strides_{};

    std::array<index_t, sizeof...(ES)> elem_strides_{}; // scalar strides for element dims

    bool runtime_is_contig_ = false;
    bool elem_is_contig_    = false;

    element_accessor() = default;

    static element_accessor
    from_tensor(const torch::Tensor &t) {
        validate(t);

        element_accessor a;
        a.data_              = static_cast<scalar_type *>(t.data_ptr<scalar_type>());
        a.runtime_is_contig_ = t.is_contiguous();

        for (size_t d = 0; d < IndexRank; ++d) {
            a.index_sizes_[d] = element_accessor_detail::checked_narrow_int<index_t>(
                t.size((int64_t)d), "index size");
            a.index_strides_[d] = element_accessor_detail::checked_narrow_int<index_t>(
                t.stride((int64_t)d), "index stride");
        }

        if constexpr (sizeof...(ES) > 0) {
            for (size_t e = 0; e < sizeof...(ES); ++e) {
                a.elem_strides_[e] = element_accessor_detail::checked_narrow_int<index_t>(
                    t.stride(static_cast<int64_t>(IndexRank + e)), "element stride");
            }
        }

        a.elem_is_contig_ = a.runtime_is_contig_ ? true : compute_elem_contig_(t);

        if constexpr (element_accessor_detail::is_cuda_like<D>()) {
            for (int64_t d = 0; d < t.dim(); ++d) {
                (void)element_accessor_detail::checked_narrow_int<index_t>(t.size(d), "size");
                (void)element_accessor_detail::checked_narrow_int<index_t>(t.stride(d), "stride");
            }
        }

        return a;
    }

    static void
    validate(const torch::Tensor &t) {
        TORCH_CHECK(t.defined(), "dispatch::element_accessor(strided): tensor is undefined");
        TORCH_CHECK(t.layout() == torch::kStrided,
                    "dispatch::element_accessor(strided): expected strided layout");
        TORCH_CHECK(t.device().type() == D, "dispatch::element_accessor(strided): device mismatch");
        TORCH_CHECK(t.scalar_type() == S, "dispatch::element_accessor(strided): dtype mismatch");

        constexpr int64_t want_dim = static_cast<int64_t>(IndexRank + elem_rank());
        TORCH_CHECK(t.dim() == want_dim,
                    "dispatch::element_accessor(strided): rank mismatch. Got dim=",
                    t.dim(),
                    " expected ",
                    want_dim);

        if constexpr (elem_rank() > 0) {
            auto dims = element_accessor_detail::shape_dims_i64<shape_type>::value();
            for (size_t e = 0; e < elem_rank(); ++e) {
                TORCH_CHECK(
                    t.size(static_cast<int64_t>(IndexRank + e)) == dims[e],
                    "dispatch::element_accessor(strided): element shape mismatch at trailing dim ",
                    e,
                    ". Got ",
                    t.size(static_cast<int64_t>(IndexRank + e)),
                    " expected ",
                    dims[e]);
            }
        }
    }

  private:
    static bool
    compute_elem_contig_(const torch::Tensor &t) {
        if constexpr (elem_rank() == 0) {
            return true;
        } else {
            auto expected =
                element_accessor_detail::shape_expected_contig_strides_i64<shape_type>::value();
            for (size_t e = 0; e < elem_rank(); ++e) {
                const int64_t got = t.stride(static_cast<int64_t>(IndexRank + e));
                if (got != expected[e])
                    return false;
            }
            return true;
        }
    }

    __hostdev__ C10_ALWAYS_INLINE value_type
    load_one_(const scalar_type *base_elem) const {
        constexpr size_t EN = elem_numel();

        if constexpr (is_scalar()) {
            return *base_elem;
        } else {
            if (elem_is_contig_) {
                return element_accessor_detail::load_packed_contig<scalar_type, EN>(base_elem);
            }
            value_type out{};
            size_t linear = 0;
            element_accessor_detail::element_loop<0, ES...>::load(
                base_elem, elem_strides_.data(), out, linear);
            return out;
        }
    }

    __hostdev__ C10_ALWAYS_INLINE void
    store_one_(scalar_type *base_elem, const value_type &v) const {
        constexpr size_t EN = elem_numel();

        if constexpr (is_scalar()) {
            *base_elem = v;
        } else {
            if (elem_is_contig_) {
                element_accessor_detail::store_packed_contig<scalar_type, EN>(base_elem, v);
                return;
            }
            size_t linear = 0;
            element_accessor_detail::element_loop<0, ES...>::store(
                base_elem, elem_strides_.data(), v, linear);
        }
    }

  public:
    // ----------------------------
    // linear_block<Vec>
    // ----------------------------
    template <typename Block>
        requires { typename Block::linear_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE auto
    load(const Block &block) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        constexpr size_t BS = Block::size();

        // runtime contig fast path: treat like contiguous in element units
        if (runtime_is_contig_) {
            constexpr size_t EN = elem_numel();
            element_accessor_detail::block_storage<value_type, BS> out;

            const index_t start = static_cast<index_t>(block.base);
            const scalar_type *C10_RESTRICT p =
                data_ + static_cast<ptrdiff_t>(start) * static_cast<ptrdiff_t>(EN);

#if !defined(__CUDA_ARCH__)
            std::memcpy(static_cast<void *>(out.data()),
                        static_cast<const void *>(p),
                        sizeof(value_type) * BS);
#else
            if constexpr (is_scalar()) {
                DISPATCH_UNROLL
                for (size_t i = 0; i < BS; ++i)
                    out[i] = p[i];
            } else {
                DISPATCH_UNROLL
                for (size_t i = 0; i < BS; ++i) {
                    out[i] = element_accessor_detail::load_packed_contig<scalar_type, EN>(
                        p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(EN));
                }
            }
#endif
            return out;
        }

        element_accessor_detail::block_storage<value_type, BS> out;

        // Initial multi-index from linear index (div/mod only once per block)
        std::array<index_t, IndexRank> idx{};
        index_t offset = 0;

        if constexpr (IndexRank > 0) {
            index_t tmp = static_cast<index_t>(block.base);
            DISPATCH_UNROLL
            for (int d = static_cast<int>(IndexRank) - 1; d >= 0; --d) {
                const index_t sz = index_sizes_[static_cast<size_t>(d)];
                const index_t id = (sz == 0) ? 0 : (tmp % sz);
                tmp              = (sz == 0) ? 0 : (tmp / sz);

                idx[static_cast<size_t>(d)] = id;
                offset += id * index_strides_[static_cast<size_t>(d)];
            }
        }

        DISPATCH_UNROLL
        for (size_t i = 0; i < BS; ++i) {
            out[i] = load_one_(data_ + static_cast<ptrdiff_t>(offset));

            if constexpr (IndexRank > 0) {
                // carry increment (row-major)
                for (int d = static_cast<int>(IndexRank) - 1; d >= 0; --d) {
                    const size_t ud = static_cast<size_t>(d);
                    idx[ud] += 1;
                    offset += index_strides_[ud];
                    if (idx[ud] < index_sizes_[ud])
                        break;

                    idx[ud] = 0;
                    offset -= index_sizes_[ud] * index_strides_[ud];
                }
            }
        }

        return out;
    }

    template <typename Block>
        requires { typename Block::linear_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE void
    store(const Block &block,
          const element_accessor_detail::block_storage<value_type, Block::size()> &buf) const {
        constexpr size_t BS = Block::size();

        if (runtime_is_contig_) {
            constexpr size_t EN = elem_numel();

            const index_t start = static_cast<index_t>(block.base);
            scalar_type *C10_RESTRICT p =
                data_ + static_cast<ptrdiff_t>(start) * static_cast<ptrdiff_t>(EN);

#if !defined(__CUDA_ARCH__)
            std::memcpy(static_cast<void *>(p),
                        static_cast<const void *>(buf.data()),
                        sizeof(value_type) * BS);
#else
            if constexpr (is_scalar()) {
                DISPATCH_UNROLL
                for (size_t i = 0; i < BS; ++i)
                    p[i] = buf[i];
            } else {
                DISPATCH_UNROLL
                for (size_t i = 0; i < BS; ++i) {
                    element_accessor_detail::store_packed_contig<scalar_type, EN>(
                        p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(EN), buf[i]);
                }
            }
#endif
            return;
        }

        std::array<index_t, IndexRank> idx{};
        index_t offset = 0;

        if constexpr (IndexRank > 0) {
            index_t tmp = static_cast<index_t>(block.base);
            DISPATCH_UNROLL
            for (int d = static_cast<int>(IndexRank) - 1; d >= 0; --d) {
                const index_t sz = index_sizes_[static_cast<size_t>(d)];
                const index_t id = (sz == 0) ? 0 : (tmp % sz);
                tmp              = (sz == 0) ? 0 : (tmp / sz);

                idx[static_cast<size_t>(d)] = id;
                offset += id * index_strides_[static_cast<size_t>(d)];
            }
        }

        DISPATCH_UNROLL
        for (size_t i = 0; i < BS; ++i) {
            store_one_(data_ + static_cast<ptrdiff_t>(offset), buf[i]);

            if constexpr (IndexRank > 0) {
                for (int d = static_cast<int>(IndexRank) - 1; d >= 0; --d) {
                    const size_t ud = static_cast<size_t>(d);
                    idx[ud] += 1;
                    offset += index_strides_[ud];
                    if (idx[ud] < index_sizes_[ud])
                        break;

                    idx[ud] = 0;
                    offset -= index_sizes_[ud] * index_strides_[ud];
                }
            }
        }
    }

    // ----------------------------
    // tile_block<...>
    // ----------------------------
    template <typename Block>
        requires { typename Block::tile_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE auto
    load(const Block &block) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        static_assert(Block::rank() == IndexRank, "tile_block rank must match IndexRank");

        using BShape = typename Block::shape_type;
        using BAxes  = typename Block::axis_map;

        using OuterShape = typename element_accessor_detail::split_last_extents<BShape>::outer;
        using OuterAxes  = typename element_accessor_detail::split_last_indices<BAxes>::outer;

        constexpr size_t InnerExtent = element_accessor_detail::split_last_extents<BShape>::inner();
        constexpr size_t InnerAxis   = element_accessor_detail::split_last_indices<BAxes>::inner();
        constexpr size_t EN          = elem_numel();

        element_accessor_detail::block_storage<value_type, Block::size()> out;

        index_t base_offset = 0;
        DISPATCH_UNROLL
        for (size_t d = 0; d < IndexRank; ++d) {
            base_offset += static_cast<index_t>(block.origin[d]) * index_strides_[d];
        }

        const index_t inner_stride = index_strides_[InnerAxis];

        size_t linear = 0;
        element_accessor_detail::outer_span_loop<InnerExtent, OuterShape, OuterAxes>::run(
            index_strides_.data(),
            base_offset,
            linear,
            [&](index_t span_offset, size_t linear_base) __hostdev__ {
                const scalar_type *C10_RESTRICT p = data_ + static_cast<ptrdiff_t>(span_offset);

#if !defined(__CUDA_ARCH__)
                // If element dims are contiguous (or scalar) AND inner_stride steps exactly one
                // element, memcpy inner span as bytes.
                if ((is_scalar() || elem_is_contig_) && inner_stride == static_cast<index_t>(EN)) {
                    std::memcpy(static_cast<void *>(out.data() + linear_base),
                                static_cast<const void *>(p),
                                sizeof(value_type) * InnerExtent);
                    return;
                }
#endif
                DISPATCH_UNROLL
                for (size_t i = 0; i < InnerExtent; ++i) {
                    const scalar_type *pe =
                        p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride);
                    out[linear_base + i] = load_one_(pe);
                }
            });

        return out;
    }

    template <typename Block>
        requires { typename Block::tile_block_tag; }
    __hostdev__ C10_ALWAYS_INLINE void
    store(const Block &block,
          const element_accessor_detail::block_storage<value_type, Block::size()> &buf) const {
        static_assert(Block::rank() == IndexRank, "tile_block rank must match IndexRank");

        using BShape = typename Block::shape_type;
        using BAxes  = typename Block::axis_map;

        using OuterShape = typename element_accessor_detail::split_last_extents<BShape>::outer;
        using OuterAxes  = typename element_accessor_detail::split_last_indices<BAxes>::outer;

        constexpr size_t InnerExtent = element_accessor_detail::split_last_extents<BShape>::inner();
        constexpr size_t InnerAxis   = element_accessor_detail::split_last_indices<BAxes>::inner();
        constexpr size_t EN          = elem_numel();

        index_t base_offset = 0;
        DISPATCH_UNROLL
        for (size_t d = 0; d < IndexRank; ++d) {
            base_offset += static_cast<index_t>(block.origin[d]) * index_strides_[d];
        }

        const index_t inner_stride = index_strides_[InnerAxis];

        size_t linear = 0;
        element_accessor_detail::outer_span_loop<InnerExtent, OuterShape, OuterAxes>::run(
            index_strides_.data(),
            base_offset,
            linear,
            [&](index_t span_offset, size_t linear_base) __hostdev__ {
                scalar_type *C10_RESTRICT p = data_ + static_cast<ptrdiff_t>(span_offset);

#if !defined(__CUDA_ARCH__)
                if ((is_scalar() || elem_is_contig_) && inner_stride == static_cast<index_t>(EN)) {
                    std::memcpy(static_cast<void *>(p),
                                static_cast<const void *>(buf.data() + linear_base),
                                sizeof(value_type) * InnerExtent);
                    return;
                }
#endif
                DISPATCH_UNROLL
                for (size_t i = 0; i < InnerExtent; ++i) {
                    scalar_type *pe =
                        p + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(inner_stride);
                    store_one_(pe, buf[linear_base + i]);
                }
            });
    }

    template <typename Block>
    __hostdev__ C10_ALWAYS_INLINE auto
    prepare(const Block &) const
        -> element_accessor_detail::block_storage<value_type, Block::size()> {
        element_accessor_detail::block_storage<value_type, Block::size()> out;
        return out;
    }
};

//======================================================================================
// Scheduling helper: iterate over linear blocks of Vec, with scalar tail fallback.
//======================================================================================
template <size_t Vec, typename Tag, typename F>
void
for_each_linear_blocks(Tag t, int64_t N, F f) {
    int64_t blocks = N / static_cast<int64_t>(Vec);
    int64_t tail   = N - blocks * static_cast<int64_t>(Vec);

    for_each(t, blocks, [=] __hostdev__(Tag tt, int64_t bi) mutable {
        element_accessor_detail::linear_block<Vec> blk;
        blk.base = bi * static_cast<int64_t>(Vec);
        f(tt, blk);
    });

    // Tail handled as Vec==1 blocks to avoid masks inside accessor store().
    if (tail) {
        for_each(t, tail, [=] __hostdev__(Tag tt, int64_t r) mutable {
            element_accessor_detail::linear_block<1> blk;
            blk.base = blocks * static_cast<int64_t>(Vec) + r;
            f(tt, blk);
        });
    }
}

//======================================================================================
// Example: affine_xform implemented via element_accessor + linear blocks
//
// y = R @ x + t
// R: (N,3,3)
// T: (N,3)
// x: (N,3)
// out: (N,3)
//
// Broadcasting is handled outside the accessor via expand() (stride-0 views).
//======================================================================================

struct affine_xform_block_op {
    template <typename TagT, typename RBlock, typename VBlock, typename YBlock>
    __hostdev__ static void
    apply(TagT, const RBlock &Rb, const VBlock &tb, const VBlock &xb, YBlock &yb) {
        constexpr size_t BS = RBlock::size();

        DISPATCH_UNROLL
        for (size_t i = 0; i < BS; ++i) {
            DISPATCH_UNROLL
            for (int r = 0; r < 3; ++r) {
                auto sum = tb[i][r];
                DISPATCH_UNROLL
                for (int c = 0; c < 3; ++c) {
                    sum += Rb[i][r * 3 + c] * xb[i][c];
                }
                yb[i][r] = sum;
            }
        }
    }
};

template <torch::DeviceType dev, torch::ScalarType stype>
consteval size_t
affine_vec_width() {
    if constexpr (dev == torch::kCPU) {
        if constexpr (stype == torch::kFloat32)
            return 4;
        if constexpr (stype == torch::kFloat64)
            return 2;
        return 1;
    } else {
        // CUDA / PrivateUse1: structured element uses many scalars; keep Vec small to avoid spills.
        return 1;
    }
}

struct affine_xform2 {
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig> t,
       torch::Tensor R_tensor,
       torch::Tensor T_tensor,
       torch::Tensor x_tensor,
       torch::Tensor out_tensor) {
        auto guard = make_device_guard(t, x_tensor);

        using mat33 = extents<3, 3>;
        using vec3  = extents<3>;

        using R_acc_t = element_accessor<dev, stype, contig, 1, mat33>;
        using V_acc_t = element_accessor<dev, stype, contig, 1, vec3>;

        auto R_acc = R_acc_t::from_tensor(R_tensor);
        auto t_acc = V_acc_t::from_tensor(T_tensor);
        auto x_acc = V_acc_t::from_tensor(x_tensor);
        auto y_acc = V_acc_t::from_tensor(out_tensor);

        int64_t const N      = x_tensor.size(0);
        constexpr size_t Vec = affine_vec_width<dev, stype>();

        auto body = [=] __hostdev__(tag<dev, stype, contig> tt, auto blk) mutable {
            auto Rb = R_acc.load(blk);
            auto tb = t_acc.load(blk);
            auto xb = x_acc.load(blk);
            auto yb = y_acc.prepare(blk);

            affine_xform_block_op::apply(tt, Rb, tb, xb, yb);

            y_acc.store(blk, yb);
        };

        for_each_linear_blocks<Vec>(t, N, body);
    }

    using space = axes<axis<torch::kCPU, torch::kCUDA>, // torch_cpu_cuda_device_axis
                       axis<torch::kBFloat16,
                            torch::kFloat16,
                            torch::kFloat32,
                            torch::kFloat64>, // torch_full_float_stype_axis
                       axis<contiguity::strided, contiguity::contiguous>>; // full_contiguity_axis

    using dispatcher =
        dispatch_table<space, void(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>;
};

// Public API wrapper using your dispatcher.
// (Assumes torch_dispatch, dispatch_table, for_each, etc. are provided by your framework.)
inline torch::Tensor
example_affine_xform2(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    static affine_xform2::dispatcher const table{
        affine_xform2::dispatcher::from_op<affine_xform2>(), affine_xform2::space{}};

    TORCH_CHECK_VALUE(
        R.dim() == 3, "example_affine_xform2: R must be 3D (N,3,3), got ", R.dim(), "D");
    TORCH_CHECK_VALUE(
        T.dim() == 2, "example_affine_xform2: T must be 2D (N,3), got ", T.dim(), "D");
    TORCH_CHECK_VALUE(
        x.dim() == 2, "example_affine_xform2: x must be 2D (N,3), got ", x.dim(), "D");

    TORCH_CHECK_VALUE(R.size(1) == 3 && R.size(2) == 3,
                      "example_affine_xform2: R element shape must be (3,3), got (",
                      R.size(1),
                      ", ",
                      R.size(2),
                      ")");
    TORCH_CHECK_VALUE(T.size(1) == 3,
                      "example_affine_xform2: T element shape must be (3,), got (",
                      T.size(1),
                      ",)");
    TORCH_CHECK_VALUE(x.size(1) == 3,
                      "example_affine_xform2: x element shape must be (3,), got (",
                      x.size(1),
                      ",)");

    TORCH_CHECK_VALUE(R.scalar_type() == T.scalar_type() && T.scalar_type() == x.scalar_type(),
                      "example_affine_xform2: all tensors must have same dtype, got R=",
                      R.scalar_type(),
                      ", T=",
                      T.scalar_type(),
                      ", x=",
                      x.scalar_type());

    TORCH_CHECK_VALUE(R.device() == T.device() && T.device() == x.device(),
                      "example_affine_xform2: all tensors must be on same device");

    int64_t const N = x.size(0);
    if (N == 0) {
        return torch::empty({0, 3}, x.options());
    }

    // Broadcast via expand() (stride-0, no copy). This keeps accessor simpler (no broadcast rules
    // inside).
    if (R.size(0) == 1 && N > 1)
        R = R.expand({N, 3, 3});
    if (T.size(0) == 1 && N > 1)
        T = T.expand({N, 3});

    auto out = torch::empty({N, 3}, x.options());

    auto const dev    = x.device().type();
    auto const stype  = x.scalar_type();
    auto const contig = combined_contiguity(R, T, x, out);

    torch_dispatch(
        "example_affine_xform2", table, std::make_tuple(dev, stype, contig), R, T, x, out);

    return out;
}

} // namespace dispatch
