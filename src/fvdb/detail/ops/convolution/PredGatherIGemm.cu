// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PredGatherIGemm.cu -- Predicated-gather implicit-GEMM sparse convolution.
//
// SM80 CUTLASS/CuTe IGEMM that processes one output leaf node per CTA,
// using predicated cp.async gather loads for the activation (B) matrix.
//
// Derived from the Sifakis v2 reference implementation; stripped of all
// internal validation, reference kernels, and test harness.  Geometry
// (kernel size and stride) is templatized.

// NOTE: <torch/types.h> MUST precede CuTe / CUTLASS headers to avoid
// CCCL include-order issues between toolkit versions.  See sifakis_ref_2.cu
// for the full explanation.
#include "dispatch/detail/core_types.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/with_value.h"

#include <fvdb/detail/ops/convolution/PredGatherIGemm.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>

// ============================================================================
// CUTLASS namespace extensions (must live in cutlass::gemm)
// ============================================================================

namespace cutlass::gemm {
using namespace cute;

template <int Stages_> struct MainloopSm80CpAsyncPredGatherB {
    constexpr static int Stages = Stages_;
    using ArchTag               = arch::Sm80;
    using Schedule              = KernelMultistage;
    using ClusterShape          = Shape<_1, _1, _1>;
};

} // namespace cutlass::gemm

// ============================================================================
// CollectiveMma specialization (must live in cutlass::gemm::collective)
// ============================================================================

namespace cutlass::gemm::collective {
using namespace cute;

template <int Stages,
          class TileShape_,
          class ElementA_,
          class StrideA_,
          class ElementB_,
          class StrideB_,
          class TiledMma_,
          class GmemTiledCopyA_,
          class SmemLayoutAtomA_,
          class SmemCopyAtomA_,
          class TransformA_,
          class GmemTiledCopyB_,
          class SmemLayoutAtomB_,
          class SmemCopyAtomB_,
          class TransformB_>
struct CollectiveMma<MainloopSm80CpAsyncPredGatherB<Stages>,
                     TileShape_,
                     ElementA_,
                     StrideA_,
                     ElementB_,
                     StrideB_,
                     TiledMma_,
                     GmemTiledCopyA_,
                     SmemLayoutAtomA_,
                     SmemCopyAtomA_,
                     TransformA_,
                     GmemTiledCopyB_,
                     SmemLayoutAtomB_,
                     SmemCopyAtomB_,
                     TransformB_> {
    using DispatchPolicy     = MainloopSm80CpAsyncUnpredicated<Stages>;
    using TileShape          = TileShape_;
    using ElementA           = ElementA_;
    using StrideA            = StrideA_;
    using ElementB           = ElementB_;
    using StrideB            = StrideB_;
    using TiledMma           = TiledMma_;
    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyA     = GmemTiledCopyA_;
    using GmemTiledCopyB     = GmemTiledCopyB_;
    using SmemLayoutAtomA    = SmemLayoutAtomA_;
    using SmemLayoutAtomB    = SmemLayoutAtomB_;
    using SmemCopyAtomA      = SmemCopyAtomA_;
    using SmemCopyAtomB      = SmemCopyAtomB_;
    using TransformA         = TransformA_;
    using TransformB         = TransformB_;
    using ArchTag            = typename DispatchPolicy::ArchTag;
    using CtaShape_MNK       = TileShape;

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

    static_assert(DispatchPolicy::Stages >= 2,
                  "CpAsync mainloop must have at least 2 stages in the pipeline.");

    struct SharedStorage {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    };

    struct Arguments {
        ElementA const *ptr_A;
        StrideA dA;
        ElementB const *ptr_B;
        StrideB dB;
    };

    using Params = Arguments;

    CollectiveMma() = default;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
        return args;
    }

    template <class FrgTensorD,
              class TensorA,
              class TensorB,
              class TensorP,
              class FrgTensorC,
              class KTileIterator,
              class ResidueMNK>
    CUTLASS_DEVICE void
    operator()(FrgTensorD &accum,
               TensorA gA,
               TensorB gB,
               TensorP sP,
               FrgTensorC const &src_accum,
               KTileIterator k_tile_iter,
               int k_tile_count,
               ResidueMNK residue_mnk,
               int thread_idx,
               char *smem_buf) {
        using namespace cute;

        static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
        static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
        static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
        static_assert(is_smem<TensorP>::value, "Gather predicate tensor must be smem resident.");
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3,
                      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");
        static_assert(cute::rank(SmemLayoutB{}) == 3,
                      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");

        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
        Tensor sA              = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
        Tensor sB              = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

        CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));
        CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));
        CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));
        CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sP));
        CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));
        CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sP));
        CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));

        GmemTiledCopyA gmem_tiled_copy_A;
        GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
        Tensor tBsP = gmem_thr_copy_B.partition_S(sP);

        (void)residue_mnk;

        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages - 1; ++k_pipe) {
            copy(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe));
            copy_if(gmem_tiled_copy_B,
                    tBsP(_, _, _, *k_tile_iter),
                    tBgB(_, _, _, *k_tile_iter),
                    tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) {
                ++k_tile_iter;
            }
        }

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA  = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB  = thr_mma.partition_fragment_B(sB(_, _, 0));

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_A) == size(tiled_mma));
        CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_B) == size(tiled_mma));

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
        Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
        Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

        int smem_pipe_read  = 0;
        int smem_pipe_write = DispatchPolicy::Stages - 1;

        Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
        Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

        auto K_BLOCK_MAX = size<2>(tCrA);

        if (K_BLOCK_MAX > 1) {
            cp_async_wait<DispatchPolicy::Stages - 2>();
            __syncthreads();

            copy(smem_tiled_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_copy_view(_, _, Int<0>{}));
            copy(smem_tiled_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_copy_view(_, _, Int<0>{}));
        }

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(DispatchPolicy::Stages - 1)) {
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    tCsB_p = tCsB(_, _, _, smem_pipe_read);
                    cp_async_wait<DispatchPolicy::Stages - 2>();
                    __syncthreads();
                }

                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                copy(smem_tiled_copy_A,
                     tCsA_p(_, _, k_block_next),
                     tCrA_copy_view(_, _, k_block_next));
                copy(smem_tiled_copy_B,
                     tCsB_p(_, _, k_block_next),
                     tCrB_copy_view(_, _, k_block_next));

                if (k_block == 0) {
                    copy(gmem_tiled_copy_A,
                         tAgA(_, _, _, *k_tile_iter),
                         tAsA(_, _, _, smem_pipe_write));
                    copy_if(gmem_tiled_copy_B,
                            tBsP(_, _, _, *k_tile_iter),
                            tBgB(_, _, _, *k_tile_iter),
                            tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();

                    --k_tile_count;
                    if (k_tile_count > 0) {
                        ++k_tile_iter;
                    }

                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read =
                        (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
                }

                cute::transform(tCrA(_, _, k_block), TransformA{});
                cute::transform(tCrB(_, _, k_block), TransformB{});
                cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), src_accum);
            });
        }

        cp_async_wait<0>();
        __syncthreads();
    }
};

} // namespace cutlass::gemm::collective

// ============================================================================
// Gather/scatter composed-layout utilities
// ============================================================================

namespace pred_gather_igemm {
namespace gather_util {

using namespace cute;

template <class Index, int Offset = 0> struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr Index
    operator()(I i) const {
        return indices_[i] + Index(Offset);
    }

    CUTE_HOST_DEVICE friend void
    print(IndexedGather const &) {
        cute::print("Indexed");
    }

    Index const *indices_;
};

template <class Func, class Stride> struct CustomStride {
    CUTE_HOST_DEVICE constexpr CustomStride(Func const &func, Stride const &stride)
        : func_(func), stride_(stride) {}

    template <class I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(I i, CustomStride const &s) {
        return s.func_(i) * s.stride_;
    }

    template <class I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(CustomStride const &s, I i) {
        return s.func_(i) * s.stride_;
    }

    CUTE_HOST_DEVICE friend void
    print(CustomStride const &s) {
        cute::print("Custom{");
        print(s.func_);
        cute::print(",");
        print(s.stride_);
        cute::print("}");
    }

    template <class Div>
    CUTE_HOST_DEVICE constexpr friend auto
    safe_div(CustomStride const &s, Div const &div) {
        return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_,
                                                                      safe_div(s.stride_, div));
    }

    template <class Shape>
    CUTE_HOST_DEVICE constexpr friend auto
    make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }

    Func func_;
    Stride stride_;
};

} // namespace gather_util
} // namespace pred_gather_igemm

// ============================================================================
// CuTe upcast overloads for composed gather layouts
// ============================================================================

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
upcast(Shape const &shape, Stride const &stride) {
    if constexpr (is_tuple<Shape>::value) {
        return transform_layout(
            shape, stride, [](auto const &s, auto const &d) { return upcast<N, I>(s, d); });
    } else if constexpr (is_scaled_basis<Stride>::value) {
        if constexpr (Stride::mode() == I) {
            return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
        } else {
            return make_layout(shape, stride);
        }
    } else {
        return upcast<N>(shape, stride);
    }
    CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>, Offset, Layout<Shape, Stride>> const &layout) {
    auto idx =
        find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    auto outer      = upcast<N>(layout.layout_a());
    auto offset =
        as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));
    auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());
    return composition(outer, offset, inner);
}

} // namespace cute

// ============================================================================
// Geometry, layouts, and kernel operator
// ============================================================================

namespace pred_gather_igemm {

using namespace cute;

// ---------------------------------------------------------------------------
// Templatized convolution geometry
// ---------------------------------------------------------------------------

template <int T_, int R_, int S_, int STx_, int STy_, int STz_, int KoutDenom_> struct Geometry {
    static constexpr int T   = T_;
    static constexpr int R   = R_;
    static constexpr int S   = S_;
    static constexpr int STx = STx_;
    static constexpr int STy = STy_;
    static constexpr int STz = STz_;

    static_assert(KoutDenom_ == 32 || KoutDenom_ == 128, "KoutDenom must be 32 or 128");

    static constexpr int Z = 4;
    static constexpr int P = 2;
    static constexpr int Q = 2;

    int c, k;
    int dx, dy, dz;

    __hostdev__ int
    C() const {
        return c;
    }
    __hostdev__ int
    K() const {
        return k;
    }
    __hostdev__ int
    Dx() const {
        return dx;
    }
    __hostdev__ int
    Dy() const {
        return dy;
    }
    __hostdev__ int
    Dz() const {
        return dz;
    }

    static constexpr int TC = 32;
    static constexpr int TK = KoutDenom_;

    static constexpr int ZZ = 1;
    static constexpr int PP = 2;
    static constexpr int QQ = 2;

    static constexpr int Bx = 8 / Z;
    static constexpr int By = 8 / P;
    static constexpr int Bz = 8 / Q;

    static constexpr int Cx = 8 / (ZZ * Z);
    static constexpr int Cy = 8 / (PP * P);
    static constexpr int Cz = 8 / (QQ * Q);

    static constexpr int CHx = (ZZ * Z - 1) * STx + T;
    static constexpr int CHy = (PP * P - 1) * STy + R;
    static constexpr int CHz = (QQ * Q - 1) * STz + S;

    static constexpr int CVx = ZZ * Z;
    static constexpr int CVy = PP * P;
    static constexpr int CVz = QQ * Q;

    static constexpr int
    VoxelsPerLeafnodeNoHalo() {
        return 512;
    }
    static constexpr int
    VoxelsPerClusterNoHalo() {
        return CVx * CVy * CVz;
    }
    static constexpr int
    VoxelsPerClusterWithHalo() {
        return CHx * CHy * CHz;
    }
};

// ---------------------------------------------------------------------------
// CuTe layout construction for the IGEMM
// ---------------------------------------------------------------------------

template <class SettingsT> struct Layouts {
    static constexpr auto Z   = Int<SettingsT::Z>{};
    static constexpr auto P   = Int<SettingsT::P>{};
    static constexpr auto Q   = Int<SettingsT::Q>{};
    static constexpr auto CVy = Int<SettingsT::CVy>{};
    static constexpr auto CVz = Int<SettingsT::CVz>{};

    SettingsT geometry{};

    __hostdev__
    Layouts(SettingsT g = {})
        : geometry(g) {}

    __hostdev__ auto
    clusterActivationComposedGatherLayout(const uint64_t *gather_idx_buf) {
        auto EG                        = E<0>{};
        auto EC                        = E<1>{};
        auto C                         = geometry.C();
        auto T                         = Int<SettingsT::T>{};
        auto R                         = Int<SettingsT::R>{};
        auto S                         = Int<SettingsT::S>{};
        auto ZZ                        = Int<SettingsT::ZZ>{};
        auto PP                        = Int<SettingsT::PP>{};
        auto QQ                        = Int<SettingsT::QQ>{};
        auto CHy                       = Int<SettingsT::CHy>{};
        auto CHz                       = Int<SettingsT::CHz>{};
        auto STx                       = Int<SettingsT::STx>{};
        auto STy                       = Int<SettingsT::STy>{};
        auto STz                       = Int<SettingsT::STz>{};
        auto xformed_act_logical_inner = make_layout(
            make_shape(make_shape(make_shape(ZZ, PP, QQ), Z, P, Q), make_shape(C, T, R, S)),
            make_stride(
                make_stride(make_stride(CHy * CHz * Z * STx * EG, CHz * P * STy * EG, Q * STz * EG),
                            CHy * CHz * STx * EG,
                            CHz * STy * EG,
                            STz * EG),
                make_stride(EC, CHy * CHz * EG, CHz * EG, EG)));

        auto xformed_act_gather_outer =
            make_layout(make_shape(_1{}, _1{}),
                        make_stride(
                            gather_util::CustomStride{
                                gather_util::IndexedGather<uint64_t, -1>{gather_idx_buf}, C},
                            _1{}));

        return composition(
            xformed_act_gather_outer, make_arithmetic_tuple(_0{}, _0{}), xformed_act_logical_inner);
    }

    __hostdev__ auto
    clusterActivationIndexLayout() {
        auto C   = geometry.C();
        auto T   = Int<SettingsT::T>{};
        auto R   = Int<SettingsT::R>{};
        auto S   = Int<SettingsT::S>{};
        auto ZZ  = Int<SettingsT::ZZ>{};
        auto PP  = Int<SettingsT::PP>{};
        auto QQ  = Int<SettingsT::QQ>{};
        auto CHy = Int<SettingsT::CHy>{};
        auto CHz = Int<SettingsT::CHz>{};
        auto STx = Int<SettingsT::STx>{};
        auto STy = Int<SettingsT::STy>{};
        auto STz = Int<SettingsT::STz>{};
        return make_layout(
            make_shape(make_shape(make_shape(ZZ, PP, QQ), Z, P, Q), make_shape(C, T, R, S)),
            make_stride(make_stride(make_stride(CHy * CHz * Z * STx, CHz * P * STy, Q * STz),
                                    CHy * CHz * STx,
                                    CHz * STy,
                                    STz),
                        make_stride(_0{}, CHy * CHz, CHz, _1{})));
    }

    __hostdev__ static auto
    clusterActivationPredicateStride() {
        auto CHy = Int<SettingsT::CHy>{};
        auto CHz = Int<SettingsT::CHz>{};
        auto STx = Int<SettingsT::STx>{};
        auto STy = Int<SettingsT::STy>{};
        auto STz = Int<SettingsT::STz>{};
        return make_stride(make_stride(make_stride(CHy * CHz * Z * STx, CHz * P * STy, Q * STz),
                                       CHy * CHz * STx,
                                       CHz * STy,
                                       STz),
                           make_stride(_0{}, _0{}, _0{}, _0{}),
                           make_stride(_0{}, CHy * CHz, CHz, _1{}));
    }

    __hostdev__ auto
    filterLayout() {
        auto C = geometry.C();
        auto K = geometry.K();
        auto T = Int<SettingsT::T>{};
        auto R = Int<SettingsT::R>{};
        auto S = Int<SettingsT::S>{};
        return make_ordered_layout(make_shape(K, make_shape(C, T, R, S)),
                                   tuple<_1, tuple<_0, _4, _3, _2>>{});
    }

    __hostdev__ auto
    outputComposedScatterLayout(const uint64_t *scatter_idx_buf) {
        auto ES = E<0>{};
        auto EC = E<1>{};
        auto K  = geometry.K();
        auto Bx = Int<SettingsT::Bx>{};
        auto By = Int<SettingsT::By>{};
        auto Bz = Int<SettingsT::Bz>{};
        auto xformed_out_logical_inner =
            make_layout(make_shape(K, make_shape(make_shape(Bx, By, Bz), Z, P, Q)),
                        make_stride(EC,
                                    make_stride(make_stride(_64{} * Z * ES, _8() * P * ES, Q * ES),
                                                _64{} * ES,
                                                _8{} * ES,
                                                ES)));
        auto xformed_out_scatter_outer =
            make_layout(make_shape(_1{}, _1{}),
                        make_stride(
                            gather_util::CustomStride{
                                gather_util::IndexedGather<uint64_t, -1>{scatter_idx_buf}, K},
                            _1{}));
        return composition(xformed_out_scatter_outer,
                           make_arithmetic_tuple(_0{}, _0{}),
                           xformed_out_logical_inner);
    }

    __hostdev__ auto
    outputIndexLayout() {
        auto K  = geometry.K();
        auto Bx = Int<SettingsT::Bx>{};
        auto By = Int<SettingsT::By>{};
        auto Bz = Int<SettingsT::Bz>{};
        return make_layout(
            make_shape(K, make_shape(make_shape(Bx, By, Bz), Z, P, Q)),
            make_stride(_0{}, make_stride(make_stride(_64{} * Z, _8{} * P, Q), _64{}, _8{}, _1{})));
    }

    __hostdev__ static auto
    clusterOutputPredicateStride() {
        return make_stride(
            _0{}, make_stride(make_stride(CVy * CVz * Z, CVz * P, Q), CVy * CVz, CVz, _1{}));
    }
};

// ---------------------------------------------------------------------------
// SparseFpropSm80 -- the IGEMM kernel operator
// ---------------------------------------------------------------------------

template <class SettingsT> struct SparseFpropSm80 {
    using Z = Int<SettingsT::Z>;
    using P = Int<SettingsT::P>;
    using Q = Int<SettingsT::Q>;

    using ZZ = Int<SettingsT::ZZ>;
    using PP = Int<SettingsT::PP>;
    using QQ = Int<SettingsT::QQ>;

    using Cx = Int<SettingsT::Cx>;
    using Cy = Int<SettingsT::Cy>;
    using Cz = Int<SettingsT::Cz>;

    using CHx = Int<SettingsT::CHx>;
    using CHy = Int<SettingsT::CHy>;
    using CHz = Int<SettingsT::CHz>;

    using CVx = Int<SettingsT::CVx>;
    using CVy = Int<SettingsT::CVy>;
    using CVz = Int<SettingsT::CVz>;

    using Tiler_K  = Int<SettingsT::TK>;
    using Tiler_C  = Int<SettingsT::TC>;
    using Tiler_N  = Shape<ZZ, PP, QQ>;
    using TileM    = Tiler_K;
    using TileN    = Shape<Tiler_N, Z, P, Q>;
    using TileK    = Shape<Tiler_C, _1, _1, _1>;
    using PIPE     = _3;
    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct = Shape<TileN, TileK>;
    using TilerOut = Shape<TileM, TileN>;

    using TileSizeM = Int<size(TileM{})>;
    using TileSizeN = Int<size(TileN{})>;

    using ElementFlt = tfloat32_t;
    using ElementAct = tfloat32_t;
    using ElementOut = float;

    using ClusterShape       = Shape<Cx, Cy, Cz>;
    using ClusterHaloLayout  = decltype(make_layout(Shape<CHx, CHy, CHz>{}, GenRowMajor{}));
    using ClusterVoxelLayout = decltype(make_layout(Shape<CVx, CVy, CVz>{}, GenRowMajor{}));

    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
                              Layout<Shape<_2, _2, _1>>,
                              Tile<_32, _32, Underscore>>;

    static constexpr int MaxThreadsPerBlock         = size(TiledMma{});
    static constexpr int MinBlocksPerMultiprocessor = 1;

    struct SharedStorage {
        union {
            struct {
                ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
                ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
            } mainloop;

            struct {
                ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
            } epilogue;
        };

        uint64_t sBIdxMatrix[SettingsT::VoxelsPerClusterWithHalo()];
        uint64_t sCIdxMatrix[SettingsT::VoxelsPerLeafnodeNoHalo()];
        bool sBPredMatrix[SettingsT::VoxelsPerClusterWithHalo()];
        bool sCPredMatrix[SettingsT::VoxelsPerClusterNoHalo()];
    };

    using GmemTiledCopyFlt =
        decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementFlt>{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _4>>{}));

    using SmemLayoutAtomFlt = decltype(composition(
        Swizzle<1, 2, 3>{}, Layout<Shape<_8, Shape<_4, _2>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomFlt = Copy_Atom<SM75_U32x4_LDSM_N, ElementFlt>;

    using GmemTiledCopyAct = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, ElementAct>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{}));

    using SmemLayoutAtomAct = decltype(composition(
        Swizzle<1, 2, 3>{}, Layout<Shape<_8, Shape<_4, _2>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomAct = Copy_Atom<SM75_U32x4_LDSM_N, ElementAct>;

    using GmemTiledCopyOut =
        decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAct>{},
                                 Layout<Shape<_8, _16>, Stride<_1, _8>>{},
                                 Layout<Shape<_4, _1>>{}));

    using SmemCopyAtomOut = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

    using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

    SettingsT geometry{};

    __hostdev__
    SparseFpropSm80(SettingsT g = {})
        : geometry(g) {}

    template <class BuildT, class EngineFlt, class LayoutFlt>
    void __device__
    operator()(cute::Tensor<EngineFlt, LayoutFlt> mFlt,
               const nanovdb::NanoGrid<BuildT> *mActGrid,
               const nanovdb::NanoGrid<BuildT> *mOutGrid,
               const float *actData,
               float *outData,
               char *smem_buf) const {
        using namespace cute;
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
            cutlass::gemm::MainloopSm80CpAsyncPredGatherB<PIPE::value>,
            Shape<TileM, TileN, TileK>,
            ElementFlt,
            Underscore,
            ElementAct,
            Underscore,
            TiledMma,
            GmemTiledCopyFlt,
            SmemLayoutAtomFlt,
            SmemCopyAtomFlt,
            cute::identity,
            GmemTiledCopyAct,
            SmemLayoutAtomAct,
            SmemCopyAtomAct,
            cute::identity>;

        int leafID = blockIdx.x;

        const auto &outLeaf = mOutGrid->tree().template getFirstNode<0>()[leafID];
        auto sCIdx_ptr      = &reinterpret_cast<SharedStorage *>(smem_buf)->sCIdxMatrix[0];
        for (int v = 0; v < SettingsT::VoxelsPerLeafnodeNoHalo(); v += MaxThreadsPerBlock)
            sCIdx_ptr[v + threadIdx.x] = outLeaf.getValue(v + threadIdx.x);
        const auto &actTree         = mActGrid->tree();
        auto sBIdx_ptr              = &reinterpret_cast<SharedStorage *>(smem_buf)->sBIdxMatrix[0];
        const auto outputLeafOrigin = outLeaf.origin();
        const auto actLeafOrigin    = nanovdb::Coord(outputLeafOrigin[0] * SettingsT::STx,
                                                  outputLeafOrigin[1] * SettingsT::STy,
                                                  outputLeafOrigin[2] * SettingsT::STz);

        Layouts<SettingsT> layouts(geometry);
        Tensor gOut =
            make_tensor(make_gmem_ptr(outData), layouts.outputComposedScatterLayout(sCIdx_ptr));
        Tensor sOutIdx = make_tensor(make_smem_ptr(sCIdx_ptr), layouts.outputIndexLayout());

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});

        Tensor gA_mk    = local_tile(mFlt, TilerFlt{}, make_coord(_, _));
        Tensor gC_mn    = local_tile(gOut, TilerOut{}, make_coord(_, _));
        Tensor sCIdx_mn = local_tile(sOutIdx, TilerOut{}, make_coord(_, _));

        for (int m_coord = 0; m_coord < size<2>(gA_mk); ++m_coord)
            for (int clusterID = 0; clusterID < size(ClusterShape{}); ++clusterID) {
                clear(accum);

                auto clusterCoord = idx2crd(clusterID, ClusterShape{});
                auto n_coord      = make_tuple(clusterCoord, _0{}, _0{}, _0{});

                const auto clusterFilterOrigin = actLeafOrigin.offsetBy(
                    get<0>(clusterCoord) * SettingsT::ZZ * SettingsT::Z * SettingsT::STx +
                        geometry.Dx(),
                    get<1>(clusterCoord) * SettingsT::PP * SettingsT::P * SettingsT::STy +
                        geometry.Dy(),
                    get<2>(clusterCoord) * SettingsT::QQ * SettingsT::Q * SettingsT::STz +
                        geometry.Dz());
                for (int v = 0; v < SettingsT::VoxelsPerClusterWithHalo(); v += MaxThreadsPerBlock)
                    if ((v + threadIdx.x) < SettingsT::VoxelsPerClusterWithHalo()) {
                        auto [i, j, k] = idx2crd(v + threadIdx.x,
                                                 shape(ClusterHaloLayout{}),
                                                 stride(ClusterHaloLayout{}));
                        sBIdx_ptr[v + threadIdx.x] =
                            actTree.getValue(clusterFilterOrigin.offsetBy(i, j, k));
                    }

                __syncthreads();

                Tensor gAct = make_tensor(make_gmem_ptr(actData),
                                          layouts.clusterActivationComposedGatherLayout(sBIdx_ptr));
                Tensor sActIdx =
                    make_tensor(make_smem_ptr(sBIdx_ptr), layouts.clusterActivationIndexLayout());

                Tensor gA    = gA_mk(_, _, m_coord, _);
                Tensor gB    = local_tile(gAct, TilerAct{}, make_coord(_0{}, _));
                Tensor sBIdx = local_tile(sActIdx, TilerAct{}, make_coord(_0{}, _));
                Tensor gC    = gC_mn(_, _, m_coord, n_coord);
                Tensor sCIdx = sCIdx_mn(_, _, m_coord, n_coord);

                auto sBPred_ptr = &reinterpret_cast<SharedStorage *>(smem_buf)->sBPredMatrix[0];
                Tensor sBPred   = make_tensor(make_smem_ptr(sBPred_ptr),
                                            shape(sBIdx),
                                            Layouts<SettingsT>::clusterActivationPredicateStride());
                for (int v = 0; v < SettingsT::VoxelsPerClusterWithHalo(); v += MaxThreadsPerBlock)
                    if (v + threadIdx.x < SettingsT::VoxelsPerClusterWithHalo())
                        sBPred_ptr[v + threadIdx.x] = (bool)sBIdx_ptr[v + threadIdx.x];

                auto sCPred_ptr = &reinterpret_cast<SharedStorage *>(smem_buf)->sCPredMatrix[0];
                Tensor sCPred   = make_tensor(make_smem_ptr(sCPred_ptr),
                                            shape(sCIdx),
                                            Layouts<SettingsT>::clusterOutputPredicateStride());
                for (int v = 0; v < SettingsT::VoxelsPerClusterNoHalo(); v += MaxThreadsPerBlock)
                    if (v + threadIdx.x < SettingsT::VoxelsPerClusterNoHalo()) {
                        auto [i, j, k] = idx2crd(v + threadIdx.x,
                                                 shape(ClusterVoxelLayout{}),
                                                 stride(ClusterVoxelLayout{}));
                        auto coord     = make_tuple(0, make_tuple(make_tuple(0, 0, 0), i, j, k));
                        sCPred(coord)  = sCIdx(coord);
                    }

                __syncthreads();

                auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
                int k_tile_count = size<2>(gA);

                CollectiveMainloop collective_mma;
                collective_mma(accum,
                               gA,
                               gB,
                               sBPred,
                               accum,
                               k_tile_iter,
                               k_tile_count,
                               Underscore{},
                               threadIdx.x,
                               smem_buf);

                // Epilogue: smem staging for predicated output scatter
                SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
                Tensor sC =
                    make_tensor(make_smem_ptr(&storage.epilogue.sCMatrix[0]), SmemLayoutOut{});

                auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomOut{}, tiled_mma);
                auto smem_thr_copy_C   = smem_tiled_copy_C.get_slice(threadIdx.x);
                auto tCrC              = smem_thr_copy_C.retile_S(accum);
                auto tCsC              = smem_thr_copy_C.partition_D(sC);
                copy(smem_tiled_copy_C, tCrC, tCsC);

                __syncthreads();

                GmemTiledCopyOut gmem_tiled_copy_C;
                auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
                auto tDsC            = gmem_thr_copy_C.partition_S(sC);
                auto tDgC            = gmem_thr_copy_C.partition_D(gC);
                auto tDsCPred        = gmem_thr_copy_C.partition_D(sCPred);
                copy_if(gmem_tiled_copy_C, tDsCPred, tDsC, tDgC);

                __syncthreads();
            }
    }
};

// ---------------------------------------------------------------------------
// Kernel launch wrapper
// ---------------------------------------------------------------------------

template <typename Operator, typename BuildT, typename FilterTensor>
__global__ void
    __launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
    kernelEntrypoint(FilterTensor mFlt,
                     const nanovdb::NanoGrid<BuildT> *mActGrid,
                     const nanovdb::NanoGrid<BuildT> *mOutGrid,
                     const float *actData,
                     float *outData,
                     Operator op) {
    extern __shared__ char smem_buf[];
    op(mFlt, mActGrid, mOutGrid, actData, outData, smem_buf);
}

} // namespace pred_gather_igemm

// ============================================================================
// Dispatch axis types for kernel size and stride
// ============================================================================
// The dispatch framework's tag machinery (is_with_type, tag_get, etc.) relies
// on enum/integer NTTPs. C++20 class-type NTTPs (named<>, structs) fail with
// gcc 13 / nvcc due to partial-specialization matching issues, so we use
// typed enums whose underlying values are the actual kernel sizes and strides.

namespace fvdb {
namespace detail {
namespace ops {

enum class conv_kernel_size : int { k3 = 3, k5 = 5, k7 = 7 };
enum class conv_stride : int { s1 = 1, s2 = 2 };
enum class conv_kout_denom : int { K32 = 32, K128 = 128 };

} // namespace ops
} // namespace detail
} // namespace fvdb

namespace dispatch {

template <> struct type_label<fvdb::detail::ops::conv_kernel_size> {
    static consteval auto
    value() {
        return fixed_label("conv.kernel_size");
    }
};

template <> struct type_label<fvdb::detail::ops::conv_stride> {
    static consteval auto
    value() {
        return fixed_label("conv.stride");
    }
};

template <> struct type_label<fvdb::detail::ops::conv_kout_denom> {
    static consteval auto
    value() {
        return fixed_label("conv.kout_denom");
    }
};

} // namespace dispatch

// ============================================================================
// fVDB entry point with dispatch over kernel size and stride
// ============================================================================

namespace fvdb {
namespace detail {
namespace ops {

using kernel_size_axis =
    dispatch::axis<conv_kernel_size::k3, conv_kernel_size::k5, conv_kernel_size::k7>;
using stride_axis     = dispatch::axis<conv_stride::s1, conv_stride::s2>;
using kout_denom_axis = dispatch::axis<conv_kout_denom::K32, conv_kout_denom::K128>;

struct pred_gather_igemm_op {
    template <typename Tag>
    static torch::Tensor
    op(Tag tg,
       torch::Tensor features,
       torch::Tensor weights,
       GridBatchData const &feature_grid,
       GridBatchData const &output_grid) {
        using namespace pred_gather_igemm;
        constexpr int ks         = static_cast<int>(dispatch::tag_get<conv_kernel_size>(tg));
        constexpr int st         = static_cast<int>(dispatch::tag_get<conv_stride>(tg));
        constexpr int kout_denom = static_cast<int>(dispatch::tag_get<conv_kout_denom>(tg));
        using GeomT              = Geometry<ks, ks, ks, st, st, st, kout_denom>;
        using ConvOp             = SparseFpropSm80<GeomT>;

        const c10::cuda::CUDAGuard device_guard(features.device());

        const int64_t C = features.size(1);
        const int64_t K = weights.size(0);

        const int64_t N_out            = output_grid.totalVoxels();
        const uint32_t outputLeafCount = output_grid.numLeavesAt(0);

        auto filter_igemm = weights.permute({2, 3, 4, 0, 1}).contiguous();

        auto opts   = torch::dtype(torch::kFloat32).device(features.device());
        auto output = torch::zeros({N_out, K}, opts);

        auto *nanoInputGrid =
            feature_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
        auto *nanoOutputGrid =
            output_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
        TORCH_CHECK(nanoInputGrid != nullptr, "Failed to get device input grid");
        TORCH_CHECK(nanoOutputGrid != nullptr, "Failed to get device output grid");

        GeomT geometry{};
        geometry.c  = static_cast<int>(C);
        geometry.k  = static_cast<int>(K);
        geometry.dx = -(GeomT::T / 2);
        geometry.dy = -(GeomT::R / 2);
        geometry.dz = -(GeomT::S / 2);

        Layouts<GeomT> layouts(geometry);
        auto tFilter = cute::make_tensor(cute::make_gmem_ptr(filter_igemm.data_ptr<float>()),
                                         layouts.filterLayout());

        ConvOp conv_op(geometry);

        constexpr size_t smem_size = sizeof(typename ConvOp::SharedStorage);
        cudaStream_t stream        = at::cuda::getCurrentCUDAStream();

        cudaFuncSetAttribute(kernelEntrypoint<ConvOp, nanovdb::ValueOnIndex, decltype(tFilter)>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);

        kernelEntrypoint<ConvOp, nanovdb::ValueOnIndex, decltype(tFilter)>
            <<<outputLeafCount, ConvOp::MaxThreadsPerBlock, smem_size, stream>>>(
                tFilter,
                nanoInputGrid,
                nanoOutputGrid,
                features.data_ptr<float>(),
                output.data_ptr<float>(),
                conv_op);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
    }

    using space      = dispatch::axes<kernel_size_axis, stride_axis, kout_denom_axis>;
    using subspaces  = dispatch::coverage<space>;
    using dispatcher = dispatch::dispatch_table<
        space,
        torch::Tensor(torch::Tensor, torch::Tensor, GridBatchData const &, GridBatchData const &)>;
};

torch::Tensor
predGatherIGemmSparseConv(torch::Tensor features,
                          torch::Tensor weights,
                          GridBatchData const &feature_grid,
                          GridBatchData const &output_grid,
                          int kernel_size,
                          int stride) {
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "features must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(features.dim() == 2, "features must be 2-D [N_in, C]");
    TORCH_CHECK(weights.dim() == 5, "weights must be 5-D [K, C, T, R, S]");
    TORCH_CHECK(features.is_contiguous(), "features must be contiguous");

    TORCH_CHECK(kernel_size == 3 || kernel_size == 5 || kernel_size == 7,
                "PredGatherIGemm supports kernel sizes 3, 5, 7; got ",
                kernel_size);
    TORCH_CHECK(stride == 1 || stride == 2, "PredGatherIGemm supports strides 1, 2; got ", stride);

    const int64_t C = features.size(1);
    const int64_t K = weights.size(0);

    TORCH_CHECK(weights.size(1) == C, "weights C dimension must match features");
    TORCH_CHECK(weights.size(2) == kernel_size && weights.size(3) == kernel_size &&
                    weights.size(4) == kernel_size,
                "weights spatial dimensions must be ",
                kernel_size,
                "x",
                kernel_size,
                "x",
                kernel_size);
    TORCH_CHECK(C % 32 == 0, "Input channels must be a multiple of 32, got ", C);
    TORCH_CHECK(K % 32 == 0, "Output channels must be a multiple of 32, got ", K);

    TORCH_CHECK(feature_grid.batchSize() == 1, "PredGatherIGemm currently supports batch size 1");
    TORCH_CHECK(output_grid.batchSize() == 1, "PredGatherIGemm currently supports batch size 1");
    TORCH_CHECK(feature_grid.totalVoxels() == features.size(0),
                "feature_grid voxel count (",
                feature_grid.totalVoxels(),
                ") must match features row count (",
                features.size(0),
                ")");

    static auto const table =
        dispatch::dispatch_table_from_op<pred_gather_igemm_op>("pred_gather_igemm_sparse_conv");

    // We've already checked that K is a multiple of 32, so we can use that as the kout_denom, but
    // if it is a multiple of 128, we can use that instead.
    conv_kout_denom kout_denom = (K % 128 == 0) ? conv_kout_denom::K128 : conv_kout_denom::K32;

    return table.select(dispatch::dispatch_set{
        static_cast<conv_kernel_size>(kernel_size), static_cast<conv_stride>(stride), kout_denom})(
        features, weights, feature_grid, output_grid);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
