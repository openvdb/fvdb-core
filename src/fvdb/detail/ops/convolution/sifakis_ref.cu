// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// sifakis_ref.cu -- Sifakis reference implementation of gather-tensor
// CUTLASS/CuTe IGEMM sparse convolution on NanoVDB grids.

// NOTE: <torch/types.h> MUST precede CuTe / CUTLASS headers.  This file
// has no other torch includes to do the job implicitly.  nvcc 13.1 causes
// CUTLASS 4.x to use <cccl/cuda/std/...> from the local CUDA toolkit, but
// those headers resolve internal <cuda/std/...> includes against the conda
// CUDA 12.9 copies (via -isystem), which lack the
// _LIBCUDACXX_HAS_SPACESHIP_OPERATOR macro.  Including <torch/types.h>
// first loads the 12.9 <cuda/std/utility> (via c10) and sets its include
// guard, preventing the mismatched 13.1 cccl/ variant from ever being
// processed.  <cuda_runtime.h> alone is not sufficient -- it does not pull
// in CCCL C++ headers.  See also CutlassGroupedGemm.cu.
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <torch/types.h>

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/collective/sm103_kernel_type.hpp>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/util/print_error.hpp>

#include <random>
#include <vector>

namespace example {

using namespace cute;

// Empty type used to disable gather/scatter for a GEMM argument
struct NoGather {
    template <class... Ts> NoGather(Ts...){};
};

/// Function object that applies an index to its argument
template <class Index> struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr Index
    operator()(I i) const {
        return indices_[i];
    }

    CUTE_HOST_DEVICE friend void
    print(IndexedGather const &s) {
        cute::print("Indexed");
    }

    Index const *indices_;
};

/// Function object that applies a stride to its argument
/// Example: StridedFunc<int,_2> gathers every other row/column
template <class Stride> struct StridedGather {
    CUTE_HOST_DEVICE constexpr StridedGather(Stride stride = {}) : stride_(stride) {}

    template <class I>
    CUTE_HOST_DEVICE constexpr auto
    operator()(I i) const {
        return i * stride_;
    }

    CUTE_HOST_DEVICE friend void
    print(StridedGather const &s) {
        cute::print("Strided{");
        print(s.stride_);
        cute::print("}");
    }

    Stride stride_;
};

/// Custom stride object that applies a function followed by a stride
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

    // Circumvent the requirement on make_layout that shape and stride are integral
    template <class Shape>
    CUTE_HOST_DEVICE constexpr friend auto
    make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }

    Func func_;
    Stride stride_;
};

template <class Stride, class Func>
CUTLASS_HOST_DEVICE auto
make_custom_stride_layout(Stride const &stride, Func &&func) {
    // Use a dummy shape and replace the first non-unit stride with a custom gather stride
    auto idx        = find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    return make_layout(
        repeat_like(stride, _1{}),
        replace<I>(stride, CustomStride{static_cast<Func &&>(func), get<I>(stride)}));
}

/// Helper function to optionally create a gather tensor
template <class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE auto
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func) {
    if constexpr (not cutlass::platform::is_same<remove_cvref_t<Func>, NoGather>::value) {
        Layout matrix_layout = make_identity_layout(shape);
        auto offset          = as_arithmetic_tuple(repeat_like(shape, _0{}));
        Layout gather_layout = make_custom_stride_layout(stride, static_cast<Func &&>(func));
        return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
    } else {
        return make_tensor(iter, shape, stride);
    }
}

} // namespace example

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
    // Find index of the stride-1 mode - that is the only one that requires updating inner shape and
    // offset
    auto idx =
        find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;

    // Upcast the outer layout (works as expected)
    auto outer = upcast<N>(layout.layout_a());

    // Upcast the accumulated offset along stride-1 mode
    auto offset =
        as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

    // Upcast the inner layout's shape along stride-1 mode
    auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());

    return composition(outer, offset, inner);
}

} // namespace cute

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// DISPATCH_POLICY_CUSTOM.HPP
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template <int Stages_> struct MainloopSm80CpAsyncUnpredicatedCustom {
    constexpr static int Stages = Stages_;
    using ArchTag               = arch::Sm80;
    using Schedule              = KernelMultistage;
    using ClusterShape          = Shape<_1, _1, _1>;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
// SM80_MMA_MULTISTAGE_CUSTOM.HPP
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

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
struct CollectiveMma<MainloopSm80CpAsyncUnpredicatedCustom<Stages>,
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
    //
    // Type Aliases
    //
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
    // Follow the change in TestSmall: TileShape switch to CtaShape
    // For sm80 arch, CtaShape should equal to TileShape
    using CtaShape_MNK = TileShape;

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

    // Host side kernel arguments
    struct Arguments {
        ElementA const *ptr_A;
        StrideA dA;
        ElementB const *ptr_B;
        StrideB dB;
    };

    // Device side kernel params
    using Params = Arguments;

    //
    // Methods
    //

    CollectiveMma() = default;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const &_, Arguments const &args, void *workspace) {
        (void)workspace;
        return args;
    }

    /// Perform a collective-scoped matrix multiply-accumulate
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

        // Construct shared memory tiles
        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
        Tensor sA =
            make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        Tensor sB =
            make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

        CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                     // BLK_M
        CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                     // BLK_K
        CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                     // BLK_N
        CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sP));                     // BLK_N
        CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                     // BLK_K
        CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sP));                     // BLK_K
        CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                     // BLK_K
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));   // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));   // PIPE

        // Partition the copying of A and B tiles across the threads
        GmemTiledCopyA gmem_tiled_copy_A;
        GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        Tensor tAgA = gmem_thr_copy_A.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,PIPE)
        Tensor tBsP = gmem_thr_copy_B.partition_S(sP); // (BCPY,BCPY_N,BCPY_K,k)

        //
        // PREDICATES
        //

        (void)residue_mnk;
        // assert(residue_mnk == make_tuple(0,0,0));

        //
        // PREFETCH
        //

        // Start async loads for all pipes but the last
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

        //
        // MMA Atom partitioning
        //

        // Tile MMA compute thread partitions and allocate accumulators
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA  = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA,MMA_M,MMA_K)
        Tensor tCrB  = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum)); // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));     // MMA_N
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum)); // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));      // MMA_K
        CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_A) == size(tiled_mma));
        CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_B) == size(tiled_mma));

        //
        // Copy Atom retiling
        //

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA            = smem_thr_copy_A.partition_S(sA);       // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);        // (CPY,CPY_M,CPY_K)
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view)); // CPY_K

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB            = smem_thr_copy_B.partition_S(sB);       // (CPY,CPY_N,CPY_K,PIPE)
        Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);        // (CPY,CPY_N,CPY_K)
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // CPY_N
        CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view)); // CPY_K

        //
        // PIPELINED MAIN LOOP
        //

        // Current pipe index in smem to read from
        int smem_pipe_read = 0;
        // Current pipe index in smem to write to
        int smem_pipe_write = DispatchPolicy::Stages - 1;

        Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
        Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

        // Size of the register pipeline
        auto K_BLOCK_MAX = size<2>(tCrA);

        // PREFETCH register pipeline
        if (K_BLOCK_MAX > 1) {
            // Wait until our first prefetched tile is loaded in
            cp_async_wait<DispatchPolicy::Stages - 2>();
            __syncthreads();

            // Prefetch the first rmem from the first k-tile
            copy(smem_tiled_copy_A, tCsA_p(_, _, Int<0>{}), tCrA_copy_view(_, _, Int<0>{}));
            copy(smem_tiled_copy_B, tCsB_p(_, _, Int<0>{}), tCrB_copy_view(_, _, Int<0>{}));
        }

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(DispatchPolicy::Stages - 1)) {
            // Pipeline the outer products with a static for loop.
            //
            // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    // Slice the smem_pipe_read smem
                    tCsA_p = tCsA(_, _, _, smem_pipe_read);
                    tCsB_p = tCsB(_, _, _, smem_pipe_read);

                    // Commit the smem for smem_pipe_read
                    cp_async_wait<DispatchPolicy::Stages - 2>();
                    __syncthreads();
                }

                // Load A, B shmem->regs for k_block+1
                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
                copy(smem_tiled_copy_A,
                     tCsA_p(_, _, k_block_next),
                     tCrA_copy_view(_, _, k_block_next));
                copy(smem_tiled_copy_B,
                     tCsB_p(_, _, k_block_next),
                     tCrB_copy_view(_, _, k_block_next));
                // Copy gmem to smem before computing gemm on each k-pipe
                if (k_block == 0) {
                    copy(gmem_tiled_copy_A,
                         tAgA(_, _, _, *k_tile_iter),
                         tAsA(_, _, _, smem_pipe_write));
                    copy_if(gmem_tiled_copy_B,
                            tBsP(_, _, _, *k_tile_iter),
                            tBgB(_, _, _, *k_tile_iter),
                            tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();

                    // Advance the tile
                    --k_tile_count;
                    if (k_tile_count > 0) {
                        ++k_tile_iter;
                    }

                    // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read =
                        (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
                }

                // Transform before compute
                cute::transform(tCrA(_, _, k_block), TransformA{});
                cute::transform(tCrB(_, _, k_block), TransformB{});
                // Thread-level register gemm for k_block
                cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrB(_, _, k_block), src_accum);
            });
        }

        cp_async_wait<0>();
        __syncthreads();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
// AMPERE_CONV_KERNEL.H
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------

// #include "dispatch_policy_custom.hpp"
// #include "sm80_mma_multistage_custom.hpp"
// #include "gather_tensor.hpp"

using namespace cute;

template <class SettingsT> struct IGEMM_Layouts {
    static constexpr auto T   = Int<SettingsT::T>{};
    static constexpr auto R   = Int<SettingsT::R>{};
    static constexpr auto S   = Int<SettingsT::S>{};
    static constexpr auto Z   = Int<SettingsT::Z>{};
    static constexpr auto P   = Int<SettingsT::P>{};
    static constexpr auto Q   = Int<SettingsT::Q>{};
    static constexpr auto C   = Int<SettingsT::C>{};
    static constexpr auto K   = Int<SettingsT::K>{};
    static constexpr auto Bx  = Int<SettingsT::Bx>{};
    static constexpr auto By  = Int<SettingsT::By>{};
    static constexpr auto Bz  = Int<SettingsT::Bz>{};
    static constexpr auto Hx  = Int<SettingsT::Hx>{};
    static constexpr auto Hy  = Int<SettingsT::Hy>{};
    static constexpr auto Hz  = Int<SettingsT::Hz>{};
    static constexpr auto CHx = Int<SettingsT::CHx>{};
    static constexpr auto CHy = Int<SettingsT::CHy>{};
    static constexpr auto CHz = Int<SettingsT::CHz>{};
    static constexpr auto CVx = Int<SettingsT::CVx>{};
    static constexpr auto CVy = Int<SettingsT::CVy>{};
    static constexpr auto CVz = Int<SettingsT::CVz>{};

    __hostdev__ static auto
    activationComposedGatherLayout(const uint64_t *gather_idx_buf) {
        // Input gather layout
        // inner_layout(make_coord((nzpq), (csrt))) => (idx_buffer_idx, dense_c_idx)
        auto EG                        = E<0>{}; // Gather basis     (1,0) (idx_buffer_idx)
        auto EC                        = E<1>{}; // Contiguous basis (0,1) (dense_offset)
        auto xformed_act_logical_inner = make_layout(
            make_shape(make_shape(make_shape(Bx, By, Bz), Z, P, Q), make_shape(C, T, R, S)),
            make_stride(
                make_stride(
                    make_stride(Hy * Hz * Z * EG, Hz * P * EG, Q * EG), Hy * Hz * EG, Hz * EG, EG),
                make_stride(EC, Hy * Hz * EG, Hz * EG, EG)));

        // outer_layout(make_coord(idx_buffer_idx, dense_c_idx)) => idx
        // IndexedGather obtains idx by applying (gmem_base_ptr + gather_idx_buf[idx_buffer_idx] +
        // dense_offset)
        auto xformed_act_gather_outer =
            make_layout(make_shape(_1{}, _1{}),
                        make_stride(example::CustomStride{example::IndexedGather{gather_idx_buf},
                                                          Int<SettingsT::C>{}},
                                    _1{}));

        // Compose the inner and outer layouts
        // gather_composed(make_coord((nzpq), (csrt))) => idx
        return composition(
            xformed_act_gather_outer, make_arithmetic_tuple(_0{}, _0{}), xformed_act_logical_inner);
    }

    __hostdev__ static auto
    activationIndexLayout() {
        // Input gather index layout
        // gather_layout_index(make_coord((ndhw), c)) => buffer_idx
        return make_layout(
            make_shape(make_shape(make_shape(Bx, By, Bz), Z, P, Q), make_shape(C, T, R, S)),
            make_stride(make_stride(make_stride(Hy * Hz * Z, Hz * P, Q), Hy * Hz, Hz, _1{}),
                        make_stride(_0{}, Hy * Hz, Hz, _1{})));
    }

    __hostdev__ static auto
    clusterActivationPredicateStride() {
        // Input gather index layout
        // gather_layout_index(make_coord((ndhw), c)) => buffer_idx
        //     make_shape (make_shape (make_shape (       Bx,    By, Bz),       Z,   P,    Q),
        //     make_shape (  KK,  _1,  _1,  _1), make_shape (  BC,       T,   R,    S)),
        return make_stride(
            make_stride(make_stride(CHy * CHz * Z, CHz * P, Q), CHy * CHz, CHz, _1{}),
            make_stride(_0{}, _0{}, _0{}, _0{}),
            make_stride(_0{}, CHy * CHz, CHz, _1{}));
    }

    __hostdev__ static auto
    filterLayout() {
        return make_ordered_layout(make_shape(K, make_shape(C, T, R, S)),
                                   tuple<_1, tuple<_0, _4, _3, _2>>{});
    }

    __hostdev__ static auto
    outputComposedScatterLayout(const uint64_t *scatter_idx_buf) {
        // Output scatter layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        auto ES = E<0>{}; // Scatter basis    (1,0) (idx_buffer_idx)
        auto EC = E<1>{}; // Contiguous basis (0,1) (dense_offset)
        auto xformed_out_logical_inner =
            make_layout(make_shape(K, make_shape(make_shape(Bx, By, Bz), Z, P, Q)),
                        make_stride(EC,
                                    make_stride(make_stride(_64{} * Z * ES, _8() * P * ES, Q * ES),
                                                _64{} * ES,
                                                _8{} * ES,
                                                ES)));
        auto xformed_out_scatter_outer =
            make_layout(make_shape(_1{}, _1{}),
                        make_stride(example::CustomStride{example::IndexedGather{scatter_idx_buf},
                                                          Int<SettingsT::K>{}},
                                    _1{}));
        return composition(xformed_out_scatter_outer,
                           make_arithmetic_tuple(_0{}, _0{}),
                           xformed_out_logical_inner);
    }

    __hostdev__ static auto
    outputIndexLayout() {
        // Output scatter index layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        return make_layout(
            make_shape(K, make_shape(make_shape(Bx, By, Bz), Z, P, Q)),
            make_stride(_0{}, make_stride(make_stride(_64{} * Z, _8{} * P, Q), _64{}, _8{}, _1{})));
    }

    __hostdev__ static auto
    clusterOutputPredicateStride() {
        // Output scatter index layout
        // scatter_layout_index(k, make_coord((nzpq))) => buffer_idx
        //     make_shape (  BK, make_shape (make_shape (       Bx,    By, Bz),       Z,   P, Q)),
        return make_stride(
            _0{}, make_stride(make_stride(CVy * CVz * Z, CVz * P, Q), CVy * CVz, CVz, _1{}));
    }
};

template <class SettingsT> struct AmperePredicatedFprop {
    //
    // Static config for conv problem shape
    //

    using T = Int<SettingsT::T>;
    using R = Int<SettingsT::R>;
    using S = Int<SettingsT::S>;

    using Z = Int<SettingsT::Z>;
    using P = Int<SettingsT::P>;
    using Q = Int<SettingsT::Q>;

    using ZZ = Int<SettingsT::ZZ>;
    using PP = Int<SettingsT::PP>;
    using QQ = Int<SettingsT::QQ>;

    using Cx = Int<SettingsT::Cx>;
    using Cy = Int<SettingsT::Cy>;
    using Cz = Int<SettingsT::Cz>;

    using Hx = Int<SettingsT::Hx>;
    using Hy = Int<SettingsT::Hy>;
    using Hz = Int<SettingsT::Hz>;

    using CHx = Int<SettingsT::CHx>;
    using CHy = Int<SettingsT::CHy>;
    using CHz = Int<SettingsT::CHz>;

    using CVx = Int<SettingsT::CVx>;
    using CVy = Int<SettingsT::CVy>;
    using CVz = Int<SettingsT::CVz>;

    using C = Int<SettingsT::C>;
    using K = Int<SettingsT::K>;

    // Tiler config
    using Tiler_K  = decltype(cute::min(K{}, _32{}));
    using Tiler_C  = decltype(cute::min(C{}, _32{}));
    using Tiler_N  = Shape<ZZ, PP, QQ>;
    using TileM    = Tiler_K;
    using TileN    = Shape<Tiler_N, Z, P, Q>;
    using TileK    = Shape<Tiler_C, _1, _1, _1>;
    using PIPE     = _3;
    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct = Shape<TileN, TileK>;
    using TilerOut = Shape<TileM, TileN>;

    using TileSizeM             = Int<size(TileM{})>;
    using TileSizeN             = Int<size(TileN{})>;
    using TileSizeK             = Int<size(TileK{})>;
    static constexpr int Stages = PIPE::value;

    using ElementFlt = tfloat32_t;
    using ElementAct = tfloat32_t;
    using ElementOut = float;

    using ClusterShape       = Shape<Cx, Cy, Cz>;
    using HaloLayout         = decltype(make_layout(Shape<Hx, Hy, Hz>{}, GenRowMajor{}));
    using ClusterHaloLayout  = decltype(make_layout(Shape<CHx, CHy, CHz>{}, GenRowMajor{}));
    using ClusterVoxelLayout = decltype(make_layout(Shape<CVx, CVy, CVz>{}, GenRowMajor{}));

    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
                              Layout<Shape<_2, _2, _1>>,
                              Tile<_32, _32, _8>>;

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

        uint64_t sBIdxMatrix[SettingsT::VoxelsPerLeafnodeWithHalo()];
        uint64_t sCIdxMatrix[SettingsT::VoxelsPerLeafnodeNoHalo()];
        bool sBPredMatrix[SettingsT::VoxelsPerClusterWithHalo()];
        bool sCPredMatrix[SettingsT::VoxelsPerClusterNoHalo()];
    };

    //
    // Stencil tensor
    //

    using GmemLayoutFlt = decltype(make_ordered_layout(Shape<K, Shape<C, T, R, S>>{},
                                                       tuple<_1, tuple<_0, _4, _3, _2>>{}));

    // We have 64 elements * 32b each in the major mode that we can vectorize
    // Max vector size is 128b, so lay 16 threads along the major mode with a vector size of 4
    // Rest along the minor mode
    using GmemTiledCopyFlt =
        decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementFlt>{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _4>>{}));

    // Following layout is also correct, but trades off dynamic strides in the slice for bank
    // conflict free accesses using SmemLayoutFlt = decltype(
    //     composition(Swizzle<3,2,3>{},
    //                 make_ordered_layout(
    //                     Shape<TileSizeM,TileSizeK,PIPE>{},
    //                     tuple<       _1,       _0,  _2>{})));

    using SmemLayoutAtomFlt = decltype(composition(
        Swizzle<1, 2, 3>{}, Layout<Shape<_8, Shape<_4, _2>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomFlt = Copy_Atom<SM75_U32x4_LDSM_N, ElementFlt>;

    //
    // Activation tensor
    //

    // Activation tensor is major in the contraction mode, so vectorize that mode first
    // Then lay out the rest of the threads along the other mode
    using GmemTiledCopyAct = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, ElementAct>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _4>>{}));

    // Following layout is also correct, but trades off dynamic strides in the slice for bank
    // conflict free accesses using SmemLayoutAct = decltype(
    //     composition(Swizzle<3,2,3>{},
    //                 make_ordered_layout(
    //                     Shape<TileSizeN,TileSizeK,PIPE>{},
    //                     tuple<       _1,       _0,  _2>{})));

    using SmemLayoutAtomAct = decltype(composition(
        Swizzle<1, 2, 3>{}, Layout<Shape<_8, Shape<_4, _2>>, Stride<_4, Stride<_1, _32>>>{}));

    using SmemCopyAtomAct = Copy_Atom<SM75_U32x4_LDSM_N, ElementAct>;

    //
    // Output tensor
    //

    using GmemTiledCopyOut =
        decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAct>{},
                                 Layout<Shape<_8, _16>, Stride<_1, _8>>{},
                                 Layout<Shape<_4, _1>>{}));

    using SmemCopyAtomOut = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

    // This can be optimized to make accesses BCF, but we use a col-major layout here to show off
    // composability
    using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

    //
    // Conv functor (predicated IGEMM)
    //
    template <class BuildT, class EngineFlt>
    // class ActivationTensor, class ActivationIndexTensor, class OutputTensor, class
    // OutputIndexTensor>
    void __device__
    operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt, // (K,(C,T,R,S))
               const nanovdb::NanoGrid<BuildT> *mActGrid,
               const nanovdb::NanoGrid<BuildT> *mOutGrid,
               const float *actData,
               float *outData,
               char *smem_buf) const {
        using namespace cute;
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
            cutlass::gemm::MainloopSm80CpAsyncUnpredicatedCustom<PIPE::value>,
            Shape<TileM, TileN, TileK>,
            ElementFlt,
            Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
            ElementAct,
            Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
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

        // Populate activation (gather) and output (scatter) indices
        const auto &outLeaf = mOutGrid->tree().template getFirstNode<0>()[leafID];
        auto sCIdx_ptr      = &reinterpret_cast<SharedStorage *>(smem_buf)->sCIdxMatrix[0];
        for (int v = 0; v < SettingsT::VoxelsPerLeafnodeNoHalo(); v += MaxThreadsPerBlock)
            sCIdx_ptr[v + threadIdx.x] = outLeaf.getValue(v + threadIdx.x);
        const auto &actTree = mActGrid->tree();
        auto sBIdx_ptr      = &reinterpret_cast<SharedStorage *>(smem_buf)->sBIdxMatrix[0];
        const auto filterOrigin =
            outLeaf.origin().offsetBy(SettingsT::Dx, SettingsT::Dy, SettingsT::Dz);
        for (int v = 0; v < SettingsT::VoxelsPerLeafnodeWithHalo(); v += MaxThreadsPerBlock)
            if ((v + threadIdx.x) < SettingsT::VoxelsPerLeafnodeWithHalo()) {
                auto [i, j, k] =
                    idx2crd(v + threadIdx.x, shape(HaloLayout{}), stride(HaloLayout{}));
                sBIdx_ptr[v + threadIdx.x] = actTree.getValue(filterOrigin.offsetBy(i, j, k));
            }

        __syncthreads();

        Tensor gAct =
            make_tensor(make_gmem_ptr(actData),
                        IGEMM_Layouts<SettingsT>::activationComposedGatherLayout(sBIdx_ptr));
        Tensor sActIdx = make_tensor(make_smem_ptr(sBIdx_ptr),
                                     IGEMM_Layouts<SettingsT>::activationIndexLayout());
        Tensor gOut    = make_tensor(make_gmem_ptr(outData),
                                  IGEMM_Layouts<SettingsT>::outputComposedScatterLayout(sCIdx_ptr));
        Tensor sOutIdx =
            make_tensor(make_smem_ptr(sCIdx_ptr), IGEMM_Layouts<SettingsT>::outputIndexLayout());

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});

        // Set up tensors
        // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of
        // higher dynamic range in NDHW
        Tensor gA_mk    = local_tile(mFlt, TilerFlt{}, make_coord(_, _));    // (BLK_M,BLK_K,m',k')
        Tensor gB_nk    = local_tile(gAct, TilerAct{}, make_coord(_, _));    // (BLK_N,BLK_K,n',_1)
        Tensor sBIdx_nk = local_tile(sActIdx, TilerAct{}, make_coord(_, _)); // (BLK_N,BLK_K,n',_1)
        Tensor gC_mn    = local_tile(gOut, TilerOut{}, make_coord(_, _));    // (BLK_M,BLK_N,m',n')
        Tensor sCIdx_mn = local_tile(sOutIdx, TilerOut{}, make_coord(_, _)); // (BLK_M,BLK_N,m',n')

        for (int m_coord = 0; m_coord < size<2>(gA_mk); ++m_coord)
            for (int clusterID = 0; clusterID < size(ClusterShape{}); ++clusterID) {
                clear(accum);

                auto clusterCoord = idx2crd(clusterID, ClusterShape{});
                auto n_coord      = make_tuple(clusterCoord, _0{}, _0{}, _0{});

                Tensor gA    = gA_mk(_, _, m_coord, _);          // (BLK_M,BLK_K,k')
                Tensor gB    = gB_nk(_, _, n_coord, _);          // (BLK_N,BLK_K,_1)
                Tensor sBIdx = sBIdx_nk(_, _, n_coord, _);       // (BLK_N,BLK_K,_1)
                Tensor gC    = gC_mn(_, _, m_coord, n_coord);    // (BLK_M,BLK_N)
                Tensor sCIdx = sCIdx_mn(_, _, m_coord, n_coord); // (BLK_M,BLK_N)

                // Build gather predicate tensors in SMEM

                auto sBPred_ptr = &reinterpret_cast<SharedStorage *>(smem_buf)->sBPredMatrix[0];
                Tensor sBPred =
                    make_tensor(make_smem_ptr(sBPred_ptr),
                                shape(sBIdx),
                                IGEMM_Layouts<SettingsT>::clusterActivationPredicateStride());
                for (int v = 0; v < SettingsT::VoxelsPerClusterWithHalo(); v += MaxThreadsPerBlock)
                    if (v + threadIdx.x < SettingsT::VoxelsPerClusterWithHalo()) {
                        auto [i, j, k] = idx2crd(v + threadIdx.x,
                                                 shape(ClusterHaloLayout{}),
                                                 stride(ClusterHaloLayout{}));
                        auto coord     = make_tuple(make_tuple(make_tuple(0, 0, 0), i, j, k),
                                                make_tuple(0, 0, 0, 0),
                                                make_tuple(0, 0, 0, 0));
                        sBPred(coord)  = sBIdx(coord);
                    }

                auto sCPred_ptr = &reinterpret_cast<SharedStorage *>(smem_buf)->sCPredMatrix[0];
                Tensor sCPred =
                    make_tensor(make_smem_ptr(sCPred_ptr),
                                shape(sCIdx),
                                IGEMM_Layouts<SettingsT>::clusterOutputPredicateStride());
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
                               Underscore{}, // no residue since we do not support predication
                               threadIdx.x,
                               smem_buf);

                //
                // Epilogue
                //

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

                __syncthreads(); // necessary if the predicate tensors are built once per leafnode
            }
        // __syncthreads(); // should be safe to synchronize only here; predicates built at every
        // iteration
    }
};

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
// SPARSE_CONVOLUTION_IGEMM_NANOVDB_CUDA_KERNELS.CU
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// #include "ampere_conv_kernel.h"

#define USE_HIERARCHICAL_BLOCK_TRAVERSAL

template <typename T>
bool
bufferCheck(const T *deviceBuffer, const T *hostBuffer, size_t elem_count) {
    T *tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i = 0; same && i < elem_count; ++i) {
        same = (tmpBuffer[i] == hostBuffer[i]);
    }
    delete[] tmpBuffer;
    return same;
}

struct IGEMM_Geometry {
    //
    // Convolution geometry
    //

    static constexpr int T = 3;         // X-dimension of convolution filter
    static constexpr int R = 3;         // Y-dimension of convolution filter
    static constexpr int S = 3;         // Z-dimension of convolution filter

    static constexpr int Z = 4;         // X-dimension of output block
    static constexpr int P = 2;         // Y-dimension of output block
    static constexpr int Q = 2;         // Z-dimension of output block

    static constexpr int D = Z + T - 1; // X-dimension of input block (inluding halo)
    static constexpr int H = P + R - 1; // Y-dimension of input block (inluding halo)
    static constexpr int W = Q + S - 1; // Z-dimension of input block (inluding halo)

    static constexpr int C = 64;        // Input feature dimension
    static constexpr int K = 128;       // Output feature dimension

    //
    // Leaf node geometry
    //

    static constexpr int ZZ =
        1; // Blocks of size (Z,P,Q) are grouped into "clusters" in a (ZZ,PP,QQ) arrangement
    static constexpr int PP = 2; // I.e. ZZ blocks are grouped along the X-dimension, PP along the
                                 // Y- and QQ along the Z-dimension
    static constexpr int QQ = 2; // The total voxel size of a cluster will be (ZZ*Z,PP*P,QQ*Q)

    static constexpr int Bx = 8 / Z;        // Block count along X-dimension of leaf node
    static constexpr int By = 8 / P;        // Block count along Y-dimension of leaf node
    static constexpr int Bz = 8 / Q;        // Block count along Z-dimension of leaf node

    static constexpr int Cx = 8 / (ZZ * Z); // Cluster count along X-dimension of leaf node
    static constexpr int Cy = 8 / (PP * P); // Cluster count along Y-dimension of leaf node
    static constexpr int Cz = 8 / (QQ * Q); // Cluster count along Z-dimension of leaf node

    static constexpr int Hx =
        T + 7; // X-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hy =
        R + 7; // Y-dimension of leaf node domain, enlarged by the necessary halo for convolution
    static constexpr int Hz =
        S + 7; // Z-dimension of leaf node domain, enlarged by the necessary halo for convolution

    static constexpr int CHx =
        ZZ * Z + T -
        1; // Cluster halo (voxel width, plus halo of one cluster) count along X-dimension
    static constexpr int CHy =
        PP * P + R -
        1; // Cluster halo (voxel width, plus halo of one cluster) count along Y-dimension
    static constexpr int CHz =
        QQ * Q + S -
        1; // Cluster halo (voxel width, plus halo of one cluster) count along Z-dimension

    static constexpr int CVx = ZZ * Z; // Voxel count per cluster along X-dimension
    static constexpr int CVy = PP * P; // Voxel count per cluster along Y-dimension
    static constexpr int CVz = QQ * Q; // Voxel count per cluster along Z-dimension

    static constexpr int
    VoxelsPerLeafnodeNoHalo() {
        return 512;
    }
    static constexpr int
    VoxelsPerLeafnodeWithHalo() {
        return Hx * Hy * Hz;
    }

    static constexpr int
    VoxelsPerClusterNoHalo() {
        return CVx * CVy * CVz;
    }
    static constexpr int
    VoxelsPerClusterWithHalo() {
        return CHx * CHy * CHz;
    }

    //
    // Filter offset (coordinate offset in the input domain that the [0,0,0] filter spoke
    // corresponds to)
    //

    static constexpr int Dx =
        -1; // X-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dy =
        -1; // Y-coordinate offset of the minimum corner of the convolution filter
    static constexpr int Dz =
        -1; // Z-coordinate offset of the minimum corner of the convolution filter
};

template <class GeometryT, int Di, int Do, class ValueType>
void
SparseConvolveCPUReference(nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
                           nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstGrid,
                           const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
                           const ValueType (*inputArray)[Di],
                           ValueType (*outputArray)[Do]) {
    auto dstLeafCount = dstGrid->nodeCount<0>();
    auto srcAcc       = srcGrid->getAccessor();
#pragma omp parallel for firstprivate(srcAcc)
    for (int dstLeafID = 0; dstLeafID < dstLeafCount; ++dstLeafID) {
        auto &dstLeaf = dstGrid->tree().getFirstLeaf()[dstLeafID];
        for (auto dstLeafIt = dstLeaf.cbeginValueOn(); dstLeafIt; ++dstLeafIt) {
            const auto dstIndex = *dstLeafIt;
            const auto dstCoord = dstLeafIt.getCoord();
            for (int i = 0; i < Do; ++i)
                outputArray[dstIndex][i] = 0.f;
            for (int di = 0; di < GeometryT::T; ++di)
                for (int dj = 0; dj < GeometryT::R; ++dj)
                    for (int dk = 0; dk < GeometryT::S; ++dk) {
                        const auto srcCoord = dstCoord.offsetBy(
                            di + GeometryT::Dx, dj + GeometryT::Dy, dk + GeometryT::Dz);
                        const auto srcIndex = srcAcc.getValue(srcCoord);
                        if (srcIndex)
                            for (int out = 0; out < Do; ++out)
                                for (int in = 0; in < Di; ++in)
                                    outputArray[dstIndex][out] +=
                                        filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
                    }
        }
    }
}

template <typename Functor>
__global__ void
lambda_kernel_wrapper(Functor func) {
    func();
}

template <class GeometryT, int Di, int Do, class ValueType>
void
SparseConvolveCudaReference(uint32_t dstLeafCount,
                            nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
                            nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstGrid,
                            const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
                            const ValueType (*inputArray)[Di],
                            ValueType (*outputArray)[Do],
                            cudaStream_t stream = 0) {
    auto convolver = [=] __device__() {
        int dstLeafID = blockIdx.x;
        int out       = threadIdx.x;
        auto &dstLeaf = dstGrid->tree().getFirstLeaf()[dstLeafID];
        for (auto dstLeafIt = dstLeaf.cbeginValueOn(); dstLeafIt; ++dstLeafIt) {
            const auto dstIndex        = *dstLeafIt;
            const auto dstCoord        = dstLeafIt.getCoord();
            outputArray[dstIndex][out] = 0.f;
            for (int di = 0; di < GeometryT::T; ++di)
                for (int dj = 0; dj < GeometryT::R; ++dj)
                    for (int dk = 0; dk < GeometryT::S; ++dk) {
                        const auto srcCoord = dstCoord.offsetBy(
                            di + GeometryT::Dx, dj + GeometryT::Dy, dk + GeometryT::Dz);
                        const auto srcIndex = srcGrid->tree().getValue(srcCoord);
                        if (srcIndex)
                            for (int in = 0; in < Di; ++in)
                                outputArray[dstIndex][out] +=
                                    filter[di][dj][dk][out][in] * inputArray[srcIndex][in];
                    }
        }
    };

    lambda_kernel_wrapper<<<dstLeafCount, Do, 0, stream>>>(convolver);
}

template <class GeometryT, int Di, int Do, class ValueType>
void
SparseConvolveScatterGatherMapsReference(
    uint64_t (*gather_idx_buf)[GeometryT::D][GeometryT::H][GeometryT::W],
    uint64_t (*scatter_idx_buf)[GeometryT::Z][GeometryT::P][GeometryT::Q],
    const std::size_t blockCount,
    const ValueType (*filter)[GeometryT::R][GeometryT::S][Do][Di],
    const ValueType (*inputArray)[Di],
    ValueType (*outputArray)[Do]) {
    auto convolver = [=] __device__() {
        int blockID         = blockIdx.x;
        auto gatherIndices  = gather_idx_buf[blockID];
        auto scatterIndices = scatter_idx_buf[blockID];
        int out             = threadIdx.x;

        for (int i = 0; i < GeometryT::Z; ++i)
            for (int j = 0; j < GeometryT::P; ++j)
                for (int k = 0; k < GeometryT::Q; ++k) {
                    const auto dstIndex = scatterIndices[i][j][k];
                    if (dstIndex) {
                        outputArray[dstIndex][out] = 0.f;
                        for (int di = 0; di < GeometryT::T; ++di)
                            for (int dj = 0; dj < GeometryT::R; ++dj)
                                for (int dk = 0; dk < GeometryT::S; ++dk) {
                                    const auto srcIndex = gatherIndices[i + di][j + dj][k + dk];
                                    if (srcIndex)
                                        for (int in = 0; in < Di; ++in)
                                            outputArray[dstIndex][out] +=
                                                filter[di][dj][dk][out][in] *
                                                inputArray[srcIndex][in];
                                }
                    }
                }
    };

    lambda_kernel_wrapper<<<blockCount, Do>>>(convolver);
}

template <int Do, class ValueType>
void
ResultCompare(const std::size_t size,
              const ValueType (*outputArray1)[Do],
              const ValueType (*outputArray2)[Do]) {
    ValueType result = 0.f;
#pragma omp parallel for reduction(max : result)
    for (size_t i = 0; i < size; i++)
        for (int j = 0; j < Do; j++)
            result = std::max(result, std::abs(outputArray1[i][j] - outputArray2[i][j]));
    std::cout << "Discrepancy = " << result << std::endl;
}

template <class Operator, class BuildT, class FilterTensor>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor) void
kernel_entrypoint_custom(FilterTensor mFlt,
                         const nanovdb::NanoGrid<BuildT> *inputGrid,
                         const nanovdb::NanoGrid<BuildT> *outputGrid,
                         const float *inputData,
                         float *outputData) {
    extern __shared__ char smem_buf[];
    Operator op;
    op(mFlt, inputGrid, outputGrid, inputData, outputData, smem_buf);
}

template <class BufferT>
void
printGridDiagnostics(nanovdb::GridHandle<BufferT> &handle) {
    using BuildT = nanovdb::ValueOnIndex;

    auto deviceGrid = handle.template deviceGrid<BuildT>();
    if (!deviceGrid)
        throw std::logic_error("No GPU grid found in printGridDiagnostics()");

    auto valueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(deviceGrid);
    auto treeData   = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(deviceGrid);
    auto gridSize   = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getGridSize(deviceGrid);
    auto indexBBox =
        nanovdb::util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(deviceGrid, treeData);

    std::cout << "======= Grid info =======" << std::endl;
    std::cout << "Allocated values         : " << valueCount << std::endl;
    std::cout << "Active voxels            : " << treeData.mVoxelCount << std::endl;
    auto minCorner = indexBBox.min(), maxCorner = indexBBox.max();
    std::cout << "Index-space bounding box : [" << minCorner.x() << "," << minCorner.y() << ","
              << minCorner.z() << "] -> [" << maxCorner.x() << "," << maxCorner.y() << ","
              << maxCorner.z() << "]" << std::endl;
    std::cout << "Leaf nodes               : " << treeData.mNodeCount[0] << std::endl;
    std::cout << "Lower internal nodes     : " << treeData.mNodeCount[1] << std::endl;
    std::cout << "Upper internal nodes     : " << treeData.mNodeCount[2] << std::endl;
    std::cout << "Leaf-level occupancy     : "
              << 100.f * (float)(treeData.mVoxelCount) / (float)(treeData.mNodeCount[0] * 512)
              << "%" << std::endl;
    std::cout << "Memory usage             : " << gridSize << " bytes" << std::endl;
}

void
mainSparseConvolutionIGEMM(const std::vector<nanovdb::Coord> &inputPoints,
                           const std::vector<nanovdb::Coord> &outputPoints,
                           uint32_t benchmark_iters) {
    using BuildT            = nanovdb::ValueOnIndex;
    using BufferT           = nanovdb::cuda::DeviceBuffer;
    static constexpr int Di = IGEMM_Geometry::C;
    static constexpr int Do = IGEMM_Geometry::K;
    using inputArrayT       = float(&)[][Di];
    using outputArrayT      = float(&)[][Do];
    using filterT = float(&)[IGEMM_Geometry::T][IGEMM_Geometry::R][IGEMM_Geometry::S][Do][Di];

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    { // Scope ensures PointsToGrid/MergeGrids/GridHandle destructors run while stream is alive
        nanovdb::util::cuda::Timer gpuTimer(stream);

        gpuTimer.start("Building input grid");
        auto inputBuffer =
            BufferT::create(inputPoints.size() * sizeof(nanovdb::Coord), nullptr, 0, stream);
        cudaCheck(cudaMemcpyAsync(inputBuffer.deviceData(),
                                  inputPoints.data(),
                                  inputPoints.size() * sizeof(nanovdb::Coord),
                                  cudaMemcpyHostToDevice,
                                  stream));
        cudaCheck(cudaStreamSynchronize(stream));
        nanovdb::tools::cuda::PointsToGrid<BuildT> converter(1.0, nanovdb::Vec3d(0.0), stream);
        converter.setChecksum(nanovdb::CheckMode::Default);
        auto inputHandle = converter.getHandle<nanovdb::Coord *, BufferT>(
            static_cast<nanovdb::Coord *>(inputBuffer.deviceData()), inputPoints.size());
        auto inputGridDev = inputHandle.deviceGrid<BuildT>();
        gpuTimer.stop();

        std::cout << "Input Grid Diagnostics:" << std::endl;
        printGridDiagnostics(inputHandle);

        gpuTimer.start("Building output grid");
        auto outputBuffer =
            BufferT::create(outputPoints.size() * sizeof(nanovdb::Coord), nullptr, 0, stream);
        cudaCheck(cudaMemcpyAsync(outputBuffer.deviceData(),
                                  outputPoints.data(),
                                  outputPoints.size() * sizeof(nanovdb::Coord),
                                  cudaMemcpyHostToDevice,
                                  stream));
        cudaCheck(cudaStreamSynchronize(stream));
        converter.setChecksum(nanovdb::CheckMode::Default);
        auto outputHandle = converter.getHandle<nanovdb::Coord *, BufferT>(
            static_cast<nanovdb::Coord *>(outputBuffer.deviceData()), outputPoints.size());
        auto outputGridDev = outputHandle.deviceGrid<BuildT>();
        gpuTimer.stop();

        std::cout << "Output Grid Diagnostics:" << std::endl;
        printGridDiagnostics(outputHandle);

        // Download grids to host for CPU-side index building
        inputHandle.deviceDownload(0, stream, false);
        outputHandle.deviceDownload(0, stream, false);
        cudaCheck(cudaStreamSynchronize(stream));
        auto inputGridHost  = inputHandle.grid<BuildT>();
        auto outputGridHost = outputHandle.grid<BuildT>();

        gpuTimer.start("Merging input/output grids (for testing)");
        nanovdb::tools::cuda::MergeGrids<BuildT> merger(inputGridDev, outputGridDev, stream);
        merger.setChecksum(nanovdb::CheckMode::Default);
        merger.setVerbose(0);
        auto mergedHandle = merger.getHandle();
        gpuTimer.stop();

        std::cout << "Merged Grid Diagnostics:" << std::endl;
        printGridDiagnostics(mergedHandle);

        // Allocate and initialize benchmark data

        std::random_device rd;
        // std::mt19937 generator(rd());
        std::mt19937 generator(23456);
        std::uniform_int_distribution<int> distribution(-256, 256);

        gpuTimer.start("Initializing input (activation) data");
        auto inputValueCount =
            nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(inputGridDev);
        auto inputVoxelCount =
            nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(inputGridDev);
        std::vector<float> h_inputData(inputValueCount * Di);
        auto inputArray = reinterpret_cast<inputArrayT>(*h_inputData.data());
        for (int i = 0; i < Di; i++)
            inputArray[0][i] = 0.f;
#pragma omp parallel for
        for (size_t v = 0; v <= inputVoxelCount; v++)
            for (int i = 0; i < Di; i++)
                inputArray[v][i] = ((float)distribution(generator)) /
                                   256.0f; // Use only up to 7 bits in the mantissa
        float *d_inputData = nullptr;
        cudaCheck(cudaMalloc(&d_inputData, h_inputData.size() * sizeof(float)));
        cudaCheck(cudaMemcpyAsync(d_inputData,
                                  h_inputData.data(),
                                  h_inputData.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
        gpuTimer.stop();

        gpuTimer.start("Initializing output (including reference) data");
        auto outputValueCount =
            nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(outputGridDev);
        auto outputVoxelCount =
            nanovdb::util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(outputGridDev);
        std::vector<float> h_outputData(outputValueCount * Do);
        auto outputArray = reinterpret_cast<outputArrayT>(*h_outputData.data());
        std::vector<float> h_outputReferenceData(outputValueCount * Do);
        auto outputReferenceArray = reinterpret_cast<outputArrayT>(*h_outputReferenceData.data());
#pragma omp parallel for
        for (size_t v = 1; v <= inputValueCount; v++)
            for (int i = 0; i < Di; i++)
                outputArray[v][i] = outputReferenceArray[v][i] = 0.f;
        for (int i = 0; i < Di; i++)
            outputArray[0][i] = outputReferenceArray[0][i] =
                ((float)distribution(generator)) / 256.0f; // Use only up to 7 bits in the mantissa
        float *d_outputData = nullptr;
        cudaCheck(cudaMalloc(&d_outputData, h_outputData.size() * sizeof(float)));
        cudaCheck(cudaMemcpyAsync(d_outputData,
                                  h_outputData.data(),
                                  h_outputData.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
        float *d_outputReferenceData = nullptr;
        cudaCheck(cudaMalloc(&d_outputReferenceData, h_outputReferenceData.size() * sizeof(float)));
        cudaCheck(cudaMemcpyAsync(d_outputReferenceData,
                                  h_outputReferenceData.data(),
                                  h_outputReferenceData.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
        gpuTimer.stop();

        gpuTimer.start("Initializing filter data");
        const size_t filterElemCount = 3 * 3 * 3 * Do * Di;
        std::vector<float> h_filterData(filterElemCount);
        auto filter = reinterpret_cast<filterT>(*h_filterData.data());
#pragma omp parallel for
        for (size_t i = 0; i < filterElemCount; i++)
            h_filterData[i] =
                ((float)distribution(generator)) / 256.0f; // Use only up to 7 bits in the mantissa
        float *d_filterData = nullptr;
        cudaCheck(cudaMalloc(&d_filterData, filterElemCount * sizeof(float)));
        cudaCheck(cudaMemcpyAsync(d_filterData,
                                  h_filterData.data(),
                                  filterElemCount * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
        gpuTimer.stop();

        gpuTimer.start("Initializing scatter indices");
        auto outputLeafCount = outputGridHost->tree().nodeCount(0);
        auto blockCount =
            outputLeafCount * IGEMM_Geometry::Bx * IGEMM_Geometry::By * IGEMM_Geometry::Bz;

        using ConvOp = AmperePredicatedFprop<IGEMM_Geometry>;
#ifdef USE_HIERARCHICAL_BLOCK_TRAVERSAL
        auto leafShape = make_shape(
            Int<IGEMM_Geometry::Bx>{}, Int<IGEMM_Geometry::By>{}, Int<IGEMM_Geometry::Bz>{});
        auto blockedLeafShape  = shape(zipped_divide(make_layout(leafShape), ConvOp::Tiler_N{}));
        auto blockedLeafLayout = make_ordered_layout(
            blockedLeafShape,
            make_tuple(make_tuple(_2{}, _1{}, _0{}), make_tuple(_5{}, _4{}, _3{})));
#endif

        auto outputVoxelsPerBlock = IGEMM_Geometry::Z * IGEMM_Geometry::P * IGEMM_Geometry::Q;
        using ScatterIndexLegacyT =
            uint64_t[IGEMM_Geometry::Z][IGEMM_Geometry::P][IGEMM_Geometry::Q];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
        using ScatterIndexArrayLegacyT =
            ScatterIndexLegacyT[IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
#else
        using ScatterIndexArrayLegacyT =
            ScatterIndexLegacyT[IGEMM_Geometry::Bx * IGEMM_Geometry::By * IGEMM_Geometry::Bz];
#endif
        std::vector<uint64_t> scatterIndexDataLegacy(blockCount * outputVoxelsPerBlock);
        auto scatterIndexArrayLegacy =
            reinterpret_cast<ScatterIndexArrayLegacyT *>(scatterIndexDataLegacy.data());

        using ScatterIndexArrayT = uint64_t[8][8][8];
        std::vector<uint64_t> scatterIndexData(outputLeafCount * 512);
        auto scatterIndexArray = reinterpret_cast<ScatterIndexArrayT *>(scatterIndexData.data());

#pragma omp parallel for
        for (uint32_t l = 0; l < outputLeafCount; l++) {
            auto &leaf = outputGridHost->tree().getFirstLeaf()[l];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
            for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
                for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
                    for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
#else
            for (int bbi = 0; bbi < shape<1, 0>(blockedLeafLayout); ++bbi)
                for (int bbj = 0; bbj < shape<1, 1>(blockedLeafLayout); ++bbj)
                    for (int bbk = 0; bbk < shape<1, 2>(blockedLeafLayout); ++bbk)
                        for (int bii = 0; bii < shape<0, 0>(blockedLeafLayout); ++bii)
                            for (int bjj = 0; bjj < shape<0, 1>(blockedLeafLayout); ++bjj)
                                for (int bkk = 0; bkk < shape<0, 2>(blockedLeafLayout); ++bkk) {
                                    int bi = bbi * shape<0, 0>(blockedLeafLayout) + bii;
                                    int bj = bbj * shape<0, 1>(blockedLeafLayout) + bjj;
                                    int bk = bbk * shape<0, 2>(blockedLeafLayout) + bkk;
#endif
                        nanovdb::Coord blockOffset(
                            bi * IGEMM_Geometry::Z, bj * IGEMM_Geometry::P, bk * IGEMM_Geometry::Q);
                        for (int i = 0; i < IGEMM_Geometry::Z; i++)
                            for (int j = 0; j < IGEMM_Geometry::P; j++)
                                for (int k = 0; k < IGEMM_Geometry::Q; k++) {
                                    auto localCoord = blockOffset.offsetBy(i, j, k);
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
                                    scatterIndexArrayLegacy[l][bi][bj][bk][i][j][k] =
                                        leaf.getValue(localCoord);
#else
                                                scatterIndexArrayLegacy[l][blockedLeafLayout(
                                                    make_tuple(bii, bjj, bkk),
                                                    make_tuple(bbi, bbj, bbk))][i][j][k] =
                                                    leaf.getValue(localCoord);
#endif
                                }
                    }
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    for (int k = 0; k < 8; k++) {
                        scatterIndexArray[l][i][j][k] = leaf.getValue(nanovdb::Coord(i, j, k));
                    }
        }
        gpuTimer.stop();

        gpuTimer.start("Initializing gather indices");
        auto inputVoxelsPerBlock = IGEMM_Geometry::D * IGEMM_Geometry::H * IGEMM_Geometry::W;
        using GatherIndexLegacyT =
            uint64_t[IGEMM_Geometry::D][IGEMM_Geometry::H][IGEMM_Geometry::W];
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
        using GatherIndexArrayLegacyT =
            GatherIndexLegacyT[IGEMM_Geometry::Bx][IGEMM_Geometry::By][IGEMM_Geometry::Bz];
#else
        using GatherIndexArrayLegacyT =
            GatherIndexLegacyT[IGEMM_Geometry::Bx * IGEMM_Geometry::By * IGEMM_Geometry::Bz];
#endif
        std::vector<uint64_t> gatherIndexDataLegacy(blockCount * inputVoxelsPerBlock);
        auto gatherIndexArrayLegacy =
            reinterpret_cast<GatherIndexArrayLegacyT *>(gatherIndexDataLegacy.data());

        using GatherIndexArrayT =
            uint64_t[IGEMM_Geometry::Hx][IGEMM_Geometry::Hy][IGEMM_Geometry::Hz];
        std::vector<uint64_t> gatherIndexData(outputLeafCount * IGEMM_Geometry::Hx *
                                              IGEMM_Geometry::Hy * IGEMM_Geometry::Hz);
        auto gatherIndexArray = reinterpret_cast<GatherIndexArrayT *>(gatherIndexData.data());

#pragma omp parallel for
        for (uint32_t l = 0; l < outputLeafCount; l++) {
            auto &outputLeaf  = outputGridHost->tree().getFirstLeaf()[l];
            const auto origin = outputLeaf.origin();
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
            for (int bi = 0; bi < IGEMM_Geometry::Bx; bi++)
                for (int bj = 0; bj < IGEMM_Geometry::By; bj++)
                    for (int bk = 0; bk < IGEMM_Geometry::Bz; bk++) {
#else
            for (int bbi = 0; bbi < shape<1, 0>(blockedLeafLayout); ++bbi)
                for (int bbj = 0; bbj < shape<1, 1>(blockedLeafLayout); ++bbj)
                    for (int bbk = 0; bbk < shape<1, 2>(blockedLeafLayout); ++bbk)
                        for (int bii = 0; bii < shape<0, 0>(blockedLeafLayout); ++bii)
                            for (int bjj = 0; bjj < shape<0, 1>(blockedLeafLayout); ++bjj)
                                for (int bkk = 0; bkk < shape<0, 2>(blockedLeafLayout); ++bkk) {
                                    int bi = bbi * shape<0, 0>(blockedLeafLayout) + bii;
                                    int bj = bbj * shape<0, 1>(blockedLeafLayout) + bjj;
                                    int bk = bbk * shape<0, 2>(blockedLeafLayout) + bkk;
#endif
                        nanovdb::Coord blockOffset(
                            bi * IGEMM_Geometry::Z, bj * IGEMM_Geometry::P, bk * IGEMM_Geometry::Q);
                        for (int i = 0; i < IGEMM_Geometry::D; i++)
                            for (int j = 0; j < IGEMM_Geometry::H; j++)
                                for (int k = 0; k < IGEMM_Geometry::W; k++) {
                                    auto localCoord  = blockOffset.offsetBy(i + IGEMM_Geometry::Dx,
                                                                           j + IGEMM_Geometry::Dy,
                                                                           k + IGEMM_Geometry::Dz);
                                    auto globalCoord = origin + localCoord;
#ifndef USE_HIERARCHICAL_BLOCK_TRAVERSAL
                                    gatherIndexArrayLegacy[l][bi][bj][bk][i][j][k] =
                                        inputGridHost->tree().getValue(globalCoord);
#else
                                                gatherIndexArrayLegacy[l][blockedLeafLayout(
                                                    make_tuple(bii, bjj, bkk),
                                                    make_tuple(bbi, bbj, bbk))][i][j][k] =
                                                    inputGridHost->tree().getValue(globalCoord);
#endif
                                }
                    }

            auto offsetOrigin =
                origin.offsetBy(IGEMM_Geometry::Dx, IGEMM_Geometry::Dy, IGEMM_Geometry::Dz);
            for (int i = 0; i < IGEMM_Geometry::Hx; ++i)
                for (int j = 0; j < IGEMM_Geometry::Hy; ++j)
                    for (int k = 0; k < IGEMM_Geometry::Hz; ++k)
                        gatherIndexArray[l][i][j][k] =
                            inputGridHost->tree().getValue(offsetOrigin + nanovdb::Coord(i, j, k));
        }
        gpuTimer.stop();

        cudaCheck(cudaStreamSynchronize(stream));

        gpuTimer.start("Reference (GPU) execution");
        SparseConvolveCudaReference<IGEMM_Geometry, Di, Do>(
            outputLeafCount,
            inputGridDev,
            outputGridDev,
            reinterpret_cast<const float(*)[IGEMM_Geometry::R][IGEMM_Geometry::S][Do][Di]>(
                d_filterData),
            reinterpret_cast<const float(*)[Di]>(d_inputData),
            reinterpret_cast<float(*)[Do]>(d_outputReferenceData),
            stream);
        gpuTimer.stop();

#if 0
    // CPU version; may be extremely slow for all but the smallest resolutions
    gpuTimer.start("Reference (CPU) execution");
    SparseConvolveCPUReference<IGEMM_Geometry, Di, Do>(
        inputGridHost,
        outputGridHost,
        filter,
        inputArray,
        outputArray
    );
    gpuTimer.stop();
#endif

#if 0
    // Scatter-gather reference requires device copies of index arrays (host-only above)
    gpuTimer.start("Reference (Gather-Scatter) execution");
    SparseConvolveScatterGatherMapsReference<IGEMM_Geometry, Di, Do>(
        reinterpret_cast<GatherIndexLegacyT*>(gatherIndexArrayLegacy),
        reinterpret_cast<ScatterIndexLegacyT*>(scatterIndexArrayLegacy),
        blockCount,
        filter,
        inputArray,
        outputArray
    );
    gpuTimer.stop();

    ResultCompare<Do>(
        outputValueCount,
        outputArray,
        outputReferenceArray
    );
#endif

        IGEMM_Layouts<IGEMM_Geometry> layouts;

        Tensor tFilter = make_tensor(make_gmem_ptr(d_filterData), layouts.filterLayout());

        constexpr size_t smem_size =
            sizeof(typename AmperePredicatedFprop<IGEMM_Geometry>::SharedStorage);
        std::cout << "smem_size = " << smem_size << std::endl;

        cudaCheck(
            cudaFuncSetAttribute(kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>,
                                                          BuildT,
                                                          decltype(tFilter)>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size));

        int num_iterations = 10;
        for (int i = 0; i < num_iterations; ++i) {
            gpuTimer.start("Scatter-Gather Cutlass IGEMM (GPU) execution");
            kernel_entrypoint_custom<AmperePredicatedFprop<IGEMM_Geometry>,
                                     BuildT,
                                     decltype(tFilter)>
                <<<outputLeafCount,
                   AmperePredicatedFprop<IGEMM_Geometry>::MaxThreadsPerBlock,
                   smem_size,
                   stream>>>(tFilter, inputGridDev, outputGridDev, d_inputData, d_outputData);
            gpuTimer.stop();
        }

        cudaCheck(cudaStreamSynchronize(stream));
        cudaCheck(cudaMemcpy(h_outputData.data(),
                             d_outputData,
                             h_outputData.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_outputReferenceData.data(),
                             d_outputReferenceData,
                             h_outputReferenceData.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

        ResultCompare<Do>(outputValueCount, outputArray, outputReferenceArray);

        cudaCheck(cudaFree(d_inputData));
        cudaCheck(cudaFree(d_outputData));
        cudaCheck(cudaFree(d_outputReferenceData));
        cudaCheck(cudaFree(d_filterData));
    } // CUDA objects destroyed here, stream still alive
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaStreamDestroy(stream));
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// SPARSE_CONVOLUTION_IGEMM_NANOVDB_CUDA.CPP
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

// the following files are from OpenVDB
// #include <openvdb/tools/Morphology.h>
// #include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
// #include <nanovdb/cuda/DeviceBuffer.h>  (included above)
// #include <nanovdb/tools/CreateNanoGrid.h>

void mainSparseConvolutionIGEMM(const std::vector<nanovdb::Coord> &inputPoints,
                                const std::vector<nanovdb::Coord> &outputPoints,
                                uint32_t benchmark_iters);

uint32_t
coordinate_bitpack(uint32_t x) {
    x &= 0x49249249; // keep only one every 3 bits
    x |= (x >> 2);
    x &= 0xc30c30c3; // Pack into pairs of bits
    x |= (x >> 4);
    x &= 0x0f00f00f; // Pack into quadruples of bits
    x |= (x >> 8);
    x &= 0xff0000ff; // Pack into quadruples of bits
    x |= (x >> 16);
    x &= 0x0000ffff; // Pack into 16-tuples (actually, 11 max) of bits
    return x;
}

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int
test_sparse_convolution_igemm_nanovdb_cuda(int benchmark_iters = 10) {
    std::random_device rd;
    // std::mt19937 generator(rd());
    std::mt19937 generator(12345);

    // static const int ambient_voxels = 16*1024;
    static const int ambient_voxels     = 1024 * 1024 * 2;
    static const float input_occupancy  = .75f;
    static const float output_occupancy = .75f;
    static const float overlap          = .65f;
    nanovdb::Coord offset(0, 0, 0);

    // Mark input voxels at requested occupancy
    int target_input_voxels = (int)(input_occupancy * (float)ambient_voxels);
    std::vector<bool> voxmap(ambient_voxels);
    int active_voxels = 0;
    std::uniform_int_distribution<int> distribution(0, ambient_voxels - 1);
    while (active_voxels < target_input_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }

    // Convert to coordinates
    std::vector<nanovdb::Coord> inputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if (voxmap[i]) {
            int x = coordinate_bitpack(i & 0x49249249);
            int y = coordinate_bitpack((i >> 1) & 0x49249249);
            int z = coordinate_bitpack((i >> 2) & 0x49249249);
            inputPoints.emplace_back(nanovdb::Coord(x, y, z) + offset);
        }
    std::cout << inputPoints.size() << " input voxels generated" << std::endl;

    // Discard voxels until desired level of overlap
    int target_overlap_voxels = (int)(overlap * (float)ambient_voxels);
    while (active_voxels > target_overlap_voxels) {
        int i = distribution(generator);
        if (voxmap[i]) {
            voxmap[i] = false;
            active_voxels--;
        }
    }
    // Then sample more voxels until desired occupancy is met
    int target_output_voxels = (int)(output_occupancy * (float)ambient_voxels);
    while (active_voxels < target_output_voxels) {
        int i = distribution(generator);
        if (!voxmap[i]) {
            voxmap[i] = true;
            active_voxels++;
        }
    }
    // Convert to coordinates
    std::vector<nanovdb::Coord> outputPoints;
    for (int i = 0; i < ambient_voxels; i++)
        if (voxmap[i]) {
            int x = coordinate_bitpack(i & 0x49249249);
            int y = coordinate_bitpack((i >> 1) & 0x49249249);
            int z = coordinate_bitpack((i >> 2) & 0x49249249);
            outputPoints.emplace_back(nanovdb::Coord(x, y, z) + offset);
        }
    std::cout << outputPoints.size() << " output voxels generated" << std::endl;

    mainSparseConvolutionIGEMM(inputPoints, outputPoints, benchmark_iters);

    return 0;
}

void
sifakisRefSparseConv(uint32_t outputLeafCount,
                     const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                     const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                     const float *d_filterData,
                     const float *d_inputData,
                     float *d_outputData,
                     cudaStream_t stream) {
    using G = IGEMM_Geometry;
    SparseConvolveCudaReference<G, G::C, G::K>(
        outputLeafCount,
        const_cast<nanovdb::NanoGrid<nanovdb::ValueOnIndex> *>(inputGrid),
        const_cast<nanovdb::NanoGrid<nanovdb::ValueOnIndex> *>(outputGrid),
        reinterpret_cast<const float(*)[G::R][G::S][G::K][G::C]>(d_filterData),
        reinterpret_cast<const float(*)[G::C]>(d_inputData),
        reinterpret_cast<float(*)[G::K]>(d_outputData),
        stream);
}

void
sifakisIGemmConv(uint32_t outputLeafCount,
                 const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                 const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                 const float *d_filterData,
                 const float *d_inputData,
                 float *d_outputData,
                 cudaStream_t stream) {
    using BuildT = nanovdb::ValueOnIndex;
    using ConvOp = AmperePredicatedFprop<IGEMM_Geometry>;

    IGEMM_Layouts<IGEMM_Geometry> layouts;
    auto tFilter = make_tensor(make_gmem_ptr(d_filterData), layouts.filterLayout());

    constexpr size_t smem_size = sizeof(typename ConvOp::SharedStorage);
    cudaFuncSetAttribute(kernel_entrypoint_custom<ConvOp, BuildT, decltype(tFilter)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    kernel_entrypoint_custom<ConvOp, BuildT, decltype(tFilter)>
        <<<outputLeafCount, ConvOp::MaxThreadsPerBlock, smem_size, stream>>>(
            tFilter, inputGrid, outputGrid, d_inputData, d_outputData);
}
