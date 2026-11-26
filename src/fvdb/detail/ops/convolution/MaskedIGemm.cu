// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// MaskedIGemm.cu -- Masked implicit-GEMM sparse convolution.
//
// Kernels use SM80 warp-level tensor-core instructions (cp.async, ldmatrix,
// mma.sync m16n8k16), which also run on SM89/SM90/SM100/SM120.
//
// Forward and input gradient share one output-stationary kernel:
//   out[o, n] = sum_{u in taps(tile)} sum_k A[nbr[o, u], k] * B_u[k, n]
// Forward reads A = features and B_u = W[u] (C_in x C_out). The input gradient
// reads A = grad_output over the reverse map (input rows to the output rows that
// read them) with B_u = W[u]^T. A submanifold map (same grid, stride 1, odd
// kernel) is its own reverse with mirrored taps, since nbr[o, u] = i iff
// nbr[i, K-1-u] = o, so it reads the forward map with B_u = W[K-1-u]^T.
//
// The weight gradient reduces over each tap's active (input, output) pairs:
//   dW[u] = sum_p features[pairIn[p]]^T grad_output[pairOut[p]]
//

#include "dispatch/detail/core_types.h"
#include "dispatch/torch/types.h"

#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/dispatch/ForEachActiveVoxel.cuh>
#include <fvdb/detail/dispatch/GridAccessor.h>
#include <fvdb/detail/ops/convolution/ConvolutionGeometry.h>
#include <fvdb/detail/ops/convolution/MaskedIGemm.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

// CuTe after torch/types.h, as in PredGatherIGemm.cu, for CCCL include order.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Set only by tests (setMaskedIGemmTestOverrides); unset fields leave the heuristics in charge.
constinit MaskedIGemmTestOverrides gTestOverrides;

// Element types the tensor-core kernels run.
template <typename T>
concept MmaElement =
    std::same_as<T, __half> || std::same_as<T, __nv_bfloat16> || std::same_as<T, float>;

// Tap masks hold kMaskWordBits taps per 32-bit word.
constexpr int kMaskWordBits = 32;

__host__ __device__ constexpr int
maskWordsFor(int64_t taps) {
    return int((taps + kMaskWordBits - 1) / kMaskWordBits);
}

__host__ __device__ __forceinline__ uint32_t
bitsBelow(int b) {
    return (1u << b) - 1u;
}

// Grid-stride kernels launch blocks of this many threads (see gridStrideBlocks).
constexpr int kGridStrideThreads = 256;

// cp.async copies 16-byte chunks, so the kernels need 16-byte-aligned operands. A contiguous view
// can still start at any element (a slice of a flat buffer, say); those get fresh storage.
torch::Tensor
aligned16(torch::Tensor t) {
    t = t.contiguous();
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 ? t : t.clone();
}

// =============================================================================
// Weight rows
// =============================================================================
//
// Weights arrive as a (C_out, C_in, k0, k1, k2) view. nn.SparseConv3d stores
// them as (k2, k1, k0, C_in, C_out), so the (K, C_in, C_out) row view is free
// for module weights. Topology taps run k0 slowest; weight rows run k2 slowest.

torch::Tensor
weightRows(torch::Tensor const &weights) {
    int64_t const K = weights.size(2) * weights.size(3) * weights.size(4);
    return weights.permute({4, 3, 2, 1, 0})
        .reshape({K, weights.size(1), weights.size(0)})
        .contiguous();
}

// The weight gradient in the weights' own layout: a free (C_out, C_in, k0, k1, k2) view of the
// rows for module weights, which are stored as (k2, k1, k0, C_in, C_out), and one copy into a
// contiguous tensor for any other layout, so callers can flatten it.
torch::Tensor
weightGradFromRows(torch::Tensor const &rows, torch::Tensor const &weights) {
    auto const grad =
        rows.view({weights.size(4), weights.size(3), weights.size(2), rows.size(1), rows.size(2)})
            .permute({4, 3, 2, 1, 0});
    return weights.permute({4, 3, 2, 1, 0}).is_contiguous() ? grad : grad.contiguous();
}

// flip maps tap k to the weights of tap K-1-k, its mirror image. A submanifold transposed
// convolution is the forward convolution with every tap flipped.
struct TapToWeightRow {
    int k1, k2, k0;
    bool flip;

    __host__ __device__ int
    operator()(int tap) const {
        if (flip)
            tap = k0 * k1 * k2 - 1 - tap;
        int const t0 = tap / (k1 * k2);
        int const t1 = (tap / k2) % k1;
        int const t2 = tap % k2;
        return (t2 * k1 + t1) * k0 + t0;
    }
};

// =============================================================================
// Tensor-core and async-copy instructions (CuTe arch atoms)
// =============================================================================
//
// SM80 instructions. Architectures below SM80 compile CuTe's invalid-path stubs; the host never
// launches these kernels there (maskedIGemmSupports checks the device).

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define FVDB_MIGEMM_SM80 1
#else
#define FVDB_MIGEMM_SM80 0
#endif

// 16-byte async copy; zero-fills the destination when pred is false.
__device__ __forceinline__ void
cpAsync16(void *dst, void const *src, bool pred) {
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *static_cast<cute::uint128_t const *>(src), *static_cast<cute::uint128_t *>(dst), pred);
}

// Four 8x8 16-bit tiles from shared memory, plain or transposed.
__device__ __forceinline__ void
ldmatrixX4(uint32_t (&r)[4], void const *src) {
    cute::SM75_U32x4_LDSM_N::copy(
        *static_cast<cute::uint128_t const *>(src), r[0], r[1], r[2], r[3]);
}

__device__ __forceinline__ void
ldmatrixX4Trans(uint32_t (&r)[4], void const *src) {
    cute::SM75_U16x8_LDSM_T::copy(
        *static_cast<cute::uint128_t const *>(src), r[0], r[1], r[2], r[3]);
}

// c += a * b on one m16n8 tile. float32 runs as TF32 (m16n8k8) with operands rounded after loading.
template <typename T> struct MmaAtom;
template <> struct MmaAtom<__half> {
    using type = cute::SM80_16x8x16_F32F16F16F32_TN;
};
template <> struct MmaAtom<__nv_bfloat16> {
    using type = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};
template <> struct MmaAtom<float> {
    using type = cute::SM80_16x8x8_F32TF32TF32F32_TN;
};

template <typename T>
__device__ __forceinline__ void
mma(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
    MmaAtom<T>::type::fma(
        c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3]);
}

// Per-element-type tile constants. Every k-tile row is 64 bytes (four 16-byte
// chunks), so shared-memory layouts and ldmatrix addressing match across types,
// and each k-tile holds two MMA k-steps.
template <MmaElement T> struct Elem {
    static constexpr int CHUNK = 16 / int(sizeof(T)); // elements per 16-byte chunk
    static constexpr int BK    = 4 * CHUNK;           // k-tile width in elements
    static constexpr int MMA_K = BK / 2;              // MMA k per instruction
};

// Round a loaded fragment register to TF32 (no-op for 16-bit types). Adding half an ulp lets
// the tensor core's truncation of the low 13 bits round to nearest, ties away from zero, as
// cvt.rna.tf32.f32 does, for less issue cost (CUTLASS's round_half_ulp_truncate). Inf and NaN
// skip the add, which would carry a NaN payload into the sign bit.
template <typename T>
__device__ __forceinline__ uint32_t
roundToTf32(uint32_t x) {
    if constexpr (sizeof(T) == 4) {
        if (isfinite(__uint_as_float(x)))
            x += 0x1000u;
    }
    return x;
}

// Fragments for 3xTF32 stay unrounded until mmaTiles splits them.
template <typename T, bool Use3xTf32>
__device__ __forceinline__ uint32_t
roundUnless3xTf32(uint32_t x) {
    if constexpr (Use3xTf32)
        return x;
    else
        return roundToTf32<T>(x);
}

// Magnitudes from here up round to inf in TF32.
constexpr uint32_t kTf32RoundingOverflow = 0x7f7ff000u;

// Split a float32 into TF32 parts: big is the value rounded to TF32 and small the rounded
// remainder, so big + small keeps about 22 of float32's 24 mantissa bits. Inf, NaN and the
// finite values that would round to inf pass through whole with no remainder: the tensor core
// truncates them, so those few values near FLT_MAX get TF32 precision. Folding them into one
// compare keeps the split's cost; testing for them separately slowed 3xTF32 layers by ~25%.
__device__ __forceinline__ void
splitTf32(uint32_t x, uint32_t &big, uint32_t &small) {
    float const f = __uint_as_float(x);
    if (!(fabsf(f) < __uint_as_float(kTf32RoundingOverflow))) {
        big   = x;
        small = 0u;
        return;
    }
    big   = (x + 0x1000u) & 0xffffe000u;
    small = roundToTf32<float>(__float_as_uint(f - __uint_as_float(big)));
}

// acc += A * B over a warp's MT x NT fragments. Use3xTf32 (float32 only) computes 3xTF32:
// small * big, big * small, then big * big, which carries ~22 mantissa bits through TF32
// tensor cores. The small * small term is below float32 rounding and is dropped. The tensor
// core truncates while adding into its accumulator, so the products go into zeroed temporaries
// and reach acc through a rounding float32 add; otherwise the truncation, relative to the whole
// running sum, dominates the error. The main product stays separate from the cross terms: when
// it is infinite (an infinite operand), a cross term can be inf * 0 = NaN, and IEEE float32
// would give the infinity, so a non-finite main product is added alone.
template <typename T, bool Use3xTf32, int MT, int NT>
__device__ __forceinline__ void
mmaTiles(float (&acc)[MT][NT][4], uint32_t (&af)[MT][4], uint32_t (&bf)[NT][2]);

template <typename T> __device__ __forceinline__ uint32_t packPair(float x, float y);

template <>
__device__ __forceinline__ uint32_t
packPair<__half>(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
}

template <>
__device__ __forceinline__ uint32_t
packPair<__nv_bfloat16>(float x, float y) {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
}

// Store two adjacent output elements.
template <typename T>
__device__ __forceinline__ void
storePair(T *dst, float x, float y) {
    if constexpr (sizeof(T) == 4)
        *reinterpret_cast<float2 *>(dst) = make_float2(x, y);
    else
        *reinterpret_cast<uint32_t *>(dst) = packPair<T>(x, y);
}

__device__ __forceinline__ float
toFloat(float x) {
    return x;
}

__device__ __forceinline__ float
toFloat(__half x) {
    return __half2float(x);
}

__device__ __forceinline__ float
toFloat(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Shared-memory swizzles, in units of 16-byte chunks.
//   Rows of 4 chunks (32 16-bit or 16 float32 elements): XOR with bits 1-2 of the row.
//   Rows of 8+ chunks: XOR the low 3 chunk bits with the row's low 3 bits.
__device__ __forceinline__ int
swizzle4(int row, int chunk) {
    return chunk ^ ((row >> 1) & 3);
}

__device__ __forceinline__ int
swizzle8(int row, int chunk) {
    return chunk ^ (row & 7);
}

// [k][m|n] tiles. 16-bit types read them with ldmatrix.trans and use swizzle8. float32 reads
// them element-wise, with lane (g, t) reading rows t and t + 4; XOR with bits 0-1 of the row,
// shifted past the chunk's low bit, so those 32 lanes hit distinct banks.
template <typename T>
__device__ __forceinline__ int
swizzleKN(int row, int chunk) {
    if constexpr (sizeof(T) == 4)
        return chunk ^ ((row & 3) << 1);
    else
        return swizzle8(row, chunk);
}

// [k][m|n] tiles W elements wide. 16-bit rows of an odd multiple of 4 chunks (W = 32, 96, ...)
// alternate between the two halves of the 128-byte bank window, so swizzle4 keeps 8 consecutive
// rows on distinct banks; rows of a multiple of 8 chunks use swizzleKN.
template <typename T, int W>
__device__ __forceinline__ int
swizzleRow(int row, int chunk) {
    constexpr int chunks = W * int(sizeof(T)) / 16;
    static_assert(chunks % 4 == 0, "rows must span a multiple of 4 chunks");
    static_assert(sizeof(T) != 4 || chunks % 8 == 0,
                  "float32 rows must span a multiple of 8 chunks");
    if constexpr (sizeof(T) == 2 && chunks % 8 != 0)
        return swizzle4(row, chunk);
    else
        return swizzleKN<T>(row, chunk);
}

template <typename T, bool Use3xTf32, int MT, int NT>
__device__ __forceinline__ void
mmaTiles(float (&acc)[MT][NT][4], uint32_t (&af)[MT][4], uint32_t (&bf)[NT][2]) {
    if constexpr (Use3xTf32) {
        static_assert(sizeof(T) == 4, "3xTF32 applies to float32");
        uint32_t as[MT][4], bs[NT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                splitTf32(af[mt][e], af[mt][e], as[mt][e]);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                splitTf32(bf[nt][h], bf[nt][h], bs[nt][h]);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float cross[4] = {0.f, 0.f, 0.f, 0.f}, main[4] = {0.f, 0.f, 0.f, 0.f};
                mma<T>(cross, as[mt], bf[nt][0], bf[nt][1]);
                mma<T>(cross, af[mt], bs[nt][0], bs[nt][1]);
                mma<T>(main, af[mt], bf[nt][0], bf[nt][1]);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[mt][nt][e] += isfinite(main[e]) ? main[e] + cross[e] : main[e];
            }
    } else {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                mma<T>(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
    }
}

// =============================================================================
// Output-stationary masked implicit GEMM (forward and input gradient)
// =============================================================================

constexpr int kMaxTaps = kMaskWordBits; // one mask word per row
// Large kernels (more than kMaxTaps taps) stage the neighbor columns of their taps in groups of
// kStagedTaps.
constexpr int kStagedTaps = 32;

template <int BM_, int BN_, int WM_, int WN_, int STAGES_> struct TileConfig {
    static constexpr int BM      = BM_;
    static constexpr int BN      = BN_;
    static constexpr int WM      = WM_;
    static constexpr int WN      = WN_;
    static constexpr int STAGES  = STAGES_;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT      = BM / WM / 16; // m16 tiles per warp
    static constexpr int NT      = BN / WN / 8;  // n8 tiles per warp
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 8) == 0, "warp tiles must cover the CTA tile");
    static_assert(NT % 2 == 0, "B fragments load two n8 tiles at a time");
    static_assert(BN >= 32, "B rows must span at least 4 chunks");
};

template <typename Cfg, typename T, bool LargeKernel = false>
constexpr int
smemBytes(int K) {
    int const tiles = Cfg::STAGES * (Cfg::BM + Cfg::BN) * Elem<T>::BK * int(sizeof(T));
    if constexpr (LargeKernel)
        return tiles + Cfg::BM * kStagedTaps * 4 + Cfg::BM * 4 + 2 * K * 4 + maskWordsFor(K) * 4 +
               16;
    else
        return tiles + Cfg::BM * K * 4 + Cfg::BM * 4 + 2 * kMaxTaps * 4 + 16;
}

// Channel counts are multiples of 8, so every 16-byte chunk lies wholly inside or outside the
// channels. The last k-tile and the last n-tile may be partial: loads past Kc or Nc zero-fill
// and stores past Nc are skipped.
//
// KMajorWeights selects the weight operand layout per tap:
//   false (forward):        B[k][n] at W + row * Kc * Nc + k * Nc + n
//   true  (input gradient): B[k][n] at W + row * Kc * Nc + n * Kc + k
// With splits > 1, CTA z of a tile runs its share of the tile's (tap, k-tile)
// iterations and adds float32 partials into outAcc instead of writing out.
//
// LargeKernel runs kernels of more than kMaxTaps taps: neighborMap is compact and rowMaskWords
// holds a (mask, base) pair per 32-tap word (see largeKernelTopology), and the neighbor columns of
// the tile's taps are staged in groups of kStagedTaps as the loader reaches them.
template <typename Cfg, MmaElement T, bool KMajorWeights, bool Use3xTf32, bool LargeKernel = false>
__global__ void __launch_bounds__(Cfg::THREADS)
maskedIGemmKernel(T const *__restrict__ A,
                  T const *__restrict__ W,
                  T const *__restrict__ bias,
                  T *__restrict__ out,
                  float *__restrict__ outAcc,
                  int32_t const *__restrict__ neighborMap,
                  int32_t const *__restrict__ sortedRows,
                  int32_t const *__restrict__ rowMaskWords,
                  int N,
                  int Kc,
                  int Nc,
                  int K,
                  int splits,
                  TapToWeightRow tapToRow) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Elem<T>::BK, STAGES = Cfg::STAGES;
    constexpr int CHUNK_ELEMS = Elem<T>::CHUNK, MMA_K = Elem<T>::MMA_K;
    constexpr int THREADS = Cfg::THREADS, MT = Cfg::MT, NT = Cfg::NT;
    constexpr int A_STAGE = BM * BK, B_STAGE = BK * BN;

    [[maybe_unused]] int const maskWords = maskWordsFor(K); // LargeKernel only
    constexpr int TAP_SLOTS              = LargeKernel ? 0 : kMaxTaps;
    int const tapSlots                   = LargeKernel ? K : TAP_SLOTS;

    extern __shared__ __align__(128) unsigned char smem[];
    T *sA               = reinterpret_cast<T *>(smem);
    T *sB               = sA + STAGES * A_STAGE;
    int32_t *sNbr       = reinterpret_cast<int32_t *>(sB + STAGES * B_STAGE);
    int32_t *sRows      = sNbr + BM * (LargeKernel ? kStagedTaps : K);
    int32_t *sTaps      = sRows + BM;
    int32_t *sWeightRow = sTaps + tapSlots;
    uint32_t *sMask     = reinterpret_cast<uint32_t *>(sWeightRow + tapSlots);
    int32_t *sNTaps     = reinterpret_cast<int32_t *>(sMask + (LargeKernel ? maskWords : 1));

    int const tid    = threadIdx.x;
    int const lane   = tid & 31;
    int const warp   = tid >> 5;
    int const nTiles = (Nc + BN - 1) / BN;
    int const nTile  = blockIdx.x % nTiles;
    int const mTile  = blockIdx.x / nTiles;
    int const row0   = mTile * BM;
    int const n0     = nTile * BN;

    // ---- Stage the tile's rows, tap union, and neighbor columns of the union ----
    // (LargeKernel stages the neighbor columns per tap group, as load() reaches them.)
    if constexpr (LargeKernel) {
        for (int w = tid; w < maskWords; w += THREADS)
            sMask[w] = 0;
    } else if (tid == 0) {
        *sMask = 0;
    }
    __syncthreads();
    for (int r = tid; r < BM; r += THREADS) {
        int const o = (row0 + r < N) ? sortedRows[row0 + r] : -1;
        sRows[r]    = o;
        if constexpr (LargeKernel) {
            for (int w = 0; w < maskWords; ++w) {
                uint32_t m = o >= 0 ? uint32_t(rowMaskWords[(int64_t(o) * maskWords + w) * 2]) : 0u;
#if FVDB_MIGEMM_SM80
                m = __reduce_or_sync(__activemask(), m);
#endif
                if (lane == 0 && m)
                    atomicOr(sMask + w, m);
            }
        } else {
            uint32_t m = o >= 0 ? uint32_t(rowMaskWords[o]) : 0u;
#if FVDB_MIGEMM_SM80
            m = __reduce_or_sync(__activemask(), m);
#endif
            if (lane == 0 && m)
                atomicOr(sMask, m);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int n = 0;
        if constexpr (LargeKernel) {
            for (int w = 0; w < maskWords; ++w) {
                uint32_t m = sMask[w];
                while (m) {
                    int const tap   = 32 * w + __ffs(m) - 1;
                    sTaps[n]        = tap;
                    sWeightRow[n++] = tapToRow(tap);
                    m &= m - 1;
                }
            }
        } else {
            uint32_t m = *sMask;
            while (m) {
                int const tap   = __ffs(m) - 1;
                sTaps[n]        = tap;
                sWeightRow[n++] = tapToRow(tap);
                m &= m - 1;
            }
        }
        *sNTaps = n;
    }
    __syncthreads();
    int const nTaps = *sNTaps;
    if constexpr (!LargeKernel) {
        for (int i = tid; i < BM * nTaps; i += THREADS) {
            int const r = i / nTaps;
            int const o = sRows[r];
            sNbr[i]     = o >= 0 ? neighborMap[int64_t(o) * K + sTaps[i - r * nTaps]] : -1;
        }
        __syncthreads();
    }

    int const kTiles = (Kc + BK - 1) / BK;
    int const total  = nTaps * kTiles;
    int const itBase = int(int64_t(total) * blockIdx.y / splits);
    int const nIter  = int(int64_t(total) * (blockIdx.y + 1) / splits) - itBase;
    if (splits > 1 && nIter <= 0)
        return;

    // ---- Async tile loads ----
    // Each thread copies A_CHUNKS 16-byte chunks of A (rows aRow + i * A_RSTEP, chunk aChunk) and
    // B_CHUNKS of B per stage. The loader walks (tap, k-tile) in order: pointers advance one
    // k-tile per load and are rebuilt from the neighbor columns and weight rows only when the
    // tap changes, so the loop does no division.
    constexpr int A_CHUNKS = BM * 4 / THREADS, A_RSTEP = THREADS / 4;
    constexpr int B_CHUNKS = BN * 4 / THREADS;
    constexpr int B_ROW    = KMajorWeights ? 4 : BN / CHUNK_ELEMS; // chunks per B shared-memory row
    // When the threads tile B rows, each thread's chunks share a column and step by B_RSTEP
    // rows; otherwise (BN = 96: B rows of 12 or 24 chunks) each chunk keeps its own offsets.
    constexpr bool B_TILED = THREADS % B_ROW == 0;
    constexpr int B_RSTEP  = B_TILED ? THREADS / B_ROW : 0;
    static_assert(BM * 4 % THREADS == 0 && BN * 4 % THREADS == 0, "chunks must tile the CTA");
    static_assert(A_RSTEP % 8 == 0, "per-thread swizzles must repeat");
    static_assert(B_TILED || !KMajorWeights, "k-major B rows are 4 chunks");

    int const aRow = tid >> 2, aChunk = tid & 3;
    int const bRowIdx = tid / B_ROW, bChunk = tid % B_ROW;
    int const aDst = aRow * BK + swizzle4(aRow, aChunk) * CHUNK_ELEMS;
    // B source offset of chunk 0 within a tap's weight block, the step between chunks, and
    // the step between k-tiles.
    int64_t const bOff       = KMajorWeights ? int64_t(n0 + bRowIdx) * Kc + bChunk * CHUNK_ELEMS
                                             : int64_t(bRowIdx) * Nc + n0 + bChunk * CHUNK_ELEMS;
    int64_t const bStep      = KMajorWeights ? int64_t(B_RSTEP) * Kc : int64_t(B_RSTEP) * Nc;
    int64_t const bKTileStep = KMajorWeights ? int64_t(BK) : int64_t(BK) * Nc;
    int bDst[B_TILED ? 1 : B_CHUNKS], bSrc[B_TILED ? 1 : B_CHUNKS];
    int bRow[B_TILED ? 1 : B_CHUNKS];
    bool bColOk[B_TILED ? 1 : B_CHUNKS];
    if constexpr (!B_TILED) {
#pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            int const c = tid + i * THREADS, r = c / B_ROW, ch = c % B_ROW;
            bDst[i]   = r * BN + swizzleRow<T, BN>(r, ch) * CHUNK_ELEMS;
            bSrc[i]   = r * Nc + ch * CHUNK_ELEMS;
            bRow[i]   = r;
            bColOk[i] = n0 + ch * CHUNK_ELEMS < Nc;
        }
    }
    // Rows of B along n (k-major) or columns (n-major) that exist in this n-tile.
    bool const bColsInRange = KMajorWeights ? true : n0 + bChunk * CHUNK_ELEMS < Nc;

    int loadTap = itBase / kTiles, loadKTile = itBase - loadTap * kTiles;
    T const *aPtr[A_CHUNKS];
    bool aOk[A_CHUNKS];
    T const *bPtr = nullptr;
    // LargeKernel: sNbr holds the neighbor columns of the tap group [tapGroupBase, tapGroupBase +
    // tapGroupSize). Every thread calls load() at the same iterations, so restaging inside it can
    // use CTA barriers.
    [[maybe_unused]] int tapGroupBase = 0, tapGroupSize = 0;
    [[maybe_unused]] auto stageTapGroup = [&](int first) {
        __syncthreads(); // every thread has read the previous tap group
        tapGroupBase = first;
        tapGroupSize = min(kStagedTaps, nTaps - first);
        for (int i = tid; i < BM * tapGroupSize; i += THREADS) {
            int const r = i / tapGroupSize;
            int const o = sRows[r];
            int32_t v   = -1;
            if (o >= 0) {
                // Compact map: rowMaskWords holds (mask, base) per 32-tap word.
                int const tap    = sTaps[first + i - r * tapGroupSize];
                int const bit    = tap % kMaskWordBits;
                int64_t const e  = (int64_t(o) * maskWords + tap / kMaskWordBits) * 2;
                uint32_t const m = uint32_t(rowMaskWords[e]);
                if ((m >> bit) & 1u)
                    v = neighborMap[rowMaskWords[e + 1] + __popc(m & bitsBelow(bit))];
            }
            sNbr[i] = v;
        }
        __syncthreads();
    };
    auto setTap = [&]() {
#pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            int const src = LargeKernel
                                ? sNbr[(aRow + i * A_RSTEP) * tapGroupSize + loadTap - tapGroupBase]
                                : sNbr[(aRow + i * A_RSTEP) * nTaps + loadTap];
            aOk[i]        = src >= 0;
            aPtr[i] = A + (aOk[i] ? int64_t(src) * Kc + aChunk * CHUNK_ELEMS : 0) + loadKTile * BK;
        }
        bPtr = W + int64_t(sWeightRow[loadTap]) * Kc * Nc + (B_TILED ? bOff : n0) +
               loadKTile * bKTileStep;
    };
    if (nIter > 0) {
        if constexpr (LargeKernel)
            stageTapGroup(loadTap);
        setTap();
    }

    auto load = [&](int stage) {
        int const k0         = loadKTile * BK;
        T *a                 = sA + stage * A_STAGE;
        bool const aKInRange = k0 + aChunk * CHUNK_ELEMS < Kc;
#pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            bool const ok = aOk[i] && aKInRange;
            cpAsync16(a + aDst + i * A_RSTEP * BK, ok ? aPtr[i] : A, ok);
            aPtr[i] += BK;
        }
        T *b = sB + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            if constexpr (!B_TILED) {
                bool const ok = bColOk[i] && k0 + bRow[i] < Kc;
                cpAsync16(b + bDst[i], ok ? bPtr + bSrc[i] : W, ok);
            } else {
                int const r = bRowIdx + i * B_RSTEP;
                if constexpr (KMajorWeights) {
                    bool const ok = n0 + r < Nc && k0 + bChunk * CHUNK_ELEMS < Kc;
                    cpAsync16(b + r * BK + swizzle4(r, bChunk) * CHUNK_ELEMS,
                              ok ? bPtr + i * bStep : W,
                              ok);
                } else {
                    bool const ok = bColsInRange && k0 + r < Kc;
                    cpAsync16(b + r * BN + swizzleRow<T, BN>(r, bChunk) * CHUNK_ELEMS,
                              ok ? bPtr + i * bStep : W,
                              ok);
                }
            }
        }
        bPtr += bKTileStep;
        if (++loadKTile == kTiles) {
            loadKTile = 0;
            if (++loadTap < nTaps) {
                if constexpr (LargeKernel) {
                    if (loadTap == tapGroupBase + tapGroupSize)
                        stageTapGroup(loadTap);
                }
                setTap();
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0.f;

    int const wm0 = (warp / Cfg::WN) * (BM / Cfg::WM);
    int const wn0 = (warp % Cfg::WN) * (BN / Cfg::WN);

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nIter)
            load(s);
        cute::cp_async_fence();
    }

    for (int it = 0; it < nIter; ++it) {
        cute::cp_async_wait<STAGES - 2>();
        __syncthreads();
        int const next = it + STAGES - 1;
        if (next < nIter)
            load(next % STAGES);
        cute::cp_async_fence();

        T const *a = sA + (it % STAGES) * A_STAGE;
        T const *b = sB + (it % STAGES) * B_STAGE;
        // Each k-step spans two 16-byte chunks for every element type; for float32,
        // ldmatrix's 8x8 16-bit tiles are 8x4 float tiles, which match TF32 fragments.
#pragma unroll
        for (int kk = 0; kk < BK / MMA_K; ++kk) {
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int const r  = wm0 + mt * 16 + (lane & 15);
                int const ch = kk * 2 + (lane >> 4);
                ldmatrixX4(af[mt], a + r * BK + swizzle4(r, ch) * CHUNK_ELEMS);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    af[mt][e] = roundUnless3xTf32<T, Use3xTf32>(af[mt][e]);
            }
            uint32_t bf[NT][2];
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t t[4];
                if constexpr (KMajorWeights) {
                    int const nr = wn0 + np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    int const ch = kk * 2 + ((lane >> 3) & 1);
                    ldmatrixX4(t, b + nr * BK + swizzle4(nr, ch) * CHUNK_ELEMS);
                } else if constexpr (sizeof(T) == 4) {
                    // ldmatrix cannot transpose 32-bit data; read N-major fragments
                    // element-wise: b0 = B[t][g], b1 = B[t + 4][g].
                    int const g = lane >> 2, k = kk * MMA_K + (lane & 3);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int const n  = wn0 + np * 16 + h * 8 + g;
                        auto const e = [&](int kr) {
                            return __float_as_uint(
                                b[kr * BN + swizzleRow<T, BN>(kr, n / CHUNK_ELEMS) * CHUNK_ELEMS +
                                  n % CHUNK_ELEMS]);
                        };
                        t[2 * h]     = e(k);
                        t[2 * h + 1] = e(k + 4);
                    }
                } else {
                    int const kr = kk * MMA_K + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int const ch = (wn0 + np * 16) / CHUNK_ELEMS + (lane >> 4);
                    ldmatrixX4Trans(t, b + kr * BN + swizzleRow<T, BN>(kr, ch) * CHUNK_ELEMS);
                }
                bf[2 * np][0]     = roundUnless3xTf32<T, Use3xTf32>(t[0]);
                bf[2 * np][1]     = roundUnless3xTf32<T, Use3xTf32>(t[1]);
                bf[2 * np + 1][0] = roundUnless3xTf32<T, Use3xTf32>(t[2]);
                bf[2 * np + 1][1] = roundUnless3xTf32<T, Use3xTf32>(t[3]);
            }
            mmaTiles<T, Use3xTf32>(acc, af, bf);
        }
    }
    cute::cp_async_wait<0>();

    // ---- Epilogue: each output row is owned by exactly one tile ----
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int const r = wm0 + mt * 16 + (lane >> 2) + h * 8;
            int const o = sRows[r];
            if (o < 0)
                continue;
            int const col      = n0 + wn0 + (lane & 3) * 2;
            int64_t const base = int64_t(o) * Nc + col;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if (col + nt * 8 >= Nc)
                    continue;
                if (splits > 1) {
                    atomicAdd(outAcc + base + nt * 8, acc[mt][nt][2 * h]);
                    atomicAdd(outAcc + base + nt * 8 + 1, acc[mt][nt][2 * h + 1]);
                } else {
                    float b0 = 0.f, b1 = 0.f;
                    if (bias) {
                        b0 = toFloat(bias[col + nt * 8]);
                        b1 = toFloat(bias[col + nt * 8 + 1]);
                    }
                    storePair<T>(
                        out + base + nt * 8, acc[mt][nt][2 * h] + b0, acc[mt][nt][2 * h + 1] + b1);
                }
            }
        }
    }
}

// =============================================================================
// Weight gradient
// =============================================================================
//
// CTA (m tile of C_in, n tile of C_out, tap u, split s) accumulates
//   dW[u][m][n] += sum_p features[pairIn[p], m] * grad_output[pairOut[p], n]
// over its share of tap u's pairs, then adds into a float32 buffer. Both
// operands are gathered along the reduction dimension. Warp 0 fetches each
// stage's pair indices into a shared-memory ring one iteration ahead, so the
// data loads never wait on a dependent global load.

// BK is the rows reduced per pipeline stage (at most 32, one per lane of warp 0).
template <int BM_, int BN_, int WM_, int WN_, int STAGES_, int BK_ = 32> struct WgradConfig {
    static constexpr int BM      = BM_;
    static constexpr int BN      = BN_;
    static constexpr int BK      = BK_;
    static constexpr int WM      = WM_;
    static constexpr int WN      = WN_;
    static constexpr int STAGES  = STAGES_;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT      = BM / WM / 16;
    static constexpr int NT      = BN / WN / 8;
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 8) == 0, "warp tiles must cover the CTA tile");
    static_assert(NT % 2 == 0 && BM >= 32 && BN >= 32 && BK <= 32, "tile shape");
};

template <typename Cfg, typename T>
constexpr int
wgradSmemBytes() {
    return Cfg::STAGES * (Cfg::BK * Cfg::BM + Cfg::BK * Cfg::BN) * int(sizeof(T)) +
           Cfg::STAGES * 2 * Cfg::BK * 4;
}

template <typename Cfg, MmaElement T, bool Use3xTf32>
__global__ void __launch_bounds__(Cfg::THREADS)
maskedIGemmWgradKernel(T const *__restrict__ X,
                       T const *__restrict__ dY,
                       float *__restrict__ dWAcc,
                       T *__restrict__ dWOut,
                       int32_t const *__restrict__ pairIn,
                       int32_t const *__restrict__ pairOut,
                       int32_t const *__restrict__ pairOffsets,
                       int32_t const *__restrict__ pairBlockOffsets,
                       int rowBlocks,
                       int Ci,
                       int Co,
                       int splits,
                       TapToWeightRow tapToRow) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, STAGES = Cfg::STAGES;
    constexpr int CHUNK_ELEMS = Elem<T>::CHUNK, MMA_K = Elem<T>::MMA_K;
    constexpr int THREADS = Cfg::THREADS, MT = Cfg::MT, NT = Cfg::NT;
    constexpr int A_STAGE = BK * BM, B_STAGE = BK * BN;

    extern __shared__ __align__(128) unsigned char smem[];
    T *sA         = reinterpret_cast<T *>(smem);
    T *sB         = sA + STAGES * A_STAGE;
    int32_t *sIdx = reinterpret_cast<int32_t *>(sB + STAGES * B_STAGE); // [STAGES][2][BK]

    int const tid    = threadIdx.x;
    int const lane   = tid & 31;
    int const warp   = tid >> 5;
    int const nTiles = (Co + BN - 1) / BN;
    int const nTile  = blockIdx.x % nTiles;
    int const mTile  = blockIdx.x / nTiles;
    int const tap    = blockIdx.y;
    int const m0     = mTile * BM;
    int const n0     = nTile * BN;

    // Each split takes an equal share of this tap's pairs, or, with pairBlockOffsets, an equal
    // share of the output rows so that concurrent CTAs of every tap read the same rows.
    int pFirst, pEnd;
    if (pairBlockOffsets) {
        int const b0     = int(int64_t(rowBlocks) * blockIdx.z / splits);
        int const b1     = int(int64_t(rowBlocks) * (blockIdx.z + 1) / splits);
        int32_t const *o = pairBlockOffsets + int64_t(tap) * rowBlocks;
        pFirst           = o[b0];
        pEnd             = b1 < rowBlocks ? o[b1] : pairOffsets[tap + 1];
    } else {
        int const pBegin = pairOffsets[tap];
        int const steps  = (pairOffsets[tap + 1] - pBegin + BK - 1) / BK;
        pFirst           = pBegin + int(int64_t(steps) * blockIdx.z / splits) * BK;
        pEnd             = std::min(pairOffsets[tap + 1],
                        pBegin + int(int64_t(steps) * (blockIdx.z + 1) / splits) * BK);
    }
    int const nIter = (pEnd - pFirst + BK - 1) / BK;
    // With one split, this CTA is the only writer of its tile and stores it even when empty.
    if (splits > 1 && nIter <= 0)
        return;

    // Pair indices of step it, for lane `lane` of warp 0.
    auto fetchIdx = [&](int it, int &in, int &out) {
        int const p   = pFirst + it * BK + lane;
        bool const ok = lane < BK && it < nIter && p < pEnd;
        in            = ok ? pairIn[p] : -1;
        out           = ok ? pairOut[p] : -1;
    };
    auto storeIdx = [&](int it, int in, int out) {
        if (lane >= BK)
            return;
        int32_t *slot   = sIdx + (it % STAGES) * 2 * BK;
        slot[lane]      = in;
        slot[BK + lane] = out;
    };

    auto load = [&](int stage, int it) {
        int32_t const *slot = sIdx + (it % STAGES) * 2 * BK;
        T *a                = sA + stage * A_STAGE;
        T *b                = sB + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < (BK * BM / CHUNK_ELEMS + THREADS - 1) / THREADS; ++i) {
            int const c = tid + i * THREADS;
            if (c < BK * BM / CHUNK_ELEMS) {
                int const kr  = c / (BM / CHUNK_ELEMS);
                int const ch  = c % (BM / CHUNK_ELEMS);
                int const src = slot[kr];
                bool const ok = src >= 0 && m0 + ch * CHUNK_ELEMS < Ci;
                T const *g    = X + (ok ? int64_t(src) * Ci + m0 + ch * CHUNK_ELEMS : 0);
                cpAsync16(a + kr * BM + swizzleRow<T, BM>(kr, ch) * CHUNK_ELEMS, g, ok);
            }
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / CHUNK_ELEMS + THREADS - 1) / THREADS; ++i) {
            int const c = tid + i * THREADS;
            if (c < BK * BN / CHUNK_ELEMS) {
                int const kr  = c / (BN / CHUNK_ELEMS);
                int const ch  = c % (BN / CHUNK_ELEMS);
                int const o   = slot[BK + kr];
                bool const ok = o >= 0 && n0 + ch * CHUNK_ELEMS < Co;
                T const *g    = dY + (ok ? int64_t(o) * Co + n0 + ch * CHUNK_ELEMS : 0);
                cpAsync16(b + kr * BN + swizzleRow<T, BN>(kr, ch) * CHUNK_ELEMS, g, ok);
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[i][j][e] = 0.f;

    int const wm0 = (warp / Cfg::WN) * (BM / Cfg::WM);
    int const wn0 = (warp % Cfg::WN) * (BN / Cfg::WN);

    // float32 fragments are read element-wise (ldmatrix cannot transpose 32-bit data): lane
    // (g, t) reads rows t and t + 4 of each k-step. The swizzle depends only on row & 3 == t,
    // so each column's offset within a stage is fixed per thread, and the k-step and row half
    // are immediate offsets.
    int aOff[MT][2], bOff[NT];
    if constexpr (sizeof(T) == 4) {
        int const g = lane >> 2, t = lane & 3;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                int const m = wm0 + mt * 16 + g + 8 * r;
                aOff[mt][r] =
                    t * BM + swizzleRow<T, BM>(t, m / CHUNK_ELEMS) * CHUNK_ELEMS + m % CHUNK_ELEMS;
            }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int const n = wn0 + nt * 8 + g;
            bOff[nt] =
                t * BN + swizzleRow<T, BN>(t, n / CHUNK_ELEMS) * CHUNK_ELEMS + n % CHUNK_ELEMS;
        }
        // Keep the offsets in registers; otherwise ptxas rebuilds them from the lane index
        // in every iteration.
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            asm("" : "+r"(aOff[mt][0]), "+r"(aOff[mt][1]));
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            asm("" : "+r"(bOff[nt]));
    }

    if (warp == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            int in, out;
            fetchIdx(s, in, out);
            storeIdx(s, in, out);
        }
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nIter)
            load(s, s);
        cute::cp_async_fence();
    }

    for (int it = 0; it < nIter; ++it) {
        cute::cp_async_wait<STAGES - 2>();
        __syncthreads();
        int const next = it + STAGES - 1;
        if (next < nIter)
            load(next % STAGES, next);
        cute::cp_async_fence();

        // Indices for step it + STAGES reuse the slot of step it, whose loads were issued
        // long ago. Fetch now, store after the MMAs so the fetch latency overlaps them.
        int pfIn = -1, pfOut = -1;
        if (warp == 0)
            fetchIdx(it + STAGES, pfIn, pfOut);

        T const *a = sA + (it % STAGES) * A_STAGE;
        T const *b = sB + (it % STAGES) * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK / MMA_K; ++kk) {
            uint32_t af[MT][4];
            uint32_t bf[NT][2];
            if constexpr (sizeof(T) == 4) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int const k = kk * MMA_K + 4 * h;
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        af[mt][2 * h] = roundUnless3xTf32<T, Use3xTf32>(
                            __float_as_uint(a[aOff[mt][0] + k * BM]));
                        af[mt][2 * h + 1] = roundUnless3xTf32<T, Use3xTf32>(
                            __float_as_uint(a[aOff[mt][1] + k * BM]));
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                        bf[nt][h] =
                            roundUnless3xTf32<T, Use3xTf32>(__float_as_uint(b[bOff[nt] + k * BN]));
                }
            } else {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    int const kr = kk * MMA_K + (lane & 7) + ((lane >> 4) & 1) * 8;
                    int const ch = (wm0 + mt * 16) / CHUNK_ELEMS + ((lane >> 3) & 1);
                    ldmatrixX4Trans(af[mt], a + kr * BM + swizzleRow<T, BM>(kr, ch) * CHUNK_ELEMS);
                }
#pragma unroll
                for (int np = 0; np < NT / 2; ++np) {
                    uint32_t t[4];
                    int const kr = kk * MMA_K + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int const ch = (wn0 + np * 16) / CHUNK_ELEMS + (lane >> 4);
                    ldmatrixX4Trans(t, b + kr * BN + swizzleRow<T, BN>(kr, ch) * CHUNK_ELEMS);
                    bf[2 * np][0]     = t[0];
                    bf[2 * np][1]     = t[1];
                    bf[2 * np + 1][0] = t[2];
                    bf[2 * np + 1][1] = t[3];
                }
            }
            mmaTiles<T, Use3xTf32>(acc, af, bf);
        }

        if (warp == 0)
            storeIdx(it + STAGES, pfIn, pfOut);
    }
    cute::cp_async_wait<0>();

    int64_t const tapBase = int64_t(tapToRow(tap)) * Ci * Co;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int const m = m0 + wm0 + mt * 16 + (lane >> 2) + h * 8;
            if (m >= Ci)
                continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int const n = n0 + wn0 + nt * 8 + (lane & 3) * 2;
                if (n >= Co)
                    continue;
                int64_t const idx = tapBase + int64_t(m) * Co + n;
                if (splits > 1) {
                    atomicAdd(dWAcc + idx, acc[mt][nt][2 * h]);
                    atomicAdd(dWAcc + idx + 1, acc[mt][nt][2 * h + 1]);
                } else {
                    storePair<T>(dWOut + idx, acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
                }
            }
        }
    }
}

// =============================================================================
// Neighbor-map builders
// =============================================================================
//
// Both write, per output row o: nbr[o, k] (input row or -1), the tap mask, and
// the mask's Gray-code rank (prefix XOR), which orders rows as FlexGEMM does.

__host__ __device__ inline uint32_t
grayRank(uint32_t mask) {
    mask ^= mask >> 1;
    mask ^= mask >> 2;
    mask ^= mask >> 4;
    mask ^= mask >> 8;
    mask ^= mask >> 16;
    return mask;
}

// Leaf-staged builder. One CTA per VoxelBlockManager block of 128 active voxels
// in leaf order, which span a run of consecutive leaves. The CTA probes each of
// those leaves' 27 neighbor leaves once into shared memory, then every voxel
// resolves its taps with a mask test and the leaf's prefix-sum popcount instead
// of walking the tree. Every tap offset must stay within one leaf (|d| <= 8).
constexpr int kStagedWidth = VbmCache::kBlockWidth;

__global__ void __launch_bounds__(kStagedWidth)
stagedNeighborMapKernel(nanovdb::OnIndexGrid const *__restrict__ grid,
                        uint32_t const *__restrict__ firstLeafID,
                        uint64_t const *__restrict__ jumpMap,
                        uint64_t firstOffset,
                        uint64_t lastOffset,
                        int64_t base,
                        ConvolutionGeometry geometry,
                        int K,
                        int32_t *__restrict__ nbr,
                        int32_t *__restrict__ code,
                        int32_t *__restrict__ masks) {
    using VbmT          = nanovdb::tools::cuda::VoxelBlockManager<VbmCache::kLog2BlockWidth>;
    constexpr int WORDS = VbmCache::kJumpMapWordCount;

    __shared__ int32_t sLeaves[kStagedWidth][27]; // neighbor leaf index, -1 if absent
    __shared__ int sNumLeaves;

    uint64_t const *jump = jumpMap + uint64_t(blockIdx.x) * WORDS;
    uint32_t const leaf0 = firstLeafID[blockIdx.x];
    auto const *leaves   = grid->tree().template getFirstNode<0>();
    auto const &root     = grid->tree().root();
    uint64_t const first = firstOffset + uint64_t(blockIdx.x) * kStagedWidth;

    if (threadIdx.x == 0) {
        int n = 1; // a jump bit marks each later leaf starting inside the block
        for (int w = 0; w < WORDS; ++w)
            n += __popcll(jump[w]);
        sNumLeaves = n;
    }
    __syncthreads();
    int const numLeaves = sNumLeaves;
    for (int i = threadIdx.x; i < numLeaves * 27; i += kStagedWidth) {
        int const j  = i / 27;
        int const n  = i - j * 27;
        auto const o = leaves[leaf0 + j].origin();
        auto const *l =
            root.probeLeaf(o.offsetBy((n / 9 - 1) * 8, ((n / 3) % 3 - 1) * 8, (n % 3 - 1) * 8));
        sLeaves[j][n] = l ? int32_t(l - leaves) : -1;
    }
    __syncthreads();

    if (first + threadIdx.x > lastOffset)
        return;
    uint32_t leafIndex;
    uint16_t voxelOffset;
    VbmT::decodeInverseMap(grid, leaf0, jump, first, int(threadIdx.x), leafIndex, voxelOffset);
    if (leafIndex == VbmT::UnusedLeafIndex)
        __trap(); // the lastOffset guard makes this unreachable; never leave a row unwritten

    int const j = int(leafIndex - leaf0);
    int const x = voxelOffset >> 6, y = (voxelOffset >> 3) & 7, z = voxelOffset & 7;
    // Sequential OnIndex invariant: this voxel's value index is its slot.
    int64_t const row = base + int64_t(first + threadIdx.x) - 1;
    int32_t *out      = nbr + row * K;
    uint32_t mask     = 0;
    for (int k = 0; k < K; ++k) {
        nanovdb::Coord const d = geometry.tapOffset(geometry.tapCoord(k));
        int const nx = x + d[0], ny = y + d[1], nz = z + d[2];
        int const n  = ((nx + 8) / 8) * 9 + ((ny + 8) / 8) * 3 + (nz + 8) / 8;
        int const li = sLeaves[j][n];
        int32_t idx  = -1;
        if (li >= 0) {
            uint32_t const off = uint32_t(((nx & 7) << 6) | ((ny & 7) << 3) | (nz & 7));
            auto const &leaf   = leaves[li];
            if (leaf.isActive(off)) {
                idx = int32_t(base + int64_t(leaf.getValue(off)) - 1);
                mask |= 1u << k;
            }
        }
        out[k] = idx;
    }
    code[row]  = int32_t(grayRank(mask));
    masks[row] = int32_t(mask);
}

void
buildNeighborMapStaged(GridBatchData const &grid,
                       ConvolutionGeometry const &geometry,
                       int K,
                       int32_t *nbr,
                       int32_t *code,
                       int32_t *masks,
                       cudaStream_t stream) {
    for (int64_t bi = 0; bi < grid.batchSize(); ++bi) {
        auto const vbm = grid.vbmAt(bi);
        if (vbm.blockCount == 0)
            continue;
        stagedNeighborMapKernel<<<vbm.blockCount, kStagedWidth, 0, stream>>>(
            grid.deviceGridPtrAt(bi),
            vbm.firstLeafID,
            vbm.jumpMap,
            vbm.firstOffset,
            vbm.lastOffset,
            grid.cumVoxelsAt(bi),
            geometry,
            K,
            nbr,
            code,
            masks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Leaf-scan builder: one thread per output voxel, K probes of the feature grid's tree.
void
buildNeighborMapScan(GridBatchData const &featureGrid,
                     GridBatchData const &outputGrid,
                     ConvolutionGeometry const &geometry,
                     int K,
                     int32_t *nbr,
                     int32_t *code,
                     int32_t *masks) {
    using CudaTag  = ::dispatch::tag<torch::kCUDA>;
    auto const acc = dispatch::make_grid_accessor(CudaTag{}, featureGrid);
    dispatch::forEachActiveVoxel(CudaTag{},
                                 outputGrid,
                                 [=] __hostdev__(CudaTag,
                                                 JIdxType batchIdx,
                                                 nanovdb::Coord ijk,
                                                 int64_t voxelIdx,
                                                 GridBatchData::Accessor) {
                                     auto treeAcc       = acc.grid(batchIdx)->getAccessor();
                                     int64_t const base = acc.voxelOffset(batchIdx);
                                     int32_t *row       = nbr + voxelIdx * K;
                                     uint32_t mask      = 0;
                                     for (int k = 0; k < K; ++k) {
                                         nanovdb::Coord const probe =
                                             geometry.fineFromCoarse(ijk, geometry.tapCoord(k));
                                         int32_t idx = -1;
                                         if (treeAcc.isActive(probe)) {
                                             idx = int32_t(base + treeAcc.getValue(probe) - 1);
                                             mask |= 1u << k;
                                         }
                                         row[k] = idx;
                                     }
                                     code[voxelIdx]  = int32_t(grayRank(mask));
                                     masks[voxelIdx] = int32_t(mask);
                                 });
}

// Reverse map: rev[i, k] = o where nbr[o, k] = i. The tap fixes the output coordinate,
// so each (input row, tap) has at most one output row and the scatter has no conflicts.
__global__ void
reverseMapKernel(int32_t const *__restrict__ nbr, int64_t total, int K, int32_t *__restrict__ rev) {
    for (int64_t e = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; e < total;
         e += int64_t(gridDim.x) * blockDim.x) {
        int32_t const i = nbr[e];
        if (i >= 0)
            rev[int64_t(i) * K + e % K] = int32_t(e / K);
    }
}

__global__ void
rowMaskKernel(int32_t const *__restrict__ map,
              int64_t N,
              int K,
              int32_t *__restrict__ code,
              int32_t *__restrict__ masks) {
    for (int64_t row = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; row < N;
         row += int64_t(gridDim.x) * blockDim.x) {
        uint32_t mask = 0;
        for (int k = 0; k < K; ++k)
            mask |= uint32_t(map[row * K + k] >= 0) << k;
        code[row]  = int32_t(grayRank(mask));
        masks[row] = int32_t(mask);
    }
}

int
gridStrideBlocks(int64_t work) {
    return int(std::clamp<int64_t>(
        (work + kGridStrideThreads - 1) / kGridStrideThreads, 1, int64_t(1) << 16));
}

// =============================================================================
// Weight-gradient pair lists
// =============================================================================
//
// Per tap, the (input, output) pairs with output rows ascending. Two passes over
// blocks of rows: count each tap's pairs per block, then write each pair at its
// tap's offset plus its block's offset plus its rank within the block.

constexpr int kPairBlock = 256;

__global__ void
pairCountKernel(int32_t const *__restrict__ rowMasks,
                int N,
                int K,
                int32_t *__restrict__ blockCounts) {
    __shared__ int32_t warpCounts[kPairBlock / 32][kMaxTaps];
    int const row    = blockIdx.x * kPairBlock + threadIdx.x;
    int const warp   = threadIdx.x >> 5;
    uint32_t const m = row < N ? uint32_t(rowMasks[row]) : 0u;
    for (int k = 0; k < K; ++k) {
        uint32_t const bits = __ballot_sync(0xffffffffu, (m >> k) & 1u);
        if ((threadIdx.x & 31) == 0)
            warpCounts[warp][k] = __popc(bits);
    }
    __syncthreads();
    if (threadIdx.x < K) {
        int sum = 0;
        for (int w = 0; w < kPairBlock / 32; ++w)
            sum += warpCounts[w][threadIdx.x];
        blockCounts[int64_t(threadIdx.x) * gridDim.x + blockIdx.x] = sum;
    }
}

// blockOffsets[k * blocks + b]: start of block b's tap-k pairs (tap offset included).
__global__ void
pairFillKernel(int32_t const *__restrict__ rowMasks,
               int32_t const *__restrict__ nbr,
               int32_t const *__restrict__ blockOffsets,
               int N,
               int K,
               int32_t *__restrict__ pairIn,
               int32_t *__restrict__ pairOut) {
    __shared__ int32_t warpCounts[kPairBlock / 32][kMaxTaps];
    int const row    = blockIdx.x * kPairBlock + threadIdx.x;
    int const warp   = threadIdx.x >> 5;
    int const lane   = threadIdx.x & 31;
    uint32_t const m = row < N ? uint32_t(rowMasks[row]) : 0u;
    for (int k = 0; k < K; ++k) {
        uint32_t const bits = __ballot_sync(0xffffffffu, (m >> k) & 1u);
        if (lane == 0)
            warpCounts[warp][k] = __popc(bits);
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        uint32_t const bits = __ballot_sync(0xffffffffu, (m >> k) & 1u);
        if (!((m >> k) & 1u))
            continue;
        int pos =
            blockOffsets[int64_t(k) * gridDim.x + blockIdx.x] + __popc(bits & bitsBelow(lane));
        for (int w = 0; w < warp; ++w)
            pos += warpCounts[w][k];
        pairOut[pos] = row;
        pairIn[pos]  = nbr[int64_t(row) * K + k];
    }
}

// =============================================================================
// Launchers
// =============================================================================

int
smCount() {
    int device = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms;
}

// The global Gray order groups rows with similar tap masks but scatters spatial neighbors
// across the grid. When the gathered tensor is several times L2, windows of nearby rows each
// sorted by Gray rank keep the neighbor gathers in L2 at the cost of slightly larger tile
// unions.
constexpr int64_t kLocalWindow = int64_t(1) << 16;

// sortedRows regrouped into windows of kLocalWindow nearby rows, each in Gray order. Built with
// the topology, so no execute call writes topology state another stream might read. Undefined
// for maps of at most one window, whose windowed order is the global one.
torch::Tensor
windowedRowOrder(torch::Tensor const &sortedRows) {
    if (sortedRows.numel() <= kLocalWindow)
        return {};
    auto const window = torch::div(sortedRows, kLocalWindow, "floor");
    auto const perm   = std::get<1>(torch::sort(window, std::optional<bool>(true), 0));
    return sortedRows.index_select(0, perm);
}

// gridDim.z limit; the weight gradient's splits run along z.
constexpr int64_t kMaxGridZ = 65535;

torch::Tensor const &
rowOrder(MaskedIGemmTopology const &topo, int64_t gatheredBytes) {
    int device = 0, l2 = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    bool local = topo.outputTotalVoxels > kLocalWindow && gatheredBytes >= 3 * int64_t(l2);
    if (gTestOverrides.localRowOrder)
        local = *gTestOverrides.localRowOrder;
    return local && topo.localRows.defined() ? topo.localRows : topo.sortedRows;
}

template <typename Cfg, typename T, bool KMajorWeights, bool Use3xTf32, bool LargeKernel = false>
void
launchConv(torch::Tensor const &A,
           torch::Tensor const &Wrows,
           torch::Tensor const &bias,
           torch::Tensor &out,
           MaskedIGemmTopology const &topo,
           bool flipTaps,
           int Kc,
           int Nc,
           cudaStream_t stream) {
    int const N    = int(topo.outputTotalVoxels);
    int const K    = int(topo.kernelVolume);
    int const smem = smemBytes<Cfg, T, LargeKernel>(K);
    auto kernel    = maskedIGemmKernel<Cfg, T, KMajorWeights, Use3xTf32, LargeKernel>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int const tiles = ((N + Cfg::BM - 1) / Cfg::BM) * ((Nc + Cfg::BN - 1) / Cfg::BN);

    // Split each tile's (tap, k-tile) iterations to fill the last wave. Cost is in k-tile units:
    // each wave runs blocks of iters / s k-tiles plus a fixed prologue and epilogue (a float32
    // atomic tile when split), and splitting adds a zero-fill and copy pass over the output.
    constexpr double kBlockOverhead = 24.0;
    constexpr double kSplitPass     = 1.1;
    int perSm                       = 0;
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, Cfg::THREADS, smem));
    int const slots     = std::max(1, perSm) * smCount();
    int const kTiles    = (Kc + Elem<T>::BK - 1) / Elem<T>::BK;
    double const iters  = double(K) * kTiles;
    int const maxSplits = std::min(64, 4 * kTiles);
    auto const cost     = [&](int s) {
        int64_t const waves = (int64_t(tiles) * s + slots - 1) / slots;
        return waves * (iters / s + kBlockOverhead) * (s > 1 ? kSplitPass : 1.0);
    };
    int splits  = 1;
    double best = cost(1);
    for (int s = 2; s <= maxSplits; ++s) {
        if (double const c = cost(s); c < best) {
            best   = c;
            splits = s;
        }
    }
    if (gTestOverrides.splits)
        splits = std::clamp(*gTestOverrides.splits, 1, maxSplits);
    torch::Tensor acc;
    if (splits > 1)
        acc = torch::zeros({N, Nc}, out.options().dtype(torch::kFloat32));

    TapToWeightRow const ttr{.k1   = topo.kernelSize[1],
                             .k2   = topo.kernelSize[2],
                             .k0   = topo.kernelSize[0],
                             .flip = flipTaps};
    dim3 const grid(tiles, splits);
    kernel<<<grid, Cfg::THREADS, smem, stream>>>(
        reinterpret_cast<T const *>(A.data_ptr()),
        reinterpret_cast<T const *>(Wrows.data_ptr()),
        bias.defined() ? reinterpret_cast<T const *>(bias.data_ptr()) : nullptr,
        reinterpret_cast<T *>(out.data_ptr()),
        splits > 1 ? acc.data_ptr<float>() : nullptr,
        topo.neighborMap.data_ptr<int32_t>(),
        rowOrder(topo, A.numel() * A.element_size()).data_ptr<int32_t>(),
        (LargeKernel ? topo.rowWords : topo.rowMasks).data_ptr<int32_t>(),
        N,
        Kc,
        Nc,
        K,
        splits,
        ttr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1)
        out.copy_(bias.defined() ? acc + bias : acc);
}

// Default tile shape by output width.
template <typename T, bool KMajorWeights, bool Use3xTf32, bool LargeKernel>
void
dispatchConvTiles(torch::Tensor const &A,
                  torch::Tensor const &Wrows,
                  torch::Tensor const &bias,
                  torch::Tensor &out,
                  MaskedIGemmTopology const &topo,
                  bool flipTaps,
                  int Kc,
                  int Nc,
                  cudaStream_t stream) {
    // Two stages keep two CTAs resident per SM, which beats deeper pipelines here.
    // Channel counts that are not multiples of 32 pick tiles as their round-up would, so a
    // narrow output (48, say) takes one wider tile instead of re-gathering its rows per tile.
    int const Np = (Nc + 31) / 32 * 32;
    if (Np % 128 == 0) {
        launchConv<TileConfig<128, 128, 2, 2, 2>, T, KMajorWeights, Use3xTf32, LargeKernel>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else if (Np % 64 == 0) {
        launchConv<TileConfig<128, 64, 4, 1, 2>, T, KMajorWeights, Use3xTf32, LargeKernel>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else if (Np % 96 == 0) {
        launchConv<TileConfig<128, 96, 4, 1, 2>, T, KMajorWeights, Use3xTf32, LargeKernel>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else {
        // 32-wide stages are small enough for a third stage at two CTAs per SM.
        launchConv<TileConfig<128, 32, 4, 1, 3>, T, KMajorWeights, Use3xTf32, LargeKernel>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    }
}

// Dense (at most kMaxTaps taps) or compact (large kernel) neighbor-map instantiations.
template <typename T, bool KMajorWeights, bool Use3xTf32>
void
dispatchConv(torch::Tensor const &A,
             torch::Tensor const &Wrows,
             torch::Tensor const &bias,
             torch::Tensor &out,
             MaskedIGemmTopology const &topo,
             bool flipTaps,
             int Kc,
             int Nc,
             cudaStream_t stream) {
    if (topo.kernelVolume > kMaxTaps) {
        dispatchConvTiles<T, KMajorWeights, Use3xTf32, true>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
        return;
    }
    dispatchConvTiles<T, KMajorWeights, Use3xTf32, false>(
        A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
}

template <typename Cfg, typename T, bool Use3xTf32>
torch::Tensor
launchWgrad(torch::Tensor const &X,
            torch::Tensor const &dY,
            MaskedIGemmTopology const &topo,
            int Ci,
            int Co,
            cudaStream_t stream) {
    int const K    = int(topo.kernelVolume);
    int const smem = wgradSmemBytes<Cfg, T>();
    auto kernel    = maskedIGemmWgradKernel<Cfg, T, Use3xTf32>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    int const sms       = smCount();
    int const tiles     = ((Ci + Cfg::BM - 1) / Cfg::BM) * ((Co + Cfg::BN - 1) / Cfg::BN);
    int64_t const steps = std::max<int64_t>(1, topo.pairIn.numel() / std::max(1, K) / Cfg::BK);
    // Many short CTAs keep the last wave small, but each split adds a tile of float32 atomics,
    // so a pair split keeps at least 128 steps per CTA (a row split, below, at least 32).
    int64_t const target = int64_t(64) * sms;
    int64_t splits       = std::clamp<int64_t>(
        (target + tiles * K - 1) / (tiles * K), 1, std::max<int64_t>(1, steps / 128));

    // Each row is read by up to K taps, which hit L2 only if they run close together. When X and
    // dY are several times L2, split by output rows rather than pairs, so the K taps of a split
    // read the same rows, and use enough splits that the rows read by the
    // resident / (tiles * K) concurrently running splits fit in a quarter of L2.
    int perSm = 0, device = 0, l2 = 0;
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, Cfg::THREADS, smem));
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    double const groups = double(std::max(1, perSm) * sms) / double(tiles * K);
    double const bytes  = double(X.numel() + dY.numel()) * X.element_size();
    auto const local    = int64_t(std::ceil(groups * bytes / (l2 / 4.0)));
    int const rowBlocks = int(topo.pairBlockOffsets.size(1));
    bool rowSplit       = local > splits && rowBlocks > 1;
    if (rowSplit)
        splits = std::min<int64_t>({local, std::max<int64_t>(1, steps / 32), rowBlocks});
    if (gTestOverrides.weightRowSplits && rowBlocks > 0) {
        rowSplit = true;
        splits   = std::clamp<int64_t>(*gTestOverrides.weightRowSplits, 1, rowBlocks);
    }

    torch::Tensor rows, acc;
    if (splits > 1)
        acc = torch::zeros({K, Ci, Co}, X.options().dtype(torch::kFloat32));
    else
        rows = torch::empty({K, Ci, Co}, X.options());

    TapToWeightRow const ttr{.k1   = topo.kernelSize[1],
                             .k2   = topo.kernelSize[2],
                             .k0   = topo.kernelSize[0],
                             .flip = false};
    splits = std::min<int64_t>(splits, kMaxGridZ);
    dim3 const grid(tiles, K, unsigned(splits));
    kernel<<<grid, Cfg::THREADS, smem, stream>>>(
        reinterpret_cast<T const *>(X.data_ptr()),
        reinterpret_cast<T const *>(dY.data_ptr()),
        splits > 1 ? acc.data_ptr<float>() : nullptr,
        splits > 1 ? nullptr : reinterpret_cast<T *>(rows.data_ptr()),
        topo.pairIn.data_ptr<int32_t>(),
        topo.pairOut.data_ptr<int32_t>(),
        topo.pairOffsets.data_ptr<int32_t>(),
        rowSplit ? topo.pairBlockOffsets.data_ptr<int32_t>() : nullptr,
        rowBlocks,
        Ci,
        Co,
        int(splits),
        ttr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return splits > 1 ? acc.to(X.scalar_type()) : rows;
}

int
round32(int c) {
    return (c + 31) / 32 * 32;
}

// Channel counts not divisible by 128 use 64-, 96- or 32-wide tiles on each side, whichever
// divides the count rounded up to 32 first, so each CTA gathers its rows once per 64 or 96
// channels where possible.
template <typename T, int BM, int WM, bool Use3xTf32>
torch::Tensor
launchWgradByCo(torch::Tensor const &X,
                torch::Tensor const &dY,
                MaskedIGemmTopology const &topo,
                int Ci,
                int Co,
                cudaStream_t stream) {
    if (round32(Co) % 64 == 0)
        return launchWgrad<WgradConfig<BM, 64, WM, 2, 2>, T, Use3xTf32>(
            X, dY, topo, Ci, Co, stream);
    if (round32(Co) % 96 == 0)
        return launchWgrad<WgradConfig<BM, 96, WM, 2, 2>, T, Use3xTf32>(
            X, dY, topo, Ci, Co, stream);
    return launchWgrad<WgradConfig<BM, 32, WM, 1, 2>, T, Use3xTf32>(X, dY, topo, Ci, Co, stream);
}

template <typename T, bool Use3xTf32>
torch::Tensor
launchWgradNarrow(torch::Tensor const &X,
                  torch::Tensor const &dY,
                  MaskedIGemmTopology const &topo,
                  int Ci,
                  int Co,
                  cudaStream_t stream) {
    if (round32(Ci) % 64 == 0)
        return launchWgradByCo<T, 64, 2, Use3xTf32>(X, dY, topo, Ci, Co, stream);
    if (round32(Ci) % 96 == 0)
        return launchWgradByCo<T, 96, 2, Use3xTf32>(X, dY, topo, Ci, Co, stream);
    return launchWgradByCo<T, 32, 1, Use3xTf32>(X, dY, topo, Ci, Co, stream);
}

template <typename T, bool Use3xTf32>
torch::Tensor
dispatchWgrad(torch::Tensor const &X,
              torch::Tensor const &dY,
              MaskedIGemmTopology const &topo,
              int Ci,
              int Co,
              cudaStream_t stream) {
    if constexpr (sizeof(T) == 4) {
        // float32 reads fragments element-wise, so wide warp tiles amortize the loads;
        // at 128x128, 16-row stages keep two CTAs resident per SM.
        if (round32(Ci) % 128 == 0 && round32(Co) % 128 == 0)
            return launchWgrad<WgradConfig<128, 128, 2, 2, 3, 16>, T, Use3xTf32>(
                X, dY, topo, Ci, Co, stream);
        return launchWgradNarrow<T, Use3xTf32>(X, dY, topo, Ci, Co, stream);
    } else {
        if (round32(Ci) % 128 == 0 && round32(Co) % 128 == 0)
            return launchWgrad<WgradConfig<128, 128, 2, 2, 3>, T, Use3xTf32>(
                X, dY, topo, Ci, Co, stream);
        return launchWgradNarrow<T, Use3xTf32>(X, dY, topo, Ci, Co, stream);
    }
}

// The relations between topology fields that the launchers rely on.
void
validateTopology(MaskedIGemmTopology const &t, char const *name) {
    int64_t const K = int64_t(t.kernelSize[0]) * t.kernelSize[1] * t.kernelSize[2];
    TORCH_CHECK(
        t.kernelVolume == K, name, ": topology kernel volume does not match its kernel size");
    bool const large = K > kMaxTaps;
    TORCH_CHECK(t.rowWords.defined() == large && t.neighborOffsets.defined() == large,
                name,
                ": topology has the wrong neighbor-map layout for a kernel of ",
                K,
                " taps");
    if (large) {
        TORCH_CHECK(t.neighborOffsets.numel() == t.outputTotalVoxels + 1 && t.rowWords.dim() == 3 &&
                        t.rowWords.size(0) == t.outputTotalVoxels &&
                        t.rowWords.size(1) == maskWordsFor(K) && t.rowWords.size(2) == 2,
                    name,
                    ": compact neighbor map has inconsistent shapes");
    } else {
        TORCH_CHECK(t.neighborMap.dim() == 2 && t.neighborMap.size(0) == t.outputTotalVoxels &&
                        t.neighborMap.size(1) == K && t.rowMasks.numel() == t.outputTotalVoxels,
                    name,
                    ": neighbor map has inconsistent shapes");
    }
    TORCH_CHECK(
        t.sortedRows.numel() == t.outputTotalVoxels, name, ": row order has the wrong length");
}

void
checkConv(torch::Tensor const &features,
          torch::Tensor const &weights,
          MaskedIGemmTopology const &topo,
          bool transposed,
          char const *name) {
    validateTopology(topo, name);
    if (topo.reverse)
        validateTopology(*topo.reverse, name);
    int64_t const inRows = transposed ? topo.outputTotalVoxels : topo.featureTotalVoxels;
    TORCH_CHECK(features.is_cuda(), name, ": features must be on CUDA");
    TORCH_CHECK(weights.device() == features.device() &&
                    topo.neighborMap.device() == features.device(),
                name,
                ": features, weights and topology must be on the same device");
    TORCH_CHECK(weights.scalar_type() == features.scalar_type(),
                name,
                ": weights must have the features' dtype ",
                features.scalar_type());
    TORCH_CHECK(features.dim() == 2 && features.size(0) == inRows,
                name,
                ": features must be [",
                inRows,
                ", C_in]");
    TORCH_CHECK(weights.dim() == 5 && weights.size(1) == features.size(1),
                name,
                ": weights must be [C_out, C_in, k0, k1, k2] with C_in matching features");
    TORCH_CHECK(weights.size(2) == topo.kernelSize[0] && weights.size(3) == topo.kernelSize[1] &&
                    weights.size(4) == topo.kernelSize[2],
                name,
                ": weights spatial dims must match the topology kernel size");
    TORCH_CHECK(maskedIGemmSupports(
                    weights.size(1), weights.size(0), features.scalar_type(), features.device()),
                name,
                ": unsupported channels (",
                weights.size(1),
                ", ",
                weights.size(0),
                ") or dtype ",
                features.scalar_type());
}

// A row map for the conv kernel, applied with or without mirrored taps.
struct RowMap {
    MaskedIGemmTopology const &topo;
    bool flip;
};

// Rows of the forward convolution's input: the reverse map, or for a submanifold map the
// forward map itself, since tap K-1-k of input row i reaches the row tap k reads from.
RowMap
inputRows(MaskedIGemmTopology const &topo) {
    return topo.reverse ? RowMap{*topo.reverse, false} : RowMap{topo, true};
}

template <bool KMajorWeights>
void
runConv(torch::Tensor const &A,
        torch::Tensor const &Wrows,
        torch::Tensor const &bias,
        torch::Tensor &out,
        RowMap map,
        int Kc,
        int Nc,
        bool precise,
        cudaStream_t stream) {
    auto const st = A.scalar_type();
    if (st == torch::kFloat16)
        dispatchConv<__half, KMajorWeights, false>(
            A, Wrows, bias, out, map.topo, map.flip, Kc, Nc, stream);
    else if (st == torch::kBFloat16)
        dispatchConv<__nv_bfloat16, KMajorWeights, false>(
            A, Wrows, bias, out, map.topo, map.flip, Kc, Nc, stream);
    else if (precise)
        dispatchConv<float, KMajorWeights, true>(
            A, Wrows, bias, out, map.topo, map.flip, Kc, Nc, stream);
    else
        dispatchConv<float, KMajorWeights, false>(
            A, Wrows, bias, out, map.topo, map.flip, Kc, Nc, stream);
}

// Per-tap X^T dY over the topology's pairs, X gathered by input row and dY by output row.
torch::Tensor
runWgrad(torch::Tensor const &X,
         torch::Tensor const &dY,
         MaskedIGemmTopology const &topo,
         int Ci,
         int Co,
         bool precise,
         cudaStream_t stream) {
    if (topo.pairIn.numel() == 0)
        return torch::zeros({topo.kernelVolume, Ci, Co}, X.options());
    auto const st = X.scalar_type();
    if (st == torch::kFloat16)
        return dispatchWgrad<__half, false>(X, dY, topo, Ci, Co, stream);
    if (st == torch::kBFloat16)
        return dispatchWgrad<__nv_bfloat16, false>(X, dY, topo, Ci, Co, stream);
    if (precise)
        return dispatchWgrad<float, true>(X, dY, topo, Ci, Co, stream);
    return dispatchWgrad<float, false>(X, dY, topo, Ci, Co, stream);
}

// Column sums of a row-major [N, C] tensor, accumulated in float32. Each thread loads 16-byte
// column chunks from a strided set of rows; a block reduces its rows in shared memory and adds
// one partial per column into out.
constexpr int kColumnSumThreads = 256;

template <MmaElement T>
__global__ void __launch_bounds__(kColumnSumThreads)
columnSumKernel(T const *__restrict__ x, int64_t N, int C, float *__restrict__ out) {
    constexpr int V  = 16 / sizeof(T);
    int const chunks = C / V;
    int const lanes  = kColumnSumThreads / chunks;
    int const chunk  = threadIdx.x % chunks;
    int const lane   = threadIdx.x / chunks;
    __shared__ float part[kColumnSumThreads * V];

    float acc[V] = {};
    if (lane < lanes) {
        for (int64_t row = int64_t(blockIdx.x) * lanes + lane; row < N;
             row += int64_t(gridDim.x) * lanes) {
            T v[V];
            *reinterpret_cast<uint4 *>(v) =
                *reinterpret_cast<uint4 const *>(x + row * C + chunk * V);
#pragma unroll
            for (int i = 0; i < V; ++i)
                acc[i] += toFloat(v[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < V; ++i)
        part[threadIdx.x * V + i] = acc[i];
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += kColumnSumThreads) {
        float sum = 0.0f;
        for (int l = 0; l < lanes; ++l)
            sum += part[(l * chunks + c / V) * V + c % V];
        atomicAdd(out + c, sum);
    }
}

// =============================================================================
// Compact neighbor maps (kernels of more than kMaxTaps taps)
// =============================================================================
//
// A dense [N, K] map grows with K: 343 taps over millions of rows is tens of GiB. Large kernels
// keep only each row's present neighbors, in tap order, at neighborMap[neighborOffsets[o] ...].
// rowWords[o, w] = (mask, base) for taps 32w..32w+31: the word's tap bits and the compact index of
// its first present tap, so tap t's neighbor is neighborMap[base + popc(mask & below(t % 32))].
// Rows are probed a batch at a time into a dense scratch and compacted, so the dense map never
// exists in full and every tap is probed once.

constexpr int64_t kProbeBatchBytes = int64_t(256) << 20; // dense scratch per probe batch

// Probe rows [r0, r1) of rowGrid into scratch[(row - r0) * K + k] (a probeGrid row or -1). The
// forward map's rows are output voxels probing input voxels at fine = stride * coarse + tap -
// padding; the reverse map's rows are input voxels probing the output voxel that reads them.
template <bool Reverse>
void
probeBatchScan(GridBatchData const &rowGrid,
               GridBatchData const &probeGrid,
               ConvolutionGeometry const &geometry,
               int K,
               int64_t r0,
               int64_t r1,
               int32_t *scratch) {
    using CudaTag  = ::dispatch::tag<torch::kCUDA>;
    auto const acc = dispatch::make_grid_accessor(CudaTag{}, probeGrid);
    dispatch::forEachActiveVoxel(CudaTag{},
                                 rowGrid,
                                 [=] __hostdev__(CudaTag,
                                                 JIdxType batchIdx,
                                                 nanovdb::Coord ijk,
                                                 int64_t voxelIdx,
                                                 GridBatchData::Accessor) {
                                     if (voxelIdx < r0 || voxelIdx >= r1)
                                         return;
                                     auto treeAcc       = acc.grid(batchIdx)->getAccessor();
                                     int64_t const base = acc.voxelOffset(batchIdx);
                                     int32_t *row       = scratch + (voxelIdx - r0) * K;
                                     for (int k = 0; k < K; ++k) {
                                         // Computed here: nvcc rejects a lambda's first capture
                                         // inside if constexpr.
                                         nanovdb::Coord const tap = geometry.tapCoord(k);
                                         nanovdb::Coord probe;
                                         if constexpr (Reverse) {
                                             if (!geometry.coarseFromFine(ijk, tap, probe)) {
                                                 row[k] = -1;
                                                 continue;
                                             }
                                         } else {
                                             probe = geometry.fineFromCoarse(ijk, tap);
                                         }
                                         row[k] = treeAcc.isActive(probe)
                                                      ? int32_t(base + treeAcc.getValue(probe) - 1)
                                                      : -1;
                                     }
                                 });
}

// Tap masks (rowWords[row, w].mask) and degrees of a probed batch, one warp per row: lane b reads
// tap 32w + b, so the warp's ballot is the row's mask word w.
__global__ void
batchMaskKernel(int32_t const *__restrict__ scratch,
                int64_t rows,
                int K,
                int W,
                int32_t *__restrict__ rowWords,
                int32_t *__restrict__ degrees) {
    int const lane      = threadIdx.x & 31;
    int64_t const warps = int64_t(gridDim.x) * (blockDim.x / 32);
    for (int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32; row < rows;
         row += warps) {
        int32_t const *r = scratch + row * K;
        int d            = 0;
        for (int w = 0; w < W; ++w) {
            int const k         = 32 * w + lane;
            uint32_t const bits = __ballot_sync(0xffffffffu, k < K && r[k] >= 0);
            if (lane == 0)
                rowWords[(row * W + w) * 2] = int32_t(bits);
            d += __popc(bits);
        }
        if (lane == 0)
            degrees[row] = d;
    }
}

// Write a probed batch's present neighbors at their batch-local offsets, one warp per row.
__global__ void
batchCompactKernel(int32_t const *__restrict__ scratch,
                   int64_t rows,
                   int K,
                   int32_t const *__restrict__ start,
                   int32_t *__restrict__ out) {
    int const lane      = threadIdx.x & 31;
    int64_t const warps = int64_t(gridDim.x) * (blockDim.x / 32);
    for (int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32; row < rows;
         row += warps) {
        int32_t const *r = scratch + row * K;
        int32_t pos      = start[row];
        for (int k0 = 0; k0 < K; k0 += 32) {
            int32_t const v     = k0 + lane < K ? r[k0 + lane] : -1;
            uint32_t const bits = __ballot_sync(0xffffffffu, v >= 0);
            if (v >= 0)
                out[pos + __popc(bits & bitsBelow(lane))] = v;
            pos += __popc(bits);
        }
    }
}

// rowWords[row, w].base from the row offsets and the masks of the words before w.
__global__ void
rowBaseKernel(int32_t const *__restrict__ offsets,
              int64_t N,
              int W,
              int32_t *__restrict__ rowWords) {
    for (int64_t row = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; row < N;
         row += int64_t(gridDim.x) * blockDim.x) {
        int32_t base = offsets[row];
        for (int w = 0; w < W; ++w) {
            rowWords[(row * W + w) * 2 + 1] = base;
            base += __popc(uint32_t(rowWords[(row * W + w) * 2]));
        }
    }
}

// keys[j, row] packs Gray-rank words 2j (low) and 2j+1 (high) of the row's whole tap mask into a
// signed int64 that sorts like the unsigned 64-bit value. A rank bit is the XOR of the mask bits at
// and above it, so a word's rank is its own prefix XOR flipped by the parity of the words above it.
__global__ void
multiWordGrayKeyKernel(int32_t const *__restrict__ rowWords,
                       int64_t N,
                       int W,
                       int64_t *__restrict__ keys) {
    for (int64_t row = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; row < N;
         row += int64_t(gridDim.x) * blockDim.x) {
        uint32_t parity = 0, high = 0;
        for (int w = W - 1; w >= 0; --w) {
            uint32_t const m    = uint32_t(rowWords[(row * W + w) * 2]);
            uint32_t const rank = grayRank(m) ^ (parity ? 0xffffffffu : 0u);
            parity ^= __popc(m) & 1u;
            if (w & 1) {
                high = rank;
            } else {
                // high is still 0 here when w == W - 1: words above W - 1 are zero.
                keys[int64_t(w / 2) * N + row] =
                    int64_t((uint64_t(high ^ 0x80000000u) << 32) | rank);
            }
        }
    }
}

// Rows ordered by the Gray rank of their whole tap mask: stable sorts from the least to the
// most significant 64-bit key.
torch::Tensor
multiWordRowOrder(torch::Tensor const &rowWords, int64_t N, int64_t W, cudaStream_t stream) {
    auto keys = torch::empty({(W + 1) / 2, N}, rowWords.options().dtype(torch::kInt64));
    if (N > 0) {
        multiWordGrayKeyKernel<<<gridStrideBlocks(N), kGridStrideThreads, 0, stream>>>(
            rowWords.data_ptr<int32_t>(), N, int(W), keys.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto perm = torch::arange(N, keys.options());
    for (int64_t j = 0; j < keys.size(0); ++j) {
        auto const order =
            std::get<1>(torch::sort(keys[j].index_select(0, perm), std::optional<bool>(true), 0));
        perm = perm.index_select(0, order);
    }
    return perm.to(torch::kInt32);
}

// Weight-gradient pair lists from the masks and compact map, as pairCountKernel/pairFillKernel.
// Dynamic shared memory holds the per-warp counts: kPairBlock / 32 * K int32.
__global__ void
pairCountMultiWordKernel(
    int32_t const *__restrict__ rowWords, int N, int K, int W, int32_t *__restrict__ blockCounts) {
    extern __shared__ int32_t warpCounts[]; // [kPairBlock / 32][K]
    int const row  = blockIdx.x * kPairBlock + threadIdx.x;
    int const warp = threadIdx.x >> 5;
    for (int w = 0; w < W; ++w) {
        uint32_t const m = row < N ? uint32_t(rowWords[(int64_t(row) * W + w) * 2]) : 0u;
        int const kEnd   = min(32, K - 32 * w);
        for (int b = 0; b < kEnd; ++b) {
            uint32_t const bits = __ballot_sync(0xffffffffu, (m >> b) & 1u);
            if ((threadIdx.x & 31) == 0)
                warpCounts[warp * K + 32 * w + b] = __popc(bits);
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += kPairBlock) {
        int sum = 0;
        for (int v = 0; v < kPairBlock / 32; ++v)
            sum += warpCounts[v * K + k];
        blockCounts[int64_t(k) * gridDim.x + blockIdx.x] = sum;
    }
}

__global__ void
pairFillMultiWordKernel(int32_t const *__restrict__ rowWords,
                        int32_t const *__restrict__ nbr,
                        int32_t const *__restrict__ offsets,
                        int32_t const *__restrict__ blockOffsets,
                        int N,
                        int K,
                        int W,
                        int32_t *__restrict__ pairIn,
                        int32_t *__restrict__ pairOut) {
    extern __shared__ int32_t warpCounts[]; // [kPairBlock / 32][K]
    int const row  = blockIdx.x * kPairBlock + threadIdx.x;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    for (int w = 0; w < W; ++w) {
        uint32_t const m = row < N ? uint32_t(rowWords[(int64_t(row) * W + w) * 2]) : 0u;
        int const kEnd   = min(32, K - 32 * w);
        for (int b = 0; b < kEnd; ++b) {
            uint32_t const bits = __ballot_sync(0xffffffffu, (m >> b) & 1u);
            if (lane == 0)
                warpCounts[warp * K + 32 * w + b] = __popc(bits);
        }
    }
    __syncthreads();
    int32_t src = row < N ? offsets[row] : 0; // this row's next compact entry
    for (int w = 0; w < W; ++w) {
        uint32_t const m = row < N ? uint32_t(rowWords[(int64_t(row) * W + w) * 2]) : 0u;
        int const kEnd   = min(32, K - 32 * w);
        for (int b = 0; b < kEnd; ++b) {
            int const k         = 32 * w + b;
            uint32_t const bits = __ballot_sync(0xffffffffu, (m >> b) & 1u);
            if (!((m >> b) & 1u))
                continue;
            int pos =
                blockOffsets[int64_t(k) * gridDim.x + blockIdx.x] + __popc(bits & bitsBelow(lane));
            for (int v = 0; v < warp; ++v)
                pos += warpCounts[v * K + k];
            pairOut[pos] = row;
            pairIn[pos]  = nbr[src++];
        }
    }
}

struct CompactMap {
    torch::Tensor nbr;      ///< int32 [P], present neighbors per row in tap order
    torch::Tensor offsets;  ///< int32 [rows + 1]
    torch::Tensor rowWords; ///< int32 [rows, W, 2], (mask, base) per 32-tap word
    torch::Tensor sorted;   ///< int32 [rows], rows by Gray rank of their tap mask
};

// Probe rowGrid's rows a batch at a time into a dense scratch, compact each batch, then derive the
// row offsets, word bases and Gray-rank row order.
template <bool Reverse>
CompactMap
buildCompactMap(GridBatchData const &rowGrid,
                GridBatchData const &probeGrid,
                ConvolutionGeometry const &geometry,
                cudaStream_t stream) {
    int64_t const K    = geometry.kernelVolume();
    int64_t const W    = maskWordsFor(K);
    int64_t const rows = rowGrid.totalVoxels();
    auto const i32     = torch::dtype(torch::kInt32).device(rowGrid.device());

    auto rowWords = torch::empty({rows, W, 2}, i32);
    auto degrees  = torch::empty({rows}, i32);
    std::vector<torch::Tensor> parts;
    int64_t const batchRows =
        std::max<int64_t>(1, kProbeBatchBytes / (K * int64_t(sizeof(int32_t))));
    auto scratch = torch::empty({std::min(batchRows, rows), K}, i32);
    for (int64_t r0 = 0; r0 < rows; r0 += batchRows) {
        int64_t const r1 = std::min(rows, r0 + batchRows), n = r1 - r0;
        probeBatchScan<Reverse>(
            rowGrid, probeGrid, geometry, int(K), r0, r1, scratch.data_ptr<int32_t>());
        batchMaskKernel<<<gridStrideBlocks(n * 32), kGridStrideThreads, 0, stream>>>(
            scratch.data_ptr<int32_t>(),
            n,
            int(K),
            int(W),
            rowWords.data_ptr<int32_t>() + r0 * W * 2,
            degrees.data_ptr<int32_t>() + r0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        torch::Tensor const d = degrees.slice(0, r0, r1);
        // Exclusive offsets within the batch; a batch holds fewer than 2^31 entries.
        torch::Tensor const start = torch::cumsum(d, 0, torch::kInt32) - d;
        torch::Tensor const last  = start[n - 1] + d[n - 1];
        int64_t const size        = last.item<int64_t>();
        torch::Tensor part        = torch::empty({size}, i32);
        if (size > 0) {
            batchCompactKernel<<<gridStrideBlocks(n * 32), kGridStrideThreads, 0, stream>>>(
                scratch.data_ptr<int32_t>(),
                n,
                int(K),
                start.data_ptr<int32_t>(),
                part.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        parts.push_back(std::move(part));
    }
    scratch  = torch::Tensor();
    auto nbr = parts.empty() ? torch::empty({0}, i32) : torch::cat(parts);
    parts    = {};

    auto offsets = torch::zeros({rows + 1}, i32);
    if (rows > 0) {
        TORCH_CHECK(nbr.size(0) < std::numeric_limits<int32_t>::max(),
                    "maskedIGemmSparseConvTopology: neighbor pairs exceed the int32 index");
        offsets.slice(0, 1).copy_(torch::cumsum(degrees, 0, torch::kInt32));
        rowBaseKernel<<<gridStrideBlocks(rows), kGridStrideThreads, 0, stream>>>(
            offsets.data_ptr<int32_t>(), rows, int(W), rowWords.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto sorted = multiWordRowOrder(rowWords, rows, W, stream);
    return {nbr, offsets, rowWords, sorted};
}

MaskedIGemmTopology
largeKernelTopology(GridBatchData const &featureGrid,
                    GridBatchData const &outputGrid,
                    ConvolutionGeometry const &geometry,
                    nanovdb::Coord kernelSize,
                    bool submanifold) {
    int64_t const K   = geometry.kernelVolume();
    int64_t const W   = maskWordsFor(K);
    int64_t const N   = outputGrid.totalVoxels();
    int64_t const NIn = featureGrid.totalVoxels();
    auto const i32    = torch::dtype(torch::kInt32).device(outputGrid.device());
    auto stream       = c10::cuda::getCurrentCUDAStream();

    // ---- Forward map: output rows, each listing the input rows it reads ----
    CompactMap const fwd = buildCompactMap<false>(outputGrid, featureGrid, geometry, stream);
    auto const &nbr      = fwd.nbr;
    auto const &offsets  = fwd.offsets;
    auto const &rowWords = fwd.rowWords;
    int64_t const P      = nbr.size(0);

    // ---- Weight-gradient pairs per tap, output rows ascending ----
    int64_t const blocks = (N + kPairBlock - 1) / kPairBlock;
    auto blockCounts     = torch::zeros({K, blocks}, i32);
    int const pairSmem   = int(kPairBlock / 32 * K * sizeof(int32_t));
    if (pairSmem > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            pairCountMultiWordKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pairSmem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            pairFillMultiWordKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pairSmem));
    }
    if (blocks > 0) {
        pairCountMultiWordKernel<<<blocks, kPairBlock, pairSmem, stream>>>(
            rowWords.data_ptr<int32_t>(), int(N), int(K), int(W), blockCounts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto pairOffsets = torch::zeros({K + 1}, i32);
    pairOffsets.slice(0, 1).copy_(torch::cumsum(blockCounts.sum(1, false, torch::kInt64), 0));
    auto blockOffsets = (torch::cumsum(blockCounts, 1, torch::kInt32) - blockCounts) +
                        pairOffsets.slice(0, 0, K).unsqueeze(1);
    auto pairIn  = torch::empty({P}, i32);
    auto pairOut = torch::empty({P}, i32);
    if (blocks > 0 && P > 0) {
        pairFillMultiWordKernel<<<blocks, kPairBlock, pairSmem, stream>>>(
            rowWords.data_ptr<int32_t>(),
            nbr.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            blockOffsets.data_ptr<int32_t>(),
            int(N),
            int(K),
            int(W),
            pairIn.data_ptr<int32_t>(),
            pairOut.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    MaskedIGemmTopology topo{.neighborMap        = nbr,
                             .sortedRows         = fwd.sorted,
                             .localRows          = windowedRowOrder(fwd.sorted),
                             .rowMasks           = rowWords.select(2, 0),
                             .pairIn             = pairIn,
                             .pairOut            = pairOut,
                             .pairOffsets        = pairOffsets,
                             .pairBlockOffsets   = blockOffsets,
                             .outputTotalVoxels  = N,
                             .featureTotalVoxels = NIn,
                             .kernelVolume       = K,
                             .kernelSize         = kernelSize,
                             .reverse            = nullptr,
                             .neighborOffsets    = offsets,
                             .rowWords           = rowWords};
    if (submanifold)
        return topo;

    // ---- Reverse map: input rows, each listing the output rows that read it ----
    CompactMap const rev = buildCompactMap<true>(featureGrid, outputGrid, geometry, stream);
    auto const none      = torch::empty({0}, i32);
    topo.reverse         = std::make_shared<MaskedIGemmTopology const>(
        MaskedIGemmTopology{.neighborMap        = rev.nbr,
                                    .sortedRows         = rev.sorted,
                                    .localRows          = windowedRowOrder(rev.sorted),
                                    .rowMasks           = rev.rowWords.select(2, 0),
                                    .pairIn             = none,
                                    .pairOut            = none,
                                    .pairOffsets        = none,
                                    .pairBlockOffsets   = none,
                                    .outputTotalVoxels  = NIn,
                                    .featureTotalVoxels = N,
                                    .kernelVolume       = K,
                                    .kernelSize         = kernelSize,
                                    .reverse            = nullptr,
                                    .neighborOffsets    = rev.offsets,
                                    .rowWords           = rev.rowWords});
    return topo;
}

} // namespace

// =============================================================================
// Public entry points
// =============================================================================

void
setMaskedIGemmTestOverrides(MaskedIGemmTestOverrides const &overrides) {
    TORCH_CHECK(!overrides.splits || *overrides.splits >= 1, "splits must be at least 1");
    TORCH_CHECK(!overrides.weightRowSplits || *overrides.weightRowSplits >= 1,
                "weight_row_splits must be at least 1");
    gTestOverrides = overrides;
}

torch::Tensor
maskedIGemmColumnSum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, "maskedIGemmColumnSum: x must be a 2-D CUDA tensor");
    int64_t const N = x.size(0), C = x.size(1);
    int const V     = 16 / int(x.element_size());
    bool const fast = (x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16 ||
                       x.scalar_type() == torch::kFloat32) &&
                      C > 0 && C % V == 0 && C / V <= kColumnSumThreads;
    // Below ~1M elements the three launches here cost more than torch's single reduction.
    if (!fast || N * C < (int64_t(1) << 20))
        return x.sum(0);
    c10::cuda::CUDAGuard guard(x.device());
    x = x.contiguous();
    if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0)
        return x.sum(0);
    auto out         = torch::zeros({C}, x.options().dtype(torch::kFloat32));
    int const lanes  = kColumnSumThreads / int(C / V);
    int const blocks = int(std::clamp<int64_t>((N + lanes - 1) / lanes, 1, 4 * int64_t(smCount())));
    auto const stream = c10::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == torch::kFloat16)
        columnSumKernel<__half><<<blocks, kColumnSumThreads, 0, stream>>>(
            reinterpret_cast<__half const *>(x.data_ptr()), N, int(C), out.data_ptr<float>());
    else if (x.scalar_type() == torch::kBFloat16)
        columnSumKernel<__nv_bfloat16><<<blocks, kColumnSumThreads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 const *>(x.data_ptr()),
            N,
            int(C),
            out.data_ptr<float>());
    else
        columnSumKernel<float><<<blocks, kColumnSumThreads, 0, stream>>>(
            x.data_ptr<float>(), N, int(C), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.to(x.scalar_type());
}

bool
maskedIGemmSupports(int64_t inChannels,
                    int64_t outChannels,
                    torch::ScalarType dtype,
                    torch::Device device) {
    bool const typeOk =
        dtype == torch::kFloat16 || dtype == torch::kBFloat16 || dtype == torch::kFloat32;
    if (!typeOk || !device.is_cuda() || inChannels <= 0 || outChannels <= 0 ||
        inChannels % 8 != 0 || outChannels % 8 != 0)
        return false;
    int index = device.has_index() ? device.index() : 0;
    if (!device.has_index())
        C10_CUDA_CHECK(cudaGetDevice(&index));
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, index));
    return major >= 8;
}

MaskedIGemmTopology
maskedIGemmSparseConvTopology(GridBatchData const &featureGrid,
                              GridBatchData const &outputGrid,
                              nanovdb::Coord kernelSize,
                              nanovdb::Coord stride) {
    TORCH_CHECK(outputGrid.device().is_cuda(),
                "maskedIGemmSparseConvTopology: grids must be on CUDA");
    TORCH_CHECK(featureGrid.device() == outputGrid.device(),
                "maskedIGemmSparseConvTopology: grids must be on the same device");
    TORCH_CHECK(featureGrid.batchSize() == outputGrid.batchSize(),
                "maskedIGemmSparseConvTopology: grids must have the same batch size");
    bool odd = true;
    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(kernelSize[d] > 0 && stride[d] > 0,
                    "maskedIGemmSparseConvTopology: kernel sizes and strides must be positive");
        odd &= kernelSize[d] % 2 == 1;
    }
    // A submanifold map is its own reverse with mirrored taps; other maps store the reverse.
    bool const submanifold = &featureGrid == &outputGrid && stride == nanovdb::Coord(1) && odd;
    ConvolutionGeometry const geometry(kernelSize, stride);
    int64_t const K   = geometry.kernelVolume();
    int64_t const N   = outputGrid.totalVoxels();
    int64_t const NIn = featureGrid.totalVoxels();
    // Kernels index rows as int32 and probe up to one block of rows past N.
    constexpr int64_t kMaxRows = std::numeric_limits<int32_t>::max() - kPairBlock;
    TORCH_CHECK(N <= kMaxRows && NIn <= kMaxRows, "maskedIGemmSparseConvTopology: too many voxels");

    c10::cuda::CUDAGuard guard(outputGrid.device());
    // Kernels of more than kMaxTaps taps keep compact maps (largeKernelTopology). Their kernels
    // hold per-tap state in shared memory, which bounds the kernel volume.
    if (K > kMaxTaps) {
        int device = 0, optin = 0;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        C10_CUDA_CHECK(
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        int64_t const need =
            std::max<int64_t>(int64_t(kPairBlock / 32) * K * int64_t(sizeof(int32_t)),
                              smemBytes<TileConfig<128, 128, 2, 2, 2>, float, true>(int(K)));
        TORCH_CHECK(need <= optin,
                    "maskedIGemmSparseConvTopology: a kernel of ",
                    K,
                    " taps needs ",
                    need,
                    " bytes of shared memory per block; this device allows ",
                    optin);
    }
    if (K > kMaxTaps)
        return largeKernelTopology(featureGrid, outputGrid, geometry, kernelSize, submanifold);
    auto const i32 = torch::dtype(torch::kInt32).device(outputGrid.device());
    auto nbr       = torch::empty({N, K}, i32);
    auto code      = torch::empty({N}, i32);
    auto rowMasks  = torch::empty({N}, i32);
    auto stream    = c10::cuda::getCurrentCUDAStream();

    if (N > 0) {
        // The staged builder needs a submanifold map with every tap within one leaf of its
        // voxel. Below ~128K voxels the VoxelBlockManager's cold build (a host sync) costs more
        // than the probes it saves.
        bool withinLeaf = submanifold;
        for (int d = 0; d < 3; ++d)
            withinLeaf &= (kernelSize[d] - 1) / 2 <= int(nanovdb::OnIndexTree::LeafNodeType::DIM);
        bool scan = !withinLeaf || N < (int64_t(1) << 17);
        if (gTestOverrides.stagedBuilder)
            scan = !(*gTestOverrides.stagedBuilder && withinLeaf);
        if (scan) {
            buildNeighborMapScan(featureGrid,
                                 outputGrid,
                                 geometry,
                                 int(K),
                                 nbr.data_ptr<int32_t>(),
                                 code.data_ptr<int32_t>(),
                                 rowMasks.data_ptr<int32_t>());
        } else {
            buildNeighborMapStaged(outputGrid,
                                   geometry,
                                   int(K),
                                   nbr.data_ptr<int32_t>(),
                                   code.data_ptr<int32_t>(),
                                   rowMasks.data_ptr<int32_t>(),
                                   stream);
        }
    }

    auto sortedRows = torch::argsort(code).to(torch::kInt32);
    // Weight-gradient pairs per tap, output rows ascending (leaf order).
    int64_t const blocks = (N + kPairBlock - 1) / kPairBlock;
    auto blockCounts     = torch::zeros({K, blocks}, i32); // tap-major, so scans run along rows
    if (blocks > 0) {
        pairCountKernel<<<blocks, kPairBlock, 0, stream>>>(
            rowMasks.data_ptr<int32_t>(), int(N), int(K), blockCounts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto tapEnds    = torch::cumsum(blockCounts.sum(1, false, torch::kInt64), 0);
    int64_t const P = tapEnds[K - 1].item<int64_t>();
    TORCH_CHECK(P < std::numeric_limits<int32_t>::max(),
                "maskedIGemmSparseConvTopology: ",
                P,
                " neighbor pairs exceed the int32 pair index");
    auto offsets = torch::zeros({K + 1}, i32);
    offsets.slice(0, 1).copy_(tapEnds);
    auto blockOffsets = (torch::cumsum(blockCounts, 1, torch::kInt32) - blockCounts) +
                        offsets.slice(0, 0, K).unsqueeze(1);
    auto pairIn  = torch::empty({P}, i32);
    auto pairOut = torch::empty({P}, i32);
    if (blocks > 0 && P > 0) {
        pairFillKernel<<<blocks, kPairBlock, 0, stream>>>(rowMasks.data_ptr<int32_t>(),
                                                          nbr.data_ptr<int32_t>(),
                                                          blockOffsets.data_ptr<int32_t>(),
                                                          int(N),
                                                          int(K),
                                                          pairIn.data_ptr<int32_t>(),
                                                          pairOut.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    MaskedIGemmTopology topo{.neighborMap        = nbr,
                             .sortedRows         = sortedRows,
                             .localRows          = windowedRowOrder(sortedRows),
                             .rowMasks           = rowMasks,
                             .pairIn             = pairIn,
                             .pairOut            = pairOut,
                             .pairOffsets        = offsets,
                             .pairBlockOffsets   = blockOffsets,
                             .outputTotalVoxels  = N,
                             .featureTotalVoxels = NIn,
                             .kernelVolume       = K,
                             .kernelSize         = kernelSize,
                             .reverse            = nullptr};
    if (submanifold)
        return topo;

    auto rev      = torch::full({NIn, K}, -1, i32);
    auto revCode  = torch::empty({NIn}, i32);
    auto revMasks = torch::empty({NIn}, i32);
    if (N > 0 && NIn > 0) {
        reverseMapKernel<<<gridStrideBlocks(N * K), kGridStrideThreads, 0, stream>>>(
            nbr.data_ptr<int32_t>(), N * K, int(K), rev.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (NIn > 0) {
        rowMaskKernel<<<gridStrideBlocks(NIn), kGridStrideThreads, 0, stream>>>(
            rev.data_ptr<int32_t>(),
            NIn,
            int(K),
            revCode.data_ptr<int32_t>(),
            revMasks.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto const none          = torch::empty({0}, i32);
    auto const revSortedRows = torch::argsort(revCode).to(torch::kInt32);
    topo.reverse             = std::make_shared<MaskedIGemmTopology const>(
        MaskedIGemmTopology{.neighborMap        = rev,
                                        .sortedRows         = revSortedRows,
                                        .localRows          = windowedRowOrder(revSortedRows),
                                        .rowMasks           = revMasks,
                                        .pairIn             = none,
                                        .pairOut            = none,
                                        .pairOffsets        = none,
                                        .pairBlockOffsets   = none,
                                        .outputTotalVoxels  = NIn,
                                        .featureTotalVoxels = N,
                                        .kernelVolume       = K,
                                        .kernelSize         = kernelSize,
                                        .reverse            = nullptr});
    return topo;
}

torch::Tensor
maskedIGemmSparseConv(torch::Tensor features,
                      torch::Tensor weights,
                      std::optional<torch::Tensor> bias,
                      MaskedIGemmTopology const &topo,
                      bool transposed,
                      bool precise) {
    checkConv(features, weights, topo, transposed, "maskedIGemmSparseConv");
    c10::cuda::CUDAGuard guard(features.device());
    auto const st = features.scalar_type();
    features      = aligned16(features);
    auto Wrows    = aligned16(weightRows(weights));
    int const Ci = int(weights.size(1)), Co = int(weights.size(0));
    torch::Tensor b;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Co,
                    "maskedIGemmSparseConv: bias must be [C_out]");
        TORCH_CHECK(bias->device() == features.device() && bias->scalar_type() == st,
                    "maskedIGemmSparseConv: bias must match the features' device and dtype");
        b = bias->contiguous();
    }
    // The transpose runs the forward kernel over the input rows' map.
    RowMap const map = transposed ? inputRows(topo) : RowMap{topo, false};
    auto out         = torch::empty({map.topo.outputTotalVoxels, Co}, features.options());
    if (map.topo.outputTotalVoxels > 0)
        runConv<false>(
            features, Wrows, b, out, map, Ci, Co, precise, c10::cuda::getCurrentCUDAStream());
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
maskedIGemmSparseConvBackward(torch::Tensor gradOutput,
                              torch::Tensor features,
                              torch::Tensor weights,
                              MaskedIGemmTopology const &topo,
                              bool needFeatures,
                              bool needWeights,
                              bool transposed,
                              bool precise) {
    checkConv(features, weights, topo, transposed, "maskedIGemmSparseConvBackward");
    int64_t const outRows = transposed ? topo.featureTotalVoxels : topo.outputTotalVoxels;
    TORCH_CHECK(gradOutput.dim() == 2 && gradOutput.size(0) == outRows &&
                    gradOutput.size(1) == weights.size(0),
                "maskedIGemmSparseConvBackward: grad_output must be [N_out, C_out]");
    TORCH_CHECK(gradOutput.device() == features.device(),
                "maskedIGemmSparseConvBackward: grad_output must be on the features' device");
    c10::cuda::CUDAGuard guard(features.device());
    auto const st = features.scalar_type();
    features      = aligned16(features);
    gradOutput    = aligned16(gradOutput.to(st));
    int const Ci = int(weights.size(1)), Co = int(weights.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();

    torch::Tensor gradFeatures, gradWeights;
    if (needFeatures) {
        // The input gradient runs the transposed weights over the other direction's map.
        RowMap const map = transposed ? RowMap{topo, false} : inputRows(topo);
        gradFeatures     = torch::empty({features.size(0), Ci}, features.options());
        if (map.topo.outputTotalVoxels > 0)
            runConv<true>(gradOutput,
                          aligned16(weightRows(weights)),
                          {},
                          gradFeatures,
                          map,
                          Co,
                          Ci,
                          precise,
                          stream);
    }
    if (needWeights) {
        // Pairs gather input rows from the forward convolution's input. For the transpose
        // that is grad_output, so swap the operands and transpose each tap's product.
        auto const gradRows = transposed
                                  ? runWgrad(gradOutput, features, topo, Co, Ci, precise, stream)
                                        .transpose(1, 2)
                                        .contiguous()
                                  : runWgrad(features, gradOutput, topo, Ci, Co, precise, stream);
        gradWeights         = weightGradFromRows(gradRows, weights);
    }
    return {gradFeatures, gradWeights};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
