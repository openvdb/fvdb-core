// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// MaskedIGemm.cu -- Masked implicit-GEMM submanifold sparse convolution.
//
// Kernels use SM80 warp-level tensor-core instructions (cp.async, ldmatrix,
// mma.sync m16n8k16), which also run on SM89/SM90/SM100/SM120.
//
// Forward and input gradient share one output-stationary kernel:
//   out[o, n] = sum_{u in taps(tile)} sum_k A[nbr[o, u], k] * B_u[k, n]
// Forward reads A = features and B_u = W[u] (C_in x C_out). The input gradient
// reads A = grad_output and B_u = W[K-1-u]^T, which is exact for submanifold
// convolution with odd kernel sizes because nbr[o, u] = i iff nbr[i, K-1-u] = o.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

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

torch::Tensor
weightsFromRows(torch::Tensor const &rows, nanovdb::Coord const &ks) {
    return rows.view({ks[2], ks[1], ks[0], rows.size(1), rows.size(2)}).permute({4, 3, 2, 1, 0});
}

// flip maps tap k to the weights of tap K-1-k, its mirror image. A submanifold transposed
// convolution is the forward convolution with every tap flipped.
struct TapToRow {
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
// PTX helpers
// =============================================================================
//
// SM80 instructions. Architectures below SM80 compile empty stubs; the host
// never launches these kernels there (maskedIGemmSupports checks the device).

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define FVDB_MIGEMM_SM80 1
#else
#define FVDB_MIGEMM_SM80 0
#endif

__device__ __forceinline__ uint32_t
smemAddr(void const *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async copy; zero-fills the destination when pred is false.
__device__ __forceinline__ void
cpAsync16(uint32_t dst, void const *src, bool pred) {
#if FVDB_MIGEMM_SM80
    int const bytes = pred ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#endif
}

__device__ __forceinline__ void
cpAsyncCommit() {
#if FVDB_MIGEMM_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void
cpAsyncWait() {
#if FVDB_MIGEMM_SM80
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void
ldsm4(uint32_t (&r)[4], uint32_t addr) {
#if FVDB_MIGEMM_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void
ldsm4T(uint32_t (&r)[4], uint32_t addr) {
#if FVDB_MIGEMM_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#endif
}

template <typename T> struct Mma;

template <> struct Mma<__half> {
    __device__ __forceinline__ static void
    run(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
#if FVDB_MIGEMM_SM80
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }
};

template <> struct Mma<__nv_bfloat16> {
    __device__ __forceinline__ static void
    run(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
#if FVDB_MIGEMM_SM80
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }
};

// float32 runs as TF32: m16n8k8 with operands rounded to TF32 after loading.
template <> struct Mma<float> {
    __device__ __forceinline__ static void
    run(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
#if FVDB_MIGEMM_SM80
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }
};

// Per-element-type tile constants. Every k-tile row is 64 bytes (four 16-byte
// chunks), so shared-memory layouts and ldmatrix addressing match across types,
// and each k-tile holds two MMA k-steps.
template <typename T> struct Elem {
    static constexpr int CHUNK = 16 / int(sizeof(T)); // elements per 16-byte chunk
    static constexpr int BK    = 4 * CHUNK;           // k-tile width in elements
    static constexpr int KSTEP = BK / 2;              // MMA k per instruction
};

// Round a loaded fragment register to TF32 (no-op for 16-bit types). Adding half an ulp lets
// the tensor core's truncation of the low 13 bits round to nearest, ties away from zero, as
// cvt.rna.tf32.f32 does, for less issue cost (CUTLASS's round_half_ulp_truncate). Inf and NaN
// skip the add, which would carry a NaN payload into the sign bit.
template <typename T>
__device__ __forceinline__ uint32_t
prep(uint32_t x) {
    if constexpr (sizeof(T) == 4) {
        if (isfinite(__uint_as_float(x)))
            x += 0x1000u;
    }
    return x;
}

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
//   Rows of 4 chunks (32 elements): XOR with bits 1-2 of the row.
//   Rows of 8+ chunks: XOR the low 3 chunk bits with the row's low 3 bits.
__device__ __forceinline__ int
swz4(int row, int chunk) {
    return chunk ^ ((row >> 1) & 3);
}

__device__ __forceinline__ int
swz8(int row, int chunk) {
    return chunk ^ (row & 7);
}

// [k][m|n] tiles. 16-bit types read them with ldmatrix.trans and use swz8. float32 reads
// them element-wise, with lane (g, t) reading rows t and t + 4; XOR with bits 0-1 of the row,
// shifted past the chunk's low bit, so those 32 lanes hit distinct banks.
template <typename T>
__device__ __forceinline__ int
swzKN(int row, int chunk) {
    if constexpr (sizeof(T) == 4)
        return chunk ^ ((row & 3) << 1);
    else
        return swz8(row, chunk);
}

// [k][m|n] tiles W elements wide. 16-bit rows of an odd multiple of 4 chunks (W = 32, 96, ...)
// alternate between the two halves of the 128-byte bank window, so swz4 keeps 8 consecutive
// rows on distinct banks; rows of a multiple of 8 chunks use swzKN.
template <typename T, int W>
__device__ __forceinline__ int
swzRow(int row, int chunk) {
    constexpr int chunks = W * int(sizeof(T)) / 16;
    static_assert(chunks % 4 == 0, "rows must span a multiple of 4 chunks");
    if constexpr (sizeof(T) == 2 && chunks % 8 != 0)
        return swz4(row, chunk);
    else
        return swzKN<T>(row, chunk);
}

// =============================================================================
// Output-stationary masked implicit GEMM (forward and input gradient)
// =============================================================================

constexpr int kMaxTaps = 32;

template <int BM_, int BN_, int WM_, int WN_, int STAGES_> struct TileConfig {
    static constexpr int BM      = BM_;
    static constexpr int BN      = BN_;
    static constexpr int WM      = WM_;
    static constexpr int WN      = WN_;
    static constexpr int STAGES  = STAGES_;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT      = BM / WM / 16; // m16 tiles per warp
    static constexpr int NT      = BN / WN / 8;  // n8 tiles per warp
    static_assert(NT % 2 == 0, "B fragments load two n8 tiles at a time");
    static_assert(BN >= 32, "B rows must span at least 4 chunks");
};

template <typename Cfg, typename T>
constexpr int
smemBytes(int K) {
    return Cfg::STAGES * (Cfg::BM + Cfg::BN) * Elem<T>::BK * int(sizeof(T)) + Cfg::BM * K * 4 +
           Cfg::BM * 4 + 2 * kMaxTaps * 4 + 16;
}

// B_KMAJOR selects the weight operand layout per tap:
//   false (forward):        B[k][n] at W + row * Kc * Nc + k * Nc + n
//   true  (input gradient): B[k][n] at W + row * Kc * Nc + n * Kc + k
// FLIP reads tap u's weights from tap K-1-u (input gradient).
// With splits > 1, CTA z of a tile runs its share of the tile's (tap, k-tile)
// iterations and adds float32 partials into outAcc instead of writing out.
template <typename Cfg, typename T, bool B_KMAJOR, bool FLIP>
__global__ void __launch_bounds__(Cfg::THREADS)
maskedIGemmKernel(T const *__restrict__ A,
                  T const *__restrict__ W,
                  T const *__restrict__ bias,
                  T *__restrict__ out,
                  float *__restrict__ outAcc,
                  int32_t const *__restrict__ nbr,
                  int32_t const *__restrict__ sortedRows,
                  int32_t const *__restrict__ rowMasks,
                  int N,
                  int Kc,
                  int Nc,
                  int K,
                  int splits,
                  TapToRow tapToRow) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Elem<T>::BK, STAGES = Cfg::STAGES;
    constexpr int CE = Elem<T>::CHUNK, KSTEP = Elem<T>::KSTEP;
    constexpr int THREADS = Cfg::THREADS, MT = Cfg::MT, NT = Cfg::NT;
    constexpr int A_STAGE = BM * BK, B_STAGE = BK * BN;

    extern __shared__ __align__(128) unsigned char smem[];
    T *sA           = reinterpret_cast<T *>(smem);
    T *sB           = sA + STAGES * A_STAGE;
    int32_t *sNbr   = reinterpret_cast<int32_t *>(sB + STAGES * B_STAGE);
    int32_t *sRows  = sNbr + BM * K;
    int32_t *sTaps  = sRows + BM;
    int32_t *sWRow  = sTaps + kMaxTaps;
    uint32_t *sMask = reinterpret_cast<uint32_t *>(sWRow + kMaxTaps);
    int32_t *sNTaps = reinterpret_cast<int32_t *>(sMask + 1);

    int const tid   = threadIdx.x;
    int const lane  = tid & 31;
    int const warp  = tid >> 5;
    int const nTile = blockIdx.x % (Nc / BN);
    int const mTile = blockIdx.x / (Nc / BN);
    int const row0  = mTile * BM;
    int const n0    = nTile * BN;

    // ---- Stage the tile's rows, tap union, and neighbor columns of the union ----
    if (tid == 0)
        *sMask = 0;
    __syncthreads();
    for (int r = tid; r < BM; r += THREADS) {
        int const o = (row0 + r < N) ? sortedRows[row0 + r] : -1;
        sRows[r]    = o;
        uint32_t m  = o >= 0 ? uint32_t(rowMasks[o]) : 0u;
#if FVDB_MIGEMM_SM80
        m = __reduce_or_sync(__activemask(), m);
#endif
        if (lane == 0 && m)
            atomicOr(sMask, m);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t m = *sMask;
        int n      = 0;
        while (m) {
            int const tap = __ffs(m) - 1;
            sTaps[n]      = tap;
            sWRow[n++]    = tapToRow(FLIP ? K - 1 - tap : tap);
            m &= m - 1;
        }
        *sNTaps = n;
    }
    __syncthreads();
    int const nTaps = *sNTaps;
    for (int i = tid; i < BM * nTaps; i += THREADS) {
        int const r = i / nTaps;
        int const o = sRows[r];
        sNbr[i]     = o >= 0 ? nbr[int64_t(o) * K + sTaps[i - r * nTaps]] : -1;
    }
    __syncthreads();

    int const kTiles = Kc / BK;
    int const total  = nTaps * kTiles;
    int const itBase = int(int64_t(total) * blockIdx.y / splits);
    int const nIter  = int(int64_t(total) * (blockIdx.y + 1) / splits) - itBase;
    if (splits > 1 && nIter <= 0)
        return;

    // ---- Async tile loads ----
    // Each thread copies A_CHUNKS 16-byte chunks of A (rows aR + i * A_RSTEP, chunk aCh) and
    // B_CHUNKS of B per stage. The loader walks (tap, k-tile) in order: pointers advance one
    // k-tile per load and are rebuilt from the neighbor columns and weight rows only when the
    // tap changes, so the loop does no division.
    constexpr int A_CHUNKS = BM * 4 / THREADS, A_RSTEP = THREADS / 4;
    constexpr int B_CHUNKS = BN * 4 / THREADS;
    constexpr int B_ROW    = B_KMAJOR ? 4 : BN / CE; // chunks per B shared-memory row
    // When the threads tile B rows, each thread's chunks share a column and step by B_RSTEP
    // rows; otherwise (B rows of 12 chunks) each chunk keeps its own offsets.
    constexpr bool B_TILED = THREADS % B_ROW == 0;
    constexpr int B_RSTEP  = B_TILED ? THREADS / B_ROW : 0;
    static_assert(BM * 4 % THREADS == 0 && BN * 4 % THREADS == 0, "chunks must tile the CTA");
    static_assert(A_RSTEP % 8 == 0, "per-thread swizzles must repeat");
    static_assert(B_TILED || !B_KMAJOR, "k-major B rows are 4 chunks");

    int const aR = tid >> 2, aCh = tid & 3;
    int const bR = tid / B_ROW, bCh = tid % B_ROW;
    int const aDst = aR * BK + swz4(aR, aCh) * CE;
    // B source offset of chunk 0 within a tap's weight block, the step between chunks, and
    // the step between k-tiles.
    int64_t const bOff =
        B_KMAJOR ? int64_t(n0 + bR) * Kc + bCh * CE : int64_t(bR) * Nc + n0 + bCh * CE;
    int64_t const bStep = B_KMAJOR ? int64_t(B_RSTEP) * Kc : int64_t(B_RSTEP) * Nc;
    int64_t const bKt   = B_KMAJOR ? int64_t(BK) : int64_t(BK) * Nc;
    int bDst[B_TILED ? 1 : B_CHUNKS], bSrc[B_TILED ? 1 : B_CHUNKS];
    if constexpr (!B_TILED) {
#pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            int const c = tid + i * THREADS, r = c / B_ROW, ch = c % B_ROW;
            bDst[i] = r * BN + swzRow<T, BN>(r, ch) * CE;
            bSrc[i] = r * Nc + ch * CE;
        }
    }

    int ldTi = itBase / kTiles, ldKt = itBase - ldTi * kTiles;
    T const *aPtr[A_CHUNKS];
    bool aOk[A_CHUNKS];
    T const *bPtr = nullptr;
    auto setTap   = [&]() {
#pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            int const src = sNbr[(aR + i * A_RSTEP) * nTaps + ldTi];
            aOk[i]        = src >= 0;
            aPtr[i]       = A + (aOk[i] ? int64_t(src) * Kc + aCh * CE : 0) + ldKt * BK;
        }
        bPtr = W + int64_t(sWRow[ldTi]) * Kc * Nc + (B_TILED ? bOff : n0) + ldKt * bKt;
    };
    if (nIter > 0)
        setTap();

    auto load = [&](int stage) {
        T *a = sA + stage * A_STAGE;
#pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            cpAsync16(smemAddr(a + aDst + i * A_RSTEP * BK), aPtr[i], aOk[i]);
            aPtr[i] += BK;
        }
        T *b = sB + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            if constexpr (!B_TILED) {
                cpAsync16(smemAddr(b + bDst[i]), bPtr + bSrc[i], true);
            } else {
                int const r = bR + i * B_RSTEP;
                if constexpr (B_KMAJOR)
                    cpAsync16(smemAddr(b + r * BK + swz4(r, bCh) * CE), bPtr + i * bStep, true);
                else
                    cpAsync16(
                        smemAddr(b + r * BN + swzRow<T, BN>(r, bCh) * CE), bPtr + i * bStep, true);
            }
        }
        bPtr += bKt;
        if (++ldKt == kTiles) {
            ldKt = 0;
            if (++ldTi < nTaps)
                setTap();
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
        cpAsyncCommit();
    }

    for (int it = 0; it < nIter; ++it) {
        cpAsyncWait<STAGES - 2>();
        __syncthreads();
        int const next = it + STAGES - 1;
        if (next < nIter)
            load(next % STAGES);
        cpAsyncCommit();

        T const *a = sA + (it % STAGES) * A_STAGE;
        T const *b = sB + (it % STAGES) * B_STAGE;
        // Each k-step spans two 16-byte chunks for every element type; for float32,
        // ldmatrix's 8x8 16-bit tiles are 8x4 float tiles, which match TF32 fragments.
#pragma unroll
        for (int kk = 0; kk < BK / KSTEP; ++kk) {
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int const r  = wm0 + mt * 16 + (lane & 15);
                int const ch = kk * 2 + (lane >> 4);
                ldsm4(af[mt], smemAddr(a + r * BK + swz4(r, ch) * CE));
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    af[mt][e] = prep<T>(af[mt][e]);
            }
            uint32_t bf[NT][2];
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t t[4];
                if constexpr (B_KMAJOR) {
                    int const nr = wn0 + np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    int const ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm4(t, smemAddr(b + nr * BK + swz4(nr, ch) * CE));
                } else if constexpr (sizeof(T) == 4) {
                    // ldmatrix cannot transpose 32-bit data; read N-major fragments
                    // element-wise: b0 = B[t][g], b1 = B[t + 4][g].
                    int const g = lane >> 2, k = kk * KSTEP + (lane & 3);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int const n  = wn0 + np * 16 + h * 8 + g;
                        auto const e = [&](int kr) {
                            return __float_as_uint(
                                b[kr * BN + swzRow<T, BN>(kr, n / CE) * CE + n % CE]);
                        };
                        t[2 * h]     = e(k);
                        t[2 * h + 1] = e(k + 4);
                    }
                } else {
                    int const kr = kk * KSTEP + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int const ch = (wn0 + np * 16) / CE + (lane >> 4);
                    ldsm4T(t, smemAddr(b + kr * BN + swzRow<T, BN>(kr, ch) * CE));
                }
                bf[2 * np][0]     = prep<T>(t[0]);
                bf[2 * np][1]     = prep<T>(t[1]);
                bf[2 * np + 1][0] = prep<T>(t[2]);
                bf[2 * np + 1][1] = prep<T>(t[3]);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    Mma<T>::run(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
        }
    }
    cpAsyncWait<0>();

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
    static_assert(NT % 2 == 0 && BM >= 32 && BN >= 32 && BK <= 32, "tile shape");
};

template <typename Cfg, typename T>
constexpr int
wgradSmemBytes() {
    return Cfg::STAGES * (Cfg::BK * Cfg::BM + Cfg::BK * Cfg::BN) * int(sizeof(T)) +
           Cfg::STAGES * 2 * Cfg::BK * 4;
}

template <typename Cfg, typename T>
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
                       TapToRow tapToRow) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, STAGES = Cfg::STAGES;
    constexpr int CE = Elem<T>::CHUNK, KSTEP = Elem<T>::KSTEP;
    constexpr int THREADS = Cfg::THREADS, MT = Cfg::MT, NT = Cfg::NT;
    constexpr int A_STAGE = BK * BM, B_STAGE = BK * BN;

    extern __shared__ __align__(128) unsigned char smem[];
    T *sA         = reinterpret_cast<T *>(smem);
    T *sB         = sA + STAGES * A_STAGE;
    int32_t *sIdx = reinterpret_cast<int32_t *>(sB + STAGES * B_STAGE); // [STAGES][2][BK]

    int const tid   = threadIdx.x;
    int const lane  = tid & 31;
    int const warp  = tid >> 5;
    int const nTile = blockIdx.x % (Co / BN);
    int const mTile = blockIdx.x / (Co / BN);
    int const tap   = blockIdx.y;
    int const m0    = mTile * BM;
    int const n0    = nTile * BN;

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
        for (int i = 0; i < (BK * BM / CE + THREADS - 1) / THREADS; ++i) {
            int const c = tid + i * THREADS;
            if (c < BK * BM / CE) {
                int const kr  = c / (BM / CE);
                int const ch  = c % (BM / CE);
                int const src = slot[kr];
                T const *g    = X + (src >= 0 ? int64_t(src) * Ci + m0 + ch * CE : 0);
                cpAsync16(smemAddr(a + kr * BM + swzRow<T, BM>(kr, ch) * CE), g, src >= 0);
            }
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / CE + THREADS - 1) / THREADS; ++i) {
            int const c = tid + i * THREADS;
            if (c < BK * BN / CE) {
                int const kr = c / (BN / CE);
                int const ch = c % (BN / CE);
                int const o  = slot[BK + kr];
                T const *g   = dY + (o >= 0 ? int64_t(o) * Co + n0 + ch * CE : 0);
                cpAsync16(smemAddr(b + kr * BN + swzRow<T, BN>(kr, ch) * CE), g, o >= 0);
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
                aOff[mt][r] = t * BM + swzRow<T, BM>(t, m / CE) * CE + m % CE;
            }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int const n = wn0 + nt * 8 + g;
            bOff[nt]    = t * BN + swzRow<T, BN>(t, n / CE) * CE + n % CE;
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
        cpAsyncCommit();
    }

    for (int it = 0; it < nIter; ++it) {
        cpAsyncWait<STAGES - 2>();
        __syncthreads();
        int const next = it + STAGES - 1;
        if (next < nIter)
            load(next % STAGES, next);
        cpAsyncCommit();

        // Indices for step it + STAGES reuse the slot of step it, whose loads were issued
        // long ago. Fetch now, store after the MMAs so the fetch latency overlaps them.
        int pfIn = -1, pfOut = -1;
        if (warp == 0)
            fetchIdx(it + STAGES, pfIn, pfOut);

        T const *a = sA + (it % STAGES) * A_STAGE;
        T const *b = sB + (it % STAGES) * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK / KSTEP; ++kk) {
            uint32_t af[MT][4];
            uint32_t bf[NT][2];
            if constexpr (sizeof(T) == 4) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int const k = kk * KSTEP + 4 * h;
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        af[mt][2 * h]     = prep<T>(__float_as_uint(a[aOff[mt][0] + k * BM]));
                        af[mt][2 * h + 1] = prep<T>(__float_as_uint(a[aOff[mt][1] + k * BM]));
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                        bf[nt][h] = prep<T>(__float_as_uint(b[bOff[nt] + k * BN]));
                }
            } else {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    int const kr = kk * KSTEP + (lane & 7) + ((lane >> 4) & 1) * 8;
                    int const ch = (wm0 + mt * 16) / CE + ((lane >> 3) & 1);
                    ldsm4T(af[mt], smemAddr(a + kr * BM + swzRow<T, BM>(kr, ch) * CE));
                }
#pragma unroll
                for (int np = 0; np < NT / 2; ++np) {
                    uint32_t t[4];
                    int const kr = kk * KSTEP + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int const ch = (wn0 + np * 16) / CE + (lane >> 4);
                    ldsm4T(t, smemAddr(b + kr * BN + swzRow<T, BN>(kr, ch) * CE));
                    bf[2 * np][0]     = t[0];
                    bf[2 * np][1]     = t[1];
                    bf[2 * np + 1][0] = t[2];
                    bf[2 * np + 1][1] = t[3];
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    Mma<T>::run(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
        }

        if (warp == 0)
            storeIdx(it + STAGES, pfIn, pfOut);
    }
    cpAsyncWait<0>();

    int64_t const tapBase = int64_t(tapToRow(tap)) * Ci * Co;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int const m = m0 + wm0 + mt * 16 + (lane >> 2) + h * 8;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int const n       = n0 + wn0 + nt * 8 + (lane & 3) * 2;
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
        return; // defensive; the lastOffset guard makes this unreachable

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
        auto const vbm = grid.vbmCache().get(grid, bi);
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

// Leaf-scan builder: one thread per leaf slot, K tree probes per active voxel.
void
buildNeighborMapScan(GridBatchData const &grid,
                     ConvolutionGeometry const &geometry,
                     int K,
                     int32_t *nbr,
                     int32_t *code,
                     int32_t *masks) {
    using CudaTag  = ::dispatch::tag<torch::kCUDA>;
    auto const acc = dispatch::make_grid_accessor(CudaTag{}, grid);
    dispatch::forEachActiveVoxel(CudaTag{},
                                 grid,
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
            blockOffsets[int64_t(k) * gridDim.x + blockIdx.x] + __popc(bits & ((1u << lane) - 1u));
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
// unions. FVDB_MIGEMM_ORDER=global|local overrides the choice.
constexpr int64_t kLocalWindow = int64_t(1) << 16;

torch::Tensor const &
rowOrder(MaskedIGemmTopology const &topo, int64_t gatheredBytes) {
    int device = 0, l2 = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    bool local = topo.totalVoxels > kLocalWindow && gatheredBytes >= 3 * int64_t(l2);
    if (char const *env = std::getenv("FVDB_MIGEMM_ORDER"))
        local = std::string(env) == "local";
    if (!local)
        return topo.sortedRows;
    if (!topo.localRows.defined()) {
        auto const window = torch::div(topo.sortedRows, kLocalWindow, "floor");
        auto const perm   = std::get<1>(torch::sort(window, std::optional<bool>(true), 0));
        topo.localRows    = topo.sortedRows.index_select(0, perm);
    }
    return topo.localRows;
}

template <typename Cfg, typename T, bool B_KMAJOR, bool FLIP>
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
    int const N    = int(topo.totalVoxels);
    int const K    = int(topo.kernelVolume);
    int const smem = smemBytes<Cfg, T>(K);
    auto kernel    = maskedIGemmKernel<Cfg, T, B_KMAJOR, FLIP>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int const tiles = ((N + Cfg::BM - 1) / Cfg::BM) * (Nc / Cfg::BN);

    // Split each tile's (tap, k-tile) iterations to fill the last wave. Cost is in k-tile units:
    // each wave runs blocks of iters / s k-tiles plus a fixed prologue and epilogue (a float32
    // atomic tile when split), and splitting adds a zero-fill and copy pass over the output.
    // FVDB_MIGEMM_SPLIT forces the split count for tuning.
    constexpr double kBlockOverhead = 24.0;
    constexpr double kSplitPass     = 1.1;
    int perSm                       = 0;
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, Cfg::THREADS, smem));
    int const slots     = std::max(1, perSm) * smCount();
    int const kTiles    = Kc / Elem<T>::BK;
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
    if (char const *env = std::getenv("FVDB_MIGEMM_SPLIT"))
        splits = std::clamp(std::atoi(env), 1, maxSplits);
    torch::Tensor acc;
    if (splits > 1)
        acc = torch::zeros({N, Nc}, out.options().dtype(torch::kFloat32));

    TapToRow const ttr{topo.kernelSize[1], topo.kernelSize[2], topo.kernelSize[0], flipTaps};
    dim3 const grid(tiles, splits);
    kernel<<<grid, Cfg::THREADS, smem, stream>>>(
        reinterpret_cast<T const *>(A.data_ptr()),
        reinterpret_cast<T const *>(Wrows.data_ptr()),
        bias.defined() ? reinterpret_cast<T const *>(bias.data_ptr()) : nullptr,
        reinterpret_cast<T *>(out.data_ptr()),
        splits > 1 ? acc.data_ptr<float>() : nullptr,
        topo.neighborMap.data_ptr<int32_t>(),
        rowOrder(topo, int64_t(N) * Kc * A.element_size()).data_ptr<int32_t>(),
        topo.rowMasks.data_ptr<int32_t>(),
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

// Tile shape by output width. FVDB_MIGEMM_CFG=<BM>x<BN>x<STAGES> overrides it for tuning.
template <typename T, bool B_KMAJOR, bool FLIP>
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
    {
        char const *env       = std::getenv("FVDB_MIGEMM_CFG");
        std::string const cfg = env ? env : "";
#define FVDB_MIGEMM_TRY(BM, BN, WM, WN, ST)                            \
    if (cfg == #BM "x" #BN "x" #ST && Nc % BN == 0) {                  \
        launchConv<TileConfig<BM, BN, WM, WN, ST>, T, B_KMAJOR, FLIP>( \
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);      \
        return;                                                        \
    }
        FVDB_MIGEMM_TRY(128, 256, 2, 4, 3)
        FVDB_MIGEMM_TRY(128, 256, 2, 4, 2)
        FVDB_MIGEMM_TRY(64, 256, 1, 4, 3)
        FVDB_MIGEMM_TRY(64, 256, 1, 4, 2)
        FVDB_MIGEMM_TRY(128, 128, 2, 2, 4)
        FVDB_MIGEMM_TRY(128, 128, 2, 2, 3)
        FVDB_MIGEMM_TRY(128, 128, 2, 2, 2)
        FVDB_MIGEMM_TRY(64, 128, 2, 2, 4)
        FVDB_MIGEMM_TRY(64, 128, 2, 2, 3)
        FVDB_MIGEMM_TRY(128, 64, 4, 1, 4)
        FVDB_MIGEMM_TRY(128, 64, 4, 1, 3)
        FVDB_MIGEMM_TRY(128, 64, 4, 1, 2)
        FVDB_MIGEMM_TRY(64, 64, 2, 2, 4)
        FVDB_MIGEMM_TRY(64, 64, 2, 2, 3)
        FVDB_MIGEMM_TRY(128, 96, 4, 1, 2)
        FVDB_MIGEMM_TRY(128, 96, 4, 1, 3)
        FVDB_MIGEMM_TRY(128, 32, 4, 1, 3)
        FVDB_MIGEMM_TRY(128, 32, 4, 1, 2)
        FVDB_MIGEMM_TRY(64, 32, 2, 1, 3)
        FVDB_MIGEMM_TRY(64, 32, 2, 1, 2)
#undef FVDB_MIGEMM_TRY
    }
    // Two stages keep two CTAs resident per SM, which beats deeper pipelines here.
    if (Nc % 128 == 0) {
        launchConv<TileConfig<128, 128, 2, 2, 2>, T, B_KMAJOR, FLIP>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else if (Nc % 64 == 0) {
        launchConv<TileConfig<128, 64, 4, 1, 2>, T, B_KMAJOR, FLIP>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else if (Nc % 96 == 0) {
        launchConv<TileConfig<128, 96, 4, 1, 2>, T, B_KMAJOR, FLIP>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    } else {
        // 32-wide stages are small enough for a third stage at two CTAs per SM.
        launchConv<TileConfig<128, 32, 4, 1, 3>, T, B_KMAJOR, FLIP>(
            A, Wrows, bias, out, topo, flipTaps, Kc, Nc, stream);
    }
}

template <typename T>
void
convForward(torch::Tensor const &features,
            torch::Tensor const &Wrows,
            torch::Tensor const &bias,
            torch::Tensor &out,
            MaskedIGemmTopology const &topo,
            bool flipTaps,
            int Ci,
            int Co,
            cudaStream_t stream) {
    dispatchConv<T, false, false>(features, Wrows, bias, out, topo, flipTaps, Ci, Co, stream);
}

template <typename Cfg, typename T>
torch::Tensor
launchWgrad(torch::Tensor const &X,
            torch::Tensor const &dY,
            MaskedIGemmTopology const &topo,
            bool flipTaps,
            int Ci,
            int Co,
            cudaStream_t stream) {
    int const K    = int(topo.kernelVolume);
    int const smem = wgradSmemBytes<Cfg, T>();
    auto kernel    = maskedIGemmWgradKernel<Cfg, T>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    int const sms       = smCount();
    int const tiles     = (Ci / Cfg::BM) * (Co / Cfg::BN);
    int64_t const steps = std::max<int64_t>(1, topo.pairIn.numel() / std::max(1, K) / Cfg::BK);
    // Many short CTAs keep the last wave small, but each split adds a tile of float32 atomics,
    // so every CTA keeps at least 128 steps. FVDB_MIGEMM_WSPLIT overrides CTAs per SM for tuning.
    char const *env      = std::getenv("FVDB_MIGEMM_WSPLIT");
    int64_t const target = int64_t(env ? std::max(1, std::atoi(env)) : 64) * sms;
    int64_t splits       = std::clamp<int64_t>(
        (target + tiles * K - 1) / (tiles * K), 1, std::max<int64_t>(1, steps / 128));

    // Each row is read by up to K taps, which hit L2 only if they run close together. When X and
    // dY are several times L2, split by output rows rather than pairs, so the K taps of a split
    // read the same rows, and use enough splits that the rows read by the resident
    // resident / (tiles * K) splits fit in a quarter of L2.
    int perSm = 0, device = 0, l2 = 0;
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, Cfg::THREADS, smem));
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    double const groups = double(std::max(1, perSm) * sms) / double(tiles * K);
    double const bytes  = double(topo.totalVoxels) * (Ci + Co) * X.element_size();
    auto const local    = int64_t(std::ceil(groups * bytes / (l2 / 4.0)));
    int const rowBlocks = int(topo.pairBlockOffsets.size(1));
    bool rowSplit       = !env && local > splits && rowBlocks > 1;
    if (rowSplit)
        splits = std::min<int64_t>({local, std::max<int64_t>(1, steps / 32), rowBlocks});
    // FVDB_MIGEMM_WROWS=<splits> forces row splitting, for tests and tuning.
    if (char const *rows = std::getenv("FVDB_MIGEMM_WROWS"); rows && rowBlocks > 0) {
        rowSplit = true;
        splits   = std::clamp<int64_t>(std::atoi(rows), 1, rowBlocks);
    }

    torch::Tensor rows, acc;
    if (splits > 1)
        acc = torch::zeros({K, Ci, Co}, X.options().dtype(torch::kFloat32));
    else
        rows = torch::empty({K, Ci, Co}, X.options());

    TapToRow const ttr{topo.kernelSize[1], topo.kernelSize[2], topo.kernelSize[0], flipTaps};
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

// Channel counts not divisible by 128 use 64-, 96- or 32-wide tiles on each side, whichever
// divides first, so each CTA gathers its rows once per 64 or 96 channels where possible.
template <typename T, int BM, int WM>
torch::Tensor
launchWgradByCo(torch::Tensor const &X,
                torch::Tensor const &dY,
                MaskedIGemmTopology const &topo,
                bool flipTaps,
                int Ci,
                int Co,
                cudaStream_t stream) {
    if (Co % 64 == 0)
        return launchWgrad<WgradConfig<BM, 64, WM, 2, 2>, T>(X, dY, topo, flipTaps, Ci, Co, stream);
    if (Co % 96 == 0)
        return launchWgrad<WgradConfig<BM, 96, WM, 2, 2>, T>(X, dY, topo, flipTaps, Ci, Co, stream);
    return launchWgrad<WgradConfig<BM, 32, WM, 1, 2>, T>(X, dY, topo, flipTaps, Ci, Co, stream);
}

template <typename T>
torch::Tensor
launchWgradNarrow(torch::Tensor const &X,
                  torch::Tensor const &dY,
                  MaskedIGemmTopology const &topo,
                  bool flipTaps,
                  int Ci,
                  int Co,
                  cudaStream_t stream) {
    if (Ci % 64 == 0)
        return launchWgradByCo<T, 64, 2>(X, dY, topo, flipTaps, Ci, Co, stream);
    if (Ci % 96 == 0)
        return launchWgradByCo<T, 96, 2>(X, dY, topo, flipTaps, Ci, Co, stream);
    return launchWgradByCo<T, 32, 1>(X, dY, topo, flipTaps, Ci, Co, stream);
}

template <typename T>
torch::Tensor
dispatchWgrad(torch::Tensor const &X,
              torch::Tensor const &dY,
              MaskedIGemmTopology const &topo,
              bool flipTaps,
              int Ci,
              int Co,
              cudaStream_t stream) {
    if constexpr (sizeof(T) == 4) {
        // float32 reads fragments element-wise, so wide warp tiles amortize the loads;
        // at 128x128, 16-row stages keep two CTAs resident per SM.
        if (Ci % 128 == 0 && Co % 128 == 0)
            return launchWgrad<WgradConfig<128, 128, 2, 2, 3, 16>, T>(
                X, dY, topo, flipTaps, Ci, Co, stream);
        return launchWgradNarrow<T>(X, dY, topo, flipTaps, Ci, Co, stream);
    } else {
        char const *env       = std::getenv("FVDB_MIGEMM_WCFG");
        std::string const cfg = env ? env : "";
#define FVDB_MIGEMM_TRYW(BM, BN, WM, WN, ST)                          \
    if (cfg == #BM "x" #BN "x" #ST && Ci % BM == 0 && Co % BN == 0) { \
        return launchWgrad<WgradConfig<BM, BN, WM, WN, ST>, T>(       \
            X, dY, topo, flipTaps, Ci, Co, stream);                   \
    }
        FVDB_MIGEMM_TRYW(128, 128, 2, 2, 4)
        FVDB_MIGEMM_TRYW(128, 128, 2, 2, 3)
        FVDB_MIGEMM_TRYW(128, 128, 2, 2, 2)
        FVDB_MIGEMM_TRYW(128, 64, 2, 2, 4)
        FVDB_MIGEMM_TRYW(128, 64, 2, 2, 3)
        FVDB_MIGEMM_TRYW(64, 128, 2, 2, 4)
        FVDB_MIGEMM_TRYW(64, 128, 2, 2, 3)
        FVDB_MIGEMM_TRYW(64, 64, 2, 2, 4)
        FVDB_MIGEMM_TRYW(64, 64, 2, 2, 3)
        FVDB_MIGEMM_TRYW(64, 64, 2, 2, 2)
#undef FVDB_MIGEMM_TRYW
        if (Ci % 128 == 0 && Co % 128 == 0)
            return launchWgrad<WgradConfig<128, 128, 2, 2, 3>, T>(
                X, dY, topo, flipTaps, Ci, Co, stream);
        return launchWgradNarrow<T>(X, dY, topo, flipTaps, Ci, Co, stream);
    }
}

void
checkConv(torch::Tensor const &features,
          torch::Tensor const &weights,
          MaskedIGemmTopology const &topo,
          char const *name) {
    TORCH_CHECK(features.is_cuda(), name, ": features must be on CUDA");
    TORCH_CHECK(weights.device() == features.device() &&
                    topo.neighborMap.device() == features.device(),
                name,
                ": features, weights and topology must be on the same device");
    TORCH_CHECK(weights.scalar_type() == features.scalar_type(),
                name,
                ": weights must have the features' dtype ",
                features.scalar_type());
    TORCH_CHECK(features.dim() == 2 && features.size(0) == topo.totalVoxels,
                name,
                ": features must be [",
                topo.totalVoxels,
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

} // namespace

// =============================================================================
// Public entry points
// =============================================================================

bool
maskedIGemmSupports(int64_t inChannels,
                    int64_t outChannels,
                    torch::ScalarType dtype,
                    torch::Device device) {
    bool const typeOk =
        dtype == torch::kFloat16 || dtype == torch::kBFloat16 || dtype == torch::kFloat32;
    if (!typeOk || !device.is_cuda() || inChannels <= 0 || outChannels <= 0 ||
        inChannels % 32 != 0 || outChannels % 32 != 0)
        return false;
    int index = device.has_index() ? device.index() : 0;
    if (!device.has_index())
        C10_CUDA_CHECK(cudaGetDevice(&index));
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, index));
    return major >= 8;
}

MaskedIGemmTopology
maskedIGemmTopology(GridBatchData const &grid, nanovdb::Coord kernelSize) {
    TORCH_CHECK(grid.device().is_cuda(), "maskedIGemmTopology: grid must be on CUDA");
    for (int d = 0; d < 3; ++d) {
        TORCH_CHECK(kernelSize[d] > 0 && kernelSize[d] % 2 == 1,
                    "maskedIGemmTopology: kernel sizes must be positive and odd");
    }
    ConvolutionGeometry const geometry(kernelSize, nanovdb::Coord(1));
    int64_t const K = geometry.kernelVolume();
    TORCH_CHECK(K <= kMaxTaps, "maskedIGemmTopology: kernel volume must be at most ", kMaxTaps);
    int64_t const N = grid.totalVoxels();
    TORCH_CHECK(N < std::numeric_limits<int32_t>::max(), "maskedIGemmTopology: too many voxels");

    c10::cuda::CUDAGuard guard(grid.device());
    auto const i32 = torch::dtype(torch::kInt32).device(grid.device());
    auto nbr       = torch::empty({N, K}, i32);
    auto code      = torch::empty({N}, i32);
    auto rowMasks  = torch::empty({N}, i32);

    if (N > 0) {
        // The staged builder needs every tap within one leaf of its voxel. Below ~128K
        // voxels the VoxelBlockManager's cold build (a host sync) costs more than the
        // probes it saves. FVDB_MIGEMM_BUILDER=scan|staged overrides the choice.
        bool withinLeaf = true;
        for (int d = 0; d < 3; ++d)
            withinLeaf &= (kernelSize[d] - 1) / 2 <= 8;
        char const *env          = std::getenv("FVDB_MIGEMM_BUILDER");
        std::string const choice = env ? env : "";
        bool scan                = !withinLeaf || N < (int64_t(1) << 17);
        if (withinLeaf && choice == "staged")
            scan = false;
        else if (choice == "scan")
            scan = true;
        if (scan) {
            buildNeighborMapScan(grid,
                                 geometry,
                                 int(K),
                                 nbr.data_ptr<int32_t>(),
                                 code.data_ptr<int32_t>(),
                                 rowMasks.data_ptr<int32_t>());
        } else {
            buildNeighborMapStaged(grid,
                                   geometry,
                                   int(K),
                                   nbr.data_ptr<int32_t>(),
                                   code.data_ptr<int32_t>(),
                                   rowMasks.data_ptr<int32_t>(),
                                   c10::cuda::getCurrentCUDAStream());
        }
    }

    auto sortedRows = torch::argsort(code).to(torch::kInt32);

    // Weight-gradient pairs per tap, output rows ascending (leaf order).
    int64_t const blocks = (N + kPairBlock - 1) / kPairBlock;
    auto blockCounts     = torch::zeros({K, blocks}, i32); // tap-major, so scans run along rows
    auto stream          = c10::cuda::getCurrentCUDAStream();
    if (blocks > 0) {
        pairCountKernel<<<blocks, kPairBlock, 0, stream>>>(
            rowMasks.data_ptr<int32_t>(), int(N), int(K), blockCounts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto tapEnds    = torch::cumsum(blockCounts.sum(1, false, torch::kInt64), 0);
    int64_t const P = tapEnds[K - 1].item<int64_t>();
    TORCH_CHECK(P < std::numeric_limits<int32_t>::max(),
                "maskedIGemmTopology: ",
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

    return MaskedIGemmTopology{nbr,
                               sortedRows,
                               torch::Tensor(),
                               rowMasks,
                               pairIn,
                               pairOut,
                               offsets,
                               blockOffsets,
                               N,
                               K,
                               kernelSize};
}

torch::Tensor
maskedIGemmConv(torch::Tensor features,
                torch::Tensor weights,
                std::optional<torch::Tensor> bias,
                MaskedIGemmTopology const &topo,
                bool flipTaps) {
    checkConv(features, weights, topo, "maskedIGemmConv");
    c10::cuda::CUDAGuard guard(features.device());
    auto const st = features.scalar_type();
    features      = features.contiguous();
    auto Wrows    = weightRows(weights);
    int const Ci = int(weights.size(1)), Co = int(weights.size(0));
    torch::Tensor b;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Co,
                    "maskedIGemmConv: bias must be [C_out]");
        TORCH_CHECK(bias->device() == features.device() && bias->scalar_type() == st,
                    "maskedIGemmConv: bias must match the features' device and dtype");
        b = bias->contiguous();
    }
    auto out = torch::empty({topo.totalVoxels, Co}, features.options());
    if (topo.totalVoxels == 0)
        return out;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (st == torch::kFloat16)
        convForward<__half>(features, Wrows, b, out, topo, flipTaps, Ci, Co, stream);
    else if (st == torch::kBFloat16)
        convForward<__nv_bfloat16>(features, Wrows, b, out, topo, flipTaps, Ci, Co, stream);
    else
        convForward<float>(features, Wrows, b, out, topo, flipTaps, Ci, Co, stream);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
maskedIGemmConvBackward(torch::Tensor gradOutput,
                        torch::Tensor features,
                        torch::Tensor weights,
                        MaskedIGemmTopology const &topo,
                        bool needFeatures,
                        bool needWeights,
                        bool flipTaps) {
    checkConv(features, weights, topo, "maskedIGemmConvBackward");
    TORCH_CHECK(gradOutput.dim() == 2 && gradOutput.size(0) == topo.totalVoxels &&
                    gradOutput.size(1) == weights.size(0),
                "maskedIGemmConvBackward: grad_output must be [N, C_out]");
    TORCH_CHECK(gradOutput.device() == features.device(),
                "maskedIGemmConvBackward: grad_output must be on the features' device");
    c10::cuda::CUDAGuard guard(features.device());
    auto const st = features.scalar_type();
    features      = features.contiguous();
    gradOutput    = gradOutput.to(st).contiguous();
    int const Ci = int(weights.size(1)), Co = int(weights.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();

    torch::Tensor gradFeatures, gradWeights;
    if (needFeatures) {
        auto Wrows   = weightRows(weights);
        gradFeatures = torch::empty({topo.totalVoxels, Ci}, features.options());
        if (topo.totalVoxels > 0) {
            if (st == torch::kFloat16)
                dispatchConv<__half, true, true>(
                    gradOutput, Wrows, {}, gradFeatures, topo, flipTaps, Co, Ci, stream);
            else if (st == torch::kBFloat16)
                dispatchConv<__nv_bfloat16, true, true>(
                    gradOutput, Wrows, {}, gradFeatures, topo, flipTaps, Co, Ci, stream);
            else
                dispatchConv<float, true, true>(
                    gradOutput, Wrows, {}, gradFeatures, topo, flipTaps, Co, Ci, stream);
        }
    }
    if (needWeights) {
        torch::Tensor gradRows;
        if (topo.totalVoxels > 0) {
            if (st == torch::kFloat16)
                gradRows =
                    dispatchWgrad<__half>(features, gradOutput, topo, flipTaps, Ci, Co, stream);
            else if (st == torch::kBFloat16)
                gradRows = dispatchWgrad<__nv_bfloat16>(
                    features, gradOutput, topo, flipTaps, Ci, Co, stream);
            else
                gradRows =
                    dispatchWgrad<float>(features, gradOutput, topo, flipTaps, Ci, Co, stream);
        } else {
            gradRows = torch::zeros({topo.kernelVolume, Ci, Co}, features.options());
        }
        gradWeights = weightsFromRows(gradRows, topo.kernelSize);
    }
    return {gradFeatures, gradWeights};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
