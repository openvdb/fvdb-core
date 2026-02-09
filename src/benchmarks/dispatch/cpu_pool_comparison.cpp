// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CPU Thread Pool Comparison Benchmark
//
// Compares thread pool implementations with SIMD GELU:
// 1. queue_pool (retired) - Simple queue-based pool (benchmark baseline)
// 2. spin_pool (retired) - Spin-wait static pool (benchmark baseline)
// 3. broadcast_pool - Static partitioning with broadcast-style dispatch
// 4. work_stealing_pool - Chase-Lev deque work stealing (for unbalanced workloads)
//
// Also compares against torch::gelu, OpenMP, and at::parallel_for as baselines.
//
// Build: ./build.sh install benchmarks
// Run:   ./build/.../gbenchmarks/cpu_pool_comparison
//

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>
#include <dispatch/thread_pool.h>

#include "queue_pool.h"
#include "spin_pool.h"

// OpenMP GELU implementation (in separate .cpp for proper pragma support)
#include "omp_gelu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

// ============================================================================
// Configuration
// ============================================================================

// Set thread count to match typical PyTorch default for fair comparison
constexpr int kNumThreads = 16;

static void
SetupThreads() {
    // Force PyTorch and ATen to use consistent thread count
    torch::set_num_threads(kNumThreads);
    at::set_num_threads(kNumThreads);
}

// ============================================================================
// SIMD GELU Implementation
// ============================================================================
// Uses exp-based formula: GELU(x) = x / (1 + exp(-2u))
// where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// This avoids tanh() which can be slower on some architectures.

template <typename T>
inline at::vec::Vectorized<T>
gelu_simd_op(const at::vec::Vectorized<T> &x) {
    using Vec = at::vec::Vectorized<T>;

    // kAlphaTwo = sqrt(2/pi) * 2 = 1.5957691216...
    // We multiply by 2 to set up the exp(-2u) term
    const Vec kAlphaTwo(static_cast<T>(1.5957691216057307117597842397375));
    const Vec kBeta(static_cast<T>(0.044715));
    const Vec kOne(static_cast<T>(1.0));

    // Calculate 2u = 2 * sqrt(2/pi) * (x + 0.044715 * x^3)
    Vec x_sq  = x * x;
    Vec inner = at::vec::fmadd(kBeta, x_sq * x, x);
    Vec two_u = kAlphaTwo * inner;

    // Calculate x / (1 + exp(-2u))
    // Using .neg() because some SIMD libs don't overload unary minus efficiently
    Vec exp_val = two_u.neg().exp();

    // This handles edge cases correctly:
    // +inf: exp->0, result->x
    // -inf: exp->inf, result->0
    return x / (kOne + exp_val);
}

// ============================================================================
// Pool-specific GELU implementations using SIMD
// ============================================================================

template <typename Pool>
void
gelu_with_pool(Pool &pool, const float *in_ptr, float *out_ptr, int64_t numel) {
    using Vec                 = at::vec::Vectorized<float>;
    constexpr int64_t vec_len = Vec::size();

    pool.parallel_for(int64_t{0}, numel, [in_ptr, out_ptr, vec_len](int64_t begin, int64_t end) {
        int64_t i     = begin;
        int64_t limit = end - (end - begin) % vec_len;

        // Main SIMD loop
        for (; i + vec_len <= limit; i += vec_len) {
            Vec data = Vec::loadu(in_ptr + i);
            data     = gelu_simd_op(data);
            data.store(out_ptr + i);
        }

        // Tail elements
        if (i < end) {
            int64_t remaining = end - i;
            Vec data          = Vec::loadu(in_ptr + i, remaining);
            data              = gelu_simd_op(data);
            data.store(out_ptr + i, remaining);
        }
    });
}

// Wrapper functions for each pool
void
gelu_queue_pool(torch::Tensor in, torch::Tensor out) {
    gelu_with_pool(dispatch_bench::queue_pool::instance(),
                   in.data_ptr<float>(),
                   out.data_ptr<float>(),
                   in.numel());
}

void
gelu_spin_pool(torch::Tensor in, torch::Tensor out) {
    gelu_with_pool(
        dispatch_bench::spin_pool::instance(), in.data_ptr<float>(), out.data_ptr<float>(), in.numel());
}

void
gelu_broadcast_pool(torch::Tensor in, torch::Tensor out) {
    gelu_with_pool(dispatch::broadcast_pool::instance(),
                   in.data_ptr<float>(),
                   out.data_ptr<float>(),
                   in.numel());
}

void
gelu_work_stealing_pool(torch::Tensor in, torch::Tensor out) {
    gelu_with_pool(dispatch::work_stealing_pool::instance(),
                   in.data_ptr<float>(),
                   out.data_ptr<float>(),
                   in.numel());
}

// ============================================================================
// Serial GELU baseline (no threading)
// ============================================================================

void
gelu_serial(float const *in_ptr, float *out_ptr, int64_t numel) {
    using Vec                 = at::vec::Vectorized<float>;
    constexpr int64_t vec_len = Vec::size();

    int64_t i     = 0;
    int64_t limit = numel - numel % vec_len;

    for (; i + vec_len <= limit; i += vec_len) {
        Vec data = Vec::loadu(in_ptr + i);
        data     = gelu_simd_op(data);
        data.store(out_ptr + i);
    }

    if (i < numel) {
        int64_t remaining = numel - i;
        Vec data          = Vec::loadu(in_ptr + i, remaining);
        data              = gelu_simd_op(data);
        data.store(out_ptr + i, remaining);
    }
}

// ============================================================================
// OpenMP-based GELU implementations
// ============================================================================
// Note: The raw OpenMP implementation is in omp_gelu.cpp (compiled by g++)
// to ensure OpenMP pragmas are processed correctly. nvcc doesn't handle
// #pragma omp properly even with -Xcompiler=-fopenmp.

// at::parallel_for (uses PyTorch's internal threading - may be TBB or OpenMP)
void
gelu_at_parallel_for(const float *in_ptr, float *out_ptr, int64_t numel) {
    using Vec                    = at::vec::Vectorized<float>;
    constexpr int64_t vec_len    = Vec::size();
    constexpr int64_t grain_size = 2048;

    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
        int64_t i     = begin;
        int64_t limit = end - (end - begin) % vec_len;

        // Main SIMD loop
        for (; i + vec_len <= limit; i += vec_len) {
            Vec data = Vec::loadu(in_ptr + i);
            data     = gelu_simd_op(data);
            data.store(out_ptr + i);
        }

        // Tail elements
        if (i < end) {
            int64_t remaining = end - i;
            Vec data          = Vec::loadu(in_ptr + i, remaining);
            data              = gelu_simd_op(data);
            data.store(out_ptr + i, remaining);
        }
    });
}

// ============================================================================
// Helper: Create test tensors
// ============================================================================

inline torch::Tensor
make_contiguous_tensor(int64_t size, torch::ScalarType dtype, torch::Device device) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    return torch::randn({size}, options);
}

// ============================================================================
// Benchmarks: Torch baseline
// ============================================================================
// For large N (10M floats = 40MB), the data exceeds L3 cache (32MB),
// so simply iterating guarantees RAM hits. No need for PauseTiming tricks.

static void
BM_Torch_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);

    // Allocate once - 40MB buffer exceeds L3 cache, guaranteeing RAM access
    auto input  = torch::randn({numel}, torch::kFloat32);
    auto output = torch::empty_like(input);

    for (auto _: state) {
        at::gelu_out(output, input, "none");
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

// ============================================================================
// Benchmarks: Individual pool implementations
// ============================================================================

static void
BM_QueuePool_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);

    // Warm up pool
    (void)dispatch_bench::queue_pool::instance();

    for (auto _: state) {
        gelu_queue_pool(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_SpinPool_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);

    // Warm up pool
    (void)dispatch_bench::spin_pool::instance();

    for (auto _: state) {
        gelu_spin_pool(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_BroadcastPool_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);

    // Warm up pool
    (void)dispatch::broadcast_pool::instance();

    for (auto _: state) {
        gelu_broadcast_pool(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_WorkStealingPool_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);

    // Warm up pool
    (void)dispatch::work_stealing_pool::instance();

    for (auto _: state) {
        gelu_work_stealing_pool(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

// ============================================================================
// Benchmark: Serial baseline (no threading)
// ============================================================================

static void
BM_Serial_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        gelu_serial(in_ptr, out_ptr, numel);
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

// ============================================================================
// Benchmarks: OpenMP implementations
// ============================================================================

static void
BM_OpenMP_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    // Print thread count once on first benchmark
    static bool printed = false;
    if (!printed) {
        printf("OpenMP threads: %d, PyTorch threads: %d\n",
               omp_gelu::get_num_threads(),
               torch::get_num_threads());
        printed = true;
    }

    for (auto _: state) {
        omp_gelu::gelu_openmp(in_ptr, out_ptr, numel);
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_ATParallelFor_CPU_NoAlloc(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel}, torch::kFloat32);
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        gelu_at_parallel_for(in_ptr, out_ptr, numel);
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * numel);
}

// ============================================================================
// Register benchmarks
// ============================================================================

// Test sizes spanning small to large:
// 1K, 4K (serial threshold), 10K, 100K (medium threshold), 1M (large threshold), 10M
//
// UseRealTime() makes Google Benchmark use wall clock time instead of CPU time
// for throughput calculations. This is essential for fair comparison of parallel
// implementations - CPU time measures differently for different threading models.
#define BENCHMARK_SIZES    \
    ->Args({1024})         \
        ->Args({4096})     \
        ->Args({10000})    \
        ->Args({100000})   \
        ->Args({1000000})  \
        ->Args({10000000}) \
        ->Args({10485760}) \
        ->UseRealTime()

// Register all benchmarks
BENCHMARK(BM_Torch_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_Serial_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_QueuePool_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_SpinPool_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_BroadcastPool_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_WorkStealingPool_CPU_NoAlloc) BENCHMARK_SIZES;

// OpenMP benchmark (uses external .cpp implementation for proper pragma support)
BENCHMARK(BM_OpenMP_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_ATParallelFor_CPU_NoAlloc) BENCHMARK_SIZES;

} // namespace

// Main is provided by benchmark::benchmark_main via linking
