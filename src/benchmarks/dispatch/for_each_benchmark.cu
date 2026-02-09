// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// for_each + flat views performance benchmarks.
//
// Measures dispatch::for_each with flat views across:
//   contiguity: contiguous vs strided (stride-2 slice)
//   device:     CPU vs CUDA
//   dtype:      float32 vs float64
//
// SoL baselines: memcpy, torch::softplus
//
// All tensors pre-allocated — allocation is never measured.
// Each benchmark runs a warmup call before the timing loop to prime:
//   - CPU thread pool construction
//   - CUDA context and kernel JIT
//   - Page faults and cache state
// Thread count is normalized via SetupThreads() for fair comparison.
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>
#include <dispatch/torch/dispatch.h>
#include <dispatch/torch/for_each.h>
#include <dispatch/torch/views.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// ============================================================================
// Configuration
// ============================================================================

constexpr int kNumThreads = 16;

static void
SetupThreads() {
    torch::set_num_threads(kNumThreads);
    at::set_num_threads(kNumThreads);
}

// ============================================================================
// Softplus scalar function
// ============================================================================

template <typename T>
__hostdev__ T
softplus_scalar(T x, T beta, T threshold) {
    T const bx = beta * x;
    return (bx > threshold) ? x : log1p(exp(bx)) / beta;
}

// ============================================================================
// CUDA kernel wrappers (device lambdas must be in free functions)
// ============================================================================

template <torch::ScalarType Stype, dispatch::contiguity Contig>
void
cuda_softplus(torch::Tensor const &input, torch::Tensor &output) {
    using namespace dispatch;
    using Tag = tag<torch::kCUDA, Stype, Contig>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});
    using scalar_t        = torch_scalar_cpp_type_t<stype>;

    auto in  = flat_in<dev, stype, contig>(input);
    auto out = flat_out<dev, stype, contig>(output);

    auto const beta      = static_cast<scalar_t>(1.0);
    auto const threshold = static_cast<scalar_t>(20.0);

    for_each(Tag{}, input.numel(), [=] __device__(Tag, int64_t idx) {
        out[idx] = softplus_scalar(in[idx], beta, threshold);
    });
}

// ============================================================================
// CPU softplus via for_each
// ============================================================================

template <torch::ScalarType Stype, dispatch::contiguity Contig>
void
cpu_softplus(torch::Tensor const &input, torch::Tensor &output) {
    using namespace dispatch;
    using Tag = tag<torch::kCPU, Stype, Contig>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});
    using scalar_t        = torch_scalar_cpp_type_t<stype>;

    auto in  = flat_in<dev, stype, contig>(input);
    auto out = flat_out<dev, stype, contig>(output);

    auto const beta      = static_cast<scalar_t>(1.0);
    auto const threshold = static_cast<scalar_t>(20.0);

    for_each(Tag{}, input.numel(), [=](Tag, int64_t idx) {
        out[idx] = softplus_scalar(in[idx], beta, threshold);
    });
}

// ============================================================================
// Helpers
// ============================================================================

inline torch::Tensor
make_contiguous(int64_t n, torch::ScalarType dtype, torch::Device dev) {
    return torch::randn({n}, torch::TensorOptions().dtype(dtype).device(dev));
}

inline torch::Tensor
make_strided(int64_t n, torch::ScalarType dtype, torch::Device dev) {
    auto full = torch::randn({n * 2}, torch::TensorOptions().dtype(dtype).device(dev));
    return full.slice(0, 0, n * 2, 2);
}

// ============================================================================
// Benchmarks: for_each softplus
// ============================================================================

// --- CPU Float32 ---

static void
BM_ForEach_Softplus_Contiguous_CPU_Float32(benchmark::State &state) {
    SetupThreads();
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);

    // Warm up: pool construction + cache priming
    cpu_softplus<torch::kFloat32, dispatch::contiguity::contiguous>(input, output);

    for (auto _: state) {
        cpu_softplus<torch::kFloat32, dispatch::contiguity::contiguous>(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_ForEach_Softplus_Strided_CPU_Float32(benchmark::State &state) {
    SetupThreads();
    int64_t const n = state.range(0);
    auto input      = make_strided(n, torch::kFloat32, torch::kCPU);
    auto output     = make_strided(n, torch::kFloat32, torch::kCPU);

    cpu_softplus<torch::kFloat32, dispatch::contiguity::strided>(input, output);

    for (auto _: state) {
        cpu_softplus<torch::kFloat32, dispatch::contiguity::strided>(input, output);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- CPU Float64 ---

static void
BM_ForEach_Softplus_Contiguous_CPU_Float64(benchmark::State &state) {
    SetupThreads();
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat64, torch::kCPU);
    auto output     = torch::empty_like(input);

    cpu_softplus<torch::kFloat64, dispatch::contiguity::contiguous>(input, output);

    for (auto _: state) {
        cpu_softplus<torch::kFloat64, dispatch::contiguity::contiguous>(input, output);
        benchmark::DoNotOptimize(output.data_ptr<double>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_ForEach_Softplus_Strided_CPU_Float64(benchmark::State &state) {
    SetupThreads();
    int64_t const n = state.range(0);
    auto input      = make_strided(n, torch::kFloat64, torch::kCPU);
    auto output     = make_strided(n, torch::kFloat64, torch::kCPU);

    cpu_softplus<torch::kFloat64, dispatch::contiguity::strided>(input, output);

    for (auto _: state) {
        cpu_softplus<torch::kFloat64, dispatch::contiguity::strided>(input, output);
        benchmark::DoNotOptimize(output.data_ptr<double>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- CUDA Float32 ---

static void
BM_ForEach_Softplus_Contiguous_CUDA_Float32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCUDA);
    auto output     = torch::empty_like(input);

    // Warm up: CUDA context + kernel JIT + cache priming
    cuda_softplus<torch::kFloat32, dispatch::contiguity::contiguous>(input, output);
    torch::cuda::synchronize();

    for (auto _: state) {
        cuda_softplus<torch::kFloat32, dispatch::contiguity::contiguous>(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_ForEach_Softplus_Strided_CUDA_Float32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_strided(n, torch::kFloat32, torch::kCUDA);
    auto output     = make_strided(n, torch::kFloat32, torch::kCUDA);

    cuda_softplus<torch::kFloat32, dispatch::contiguity::strided>(input, output);
    torch::cuda::synchronize();

    for (auto _: state) {
        cuda_softplus<torch::kFloat32, dispatch::contiguity::strided>(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- CUDA Float64 ---

static void
BM_ForEach_Softplus_Contiguous_CUDA_Float64(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat64, torch::kCUDA);
    auto output     = torch::empty_like(input);

    cuda_softplus<torch::kFloat64, dispatch::contiguity::contiguous>(input, output);
    torch::cuda::synchronize();

    for (auto _: state) {
        cuda_softplus<torch::kFloat64, dispatch::contiguity::contiguous>(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<double>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_ForEach_Softplus_Strided_CUDA_Float64(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_strided(n, torch::kFloat64, torch::kCUDA);
    auto output     = make_strided(n, torch::kFloat64, torch::kCUDA);

    cuda_softplus<torch::kFloat64, dispatch::contiguity::strided>(input, output);
    torch::cuda::synchronize();

    for (auto _: state) {
        cuda_softplus<torch::kFloat64, dispatch::contiguity::strided>(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<double>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// SoL Baselines
// ============================================================================

static void
BM_SoL_Memcpy_CPU_Float32(benchmark::State &state) {
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);

    // Warm up: page faults + cache
    std::memcpy(output.data_ptr<float>(),
                input.data_ptr<float>(),
                static_cast<std::size_t>(n) * sizeof(float));

    for (auto _: state) {
        std::memcpy(output.data_ptr<float>(),
                    input.data_ptr<float>(),
                    static_cast<std::size_t>(n) * sizeof(float));
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_SoL_Memcpy_CUDA_Float32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCUDA);
    auto output     = torch::empty_like(input);

    cudaMemcpy(output.data_ptr<float>(),
               input.data_ptr<float>(),
               static_cast<std::size_t>(n) * sizeof(float),
               cudaMemcpyDeviceToDevice);
    torch::cuda::synchronize();

    for (auto _: state) {
        cudaMemcpy(output.data_ptr<float>(),
                   input.data_ptr<float>(),
                   static_cast<std::size_t>(n) * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_SoL_TorchSoftplus_CPU_Float32(benchmark::State &state) {
    SetupThreads();
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);

    // Warm up: torch internals + thread pool
    // Use softplus_out to pre-allocate — matches our for_each benchmarks (no alloc in loop)
    at::softplus_out(output, input, 1.0, 20.0);

    for (auto _: state) {
        at::softplus_out(output, input, 1.0, 20.0);
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void
BM_SoL_TorchSoftplus_CUDA_Float32(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCUDA);
    auto output     = torch::empty_like(input);

    at::softplus_out(output, input, 1.0, 20.0);
    torch::cuda::synchronize();

    for (auto _: state) {
        at::softplus_out(output, input, 1.0, 20.0);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Raw diagnostic baselines — isolate where time goes
// ============================================================================

// Raw scalar softplus for diagnostic baselines (no dispatch, no views, no parallelism)
inline float
raw_softplus_scalar(float x) {
    float const bx = x; // beta = 1.0
    return (bx > 20.0f) ? x : log1pf(expf(bx));
}

// Serial loop: measures per-element compute cost on raw pointers (no parallelism)
static void
BM_Diag_SerialLoop_CPU_Float32(benchmark::State &state) {
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);
    float *in_ptr   = input.data_ptr<float>();
    float *out_ptr  = output.data_ptr<float>();

    // Warm up
    for (int64_t i = 0; i < n; ++i)
        out_ptr[i] = raw_softplus_scalar(in_ptr[i]);

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = raw_softplus_scalar(in_ptr[i]);
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// Copy loop: measures pure memory bandwidth (no compute)
static void
BM_Diag_CopyLoop_CPU_Float32(benchmark::State &state) {
    int64_t const n = state.range(0);
    auto input      = make_contiguous(n, torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);
    float *in_ptr   = input.data_ptr<float>();
    float *out_ptr  = output.data_ptr<float>();

    for (int64_t i = 0; i < n; ++i)
        out_ptr[i] = in_ptr[i];

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = in_ptr[i];
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Register benchmarks
// ============================================================================

#define BENCHMARK_SIZES \
    ->Args({10000})->Args({100000})->Args({1000000})->Args({10000000})->UseRealTime()

// for_each softplus
BENCHMARK(BM_ForEach_Softplus_Contiguous_CPU_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Strided_CPU_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Contiguous_CUDA_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Strided_CUDA_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Contiguous_CPU_Float64) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Strided_CPU_Float64) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Contiguous_CUDA_Float64) BENCHMARK_SIZES;
BENCHMARK(BM_ForEach_Softplus_Strided_CUDA_Float64) BENCHMARK_SIZES;

// SoL baselines (pre-allocated output, apples-to-apples with for_each)
BENCHMARK(BM_SoL_Memcpy_CPU_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_SoL_Memcpy_CUDA_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_SoL_TorchSoftplus_CPU_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_SoL_TorchSoftplus_CUDA_Float32) BENCHMARK_SIZES;

// Diagnostic baselines (CPU only — isolate compute vs memory vs parallelism)
BENCHMARK(BM_Diag_CopyLoop_CPU_Float32) BENCHMARK_SIZES;
BENCHMARK(BM_Diag_SerialLoop_CPU_Float32) BENCHMARK_SIZES;

} // namespace
