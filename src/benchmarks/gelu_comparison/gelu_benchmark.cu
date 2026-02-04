// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// GELU Performance Comparison Benchmark
//
// Compares three implementations:
// 1. torch::gelu - PyTorch's native GELU
// 2. gelu_old - Old-style fvdb ForEach implementation
// 3. example_gelu_for_each - New dispatch framework implementation
//
// Build: ./build.sh install benchmarks
// Run:   ./build/.../gbenchmarks/gelu_comparison
//

#include "gelu_old.cuh"

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>
#include <dispatch/thread_pool.h>
#include <examples/gelu_for_each.h>
#include <examples/gelu_scalar.h>

#include <cmath>

namespace {

// ============================================================================
// Raw scalar GELU for diagnostic baselines (no dispatch, no views)
// ============================================================================

inline float
raw_gelu_scalar(float x) {
    constexpr float sqrt2 = 1.4142135623730951f;
    return 0.5f * x * (1.0f + std::erf(x / sqrt2));
}

// ============================================================================
// Helper: Create test tensors
// ============================================================================

inline torch::Tensor
make_contiguous_tensor(int64_t size, torch::ScalarType dtype, torch::Device device) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    return torch::randn({size}, options);
}

inline torch::Tensor
make_strided_tensor(int64_t size, torch::ScalarType dtype, torch::Device device) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    // Create a larger tensor and slice with stride=2
    auto full = torch::randn({size * 2}, options);
    return full.slice(0, 0, size * 2, 2);
}

// ============================================================================
// Benchmark: torch::gelu (PyTorch native)
// ============================================================================

static void
BM_TorchGelu_Contiguous_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = torch::gelu(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_TorchGelu_Strided_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = torch::gelu(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_TorchGelu_Contiguous_CPU(benchmark::State &state) {
    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);

    for (auto _: state) {
        auto result = torch::gelu(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// ============================================================================
// Benchmark: gelu_old (Old-style fvdb ForEach)
// ============================================================================

static void
BM_GeluOld_Contiguous_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = gelu_comparison::gelu_old(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluOld_Contiguous_CPU(benchmark::State &state) {
    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);

    for (auto _: state) {
        auto result = gelu_comparison::gelu_old(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// Note: gelu_old does NOT support strided tensors - it requires contiguous input.
// This is one of the limitations of the old approach.

// ============================================================================
// Benchmark: example_gelu_for_each (New dispatch framework)
// ============================================================================

static void
BM_GeluNew_Contiguous_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Strided_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Contiguous_CPU(benchmark::State &state) {
    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);

    // Ensure thread pool is created before timing
    (void)dispatch::thread_pool::instance();

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Strided_CPU(benchmark::State &state) {
    auto tensor = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCPU);

    // Ensure thread pool is created before timing
    (void)dispatch::thread_pool::instance();

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// ============================================================================
// Benchmark: In-place variants
// ============================================================================

static void
BM_TorchGelu_InPlace_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        // torch::gelu doesn't have an in-place variant, use gelu_
        torch::gelu_(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(tensor.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluOld_InPlace_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        gelu_comparison::gelu_old_(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(tensor.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_InPlace_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        dispatch_examples::example_gelu_for_each_(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(tensor.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// ============================================================================
// Benchmark: Half precision (Float16)
// ============================================================================

static void
BM_TorchGelu_Float16_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat16, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = torch::gelu(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluOld_Float16_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat16, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = gelu_comparison::gelu_old(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Float16_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto tensor = make_contiguous_tensor(state.range(0), torch::kFloat16, torch::kCUDA);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// ============================================================================
// Benchmark: NoAlloc variants (pre-allocated output tensors)
// These isolate kernel performance from allocation overhead
// ============================================================================

static void
BM_TorchGelu_Contiguous_CUDA_NoAlloc(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto input  = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    auto output = torch::empty_like(input);
    torch::cuda::synchronize();

    for (auto _: state) {
        torch::gelu_outf(input, "none", output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Contiguous_CUDA_NoAlloc(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto input  = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    auto output = torch::empty_like(input);
    torch::cuda::synchronize();

    for (auto _: state) {
        dispatch_examples::example_gelu_for_each_out(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_TorchGelu_Contiguous_CPU_NoAlloc(benchmark::State &state) {
    auto input  = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);
    auto output = torch::empty_like(input);

    for (auto _: state) {
        torch::gelu_outf(input, "none", output);
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Contiguous_CPU_NoAlloc(benchmark::State &state) {
    auto input  = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);
    auto output = torch::empty_like(input);

    // Ensure thread pool is created before timing
    (void)dispatch::thread_pool::instance();

    for (auto _: state) {
        dispatch_examples::example_gelu_for_each_out(input, output);
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_TorchGelu_Strided_CUDA_NoAlloc(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto input  = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    auto output = torch::empty_like(input);
    torch::cuda::synchronize();

    for (auto _: state) {
        torch::gelu_outf(input, "none", output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Strided_CUDA_NoAlloc(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }

    auto input  = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCUDA);
    auto output = torch::empty_like(input);
    torch::cuda::synchronize();

    for (auto _: state) {
        dispatch_examples::example_gelu_for_each_out(input, output);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(output.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// ============================================================================
// RAW DIAGNOSTIC BENCHMARKS - validate timing infrastructure
// These use raw pointer loops, no dispatch, no views, no parallelism
// ============================================================================

// Raw serial loop on CPU - absolute baseline
static void
BM_Raw_SerialLoop_CPU(benchmark::State &state) {
    const int64_t n = state.range(0);
    std::vector<float> input(n), output(n);
    for (int64_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) * 0.001f;
    }

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            output[i] = raw_gelu_scalar(input[i]);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

// Raw serial loop using torch tensors but raw pointer access
static void
BM_Raw_TorchTensor_SerialLoop_CPU(benchmark::State &state) {
    auto input      = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);
    float *in_ptr   = input.data_ptr<float>();
    float *out_ptr  = output.data_ptr<float>();
    const int64_t n = input.numel();

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = raw_gelu_scalar(in_ptr[i]);
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

// Just measure overhead of iterating (no computation)
static void
BM_Raw_LoopOverhead_CPU(benchmark::State &state) {
    const int64_t n = state.range(0);
    std::vector<float> input(n), output(n);

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            output[i] = input[i]; // Just copy, no erf
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

// Using dispatch gelu_scalar directly (tests the scalar function itself)
static void
BM_Raw_DispatchGeluScalar_CPU(benchmark::State &state) {
    auto input      = make_contiguous_tensor(state.range(0), torch::kFloat32, torch::kCPU);
    auto output     = torch::empty_like(input);
    float *in_ptr   = input.data_ptr<float>();
    float *out_ptr  = output.data_ptr<float>();
    const int64_t n = input.numel();

    for (auto _: state) {
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = dispatch_examples::gelu_scalar(in_ptr[i]);
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Register benchmarks
// ============================================================================

// Tensor sizes: 1K, 10K, 100K, 1M, 10M
#define BENCHMARK_SIZES ->RangeMultiplier(10)->Range(1024, 10 * 1024 * 1024)

// CUDA Contiguous - Float32
BENCHMARK(BM_TorchGelu_Contiguous_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluOld_Contiguous_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Contiguous_CUDA) BENCHMARK_SIZES;

// CUDA Strided - Float32 (only torch and new dispatch support strided)
BENCHMARK(BM_TorchGelu_Strided_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Strided_CUDA) BENCHMARK_SIZES;

// CPU Contiguous - Float32
BENCHMARK(BM_TorchGelu_Contiguous_CPU) BENCHMARK_SIZES;
BENCHMARK(BM_GeluOld_Contiguous_CPU) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Contiguous_CPU) BENCHMARK_SIZES;

// CPU Strided - Float32
BENCHMARK(BM_GeluNew_Strided_CPU) BENCHMARK_SIZES;

// In-place variants - CUDA Float32
BENCHMARK(BM_TorchGelu_InPlace_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluOld_InPlace_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_InPlace_CUDA) BENCHMARK_SIZES;

// Float16 - CUDA
BENCHMARK(BM_TorchGelu_Float16_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluOld_Float16_CUDA) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Float16_CUDA) BENCHMARK_SIZES;

// NoAlloc variants - isolate kernel performance from allocation
BENCHMARK(BM_TorchGelu_Contiguous_CUDA_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Contiguous_CUDA_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_TorchGelu_Strided_CUDA_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Strided_CUDA_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_TorchGelu_Contiguous_CPU_NoAlloc) BENCHMARK_SIZES;
BENCHMARK(BM_GeluNew_Contiguous_CPU_NoAlloc) BENCHMARK_SIZES;

// Raw diagnostic benchmarks - validate timing
BENCHMARK(BM_Raw_LoopOverhead_CPU) BENCHMARK_SIZES;
BENCHMARK(BM_Raw_SerialLoop_CPU) BENCHMARK_SIZES;
BENCHMARK(BM_Raw_TorchTensor_SerialLoop_CPU) BENCHMARK_SIZES;
BENCHMARK(BM_Raw_DispatchGeluScalar_CPU) BENCHMARK_SIZES;

} // namespace

// Main is provided by benchmark::benchmark_main via linking
