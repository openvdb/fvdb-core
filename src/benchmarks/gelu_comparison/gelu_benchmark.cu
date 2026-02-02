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
#include <examples/gelu_for_each.h>

namespace {

// ============================================================================
// Helper: Ensure CUDA operations complete before timing ends
// ============================================================================

inline void
sync_if_cuda(torch::Device device) {
    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }
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

    for (auto _: state) {
        auto result = dispatch_examples::example_gelu_for_each(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void
BM_GeluNew_Strided_CPU(benchmark::State &state) {
    auto tensor = make_strided_tensor(state.range(0), torch::kFloat32, torch::kCPU);

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

} // namespace

// Main is provided by benchmark::benchmark_main via linking
