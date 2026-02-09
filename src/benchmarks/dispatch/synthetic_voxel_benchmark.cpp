// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic benchmarks to test parallel_for implementations with
// compute-bound workloads that avoid PyTorch's hyper-optimized kernels.

#include "queue_pool.h"
#include "spin_pool.h"

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <benchmark/benchmark.h>
#include <dispatch/thread_pool.h>

#include <cmath>

// ----------------------------------------------------------------------------
// CONFIGURATION
// ----------------------------------------------------------------------------

constexpr int kNumThreads = 16;

static void
SetupThreads() {
    torch::set_num_threads(kNumThreads);
    at::set_num_threads(kNumThreads);
}

// ----------------------------------------------------------------------------
// SYNTHETIC KERNELS
// ----------------------------------------------------------------------------

// 1. Uniform Heavy Compute
// Simulates a costly operation (e.g., ray marching 50 steps) on every voxel.
template <typename T>
inline void
heavy_compute_kernel(const T *in, T *out, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        T val = in[i];
        // 50 iterations of FMA - high instruction dependency
        for (int k = 0; k < 50; ++k) {
            val = val * 0.999f + 0.001f;
        }
        out[i] = val;
    }
}

// 2. Unbalanced Compute (The "VDB Killer")
// Simulates sparse data: 10% of voxels are "active" (heavy), 90% are empty (cheap).
// This wrecks static partitioning (OpenMP) and tests work-stealing.
template <typename T>
inline void
unbalanced_kernel(const T *in, T *out, int64_t size, int64_t offset) {
    for (int64_t i = 0; i < size; ++i) {
        // Deterministic pseudo-randomness based on index
        bool active = ((i + offset) % 10) == 0;

        T val          = in[i];
        int iterations = active ? 200 : 1; // Heavy vs Trivial

        for (int k = 0; k < iterations; ++k) {
            val = val * 0.999f + 0.001f;
        }
        out[i] = val;
    }
}

// ----------------------------------------------------------------------------
// BENCHMARKS: UNIFORM HEAVY COMPUTE
// ----------------------------------------------------------------------------

static void
BM_Uniform_Serial(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        heavy_compute_kernel(in_ptr, out_ptr, numel);
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_OpenMP(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numel; ++i) {
            float val = in_ptr[i];
            for (int k = 0; k < 50; ++k)
                val = val * 0.999f + 0.001f;
            out_ptr[i] = val;
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_QueuePool(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch_bench::queue_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                float val = ip[i];
                for (int k = 0; k < 50; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_SpinPool(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch_bench::spin_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            float const *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                float val = ip[i];
                for (int k = 0; k < 50; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_Broadcast(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch::broadcast_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                float val = ip[i];
                for (int k = 0; k < 50; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_WorkStealing(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch::work_stealing_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                float val = ip[i];
                for (int k = 0; k < 50; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Uniform_ATParallel(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    for (auto _: state) {
        at::parallel_for(0, numel, 2048, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                float val = ip[i];
                for (int k = 0; k < 50; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

// ----------------------------------------------------------------------------
// BENCHMARKS: UNBALANCED (SPARSE) COMPUTE
// ----------------------------------------------------------------------------

static void
BM_Unbalanced_Serial(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        unbalanced_kernel(in_ptr, out_ptr, numel, 0);
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_OpenMP_Static(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        // Static schedule fails hard on unbalanced data
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numel; ++i) {
            bool active = (i % 10) == 0;
            float val   = in_ptr[i];
            int iter    = active ? 200 : 1;
            for (int k = 0; k < iter; ++k)
                val = val * 0.999f + 0.001f;
            out_ptr[i] = val;
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_OpenMP_Dynamic(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);
    const float *in_ptr = input.data_ptr<float>();
    float *out_ptr      = output.data_ptr<float>();

    for (auto _: state) {
        // Dynamic schedule handles unbalanced better but has overhead
#pragma omp parallel for schedule(dynamic, 1024)
        for (int64_t i = 0; i < numel; ++i) {
            bool active = (i % 10) == 0;
            float val   = in_ptr[i];
            int iter    = active ? 200 : 1;
            for (int k = 0; k < iter; ++k)
                val = val * 0.999f + 0.001f;
            out_ptr[i] = val;
        }
        benchmark::DoNotOptimize(out_ptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_QueuePool(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch_bench::queue_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                bool active = (i % 10) == 0;
                float val   = ip[i];
                int iter    = active ? 200 : 1;
                for (int k = 0; k < iter; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_SpinPool(benchmark::State &state) {
    SetupThreads();
    int64_t const numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch_bench::spin_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            float const *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                bool active = (i % 10) == 0;
                float val   = ip[i];
                int iter    = active ? 200 : 1;
                for (int k = 0; k < iter; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_Broadcast(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch::broadcast_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                bool active = (i % 10) == 0;
                float val   = ip[i];
                int iter    = active ? 200 : 1;
                for (int k = 0; k < iter; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_WorkStealing(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    auto &pool = dispatch::work_stealing_pool::instance();

    for (auto _: state) {
        pool.parallel_for(int64_t{0}, numel, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                bool active = (i % 10) == 0;
                float val   = ip[i];
                int iter    = active ? 200 : 1;
                for (int k = 0; k < iter; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

static void
BM_Unbalanced_ATParallel(benchmark::State &state) {
    SetupThreads();
    const int64_t numel = state.range(0);
    auto input          = torch::randn({numel});
    auto output         = torch::empty_like(input);

    for (auto _: state) {
        at::parallel_for(0, numel, 2048, [&](int64_t begin, int64_t end) {
            const float *ip = input.data_ptr<float>();
            float *op       = output.data_ptr<float>();
            for (int64_t i = begin; i < end; ++i) {
                bool active = (i % 10) == 0;
                float val   = ip[i];
                int iter    = active ? 200 : 1;
                for (int k = 0; k < iter; ++k)
                    val = val * 0.999f + 0.001f;
                op[i] = val;
            }
        });
        benchmark::DoNotOptimize(output.data_ptr<float>());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * numel);
}

// ----------------------------------------------------------------------------
// Register Benchmarks
// ----------------------------------------------------------------------------

#define BENCH_SIZES        \
    ->Args({100000})       \
        ->Args({1000000})  \
        ->Args({10000000}) \
        ->UseRealTime()    \
        ->Unit(benchmark::kMillisecond)

// Uniform workload
BENCHMARK(BM_Uniform_Serial) BENCH_SIZES;
BENCHMARK(BM_Uniform_OpenMP) BENCH_SIZES;
BENCHMARK(BM_Uniform_QueuePool) BENCH_SIZES;
BENCHMARK(BM_Uniform_SpinPool) BENCH_SIZES;
BENCHMARK(BM_Uniform_Broadcast) BENCH_SIZES;
BENCHMARK(BM_Uniform_WorkStealing) BENCH_SIZES;
BENCHMARK(BM_Uniform_ATParallel) BENCH_SIZES;

// Unbalanced workload
BENCHMARK(BM_Unbalanced_Serial) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_OpenMP_Static) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_OpenMP_Dynamic) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_QueuePool) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_SpinPool) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_Broadcast) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_WorkStealing) BENCH_SIZES;
BENCHMARK(BM_Unbalanced_ATParallel) BENCH_SIZES;

BENCHMARK_MAIN();
