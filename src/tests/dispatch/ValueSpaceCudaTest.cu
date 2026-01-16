// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Ensure ValueSpace.h compiles with nvcc and basic consteval operations
// work in device code. This is NOT an exhaustive test of ValueSpace functionality
// (see ValueSpaceTest.cpp for that). This file tests:
// - Header compiles with nvcc
// - consteval calls work inside kernels
// - Visitor-driven host/device dispatch pattern (the primary use case)

#include <fvdb/detail/dispatch/ValueSpace.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test enums and axis definitions
// =============================================================================

enum class Device { CPU, CUDA };
enum class DType { Float32, Float64, Int32 };

using DeviceAxis = Values<Device::CPU, Device::CUDA>;
using DTypeAxis  = Values<DType::Float32, DType::Float64, DType::Int32>;

// =============================================================================
// Test 1: Basic consteval operations in device code
// =============================================================================
// Smoke test that consteval trait queries compile and execute in kernels.

__global__ void
test_consteval_traits_kernel(size_t *results) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    results[0] = Rank_v<Space>();
    results[1] = Numel_v<Space>();
    results[2] = AxisSize_v<DeviceAxis>();
    results[3] = AxisSize_v<DTypeAxis>();
}

TEST(ValueSpaceCuda, ConstevalTraitsInKernel) {
    constexpr size_t num_results = 4;

    size_t *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_results * sizeof(size_t)), cudaSuccess);

    test_consteval_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, num_results> h_results{};
    ASSERT_EQ(
        cudaMemcpy(
            h_results.data(), d_results, num_results * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{2}) << "Rank<Space>";
    EXPECT_EQ(h_results[1], size_t{6}) << "Numel<Space> (2 devices * 3 dtypes)";
    EXPECT_EQ(h_results[2], size_t{2}) << "AxisSize<DeviceAxis>";
    EXPECT_EQ(h_results[3], size_t{3}) << "AxisSize<DTypeAxis>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 2: Per-coordinate kernel instantiation via visit_value_space
// =============================================================================
// The primary dispatch pattern: visit_value_space iterates over all coordinates
// at compile time, and the visitor launches a kernel for each coordinate.

template <Device D, DType T>
__global__ void
per_coord_kernel(size_t *output) {
    using Space              = ValueAxes<DeviceAxis, DTypeAxis>;
    using Coord              = Values<D, T>;
    using Pt                 = PointFromCoord_t<Space, Coord>;
    constexpr size_t lin_idx = LinearIndexFromPoint_v<IndexSpaceOf_t<Space>, Pt>();

    output[lin_idx] = lin_idx;
}

struct PerCoordKernelLauncher {
    size_t *d_output;

    template <Device D, DType T>
    void
    operator()(Values<D, T>) const {
        per_coord_kernel<D, T><<<1, 1>>>(d_output);
    }
};

TEST(ValueSpaceCuda, PerCoordKernelInstantiation) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    static_assert(Numel_v<Space>() == 6); // 2 devices * 3 dtypes

    size_t *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, Numel_v<Space>() * sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0xFF, Numel_v<Space>() * sizeof(size_t)), cudaSuccess);

    // Launches 6 different kernel instantiations
    visit_value_space(PerCoordKernelLauncher{d_output}, Space{});
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<size_t> h_output(Numel_v<Space>());
    ASSERT_EQ(
        cudaMemcpy(
            h_output.data(), d_output, Numel_v<Space>() * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (size_t i = 0; i < Numel_v<Space>(); ++i) {
        EXPECT_EQ(h_output[i], i) << "Coordinate " << i;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 3: Host/Device dispatch pattern
// =============================================================================
// Tests the realistic pattern where a visitor dispatches to either host or
// device code based on a Device enum value in the coordinate.

template <DType T>
__global__ void
dtype_kernel(size_t *output, size_t idx) {
    output[idx] = static_cast<size_t>(T);
}

template <DType T>
void
dtype_host(size_t *output, size_t idx) {
    output[idx] = static_cast<size_t>(T) + 100; // +100 to distinguish from device
}

struct HostDeviceDispatcher {
    size_t *d_output;
    size_t *h_output;
    mutable size_t call_idx = 0;

    template <Device D, DType T>
    void
    operator()(Values<D, T>) const {
        if constexpr (D == Device::CPU) {
            dtype_host<T>(h_output, call_idx);
        } else {
            static_assert(D == Device::CUDA);
            dtype_kernel<T><<<1, 1>>>(d_output, call_idx);
        }
        ++call_idx;
    }
};

TEST(ValueSpaceCuda, HostDeviceDispatch) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // Visitation order: (CPU, F32), (CPU, F64), (CPU, I32), (CUDA, F32), (CUDA, F64), (CUDA, I32)

    constexpr size_t total = Numel_v<Space>();

    size_t *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, total * sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0xFF, total * sizeof(size_t)), cudaSuccess);

    std::vector<size_t> h_output(total, 0xFF);

    visit_value_space(HostDeviceDispatcher{d_output, h_output.data()}, Space{});
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<size_t> d_results(total);
    ASSERT_EQ(
        cudaMemcpy(d_results.data(), d_output, total * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    // CPU calls (indices 0-2): should have +100 offset
    constexpr size_t num_dtypes = AxisSize_v<DTypeAxis>();
    for (size_t i = 0; i < num_dtypes; ++i) {
        EXPECT_EQ(h_output[i], i + 100) << "CPU dispatch at index " << i;
    }

    // CUDA calls (indices 3-5): raw DType ordinal
    for (size_t i = 0; i < num_dtypes; ++i) {
        size_t cuda_idx = num_dtypes + i;
        EXPECT_EQ(d_results[cuda_idx], i) << "CUDA dispatch at index " << cuda_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

} // namespace dispatch
} // namespace fvdb
