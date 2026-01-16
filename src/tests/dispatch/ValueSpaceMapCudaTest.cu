// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Ensure ValueSpaceMap.h compiles with nvcc. This is NOT an exhaustive
// test of ValueSpaceMap functionality (see ValueSpaceMapTest.cpp for that).
// This file tests:
// - Header compiles with nvcc
// - ValueSpaceMapKey, Hash, Equal compile correctly
// - create_and_store works with a factory that launches CUDA kernels
// - Host/Device dispatch pattern using ValueSpaceMap

#include <fvdb/detail/dispatch/ValueSpaceMap.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test enums and axis definitions
// =============================================================================

enum class Device { CPU, GPU };
enum class DType { Float32, Float64, Int32 };

using DeviceAxis = Values<Device::CPU, Device::GPU>;
using DTypeAxis  = Values<DType::Float32, DType::Float64, DType::Int32>;
using IntAxis    = Values<1, 2, 4, 8>;

// =============================================================================
// Test 1: Basic type instantiation smoke test
// =============================================================================
// Verify all ValueSpaceMap components compile with nvcc.

TEST(ValueSpaceMapCuda, TypeInstantiation) {
    // ValueSpaceMapKey
    using Space1D = ValueAxes<IntAxis>;
    using Key1D   = ValueSpaceMapKey<Space1D>;
    static_assert(std::is_same_v<typename Key1D::space_type, Space1D>);
    static_assert(std::is_same_v<typename Key1D::coord_type, std::tuple<int>>);

    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;
    using Key2D   = ValueSpaceMapKey<Space2D>;
    static_assert(std::is_same_v<typename Key2D::space_type, Space2D>);
    static_assert(std::is_same_v<typename Key2D::coord_type, std::tuple<Device, DType>>);

    // ValueSpaceMapHash
    using Hash2D = ValueSpaceMapHash<Space2D>;
    static_assert(std::is_same_v<typename Hash2D::space_type, Space2D>);

    // ValueSpaceMapEqual
    using Equal2D = ValueSpaceMapEqual<Space2D>;
    static_assert(std::is_same_v<typename Equal2D::space_type, Space2D>);

    // ValueSpaceMap_t
    using Map2D = ValueSpaceMap_t<Space2D, int>;
    static_assert(std::is_same_v<typename Map2D::key_type, Key2D>);

    SUCCEED();
}

// =============================================================================
// Test 2: Key construction and linear index computation
// =============================================================================

TEST(ValueSpaceMapCuda, KeyConstruction) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    using Key   = ValueSpaceMapKey<Space>;

    // Construct from tuple
    Key k00(std::make_tuple(Device::CPU, DType::Float32));
    Key k10(std::make_tuple(Device::GPU, DType::Float32));

    EXPECT_EQ(k00.linear_index, 0u);
    EXPECT_EQ(k10.linear_index, 3u);

    // Construct from ValuePack
    Key kPack(Values<Device::GPU, DType::Float64>{});
    EXPECT_EQ(kPack.linear_index, 4u);
}

// =============================================================================
// Test 3: Map operations compile and work
// =============================================================================

TEST(ValueSpaceMapCuda, BasicMapOperations) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    ValueSpaceMap_t<Space, std::string> map;

    // Emplace with tuple
    map.emplace(std::make_tuple(Device::CPU, DType::Float32), "cpu-f32");
    map.emplace(std::make_tuple(Device::GPU, DType::Float64), "gpu-f64");

    // Emplace with ValuePack
    map.emplace(Values<Device::GPU, DType::Int32>{}, "gpu-i32");

    EXPECT_EQ(map.size(), 3u);

    // Find with tuple
    auto it1 = map.find(std::make_tuple(Device::CPU, DType::Float32));
    ASSERT_NE(it1, map.end());
    EXPECT_EQ(it1->second, "cpu-f32");

    // Find with ValuePack
    auto it2 = map.find(Values<Device::GPU, DType::Float64>{});
    ASSERT_NE(it2, map.end());
    EXPECT_EQ(it2->second, "gpu-f64");

    // Find missing returns end (graceful failure for valid coord not in map)
    EXPECT_EQ(map.find(std::make_tuple(Device::CPU, DType::Float64)), map.end());
}

// =============================================================================
// Test 4: create_and_store with subspace
// =============================================================================

TEST(ValueSpaceMapCuda, CreateAndStoreSubspace) {
    using Space    = ValueAxes<DeviceAxis, DTypeAxis>;
    using SubSpace = ValueAxes<Values<Device::CPU>, DTypeAxis>; // CPU only
    ValueSpaceMap_t<Space, int> map;

    int counter  = 0;
    auto factory = [&counter](auto) { return ++counter; };

    create_and_store(map, factory, SubSpace{});

    // Should have 3 entries (CPU x 3 dtypes)
    EXPECT_EQ(map.size(), 3u);
    EXPECT_NE(map.find(std::make_tuple(Device::CPU, DType::Float32)), map.end());
    EXPECT_NE(map.find(std::make_tuple(Device::CPU, DType::Float64)), map.end());
    EXPECT_NE(map.find(std::make_tuple(Device::CPU, DType::Int32)), map.end());

    // GPU entries not stored
    EXPECT_EQ(map.find(std::make_tuple(Device::GPU, DType::Float32)), map.end());
}

// =============================================================================
// Test 5: CUDA kernel integration - factory launches kernel
// =============================================================================
// This is the key test: create_and_store with a factory that launches CUDA kernels.

__global__ void
fillKernel(int *output, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = value;
    }
}

// A simple result holder for dispatch testing
struct DispatchResult {
    int *d_data;
    int n;
    int expectedValue;

    DispatchResult(int val, int count) : n(count), expectedValue(val) {
        cudaMalloc(&d_data, n * sizeof(int));
        cudaMemset(d_data, 0, n * sizeof(int));
    }

    ~DispatchResult() {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    // Non-copyable
    DispatchResult(const DispatchResult &)            = delete;
    DispatchResult &operator=(const DispatchResult &) = delete;

    // Movable
    DispatchResult(DispatchResult &&other) noexcept
        : d_data(other.d_data), n(other.n), expectedValue(other.expectedValue) {
        other.d_data = nullptr;
    }
    DispatchResult &
    operator=(DispatchResult &&other) noexcept {
        if (this != &other) {
            if (d_data)
                cudaFree(d_data);
            d_data        = other.d_data;
            n             = other.n;
            expectedValue = other.expectedValue;
            other.d_data  = nullptr;
        }
        return *this;
    }

    bool
    verify() const {
        if (!d_data)
            return false;
        std::vector<int> h_data(n);
        cudaMemcpy(h_data.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            if (h_data[i] != expectedValue)
                return false;
        }
        return true;
    }
};

// Factory that creates DispatchResult with kernel launch for GPU, host fill for CPU
struct DeviceDTypeFactory {
    static constexpr int N = 256;

    template <Device D, DType T>
    DispatchResult
    operator()(Values<D, T>) const {
        // Use dtype ordinal + device offset as the fill value
        int value = static_cast<int>(T) + (D == Device::GPU ? 100 : 0);

        DispatchResult result(value, N);

        if constexpr (D == Device::GPU) {
            // Launch kernel
            const int blockSize = 64;
            const int numBlocks = (N + blockSize - 1) / blockSize;
            fillKernel<<<numBlocks, blockSize>>>(result.d_data, value, N);
        } else {
            // CPU path: fill via memset (for simplicity, just use CUDA API from host)
            // In practice you'd do this differently, but this tests the dispatch pattern
            std::vector<int> h_data(N, value);
            cudaMemcpy(result.d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        }

        return result;
    }
};

TEST(ValueSpaceMapCuda, CreateAndStoreWithKernelLaunch) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    ValueSpaceMap_t<Space, DispatchResult> map;

    DeviceDTypeFactory factory;
    create_and_store(map, factory, Space{});

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Should have all 6 entries (2 devices x 3 dtypes)
    EXPECT_EQ(map.size(), 6u);

    // Verify GPU entries launched kernels correctly
    auto itGpuF32 = map.find(Values<Device::GPU, DType::Float32>{});
    ASSERT_NE(itGpuF32, map.end());
    EXPECT_TRUE(itGpuF32->second.verify()) << "GPU Float32 failed";

    auto itGpuF64 = map.find(Values<Device::GPU, DType::Float64>{});
    ASSERT_NE(itGpuF64, map.end());
    EXPECT_TRUE(itGpuF64->second.verify()) << "GPU Float64 failed";

    auto itGpuI32 = map.find(Values<Device::GPU, DType::Int32>{});
    ASSERT_NE(itGpuI32, map.end());
    EXPECT_TRUE(itGpuI32->second.verify()) << "GPU Int32 failed";

    // Verify CPU entries
    auto itCpuF32 = map.find(Values<Device::CPU, DType::Float32>{});
    ASSERT_NE(itCpuF32, map.end());
    EXPECT_TRUE(itCpuF32->second.verify()) << "CPU Float32 failed";
}

// =============================================================================
// Test 6: Dispatch table pattern with function pointers
// =============================================================================
// Simulates a dispatch table where each entry is a function pointer.

using DispatchFn = void (*)(int *, int, int);

__global__ void
gpuFillKernel(int *output, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = value;
    }
}

template <Device D, DType T> void dispatchImpl(int *output, int value, int n);

template <>
void
dispatchImpl<Device::CPU, DType::Float32>(int *output, int value, int n) {
    std::vector<int> h_data(n, value + 1000);
    cudaMemcpy(output, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

template <>
void
dispatchImpl<Device::CPU, DType::Float64>(int *output, int value, int n) {
    std::vector<int> h_data(n, value + 2000);
    cudaMemcpy(output, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

template <>
void
dispatchImpl<Device::GPU, DType::Float32>(int *output, int value, int n) {
    const int blockSize = 64;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    gpuFillKernel<<<numBlocks, blockSize>>>(output, value + 3000, n);
}

template <>
void
dispatchImpl<Device::GPU, DType::Float64>(int *output, int value, int n) {
    const int blockSize = 64;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    gpuFillKernel<<<numBlocks, blockSize>>>(output, value + 4000, n);
}

struct DispatchFnFactory {
    template <Device D, DType T>
    DispatchFn
    operator()(Values<D, T>) const {
        return &dispatchImpl<D, T>;
    }
};

TEST(ValueSpaceMapCuda, DispatchTableWithFunctionPointers) {
    // Subspace: only CPU and GPU with Float32 and Float64
    using Space    = ValueAxes<DeviceAxis, DTypeAxis>;
    using SubSpace = ValueAxes<DeviceAxis, Values<DType::Float32, DType::Float64>>;
    ValueSpaceMap_t<Space, DispatchFn> dispatchTable;

    DispatchFnFactory factory;
    create_and_store(dispatchTable, factory, SubSpace{});

    EXPECT_EQ(dispatchTable.size(), 4u);

    // Allocate output buffer
    constexpr int N = 128;
    int *d_output   = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, N * sizeof(int)), cudaSuccess);

    // Dispatch to GPU Float32
    {
        auto it = dispatchTable.find(std::make_tuple(Device::GPU, DType::Float32));
        ASSERT_NE(it, dispatchTable.end());
        it->second(d_output, 42, N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<int> h_result(N);
        cudaMemcpy(h_result.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
        EXPECT_EQ(h_result[0], 42 + 3000);
    }

    // Dispatch to CPU Float64
    {
        auto it = dispatchTable.find(std::make_tuple(Device::CPU, DType::Float64));
        ASSERT_NE(it, dispatchTable.end());
        it->second(d_output, 7, N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<int> h_result(N);
        cudaMemcpy(h_result.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
        EXPECT_EQ(h_result[0], 7 + 2000);
    }

    // Int32 not in subspace - should not be in dispatch table
    EXPECT_EQ(dispatchTable.find(std::make_tuple(Device::GPU, DType::Int32)), dispatchTable.end());

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 7: Hash and Equal with ValuePack in device code context
// =============================================================================

TEST(ValueSpaceMapCuda, HashAndEqualWithValuePack) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    ValueSpaceMapHash<Space> hash;
    ValueSpaceMapEqual<Space> equal;

    // Hash with ValuePack should match hash with tuple
    auto hashPack  = hash(Values<Device::GPU, DType::Float32>{});
    auto hashTuple = hash(std::make_tuple(Device::GPU, DType::Float32));
    EXPECT_EQ(hashPack, hashTuple);
    EXPECT_EQ(hashPack, 3u); // GPU=1, Float32=0 -> linear index 3

    // Equal with ValuePack
    using Key = ValueSpaceMapKey<Space>;
    Key key(std::make_tuple(Device::GPU, DType::Float32));

    EXPECT_TRUE(equal(Values<Device::GPU, DType::Float32>{}, key));
    EXPECT_TRUE(equal(key, Values<Device::GPU, DType::Float32>{}));
    EXPECT_FALSE(equal(Values<Device::CPU, DType::Float32>{}, key));
}

// =============================================================================
// Test 8: Three-axis space compiles
// =============================================================================

TEST(ValueSpaceMapCuda, ThreeAxisSpace) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;
    ValueSpaceMap_t<Space, int> map;

    // Insert and find with 3-tuple
    map.emplace(std::make_tuple(Device::GPU, DType::Float32, 4), 42);

    auto it = map.find(std::make_tuple(Device::GPU, DType::Float32, 4));
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->second, 42);

    // With ValuePack
    map.emplace(Values<Device::CPU, DType::Int32, 8>{}, 99);
    auto it2 = map.find(Values<Device::CPU, DType::Int32, 8>{});
    ASSERT_NE(it2, map.end());
    EXPECT_EQ(it2->second, 99);
}

// =============================================================================
// Test 9: CreateAndStoreVisitor compiles with nvcc
// =============================================================================

TEST(ValueSpaceMapCuda, CreateAndStoreVisitorCompiles) {
    using Space = ValueAxes<IntAxis>;
    ValueSpaceMap_t<Space, int> map;

    auto factory = [](auto coord) { return std::get<0>(coordToTuple(coord)) * 2; };

    // CreateAndStoreVisitor is used internally by create_and_store
    // Just verify it works
    create_and_store(map, factory, Space{});

    EXPECT_EQ(map.size(), 4u);
    EXPECT_EQ(map.find(Values<1>{})->second, 2);
    EXPECT_EQ(map.find(Values<2>{})->second, 4);
    EXPECT_EQ(map.find(Values<4>{})->second, 8);
    EXPECT_EQ(map.find(Values<8>{})->second, 16);
}

} // namespace dispatch
} // namespace fvdb
