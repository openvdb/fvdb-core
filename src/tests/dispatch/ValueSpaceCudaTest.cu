// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Ensure ValueSpace.h and underlying machinery (Values.h, IndexSpace.h)
// compile correctly with nvcc and work in device code. This is NOT an exhaustive
// test of ValueSpace functionality (see ValueSpaceTest.cpp for that). Instead,
// this tests a representative cross-section to catch nvcc-specific issues like:
// - Template instantiation explosions
// - Concept/requires clause compatibility
// - consteval function usage in device code
// - Visitor-driven kernel instantiation patterns

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
using IntAxis    = Values<1, 2, 4>;

// =============================================================================
// Test 1: Per-coordinate kernel instantiation via visit_value_space
// =============================================================================
// This is the primary pattern used in dispatch: visit_value_space iterates
// over all coordinates at compile time, and the visitor launches a kernel
// instantiated for each specific coordinate.

template <Device D, DType T>
__global__ void
per_coord_value_kernel(size_t *output) {
    using Space              = ValueAxes<DeviceAxis, DTypeAxis>;
    using Coord              = Values<D, T>;
    using Pt                 = PointFromCoord_t<Space, Coord>;
    constexpr size_t lin_idx = LinearIndexFromPoint_v<IndexSpaceOf_t<Space>, Pt>();

    // Write: [linear_index, device_ordinal, dtype_ordinal]
    output[lin_idx * 3 + 0] = lin_idx;
    output[lin_idx * 3 + 1] = static_cast<size_t>(D);
    output[lin_idx * 3 + 2] = static_cast<size_t>(T);
}

struct ValueKernelLauncher {
    size_t *d_output;

    template <Device D, DType T>
    void
    operator()(Values<D, T>) const {
        per_coord_value_kernel<D, T><<<1, 1>>>(d_output);
    }
};

TEST(ValueSpaceCuda, PerCoordKernelInstantiation) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    static_assert(Rank_v<Space>() == 2);
    static_assert(Numel_v<Space>() == 6); // 2 devices * 3 dtypes

    // Allocate device memory: 3 values per coordinate
    size_t *d_output         = nullptr;
    const size_t output_size = Numel_v<Space>() * 3 * sizeof(size_t);
    ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0xFF, output_size), cudaSuccess);

    // Visit all coordinates, launching one kernel per (Device, DType) combination.
    // This instantiates 6 different kernels.
    visit_value_space(ValueKernelLauncher{d_output}, Space{});

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    std::vector<size_t> h_output(Numel_v<Space>() * 3);
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify each coordinate wrote correct values
    // Row-major order: Device varies slowest, DType fastest
    for (size_t lin_idx = 0; lin_idx < Numel_v<Space>(); ++lin_idx) {
        constexpr size_t dtype_count = AxisSize_v<DTypeAxis>();
        const size_t expected_device = lin_idx / dtype_count;
        const size_t expected_dtype  = lin_idx % dtype_count;

        EXPECT_EQ(h_output[lin_idx * 3 + 0], lin_idx) << "Linear index mismatch at " << lin_idx;
        EXPECT_EQ(h_output[lin_idx * 3 + 1], expected_device)
            << "Device mismatch at linear index " << lin_idx;
        EXPECT_EQ(h_output[lin_idx * 3 + 2], expected_dtype)
            << "DType mismatch at linear index " << lin_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 2: ValueSpace traits in device code
// =============================================================================

__global__ void
test_value_space_traits_kernel(size_t *results) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;
    using Space3D = ValueAxes<DeviceAxis, DTypeAxis, IntAxis>;

    results[0] = Rank_v<Space1D>();
    results[1] = Rank_v<Space2D>();
    results[2] = Rank_v<Space3D>();
    results[3] = Numel_v<Space1D>();
    results[4] = Numel_v<Space2D>();
    results[5] = Numel_v<Space3D>();
}

TEST(ValueSpaceCuda, TraitsOnDevice) {
    constexpr size_t num_tests = 6;

    size_t *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(size_t)), cudaSuccess);

    test_value_space_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{1}) << "Rank<Space1D>";
    EXPECT_EQ(h_results[1], size_t{2}) << "Rank<Space2D>";
    EXPECT_EQ(h_results[2], size_t{3}) << "Rank<Space3D>";
    EXPECT_EQ(h_results[3], size_t{2}) << "Numel<Space1D>";  // 2 devices
    EXPECT_EQ(h_results[4], size_t{6}) << "Numel<Space2D>";  // 2 * 3
    EXPECT_EQ(h_results[5], size_t{18}) << "Numel<Space3D>"; // 2 * 3 * 3

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 3: Compile-time coordinate conversion in device code
// =============================================================================
// Tests CoordFromPoint and PointFromCoord round-trip at compile time.

template <typename Space, size_t... Is>
__global__ void
test_coord_round_trip_kernel(size_t *result) {
    using Pt       = Point<Is...>;
    using Coord    = CoordFromPoint_t<Space, Pt>;
    using PtBack   = PointFromCoord_t<Space, Coord>;
    using IdxSpace = IndexSpaceOf_t<Space>;
    *result        = LinearIndexFromPoint_v<IdxSpace, PtBack>();
}

TEST(ValueSpaceCuda, CoordConversionOnDevice) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    size_t *d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(size_t)), cudaSuccess);

    size_t h_result{};

    // Test Point<0, 0> -> Values<CPU, Float32> -> Point<0, 0>
    test_coord_round_trip_kernel<Space, 0, 0><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{0});

    // Test Point<1, 2> -> Values<CUDA, Int32> -> Point<1, 2>
    test_coord_round_trip_kernel<Space, 1, 2><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{5}); // linear index of (1, 2) in 2x3 space

    ASSERT_EQ(cudaFree(d_result), cudaSuccess);
}

// =============================================================================
// Test 4: ValueAxis concept test helpers in device code
// =============================================================================

__global__ void
test_value_axis_concepts_kernel(bool *results) {
    using ValidAxis     = Values<1, 2, 3>;
    using EmptyAxis     = Values<>;
    using DuplicateAxis = Values<1, 2, 1>;
    using MixedAxis     = Values<1, 'x', true>;

    results[0] = is_value_axis<ValidAxis>();
    results[1] = is_value_axis<DeviceAxis>();
    results[2] = is_value_axis<DTypeAxis>();
    results[3] = !is_value_axis<EmptyAxis>();
    results[4] = !is_value_axis<DuplicateAxis>();
    results[5] = !is_value_axis<MixedAxis>();
    results[6] = !is_value_axis<int>();
}

TEST(ValueSpaceCuda, AxisConceptsOnDevice) {
    constexpr size_t num_tests = 7;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_value_axis_concepts_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "is_value_axis<ValidAxis>";
    EXPECT_TRUE(h_results[1]) << "is_value_axis<DeviceAxis>";
    EXPECT_TRUE(h_results[2]) << "is_value_axis<DTypeAxis>";
    EXPECT_TRUE(h_results[3]) << "!is_value_axis<EmptyAxis>";
    EXPECT_TRUE(h_results[4]) << "!is_value_axis<DuplicateAxis>";
    EXPECT_TRUE(h_results[5]) << "!is_value_axis<MixedAxis>";
    EXPECT_TRUE(h_results[6]) << "!is_value_axis<int>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 5: ValueAxis traits in device code
// =============================================================================

__global__ void
test_value_axis_traits_kernel(size_t *results) {
    results[0] = AxisSize_v<DeviceAxis>();
    results[1] = AxisSize_v<DTypeAxis>();
    results[2] = AxisSize_v<IntAxis>();
    results[3] = static_cast<size_t>(AxisElement_v<DeviceAxis, 0>());
    results[4] = static_cast<size_t>(AxisElement_v<DeviceAxis, 1>());
    results[5] = AxisContains_v<IntAxis, 2>() ? 1 : 0;
    results[6] = AxisContains_v<IntAxis, 3>() ? 1 : 0; // 3 not in {1, 2, 4}
    results[7] = AxisIndex_v<DTypeAxis, DType::Int32>();
}

TEST(ValueSpaceCuda, AxisTraitsOnDevice) {
    constexpr size_t num_tests = 8;

    size_t *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(size_t)), cudaSuccess);

    test_value_axis_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{2}) << "AxisSize<DeviceAxis>";
    EXPECT_EQ(h_results[1], size_t{3}) << "AxisSize<DTypeAxis>";
    EXPECT_EQ(h_results[2], size_t{3}) << "AxisSize<IntAxis>";
    EXPECT_EQ(h_results[3], static_cast<size_t>(Device::CPU)) << "AxisElement<DeviceAxis, 0>";
    EXPECT_EQ(h_results[4], static_cast<size_t>(Device::CUDA)) << "AxisElement<DeviceAxis, 1>";
    EXPECT_EQ(h_results[5], size_t{1}) << "AxisContains<IntAxis, 2>";
    EXPECT_EQ(h_results[6], size_t{0}) << "!AxisContains<IntAxis, 3>";
    EXPECT_EQ(h_results[7], size_t{2}) << "AxisIndex<DTypeAxis, Int32>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 6: ValueSpace concept test helpers in device code
// =============================================================================

__global__ void
test_value_space_concepts_kernel(bool *results) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;

    results[0] = is_value_space<Space1D>();
    results[1] = is_value_space<Space2D>();
    results[2] = !is_value_space<DeviceAxis>(); // Axis, not space
    results[3] = !is_value_space<int>();
}

TEST(ValueSpaceCuda, SpaceConceptsOnDevice) {
    constexpr size_t num_tests = 4;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_value_space_concepts_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "is_value_space<Space1D>";
    EXPECT_TRUE(h_results[1]) << "is_value_space<Space2D>";
    EXPECT_TRUE(h_results[2]) << "!is_value_space<DeviceAxis>";
    EXPECT_TRUE(h_results[3]) << "!is_value_space<int>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 7: SpaceContains and CoordTypesMatch in device code
// =============================================================================

__global__ void
test_space_membership_kernel(bool *results) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Valid coordinates
    results[0] = space_contains_v<Space, Values<Device::CPU, DType::Float32>>();
    results[1] = space_contains_v<Space, Values<Device::CUDA, DType::Int32>>();

    // Type matching (coord types match even if values hypothetically invalid)
    results[2] = coord_types_match_v<Space, Values<Device::CPU, DType::Float64>>();

    // SpaceContains implies CoordTypesMatch
    results[3] = SpaceContains<Space, Values<Device::CUDA, DType::Float64>>;
    results[4] = CoordTypesMatch<Space, Values<Device::CUDA, DType::Float64>>;
}

TEST(ValueSpaceCuda, SpaceMembershipOnDevice) {
    constexpr size_t num_tests = 5;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_space_membership_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "space_contains<CPU, Float32>";
    EXPECT_TRUE(h_results[1]) << "space_contains<CUDA, Int32>";
    EXPECT_TRUE(h_results[2]) << "coord_types_match<CPU, Float64>";
    EXPECT_TRUE(h_results[3]) << "SpaceContains<CUDA, Float64>";
    EXPECT_TRUE(h_results[4]) << "CoordTypesMatch<CUDA, Float64>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 8: Host/Device dispatch pattern via visitor
// =============================================================================
// This tests the pattern where a visitor dispatches to either host or device
// code based on a Device enum value in the coordinate.

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

struct DeviceDispatchVisitor {
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

TEST(ValueSpaceCuda, DeviceDispatchPattern) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // Order: (CPU, F32), (CPU, F64), (CPU, I32), (CUDA, F32), (CUDA, F64), (CUDA, I32)

    constexpr size_t total   = Numel_v<Space>();
    const size_t output_size = total * sizeof(size_t);

    size_t *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0xFF, output_size), cudaSuccess);

    std::vector<size_t> h_output(total, 0xFF);

    visit_value_space(DeviceDispatchVisitor{d_output, h_output.data()}, Space{});
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<size_t> d_results(total);
    ASSERT_EQ(cudaMemcpy(d_results.data(), d_output, output_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify CPU calls (indices 0-2): should have +100
    for (size_t i = 0; i < AxisSize_v<DTypeAxis>(); ++i) {
        EXPECT_EQ(h_output[i], i + 100) << "CPU dispatch at index " << i;
    }

    // Verify CUDA calls (indices 3-5): should be raw DType ordinal
    for (size_t i = 0; i < AxisSize_v<DTypeAxis>(); ++i) {
        size_t cuda_idx = AxisSize_v<DTypeAxis>() + i;
        EXPECT_EQ(d_results[cuda_idx], i) << "CUDA dispatch at index " << cuda_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 9: CoordFromLinearIndex in device code
// =============================================================================

template <typename Space, size_t linearIndex>
__global__ void
test_coord_from_linear_index_kernel(size_t *result) {
    using Coord    = CoordFromLinearIndex_t<Space, linearIndex>;
    using IdxSpace = IndexSpaceOf_t<Space>;
    using Pt       = PointFromCoord_t<Space, Coord>;
    // Return the linear index from the resulting coord to verify round-trip
    *result = LinearIndexFromPoint_v<IdxSpace, Pt>();
}

TEST(ValueSpaceCuda, CoordFromLinearIndexOnDevice) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    size_t *d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(size_t)), cudaSuccess);

    size_t h_result{};

    // Test linear index 0 -> coord -> back to linear index 0
    test_coord_from_linear_index_kernel<Space, 0><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{0});

    // Test linear index 3 (second device, first dtype)
    test_coord_from_linear_index_kernel<Space, 3><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{3});

    // Test linear index 5 (last valid index in 2x3 space)
    test_coord_from_linear_index_kernel<Space, 5><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{5});

    ASSERT_EQ(cudaFree(d_result), cudaSuccess);
}

__global__ void
test_coord_from_linear_index_traits_kernel(bool *results) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA}, DTypeAxis = {Float32, Float64, Int32}
    // Linear 0 -> (0,0) -> Values<CPU, Float32>
    // Linear 1 -> (0,1) -> Values<CPU, Float64>
    // Linear 3 -> (1,0) -> Values<CUDA, Float32>
    // Linear 5 -> (1,2) -> Values<CUDA, Int32>

    results[0] =
        std::is_same_v<CoordFromLinearIndex_t<Space, 0>, Values<Device::CPU, DType::Float32>>;
    results[1] =
        std::is_same_v<CoordFromLinearIndex_t<Space, 1>, Values<Device::CPU, DType::Float64>>;
    results[2] =
        std::is_same_v<CoordFromLinearIndex_t<Space, 3>, Values<Device::CUDA, DType::Float32>>;
    results[3] =
        std::is_same_v<CoordFromLinearIndex_t<Space, 5>, Values<Device::CUDA, DType::Int32>>;

    // Verify CoordFromLinearIndex matches CoordFromPoint(PointFromLinearIndex)
    using IdxSpace = IndexSpaceOf_t<Space>;
    results[4]     = std::is_same_v<CoordFromLinearIndex_t<Space, 2>,
                                    CoordFromPoint_t<Space, PointFromLinearIndex_t<IdxSpace, 2>>>;
    results[5]     = std::is_same_v<CoordFromLinearIndex_t<Space, 4>,
                                    CoordFromPoint_t<Space, PointFromLinearIndex_t<IdxSpace, 4>>>;
}

TEST(ValueSpaceCuda, CoordFromLinearIndexTraitsOnDevice) {
    constexpr size_t num_tests = 6;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_coord_from_linear_index_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "CoordFromLinearIndex<0> == Values<CPU, Float32>";
    EXPECT_TRUE(h_results[1]) << "CoordFromLinearIndex<1> == Values<CPU, Float64>";
    EXPECT_TRUE(h_results[2]) << "CoordFromLinearIndex<3> == Values<CUDA, Float32>";
    EXPECT_TRUE(h_results[3]) << "CoordFromLinearIndex<5> == Values<CUDA, Int32>";
    EXPECT_TRUE(h_results[4]) << "CoordFromLinearIndex matches CoordFromPoint chain (idx 2)";
    EXPECT_TRUE(h_results[5]) << "CoordFromLinearIndex matches CoordFromPoint chain (idx 4)";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 10: LinearIndexFromCoord in device code
// =============================================================================

template <typename Space, typename Coord>
__global__ void
test_linear_index_from_coord_kernel(size_t *result) {
    *result = LinearIndexFromCoord_v<Space, Coord>();
}

TEST(ValueSpaceCuda, LinearIndexFromCoordOnDevice) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    size_t *d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(size_t)), cudaSuccess);

    size_t h_result{};

    // Test Values<CPU, Float32> -> linear index 0
    test_linear_index_from_coord_kernel<Space, Values<Device::CPU, DType::Float32>>
        <<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{0});

    // Test Values<CUDA, Float32> -> linear index 3
    test_linear_index_from_coord_kernel<Space, Values<Device::CUDA, DType::Float32>>
        <<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{3});

    // Test Values<CUDA, Int32> -> linear index 5
    test_linear_index_from_coord_kernel<Space, Values<Device::CUDA, DType::Int32>>
        <<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{5});

    ASSERT_EQ(cudaFree(d_result), cudaSuccess);
}

__global__ void
test_linear_index_from_coord_traits_kernel(size_t *results) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;
    // DeviceAxis = {CPU, CUDA}, DTypeAxis = {Float32, Float64, Int32}
    // Values<CPU, Float32> -> 0
    // Values<CPU, Float64> -> 1
    // Values<CUDA, Float32> -> 3
    // Values<CUDA, Int32> -> 5

    results[0] = LinearIndexFromCoord_v<Space, Values<Device::CPU, DType::Float32>>();
    results[1] = LinearIndexFromCoord_v<Space, Values<Device::CPU, DType::Float64>>();
    results[2] = LinearIndexFromCoord_v<Space, Values<Device::CUDA, DType::Float32>>();
    results[3] = LinearIndexFromCoord_v<Space, Values<Device::CUDA, DType::Int32>>();

    // Verify LinearIndexFromCoord matches LinearIndexFromPoint(PointFromCoord)
    using IdxSpace = IndexSpaceOf_t<Space>;
    results[4] =
        LinearIndexFromPoint_v<IdxSpace,
                               PointFromCoord_t<Space, Values<Device::CPU, DType::Int32>>>();
    results[5] =
        LinearIndexFromPoint_v<IdxSpace,
                               PointFromCoord_t<Space, Values<Device::CUDA, DType::Float64>>>();
}

TEST(ValueSpaceCuda, LinearIndexFromCoordTraitsOnDevice) {
    constexpr size_t num_tests = 6;
    size_t *d_results          = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(size_t)), cudaSuccess);

    test_linear_index_from_coord_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // std::array<size_t, num_tests> h_results{};
    std::vector<size_t> h_results;
    h_results.resize(num_tests, 0);
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{0}) << "LinearIndexFromCoord<CPU, Float32>";
    EXPECT_EQ(h_results[1], size_t{1}) << "LinearIndexFromCoord<CPU, Float64>";
    EXPECT_EQ(h_results[2], size_t{3}) << "LinearIndexFromCoord<CUDA, Float32>";
    EXPECT_EQ(h_results[3], size_t{5}) << "LinearIndexFromCoord<CUDA, Int32>";
    // Verify chain matches: LinearIndexFromCoord == LinearIndexFromPoint(PointFromCoord)
    EXPECT_EQ(h_results[4], size_t{2}) << "LinearIndexFromPoint chain (CPU, Int32)";
    EXPECT_EQ(h_results[5], size_t{4}) << "LinearIndexFromPoint chain (CUDA, Float64)";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

__global__ void
test_linear_index_round_trip_kernel(bool *results) {
    using Space = ValueAxes<DeviceAxis, DTypeAxis>;

    // Verify LinearIndexFromCoord(CoordFromLinearIndex(i)) == i
    results[0] = LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 0>>() == 0;
    results[1] = LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 3>>() == 3;
    results[2] = LinearIndexFromCoord_v<Space, CoordFromLinearIndex_t<Space, 5>>() == 5;

    // Verify CoordFromLinearIndex(LinearIndexFromCoord(c)) == c
    using Coord1 = Values<Device::CPU, DType::Float32>;
    using Coord2 = Values<Device::CUDA, DType::Int32>;
    results[3] =
        std::is_same_v<CoordFromLinearIndex_t<Space, LinearIndexFromCoord_v<Space, Coord1>()>,
                       Coord1>;
    results[4] =
        std::is_same_v<CoordFromLinearIndex_t<Space, LinearIndexFromCoord_v<Space, Coord2>()>,
                       Coord2>;
}

TEST(ValueSpaceCuda, LinearIndexRoundTripOnDevice) {
    constexpr size_t num_tests = 5;
    bool *d_results            = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_linear_index_round_trip_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // std::array<bool, num_tests> h_results{};
    //  std::vector<bool> h_results;
    //  h_results.resize(num_tests, false);
    bool h_results[num_tests] = {false, false, false, false, false};
    ASSERT_EQ(
        cudaMemcpy((void *)h_results, d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "LinearIndexFromCoord(CoordFromLinearIndex(0)) == 0";
    EXPECT_TRUE(h_results[1]) << "LinearIndexFromCoord(CoordFromLinearIndex(3)) == 3";
    EXPECT_TRUE(h_results[2]) << "LinearIndexFromCoord(CoordFromLinearIndex(5)) == 5";
    EXPECT_TRUE(h_results[3]) << "CoordFromLinearIndex(LinearIndexFromCoord(Coord1)) == Coord1";
    EXPECT_TRUE(h_results[4]) << "CoordFromLinearIndex(LinearIndexFromCoord(Coord2)) == Coord2";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 11: IndexSpaceOf mapping in device code
// =============================================================================

__global__ void
test_index_space_of_kernel(size_t *results) {
    using Space1D = ValueAxes<DeviceAxis>;
    using Space2D = ValueAxes<DeviceAxis, DTypeAxis>;

    using IdxSpace1D = IndexSpaceOf_t<Space1D>;
    using IdxSpace2D = IndexSpaceOf_t<Space2D>;

    // Verify the index spaces have correct numels
    results[0] = Numel_v<IdxSpace1D>();
    results[1] = Numel_v<IdxSpace2D>();
    results[2] = Rank_v<IdxSpace1D>();
    results[3] = Rank_v<IdxSpace2D>();
}

TEST(ValueSpaceCuda, IndexSpaceOfOnDevice) {
    constexpr size_t num_tests = 4;

    size_t *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(size_t)), cudaSuccess);

    test_index_space_of_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{2}) << "Numel<IdxSpace1D>";
    EXPECT_EQ(h_results[1], size_t{6}) << "Numel<IdxSpace2D>";
    EXPECT_EQ(h_results[2], size_t{1}) << "Rank<IdxSpace1D>";
    EXPECT_EQ(h_results[3], size_t{2}) << "Rank<IdxSpace2D>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

} // namespace dispatch
} // namespace fvdb
