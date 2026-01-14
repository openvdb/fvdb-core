// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/AxisOuterProduct.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test axis types: Device (HOST/DEVICE) x ScalarType (representative set)
// =============================================================================

enum class Device { kHOST, kDEVICE };
enum class ScalarType { kFloat, kDouble, kHalf, kInt32, kInt64, kInt16, kInt8, kUInt8 };

using DeviceAxis     = DimensionalAxis<Device::kHOST, Device::kDEVICE>;
using ScalarTypeAxis = DimensionalAxis<ScalarType::kFloat,
                                       ScalarType::kDouble,
                                       ScalarType::kHalf,
                                       ScalarType::kInt32,
                                       ScalarType::kInt64,
                                       ScalarType::kInt16,
                                       ScalarType::kInt8,
                                       ScalarType::kUInt8>;

using TestSpace = AxisOuterProduct<DeviceAxis, ScalarTypeAxis>;

static_assert(TestSpace::rank == 2);
static_assert(TestSpace::numel == 16); // 2 devices * 8 scalar types

// =============================================================================
// Device kernel: one instantiation per ScalarType, tagged with compile-time index
// =============================================================================

template <size_t ScalarTypeIndex>
__global__ void
scalar_type_kernel(size_t *output, size_t output_offset) {
    output[output_offset * 2 + 0] = static_cast<size_t>(Device::kDEVICE);
    output[output_offset * 2 + 1] = ScalarTypeIndex;
}

// =============================================================================
// Host function: one instantiation per ScalarType (no kernel launch)
// =============================================================================

template <size_t ScalarTypeIndex>
void
scalar_type_host(size_t *output, size_t output_offset) {
    output[output_offset * 2 + 0] = static_cast<size_t>(Device::kHOST);
    output[output_offset * 2 + 1] = ScalarTypeIndex;
}

// =============================================================================
// Visitor that dispatches to host function or device kernel based on Device value
// =============================================================================

struct DeviceAwareVisitor {
    size_t *d_output;
    size_t *h_output;
    mutable size_t call_index = 0;

    template <Device D, ScalarType S>
    void
    operator()(AnyTypeValuePack<D, S>) const {
        // Get the scalar type index at compile time
        constexpr auto maybe_stype_idx = ScalarTypeAxis::index_of_value(S);
        static_assert(maybe_stype_idx.has_value());
        constexpr size_t stype_idx = *maybe_stype_idx;

        if constexpr (D == Device::kHOST) {
            // Host path: direct CPU execution, no kernel
            scalar_type_host<stype_idx>(h_output, call_index);
        } else {
            // Device path: launch a CUDA kernel
            static_assert(D == Device::kDEVICE);
            scalar_type_kernel<stype_idx><<<1, 1>>>(d_output, call_index);
        }
        ++call_index;
    }
};

TEST(AxisOuterProductCuda, PerCoordInstantiation) {
    // Allocate device memory for DEVICE results
    size_t *d_output         = nullptr;
    const size_t output_size = TestSpace::numel * 2 * sizeof(size_t);
    ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0xFF, output_size), cudaSuccess);

    // Host memory for HOST results
    std::vector<size_t> h_output(TestSpace::numel * 2, 0xFF);

    // Visit all (Device, ScalarType) combinations
    TestSpace::visit(DeviceAwareVisitor{d_output, h_output.data()});

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy device results to host
    std::vector<size_t> d_results(TestSpace::numel * 2);
    ASSERT_EQ(cudaMemcpy(d_results.data(), d_output, output_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify results: visit order is row-major (device varies slowest, stype fastest)
    // Indices 0-7: HOST + stype 0-7
    // Indices 8-15: DEVICE + stype 0-7
    for (size_t stype_idx = 0; stype_idx < ScalarTypeAxis::size; ++stype_idx) {
        // HOST results (indices 0-7)
        size_t host_idx = stype_idx;
        EXPECT_EQ(h_output[host_idx * 2 + 0], static_cast<size_t>(Device::kHOST))
            << "HOST device mismatch at stype " << stype_idx;
        EXPECT_EQ(h_output[host_idx * 2 + 1], stype_idx)
            << "HOST stype index mismatch at " << stype_idx;

        // DEVICE results (indices 8-15)
        size_t device_idx = ScalarTypeAxis::size + stype_idx;
        EXPECT_EQ(d_results[device_idx * 2 + 0], static_cast<size_t>(Device::kDEVICE))
            << "DEVICE device mismatch at stype " << stype_idx;
        EXPECT_EQ(d_results[device_idx * 2 + 1], stype_idx)
            << "DEVICE stype index mismatch at " << stype_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test that AxisOuterProduct type computations work in device code
// =============================================================================

template <typename Space>
__global__ void
test_space_properties_kernel(size_t *rank_out, size_t *numel_out) {
    *rank_out  = Space::rank;
    *numel_out = Space::numel;
}

TEST(AxisOuterProductCuda, SpacePropertiesOnDevice) {
    size_t *d_rank  = nullptr;
    size_t *d_numel = nullptr;
    ASSERT_EQ(cudaMalloc(&d_rank, sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_numel, sizeof(size_t)), cudaSuccess);

    test_space_properties_kernel<TestSpace><<<1, 1>>>(d_rank, d_numel);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t h_rank = 0, h_numel = 0;
    ASSERT_EQ(cudaMemcpy(&h_rank, d_rank, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_numel, d_numel, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_rank, TestSpace::rank);
    EXPECT_EQ(h_numel, TestSpace::numel);

    ASSERT_EQ(cudaFree(d_rank), cudaSuccess);
    ASSERT_EQ(cudaFree(d_numel), cudaSuccess);
}

// =============================================================================
// Test coord_from_indices_type on device (compile-time type computation)
// =============================================================================

template <typename Space, typename IndexSeq>
__global__ void
test_coord_from_indices_kernel(size_t *device_val_out, size_t *stype_val_out) {
    using CoordType = typename Space::template coord_from_indices_type<IndexSeq>;
    // Extract the compile-time values from the coord type
    constexpr auto values = CoordType::value_tuple;
    *device_val_out       = static_cast<size_t>(std::get<0>(values));
    *stype_val_out        = static_cast<size_t>(std::get<1>(values));
}

TEST(AxisOuterProductCuda, CoordFromIndicesOnDevice) {
    size_t *d_device_val = nullptr;
    size_t *d_stype_val  = nullptr;
    ASSERT_EQ(cudaMalloc(&d_device_val, sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_stype_val, sizeof(size_t)), cudaSuccess);

    // Test index (1, 3) -> (DEVICE, kInt32)
    using Indices = std::index_sequence<1, 3>;
    test_coord_from_indices_kernel<TestSpace, Indices><<<1, 1>>>(d_device_val, d_stype_val);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t h_device_val = 0, h_stype_val = 0;
    ASSERT_EQ(cudaMemcpy(&h_device_val, d_device_val, sizeof(size_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_stype_val, d_stype_val, sizeof(size_t), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(h_device_val, static_cast<size_t>(Device::kDEVICE));
    EXPECT_EQ(h_stype_val, static_cast<size_t>(ScalarType::kInt32));

    ASSERT_EQ(cudaFree(d_device_val), cudaSuccess);
    ASSERT_EQ(cudaFree(d_stype_val), cudaSuccess);
}

} // namespace dispatch
} // namespace fvdb
