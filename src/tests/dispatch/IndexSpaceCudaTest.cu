// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/IndexSpace.h>
#include <fvdb/detail/dispatch/Values.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Test 1: Per-coordinate kernel instantiation via visit_index_space
// =============================================================================

// A kernel that is instantiated per coordinate.
// The coordinate is encoded as compile-time template parameters via Indices<I, J>.
// Each instantiation writes: [linear_index, I, J] to the output at linear_index * 3.
template <size_t I, size_t J>
__global__ void
per_coord_kernel(size_t *output) {
    // Compute linear index from compile-time coordinates using row-major order
    using Space              = Sizes<2, 8>;
    constexpr size_t lin_idx = LinearIndexFromIndices_v<Space, Indices<I, J>>();

    // Write identifying data: the linear index and the coordinate components
    output[lin_idx * 3 + 0] = lin_idx;
    output[lin_idx * 3 + 1] = I;
    output[lin_idx * 3 + 2] = J;
}

// Visitor that launches a kernel for each coordinate in the space.
// The coordinate comes in as Indices<I, J>.
struct KernelLauncher {
    size_t *d_output;

    template <size_t I, size_t J>
    void
    operator()(Indices<I, J>) const {
        per_coord_kernel<I, J><<<1, 1>>>(d_output);
    }
};

TEST(IndexSpaceCuda, PerCoordKernelInstantiation) {
    using Space = Sizes<2, 8>;

    static_assert(Rank_v<Space>() == 2);
    static_assert(Numel_v<Space>() == 16);

    // Allocate device memory: 3 values per coordinate (linear_idx, I, J)
    size_t *d_output         = nullptr;
    const size_t output_size = Numel_v<Space>() * 3 * sizeof(size_t);
    ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0, output_size), cudaSuccess);

    // Visit all coordinates in the space, launching one kernel per coordinate.
    // This instantiates 16 different kernels (one for each (I, J) pair).
    visit_index_space(KernelLauncher{d_output}, Space{});

    // Synchronize and check for errors
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back to host
    std::vector<size_t> h_output(Numel_v<Space>() * 3);
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify each coordinate wrote the correct values
    for (size_t lin_idx = 0; lin_idx < Numel_v<Space>(); ++lin_idx) {
        // For Sizes<2, 8>: row-major layout
        constexpr size_t dim1   = 8;
        const size_t expected_I = lin_idx / dim1;
        const size_t expected_J = lin_idx % dim1;

        EXPECT_EQ(h_output[lin_idx * 3 + 0], lin_idx)
            << "Linear index mismatch at coordinate (" << expected_I << ", " << expected_J << ")";
        EXPECT_EQ(h_output[lin_idx * 3 + 1], expected_I)
            << "I coordinate mismatch at linear index " << lin_idx;
        EXPECT_EQ(h_output[lin_idx * 3 + 2], expected_J)
            << "J coordinate mismatch at linear index " << lin_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 2: Rank and Numel in device code
// =============================================================================

__global__ void
test_rank_numel_kernel(size_t *rank_out, size_t *numel_out) {
    using Space = Sizes<2, 8>;
    *rank_out   = Rank_v<Space>();
    *numel_out  = Numel_v<Space>();
}

TEST(IndexSpaceCuda, RankNumelOnDevice) {
    using Space = Sizes<2, 8>;

    size_t *d_rank  = nullptr;
    size_t *d_numel = nullptr;
    ASSERT_EQ(cudaMalloc(&d_rank, sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_numel, sizeof(size_t)), cudaSuccess);

    test_rank_numel_kernel<<<1, 1>>>(d_rank, d_numel);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t h_rank{}, h_numel{};
    ASSERT_EQ(cudaMemcpy(&h_rank, d_rank, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_numel, d_numel, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_rank, Rank_v<Space>());
    EXPECT_EQ(h_numel, Numel_v<Space>());

    ASSERT_EQ(cudaFree(d_rank), cudaSuccess);
    ASSERT_EQ(cudaFree(d_numel), cudaSuccess);
}

// =============================================================================
// Test 3: Compile-time coordinate conversion in device code
// =============================================================================

// IndicesFromLinearIndex and LinearIndexFromIndices are compile-time only,
// so we test them via template instantiation in device code.
template <typename Space, size_t LinearIndex>
__global__ void
test_coord_round_trip_kernel(size_t *result) {
    using Coord           = IndicesFromLinearIndex_t<Space, LinearIndex>;
    constexpr size_t back = LinearIndexFromIndices_v<Space, Coord>();
    *result               = back;
}

TEST(IndexSpaceCuda, CoordConversionOnDevice) {
    using Space = Sizes<2, 8>;

    size_t *d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(size_t)), cudaSuccess);

    size_t h_result{};

    // Test index 0
    test_coord_round_trip_kernel<Space, 0><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{0});

    // Test index 7 (last in first row)
    test_coord_round_trip_kernel<Space, 7><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{7});

    // Test index 8 (first in second row)
    test_coord_round_trip_kernel<Space, 8><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{8});

    // Test index 15 (last element)
    test_coord_round_trip_kernel<Space, 15><<<1, 1>>>(d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_result, size_t{15});

    ASSERT_EQ(cudaFree(d_result), cudaSuccess);
}

// =============================================================================
// Test 4: visit_index_spaces with multiple spaces
// =============================================================================

template <size_t I>
__global__ void
per_1d_coord_kernel(size_t *output, size_t base_offset) {
    output[base_offset + I] = I + 100; // Add 100 to distinguish values
}

struct Multi1DKernelLauncher {
    size_t *d_output;
    size_t base_offset;

    template <size_t I>
    void
    operator()(Indices<I>) const {
        per_1d_coord_kernel<I><<<1, 1>>>(d_output, base_offset);
    }
};

TEST(IndexSpaceCuda, VisitIndexSpaces) {
    using Space1 = Sizes<3>;
    using Space2 = Sizes<4>;

    constexpr size_t total_size = Numel_v<Space1>() + Numel_v<Space2>();

    size_t *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, total_size * sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0, total_size * sizeof(size_t)), cudaSuccess);

    // Use visit_index_spaces to visit both spaces with different base offsets
    // Note: visit_index_spaces calls same visitor on all spaces, so we test sequentially
    visit_index_space(Multi1DKernelLauncher{d_output, 0}, Space1{});
    visit_index_space(Multi1DKernelLauncher{d_output, Numel_v<Space1>()}, Space2{});

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<size_t> h_output(total_size);
    ASSERT_EQ(
        cudaMemcpy(h_output.data(), d_output, total_size * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    // Check Space1 values (indices 0, 1, 2 -> values 100, 101, 102)
    for (size_t i = 0; i < Numel_v<Space1>(); ++i) {
        EXPECT_EQ(h_output[i], i + 100) << "Space1 mismatch at index " << i;
    }

    // Check Space2 values (indices 0, 1, 2, 3 -> values 100, 101, 102, 103)
    for (size_t i = 0; i < Numel_v<Space2>(); ++i) {
        EXPECT_EQ(h_output[Numel_v<Space1>() + i], i + 100) << "Space2 mismatch at index " << i;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// =============================================================================
// Test 5: IndexSpace concept test helpers in device code
// =============================================================================

__global__ void
test_index_space_concepts_kernel(bool *results) {
    using TensorSpace   = Sizes<2, 8>;
    using ScalarSpace   = Sizes<>;
    using EmptySpace    = Sizes<0>;
    using NonEmptySpace = Sizes<2, 8>;

    // These are consteval so they work at compile time in device code
    results[0] = is_index_space<TensorSpace>();
    results[1] = is_tensor_index_space<TensorSpace>();
    results[2] = is_scalar_index_space<ScalarSpace>();
    results[3] = is_non_empty_index_space<NonEmptySpace>();
    results[4] = !is_non_empty_index_space<EmptySpace>();
    results[5] = is_same_rank<Sizes<2, 3>, Sizes<4, 5>>();
}

TEST(IndexSpaceCuda, ConceptTestHelpersOnDevice) {
    constexpr size_t num_tests = 6;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_index_space_concepts_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "is_index_space<TensorSpace>";
    EXPECT_TRUE(h_results[1]) << "is_tensor_index_space<TensorSpace>";
    EXPECT_TRUE(h_results[2]) << "is_scalar_index_space<ScalarSpace>";
    EXPECT_TRUE(h_results[3]) << "is_non_empty_index_space<NonEmptySpace>";
    EXPECT_TRUE(h_results[4]) << "!is_non_empty_index_space<EmptySpace>";
    EXPECT_TRUE(h_results[5]) << "is_same_rank<Sizes<2,3>, Sizes<4,5>>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 6: Values.h concept test helpers in device code
// =============================================================================

__global__ void
test_values_concepts_kernel(bool *results) {
    using Pack       = Values<1, 2, 3>;
    using EmptyPack  = Values<>;
    using MixedPack  = Values<1, 2.0f, 'c'>;
    using UniquePack = Values<1, 2, 3>;
    using DupPack    = Values<1, 2, 1>;

    results[0] = is_value_pack<Pack>();
    results[1] = is_non_empty_value_pack<Pack>();
    results[2] = is_empty_value_pack<EmptyPack>();
    results[3] = is_same_type_value_pack<Pack>();
    results[4] = !is_same_type_value_pack<MixedPack>();
    results[5] = is_unique_value_pack<UniquePack>();
    results[6] = !is_unique_value_pack<DupPack>();
}

TEST(IndexSpaceCuda, ValuesConceptsOnDevice) {
    constexpr size_t num_tests = 7;

    bool *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(bool)), cudaSuccess);

    test_values_concepts_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<bool, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(bool), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_TRUE(h_results[0]) << "is_value_pack<Pack>";
    EXPECT_TRUE(h_results[1]) << "is_non_empty_value_pack<Pack>";
    EXPECT_TRUE(h_results[2]) << "is_empty_value_pack<EmptyPack>";
    EXPECT_TRUE(h_results[3]) << "is_same_type_value_pack<Pack>";
    EXPECT_TRUE(h_results[4]) << "!is_same_type_value_pack<MixedPack>";
    EXPECT_TRUE(h_results[5]) << "is_unique_value_pack<UniquePack>";
    EXPECT_TRUE(h_results[6]) << "!is_unique_value_pack<DupPack>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

// =============================================================================
// Test 7: PackSize and PackElement in device code
// =============================================================================

__global__ void
test_values_traits_kernel(size_t *results) {
    using Pack = Values<10, 20, 30>;

    results[0] = PackSize_v<Pack>();
    results[1] = PackElement_v<Pack, 0>();
    results[2] = PackElement_v<Pack, 1>();
    results[3] = PackElement_v<Pack, 2>();
    results[4] = PackHeadValue_v<Pack>();
}

TEST(IndexSpaceCuda, ValuesTraitsOnDevice) {
    constexpr size_t num_tests = 5;

    size_t *d_results = nullptr;
    ASSERT_EQ(cudaMalloc(&d_results, num_tests * sizeof(size_t)), cudaSuccess);

    test_values_traits_kernel<<<1, 1>>>(d_results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, num_tests> h_results{};
    ASSERT_EQ(
        cudaMemcpy(h_results.data(), d_results, num_tests * sizeof(size_t), cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(h_results[0], size_t{3}) << "PackSize_v<Pack>";
    EXPECT_EQ(h_results[1], size_t{10}) << "PackElement_v<Pack, 0>";
    EXPECT_EQ(h_results[2], size_t{20}) << "PackElement_v<Pack, 1>";
    EXPECT_EQ(h_results[3], size_t{30}) << "PackElement_v<Pack, 2>";
    EXPECT_EQ(h_results[4], size_t{10}) << "PackHeadValue_v<Pack>";

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

} // namespace dispatch
} // namespace fvdb
