// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#if 0

#include <fvdb/detail/dispatch/IndexSpace.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace fvdb {
namespace dispatch {

// A kernel that is instantiated per coordinate.
// The coordinate is encoded as compile-time template parameters (I, J).
// Each instantiation writes: [linear_index, I, J] to the output at linear_index * 3.
template <size_t I, size_t J>
__global__ void
per_coord_kernel(size_t *output) {
    // Compute linear index from compile-time coordinates using row-major order for 2xN space
    // For IndexSpace<2, 8>: linear_index = I * 8 + J
    using Space              = IndexSpace<2, 8>;
    constexpr auto strides   = Space::get_strides();
    constexpr size_t lin_idx = I * strides[0] + J * strides[1];

    // Write identifying data: the linear index and the coordinate components
    output[lin_idx * 3 + 0] = lin_idx;
    output[lin_idx * 3 + 1] = I;
    output[lin_idx * 3 + 2] = J;
}

// Visitor that launches a kernel for each coordinate in the space.
// The coordinate comes in as an std::index_sequence<I, J>.
struct KernelLauncher {
    size_t *d_output;

    template <size_t I, size_t J>
    void
    operator()(std::index_sequence<I, J>) const {
        per_coord_kernel<I, J><<<1, 1>>>(d_output);
    }
};

TEST(IndexSpaceCuda, PerCoordKernelInstantiation) {
    using Space = IndexSpace<2, 8>;

    static_assert(Space::rank == 2);
    static_assert(Space::numel == 16);

    // Allocate device memory: 3 values per coordinate (linear_idx, I, J)
    size_t *d_output         = nullptr;
    const size_t output_size = Space::numel * 3 * sizeof(size_t);
    ASSERT_EQ(cudaMalloc(&d_output, output_size), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_output, 0, output_size), cudaSuccess);

    // Visit all coordinates in the space, launching one kernel per coordinate.
    // This instantiates 16 different kernels (one for each (I, J) pair).
    Space::visit(KernelLauncher{d_output});

    // Synchronize and check for errors
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back to host
    std::vector<size_t> h_output(Space::numel * 3);
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify each coordinate wrote the correct values
    for (size_t lin_idx = 0; lin_idx < Space::numel; ++lin_idx) {
        constexpr auto shape    = Space::get_shape();
        const size_t expected_I = lin_idx / shape[1]; // lin_idx / 8
        const size_t expected_J = lin_idx % shape[1]; // lin_idx % 8

        EXPECT_EQ(h_output[lin_idx * 3 + 0], lin_idx)
            << "Linear index mismatch at coordinate (" << expected_I << ", " << expected_J << ")";
        EXPECT_EQ(h_output[lin_idx * 3 + 1], expected_I)
            << "I coordinate mismatch at linear index " << lin_idx;
        EXPECT_EQ(h_output[lin_idx * 3 + 2], expected_J)
            << "J coordinate mismatch at linear index " << lin_idx;
    }

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// Test that get_shape() and get_strides() work correctly in device code
__global__ void
test_shape_strides_kernel(size_t *shape_out, size_t *strides_out) {
    using Space            = IndexSpace<2, 8>;
    constexpr auto shape   = Space::get_shape();
    constexpr auto strides = Space::get_strides();

    shape_out[0]   = shape[0];
    shape_out[1]   = shape[1];
    strides_out[0] = strides[0];
    strides_out[1] = strides[1];
}

TEST(IndexSpaceCuda, GetShapeStridesOnDevice) {
    using Space = IndexSpace<2, 8>;

    size_t *d_shape   = nullptr;
    size_t *d_strides = nullptr;
    ASSERT_EQ(cudaMalloc(&d_shape, 2 * sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_strides, 2 * sizeof(size_t)), cudaSuccess);

    test_shape_strides_kernel<<<1, 1>>>(d_shape, d_strides);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::array<size_t, 2> h_shape{}, h_strides{};
    ASSERT_EQ(cudaMemcpy(h_shape.data(), d_shape, 2 * sizeof(size_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_strides.data(), d_strides, 2 * sizeof(size_t), cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Verify shape: {2, 8}
    EXPECT_EQ(h_shape[0], size_t{2});
    EXPECT_EQ(h_shape[1], size_t{8});

    // Verify strides: {8, 1} (row-major)
    EXPECT_EQ(h_strides[0], size_t{8});
    EXPECT_EQ(h_strides[1], size_t{1});

    ASSERT_EQ(cudaFree(d_shape), cudaSuccess);
    ASSERT_EQ(cudaFree(d_strides), cudaSuccess);
}

// Test linear_index and coord functions work on device
__global__ void
test_coord_conversion_kernel(size_t *results, size_t numel) {
    using Space = IndexSpace<2, 8>;

    for (size_t i = 0; i < numel; ++i) {
        // Round-trip: linear_index -> coord -> linear_index
        auto c         = Space::coord(i);
        size_t back    = Space::linear_index(c);
        results[i * 2] = (back == i) ? 1 : 0; // success flag

        // Also verify coord components are in-bounds
        constexpr auto shape = Space::get_shape();
        bool in_bounds       = (c[0] < shape[0]) && (c[1] < shape[1]);
        results[i * 2 + 1]   = in_bounds ? 1 : 0;
    }
}

TEST(IndexSpaceCuda, CoordConversionOnDevice) {
    using Space = IndexSpace<2, 8>;

    size_t *d_results         = nullptr;
    const size_t results_size = Space::numel * 2 * sizeof(size_t);
    ASSERT_EQ(cudaMalloc(&d_results, results_size), cudaSuccess);

    test_coord_conversion_kernel<<<1, 1>>>(d_results, Space::numel);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<size_t> h_results(Space::numel * 2);
    ASSERT_EQ(cudaMemcpy(h_results.data(), d_results, results_size, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (size_t i = 0; i < Space::numel; ++i) {
        EXPECT_EQ(h_results[i * 2], size_t{1}) << "Round-trip failed at linear index " << i;
        EXPECT_EQ(h_results[i * 2 + 1], size_t{1}) << "Coord out of bounds at linear index " << i;
    }

    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
}

} // namespace dispatch
} // namespace fvdb

#endif
