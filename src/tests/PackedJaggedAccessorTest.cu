// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <gtest/gtest.h>

using namespace fvdb;

// Test fixture for PackedJaggedAccessor tests
class PackedJaggedAccessorTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        // Create test data on GPU by default
        data_1d = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, torch::kCUDA);
        data_2d =
            torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}, torch::kCUDA);

        // Create sample tensors for testing
        tensor_a = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kCUDA);
        tensor_b = torch::tensor({4.0f, 5.0f}, torch::kCUDA);
        tensor_c = torch::tensor({6.0f, 7.0f, 8.0f, 9.0f}, torch::kCUDA);

        // Create JaggedTensor with specific structure
        std::vector<int64_t> lsizes = {3, 2, 3};
        jt_1d                       = JaggedTensor(lsizes, data_1d);

        // 2D jagged tensor
        std::vector<int64_t> lsizes_2d = {2, 2};
        jt_2d                          = JaggedTensor(lsizes_2d, data_2d);
    }

    torch::Tensor data_1d;
    torch::Tensor data_2d;
    torch::Tensor tensor_a, tensor_b, tensor_c;
    JaggedTensor jt_1d;
    JaggedTensor jt_2d;
};

// CUDA kernel to test PackedJaggedAccessor basic functionality (templated)
template <typename AccessorType>
__global__ void
test_packed_accessor_basic_kernel(AccessorType accessor, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < accessor.elementCount()) {
        // Test batch indexing
        int batch_idx = accessor.batchIdx(idx);
        results[idx]  = batch_idx;

        // Test offset access for the batch this element belongs to
        int start                                  = accessor.offsetStart(batch_idx);
        int end                                    = accessor.offsetEnd(batch_idx);
        results[idx + accessor.elementCount()]     = start;
        results[idx + 2 * accessor.elementCount()] = end;
    }
}

// Test PackedJaggedAccessor32 basic functionality
TEST_F(PackedJaggedAccessorTest, PackedAccessor32Basic) {
    auto accessor = jt_1d.packed_accessor32<float, 1>();

    // Test basic properties
    EXPECT_EQ(accessor.elementCount(), 8);

    // Test batch indexing and offset access through CUDA kernel
    auto results = torch::zeros({24}, torch::kInt32)
                       .cuda(); // 8 batch indices + 8 start offsets + 8 end offsets

    int block_size = 256;
    int grid_size  = (accessor.elementCount() + block_size - 1) / block_size;
    test_packed_accessor_basic_kernel<decltype(accessor)>
        <<<grid_size, block_size>>>(accessor, results.data_ptr<int>());

    cudaDeviceSynchronize();

    auto host_results = results.cpu();

    // Check batch indices
    EXPECT_EQ(host_results[0].item<int>(), 0); // First tensor
    EXPECT_EQ(host_results[1].item<int>(), 0); // First tensor
    EXPECT_EQ(host_results[2].item<int>(), 0); // First tensor
    EXPECT_EQ(host_results[3].item<int>(), 1); // Second tensor
    EXPECT_EQ(host_results[4].item<int>(), 1); // Second tensor
    EXPECT_EQ(host_results[5].item<int>(), 2); // Third tensor
    EXPECT_EQ(host_results[6].item<int>(), 2); // Third tensor
    EXPECT_EQ(host_results[7].item<int>(), 2); // Third tensor

    // Check offset starts
    EXPECT_EQ(host_results[8].item<int>(), 0);  // Tensor 0 start
    EXPECT_EQ(host_results[9].item<int>(), 0);  // Tensor 0 start
    EXPECT_EQ(host_results[10].item<int>(), 0); // Tensor 0 start
    EXPECT_EQ(host_results[11].item<int>(), 3); // Tensor 1 start
    EXPECT_EQ(host_results[12].item<int>(), 3); // Tensor 1 start
    EXPECT_EQ(host_results[13].item<int>(), 5); // Tensor 2 start
    EXPECT_EQ(host_results[14].item<int>(), 5); // Tensor 2 start
    EXPECT_EQ(host_results[15].item<int>(), 5); // Tensor 2 start

    // Check offset ends
    EXPECT_EQ(host_results[16].item<int>(), 3); // Tensor 0 end
    EXPECT_EQ(host_results[17].item<int>(), 3); // Tensor 0 end
    EXPECT_EQ(host_results[18].item<int>(), 3); // Tensor 0 end
    EXPECT_EQ(host_results[19].item<int>(), 5); // Tensor 1 end
    EXPECT_EQ(host_results[20].item<int>(), 5); // Tensor 1 end
    EXPECT_EQ(host_results[21].item<int>(), 8); // Tensor 2 end
    EXPECT_EQ(host_results[22].item<int>(), 8); // Tensor 2 end
    EXPECT_EQ(host_results[23].item<int>(), 8); // Tensor 2 end
}

// Test PackedJaggedAccessor64 basic functionality
TEST_F(PackedJaggedAccessorTest, PackedAccessor64Basic) {
    auto accessor = jt_1d.packed_accessor64<float, 1>();

    // Test basic properties
    EXPECT_EQ(accessor.elementCount(), 8);

    // Test batch indexing and offset access through CUDA kernel
    auto results = torch::zeros({24}, torch::kInt32)
                       .cuda(); // 8 batch indices + 8 start offsets + 8 end offsets

    int block_size = 256;
    int grid_size  = (accessor.elementCount() + block_size - 1) / block_size;
    test_packed_accessor_basic_kernel<decltype(accessor)>
        <<<grid_size, block_size>>>(accessor, results.data_ptr<int>());

    cudaDeviceSynchronize();

    auto host_results = results.cpu();

    // Check batch indices
    EXPECT_EQ(host_results[0].item<int>(), 0);
    EXPECT_EQ(host_results[3].item<int>(), 1);
    EXPECT_EQ(host_results[5].item<int>(), 2);

    // Check offset starts
    EXPECT_EQ(host_results[8].item<int>(), 0);
    EXPECT_EQ(host_results[11].item<int>(), 3);
    EXPECT_EQ(host_results[13].item<int>(), 5);

    // Check offset ends
    EXPECT_EQ(host_results[16].item<int>(), 3);
    EXPECT_EQ(host_results[19].item<int>(), 5);
    EXPECT_EQ(host_results[21].item<int>(), 8);
}

// Test data access through packed accessors (host-side validation)
TEST_F(PackedJaggedAccessorTest, DataAccess) {
    auto packed_accessor = jt_1d.packed_accessor32<float, 1>();

    // Test element count
    EXPECT_EQ(packed_accessor.elementCount(), 8);

    // Test data access - use host-side tensor for validation
    auto host_data = jt_1d.jdata().cpu();
    EXPECT_EQ(host_data.size(0), 8);

    // Test reading values from host tensor
    EXPECT_EQ(host_data[0].item<float>(), 1.0f);
    EXPECT_EQ(host_data[1].item<float>(), 2.0f);
    EXPECT_EQ(host_data[2].item<float>(), 3.0f);
    EXPECT_EQ(host_data[3].item<float>(), 4.0f);
    EXPECT_EQ(host_data[4].item<float>(), 5.0f);
    EXPECT_EQ(host_data[5].item<float>(), 6.0f);
    EXPECT_EQ(host_data[6].item<float>(), 7.0f);
    EXPECT_EQ(host_data[7].item<float>(), 8.0f);
}

// Test 2D jagged tensor accessor
TEST_F(PackedJaggedAccessorTest, TwoDimensionalAccessor) {
    auto packed_accessor = jt_2d.packed_accessor32<float, 2>();

    // Test element count
    EXPECT_EQ(packed_accessor.elementCount(), 4);

    // Test data access for 2D tensor - use host-side tensor for validation
    auto host_data = jt_2d.jdata().cpu();
    EXPECT_EQ(host_data.size(0), 4); // 4 rows
    EXPECT_EQ(host_data.size(1), 2); // 2 columns

    // Test reading 2D values from host tensor
    EXPECT_EQ(host_data[0][0].item<float>(), 1.0f);
    EXPECT_EQ(host_data[0][1].item<float>(), 2.0f);
    EXPECT_EQ(host_data[1][0].item<float>(), 3.0f);
    EXPECT_EQ(host_data[1][1].item<float>(), 4.0f);
    EXPECT_EQ(host_data[2][0].item<float>(), 5.0f);
    EXPECT_EQ(host_data[2][1].item<float>(), 6.0f);
    EXPECT_EQ(host_data[3][0].item<float>(), 7.0f);
    EXPECT_EQ(host_data[3][1].item<float>(), 8.0f);
}

// Test edge cases with empty tensors
TEST_F(PackedJaggedAccessorTest, EmptyTensorAccessor) {
    auto empty_tensor = torch::empty({0}, torch::kCUDA);
    JaggedTensor jt_empty(empty_tensor);

    auto packed_accessor = jt_empty.packed_accessor32<float, 1>();

    // Test element count
    EXPECT_EQ(packed_accessor.elementCount(), 0);
}

// Test single element accessor
TEST_F(PackedJaggedAccessorTest, SingleElementAccessor) {
    auto single_data = torch::tensor({42.0f}, torch::kCUDA);
    JaggedTensor jt_single(single_data);

    auto packed_accessor = jt_single.packed_accessor32<float, 1>();

    // Test element count
    EXPECT_EQ(packed_accessor.elementCount(), 1);

    // Test data access
    auto host_data = jt_single.jdata().cpu();
    EXPECT_EQ(host_data[0].item<float>(), 42.0f);
}

// Test accessor consistency across different accessor types
TEST_F(PackedJaggedAccessorTest, AccessorConsistency) {
    auto packed_accessor32 = jt_1d.packed_accessor32<float, 1>();
    auto packed_accessor64 = jt_1d.packed_accessor64<float, 1>();

    // Both accessors should have the same element count
    EXPECT_EQ(packed_accessor32.elementCount(), packed_accessor64.elementCount());
}

// Test nested jagged tensor accessor
TEST_F(PackedJaggedAccessorTest, NestedJaggedTensorAccessor) {
    auto tensor_small  = torch::tensor({1.0f}, torch::kCUDA);
    auto tensor_medium = torch::tensor({2.0f, 3.0f}, torch::kCUDA);

    std::vector<std::vector<torch::Tensor>> nested_tensors = {{tensor_a, tensor_b}, {tensor_small}};

    JaggedTensor jt_nested(nested_tensors);

    auto packed_accessor = jt_nested.packed_accessor32<float, 1>();

    // Test element count
    EXPECT_EQ(packed_accessor.elementCount(), 6); // 3 + 2 + 1 = 6
}

// Test different data types
TEST_F(PackedJaggedAccessorTest, DifferentDataTypes) {
    // Test with double precision - ensure tensor stays double
    auto double_data =
        torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::dtype(torch::kDouble).device(torch::kCUDA));
    std::vector<int64_t> lsizes = {2, 2};
    JaggedTensor double_jt(lsizes, double_data);

    auto double_accessor = double_jt.packed_accessor32<double, 1>();
    EXPECT_EQ(double_accessor.elementCount(), 4);

    // Test with integer data - ensure tensor stays int
    auto int_data =
        torch::tensor({1, 2, 3, 4, 5}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    std::vector<int64_t> int_lsizes = {3, 2};
    JaggedTensor int_jt(int_lsizes, int_data);

    auto int_accessor = int_jt.packed_accessor32<int, 1>();
    EXPECT_EQ(int_accessor.elementCount(), 5);
}

// CUDA kernel for testing PackedJaggedAccessor in GPU context
__global__ void
simple_jagged_kernel(PackedJaggedAccessor32<float, 1> accessor, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < accessor.elementCount()) {
        accessor.data()[idx] *= multiplier;
    }
}

// Test CUDA kernel usage
TEST_F(PackedJaggedAccessorTest, CUDAKernelUsage) {
    auto packed_accessor = jt_1d.packed_accessor32<float, 1>();

    // Launch kernel to multiply all elements by 2.0
    int block_size = 256;
    int grid_size  = (packed_accessor.elementCount() + block_size - 1) / block_size;
    simple_jagged_kernel<<<grid_size, block_size>>>(packed_accessor, 2.0f);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Check that values were modified correctly using host-side validation
    auto host_data = jt_1d.jdata().cpu();
    EXPECT_EQ(host_data[0].item<float>(), 2.0f);  // 1.0 * 2.0
    EXPECT_EQ(host_data[1].item<float>(), 4.0f);  // 2.0 * 2.0
    EXPECT_EQ(host_data[2].item<float>(), 6.0f);  // 3.0 * 2.0
    EXPECT_EQ(host_data[3].item<float>(), 8.0f);  // 4.0 * 2.0
    EXPECT_EQ(host_data[4].item<float>(), 10.0f); // 5.0 * 2.0
    EXPECT_EQ(host_data[5].item<float>(), 12.0f); // 6.0 * 2.0
    EXPECT_EQ(host_data[6].item<float>(), 14.0f); // 7.0 * 2.0
    EXPECT_EQ(host_data[7].item<float>(), 16.0f); // 8.0 * 2.0
}

// Test CUDA kernel with 2D accessor
__global__ void
simple_2d_jagged_kernel(PackedJaggedAccessor32<float, 2> accessor, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < accessor.elementCount()) {
        for (int j = 0; j < accessor.data().size(1); ++j) {
            accessor.data()[idx][j] *= multiplier;
        }
    }
}

// Test 2D CUDA kernel usage
TEST_F(PackedJaggedAccessorTest, CUDAKernelUsage2D) {
    auto packed_accessor = jt_2d.packed_accessor32<float, 2>();

    // Launch kernel to multiply all elements by 3.0
    int block_size = 256;
    int grid_size  = (packed_accessor.elementCount() + block_size - 1) / block_size;
    simple_2d_jagged_kernel<<<grid_size, block_size>>>(packed_accessor, 3.0f);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Check that values were modified correctly using host-side validation
    auto host_data = jt_2d.jdata().cpu();
    EXPECT_EQ(host_data[0][0].item<float>(), 3.0f);  // 1.0 * 3.0
    EXPECT_EQ(host_data[0][1].item<float>(), 6.0f);  // 2.0 * 3.0
    EXPECT_EQ(host_data[1][0].item<float>(), 9.0f);  // 3.0 * 3.0
    EXPECT_EQ(host_data[1][1].item<float>(), 12.0f); // 4.0 * 3.0
    EXPECT_EQ(host_data[2][0].item<float>(), 15.0f); // 5.0 * 3.0
    EXPECT_EQ(host_data[2][1].item<float>(), 18.0f); // 6.0 * 3.0
    EXPECT_EQ(host_data[3][0].item<float>(), 21.0f); // 7.0 * 3.0
    EXPECT_EQ(host_data[3][1].item<float>(), 24.0f); // 8.0 * 3.0
}
