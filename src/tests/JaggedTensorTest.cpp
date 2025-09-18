// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <gtest/gtest.h>

using namespace fvdb;

// Test fixture for JaggedTensor tests
class JaggedTensorTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        // Create sample data for testing
        sample_data_1d = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        sample_data_2d =
            torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}});

        // Create sample tensors for list construction
        tensor_a = torch::tensor({1.0f, 2.0f, 3.0f});
        tensor_b = torch::tensor({4.0f, 5.0f});
        tensor_c = torch::tensor({6.0f, 7.0f, 8.0f, 9.0f});
    }

    torch::Tensor sample_data_1d;
    torch::Tensor sample_data_2d;
    torch::Tensor tensor_a, tensor_b, tensor_c;
};

// Test empty JaggedTensor constructor
TEST_F(JaggedTensorTest, EmptyConstructor) {
    JaggedTensor jt;

    EXPECT_EQ(jt.num_tensors(), 0);
    EXPECT_EQ(jt.num_outer_lists(), 0);
    EXPECT_EQ(jt.element_count(), 0);
    EXPECT_EQ(jt.ldim(), 1);
    EXPECT_EQ(jt.joffsets().size(0), 1);
    EXPECT_EQ(jt.jidx().size(0), 0);
    EXPECT_EQ(jt.jlidx().size(0), 0);
}

// Test single tensor constructor
TEST_F(JaggedTensorTest, SingleTensorConstructor) {
    JaggedTensor jt(tensor_a);

    EXPECT_EQ(jt.num_tensors(), 1);
    EXPECT_EQ(jt.num_outer_lists(), 1);
    EXPECT_EQ(jt.element_count(), 3);
    EXPECT_EQ(jt.ldim(), 1);
    EXPECT_TRUE(torch::equal(jt.jdata(), tensor_a));

    // Check offsets: should be [0, 3]
    auto offsets = jt.joffsets();
    EXPECT_EQ(offsets.size(0), 2);
    EXPECT_EQ(offsets[0].item<int64_t>(), 0);
    EXPECT_EQ(offsets[1].item<int64_t>(), 3);
}

// Test vector of tensors constructor
TEST_F(JaggedTensorTest, VectorOfTensorsConstructor) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    EXPECT_EQ(jt.num_tensors(), 3);
    EXPECT_EQ(jt.num_outer_lists(), 3);
    EXPECT_EQ(jt.element_count(), 9); // 3 + 2 + 4
    EXPECT_EQ(jt.ldim(), 1);

    // Check that data is concatenated correctly
    auto expected_data = torch::cat({tensor_a, tensor_b, tensor_c}, 0);
    EXPECT_TRUE(torch::equal(jt.jdata(), expected_data));

    // Check offsets: should be [0, 3, 5, 9]
    auto offsets = jt.joffsets();
    EXPECT_EQ(offsets.size(0), 4);
    EXPECT_EQ(offsets[0].item<int64_t>(), 0);
    EXPECT_EQ(offsets[1].item<int64_t>(), 3);
    EXPECT_EQ(offsets[2].item<int64_t>(), 5);
    EXPECT_EQ(offsets[3].item<int64_t>(), 9);
}

// Test lsizes constructor
TEST_F(JaggedTensorTest, LSizesConstructor) {
    std::vector<int64_t> lsizes = {3, 2, 1};
    auto data                   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    JaggedTensor jt(lsizes, data);

    EXPECT_EQ(jt.num_tensors(), 3);
    EXPECT_EQ(jt.element_count(), 6);
    EXPECT_TRUE(torch::equal(jt.jdata(), data));

    auto computed_lsizes = jt.lsizes1();
    EXPECT_EQ(computed_lsizes.size(), 3ul);
    EXPECT_EQ(computed_lsizes[0], 3);
    EXPECT_EQ(computed_lsizes[1], 2);
    EXPECT_EQ(computed_lsizes[2], 1);
}

// Test vector of vector of tensors constructor (nested lists)
TEST_F(JaggedTensorTest, NestedListsConstructor) {
    auto tensor_small  = torch::tensor({1.0f});
    auto tensor_medium = torch::tensor({2.0f, 3.0f});

    std::vector<std::vector<torch::Tensor>> nested_tensors = {
        {tensor_a, tensor_b}, {tensor_small}, {tensor_medium, tensor_c}};

    JaggedTensor jt(nested_tensors);

    EXPECT_EQ(jt.ldim(), 2);
    EXPECT_EQ(jt.num_outer_lists(), 3);
    EXPECT_EQ(jt.num_tensors(), 5);    // 2 + 1 + 2
    EXPECT_EQ(jt.element_count(), 12); // 3 + 2 + 1 + 2 + 4
}

// Test unbind1 functionality
TEST_F(JaggedTensorTest, Unbind1) {
    std::vector<torch::Tensor> original_tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(original_tensors);

    auto unbound_tensors = jt.unbind1();

    EXPECT_EQ(unbound_tensors.size(), 3ul);
    EXPECT_TRUE(torch::equal(unbound_tensors[0], tensor_a));
    EXPECT_TRUE(torch::equal(unbound_tensors[1], tensor_b));
    EXPECT_TRUE(torch::equal(unbound_tensors[2], tensor_c));
}

// Test jreshape functionality
TEST_F(JaggedTensorTest, JReshape) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Reshape to different lsizes
    std::vector<int64_t> new_lsizes = {4, 5};
    auto reshaped                   = jt.jreshape(new_lsizes);

    EXPECT_EQ(reshaped.num_tensors(), 2);
    EXPECT_EQ(reshaped.element_count(), 9);

    auto computed_lsizes = reshaped.lsizes1();
    EXPECT_EQ(computed_lsizes[0], 4);
    EXPECT_EQ(computed_lsizes[1], 5);
}

// Test jflatten functionality
TEST_F(JaggedTensorTest, JFlatten) {
    auto tensor_small                                      = torch::tensor({1.0f});
    std::vector<std::vector<torch::Tensor>> nested_tensors = {{tensor_a, tensor_b}, {tensor_small}};

    JaggedTensor jt(nested_tensors);
    EXPECT_EQ(jt.ldim(), 2);

    // Flatten the outer dimension
    auto flattened = jt.jflatten(0);
    EXPECT_EQ(flattened.ldim(), 1);
    EXPECT_EQ(flattened.num_tensors(), 3); // Should have 3 individual tensors
}

// Test indexing functionality
TEST_F(JaggedTensorTest, Indexing) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Index with single integer
    auto indexed = jt.index(1);
    EXPECT_EQ(indexed.num_tensors(), 1);
    EXPECT_TRUE(torch::equal(indexed.jdata(), tensor_b));

    // Index with slice
    auto sliced = jt.index(0, 2, 1);
    EXPECT_EQ(sliced.num_tensors(), 2);
    EXPECT_EQ(sliced.element_count(), 5); // 3 + 2
}

// Test indexing functionality with JaggedTensor
TEST_F(JaggedTensorTest, IndexingWithJaggedTensor) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Create indices for each tensor in the JaggedTensor
    // tensor_a has 3 elements, tensor_b has 2 elements, tensor_c has 4 elements
    auto indices_a = torch::tensor({0, 2}, torch::kInt64); // Select elements 0 and 2 from tensor_a
    auto indices_b = torch::tensor({1}, torch::kInt64);    // Select element 1 from tensor_b
    auto indices_c =
        torch::tensor({0, 1, 3}, torch::kInt64);           // Select elements 0, 1, 3 from tensor_c

    std::vector<torch::Tensor> index_tensors = {indices_a, indices_b, indices_c};
    JaggedTensor indices_jt(index_tensors);

    auto indexed_jt = jt.index(indices_jt);
    EXPECT_EQ(indexed_jt.num_tensors(), 3);
    EXPECT_EQ(indexed_jt.element_count(), 6); // 2 + 1 + 3 = 6 elements total
}

// Test arithmetic operations
TEST_F(JaggedTensorTest, ArithmeticOperations) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt1(tensors);
    JaggedTensor jt2(tensors);

    // Test JaggedTensor + JaggedTensor
    auto sum_result   = jt1 + jt2;
    auto expected_sum = torch::cat({tensor_a + tensor_a, tensor_b + tensor_b}, 0);
    EXPECT_TRUE(torch::equal(sum_result.jdata(), expected_sum));

    // Test JaggedTensor + scalar
    auto add_scalar_result   = jt1 + 1.0f;
    auto expected_add_scalar = torch::cat({tensor_a + 1.0f, tensor_b + 1.0f}, 0);
    EXPECT_TRUE(torch::equal(add_scalar_result.jdata(), expected_add_scalar));

    // Test JaggedTensor - scalar
    auto sub_result   = jt1 - 1.0f;
    auto expected_sub = torch::cat({tensor_a - 1.0f, tensor_b - 1.0f}, 0);
    EXPECT_TRUE(torch::equal(sub_result.jdata(), expected_sub));

    // Test JaggedTensor * scalar
    auto mult_result   = jt1 * 2.0f;
    auto expected_mult = torch::cat({tensor_a * 2.0f, tensor_b * 2.0f}, 0);
    EXPECT_TRUE(torch::equal(mult_result.jdata(), expected_mult));

    // Test JaggedTensor / scalar
    auto div_result   = jt1 / 2.0f;
    auto expected_div = torch::cat({tensor_a / 2.0f, tensor_b / 2.0f}, 0);
    EXPECT_TRUE(torch::equal(div_result.jdata(), expected_div));

    // Test unary negation
    auto neg_result   = -jt1;
    auto expected_neg = torch::cat({-tensor_a, -tensor_b}, 0);
    EXPECT_TRUE(torch::equal(neg_result.jdata(), expected_neg));

    // Test JaggedTensor + tensor
    auto tensor_other       = torch::tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    auto tensor_op_result   = jt1 + tensor_other;
    auto expected_tensor_op = torch::cat(
        {tensor_a + tensor_other.slice(0, 0, 3), tensor_b + tensor_other.slice(0, 3, 5)}, 0);
    EXPECT_TRUE(torch::equal(tensor_op_result.jdata(), expected_tensor_op));
}

// Test reduction operations
TEST_F(JaggedTensorTest, ReductionOperations) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Test jsum
    {
        auto sum_result = jt.jsum();
        EXPECT_EQ(sum_result.num_tensors(), 3);

        auto sum_tensors = sum_result.unbind1();
        EXPECT_TRUE(torch::allclose(sum_tensors[0], tensor_a.sum()));
        EXPECT_TRUE(torch::allclose(sum_tensors[1], tensor_b.sum()));
        EXPECT_TRUE(torch::allclose(sum_tensors[2], tensor_c.sum()));
    }

    // Test jmin - returns [min_values, argmin_indices]
    {
        auto min_result = jt.jmin();
        EXPECT_EQ(min_result.size(), 2ul); // Two JaggedTensors: values and indices

        auto min_values  = min_result[0].unbind1();
        auto min_indices = min_result[1].unbind1();
        EXPECT_EQ(min_values.size(), 3ul);
        EXPECT_EQ(min_indices.size(), 3ul);

        // Check that min values are correct for each tensor
        EXPECT_TRUE(torch::allclose(min_values[0], tensor_a.min()));
        EXPECT_TRUE(torch::allclose(min_values[1], tensor_b.min()));
        EXPECT_TRUE(torch::allclose(min_values[2], tensor_c.min()));
    }

    // Test jmax - returns [max_values, argmax_indices]
    {
        auto max_result = jt.jmax();
        EXPECT_EQ(max_result.size(), 2ul); // Two JaggedTensors: values and indices

        auto max_values  = max_result[0].unbind1();
        auto max_indices = max_result[1].unbind1();
        EXPECT_EQ(max_values.size(), 3ul);
        EXPECT_EQ(max_indices.size(), 3ul);

        // Check that max values are correct for each tensor
        EXPECT_TRUE(torch::allclose(max_values[0], tensor_a.max()));
        EXPECT_TRUE(torch::allclose(max_values[1], tensor_b.max()));
        EXPECT_TRUE(torch::allclose(max_values[2], tensor_c.max()));
    }
}

// Test device and dtype operations
TEST_F(JaggedTensorTest, DeviceAndDtype) {
    JaggedTensor jt(tensor_a);

    EXPECT_TRUE(jt.is_cpu());
    EXPECT_FALSE(jt.is_cuda());
    EXPECT_EQ(jt.scalar_type(), torch::kFloat32);

    // Test dtype conversion
    auto double_jt = jt.to(torch::kFloat64);
    EXPECT_EQ(double_jt.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(torch::equal(double_jt.jdata(), tensor_a.to(torch::kFloat64)));
}

// Test jagged_like functionality
TEST_F(JaggedTensorTest, JaggedLike) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    auto new_data = torch::ones_like(jt.jdata()) * 10.0f;
    auto new_jt   = jt.jagged_like(new_data);

    EXPECT_EQ(new_jt.num_tensors(), jt.num_tensors());
    EXPECT_EQ(new_jt.element_count(), jt.element_count());
    EXPECT_TRUE(torch::equal(new_jt.jdata(), new_data));

    // Structure should be the same
    EXPECT_TRUE(torch::equal(new_jt.joffsets(), jt.joffsets()));
    EXPECT_TRUE(torch::equal(new_jt.jidx(), jt.jidx()));
}

// Test accessors
TEST_F(JaggedTensorTest, Accessors) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Test regular accessor
    auto accessor = jt.accessor<float, 1>();
    EXPECT_EQ(accessor.elementCount(), 9);
    EXPECT_EQ(accessor.batchIdx(0), 0);
    EXPECT_EQ(accessor.batchIdx(3), 1);
    EXPECT_EQ(accessor.batchIdx(5), 2);

    EXPECT_EQ(accessor.offsetStart(0), 0);
    EXPECT_EQ(accessor.offsetStart(1), 3);
    EXPECT_EQ(accessor.offsetStart(2), 5);

    EXPECT_EQ(accessor.offsetEnd(0), 3);
    EXPECT_EQ(accessor.offsetEnd(1), 5);
    EXPECT_EQ(accessor.offsetEnd(2), 9);
}

// Test packed accessors
TEST_F(JaggedTensorTest, PackedAccessors) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Test 32-bit packed accessor
    auto packed_accessor32 = jt.packed_accessor32<float, 1>();
    EXPECT_EQ(packed_accessor32.elementCount(), 9);
    EXPECT_EQ(packed_accessor32.batchIdx(0), 0);
    EXPECT_EQ(packed_accessor32.offsetStart(0), 0);
    EXPECT_EQ(packed_accessor32.offsetEnd(0), 3);

    // Test 64-bit packed accessor
    auto packed_accessor64 = jt.packed_accessor64<float, 1>();
    EXPECT_EQ(packed_accessor64.elementCount(), 9);
    EXPECT_EQ(packed_accessor64.batchIdx(0), 0);
    EXPECT_EQ(packed_accessor64.offsetStart(0), 0);
    EXPECT_EQ(packed_accessor64.offsetEnd(0), 3);
}

// Test data modification through accessors
TEST_F(JaggedTensorTest, AccessorDataModification) {
    auto mutable_data           = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    std::vector<int64_t> lsizes = {2, 3};
    JaggedTensor jt(lsizes, mutable_data);

    auto packed_accessor = jt.packed_accessor32<float, 1>();

    // Modify data through accessor
    packed_accessor.data()[0] = 10.0f;
    packed_accessor.data()[1] = 20.0f;

    // Check that original data was modified
    EXPECT_EQ(jt.jdata()[0].item<float>(), 10.0f);
    EXPECT_EQ(jt.jdata()[1].item<float>(), 20.0f);
}

// Test validation
TEST_F(JaggedTensorTest, Validation) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    // Should not throw for valid tensor
    EXPECT_NO_THROW(jt.check_valid());

    // Test empty tensor validation - create empty tensor using existing constructor
    // Create an empty tensor with the same device as our fixture tensors
    auto empty_tensor = torch::empty({0}, tensor_a.options());
    JaggedTensor empty_jt(empty_tensor);

    EXPECT_NO_THROW(empty_jt.check_valid());

    // Test that default-constructed JaggedTensor fails validation
    // (it creates tensors without proper devices)
    JaggedTensor default_jt;
    EXPECT_THROW(default_jt.check_valid(), c10::Error);
}

// Test static factory methods
TEST_F(JaggedTensorTest, StaticFactoryMethods) {
    // Test from_data_indices_and_list_ids
    auto data     = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto indices  = torch::tensor({0, 0, 1, 1, 2}, torch::kInt32);
    auto list_ids = torch::tensor({0, 1, 2}, torch::kInt32);

    auto jt = JaggedTensor::from_data_indices_and_list_ids(data, indices, list_ids, 3);

    EXPECT_EQ(jt.num_tensors(), 3);
    EXPECT_EQ(jt.element_count(), 5);
    EXPECT_TRUE(torch::equal(jt.jdata(), data));

    // Test from_data_offsets_and_list_ids
    // list_ids must be 2D with shape [num_tensors, ?]
    auto offsets     = torch::tensor({0, 2, 4, 5}, torch::kInt64);
    auto list_ids_2d = torch::tensor({{0}, {1}, {2}}, torch::kInt32); // Shape [3, 1]
    auto jt2         = JaggedTensor::from_data_offsets_and_list_ids(data, offsets, list_ids_2d);

    EXPECT_EQ(jt2.num_tensors(), 3);
    EXPECT_EQ(jt2.element_count(), 5);
    EXPECT_TRUE(torch::equal(jt2.jdata(), data));
}

// Test jcat functionality
TEST_F(JaggedTensorTest, JCat) {
    std::vector<torch::Tensor> tensors_a = {tensor_a, tensor_b};
    std::vector<torch::Tensor> tensors_b = {tensor_c};

    JaggedTensor jt_a(tensors_a);
    JaggedTensor jt_b(tensors_b);

    // Test concatenation as lists (dim = nullopt)
    auto concatenated = JaggedTensor::jcat({jt_a, jt_b}, std::nullopt);
    EXPECT_EQ(concatenated.num_tensors(), 3);
    EXPECT_EQ(concatenated.element_count(), 9); // 3 + 2 + 4

    // Test concatenation along dimension 0
    // For this test, we need tensors with compatible shapes
    auto tensor_compat_a = torch::tensor({1.0f, 2.0f});
    auto tensor_compat_b = torch::tensor({3.0f, 4.0f});
    auto tensor_compat_c = torch::tensor({5.0f, 6.0f});

    JaggedTensor jt_compat_a({tensor_compat_a});
    JaggedTensor jt_compat_b({tensor_compat_b});

    auto dim_concat = JaggedTensor::jcat({jt_compat_a, jt_compat_b}, 0);
    EXPECT_EQ(dim_concat.num_tensors(), 1);
    EXPECT_EQ(dim_concat.element_count(), 4);
}

// Test edge cases
TEST_F(JaggedTensorTest, EdgeCases) {
    // Test with empty tensors in list
    auto empty_tensor                             = torch::empty({0});
    std::vector<torch::Tensor> tensors_with_empty = {tensor_a, empty_tensor, tensor_b};

    EXPECT_NO_THROW({
        JaggedTensor jt(tensors_with_empty);
        EXPECT_EQ(jt.num_tensors(), 3);
        EXPECT_EQ(jt.element_count(), 5); // 3 + 0 + 2
    });

    // Test with single element tensors
    auto single_element = torch::tensor({42.0f});
    JaggedTensor jt_single(single_element);
    EXPECT_EQ(jt_single.num_tensors(), 1);
    EXPECT_EQ(jt_single.element_count(), 1);
}

// Test copy and clone operations
TEST_F(JaggedTensorTest, CopyAndClone) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    // Test clone
    auto cloned = jt.clone();
    EXPECT_TRUE(torch::equal(cloned.jdata(), jt.jdata()));
    EXPECT_EQ(cloned.num_tensors(), jt.num_tensors());

    // Modify original and ensure clone is independent
    jt.jdata()[0] = 999.0f;
    EXPECT_FALSE(torch::equal(cloned.jdata(), jt.jdata()));

    // Test contiguous
    auto contiguous = jt.contiguous();
    EXPECT_TRUE(contiguous.is_contiguous());
}

// Test requires_grad functionality
TEST_F(JaggedTensorTest, RequiresGrad) {
    auto grad_tensor = torch::tensor({1.0f, 2.0f, 3.0f}, torch::requires_grad(true));
    JaggedTensor jt(grad_tensor);

    EXPECT_TRUE(jt.requires_grad());

    auto no_grad_jt = jt.set_requires_grad(false);
    EXPECT_FALSE(no_grad_jt.requires_grad());

    auto detached = jt.detach();
    EXPECT_FALSE(detached.requires_grad());
}

// Test comparison operators
TEST_F(JaggedTensorTest, ComparisonOperators) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    // Test greater than
    auto gt_result   = jt > 2.0f;
    auto expected_gt = torch::cat({tensor_a > 2.0f, tensor_b > 2.0f}, 0);
    EXPECT_TRUE(torch::equal(gt_result.jdata(), expected_gt));

    // Test less than
    auto lt_result   = jt < 3.0f;
    auto expected_lt = torch::cat({tensor_a < 3.0f, tensor_b < 3.0f}, 0);
    EXPECT_TRUE(torch::equal(lt_result.jdata(), expected_lt));

    // Test equal
    auto eq_result   = jt == 2.0f;
    auto expected_eq = torch::cat({tensor_a == 2.0f, tensor_b == 2.0f}, 0);
    EXPECT_TRUE(torch::equal(eq_result.jdata(), expected_eq));

    // Test not equal
    auto ne_result   = jt != 1.0f;
    auto expected_ne = torch::cat({tensor_a != 1.0f, tensor_b != 1.0f}, 0);
    EXPECT_TRUE(torch::equal(ne_result.jdata(), expected_ne));
}

// Test mathematical functions
TEST_F(JaggedTensorTest, MathematicalFunctions) {
    auto positive_data          = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f, 25.0f});
    std::vector<int64_t> lsizes = {3, 2};
    JaggedTensor jt(lsizes, positive_data);

    // Test sqrt
    auto sqrt_result   = jt.sqrt();
    auto expected_sqrt = torch::sqrt(positive_data);
    EXPECT_TRUE(torch::allclose(sqrt_result.jdata(), expected_sqrt));

    // Test abs
    auto negative_data = torch::tensor({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f});
    JaggedTensor jt_neg(lsizes, negative_data);
    auto abs_result   = jt_neg.abs();
    auto expected_abs = torch::abs(negative_data);
    EXPECT_TRUE(torch::equal(abs_result.jdata(), expected_abs));

    // Test floor
    auto float_data = torch::tensor({1.7f, 2.3f, 3.9f, 4.1f, 5.8f});
    JaggedTensor jt_float(lsizes, float_data);
    auto floor_result   = jt_float.floor();
    auto expected_floor = torch::floor(float_data);
    EXPECT_TRUE(torch::equal(floor_result.jdata(), expected_floor));

    // Test ceil
    auto ceil_result   = jt_float.ceil();
    auto expected_ceil = torch::ceil(float_data);
    EXPECT_TRUE(torch::equal(ceil_result.jdata(), expected_ceil));

    // Test round
    auto round_result   = jt_float.round();
    auto expected_round = torch::round(float_data);
    EXPECT_TRUE(torch::equal(round_result.jdata(), expected_round));
}

// Test in-place operations
TEST_F(JaggedTensorTest, InPlaceOperations) {
    auto mutable_data           = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    std::vector<int64_t> lsizes = {3, 2};
    JaggedTensor jt(lsizes, mutable_data);

    // Test in-place addition
    auto jt_copy = jt.clone();
    jt_copy += 2.0f;
    auto expected_add = mutable_data + 2.0f;
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_add));

    // Test in-place multiplication
    jt_copy *= 3.0f;
    auto expected_mult = expected_add * 3.0f;
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_mult));

    // Test in-place subtraction
    jt_copy -= 1.0f;
    auto expected_sub = expected_mult - 1.0f;
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_sub));

    // Test in-place division
    jt_copy /= 2.0f;
    auto expected_div = expected_sub / 2.0f;
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_div));
}

// Test additional utility methods
TEST_F(JaggedTensorTest, AdditionalUtilityMethods) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Test numel
    EXPECT_EQ(jt.numel(), 9); // 3 + 2 + 4

    // Test is_complex
    EXPECT_FALSE(jt.is_complex());

    // Test is_floating_point
    EXPECT_TRUE(jt.is_floating_point());

    // Test is_signed
    EXPECT_TRUE(jt.is_signed());

    // Test get_device
    EXPECT_EQ(jt.get_device(), -1); // CPU device

    // Test options
    auto options = jt.options();
    EXPECT_EQ(options.dtype(), torch::kFloat32);
    EXPECT_EQ(options.device(), torch::kCPU);

    // Test layout
    EXPECT_EQ(jt.layout(), torch::kStrided);
}

// Test additional indexing and reshaping
TEST_F(JaggedTensorTest, AdditionalIndexingAndReshaping) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Test jreshape_as
    auto other_data =
        torch::tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f});
    JaggedTensor other_jt(tensors); // Same structure
    auto reshaped = jt.jreshape_as(other_jt);
    EXPECT_EQ(reshaped.num_tensors(), jt.num_tensors());
    EXPECT_EQ(reshaped.element_count(), jt.element_count());

    // Test jsqueeze
    auto squeezed = jt.jsqueeze();
    EXPECT_EQ(squeezed.num_tensors(), jt.num_tensors());
    EXPECT_EQ(squeezed.element_count(), jt.element_count());

    // Test rsize, rdim, rsizes
    EXPECT_EQ(jt.rdim(), 1);   // 1D data
    EXPECT_EQ(jt.rsize(0), 9); // Total elements
    auto rsizes = jt.rsizes();
    EXPECT_EQ(rsizes.size(), 1ul);
    EXPECT_EQ(rsizes[0], 9);
}

// Test rmask functionality
TEST_F(JaggedTensorTest, RMask) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt(tensors);

    // Create a mask that selects some elements
    auto mask =
        torch::tensor({true, false, true, true, false, true, false, true, true}, torch::kBool);
    auto masked_jt = jt.rmask(mask);

    // Check that masked tensor has correct number of elements
    EXPECT_EQ(masked_jt.element_count(), 6); // 6 true values in mask
    EXPECT_TRUE(masked_jt.num_tensors() > 0);
}

// Test additional arithmetic with different types
TEST_F(JaggedTensorTest, ArithmeticWithDifferentTypes) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    // Test with integer
    auto int_result   = jt + 5;
    auto expected_int = torch::cat({tensor_a + 5, tensor_b + 5}, 0);
    EXPECT_TRUE(torch::equal(int_result.jdata(), expected_int));

    // Test with float
    auto float_result   = jt * 2.5f;
    auto expected_float = torch::cat({tensor_a * 2.5f, tensor_b * 2.5f}, 0);
    EXPECT_TRUE(torch::allclose(float_result.jdata(), expected_float));

    // Test with tensor
    auto tensor_other    = torch::tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    auto tensor_result   = jt - tensor_other;
    auto expected_tensor = torch::cat(
        {tensor_a - tensor_other.slice(0, 0, 3), tensor_b - tensor_other.slice(0, 3, 5)}, 0);
    EXPECT_TRUE(torch::equal(tensor_result.jdata(), expected_tensor));
}

// Test additional comparison operators
TEST_F(JaggedTensorTest, AdditionalComparisonOperators) {
    std::vector<torch::Tensor> tensors = {tensor_a, tensor_b};
    JaggedTensor jt(tensors);

    // Test greater than or equal
    auto ge_result   = jt >= 2.0f;
    auto expected_ge = torch::cat({tensor_a >= 2.0f, tensor_b >= 2.0f}, 0);
    EXPECT_TRUE(torch::equal(ge_result.jdata(), expected_ge));

    // Test less than or equal
    auto le_result   = jt <= 3.0f;
    auto expected_le = torch::cat({tensor_a <= 3.0f, tensor_b <= 3.0f}, 0);
    EXPECT_TRUE(torch::equal(le_result.jdata(), expected_le));
}

// Test additional mathematical operations
TEST_F(JaggedTensorTest, AdditionalMathematicalOperations) {
    auto data                   = torch::tensor({2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::vector<int64_t> lsizes = {3, 2};
    JaggedTensor jt(lsizes, data);

    // Test pow
    auto pow_result   = jt.pow(2);
    auto expected_pow = torch::pow(data, 2);
    EXPECT_TRUE(torch::equal(pow_result.jdata(), expected_pow));

    // Test floordiv
    auto floordiv_result   = jt.floordiv(2.0f);
    auto expected_floordiv = torch::floor(data / 2.0f);
    EXPECT_TRUE(torch::equal(floordiv_result.jdata(), expected_floordiv));

    // Test modulo
    auto mod_result   = jt % 3;
    auto expected_mod = torch::fmod(data, 3.0f);
    EXPECT_TRUE(torch::allclose(mod_result.jdata(), expected_mod));
}

// Test in-place operations
TEST_F(JaggedTensorTest, InPlaceMathematicalOperations) {
    auto data                   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    std::vector<int64_t> lsizes = {3, 2};
    JaggedTensor jt(lsizes, data);

    // Test in-place pow
    auto jt_copy = jt.clone();
    jt_copy.poweq(2);
    auto expected_pow = torch::pow(data, 2);
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_pow));

    // Test in-place floordiv
    jt_copy = jt.clone();
    jt_copy.floordiveq(2.0f);
    auto expected_floordiv = torch::floor(data / 2.0f);
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_floordiv));

    // Test in-place modulo
    jt_copy = jt.clone();
    jt_copy %= 3;
    auto expected_mod = torch::fmod(data, 3.0f);
    EXPECT_TRUE(torch::allclose(jt_copy.jdata(), expected_mod));
}

// Test in-place mathematical functions
TEST_F(JaggedTensorTest, InPlaceMathematicalFunctions) {
    auto data                   = torch::tensor({1.7f, 2.3f, 3.9f, 4.1f, 5.8f});
    std::vector<int64_t> lsizes = {3, 2};
    JaggedTensor jt(lsizes, data);

    // Test in-place sqrt
    auto jt_copy = jt.clone();
    jt_copy.sqrt_();
    auto expected_sqrt = torch::sqrt(data);
    EXPECT_TRUE(torch::allclose(jt_copy.jdata(), expected_sqrt));

    // Test in-place abs
    auto neg_data = torch::tensor({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f});
    JaggedTensor jt_neg(lsizes, neg_data);
    jt_neg.abs_();
    auto expected_abs = torch::abs(neg_data);
    EXPECT_TRUE(torch::equal(jt_neg.jdata(), expected_abs));

    // Test in-place round
    jt_copy = jt.clone();
    jt_copy.round_();
    auto expected_round = torch::round(data);
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_round));

    // Test in-place floor
    jt_copy = jt.clone();
    jt_copy.floor_();
    auto expected_floor = torch::floor(data);
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_floor));

    // Test in-place ceil
    jt_copy = jt.clone();
    jt_copy.ceil_();
    auto expected_ceil = torch::ceil(data);
    EXPECT_TRUE(torch::equal(jt_copy.jdata(), expected_ceil));
}

// Test static utility functions
TEST_F(JaggedTensorTest, StaticUtilityFunctions) {
    // Test jidx_from_joffsets
    auto offsets = torch::tensor({0, 3, 5, 9}, torch::kInt64);
    auto jidx    = JaggedTensor::jidx_from_joffsets(offsets, 9);

    // Expected jidx: [0, 0, 0, 1, 1, 2, 2, 2, 2]
    // (3 elements for tensor 0, 2 elements for tensor 1, 4 elements for tensor 2)
    auto expected_jidx = torch::tensor({0, 0, 0, 1, 1, 2, 2, 2, 2}, torch::kInt32);
    EXPECT_TRUE(torch::equal(jidx, expected_jidx));

    // Test joffsets_from_jidx_and_jdata
    auto data             = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    auto computed_offsets = JaggedTensor::joffsets_from_jidx_and_jdata(jidx, data, 3);

    // Expected offsets: [0, 3, 5, 9]
    auto expected_offsets = torch::tensor({0, 3, 5, 9}, torch::kInt64);
    EXPECT_TRUE(torch::equal(computed_offsets, expected_offsets));

    // Test round-trip consistency
    auto round_trip_jidx = JaggedTensor::jidx_from_joffsets(computed_offsets, data.size(0));
    EXPECT_TRUE(torch::equal(round_trip_jidx, jidx));

    // Test edge case with single tensor
    auto single_offsets       = torch::tensor({0, 5}, torch::kInt64);
    auto single_jidx          = JaggedTensor::jidx_from_joffsets(single_offsets, 5);
    auto expected_single_jidx = torch::tensor({0, 0, 0, 0, 0}, torch::kInt32);
    EXPECT_TRUE(torch::equal(single_jidx, expected_single_jidx));

    // Test edge case with empty tensor
    auto empty_offsets = torch::tensor({0, 0}, torch::kInt64);
    auto empty_jidx    = JaggedTensor::jidx_from_joffsets(empty_offsets, 0);
    EXPECT_EQ(empty_jidx.size(0), 0);
}

// Test additional list dimension and size methods
TEST_F(JaggedTensorTest, ListDimensionAndSizeMethods) {
    // Test lsizes2 with nested jagged tensor
    auto tensor_small  = torch::tensor({1.0f});
    auto tensor_medium = torch::tensor({2.0f, 3.0f});
    auto tensor_large  = torch::tensor({4.0f, 5.0f, 6.0f});
    auto tensor_extra  = torch::tensor({7.0f, 8.0f, 9.0f, 10.0f});

    std::vector<std::vector<torch::Tensor>> nested_tensors = {
        {tensor_large, tensor_medium}, {tensor_small}, {tensor_medium, tensor_extra}};

    JaggedTensor jt_nested(nested_tensors);

    // Test ldim for nested tensor
    EXPECT_EQ(jt_nested.ldim(), 2);

    // Test lsizes2
    auto lsizes2 = jt_nested.lsizes2();
    EXPECT_EQ(lsizes2.size(), 3ul);    // 3 outer lists
    EXPECT_EQ(lsizes2[0].size(), 2ul); // First list has 2 tensors
    EXPECT_EQ(lsizes2[1].size(), 1ul); // Second list has 1 tensor
    EXPECT_EQ(lsizes2[2].size(), 2ul); // Third list has 2 tensors

    // Check specific sizes
    EXPECT_EQ(lsizes2[0][0], 3); // tensor_large has 3 elements
    EXPECT_EQ(lsizes2[0][1], 2); // tensor_medium has 2 elements
    EXPECT_EQ(lsizes2[1][0], 1); // tensor_small has 1 element
    EXPECT_EQ(lsizes2[2][0], 2); // tensor_medium has 2 elements
    EXPECT_EQ(lsizes2[2][1], 4); // tensor_extra has 4 elements

    // Test unbind2
    auto unbind2_result = jt_nested.unbind2();
    EXPECT_EQ(unbind2_result.size(), 3ul);    // 3 outer lists
    EXPECT_EQ(unbind2_result[0].size(), 2ul); // First list has 2 tensors
    EXPECT_EQ(unbind2_result[1].size(), 1ul); // Second list has 1 tensor
    EXPECT_EQ(unbind2_result[2].size(), 2ul); // Third list has 2 tensors

    // Verify the tensors in unbind2 result
    EXPECT_TRUE(torch::equal(unbind2_result[0][0], tensor_large));
    EXPECT_TRUE(torch::equal(unbind2_result[0][1], tensor_medium));
    EXPECT_TRUE(torch::equal(unbind2_result[1][0], tensor_small));
    EXPECT_TRUE(torch::equal(unbind2_result[2][0], tensor_medium));
    EXPECT_TRUE(torch::equal(unbind2_result[2][1], tensor_extra));
}

// Test element shape and dimension methods
TEST_F(JaggedTensorTest, ElementShapeAndDimensionMethods) {
    // Test with 1D tensors
    std::vector<torch::Tensor> tensors_1d = {tensor_a, tensor_b, tensor_c};
    JaggedTensor jt_1d(tensors_1d);

    EXPECT_EQ(jt_1d.edim(), 0);       // 1D tensors have 0 additional dimensions
    auto esizes_1d = jt_1d.esizes();
    EXPECT_EQ(esizes_1d.size(), 0ul); // No additional dimensions

    // Test with 2D tensors - ensure compatible shapes
    auto tensor_2d_a = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto tensor_2d_b = torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}); // Same shape as tensor_2d_a
    std::vector<torch::Tensor> tensors_2d = {tensor_2d_a, tensor_2d_b};
    JaggedTensor jt_2d(tensors_2d);

    EXPECT_EQ(jt_2d.edim(), 1); // 2D tensors have 1 additional dimension
    auto esizes_2d = jt_2d.esizes();
    EXPECT_EQ(esizes_2d.size(), 1ul);
    EXPECT_EQ(esizes_2d[0], 2); // Each element has 2 columns

    // Test with 3D tensors - ensure compatible shapes
    auto tensor_3d_a = torch::tensor({{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    auto tensor_3d_b =
        torch::tensor({{{9.0f, 10.0f}, {11.0f, 12.0f}}}); // Same shape as tensor_3d_a
    std::vector<torch::Tensor> tensors_3d = {tensor_3d_a, tensor_3d_b};
    JaggedTensor jt_3d(tensors_3d);

    EXPECT_EQ(jt_3d.edim(), 2); // 3D tensors have 2 additional dimensions
    auto esizes_3d = jt_3d.esizes();
    EXPECT_EQ(esizes_3d.size(), 2ul);
    EXPECT_EQ(esizes_3d[0], 2); // First additional dimension
    EXPECT_EQ(esizes_3d[1], 2); // Second additional dimension
}

// Test error cases for dimension methods
TEST_F(JaggedTensorTest, DimensionMethodErrorCases) {
    // Test lsizes1 on nested tensor (should throw)
    auto tensor_small                                      = torch::tensor({1.0f});
    auto tensor_medium                                     = torch::tensor({2.0f, 3.0f});
    std::vector<std::vector<torch::Tensor>> nested_tensors = {{tensor_a, tensor_b}, {tensor_small}};
    JaggedTensor jt_nested(nested_tensors);

    EXPECT_THROW(jt_nested.lsizes1(), c10::Error); // Should throw for ldim != 1

    // Test lsizes2 on flat tensor (should throw)
    std::vector<torch::Tensor> flat_tensors = {tensor_a, tensor_b};
    JaggedTensor jt_flat(flat_tensors);

    EXPECT_THROW(jt_flat.lsizes2(), c10::Error); // Should throw for ldim != 2

    // Test unbind2 on flat tensor (should throw)
    EXPECT_THROW(jt_flat.unbind2(), c10::Error); // Should throw for ldim != 2
}
