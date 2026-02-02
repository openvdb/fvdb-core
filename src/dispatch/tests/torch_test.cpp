// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <type_traits>

namespace dispatch {

// =============================================================================
// torch_types.h - Type mappings
// =============================================================================

TEST(TorchScalarCppType, Float) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kFloat>, float>);
}

TEST(TorchScalarCppType, Double) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kDouble>, double>);
}

TEST(TorchScalarCppType, Int) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kInt>, int32_t>);
}

TEST(TorchScalarCppType, Long) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kLong>, int64_t>);
}

TEST(TorchScalarCppType, Half) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kHalf>, at::Half>);
}

TEST(TorchScalarCppType, BFloat16) {
    static_assert(std::is_same_v<torch_scalar_cpp_type_t<torch::kBFloat16>, at::BFloat16>);
}

// =============================================================================
// torch_types.h - Device axes
// =============================================================================

TEST(TorchDeviceAxes, CpuCudaAxis) {
    static_assert(is_axis_v<torch_cpu_cuda_device_axis>());
    static_assert(extent_v<torch_cpu_cuda_device_axis>() == 2);
}

TEST(TorchDeviceAxes, FullDeviceAxis) {
    static_assert(is_axis_v<torch_full_device_axis>());
    static_assert(extent_v<torch_full_device_axis>() == 3); // CPU, CUDA, PrivateUse1
}

// =============================================================================
// torch_types.h - Scalar type axes
// =============================================================================

TEST(TorchScalarTypeAxes, FullFloatStypeAxis) {
    static_assert(is_axis_v<torch_full_float_stype_axis>());
    static_assert(extent_v<torch_full_float_stype_axis>() ==
                  4); // BFloat16, Float16, Float32, Float64
}

TEST(TorchScalarTypeAxes, BuiltinFloatStypeAxis) {
    static_assert(is_axis_v<torch_builtin_float_stype_axis>());
    static_assert(extent_v<torch_builtin_float_stype_axis>() == 2); // Float32, Float64
}

TEST(TorchScalarTypeAxes, FullSignedIntStypeAxis) {
    static_assert(is_axis_v<torch_full_signed_int_stype_axis>());
    static_assert(extent_v<torch_full_signed_int_stype_axis>() == 4); // Int8, Int16, Int32, Int64
}

TEST(TorchScalarTypeAxes, FullNumericStypeAxis) {
    static_assert(is_axis_v<torch_full_numeric_stype_axis>());
    static_assert(extent_v<torch_full_numeric_stype_axis>() == 8);
}

// =============================================================================
// torch_types.h - Scalar type concepts
// =============================================================================

TEST(TorchScalarTypeConcepts, IntegerStype) {
    static_assert(torch_integer_stype<torch::kInt>);
    static_assert(torch_integer_stype<torch::kLong>);
    static_assert(!torch_integer_stype<torch::kFloat>);
    static_assert(!torch_integer_stype<torch::kDouble>);
}

TEST(TorchScalarTypeConcepts, FloatStype) {
    static_assert(torch_float_stype<torch::kFloat>);
    static_assert(torch_float_stype<torch::kDouble>);
    static_assert(torch_float_stype<torch::kHalf>);
    static_assert(torch_float_stype<torch::kBFloat16>);
    static_assert(!torch_float_stype<torch::kInt>);
    static_assert(!torch_float_stype<torch::kLong>);
}

// =============================================================================
// torch_types.h - Runtime checks
// =============================================================================

TEST(TorchRuntimeChecks, IsTorchIntegerStype) {
    EXPECT_TRUE(is_torch_integer_stype(torch::kInt));
    EXPECT_TRUE(is_torch_integer_stype(torch::kLong));
    EXPECT_TRUE(is_torch_integer_stype(torch::kByte));
    EXPECT_FALSE(is_torch_integer_stype(torch::kFloat));
    EXPECT_FALSE(is_torch_integer_stype(torch::kDouble));
}

TEST(TorchRuntimeChecks, IsTorchFloatStype) {
    EXPECT_TRUE(is_torch_float_stype(torch::kFloat));
    EXPECT_TRUE(is_torch_float_stype(torch::kDouble));
    EXPECT_TRUE(is_torch_float_stype(torch::kHalf));
    EXPECT_TRUE(is_torch_float_stype(torch::kBFloat16));
    EXPECT_FALSE(is_torch_float_stype(torch::kInt));
    EXPECT_FALSE(is_torch_float_stype(torch::kLong));
}

// =============================================================================
// torch.h - Stringification
// =============================================================================

TEST(TorchCoordToString, DeviceType) {
    std::string s = torch_coord_to_string(torch::kCPU);
    EXPECT_TRUE(s.find("CPU") != std::string::npos || s.find("cpu") != std::string::npos);
}

TEST(TorchCoordToString, ScalarType) {
    std::string s = torch_coord_to_string(torch::kFloat);
    EXPECT_TRUE(s.find("Float") != std::string::npos || s.find("float") != std::string::npos);
}

TEST(TorchCoordToString, Placement) {
    std::string s = torch_coord_to_string(placement::in_place);
    EXPECT_TRUE(s.find("in_place") != std::string::npos);
}

TEST(TorchCoordToString, Determinism) {
    std::string s = torch_coord_to_string(determinism::required);
    EXPECT_TRUE(s.find("required") != std::string::npos);
}

TEST(TorchCoordToString, Contiguity) {
    std::string s = torch_coord_to_string(contiguity::contiguous);
    EXPECT_TRUE(s.find("contiguous") != std::string::npos);
}

TEST(TorchFormatDispatchCoords, SingleCoordinate) {
    auto coords   = std::make_tuple(torch::kCPU);
    std::string s = torch_format_dispatch_coords(coords);
    EXPECT_FALSE(s.empty());
}

TEST(TorchFormatDispatchCoords, MultipleCoordinates) {
    auto coords   = std::make_tuple(torch::kCPU, torch::kFloat, placement::in_place);
    std::string s = torch_format_dispatch_coords(coords);
    EXPECT_FALSE(s.empty());
    // Should contain commas
    EXPECT_TRUE(s.find(',') != std::string::npos);
}

// =============================================================================
// torch.h - Contiguity extraction
// =============================================================================

TEST(TorchGetContiguity, ContiguousTensor) {
    auto tensor = torch::zeros({2, 3});
    EXPECT_TRUE(tensor.is_contiguous());
    EXPECT_EQ(torch_get_contiguity(tensor), contiguity::contiguous);
}

TEST(TorchGetContiguity, StridedTensor) {
    auto tensor  = torch::zeros({2, 3});
    auto strided = tensor.slice(0, 0, 2, 1); // Should still be contiguous
    EXPECT_EQ(torch_get_contiguity(strided), contiguity::contiguous);

    // Create a truly strided tensor
    auto transposed = tensor.t();
    if (!transposed.is_contiguous()) {
        EXPECT_EQ(torch_get_contiguity(transposed), contiguity::strided);
    }
}

// =============================================================================
// torch.h - Dispatch wrapper
// =============================================================================

TEST(TorchDispatch, SuccessPath) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    auto result = torch_dispatch("test_func", table, std::make_tuple(1));
    EXPECT_EQ(result, 42);
}

TEST(TorchDispatch, ErrorPath) {
    using A     = axis<1>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    // Try to dispatch to invalid coordinate - should throw TORCH_CHECK_VALUE
    EXPECT_THROW(torch_dispatch("test_func", table, std::make_tuple(99)), c10::Error);
}

// =============================================================================
// dispatch.h - combined_contiguity
// =============================================================================

TEST(CombinedContiguity, AllContiguous) {
    auto t1 = torch::zeros({2, 3});
    auto t2 = torch::zeros({2, 3});
    EXPECT_TRUE(t1.is_contiguous());
    EXPECT_TRUE(t2.is_contiguous());
    EXPECT_EQ(combined_contiguity(t1, t2), contiguity::contiguous);
}

TEST(CombinedContiguity, OneStrided) {
    auto t1       = torch::zeros({2, 3});
    auto t2       = torch::zeros({3, 2}).t(); // transposed = strided
    bool strided2 = !t2.is_contiguous();
    if (strided2) {
        EXPECT_EQ(combined_contiguity(t1, t2), contiguity::strided);
    }
}

TEST(CombinedContiguity, ThreeTensors) {
    auto t1 = torch::zeros({2, 3});
    auto t2 = torch::zeros({2, 3});
    auto t3 = torch::zeros({2, 3});
    EXPECT_EQ(combined_contiguity(t1, t2, t3), contiguity::contiguous);
}

} // namespace dispatch
