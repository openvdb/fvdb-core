// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Single .cu file that exercises all library templates under nvcc
// to catch compiler-specific issues. Excludes for_each (phase 3).

#include "dispatch/axes_map.h"
#include "dispatch/dispatch_set.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"
#include "dispatch/visit_spaces.h"
#include "dispatch/with_value.h"

#include <torch/torch.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace dispatch {

// =============================================================================
// dispatch_table.h templates
// =============================================================================

struct cuda_test_op {
    template <typename Coord>
    static int
    op(Coord, int x) {
        return x * 2;
    }
};

TEST(NvccSmoke, DispatchTableTemplates) {
    using TestAxes = axes<torch_cpu_cuda_device_axis>;
    using Table    = dispatch_table<TestAxes, int(int)>;

    auto factory = Table::from_op<cuda_test_op>();
    Table table("nvcc_smoke", factory, TestAxes{});

    auto fn = table.select(dispatch_set{torch::kCPU});
    EXPECT_EQ(fn(5), 10);
}

// CUDA kernel that uses dispatch types
__global__ void
test_dispatch_kernel(int *result) {
    // Just verify we can use the types in device code
    using A    = axis<placement::in_place, placement::out_of_place>;
    using Axes = axes<A>;
    // Types compile — that's what we're testing
    (void)sizeof(Axes);
    *result = 42;
}

TEST(NvccSmoke, DispatchTypesInDeviceCode) {
    int *d_result;
    cudaError_t err = cudaMalloc(&d_result, sizeof(int));
    ASSERT_EQ(cudaSuccess, err);

    test_dispatch_kernel<<<1, 1>>>(d_result);

    err = cudaGetLastError();
    ASSERT_EQ(cudaSuccess, err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(cudaSuccess, err);

    int h_result;
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, err);

    EXPECT_EQ(h_result, 42);

    err = cudaFree(d_result);
    ASSERT_EQ(cudaSuccess, err);
}

// =============================================================================
// Core type templates
// =============================================================================

TEST(NvccSmoke, CoreTypeTemplates) {
    using E    = extents<2, 3>;
    using I    = indices<0, 1>;
    using T    = tag<placement::in_place>;
    using A    = axis<placement::in_place, placement::out_of_place>;
    using MyAxes = axes<A>;

    static_assert(is_extents_v<E>());
    static_assert(is_indices_v<I>());
    static_assert(is_tag_v<T>());
    static_assert(is_axis_v<A>());
    static_assert(is_axes_v<MyAxes>());

    SUCCEED();
}

// =============================================================================
// axes_map.h and visit_spaces.h
// =============================================================================

TEST(NvccSmoke, AxesMapCompiles) {
    using TestAxes = axes<full_placement_axis, full_determinism_axis>;
    axes_map<TestAxes, int> map;

    map.emplace(tag<placement::in_place, determinism::required>{}, 10);
    EXPECT_EQ(map.size(), 1u);
}

TEST(NvccSmoke, VisitSpacesCompiles) {
    using TestAxes = axes<full_placement_axis>;
    int count      = 0;

    auto visitor = [&count](auto /*tag*/) { ++count; };

    visit_axes_space(visitor, TestAxes{});
    EXPECT_EQ(count, 2);
}

// =============================================================================
// with_value concepts
// =============================================================================

TEST(NvccSmoke, WithValueConcepts) {
    using T = tag<placement::in_place, determinism::required>;

    static_assert(with_value<T, placement::in_place>);
    static_assert(!with_value<T, placement::out_of_place>);
    static_assert(with_type<T, placement>);
    static_assert(with_type<T, determinism>);
    static_assert(!with_type<T, contiguity>);

    SUCCEED();
}

// =============================================================================
// Integration: All headers together
// =============================================================================

TEST(NvccSmoke, AllHeadersTogether) {
    using DeviceAxis = torch_cpu_cuda_device_axis;
    using StypeAxis  = torch_builtin_float_stype_axis;
    using TestAxes   = axes<DeviceAxis, StypeAxis>;
    using Table      = dispatch_table<TestAxes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table("nvcc_integration", factory, TestAxes{});

    auto fn = table.select(dispatch_set{torch::kCPU, torch::kFloat});
    EXPECT_EQ(fn(), 42);
}

} // namespace dispatch
