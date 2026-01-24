// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Purpose: Single .cu file that exercises all library templates under nvcc
// to catch compiler-specific issues.

#include "dispatch/axes_map.h"
#include "dispatch/dispatch_table.h"
#include "dispatch/torch.h"
#include "dispatch/torch_types.h"
#include "dispatch/types.h"
#include "dispatch/visit_spaces.h"

#include <torch/torch.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace dispatch {

// =============================================================================
// dispatch_table.h templates
// =============================================================================

struct CudaTestOp {
    template <typename Coord>
    static int
    op(Coord, int x) {
        return x * 2;
    }
};

TEST(NvccSmoke, DispatchTableTemplates) {
    using A     = axis<1, 2>;
    using Axes  = axes<A>;
    using Table = dispatch_table<Axes, int(int)>;

    // Verify dispatch_table templates compile under nvcc
    auto factory = Table::from_op<CudaTestOp>();
    Table table(factory, Axes{});

    // Exercise tag/axes types in device-compiled code
    auto result = table(std::make_tuple(1), 5);
    EXPECT_EQ(result, 10);
}

// CUDA kernel that uses dispatch table types
__global__ void
test_dispatch_kernel(int *result) {
    // Just verify we can use the types in device code
    using A    = axis<1>;
    using Axes = axes<A>;
    // Types compile - that's what we're testing
    *result = 42;
}

TEST(NvccSmoke, DispatchTableInDeviceCode) {
    int *d_result;
    cudaMalloc(&d_result, sizeof(int));

    test_dispatch_kernel<<<1, 1>>>(d_result);

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 42);

    cudaFree(d_result);
}

// =============================================================================
// torch.h CUDA paths
// =============================================================================

TEST(NvccSmoke, TorchAccessorCudaOverloads) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Verify torch_accessor() CUDA overloads instantiate under nvcc
    auto tensor =
        torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch_cuda_tensor<torch::kFloat, 2> ct(tensor);

    // This should compile with nvcc
    auto accessor = torch_accessor(ct);
    EXPECT_EQ(accessor.size(0), 2);
    EXPECT_EQ(accessor.size(1), 3);
}

TEST(NvccSmoke, TorchConcreteTensorCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Exercise torch_concrete_tensor with CUDA device tags
    auto tensor =
        torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch_cuda_tensor<torch::kFloat, 2> ct(tensor);

    EXPECT_EQ(ct.tensor.device().type(), torch::kCUDA);
    EXPECT_EQ(ct.tensor.scalar_type(), torch::kFloat);
}

// =============================================================================
// types.h and detail.h
// =============================================================================

TEST(NvccSmoke, CoreTypeTemplates) {
    // Verify core type templates instantiate under nvcc
    using E    = extents<2, 3>;
    using I    = indices<0, 1>;
    using T    = tag<1, 2>;
    using A    = axis<1, 2>;
    using Axes = axes<A>;

    static_assert(is_extents_v<E>());
    static_assert(is_indices_v<I>());
    static_assert(is_tag_v<T>());
    static_assert(is_axis_v<A>());
    static_assert(is_axes_v<Axes>());

    SUCCEED();
}

// =============================================================================
// axes_map.h and visit_spaces.h
// =============================================================================

TEST(NvccSmoke, AxesMapCompiles) {
    // Verify map operations compile (host-side, but header included in nvcc)
    using A1   = axis<1, 2>;
    using A2   = axis<3, 4>;
    using Axes = axes<A1, A2>;
    axes_map<Axes, int> map;

    map.emplace(tag<1, 3>{}, 10);
    EXPECT_EQ(map.size(), 1u);
}

TEST(NvccSmoke, VisitSpacesCompiles) {
    // Verify visitation templates compile
    using A    = axis<1, 2>;
    using Axes = axes<A>;
    int count  = 0;

    auto visitor = [&count](auto /*tag*/) { ++count; };

    visit_axes_space(visitor, Axes{});
    EXPECT_EQ(count, 2);
}

// =============================================================================
// Integration: All headers together
// =============================================================================

TEST(NvccSmoke, AllHeadersTogether) {
    // Exercise all headers together to catch any interaction issues
    using DeviceAxis = torch_cpu_cuda_device_axis;
    using StypeAxis  = torch_builtin_float_stype_axis;
    using Axes       = axes<DeviceAxis, StypeAxis>;
    using Table      = dispatch_table<Axes, int()>;

    auto factory = [](auto coord) -> int (*)() { return []() { return 42; }; };

    Table table(factory, Axes{});

    // Should compile and work
    auto result = table(std::make_tuple(torch::kCPU, torch::kFloat));
    EXPECT_EQ(result, 42);
}

} // namespace dispatch
