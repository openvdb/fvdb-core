// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// TINY CUDA dispatch table test to isolate core dump / memory explosion.
// This is the minimal test case to find what triggers the problem.
//
// The full DispatchTableCudaTest.cu is core dumping with massive memory usage,
// suggesting a template instantiation explosion. This file progressively builds
// up from the absolute minimum to find the breaking point.

#include <fvdb/detail/dispatch/SparseDispatchTable.h>
#include <fvdb/detail/dispatch/TorchDispatch.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

namespace fvdb {
namespace dispatch {

// =============================================================================
// TEST 0: Does nvcc even work with gtest?
// =============================================================================

TEST(TinyCudaDispatch, NvccWorks) {
    EXPECT_TRUE(true);
}

// =============================================================================
// TEST 1: Minimal CUDA kernel (no dispatch table at all)
// =============================================================================

__global__ void
trivialKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

TEST(TinyCudaDispatch, TrivialKernelWorks) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto t = torch::zeros({16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    trivialKernel<<<1, 16>>>(t.data_ptr<float>(), 16);
    cudaDeviceSynchronize();

    auto expected = torch::ones({16}, t.options());
    EXPECT_TRUE(torch::allclose(t, expected));
}

// =============================================================================
// TEST 2: Single-axis, single-value space (1 combination)
// =============================================================================

// Invoker: just one specialization
template <torch::ScalarType Dtype> struct SingleDtypeInvoker {
    static void
    invoke(torch::Tensor &t) {
        // Just mark the tensor as processed by setting first element
        t[0] = 42.0f;
    }
};

template <auto... Vs> using SingleDtypeInst = InvokeToGet<SingleDtypeInvoker, Vs...>;

using SingleValueSpace = AxisOuterProduct<TorchScalarTypeAxis<torch::kFloat32>>;

struct SingleDtypeEncoder {
    static std::tuple<torch::ScalarType>
    encode(torch::Tensor &t) {
        return {t.scalar_type()};
    }
};

struct SingleDtypeUnbound {
    [[noreturn]] static void
    unbound(torch::ScalarType) {
        throw std::runtime_error("Unbound");
    }
};

TEST(TinyCudaDispatch, SingleAxisSingleValue) {
    // 1 dtype = 1 combination total
    using Generators = SubspaceGenerator<SingleDtypeInst, SingleValueSpace>;

    auto dispatcher = build_dispatcher<SingleValueSpace,
                                       Generators,
                                       SingleDtypeEncoder,
                                       SingleDtypeUnbound,
                                       void,
                                       torch::Tensor &>();

    auto t = torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32));
    dispatcher(t);
    EXPECT_EQ(t[0].item<float>(), 42.0f);
}

// =============================================================================
// TEST 3: Two-axis, single value each (1 combination, but 2D lookup)
// =============================================================================

template <torch::DeviceType Device, torch::ScalarType Dtype> struct TwoAxisInvoker {
    static void
    invoke(torch::Tensor &t) {
        t[0] = 99.0f;
    }
};

template <auto... Vs> using TwoAxisInst = InvokeToGet<TwoAxisInvoker, Vs...>;

using TwoAxisSingleSpace =
    AxisOuterProduct<TorchDeviceDispatchAxis<torch::kCPU>, TorchScalarTypeAxis<torch::kFloat32>>;

struct TwoAxisEncoder {
    static std::tuple<torch::DeviceType, torch::ScalarType>
    encode(torch::Tensor &t) {
        return {t.device().type(), t.scalar_type()};
    }
};

struct TwoAxisUnbound {
    [[noreturn]] static void
    unbound(torch::DeviceType, torch::ScalarType) {
        throw std::runtime_error("Unbound");
    }
};

TEST(TinyCudaDispatch, TwoAxisSingleValueEach) {
    // 1 device × 1 dtype = 1 combination
    using Generators = SubspaceGenerator<TwoAxisInst, TwoAxisSingleSpace>;

    auto dispatcher = build_dispatcher<TwoAxisSingleSpace,
                                       Generators,
                                       TwoAxisEncoder,
                                       TwoAxisUnbound,
                                       void,
                                       torch::Tensor &>();

    auto t = torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    dispatcher(t);
    EXPECT_EQ(t[0].item<float>(), 99.0f);
}

// =============================================================================
// TEST 4: Two devices (CPU + CUDA), one dtype (2 combinations)
// =============================================================================

template <torch::DeviceType Device, torch::ScalarType Dtype> struct CpuCudaInvoker;

template <torch::ScalarType Dtype> struct CpuCudaInvoker<torch::kCPU, Dtype> {
    static void
    invoke(torch::Tensor &t) {
        t[0] = 1.0f; // CPU marker
    }
};

template <torch::ScalarType Dtype> struct CpuCudaInvoker<torch::kCUDA, Dtype> {
    static void
    invoke(torch::Tensor &t) {
        // For CUDA, we need to do this on host (accessor won't work on CUDA tensor directly)
        t.fill_(2.0f); // CUDA marker - use fill_ which works on CUDA
    }
};

template <auto... Vs> using CpuCudaInst = InvokeToGet<CpuCudaInvoker, Vs...>;

using TwoDeviceSpace = AxisOuterProduct<TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>,
                                        TorchScalarTypeAxis<torch::kFloat32>>;

TEST(TinyCudaDispatch, TwoDevicesOneType) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // 2 devices × 1 dtype = 2 combinations
    using Generators = SubspaceGenerator<CpuCudaInst, TwoDeviceSpace>;

    auto dispatcher = build_dispatcher<TwoDeviceSpace,
                                       Generators,
                                       TwoAxisEncoder,
                                       TwoAxisUnbound,
                                       void,
                                       torch::Tensor &>();

    // Test CPU path
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        dispatcher(t);
        EXPECT_EQ(t[0].item<float>(), 1.0f);
    }

    // Test CUDA path
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        dispatcher(t);
        torch::cuda::synchronize();
        EXPECT_EQ(t[0].item<float>(), 2.0f);
    }
}

// =============================================================================
// TEST 5: Add actual CUDA kernel dispatch (2 combinations with real kernel)
// =============================================================================

__global__ void
markerKernel(float *data, float marker) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] = marker;
    }
}

template <torch::DeviceType Device, torch::ScalarType Dtype> struct KernelInvoker;

template <torch::ScalarType Dtype> struct KernelInvoker<torch::kCPU, Dtype> {
    static void
    invoke(torch::Tensor &t) {
        t[0] = 100.0f; // CPU marker
    }
};

template <torch::ScalarType Dtype> struct KernelInvoker<torch::kCUDA, Dtype> {
    static void
    invoke(torch::Tensor &t) {
        c10::cuda::CUDAGuard guard(t.device());
        auto stream = c10::cuda::getCurrentCUDAStream();
        markerKernel<<<1, 1, 0, stream.stream()>>>(t.data_ptr<float>(), 200.0f);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

template <auto... Vs> using KernelInst = InvokeToGet<KernelInvoker, Vs...>;

TEST(TinyCudaDispatch, RealKernelDispatch) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    using Generators = SubspaceGenerator<KernelInst, TwoDeviceSpace>;

    auto dispatcher = build_dispatcher<TwoDeviceSpace,
                                       Generators,
                                       TwoAxisEncoder,
                                       TwoAxisUnbound,
                                       void,
                                       torch::Tensor &>();

    // Test CPU
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        dispatcher(t);
        EXPECT_EQ(t[0].item<float>(), 100.0f);
    }

    // Test CUDA
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        dispatcher(t);
        torch::cuda::synchronize();
        EXPECT_EQ(t[0].item<float>(), 200.0f);
    }
}

// =============================================================================
// TEST 6: Two devices × two dtypes (4 combinations) - this is where trouble may start
// =============================================================================

template <torch::DeviceType Device, torch::ScalarType Dtype> struct FourComboInvoker;

template <> struct FourComboInvoker<torch::kCPU, torch::kFloat32> {
    static void
    invoke(torch::Tensor &t) {
        t.fill_(1.0f);
    }
};

template <> struct FourComboInvoker<torch::kCPU, torch::kFloat64> {
    static void
    invoke(torch::Tensor &t) {
        t.fill_(2.0);
    }
};

template <> struct FourComboInvoker<torch::kCUDA, torch::kFloat32> {
    static void
    invoke(torch::Tensor &t) {
        t.fill_(3.0f);
    }
};

template <> struct FourComboInvoker<torch::kCUDA, torch::kFloat64> {
    static void
    invoke(torch::Tensor &t) {
        t.fill_(4.0);
    }
};

template <auto... Vs> using FourComboInst = InvokeToGet<FourComboInvoker, Vs...>;

using FourComboSpace = AxisOuterProduct<TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>,
                                        TorchScalarTypeAxis<torch::kFloat32, torch::kFloat64>>;

TEST(TinyCudaDispatch, TwoDevicesTwoDtypes) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // 2 devices × 2 dtypes = 4 combinations
    using Generators = SubspaceGenerator<FourComboInst, FourComboSpace>;

    auto dispatcher = build_dispatcher<FourComboSpace,
                                       Generators,
                                       TwoAxisEncoder,
                                       TwoAxisUnbound,
                                       void,
                                       torch::Tensor &>();

    // CPU float32
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        dispatcher(t);
        EXPECT_EQ(t[0].item<float>(), 1.0f);
    }

    // CPU float64
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
        dispatcher(t);
        EXPECT_EQ(t[0].item<double>(), 2.0);
    }

    // CUDA float32
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        dispatcher(t);
        torch::cuda::synchronize();
        EXPECT_EQ(t[0].item<float>(), 3.0f);
    }

    // CUDA float64
    {
        auto t =
            torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
        dispatcher(t);
        torch::cuda::synchronize();
        EXPECT_EQ(t[0].item<double>(), 4.0);
    }
}

} // namespace dispatch
} // namespace fvdb
