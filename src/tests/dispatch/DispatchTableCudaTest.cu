// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// CUDA-specific tests for the dispatch table mechanism.
// These tests verify that the dispatch table works correctly when:
// 1. Dispatching to actual CUDA kernels
// 2. Using torch scalar types (including bfloat16, float16)
// 3. Dispatching across multiple axes (device × dtype)
// 4. Using InvokeToGet with CUDA code compiled by nvcc
//
// The goal is to reveal nvcc-specific issues with template instantiation,
// design flaws in the dispatch mechanism, and implementation errors.

#include <fvdb/detail/dispatch/SparseDispatchTable.h>
#include <fvdb/detail/dispatch/TorchDispatch.h>
#include <fvdb/detail/dispatch/TorchTags.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// =============================================================================
// Type Mapping Utilities
// =============================================================================

// Map torch::ScalarType to C++ type at compile time
template <torch::ScalarType S>
using ScalarCppType = typename c10::impl::ScalarTypeToCPPType<S>::type;

// =============================================================================
// CUDA Kernels for SAXPY (y = alpha * x + y)
// =============================================================================

template <typename T>
__global__ void
saxpyKernel(T alpha, const T *__restrict__ x, T *__restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// CPU implementation for comparison
template <typename T>
void
saxpyCpu(T alpha, const T *x, T *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// =============================================================================
// Dispatch Axes Definitions
// =============================================================================

// Device axis: CPU and CUDA
using DeviceAxis = TorchDeviceDispatchAxis<torch::kCPU, torch::kCUDA>;

// Common floating-point types
using FloatDtypeAxis =
    TorchScalarTypeAxis<torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16>;

// Integer types for testing
using IntDtypeAxis = TorchScalarTypeAxis<torch::kInt32, torch::kInt64>;

// Combined axis spaces
using DeviceDtypeSpace    = AxisOuterProduct<DeviceAxis, FloatDtypeAxis>;
using DeviceIntDtypeSpace = AxisOuterProduct<DeviceAxis, IntDtypeAxis>;
using FloatDtypeOnlySpace = AxisOuterProduct<FloatDtypeAxis>;

// =============================================================================
// SAXPY Invoker Templates
// =============================================================================
// These templates are parameterized by device and dtype, and contain the
// actual implementation logic.

template <torch::DeviceType Device, torch::ScalarType Dtype> struct SaxpyInvoker;

// CPU specialization
template <torch::ScalarType Dtype> struct SaxpyInvoker<torch::kCPU, Dtype> {
    using scalar_t = ScalarCppType<Dtype>;

    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, double alpha) {
        TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
        TORCH_CHECK(y.device().is_cpu(), "y must be CPU tensor");
        TORCH_CHECK(x.scalar_type() == Dtype, "x dtype mismatch");
        TORCH_CHECK(y.scalar_type() == Dtype, "y dtype mismatch");

        auto x_acc = x.accessor<scalar_t, 1>();
        auto y_acc = y.accessor<scalar_t, 1>();
        int64_t n  = x.size(0);

        scalar_t a = static_cast<scalar_t>(alpha);
        for (int64_t i = 0; i < n; ++i) {
            y_acc[i] = a * x_acc[i] + y_acc[i];
        }
    }
};

// CUDA specialization
template <torch::ScalarType Dtype> struct SaxpyInvoker<torch::kCUDA, Dtype> {
    using scalar_t = ScalarCppType<Dtype>;

    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, double alpha) {
        TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
        TORCH_CHECK(y.device().is_cuda(), "y must be CUDA tensor");
        TORCH_CHECK(x.scalar_type() == Dtype, "x dtype mismatch");
        TORCH_CHECK(y.scalar_type() == Dtype, "y dtype mismatch");

        c10::cuda::CUDAGuard guard(x.device());
        auto stream = c10::cuda::getCurrentCUDAStream();

        int64_t n           = x.size(0);
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;

        scalar_t a = static_cast<scalar_t>(alpha);
        saxpyKernel<scalar_t><<<numBlocks, blockSize, 0, stream.stream()>>>(
            a, x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

// =============================================================================
// InvokeToGet Instantiator for SAXPY
// =============================================================================
// This tests the InvokeToGet adapter pattern with CUDA code.

template <auto... Vs> using SaxpyInstantiator = InvokeToGet<SaxpyInvoker, Vs...>;

// =============================================================================
// Encoders
// =============================================================================

struct DeviceDtypeEncoder {
    static std::tuple<torch::DeviceType, torch::ScalarType>
    encode(torch::Tensor &y, const torch::Tensor &x, double /*alpha*/) {
        return std::make_tuple(x.device().type(), x.scalar_type());
    }
};

// =============================================================================
// Unbound Handlers
// =============================================================================

struct SaxpyUnboundHandler {
    [[noreturn]] static void
    unbound(torch::DeviceType device, torch::ScalarType dtype) {
        throw std::runtime_error(
            "SAXPY not implemented for device=" + std::string(c10::DeviceTypeName(device)) +
            " dtype=" + std::string(c10::toString(dtype)));
    }
};

// =============================================================================
// Single Dtype Test Infrastructure
// =============================================================================

using SingleDtypeSpace = AxisOuterProduct<TorchScalarTypeAxis<torch::kFloat32>>;

struct SingleDtypeEncoder {
    static std::tuple<torch::ScalarType>
    encode(torch::Tensor & /*y*/, const torch::Tensor &x, double /*alpha*/) {
        return std::make_tuple(x.scalar_type());
    }
};

struct SingleDtypeUnbound {
    [[noreturn]] static void
    unbound(torch::ScalarType dtype) {
        throw std::runtime_error("Unsupported dtype: " + std::string(c10::toString(dtype)));
    }
};

template <torch::ScalarType Dtype> struct CudaFloat32Invoker {
    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, double alpha) {
        SaxpyInvoker<torch::kCUDA, Dtype>::invoke(y, x, alpha);
    }
};

template <auto... Vs> using CudaSaxpyInst = InvokeToGet<CudaFloat32Invoker, Vs...>;

// =============================================================================
// Three-Axis SAXPY Infrastructure
// =============================================================================

// A three-axis invoker: device × dtype × bool (e.g., in-place vs out-of-place)
template <torch::DeviceType Device, torch::ScalarType Dtype, bool InPlace>
struct ThreeAxisSaxpyInvoker;

// CPU in-place
template <torch::ScalarType Dtype> struct ThreeAxisSaxpyInvoker<torch::kCPU, Dtype, true> {
    using scalar_t = ScalarCppType<Dtype>;

    static torch::Tensor
    invoke(const torch::Tensor &x, torch::Tensor y, double alpha) {
        auto x_acc = x.accessor<scalar_t, 1>();
        auto y_acc = y.accessor<scalar_t, 1>();
        scalar_t a = static_cast<scalar_t>(alpha);
        for (int64_t i = 0; i < x.size(0); ++i) {
            y_acc[i] = a * x_acc[i] + y_acc[i];
        }
        return y;
    }
};

// CPU out-of-place
template <torch::ScalarType Dtype> struct ThreeAxisSaxpyInvoker<torch::kCPU, Dtype, false> {
    using scalar_t = ScalarCppType<Dtype>;

    static torch::Tensor
    invoke(const torch::Tensor &x, torch::Tensor y, double alpha) {
        auto result = y.clone();
        auto x_acc  = x.accessor<scalar_t, 1>();
        auto r_acc  = result.accessor<scalar_t, 1>();
        scalar_t a  = static_cast<scalar_t>(alpha);
        for (int64_t i = 0; i < x.size(0); ++i) {
            r_acc[i] = a * x_acc[i] + r_acc[i];
        }
        return result;
    }
};

// CUDA in-place
template <torch::ScalarType Dtype> struct ThreeAxisSaxpyInvoker<torch::kCUDA, Dtype, true> {
    using scalar_t = ScalarCppType<Dtype>;

    static torch::Tensor
    invoke(const torch::Tensor &x, torch::Tensor y, double alpha) {
        c10::cuda::CUDAGuard guard(x.device());
        auto stream         = c10::cuda::getCurrentCUDAStream();
        int64_t n           = x.size(0);
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_t a          = static_cast<scalar_t>(alpha);
        saxpyKernel<scalar_t><<<numBlocks, blockSize, 0, stream.stream()>>>(
            a, x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
};

// CUDA out-of-place
template <torch::ScalarType Dtype> struct ThreeAxisSaxpyInvoker<torch::kCUDA, Dtype, false> {
    using scalar_t = ScalarCppType<Dtype>;

    static torch::Tensor
    invoke(const torch::Tensor &x, torch::Tensor y, double alpha) {
        auto result = y.clone();
        c10::cuda::CUDAGuard guard(x.device());
        auto stream         = c10::cuda::getCurrentCUDAStream();
        int64_t n           = x.size(0);
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_t a          = static_cast<scalar_t>(alpha);
        saxpyKernel<scalar_t><<<numBlocks, blockSize, 0, stream.stream()>>>(
            a, x.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
};

template <auto... Vs> using ThreeAxisSaxpyInst = InvokeToGet<ThreeAxisSaxpyInvoker, Vs...>;

using ThreeAxisSpace =
    AxisOuterProduct<DeviceAxis, FloatDtypeAxis, SameTypeUniqueValuePack<true, false>>;

struct ThreeAxisEncoder {
    static std::tuple<torch::DeviceType, torch::ScalarType, bool>
    encode(const torch::Tensor &x, torch::Tensor /*y*/, double /*alpha*/, bool inplace) {
        return std::make_tuple(x.device().type(), x.scalar_type(), inplace);
    }
};

struct ThreeAxisUnbound {
    [[noreturn]] static void
    unbound(torch::DeviceType device, torch::ScalarType dtype, bool inplace) {
        throw std::runtime_error("Unsupported: device=" + std::string(c10::DeviceTypeName(device)) +
                                 " dtype=" + std::string(c10::toString(dtype)) +
                                 " inplace=" + std::to_string(inplace));
    }
};

// =============================================================================
// Integer SAXPY Infrastructure
// =============================================================================

template <torch::DeviceType Device, torch::ScalarType Dtype> struct IntSaxpyInvoker;

template <torch::ScalarType Dtype> struct IntSaxpyInvoker<torch::kCPU, Dtype> {
    using scalar_t = ScalarCppType<Dtype>;

    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, int64_t alpha) {
        auto x_acc = x.accessor<scalar_t, 1>();
        auto y_acc = y.accessor<scalar_t, 1>();
        scalar_t a = static_cast<scalar_t>(alpha);
        for (int64_t i = 0; i < x.size(0); ++i) {
            y_acc[i] = a * x_acc[i] + y_acc[i];
        }
    }
};

template <torch::ScalarType Dtype> struct IntSaxpyInvoker<torch::kCUDA, Dtype> {
    using scalar_t = ScalarCppType<Dtype>;

    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, int64_t alpha) {
        c10::cuda::CUDAGuard guard(x.device());
        auto stream         = c10::cuda::getCurrentCUDAStream();
        int64_t n           = x.size(0);
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_t a          = static_cast<scalar_t>(alpha);
        saxpyKernel<scalar_t><<<numBlocks, blockSize, 0, stream.stream()>>>(
            a, x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

template <auto... Vs> using IntSaxpyInst = InvokeToGet<IntSaxpyInvoker, Vs...>;

struct IntDeviceDtypeEncoder {
    static std::tuple<torch::DeviceType, torch::ScalarType>
    encode(torch::Tensor & /*y*/, const torch::Tensor &x, int64_t /*alpha*/) {
        return std::make_tuple(x.device().type(), x.scalar_type());
    }
};

struct IntSaxpyUnbound {
    [[noreturn]] static void
    unbound(torch::DeviceType device, torch::ScalarType dtype) {
        throw std::runtime_error(
            "Int SAXPY not supported for device=" + std::string(c10::DeviceTypeName(device)) +
            " dtype=" + std::string(c10::toString(dtype)));
    }
};

// =============================================================================
// Partial Binding Test Infrastructure
// =============================================================================

using CudaOnlySpace =
    AxisOuterProduct<TorchDeviceDispatchAxis<torch::kCUDA>, TorchScalarTypeAxis<torch::kFloat32>>;

template <torch::DeviceType Device, torch::ScalarType Dtype> struct CudaOnlySaxpyInvoker {
    static void
    invoke(torch::Tensor &y, const torch::Tensor &x, double alpha) {
        SaxpyInvoker<Device, Dtype>::invoke(y, x, alpha);
    }
};

template <auto... Vs> using CudaOnlyInst = InvokeToGet<CudaOnlySaxpyInvoker, Vs...>;

// =============================================================================
// Test Fixture
// =============================================================================

class DispatchTableCudaTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    // Helper to create test tensors
    template <torch::ScalarType Dtype>
    std::pair<torch::Tensor, torch::Tensor>
    createTestTensors(torch::Device device, int64_t n = 1024) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto x       = torch::randn({n}, options);
        auto y       = torch::randn({n}, options);
        // Convert dtype and move to device
        return {x.to(Dtype).to(device), y.to(Dtype).to(device)};
    }

    // Verify SAXPY result with tolerance appropriate for the dtype
    void
    verifySaxpy(const torch::Tensor &y_result,
                const torch::Tensor &y_original,
                const torch::Tensor &x,
                double alpha,
                double rtol = 1e-4,
                double atol = 1e-5) {
        // Compute expected on CPU in float64 for accuracy
        auto x_f64        = x.to(torch::kFloat64).to(torch::kCPU);
        auto y_orig_f64   = y_original.to(torch::kFloat64).to(torch::kCPU);
        auto expected_f64 = alpha * x_f64 + y_orig_f64;

        auto result_f64 = y_result.to(torch::kFloat64).to(torch::kCPU);
        EXPECT_TRUE(torch::allclose(result_f64, expected_f64, rtol, atol))
            << "SAXPY result mismatch. Max diff: "
            << (result_f64 - expected_f64).abs().max().item<double>();
    }
};

// =============================================================================
// Basic CUDA Dispatch Tests
// =============================================================================

TEST_F(DispatchTableCudaTest, SingleDtypeDispatchFloat32Cuda) {
    // Test dispatching a single dtype on CUDA
    using Generators = SubspaceGenerator<CudaSaxpyInst, SingleDtypeSpace>;

    auto dispatcher = build_dispatcher<SingleDtypeSpace,
                                       Generators,
                                       SingleDtypeEncoder,
                                       SingleDtypeUnbound,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
    auto y_orig = y.clone();

    dispatcher(y, x, 2.0);
    torch::cuda::synchronize();

    verifySaxpy(y, y_orig, x, 2.0);
}

TEST_F(DispatchTableCudaTest, DeviceDtypeDispatchFloat32) {
    // Test full device × dtype dispatch for float32
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Test CUDA
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.5);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 2.5);
    }

    // Test CPU
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCPU);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.5);
        verifySaxpy(y, y_orig, x, 2.5);
    }
}

TEST_F(DispatchTableCudaTest, DeviceDtypeDispatchFloat64) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Test CUDA
    {
        auto [x, y] = createTestTensors<torch::kFloat64>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 3.14);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 3.14, 1e-10, 1e-12);
    }

    // Test CPU
    {
        auto [x, y] = createTestTensors<torch::kFloat64>(torch::kCPU);
        auto y_orig = y.clone();
        dispatcher(y, x, 3.14);
        verifySaxpy(y, y_orig, x, 3.14, 1e-10, 1e-12);
    }
}

// =============================================================================
// Half-Precision Type Tests (bfloat16, float16)
// =============================================================================

TEST_F(DispatchTableCudaTest, DeviceDtypeDispatchFloat16) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Test CUDA float16
    {
        auto [x, y] = createTestTensors<torch::kFloat16>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 1.5);
        torch::cuda::synchronize();
        // float16 has less precision
        verifySaxpy(y, y_orig, x, 1.5, 1e-2, 1e-2);
    }
}

TEST_F(DispatchTableCudaTest, DeviceDtypeDispatchBFloat16) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Test CUDA bfloat16
    {
        auto [x, y] = createTestTensors<torch::kBFloat16>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 1.5);
        torch::cuda::synchronize();
        // bfloat16 has even less precision than float16
        verifySaxpy(y, y_orig, x, 1.5, 5e-2, 5e-2);
    }
}

// Test all half-precision types in one dispatcher
TEST_F(DispatchTableCudaTest, AllHalfPrecisionTypesInOneDispatcher) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Dispatch to float16
    {
        auto [x, y] = createTestTensors<torch::kFloat16>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.0);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 2.0, 1e-2, 1e-2);
    }

    // Dispatch to bfloat16 - same dispatcher, different dtype
    {
        auto [x, y] = createTestTensors<torch::kBFloat16>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.0);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 2.0, 5e-2, 5e-2);
    }

    // Dispatch to float32 - same dispatcher, different dtype
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.0);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 2.0);
    }
}

// =============================================================================
// Three-Axis Dispatch Tests
// =============================================================================

TEST_F(DispatchTableCudaTest, ThreeAxisDispatch) {
    using Generators = SubspaceGenerator<ThreeAxisSaxpyInst, ThreeAxisSpace>;

    auto dispatcher = build_dispatcher<ThreeAxisSpace,
                                       Generators,
                                       ThreeAxisEncoder,
                                       ThreeAxisUnbound,
                                       torch::Tensor,
                                       const torch::Tensor &,
                                       torch::Tensor,
                                       double,
                                       bool>();

    // Test in-place CUDA float32
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        auto y_orig = y.clone();
        auto result = dispatcher(x, y, 2.0, true);
        torch::cuda::synchronize();

        // In-place: result should be same memory as y
        EXPECT_EQ(result.data_ptr(), y.data_ptr());
        verifySaxpy(result, y_orig, x, 2.0);
    }

    // Test out-of-place CUDA float32
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        auto y_orig = y.clone();
        auto result = dispatcher(x, y, 2.0, false);
        torch::cuda::synchronize();

        // Out-of-place: result should be different memory
        EXPECT_NE(result.data_ptr(), y.data_ptr());
        // y should be unchanged
        EXPECT_TRUE(torch::equal(y, y_orig));
        verifySaxpy(result, y_orig, x, 2.0);
    }

    // Test in-place CPU bfloat16
    {
        auto [x, y] = createTestTensors<torch::kBFloat16>(torch::kCPU);
        auto y_orig = y.clone();
        auto result = dispatcher(x, y, 1.5, true);

        EXPECT_EQ(result.data_ptr(), y.data_ptr());
        verifySaxpy(result, y_orig, x, 1.5, 5e-2, 5e-2);
    }
}

// =============================================================================
// Static Dispatch Table Tests
// =============================================================================

TEST_F(DispatchTableCudaTest, StaticDispatcherWorks) {
    // Test that dispatchers can be stored as static const (common pattern)
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    static const auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                                    Generators,
                                                    DeviceDtypeEncoder,
                                                    SaxpyUnboundHandler,
                                                    void,
                                                    torch::Tensor &,
                                                    const torch::Tensor &,
                                                    double>();

    auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
    auto y_orig = y.clone();

    dispatcher(y, x, 4.0);
    torch::cuda::synchronize();

    verifySaxpy(y, y_orig, x, 4.0);
}

// =============================================================================
// Integer Type Dispatch Tests
// =============================================================================

TEST_F(DispatchTableCudaTest, IntegerTypeDispatch) {
    using Generators = SubspaceGenerator<IntSaxpyInst, DeviceIntDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceIntDtypeSpace,
                                       Generators,
                                       IntDeviceDtypeEncoder,
                                       IntSaxpyUnbound,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       int64_t>();

    // Test int32 on CUDA
    {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto x       = torch::randint(0, 100, {1024}, options);
        auto y       = torch::randint(0, 100, {1024}, options);
        auto y_orig  = y.clone();

        dispatcher(y, x, 2);
        torch::cuda::synchronize();

        auto expected = 2 * x + y_orig;
        EXPECT_TRUE(torch::equal(y, expected));
    }

    // Test int64 on CPU
    {
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto x       = torch::randint(0, 100, {1024}, options);
        auto y       = torch::randint(0, 100, {1024}, options);
        auto y_orig  = y.clone();

        dispatcher(y, x, 3);

        auto expected = 3 * x + y_orig;
        EXPECT_TRUE(torch::equal(y, expected));
    }
}

// =============================================================================
// Partial Binding Tests (Some combinations not bound)
// =============================================================================

TEST_F(DispatchTableCudaTest, PartialBindingThrowsForUnbound) {
    // Only bind CUDA float32, but use full space
    using Generators = SubspaceGenerator<CudaOnlyInst, CudaOnlySpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace, // Full space
                                       Generators,       // But only CUDA×float32 bound
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // CUDA float32 should work
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        EXPECT_NO_THROW(dispatcher(y, x, 2.0));
        torch::cuda::synchronize();
    }

    // CPU float32 should throw (not bound)
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCPU);
        EXPECT_THROW(dispatcher(y, x, 2.0), std::runtime_error);
    }

    // CUDA float64 should throw (not bound)
    {
        auto [x, y] = createTestTensors<torch::kFloat64>(torch::kCUDA);
        EXPECT_THROW(dispatcher(y, x, 2.0), std::runtime_error);
    }
}

// =============================================================================
// Function Pointer Address Verification
// =============================================================================

TEST_F(DispatchTableCudaTest, InvokeToGetPreservesAddressInCuda) {
    // Verify InvokeToGet returns the exact address of the invoke function
    auto fromInvokeToGet = InvokeToGet<SaxpyInvoker, torch::kCUDA, torch::kFloat32>::get();
    auto direct          = &SaxpyInvoker<torch::kCUDA, torch::kFloat32>::invoke;

    EXPECT_EQ(fromInvokeToGet, direct);
}

// =============================================================================
// Large Dispatch Space Test
// =============================================================================

TEST_F(DispatchTableCudaTest, LargeDispatchSpaceCompiles) {
    // Test with a larger dispatch space to ensure no compile-time issues
    using LargeDtypeAxis = TorchScalarTypeAxis<torch::kFloat32,
                                               torch::kFloat64,
                                               torch::kFloat16,
                                               torch::kBFloat16,
                                               torch::kInt32,
                                               torch::kInt64,
                                               torch::kInt16,
                                               torch::kInt8>;

    // 2 devices × 8 dtypes = 16 combinations
    using LargeSpace = AxisOuterProduct<DeviceAxis, LargeDtypeAxis>;

    // Just verify the space compiles and has correct size
    static_assert(LargeSpace::size == 16, "Large space should have 16 combinations");
    static_assert(LargeSpace::num_axes == 2, "Large space should have 2 axes");

    // Verify index_of_values works
    auto idx = LargeSpace::index_of_values(torch::kCUDA, torch::kFloat32);
    EXPECT_TRUE(idx.has_value());
}

// =============================================================================
// Multiple Dispatchers Test
// =============================================================================

TEST_F(DispatchTableCudaTest, MultipleDifferentDispatchers) {
    // Create two different dispatchers to ensure no static state collision

    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher1 = build_dispatcher<DeviceDtypeSpace,
                                        Generators,
                                        DeviceDtypeEncoder,
                                        SaxpyUnboundHandler,
                                        void,
                                        torch::Tensor &,
                                        const torch::Tensor &,
                                        double>();

    auto dispatcher2 = build_dispatcher<DeviceDtypeSpace,
                                        Generators,
                                        DeviceDtypeEncoder,
                                        SaxpyUnboundHandler,
                                        void,
                                        torch::Tensor &,
                                        const torch::Tensor &,
                                        double>();

    // Both should work independently
    auto [x1, y1] = createTestTensors<torch::kFloat32>(torch::kCUDA);
    auto y1_orig  = y1.clone();
    dispatcher1(y1, x1, 2.0);

    auto [x2, y2] = createTestTensors<torch::kFloat32>(torch::kCUDA);
    auto y2_orig  = y2.clone();
    dispatcher2(y2, x2, 3.0);

    torch::cuda::synchronize();

    verifySaxpy(y1, y1_orig, x1, 2.0);
    verifySaxpy(y2, y2_orig, x2, 3.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(DispatchTableCudaTest, SmallTensorDispatch) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Very small tensor (edge case for block size)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto x       = torch::randn({1}, options);
    auto y       = torch::randn({1}, options);
    auto y_orig  = y.clone();

    dispatcher(y, x, 5.0);
    torch::cuda::synchronize();

    verifySaxpy(y, y_orig, x, 5.0);
}

TEST_F(DispatchTableCudaTest, LargeTensorDispatch) {
    using Generators = SubspaceGenerator<SaxpyInstantiator, DeviceDtypeSpace>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // Large tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto x       = torch::randn({1000000}, options);
    auto y       = torch::randn({1000000}, options);
    auto y_orig  = y.clone();

    dispatcher(y, x, 0.5);
    torch::cuda::synchronize();

    verifySaxpy(y, y_orig, x, 0.5);
}

// =============================================================================
// PointGenerator with CUDA Test
// =============================================================================

TEST_F(DispatchTableCudaTest, PointGeneratorWithCuda) {
    // Test PointGenerator explicitly binding specific combinations
    using CudaFloat32Point = PointGenerator<SaxpyInstantiator, torch::kCUDA, torch::kFloat32>;
    using CpuFloat64Point  = PointGenerator<SaxpyInstantiator, torch::kCPU, torch::kFloat64>;
    using Generators       = GeneratorList<CudaFloat32Point, CpuFloat64Point>;

    auto dispatcher = build_dispatcher<DeviceDtypeSpace,
                                       Generators,
                                       DeviceDtypeEncoder,
                                       SaxpyUnboundHandler,
                                       void,
                                       torch::Tensor &,
                                       const torch::Tensor &,
                                       double>();

    // CUDA float32 should work
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCUDA);
        auto y_orig = y.clone();
        dispatcher(y, x, 2.0);
        torch::cuda::synchronize();
        verifySaxpy(y, y_orig, x, 2.0);
    }

    // CPU float64 should work
    {
        auto [x, y] = createTestTensors<torch::kFloat64>(torch::kCPU);
        auto y_orig = y.clone();
        dispatcher(y, x, 3.0);
        verifySaxpy(y, y_orig, x, 3.0, 1e-10, 1e-12);
    }

    // CPU float32 should throw (not bound)
    {
        auto [x, y] = createTestTensors<torch::kFloat32>(torch::kCPU);
        EXPECT_THROW(dispatcher(y, x, 2.0), std::runtime_error);
    }
}

// =============================================================================
// Type Deduction Tests for CUDA
// =============================================================================

TEST_F(DispatchTableCudaTest, TypeDeductionForCudaInvoker) {
    // Verify type deduction works correctly for CUDA invokers
    using ExpectedFPtr = FunctionPtr<void, torch::Tensor &, const torch::Tensor &, double>;

    // Verify SaxpyInstantiator returns the expected type
    static_assert(std::is_same_v<decltype(SaxpyInstantiator<torch::kCUDA, torch::kFloat32>::get()),
                                 ExpectedFPtr>);
    static_assert(std::is_same_v<decltype(SaxpyInstantiator<torch::kCPU, torch::kFloat32>::get()),
                                 ExpectedFPtr>);

    // For half types
    static_assert(std::is_same_v<decltype(SaxpyInstantiator<torch::kCUDA, torch::kFloat16>::get()),
                                 ExpectedFPtr>);
    static_assert(std::is_same_v<decltype(SaxpyInstantiator<torch::kCUDA, torch::kBFloat16>::get()),
                                 ExpectedFPtr>);
}

} // namespace dispatch
} // namespace fvdb
