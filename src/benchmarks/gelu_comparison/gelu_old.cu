// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Old-style GELU implementation using fvdb's ForEach utilities.
// This demonstrates the traditional approach for comparison with the new dispatch framework.
//

#include "gelu_old.cuh"

#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <numbers>

namespace gelu_comparison {

// ============================================================================
// GELU scalar computation (same as dispatch_examples::gelu_scalar)
// ============================================================================

template <typename T>
__host__ __device__ inline T
gelu_scalar_impl(T x) {
    constexpr T sqrt2 = T(1.4142135623730951);
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, float>) {
        return T(0.5) * x * (T(1.0) + ::erff(x / sqrt2));
    } else {
        return T(0.5) * x * (T(1.0) + ::erf(x / sqrt2));
    }
#else
    return T(0.5) * x * (T(1.0) + std::erf(x / sqrt2));
#endif
}

// Specializations for half types
__host__ __device__ inline at::Half
gelu_scalar_impl(at::Half x) {
    return at::Half(gelu_scalar_impl(static_cast<float>(x)));
}

__host__ __device__ inline at::BFloat16
gelu_scalar_impl(at::BFloat16 x) {
    return at::BFloat16(gelu_scalar_impl(static_cast<float>(x)));
}

// ============================================================================
// Old-style GELU functor for ForEach
// ============================================================================
// The old ForEach pattern requires a functor with signature:
//   (int64_t elementIdx, int64_t channelIdx, accessor, ...args)
// This is less ergonomic than the new dispatch framework.

template <typename ScalarT> struct GeluOldFunctor {
    using InAcc  = torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits>;
    using OutAcc = torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits>;

    __host__ __device__ void
    operator()(int64_t elementIdx, int64_t /*channelIdx*/, InAcc inAcc, OutAcc outAcc) const {
        outAcc[elementIdx] = gelu_scalar_impl(inAcc[elementIdx]);
    }
};

// In-place version
template <typename ScalarT> struct GeluOldInPlaceFunctor {
    using Acc = torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits>;

    __host__ __device__ void
    operator()(int64_t elementIdx, int64_t /*channelIdx*/, Acc acc) const {
        acc[elementIdx] = gelu_scalar_impl(acc[elementIdx]);
    }
};

// ============================================================================
// CUDA implementation
// ============================================================================

template <typename ScalarT>
void
gelu_old_cuda_impl(torch::Tensor input, torch::Tensor output) {
    // Note: forEachTensorElementChannelCUDA automatically creates an accessor from `input`
    // and passes it to the functor as the first accessor argument. We pass `outAcc` as an
    // extra arg so the functor receives (elementIdx, channelIdx, inputAcc, outAcc).
    auto outAcc = output.packed_accessor32<ScalarT, 1, torch::RestrictPtrTraits>();

    fvdb::forEachTensorElementChannelCUDA<ScalarT, 1>(1024,  // numThreads
                                                      1,     // numChannels (treating as 1D)
                                                      input, // tensor to iterate over
                                                      GeluOldFunctor<ScalarT>{}, // functor
                                                      outAcc // extra arg passed to functor
    );
}

template <typename ScalarT>
void
gelu_old_cuda_inplace_impl(torch::Tensor tensor) {
    // Note: forEachTensorElementChannelCUDA automatically creates an accessor from the tensor
    // and passes it to the functor. We don't need to pass an extra accessor for in-place ops.
    fvdb::forEachTensorElementChannelCUDA<ScalarT, 1>(
        1024, 1, tensor, GeluOldInPlaceFunctor<ScalarT>{});
}

// ============================================================================
// CPU implementation
// ============================================================================

template <typename ScalarT>
void
gelu_old_cpu_impl(torch::Tensor input, torch::Tensor output) {
    auto inAcc  = input.accessor<ScalarT, 1>();
    auto outAcc = output.accessor<ScalarT, 1>();

    const int64_t n = input.size(0);
    for (int64_t i = 0; i < n; ++i) {
        outAcc[i] = gelu_scalar_impl(inAcc[i]);
    }
}

template <typename ScalarT>
void
gelu_old_cpu_inplace_impl(torch::Tensor tensor) {
    auto acc        = tensor.accessor<ScalarT, 1>();
    const int64_t n = tensor.size(0);
    for (int64_t i = 0; i < n; ++i) {
        acc[i] = gelu_scalar_impl(acc[i]);
    }
}

// ============================================================================
// Public API
// ============================================================================

torch::Tensor
gelu_old(torch::Tensor input) {
    // Old way requires explicit checks and manual dispatch
    TORCH_CHECK(input.is_contiguous(), "gelu_old requires contiguous input");
    TORCH_CHECK(input.dim() == 1, "gelu_old requires 1D input for this benchmark");

    if (input.numel() == 0) {
        return torch::empty_like(input);
    }

    auto output = torch::empty_like(input);

    // Old way: AT_DISPATCH_FLOATING_TYPES_AND2 for dtype dispatch
    // Then manual device dispatch
    if (input.is_cuda()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                        at::ScalarType::BFloat16,
                                        input.scalar_type(),
                                        "gelu_old_cuda",
                                        [&] { gelu_old_cuda_impl<scalar_t>(input, output); });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                        at::ScalarType::BFloat16,
                                        input.scalar_type(),
                                        "gelu_old_cpu",
                                        [&] { gelu_old_cpu_impl<scalar_t>(input, output); });
    }

    return output;
}

torch::Tensor
gelu_old_(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "gelu_old_ requires contiguous input");
    TORCH_CHECK(input.dim() == 1, "gelu_old_ requires 1D input for this benchmark");

    if (input.numel() == 0) {
        return input;
    }

    if (input.is_cuda()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                        at::ScalarType::BFloat16,
                                        input.scalar_type(),
                                        "gelu_old_cuda_inplace",
                                        [&] { gelu_old_cuda_inplace_impl<scalar_t>(input); });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                        at::ScalarType::BFloat16,
                                        input.scalar_type(),
                                        "gelu_old_cpu_inplace",
                                        [&] { gelu_old_cpu_inplace_impl<scalar_t>(input); });
    }

    return input;
}

} // namespace gelu_comparison
