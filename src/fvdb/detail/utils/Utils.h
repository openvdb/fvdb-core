// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_UTILS_H
#define FVDB_DETAIL_UTILS_UTILS_H

#include <fvdb/TypeTraits.h>
#include <fvdb/detail/utils/nanovdb/ActiveVoxelIterator.h>
#include <fvdb/detail/utils/nanovdb/HDDAIterators.h>
#include <fvdb/detail/utils/nanovdb/TorchNanoConversions.h>

#include <ATen/Dispatch_v2.h>
#include <torch/types.h>
#include <torch/version.h>

#include <iostream>
#include <memory>
#include <string>

// A bunch of things defined to make intellisense work with nvcc
#if defined(NDEVELOP_IDE_ONLY)
namespace torch {
template <typename T> struct RestrictPtrTraits {
    typedef T *__restrict__ PtrType;
};
} // namespace torch
#endif

/// @brief Given a torch::Device, define DeviceTag to torch::kCPU or torch::kCUDA.
///        This macro calls the passed in function with the typedef DeviceTag to the correct device
///        tag.
#define FVDB_DISPATCH_KERNEL_DEVICE(DEVICE, ...)                           \
    [&]() {                                                                \
        if (DEVICE.is_cpu()) {                                             \
            static constexpr c10::DeviceType DeviceTag = torch::kCPU;      \
            return __VA_ARGS__();                                          \
        } else if (DEVICE.is_cuda()) {                                     \
            static constexpr c10::DeviceType DeviceTag = torch::kCUDA;     \
            return __VA_ARGS__();                                          \
        } else {                                                           \
            TORCH_CHECK(false, "Only CPU and CUDA devices are supported"); \
        }                                                                  \
    }()

/// @brief Given a torch::Device, define DeviceTag to torch::kCPU, torch::kCUDA, or
/// torch::kPrivateUse1.
///        This macro calls the passed in function with the typedef DeviceTag to the correct device
///        tag.
#define FVDB_DISPATCH_KERNEL(DEVICE, ...)                                                \
    [&]() {                                                                              \
        if (DEVICE.is_cpu()) {                                                           \
            static constexpr c10::DeviceType DeviceTag = torch::kCPU;                    \
            return __VA_ARGS__();                                                        \
        } else if (DEVICE.is_cuda()) {                                                   \
            static constexpr c10::DeviceType DeviceTag = torch::kCUDA;                   \
            return __VA_ARGS__();                                                        \
        } else if (DEVICE.is_privateuseone()) {                                          \
            static constexpr c10::DeviceType DeviceTag = torch::kPrivateUse1;            \
            return __VA_ARGS__();                                                        \
        } else {                                                                         \
            TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported"); \
        }                                                                                \
    }()

namespace fvdb {
namespace detail {

/// @brief Convert a 1d tensor of integer values into an std:vector<int64_t>
/// @param shapeTensor a 1D tensor of integer values
/// @return An std::vector<int64_t> with the same values as the input tensor
inline std::vector<int64_t>
intTensor1DToStdVector(torch::Tensor shapeTensor) {
    return AT_DISPATCH_V2(
        shapeTensor.scalar_type(),
        "tensorToShape",
        AT_WRAP([&]() {
            TORCH_CHECK(shapeTensor.dim() == 1, "shapeTensor must be a 1D tensor");
            TORCH_CHECK(!shapeTensor.is_floating_point(), "shapeTensor must be an integer tensor");
            auto acc = shapeTensor.accessor<scalar_t, 1>();
            std::vector<int64_t> outShape(acc.size(0));
            for (int64_t i = 0; i < acc.size(0); i += 1) {
                outShape[i] = (int64_t)acc[i];
            }
            return outShape;
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));
}

/// @brief Return an std::vector<int64_t> representing the shape of a tensor which is forned by
/// removing the first
///        N dimensions of the input tensor and replacing them with s0
///        For example lets' say inTensor has shape [X, Y, Z], then spliceShape({A, B}, inTensor, 2)
///        will return [A, B, Z] and spliceShape({A, B}, inTensor, 1) will return [A, B, Y, Z]
/// @param s0 The shape values to splice in
/// @param inTensor The tensor whose shape to splice
/// @param start How many dimensions to remove from the shape of inTensor
/// @return An std::vector<int64_t> representing the shape of the spliced tensor
inline std::vector<int64_t>
spliceShape(std::vector<int64_t> s0, const torch::Tensor &inTensor, int start = 1) {
    TORCH_CHECK(start >= 0 && start <= inTensor.dim(),
                "start must be in range [0, inTensor.dim()]");
    std::vector<int64_t> outSize(s0.size() + inTensor.dim() - start);
    for (size_t i = 0; i < s0.size(); i += 1) {
        outSize[i] = s0[i];
    }

    for (int64_t i = start; i < inTensor.dim(); i += 1) {
        outSize[i + s0.size() - start] = inTensor.size(i);
    }
    return outSize;
}

/// @brief Return a view of the input tensor with all but first ndim dimensions coalesced into a
/// single dimension
///        this is similar to inTensor.view({inTensor.size(0), ..., inTensor.size(ndim - 1), -1})
///        but it handles the case when inTensor.size(0)*...*inTensor.size(ndim - 1) == 0
/// @param inTensor The tensor to coalesce
/// @return A view of the input tensor with all but first dimensions coalesced into a single
/// dimension
inline torch::Tensor
featureCoalescedView(const torch::Tensor &inTensor, int64_t ndim = 1) {
    std::vector<int64_t> outSize;
    for (int64_t i = 0; i < ndim; ++i) {
        outSize.push_back(inTensor.size(i));
    }
    int64_t numOthers = 1;
    for (int64_t i = ndim; i < inTensor.dim(); ++i) {
        numOthers *= inTensor.size(i);
    }
    outSize.push_back(numOthers);
    torch::Tensor outTensor = inTensor.view(outSize);
    TORCH_CHECK(inTensor.storage().is_alias_of(outTensor.storage()), "output should be a view!");
    return outTensor;
}

/// @brief Return a view of the input tensor with dimensions between the first `ndim` and the last
///        `trailingdim` dimensions coalesced into a single middle dimension. The leading `ndim`
///        and trailing `trailingdim` dimensions are preserved.
///        For example, if inTensor has shape [A, B, C, D, E, F, G], ndim=1 and trailingdim=3,
///        the output shape is [A, (B*C*D), E, F, G].
///        When there are no middle dimensions to coalesce (i.e. ndim + trailingdim ==
///        inTensor.dim()), the shape is unchanged.
/// @param inTensor The tensor to reshape as a view
/// @param ndim The number of leading dimensions to preserve (may be 0)
/// @param trailingdim The number of trailing dimensions to preserve (may be 0)
/// @return A view of the input tensor with the middle block coalesced into a single dimension
inline torch::Tensor
featureCoalescedViewTrailing(const torch::Tensor &inTensor,
                             int64_t ndim        = 1,
                             int64_t trailingdim = 0) {
    TORCH_CHECK(ndim >= 0, "ndim must be non-negative");
    TORCH_CHECK(trailingdim >= 0, "trailingdim must be non-negative");
    TORCH_CHECK(ndim + trailingdim <= inTensor.dim(),
                "ndim + trailingdim must be <= inTensor.dim()");

    const int64_t totalDims       = inTensor.dim();
    const int64_t midStart        = ndim;
    const int64_t midEndExclusive = totalDims - trailingdim; // may equal midStart
    const int64_t midCount        = midEndExclusive - midStart;

    // Fast path: nothing to coalesce
    if (midCount == 0 || midCount == 1) {
        return inTensor;
    }

    std::vector<int64_t> outSize;
    outSize.reserve(ndim + 1 + trailingdim);

    // Preserve leading dimensions
    for (int64_t i = 0; i < ndim; ++i) {
        outSize.push_back(inTensor.size(i));
    }

    // Coalesce middle dimensions into one
    int64_t middleProduct = 1;
    for (int64_t i = midStart; i < midEndExclusive; ++i) {
        middleProduct *= inTensor.size(i);
    }
    outSize.push_back(middleProduct);

    // Preserve trailing dimensions
    for (int64_t i = midEndExclusive; i < totalDims; ++i) {
        outSize.push_back(inTensor.size(i));
    }

    torch::Tensor outTensor = inTensor.view(outSize);
    TORCH_CHECK(inTensor.storage().is_alias_of(outTensor.storage()), "output should be a view!");
    return outTensor;
}

/// @brief Convert a tensor of shape [B, 3] or [3] representing a batch of coordinates or a single
/// coordinate into a
///        tensor of shape [B, 3] (if the input has shape [B, 3], this is a no-op)
/// @param coordOrBatch A tensor of shape [B, 3] or [3]
/// @param batchSize The size of the batch
/// @return A tensor of shape [B, 3]
inline torch::Tensor
coordTensorToBatch(const torch::Tensor &coordOrBatch, int64_t batchSize) {
    if (coordOrBatch.dim() == 1) {
        TORCH_CHECK_VALUE(coordOrBatch.size(0) == 3,
                          "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                              std::to_string(coordOrBatch.size(0)) + ",]");
        return coordOrBatch.unsqueeze(0).repeat({batchSize, 1});
    } else {
        TORCH_CHECK_VALUE(coordOrBatch.dim() == 2,
                          "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                              std::to_string(coordOrBatch.size(0)) + ", " +
                              std::to_string(coordOrBatch.size(1)) + "]");
        TORCH_CHECK_VALUE(coordOrBatch.size(0) == batchSize,
                          "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                              std::to_string(coordOrBatch.size(0)) + ", " +
                              std::to_string(coordOrBatch.size(1)) + "]");
        TORCH_CHECK_VALUE(coordOrBatch.size(1) == 3,
                          "Expected coordOrBatch to have shape [3,] or [B, 3] but got shape = [" +
                              std::to_string(coordOrBatch.size(0)) + ", " +
                              std::to_string(coordOrBatch.size(1)) + "]");
        return coordOrBatch;
    }
}

} // namespace detail
} // namespace fvdb

// std::cout and std::cerr for shapes
inline std::ostream &
operator<<(std::ostream &os, at::IntArrayRef c) {
    os << "[";
    for (size_t i = 0; i < c.size(); i += 1) {
        os << c[i];
        if (i < c.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

#endif // FVDB_DETAIL_UTILS_UTILS_H
