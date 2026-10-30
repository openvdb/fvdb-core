// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/CopyImageToDevice.h>
#include <fvdb/detail/utils/gsplat/ImagePartition.cuh>

#include <cstdint>
#include <limits>
#include <vector>

namespace fvdb::detail::ops {

namespace {

void
checkImage(const torch::Tensor &image) {
    TORCH_CHECK_VALUE(image.is_cpu(), "Image copy requires a CPU source");
    TORCH_CHECK_VALUE(image.layout() == torch::kStrided && image.dim() == 4 &&
                          image.is_contiguous(),
                      "Image copy expects contiguous NCHW storage");
    TORCH_CHECK_VALUE(!image.requires_grad(), "Image copy does not support source gradients");
    TORCH_CHECK_VALUE(image.numel(), "Image copy requires nonempty image batches");
    TORCH_CHECK_VALUE(image.numel() <= std::numeric_limits<int>::max(),
                      "Image copy batch is too large");
    for (const auto size: image.sizes()) {
        TORCH_CHECK_VALUE(size <= std::numeric_limits<int>::max() - kImageBlockHeight,
                          "Image copy dimension is too large");
    }
}

} // namespace

torch::Tensor
copyImageToDeviceCUDA(const torch::Tensor &image, torch::Device device) {
    checkImage(image);
    return image.to(device);
}

torch::Tensor
copyImageToDevicePrivateUse1(const torch::Tensor &image, torch::Device device) {
    checkImage(image);
    auto result  = torch::empty(image.sizes(), image.options().device(device));
    const int B  = image.size(0);
    const int CH = image.size(1);
    const int H  = image.size(2);
    const int W  = image.size(3);

    prefetchImages({}, {result});

    const auto *sourceData = static_cast<const uint8_t *>(image.const_data_ptr());
    auto *targetData       = static_cast<uint8_t *>(result.data_ptr());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream      = c10::cuda::getCurrentCUDAStream(deviceId);
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        std::vector<void *> pointers;
        std::vector<size_t> sizes;
        appendImagePrefetchRanges(
            pointers, sizes, {result}, chunk.tileRowOffset, chunk.tileRowCount, B, CH, H, W);
        // Copy full-width tile rows using the same ownership as the image kernels.
        for (const auto rangeId: c10::irange(pointers.size())) {
            const auto offset = static_cast<uint8_t *>(pointers[rangeId]) - targetData;
            C10_CUDA_CHECK(cudaMemcpyAsync(
                pointers[rangeId], sourceData + offset, sizes[rangeId], cudaMemcpyDefault, stream));
        }
    }
    // Consumers may read halo rows uploaded by neighboring GPUs.
    mergeStreams();
    return result;
}

} // namespace fvdb::detail::ops
