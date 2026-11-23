// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/FusedL1.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/gsplat/ImagePartition.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/types.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace fvdb {

namespace detail {

namespace ops {

namespace {

namespace cg = cooperative_groups;

// ------------------------------------------
// Block Dimensions (shared with FusedSSIM)
// ------------------------------------------
#define BLOCK_X kImageBlockWidth
#define BLOCK_Y kImageBlockHeight

// ------------------------------------------
// Forward Kernel: Fused L1 Loss
//  - Reduces each tile over all channels without materializing an L1 map.
//  - Divides by the full element count, including uneven final tiles.
//  - Uses the same NCHW storage and spatial tile ownership as SSIM.
// ------------------------------------------
__global__ void
fusedL1Kernel(int localToGlobalOffset,
              int B,
              int H,
              int W,
              int CH,
              const float *__restrict__ img1,
              const float *__restrict__ img2,
              float *__restrict__ partial_sums) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx        = globalGroupIndex.z;
    const int pix_y       = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x       = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int64_t pix_id  = static_cast<int64_t>(pix_y) * W + pix_x;
    const int64_t num_pix = static_cast<int64_t>(H) * W;

    const float normalization = 1.0f / (static_cast<int64_t>(B) * CH * num_pix);
    float loss                = 0.0f;
    if (pix_y < H && pix_x < W) {
        for (int c = 0; c < CH; ++c) {
            const int64_t global_idx = (static_cast<int64_t>(bIdx) * CH + c) * num_pix + pix_id;
            loss += fabsf(img1[global_idx] - img2[global_idx]) * normalization;
        }
    }

    // Every thread participates; padding contributes zero to the sum.
    using BlockReduce =
        cub::BlockReduce<float, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const float tile_loss = BlockReduce(reduction_storage).Sum(loss);
    if (block.thread_rank() == 0) {
        partial_sums[block.group_index().x] = tile_loss;
    }
}

// ------------------------------------------
// Backward Kernel: Fused L1
//  - Recomputes the sign of the residual.
//  - Writes gradients directly in the same
//    spatial tiles as the forward kernel.
// ------------------------------------------
__global__ void
fusedL1BackwardKernel(int localToGlobalOffset,
                      int B,
                      int H,
                      int W,
                      int CH,
                      const float *__restrict__ img1,
                      const float *__restrict__ img2,
                      const float *__restrict__ dL_dloss,
                      float *__restrict__ dL_dimg1,
                      float *__restrict__ dL_dimg2) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx        = globalGroupIndex.z;
    const int pix_y       = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x       = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int64_t pix_id  = static_cast<int64_t>(pix_y) * W + pix_x;
    const int64_t num_pix = static_cast<int64_t>(H) * W;

    if (pix_y < H && pix_x < W) {
        const float scale = *dL_dloss / static_cast<float>(static_cast<int64_t>(B) * CH * num_pix);
        for (int c = 0; c < CH; ++c) {
            const int64_t global_idx = (static_cast<int64_t>(bIdx) * CH + c) * num_pix + pix_id;
            const float difference   = img1[global_idx] - img2[global_idx];
            // Match PyTorch's real sign, including zero and NaN residuals.
            const float sign     = static_cast<float>((0.0f < difference) - (difference < 0.0f));
            const float gradient = scale * sign;
            if (dL_dimg1) {
                dL_dimg1[global_idx] = gradient;
            }
            if (dL_dimg2) {
                dL_dimg2[global_idx] = -gradient;
            }
        }
    }
}

void
checkL1Images(const torch::Tensor &img1, const torch::Tensor &img2) {
    TORCH_CHECK_VALUE(img1.dim() == 4 && img2.dim() == 4, "Fused L1 expects NCHW image batches");
    TORCH_CHECK_VALUE(img1.sizes() == img2.sizes(), "Fused L1 requires identical image shapes");
    TORCH_CHECK_VALUE(img1.device() == img2.device(),
                      "Fused L1 requires images on the same device");
    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat && img2.scalar_type() == torch::kFloat,
                      "Fused L1 only supports float32 images");
    TORCH_CHECK_VALUE(img1.is_contiguous() && img2.is_contiguous(),
                      "Fused L1 expects contiguous NCHW storage");
    for (const auto size: img1.sizes()) {
        TORCH_CHECK_VALUE(size <= std::numeric_limits<int>::max() - BLOCK_Y,
                          "Fused L1 image dimension is too large");
    }
    if (!img1.numel()) {
        return;
    }
    const int64_t tile_count = img1.size(0) * ((img1.size(2) + BLOCK_Y - 1) / BLOCK_Y) *
                               ((img1.size(3) + BLOCK_X - 1) / BLOCK_X);
    TORCH_CHECK_VALUE(tile_count <= std::numeric_limits<int>::max(),
                      "Fused L1 image tile count is too large");
}

void
checkL1LossGradient(const torch::Tensor &img1, const torch::Tensor &dL_dloss) {
    TORCH_CHECK_VALUE(dL_dloss.dim() == 0, "Fused L1 expects a scalar loss gradient");
    TORCH_CHECK_VALUE(dL_dloss.device() == img1.device() &&
                          dL_dloss.scalar_type() == img1.scalar_type(),
                      "Fused L1 loss gradient must have the images' device and dtype");
}

void
forwardL1Tiles(const torch::Tensor &img1,
               const torch::Tensor &img2,
               int blockOffset,
               int blockCount,
               float *partial_sums,
               cudaStream_t stream) {
    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedL1Kernel<<<grid, block, 0, stream>>>(blockOffset,
                                              img1.size(0),
                                              img1.size(2),
                                              img1.size(3),
                                              img1.size(1),
                                              img1.const_data_ptr<float>(),
                                              img2.const_data_ptr<float>(),
                                              partial_sums);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns the mean L1 loss over B, CH, H, W.
// ------------------------------------------
torch::Tensor
fusedL1CUDA(const torch::Tensor &img1, const torch::Tensor &img2) {
    checkL1Images(img1, img2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const auto stream = at::cuda::getCurrentCUDAStream(img1.device().index());
    const int B       = img1.size(0);
    const int H       = img1.size(2);
    const int W       = img1.size(3);

    if (!img1.numel()) {
        return torch::empty({}, img1.options()).fill_(std::numeric_limits<float>::quiet_NaN());
    }
    const int blockCount = ((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B;
    auto loss            = torch::empty({}, img1.options());
    auto partial_sums    = torch::empty({blockCount}, img1.options());
    forwardL1Tiles(img1, img2, 0, blockCount, partial_sums.data_ptr<float>(), stream);
    FVDB_CUB_WRAPPER(cub::DeviceReduce::Sum,
                     partial_sums.data_ptr<float>(),
                     loss.data_ptr<float>(),
                     blockCount,
                     stream);
    return loss;
}

// ------------------------------------------
// PyTorch Interface (Backward)
//   Returns the requested image gradients.
// ------------------------------------------
std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>>
fusedL1BackwardCUDA(const torch::Tensor &img1,
                    const torch::Tensor &img2,
                    const torch::Tensor &dL_dloss,
                    bool need_img1,
                    bool need_img2) {
    checkL1Images(img1, img2);
    checkL1LossGradient(img1, dL_dloss);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const auto stream = at::cuda::getCurrentCUDAStream(img1.device().index());
    int B             = img1.size(0);
    int CH            = img1.size(1);
    int H             = img1.size(2);
    int W             = img1.size(3);

    auto dL_dimg1 = need_img1 ? std::make_optional(torch::empty_like(img1)) : std::nullopt;
    auto dL_dimg2 = need_img2 ? std::make_optional(torch::empty_like(img2)) : std::nullopt;
    if (img1.numel() && (need_img1 || need_img2)) {
        dim3 grid(((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B);
        dim3 block(BLOCK_X, BLOCK_Y);
        fusedL1BackwardKernel<<<grid, block, 0, stream>>>(
            0,
            B,
            H,
            W,
            CH,
            img1.const_data_ptr<float>(),
            img2.const_data_ptr<float>(),
            dL_dloss.const_data_ptr<float>(),
            dL_dimg1 ? dL_dimg1->data_ptr<float>() : nullptr,
            dL_dimg2 ? dL_dimg2->data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(dL_dimg1, dL_dimg2);
}

// ------------------------------------------
// PyTorch Interface (Multi-GPU Forward)
//   Reduces the L1 loss using the same spatial rows as SSIM.
//   Only the scalar partial sums cross devices.
// ------------------------------------------
torch::Tensor
fusedL1PrivateUse1(const torch::Tensor &img1, const torch::Tensor &img2) {
    checkL1Images(img1, img2);
    const int B = img1.size(0);
    const int H = img1.size(2);
    const int W = img1.size(3);

    if (!img1.numel()) {
        return torch::empty({}, img1.options()).fill_(std::numeric_limits<float>::quiet_NaN());
    }
    auto loss = torch::empty({}, img1.options());

    prefetchImages({img1, img2}, {});
    std::vector<torch::Tensor> partial_losses;
    partial_losses.reserve(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        const auto chunk    = imageBlockChunk(B, H, W, deviceId);
        float *partial_loss = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&partial_loss, sizeof(float), stream));
        // The CUDA device is only metadata for NCCL; from_blob does not allocate storage.
        partial_losses.push_back(
            torch::from_blob(partial_loss, {1}, img1.options().device(torch::kCUDA, deviceId)));
        if (!chunk.blockCount) {
            C10_CUDA_CHECK(cudaMemsetAsync(partial_loss, 0, sizeof(float), stream));
            continue;
        }
        float *partial_sums = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&partial_sums, chunk.blockCount * sizeof(float), stream));
        forwardL1Tiles(img1, img2, chunk.blockOffset, chunk.blockCount, partial_sums, stream);
        FVDB_CUB_WRAPPER_ASYNC(
            stream, cub::DeviceReduce::Sum, partial_sums, partial_loss, chunk.blockCount);
        C10_CUDA_CHECK(cudaFreeAsync(partial_sums, stream));
    }
    auto nccl_loss =
        torch::from_blob(loss.data_ptr<float>(), {1}, partial_losses.front().options());
    torch::cuda::nccl::reduce(partial_losses, nccl_loss);
    // Release each scalar on its producing stream after NCCL has consumed it.
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaFreeAsync(partial_losses[deviceId].data_ptr<float>(), stream));
    }
    mergeStreams();
    return loss;
}

// ------------------------------------------
// PyTorch Interface (Multi-GPU Backward)
//   Prefetches and writes image gradients
//   using SSIM's spatial tile ownership.
// ------------------------------------------
std::tuple<std::optional<torch::Tensor>, std::optional<torch::Tensor>>
fusedL1BackwardPrivateUse1(const torch::Tensor &img1,
                           const torch::Tensor &img2,
                           const torch::Tensor &dL_dloss,
                           bool need_img1,
                           bool need_img2) {
    checkL1Images(img1, img2);
    checkL1LossGradient(img1, dL_dloss);
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    auto dL_dimg1 = need_img1 ? std::make_optional(torch::empty_like(img1)) : std::nullopt;
    auto dL_dimg2 = need_img2 ? std::make_optional(torch::empty_like(img2)) : std::nullopt;
    if (!img1.numel() || (!need_img1 && !need_img2)) {
        return std::make_tuple(dL_dimg1, dL_dimg2);
    }
    std::vector<torch::Tensor> inputs = {img1, img2};
    std::vector<torch::Tensor> outputs;
    if (dL_dimg1) {
        outputs.emplace_back(*dL_dimg1);
    }
    if (dL_dimg2) {
        outputs.emplace_back(*dL_dimg2);
    }
    prefetchImages(inputs, outputs);
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        if (chunk.blockCount) {
            dim3 grid(chunk.blockCount);
            dim3 block(BLOCK_X, BLOCK_Y);
            fusedL1BackwardKernel<<<grid, block, 0, stream>>>(
                chunk.blockOffset,
                B,
                H,
                W,
                CH,
                img1.const_data_ptr<float>(),
                img2.const_data_ptr<float>(),
                dL_dloss.const_data_ptr<float>(),
                dL_dimg1 ? dL_dimg1->data_ptr<float>() : nullptr,
                dL_dimg2 ? dL_dimg2->data_ptr<float>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    mergeStreams();
    return std::make_tuple(dL_dimg1, dL_dimg2);
}

} // namespace ops

} // namespace detail

} // namespace fvdb
