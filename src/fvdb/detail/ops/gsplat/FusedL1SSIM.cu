// This file contains source code from the fused-ssim library obtained from
// https://github.com/rahul-goel/fused-ssim. The fused-ssim library is licensed under the MIT
// License. Refer to ORSB 5512107 for more. Original license text follows.

// Copyright (c) 2024 Rahul Goel
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/FusedL1SSIM.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/gsplat/ImagePartition.cuh>

#include <ATen/native/Lerp.h>
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
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float cGauss[11] = {0.001028380123898387f,
                                 0.0075987582094967365f,
                                 0.036000773310661316f,
                                 0.10936068743467331f,
                                 0.21300552785396576f,
                                 0.26601171493530273f,
                                 0.21300552785396576f,
                                 0.10936068743467331f,
                                 0.036000773310661316f,
                                 0.0075987582094967365f,
                                 0.001028380123898387f};

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X kImageBlockWidth
#define BLOCK_Y kImageBlockHeight
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float
getPixelValue(const float *img, int b, int c, int y, int x, int CH, int H, int W) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// Forward Kernel: Combined L1 and SSIM Loss
//  - Reuses SSIM's shared image tile for L1.
//  - Blends each pixel's losses and reduces over the tile and all channels.
//  - Saves SSIM partial derivatives for backward without materializing loss maps.
// ------------------------------------------
__global__ void
fusedL1SSIMKernel(int localToGlobalOffset,
                  int B,
                  int H,
                  int W,
                  int CH,
                  float ssim_weight,
                  float C1,
                  float C2,
                  const float *__restrict__ img1,
                  const float *__restrict__ img2,
                  float *__restrict__ loss_tiles,
                  float *__restrict__ dm_dmu1,
                  float *__restrict__ dm_dsigma1_sq,
                  float *__restrict__ dm_dsigma12) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx    = globalGroupIndex.z; // batch index
    const int pix_y   = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x   = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int pix_id  = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ float xconv[CONV_Y][CONV_X][5];

    const float normalization = 1.0f / (static_cast<int64_t>(B) * CH * num_pix);
    float loss                = 0.0f;

    // Each block owns one spatial tile and loops over its channels:
    for (int c = 0; c < CH; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads  = BLOCK_X * BLOCK_Y;
            const int steps    = (tileSize + threads - 1) / threads;

            const int tileStartY = globalGroupIndex.y * BLOCK_Y;
            const int tileStartX = globalGroupIndex.x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / SHARED_X;
                    int local_x = tid % SHARED_X;
                    int gy      = tileStartY + local_y - HALO;
                    int gx      = tileStartX + local_x - HALO;

                    float X = getPixelValue(img1, bIdx, c, gy, gx, CH, H, W);
                    float Y = getPixelValue(img2, bIdx, c, gy, gx, CH, H, W);

                    sTile[local_y][local_x][0] = X;
                    sTile[local_y][local_x][1] = Y;
                }
            }
        }
        block.sync();

        // Reuse the loaded center pixels for L1. Read them before the horizontal
        // pass's barrier, so the next channel cannot overwrite sTile too early.
        const float l1_value = fabsf(sTile[threadIdx.y + HALO][threadIdx.x + HALO][0] -
                                     sTile[threadIdx.y + HALO][threadIdx.x + HALO][1]);

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO; // skip left halo

            float sumX  = 0.f;
            float sumX2 = 0.f;
            float sumY  = 0.f;
            float sumY2 = 0.f;
            float sumXY = 0.f;

            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w      = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];

                sumX += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float wc      = cGauss[HALO];
                sumX += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
            }

            // Write out partial sums
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX  = 0.f;
                sumX2 = 0.f;
                sumY  = 0.f;
                sumY2 = 0.f;
                sumXY = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w      = cGauss[HALO - d];
                    float Xleft  = sTile[ly2][lx - d][0];
                    float Yleft  = sTile[ly2][lx - d][1];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];

                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w    = cGauss[HALO - d];
                float *top = xconv[ly - d][lx];
                float *bot = xconv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            // center
            {
                float wC   = cGauss[HALO];
                float *ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1    = out0;
                float mu2    = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1 * mu2;

                float A  = mu1_sq + mu2_sq + C1;
                float B  = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                loss += at::native::lerp(l1_value, 1.0f - val, ssim_weight) * normalization;

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;

                if (dm_dmu1) {
                    // partial derivatives
                    float d_m_dmu1 =
                        ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) -
                         (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }

    // Padding contributes zero, but every thread participates in the reduction.
    using BlockReduce =
        cub::BlockReduce<float, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const float tile_loss = BlockReduce(reduction_storage).Sum(loss);
    if (block.thread_rank() == 0) {
        loss_tiles[block.group_index().x] = tile_loss;
    }
}

// ------------------------------------------
// Backward Kernel: Combined Image Gradient
//  - Applies the normalized upstream scalar while loading SSIM derivatives.
//  - Adds the weighted L1 sign term after the SSIM convolution.
//  - Writes one image gradient without intermediate gradient maps.
// ------------------------------------------
__global__ void
fusedL1SSIMBackwardKernel(int localToGlobalOffset,
                          int B,
                          int H,
                          int W,
                          int CH,
                          float ssim_weight,
                          const float *__restrict__ img1,
                          const float *__restrict__ img2,
                          const float *__restrict__ grad_loss,
                          float *__restrict__ dL_dimg1,
                          const float *__restrict__ dm_dmu1,
                          const float *__restrict__ dm_dsigma1_sq,
                          const float *__restrict__ dm_dsigma12) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx    = globalGroupIndex.z; // batch index
    const int pix_y   = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x   = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int pix_id  = pix_y * W + pix_x;
    const int num_pix = H * W;

    // The mean supplies one scalar gradient to every valid SSIM-map pixel.
    const float scale = *grad_loss / static_cast<float>(static_cast<int64_t>(B) * CH * num_pix);

    // Shared memory for the fused data:
    // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
    __shared__ float sData[3][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = getPixelValue(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = getPixelValue(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // (1) Load + fuse multiplication
        {
            const int start_y = globalGroupIndex.y * BLOCK_Y;
            const int start_x = globalGroupIndex.x * BLOCK_X;

            int tid          = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id      = tid / 32;
            int lane_id      = tid % 32;
            int totalThreads = BLOCK_X * BLOCK_Y;
            int num_warps    = (totalThreads + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                int gy = start_y + row - HALO;
                for (int col = lane_id; col < SHARED_X; col += 32) {
                    int gx = start_x + col - HALO;

                    float chain = (gx >= 0 && gx < W && gy >= 0 && gy < H) ? scale : 0.0f;
                    float vmu   = getPixelValue(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                    float vs1   = getPixelValue(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                    float vs12  = getPixelValue(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);

                    sData[0][row][col] = vmu * chain;
                    sData[1][row][col] = vs1 * chain;
                    sData[2][row][col] = vs12 * chain;
                }
            }
        }
        block.sync();

        // (2) Horizontal pass
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w     = cGauss[HALO - d];
                        float left0 = sData[0][yy][lx - d];
                        float left1 = sData[1][yy][lx - d];
                        float left2 = sData[2][yy][lx - d];

                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    // center
                    {
                        float wc = cGauss[HALO];
                        float c0 = sData[0][yy][lx];
                        float c1 = sData[1][yy][lx];
                        float c2 = sData[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }

                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w    = cGauss[HALO - d];
                float *top = sScratch[ly - d][lx];
                float *bot = sScratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                float wc   = cGauss[HALO];
                float *ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            // Combine the SSIM derivative with L1 using the same center pixels.
            const float ssim_gradient = sum0 + (2.f * p1) * sum1 + p2 * sum2;
            const float difference    = p1 - p2;
            // Match the existing L1 subgradient for zero and NaN residuals.
            const float sign        = static_cast<float>((0.0f < difference) - (difference < 0.0f));
            const float l1_gradient = scale * sign;

            int out_idx       = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = (1.0f - ssim_weight) * l1_gradient - ssim_weight * ssim_gradient;
        }
        block.sync();
    }
}

void
checkImages(const torch::Tensor &img1, const torch::Tensor &img2) {
    TORCH_CHECK_VALUE(img1.dim() == 4 && img2.dim() == 4,
                      "Fused L1/SSIM expects NCHW image batches");
    TORCH_CHECK_VALUE(img1.sizes() == img2.sizes(),
                      "Fused L1/SSIM requires identical image shapes");
    TORCH_CHECK_VALUE(img1.device() == img2.device(),
                      "Fused L1/SSIM requires images on the same device");
    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat && img2.scalar_type() == torch::kFloat,
                      "Fused L1/SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img1.is_contiguous() && img2.is_contiguous(),
                      "Fused L1/SSIM expects contiguous NCHW storage");
    TORCH_CHECK_VALUE(img1.numel(), "Fused L1/SSIM requires nonempty image batches");
    // The image kernels index the complete image batch with signed ints.
    TORCH_CHECK_VALUE(img1.numel() <= std::numeric_limits<int>::max(),
                      "Fused L1/SSIM image batch is too large");
    for (const auto size: img1.sizes()) {
        TORCH_CHECK_VALUE(size <= std::numeric_limits<int>::max() - BLOCK_Y,
                          "Fused L1/SSIM image dimension is too large");
    }
}

void
checkBackwardInputs(const torch::Tensor &img1,
                    const torch::Tensor &grad_loss,
                    const torch::TensorList &derivatives) {
    TORCH_CHECK_VALUE(
        grad_loss.dim() == 0 && grad_loss.device() == img1.device() &&
            grad_loss.scalar_type() == img1.scalar_type(),
        "Fused L1/SSIM requires a scalar loss gradient with the images' device and dtype");
    for (const auto &derivative: derivatives) {
        TORCH_CHECK_VALUE(
            derivative.sizes() == img1.sizes() && derivative.device() == img1.device() &&
                derivative.scalar_type() == img1.scalar_type() && derivative.is_contiguous(),
            "Fused L1/SSIM requires contiguous saved derivatives matching the images");
    }
}

// Prefetch all images and intermediates once per pass, on the same tile rows.
std::vector<cudaEvent_t>
prefetchImages(const torch::TensorList &images) {
    const auto &image = images.front();
    const int B       = image.size(0);
    const int CH      = image.size(1);
    const int H       = image.size(2);
    const int W       = image.size(3);
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getStreamFromPool(false, deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        if (chunk.blockCount) {
            std::vector<void *> pointers;
            std::vector<size_t> sizes;
            appendImagePrefetchRanges(
                pointers, sizes, images, chunk.tileRowOffset, chunk.tileRowCount, B, CH, H, W);
            memPrefetchBatchAsync(pointers, sizes, deviceId, stream);
        }
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }
    return events;
}

void
forwardTiles(const torch::Tensor &img1,
             const torch::Tensor &img2,
             const torch::Tensor &dm_dmu1,
             const torch::Tensor &dm_dsigma1_sq,
             const torch::Tensor &dm_dsigma12,
             float ssim_weight,
             float C1,
             float C2,
             int blockOffset,
             int blockCount,
             float *loss,
             cudaStream_t stream) {
    if (!blockCount) {
        C10_CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), stream));
        return;
    }
    const int B  = img1.size(0);
    const int CH = img1.size(1);
    const int H  = img1.size(2);
    const int W  = img1.size(3);
    // Keep tile contributions on the GPU that produced them.
    auto tile_losses = torch::empty(
        {blockCount}, img1.options().device(torch::kCUDA, c10::cuda::current_device()));
    auto *loss_tiles = tile_losses.data_ptr<float>();

    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedL1SSIMKernel<<<grid, block, 0, stream>>>(
        blockOffset,
        B,
        H,
        W,
        CH,
        ssim_weight,
        C1,
        C2,
        img1.const_data_ptr<float>(),
        img2.const_data_ptr<float>(),
        loss_tiles,
        dm_dmu1.numel() ? dm_dmu1.data_ptr<float>() : nullptr,
        dm_dsigma1_sq.numel() ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        dm_dsigma12.numel() ? dm_dsigma12.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    FVDB_CUB_WRAPPER_ASYNC(stream, cub::DeviceReduce::Sum, loss_tiles, loss, blockCount);
}

void
backwardTiles(const torch::Tensor &img1,
              const torch::Tensor &img2,
              const torch::Tensor &grad_loss,
              const torch::Tensor &dm_dmu1,
              const torch::Tensor &dm_dsigma1_sq,
              const torch::Tensor &dm_dsigma12,
              const torch::Tensor &grad_img1,
              float ssim_weight,
              int blockOffset,
              int blockCount,
              cudaStream_t stream) {
    if (!blockCount) {
        return;
    }
    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedL1SSIMBackwardKernel<<<grid, block, 0, stream>>>(blockOffset,
                                                          img1.size(0),
                                                          img1.size(2),
                                                          img1.size(3),
                                                          img1.size(1),
                                                          ssim_weight,
                                                          img1.const_data_ptr<float>(),
                                                          img2.const_data_ptr<float>(),
                                                          grad_loss.const_data_ptr<float>(),
                                                          grad_img1.data_ptr<float>(),
                                                          dm_dmu1.const_data_ptr<float>(),
                                                          dm_dsigma1_sq.const_data_ptr<float>(),
                                                          dm_dsigma12.const_data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// ------------------------------------------
// PyTorch Interfaces (Forward)
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMCUDA(const torch::Tensor &img1,
                const torch::Tensor &img2,
                double ssim_weight,
                double C1,
                double C2,
                bool train) {
    checkImages(img1, img2);
    const c10::cuda::CUDAGuard device_guard(img1.device().index());
    const int B        = img1.size(0);
    const int H        = img1.size(2);
    const int W        = img1.size(3);
    auto loss          = torch::empty({}, img1.options());
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    auto stream          = c10::cuda::getCurrentCUDAStream(img1.device().index());
    const int blockCount = B * ((H + BLOCK_Y - 1) / BLOCK_Y) * ((W + BLOCK_X - 1) / BLOCK_X);
    forwardTiles(img1,
                 img2,
                 dm_dmu1,
                 dm_dsigma1_sq,
                 dm_dsigma12,
                 static_cast<float>(ssim_weight),
                 static_cast<float>(C1),
                 static_cast<float>(C2),
                 0,
                 blockCount,
                 loss.data_ptr<float>(),
                 stream);
    return std::make_tuple(loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMPrivateUse1(const torch::Tensor &img1,
                       const torch::Tensor &img2,
                       double ssim_weight,
                       double C1,
                       double C2,
                       bool train) {
    checkImages(img1, img2);
    const int B        = img1.size(0);
    const int H        = img1.size(2);
    const int W        = img1.size(3);
    auto loss          = torch::empty({}, img1.options());
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    std::vector<torch::Tensor> images = {img1, img2};
    if (train) {
        images.insert(images.end(), {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    }
    auto events = prefetchImages(images);
    std::vector<torch::Tensor> partial_losses;
    partial_losses.reserve(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        partial_losses.push_back(torch::empty({1}, img1.options().device(torch::kCUDA, deviceId)));
        forwardTiles(img1,
                     img2,
                     dm_dmu1,
                     dm_dsigma1_sq,
                     dm_dsigma12,
                     static_cast<float>(ssim_weight),
                     static_cast<float>(C1),
                     static_cast<float>(C2),
                     chunk.blockOffset,
                     chunk.blockCount,
                     partial_losses.back().data_ptr<float>(),
                     stream);
    }
    auto nccl_loss =
        torch::from_blob(loss.data_ptr<float>(), {1}, partial_losses.front().options());
    torch::cuda::nccl::reduce(partial_losses, nccl_loss);
    mergeStreams();
    return std::make_tuple(loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// PyTorch Interfaces (Backward)
// ------------------------------------------
torch::Tensor
fusedL1SSIMBackwardCUDA(const torch::Tensor &img1,
                        const torch::Tensor &img2,
                        const torch::Tensor &grad_loss,
                        const torch::Tensor &dm_dmu1,
                        const torch::Tensor &dm_dsigma1_sq,
                        const torch::Tensor &dm_dsigma12,
                        double ssim_weight,
                        double C1,
                        double C2) {
    checkImages(img1, img2);
    checkBackwardInputs(img1, grad_loss, {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    const c10::cuda::CUDAGuard device_guard(img1.device().index());
    const int B    = img1.size(0);
    const int H    = img1.size(2);
    const int W    = img1.size(3);
    auto grad_img1 = torch::empty_like(img1);

    auto stream          = c10::cuda::getCurrentCUDAStream(img1.device().index());
    const int blockCount = B * ((H + BLOCK_Y - 1) / BLOCK_Y) * ((W + BLOCK_X - 1) / BLOCK_X);
    backwardTiles(img1,
                  img2,
                  grad_loss,
                  dm_dmu1,
                  dm_dsigma1_sq,
                  dm_dsigma12,
                  grad_img1,
                  static_cast<float>(ssim_weight),
                  0,
                  blockCount,
                  stream);
    return grad_img1;
}

torch::Tensor
fusedL1SSIMBackwardPrivateUse1(const torch::Tensor &img1,
                               const torch::Tensor &img2,
                               const torch::Tensor &grad_loss,
                               const torch::Tensor &dm_dmu1,
                               const torch::Tensor &dm_dsigma1_sq,
                               const torch::Tensor &dm_dsigma12,
                               double ssim_weight,
                               double C1,
                               double C2) {
    checkImages(img1, img2);
    checkBackwardInputs(img1, grad_loss, {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    const int B    = img1.size(0);
    const int H    = img1.size(2);
    const int W    = img1.size(3);
    auto grad_img1 = torch::empty_like(img1);

    auto events = prefetchImages({img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, grad_img1});
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        backwardTiles(img1,
                      img2,
                      grad_loss,
                      dm_dmu1,
                      dm_dsigma1_sq,
                      dm_dsigma12,
                      grad_img1,
                      static_cast<float>(ssim_weight),
                      chunk.blockOffset,
                      chunk.blockCount,
                      stream);
    }
    mergeStreams();
    return grad_img1;
}

} // namespace ops

} // namespace detail

} // namespace fvdb
