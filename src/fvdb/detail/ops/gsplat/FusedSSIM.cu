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
#include <fvdb/detail/ops/gsplat/FusedSSIM.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cooperative_groups.h>

#include <algorithm>

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
#define BLOCK_X 16
#define BLOCK_Y 16
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
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void
fusedSSIMKernel(int localToGlobalOffset,
                int B,
                int H,
                int W,
                int CH,
                float C1,
                float C2,
                const float *__restrict__ img1,
                const float *__restrict__ img2,
                float *__restrict__ ssim_map,
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

    // Each block processes B x C sub-batches. We loop over channels:
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

                int global_idx       = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

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
}

// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
//    and dL/dmap (the gradient from above).
// ------------------------------------------
__global__ void
fusedSSIMBackwardKernel(int localToGlobalOffset,
                        int B,
                        int H,
                        int W,
                        int CH,
                        float C1,
                        float C2,
                        const float *__restrict__ img1,
                        const float *__restrict__ img2,
                        const float *__restrict__ dL_dmap,
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

                    float chain = getPixelValue(dL_dmap, bIdx, c, gy, gx, CH, H, W);
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

            // final accumulation
            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2)*sum2;

            int out_idx       = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }
        block.sync();
    }
}

} // namespace

// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedSSIMCUDA(
    double C1, double C2, const torch::Tensor &img1, const torch::Tensor &img2, bool train) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const auto stream = at::cuda::getCurrentCUDAStream(img1.device().index());
    int B             = img1.size(0);
    int CH            = img1.size(1);
    int H             = img1.size(2);
    int W             = img1.size(3);

    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img2.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");

    // Launch config
    dim3 grid(((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Output SSIM map
    auto ssim_map = torch::empty_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    fusedSSIMKernel<<<grid, block, 0, stream>>>(0,
                                                B,
                                                H,
                                                W,
                                                CH,
                                                static_cast<float>(C1),
                                                static_cast<float>(C2),
                                                img1.contiguous().const_data_ptr<float>(),
                                                img2.contiguous().const_data_ptr<float>(),
                                                ssim_map.data_ptr<float>(),
                                                train ? dm_dmu1.data_ptr<float>() : nullptr,
                                                train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
                                                train ? dm_dsigma12.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
torch::Tensor
fusedSSIMBackwardCUDA(double C1,
                      double C2,
                      const torch::Tensor &img1,
                      const torch::Tensor &img2,
                      const torch::Tensor &dL_dmap,
                      const torch::Tensor &dm_dmu1,
                      const torch::Tensor &dm_dsigma1_sq,
                      const torch::Tensor &dm_dsigma12) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const auto stream = at::cuda::getCurrentCUDAStream(img1.device().index());
    int B             = img1.size(0);
    int CH            = img1.size(1);
    int H             = img1.size(2);
    int W             = img1.size(3);

    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img2.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");

    auto dL_dimg1 = torch::empty_like(img1);

    dim3 grid(((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedSSIMBackwardKernel<<<grid, block, 0, stream>>>(
        0,
        B,
        H,
        W,
        CH,
        static_cast<float>(C1),
        static_cast<float>(C2),
        img1.contiguous().const_data_ptr<float>(),
        img2.contiguous().const_data_ptr<float>(),
        dL_dmap.contiguous().const_data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().const_data_ptr<float>(),
        dm_dsigma1_sq.contiguous().const_data_ptr<float>(),
        dm_dsigma12.contiguous().const_data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return dL_dimg1;
}

// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedSSIMPrivateUse1(
    double C1, double C2, const torch::Tensor &img1, const torch::Tensor &img2, bool train) {
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img2.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");

    // Output SSIM map
    auto ssim_map = torch::empty_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    auto img1_ = img1.contiguous();
    auto img2_ = img2.contiguous();

    const auto globalBlockCount = ((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B;
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        constexpr size_t kAlignment = kPageSize / (sizeof(float) * BLOCK_X * BLOCK_Y);
        int localBlockOffset, localBlockCount;
        std::tie(localBlockOffset, localBlockCount) =
            alignedChunk(kAlignment, globalBlockCount, deviceId);

        if (localBlockCount) {
            // auto localElementOffset = localBlockOffset * BLOCK_X * BLOCK_Y * CH;
            // auto localElementCount  = localBlockCount * BLOCK_X * BLOCK_Y * CH;
            // if (localElementOffset + localElementCount > img1_.numel()) {
            //     localElementOffset = std::min(localElementOffset, static_cast<int>(img1_.numel()));
            //     localElementCount  = std::min(localElementCount,
            //                                  static_cast<int>(img1_.numel()) - localElementOffset);
            // }
            // nanovdb::util::cuda::memPrefetchAsync(img1_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(img2_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);

            // Launch config
            dim3 grid(localBlockCount);
            dim3 block(BLOCK_X, BLOCK_Y);

            fusedSSIMKernel<<<grid, block, 0, stream>>>(
                localBlockOffset,
                B,
                H,
                W,
                CH,
                static_cast<float>(C1),
                static_cast<float>(C2),
                img1_.const_data_ptr<float>(),
                img2_.const_data_ptr<float>(),
                ssim_map.data_ptr<float>(),
                train ? dm_dmu1.data_ptr<float>() : nullptr,
                train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
                train ? dm_dsigma12.data_ptr<float>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
torch::Tensor
fusedSSIMBackwardPrivateUse1(double C1,
                             double C2,
                             const torch::Tensor &img1,
                             const torch::Tensor &img2,
                             const torch::Tensor &dL_dmap,
                             const torch::Tensor &dm_dmu1,
                             const torch::Tensor &dm_dsigma1_sq,
                             const torch::Tensor &dm_dsigma12) {
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img2.scalar_type() == torch::kFloat,
                      "Fused SSIM only supports float32 images");

    auto dL_dimg1 = torch::empty_like(img1);

    auto img1_ = img1.contiguous();
    auto img2_ = img2.contiguous();

    auto dL_dmap_       = dL_dmap.contiguous();
    auto dm_dmu1_       = dm_dmu1.contiguous();
    auto dm_dsigma1_sq_ = dm_dsigma1_sq.contiguous();
    auto dm_dsigma12_   = dm_dsigma12.contiguous();

    const auto globalBlockCount = ((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y) * B;
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        constexpr size_t kAlignment = kPageSize / (sizeof(float) * BLOCK_X * BLOCK_Y);
        int localBlockOffset, localBlockCount;
        std::tie(localBlockOffset, localBlockCount) =
            alignedChunk(kAlignment, globalBlockCount, deviceId);

        if (localBlockCount) {
            // auto localElementOffset = localBlockOffset * BLOCK_X * BLOCK_Y * CH;
            // auto localElementCount  = localBlockCount * BLOCK_X * BLOCK_Y * CH;
            // if (localElementOffset + localElementCount > img1_.numel()) {
            //     localElementOffset = std::min(localElementOffset, static_cast<int>(img1_.numel()));
            //     localElementCount  = std::min(localElementCount,
            //                                  static_cast<int>(img1_.numel()) - localElementOffset);
            // }
            // nanovdb::util::cuda::memPrefetchAsync(img1_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(img2_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(dL_dmap_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(dm_dmu1_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(dm_dsigma1_sq_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);
            // nanovdb::util::cuda::memPrefetchAsync(dm_dsigma12_.const_data_ptr<float>() +
            //                                           localElementOffset,
            //                                       localElementCount * sizeof(float),
            //                                       deviceId,
            //                                       stream);

            // Launch config
            dim3 grid(localBlockCount);
            dim3 block(BLOCK_X, BLOCK_Y);

            fusedSSIMBackwardKernel<<<grid, block, 0, stream>>>(
                localBlockOffset,
                B,
                H,
                W,
                CH,
                static_cast<float>(C1),
                static_cast<float>(C2),
                img1_.const_data_ptr<float>(),
                img2_.const_data_ptr<float>(),
                dL_dmap_.const_data_ptr<float>(),
                dL_dimg1.data_ptr<float>(),
                dm_dmu1_.const_data_ptr<float>(),
                dm_dsigma1_sq_.const_data_ptr<float>(),
                dm_dsigma12_.const_data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }

    return dL_dimg1;
}

} // namespace ops

} // namespace detail

} // namespace fvdb
