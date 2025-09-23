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
#ifndef FVDB_DETAIL_OPS_GSPLAT_FUSEDSSIM_H
#define FVDB_DETAIL_OPS_GSPLAT_FUSEDSSIM_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fusedSSIMCUDA(
    double C1, double C2, const torch::Tensor &img1, const torch::Tensor &img2, bool train);

torch::Tensor fusedSSIMBackwardCUDA(double C1,
                                    double C2,
                                    const torch::Tensor &img1,
                                    const torch::Tensor &img2,
                                    const torch::Tensor &dL_dmap,
                                    const torch::Tensor &dm_dmu1,
                                    const torch::Tensor &dm_dsigma1_sq,
                                    const torch::Tensor &dm_dsigma12);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fusedSSIMPrivateUse1(
    double C1, double C2, const torch::Tensor &img1, const torch::Tensor &img2, bool train);

torch::Tensor fusedSSIMBackwardPrivateUse1(double C1,
                                           double C2,
                                           const torch::Tensor &img1,
                                           const torch::Tensor &img2,
                                           const torch::Tensor &dL_dmap,
                                           const torch::Tensor &dm_dmu1,
                                           const torch::Tensor &dm_dsigma1_sq,
                                           const torch::Tensor &dm_dsigma12);

} // namespace ops

} // namespace detail

} // namespace fvdb

#endif
