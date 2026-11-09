// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/FusedL1SSIM.h>

#include <torch/library.h>

TORCH_LIBRARY_IMPL(fvdb, CUDA, m) {
    m.impl("_fused_l1_ssim", &fvdb::detail::ops::fusedL1SSIMCUDA);
    m.impl("_fused_l1_ssim_backward", &fvdb::detail::ops::fusedL1SSIMBackwardCUDA);
}

TORCH_LIBRARY_IMPL(fvdb, PrivateUse1, m) {
    m.impl("_fused_l1_ssim", &fvdb::detail::ops::fusedL1SSIMPrivateUse1);
    m.impl("_fused_l1_ssim_backward", &fvdb::detail::ops::fusedL1SSIMBackwardPrivateUse1);
}
