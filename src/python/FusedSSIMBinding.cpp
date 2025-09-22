// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/FusedSSIM.h>

#include <torch/library.h>

TORCH_LIBRARY_IMPL(fvdb, CUDA, m) {
    m.impl("_fused_ssim", &fvdb::detail::ops::fusedssim_cuda);
    m.impl("_fused_ssim_backward", &fvdb::detail::ops::fusedssim_backward_cuda);
}

TORCH_LIBRARY_IMPL(fvdb, PrivateUse1, m) {
    m.impl("_fused_ssim", &fvdb::detail::ops::fusedssim_privateuse1);
    m.impl("_fused_ssim_backward", &fvdb::detail::ops::fusedssim_backward_privateuse1);
}
