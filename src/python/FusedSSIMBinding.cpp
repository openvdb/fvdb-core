// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/ops/gsplat/FusedSSIM.h>

#include <torch/library.h>

TORCH_LIBRARY_IMPL(fvdb, CUDA, m) {
    m.impl("_fused_ssim", &fvdb::detail::ops::fusedSSIMCUDA);
    m.impl("_fused_ssim_backward", &fvdb::detail::ops::fusedSSIMBackwardCUDA);
}

TORCH_LIBRARY_IMPL(fvdb, PrivateUse1, m) {
    m.impl("_fused_ssim", &fvdb::detail::ops::fusedSSIMPrivateUse1);
    m.impl("_fused_ssim_backward", &fvdb::detail::ops::fusedSSIMBackwardPrivateUse1);
}
