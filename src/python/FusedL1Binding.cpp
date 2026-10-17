// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/FusedL1.h>

#include <torch/library.h>

TORCH_LIBRARY_IMPL(fvdb, CUDA, m) {
    m.impl("_fused_l1_loss", &fvdb::detail::ops::fusedL1CUDA);
    m.impl("_fused_l1_loss_backward", &fvdb::detail::ops::fusedL1BackwardCUDA);
}

TORCH_LIBRARY_IMPL(fvdb, PrivateUse1, m) {
    m.impl("_fused_l1_loss", &fvdb::detail::ops::fusedL1PrivateUse1);
    m.impl("_fused_l1_loss_backward", &fvdb::detail::ops::fusedL1BackwardPrivateUse1);
}
