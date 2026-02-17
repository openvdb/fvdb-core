// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEOPTIONALINPUTS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEOPTIONALINPUTS_H

#include <c10/util/Exception.h>
#include <torch/types.h>

#include <optional>

namespace fvdb::detail::ops {

struct PreparedRasterOptionalInputs {
    const float *backgrounds = nullptr;
    const bool *masks        = nullptr;
    torch::Tensor backgroundsContig;
    torch::Tensor masksContig;
};

inline PreparedRasterOptionalInputs
prepareRasterOptionalInputs(const torch::Tensor &features,
                            const int64_t C,
                            const int64_t tileExtentH,
                            const int64_t tileExtentW,
                            const int64_t numChannels,
                            const at::optional<torch::Tensor> &backgrounds,
                            const at::optional<torch::Tensor> &masks) {
    PreparedRasterOptionalInputs out;

    if (backgrounds.has_value()) {
        TORCH_CHECK(backgrounds.value().is_cuda(), "backgrounds must be CUDA");
        TORCH_CHECK(backgrounds.value().device() == features.device(),
                    "backgrounds must be on the same device as features");
        TORCH_CHECK(backgrounds.value().scalar_type() == torch::kFloat32,
                    "backgrounds must have dtype=float32");
        TORCH_CHECK(backgrounds.value().sizes() == torch::IntArrayRef({C, numChannels}),
                    "backgrounds must have shape [C, NUM_CHANNELS]");
        out.backgroundsContig = backgrounds.value().contiguous();
        out.backgrounds       = out.backgroundsContig.data_ptr<float>();
    }

    if (masks.has_value()) {
        TORCH_CHECK(masks.value().is_cuda(), "masks must be CUDA");
        TORCH_CHECK(masks.value().device() == features.device(),
                    "masks must be on the same device as features");
        TORCH_CHECK(masks.value().scalar_type() == torch::kBool, "masks must have dtype=bool");
        TORCH_CHECK(masks.value().sizes() == torch::IntArrayRef({C, tileExtentH, tileExtentW}),
                    "masks must have shape [C, tileExtentH, tileExtentW]");
        out.masksContig = masks.value().contiguous();
        out.masks       = out.masksContig.data_ptr<bool>();
    }

    return out;
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRASTERIZEOPTIONALINPUTS_H
