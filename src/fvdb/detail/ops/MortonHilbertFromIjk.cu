// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/MortonHilbertFromIjk.h>
#include <fvdb/detail/utils/HilbertCode.h>
#include <fvdb/detail/utils/MortonCode.h>
#include <fvdb/detail/utils/SimpleOpHelper.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <torch::DeviceType DeviceTag>
struct MortonProcessor : public BasePerElementProcessor<DeviceTag,
                                                        MortonProcessor<DeviceTag>,
                                                        FixedElementType<int32_t, 3>,
                                                        ScalarElementType<int64_t>> {
    // Per-element callback which computes the morton code for each ijk element in the tensor
    __hostdev__ void
    perElement(int64_t const element_idx, auto in_accessor, auto out_accessor) const {
        auto &&in = in_accessor[element_idx];

        // Coordinate from in
        auto const i = static_cast<uint32_t>(in[0]);
        auto const j = static_cast<uint32_t>(in[1]);
        auto const k = static_cast<uint32_t>(in[2]);

        out_accessor[element_idx] = static_cast<int64_t>(utils::morton(i, j, k));
    }
};

template <torch::DeviceType DeviceTag>
struct HilbertProcessor : public BasePerElementProcessor<DeviceTag,
                                                         HilbertProcessor<DeviceTag>,
                                                         FixedElementType<int32_t, 3>,
                                                         ScalarElementType<int64_t>> {
    // Per-element callback which computes the hilbert code for each ijk element in the tensor
    __hostdev__ void
    perElement(int64_t const element_idx, auto in_accessor, auto out_accessor) const {
        auto &&in = in_accessor[element_idx];

        // Coordinate from in
        auto const i = static_cast<uint32_t>(in[0]);
        auto const j = static_cast<uint32_t>(in[1]);
        auto const k = static_cast<uint32_t>(in[2]);

        out_accessor[element_idx] = static_cast<int64_t>(utils::hilbert(i, j, k));
    }
};

} // End anonymous namespace

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchMortonFromIjk(torch::Tensor ijk) {
    TORCH_CHECK_VALUE(ijk.dim() == 1, "ijk must be a 1D tensor");
    TORCH_CHECK_VALUE(ijk.size(1) == 3, "ijk must have 3 dimensions");
    TORCH_CHECK_VALUE(ijk.scalar_type() == torch::kInt32, "ijk must be int32");
    return MortonProcessor<DeviceTag>{}.execute(ijk);
}

torch::Tensor
mortonFromIjk(torch::Tensor ijk) {
    if (ijk.device().is_cuda()) {
        return dispatchMortonFromIjk<torch::kCUDA>(ijk);
    } else if (ijk.device().is_privateuseone()) {
        return dispatchMortonFromIjk<torch::kPrivateUse1>(ijk);
    } else {
        return dispatchMortonFromIjk<torch::kCPU>(ijk);
    }
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchHilbertFromIjk(torch::Tensor ijk) {
    TORCH_CHECK_VALUE(ijk.dim() == 1, "ijk must be a 1D tensor");
    TORCH_CHECK_VALUE(ijk.size(1) == 3, "ijk must have 3 dimensions");
    TORCH_CHECK_VALUE(ijk.scalar_type() == torch::kInt32, "ijk must be int32");
    return HilbertProcessor<DeviceTag>{}.execute(ijk);
}

torch::Tensor
hilbertFromIjk(torch::Tensor ijk) {
    if (ijk.device().is_cuda()) {
        return dispatchHilbertFromIjk<torch::kCUDA>(ijk);
    } else if (ijk.device().is_privateuseone()) {
        return dispatchHilbertFromIjk<torch::kPrivateUse1>(ijk);
    } else {
        return dispatchHilbertFromIjk<torch::kCPU>(ijk);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
