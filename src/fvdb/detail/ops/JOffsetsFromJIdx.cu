// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/JOffsetsFromJIdx.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/cub.cuh>

namespace fvdb {
namespace detail {
namespace ops {

template <typename T>
__global__ void
setZero(T *thingToSet) {
    *thingToSet = 0;
}

torch::Tensor
joffsetsFromJIdx(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors) {
    TORCH_CHECK_VALUE(jidx.dim() == 1, "jidx must be a 1D tensor");

    if (jidx.size(0) == 0 && numTensors == 1) {
        torch::Tensor ret = torch::empty(
            {2},
            torch::TensorOptions()
                .dtype(JOffsetsScalarType)
                .pinned_memory(jdata.device().is_cuda() || jdata.device().is_privateuseone()));
        auto acc = ret.accessor<JOffsetsType, 1>();
        acc[0]   = 0;
        acc[1]   = jdata.size(0);
        return ret.to(jdata.device());
    }

    // Get the number of unique batch indices assuming jidx is always sorted
    // It should be of the form [0, ..., 0, 1, ..., 1, 3, ..., 3, ...]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> uniqueRes =
        torch::unique_dim(jidx, 0, false, false, true);
    torch::Tensor uniqueBatchValues = std::get<0>(uniqueRes); // [0, 1, 3, ...]
    torch::Tensor uniqueBatchCounts = std::get<2>(uniqueRes); // [n0, n1, n3, ...]

    torch::Tensor fullBatchCounts =
        torch::full({numTensors + 1},
                    0,
                    torch::TensorOptions().dtype(JOffsetsScalarType).device(jdata.device()));
    fullBatchCounts.index({torch::indexing::Slice(1, torch::indexing::None, 1)})
        .index_put_({uniqueBatchValues}, uniqueBatchCounts);

    torch::Tensor cumOffsets = torch::cumsum(fullBatchCounts, 0, JOffsetsScalarType);
    return cumOffsets;
}

torch::Tensor
jOffsetsFromJIdx(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors) {
    TORCH_CHECK_VALUE(jidx.dim() == 1, "jidx must be a 1D tensor");
    TORCH_CHECK_VALUE(jidx.size(0) == 0 || jidx.device() == jdata.device(),
                      "jidx and jdata must be on the same device");
    c10::OptionalDeviceGuard deviceGuard(jdata.device());
    return joffsetsFromJIdx(jidx, jdata, numTensors);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
