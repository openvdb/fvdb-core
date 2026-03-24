// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/NeighborIndexes.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
neighborIndexesCallback(int32_t bidx,
                          int32_t eidx,
                          JaggedAccessor<ScalarType, 2> coords,
                          TensorAccessor<int64_t, 4> outIndex,
                          BatchGridAccessor batchAccessor,
                          nanovdb::Coord extentMin,
                          nanovdb::Coord extentMax,
                          int32_t shift) {
    const nanovdb::OnIndexGrid *gpuGrid = batchAccessor.grid(bidx);
    auto acc                            = gpuGrid->getAccessor();

    auto coord          = coords.data()[eidx];
    nanovdb::Coord ijk0 = nanovdb::Coord(coord[0], coord[1], coord[2]) << shift;

    for (int32_t i = extentMin[0]; i <= extentMax[0]; i += 1) {
        for (int32_t j = extentMin[1]; j <= extentMax[1]; j += 1) {
            for (int32_t k = extentMin[2]; k <= extentMax[2]; k += 1) {
                const nanovdb::Coord ijk = nanovdb::Coord(i, j, k) + ijk0;
                const int64_t index = acc.isActive(ijk) ? ((int64_t)acc.getValue(ijk) - 1) : -1;
                outIndex[eidx][i - extentMin[0]][j - extentMin[1]][k - extentMin[2]] = index;
            }
        }
    }
}

template <torch::DeviceType DeviceTag>
JaggedTensor
VoxelNeighborhood(const GridBatchData &batchHdl,
                  const JaggedTensor &ijk,
                  nanovdb::Coord extentMin,
                  nanovdb::Coord extentMax,
                  int32_t shift) {
    batchHdl.checkDevice(ijk);
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK(ijk.rdim() == 2,
                std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK(ijk.rsize(0) > 0, "Empty tensor (coords)");
    TORCH_CHECK(ijk.rsize(1) == 3,
                "Expected 3 dimensional coords but got points.shape[1] = " +
                    std::to_string(ijk.rsize(1)));

    for (int i = 0; i < 3; i++) {
        TORCH_CHECK(extentMin[i] <= extentMax[i],
                    "Extent min must be less than or equal to extent max");
    }
    TORCH_CHECK(shift >= 0, "Bitshift must be non-negative");
    const nanovdb::Coord extentPerAxis = (extentMax - extentMin) + nanovdb::Coord(1);
    const uint32_t numVals             = extentPerAxis[0] * extentPerAxis[1] * extentPerAxis[2];

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(ijk.device());
    torch::Tensor outIndex =
        torch::empty({ijk.rsize(0), extentPerAxis[0], extentPerAxis[1], extentPerAxis[2]}, opts);

    AT_DISPATCH_V2(
        ijk.scalar_type(),
        "VoxelNeighborhood",
        AT_WRAP([&]() {
            auto batchAcc    = gridBatchAccessor<DeviceTag>(batchHdl);
            auto outIndexAcc = tensorAccessor<DeviceTag, int64_t, 4>(outIndex);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<scalar_t, 2> ptsA) {
                    neighborIndexesCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, ptsA, outIndexAcc, batchAcc, extentMin, extentMax, shift);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, 1, ijk, cb);
            } else {
                auto cb =
                    [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                        neighborIndexesCallback<scalar_t, JaggedAcc, TorchAcc>(
                            bidx, eidx, ptsA, outIndexAcc, batchAcc, extentMin, extentMax, shift);
                    };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
            }
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));

    return ijk.jagged_like(outIndex);
}

JaggedTensor
neighborIndexes(const GridBatchData &batchHdl,
                  const JaggedTensor &coords,
                  int32_t extent,
                  int32_t shift) {
    TORCH_CHECK_VALUE(
        coords.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        coords.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(extent >= 0, "extent must be >= 0");
    nanovdb::Coord extentMin(-extent, -extent, -extent);
    nanovdb::Coord extentMax(extent, extent, extent);
    return FVDB_DISPATCH_KERNEL_DEVICE(coords.device(), [&]() {
        return VoxelNeighborhood<DeviceTag>(batchHdl, coords, extentMin, extentMax, shift);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
