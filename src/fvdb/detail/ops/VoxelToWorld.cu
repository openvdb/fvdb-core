// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/VoxelToWorld.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
voxelToWorldCallback(int32_t bidx,
                     int32_t eidx,
                     JaggedAccessor<ScalarType, 2> pts,
                     TensorAccessor<ScalarType, 2> outPts,
                     BatchGridAccessor batchAccessor,
                     bool primal) {
    const auto tx =
        primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);

    const auto pt                             = pts.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.applyInv(pt[0], pt[1], pt[2]);
    outPts[eidx][0]                           = wci[0];
    outPts[eidx][1]                           = wci[1];
    outPts[eidx][2]                           = wci[2];
}

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
voxelToWorldBackwardCallback(int32_t bidx,
                             int32_t eidx,
                             JaggedAccessor<ScalarType, 2> gradOut,
                             TensorAccessor<ScalarType, 2> outGradIn,
                             BatchGridAccessor batchAccessor,
                             bool primal) {
    const auto tx =
        primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto gradOutI = gradOut.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci =
        tx.applyInvGrad(gradOutI[0], gradOutI[1], gradOutI[2]);
    outGradIn[eidx][0] = wci[0] * gradOutI[0];
    outGradIn[eidx][1] = wci[1] * gradOutI[1];
    outGradIn[eidx][2] = wci[2] * gradOutI[2];
}

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
worldToVoxelCallback(int32_t bidx,
                     int32_t eidx,
                     JaggedAccessor<ScalarType, 2> pts,
                     TensorAccessor<ScalarType, 2> outPts,
                     BatchGridAccessor batchAccessor,
                     bool primal) {
    const auto tx =
        primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto pt                             = pts.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.apply(pt[0], pt[1], pt[2]);
    outPts[eidx][0]                           = wci[0];
    outPts[eidx][1]                           = wci[1];
    outPts[eidx][2]                           = wci[2];
}

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
worldToVoxelBackwardCallback(int32_t bidx,
                             int32_t eidx,
                             JaggedAccessor<ScalarType, 2> gradOut,
                             TensorAccessor<ScalarType, 2> outGradIn,
                             BatchGridAccessor batchAccessor,
                             bool primal) {
    const auto tx =
        primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto gradOutI                       = gradOut.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.applyGrad(gradOutI[0], gradOutI[1], gradOutI[2]);
    outGradIn[eidx][0]                        = wci[0] * gradOutI[0];
    outGradIn[eidx][1]                        = wci[1] * gradOutI[1];
    outGradIn[eidx][2]                        = wci[2] * gradOutI[2];
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
TransformPointsToGrid(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.rdim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.rsize(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();

    auto opts               = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outCoords = torch::empty({points.rsize(0), points.rsize(1)}, opts);

    auto batchAcc     = gridBatchAccessor<DeviceTag>(batchHdl);
    auto outCoordsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoords);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                voxelToWorldCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2, 512>(1, points, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                voxelToWorldCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, points, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
            voxelToWorldCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
    }

    return outCoords;
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
InvTransformPointsToGrid(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.rdim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.rsize(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();

    auto opts               = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outCoords = torch::empty({points.rsize(0), points.rsize(1)}, opts);

    auto batchAcc     = gridBatchAccessor<DeviceTag>(batchHdl);
    auto outCoordsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoords);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                worldToVoxelCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2, 512>(1, points, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                worldToVoxelCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, points, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
            worldToVoxelCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
    }

    return outCoords;
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
TransformPointsToGridBackward(const GridBatchData &batchHdl,
                              const JaggedTensor &gradOut,
                              bool isPrimal) {
    torch::Tensor outGradIn = torch::empty_like(gradOut.jdata());

    auto batchAcc     = gridBatchAccessor<DeviceTag>(batchHdl);
    auto outGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradIn);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                voxelToWorldBackwardCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2, 512>(1, gradOut, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                voxelToWorldBackwardCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, gradOut, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
            voxelToWorldBackwardCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(1, gradOut, cb);
    }

    return outGradIn;
}

template <torch::DeviceType DeviceTag, typename scalar_t>
torch::Tensor
InvTransformPointsToGridBackward(const GridBatchData &batchHdl,
                                 const JaggedTensor &gradOut,
                                 bool isPrimal) {
    torch::Tensor outGradIn = torch::empty_like(gradOut.jdata());

    auto batchAcc     = gridBatchAccessor<DeviceTag>(batchHdl);
    auto outGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradIn);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                worldToVoxelBackwardCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2, 512>(1, gradOut, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> ptsA) {
                worldToVoxelBackwardCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, gradOut, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
            worldToVoxelBackwardCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(1, gradOut, cb);
    }

    return outGradIn;
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchTransformPointsToGrid(const GridBatchData &batchHdl,
                              const JaggedTensor &points,
                              bool isPrimal) {
    return AT_DISPATCH_V2(points.scalar_type(),
                          "voxelToWorld",
                          AT_WRAP([&]() {
                              return TransformPointsToGrid<DeviceTag, scalar_t>(
                                  batchHdl, points, isPrimal);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchInvTransformPointsToGrid(const GridBatchData &batchHdl,
                                 const JaggedTensor &points,
                                 bool isPrimal) {
    return AT_DISPATCH_V2(points.scalar_type(),
                          "worldToVoxel",
                          AT_WRAP([&]() {
                              return InvTransformPointsToGrid<DeviceTag, scalar_t>(
                                  batchHdl, points, isPrimal);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchTransformPointsToGridBackward(const GridBatchData &batchHdl,
                                      const JaggedTensor &gradOut,
                                      bool isPrimal) {
    return AT_DISPATCH_V2(gradOut.scalar_type(),
                          "voxelToWorldBackward",
                          AT_WRAP([&]() {
                              return TransformPointsToGridBackward<DeviceTag, scalar_t>(
                                  batchHdl, gradOut, isPrimal);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

template <torch::DeviceType DeviceTag>
torch::Tensor
dispatchInvTransformPointsToGridBackward(const GridBatchData &batchHdl,
                                         const JaggedTensor &gradOut,
                                         bool isPrimal) {
    return AT_DISPATCH_V2(gradOut.scalar_type(),
                          "worldToVoxelBackward",
                          AT_WRAP([&]() {
                              return InvTransformPointsToGridBackward<DeviceTag, scalar_t>(
                                  batchHdl, gradOut, isPrimal);
                          }),
                          AT_EXPAND(AT_FLOATING_TYPES),
                          c10::kHalf);
}

} // anonymous namespace

torch::Tensor
voxelToWorld(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.rdim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.rsize(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();
    return FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return dispatchTransformPointsToGrid<DeviceTag>(batchHdl, points, isPrimal);
    });
}

torch::Tensor
worldToVoxel(const GridBatchData &batchHdl, const JaggedTensor &points, bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.rdim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.rsize(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();
    return FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return dispatchInvTransformPointsToGrid<DeviceTag>(batchHdl, points, isPrimal);
    });
}

torch::Tensor
voxelToWorldBackward(const GridBatchData &batchHdl, const JaggedTensor &gradOut, bool isPrimal) {
    return FVDB_DISPATCH_KERNEL(gradOut.device(), [&]() {
        return dispatchTransformPointsToGridBackward<DeviceTag>(batchHdl, gradOut, isPrimal);
    });
}

torch::Tensor
worldToVoxelBackward(const GridBatchData &batchHdl, const JaggedTensor &gradOut, bool isPrimal) {
    return FVDB_DISPATCH_KERNEL(gradOut.device(), [&]() {
        return dispatchInvTransformPointsToGridBackward<DeviceTag>(batchHdl, gradOut, isPrimal);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
