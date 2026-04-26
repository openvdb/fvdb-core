// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/SampleTrilinear.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearStencil.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

// One-thread-per-point callback: resolve the 8-corner stencil once, then iterate
// all channels with scalar loads. Works on both CPU and GPU, all scalar types.
template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
sampleTrilinearStencilCallback(int32_t bidx,
                               int32_t eidx,
                               int32_t /*cidx*/,
                               JaggedAccessor<ScalarType, 2> points,
                               TensorAccessor<ScalarType, 2> gridData,
                               BatchGridAccessor batchAccessor,
                               TensorAccessor<ScalarType, 2> outFeatures,
                               int64_t numChannels) {
    using MathType = at::opmath_type<ScalarType>;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz =
        transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                        static_cast<MathType>(pointsData[eidx][1]),
                        static_cast<MathType>(pointsData[eidx][2]));

    int64_t indices[8] = {};
    MathType weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    for (int64_t c = 0; c < numChannels; ++c) {
        MathType accum = MathType(0);
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            accum += weights[corner] * static_cast<MathType>(gridData[indices[corner]][c]);
        }
        outFeatures[eidx][c] = static_cast<ScalarType>(accum);
    }
}

namespace {

// Aligned wrapper around nanovdb::math::Vec2<T>. nanovdb's Vec2 stores a bare
// T[2] without an alignas, so its natural alignment is alignof(T) (4B for
// float, 8B for double). Reinterpreting a row as Vec2<T>* and dereferencing
// would let nvcc split the access into two scalar loads. Bumping the alignment
// to 2*sizeof(T) (8B for float, 16B for double) matches float2/double2 and is
// the contract the runtime address checks (`aligned8`/`aligned16`) rely on to
// guarantee a single LDG.64/LDG.128 transaction. Inheriting from Vec2 keeps
// arithmetic operators (`+=`, `*scalar`, ...) usable without rewriting them.
template <typename T> struct alignas(2 * sizeof(T)) AlignedVec2 : nanovdb::math::Vec2<T> {
    using Base = nanovdb::math::Vec2<T>;
    __hostdev__
    AlignedVec2()
        : Base(T(0), T(0)) {}
    __hostdev__
    AlignedVec2(const Base &b)
        : Base(b) {}
};

// Aligned wrapper around nanovdb::math::Vec4<T>. Same rationale as
// AlignedVec2: nanovdb::math::Vec4 has alignof(T) (4B for float), but a single
// LDG.128 needs the address (and the type) to be 16B aligned. Only
// instantiated for float in this TU because CUDA has no native 4-wide double
// vector with single-instruction 16-byte load semantics; doubles fall through
// to the scalar path or the Vec2 fast path.
template <typename T> struct alignas(4 * sizeof(T)) AlignedVec4 : nanovdb::math::Vec4<T> {
    using Base = nanovdb::math::Vec4<T>;
    __hostdev__
    AlignedVec4()
        : Base(T(0), T(0), T(0), T(0)) {}
    __hostdev__
    AlignedVec4(const Base &b)
        : Base(b) {}
};

} // namespace

// One-thread-per-point callback specialized for numChannels == 2. Resolves the
// 8-corner stencil once, then issues a single 8-byte (float2) or 16-byte
// (double2) vector load per corner, halving per-sample transactions relative
// to the scalar path. GPU only. Caller is responsible for alignment checks.
template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearStencilCallbackVec2(int32_t bidx,
                                   int32_t eidx,
                                   int32_t /*cidx*/,
                                   JaggedAccessor<ScalarType, 2> points,
                                   TensorAccessor<ScalarType, 2> gridData,
                                   BatchGridAccessor batchAccessor,
                                   TensorAccessor<ScalarType, 2> outFeatures) {
    using Vec = AlignedVec2<ScalarType>;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<ScalarType> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    int64_t indices[8] = {};
    ScalarType weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    Vec accum;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        const Vec val = *reinterpret_cast<const Vec *>(&gridData[indices[corner]][0]);
        accum += val * weights[corner];
    }
    *reinterpret_cast<Vec *>(&outFeatures[eidx][0]) = accum;
}

// One-thread-per-point callback specialized for numChannels % 4 == 0. Resolves
// the 8-corner stencil once, then iterates channels in 4-wide groups with a
// single 16-byte (float4) vector load per corner per group, quartering
// per-sample transactions relative to the scalar path. GPU only, float only.
// Caller is responsible for alignment checks.
template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__device__ void
sampleTrilinearStencilCallbackVec4(int32_t bidx,
                                   int32_t eidx,
                                   int32_t /*cidx*/,
                                   JaggedAccessor<ScalarType, 2> points,
                                   TensorAccessor<ScalarType, 2> gridData,
                                   BatchGridAccessor batchAccessor,
                                   TensorAccessor<ScalarType, 2> outFeatures,
                                   int64_t numChannels) {
    using Vec = AlignedVec4<ScalarType>;

    const auto &pointsData               = points.data();
    const nanovdb::OnIndexGrid *grid     = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset             = batchAccessor.voxelOffset(bidx);
    auto gridAcc                         = grid->tree().getAccessor();

    const nanovdb::math::Vec3<ScalarType> xyz =
        transform.apply(pointsData[eidx][0], pointsData[eidx][1], pointsData[eidx][2]);

    int64_t indices[8] = {};
    ScalarType weights[8];
    const uint8_t activeMask = resolveTrilinearStencil(xyz, gridAcc, baseOffset, indices, weights);

    if (activeMask == 0)
        return;

    const int64_t numGroups = numChannels / 4;
    for (int64_t g = 0; g < numGroups; ++g) {
        const int64_t cBase = g * 4;
        Vec accum;
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const Vec val = *reinterpret_cast<const Vec *>(&gridData[indices[corner]][cBase]);
            accum += val * weights[corner];
        }
        *reinterpret_cast<Vec *>(&outFeatures[eidx][cBase]) = accum;
    }
}

template <torch::DeviceType DeviceTag, typename scalar_t>
std::vector<torch::Tensor>
SampleGridTrilinear(const GridBatchData &batchHdl,
                    const JaggedTensor &points,
                    const torch::Tensor &gridData) {
    auto opts = torch::TensorOptions()
                    .dtype(gridData.dtype())
                    .device(gridData.device())
                    .requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData).contiguous();     // [B*N, -1]
    torch::Tensor outFeatures =
        torch::zeros({points.rsize(0), gridDataReshape.size(1)}, opts).contiguous(); // [B*M, -1]
    auto outShape = spliceShape({points.rsize(0)}, gridData, 1);                     // [B*M, *]

    auto batchAcc             = gridBatchAccessor<DeviceTag>(batchHdl);
    auto gridDataAcc          = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
    auto outFeaturesAcc       = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
    const int64_t numChannels = gridDataReshape.size(1);

    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        auto dispatchForEach = [&](const auto &cb) {
            if constexpr (DeviceTag == torch::kCUDA) {
                forEachJaggedElementChannelCUDA<scalar_t, 2>(int64_t(1), points, cb);
            } else {
                forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(int64_t(1), points, cb);
            }
        };

        auto scalarCb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc64<scalar_t, 2> pts) {
                sampleTrilinearStencilCallback<scalar_t, JaggedRAcc64, TorchRAcc64>(
                    bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
            };

        if constexpr (std::is_same_v<scalar_t, float>) {
            const auto gridDataAddr =
                reinterpret_cast<uintptr_t>(gridDataReshape.data_ptr<float>());
            const auto outAddr   = reinterpret_cast<uintptr_t>(outFeatures.data_ptr<float>());
            const bool aligned8  = (gridDataAddr % 8 == 0) && (outAddr % 8 == 0);
            const bool aligned16 = (gridDataAddr % 16 == 0) && (outAddr % 16 == 0);
            if (numChannels == 2 && aligned8) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    sampleTrilinearStencilCallbackVec2<float, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                dispatchForEach(cb);
            } else if (numChannels >= 4 && numChannels % 4 == 0 && aligned16) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<float, 2> pts) {
                    sampleTrilinearStencilCallbackVec4<float, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
                };
                dispatchForEach(cb);
            } else {
                dispatchForEach(scalarCb);
            }
        } else if constexpr (std::is_same_v<scalar_t, double>) {
            const auto gridDataAddr =
                reinterpret_cast<uintptr_t>(gridDataReshape.data_ptr<double>());
            const auto outAddr   = reinterpret_cast<uintptr_t>(outFeatures.data_ptr<double>());
            const bool aligned16 = (gridDataAddr % 16 == 0) && (outAddr % 16 == 0);
            if (numChannels == 2 && aligned16) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc64<double, 2> pts) {
                    sampleTrilinearStencilCallbackVec2<double, JaggedRAcc64, TorchRAcc64>(
                        bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                dispatchForEach(cb);
            } else {
                dispatchForEach(scalarCb);
            }
        } else {
            dispatchForEach(scalarCb);
        }
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
            sampleTrilinearStencilCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, numChannels);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(int64_t(1), points, cb);
    }

    return {outFeatures.reshape(outShape)};
}

template <torch::DeviceType DeviceTag>
std::vector<torch::Tensor>
dispatchSampleGridTrilinear(const GridBatchData &batchHdl,
                            const JaggedTensor &points,
                            const torch::Tensor &gridData) {
    return AT_DISPATCH_V2(
        points.scalar_type(),
        "SampleGridTrilinear",
        AT_WRAP(
            [&] { return SampleGridTrilinear<DeviceTag, scalar_t>(batchHdl, points, gridData); }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kCPU>(const GridBatchData &,
                                                                             const JaggedTensor &,
                                                                             const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kCUDA>(
    const GridBatchData &, const JaggedTensor &, const torch::Tensor &);
template std::vector<torch::Tensor> dispatchSampleGridTrilinear<torch::kPrivateUse1>(
    const GridBatchData &, const JaggedTensor &, const torch::Tensor &);

std::vector<torch::Tensor>
sampleTrilinear(const GridBatchData &batchHdl,
                const JaggedTensor &points,
                const torch::Tensor &gridData) {
    batchHdl.checkNonEmptyGrid();
    TORCH_CHECK_VALUE(points.device() == gridData.device(),
                      "points and data must be on the same device");
    batchHdl.checkDevice(points);
    batchHdl.checkDevice(gridData);
    points.check_valid();
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == gridData.dtype(), "all tensors must have the same type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      "Expected points to have shape [B*M, 3] (wrong number of dimensions)");
    TORCH_CHECK(points.numel() > 0, "Empty tensor (points)");
    TORCH_CHECK(points.rsize(1) == 3, "points must have shape [B, M, 3] (points must be 3D)");
    TORCH_CHECK_TYPE(gridData.is_floating_point(), "data must have a floating point type");
    TORCH_CHECK_VALUE(gridData.dim() >= 2,
                      "Expected data to have shape [N, *] (at least 2 dimensions)");
    TORCH_CHECK(gridData.numel() > 0, "Empty tensor (data)");
    TORCH_CHECK(gridData.size(0) == batchHdl.totalVoxels(),
                "grid_data must have one value per voxel (shape [N, *]) (wrong first dimension)");
    return FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return dispatchSampleGridTrilinear<DeviceTag>(batchHdl, points, gridData);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
