// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/ReadFromDense.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ inline void
readFromDenseCminorVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor batchHandle,
    torch::PackedTensorAccessor64<ScalarType, 5, torch::RestrictPtrTraits>
        inDenseTensor, // [B, X, Y, Z, C]
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> denseOrigins, // [B, 3]
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits>
        outSparseTensor                                                               // [B*N, C]
) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(
        inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
    const nanovdb::Coord denseOrigin(
        denseOrigins[batchIdx][0], denseOrigins[batchIdx][1], denseOrigins[batchIdx][2]);
    const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
    const int64_t baseOffset = batchHandle.voxelOffset(batchIdx);

    const LeafNodeT &leaf       = gpuGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);

    const bool isActive = leaf.isActive(voxelIdx);

    const nanovdb::Coord ijk = voxIjk - denseOrigin;
    const int64_t offset     = baseOffset + leaf.getValue(voxelIdx) - 1;

    if (isActive && bbox.isInside(voxIjk)) {
        outSparseTensor[offset][channelIdx] =
            inDenseTensor[batchIdx][ijk[0]][ijk[1]][ijk[2]][channelIdx];
    }
}

template <typename ScalarType>
__hostdev__ inline void
readFromDenseCmajorVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor batchHandle,
    torch::PackedTensorAccessor64<ScalarType, 5, torch::RestrictPtrTraits>
        inDenseTensor, // [B, C, X, Y, Z]
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> denseOrigins, // [B, 3]
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits>
        outSparseTensor                                                               // [B*N, C]
) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(
        inDenseTensor.size(2), inDenseTensor.size(3), inDenseTensor.size(4));
    const nanovdb::Coord denseOrigin(
        denseOrigins[batchIdx][0], denseOrigins[batchIdx][1], denseOrigins[batchIdx][2]);
    const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
    const int64_t baseOffset = batchHandle.voxelOffset(batchIdx);

    const LeafNodeT &leaf       = gpuGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);

    const bool isActive = leaf.isActive(voxelIdx);

    const nanovdb::Coord ijk = voxIjk - denseOrigin;
    const int64_t offset     = baseOffset + leaf.getValue(voxelIdx) - 1;

    if (isActive && bbox.isInside(voxIjk)) {
        outSparseTensor[offset][channelIdx] =
            inDenseTensor[batchIdx][channelIdx][ijk[0]][ijk[1]][ijk[2]];
    }
}

template <typename ScalarType>
void
readFromDenseCminorCPU(const GridBatchImpl::Accessor &gridHandle,
                       const torch::TensorAccessor<ScalarType, 5> inDenseTensor,
                       const torch::TensorAccessor<int32_t, 2> denseOrigins,
                       torch::TensorAccessor<ScalarType, 2> outSparseTensor,
                       bool isContiguous) {
    for (int64_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *grid = gridHandle.grid(bi);
        const nanovdb::Coord denseOrigin(
            denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord denseDim(
            inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
        const nanovdb::CoordBBox bbox(denseOrigin,
                                      denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);
        auto inBatch             = inDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<-1>(grid->tree(), baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - denseOrigin;
                if (isContiguous) {
                    memcpy(outSparseTensor[it->second].data(),
                           inBatch[ijk[0]][ijk[1]][ijk[2]].data(),
                           outSparseTensor.size(1) * sizeof(ScalarType));
                } else {
                    for (int c = 0; c < outSparseTensor.size(1); ++c) {
                        outSparseTensor[it->second][c] = inBatch[ijk[0]][ijk[1]][ijk[2]][c];
                    }
                }
            }
        }
    }
}

template <typename ScalarType>
void
readFromDenseCmajorCPU(const GridBatchImpl::Accessor &gridHandle,
                       const torch::TensorAccessor<ScalarType, 5> inDenseTensor,
                       const torch::TensorAccessor<int32_t, 2> denseOrigins,
                       torch::TensorAccessor<ScalarType, 2> outSparseTensor) {
    for (int64_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *grid = gridHandle.grid(bi);
        const nanovdb::Coord denseOrigin(
            denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord denseDim(
            inDenseTensor.size(2), inDenseTensor.size(3), inDenseTensor.size(4));
        const nanovdb::CoordBBox bbox(denseOrigin,
                                      denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);
        auto inBatch             = inDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<-1>(grid->tree(), baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - denseOrigin;
                for (int c = 0; c < outSparseTensor.size(1); ++c) {
                    outSparseTensor[it->second][c] = inBatch[c][ijk[0]][ijk[1]][ijk[2]];
                }
            }
        }
    }
}

void
readFromDenseCminorCUDA(const GridBatchImpl &batchHdl,
                        const torch::Tensor &inDenseTensor,
                        const torch::Tensor &denseOrigins,
                        torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(
        inDenseTensor.scalar_type(),
        "readFromDenseCminor",
        AT_WRAP([&]() {
            auto inDenseAcc =
                inDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc =
                outSparseTensor.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readFromDenseCminorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc);
            };
            forEachVoxelCUDA(1024, outSparseTensor.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

void
readFromDenseCminorPrivateUse1(const GridBatchImpl &batchHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(
        inDenseTensor.scalar_type(),
        "readFromDenseCminor",
        AT_WRAP([&]() {
            auto inDenseAcc =
                inDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc =
                outSparseTensor.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readFromDenseCminorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc);
            };
            forEachVoxelPrivateUse1(outSparseTensor.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

void
readFromDenseCminorCPUDispatch(const GridBatchImpl &gridHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor) {
    bool isContiguous = inDenseTensor.is_contiguous() && outSparseTensor.is_contiguous();

    AT_DISPATCH_V2(inDenseTensor.scalar_type(),
                   "readFromDenseCminor",
                   AT_WRAP([&]() {
                       readFromDenseCminorCPU(gridHdl.hostAccessor(),
                                              inDenseTensor.accessor<scalar_t, 5>(),
                                              denseOrigins.accessor<int32_t, 2>(),
                                              outSparseTensor.accessor<scalar_t, 2>(),
                                              isContiguous);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf,
                   c10::kBFloat16);
}

void
readFromDenseCmajorCUDA(const GridBatchImpl &batchHdl,
                        const torch::Tensor &inDenseTensor,
                        const torch::Tensor &denseOrigins,
                        torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(
        inDenseTensor.scalar_type(),
        "readFromDenseCmajor",
        AT_WRAP([&]() {
            auto inDenseAcc =
                inDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc =
                outSparseTensor.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readFromDenseCmajorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc);
            };
            forEachVoxelCUDA(1024, outSparseTensor.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

void
readFromDenseCmajorPrivateUse1(const GridBatchImpl &batchHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(
        inDenseTensor.scalar_type(),
        "readFromDenseCmajor",
        AT_WRAP([&]() {
            auto inDenseAcc =
                inDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc =
                outSparseTensor.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readFromDenseCmajorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc);
            };
            forEachVoxelPrivateUse1(outSparseTensor.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

void
readFromDenseCmajorCPUDispatch(const GridBatchImpl &gridHdl,
                               const torch::Tensor &inDenseTensor,
                               const torch::Tensor &denseOrigins,
                               torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(inDenseTensor.scalar_type(),
                   "readFromDenseCmajor",
                   AT_WRAP([&]() {
                       readFromDenseCmajorCPU(gridHdl.hostAccessor(),
                                              inDenseTensor.accessor<scalar_t, 5>(),
                                              denseOrigins.accessor<int32_t, 2>(),
                                              outSparseTensor.accessor<scalar_t, 2>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf,
                   c10::kBFloat16);
}

torch::Tensor
readFromDenseCminor(const GridBatchImpl &batchHdl,
                    const torch::Tensor &denseData,
                    const torch::Tensor &denseOrigins) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(denseData);
    TORCH_CHECK_VALUE(denseData.dim() > 4, "dense data must have shape [B, X, Y, Z, C]");
    TORCH_CHECK_VALUE(denseData.size(0) == batchHdl.batchSize(),
                      "dense data must have shape [B, X, Y, Z, *]");
    TORCH_CHECK_VALUE(denseOrigins.dim() == 2, "denseOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(0) == batchHdl.batchSize(),
                      "denseOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(1) == 3, "denseOrigins must have shape [B, 3]");

    torch::Tensor denseDataContig  = denseData.contiguous();
    torch::Tensor denseDataReshape = featureCoalescedView(denseDataContig, 4);

    torch::Tensor outSparseTensor =
        torch::zeros({batchHdl.totalVoxels(), denseDataReshape.size(4)}, denseData.options());

    if (batchHdl.device().is_cuda()) {
        readFromDenseCminorCUDA(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    } else if (batchHdl.device().is_privateuseone()) {
        readFromDenseCminorPrivateUse1(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    } else {
        readFromDenseCminorCPUDispatch(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    }

    return outSparseTensor;
}

torch::Tensor
readFromDenseCmajor(const GridBatchImpl &batchHdl,
                    const torch::Tensor &denseData,
                    const torch::Tensor &denseOrigins) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(denseData);
    TORCH_CHECK_VALUE(denseData.dim() > 4, "dense data must have shape [B, *, X, Y, Z]");
    TORCH_CHECK_VALUE(denseData.size(0) == batchHdl.batchSize(),
                      "dense data must have shape [B, *, X, Y, Z]");
    TORCH_CHECK_VALUE(denseOrigins.dim() == 2, "denseOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(0) == batchHdl.batchSize(),
                      "denseOrigins must have shape [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(1) == 3, "denseOrigins must have shape [B, 3]");

    torch::Tensor denseDataContig  = denseData.contiguous();
    torch::Tensor denseDataReshape = featureCoalescedViewTrailing(denseDataContig, 1, 3);

    torch::Tensor outSparseTensor =
        torch::zeros({batchHdl.totalVoxels(), denseDataReshape.size(1)}, denseData.options());

    if (batchHdl.device().is_cuda()) {
        readFromDenseCmajorCUDA(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    } else if (batchHdl.device().is_privateuseone()) {
        readFromDenseCmajorPrivateUse1(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    } else {
        readFromDenseCmajorCPUDispatch(batchHdl, denseDataReshape, denseOrigins, outSparseTensor);
    }

    return outSparseTensor;
}

} // namespace ops
} // namespace detail
} // namespace fvdb
