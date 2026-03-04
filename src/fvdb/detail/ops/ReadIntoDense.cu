// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/ReadIntoDense.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ inline void
readIntoDenseCminorVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor batchHandle,
    // [B, 3]
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> denseOrigins, // [B, 3]
    // [B*N, C]
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits> inSparseTensor,
    // [B, X, Y, Z, C]
    torch::PackedTensorAccessor64<ScalarType, 5, torch::RestrictPtrTraits> outDenseTensor) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(
        outDenseTensor.size(1), outDenseTensor.size(2), outDenseTensor.size(3));
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
        outDenseTensor[batchIdx][ijk[0]][ijk[1]][ijk[2]][channelIdx] =
            inSparseTensor[offset][channelIdx];
    }
}

template <typename ScalarType>
__hostdev__ inline void
readIntoDenseCmajorVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor batchHandle,
    // [B, 3]
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> denseOrigins,
    // [B*N, C]
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits> inSparseTensor,
    // [B, C, X, Y, Z]
    torch::PackedTensorAccessor64<ScalarType, 5, torch::RestrictPtrTraits> outDenseTensor) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(
        outDenseTensor.size(2), outDenseTensor.size(3), outDenseTensor.size(4));
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
        outDenseTensor[batchIdx][channelIdx][ijk[0]][ijk[1]][ijk[2]] =
            inSparseTensor[offset][channelIdx];
    }
}

template <typename ScalarType>
void
readIntoDenseCminorCPU(const GridBatchImpl::Accessor &gridHandle,
                       const torch::TensorAccessor<ScalarType, 2> inGridData,
                       const torch::TensorAccessor<int32_t, 2> denseOrigins,
                       torch::TensorAccessor<ScalarType, 5> outDenseTensor,
                       bool isContiguous) {
    for (int64_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *grid = gridHandle.grid(bi);

        const nanovdb::Coord bbmin(denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord bbsize(
            outDenseTensor.size(1), outDenseTensor.size(2), outDenseTensor.size(3));
        const nanovdb::CoordBBox bbox(bbmin, bbmin + bbsize - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);

        auto outBatch = outDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<-1>(grid->tree(), baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - bbox.min();

                if (isContiguous) {
                    memcpy(outBatch[ijk[0]][ijk[1]][ijk[2]].data(),
                           inGridData[it->second].data(),
                           inGridData.size(1) * sizeof(ScalarType));
                } else {
                    for (int c = 0; c < inGridData.size(1); ++c) {
                        outBatch[ijk[0]][ijk[1]][ijk[2]][c] = inGridData[it->second][c];
                    }
                }
            }
        }
    }
}

template <typename ScalarType>
void
readIntoDenseCmajorCPU(const GridBatchImpl::Accessor &gridHandle,
                       const torch::TensorAccessor<ScalarType, 2> inGridData,
                       const torch::TensorAccessor<int32_t, 2> denseOrigins,
                       torch::TensorAccessor<ScalarType, 5> outDenseTensor) {
    for (int64_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *grid = gridHandle.grid(bi);

        const nanovdb::Coord bbmin(denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord bbsize(
            outDenseTensor.size(2), outDenseTensor.size(3), outDenseTensor.size(4));
        const nanovdb::CoordBBox bbox(bbmin, bbmin + bbsize - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);

        auto outBatch = outDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<-1>(grid->tree(), baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - bbox.min();

                for (int c = 0; c < inGridData.size(1); ++c) {
                    outBatch[c][ijk[0]][ijk[1]][ijk[2]] = inGridData[it->second][c];
                }
            }
        }
    }
}

template <>
void
dispatchReadIntoDenseCminor<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                          const torch::Tensor &inGridData,
                                          const torch::Tensor &denseOrigins,
                                          torch::Tensor &outDenseTensor) {
    AT_DISPATCH_V2(
        outDenseTensor.scalar_type(),
        "readIntoDenseCminor",
        AT_WRAP([&]() {
            auto outDenseAcc =
                outDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto inGridDataAcc =
                inGridData.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readIntoDenseCminorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, denseOriginsAcc, inGridDataAcc, outDenseAcc);
            };
            forEachVoxelCUDA(1024, inGridData.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

template <>
void
dispatchReadIntoDenseCminor<torch::kPrivateUse1>(const GridBatchImpl &batchHdl,
                                                 const torch::Tensor &inGridData,
                                                 const torch::Tensor &denseOrigins,
                                                 torch::Tensor &outDenseTensor) {
    AT_DISPATCH_V2(
        outDenseTensor.scalar_type(),
        "readIntoDenseCminor",
        AT_WRAP([&]() {
            auto outDenseAcc =
                outDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto inGridDataAcc =
                inGridData.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readIntoDenseCminorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, denseOriginsAcc, inGridDataAcc, outDenseAcc);
            };
            forEachVoxelPrivateUse1(inGridData.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

template <>
void
dispatchReadIntoDenseCminor<torch::kCPU>(const GridBatchImpl &gridHdl,
                                         const torch::Tensor &inGridData,
                                         const torch::Tensor &denseOrigins,
                                         torch::Tensor &outDenseTensor) {
    bool isContiguous = inGridData.is_contiguous() && outDenseTensor.is_contiguous();

    AT_DISPATCH_V2(outDenseTensor.scalar_type(),
                   "readIntoDenseCminor",
                   AT_WRAP([&]() {
                       readIntoDenseCminorCPU(gridHdl.hostAccessor(),
                                              inGridData.accessor<scalar_t, 2>(),
                                              denseOrigins.accessor<int32_t, 2>(),
                                              outDenseTensor.accessor<scalar_t, 5>(),
                                              isContiguous);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf,
                   c10::kBFloat16);
}

template <>
void
dispatchReadIntoDenseCmajor<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                          const torch::Tensor &inGridData,
                                          const torch::Tensor &denseOrigins,
                                          torch::Tensor &outDenseTensor) {
    AT_DISPATCH_V2(
        outDenseTensor.scalar_type(),
        "readIntoDenseCmajor",
        AT_WRAP([&]() {
            auto outDenseAcc =
                outDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto inGridDataAcc =
                inGridData.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readIntoDenseCmajorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, denseOriginsAcc, inGridDataAcc, outDenseAcc);
            };
            forEachVoxelCUDA(1024, inGridData.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

template <>
void
dispatchReadIntoDenseCmajor<torch::kPrivateUse1>(const GridBatchImpl &batchHdl,
                                                 const torch::Tensor &inGridData,
                                                 const torch::Tensor &denseOrigins,
                                                 torch::Tensor &outDenseTensor) {
    AT_DISPATCH_V2(
        outDenseTensor.scalar_type(),
        "readIntoDenseCmajor",
        AT_WRAP([&]() {
            auto outDenseAcc =
                outDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto inGridDataAcc =
                inGridData.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor batchAcc) {
                readIntoDenseCmajorVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, denseOriginsAcc, inGridDataAcc, outDenseAcc);
            };
            forEachVoxelPrivateUse1(inGridData.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

template <>
void
dispatchReadIntoDenseCmajor<torch::kCPU>(const GridBatchImpl &gridHdl,
                                         const torch::Tensor &inGridData,
                                         const torch::Tensor &denseOrigins,
                                         torch::Tensor &outDenseTensor) {
    AT_DISPATCH_V2(outDenseTensor.scalar_type(),
                   "readIntoDenseCmajor",
                   AT_WRAP([&]() {
                       readIntoDenseCmajorCPU(gridHdl.hostAccessor(),
                                              inGridData.accessor<scalar_t, 2>(),
                                              denseOrigins.accessor<int32_t, 2>(),
                                              outDenseTensor.accessor<scalar_t, 5>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf,
                   c10::kBFloat16);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
