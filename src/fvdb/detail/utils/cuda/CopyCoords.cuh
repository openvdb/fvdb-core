// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_COPYCOORDS_CUH
#define FVDB_DETAIL_UTILS_CUDA_COPYCOORDS_CUH

#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {

/// Write all ijk coordinates within a bounding box (offset by ijk0) into output tensors.
__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord &ijk0,
           const nanovdb::CoordBBox &bbox,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    int32_t count = 0;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk                      = ijk0 + nanovdb::Coord(di, dj, dk);
                outIJK[base + count][0]  = ijk[0];
                outIJK[base + count][1]  = ijk[1];
                outIJK[base + count][2]  = ijk[2];
                outIJKBIdx[base + count] = bidx;
                count += 1;
            }
        }
    }
}

/// Overload taking a size Coord instead of a CoordBBox (builds bbox as [0, size-1]).
__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord size,
           const nanovdb::Coord &ijk0,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    return copyCoords(bidx,
                      base,
                      ijk0,
                      nanovdb::CoordBBox(nanovdb::Coord(0), size - nanovdb::Coord(1)),
                      outIJK,
                      outIJKBIdx);
}

/// Write a single coordinate if all voxels in the bbox (offset by ijk0) are active.
__device__ inline void
copyCoordsWithoutBorder(
    const typename nanovdb::DefaultReadAccessor<nanovdb::ValueOnIndex> gridAccessor,
    const fvdb::JIdxType bidx,
    const int64_t base,
    const nanovdb::Coord &ijk0,
    const nanovdb::CoordBBox &bbox,
    const TorchRAcc64<int64_t, 1> packInfoBase,
    TorchRAcc64<int32_t, 2> outIJK,
    TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk    = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }
    if (active) {
        int64_t outBase     = packInfoBase[base];
        outIJK[outBase][0]  = ijk0[0];
        outIJK[outBase][1]  = ijk0[1];
        outIJK[outBase][2]  = ijk0[2];
        outIJKBIdx[outBase] = bidx;
    }
}

/// Count 1 if all voxels in the bbox (offset by ijk0) are active, 0 otherwise.
__device__ inline void
countCoordsWithoutBorder(
    const typename nanovdb::DefaultReadAccessor<nanovdb::ValueOnIndex> gridAccessor,
    const fvdb::JIdxType bidx,
    const int64_t base,
    const nanovdb::Coord &ijk0,
    const nanovdb::CoordBBox &bbox,
    TorchRAcc64<int64_t, 1> outCounter) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk    = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }

    outCounter[base] = active ? 1 : 0;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_COPYCOORDS_CUH
