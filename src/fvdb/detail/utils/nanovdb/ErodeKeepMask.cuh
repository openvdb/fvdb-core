// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/*!
    \file fvdb/detail/utils/nanovdb/ErodeKeepMask.cuh

    \brief Keep masks of a one-sided unit erosion of NanoVDB IndexGrids, for exclude-border
           padding (`dual_grid` / `build_padded_grid` with `exclude_border=True`).
*/

#ifndef FVDB_DETAIL_UTILS_NANOVDB_ERODEKEEPMASK_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_ERODEKEEPMASK_CUH

#include <nanovdb/NanoVDB.h>

#include <cstddef>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace morphology {

/// @brief Computes, for one source leaf, the 512-bit "keep" mask of the one-sided erosion
///        S (-) {0,1}^3 (positive) or S (-) {-1,0}^3 (negative): keep(c) is set iff every voxel
///        c+d, d in the octant, is active in the source grid.
///
/// The result is written to a per-source-leaf leaf mask, the keep masks of
/// BatchedTopologyBuilder's Prune pass. Since the octant always contains the origin, the keep
/// mask is a subset of the source activity mask, so pruning the source to it is exact. This
/// matches the CPU path, `buildPaddedGridFromGridWithoutBorderCPU` (BuildPaddedGrid.cu): a voxel
/// survives iff its entire [bmin,bmax]^3 box neighborhood is active. Called once per source leaf.
template <typename BuildT, bool Positive> struct ErodeKeepMaskFunctor {
    using LeafMaskT = nanovdb::Mask<nanovdb::NanoLeaf<BuildT>::LOG2DIM>;
    // The word shifts below assume 8^3 leaves: 8 words of 64 bits, word = x, byte = y, bit = z.
    static_assert(LeafMaskT::WORD_COUNT == 8, "ErodeKeepMaskFunctor assumes 8x64-bit leaf masks");

    __device__ void
    operator()(std::size_t leafID,
               const nanovdb::NanoGrid<BuildT> *srcGrid,
               LeafMaskT *d_keepMasks) const {
        const auto &srcTree   = srcGrid->tree();
        const auto &leaf      = srcTree.template getFirstNode<0>()[leafID];
        const auto leafOrigin = leaf.origin();

        uint64_t originalWordsShifted[10][3][3] = {};
        // Logical indices (x-voxel offset in [-1,8], y/z-block in [-1,1]) are centered onto the
        // [10][3][3] array by a +1 offset, avoiding the negative-subscript UB of a middle-element
        // reinterpret_cast reference.
        auto originalWords = [&](int i, int dBj, int dBk) -> uint64_t & {
            return originalWordsShifted[i + 1][dBj + 1][dBk + 1];
        };

        // Erosion gathers the opposite octant of padding:
        //   positive octant keep(c) = AND_{d in {0,1}^3} src(c+d) -> read blocks { 0,1}^3
        //   negative octant keep(c) = AND_{d in {-1,0}^3} src(c+d) -> read blocks {-1,0}^3
        constexpr int blockLo = Positive ? 0 : -1;
        constexpr int blockHi = Positive ? 1 : 0;
        for (int dBi = blockLo; dBi <= blockHi; dBi++)
            for (int dBj = blockLo; dBj <= blockHi; dBj++)
                for (int dBk = blockLo; dBk <= blockHi; dBk++) {
                    auto neighborOrigin = leafOrigin.offsetBy(dBi * 8, dBj * 8, dBk * 8);
                    if (auto neighborLeafPtr = srcTree.root().probeLeaf(neighborOrigin)) {
                        auto neighborWords = neighborLeafPtr->valueMask().words();
                        if (dBi == -1)
                            originalWords(-1, dBj, dBk) = neighborWords[7];
                        else if (dBi == 1)
                            originalWords(8, dBj, dBk) = neighborWords[0];
                        else
                            for (int i = 0; i < 8; i++)
                                originalWords(i, dBj, dBk) = neighborWords[i];
                    }
                }

        auto keepWords = d_keepMasks[leafID].words();
        if constexpr (Positive) {
            // Erode toward +z: keep(c) &= src(c+(0,0,1))
            for (int i = 0; i <= 8; i++)
                for (int dBj = 0; dBj <= 1; dBj++) {
                    uint64_t w = originalWords(i, dBj, 0);
                    w &= ((originalWords(i, dBj, 0) & 0xfefefefefefefefeUL) >> 1) |
                         ((originalWords(i, dBj, 1) & 0x0101010101010101UL) << 7);
                    originalWords(i, dBj, 0) = w;
                }
            // Erode toward +y
            for (int i = 0; i <= 8; i++) {
                uint64_t w = originalWords(i, 0, 0);
                w &= ((originalWords(i, 0, 0) & 0xffffffffffffff00UL) >> 8) |
                     ((originalWords(i, 1, 0) & 0x00000000000000ffUL) << 56);
                originalWords(i, 0, 0) = w;
            }
            // Erode toward +x
            for (int i = 0; i <= 7; i++)
                keepWords[i] = originalWords(i, 0, 0) & originalWords(i + 1, 0, 0);
        } else {
            // Erode toward -z: keep(c) &= src(c+(0,0,-1))
            for (int i = -1; i <= 7; i++)
                for (int dBj = -1; dBj <= 0; dBj++) {
                    uint64_t w = originalWords(i, dBj, 0);
                    w &= ((originalWords(i, dBj, 0) & 0x7f7f7f7f7f7f7f7fUL) << 1) |
                         ((originalWords(i, dBj, -1) & 0x8080808080808080UL) >> 7);
                    originalWords(i, dBj, 0) = w;
                }
            // Erode toward -y
            for (int i = -1; i <= 7; i++) {
                uint64_t w = originalWords(i, 0, 0);
                w &= ((originalWords(i, 0, 0) & 0x00ffffffffffffffUL) << 8) |
                     ((originalWords(i, -1, 0) & 0xff00000000000000UL) >> 56);
                originalWords(i, 0, 0) = w;
            }
            // Erode toward -x
            for (int i = 0; i <= 7; i++)
                keepWords[i] = originalWords(i, 0, 0) & originalWords(i - 1, 0, 0);
        }
    }
};

} // namespace morphology
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_ERODEKEEPMASK_CUH
