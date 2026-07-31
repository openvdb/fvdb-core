// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/*!
    \file fvdb/detail/utils/nanovdb/PadGrid.cuh

    \brief One-sided (octant) morphological padding of NanoVDB IndexGrids on the device.

    The operation is a *one-sided* (positive- or negative-octant) unit dilation, i.e. the
    Minkowski sum S (+) {0,1}^3 (positive octant) or S (+) {-1,0}^3 (negative octant).
    A general padded grid S (+) [bmin,bmax]^3 is obtained by composing `bmax` positive
    passes and `-bmin` negative passes (Minkowski sums by boxes compose). `dual_grid` is
    exactly one positive pass.

    It is closely modeled on `nanovdb::tools::cuda::DilateGrid` (symmetric 26-connected
    dilation): the driver, root speculation, and TopologyBuilder pipeline are reused
    essentially verbatim. Only two things change relative to `DilateGrid`:
      1. The internal-node functor uses a one-sided 8-direction stencil instead of the
         symmetric 26-direction stencil (`padNeighborMaskStencil` vs `neighborMaskStencil`).
         The full 27-way MaskShift scatter and the full 26-neighborhood root speculation
         are retained unchanged -- feeding a one-sided stencil into them yields a one-sided
         dilation, and any speculatively-introduced tiles that end up empty are pruned.
      2. The leaf-node functor shifts the activity masks in a single octant instead of all
         26 neighbor directions.

*/

#ifndef FVDB_DETAIL_UTILS_NANOVDB_PADGRID_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_PADGRID_CUH

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <cub/cub.cuh>

#include <map>

namespace fvdb {
namespace detail {
namespace morphology {

// Pull in the NanoVDB names used by the (near-verbatim) functor/driver bodies below so
// that they read like their `nanovdb::tools::cuda::DilateGrid` counterparts.
using namespace nanovdb;
using namespace nanovdb::util;
using namespace nanovdb::util::morphology;
using namespace nanovdb::util::morphology::cuda;

// ---------------------------------------------------------------------------------------
// One-sided neighbor stencil
// ---------------------------------------------------------------------------------------

/// @brief One-sided analogue of `nanovdb::util::morphology::neighborMaskStencil`.
///
/// Given a leaf's 512-bit activity mask, returns a 27-bit mask (in the same
/// `NearestNeighborBitMask<di,dj,dk>` encoding used by the symmetric stencil) with a bit
/// set for each neighboring leaf block that the one-sided unit dilation would touch. For
/// the positive octant only the 8 offsets (di,dj,dk) in {0,1}^3 can be set; for the
/// negative octant only the 8 offsets in {-1,0}^3. Every predicate below is copied
/// verbatim from the corresponding line of `neighborMaskStencil` (Mask<3> word layout:
/// word index = local x, byte within word = local y, bit within byte = local z).
template <bool Positive>
__hostdev__ inline uint32_t
padNeighborMaskStencil(const nanovdb::Mask<3> &mask) {
    uint32_t result     = 0;
    auto words          = mask.words();
    uint64_t allWordsOr = 0;
    for (int i = 0; i < 8; i++)
        allWordsOr |= words[i];

    // Center (always included when the leaf is non-empty)
    if (allWordsOr)
        result |= NearestNeighborBitMask<0, 0, 0>::value;

    if constexpr (Positive) {
        // Faces toward +x/+y/+z
        if (words[7])
            result |= NearestNeighborBitMask<1, 0, 0>::value;
        if (allWordsOr & UINT64_C(0xff00000000000000))
            result |= NearestNeighborBitMask<0, 1, 0>::value;
        if (allWordsOr & UINT64_C(0x8080808080808080))
            result |= NearestNeighborBitMask<0, 0, 1>::value;
        // Edges
        if (words[7] & UINT64_C(0xff00000000000000))
            result |= NearestNeighborBitMask<1, 1, 0>::value;
        if (words[7] & UINT64_C(0x8080808080808080))
            result |= NearestNeighborBitMask<1, 0, 1>::value;
        if (allWordsOr & UINT64_C(0x8000000000000000))
            result |= NearestNeighborBitMask<0, 1, 1>::value;
        // Vertex
        if (words[7] & UINT64_C(0x8000000000000000))
            result |= NearestNeighborBitMask<1, 1, 1>::value;
    } else {
        // Faces toward -x/-y/-z
        if (words[0])
            result |= NearestNeighborBitMask<-1, 0, 0>::value;
        if (allWordsOr & UINT64_C(0x00000000000000ff))
            result |= NearestNeighborBitMask<0, -1, 0>::value;
        if (allWordsOr & UINT64_C(0x0101010101010101))
            result |= NearestNeighborBitMask<0, 0, -1>::value;
        // Edges
        if (words[0] & UINT64_C(0x00000000000000ff))
            result |= NearestNeighborBitMask<-1, -1, 0>::value;
        if (words[0] & UINT64_C(0x0101010101010101))
            result |= NearestNeighborBitMask<-1, 0, -1>::value;
        if (allWordsOr & UINT64_C(0x0000000000000001))
            result |= NearestNeighborBitMask<0, -1, -1>::value;
        // Vertex
        if (words[0] & UINT64_C(0x0000000000000001))
            result |= NearestNeighborBitMask<-1, -1, -1>::value;
    }
    return result;
}

// ---------------------------------------------------------------------------------------
// Internal-node functor
// ---------------------------------------------------------------------------------------

/// @brief One-sided variant of `nanovdb::util::morphology::cuda::DilateInternalNodesFunctor`.
///
/// This is a verbatim copy of that functor's body, with a single change: the per-leaf
/// spill stencil is computed with `padNeighborMaskStencil<Positive>` instead of the
/// symmetric `neighborMaskStencil<NN_FACE_EDGE_VERTEX>`. Feeding a one-sided stencil into
/// the (unchanged) 27-way MaskShift redistribution and scatter produces exactly the
/// one-sided dilated topology; the extra MaskShift calls operate on always-zero offset
/// masks and contribute nothing.
template <class BuildT, bool Positive> struct PadInternalNodesFunctor {
    // Intended to be called via nanovdb::util::cuda::operatorKernel
    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int WarpsPerBlock              = MaxThreadsPerBlock >> 5;
    static constexpr int SlicesPerLowerNode         = 8;
    static constexpr int LeafNodesPerSlice          = 4096 / SlicesPerLowerNode;

    void __device__
    operator()(const NanoGrid<BuildT> *srcGrid,
               const NanoRoot<BuildT> *dilatedRoot,
               void *upperMasks_,
               void *lowerMasks_) {
        int tID            = threadIdx.x;
        int lowerID        = blockIdx.x;
        int sliceID        = blockIdx.y;
        int threadInWarpID = threadIdx.x & 0x1f;
        int warpID         = threadIdx.x >> 5;

        using UpperMaskArrayT = Mask<5> *;
        using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
        auto upperMasks       = static_cast<UpperMaskArrayT>(upperMasks_);
        auto lowerMasks       = static_cast<LowerMaskArrayT>(lowerMasks_);

        using LowerMaskT        = Mask<4>;
        using LowerMaskStencilT = LowerMaskT(&)[3][3][3];
        __shared__ uint64_t sOffsetMasksRaw[LowerMaskT::WORD_COUNT * 27];
        __shared__ uint64_t sNeighborMasksRaw[LowerMaskT::WORD_COUNT * 27];
        auto sOffsetMasks   = reinterpret_cast<LowerMaskStencilT>(sOffsetMasksRaw[0]);
        auto sNeighborMasks = reinterpret_cast<LowerMaskStencilT>(sNeighborMasksRaw[0]);

        using WarpReduce = cub::WarpReduce<uint32_t>;
        __shared__ typename WarpReduce::TempStorage temp_storage[WarpsPerBlock];

        // TODO: Use all available threads
        if (tID < LowerMaskT::WORD_COUNT)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++) {
                        const_cast<uint64_t *>(sOffsetMasks[i][j][k].words())[tID]   = 0;
                        const_cast<uint64_t *>(sNeighborMasks[i][j][k].words())[tID] = 0;
                    }
        __syncthreads();

        const auto &srcTree = srcGrid->tree();
        const auto &lower   = srcTree.template getFirstNode<1>()[lowerID];
        auto &valueMask     = const_cast<LowerMaskT &>(lower.valueMask());

        for (std::size_t jj = sliceID * LeafNodesPerSlice; jj < (sliceID + 1) * LeafNodesPerSlice;
             jj += MaxThreadsPerBlock) {
            // Compute the mask of affected lower nodes in packed uint32_t format
            uint32_t neighborMask = 0;
            if (lower.childMask().isOn(jj + tID)) {
                auto &leaf   = *lower.data()->getChild(jj + tID);
                neighborMask = padNeighborMaskStencil<Positive>(leaf.valueMask());
            }

            // Combine information from LeafNodes processed into an offset mask
            for (int bit = 0; bit < 27; bit++) {
                uint32_t mask = (neighborMask & (1u << bit)) ? (1u << threadInWarpID) : 0;
                mask = WarpReduce(temp_storage[warpID]).Sum(mask); // Really a bitwise or, but since
                                                                   // the inputs have disjoint bits
                                                                   // set, a sum is equivalent
                auto warpMaskPtr =
                    reinterpret_cast<uint32_t *>(sOffsetMasks[0][0][bit].words()) + (jj >> 5);
                // Do we need to guard this ??
                if (threadInWarpID == 0)
                    warpMaskPtr[warpID] = mask;
                __syncthreads();
            }
        }

        // Compute neighbor masks from offset masks
        // This version is optimized for 128 threads (and requires at least that many)
        if (warpID == 0) {
            // Contribution to mask of own lower node
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,1,1)
            MaskShift<1, 1, 1>(sOffsetMasks[0][0][0], sNeighborMasks[1][1][1]);
            MaskShift<1, 1, 0>(sOffsetMasks[0][0][1], sNeighborMasks[1][1][1]);
            MaskShift<1, 1, -1>(sOffsetMasks[0][0][2], sNeighborMasks[1][1][1]);
            MaskShift<1, 0, 1>(sOffsetMasks[0][1][0], sNeighborMasks[1][1][1]);
            MaskShift<1, 0, 0>(sOffsetMasks[0][1][1], sNeighborMasks[1][1][1]);
            MaskShift<1, 0, -1>(sOffsetMasks[0][1][2], sNeighborMasks[1][1][1]);
            MaskShift<1, -1, 1>(sOffsetMasks[0][2][0], sNeighborMasks[1][1][1]);
            MaskShift<1, -1, 0>(sOffsetMasks[0][2][1], sNeighborMasks[1][1][1]);
            MaskShift<1, -1, -1>(sOffsetMasks[0][2][2], sNeighborMasks[1][1][1]);
            MaskShift<0, 1, 1>(sOffsetMasks[1][0][0], sNeighborMasks[1][1][1]);
            MaskShift<0, 1, 0>(sOffsetMasks[1][0][1], sNeighborMasks[1][1][1]);
            MaskShift<0, 1, -1>(sOffsetMasks[1][0][2], sNeighborMasks[1][1][1]);
            MaskShift<0, 0, 1>(sOffsetMasks[1][1][0], sNeighborMasks[1][1][1]);
            MaskShift<0, 0, 0>(sOffsetMasks[1][1][1], sNeighborMasks[1][1][1]);
            MaskShift<0, 0, -1>(sOffsetMasks[1][1][2], sNeighborMasks[1][1][1]);
            MaskShift<0, -1, 1>(sOffsetMasks[1][2][0], sNeighborMasks[1][1][1]);
            MaskShift<0, -1, 0>(sOffsetMasks[1][2][1], sNeighborMasks[1][1][1]);
            MaskShift<0, -1, -1>(sOffsetMasks[1][2][2], sNeighborMasks[1][1][1]);
            MaskShift<-1, 1, 1>(sOffsetMasks[2][0][0], sNeighborMasks[1][1][1]);
            MaskShift<-1, 1, 0>(sOffsetMasks[2][0][1], sNeighborMasks[1][1][1]);
            MaskShift<-1, 1, -1>(sOffsetMasks[2][0][2], sNeighborMasks[1][1][1]);
            MaskShift<-1, 0, 1>(sOffsetMasks[2][1][0], sNeighborMasks[1][1][1]);
            MaskShift<-1, 0, 0>(sOffsetMasks[2][1][1], sNeighborMasks[1][1][1]);
            MaskShift<-1, 0, -1>(sOffsetMasks[2][1][2], sNeighborMasks[1][1][1]);
            MaskShift<-1, -1, 1>(sOffsetMasks[2][2][0], sNeighborMasks[1][1][1]);
            MaskShift<-1, -1, 0>(sOffsetMasks[2][2][1], sNeighborMasks[1][1][1]);
            MaskShift<-1, -1, -1>(sOffsetMasks[2][2][2], sNeighborMasks[1][1][1]);
            // Contribution to mask of lower node at offset (-1,-1,-1)
            MaskShift<-15, -15, -15>(sOffsetMasks[0][0][0], sNeighborMasks[0][0][0]);
            // Contribution to mask of lower node at offset (-1,-1,1)
            MaskShift<-15, -15, 15>(sOffsetMasks[0][0][2], sNeighborMasks[0][0][2]);
            // Contribution to mask of lower node at offset (-1,1,-1)
            MaskShift<-15, 15, -15>(sOffsetMasks[0][2][0], sNeighborMasks[0][2][0]);
            // Contribution to mask of lower node at offset (-1,1,1)
            MaskShift<-15, 15, 15>(sOffsetMasks[0][2][2], sNeighborMasks[0][2][2]);
            // Contribution to mask of lower node at offset (1,-1,-1)
            MaskShift<15, -15, -15>(sOffsetMasks[2][0][0], sNeighborMasks[2][0][0]);
        }

        if (warpID == 1) {
            // Contribution to mask of lower node at offset (0,0,-1)
            MaskShift<1, 1, -15>(sOffsetMasks[0][0][0], sNeighborMasks[1][1][0]);
            MaskShift<1, 0, -15>(sOffsetMasks[0][1][0], sNeighborMasks[1][1][0]);
            MaskShift<1, -1, -15>(sOffsetMasks[0][2][0], sNeighborMasks[1][1][0]);
            MaskShift<0, 1, -15>(sOffsetMasks[1][0][0], sNeighborMasks[1][1][0]);
            MaskShift<0, 0, -15>(sOffsetMasks[1][1][0], sNeighborMasks[1][1][0]);
            MaskShift<0, -1, -15>(sOffsetMasks[1][2][0], sNeighborMasks[1][1][0]);
            MaskShift<-1, 1, -15>(sOffsetMasks[2][0][0], sNeighborMasks[1][1][0]);
            MaskShift<-1, 0, -15>(sOffsetMasks[2][1][0], sNeighborMasks[1][1][0]);
            MaskShift<-1, -1, -15>(sOffsetMasks[2][2][0], sNeighborMasks[1][1][0]);
            // Contribution to mask of lower node at offset (0,0,1)
            MaskShift<1, 1, 15>(sOffsetMasks[0][0][2], sNeighborMasks[1][1][2]);
            MaskShift<1, 0, 15>(sOffsetMasks[0][1][2], sNeighborMasks[1][1][2]);
            MaskShift<1, -1, 15>(sOffsetMasks[0][2][2], sNeighborMasks[1][1][2]);
            MaskShift<0, 1, 15>(sOffsetMasks[1][0][2], sNeighborMasks[1][1][2]);
            MaskShift<0, 0, 15>(sOffsetMasks[1][1][2], sNeighborMasks[1][1][2]);
            MaskShift<0, -1, 15>(sOffsetMasks[1][2][2], sNeighborMasks[1][1][2]);
            MaskShift<-1, 1, 15>(sOffsetMasks[2][0][2], sNeighborMasks[1][1][2]);
            MaskShift<-1, 0, 15>(sOffsetMasks[2][1][2], sNeighborMasks[1][1][2]);
            MaskShift<-1, -1, 15>(sOffsetMasks[2][2][2], sNeighborMasks[1][1][2]);
            // Contribution to mask of lower node at offset (-1,-1,0)
            MaskShift<-15, -15, 1>(sOffsetMasks[0][0][0], sNeighborMasks[0][0][1]);
            MaskShift<-15, -15, 0>(sOffsetMasks[0][0][1], sNeighborMasks[0][0][1]);
            MaskShift<-15, -15, -1>(sOffsetMasks[0][0][2], sNeighborMasks[0][0][1]);
            // Contribution to mask of lower node at offset (-1,1,0)
            MaskShift<-15, 15, 1>(sOffsetMasks[0][2][0], sNeighborMasks[0][2][1]);
            MaskShift<-15, 15, 0>(sOffsetMasks[0][2][1], sNeighborMasks[0][2][1]);
            MaskShift<-15, 15, -1>(sOffsetMasks[0][2][2], sNeighborMasks[0][2][1]);
            // Contribution to mask of lower node at offset (1,-1,0)
            MaskShift<15, -15, 1>(sOffsetMasks[2][0][0], sNeighborMasks[2][0][1]);
            MaskShift<15, -15, 0>(sOffsetMasks[2][0][1], sNeighborMasks[2][0][1]);
            MaskShift<15, -15, -1>(sOffsetMasks[2][0][2], sNeighborMasks[2][0][1]);
            // Contribution to mask of lower node at offset (1,1,0)
            MaskShift<15, 15, 1>(sOffsetMasks[2][2][0], sNeighborMasks[2][2][1]);
            MaskShift<15, 15, 0>(sOffsetMasks[2][2][1], sNeighborMasks[2][2][1]);
            MaskShift<15, 15, -1>(sOffsetMasks[2][2][2], sNeighborMasks[2][2][1]);
            // Contribution to mask of lower node at offset (1,-1,1)
            MaskShift<15, -15, 15>(sOffsetMasks[2][0][2], sNeighborMasks[2][0][2]);
        }

        if (warpID == 2) {
            // Contribution to mask of lower node at offset (0,-1,0)
            MaskShift<1, -15, 1>(sOffsetMasks[0][0][0], sNeighborMasks[1][0][1]);
            MaskShift<1, -15, 0>(sOffsetMasks[0][0][1], sNeighborMasks[1][0][1]);
            MaskShift<1, -15, -1>(sOffsetMasks[0][0][2], sNeighborMasks[1][0][1]);
            MaskShift<0, -15, 1>(sOffsetMasks[1][0][0], sNeighborMasks[1][0][1]);
            MaskShift<0, -15, 0>(sOffsetMasks[1][0][1], sNeighborMasks[1][0][1]);
            MaskShift<0, -15, -1>(sOffsetMasks[1][0][2], sNeighborMasks[1][0][1]);
            MaskShift<-1, -15, 1>(sOffsetMasks[2][0][0], sNeighborMasks[1][0][1]);
            MaskShift<-1, -15, 0>(sOffsetMasks[2][0][1], sNeighborMasks[1][0][1]);
            MaskShift<-1, -15, -1>(sOffsetMasks[2][0][2], sNeighborMasks[1][0][1]);
            // Contribution to mask of lower node at offset (0,1,0)
            MaskShift<1, 15, 1>(sOffsetMasks[0][2][0], sNeighborMasks[1][2][1]);
            MaskShift<1, 15, 0>(sOffsetMasks[0][2][1], sNeighborMasks[1][2][1]);
            MaskShift<1, 15, -1>(sOffsetMasks[0][2][2], sNeighborMasks[1][2][1]);
            MaskShift<0, 15, 1>(sOffsetMasks[1][2][0], sNeighborMasks[1][2][1]);
            MaskShift<0, 15, 0>(sOffsetMasks[1][2][1], sNeighborMasks[1][2][1]);
            MaskShift<0, 15, -1>(sOffsetMasks[1][2][2], sNeighborMasks[1][2][1]);
            MaskShift<-1, 15, 1>(sOffsetMasks[2][2][0], sNeighborMasks[1][2][1]);
            MaskShift<-1, 15, 0>(sOffsetMasks[2][2][1], sNeighborMasks[1][2][1]);
            MaskShift<-1, 15, -1>(sOffsetMasks[2][2][2], sNeighborMasks[1][2][1]);
            // Contribution to mask of lower node at offset (-1,0,-1)
            MaskShift<-15, 1, -15>(sOffsetMasks[0][0][0], sNeighborMasks[0][1][0]);
            MaskShift<-15, 0, -15>(sOffsetMasks[0][1][0], sNeighborMasks[0][1][0]);
            MaskShift<-15, -1, -15>(sOffsetMasks[0][2][0], sNeighborMasks[0][1][0]);
            // Contribution to mask of lower node at offset (-1,0,1)
            MaskShift<-15, 1, 15>(sOffsetMasks[0][0][2], sNeighborMasks[0][1][2]);
            MaskShift<-15, 0, 15>(sOffsetMasks[0][1][2], sNeighborMasks[0][1][2]);
            MaskShift<-15, -1, 15>(sOffsetMasks[0][2][2], sNeighborMasks[0][1][2]);
            // Contribution to mask of lower node at offset (1,0,-1)
            MaskShift<15, 1, -15>(sOffsetMasks[2][0][0], sNeighborMasks[2][1][0]);
            MaskShift<15, 0, -15>(sOffsetMasks[2][1][0], sNeighborMasks[2][1][0]);
            MaskShift<15, -1, -15>(sOffsetMasks[2][2][0], sNeighborMasks[2][1][0]);
            // Contribution to mask of lower node at offset (1,0,1)
            MaskShift<15, 1, 15>(sOffsetMasks[2][0][2], sNeighborMasks[2][1][2]);
            MaskShift<15, 0, 15>(sOffsetMasks[2][1][2], sNeighborMasks[2][1][2]);
            MaskShift<15, -1, 15>(sOffsetMasks[2][2][2], sNeighborMasks[2][1][2]);
            // Contribution to mask of lower node at offset (1,1,-1)
            MaskShift<15, 15, -15>(sOffsetMasks[2][2][0], sNeighborMasks[2][2][0]);
        }

        if (warpID == 3) {
            // Contribution to mask of lower node at offset (-1,0,0)
            MaskShift<-15, 1, 1>(sOffsetMasks[0][0][0], sNeighborMasks[0][1][1]);
            MaskShift<-15, 1, 0>(sOffsetMasks[0][0][1], sNeighborMasks[0][1][1]);
            MaskShift<-15, 1, -1>(sOffsetMasks[0][0][2], sNeighborMasks[0][1][1]);
            MaskShift<-15, 0, 1>(sOffsetMasks[0][1][0], sNeighborMasks[0][1][1]);
            MaskShift<-15, 0, 0>(sOffsetMasks[0][1][1], sNeighborMasks[0][1][1]);
            MaskShift<-15, 0, -1>(sOffsetMasks[0][1][2], sNeighborMasks[0][1][1]);
            MaskShift<-15, -1, 1>(sOffsetMasks[0][2][0], sNeighborMasks[0][1][1]);
            MaskShift<-15, -1, 0>(sOffsetMasks[0][2][1], sNeighborMasks[0][1][1]);
            MaskShift<-15, -1, -1>(sOffsetMasks[0][2][2], sNeighborMasks[0][1][1]);
            // Contribution to mask of lower node at offset (1,0,0)
            MaskShift<15, 1, 1>(sOffsetMasks[2][0][0], sNeighborMasks[2][1][1]);
            MaskShift<15, 1, 0>(sOffsetMasks[2][0][1], sNeighborMasks[2][1][1]);
            MaskShift<15, 1, -1>(sOffsetMasks[2][0][2], sNeighborMasks[2][1][1]);
            MaskShift<15, 0, 1>(sOffsetMasks[2][1][0], sNeighborMasks[2][1][1]);
            MaskShift<15, 0, 0>(sOffsetMasks[2][1][1], sNeighborMasks[2][1][1]);
            MaskShift<15, 0, -1>(sOffsetMasks[2][1][2], sNeighborMasks[2][1][1]);
            MaskShift<15, -1, 1>(sOffsetMasks[2][2][0], sNeighborMasks[2][1][1]);
            MaskShift<15, -1, 0>(sOffsetMasks[2][2][1], sNeighborMasks[2][1][1]);
            MaskShift<15, -1, -1>(sOffsetMasks[2][2][2], sNeighborMasks[2][1][1]);
            // Contribution to mask of lower node at offset (0,-1,-1)
            MaskShift<1, -15, -15>(sOffsetMasks[0][0][0], sNeighborMasks[1][0][0]);
            MaskShift<0, -15, -15>(sOffsetMasks[1][0][0], sNeighborMasks[1][0][0]);
            MaskShift<-1, -15, -15>(sOffsetMasks[2][0][0], sNeighborMasks[1][0][0]);
            // Contribution to mask of lower node at offset (0,-1,1)
            MaskShift<1, -15, 15>(sOffsetMasks[0][0][2], sNeighborMasks[1][0][2]);
            MaskShift<0, -15, 15>(sOffsetMasks[1][0][2], sNeighborMasks[1][0][2]);
            MaskShift<-1, -15, 15>(sOffsetMasks[2][0][2], sNeighborMasks[1][0][2]);
            // Contribution to mask of lower node at offset (0,1,-1)
            MaskShift<1, 15, -15>(sOffsetMasks[0][2][0], sNeighborMasks[1][2][0]);
            MaskShift<0, 15, -15>(sOffsetMasks[1][2][0], sNeighborMasks[1][2][0]);
            MaskShift<-1, 15, -15>(sOffsetMasks[2][2][0], sNeighborMasks[1][2][0]);
            // Contribution to mask of lower node at offset (0,1,1)
            MaskShift<1, 15, 15>(sOffsetMasks[0][2][2], sNeighborMasks[1][2][2]);
            MaskShift<0, 15, 15>(sOffsetMasks[1][2][2], sNeighborMasks[1][2][2]);
            MaskShift<-1, 15, 15>(sOffsetMasks[2][2][2], sNeighborMasks[1][2][2]);
            // Contribution to mask of lower node at offset (1,1,1)
            MaskShift<15, 15, 15>(sOffsetMasks[2][2][2], sNeighborMasks[2][2][2]);
        }

        __syncthreads();

        // Compose contributions to the lower-node masks of the dilated tree
        for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
                for (int dk = -1; dk <= 1; dk++) {
                    int neighborID = (di + 1) * 9 + (dj + 1) * 3 + dk + 1;
                    if ((neighborID % WarpsPerBlock) == warpID) {
                        auto neighborOrigin = lower.origin().offsetBy(di * 128, dj * 128, dk * 128);
                        auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(neighborOrigin);
                        auto &neighborMask   = sNeighborMasks[di + 1][dj + 1][dk + 1];

                        for (int tOffset = 0; tOffset < Mask<4>::WORD_COUNT; tOffset += 32) {
                            unsigned long long int computedWord =
                                neighborMask.words()[threadInWarpID + tOffset];
                            if (computedWord) {
                                auto dilatedTile = dilatedRoot->probeTile(neighborOrigin);
                                uint64_t tileChildIndex =
                                    util::PtrDiff(dilatedTile, dilatedRoot->tile(0)) /
                                    sizeof(NanoRoot<BuildT>::Tile);
                                auto &outputUpperMask = upperMasks[tileChildIndex];
                                outputUpperMask.setOnAtomic(upperChildIndex);
                                auto &outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                                util::atomicOr(
                                    const_cast<uint64_t *>(
                                        &outputLowerMask.words()[threadInWarpID + tOffset]),
                                    static_cast<uint64_t>(computedWord));
                            }
                        }
                    }
                }
        __syncthreads();
    }
};

// ---------------------------------------------------------------------------------------
// Leaf-node functor
// ---------------------------------------------------------------------------------------

/// @brief One-sided variant of `DilateLeafNodesFunctor<..., NN_FACE_EDGE_VERTEX>`.
///
/// For each destination leaf, gathers the (up to) 8 source leaves in the octant's block
/// neighborhood and OR-shifts the 512-bit activity masks in a single octant, realizing the
/// Minkowski sum S (+) {0,1}^3 (positive) or S (+) {-1,0}^3 (negative). The per-axis shift
/// expressions are exactly the positive-direction terms of `DilateLeafNodesFunctor`
/// (positive octant) or its negative-direction terms (negative octant).
template <class BuildT, bool Positive> struct PadLeafNodesFunctor {
    // Intended to be called via nanovdb::util::cuda::operatorKernel
    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int SlicesPerLowerNode         = 8;
    static constexpr int LeafNodesPerSlice          = 4096 / SlicesPerLowerNode;

    __device__ void
    operator()(const NanoGrid<BuildT> *srcGrid, NanoGrid<BuildT> *dstGrid) {
        int tID     = threadIdx.x;
        int lowerID = blockIdx.x;
        int sliceID = blockIdx.y;

        const auto &srcTree  = srcGrid->tree();
        const auto &dstTree  = dstGrid->tree();
        const auto &dstLower = dstTree.template getFirstNode<1>()[lowerID];
        for (std::size_t jj = sliceID * LeafNodesPerSlice; jj < (sliceID + 1) * LeafNodesPerSlice;
             jj += MaxThreadsPerBlock)
            if (dstLower.childMask().isOn(jj + tID)) {
                auto &dstLeaf         = *dstLower.data()->getChild(jj + tID);
                const auto leafOrigin = dstLeaf.origin();

                uint64_t originalWordsShifted[10][3][3] = {};
                using WordStencilT  = uint64_t (&)[10][3][3]; // [x-voxel offset][y-block][z-block]
                auto &originalWords = reinterpret_cast<WordStencilT>(
                    originalWordsShifted[1][1][1]);           // logical range [-1,8][-1,1][-1,1]

                // Gather source leaves in the octant's 2x2x2 block neighborhood:
                //   positive octant -> blocks at offsets {-1,0}^3 (content flows +dir)
                //   negative octant -> blocks at offsets { 0,1}^3 (content flows -dir)
                constexpr int blockLo = Positive ? -1 : 0;
                constexpr int blockHi = Positive ? 0 : 1;
                for (int dBi = blockLo; dBi <= blockHi; dBi++)
                    for (int dBj = blockLo; dBj <= blockHi; dBj++)
                        for (int dBk = blockLo; dBk <= blockHi; dBk++) {
                            auto neighborOrigin = leafOrigin.offsetBy(dBi * 8, dBj * 8, dBk * 8);
                            if (auto neighborLeafPtr = srcTree.root().probeLeaf(neighborOrigin)) {
                                auto neighborWords = neighborLeafPtr->valueMask().words();
                                if (dBi == -1)
                                    originalWords[-1][dBj][dBk] = neighborWords[7];
                                else if (dBi == 1)
                                    originalWords[8][dBj][dBk] = neighborWords[0];
                                else
                                    for (int i = 0; i < 8; i++)
                                        originalWords[i][dBj][dBk] = neighborWords[i];
                            }
                        }

                if constexpr (Positive) {
                    // Pad +z: for each block word, OR in the copy shifted one voxel toward +z
                    for (int i = -1; i <= 8; i++)
                        for (int dBj = -1; dBj <= 0; dBj++) {
                            uint64_t w = originalWords[i][dBj][0];
                            w |= (originalWords[i][dBj][0] & 0x7f7f7f7f7f7f7f7fUL) << 1;
                            w |= (originalWords[i][dBj][-1] & 0x8080808080808080UL) >> 7;
                            originalWords[i][dBj][0] = w;
                        }
                    // Pad +y
                    for (int i = -1; i <= 8; i++) {
                        uint64_t w = originalWords[i][0][0];
                        w |= (originalWords[i][0][0] & 0x00ffffffffffffffUL) << 8;
                        w |= (originalWords[i][-1][0] & 0xff00000000000000UL) >> 56;
                        originalWords[i][0][0] = w;
                    }
                    // Pad +x
                    auto paddedWords = const_cast<Mask<3> &>(dstLeaf.valueMask()).words();
                    for (int i = 0; i <= 7; i++)
                        paddedWords[i] = originalWords[i][0][0] | originalWords[i - 1][0][0];
                } else {
                    // Pad -z
                    for (int i = -1; i <= 8; i++)
                        for (int dBj = 0; dBj <= 1; dBj++) {
                            uint64_t w = originalWords[i][dBj][0];
                            w |= (originalWords[i][dBj][0] & 0xfefefefefefefefeUL) >> 1;
                            w |= (originalWords[i][dBj][1] & 0x0101010101010101UL) << 7;
                            originalWords[i][dBj][0] = w;
                        }
                    // Pad -y
                    for (int i = -1; i <= 8; i++) {
                        uint64_t w = originalWords[i][0][0];
                        w |= (originalWords[i][0][0] & 0xffffffffffffff00UL) >> 8;
                        w |= (originalWords[i][1][0] & 0x00000000000000ffUL) << 56;
                        originalWords[i][0][0] = w;
                    }
                    // Pad -x
                    auto paddedWords = const_cast<Mask<3> &>(dstLeaf.valueMask()).words();
                    for (int i = 0; i <= 7; i++)
                        paddedWords[i] = originalWords[i][0][0] | originalWords[i + 1][0][0];
                }
            }
        return;
    }
};

// ---------------------------------------------------------------------------------------
// Erosion keep-mask functor (for exclude-border padding)
// ---------------------------------------------------------------------------------------

/// @brief Computes, for each source leaf, the 512-bit "keep" mask of the one-sided
///        erosion S (-) {0,1}^3 (positive) or S (-) {-1,0}^3 (negative), i.e. keep(c) is
///        set iff every voxel c+d, d in the octant, is active in the source grid.
///
/// The result is written to a per-source-leaf `Mask<3>` sidecar suitable for feeding
/// `nanovdb::tools::cuda::PruneGrid`. Since the octant always contains the origin, the
/// keep mask is a subset of the source activity mask, so PruneGrid reproduces it exactly.
/// This matches the semantics of `paddedIJKForGridWithoutBorder`: a voxel survives iff its
/// entire [bmin,bmax]^3 box neighborhood is active. Intended for lambdaKernel dispatch
/// (one thread per source leaf).
template <typename BuildT, bool Positive> struct ErodeKeepMaskFunctor {
    __device__ void
    operator()(size_t leafID, const NanoGrid<BuildT> *srcGrid, Mask<3> *d_keepMasks) {
        const auto &srcTree   = srcGrid->tree();
        const auto &leaf      = srcTree.template getFirstNode<0>()[leafID];
        const auto leafOrigin = leaf.origin();

        uint64_t originalWordsShifted[10][3][3] = {};
        using WordStencilT                      = uint64_t (&)[10][3][3];
        auto &originalWords = reinterpret_cast<WordStencilT>(originalWordsShifted[1][1][1]);

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
                            originalWords[-1][dBj][dBk] = neighborWords[7];
                        else if (dBi == 1)
                            originalWords[8][dBj][dBk] = neighborWords[0];
                        else
                            for (int i = 0; i < 8; i++)
                                originalWords[i][dBj][dBk] = neighborWords[i];
                    }
                }

        auto keepWords = d_keepMasks[leafID].words();
        if constexpr (Positive) {
            // Erode toward +z: keep(c) &= src(c+(0,0,1))
            for (int i = 0; i <= 8; i++)
                for (int dBj = 0; dBj <= 1; dBj++) {
                    uint64_t w = originalWords[i][dBj][0];
                    w &= ((originalWords[i][dBj][0] & 0xfefefefefefefefeUL) >> 1) |
                         ((originalWords[i][dBj][1] & 0x0101010101010101UL) << 7);
                    originalWords[i][dBj][0] = w;
                }
            // Erode toward +y
            for (int i = 0; i <= 8; i++) {
                uint64_t w = originalWords[i][0][0];
                w &= ((originalWords[i][0][0] & 0xffffffffffffff00UL) >> 8) |
                     ((originalWords[i][1][0] & 0x00000000000000ffUL) << 56);
                originalWords[i][0][0] = w;
            }
            // Erode toward +x
            for (int i = 0; i <= 7; i++)
                keepWords[i] = originalWords[i][0][0] & originalWords[i + 1][0][0];
        } else {
            // Erode toward -z: keep(c) &= src(c+(0,0,-1))
            for (int i = -1; i <= 7; i++)
                for (int dBj = -1; dBj <= 0; dBj++) {
                    uint64_t w = originalWords[i][dBj][0];
                    w &= ((originalWords[i][dBj][0] & 0x7f7f7f7f7f7f7f7fUL) << 1) |
                         ((originalWords[i][dBj][-1] & 0x8080808080808080UL) >> 7);
                    originalWords[i][dBj][0] = w;
                }
            // Erode toward -y
            for (int i = -1; i <= 7; i++) {
                uint64_t w = originalWords[i][0][0];
                w &= ((originalWords[i][0][0] & 0x00ffffffffffffffUL) << 8) |
                     ((originalWords[i][-1][0] & 0xff00000000000000UL) >> 56);
                originalWords[i][0][0] = w;
            }
            // Erode toward -x
            for (int i = 0; i <= 7; i++)
                keepWords[i] = originalWords[i][0][0] & originalWords[i - 1][0][0];
        }
    }
};

// ---------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------

/// @brief One unit-octant padding pass over an OnIndex grid, driving TopologyBuilder.
///        Modeled on `nanovdb::tools::cuda::DilateGrid`; the driver, root speculation and
///        the TopologyBuilder pipeline are reused as-is, with the internal-node and
///        leaf-node stages swapped for their one-sided (`Positive`-selected) variants.
template <typename BuildT> class PadGrid {
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

  public:
    /// @param d_srcGrid      source device grid to be padded
    /// @param positiveOctant true -> pad by {0,1}^3, false -> pad by {-1,0}^3
    /// @param stream         optional CUDA stream
    PadGrid(const GridT *d_srcGrid, bool positiveOctant, cudaStream_t stream = 0)
        : mBuilder(stream), mStream(stream), mDeviceSrcGrid(d_srcGrid), mPositive(positiveOctant) {}

    void
    setChecksum(CheckMode mode = CheckMode::Disable) {
        mBuilder.mChecksum = mode;
    }

    template <typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> getHandle(const BufferT &buffer = BufferT());

  private:
    void padRoot();
    void padInternalNodes();
    void processGridTreeRoot();
    void padLeafNodes();

    tools::cuda::TopologyBuilder<BuildT> mBuilder;
    cudaStream_t mStream{0};
    const GridT *mDeviceSrcGrid;
    bool mPositive;
    TreeData mSrcTreeData;
};

template <typename BuildT>
template <typename BufferT>
GridHandle<BufferT>
PadGrid<BuildT>::getHandle(const BufferT &pool) {
    // Copy TreeData from GPU -> CPU
    cudaStreamSynchronize(mStream);
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid);

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively expand root node, allocate internal masks, dilate internal nodes
    padRoot();
    mBuilder.allocateInternalMaskBuffers(mStream);
    padInternalNodes();

    // Enumerate tree nodes and allocate the destination grid buffer
    mBuilder.countNodes(mStream);
    cudaStreamSynchronize(mStream);
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process Grid/Tree/Root, then the internal nodes of the padded result
    processGridTreeRoot();
    mBuilder.processUpperNodes(mStream);
    mBuilder.processLowerNodes(mStream);

    // Pad the leaf active masks into the new topology, then finalize
    padLeafNodes();
    mBuilder.processBBox(mStream);
    mBuilder.postProcessGridTree(mStream);

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}

template <typename BuildT>
void
PadGrid<BuildT>::padRoot() {
    // Conservatively and speculatively expands the root tile table to accommodate any new
    // root nodes introduced by the padding. This mirrors `DilateGrid::dilateRoot` verbatim
    // (a symmetric 26-connected speculation): although a one-sided pass only spills into
    // one octant, the symmetric speculation is a strict superset, so it guarantees every
    // tile the internal-node scatter might touch already exists. Speculatively introduced
    // tiles that end up empty are pruned by TopologyBuilder::countNodes.
    int device = 0;
    cudaGetDevice(&device);

    std::map<uint64_t, typename RootT::DataType::Tile> dilatedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid (and DilateGrid); it is what
    // makes the enumerated voxel order match the coordinate-list path this replaces.
    auto coordToKey = [](const Coord &ijk) -> uint64_t {
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)) |
               (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) |
               (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);
    };

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        auto deviceSrcRoot = static_cast<const RootT *>(
            util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        uint64_t rootAndUpperSize  = mSrcTreeData.mNodeOffset[1] - mSrcTreeData.mNodeOffset[3];
        auto srcRootAndUpperBuffer = nanovdb::HostBuffer::create(rootAndUpperSize);
        cudaCheck(cudaMemcpyAsync(srcRootAndUpperBuffer.data(),
                                  deviceSrcRoot,
                                  rootAndUpperSize,
                                  cudaMemcpyDeviceToHost,
                                  mStream));
        auto srcRootAndUpper = static_cast<RootT *>(srcRootAndUpperBuffer.data());

        for (uint32_t t = 0; t < srcRootAndUpper->tileCount(); t++) {
            auto srcUpper          = srcRootAndUpper->getChild(srcRootAndUpper->tile(t));
            const auto dilatedBBox = srcUpper->bbox().expandBy(1);

            static constexpr int32_t rootTileDim = UpperT::DIM; // 4096
            for (int di = -rootTileDim; di <= rootTileDim; di += rootTileDim)
                for (int dj = -rootTileDim; dj <= rootTileDim; dj += rootTileDim)
                    for (int dk = -rootTileDim; dk <= rootTileDim; dk += rootTileDim) {
                        auto testBBox = nanovdb::CoordBBox::createCube(
                            srcUpper->origin().offsetBy(di, dj, dk), rootTileDim);
                        auto sortKey = coordToKey(testBBox.min());
                        auto tileKey = RootT::CoordToKey(testBBox.min());
                        if (testBBox.hasOverlap(dilatedBBox) & (dilatedTiles.count(sortKey) == 0)) {
                            typename RootT::Tile neighborTile{tileKey};
                            dilatedTiles.emplace(sortKey, neighborTile);
                        }
                    }
        }
    }

    uint64_t rootSize          = RootT::memUsage(dilatedTiles.size());
    mBuilder.mProcessedRoot    = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto dilatedRootPtr        = static_cast<RootT *>(mBuilder.mProcessedRoot.data());
    dilatedRootPtr->mTableSize = dilatedTiles.size();
    uint32_t t                 = 0;
    for (const auto &[key, tile]: dilatedTiles)
        *dilatedRootPtr->tile(t++) = tile;
    mBuilder.mProcessedRoot.deviceUpload(device, mStream, false);
}

template <typename BuildT>
void
PadGrid<BuildT>::padInternalNodes() {
    if (mSrcTreeData.mNodeCount[1]) { // Unless it's an empty grid
        if (mPositive) {
            using Op = PadInternalNodesFunctor<BuildT, true>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1], Op::SlicesPerLowerNode, 1),
                   Op::MaxThreadsPerBlock,
                   0,
                   mStream>>>(mDeviceSrcGrid,
                              mBuilder.deviceProcessedRoot(),
                              mBuilder.deviceUpperMasks(),
                              mBuilder.deviceLowerMasks());
        } else {
            using Op = PadInternalNodesFunctor<BuildT, false>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1], Op::SlicesPerLowerNode, 1),
                   Op::MaxThreadsPerBlock,
                   0,
                   mStream>>>(mDeviceSrcGrid,
                              mBuilder.deviceProcessedRoot(),
                              mBuilder.deviceUpperMasks(),
                              mBuilder.deviceLowerMasks());
        }
    }
}

template <typename BuildT>
void
PadGrid<BuildT>::processGridTreeRoot() {
    // Copy GridData from source grid (duplicates grid name and map; others reset later)
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(),
                              mDeviceSrcGrid->data(),
                              GridT::memUsage(),
                              cudaMemcpyDeviceToDevice,
                              mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(
        1,
        tools::cuda::topology::detail::BuildGridTreeRootFunctor<BuildT>(),
        mBuilder.deviceData());
    cudaCheckError();
}

template <typename BuildT>
void
PadGrid<BuildT>::padLeafNodes() {
    if (mBuilder.data()->nodeCount[1]) { // Unless output grid is empty
        if (mPositive) {
            using Op = PadLeafNodesFunctor<BuildT, true>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1], Op::SlicesPerLowerNode, 1),
                   Op::MaxThreadsPerBlock,
                   0,
                   mStream>>>(mDeviceSrcGrid, static_cast<GridT *>(mBuilder.data()->d_bufferPtr));
        } else {
            using Op = PadLeafNodesFunctor<BuildT, false>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1], Op::SlicesPerLowerNode, 1),
                   Op::MaxThreadsPerBlock,
                   0,
                   mStream>>>(mDeviceSrcGrid, static_cast<GridT *>(mBuilder.data()->d_bufferPtr));
        }
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
}

} // namespace morphology
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_PADGRID_CUH
