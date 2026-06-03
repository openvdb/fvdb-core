// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_HDDAITERATORS_H
#define FVDB_DETAIL_UTILS_NANOVDB_HDDAITERATORS_H

#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>

#include <ATen/OpMathType.h>
#include <c10/util/Half.h>

#include <iostream>

namespace nanovdb {

namespace math {

template <> struct Delta<c10::Half> {
    __hostdev__ static c10::Half
    value() {
        return c10::Half(1e-3f);
    }
};

} // namespace math

} // namespace nanovdb

namespace fvdb {

template <typename AccT, typename ScalarT> struct HDDASegmentIterator {
  public:
    using BuildT       = typename AccT::BuildType;
    using MathType     = at::opmath_type<ScalarT>;
    using RayT         = nanovdb::math::Ray<ScalarT>;
    using RayTInternal = nanovdb::math::Ray<MathType>;
    using TimespanT    = typename RayTInternal::TimeSpan;
    using CoordT       = nanovdb::Coord;
    using HDDAT        = nanovdb::math::HDDA<RayTInternal, nanovdb::Coord>;

    using value_type        = TimespanT;
    using pointer           = value_type *;
    using reference         = value_type &;
    using iterator_category = std::forward_iterator_tag;

    HDDASegmentIterator() = delete;

    __hostdev__ bool
    isValid() const {
        return mTimespan.valid(0.0);
    }

    __hostdev__ const HDDASegmentIterator &
    operator++() {
        nextSegment();
        return *this;
    }

    __hostdev__ HDDASegmentIterator
    operator++(int) {
        HDDASegmentIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __hostdev__
    HDDASegmentIterator(const RayT &rayVox, const AccT &acc)
        : mAcc(acc) {
        mRay       = RayTInternal(nanovdb::math::Vec3<MathType>(rayVox.eye()),
                            nanovdb::math::Vec3<MathType>(rayVox.dir()),
                            static_cast<MathType>(rayVox.t0()),
                            static_cast<MathType>(rayVox.t1()));
        CoordT ijk = nanovdb::math::RoundDown<CoordT>(
            rayVox(mRay.t0() + nanovdb::math::Delta<ScalarT>::value()));
        mHdda.init(mRay, mAcc.getDim(ijk, mRay));
        nextSegment(); // Move to first segment
    }

    // Dereferencable.
    __hostdev__ const value_type &
    operator*() const {
        return mTimespan;
    }

    __hostdev__ const value_type *
    operator->() const {
        return (const value_type *)&mTimespan;
    }

  private:
    __hostdev__ bool
    nextSegment() {
        mTimespan.t0 = mRay.t1() + static_cast<ScalarT>(5.0);
        mTimespan.t1 = mRay.t1();
        do {
            // Re-align the HDDA to the correct level if mHdda's current dim
            // disagrees with the tree's dim at the current voxel. After
            // HDDA::update snaps mVoxel to the new grid level we have to
            // query isActive at the snapped voxel, so we always do the
            // isActive lookup *after* the dim check.
            const int dim = mAcc.getDim(mHdda.voxel(), mRay);
            if (mHdda.dim() != dim) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, dim);
            }
            const bool active = mAcc.isActive(mHdda.voxel());

            // Predicated TimeSpan writes: only the `leaving` break is a real
            // branch. The entering and leaving t0/t1 updates are expressed as
            // selects so rays in the same warp whose `active` state differs
            // don't diverge at the setter level.
            const bool wasValid = mTimespan.valid();
            const MathType t    = mHdda.time();

            mTimespan.t0       = (active && !wasValid) ? t : mTimespan.t0;
            const bool leaving = (!active && wasValid);
            mTimespan.t1       = leaving ? t : mTimespan.t1;
            if (leaving) {
                break;
            }
        } while (mHdda.step());

        if (!mTimespan.valid(0.0)) {
            mTimespan.t1 = fminf(mRay.t1(), mHdda.time());
        }
        // We didn't hit anything, return
        return mTimespan.valid(0.0);
    }

    const AccT &mAcc;
    RayTInternal mRay;
    HDDAT mHdda;
    TimespanT mTimespan;
};

// Iterates the active values an HDDA ray walk visits, yielding each as a
// {coordinate, [t0, t1]} pair. `LeafOnly` selects what counts as a value:
//
//   - LeafOnly == false: every active value at any node level, i.e. active
//     coarse tiles (dim > 1) as well as active leaf voxels (dim == 1). The
//     caller must branch on `getDim()` to tell them apart. Exposed as the
//     `HDDAActiveValueIterator` alias.
//   - LeafOnly == true: only active leaf voxels (dim == 1); active tiles are
//     skipped in a single coarse HDDA step. This mirrors the narrow-band
//     inner loop of `nanovdb::ZeroCrossing`. Exposed as the
//     `HDDALeafVoxelIterator` alias. Because every yielded value is a leaf
//     voxel, a per-voxel buffer indexed by `getValue(ijk) - 1` is always
//     in-bounds; ops with per-active-value buffers must use the active-value
//     alias and handle tiles explicitly.
//
// `LeafOnly` is a compile-time flag, so NVCC prunes the unused branch and the
// two aliases compile to distinct, fully specialised kernels.
template <typename AccT, typename ScalarT, bool LeafOnly> struct HDDAValueIteratorImpl {
    using MathType = at::opmath_type<ScalarT>;
    struct PairT {
        nanovdb::Coord first;
        typename nanovdb::math::Ray<MathType>::TimeSpan second;
    };
    using BuildT       = typename AccT::BuildType;
    using RayT         = nanovdb::math::Ray<ScalarT>;
    using RayTInternal = nanovdb::math::Ray<MathType>;
    using TimespanT    = typename RayTInternal::TimeSpan;
    using CoordT       = nanovdb::Coord;
    using HDDAT        = nanovdb::math::HDDA<RayTInternal, nanovdb::Coord>;

    using value_type        = PairT;
    using pointer           = value_type *;
    using reference         = value_type &;
    using iterator_category = std::forward_iterator_tag;

    HDDAValueIteratorImpl() = delete;

    __hostdev__
    HDDAValueIteratorImpl(const RayT &rayVox, const AccT &acc)
        : mAcc(acc) {
        mRay = RayTInternal(nanovdb::math::Vec3<MathType>(rayVox.eye()),
                            nanovdb::math::Vec3<MathType>(rayVox.dir()),
                            static_cast<MathType>(rayVox.t0()),
                            static_cast<MathType>(rayVox.t1()));

        CoordT ijk = mRay(mRay.t0() + nanovdb::math::Delta<ScalarT>::value()).floor();
        mHdda.init(mRay, mAcc.getDim(ijk, mRay));
        mIsValid = nextVoxel();
    }

    __hostdev__ bool
    isValid() const {
        return mIsValid;
    }

    __hostdev__ const value_type &
    operator*() const {
        return mData;
    }

    __hostdev__ const value_type *
    operator->() const {
        return (const value_type *)&mData;
    }

    __hostdev__ const HDDAValueIteratorImpl &
    operator++() {
        mIsValid = nextVoxel();
        return *this;
    }

    __hostdev__ HDDAValueIteratorImpl
    operator++(int) {
        HDDAValueIteratorImpl tmp = *this;
        ++(*this);
        return tmp;
    }

  private:
    __hostdev__ bool
    nextVoxel() {
        do {
            // Re-align the HDDA to the correct level. mHdda's level needs up
            // to three passes to stabilise (4096 -> 128 -> 8 -> 1) and on
            // each pass we have to re-query getDim at the snapped voxel.
            // Bound the loop at three iterations so a degenerate accessor
            // can't spin forever; in practice the loop converges in <= 3
            // passes by construction of the level hierarchy.
            int dim = mAcc.getDim(mHdda.voxel(), mRay);
            for (int pass = 0; pass < 3 && mHdda.dim() != dim; ++pass) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, dim);
                dim = mAcc.getDim(mHdda.voxel(), mRay);
            }

            // `dim` (computed by the realign loop above) is the tree's node
            // size at the current voxel: dim == 1 is a leaf voxel, dim > 1 is
            // a coarse tile. isActive returns true for both. When LeafOnly,
            // skip tiles so we yield only leaf voxels; the trailing
            // mHdda.step() then jumps over the whole tile in one coarse step.
            // When !LeafOnly the gate collapses to the original isActive check
            // (no extra cost — dim is already in hand).
            const bool isLeaf = (dim == 1);
            if ((!LeafOnly || isLeaf) && mAcc.isActive(mHdda.voxel())) {
                mData = {mHdda.voxel(), TimespanT(mHdda.time(), mHdda.next())};
                mHdda.step();
                return true;
            }
        } while (mHdda.step());

        // We didn't find any active voxels, return
        return false;
    }

    bool mIsValid = false;
    const AccT &mAcc;
    RayTInternal mRay;
    HDDAT mHdda;
    value_type mData;
};

// Yields every active value the ray walk visits (leaf voxels AND coarse
// tiles); the caller distinguishes them via getDim(). See
// HDDAValueIteratorImpl above.
template <typename AccT, typename ScalarT>
using HDDAActiveValueIterator = HDDAValueIteratorImpl<AccT, ScalarT, /*LeafOnly=*/false>;

// Yields only active leaf voxels (dim == 1), skipping coarse tiles. Use when
// the per-value buffer has exactly one entry per active voxel (e.g. a
// level-set scalar field). Mirrors nanovdb::ZeroCrossing's narrow-band walk.
template <typename AccT, typename ScalarT>
using HDDALeafVoxelIterator = HDDAValueIteratorImpl<AccT, ScalarT, /*LeafOnly=*/true>;

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_HDDAITERATORS_H
