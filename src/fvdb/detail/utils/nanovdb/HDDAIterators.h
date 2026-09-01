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
#include <type_traits>

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
            // Fused getDim + isActive: a single Root->Internal->Leaf descent (or single
            // accessor-cache lookup) produces both the HDDA step size and the active state of the
            // current voxel — see ReadAccessor::getDimAndActive in NanoVDB.h. In the common case
            // where mHdda is already at the right level this saves one full tree walk per
            // iteration. The default ActiveExact policy is required here: this iterator reads
            // `active` unconditionally, including at coarse-tile levels, so a skip-flag
            // short-circuit that leaves `active` unspecified (ActiveOnLeafOnly) would be wrong.
            //
            // If we do need to re-align the HDDA to a different level, HDDA::update snaps mVoxel
            // to the new grid (mVoxel = RoundDown & ~(dim-1)), so the pre-update active bit may no
            // longer refer to the current voxel. Re-query after update so `active` is always
            // consistent with the voxel we'll read in the TimeSpan logic below. This restores the
            // two-descent cost only on level-change iterations (which are rare compared to
            // same-level stepping).
            auto dimActive = mAcc.getDimAndActive(mHdda.voxel(), mRay);
            if (mHdda.dim() != static_cast<int>(dimActive.dim())) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, static_cast<int>(dimActive.dim()));
                dimActive = mAcc.getDimAndActive(mHdda.voxel(), mRay);
            }
            const bool active = dimActive.active();

            // Predicated TimeSpan writes: only the `leaving` break is a real branch. The entering
            // and leaving t0/t1 updates are expressed as selects so rays in the same warp whose
            // `active` state differs don't diverge at the setter level.
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

// Iterates the active values an HDDA ray walk visits, yielding each as a {coordinate, [t0, t1]}
// pair. `LeafOnly` selects what counts as a value:
//
//   - LeafOnly == false: every active value at any node level, i.e. active coarse tiles (dim > 1)
//     as well as active leaf voxels (dim == 1). The caller must branch on `getDim()` to tell them
//     apart. Exposed as the `HDDAActiveValueIterator` alias.
//   - LeafOnly == true: only active leaf voxels (dim == 1); active tiles are skipped in a single
//     coarse HDDA step. This mirrors the narrow-band inner loop of `nanovdb::ZeroCrossing`. Exposed
//     as the `HDDALeafVoxelIterator` alias. Because every yielded value is a leaf voxel, a
//     per-voxel buffer indexed by `getValue(ijk) - 1` is always in-bounds; ops with
//     per-active-value buffers must use the active-value alias and handle tiles explicitly.
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
            // Fused getDim + isActive convergence: ReadAccessor::getDimAndActive collapses the
            // per-pass getDim and the post-convergence isActive into a single
            // Root->Internal->Leaf descent per iteration — `active` comes "for free" along with
            // the dim that drives the convergence check.
            //
            // Policy choice: when LeafOnly the gate below only consults `active` after checking
            // dim == 1, which is exactly the ActiveOnLeafOnly contract — on a skip-flag
            // short-circuit the returned dim exceeds 1, the gate rejects on dim alone, and the
            // (unspecified) active bit is never read. That preserves getDim's skip-flag fast path.
            // When !LeafOnly the gate reads `active` at every level (active coarse tiles must be
            // yielded), so ActiveExact is required — it matches separate getDim + isActive
            // byte-for-byte.
            using PolicyT =
                std::conditional_t<LeafOnly, nanovdb::ActiveOnLeafOnly, nanovdb::ActiveExact>;

            // Re-align the HDDA to the tree level at the current voxel. A single update can leave
            // the HDDA one level short when a step crosses several node levels at once, so we retry
            // up to three passes (root -> upper -> lower -> leaf is the deepest transition). Each
            // re-query runs at the voxel HDDA::update just snapped to, so on loop exit `dimActive`
            // always describes the current mHdda.voxel(). The `#pragma unroll` is load-bearing:
            // without it ptxas keeps this bounded loop rolled (the body is a fused tree-walk plus
            // an update, too big to auto-unroll) and emits a data-dependent backedge that adds
            // warp divergence at level transitions. Forcing the unroll turns the <=3 passes into
            // predicated straight-line code with no backedge.
            auto dimActive = mAcc.template getDimAndActive<PolicyT>(mHdda.voxel(), mRay);
            // Emit the pragma only in the CUDA device pass: it's meaningless on the host and the
            // host compiler treats unknown pragmas as errors (-Werror=unknown-pragmas).
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (int pass = 0; pass < 3 && mHdda.dim() != static_cast<int>(dimActive.dim());
                 ++pass) {
                mRay.setMinTime(mHdda.time());
                mHdda.update(mRay, static_cast<int>(dimActive.dim()));
                dimActive = mAcc.template getDimAndActive<PolicyT>(mHdda.voxel(), mRay);
            }

            // dim == 1 is a leaf voxel, dim > 1 a coarse tile; active is true for both under
            // ActiveExact. When LeafOnly, skip tiles so we yield only leaf voxels (the trailing
            // mHdda.step() jumps the whole tile in one coarse step); when !LeafOnly the gate
            // collapses to the original isActive check.
            const bool isLeaf = (static_cast<int>(dimActive.dim()) == 1);
            if ((!LeafOnly || isLeaf) && dimActive.active()) {
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

// Yields every active value the ray walk visits (leaf voxels AND coarse tiles); the caller
// distinguishes them via getDim(). See HDDAValueIteratorImpl above.
template <typename AccT, typename ScalarT>
using HDDAActiveValueIterator = HDDAValueIteratorImpl<AccT, ScalarT, /*LeafOnly=*/false>;

// Yields only active leaf voxels (dim == 1), skipping coarse tiles. Use when the per-value buffer
// has exactly one entry per active voxel (e.g. a level-set scalar field). Mirrors
// nanovdb::ZeroCrossing's narrow-band walk.
template <typename AccT, typename ScalarT>
using HDDALeafVoxelIterator = HDDAValueIteratorImpl<AccT, ScalarT, /*LeafOnly=*/true>;

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_HDDAITERATORS_H
