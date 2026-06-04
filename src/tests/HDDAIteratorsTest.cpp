// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// HDDAIteratorsTest.cpp -- Contract tests for the HDDA ray-walk iterators in
// src/fvdb/detail/utils/nanovdb/HDDAIterators.h.
//
// These pin the distinction between the two iterator aliases:
//
//   HDDAActiveValueIterator  yields every active value the ray walk visits,
//                            i.e. active coarse tiles (getDim > 1) as well as
//                            active leaf voxels (getDim == 1).
//   HDDALeafVoxelIterator    yields ONLY active leaf voxels (getDim == 1),
//                            skipping active tiles in a single coarse step.
//
// This matters because ray_implicit_intersection indexes a per-voxel gridScalars buffer by
// getValue(ijk)-1, which is only valid for leaf voxels. fvdb's own grids are leaf-only, but a
// grid loaded via from_nanovdb may have been built by an external tool and contain active tiles;
// the leaf iterator is what keeps ray_implicit_intersection correct (and in-bounds) for those.
//
// The iterators are __hostdev__ and driven here entirely on the host over a NanoGrid<float> built
// with NanoVDB's build tools, so no CUDA launch or fvdb OnIndexGrid is required -- the iterators
// only touch the accessor's getDim()/isActive(), which every NanoVDB accessor provides.

#include <fvdb/detail/utils/nanovdb/HDDAIterators.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace {

using Rayf = nanovdb::math::Ray<float>;

struct Yielded {
    nanovdb::Coord ijk;
    int dim;
};

// Walk `ray` with HDDAActiveValueIterator, recording each yielded voxel and the tree dim at that
// voxel (1 == leaf, > 1 == coarse tile).
template <typename AccT>
std::vector<Yielded>
collectActiveValues(const Rayf &ray, const AccT &acc) {
    std::vector<Yielded> out;
    for (auto it = fvdb::HDDAActiveValueIterator<AccT, float>(ray, acc); it.isValid(); ++it) {
        const nanovdb::Coord ijk = it->first;
        out.push_back({ijk, static_cast<int>(acc.getDim(ijk, ray))});
    }
    return out;
}

// Same, but with HDDALeafVoxelIterator (leaf voxels only).
template <typename AccT>
std::vector<Yielded>
collectLeafVoxels(const Rayf &ray, const AccT &acc) {
    std::vector<Yielded> out;
    for (auto it = fvdb::HDDALeafVoxelIterator<AccT, float>(ray, acc); it.isValid(); ++it) {
        const nanovdb::Coord ijk = it->first;
        out.push_back({ijk, static_cast<int>(acc.getDim(ijk, ray))});
    }
    return out;
}

int
countTiles(const std::vector<Yielded> &ys) {
    int n = 0;
    for (const auto &y: ys) {
        if (y.dim > 1) {
            ++n;
        }
    }
    return n;
}

} // namespace

// On a region that contains only active leaf voxels, the two iterators must agree exactly: the
// leaf-only gate must not drop any genuine leaf voxel.
TEST(HDDAIterators, LeafIteratorMatchesActiveValueIteratorOnLeafOnlyRegion) {
    using SrcGridT = nanovdb::tools::build::Grid<float>;
    SrcGridT srcGrid(0.0f);
    auto srcAcc = srcGrid.getAccessor();

    // A short contiguous run of active leaf voxels along +x at (x, 0, 0).
    for (int x = 0; x <= 5; ++x) {
        srcAcc.setValue(nanovdb::Coord(x, 0, 0), x < 3 ? 1.0f : -1.0f);
    }

    auto handle   = nanovdb::tools::createNanoGrid(srcGrid);
    auto *dstGrid = handle.template grid<float>();
    ASSERT_NE(dstGrid, nullptr);
    auto acc = dstGrid->getAccessor();

    // Ray along +x through the centre of the (x, 0, 0) cells (index space).
    const Rayf ray(nanovdb::math::Vec3<float>(-1.0f, 0.5f, 0.5f),
                   nanovdb::math::Vec3<float>(1.0f, 0.0f, 0.0f),
                   0.0f,
                   10.0f);

    const std::vector<Yielded> activeValues = collectActiveValues(ray, acc);
    const std::vector<Yielded> leafVoxels   = collectLeafVoxels(ray, acc);

    // No tiles in a pure leaf region, so the two walks must be identical.
    EXPECT_EQ(countTiles(activeValues), 0);
    EXPECT_EQ(countTiles(leafVoxels), 0);
    ASSERT_EQ(activeValues.size(), leafVoxels.size());
    ASSERT_GE(leafVoxels.size(), 1u) << "ray should have crossed active leaf voxels";
    for (size_t i = 0; i < leafVoxels.size(); ++i) {
        EXPECT_EQ(activeValues[i].ijk, leafVoxels[i].ijk);
        EXPECT_EQ(leafVoxels[i].dim, 1);
    }
}

// A ray crossing an active coarse tile must surface the tile in the active-value walk (getDim >
// 1) but be skipped entirely by the leaf walk.
TEST(HDDAIterators, LeafIteratorSkipsActiveTiles) {
    using SrcGridT = nanovdb::tools::build::Grid<float>;
    SrcGridT srcGrid(0.0f);

    // addTile<1> marks a whole lower-internal-node region (128^3 voxels) active as a single tile --
    // getDim inside it returns the tile dim (128), not 1. The coord is tile-aligned
    // (1024 = 8 * 128).
    const nanovdb::Coord tileOrigin(1024, 1024, 1024);
    srcGrid.tree().root().template addTile<1>(tileOrigin, 5.0f, true);

    auto handle   = nanovdb::tools::createNanoGrid(srcGrid);
    auto *dstGrid = handle.template grid<float>();
    ASSERT_NE(dstGrid, nullptr);
    auto acc = dstGrid->getAccessor();

    // Ray along +x passing through the interior of the tile region ([1024, 1151]^3) at
    // y = z = 1088.
    const Rayf ray(nanovdb::math::Vec3<float>(1000.0f, 1088.0f, 1088.0f),
                   nanovdb::math::Vec3<float>(1.0f, 0.0f, 0.0f),
                   0.0f,
                   200.0f);

    const std::vector<Yielded> activeValues = collectActiveValues(ray, acc);
    const std::vector<Yielded> leafVoxels   = collectLeafVoxels(ray, acc);

    // The active-value walk must see the tile (at least one getDim > 1 entry)...
    EXPECT_GE(countTiles(activeValues), 1)
        << "HDDAActiveValueIterator should surface the active tile";
    // ...and the leaf walk must skip it entirely (no tile, and since the region has no leaf voxels,
    // nothing at all).
    EXPECT_EQ(countTiles(leafVoxels), 0) << "HDDALeafVoxelIterator must not yield coarse tiles";
    EXPECT_EQ(leafVoxels.size(), 0u)
        << "tile region has no leaf voxels, so the leaf walk should be empty";
}
