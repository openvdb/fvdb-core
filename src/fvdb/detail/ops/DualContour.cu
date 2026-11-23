// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Dual-contouring (DC + QEF) mesher: meshes an OnIndex grid carrying a narrow-band SDF into a
// triangle mesh, with QEF vertex placement and optional cluster-collapse decimation. The 3x3x3 box
// stencil is gathered once per voxel via the NanoVDB VoxelBlockManager. The whole batch is meshed
// together (in chunks of consecutive grids bounded by kMaxChunkVoxels), so the number of host
// synchronizations and kernel launches does not grow with the number of grids.
//
// This follows the dual-contouring approach of OpenVDB's tools/VolumeToMesh.h, but places vertices
// by minimising a quadratic error function (classic DC) rather than averaging the edge crossings
// (the "mass point" VolumeToMesh uses by default), and decimates by simple cluster-collapse rather
// than VolumeToMesh's seam-stitched octree region merge.
//
// References:
//   [Ju et al. 2002]      T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite
//                         Data", ACM TOG 21(3) (SIGGRAPH 2002), 339-346. The core method: one
//                         vertex per cell minimising a QEF over the edge Hermite data (crossings +
//                         normals), the dual connectivity, the numerically-stable centroid-biased
//                         QEF, and the octree-based adaptive simplification our decimation is a
//                         simplified form of.
//   [Garland & Heckbert 1997]  M. Garland, P. Heckbert, "Surface Simplification Using Quadric Error
//                         Metrics", SIGGRAPH 1997, 209-216. The quadric error metric (A = sum n
//                         n^T) that the vertex placement minimises.
//   [Kobbelt et al. 2001] L. Kobbelt, M. Botsch, U. Schwanecke, H.-P. Seidel, "Feature Sensitive
//                         Surface Extraction from Volume Data", SIGGRAPH 2001. Placing the vertex
//                         at the intersection of the edge-crossing tangent planes (the feature
//                         point).
//   [Lorensen & Cline 1987]  W. Lorensen, H. Cline, "Marching Cubes: A High Resolution 3D Surface
//                         Construction Algorithm", SIGGRAPH 1987, 163-169. The sign-based
//                         isosurface extraction that dual contouring is the dual of.
//
#include <fvdb/VoxelCoordTransform.h>
#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/ops/DualContour.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

using OnIndexGridT = nanovdb::NanoGrid<nanovdb::ValueOnIndex>;

// The batch's cached VoxelBlockManagers (GridBatchData::vbm) are decoded one VBM block per CUDA
// block of kBlockWidth threads, one thread per block slot. (A grid-stride form with 256-thread
// blocks doubled the gather kernel's register count and ran ~25% slower.)
static constexpr int kLog2BlockWidth = VbmCache::kLog2BlockWidth;
static constexpr int kBlockWidth     = VbmCache::kBlockWidth;

// Grids are meshed together in chunks of consecutive grids holding at most this many active voxels
// (a single larger grid forms its own chunk). Each chunk costs a constant number of host syncs (the
// surface-cell compaction, the triangle count, the vertex prune, and the cluster relabels when
// decimating) and ~89 bytes of scratch per voxel, so this bounds the scratch at ~370 MB while a
// chunk's work (milliseconds) still dwarfs its sync overhead (tens of microseconds). Chunk-local
// value indices are int32, so a chunk must also stay below 2^31 voxels.
static constexpr int64_t kMaxChunkVoxels = int64_t(1) << 22;

// Per-grid decimation-key parameters, one row of int64 per grid in the batch (see
// buildGridKeyParams): the grid's surface-cell coord minimum (the key origin), its key-space spans,
// and the base of its key range, so that cluster keys of different grids never collide.
enum GridKeyParam : int {
    kKeyOriginX = 0,
    kKeyOriginY,
    kKeyOriginZ,
    kKeySpanY,
    kKeySpanZ,
    kKeyFineCount, // number of fine (per-voxel) keys of the grid: spanX * spanY * spanZ
    kKeyBase,      // first key of the grid's [kKeyBase, kKeyBase + 2 * kKeyFineCount) key range
    kNumGridKeyParams
};

// ------------------------- box-stencil geometry (compile-time) -------------------------

/// @brief Compile-time description of the 3x3x3 box stencil used by the mesher.
///
/// A box-stencil neighbour is addressed by a spoke index in [0,27):
/// `spoke = (di+1)*9 + (dj+1)*3 + (dk+1)`; the centre (the voxel itself) is spoke 13. The mesher
/// only touches a subset of the 27 spokes (8 cube corners, 6 faces, 3 connectivity fans), so the
/// gather emits just those `numColumns` columns. All of this is static, so it is computed once at
/// compile time into `kGeometry` and passed by value into the kernels (no `__constant__` /
/// per-call setup).
struct BoxStencilGeometry {
    int numColumns{0};
    int usedSpokes[27]{};      // the `numColumns` gathered spokes (sorted, unique)
    int cornerColumn[8]{};     // column of each cube corner in the gathered (numVoxels,numColumns)
                               // table
    int faceColumn[6]{};       // column of each face neighbour (-x,+x,-y,+y,-z,+z)
    int centerColumn{0};       // column of spoke 13 (the voxel itself)
    int fanForwardColumn[3]{}; // forward-edge spoke column for the x/y/z minimal edges
    int fanColumn[3][4]{};     // the 4 surrounding-cell spoke columns per edge fan
    int edgeCornerA[12]{};     // the 12 cube edges as corner-index pairs (A,B)
    int edgeCornerB[12]{};
    int cornerSpoke[8]{}; // raw spoke of each corner (the fused gather reads the stencil directly)
    int faceSpoke[6]{};
    float cornerOffset[8][3]{}; // the 8 corner offsets in [0,1]^3
};

constexpr int
spokeIndex(int di, int dj, int dk) {
    return (di + 1) * 9 + (dj + 1) * 3 + (dk + 1);
}

/// @brief Column of `spoke` within the sorted-unique `usedSpokes[0..numColumns)`, or -1 if absent.
/// Binary search (std::lower_bound is constexpr in C++20) since usedSpokes is sorted.
constexpr int
columnOfSpoke(const int *usedSpokes, int numColumns, int spoke) {
    const int *end   = usedSpokes + numColumns;
    const int *found = std::lower_bound(usedSpokes, end, spoke);
    return (found != end && *found == spoke) ? int(found - usedSpokes) : -1;
}

constexpr BoxStencilGeometry
makeBoxStencilGeometry() {
    BoxStencilGeometry geometry{};
    const int cornerOffset[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    const int cubeEdge[12][2] = {{0, 1},
                                 {2, 4},
                                 {3, 5},
                                 {6, 7},
                                 {0, 2},
                                 {1, 4},
                                 {3, 6},
                                 {5, 7},
                                 {0, 3},
                                 {1, 5},
                                 {2, 6},
                                 {4, 7}};
    for (int corner = 0; corner < 8; ++corner) {
        for (int axis = 0; axis < 3; ++axis) {
            geometry.cornerOffset[corner][axis] = (float)cornerOffset[corner][axis];
        }
        geometry.cornerSpoke[corner] =
            spokeIndex(cornerOffset[corner][0], cornerOffset[corner][1], cornerOffset[corner][2]);
    }
    for (int edge = 0; edge < 12; ++edge) {
        geometry.edgeCornerA[edge] = cubeEdge[edge][0];
        geometry.edgeCornerB[edge] = cubeEdge[edge][1];
    }
    geometry.faceSpoke[0]    = spokeIndex(-1, 0, 0);
    geometry.faceSpoke[1]    = spokeIndex(1, 0, 0);
    geometry.faceSpoke[2]    = spokeIndex(0, -1, 0);
    geometry.faceSpoke[3]    = spokeIndex(0, 1, 0);
    geometry.faceSpoke[4]    = spokeIndex(0, 0, -1);
    geometry.faceSpoke[5]    = spokeIndex(0, 0, 1);
    const int fanForward[3]  = {spokeIndex(1, 0, 0), spokeIndex(0, 1, 0), spokeIndex(0, 0, 1)};
    const int fanSpoke[3][4] = {
        {13, spokeIndex(0, 0, -1), spokeIndex(0, -1, -1), spokeIndex(0, -1, 0)},
        {13, spokeIndex(0, 0, -1), spokeIndex(-1, 0, -1), spokeIndex(-1, 0, 0)},
        {13, spokeIndex(0, -1, 0), spokeIndex(-1, -1, 0), spokeIndex(-1, 0, 0)}};

    // sorted-unique union of all touched spokes -> usedSpokes[0..numColumns)
    int allSpokes[64]{};
    int spokeCount = 0;
    for (int i = 0; i < 6; ++i)
        allSpokes[spokeCount++] = geometry.faceSpoke[i];
    for (int i = 0; i < 8; ++i)
        allSpokes[spokeCount++] = geometry.cornerSpoke[i];
    for (int dir = 0; dir < 3; ++dir) {
        allSpokes[spokeCount++] = fanForward[dir];
        for (int i = 0; i < 4; ++i)
            allSpokes[spokeCount++] = fanSpoke[dir][i];
    }
    // sort then drop duplicates (std::sort / std::unique are constexpr in C++20)
    std::sort(allSpokes, allSpokes + spokeCount);
    const int numColumns = int(std::unique(allSpokes, allSpokes + spokeCount) - allSpokes);
    for (int i = 0; i < numColumns; ++i)
        geometry.usedSpokes[i] = allSpokes[i];
    geometry.numColumns = numColumns;

    for (int corner = 0; corner < 8; ++corner) {
        geometry.cornerColumn[corner] =
            columnOfSpoke(geometry.usedSpokes, numColumns, geometry.cornerSpoke[corner]);
    }
    for (int face = 0; face < 6; ++face) {
        geometry.faceColumn[face] =
            columnOfSpoke(geometry.usedSpokes, numColumns, geometry.faceSpoke[face]);
    }
    geometry.centerColumn = columnOfSpoke(geometry.usedSpokes, numColumns, 13);
    for (int dir = 0; dir < 3; ++dir) {
        geometry.fanForwardColumn[dir] =
            columnOfSpoke(geometry.usedSpokes, numColumns, fanForward[dir]);
        for (int i = 0; i < 4; ++i) {
            geometry.fanColumn[dir][i] =
                columnOfSpoke(geometry.usedSpokes, numColumns, fanSpoke[dir][i]);
        }
    }
    return geometry;
}

static constexpr BoxStencilGeometry kGeometry = makeBoxStencilGeometry();

/// @brief Solve a symmetric 3x3 SPD system A x = b (A given by its 6 unique entries) via Cramer's
/// rule.
__device__ inline void
solveSymmetric3x3(const double A[6], const double b[3], double x[3]) {
    const double a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3], a4 = A[4], a5 = A[5];
    const double cof00 = a3 * a5 - a4 * a4, cof01 = a4 * a2 - a1 * a5, cof02 = a1 * a4 - a3 * a2;
    double determinant = a0 * cof00 + a1 * cof01 + a2 * cof02;
    if (fabs(determinant) < 1e-30) {
        x[0] = x[1] = x[2] = 0.0;
        return;
    }
    const double invDet = 1.0 / determinant;
    const double cof11 = a0 * a5 - a2 * a2, cof12 = a1 * a2 - a0 * a4, cof22 = a0 * a3 - a1 * a1;
    x[0] = invDet * (cof00 * b[0] + cof01 * b[1] + cof02 * b[2]);
    x[1] = invDet * (cof01 * b[0] + cof11 * b[1] + cof12 * b[2]);
    x[2] = invDet * (cof02 * b[0] + cof12 * b[1] + cof22 * b[2]);
}

// =====================  fused VBM decode  ====================================================
/// @brief Decoding the VBM stencil is the expensive part, so each voxel does it ONCE here and
/// writes four slot-indexed outputs that every later kernel then reads in O(1) (no re-walking the
/// tree):
///   - neighborTable [valueCount x numColumns] : the gathered box-stencil neighbour slots
///   - gradient                                : central-difference SDF gradient (the surface
///                                               normal)
///   - surfaceFlag                             : 1 iff the iso surface passes through this voxel's
///                                               cube
///   - voxelCoord                              : the voxel's index-space (i,j,k)
///
/// One launch covers every grid of a chunk: CUDA block b decodes batch-wide VBM block vbBegin + b
/// (gap blocks between grids exit at once). Outputs are
/// indexed by chunk-wide slot: the voxel with batch-wide feature index f sits at slot
/// 1 + f - chunkVoxelBase, and slot 0 is the shared background (inactive neighbours of every grid
/// map to it). The box stencil only reaches voxels of the same grid, so the neighbour table, and
/// with it all connectivity, never crosses grids.
__global__ void __launch_bounds__(kBlockWidth)
gatherFusedKernel(const BatchVbm vbm,
                  const BatchVbmLayout layout,
                  uint64_t vbBegin,
                  int64_t chunkVoxelBase,
                  const float *sdf,
                  BoxStencilGeometry geometry,
                  float iso,
                  int32_t *neighborTable,
                  float *gradient,
                  uint8_t *surfaceFlag,
                  int32_t *voxelCoord) {
    using VoxelBlockManagerT = nanovdb::tools::cuda::VoxelBlockManager<kLog2BlockWidth>;
    const BatchVbmView view(
        layout, uint32_t(vbm.numGrids), vbm.blockCount, vbm.firstLeafID, vbm.jumpMap);
    const uint64_t vb = vbBegin + blockIdx.x; // this CUDA block's batch-wide VBM block
    // blocks between grids map to the grid before them and fall past its voxels below
    const uint32_t bi = view.lastGridAtOrBefore(vb);
    if (bi == BatchVbmView::UnusedGridIndex)
        return;
    const uint64_t localBlock = vb - view.blockBase(bi);
    if (localBlock * kBlockWidth + threadIdx.x >= view.voxelCount(bi))
        return; // slot past the grid's last active voxel
    const nanovdb::OnIndexGrid *grid = layout.acc.grid(bi);
    uint32_t leafIndex;
    uint16_t voxelOffset;
    VoxelBlockManagerT::template decodeInverseMap<nanovdb::ValueOnIndex>(
        grid,
        view.firstLeafID()[vb],
        view.jumpMap() + vb * VbmCache::kJumpMapWordCount,
        1 + localBlock * kBlockWidth,
        int(threadIdx.x),
        leafIndex,
        voxelOffset);
    if (leafIndex == VoxelBlockManagerT::UnusedLeafIndex)
        return; // defensive; the voxelCount guard above makes this unreachable
    // grid-local value indices (1-based, 0 = inactive); slotOf maps one to its chunk-wide slot
    // (0 = background) as it is read
    uint64_t stencil[27];
    VoxelBlockManagerT::template computeBoxStencil<nanovdb::ValueOnIndex>(
        grid, leafIndex, voxelOffset, stencil);
    const int64_t slotBase = layout.voxelOffsets[bi] - chunkVoxelBase;
    auto slotOf            = [&](int spoke) -> int64_t {
        const uint64_t valueIndex = stencil[spoke];
        return valueIndex ? int64_t(valueIndex) + slotBase : 0;
    };
    const int64_t centerIndex = slotOf(13); // this voxel's slot (>= 1)
    const int numColumns      = geometry.numColumns;

    for (int column = 0; column < numColumns; ++column) {
        neighborTable[centerIndex * numColumns + column] =
            (int32_t)slotOf(geometry.usedSpokes[column]);
    }

    // central-difference gradient (inactive face stencil==0 -> use the centre value). Computed in
    // index space; normalised, this is the correct world normal for isotropic voxels (the uniform
    // scale cancels) -- the assumption of the SDF ops this consumes. See the note in DualContour.h.
    const float centerSdf = sdf[centerIndex];
    auto faceValue        = [&](int spoke) {
        const int64_t neighbor = slotOf(spoke);
        return neighbor > 0 ? sdf[neighbor] : centerSdf;
    };
    gradient[centerIndex * 3 + 0] =
        0.5f * (faceValue(geometry.faceSpoke[1]) - faceValue(geometry.faceSpoke[0]));
    gradient[centerIndex * 3 + 1] =
        0.5f * (faceValue(geometry.faceSpoke[3]) - faceValue(geometry.faceSpoke[2]));
    gradient[centerIndex * 3 + 2] =
        0.5f * (faceValue(geometry.faceSpoke[5]) - faceValue(geometry.faceSpoke[4]));

    // surface-cell flag: the cube anchored at this voxel is a surface cell iff its 8 corners
    // straddle iso (0 < #inside < 8) AND are all active -- a full active cube is needed to place a
    // QEF vertex and to let the cell take part in quads.
    int numInsideCorners  = 0;
    bool allCornersActive = true;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        const int64_t cornerIndex = slotOf(geometry.cornerSpoke[corner]);
        numInsideCorners += (sdf[cornerIndex] < iso) ? 1 : 0;
        // corner 0 is offset (0,0,0) -- the anchor voxel itself (== centerIndex), which is always
        // active here (this thread only runs for an active voxel), so skip its redundant check.
        if (corner)
            allCornersActive &= (cornerIndex > 0);
    }
    surfaceFlag[centerIndex] =
        (allCornersActive && numInsideCorners > 0 && numInsideCorners < 8) ? 1u : 0u;

    const auto &leaf                 = grid->tree().getFirstNode<0>()[leafIndex];
    const nanovdb::Coord globalCoord = leaf.offsetToGlobalCoord(voxelOffset);
    voxelCoord[centerIndex * 3 + 0]  = globalCoord[0];
    voxelCoord[centerIndex * 3 + 1]  = globalCoord[1];
    voxelCoord[centerIndex * 3 + 2]  = globalCoord[2];
}

/// @brief One cube edge's iso-surface crossing: the linearly interpolated crossing point (in the
/// cell-local [0,1]^3 frame) and the unit surface normal there.
struct EdgeCrossing {
    float point[3];  // crossing position in the cell's [0,1]^3 frame
    float normal[3]; // unit surface normal at the crossing (from the interpolated corner gradients)
};

/// @brief Compute cube edge `edge`'s iso crossing from the cell's 8 corner SDFs/gradients. Returns
/// false (leaving `out` untouched) when both endpoints lie on the same side of `iso`, i.e. the edge
/// does not cross the surface. This is the per-edge primitive the three QEF kernels share
/// (placeQefVerticesKernel, cellNormalKernel, clusterQefAccumKernel), so the 12-edge
/// crossing/normal interpolation lives in exactly one place.
__device__ inline bool
computeEdgeCrossing(int edge,
                    const float cornerSdf[8],
                    const float cornerGradient[8][3],
                    BoxStencilGeometry geometry,
                    float iso,
                    EdgeCrossing &out) {
    const int cornerA = geometry.edgeCornerA[edge], cornerB = geometry.edgeCornerB[edge];
    const float sdfA = cornerSdf[cornerA], sdfB = cornerSdf[cornerB];
    if ((sdfA < iso) == (sdfB < iso))
        return false; // endpoints on the same side of iso -> this edge does not cross the surface
    // crossingT = fraction along A->B where the linearly-interpolated sdf reaches iso (the edge
    // root); fall back to the midpoint if the two endpoints are (numerically) equal
    const float sdfDelta = sdfB - sdfA;
    float crossingT      = (fabsf(sdfDelta) > 1e-12f) ? (iso - sdfA) / sdfDelta : 0.5f;
    crossingT            = fminf(fmaxf(crossingT, 0.f), 1.f);
    // crossing point and (un-normalised) normal, both linearly interpolated between the two corners
    float edgeNormal[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        out.point[axis] = geometry.cornerOffset[cornerA][axis] +
                          crossingT * (geometry.cornerOffset[cornerB][axis] -
                                       geometry.cornerOffset[cornerA][axis]);
        edgeNormal[axis] =
            cornerGradient[cornerA][axis] +
            crossingT * (cornerGradient[cornerB][axis] - cornerGradient[cornerA][axis]);
    }
    // normalise n_i (the eps keeps a (near-)zero gradient finite)
    const float invLen = rsqrtf(edgeNormal[0] * edgeNormal[0] + edgeNormal[1] * edgeNormal[1] +
                                edgeNormal[2] * edgeNormal[2] + 1e-24f);
    out.normal[0]      = edgeNormal[0] * invLen;
    out.normal[1]      = edgeNormal[1] * invLen;
    out.normal[2]      = edgeNormal[2] * invLen;
    return true;
}

/// @brief (QEF) Place one dual-contouring vertex per surface cell. The vertex is the point x
/// minimising the quadratic error function  E(x) = sum_i [ n_i . (x - p_i) ]^2  over the cell's
/// edge intersections, where p_i is the i-th edge zero-crossing and n_i the unit surface normal
/// there. Each term is the squared distance from x to the tangent plane at crossing i, so the
/// minimiser is the point that best fits all those planes -- it snaps onto sharp features
/// (edges/corners) that a plain crossing-average would round off. Setting dE/dx = 0 gives the 3x3
/// normal equations
///   A x = b,   A = sum_i n_i n_i^T,   b = sum_i n_i (n_i . p_i),
/// solved below (re-centred on the crossing centroid and Tikhonov-regularised for stability). Also
/// stores the centre voxel's SDF gradient as the per-vertex reference normal (used to orient the
/// triangles, and emitted as the output normal once normalised).
/// Method: [Ju et al. 2002]; the QEF is the quadric error metric of [Garland & Heckbert 1997]; the
/// tangent-plane feature point is [Kobbelt et al. 2001]. (See the reference list at the top.)
__global__ void
placeQefVerticesKernel(const int32_t *surfaceCells,
                       const int32_t *surfaceGrid,
                       int64_t numSurfaceCells,
                       const int32_t *neighborTable,
                       const float *sdf,
                       const float *gradient,
                       const int32_t *voxelCoord,
                       BoxStencilGeometry geometry,
                       float iso,
                       GridBatchData::Accessor batchAcc,
                       float *outVertices,
                       float *outRefNormal) {
    const int64_t cellOrdinal = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cellOrdinal >= numSurfaceCells)
        return;
    const int numColumns         = geometry.numColumns;
    const int32_t cellValueIndex = surfaceCells[cellOrdinal];
    // gather the 8 cube-corner SDF values and gradients (a surface cell's corners are all active)
    float cornerSdf[8];
    float cornerGradient[8][3];
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int32_t cornerIndex =
            neighborTable[int64_t(cellValueIndex) * numColumns + geometry.cornerColumn[corner]];
        cornerSdf[corner]         = sdf[cornerIndex];
        cornerGradient[corner][0] = gradient[int64_t(cornerIndex) * 3 + 0];
        cornerGradient[corner][1] = gradient[int64_t(cornerIndex) * 3 + 1];
        cornerGradient[corner][2] = gradient[int64_t(cornerIndex) * 3 + 2];
    }
    double normalMatrix[6] = {0, 0, 0, 0, 0, 0}; // sum of n n^T (6 unique entries)
    double normalRhs[3]    = {0, 0, 0};          // sum of n (n . crossing)
    double crossingSum[3]  = {0, 0, 0};          // sum of edge crossings
    int numCrossings       = 0;
    // accumulate A and b over the cell's 12 edges: each edge that crosses the iso surface
    // contributes one tangent-plane constraint (its crossing point p_i and the normal n_i there)
#pragma unroll
    for (int edge = 0; edge < 12; ++edge) {
        EdgeCrossing ec;
        if (!computeEdgeCrossing(edge, cornerSdf, cornerGradient, geometry, iso, ec))
            continue;
        const double nx = ec.normal[0], ny = ec.normal[1], nz = ec.normal[2];
        const double normalDotCrossing = nx * ec.point[0] + ny * ec.point[1] + nz * ec.point[2];
        // A += n n^T (6 unique entries), b-term normalRhs += n (n.p), and track the crossing
        // centroid
        normalMatrix[0] += nx * nx;
        normalMatrix[1] += nx * ny;
        normalMatrix[2] += nx * nz;
        normalMatrix[3] += ny * ny;
        normalMatrix[4] += ny * nz;
        normalMatrix[5] += nz * nz;
        normalRhs[0] += nx * normalDotCrossing;
        normalRhs[1] += ny * normalDotCrossing;
        normalRhs[2] += nz * normalDotCrossing;
        crossingSum[0] += ec.point[0];
        crossingSum[1] += ec.point[1];
        crossingSum[2] += ec.point[2];
        ++numCrossings;
    }
    double localVertex[3];
    double invCount            = numCrossings > 0 ? 1.0 / numCrossings : 0.0;
    double crossingCentroid[3] = {
        crossingSum[0] * invCount, crossingSum[1] * invCount, crossingSum[2] * invCount};
    if (numCrossings == 0) {
        // no bipolar edges (a real surface cell always has >= 1; defensive) -> just use the
        // centroid
        localVertex[0] = crossingCentroid[0];
        localVertex[1] = crossingCentroid[1];
        localVertex[2] = crossingCentroid[2];
    } else {
        // Solve the normal equations re-centred on the crossing centroid c (much better
        // conditioned): with y = x - c,  (A + lambda I) y = b - A c = normalRhs - A.centroid, then
        // x = c + y. The lambda = 0.05 Tikhonov term keeps the system solvable when A is
        // rank-deficient (flat cells or too few independent normals leave the surface position
        // underconstrained) and nudges those free directions gently back toward the centroid.
        double matTimesCentroid[3] = {
            normalMatrix[0] * crossingCentroid[0] + normalMatrix[1] * crossingCentroid[1] +
                normalMatrix[2] * crossingCentroid[2],
            normalMatrix[1] * crossingCentroid[0] + normalMatrix[3] * crossingCentroid[1] +
                normalMatrix[4] * crossingCentroid[2],
            normalMatrix[2] * crossingCentroid[0] + normalMatrix[4] * crossingCentroid[1] +
                normalMatrix[5] * crossingCentroid[2]};
        double rhs[3]               = {normalRhs[0] - matTimesCentroid[0],
                                       normalRhs[1] - matTimesCentroid[1],
                                       normalRhs[2] - matTimesCentroid[2]};
        double regularizedMatrix[6] = {normalMatrix[0] + 0.05,
                                       normalMatrix[1],
                                       normalMatrix[2],
                                       normalMatrix[3] + 0.05,
                                       normalMatrix[4],
                                       normalMatrix[5] + 0.05};
        double solution[3];
        solveSymmetric3x3(regularizedMatrix, rhs, solution);
        // x = centroid + y, clamped into the unit cell so the vertex never leaves its own cell
        for (int axis = 0; axis < 3; ++axis)
            localVertex[axis] = fmin(fmax(crossingCentroid[axis] + solution[axis], 0.0), 1.0);
    }
    // cell-local vertex -> world: localVertex is in [0,1]^3 within the cell, so add the cell's
    // index-space origin and map index space -> world with the cell's grid transform.
    const VoxelCoordTransform &transform = batchAcc.primalTransform(surfaceGrid[cellOrdinal]);
    const float originX                  = (float)voxelCoord[int64_t(cellValueIndex) * 3 + 0];
    const float originY                  = (float)voxelCoord[int64_t(cellValueIndex) * 3 + 1];
    const float originZ                  = (float)voxelCoord[int64_t(cellValueIndex) * 3 + 2];
    auto worldPos                    = transform.applyInv<float>(originX + (float)localVertex[0],
                                              originY + (float)localVertex[1],
                                              originZ + (float)localVertex[2]);
    outVertices[cellOrdinal * 3 + 0] = worldPos[0];
    outVertices[cellOrdinal * 3 + 1] = worldPos[1];
    outVertices[cellOrdinal * 3 + 2] = worldPos[2];
    // reference normal = the centre voxel's SDF gradient; orients the triangles and (once
    // normalised) becomes the emitted per-vertex normal.
    outRefNormal[cellOrdinal * 3 + 0] = gradient[int64_t(cellValueIndex) * 3 + 0];
    outRefNormal[cellOrdinal * 3 + 1] = gradient[int64_t(cellValueIndex) * 3 + 1];
    outRefNormal[cellOrdinal * 3 + 2] = gradient[int64_t(cellValueIndex) * 3 + 2];
}

/// @brief (connectivity) Dual-contouring connectivity is the dual of marching cubes: wherever a
/// grid edge changes sign (crosses the surface), the 4 cells sharing that edge each hold a vertex,
/// and joining those 4 vertices makes one quad straddling the edge. To emit each quad exactly once,
/// a voxel owns only its 3 "minimal" forward edges (+x/+y/+z, selected by `dir`). Returns true and
/// fills `quadVertices[4]` with the 4 surrounding cells' vertices when this voxel's `dir` edge is
/// bipolar and all 4 of those cells carry a vertex; false otherwise.
/// Dual connectivity is [Ju et al. 2002]; the sign-based isosurface it is dual to is [Lorensen &
/// Cline 1987].
__device__ inline bool
tryBuildQuad(int64_t voxelIndex,
             int dir,
             const int32_t *neighborTable,
             const float *sdf,
             const int32_t *cellVertex,
             BoxStencilGeometry geometry,
             float iso,
             int32_t quadVertices[4]) {
    if (voxelIndex == 0)
        return false;
    const int numColumns = geometry.numColumns;
    int32_t forwardIndex = neighborTable[voxelIndex * numColumns + geometry.fanForwardColumn[dir]];
    if (forwardIndex <= 0)
        return false;
    if ((sdf[voxelIndex] < iso) == (sdf[forwardIndex] < iso))
        return false; // edge is not bipolar
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int column                = geometry.fanColumn[dir][i];
        int32_t neighborCellIndex = (column == geometry.centerColumn)
                                        ? int32_t(voxelIndex)
                                        : neighborTable[voxelIndex * numColumns + column];
        int32_t vertex = cellVertex[neighborCellIndex]; // cellVertex[0] == -1 (background)
        if (vertex < 0)
            return false;
        quadVertices[i] = vertex;
    }
    return true;
}

/// @brief Diagonal-aware split of the cyclic quad; returns false if triangle `whichTriangle` is
/// degenerate (a repeated vertex -- happens only when cells contract onto one cluster).
__device__ inline bool
splitQuadTriangle(const int32_t quadVertices[4], int whichTriangle, int32_t outTriangle[3]) {
    bool useDiagonal13 =
        (quadVertices[0] == quadVertices[2]) && (quadVertices[1] != quadVertices[3]);
    if (whichTriangle == 0) {
        outTriangle[0] = quadVertices[0];
        outTriangle[1] = quadVertices[1];
        outTriangle[2] = useDiagonal13 ? quadVertices[3] : quadVertices[2];
    } else {
        outTriangle[0] = useDiagonal13 ? quadVertices[1] : quadVertices[0];
        outTriangle[1] = quadVertices[2];
        outTriangle[2] = quadVertices[3];
    }
    return outTriangle[0] != outTriangle[1] && outTriangle[1] != outTriangle[2] &&
           outTriangle[0] != outTriangle[2];
}

/// @brief (connectivity, pass 1) Count the triangles each surface cell emits; a cumsum of these
/// counts then gives each cell a contiguous, deterministic write range (no atomics, unlike a single
/// shared counter). Only surface cells can own a quad: every quad requires its 4 fan cells (one of
/// which is the owning voxel itself) to carry a vertex, so iterating surface cells is complete --
/// and far cheaper than sweeping every active voxel.
__global__ void
countTrianglesKernel(const int32_t *surfaceCells,
                     int64_t numSurfaceCells,
                     const int32_t *neighborTable,
                     const float *sdf,
                     const int32_t *cellVertex,
                     BoxStencilGeometry geometry,
                     float iso,
                     int64_t *triangleCounts) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    const int64_t voxelIndex = surfaceCells[i];
    int32_t quadVertices[4], triangle[3];
    int localCount = 0;
#pragma unroll
    for (int dir = 0; dir < 3; ++dir) {
        if (tryBuildQuad(
                voxelIndex, dir, neighborTable, sdf, cellVertex, geometry, iso, quadVertices)) {
            for (int whichTriangle = 0; whichTriangle < 2; ++whichTriangle) {
                if (splitQuadTriangle(quadVertices, whichTriangle, triangle))
                    ++localCount;
            }
        }
    }
    triangleCounts[i] = localCount;
}

/// @brief (connectivity, pass 2) Write each surface cell's triangles into its own range, starting
/// at the exclusive prefix sum `triangleOffsets[i]`.
__global__ void
writeTrianglesKernel(const int32_t *surfaceCells,
                     int64_t numSurfaceCells,
                     const int32_t *neighborTable,
                     const float *sdf,
                     const int32_t *cellVertex,
                     BoxStencilGeometry geometry,
                     float iso,
                     const int64_t *triangleOffsets,
                     int32_t *triangles) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    const int64_t voxelIndex = surfaceCells[i];
    int32_t quadVertices[4], triangle[3];
    int64_t slot = triangleOffsets[i];
#pragma unroll
    for (int dir = 0; dir < 3; ++dir) {
        if (!tryBuildQuad(
                voxelIndex, dir, neighborTable, sdf, cellVertex, geometry, iso, quadVertices))
            continue;
        for (int whichTriangle = 0; whichTriangle < 2; ++whichTriangle) {
            if (!splitQuadTriangle(quadVertices, whichTriangle, triangle))
                continue;
            triangles[slot * 3 + 0] = triangle[0];
            triangles[slot * 3 + 1] = triangle[1];
            triangles[slot * 3 + 2] = triangle[2];
            ++slot;
        }
    }
}

/// @brief (orient) Flip each triangle so its geometric normal agrees with the summed reference
/// normal.
__global__ void
orientTrianglesKernel(int64_t numTriangles,
                      const float *vertices,
                      const float *refNormal,
                      int32_t *triangles) {
    const int64_t face = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face >= numTriangles)
        return;
    using Vec3f      = nanovdb::math::Vec3<float>;
    const int32_t i0 = triangles[face * 3 + 0], i1 = triangles[face * 3 + 1],
                  i2 = triangles[face * 3 + 2];
    auto vertex      = [&](int32_t i) {
        return Vec3f(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    };
    auto normal = [&](int32_t i) {
        return Vec3f(refNormal[i * 3], refNormal[i * 3 + 1], refNormal[i * 3 + 2]);
    };
    const Vec3f p0         = vertex(i0);
    const Vec3f faceNormal = (vertex(i1) - p0).cross(vertex(i2) - p0);
    const Vec3f refSum     = normal(i0) + normal(i1) + normal(i2);
    if (faceNormal.dot(refSum) < 0.f) {
        triangles[face * 3 + 1] = i2;
        triangles[face * 3 + 2] = i1;
    }
}

// =====================  decimation (reduce / adaptivity)  ====================================
// Cluster-collapse decimation: instead of one vertex per surface cell, group cells into clusters
// and emit one vertex per cluster. `reduce = F` uses uniform F x F x F index-space blocks;
// `adaptivity` keeps feature cells at full detail and only collapses "flat" coarse blocks (those
// whose cells' edge normals are well aligned). Each cell's QEF is summed into its cluster
// (clusterQefAccumKernel) and a single merged vertex is solved per cluster (clusterSolveKernel);
// connectivity then merges for free, since quads referencing cells that collapsed to the same
// vertex become degenerate and are dropped. NOTE: the per-cluster accumulation uses double
// atomicAdd, so the summation order -- and hence the exact decimated vertex positions -- are not
// bit-reproducible run to run.
// This is a simplified, single-level form of the octree-based adaptive DC simplification of [Ju et
// al. 2002]; OpenVDB's VolumeToMesh instead does the full seam-stitched octree region merge.

/// @brief Gather a cell's 8 cube-corner SDF values and gradients from the neighbour table into
/// `cornerSdf` / `cornerGradient` (shared by the decimation kernels' per-cell QEF work).
__device__ inline void
loadCellCorners(int32_t cellValueIndex,
                const int32_t *neighborTable,
                const float *sdf,
                const float *gradient,
                BoxStencilGeometry geometry,
                float cornerSdf[8],
                float cornerGradient[8][3]) {
    const int numColumns = geometry.numColumns;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int32_t cornerIndex =
            neighborTable[int64_t(cellValueIndex) * numColumns + geometry.cornerColumn[corner]];
        cornerSdf[corner]         = sdf[cornerIndex];
        cornerGradient[corner][0] = gradient[int64_t(cornerIndex) * 3 + 0];
        cornerGradient[corner][1] = gradient[int64_t(cornerIndex) * 3 + 1];
        cornerGradient[corner][2] = gradient[int64_t(cornerIndex) * 3 + 2];
    }
}

/// @brief Per-cell flatness signal for adaptive decimation: sum the cell's unit edge-crossing
/// normals and normalize. If the crossings share a direction (a flat patch) the sum stays near unit
/// length; if they disagree (a feature) it partly cancels. Block flatness is then the mean of these
/// over a block.
__global__ void
cellNormalKernel(const int32_t *surfaceCells,
                 int64_t numSurfaceCells,
                 const int32_t *neighborTable,
                 const float *sdf,
                 const float *gradient,
                 BoxStencilGeometry geometry,
                 float iso,
                 float *cellNormals) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    float cornerSdf[8], cornerGradient[8][3];
    loadCellCorners(
        surfaceCells[i], neighborTable, sdf, gradient, geometry, cornerSdf, cornerGradient);
    float normalSum[3] = {0, 0, 0};
#pragma unroll
    for (int edge = 0; edge < 12; ++edge) {
        EdgeCrossing ec;
        if (!computeEdgeCrossing(edge, cornerSdf, cornerGradient, geometry, iso, ec))
            continue;
        normalSum[0] += ec.normal[0];
        normalSum[1] += ec.normal[1];
        normalSum[2] += ec.normal[2];
    }
    float invLen           = rsqrtf(normalSum[0] * normalSum[0] + normalSum[1] * normalSum[1] +
                          normalSum[2] * normalSum[2] + 1e-24f);
    cellNormals[i * 3 + 0] = normalSum[0] * invLen;
    cellNormals[i * 3 + 1] = normalSum[1] * invLen;
    cellNormals[i * 3 + 2] = normalSum[2] * invLen;
}

__device__ inline long long
packKey(long long x, long long y, long long z, long long spanY, long long spanZ) {
    return (x * spanY + y) * spanZ + z;
}

/// @brief A surface cell's index-space coord relative to its grid's key origin (the minimum
/// surface-cell coord of the grid), and the grid's key parameter row.
struct CellKeyFrame {
    const int64_t *params; // the grid's kNumGridKeyParams row
    long long localX, localY, localZ;
};

__device__ inline CellKeyFrame
cellKeyFrame(int64_t i,
             const int32_t *surfaceCells,
             const int32_t *surfaceGrid,
             const int32_t *voxelCoord,
             const int64_t *gridKeyParams) {
    const int64_t *params        = gridKeyParams + int64_t(surfaceGrid[i]) * kNumGridKeyParams;
    const int32_t cellValueIndex = surfaceCells[i];
    return {params,
            voxelCoord[int64_t(cellValueIndex) * 3 + 0] - params[kKeyOriginX],
            voxelCoord[int64_t(cellValueIndex) * 3 + 1] - params[kKeyOriginY],
            voxelCoord[int64_t(cellValueIndex) * 3 + 2] - params[kKeyOriginZ]};
}

/// @brief Uniform-block cluster key (also the coarse-block key reused in the adaptive pass). Each
/// grid's keys lie in its own [kKeyBase, kKeyBase + kKeyFineCount) range, so no cluster spans
/// grids.
__global__ void
clusterKeyKernel(const int32_t *surfaceCells,
                 const int32_t *surfaceGrid,
                 int64_t numSurfaceCells,
                 const int32_t *voxelCoord,
                 const int64_t *gridKeyParams,
                 int blockSize,
                 long long *clusterKeys) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    const CellKeyFrame f = cellKeyFrame(i, surfaceCells, surfaceGrid, voxelCoord, gridKeyParams);
    clusterKeys[i]       = f.params[kKeyBase] + packKey(f.localX / blockSize,
                                                  f.localY / blockSize,
                                                  f.localZ / blockSize,
                                                  f.params[kKeySpanY],
                                                  f.params[kKeySpanZ]);
}

/// @brief Adaptive cluster key: a cell in a flat block takes its coarse-block id (so the whole
/// block collapses to one vertex); a cell in a feature block takes its own per-voxel key (so it
/// stays full detail). Within the grid's key range the flat-block keys are offset by kKeyFineCount
/// so the fine and coarse key spaces can never collide.
__global__ void
clusterKeyAdaptiveKernel(const int32_t *surfaceCells,
                         const int32_t *surfaceGrid,
                         int64_t numSurfaceCells,
                         const int32_t *voxelCoord,
                         const int64_t *gridKeyParams,
                         int blockSize,
                         const int32_t *blockOfCell,
                         const uint8_t *blockIsFlat,
                         long long *clusterKeys) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    const CellKeyFrame f  = cellKeyFrame(i, surfaceCells, surfaceGrid, voxelCoord, gridKeyParams);
    const long long spanY = f.params[kKeySpanY], spanZ = f.params[kKeySpanZ];
    if (blockIsFlat[blockOfCell[i]])
        clusterKeys[i] =
            f.params[kKeyBase] + f.params[kKeyFineCount] +
            packKey(f.localX / blockSize, f.localY / blockSize, f.localZ / blockSize, spanY, spanZ);
    else
        clusterKeys[i] = f.params[kKeyBase] + packKey(f.localX, f.localY, f.localZ, spanY, spanZ);
}

/// @brief Accumulate the per-cluster QEF: each surface cell builds the same 12-edge normal-equation
/// terms as placeQefVerticesKernel (A = sum n n^T, b = sum n (n.p), and the crossing centroid), but
/// expressed in the cluster's shared origin frame and atomic-added into its cluster's running
/// totals. clusterSolveKernel then solves one vertex from each cluster's summed system. (double
/// atomicAdd -> order-dependent; see the decimation note above.)
__global__ void
clusterQefAccumKernel(const int32_t *surfaceCells,
                      const int32_t *surfaceGrid,
                      const int32_t *clusterIds,
                      int64_t numSurfaceCells,
                      const int32_t *neighborTable,
                      const float *sdf,
                      const float *gradient,
                      const int32_t *voxelCoord,
                      BoxStencilGeometry geometry,
                      float iso,
                      const int64_t *gridKeyParams,
                      double *normalMatrix,
                      double *normalRhs,
                      double *crossingSum,
                      double *crossingCount,
                      double *normalSum) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    int32_t cellValueIndex = surfaceCells[i];
    int clusterId          = clusterIds[i];
    float cornerSdf[8], cornerGradient[8][3];
    loadCellCorners(
        cellValueIndex, neighborTable, sdf, gradient, geometry, cornerSdf, cornerGradient);
    // the cell's origin in its grid's key frame (shared by every cell of a cluster)
    const CellKeyFrame frame =
        cellKeyFrame(i, surfaceCells, surfaceGrid, voxelCoord, gridKeyParams);
    double cellOriginX = (double)frame.localX, cellOriginY = (double)frame.localY,
           cellOriginZ    = (double)frame.localZ;
    double localMatrix[6] = {0, 0, 0, 0, 0, 0}, localRhs[3] = {0, 0, 0},
           localCrossingSum[3] = {0, 0, 0}, localNormalSum[3] = {0, 0, 0};
    int numCrossings = 0;
#pragma unroll
    for (int edge = 0; edge < 12; ++edge) {
        EdgeCrossing ec;
        if (!computeEdgeCrossing(edge, cornerSdf, cornerGradient, geometry, iso, ec))
            continue;
        // shift the cell-local crossing into the cluster's shared origin frame before accumulating
        const float nx = ec.normal[0], ny = ec.normal[1], nz = ec.normal[2];
        const double px = cellOriginX + ec.point[0], py = cellOriginY + ec.point[1],
                     pz                = cellOriginZ + ec.point[2],
                     normalDotCrossing = nx * px + ny * py + nz * pz;
        localMatrix[0] += nx * nx;
        localMatrix[1] += nx * ny;
        localMatrix[2] += nx * nz;
        localMatrix[3] += ny * ny;
        localMatrix[4] += ny * nz;
        localMatrix[5] += nz * nz;
        localRhs[0] += nx * normalDotCrossing;
        localRhs[1] += ny * normalDotCrossing;
        localRhs[2] += nz * normalDotCrossing;
        localCrossingSum[0] += px;
        localCrossingSum[1] += py;
        localCrossingSum[2] += pz;
        localNormalSum[0] += nx;
        localNormalSum[1] += ny;
        localNormalSum[2] += nz;
        ++numCrossings;
    }
    for (int j = 0; j < 6; ++j)
        gpuAtomicAddNoReturn(&normalMatrix[clusterId * 6 + j], localMatrix[j]);
    for (int j = 0; j < 3; ++j) {
        gpuAtomicAddNoReturn(&normalRhs[clusterId * 3 + j], localRhs[j]);
        gpuAtomicAddNoReturn(&crossingSum[clusterId * 3 + j], localCrossingSum[j]);
        gpuAtomicAddNoReturn(&normalSum[clusterId * 3 + j], localNormalSum[j]);
    }
    gpuAtomicAddNoReturn(&crossingCount[clusterId], (double)numCrossings);
}

/// @brief Solve one vertex per cluster from its accumulated QEF -- same centroid re-centring + 0.05
/// Tikhonov regularisation as placeQefVerticesKernel -- clamp it within +/- blockSize of the
/// cluster centroid, then write the world-space vertex and its reference normal (the summed cluster
/// normal).
__global__ void
clusterSolveKernel(int64_t numClusters,
                   const int32_t *clusterGrid,
                   const double *normalMatrix,
                   const double *normalRhs,
                   const double *crossingSum,
                   const double *crossingCount,
                   const double *normalSum,
                   const int64_t *gridKeyParams,
                   int blockSize,
                   GridBatchData::Accessor batchAcc,
                   float *vertices,
                   float *refNormal) {
    int64_t clusterId = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (clusterId >= numClusters)
        return;
    double count = crossingCount[clusterId], invCount = count > 0 ? 1.0 / count : 0.0;
    double crossingCentroid[3]  = {crossingSum[clusterId * 3 + 0] * invCount,
                                   crossingSum[clusterId * 3 + 1] * invCount,
                                   crossingSum[clusterId * 3 + 2] * invCount};
    double regularizedMatrix[6] = {normalMatrix[clusterId * 6 + 0] + 0.05,
                                   normalMatrix[clusterId * 6 + 1],
                                   normalMatrix[clusterId * 6 + 2],
                                   normalMatrix[clusterId * 6 + 3] + 0.05,
                                   normalMatrix[clusterId * 6 + 4],
                                   normalMatrix[clusterId * 6 + 5] + 0.05};
    double rhs[3]               = {normalRhs[clusterId * 3 + 0] + 0.05 * crossingCentroid[0],
                                   normalRhs[clusterId * 3 + 1] + 0.05 * crossingCentroid[1],
                                   normalRhs[clusterId * 3 + 2] + 0.05 * crossingCentroid[2]};
    double solution[3];
    solveSymmetric3x3(regularizedMatrix, rhs, solution);
    for (int axis = 0; axis < 3; ++axis)
        solution[axis] = fmin(fmax(solution[axis], crossingCentroid[axis] - blockSize),
                              crossingCentroid[axis] + blockSize);
    // back from the grid's key frame to index space, then to world with the grid's transform
    const int32_t grid    = clusterGrid[clusterId];
    const int64_t *params = gridKeyParams + int64_t(grid) * kNumGridKeyParams;
    auto worldPos =
        batchAcc.primalTransform(grid).applyInv<float>((float)(solution[0] + params[kKeyOriginX]),
                                                       (float)(solution[1] + params[kKeyOriginY]),
                                                       (float)(solution[2] + params[kKeyOriginZ]));
    vertices[clusterId * 3 + 0]  = worldPos[0];
    vertices[clusterId * 3 + 1]  = worldPos[1];
    vertices[clusterId * 3 + 2]  = worldPos[2];
    refNormal[clusterId * 3 + 0] = (float)normalSum[clusterId * 3 + 0];
    refNormal[clusterId * 3 + 1] = (float)normalSum[clusterId * 3 + 1];
    refNormal[clusterId * 3 + 2] = (float)normalSum[clusterId * 3 + 2];
}

/// @brief Per-grid bounds of the surface-cell coords: bounds[g] = [min x,y,z, max x,y,z] over grid
/// g's surface cells (initialised to +/-INT32_MAX by the caller). Surface cells come ordered by
/// slot and hence grouped by grid, so almost every warp holds a single grid: such a warp reduces
/// in registers and issues one atomic per bound, instead of 32 atomics contending on the same six
/// words (a warp straddling two grids, or the partial last warp, falls back to per-thread
/// atomics).
static_assert(DEFAULT_BLOCK_DIM % 32 == 0, "surfaceBoundsKernel reduces whole warps");

__global__ void
surfaceBoundsKernel(const int32_t *surfaceCells,
                    const int32_t *surfaceGrid,
                    int64_t numSurfaceCells,
                    const int32_t *voxelCoord,
                    int32_t *bounds) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numSurfaceCells)
        return;
    const int32_t grid = surfaceGrid[i];
    const int64_t cell = surfaceCells[i];
    int32_t lo[3], hi[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis)
        lo[axis] = hi[axis] = voxelCoord[cell * 3 + axis];
    int32_t *gridBounds = bounds + int64_t(grid) * 6;
    // every lane of a warp whose last index is in range got past the early return above (the
    // launch's block size is a multiple of 32), so the warp can synchronize as a whole
    constexpr unsigned kFullMask = 0xffffffffu;
    const bool fullWarp          = (i - (threadIdx.x & 31)) + 32 <= numSurfaceCells;
    if (fullWarp && __all_sync(kFullMask, grid == __shfl_sync(kFullMask, grid, 0))) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = min(lo[axis], __shfl_xor_sync(kFullMask, lo[axis], offset));
                hi[axis] = max(hi[axis], __shfl_xor_sync(kFullMask, hi[axis], offset));
            }
        }
        if ((threadIdx.x & 31) != 0)
            return;
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        atomicMin(&gridBounds[axis], lo[axis]);
        atomicMax(&gridBounds[3 + axis], hi[axis]);
    }
}

// ------------------------- host helpers -------------------------
/// @brief Dense [0,numUnique) relabel of `keys` in original order: expressed via torch::unique_dim
/// exactly as MarchingCubes does. Returns the per-element int32 labels (sorted-key order) and
/// writes the unique count to `numUnique`.
static torch::Tensor
denseRelabel(const torch::Tensor &keys, int64_t &numUnique) {
    auto result = torch::unique_dim(keys, 0, /*sorted=*/true, /*return_inverse=*/true);
    numUnique   = std::get<0>(result).size(0);
    return std::get<1>(result).to(torch::kInt32);
}

/// @brief Per-dtype TensorOptions for this op's scratch/output tensors (all on the same device).
/// Bundled so the chunk mesher takes a single argument instead of one option per dtype.
struct ScratchOptions {
    torch::TensorOptions f32, i32, i64, u8, f64;
    explicit ScratchOptions(torch::Device device)
        : f32(torch::TensorOptions().dtype(torch::kFloat32).device(device)),
          i32(torch::TensorOptions().dtype(torch::kInt32).device(device)),
          i64(torch::TensorOptions().dtype(torch::kInt64).device(device)),
          u8(torch::TensorOptions().dtype(torch::kUInt8).device(device)),
          f64(torch::TensorOptions().dtype(torch::kFloat64).device(device)) {}
};

/// @brief The per-grid decimation-key parameters ([numGrids, kNumGridKeyParams] int64, on the
/// device) from the surface cells' coords and grids, without a host sync. Each grid's key origin is
/// its minimum surface-cell coord and its spans cover its surface cells, exactly as when each grid
/// was meshed on its own; the key bases then give each grid a disjoint range of
/// 2 * kKeyFineCount keys (fine keys, then the adaptive pass's coarse keys). Grids without surface
/// cells get empty spans and an empty key range.
static torch::Tensor
buildGridKeyParams(const int32_t *surfaceCells,
                   const int32_t *surfaceGrid,
                   int64_t numSurfaceCells,
                   const int32_t *voxelCoord,
                   int64_t numGrids,
                   cudaStream_t stream,
                   const ScratchOptions &opts) {
    // per-grid [min x, min y, min z, max x, max y, max z] of the surface-cell coords
    torch::Tensor bounds = torch::empty({numGrids, 6}, opts.i32);
    bounds.narrow(1, 0, 3).fill_(std::numeric_limits<int32_t>::max());
    bounds.narrow(1, 3, 3).fill_(std::numeric_limits<int32_t>::min());
    surfaceBoundsKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                          DEFAULT_BLOCK_DIM,
                          0,
                          stream>>>(
        surfaceCells, surfaceGrid, numSurfaceCells, voxelCoord, bounds.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const torch::Tensor wide      = bounds.to(torch::kInt64);
    const torch::Tensor lo        = wide.narrow(1, 0, 3);
    const torch::Tensor hi        = wide.narrow(1, 3, 3);
    const torch::Tensor span      = (hi - lo + 1).clamp_min(0); // 0 for grids without surface cells
    const torch::Tensor fineCount = span.prod(1);
    const torch::Tensor keyRange  = 2 * fineCount;
    const torch::Tensor keyBase   = torch::cumsum(keyRange, 0) - keyRange; // exclusive prefix sum
    return torch::stack({lo.select(1, 0),
                         lo.select(1, 1),
                         lo.select(1, 2),
                         span.select(1, 1),
                         span.select(1, 2),
                         fineCount,
                         keyBase},
                        1)
        .contiguous();
}

/// @brief One chunk's mesh: vertices/normals grouped by grid (ascending), triangles holding
/// grid-local vertex indices, and the grid of every vertex and triangle.
struct ChunkMesh {
    torch::Tensor vertices;     // [V, 3] float32
    torch::Tensor normals;      // [V, 3] float32
    torch::Tensor triangles;    // [T, 3] int64, indices local to the triangle's grid
    torch::Tensor vertexGrid;   // [V] int64
    torch::Tensor triangleGrid; // [T] int64
};

/// @brief Mesh the consecutive grids [gridBegin, gridEnd) of the batch together. Slots are
/// chunk-wide (slot 1 + f - chunkVoxelBase for the voxel with batch-wide feature index f, slot 0 =
/// the "outside" sentinel used for inactive neighbours). Pipeline:
///   1. gatherFusedKernel  -- one VBM decode per voxel of every grid of the chunk -> neighbour
///                            table + gradient + surface flag + index coord
///   2. compact the surface cells (torch::nonzero) -- the cells the iso surface passes through --
///      and find each one's grid (searchsorted over the batch's voxel offsets)
///   3. place one vertex per surface cell: QEF (placeQefVerticesKernel), or when decimating group
///      the cells into per-grid clusters and solve one vertex per cluster
///   4. connectivity: one quad per sign-changing minimal grid edge -> triangles (count -> cumsum ->
///      write), each oriented outward by the SDF gradient
///   5. prune unreferenced vertices, rebase the triangles onto grid-local vertex indices, and
///      normalise the per-vertex reference normals
/// Surface cells, and hence vertices and triangles, come out ordered by slot and so grouped by
/// grid; a grid's slice is identical to meshing that grid alone (bitwise, without decimation).
/// Host syncs: steps 2, 4 and 5 once each, plus one or two cluster relabels when decimating.
static ChunkMesh
meshChunk(const GridBatchData &batchHdl,
          const BatchVbm &vbm,
          const BatchVbmLayout &layout,
          int64_t gridBegin,
          int64_t gridEnd,
          const torch::Tensor &fieldData,
          float iso,
          int reduce,
          double adaptivity,
          cudaStream_t stream,
          const ScratchOptions &opts) {
    const bool decimate          = (reduce != 1) || (adaptivity > 0.0);
    const int numColumns         = kGeometry.numColumns;
    const int64_t numGrids       = batchHdl.batchSize();
    const int64_t chunkVoxelBase = batchHdl.cumVoxelsAt(gridBegin);
    const int64_t chunkVoxelEnd =
        gridEnd < numGrids ? batchHdl.cumVoxelsAt(gridEnd) : batchHdl.totalVoxels();
    const int64_t numChunkVoxels = chunkVoxelEnd - chunkVoxelBase;
    const int64_t valueCount     = numChunkVoxels + 1; // slot 0 = background

    auto emptyChunk = [&] {
        return ChunkMesh{torch::empty({0, 3}, opts.f32),
                         torch::empty({0, 3}, opts.f32),
                         torch::empty({0, 3}, opts.i64),
                         torch::empty({0}, opts.i64),
                         torch::empty({0}, opts.i64)};
    };
    if (numChunkVoxels == 0)
        return emptyChunk();

    // value-indexed SDF: slot 0 = "outside" sentinel (>= iso) so inactive corners classify as
    // outside, slots [1..numChunkVoxels] = the chunk's slice of the field.
    torch::Tensor sdfBuf = torch::empty({valueCount}, opts.f32);
    sdfBuf.narrow(0, 0, 1).fill_(iso + 1.0f);
    sdfBuf.narrow(0, 1, numChunkVoxels).copy_(fieldData.narrow(0, chunkVoxelBase, numChunkVoxels));
    const float *sdf = sdfBuf.data_ptr<float>();

    torch::Tensor neighborTableBuf = torch::empty({valueCount * numColumns}, opts.i32);
    torch::Tensor gradientBuf      = torch::empty({valueCount * 3}, opts.f32);
    torch::Tensor surfaceFlagBuf   = torch::empty({valueCount}, opts.u8);
    torch::Tensor voxelCoordBuf    = torch::empty({valueCount * 3}, opts.i32);
    int32_t *neighborTable         = neighborTableBuf.data_ptr<int32_t>();
    float *gradient                = gradientBuf.data_ptr<float>();
    uint8_t *surfaceFlag           = surfaceFlagBuf.data_ptr<uint8_t>();
    int32_t *voxelCoord            = voxelCoordBuf.data_ptr<int32_t>();
    C10_CUDA_CHECK(cudaMemsetAsync(gradient, 0, 3 * sizeof(float), stream)); // background slot 0
    C10_CUDA_CHECK(cudaMemsetAsync(surfaceFlag, 0, sizeof(uint8_t), stream));

    // the chunk's VBM blocks: from gridBegin's first block to the next chunk's first block (or the
    // end of the batch); gap blocks between grids are skipped by the kernel.
    const uint64_t vbBegin = VbmCache::blockBase(chunkVoxelBase, gridBegin);
    const uint64_t vbEnd =
        gridEnd < numGrids ? VbmCache::blockBase(chunkVoxelEnd, gridEnd) : vbm.blockCount;
    TORCH_CHECK(vbEnd - vbBegin <= uint64_t(std::numeric_limits<int32_t>::max()),
                "dual_contour: too many VBM blocks in one chunk");
    gatherFusedKernel<<<uint32_t(vbEnd - vbBegin), kBlockWidth, 0, stream>>>(vbm,
                                                                             layout,
                                                                             vbBegin,
                                                                             chunkVoxelBase,
                                                                             sdf,
                                                                             kGeometry,
                                                                             iso,
                                                                             neighborTable,
                                                                             gradient,
                                                                             surfaceFlag,
                                                                             voxelCoord);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // compact surface cells: torch::nonzero returns the set slots in ascending order, giving a
    // stable vertex order grouped by grid (== the source's copy_if over a counting iterator).
    torch::Tensor surfaceCellsLong = torch::nonzero(surfaceFlagBuf).flatten(); // host sync
    const int64_t numSurfaceCells  = surfaceCellsLong.size(0);
    if (numSurfaceCells == 0)
        return emptyChunk();
    torch::Tensor surfaceCellBuf = surfaceCellsLong.to(torch::kInt32);
    const int32_t *surfaceCells  = surfaceCellBuf.data_ptr<int32_t>();
    // each surface cell's grid: the last grid whose first feature index is <= the cell's (known on
    // the host when the chunk holds a single grid)
    const bool singleGrid = gridEnd - gridBegin == 1;
    torch::Tensor surfaceGridLong =
        singleGrid ? torch::full({numSurfaceCells}, gridBegin, opts.i64)
                   : torch::searchsorted(batchHdl.voxelOffsets(),
                                         surfaceCellsLong + (chunkVoxelBase - 1),
                                         /*out_int32=*/false,
                                         /*right=*/true) -
                         1;
    torch::Tensor surfaceGridBuf           = surfaceGridLong.to(torch::kInt32);
    const int32_t *surfaceGrid             = surfaceGridBuf.data_ptr<int32_t>();
    const GridBatchData::Accessor batchAcc = batchHdl.deviceAccessor();

    torch::Tensor cellVertexBuf = torch::full({valueCount}, -1, opts.i32);
    int32_t *cellVertex         = cellVertexBuf.data_ptr<int32_t>();

    int64_t numVertices = 0;
    torch::Tensor vertexBuf, refNormalBuf, vertexGrid;
    if (!decimate) {
        numVertices = numSurfaceCells;
        vertexGrid  = surfaceGridLong;
        // cellVertex[surfaceCells[i]] = i (the compaction ordinal); background slots stay -1.
        cellVertexBuf.index_put_({surfaceCellsLong}, torch::arange(numSurfaceCells, opts.i32));
        vertexBuf    = torch::empty({numSurfaceCells * 3}, opts.f32);
        refNormalBuf = torch::empty({numSurfaceCells * 3}, opts.f32);
        placeQefVerticesKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                                 DEFAULT_BLOCK_DIM,
                                 0,
                                 stream>>>(surfaceCells,
                                           surfaceGrid,
                                           numSurfaceCells,
                                           neighborTable,
                                           sdf,
                                           gradient,
                                           voxelCoord,
                                           kGeometry,
                                           iso,
                                           batchAcc,
                                           vertexBuf.data_ptr<float>(),
                                           refNormalBuf.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // Cluster block size in voxels. Uniform decimation uses `reduce` directly. The curvature-
        // adaptive path (adaptivity > 0, reduce left at its default) groups cells into 8^3 coarse
        // blocks: a flat block collapses to one vertex while feature blocks stay full resolution,
        // so 8 sets only how coarse the *flat* regions become.
        const int blockSize = (adaptivity > 0.0 && reduce <= 1) ? 8 : std::max(1, reduce);
        // per-grid key origins/spans/bases from the surface-cell coords, on the device
        torch::Tensor gridKeyParamsBuf = buildGridKeyParams(
            surfaceCells, surfaceGrid, numSurfaceCells, voxelCoord, numGrids, stream, opts);
        const int64_t *gridKeyParams = gridKeyParamsBuf.data_ptr<int64_t>();

        torch::Tensor clusterKeyBuf = torch::empty({numSurfaceCells}, opts.i64);
        auto *clusterKeys = reinterpret_cast<long long *>(clusterKeyBuf.data_ptr<int64_t>());
        torch::Tensor clusterIdBuf;
        int64_t numClusters = 0;
        clusterKeyKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                           DEFAULT_BLOCK_DIM,
                           0,
                           stream>>>(surfaceCells,
                                     surfaceGrid,
                                     numSurfaceCells,
                                     voxelCoord,
                                     gridKeyParams,
                                     blockSize,
                                     clusterKeys);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (adaptivity <= 0.0) { // uniform blocks
            clusterIdBuf = denseRelabel(clusterKeyBuf, numClusters);
        } else {                 // flat blocks collapse, feature cells stay fine
            torch::Tensor cellNormalBuf = torch::empty({numSurfaceCells * 3}, opts.f32);
            cellNormalKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                               DEFAULT_BLOCK_DIM,
                               0,
                               stream>>>(surfaceCells,
                                         numSurfaceCells,
                                         neighborTable,
                                         sdf,
                                         gradient,
                                         kGeometry,
                                         iso,
                                         cellNormalBuf.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            int64_t numBlocks            = 0;
            torch::Tensor blockOfCellBuf = denseRelabel(clusterKeyBuf, numBlocks);
            // a block is "flat" when the mean magnitude of its cells' unit normals (i.e. how
            // aligned they are) clears cos(threshold); flat blocks then collapse to one vertex. The
            // scatter-add and per-block reduction are index_add_ + bincount + a vectorized
            // comparison.
            torch::Tensor blockNormalSum =
                torch::zeros({numBlocks, 3}, opts.f32)
                    .index_add_(0, blockOfCellBuf, cellNormalBuf.view({-1, 3}));
            torch::Tensor blockCellCount =
                torch::bincount(blockOfCellBuf, /*weights=*/{}, numBlocks)
                    .clamp_min(1)
                    .to(torch::kFloat32);
            const float flatCosThreshold =
                (float)std::cos(adaptivity * 60.0 * nanovdb::math::pi<double>() / 180.0);
            torch::Tensor blockFlatBuf =
                (blockNormalSum.norm(2, /*dim=*/1) / blockCellCount >= flatCosThreshold)
                    .to(torch::kUInt8);
            clusterKeyAdaptiveKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                                       DEFAULT_BLOCK_DIM,
                                       0,
                                       stream>>>(surfaceCells,
                                                 surfaceGrid,
                                                 numSurfaceCells,
                                                 voxelCoord,
                                                 gridKeyParams,
                                                 blockSize,
                                                 blockOfCellBuf.data_ptr<int32_t>(),
                                                 blockFlatBuf.data_ptr<uint8_t>(),
                                                 clusterKeys);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            clusterIdBuf = denseRelabel(clusterKeyBuf, numClusters);
        }
        int32_t *clusterIds = clusterIdBuf.data_ptr<int32_t>();
        numVertices         = numClusters;
        // cellVertex[surfaceCells[i]] = clusterIds[i] (background slots stay -1).
        cellVertexBuf.index_put_({surfaceCellsLong}, clusterIdBuf);
        if (numClusters == 0)
            return emptyChunk();
        // every cell of a cluster is in the same grid (the key ranges are disjoint per grid), so
        // this scatter is race-free in value; clusters are sorted by key and so grouped by grid.
        vertexGrid = torch::empty({numClusters}, opts.i64)
                         .index_put_({clusterIdBuf.to(torch::kInt64)}, surfaceGridLong);
        torch::Tensor clusterGridBuf   = vertexGrid.to(torch::kInt32);
        torch::Tensor normalMatrixBuf  = torch::zeros({numClusters * 6}, opts.f64),
                      normalRhsBuf     = torch::zeros({numClusters * 3}, opts.f64),
                      crossingSumBuf   = torch::zeros({numClusters * 3}, opts.f64),
                      crossingCountBuf = torch::zeros({numClusters}, opts.f64),
                      normalSumBuf     = torch::zeros({numClusters * 3}, opts.f64);
        clusterQefAccumKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                                DEFAULT_BLOCK_DIM,
                                0,
                                stream>>>(surfaceCells,
                                          surfaceGrid,
                                          clusterIds,
                                          numSurfaceCells,
                                          neighborTable,
                                          sdf,
                                          gradient,
                                          voxelCoord,
                                          kGeometry,
                                          iso,
                                          gridKeyParams,
                                          normalMatrixBuf.data_ptr<double>(),
                                          normalRhsBuf.data_ptr<double>(),
                                          crossingSumBuf.data_ptr<double>(),
                                          crossingCountBuf.data_ptr<double>(),
                                          normalSumBuf.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        vertexBuf    = torch::empty({numClusters * 3}, opts.f32);
        refNormalBuf = torch::empty({numClusters * 3}, opts.f32);
        clusterSolveKernel<<<GET_BLOCKS(numClusters, DEFAULT_BLOCK_DIM),
                             DEFAULT_BLOCK_DIM,
                             0,
                             stream>>>(numClusters,
                                       clusterGridBuf.data_ptr<int32_t>(),
                                       normalMatrixBuf.data_ptr<double>(),
                                       normalRhsBuf.data_ptr<double>(),
                                       crossingSumBuf.data_ptr<double>(),
                                       crossingCountBuf.data_ptr<double>(),
                                       normalSumBuf.data_ptr<double>(),
                                       gridKeyParams,
                                       blockSize,
                                       batchAcc,
                                       vertexBuf.data_ptr<float>(),
                                       refNormalBuf.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // connectivity: per-surface-cell triangle counts -> cumsum into contiguous write ranges ->
    // write. This is the MarchingCubes pattern; the deterministic offsets replace a shared atomic
    // append counter, and iterating surface cells (not every active voxel) keeps the launch tight.
    torch::Tensor triangleCountBuf = torch::empty({numSurfaceCells}, opts.i64);
    countTrianglesKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                           DEFAULT_BLOCK_DIM,
                           0,
                           stream>>>(surfaceCells,
                                     numSurfaceCells,
                                     neighborTable,
                                     sdf,
                                     cellVertex,
                                     kGeometry,
                                     iso,
                                     triangleCountBuf.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    torch::Tensor triangleOffsetBuf = torch::cumsum(triangleCountBuf, 0);    // inclusive prefix sum
    const int64_t numTriangles      = triangleOffsetBuf[-1].item<int64_t>(); // host sync
    if (numTriangles == 0 || numVertices == 0)
        return emptyChunk();
    triangleOffsetBuf    = torch::roll(triangleOffsetBuf, {1}); // shift -> exclusive prefix sum
    triangleOffsetBuf[0] = 0;
    torch::Tensor triangleBuf = torch::empty({numTriangles * 3}, opts.i32);
    int32_t *triangles        = triangleBuf.data_ptr<int32_t>();
    writeTrianglesKernel<<<GET_BLOCKS(numSurfaceCells, DEFAULT_BLOCK_DIM),
                           DEFAULT_BLOCK_DIM,
                           0,
                           stream>>>(surfaceCells,
                                     numSurfaceCells,
                                     neighborTable,
                                     sdf,
                                     cellVertex,
                                     kGeometry,
                                     iso,
                                     triangleOffsetBuf.data_ptr<int64_t>(),
                                     triangles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    orientTrianglesKernel<<<GET_BLOCKS(numTriangles, DEFAULT_BLOCK_DIM),
                            DEFAULT_BLOCK_DIM,
                            0,
                            stream>>>(
        numTriangles, vertexBuf.data_ptr<float>(), refNormalBuf.data_ptr<float>(), triangles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // prune unreferenced (free-point) vertices and reindex the triangles. torch::unique_dim over
    // the flattened triangle indices returns, in ascending order, the referenced vertex ids and
    // (via return_inverse) the triangles already reindexed into [0, numKeptVertices) -- the same
    // merge idiom MarchingCubes uses. Compacting the vertices/normals is then a single index_select
    // gather. The kept vertices stay grouped by grid, so subtracting each grid's first kept vertex
    // turns the chunk-wide indices into grid-local ones.
    auto unique = torch::unique_dim(triangleBuf, 0, /*sorted=*/true, /*return_inverse=*/true);
    torch::Tensor keptVertexIds  = std::get<0>(unique).to(torch::kInt64); // ascending; host sync
    const int64_t numOutVertices = keptVertexIds.size(0);
    torch::Tensor chunkTriangles = std::get<1>(unique).to(torch::kInt64).view({numTriangles, 3});

    ChunkMesh mesh;
    mesh.vertices = vertexBuf.view({numVertices, 3}).index_select(0, keptVertexIds);
    mesh.normals  = refNormalBuf.view({numVertices, 3}).index_select(0, keptVertexIds);
    mesh.normals =
        mesh.normals / mesh.normals.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-12);
    if (singleGrid) { // chunk-wide vertex indices are already grid-local
        mesh.vertexGrid   = torch::full({numOutVertices}, gridBegin, opts.i64);
        mesh.triangleGrid = torch::full({numTriangles}, gridBegin, opts.i64);
        mesh.triangles    = chunkTriangles;
    } else {
        mesh.vertexGrid   = vertexGrid.index_select(0, keptVertexIds);
        mesh.triangleGrid = mesh.vertexGrid.index_select(0, chunkTriangles.select(1, 0));
        torch::Tensor gridVertexCount = torch::bincount(mesh.vertexGrid, /*weights=*/{}, numGrids);
        torch::Tensor gridVertexBase  = torch::cumsum(gridVertexCount, 0) - gridVertexCount;
        mesh.triangles =
            chunkTriangles - gridVertexBase.index_select(0, mesh.triangleGrid).unsqueeze(1);
    }
    TORCH_INTERNAL_ASSERT(mesh.vertices.size(0) == numOutVertices);
    return mesh;
}

} // namespace

std::tuple<JaggedTensor, JaggedTensor, JaggedTensor>
dualContour(const GridBatchData &batchHdl,
            const JaggedTensor &field,
            double iso,
            int reduce,
            double adaptivity) {
    TORCH_CHECK_VALUE(
        field.ldim() == 1,
        "Expected field to have 1 list dimension (a single list of per-voxel values)");
    TORCH_CHECK_TYPE(field.is_floating_point(), "field must have a floating point type");
    TORCH_CHECK_VALUE(field.numel() == batchHdl.totalVoxels(),
                      "field value count does not match the number of voxels in the grid");
    TORCH_CHECK_VALUE(field.num_outer_lists() == batchHdl.batchSize(),
                      "field batch size does not match the grid batch size");
    batchHdl.checkDevice(field);
    TORCH_CHECK(field.device().is_cuda(), "dual_contour currently requires a CUDA device");
    adaptivity = std::clamp(adaptivity, 0.0, 1.5); // beyond 1.5 the flatness bar saturates
    reduce     = std::max(1, reduce);

    const auto device = field.device();
    c10::cuda::CUDAGuard deviceGuard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    ScratchOptions opts(device);

    torch::Tensor fieldData = field.jdata().contiguous();
    if (fieldData.dim() != 1)
        fieldData = fieldData.view({-1});
    if (fieldData.scalar_type() != torch::kFloat32)
        fieldData =
            fieldData.to(torch::kFloat32); // QEF accumulators are double; the SDF stays float32

    // The batch's VBMs (built on first use) and their placement, shared by every chunk.
    const int64_t numGrids      = batchHdl.batchSize();
    const BatchVbm &vbm         = batchHdl.vbm();
    const BatchVbmLayout layout = BatchVbmLayout::of(batchHdl);

    // Greedily group consecutive grids into chunks of at most kMaxChunkVoxels voxels; a single grid
    // larger than that forms its own chunk. Chunk-local slots are int32 (the neighbour table and
    // the surface-cell / triangle buffers), so a chunk must fit in int32 value indexing.
    std::vector<ChunkMesh> chunks;
    for (int64_t gridBegin = 0; gridBegin < numGrids;) {
        int64_t gridEnd   = gridBegin + 1;
        int64_t numVoxels = batchHdl.numVoxelsAt(gridBegin);
        while (gridEnd < numGrids && numVoxels + batchHdl.numVoxelsAt(gridEnd) <= kMaxChunkVoxels)
            numVoxels += batchHdl.numVoxelsAt(gridEnd++);
        TORCH_CHECK(numVoxels + 1 <= std::numeric_limits<int32_t>::max(),
                    "dual_contour: grid ",
                    gridBegin,
                    " has too many voxels (",
                    numVoxels,
                    ") for int32 value indexing");
        if (numVoxels > 0) {
            chunks.push_back(meshChunk(batchHdl,
                                       vbm,
                                       layout,
                                       gridBegin,
                                       gridEnd,
                                       fieldData,
                                       (float)iso,
                                       reduce,
                                       adaptivity,
                                       stream,
                                       opts));
        }
        gridBegin = gridEnd;
    }

    // concatenate the chunks (each already grouped by grid, chunks in grid order) and build the
    // jagged outputs
    std::vector<torch::Tensor> vertexList, normalList, faceList, vertexGridList, faceGridList;
    for (const ChunkMesh &chunk: chunks) {
        vertexList.push_back(chunk.vertices);
        normalList.push_back(chunk.normals);
        faceList.push_back(chunk.triangles);
        vertexGridList.push_back(chunk.vertexGrid);
        faceGridList.push_back(chunk.triangleGrid);
    }
    auto catOrEmpty = [&](const std::vector<torch::Tensor> &list,
                          std::initializer_list<int64_t> emptyShape,
                          const torch::TensorOptions &options) {
        return list.empty() ? torch::empty(emptyShape, options) : torch::cat(list, 0);
    };
    torch::Tensor allVertices = catOrEmpty(vertexList, {0, 3}, opts.f32);
    torch::Tensor allNormals  = catOrEmpty(normalList, {0, 3}, opts.f32);
    torch::Tensor allFaces    = catOrEmpty(faceList, {0, 3}, opts.i64);
    torch::Tensor vertexBatchIdx =
        catOrEmpty(vertexGridList, {0}, opts.i64).to(fvdb::JIdxScalarType);
    torch::Tensor faceBatchIdx = catOrEmpty(faceGridList, {0}, opts.i64).to(fvdb::JIdxScalarType);

    JaggedTensor retVertices = JaggedTensor::from_data_indices_and_list_ids(
        allVertices, vertexBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());
    JaggedTensor retFaces = JaggedTensor::from_data_indices_and_list_ids(
        allFaces, faceBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());
    JaggedTensor retNormals = JaggedTensor::from_data_indices_and_list_ids(
        allNormals, vertexBatchIdx, batchHdl.jlidx(), batchHdl.batchSize());
    return {retVertices, retFaces, retNormals};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
