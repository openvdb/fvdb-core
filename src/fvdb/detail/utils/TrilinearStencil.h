// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_TRILINEARSTENCIL_H
#define FVDB_DETAIL_UTILS_TRILINEARSTENCIL_H

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace fvdb {
namespace detail {

// Resolve the 8 trilinear corner indices and weights in a single pass.
// Uses NanoVDB TrilinearSampler-style coordinate traversal (increment
// one component at a time) to maximize ReadAccessor node-cache hits
// across successive lookups.
// Inactive corners receive 0 weights, enabling branchless channel loops.
// Returns a bitmask indicating which corners are active in the grid.
template <typename MathType, typename GridAccessorType>
__hostdev__ inline uint8_t
resolveTrilinearStencil(const nanovdb::math::Vec3<MathType> &xyz,
                        GridAccessorType &gridAcc,
                        int64_t baseOffset,
                        int64_t (&indices)[8],
                        MathType (&weights)[8]) {
    nanovdb::Coord ijk = xyz.floor();
    const MathType u   = xyz[0] - MathType(ijk[0]);
    const MathType v   = xyz[1] - MathType(ijk[1]);
    const MathType w   = xyz[2] - MathType(ijk[2]);
    const MathType ONE = MathType(1);
    const MathType U = ONE - u, V = ONE - v, W = ONE - w;

    uint8_t activeMask = 0;

#define FVDB_RESOLVE_CORNER(CORNER, WEIGHT)                       \
    if (gridAcc.isActive(ijk)) {                                  \
        activeMask |= (1 << (CORNER));                            \
        weights[CORNER] = (WEIGHT);                               \
        indices[CORNER] = gridAcc.getValue(ijk) - 1 + baseOffset; \
    } else {                                                      \
        weights[CORNER] = MathType(0);                            \
    }

    FVDB_RESOLVE_CORNER(0, U * V * W) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER(1, U * V * w) // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER(2, U * v * w) // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER(3, U * v * W) // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_RESOLVE_CORNER(4, u * V * W) // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER(5, u * V * w) // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER(6, u * v * w) // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER(7, u * v * W) // (i+1, j+1, k  )

#undef FVDB_RESOLVE_CORNER

    return activeMask;
}

// Resolve the 8 trilinear corner indices, interpolation weights, and spatial-gradient
// weights in a single pass. Uses NanoVDB TrilinearSampler-style coordinate traversal
// (increment one component at a time) for ReadAccessor cache efficiency.
// gradWeights[corner][dim] stores dWeight/d{u,v,w} in index space; the caller applies
// the voxel-to-world gradTransform afterward.
// Inactive corners receive 0 for all weights, enabling branchless channel loops.
// Returns a bitmask indicating which corners are active in the grid.
template <typename MathType, typename GridAccessorType>
__hostdev__ inline uint8_t
resolveTrilinearStencilWithGrad(const nanovdb::math::Vec3<MathType> &xyz,
                                GridAccessorType &gridAcc,
                                int64_t baseOffset,
                                int64_t (&indices)[8],
                                MathType (&weights)[8],
                                MathType (&gradWeights)[8][3]) {
    nanovdb::Coord ijk = xyz.floor();
    const MathType u   = xyz[0] - MathType(ijk[0]);
    const MathType v   = xyz[1] - MathType(ijk[1]);
    const MathType w   = xyz[2] - MathType(ijk[2]);
    const MathType ONE = MathType(1);
    const MathType U = ONE - u, V = ONE - v, W = ONE - w;

    uint8_t activeMask = 0;

#define FVDB_RESOLVE_CORNER_GRAD(CORNER, WT, GU, GV, GW)                 \
    if (gridAcc.isActive(ijk)) {                                         \
        activeMask |= (1 << (CORNER));                                   \
        weights[CORNER]        = (WT);                                   \
        gradWeights[CORNER][0] = (GU);                                   \
        gradWeights[CORNER][1] = (GV);                                   \
        gradWeights[CORNER][2] = (GW);                                   \
        indices[CORNER]        = gridAcc.getValue(ijk) - 1 + baseOffset; \
    } else {                                                             \
        weights[CORNER]        = MathType(0);                            \
        gradWeights[CORNER][0] = MathType(0);                            \
        gradWeights[CORNER][1] = MathType(0);                            \
        gradWeights[CORNER][2] = MathType(0);                            \
    }

    FVDB_RESOLVE_CORNER_GRAD(0, U * V * W, -V * W, -U * W, -U * V) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER_GRAD(1, U * V * w, -V * w, -U * w, U * V)  // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER_GRAD(2, U * v * w, -v * w, U * w, U * v)   // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(3, U * v * W, -v * W, U * W, -U * v)  // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(4, u * V * W, V * W, -u * W, -u * V)  // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_RESOLVE_CORNER_GRAD(5, u * V * w, V * w, -u * w, u * V)   // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_RESOLVE_CORNER_GRAD(6, u * v * w, v * w, u * w, u * v)    // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_RESOLVE_CORNER_GRAD(7, u * v * W, v * W, u * W, -u * v)   // (i+1, j+1, k  )

#undef FVDB_RESOLVE_CORNER_GRAD

    return activeMask;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_TRILINEARSTENCIL_H
