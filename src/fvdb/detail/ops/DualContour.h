// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_DUALCONTOUR_H
#define FVDB_DETAIL_OPS_DUALCONTOUR_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Dual-contour (DC + QEF) mesh an OnIndex grid carrying a narrow-band signed distance
/// field.
///
/// For each cube cell whose 8 corners are all active and whose corner signs straddle @p iso, places
/// one vertex by minimising a quadratic error function built from the 12 edge zero-crossings and
/// their interpolated SDF-gradient normals (regularised toward the crossing centroid, clamped to
/// the cell). One quad per bipolar minimal grid edge joins the 4 surrounding cells; quads are
/// triangulated and oriented outward by the SDF gradient. Optional cluster-collapse decimation
/// (@p reduce / @p adaptivity) contracts the connectivity onto fewer vertices.
/// Requires a >= ~3-voxel band for a watertight result.
///
/// @note Vertex placement and the emitted normals are computed in index space (the QEF and the
///       central-difference gradient). This is exact for isotropic voxel sizes -- the assumption of
///       the narrow-band SDF ops (reinitialize_sdf / retopologize_sdf) this op consumes, which use
///       a scalar voxel size; for strongly anisotropic voxels the emitted normals are approximate.
///
/// @param batchHdl    Grid batch defining the sparse topology.
/// @param field       Per-voxel signed field: a floating-point JaggedTensor of shape [B, -1].
/// @param iso         Isovalue of the surface to extract.
/// @param reduce      Uniform F x F x F cluster-collapse decimation block size. reduce=1 is full
///                    resolution only when adaptivity==0; when adaptivity>0 it instead sets the
///                    coarse-block size for the adaptive collapse (defaulting to 8 when reduce<=1).
/// @param adaptivity  Curvature-adaptive decimation in [0, 1.5] (0 = off). When >0, flat blocks are
///                    collapsed while features keep full detail -- so the mesh is simplified even at
///                    reduce=1.
/// @return A (vertices, faces, normals) tuple, each jagged over the grid batch: vertices and
///         normals are float32 JaggedTensors of shape [B, -1, 3] and faces is an int64 JaggedTensor
///         of shape [B, -1, 3]. faces holds grid-local triangle vertex indices; normals is the
///         normalized SDF gradient at each vertex.
std::tuple<JaggedTensor, JaggedTensor, JaggedTensor> dualContour(const GridBatchData &batchHdl,
                                                                 const JaggedTensor &field,
                                                                 double iso,
                                                                 int reduce,
                                                                 double adaptivity);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_DUALCONTOUR_H
