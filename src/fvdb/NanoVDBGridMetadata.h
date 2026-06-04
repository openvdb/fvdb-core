// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_NANOVDBGRIDMETADATA_H
#define FVDB_NANOVDBGRIDMETADATA_H

#include <nanovdb/NanoVDB.h>

#include <cstdint>
#include <string>

namespace fvdb {

/// @brief Lightweight per-grid metadata read from a .nvdb file header without loading any voxel
///        data. Useful for enumerating grids (names / types) before deciding which ones to load.
struct NanoVDBGridMetadata {
    std::string name;      ///< Grid name as stored in the file
    std::string type;      ///< nanovdb::GridType as a string (e.g. "float", "Vec3f", "OnIndex")
    std::string gridClass; ///< nanovdb::GridClass as a string (e.g. "SDF", "FOG", "TENSOR", "?")
    int64_t voxelCount{0}; ///< Number of active voxels
    nanovdb::Vec3d voxelSize{1.0, 1.0, 1.0}; ///< Voxel size in world units
    nanovdb::Coord indexBBoxMin{0, 0, 0};    ///< Inclusive min of the index bbox
    nanovdb::Coord indexBBoxMax{0, 0, 0};    ///< Inclusive max of the index bbox
};

} // namespace fvdb

#endif // FVDB_NANOVDBGRIDMETADATA_H
