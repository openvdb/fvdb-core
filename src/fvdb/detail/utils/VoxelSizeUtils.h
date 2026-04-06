// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_VOXELSIZEUTILS_H
#define FVDB_DETAIL_UTILS_VOXELSIZEUTILS_H

#include <nanovdb/NanoVDB.h>

namespace fvdb {
namespace detail {

inline nanovdb::Vec3d
coarseVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &coarseningFactor) {
    return coarseningFactor.asVec3d() * voxelSize;
}

inline nanovdb::Vec3d
coarseVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                  const nanovdb::Vec3d &voxelOrigin,
                  const nanovdb::Coord &coarseningFactor) {
    return (coarseningFactor.asVec3d() - nanovdb::Vec3d(1.0)) * voxelSize * 0.5 + voxelOrigin;
}

inline nanovdb::Vec3d
fineVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &subdivFactor) {
    return voxelSize / subdivFactor.asVec3d();
}

inline nanovdb::Vec3d
fineVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                const nanovdb::Vec3d &voxelOrigin,
                const nanovdb::Coord &subdivFactor) {
    return voxelOrigin - (subdivFactor.asVec3d() - nanovdb::Vec3d(1.0)) *
                             (voxelSize / subdivFactor.asVec3d()) * 0.5;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_VOXELSIZEUTILS_H
