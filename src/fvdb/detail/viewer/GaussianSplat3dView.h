// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_GAUSSIANSPLAT3DVIEW_H
#define FVDB_DETAIL_VIEWER_GAUSSIANSPLAT3DVIEW_H

#include <fvdb/GaussianSplat3d.h>

#include <nanovdb_editor/putil/Raster.h>

#include <string>

namespace fvdb::detail::viewer {

// Forward declaration
class Viewer;

class GaussianSplat3dView {
    friend class Viewer;

    // View can only be created by Viewer via addGaussianSplat3dView
    GaussianSplat3dView(const GaussianSplat3dView &)            = delete;
    GaussianSplat3dView &operator=(const GaussianSplat3dView &) = delete;

    std::string mViewName;

  protected:
    pnanovdb_raster_shader_params_t mParams;
    std::function<void(bool)> mSyncCallback;

  public:
    GaussianSplat3dView(const std::string &name, const Viewer &viewer);

    const float
    getNear() const {
        float nearPlane = mParams.near_plane_override;
        mSyncCallback(false);
        return nearPlane;
    }
    void
    setNear(const float near) {
        mParams.near_plane_override = near;
        mSyncCallback(true);
    }

    const float
    getFar() const {
        float farPlane = mParams.far_plane_override;
        mSyncCallback(false);
        return farPlane;
    }
    void
    setFar(const float far) {
        mParams.far_plane_override = far;
        mSyncCallback(true);
    }

    const size_t
    getTileSize() const {
        mSyncCallback(false);
        return mParams.tile_size;
    }
    void
    setTileSize(const size_t tileSize) {
        mParams.tile_size = tileSize;
        mSyncCallback(true);
    }

    const float
    getMinRadius2d() const {
        mSyncCallback(false);
        return mParams.min_radius_2d;
    }
    void
    setMinRadius2d(const float minRadius2d) {
        mParams.min_radius_2d = minRadius2d;
        mSyncCallback(true);
    }

    const float
    getEps2d() const {
        mSyncCallback(false);
        return mParams.eps2d;
    }
    void
    setEps2d(const float eps2d) {
        mParams.eps2d = eps2d;
        mSyncCallback(true);
    }

    const bool
    getAntialias() const {
        return false;
    }
    void
    setAntialias(const bool antialias) {}

    int
    getShDegreeToUse() const {
        mSyncCallback(false);
        return mParams.sh_degree_override;
    }
    void
    setShDegreeToUse(const int shDegree) {
        mParams.sh_degree_override = shDegree;
        mSyncCallback(true);
    }
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_GAUSSIANSPLAT3DVIEW_H
