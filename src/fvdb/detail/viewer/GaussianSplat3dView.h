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

    std::string mName;
    pnanovdb_editor_token_t *mSceneToken;

    void
    syncSet() {
        mSyncCallback(true);
    }

    void
    syncGet() const {
        mSyncCallback(false);
    }

  protected:
    pnanovdb_raster_shader_params_t mParams;
    std::function<void(bool)> mSyncCallback;

  public:
    GaussianSplat3dView(const std::string &name, const Viewer &viewer);

    const size_t
    getTileSize() const {
        syncGet();
        return mParams.tile_size;
    }
    void
    setTileSize(const size_t tileSize) {
        mParams.tile_size = tileSize;
        syncSet();
    }

    const float
    getMinRadius2d() const {
        syncGet();
        return mParams.min_radius_2d;
    }
    void
    setMinRadius2d(const float minRadius2d) {
        mParams.min_radius_2d = minRadius2d;
        syncSet();
    }

    const float
    getEps2d() const {
        syncGet();
        return mParams.eps2d;
    }
    void
    setEps2d(const float eps2d) {
        mParams.eps2d = eps2d;
        syncSet();
    }

    const bool
    getAntialias() const {
        return false;
    }
    void
    setAntialias(const bool antialias) {}

    int
    getShDegreeToUse() const {
        syncGet();
        return mParams.sh_degree_override;
    }
    void
    setShDegreeToUse(const int shDegree) {
        mParams.sh_degree_override = shDegree;
        syncSet();
    }

    const bool
    isShStrideRgbRgbRgb() const {
        syncGet();
        return (mParams.sh_stride_rgbrgbrgb_override != 0u);
    }
    void
    setShStrideRgbRgbRgb(bool value) {
        mParams.sh_stride_rgbrgbrgb_override = value ? 1u : 0u;
        syncSet();
    }
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_GAUSSIANSPLAT3DVIEW_H
