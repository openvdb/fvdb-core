// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_CAMERAVIEW_H
#define FVDB_DETAIL_VIEWER_CAMERAVIEW_H

#include <nanovdb_editor/putil/Editor.h>

#include <string>
#include <tuple>

namespace fvdb::detail::viewer {

// Forward declaration
class Viewer;

class CameraView {
    friend class Viewer;

    // View can only be created by Viewer via addCameraView
    CameraView(const CameraView &)            = delete;
    CameraView &operator=(const CameraView &) = delete;

    std::string mName;
    pnanovdb_editor_token_t *mSceneToken = nullptr;

  protected:
    pnanovdb_camera_view_t mView;

  public:
    explicit CameraView(const std::string &name, pnanovdb_editor_token_t *nameToken) : mName(name) {
        pnanovdb_camera_view_default(&mView);
        mView.name = nameToken;
    }

    ~CameraView() {
        delete[] mView.states;
        delete[] mView.configs;
    }

    const std::string &
    getName() const {
        return mName;
    }

    bool
    getVisible() const {
        return mView.is_visible == PNANOVDB_TRUE;
    }
    void
    setVisible(const bool visible) {
        mView.is_visible = visible ? PNANOVDB_TRUE : PNANOVDB_FALSE;
    }

    float
    getAxisLength() const {
        return mView.axis_length;
    }
    void
    setAxisLength(const float length) {
        mView.axis_length = length;
    }

    float
    getAxisThickness() const {
        return mView.axis_thickness;
    }
    void
    setAxisThickness(const float thickness) {
        mView.axis_thickness = thickness;
    }

    float
    getFrustumLineWidth() const {
        return mView.frustum_line_width;
    }
    void
    setFrustumLineWidth(const float width) {
        mView.frustum_line_width = width;
    }

    float
    getFrustumScale() const {
        return mView.frustum_scale;
    }
    void
    setFrustumScale(const float scale) {
        mView.frustum_scale = scale;
    }

    std::tuple<float, float, float>
    getFrustumColor() const {
        return std::make_tuple(mView.frustum_color.x, mView.frustum_color.y, mView.frustum_color.z);
    }
    void
    setFrustumColor(const float r, const float g, const float b) {
        mView.frustum_color.x = r;
        mView.frustum_color.y = g;
        mView.frustum_color.z = b;
    }
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_CAMERAVIEW_H
