// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_VIEWER_H
#define FVDB_DETAIL_VIEWER_VIEWER_H

#include <fvdb/GaussianSplat3d.h>
#include <fvdb/GridBatch.h>
#include <fvdb/detail/viewer/CameraView.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>

#include <torch/torch.h>

#include <nanovdb_editor/putil/Editor.h>

#include <map>
#include <string>

namespace fvdb::detail::viewer {

class Viewer {
    struct EditorContext {
        pnanovdb_compiler_t compiler;
        pnanovdb_compute_t compute;
        pnanovdb_compute_device_desc_t deviceDesc;
        pnanovdb_compute_device_manager_t *deviceManager;
        pnanovdb_compute_device_t *device;
        pnanovdb_editor_t editor;
        pnanovdb_editor_config_t config;
        pnanovdb_camera_t camera;
    };

    EditorContext mEditor;
    bool mIsEditorRunning;
    std::string mIpAddress;
    int mPort;
    std::string mCurrentSceneName;

    // Views are currently shared by all scenes and need to have unique names
    std::map<std::string, GaussianSplat3dView> mSplat3dViews;
    std::map<std::string, CameraView> mCameraViews;

    void updateCamera(const std::string &scene_name);
    void getCamera(const std::string &scene_name);

    void startServer();
    void stopServer();

  public:
    Viewer(const std::string &ipAddress,
           const int port,
           const int device_id,
           const bool verbose = false);
    ~Viewer();

    void reset();

    void setSceneName(const std::string &scene_name);

    void addScene(const std::string &scene_name);

    void removeScene(const std::string &scene_name);

    void removeView(const std::string &scene_name, const std::string &view_name);

    pnanovdb_editor_token_t *
    getToken(const std::string &name) const {
        return mEditor.editor.get_token(name.c_str());
    }

    GaussianSplat3dView &addGaussianSplat3dView(const std::string &scene_name,
                                                const std::string &name,
                                                const GaussianSplat3d &splats);
    CameraView &addCameraView(const std::string &scene_name,
                              const std::string &name,
                              const torch::Tensor &cameraToWorldMatrices,
                              const torch::Tensor &projectionMatrices,
                              const torch::Tensor &imageSizes,
                              float frustumNear,
                              float frustumFar,
                              float axisLength,
                              float axisThickness,
                              float frustumLineWidth,
                              float frustumScale,
                              const std::tuple<float, float, float> &frustumColor,
                              bool visible);

    bool
    hasGaussianSplat3dView(const std::string &name) const {
        return mSplat3dViews.find(name) != mSplat3dViews.end();
    }
    bool
    hasCameraView(const std::string &name) const {
        return mCameraViews.find(name) != mCameraViews.end();
    }

    GaussianSplat3dView &
    getGaussianSplat3dView(const std::string &name) {
        const auto it      = mSplat3dViews.find(name);
        const bool hasView = it != mSplat3dViews.end();
        TORCH_CHECK(hasView, "No GaussianSplat3dView with name '", name, "' found");

        return it->second;
    }
    CameraView &
    getCameraView(const std::string &name) {
        const auto it      = mCameraViews.find(name);
        const bool hasView = it != mCameraViews.end();
        TORCH_CHECK(hasView, "No CameraView with name '", name, "' found");

        return it->second;
    }

    std::tuple<float, float, float> cameraOrbitCenter(const std::string &scene_name);
    void setCameraOrbitCenter(const std::string &scene_name, float ox, float oy, float oz);

    std::tuple<float, float, float> cameraUpDirection(const std::string &scene_name);
    void setCameraUpDirection(const std::string &scene_name, float ux, float uy, float uz);

    std::tuple<float, float, float> cameraViewDirection(const std::string &scene_name);
    void setCameraViewDirection(const std::string &scene_name, float dx, float dy, float dz);

    float cameraOrbitRadius(const std::string &scene_name);
    void setCameraOrbitRadius(const std::string &scene_name, float radius);

    float cameraNear(const std::string &scene_name);
    void setCameraNear(const std::string &scene_name, float near);

    float cameraFar(const std::string &scene_name);
    void setCameraFar(const std::string &scene_name, float far);

    void setCameraProjectionType(const std::string &scene_name,
                                 GaussianSplat3d::ProjectionType mode);
    GaussianSplat3d::ProjectionType cameraProjectionType(const std::string &scene_name);

    std::string
    ipAddress() const {
        return mIpAddress;
    };
    int
    port() const {
        return mPort;
    };
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_VIEWER_H
