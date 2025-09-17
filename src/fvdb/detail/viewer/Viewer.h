// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_VIEWER_H
#define FVDB_DETAIL_VIEWER_VIEWER_H

#include <fvdb/GaussianSplat3d.h>
#include <fvdb/GridBatch.h>
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
        pnanovdb_raster_t raster;
        pnanovdb_camera_t camera;
        pnanovdb_editor_t editor;
        pnanovdb_editor_config_t config;
        const pnanovdb_reflect_data_type_t *rasterShaderParamsType;
    };

    EditorContext mEditor;
    bool mIsEditorRunning;
    std::string mIpAddress;
    int mPort;

    std::map<std::string, GaussianSplat3dView> mSplat3dViews;

    void updateCamera();

  public:
    Viewer(const std::string &ipAddress, const int port, const bool verbose = false);
    ~Viewer();

    GaussianSplat3dView &registerGaussianSplat3dView(const std::string &name,
                                                     const GaussianSplat3d &splats);

    void startServer();
    void stopServer();

    // Camera control methods
    void setCameraPosition(float x, float y, float z);
    std::tuple<float, float, float> getCameraPosition();

    void setCameraLookat(float x, float y, float z);
    std::tuple<float, float, float> getCameraLookat();

    void setCameraNear(float near);
    float getCameraNear();

    void setCameraFar(float far);
    float getCameraFar();

    void setCameraPose(torch::Tensor cameraToWorldMatrix);

    void setCameraEyeDirection(float x, float y, float z);
    std::tuple<float, float, float> getCameraEyeDirection();

    void setCameraEyeUp(float x, float y, float z);
    std::tuple<float, float, float> getCameraEyeUp();

    void setCameraEyeDistanceFromPosition(float distance);
    float getCameraEyeDistanceFromPosition();

    void setCameraMode(GaussianSplat3d::ProjectionType mode);
    GaussianSplat3d::ProjectionType getCameraMode();
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_VIEWER_H
