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
    std::map<std::string, CameraView> mCameraViews;

    void updateCamera();

    void startServer();
    void stopServer();

  public:
    Viewer(const std::string &ipAddress, const int port, const bool verbose = false);
    ~Viewer();

    GaussianSplat3dView &addGaussianSplat3d(const std::string &name, const GaussianSplat3d &splats);
    CameraView &addCameraView(const std::string &name,
                              const torch::Tensor &cameraToWorldMatrices,
                              const torch::Tensor &projectionMatrices);

    std::tuple<float, float, float> cameraOrbitCenter() const;
    void setCameraOrbitCenter(float ox, float oy, float oz);

    std::tuple<float, float, float> cameraUpDirection() const;
    void setCameraUpDirection(float ux, float uy, float uz);

    std::tuple<float, float, float> cameraViewDirection() const;
    void setCameraViewDirection(float dx, float dy, float dz);

    float cameraOrbitRadius() const;
    void setCameraOrbitRadius(float radius);

    float cameraNear() const;
    void setCameraNear(float near);

    float cameraFar() const;
    void setCameraFar(float far);

    void setCameraProjectionType(GaussianSplat3d::ProjectionType mode);
    GaussianSplat3d::ProjectionType cameraProjectionType() const;
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_VIEWER_H
