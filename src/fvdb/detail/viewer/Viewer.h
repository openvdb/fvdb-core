// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_VIEWER_H
#define FVDB_DETAIL_VIEWER_VIEWER_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/viewer/CameraView.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>

#include <c10/util/Exception.h>
#include <torch/types.h>

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

    struct NanoVDBView {
        std::string name;
    };

    // Views are currently shared by all scenes and need to have unique names
    std::map<std::string, GaussianSplat3dView> mSplat3dViews;
    std::map<std::string, CameraView> mCameraViews;
    std::map<std::string, NanoVDBView> mNanoVDBViews;

    void updateCamera(const std::string &scene_name);
    void getCamera(const std::string &scene_name);

    void startServer();
    void stopServer();

    // Shared implementation behind addLevelSetView / addFogVolumeView: builds an ONINDEX NanoVDB
    // buffer carrying `floatValues` as blind data and registers it under the given nanovdb-editor
    // render pipeline and shader. `render_shader` must be the shader that `render_pipeline` uses
    // (e.g. "editor/editor_surface.slang" for the surface pipeline) since add_nanovdb_3 otherwise
    // leaves the object on the editor's default scalar shader. `grid` must contain exactly one
    // grid (batchSize() == 1).
    void addNanoVDBGridView(const std::string &scene_name,
                            const std::string &name,
                            const GridBatchData &grid,
                            const JaggedTensor &floatValues,
                            pnanovdb_pipeline_type_t render_pipeline,
                            const std::string &render_shader);

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
                                                const torch::Tensor &means,
                                                const torch::Tensor &quats,
                                                const torch::Tensor &logScales,
                                                const torch::Tensor &logitOpacities,
                                                const torch::Tensor &sh0,
                                                const torch::Tensor &shN);
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

    void addImage(const std::string &scene_name,
                  const std::string &name,
                  const torch::Tensor &rgba_image,
                  int64_t width,
                  int64_t height);

    // Add a single grid (batchSize() == 1) with per-voxel float32 SDF values as a level-set
    // isosurface, rendered by the nanovdb-editor "surface" pipeline (HDDA zero-crossing).
    void addLevelSetView(const std::string &scene_name,
                         const std::string &name,
                         const GridBatchData &grid,
                         const JaggedTensor &sdf);

    // Add a single grid (batchSize() == 1) with per-voxel float32 density values as a fog volume,
    // rendered by the nanovdb-editor "render" pipeline (volumetric ray-marcher).
    void addFogVolumeView(const std::string &scene_name,
                          const std::string &name,
                          const GridBatchData &grid,
                          const JaggedTensor &density);

    bool
    hasNanoVDBView(const std::string &name) const {
        return mNanoVDBViews.find(name) != mNanoVDBViews.end();
    }

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

    float cameraFov(const std::string &scene_name);
    void setCameraFov(const std::string &scene_name, float fov_radians);

    float cameraNear(const std::string &scene_name);
    void setCameraNear(const std::string &scene_name, float near);

    float cameraFar(const std::string &scene_name);
    void setCameraFar(const std::string &scene_name, float far);

    void setCameraModel(const std::string &scene_name, fvdb::detail::ops::DistortionModel model);
    fvdb::detail::ops::DistortionModel cameraModel(const std::string &scene_name);

    std::string
    ipAddress() const {
        return mIpAddress;
    };
    int
    port() const {
        return mPort;
    };

    void waitForInteerrupt();
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_VIEWER_H
