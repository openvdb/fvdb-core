// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/io/SaveNanoVDB.h>
#include <fvdb/detail/viewer/CameraView.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>
#include <fvdb/detail/viewer/ParamViews.h>
#include <fvdb/detail/viewer/Viewer.h>

#include <c10/util/Exception.h>

#include <nanovdb_editor/putil/Raster.h>
#include <nanovdb_editor/putil/Reflect.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

inline void
pNanoLogPrint(pnanovdb_compute_log_level_t level, const char *format, ...) {
    va_list args;
    va_start(args, format);

    const char *prefix = "Unknown";
    if (level == PNANOVDB_COMPUTE_LOG_LEVEL_ERROR) {
        prefix = "Error";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_WARNING) {
        prefix = "Warning";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_INFO) {
        prefix = "Info";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_DEBUG) {
        va_end(args);
        return;
    }
    printf("Viewer %s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

inline void
pNanoLogPrintVerbose(pnanovdb_compute_log_level_t level, const char *format, ...) {
    va_list args;
    va_start(args, format);

    const char *prefix = "Unknown";
    if (level == PNANOVDB_COMPUTE_LOG_LEVEL_ERROR) {
        prefix = "Error";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_WARNING) {
        prefix = "Warning";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_INFO) {
        prefix = "Info";
    } else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_DEBUG) {
        prefix = "Debug";
    }
    printf("Viewer %s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

namespace fvdb::detail::viewer {

constexpr float DEFAULT_CAMERA_FOV_RADIANS  = 60.f * M_PI / 180.f;
constexpr float DEFAULT_CAMERA_ASPECT_RATIO = 4.f / 3.f;

inline pnanovdb_pipeline_type_t
getPipelineType(pnanovdb_editor_t &editor, const char *typeId) {
    pnanovdb_pipeline_type_t type      = 0;
    pnanovdb_editor_token_t *typeToken = editor.get_token(typeId);
    TORCH_CHECK(
        typeToken, "nanovdb-editor could not create token for pipeline type '", typeId, "'");
    TORCH_CHECK(editor.get_pipeline_type(&editor, typeToken, &type),
                "nanovdb-editor could not resolve pipeline type '",
                typeId,
                "'");
    return type;
}

void
Viewer::updateCamera(const std::string &scene_name) {
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    mEditor.editor.update_camera_2(&mEditor.editor, sceneToken, &mEditor.camera);
}

void
Viewer::getCamera(const std::string &scene_name) {
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    pnanovdb_camera_t *camera           = mEditor.editor.get_camera(&mEditor.editor, sceneToken);
    if (camera) {
        // copy POD data
        mEditor.camera = *camera;
    }
}

Viewer::Viewer(const std::string &ipAddress,
               const int port,
               const int device_id,
               const bool verbose)
    : mIpAddress(ipAddress), mPort(port) {
    mEditor.compiler = {};
    pnanovdb_compiler_load(&mEditor.compiler);

    mEditor.compute = {};
    pnanovdb_compute_load(&mEditor.compute, &mEditor.compiler);

    mEditor.deviceDesc              = {};
    mEditor.deviceDesc.device_index = device_id;
    mEditor.deviceDesc.log_print    = verbose ? pNanoLogPrintVerbose : pNanoLogPrint;

    mEditor.deviceManager = mEditor.compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    mEditor.device =
        mEditor.compute.device_interface.create_device(mEditor.deviceManager, &mEditor.deviceDesc);

    mEditor.editor = {};
    pnanovdb_editor_load(&mEditor.editor, &mEditor.compute, &mEditor.compiler);

    mEditor.config                 = {};
    mEditor.config.ip_address      = mIpAddress.c_str();
    mEditor.config.port            = port;
    mEditor.config.headless        = PNANOVDB_TRUE;
    mEditor.config.streaming       = PNANOVDB_TRUE;
    mEditor.config.ui_profile_name = "viewer";

    mIsEditorRunning = false;

    startServer();
}

Viewer::~Viewer() {
    stopServer();

    mSplat3dViews.clear();
    mCameraViews.clear();

    mEditor.compute.device_interface.destroy_device(mEditor.deviceManager, mEditor.device);
    mEditor.compute.device_interface.destroy_device_manager(mEditor.deviceManager);

    pnanovdb_editor_free(&mEditor.editor);
    pnanovdb_compute_free(&mEditor.compute);
    pnanovdb_compiler_free(&mEditor.compiler);
}

void
Viewer::reset() {
    mEditor.editor.reset(&mEditor.editor);

    mCameraViews.clear();
    mSplat3dViews.clear();
    mSliderViews.clear();
    mNumberViews.clear();
    mTextViews.clear();
    mCheckboxViews.clear();
    mSceneCustomParams.clear();
}

void
Viewer::addScene(const std::string &scene_name) {
    pnanovdb_camera_init(&mEditor.camera);
    updateCamera(scene_name);
}

void
Viewer::removeScene(const std::string &scene_name) {
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    mEditor.editor.remove(&mEditor.editor, sceneToken, nullptr);

    // Erase all camera views belonging to the removed scene
    for (auto it = mCameraViews.begin(); it != mCameraViews.end();) {
        if (it->second.mSceneToken == sceneToken) {
            it = mCameraViews.erase(it);
        } else {
            ++it;
        }
    }
    // Erase all splat 3d views belonging to the removed scene
    for (auto it = mSplat3dViews.begin(); it != mSplat3dViews.end();) {
        if (it->second.mSceneToken == sceneToken) {
            it = mSplat3dViews.erase(it);
        } else {
            ++it;
        }
    }

    // Erase all custom-scene-params widget views and specs belonging to the
    // removed scene. Widget views identify their owning scene by name (not
    // by token) since their state lives in mSceneCustomParams[scene_name].
    auto eraseByScene = [&scene_name](auto &views) {
        for (auto it = views.begin(); it != views.end();) {
            if (it->second.getSceneName() == scene_name) {
                it = views.erase(it);
            } else {
                ++it;
            }
        }
    };
    eraseByScene(mSliderViews);
    eraseByScene(mNumberViews);
    eraseByScene(mTextViews);
    eraseByScene(mCheckboxViews);
    mSceneCustomParams.erase(scene_name);
}

void
Viewer::removeView(const std::string &scene_name, const std::string &name) {
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    pnanovdb_editor_token_t *viewToken  = mEditor.editor.get_token(name.c_str());
    mEditor.editor.remove(&mEditor.editor, sceneToken, viewToken);

    mCameraViews.erase(name);
    mSplat3dViews.erase(name);

    // Removing a widget view also removes its spec from the schema and
    // resubmits the JSON so the editor's `Scene Params` window updates.
    bool removedWidget = false;
    if (mSliderViews.erase(name) > 0) {
        removedWidget = true;
    }
    if (mNumberViews.erase(name) > 0) {
        removedWidget = true;
    }
    if (mTextViews.erase(name) > 0) {
        removedWidget = true;
    }
    if (mCheckboxViews.erase(name) > 0) {
        removedWidget = true;
    }
    if (removedWidget) {
        auto it = mSceneCustomParams.find(scene_name);
        if (it != mSceneCustomParams.end()) {
            auto &specs               = it->second;
            const std::string counter = name + std::string(kSubmitCounterSuffix);
            specs.erase(std::remove_if(specs.begin(),
                                       specs.end(),
                                       [&name, &counter](const CustomParamSpec &s) {
                                           return s.name == name || s.name == counter;
                                       }),
                        specs.end());
            if (specs.empty()) {
                mSceneCustomParams.erase(it);
            } else {
                submitSchemaForScene(scene_name);
            }
        }
    }
}

fvdb::detail::viewer::GaussianSplat3dView &
Viewer::addGaussianSplat3dView(const std::string &scene_name,
                               const std::string &name,
                               const torch::Tensor &means,
                               const torch::Tensor &quats,
                               const torch::Tensor &logScales,
                               const torch::Tensor &logitOpacities,
                               const torch::Tensor &sh0,
                               const torch::Tensor &shN) {
    std::shared_ptr<pnanovdb_raster_gaussian_data_t> oldData;
    auto itPrev = mSplat3dViews.find(name);
    if (itPrev != mSplat3dViews.end()) {
        mSplat3dViews.erase(itPrev);
    }

    auto [it, inserted] = mSplat3dViews.emplace(
        std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(name, *this));

    auto makeComputeArray = [this](const torch::Tensor &tensor) -> pnanovdb_compute_array_t * {
        torch::Tensor contig = tensor.cpu().contiguous();
        size_t total_size    = 1;
        for (int i = 0; i < contig.dim(); ++i) {
            total_size *= contig.size(i);
        }
        return mEditor.compute.create_array(contig.element_size(), total_size, contig.data_ptr());
    };

    // Copy into pnano format and pass to the viewer
    pnanovdb_compute_array_t *means_arr          = makeComputeArray(means);
    pnanovdb_compute_array_t *quats_arr          = makeComputeArray(quats);
    pnanovdb_compute_array_t *logScales_arr      = makeComputeArray(logScales);
    pnanovdb_compute_array_t *logitOpacities_arr = makeComputeArray(logitOpacities);
    pnanovdb_compute_array_t *sh0_arr            = makeComputeArray(sh0);
    pnanovdb_compute_array_t *shN_arr            = makeComputeArray(shN);

    pnanovdb_compute_array_t *arrays[] = {
        means_arr, logitOpacities_arr, quats_arr, logScales_arr, sh0_arr, shN_arr};

    // Load splats into viewer
    pnanovdb_editor_gaussian_data_desc_t desc = {};
    desc.means                                = means_arr;
    desc.opacities                            = logitOpacities_arr;
    desc.quaternions                          = quats_arr;
    desc.scales                               = logScales_arr;
    desc.sh_0                                 = sh0_arr;
    desc.sh_n                                 = shN_arr;

    // Get token for this object name
    pnanovdb_editor_token_t *nameToken  = mEditor.editor.get_token(name.c_str());
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());

    // Add to editor using token-based API
    mEditor.editor.add_gaussian_data_2(&mEditor.editor, sceneToken, nameToken, &desc);

    it->second.mSceneToken = sceneToken;

    // Set up parameter synchronization using map/unmap against named object
    it->second.mSyncCallback = [this, sceneToken, nameToken, viewPtr = &it->second](bool set_data) {
        void *paramsPtr = mEditor.editor.map_params(
            &mEditor.editor, sceneToken, nameToken, viewPtr->mParams.data_type);
        if (!paramsPtr) {
            return;
        }
        if (set_data) {
            std::memcpy(paramsPtr, &viewPtr->mParams, viewPtr->mParams.data_type->element_size);
        } else {
            std::memcpy(&viewPtr->mParams, paramsPtr, viewPtr->mParams.data_type->element_size);
        }
        mEditor.editor.unmap_params(&mEditor.editor, sceneToken, nameToken);
    };

    for (pnanovdb_compute_array_t *arr: arrays) {
        mEditor.compute.destroy_array(arr);
    }

    return it->second;
}

void
Viewer::startServer() {
    if (!mIsEditorRunning) {
        mEditor.editor.start(&mEditor.editor, mEditor.device, &mEditor.config);
        mIsEditorRunning = true;
    }
}

void
Viewer::stopServer() {
    if (mIsEditorRunning) {
        mEditor.editor.stop(&mEditor.editor);
        mIsEditorRunning = false;
    }
}

void
Viewer::stop() {
    stopServer();
}

void
Viewer::waitForInteerrupt() {
    mEditor.editor.wait_for_interrupt(&mEditor.editor);
}

std::tuple<float, float, float>
Viewer::cameraOrbitCenter(const std::string &scene_name) {
    getCamera(scene_name);
    return std::make_tuple(mEditor.camera.state.position.x,
                           mEditor.camera.state.position.y,
                           mEditor.camera.state.position.z);
}
void
Viewer::setCameraOrbitCenter(const std::string &scene_name, float x, float y, float z) {
    getCamera(scene_name);
    mEditor.camera.state.position.x = x;
    mEditor.camera.state.position.y = y;
    mEditor.camera.state.position.z = z;
    updateCamera(scene_name);
}

float
Viewer::cameraOrbitRadius(const std::string &scene_name) {
    getCamera(scene_name);
    return mEditor.camera.state.eye_distance_from_position;
}
void
Viewer::setCameraOrbitRadius(const std::string &scene_name, float radius) {
    getCamera(scene_name);
    mEditor.camera.state.eye_distance_from_position = radius;
    updateCamera(scene_name);
}

std::tuple<float, float, float>
Viewer::cameraViewDirection(const std::string &scene_name) {
    getCamera(scene_name);
    return std::make_tuple(mEditor.camera.state.eye_direction.x,
                           mEditor.camera.state.eye_direction.y,
                           mEditor.camera.state.eye_direction.z);
}
void
Viewer::setCameraViewDirection(const std::string &scene_name, float x, float y, float z) {
    getCamera(scene_name);
    mEditor.camera.state.eye_direction.x = x;
    mEditor.camera.state.eye_direction.y = y;
    mEditor.camera.state.eye_direction.z = z;
    updateCamera(scene_name);
}

std::tuple<float, float, float>
Viewer::cameraUpDirection(const std::string &scene_name) {
    getCamera(scene_name);
    return std::make_tuple(mEditor.camera.state.eye_up.x,
                           mEditor.camera.state.eye_up.y,
                           mEditor.camera.state.eye_up.z);
}
void
Viewer::setCameraUpDirection(const std::string &scene_name, float x, float y, float z) {
    getCamera(scene_name);
    mEditor.camera.state.eye_up.x = x;
    mEditor.camera.state.eye_up.y = y;
    mEditor.camera.state.eye_up.z = z;
    updateCamera(scene_name);
}

float
Viewer::cameraFov(const std::string &scene_name) {
    getCamera(scene_name);
    return mEditor.camera.config.fov_angle_y;
}
void
Viewer::setCameraFov(const std::string &scene_name, float fov_radians) {
    getCamera(scene_name);
    mEditor.camera.config.fov_angle_y = fov_radians;
    updateCamera(scene_name);
}

float
Viewer::cameraNear(const std::string &scene_name) {
    getCamera(scene_name);
    return mEditor.camera.config.near_plane;
}
void
Viewer::setCameraNear(const std::string &scene_name, float near) {
    getCamera(scene_name);
    mEditor.camera.config.near_plane = near;
    updateCamera(scene_name);
}

float
Viewer::cameraFar(const std::string &scene_name) {
    getCamera(scene_name);
    return mEditor.camera.config.far_plane;
}
void
Viewer::setCameraFar(const std::string &scene_name, float far) {
    getCamera(scene_name);
    mEditor.camera.config.far_plane = far;
    updateCamera(scene_name);
}

fvdb::detail::ops::DistortionModel
Viewer::cameraModel(const std::string &scene_name) {
    getCamera(scene_name);
    return mEditor.camera.config.is_orthographic ? fvdb::detail::ops::DistortionModel::ORTHOGRAPHIC
                                                 : fvdb::detail::ops::DistortionModel::PINHOLE;
}

void
Viewer::setCameraModel(const std::string &scene_name, fvdb::detail::ops::DistortionModel model) {
    getCamera(scene_name);
    if (model == fvdb::detail::ops::DistortionModel::PINHOLE) {
        mEditor.camera.config.is_orthographic = PNANOVDB_FALSE;
    } else if (model == fvdb::detail::ops::DistortionModel::ORTHOGRAPHIC) {
        mEditor.camera.config.is_orthographic = PNANOVDB_TRUE;
    } else {
        throw std::invalid_argument(
            "Viewer currently only supports CameraModel::PINHOLE and ORTHOGRAPHIC");
    }

    updateCamera(scene_name);
}

CameraView &
Viewer::addCameraView(const std::string &scene_name,
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
                      bool visible) {
    TORCH_CHECK(cameraToWorldMatrices.dim() == 3 && cameraToWorldMatrices.size(1) == 4 &&
                    cameraToWorldMatrices.size(2) == 4,
                "camera_to_world_matrices must have shape [N, 4, 4]");
    TORCH_CHECK(projectionMatrices.dim() == 3 && projectionMatrices.size(1) == 3 &&
                    projectionMatrices.size(2) == 3,
                "projection_matrices must have shape [N, 3, 3]");

    auto itPrev = mCameraViews.find(name);
    if (itPrev != mCameraViews.end()) {
        mCameraViews.erase(itPrev);
    }

    const int64_t numCameras = cameraToWorldMatrices.size(0);
    if (imageSizes.numel() != 0) {
        TORCH_CHECK(imageSizes.dim() == 2 && imageSizes.size(0) == numCameras &&
                        imageSizes.size(1) == 2,
                    "image_sizes must have shape [N, 2] if provided. Got ",
                    imageSizes.sizes(),
                    " instead.");
    }

    pnanovdb_editor_token_t *nameToken  = mEditor.editor.get_token(name.c_str());
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());

    auto [it, inserted] = mCameraViews.emplace(std::piecewise_construct,
                                               std::forward_as_tuple(name),
                                               std::forward_as_tuple(name, nameToken));

    it->second.mSceneToken       = sceneToken;
    it->second.mView.num_cameras = numCameras;
    it->second.mView.states      = new pnanovdb_camera_state_t[it->second.mView.num_cameras];
    it->second.mView.configs     = new pnanovdb_camera_config_t[it->second.mView.num_cameras];

    for (int i = 0; i < (int)it->second.mView.num_cameras; i++) {
        torch::Tensor c2w = cameraToWorldMatrices.index({i}).contiguous().cpu();
        torch::Tensor K   = projectionMatrices.index({i}).contiguous().cpu();
        auto c2w_acc      = c2w.accessor<float, 2>();
        auto K_acc        = K.accessor<float, 2>();

        float px = c2w_acc[0][3];
        float py = c2w_acc[1][3];
        float pz = c2w_acc[2][3];

        float zx = c2w_acc[0][2];
        float zy = c2w_acc[1][2];
        float zz = c2w_acc[2][2];

        float ux = c2w_acc[0][1];
        float uy = c2w_acc[1][1];
        float uz = c2w_acc[2][1];

        // NanoVDB editor camera
        // - state.position: orbit center (c2w translation)
        // - state.eye_direction: vector from center to camera (-forward)
        // - state.eye_up: camera up
        // - state.eye_distance_from_position: orbit radius (not needed for frustum rendering)

        pnanovdb_camera_state_default(&it->second.mView.states[i], PNANOVDB_FALSE);

        it->second.mView.states[i].position                   = {px, py, pz};
        it->second.mView.states[i].eye_direction              = {zx, zy, zz};
        it->second.mView.states[i].eye_up                     = {ux, uy, uz};
        it->second.mView.states[i].eye_distance_from_position = 1.f;

        pnanovdb_camera_config_default(&it->second.mView.configs[i]);
        it->second.mView.configs[i].is_orthographic = PNANOVDB_FALSE;
        it->second.mView.configs[i].is_reverse_z    = PNANOVDB_TRUE;

        // Used for frustum visualization
        it->second.mView.configs[i].near_plane = frustumNear;
        it->second.mView.configs[i].far_plane  = frustumFar;

        // Set perspective parameters from image sizes when available
        float fy      = K_acc[1][1];
        float width   = 0.f;
        float height  = 0.f;
        bool haveDims = imageSizes.numel() != 0;
        if (haveDims) {
            torch::Tensor dims = imageSizes.index({i}).contiguous().cpu();
            auto dims_acc      = dims.accessor<float, 1>();
            height             = dims_acc[0];
            width              = dims_acc[1];
        }

        if (haveDims && height > 0.f && fy > 0.f) {
            it->second.mView.configs[i].fov_angle_y  = 2.f * std::atan(0.5f * height / fy);
            it->second.mView.configs[i].aspect_ratio = width / height;
        } else {
            it->second.mView.configs[i].fov_angle_y  = DEFAULT_CAMERA_FOV_RADIANS;
            it->second.mView.configs[i].aspect_ratio = DEFAULT_CAMERA_ASPECT_RATIO;
        }
    }

    // Set visualization parameters
    it->second.setAxisLength(axisLength);
    it->second.setAxisThickness(axisThickness);
    it->second.setFrustumLineWidth(frustumLineWidth);
    it->second.setFrustumScale(frustumScale);
    it->second.setFrustumColor(
        std::get<0>(frustumColor), std::get<1>(frustumColor), std::get<2>(frustumColor));
    it->second.setVisible(visible);

    mEditor.editor.add_camera_view_2(&mEditor.editor, sceneToken, &it->second.mView);

    return it->second;
}

void
Viewer::addImage(const std::string &scene_name,
                 const std::string &name,
                 const torch::Tensor &rgba_image,
                 int64_t width,
                 int64_t height) {
    TORCH_CHECK(rgba_image.dim() == 1, "rgba_image must be a 1D tensor of packed RGBA8 values");
    TORCH_CHECK(rgba_image.scalar_type() == torch::kUInt8 ||
                    rgba_image.scalar_type() == torch::kByte,
                "rgba_image must have dtype uint8");
    TORCH_CHECK(rgba_image.numel() == width * height * 4,
                "rgba_image must have size width * height * 4");

    torch::Tensor rgba_cpu = rgba_image.cpu().contiguous();

    pnanovdb_compute_array_t *rgba_array =
        mEditor.compute.create_array(1u, width * height * 4, rgba_cpu.data_ptr());

    pnanovdb_compute_array_t *image_nanovdb =
        mEditor.compute.nanovdb_from_image_rgba8(rgba_array, width, height);

    mEditor.compute.destroy_array(rgba_array);

    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    pnanovdb_editor_token_t *nameToken  = mEditor.editor.get_token(name.c_str());

    mEditor.editor.add_nanovdb_2(&mEditor.editor, sceneToken, nameToken, image_nanovdb);

    pnanovdb_editor_shader_name_t *mapped =
        (pnanovdb_editor_shader_name_t *)mEditor.editor.map_params(
            &mEditor.editor,
            sceneToken,
            nameToken,
            PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_editor_shader_name_t));
    if (mapped) {
        mapped->shader_name = mEditor.editor.get_token("editor/image2d.slang");
        mEditor.editor.unmap_params(&mEditor.editor, sceneToken, nameToken);
    }

    mEditor.compute.destroy_array(image_nanovdb);
}

void
Viewer::addLevelSetView(const std::string &scene_name,
                        const std::string &name,
                        const GridBatchData &grid,
                        const JaggedTensor &sdf) {
    addNanoVDBGridView(scene_name,
                       name,
                       grid,
                       sdf,
                       getPipelineType(mEditor.editor, "pnanovdb_pipeline_type_nanovdb_surface"));
}

void
Viewer::addFogVolumeView(const std::string &scene_name,
                         const std::string &name,
                         const GridBatchData &grid,
                         const JaggedTensor &density) {
    addNanoVDBGridView(scene_name,
                       name,
                       grid,
                       density,
                       getPipelineType(mEditor.editor, "pnanovdb_pipeline_type_nanovdb_render"));
}

void
Viewer::addNanoVDBGridView(const std::string &scene_name,
                           const std::string &name,
                           const GridBatchData &grid,
                           const JaggedTensor &floatValues,
                           pnanovdb_pipeline_type_t render_pipeline,
                           const std::string &render_shader) {
    // The nanovdb-editor renders exactly one grid per view (its shader decodes a single
    // grid anchored at byte offset 0 of the uploaded buffer). Callers must expand a
    // multi-grid GridBatch into one view per grid; see fvdb.viz._nanovdb_grid_view.
    TORCH_CHECK(grid.batchSize() == 1,
                "addNanoVDBGridView expects a single grid (batchSize == 1), got a batch of ",
                grid.batchSize(),
                ". The viewer renders one grid per view; add one view per grid.");

    auto handle = detail::io::toNVDBWithBlindFloat(grid, floatValues);

    // Match the editor's own NanoVDB upload convention (compute.load_nanovdb): a NanoVDB
    // buffer is a uint32 array, so element_size = 4. This becomes the GPU buffer's
    // structure_stride at upload time. NanoVDB grids are 32-byte aligned, so the buffer
    // size is always a multiple of 4.
    const uint64_t bufferBytes = handle.buffer().size();
    TORCH_CHECK(bufferBytes % sizeof(uint32_t) == 0,
                "Internal error: NanoVDB buffer size ",
                bufferBytes,
                " is not 4-byte aligned");
    pnanovdb_compute_array_t *array = mEditor.compute.create_array(
        sizeof(uint32_t), bufferBytes / sizeof(uint32_t), handle.buffer().data());

    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    pnanovdb_editor_token_t *nameToken  = mEditor.editor.get_token(name.c_str());

    mEditor.editor.add_nanovdb_3(&mEditor.editor,
                                 sceneToken,
                                 nameToken,
                                 array,
                                 getPipelineType(mEditor.editor, "pnanovdb_pipeline_type_noop"),
                                 render_pipeline);

    if (!render_shader.empty()) {
        pnanovdb_editor_shader_name_t *mapped =
            (pnanovdb_editor_shader_name_t *)mEditor.editor.map_params(
                &mEditor.editor,
                sceneToken,
                nameToken,
                PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_editor_shader_name_t));
        if (mapped) {
            mapped->shader_name = mEditor.editor.get_token(render_shader.c_str());
            mEditor.editor.unmap_params(&mEditor.editor, sceneToken, nameToken);
        }
    }

    mEditor.compute.destroy_array(array);

    mNanoVDBViews[name] = NanoVDBView{name};
}

namespace {

void
appendJsonString(std::ostringstream &os, const std::string &s) {
    os << '"';
    for (char c: s) {
        switch (c) {
        case '"': os << "\\\""; break;
        case '\\': os << "\\\\"; break;
        case '\b': os << "\\b"; break;
        case '\f': os << "\\f"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8] = {};
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c) & 0xff);
                os << buf;
            } else {
                os << c;
            }
        }
    }
    os << '"';
}

void
appendJsonFloat(std::ostringstream &os, float v) {
    if (!std::isfinite(v)) {
        // clamp to 0 to avoid a parse error
        os << "0.0";
        return;
    }
    char buf[64] = {};
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    os << buf;
    // Make sure the output is valid JSON if `%g` printed an integer-looking
    // number for a float (e.g. "1" instead of "1.0").
    bool hasDot = false;
    for (char c: std::string(buf)) {
        if (c == '.' || c == 'e' || c == 'E') {
            hasDot = true;
            break;
        }
    }
    if (!hasDot) {
        os << ".0";
    }
}

} // namespace

std::string
Viewer::buildSchemaJson(const std::vector<CustomParamSpec> &specs) {
    std::ostringstream os;
    os << "{\"SceneParams\":{";
    bool first = true;
    for (const auto &spec: specs) {
        if (!first) {
            os << ',';
        }
        first = false;
        appendJsonString(os, spec.name);
        os << ":{";

        switch (spec.kind) {
        case CustomParamKind::Slider:
        case CustomParamKind::Number: {
            os << "\"type\":\"float\",\"value\":";
            appendJsonFloat(os, spec.defaultFloat);
            if (spec.hasMin) {
                os << ",\"min\":";
                appendJsonFloat(os, spec.minValue);
            }
            if (spec.hasMax) {
                os << ",\"max\":";
                appendJsonFloat(os, spec.maxValue);
            }
            if (spec.hasStep) {
                os << ",\"step\":";
                appendJsonFloat(os, spec.stepValue);
            }
            if (spec.kind == CustomParamKind::Slider) {
                os << ",\"useSlider\":true";
            }
            break;
        }
        case CustomParamKind::Text: {
            os << "\"type\":\"string\",\"length\":" << spec.lengthChars << ",\"value\":";
            appendJsonString(os, spec.defaultText);
            if (spec.commitOnEnter) {
                os << ",\"commitOnEnter\":true";
                if (!spec.submitCounterField.empty()) {
                    os << ",\"submitCounterField\":";
                    appendJsonString(os, spec.submitCounterField);
                }
            }
            break;
        }
        case CustomParamKind::Checkbox: {
            os << "\"type\":\"bool\",\"isBool\":true,\"value\":"
               << (spec.defaultBool ? "true" : "false");
            break;
        }
        case CustomParamKind::SubmitCounter: {
            // Hidden uint32, lives alongside a Text field that has commit_on_enter set.
            os << "\"type\":\"uint32\",\"value\":0,\"hidden\":true";
            break;
        }
        }
        os << '}';
    }
    os << "}}";
    return os.str();
}

bool
Viewer::findFieldInfo(const std::string &scene_name,
                      const std::string &field_name,
                      FieldInfo &out) {
    if (!mEditor.editor.get_custom_scene_params_data_type) {
        return false;
    }
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    const pnanovdb_reflect_data_type_t *dataType =
        mEditor.editor.get_custom_scene_params_data_type(&mEditor.editor, sceneToken);
    if (!dataType) {
        return false;
    }

    for (pnanovdb_uint64_t i = 0; i < dataType->child_reflect_data_count; ++i) {
        const pnanovdb_reflect_data_t *child = dataType->child_reflect_datas + i;
        const std::string childName          = child->name ? child->name : "";
        if (childName != field_name) {
            continue;
        }
        out.name                                      = childName;
        out.offset                                    = static_cast<uint64_t>(child->data_offset);
        const pnanovdb_reflect_data_type_t *childType = child->data_type;
        if (!childType) {
            return false;
        }
        const char *typeStr = pnanovdb_reflect_type_to_string(childType->data_type);
        out.type            = typeStr ? typeStr : "unknown";
        out.size            = static_cast<uint64_t>(childType->element_size);
        uint64_t scalarSize = 0;
        switch (childType->data_type) {
        case PNANOVDB_REFLECT_TYPE_INT32: scalarSize = 4; break;
        case PNANOVDB_REFLECT_TYPE_UINT32: scalarSize = 4; break;
        case PNANOVDB_REFLECT_TYPE_FLOAT: scalarSize = 4; break;
        case PNANOVDB_REFLECT_TYPE_BOOL32: scalarSize = 4; break;
        case PNANOVDB_REFLECT_TYPE_UINT8: scalarSize = 1; break;
        case PNANOVDB_REFLECT_TYPE_UINT16: scalarSize = 2; break;
        case PNANOVDB_REFLECT_TYPE_UINT64: scalarSize = 8; break;
        case PNANOVDB_REFLECT_TYPE_CHAR: scalarSize = 1; break;
        case PNANOVDB_REFLECT_TYPE_DOUBLE: scalarSize = 8; break;
        case PNANOVDB_REFLECT_TYPE_INT64: scalarSize = 8; break;
        default: scalarSize = 0; break;
        }
        out.element_size = scalarSize > 0 ? scalarSize : out.size;
        return true;
    }
    return false;
}

bool
Viewer::readFieldBytes(const std::string &scene_name, const FieldInfo &field, void *dest) {
    if (!mEditor.editor.get_custom_scene_params_data_type || !mEditor.editor.map_params ||
        !mEditor.editor.unmap_params) {
        return false;
    }
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    const pnanovdb_reflect_data_type_t *dataType =
        mEditor.editor.get_custom_scene_params_data_type(&mEditor.editor, sceneToken);
    if (!dataType || dataType->element_size == 0) {
        return false;
    }
    void *mapped = mEditor.editor.map_params(&mEditor.editor, sceneToken, nullptr, dataType);
    if (!mapped) {
        return false;
    }
    std::memcpy(dest, static_cast<const uint8_t *>(mapped) + field.offset, field.size);
    mEditor.editor.unmap_params(&mEditor.editor, sceneToken, nullptr);
    return true;
}

bool
Viewer::writeFieldBytes(const std::string &scene_name, const FieldInfo &field, const void *src) {
    if (!mEditor.editor.get_custom_scene_params_data_type || !mEditor.editor.map_params ||
        !mEditor.editor.unmap_params) {
        return false;
    }
    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    const pnanovdb_reflect_data_type_t *dataType =
        mEditor.editor.get_custom_scene_params_data_type(&mEditor.editor, sceneToken);
    if (!dataType || dataType->element_size == 0) {
        return false;
    }
    void *mapped = mEditor.editor.map_params(&mEditor.editor, sceneToken, nullptr, dataType);
    if (!mapped) {
        return false;
    }
    std::memcpy(static_cast<uint8_t *>(mapped) + field.offset, src, field.size);
    mEditor.editor.unmap_params(&mEditor.editor, sceneToken, nullptr);
    return true;
}

const Viewer::CustomParamSpec *
Viewer::findSpec(const std::string &scene_name, const std::string &field_name) const {
    auto it = mSceneCustomParams.find(scene_name);
    if (it == mSceneCustomParams.end()) {
        return nullptr;
    }
    for (const auto &spec: it->second) {
        if (spec.name == field_name) {
            return &spec;
        }
    }
    return nullptr;
}

Viewer::CustomParamSpec &
Viewer::replaceOrAppendSpec(const std::string &scene_name, const CustomParamSpec &spec) {
    auto &specs = mSceneCustomParams[scene_name];
    for (auto &existing: specs) {
        if (existing.name == spec.name) {
            existing = spec;
            return existing;
        }
    }
    specs.push_back(spec);
    return specs.back();
}

void
Viewer::submitSchemaForScene(const std::string &scene_name) {
    auto it = mSceneCustomParams.find(scene_name);
    if (it == mSceneCustomParams.end() || it->second.empty()) {
        return;
    }
    const auto &specs = it->second;

    if (!mEditor.editor.set_custom_scene_params) {
        throw std::runtime_error(
            "set_custom_scene_params is not available in the loaded nanovdb_editor binary");
    }

    // Snapshot any currently-loaded values so they survive the schema reload.
    struct SnapshotEntry {
        std::vector<uint8_t> bytes;
        FieldInfo info;
    };
    std::unordered_map<std::string, SnapshotEntry> snapshot;
    for (const auto &spec: specs) {
        FieldInfo info{};
        if (!findFieldInfo(scene_name, spec.name, info)) {
            continue;
        }
        SnapshotEntry entry;
        entry.bytes.resize(info.size);
        entry.info = info;
        if (readFieldBytes(scene_name, info, entry.bytes.data())) {
            snapshot.emplace(spec.name, std::move(entry));
        }
    }

    const std::string json = buildSchemaJson(specs);

    pnanovdb_editor_token_t *sceneToken = mEditor.editor.get_token(scene_name.c_str());
    pnanovdb_editor_token_t *jsonToken  = mEditor.editor.get_token(json.c_str());

    char errorBuf[512] = {};
    pnanovdb_bool_t ok = mEditor.editor.set_custom_scene_params(
        &mEditor.editor, sceneToken, jsonToken, errorBuf, sizeof(errorBuf));
    if (!ok) {
        errorBuf[sizeof(errorBuf) - 1] = '\0';
        const std::string msg =
            errorBuf[0] ? std::string(errorBuf) : "set_custom_scene_params failed";
        throw std::runtime_error("Failed to load custom scene params for scene '" + scene_name +
                                 "': " + msg);
    }

    // Restore previously-snapshotted values.
    for (const auto &kv: snapshot) {
        FieldInfo info{};
        if (!findFieldInfo(scene_name, kv.first, info)) {
            continue;
        }
        if (info.type != kv.second.info.type || info.size != kv.second.info.size) {
            continue;
        }
        writeFieldBytes(scene_name, info, kv.second.bytes.data());
    }
}

std::vector<std::string>
Viewer::sceneWidgetNames(const std::string &scene_name) const {
    std::vector<std::string> names;
    auto it = mSceneCustomParams.find(scene_name);
    if (it == mSceneCustomParams.end()) {
        return names;
    }
    names.reserve(it->second.size());
    for (const auto &spec: it->second) {
        // Skip internal hidden bookkeeping fields (e.g. submit counters attached to text widgets).
        if (spec.kind == CustomParamKind::SubmitCounter) {
            continue;
        }
        names.push_back(spec.name);
    }
    return names;
}

float
Viewer::readFloatField(const std::string &scene_name, const std::string &field_name) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "float") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'float'");
    }
    float value = 0.f;
    if (!readFieldBytes(scene_name, info, &value)) {
        throw std::runtime_error("Failed to read custom scene params field '" + field_name + "'");
    }
    return value;
}

void
Viewer::writeFloatField(const std::string &scene_name, const std::string &field_name, float value) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "float") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'float'");
    }
    if (!writeFieldBytes(scene_name, info, &value)) {
        throw std::runtime_error("Failed to write custom scene params field '" + field_name + "'");
    }
}

std::string
Viewer::readStringField(const std::string &scene_name, const std::string &field_name) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "char") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'char'");
    }
    std::vector<char> buf(static_cast<size_t>(info.size));
    if (!readFieldBytes(scene_name, info, buf.data())) {
        throw std::runtime_error("Failed to read custom scene params field '" + field_name + "'");
    }
    // Decode as a NUL-terminated string.
    size_t len = 0;
    while (len < buf.size() && buf[len] != '\0') {
        ++len;
    }
    return std::string(buf.data(), len);
}

void
Viewer::writeStringField(const std::string &scene_name,
                         const std::string &field_name,
                         const std::string &value) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "char") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'char'");
    }
    const size_t capacity = static_cast<size_t>(info.size);
    if (capacity == 0) {
        throw std::runtime_error("Custom scene params field '" + field_name +
                                 "' has zero capacity");
    }
    if (value.size() > capacity - 1) {
        throw std::runtime_error("Value for custom scene params field '" + field_name +
                                 "' is too long: " + std::to_string(value.size()) + " bytes, max " +
                                 std::to_string(capacity - 1) + " (excluding NUL terminator)");
    }
    std::vector<char> buf(capacity, '\0');
    std::memcpy(buf.data(), value.data(), value.size());
    if (!writeFieldBytes(scene_name, info, buf.data())) {
        throw std::runtime_error("Failed to write custom scene params field '" + field_name + "'");
    }
}

uint32_t
Viewer::readUInt32Field(const std::string &scene_name, const std::string &field_name) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "uint32") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'uint32'");
    }
    uint32_t raw = 0;
    if (!readFieldBytes(scene_name, info, &raw)) {
        throw std::runtime_error("Failed to read custom scene params field '" + field_name + "'");
    }
    return raw;
}

bool
Viewer::readBoolField(const std::string &scene_name, const std::string &field_name) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "bool32") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'bool32'");
    }
    uint32_t raw = 0;
    if (!readFieldBytes(scene_name, info, &raw)) {
        throw std::runtime_error("Failed to read custom scene params field '" + field_name + "'");
    }
    return raw != 0u;
}

void
Viewer::writeBoolField(const std::string &scene_name, const std::string &field_name, bool value) {
    FieldInfo info{};
    if (!findFieldInfo(scene_name, field_name, info)) {
        throw std::runtime_error("No custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "'");
    }
    if (info.type != "bool32") {
        throw std::runtime_error("Custom scene params field '" + field_name + "' on scene '" +
                                 scene_name + "' has type '" + info.type + "', expected 'bool32'");
    }
    uint32_t raw = value ? 1u : 0u;
    if (!writeFieldBytes(scene_name, info, &raw)) {
        throw std::runtime_error("Failed to write custom scene params field '" + field_name + "'");
    }
}

SliderView
Viewer::addSlider(const std::string &scene_name,
                  const std::string &name,
                  float min,
                  float max,
                  float initial,
                  float step) {
    if (!(min < max)) {
        throw std::invalid_argument("Slider min must be less than max");
    }
    if (!(step > 0.f)) {
        throw std::invalid_argument("Slider step must be positive");
    }

    CustomParamSpec spec;
    spec.name         = name;
    spec.kind         = CustomParamKind::Slider;
    spec.hasMin       = true;
    spec.hasMax       = true;
    spec.hasStep      = true;
    spec.minValue     = min;
    spec.maxValue     = max;
    spec.stepValue    = step;
    spec.defaultFloat = initial;
    replaceOrAppendSpec(scene_name, spec);

    submitSchemaForScene(scene_name);

    // Replace any existing view to refresh metadata.
    mSliderViews.erase(name);
    auto [it, inserted] = mSliderViews.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(scene_name, name, this, min, max, step, initial));
    return it->second;
}

NumberView
Viewer::addNumber(const std::string &scene_name,
                  const std::string &name,
                  float initial,
                  bool hasMin,
                  float min,
                  bool hasMax,
                  float max,
                  float step) {
    if (hasMin && hasMax && !(min < max)) {
        throw std::invalid_argument("Number min must be less than max");
    }
    if (!(step > 0.f)) {
        throw std::invalid_argument("Number step must be positive");
    }

    CustomParamSpec spec;
    spec.name         = name;
    spec.kind         = CustomParamKind::Number;
    spec.hasMin       = hasMin;
    spec.hasMax       = hasMax;
    spec.hasStep      = true;
    spec.minValue     = min;
    spec.maxValue     = max;
    spec.stepValue    = step;
    spec.defaultFloat = initial;
    replaceOrAppendSpec(scene_name, spec);

    submitSchemaForScene(scene_name);

    mNumberViews.erase(name);
    auto [it, inserted] = mNumberViews.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(scene_name, name, this, hasMin, min, hasMax, max, step, initial));
    return it->second;
}

TextView
Viewer::addText(const std::string &scene_name,
                const std::string &name,
                const std::string &initial,
                int32_t maxLength,
                bool commitOnEnter) {
    if (maxLength <= 1) {
        throw std::invalid_argument("Text max_length must be at least 2 (one byte plus NUL)");
    }
    if (initial.size() > static_cast<size_t>(maxLength - 1)) {
        throw std::invalid_argument(
            "Text initial value is longer than max_length - 1 (NUL-terminated capacity)");
    }

    const std::string counterFieldName =
        commitOnEnter ? (name + std::string(kSubmitCounterSuffix)) : std::string();

    CustomParamSpec spec;
    spec.name               = name;
    spec.kind               = CustomParamKind::Text;
    spec.lengthChars        = maxLength;
    spec.defaultText        = initial;
    spec.commitOnEnter      = commitOnEnter;
    spec.submitCounterField = counterFieldName;
    replaceOrAppendSpec(scene_name, spec);

    if (commitOnEnter) {
        CustomParamSpec counter;
        counter.name = counterFieldName;
        counter.kind = CustomParamKind::SubmitCounter;
        replaceOrAppendSpec(scene_name, counter);
    } else {
        // If the user previously created this text widget with
        // commit_on_enter=true and is now turning it off, drop any leftover
        // counter spec for this widget.
        auto it = mSceneCustomParams.find(scene_name);
        if (it != mSceneCustomParams.end()) {
            auto &specs                 = it->second;
            const std::string staleName = name + std::string(kSubmitCounterSuffix);
            specs.erase(std::remove_if(specs.begin(),
                                       specs.end(),
                                       [&staleName](const CustomParamSpec &s) {
                                           return s.kind == CustomParamKind::SubmitCounter &&
                                                  s.name == staleName;
                                       }),
                        specs.end());
        }
    }

    submitSchemaForScene(scene_name);

    mTextViews.erase(name);
    auto [it, inserted] = mTextViews.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(scene_name, name, this, maxLength, initial, commitOnEnter));
    return it->second;
}

CheckboxView
Viewer::addCheckbox(const std::string &scene_name, const std::string &name, bool initial) {
    CustomParamSpec spec;
    spec.name        = name;
    spec.kind        = CustomParamKind::Checkbox;
    spec.defaultBool = initial;
    replaceOrAppendSpec(scene_name, spec);

    submitSchemaForScene(scene_name);

    mCheckboxViews.erase(name);
    auto [it, inserted] =
        mCheckboxViews.emplace(std::piecewise_construct,
                               std::forward_as_tuple(name),
                               std::forward_as_tuple(scene_name, name, this, initial));
    return it->second;
}

// ---------------------------------------------------------------------------
// SliderView / NumberView / TextView / CheckboxView implementations
// ---------------------------------------------------------------------------

float
SliderView::getValue() const {
    return mViewer->readFloatField(mSceneName, mName);
}

void
SliderView::setValue(float value) {
    mViewer->writeFloatField(mSceneName, mName, value);
}

float
NumberView::getValue() const {
    return mViewer->readFloatField(mSceneName, mName);
}

void
NumberView::setValue(float value) {
    mViewer->writeFloatField(mSceneName, mName, value);
}

std::string
TextView::getValue() const {
    return mViewer->readStringField(mSceneName, mName);
}

void
TextView::setValue(const std::string &value) {
    mViewer->writeStringField(mSceneName, mName, value);
}

uint32_t
TextView::getSubmitCounter() const {
    if (!mCommitOnEnter) {
        return 0u;
    }
    const std::string counterField = mName + std::string(kSubmitCounterSuffix);
    return mViewer->readUInt32Field(mSceneName, counterField);
}

bool
CheckboxView::getValue() const {
    return mViewer->readBoolField(mSceneName, mName);
}

void
CheckboxView::setValue(bool value) {
    mViewer->writeBoolField(mSceneName, mName, value);
}

} // namespace fvdb::detail::viewer
