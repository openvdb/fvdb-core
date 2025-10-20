// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "fvdb/detail/viewer/GaussianSplat3dView.h"

#include <fvdb/detail/viewer/Viewer.h>

#include <c10/util/Exception.h>

#include <nanovdb_editor/putil/Raster.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>

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

void
Viewer::updateCamera() {
    mEditor.editor.update_camera(&mEditor.editor, &mEditor.camera);
}

Viewer::Viewer(const std::string &ipAddress, const int port, const bool verbose)
    : mIpAddress(ipAddress), mPort(port) {
    mEditor.compiler = {};
    pnanovdb_compiler_load(&mEditor.compiler);

    mEditor.compute = {};
    pnanovdb_compute_load(&mEditor.compute, &mEditor.compiler);

    mEditor.deviceDesc           = {};
    mEditor.deviceDesc.log_print = verbose ? pNanoLogPrintVerbose : pNanoLogPrint;

    mEditor.deviceManager = mEditor.compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    mEditor.device =
        mEditor.compute.device_interface.create_device(mEditor.deviceManager, &mEditor.deviceDesc);

    mEditor.editor = {};
    pnanovdb_editor_load(&mEditor.editor, &mEditor.compute, &mEditor.compiler);

    pnanovdb_compute_queue_t *queue =
        mEditor.compute.device_interface.get_compute_queue(mEditor.device);
    if (queue == nullptr) {
        throw std::runtime_error("Failed to get compute queue");
    }

    // init raster
    mEditor.raster = {};
    pnanovdb_raster_load(&mEditor.raster, &mEditor.compute);
    pnanovdb_raster_context_t *raster_context =
        mEditor.raster.create_context(&mEditor.compute, queue);
    if (raster_context == nullptr) {
        throw std::runtime_error("Failed to create raster context");
    }
    mEditor.raster_ctx = raster_context;

    pnanovdb_camera_init(&mEditor.camera);
    mEditor.editor.update_camera(&mEditor.editor, &mEditor.camera);

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

    // Clear views before unloading editor to ensure proper cleanup order
    mSplat3dViews.clear();
    mCameraViews.clear();

    pnanovdb_compute_queue_t *queue =
        mEditor.compute.device_interface.get_compute_queue(mEditor.device);

    mEditor.raster.destroy_context(mEditor.raster.compute, queue, mEditor.raster_ctx);

    mEditor.compute.device_interface.destroy_device(mEditor.deviceManager, mEditor.device);
    mEditor.compute.device_interface.destroy_device_manager(mEditor.deviceManager);

    pnanovdb_editor_free(&mEditor.editor);
    pnanovdb_raster_free(&mEditor.raster);
    pnanovdb_compute_free(&mEditor.compute);
    pnanovdb_compiler_free(&mEditor.compiler);
}

fvdb::detail::viewer::GaussianSplat3dView &
Viewer::addGaussianSplat3dView(const std::string &name, const GaussianSplat3d &splats) {
    std::shared_ptr<pnanovdb_raster_gaussian_data_t> oldData;
    auto itPrev = mSplat3dViews.find(name);
    if (itPrev != mSplat3dViews.end()) {
        oldData = itPrev->second.mGaussianData;
        mSplat3dViews.erase(itPrev);
    }

    auto [it, inserted] = mSplat3dViews.emplace(
        std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(name, *this));

    // Get the various tensors to pass to the viewer
    torch::Tensor means          = splats.means();
    torch::Tensor quats          = splats.quats();
    torch::Tensor logScales      = splats.logScales();
    torch::Tensor logitOpacities = splats.logitOpacities();
    torch::Tensor sh0            = splats.sh0();
    torch::Tensor shN            = splats.shN();

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

    pnanovdb_compute_queue_t *queue =
        mEditor.compute.device_interface.get_device_queue(mEditor.device);

    // Load splats into viewer
    pnanovdb_raster_gaussian_data_t *pGaussianData = nullptr;
    mEditor.raster.create_gaussian_data_from_arrays(&mEditor.raster,
                                                    &mEditor.compute,
                                                    queue,
                                                    arrays,
                                                    6u,
                                                    &pGaussianData,
                                                    &it->second.mParams,
                                                    &mEditor.raster_ctx);
    if (pGaussianData == nullptr) {
        throw std::runtime_error("Failed to create gaussian data");
    }
    // printf("Created gaussian data - %p\n", pGaussianData);

    mEditor.editor.add_gaussian_data(&mEditor.editor, mEditor.raster_ctx, queue, pGaussianData);

    it->second.mGaussianData = std::shared_ptr<pnanovdb_raster_gaussian_data_t>(
        pGaussianData, [this](pnanovdb_raster_gaussian_data_t *ptr) {
            if (ptr && mEditor.device) {
                pnanovdb_compute_queue_t *deviceQueue =
                    mEditor.compute.device_interface.get_device_queue(mEditor.device);
                if (deviceQueue) {
                    mEditor.raster.destroy_gaussian_data(mEditor.raster.compute, deviceQueue, ptr);
                    // printf("Destroyed gaussian data - %p\n", ptr);
                }
            }
        });

    pnanovdb_raster_shader_params_t *params = &it->second.mParams;
    it->second.mSyncCallback                = [this, params](bool set_data) {
        mEditor.editor.sync_shader_params(
            &mEditor.editor, params, set_data ? PNANOVDB_TRUE : PNANOVDB_FALSE);
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

std::tuple<float, float, float>
Viewer::cameraOrbitCenter() const {
    return std::make_tuple(mEditor.camera.state.position.x,
                           mEditor.camera.state.position.y,
                           mEditor.camera.state.position.z);
}
void
Viewer::setCameraOrbitCenter(float x, float y, float z) {
    mEditor.camera.state.position.x = x;
    mEditor.camera.state.position.y = y;
    mEditor.camera.state.position.z = z;
    updateCamera();
}

float
Viewer::cameraOrbitRadius() const {
    return mEditor.camera.state.eye_distance_from_position;
}
void
Viewer::setCameraOrbitRadius(float radius) {
    mEditor.camera.state.eye_distance_from_position = radius;
    updateCamera();
}

std::tuple<float, float, float>
Viewer::cameraViewDirection() const {
    return std::make_tuple(mEditor.camera.state.eye_direction.x,
                           mEditor.camera.state.eye_direction.y,
                           mEditor.camera.state.eye_direction.z);
}
void
Viewer::setCameraViewDirection(float x, float y, float z) {
    mEditor.camera.state.eye_direction.x = x;
    mEditor.camera.state.eye_direction.y = y;
    mEditor.camera.state.eye_direction.z = z;
    updateCamera();
}

std::tuple<float, float, float>
Viewer::cameraUpDirection() const {
    return std::make_tuple(mEditor.camera.state.eye_up.x,
                           mEditor.camera.state.eye_up.y,
                           mEditor.camera.state.eye_up.z);
}
void
Viewer::setCameraUpDirection(float x, float y, float z) {
    mEditor.camera.state.eye_up.x = x;
    mEditor.camera.state.eye_up.y = y;
    mEditor.camera.state.eye_up.z = z;
    updateCamera();
}

float
Viewer::cameraNear() const {
    return mEditor.camera.config.near_plane;
}
void
Viewer::setCameraNear(float near) {
    mEditor.camera.config.near_plane = near;
    updateCamera();
}

float
Viewer::cameraFar() const {
    return mEditor.camera.config.far_plane;
}
void
Viewer::setCameraFar(float far) {
    mEditor.camera.config.far_plane = far;
    updateCamera();
}

GaussianSplat3d::ProjectionType
Viewer::cameraProjectionType() const {
    return mEditor.camera.config.is_orthographic ? GaussianSplat3d::ProjectionType::ORTHOGRAPHIC
                                                 : GaussianSplat3d::ProjectionType::PERSPECTIVE;
}
void
Viewer::setCameraProjectionType(GaussianSplat3d::ProjectionType mode) {
    mEditor.camera.config.is_orthographic =
        (mode == GaussianSplat3d::ProjectionType::ORTHOGRAPHIC) ? PNANOVDB_TRUE : PNANOVDB_FALSE;

    updateCamera();
}

CameraView &
Viewer::addCameraView(const std::string &name,
                      const torch::Tensor &cameraToWorldMatrices,
                      const torch::Tensor &projectionMatrices,
                      const torch::Tensor &imageSizes,
                      float frustumNear,
                      float frustumFar) {
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
                    "image_sizes must have shape [N, 2] if provided");
    }

    auto [it, inserted] = mCameraViews.emplace(
        std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(name));

    it->second.mView.num_cameras = numCameras;
    it->second.mView.states      = new pnanovdb_camera_state_t[it->second.mView.num_cameras];
    it->second.mView.configs     = new pnanovdb_camera_config_t[it->second.mView.num_cameras];

    for (int i = 0; i < (int)it->second.mView.num_cameras; i++) {
        torch::Tensor c2w = cameraToWorldMatrices.index({i}).contiguous().cpu();
        torch::Tensor K   = projectionMatrices.index({i}).contiguous().cpu();

        float px = c2w[0][3].item<float>();
        float py = c2w[1][3].item<float>();
        float pz = c2w[2][3].item<float>();

        float zx = c2w[0][2].item<float>();
        float zy = c2w[1][2].item<float>();
        float zz = c2w[2][2].item<float>();

        float ux = c2w[0][1].item<float>();
        float uy = c2w[1][1].item<float>();
        float uz = c2w[2][1].item<float>();

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
        float fy      = K[1][1].item<float>();
        float width   = 0.f;
        float height  = 0.f;
        bool haveDims = imageSizes.numel() != 0;
        if (haveDims) {
            torch::Tensor dims = imageSizes.index({i}).contiguous().cpu();
            height             = dims[0].item<float>();
            width              = dims[1].item<float>();
        }

        if (haveDims && height > 0.f && fy > 0.f) {
            it->second.mView.configs[i].fov_angle_y  = 2.f * std::atan(0.5f * height / fy);
            it->second.mView.configs[i].aspect_ratio = width / height;
        } else {
            it->second.mView.configs[i].fov_angle_y  = DEFAULT_CAMERA_FOV_RADIANS;
            it->second.mView.configs[i].aspect_ratio = DEFAULT_CAMERA_ASPECT_RATIO;
        }
    }

    mEditor.editor.add_camera_view(&mEditor.editor, &it->second.mView);
    return it->second;
}

} // namespace fvdb::detail::viewer
