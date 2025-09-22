// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/viewer/Viewer.h>

#include <c10/util/Exception.h>

#include <nanovdb_editor/putil/Raster.h>
#include <nanovdb_editor/putil/Raster.hpp>

#include <cstdarg>
#include <cstdio>

// #define TEST_RGBRGB

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
    }
    printf("%s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

namespace fvdb::detail::viewer {

void
Viewer::updateCamera() {
    mEditor.editor.add_camera(&mEditor.editor, &mEditor.camera);
}

Viewer::Viewer(const std::string &ipAddress, const int port, const bool verbose)
    : mIpAddress(ipAddress), mPort(port) {
    mEditor.compiler = {};
    pnanovdb_compiler_load(&mEditor.compiler);

    mEditor.compute = {};
    pnanovdb_compute_load(&mEditor.compute, &mEditor.compiler);

    mEditor.deviceDesc           = {};
    mEditor.deviceDesc.log_print = verbose ? pNanoLogPrint : nullptr;

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
    mEditor.editor.raster_ctx      = raster_context; // destroyed by editor
    mEditor.rasterShaderParamsType = PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_raster_shader_params_t);

    pnanovdb_camera_init(&mEditor.camera);
    mEditor.editor.add_camera(&mEditor.editor, &mEditor.camera);

    mEditor.config            = {};
    mEditor.config.ip_address = mIpAddress.c_str();
    mEditor.config.port       = port;
    mEditor.config.headless   = PNANOVDB_TRUE;
    mEditor.config.streaming  = PNANOVDB_TRUE;

    mIsEditorRunning = false;

    startServer();
}

Viewer::~Viewer() {
    stopServer();
    pnanovdb_editor_free(&mEditor.editor);

    mEditor.compute.device_interface.destroy_device(mEditor.deviceManager, mEditor.device);
    mEditor.compute.device_interface.destroy_device_manager(mEditor.deviceManager);

    pnanovdb_raster_free(&mEditor.raster);
    pnanovdb_compute_free(&mEditor.compute);
    pnanovdb_compiler_free(&mEditor.compiler);
}

fvdb::detail::viewer::GaussianSplat3dView &
Viewer::registerGaussianSplat3dView(const std::string &name, const GaussianSplat3d &splats) {
    auto [it, inserted] = mSplat3dViews.emplace(std::piecewise_construct,
                                                std::forward_as_tuple(name),
                                                std::forward_as_tuple(name, splats, *this));
    // Load splats into viewer

    // Get the various tensors to pass to the viewer
    torch::Tensor means          = splats.means();
    torch::Tensor quats          = splats.quats();
    torch::Tensor logScales      = splats.logScales();
    torch::Tensor logitOpacities = splats.logitOpacities();
    torch::Tensor sh0            = splats.sh0();
    torch::Tensor shN            = splats.shN();

    // Use RRRGGGBBB
    torch::Tensor sh = torch::cat({sh0, shN}, 1);

#ifdef TEST_RGBRGB
    int N              = means.size(0);
    auto shN_flat      = shN.reshape({N, 45});
    auto shN_R         = shN_flat.slice(1, 0, 15).unsqueeze(2);  // (N, 15, 1)
    auto shN_G         = shN_flat.slice(1, 15, 30).unsqueeze(2); // (N, 15, 1)
    auto shN_B         = shN_flat.slice(1, 30, 45).unsqueeze(2); // (N, 15, 1)
    auto shN_reordered = torch::cat({shN_R, shN_G, shN_B}, 2);   // (N, 15, 3)
    sh                 = torch::cat({sh0, shN_reordered}, 1);    // (N, 16, 3)
#endif

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
    pnanovdb_compute_array_t *sh_arr             = makeComputeArray(sh);

    pnanovdb_compute_array_t *arrays[] = {
        means_arr, logitOpacities_arr, quats_arr, logScales_arr, sh_arr};

    pnanovdb_compute_queue_t *queue =
        mEditor.compute.device_interface.get_compute_queue(mEditor.device);

    pnanovdb_raster_gaussian_data_t *gaussian_data = nullptr;
    pnanovdb_raster::get_gaussian_data(&mEditor.raster,
                                       &mEditor.compute,
                                       queue,
                                       arrays,
                                       &gaussian_data,
                                       nullptr,
                                       &mEditor.editor.raster_ctx);
    if (gaussian_data == nullptr) {
        throw std::runtime_error("Failed to create gaussian data");
    }

    mEditor.editor.add_gaussian_data(&mEditor.editor, &mEditor.raster, queue, gaussian_data);
    mEditor.editor.setup_shader_params(
        &mEditor.editor, &it->second.mParams, mEditor.rasterShaderParamsType);
    it->second.mSyncCallback = [this](bool set_data) {
        mEditor.editor.sync_shader_params(&mEditor.editor,
                                          mEditor.rasterShaderParamsType,
                                          set_data ? PNANOVDB_TRUE : PNANOVDB_FALSE);
        mEditor.editor.wait_for_shader_params_sync(&mEditor.editor, mEditor.rasterShaderParamsType);
    };
    it->second.mSyncCallback(true); // initial sync

    mEditor.compute.destroy_array(means_arr);
    mEditor.compute.destroy_array(quats_arr);
    mEditor.compute.destroy_array(logScales_arr);
    mEditor.compute.destroy_array(logitOpacities_arr);
    mEditor.compute.destroy_array(sh_arr);

    return it->second;
}

void
Viewer::startServer() {
    if (!mIsEditorRunning) {
        printf("Starting NanoVDB Editor server at %s:%d\n", mIpAddress.c_str(), mPort);
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
Viewer::setCameraPosition(float x, float y, float z) {
    mEditor.camera.state.position.x = x;
    mEditor.camera.state.position.y = y;
    mEditor.camera.state.position.z = z;
    updateCamera();
}

std::tuple<float, float, float>
Viewer::getCameraPosition() {
    return std::make_tuple(mEditor.camera.state.position.x,
                           mEditor.camera.state.position.y,
                           mEditor.camera.state.position.z);
}

void
Viewer::setCameraLookat(float x, float y, float z) {
    pnanovdb_vec3_t lookat    = {x, y, z};
    pnanovdb_vec3_t direction = {lookat.x - mEditor.camera.state.position.x,
                                 lookat.y - mEditor.camera.state.position.y,
                                 lookat.z - mEditor.camera.state.position.z};
    float length =
        sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    if (length > 0.0f) {
        mEditor.camera.state.eye_direction.x = direction.x / length;
        mEditor.camera.state.eye_direction.y = direction.y / length;
        mEditor.camera.state.eye_direction.z = direction.z / length;
    }
    updateCamera();
}

std::tuple<float, float, float>
Viewer::getCameraLookat() {
    float lookat_x = mEditor.camera.state.position.x + mEditor.camera.state.eye_direction.x;
    float lookat_y = mEditor.camera.state.position.y + mEditor.camera.state.eye_direction.y;
    float lookat_z = mEditor.camera.state.position.z + mEditor.camera.state.eye_direction.z;

    return std::make_tuple(lookat_x, lookat_y, lookat_z);
}

void
Viewer::setCameraNear(float near) {
    mEditor.camera.config.near_plane = near;
    updateCamera();
}

float
Viewer::getCameraNear() {
    return mEditor.camera.config.near_plane;
}

void
Viewer::setCameraFar(float far) {
    mEditor.camera.config.far_plane = far;
    updateCamera();
}

float
Viewer::getCameraFar() {
    return mEditor.camera.config.far_plane;
}

void
Viewer::setCameraPose(torch::Tensor cameraToWorldMatrix) {
    TORCH_CHECK(cameraToWorldMatrix.dim() == 2, "Camera matrix must be 2D");
    TORCH_CHECK(cameraToWorldMatrix.size(0) == 4 && cameraToWorldMatrix.size(1) == 4,
                "Camera matrix must be 4x4");

    // position from last column
    mEditor.camera.state.position.x = cameraToWorldMatrix[0][3].item<float>();
    mEditor.camera.state.position.y = cameraToWorldMatrix[1][3].item<float>();
    mEditor.camera.state.position.z = cameraToWorldMatrix[2][3].item<float>();

    // forward direction from negative Z axis
    mEditor.camera.state.eye_direction.x = -cameraToWorldMatrix[0][2].item<float>();
    mEditor.camera.state.eye_direction.y = -cameraToWorldMatrix[1][2].item<float>();
    mEditor.camera.state.eye_direction.z = -cameraToWorldMatrix[2][2].item<float>();

    // up direction from Y axis
    mEditor.camera.state.eye_up.x = cameraToWorldMatrix[0][1].item<float>();
    mEditor.camera.state.eye_up.y = cameraToWorldMatrix[1][1].item<float>();
    mEditor.camera.state.eye_up.z = cameraToWorldMatrix[2][1].item<float>();

    updateCamera();
}

void
Viewer::setCameraEyeDirection(float x, float y, float z) {
    mEditor.camera.state.eye_direction.x = x;
    mEditor.camera.state.eye_direction.y = y;
    mEditor.camera.state.eye_direction.z = z;
    updateCamera();
}

std::tuple<float, float, float>
Viewer::getCameraEyeDirection() {
    return std::make_tuple(mEditor.camera.state.eye_direction.x,
                           mEditor.camera.state.eye_direction.y,
                           mEditor.camera.state.eye_direction.z);
}

void
Viewer::setCameraEyeUp(float x, float y, float z) {
    mEditor.camera.state.eye_up.x = x;
    mEditor.camera.state.eye_up.y = y;
    mEditor.camera.state.eye_up.z = z;
    updateCamera();
}

std::tuple<float, float, float>
Viewer::getCameraEyeUp() {
    return std::make_tuple(mEditor.camera.state.eye_up.x,
                           mEditor.camera.state.eye_up.y,
                           mEditor.camera.state.eye_up.z);
}

void
Viewer::setCameraEyeDistanceFromPosition(float distance) {
    mEditor.camera.state.eye_distance_from_position = distance;
    updateCamera();
}

float
Viewer::getCameraEyeDistanceFromPosition() {
    return mEditor.camera.state.eye_distance_from_position;
}

void
Viewer::setCameraMode(GaussianSplat3d::ProjectionType mode) {
    mEditor.camera.config.is_orthographic =
        (mode == GaussianSplat3d::ProjectionType::ORTHOGRAPHIC) ? PNANOVDB_TRUE : PNANOVDB_FALSE;
    mEditor.camera.config.is_projection_rh = ~mEditor.camera.config.is_orthographic;

    updateCamera();
}

GaussianSplat3d::ProjectionType
Viewer::getCameraMode() {
    return mEditor.camera.config.is_orthographic ? GaussianSplat3d::ProjectionType::ORTHOGRAPHIC
                                                 : GaussianSplat3d::ProjectionType::PERSPECTIVE;
}

} // namespace fvdb::detail::viewer
