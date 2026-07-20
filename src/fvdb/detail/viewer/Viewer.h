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
#include <fvdb/detail/viewer/ParamViews.h>

#include <c10/util/Exception.h>
#include <torch/types.h>

#include <nanovdb_editor/putil/Editor.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

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

    enum class CustomParamKind {
        Slider,        ///< float, useSlider=true, has min/max/step
        Number,        ///< float, useSlider=false, optional min/max/step
        Text,          ///< char[length] string
        Checkbox,      ///< bool32 rendered as checkbox (isBool=true)
        SubmitCounter, ///< hidden uint32 bumped by editor on Enter (companion to Text fields with
                       ///< commitOnEnter=true)
    };

    struct CustomParamSpec {
        std::string name;
        CustomParamKind kind = CustomParamKind::Slider;
        // Numeric (slider/number/checkbox)
        bool hasMin        = false;
        bool hasMax        = false;
        bool hasStep       = false;
        float minValue     = 0.f;
        float maxValue     = 0.f;
        float stepValue    = 0.01f;
        float defaultFloat = 0.f;
        // Text
        int32_t lengthChars = 0;
        std::string defaultText;
        bool commitOnEnter =
            false;              ///< Text only: render with EnterReturnsTrue and bump submit counter
        std::string
            submitCounterField; ///< Text only: name of the companion SubmitCounter spec, if any
        // Bool
        bool defaultBool = false;
    };

    std::map<std::string, std::vector<CustomParamSpec>> mSceneCustomParams;
    std::map<std::string, SliderView> mSliderViews;
    std::map<std::string, NumberView> mNumberViews;
    std::map<std::string, TextView> mTextViews;
    std::map<std::string, CheckboxView> mCheckboxViews;
    std::map<std::string, NanoVDBView> mNanoVDBViews;

    void updateCamera(const std::string &scene_name);
    void getCamera(const std::string &scene_name);

    void startServer();
    void stopServer();

    // Shared implementation behind addLevelSetView / addFogVolumeView: builds an ONINDEX NanoVDB
    // buffer carrying `floatValues` as blind data and registers it under the given nanovdb-editor
    // render pipeline. `render_shader` optionally overrides the editor-selected shader when
    // non-empty. `grid` must contain exactly one grid (batchSize() == 1).
    void addNanoVDBGridView(const std::string &scene_name,
                            const std::string &name,
                            const GridBatchData &grid,
                            const JaggedTensor &floatValues,
                            pnanovdb_pipeline_type_t render_pipeline,
                            const std::string &render_shader = "");

    struct FieldInfo {
        std::string name;
        std::string type; ///< "float", "int32", "char", "bool32", ...
        uint64_t offset       = 0;
        uint64_t size         = 0;
        uint64_t element_size = 0;
    };

    static std::string buildSchemaJson(const std::vector<CustomParamSpec> &specs);

    bool
    findFieldInfo(const std::string &scene_name, const std::string &field_name, FieldInfo &out);

    bool readFieldBytes(const std::string &scene_name, const FieldInfo &field, void *dest);

    bool writeFieldBytes(const std::string &scene_name, const FieldInfo &field, const void *src);

    void submitSchemaForScene(const std::string &scene_name);

    const CustomParamSpec *findSpec(const std::string &scene_name,
                                    const std::string &field_name) const;

    CustomParamSpec &replaceOrAppendSpec(const std::string &scene_name,
                                         const CustomParamSpec &spec);

  public:
    Viewer(const std::string &ipAddress,
           const int port,
           const int device_id,
           const bool verbose = false);
    ~Viewer();

    void reset();

    void stop();

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

    SliderView addSlider(const std::string &scene_name,
                         const std::string &name,
                         float min,
                         float max,
                         float initial,
                         float step);

    NumberView addNumber(const std::string &scene_name,
                         const std::string &name,
                         float initial,
                         bool hasMin,
                         float min,
                         bool hasMax,
                         float max,
                         float step);

    TextView addText(const std::string &scene_name,
                     const std::string &name,
                     const std::string &initial,
                     int32_t maxLength,
                     bool commitOnEnter = false);

    CheckboxView addCheckbox(const std::string &scene_name, const std::string &name, bool initial);

    bool
    hasSlider(const std::string &name) const {
        return mSliderViews.find(name) != mSliderViews.end();
    }
    bool
    hasNumber(const std::string &name) const {
        return mNumberViews.find(name) != mNumberViews.end();
    }
    bool
    hasText(const std::string &name) const {
        return mTextViews.find(name) != mTextViews.end();
    }
    bool
    hasCheckbox(const std::string &name) const {
        return mCheckboxViews.find(name) != mCheckboxViews.end();
    }

    SliderView
    getSlider(const std::string &name) const {
        const auto it = mSliderViews.find(name);
        TORCH_CHECK(it != mSliderViews.end(), "No SliderView with name '", name, "' found");
        return it->second;
    }
    NumberView
    getNumber(const std::string &name) const {
        const auto it = mNumberViews.find(name);
        TORCH_CHECK(it != mNumberViews.end(), "No NumberView with name '", name, "' found");
        return it->second;
    }
    TextView
    getText(const std::string &name) const {
        const auto it = mTextViews.find(name);
        TORCH_CHECK(it != mTextViews.end(), "No TextView with name '", name, "' found");
        return it->second;
    }
    CheckboxView
    getCheckbox(const std::string &name) const {
        const auto it = mCheckboxViews.find(name);
        TORCH_CHECK(it != mCheckboxViews.end(), "No CheckboxView with name '", name, "' found");
        return it->second;
    }

    std::vector<std::string> sceneWidgetNames(const std::string &scene_name) const;

    float readFloatField(const std::string &scene_name, const std::string &field_name);
    void writeFloatField(const std::string &scene_name, const std::string &field_name, float value);

    std::string readStringField(const std::string &scene_name, const std::string &field_name);
    void writeStringField(const std::string &scene_name,
                          const std::string &field_name,
                          const std::string &value);

    bool readBoolField(const std::string &scene_name, const std::string &field_name);
    void writeBoolField(const std::string &scene_name, const std::string &field_name, bool value);

    uint32_t readUInt32Field(const std::string &scene_name, const std::string &field_name);

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
