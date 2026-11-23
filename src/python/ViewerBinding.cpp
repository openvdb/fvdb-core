// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <fvdb/detail/utils/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/viewer/CameraView.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>
#include <fvdb/detail/viewer/ParamViews.h>
#include <fvdb/detail/viewer/Viewer.h>

#include <torch/python.h>

void
bind_viewer(py::module &m) {
    py::class_<fvdb::detail::viewer::SliderView>(
        m,
        "SliderView",
        "A handle to a slider widget in the editor's `Scene Params` window. "
        "Read or write `value` to query or change the live UI state.")
        .def_property_readonly(
            "name", &fvdb::detail::viewer::SliderView::getName, "Widget field name.")
        .def_property_readonly("scene_name",
                               &fvdb::detail::viewer::SliderView::getSceneName,
                               "Name of the scene this widget belongs to.")
        .def_property_readonly(
            "min", &fvdb::detail::viewer::SliderView::getMin, "Minimum slider value.")
        .def_property_readonly(
            "max", &fvdb::detail::viewer::SliderView::getMax, "Maximum slider value.")
        .def_property_readonly(
            "step", &fvdb::detail::viewer::SliderView::getStep, "Slider step size.")
        .def_property("value",
                      &fvdb::detail::viewer::SliderView::getValue,
                      &fvdb::detail::viewer::SliderView::setValue,
                      "Current slider value (float).");

    py::class_<fvdb::detail::viewer::NumberView>(
        m, "NumberView", "A handle to a numeric input field in the editor's `Scene Params` window.")
        .def_property_readonly(
            "name", &fvdb::detail::viewer::NumberView::getName, "Widget field name.")
        .def_property_readonly("scene_name",
                               &fvdb::detail::viewer::NumberView::getSceneName,
                               "Name of the scene this widget belongs to.")
        .def_property_readonly("has_min",
                               &fvdb::detail::viewer::NumberView::hasMin,
                               "Whether a minimum bound was supplied.")
        .def_property_readonly("has_max",
                               &fvdb::detail::viewer::NumberView::hasMax,
                               "Whether a maximum bound was supplied.")
        .def_property_readonly(
            "min", &fvdb::detail::viewer::NumberView::getMin, "Minimum value, if set.")
        .def_property_readonly(
            "max", &fvdb::detail::viewer::NumberView::getMax, "Maximum value, if set.")
        .def_property_readonly(
            "step", &fvdb::detail::viewer::NumberView::getStep, "Drag widget step size.")
        .def_property("value",
                      &fvdb::detail::viewer::NumberView::getValue,
                      &fvdb::detail::viewer::NumberView::setValue,
                      "Current numeric value (float).");

    py::class_<fvdb::detail::viewer::TextView>(
        m,
        "TextView",
        "A handle to a text input field in the editor's `Scene Params` window. "
        "The widget commits the value on every keystroke, so reads return the "
        "current contents of the buffer.")
        .def_property_readonly(
            "name", &fvdb::detail::viewer::TextView::getName, "Widget field name.")
        .def_property_readonly("scene_name",
                               &fvdb::detail::viewer::TextView::getSceneName,
                               "Name of the scene this widget belongs to.")
        .def_property_readonly("max_length",
                               &fvdb::detail::viewer::TextView::getMaxLength,
                               "Maximum string capacity in bytes (including the NUL terminator).")
        .def_property_readonly(
            "commit_on_enter",
            &fvdb::detail::viewer::TextView::getCommitOnEnter,
            "Whether the widget was created with commit_on_enter=True. When true, "
            "pressing Enter bumps a per-widget submit counter that callers poll via "
            "`submit_counter` to detect commit events.")
        .def_property_readonly(
            "submit_counter",
            &fvdb::detail::viewer::TextView::getSubmitCounter,
            "Current value of the editor-side submit counter (uint32). Increments "
            "every time the user presses Enter on the input. Always 0 when the "
            "widget was created with commit_on_enter=False.")
        .def_property("value",
                      &fvdb::detail::viewer::TextView::getValue,
                      &fvdb::detail::viewer::TextView::setValue,
                      "Current string value.");

    py::class_<fvdb::detail::viewer::CheckboxView>(
        m, "CheckboxView", "A handle to a checkbox in the editor's `Scene Params` window.")
        .def_property_readonly(
            "name", &fvdb::detail::viewer::CheckboxView::getName, "Widget field name.")
        .def_property_readonly("scene_name",
                               &fvdb::detail::viewer::CheckboxView::getSceneName,
                               "Name of the scene this widget belongs to.")
        .def_property("value",
                      &fvdb::detail::viewer::CheckboxView::getValue,
                      &fvdb::detail::viewer::CheckboxView::setValue,
                      "Current checkbox value (bool).");

    py::class_<fvdb::detail::viewer::CameraView>(
        m, "CameraView", "A view object for visualizing a camera in the editor")
        .def_property("visible",
                      &fvdb::detail::viewer::CameraView::getVisible,
                      &fvdb::detail::viewer::CameraView::setVisible,
                      "Whether the camera view is visible")
        .def_property_readonly(
            "name", &fvdb::detail::viewer::CameraView::getName, "The name of this camera view")
        .def_property("axis_length",
                      &fvdb::detail::viewer::CameraView::getAxisLength,
                      &fvdb::detail::viewer::CameraView::setAxisLength,
                      "The axis length for the gizmo")
        .def_property("axis_thickness",
                      &fvdb::detail::viewer::CameraView::getAxisThickness,
                      &fvdb::detail::viewer::CameraView::setAxisThickness,
                      "The axis thickness for the gizmo")
        .def_property("frustum_line_width",
                      &fvdb::detail::viewer::CameraView::getFrustumLineWidth,
                      &fvdb::detail::viewer::CameraView::setFrustumLineWidth,
                      "The line width of the frustum")
        .def_property("frustum_scale",
                      &fvdb::detail::viewer::CameraView::getFrustumScale,
                      &fvdb::detail::viewer::CameraView::setFrustumScale,
                      "The scale of the frustum visualization, default is 1.0")
        .def_property(
            "frustum_color",
            &fvdb::detail::viewer::CameraView::getFrustumColor,
            [](fvdb::detail::viewer::CameraView &self, const std::tuple<float, float, float> &rgb) {
                self.setFrustumColor(std::get<0>(rgb), std::get<1>(rgb), std::get<2>(rgb));
            },
            "The RGB color of the frustum as a 3-tuple");

    py::class_<fvdb::detail::viewer::GaussianSplat3dView>(
        m, "GaussianSplat3dView", "A view for displaying Gaussian splat 3D data in the viewer")
        .def_property("tile_size",
                      &fvdb::detail::viewer::GaussianSplat3dView::getTileSize,
                      &fvdb::detail::viewer::GaussianSplat3dView::setTileSize,
                      "The tile size for rendering this Gaussian scene.")
        .def_property(
            "min_radius_2d",
            &fvdb::detail::viewer::GaussianSplat3dView::getMinRadius2d,
            &fvdb::detail::viewer::GaussianSplat3dView::setMinRadius2d,
            "The minimum projected pixel radius below which Gaussians will not be rendered.")
        .def_property("eps_2d",
                      &fvdb::detail::viewer::GaussianSplat3dView::getEps2d,
                      &fvdb::detail::viewer::GaussianSplat3dView::setEps2d,
                      "The 2D epsilon value for this Gaussian scene.")
        .def_property("antialias",
                      &fvdb::detail::viewer::GaussianSplat3dView::getAntialias,
                      &fvdb::detail::viewer::GaussianSplat3dView::setAntialias,
                      "Whether to enable antialiasing for this Gaussian scene.")
        .def_property(
            "sh_degree_to_use",
            &fvdb::detail::viewer::GaussianSplat3dView::getShDegreeToUse,
            &fvdb::detail::viewer::GaussianSplat3dView::setShDegreeToUse,
            "The spherical harmonics degree used to render this Gaussian scene. A value of 0 means all available spherical harmonics are used.")
        .def_property("rgb_rgb_rgb_sh",
                      &fvdb::detail::viewer::GaussianSplat3dView::isShStrideRgbRgbRgb,
                      &fvdb::detail::viewer::GaussianSplat3dView::setShStrideRgbRgbRgb,
                      "Whether the spherical harmonics data is stored in RGBRGB... order.");

    py::class_<fvdb::detail::viewer::Viewer>(
        m, "Viewer", "A viewer for displaying 3D data including Gaussian splats")
        .def(py::init<const std::string &, const int, const int, const bool>(),
             py::arg("ip_address"),
             py::arg("port"),
             py::arg("device_id"),
             py::arg("verbose"),
             "Create a new Viewer instance")
        .def("add_gaussian_splat_3d_view",
             &fvdb::detail::viewer::Viewer::addGaussianSplat3dView,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("means"),
             py::arg("quats"),
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("sh0"),
             py::arg("shN"),
             py::return_value_policy::reference_internal, // preserve reference; tie lifetime to
                                                          // parent
             "Register a Gaussian splat 3D view with the viewer (accepts raw tensors)")
        .def("has_gaussian_splat_3d_view",
             &fvdb::detail::viewer::Viewer::hasGaussianSplat3dView,
             py::arg("name"),
             "Check if a Gaussian splat 3D view with the given name exists")
        .def("get_gaussian_splat_3d_view",
             &fvdb::detail::viewer::Viewer::getGaussianSplat3dView,
             py::arg("name"),
             py::return_value_policy::reference_internal,
             "Get a Gaussian splat 3D view by name")

        .def("ip_address",
             &fvdb::detail::viewer::Viewer::ipAddress,
             "The IP address the viewer server is listening on.")

        .def("port",
             &fvdb::detail::viewer::Viewer::port,
             "The port the viewer server is listening on.")

        .def("reset", &fvdb::detail::viewer::Viewer::reset, "Reset the viewer server state")

        .def("stop",
             &fvdb::detail::viewer::Viewer::stop,
             "Stop the viewer server and join the editor render thread.")

        .def("add_scene",
             &fvdb::detail::viewer::Viewer::addScene,
             py::arg("scene_name"),
             "Add a new scene to the viewer")

        .def("remove_scene",
             &fvdb::detail::viewer::Viewer::removeScene,
             py::arg("scene_name"),
             "Remove a scene from the viewer")

        .def("remove_view",
             &fvdb::detail::viewer::Viewer::removeView,
             py::arg("scene_name"),
             py::arg("name"),
             "Remove a view from a scene")

        .def("camera_orbit_center",
             &fvdb::detail::viewer::Viewer::cameraOrbitCenter,
             py::arg("scene_name"),
             "Get the point about which the camera orbits")
        .def("set_camera_orbit_center",
             &fvdb::detail::viewer::Viewer::setCameraOrbitCenter,
             py::arg("scene_name"),
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             "Set the camera orbit center")

        .def("camera_orbit_radius",
             &fvdb::detail::viewer::Viewer::cameraOrbitRadius,
             py::arg("scene_name"),
             "Get the camera orbit radius")
        .def("set_camera_orbit_radius",
             &fvdb::detail::viewer::Viewer::setCameraOrbitRadius,
             py::arg("scene_name"),
             py::arg("radius"),
             "Set the camera orbit radius (must be positive)")

        .def("camera_up_direction",
             &fvdb::detail::viewer::Viewer::cameraUpDirection,
             py::arg("scene_name"),
             "Get the camera up vector")
        .def("set_camera_up_direction",
             &fvdb::detail::viewer::Viewer::setCameraUpDirection,
             py::arg("scene_name"),
             py::arg("ux"),
             py::arg("uy"),
             py::arg("uz"),
             "Set the camera up vector")

        .def("camera_view_direction",
             &fvdb::detail::viewer::Viewer::cameraViewDirection,
             py::arg("scene_name"),
             "Get the camera view direction")
        .def("set_camera_view_direction",
             &fvdb::detail::viewer::Viewer::setCameraViewDirection,
             py::arg("scene_name"),
             py::arg("dx"),
             py::arg("dy"),
             py::arg("dz"),
             "Set the camera view direction")

        .def("camera_fov",
             &fvdb::detail::viewer::Viewer::cameraFov,
             py::arg("scene_name"),
             "Get the camera vertical field of view in radians")
        .def("set_camera_fov",
             &fvdb::detail::viewer::Viewer::setCameraFov,
             py::arg("scene_name"),
             py::arg("fov_radians"),
             "Set the camera vertical field of view in radians")

        .def("camera_near",
             &fvdb::detail::viewer::Viewer::cameraNear,
             py::arg("scene_name"),
             "Get the camera near clipping plane")
        .def("set_camera_near",
             &fvdb::detail::viewer::Viewer::setCameraNear,
             py::arg("scene_name"),
             py::arg("near"),
             "Set the camera near clipping plane")

        .def("camera_far",
             &fvdb::detail::viewer::Viewer::cameraFar,
             py::arg("scene_name"),
             "Get the camera far clipping plane")
        .def("set_camera_far",
             &fvdb::detail::viewer::Viewer::setCameraFar,
             py::arg("scene_name"),
             py::arg("far"),
             "Set the camera far clipping plane")

        .def("camera_model",
             &fvdb::detail::viewer::Viewer::cameraModel,
             py::arg("scene_name"),
             "The viewer camera model (currently pinhole or orthographic)")
        .def(
            "set_camera_model",
            [](fvdb::detail::viewer::Viewer &viewer,
               const std::string &sceneName,
               fvdb::detail::ops::DistortionModel model) {
                if (model != fvdb::detail::ops::DistortionModel::PINHOLE &&
                    model != fvdb::detail::ops::DistortionModel::ORTHOGRAPHIC) {
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "Viewer currently only supports DistortionModel.PINHOLE and "
                                    "DistortionModel.ORTHOGRAPHIC");
                    throw py::error_already_set();
                }
                viewer.setCameraModel(sceneName, model);
            },
            py::arg("scene_name"),
            py::arg("model"),
            "Set the viewer camera model (currently pinhole or orthographic)")
        .def("add_camera_view",
             py::overload_cast<const std::string &,
                               const std::string &,
                               const torch::Tensor &,
                               const torch::Tensor &,
                               const torch::Tensor &,
                               float,
                               float,
                               float,
                               float,
                               float,
                               float,
                               const std::tuple<float, float, float> &,
                               bool>(&fvdb::detail::viewer::Viewer::addCameraView),
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("camera_to_world_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_sizes"),
             py::arg("frustum_near_plane"),
             py::arg("frustum_far_plane"),
             py::arg("axis_length"),
             py::arg("axis_thickness"),
             py::arg("frustum_line_width"),
             py::arg("frustum_scale"),
             py::arg("frustum_color"),
             py::arg("visible"),
             py::return_value_policy::reference_internal,
             "Add a named camera view from camera/world and projection matrices")
        .def("has_camera_view",
             &fvdb::detail::viewer::Viewer::hasCameraView,
             py::arg("name"),
             "Check if a camera view with the given name exists")
        .def("get_camera_view",
             &fvdb::detail::viewer::Viewer::getCameraView,
             py::arg("name"),
             py::return_value_policy::reference_internal,
             "Get a camera view by name")
        .def("add_image",
             &fvdb::detail::viewer::Viewer::addImage,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("rgba_image"),
             py::arg("width"),
             py::arg("height"),
             "Add an RGBA8 image as a 2D NanoVDB grid to the viewer. "
             "The rgba_image should be a 1D uint8 tensor of size width * height * 4 "
             "containing packed RGBA values.")
        .def("add_level_set_view",
             &fvdb::detail::viewer::Viewer::addLevelSetView,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("grid"),
             py::arg("sdf"),
             "Add a single grid with per-voxel float32 SDF values as a level-set isosurface "
             "(rendered by the nanovdb-editor surface pipeline). The grid must contain exactly "
             "one grid.")
        .def("add_fog_volume_view",
             &fvdb::detail::viewer::Viewer::addFogVolumeView,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("grid"),
             py::arg("density"),
             "Add a single grid with per-voxel float32 density values as a fog volume "
             "(rendered by the nanovdb-editor render pipeline). The grid must contain exactly "
             "one grid.")
        .def("has_nanovdb_view",
             &fvdb::detail::viewer::Viewer::hasNanoVDBView,
             py::arg("name"),
             "Check if a NanoVDB grid view with the given name exists.")
        .def("wait_for_interrupt",
             &fvdb::detail::viewer::Viewer::waitForInteerrupt,
             "Block until the viewer is interrupted by the user (Ctrl-C or closing the window)")

        .def("add_slider",
             &fvdb::detail::viewer::Viewer::addSlider,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("min"),
             py::arg("max"),
             py::arg("initial"),
             py::arg("step"),
             "Register a float slider widget in the editor's `Scene Params` window for the "
             "given scene. Returns a SliderView handle whose `value` property reads or writes "
             "the current widget value.")

        .def("add_number",
             &fvdb::detail::viewer::Viewer::addNumber,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("initial"),
             py::arg("has_min"),
             py::arg("min"),
             py::arg("has_max"),
             py::arg("max"),
             py::arg("step"),
             "Register a float numeric drag widget in the editor's `Scene Params` window for "
             "the given scene. Returns a NumberView handle whose `value` property reads or "
             "writes the current widget value.")

        .def("add_text",
             &fvdb::detail::viewer::Viewer::addText,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("initial"),
             py::arg("max_length"),
             py::arg("commit_on_enter") = false,
             "Register a text input widget in the editor's `Scene Params` window for the "
             "given scene. Returns a TextView handle whose `value` property reads or writes "
             "the current string contents. By default the widget commits on every keystroke; "
             "pass commit_on_enter=True to additionally have the editor increment a hidden "
             "submit counter when the user presses Enter, so callers can fire `on_submit` "
             "callbacks separate from per-keystroke `on_update` events.")

        .def("add_checkbox",
             &fvdb::detail::viewer::Viewer::addCheckbox,
             py::arg("scene_name"),
             py::arg("name"),
             py::arg("initial"),
             "Register a checkbox widget in the editor's `Scene Params` window for the given "
             "scene. Returns a CheckboxView handle whose `value` property reads or writes "
             "the current bool value.")

        .def("has_slider",
             &fvdb::detail::viewer::Viewer::hasSlider,
             py::arg("name"),
             "Check if a slider widget with the given name exists.")
        .def("has_number",
             &fvdb::detail::viewer::Viewer::hasNumber,
             py::arg("name"),
             "Check if a numeric widget with the given name exists.")
        .def("has_text",
             &fvdb::detail::viewer::Viewer::hasText,
             py::arg("name"),
             "Check if a text widget with the given name exists.")
        .def("has_checkbox",
             &fvdb::detail::viewer::Viewer::hasCheckbox,
             py::arg("name"),
             "Check if a checkbox widget with the given name exists.")

        .def("get_slider",
             &fvdb::detail::viewer::Viewer::getSlider,
             py::arg("name"),
             "Get a snapshot of the slider widget with the given name.")
        .def("get_number",
             &fvdb::detail::viewer::Viewer::getNumber,
             py::arg("name"),
             "Get a snapshot of the numeric widget with the given name.")
        .def("get_text",
             &fvdb::detail::viewer::Viewer::getText,
             py::arg("name"),
             "Get a snapshot of the text widget with the given name.")
        .def("get_checkbox",
             &fvdb::detail::viewer::Viewer::getCheckbox,
             py::arg("name"),
             "Get a snapshot of the checkbox widget with the given name.")

        .def("scene_widget_names",
             &fvdb::detail::viewer::Viewer::sceneWidgetNames,
             py::arg("scene_name"),
             "Return the names of all widgets currently registered on the given scene, in "
             "registration order.");
}
