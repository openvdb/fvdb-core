// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "TypeCasters.h"

#include <fvdb/GaussianSplat3d.h>
#include <fvdb/detail/viewer/CameraView.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>
#include <fvdb/detail/viewer/Viewer.h>

#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

void
bind_viewer(py::module &m) {
    py::class_<fvdb::detail::viewer::CameraView>(
        m, "CameraView", "A view object for visualizing a camera in the editor")
        .def_property("visible",
                      &fvdb::detail::viewer::CameraView::getIsVisible,
                      &fvdb::detail::viewer::CameraView::setIsVisible,
                      "Whether the camera view is visible")
        .def_property("axis_length",
                      &fvdb::detail::viewer::CameraView::getAxisLength,
                      &fvdb::detail::viewer::CameraView::setAxisLength,
                      "The axis length for the gizmo")
        .def_property("axis_thickness",
                      &fvdb::detail::viewer::CameraView::getAxisThickness,
                      &fvdb::detail::viewer::CameraView::setAxisThickness,
                      "The axis thickness for the gizmo")
        .def_property("axis_scale",
                      &fvdb::detail::viewer::CameraView::getAxisScale,
                      &fvdb::detail::viewer::CameraView::setAxisScale,
                      "The axis scale for the gizmo")
        .def_property("frustum_line_width",
                      &fvdb::detail::viewer::CameraView::getFrustumLineWidth,
                      &fvdb::detail::viewer::CameraView::setFrustumLineWidth,
                      "The line width of the frustum")
        .def_property("frustum_scale",
                      &fvdb::detail::viewer::CameraView::getFrustumScale,
                      &fvdb::detail::viewer::CameraView::setFrustumScale,
                      "The scale of the frustum visualization")
        .def("get_position", &fvdb::detail::viewer::CameraView::getPosition)
        .def("set_position", &fvdb::detail::viewer::CameraView::setPosition, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("get_eye_direction", &fvdb::detail::viewer::CameraView::getEyeDirection)
        .def("set_eye_direction", &fvdb::detail::viewer::CameraView::setEyeDirection, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("get_eye_up", &fvdb::detail::viewer::CameraView::getEyeUp)
        .def("set_eye_up", &fvdb::detail::viewer::CameraView::setEyeUp, py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property("near_plane",
                      &fvdb::detail::viewer::CameraView::getNearPlane,
                      &fvdb::detail::viewer::CameraView::setNearPlane)
        .def_property("far_plane",
                      &fvdb::detail::viewer::CameraView::getFarPlane,
                      &fvdb::detail::viewer::CameraView::setFarPlane)
        .def_property("fov_angle_y",
                      &fvdb::detail::viewer::CameraView::getFovAngleY,
                      &fvdb::detail::viewer::CameraView::setFovAngleY)
        .def_property("is_orthographic",
                      &fvdb::detail::viewer::CameraView::getIsOrthographicCam,
                      &fvdb::detail::viewer::CameraView::setIsOrthographicCam);

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
        .def_property("near",
                      &fvdb::detail::viewer::GaussianSplat3dView::getNear,
                      &fvdb::detail::viewer::GaussianSplat3dView::setNear,
                      "The near clipping plane for this Gaussian scene.")
        .def_property("far",
                      &fvdb::detail::viewer::GaussianSplat3dView::getFar,
                      &fvdb::detail::viewer::GaussianSplat3dView::setFar,
                      "The far clipping plane for this Gaussian scene.");

    py::class_<fvdb::detail::viewer::Viewer>(
        m, "Viewer", "A viewer for displaying 3D data including Gaussian splats")
        .def(py::init<const std::string &, const int, const bool>(),
             py::arg("ip_address"),
             py::arg("port"),
             py::arg("verbose"),
             "Create a new Viewer instance")
        .def(
            "add_gaussian_splat_3d",
            &fvdb::detail::viewer::Viewer::addGaussianSplat3d,
            py::arg("name"),
            py::arg("gaussian_splat_3d"),
            py::return_value_policy::reference_internal, // preserve reference; tie lifetime to
                                                         // parent
            "Register a Gaussian splat 3D view with the viewer (accepts Python or C++ GaussianSplat3d)")

        .def("camera_near",
             &fvdb::detail::viewer::Viewer::cameraNear,
             "Get the camera near clipping plane")
        .def("set_camera_near",
             &fvdb::detail::viewer::Viewer::setCameraNear,
             py::arg("near"),
             "Set the camera near clipping plane")

        .def("camera_projection_type",
             &fvdb::detail::viewer::Viewer::cameraProjectionType,
             "The camera mode (perspective or orthographic)")
        .def("set_camera_projection_type",
             &fvdb::detail::viewer::Viewer::setCameraProjectionType,
             py::arg("mode"),
             "Set the camera mode (perspective or orthographic)")
        .def("add_camera_view",
             &fvdb::detail::viewer::Viewer::addCameraView,
             py::arg("name"),
             py::return_value_policy::reference_internal,
             "Add a camera view visualization to the editor")
        .def("add_camera_view",
             py::overload_cast<const std::string &, const torch::Tensor &, const torch::Tensor &>(
                 &fvdb::detail::viewer::Viewer::addCameraView),
             py::arg("name"),
             py::arg("camera_to_world_matrices"),
             py::arg("projection_matrices"),
             py::return_value_policy::reference_internal,
             "Add a named camera view from camera/world and projection matrices");
}
