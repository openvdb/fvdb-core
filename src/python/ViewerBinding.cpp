// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GaussianSplat3d.h>
#include <fvdb/detail/viewer/GaussianSplat3dView.h>
#include <fvdb/detail/viewer/Viewer.h>

#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

void
bind_viewer(py::module &m) {
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
                      "The far clipping plane for this Gaussian scene.")
        .def("__repr__", [](const fvdb::detail::viewer::GaussianSplat3dView &self) {
            return "GaussianSplat3dView()";
        });

    py::class_<fvdb::detail::viewer::Viewer>(
        m, "Viewer", "A viewer for displaying 3D data including Gaussian splats")
        .def(py::init<const std::string &, const int, const bool>(),
             py::arg("ip_address") = "127.0.0.1",
             py::arg("port")       = 8080,
             py::arg("verbose")    = false,
             "Create a new Viewer instance")
        .def(
            "register_gaussian_splat_3d",
            [](fvdb::detail::viewer::Viewer &self,
               const std::string &name,
               py::object py_splat) -> decltype(auto) {
                auto cpp_splat = py_splat.attr("_impl").cast<fvdb::GaussianSplat3d>();
                return self.registerGaussianSplat3dView(name, cpp_splat);
            },
            py::arg("name"),
            py::arg("gaussian_scene"),
            py::return_value_policy::reference_internal, // preserve reference; tie lifetime to
                                                         // parent
            "Register a Gaussian splat 3D view with the viewer (accepts Python or C++ GaussianSplat3d)")
        .def("start_server", &fvdb::detail::viewer::Viewer::startServer, "Start the viewer")
        .def("stop_server", &fvdb::detail::viewer::Viewer::stopServer, "Stop the viewer")
        .def_property(
            "camera_position",
            &fvdb::detail::viewer::Viewer::getCameraPosition,
            [](fvdb::detail::viewer::Viewer &self, std::tuple<float, float, float> pos) {
                self.setCameraPosition(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
            },
            "The camera position as a tuple (x, y, z)")
        .def_property(
            "camera_lookat",
            &fvdb::detail::viewer::Viewer::getCameraLookat,
            [](fvdb::detail::viewer::Viewer &self, std::tuple<float, float, float> lookat) {
                self.setCameraLookat(std::get<0>(lookat), std::get<1>(lookat), std::get<2>(lookat));
            },
            "The camera look-at point as a tuple (x, y, z)")
        .def_property(
            "camera_position",
            &fvdb::detail::viewer::Viewer::getCameraPosition,
            [](fvdb::detail::viewer::Viewer &self, std::tuple<float, float, float> pos) {
                self.setCameraPosition(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
            },
            "The camera position as a tuple (x, y, z)")
        .def_property("camera_near",
                      &fvdb::detail::viewer::Viewer::getCameraNear,
                      &fvdb::detail::viewer::Viewer::setCameraNear,
                      "The camera near clipping plane")
        .def_property("camera_far",
                      &fvdb::detail::viewer::Viewer::getCameraFar,
                      &fvdb::detail::viewer::Viewer::setCameraFar,
                      "The camera far clipping plane")
        .def("camera_pose",
             &fvdb::detail::viewer::Viewer::setCameraPose,
             py::arg("camera_to_world_matrix"),
             "Set the camera pose from a 4x4 matrix")
        .def_property(
            "camera_eye_direction",
            &fvdb::detail::viewer::Viewer::getCameraEyeDirection,
            [](fvdb::detail::viewer::Viewer &self, std::tuple<float, float, float> dir) {
                self.setCameraEyeDirection(std::get<0>(dir), std::get<1>(dir), std::get<2>(dir));
            },
            "The camera eye direction vector")
        .def_property(
            "camera_eye_up",
            &fvdb::detail::viewer::Viewer::getCameraEyeUp,
            [](fvdb::detail::viewer::Viewer &self, std::tuple<float, float, float> up) {
                self.setCameraEyeUp(std::get<0>(up), std::get<1>(up), std::get<2>(up));
            },
            "The camera eye up vector")
        .def_property("camera_eye_distance_from_position",
                      &fvdb::detail::viewer::Viewer::getCameraEyeDistanceFromPosition,
                      &fvdb::detail::viewer::Viewer::setCameraEyeDistanceFromPosition,
                      "The camera eye distance from position")
        .def_property(
            "camera_mode",
            &fvdb::detail::viewer::Viewer::getCameraMode,
            [](fvdb::detail::viewer::Viewer &self, fvdb::GaussianSplat3d::ProjectionType mode) {
                self.setCameraMode(mode);
            },
            "The camera mode (perspective or orthographic)")
        .def("__repr__",
             [](const fvdb::detail::viewer::Viewer &self) { return "<fvdb.viz.Viewer>"; });
}
