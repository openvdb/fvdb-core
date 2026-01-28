// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <pybind11/cast.h>

#include "TypeCasters.h"

#include <fvdb/FVDB.h>
#include <fvdb/GaussianSplat3d.h>
#include <fvdb/detail/autograd/EvaluateSphericalHarmonics.h>

#include <torch/extension.h>

void
bind_gaussian_splat3d(py::module &m) {
    py::class_<fvdb::GaussianSplat3d::ProjectedGaussianSplats>(m, "ProjectedGaussianSplats")
        .def_property_readonly("means2d", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::means2d)
        .def_property_readonly("conics", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::conics)
        .def_property_readonly("render_quantities",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::renderQuantities)
        .def_property_readonly("depths", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::depths)
        .def_property_readonly("opacities",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::opacities)
        .def_property_readonly("radii", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::radii)
        .def_property_readonly("tile_offsets",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::offsets)
        .def_property_readonly("tile_gaussian_ids",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::gaussianIds)
        .def_property_readonly("image_width",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::imageWidth)
        .def_property_readonly("image_height",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::imageHeight)
        .def_property_readonly("near_plane",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::nearPlane)
        .def_property_readonly("far_plane",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::farPlane)
        .def_property_readonly("projection_type",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::projectionType)
        .def_property_readonly("sh_degree_to_use",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::shDegreeToUse)
        .def_property_readonly("min_radius_2d",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::minRadius2d)
        .def_property_readonly("eps_2d", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::eps2d)
        .def_property_readonly("antialias",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::antialias);

    py::class_<fvdb::GaussianSplat3d> gs3d(m, "GaussianSplat3d", "A gaussian splat scene");

    py::enum_<fvdb::GaussianSplat3d::ProjectionType>(gs3d, "ProjectionType")
        .value("PERSPECTIVE", fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE)
        .value("ORTHOGRAPHIC", fvdb::GaussianSplat3d::ProjectionType::ORTHOGRAPHIC)
        .export_values();

    gs3d.def(py::init<torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      bool,
                      bool,
                      bool>(),
             py::arg("means"),
             py::arg("quats"),
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("sh0"),
             py::arg("shN"),
             py::arg("accumulate_mean_2d_gradients"),
             py::arg("accumulate_max_2d_radii"),
             py::arg("detach"))
        .def_property_readonly("device", &fvdb::GaussianSplat3d::device)
        .def_property_readonly("dtype", &fvdb::GaussianSplat3d::scalarType)
        .def_property_readonly("sh_degree", &fvdb::GaussianSplat3d::shDegree)
        .def_property("means", &fvdb::GaussianSplat3d::means, &fvdb::GaussianSplat3d::setMeans)
        .def_property("quats", &fvdb::GaussianSplat3d::quats, &fvdb::GaussianSplat3d::setQuats)
        .def_property_readonly("scales", &fvdb::GaussianSplat3d::scales)
        .def_property(
            "log_scales", &fvdb::GaussianSplat3d::logScales, &fvdb::GaussianSplat3d::setLogScales)
        .def_property_readonly("opacities", &fvdb::GaussianSplat3d::opacities)
        .def_property("logit_opacities",
                      &fvdb::GaussianSplat3d::logitOpacities,
                      &fvdb::GaussianSplat3d::setLogitOpacities)
        .def_property("sh0", &fvdb::GaussianSplat3d::sh0, &fvdb::GaussianSplat3d::setSh0)
        .def_property("shN", &fvdb::GaussianSplat3d::shN, &fvdb::GaussianSplat3d::setShN)
        .def_property_readonly("num_gaussians", &fvdb::GaussianSplat3d::numGaussians)
        .def_property_readonly("num_sh_bases", &fvdb::GaussianSplat3d::numShBases)
        .def_property_readonly("num_channels", &fvdb::GaussianSplat3d::numChannels)
        .def_property_readonly("requires_grad", &fvdb::GaussianSplat3d::requiresGrad)
        .def_property("accumulate_max_2d_radii",
                      &fvdb::GaussianSplat3d::accumulateMax2dRadii,
                      &fvdb::GaussianSplat3d::setAccumulateMax2dRadii)
        .def_property("accumulate_mean_2d_gradients",
                      &fvdb::GaussianSplat3d::accumulateMean2dGradients,
                      &fvdb::GaussianSplat3d::setAccumulateMean2dGradients)
        .def_property_readonly("accumulated_mean_2d_gradient_norms",
                               &fvdb::GaussianSplat3d::accumulated2dMeansGradientNormsForGrad)
        .def_property_readonly("accumulated_max_2d_radii",
                               &fvdb::GaussianSplat3d::accumulatedMax2dRadiiForGrad)
        .def_property_readonly("accumulated_gradient_step_counts",
                               &fvdb::GaussianSplat3d::gradientStepCountsForGrad)
        .def_static(
            "from_state_dict",
            [](const std::unordered_map<std::string, torch::Tensor> &stateDict) {
                return fvdb::GaussianSplat3d(stateDict);
            },
            py::arg("state_dict"))
        .def("to", &fvdb::GaussianSplat3d::to, py::arg("device"), py::arg("dtype"))
        .def("detach", &fvdb::GaussianSplat3d::detach)
        .def("detach_in_place", &fvdb::GaussianSplat3d::detachInPlace)
        .def("state_dict", &fvdb::GaussianSplat3d::stateDict)
        .def("load_state_dict", &fvdb::GaussianSplat3d::loadStateDict, py::arg("state_dict"))
        .def_property("requires_grad",
                      &fvdb::GaussianSplat3d::requiresGrad,
                      &fvdb::GaussianSplat3d::setRequiresGrad)
        .def("set_state",
             &fvdb::GaussianSplat3d::setState,
             py::arg("means"),
             py::arg("quats"),
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("sh0"),
             py::arg("shN"))
        .def("save_ply", &fvdb::GaussianSplat3d::savePly, py::arg("filename"), py::arg("metadata"))
        .def_static("from_ply",
                    &fvdb::GaussianSplat3d::fromPly,
                    py::arg("filename"),
                    py::arg("device") = torch::kCPU)
        .def_static("cat",
                    &fvdb::GaussianSplat3d::cat,
                    py::arg("splats_to_cat"),
                    py::arg("accumulate_mean_2d_gradients") = false,
                    py::arg("accumulate_max_2d_radii")      = false,
                    py::arg("detach")                       = false)
        .def("reset_accumulated_gradient_state",
             &fvdb::GaussianSplat3d::resetAccumulatedGradientState)
        .def("project_gaussians_for_images",
             &fvdb::GaussianSplat3d::projectGaussiansForImages,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("project_gaussians_for_depths",
             &fvdb::GaussianSplat3d::projectGaussiansForDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("project_gaussians_for_images_and_depths",
             &fvdb::GaussianSplat3d::projectGaussiansForImagesAndDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("render_from_projected_gaussians",
             &fvdb::GaussianSplat3d::renderFromProjectedGaussians,
             py::arg("projected_gaussians"),
             py::arg("crop_width")    = -1,
             py::arg("crop_height")   = -1,
             py::arg("crop_origin_w") = -1,
             py::arg("crop_origin_h") = -1,
             py::arg("tile_size")     = 16,
             py::arg("backgrounds")   = std::nullopt)

        .def("render_images",
             &fvdb::GaussianSplat3d::renderImages,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false,
             py::arg("backgrounds")      = std::nullopt)

        .def("render_depths",
             &fvdb::GaussianSplat3d::renderDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")       = 16,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false,
             py::arg("backgrounds")     = std::nullopt)

        .def("render_images_and_depths",
             &fvdb::GaussianSplat3d::renderImagesAndDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false,
             py::arg("backgrounds")      = std::nullopt)

        .def("sparse_render_images",
             &fvdb::GaussianSplat3d::sparseRenderImages,
             py::arg("pixels_to_render"),
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("sparse_render_depths",
             &fvdb::GaussianSplat3d::sparseRenderDepths,
             py::arg("pixels_to_render"),
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")       = 16,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("sparse_render_images_and_depths",
             &fvdb::GaussianSplat3d::sparseRenderImagesAndDepths,
             py::arg("pixels_to_render"),
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("render_num_contributing_gaussians",
             &fvdb::GaussianSplat3d::renderNumContributingGaussians,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")       = 16,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("sparse_render_num_contributing_gaussians",
             &fvdb::GaussianSplat3d::sparseRenderNumContributingGaussians,
             py::arg("pixels_to_render"),
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")       = 16,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("render_contributing_gaussian_ids",
             &fvdb::GaussianSplat3d::renderContributingGaussianIds,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")    = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")          = 16,
             py::arg("min_radius_2d")      = 0.0,
             py::arg("eps_2d")             = 0.3,
             py::arg("antialias")          = false,
             py::arg("top_k_contributors") = 0)

        .def("sparse_render_contributing_gaussian_ids",
             &fvdb::GaussianSplat3d::sparseRenderContributingGaussianIds,
             py::arg("pixels_to_render"),
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")    = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")          = 16,
             py::arg("min_radius_2d")      = 0.0,
             py::arg("eps_2d")             = 0.3,
             py::arg("antialias")          = false,
             py::arg("top_k_contributors") = 0)

        .def("relocate_gaussians",
             &fvdb::GaussianSplat3d::relocateGaussians,
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("ratios"),
             py::arg("binomial_coeffs"),
             py::arg("n_max"),
             py::arg("min_opacity"))

        .def("add_noise_to_means",
             &fvdb::GaussianSplat3d::addNoiseToMeans,
             py::arg("noise_scale"),
             py::arg("t") = 0.005,
             py::arg("k") = 100.0)

        .def("index_select", &fvdb::GaussianSplat3d::indexSelect, py::arg("indices"))
        .def("mask_select", &fvdb::GaussianSplat3d::maskSelect, py::arg("mask"))
        .def("slice_select",
             &fvdb::GaussianSplat3d::sliceSelect,
             py::arg("begin"),
             py::arg("end"),
             py::arg("step"))
        .def("index_set", &fvdb::GaussianSplat3d::indexSet, py::arg("indices"), py::arg("value"))
        .def("mask_set", &fvdb::GaussianSplat3d::maskSet, py::arg("mask"), py::arg("value"))
        .def("slice_set",
             &fvdb::GaussianSplat3d::sliceSet,
             py::arg("begin"),
             py::arg("end"),
             py::arg("step"),
             py::arg("value"));

    m.def("gaussian_render_jagged",
          &fvdb::gaussianRenderJagged,
          py::arg("means"),
          py::arg("quats"),
          py::arg("scales"),
          py::arg("opacities"),
          py::arg("sh_coeffs"),
          py::arg("viewmats"),
          py::arg("Ks"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("near_plane")           = 0.01,
          py::arg("far_plane")            = 1e10,
          py::arg("sh_degree_to_use")     = 3,
          py::arg("tile_size")            = 16,
          py::arg("radius_clip")          = 0.0,
          py::arg("eps2d")                = 0.3,
          py::arg("antialias")            = false,
          py::arg("render_depth_channel") = false,
          py::arg("return_debug_info")    = false,
          py::arg("render_depth_only")    = false,
          py::arg("ortho")                = false,
          py::arg("backgrounds")          = std::nullopt);

    m.def(
        "evaluate_spherical_harmonics",
        [](int64_t shDegree,
           int64_t numCameras,
           const torch::Tensor &sh0,
           const std::optional<torch::Tensor> &shN,
           const std::optional<torch::Tensor> &viewDirections,
           const std::optional<torch::Tensor> &radii) {
            return fvdb::detail::autograd::EvaluateSphericalHarmonics::apply(
                shDegree, numCameras, viewDirections, sh0, shN, radii)[0];
        },
        R"doc(
Evaluate spherical harmonics to compute view-dependent features/colors.

This function evaluates spherical harmonics (SH) coefficients to compute
features (typically RGB colors) for a set of points, optionally considering
view directions for view-dependent appearance.

Args:
    sh_degree: Degree of spherical harmonics to use (0-3 typically).
               Degree 0 uses only sh0 (view-independent).
               Higher degrees require view_directions and shN.
    num_cameras: Number of camera views (C). The output will have shape [C, N, D].
    sh0: DC term coefficients with shape [N, 1, D] where N is the number of
         points and D is the number of feature channels.
    shN: Higher-order SH coefficients with shape [N, K-1, D] where
         K = (sh_degree+1)^2. Required when sh_degree > 0. Pass None for degree 0.
    view_directions: Unnormalized view directions with shape [C, N, 3].
                     Required when sh_degree > 0. Pass None for degree 0.
    radii: Optional projected radii with shape [C, N] (int32). When provided,
           points with radii <= 0 will output zeros (used to skip invisible
           gaussians as an optimization). Pass None to evaluate all points.

Returns:
    Tensor of shape [C, N, D] containing the evaluated features/colors.
)doc",
        py::arg("sh_degree"),
        py::arg("num_cameras"),
        py::arg("sh0"),
        py::arg("shN")             = std::nullopt,
        py::arg("view_directions") = std::nullopt,
        py::arg("radii")           = std::nullopt);
}
