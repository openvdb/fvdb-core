// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Pybind11 bindings for Gaussian splat free-function ops.
// These expose the fvdb::detail::ops functions from GaussianSplatOps.h
// as module-level functions on _fvdb_cpp, enabling the Python functional layer.
//
// Design note on accumulator mutability:
// The C++ projection backward kernel mutates three accumulator tensors in-place
// (gradient norms, max 2D radii, step counts) via atomicAdd. These support
// Gaussian densification (split/clone/prune decisions during training).
// The C++ free functions in GaussianSplatOps.h accept these as mutable
// std::optional<torch::Tensor>& refs. Here, we wrap those functions in lambdas
// that pass accumulateMean2dGradients=false and accumulateMax2dRadii=false,
// hiding the mutability and presenting a strictly functional interface to Python.
// When autograd moves to Python (Milestones 5-7), the Python GaussianSplat3d
// class will own these accumulators and pass them to the C++ backward dispatch.

#include <pybind11/stl.h>

#include <fvdb/detail/ops/gsplat/GaussianSplatOps.h>

#include <torch/extension.h>

void
bind_gaussian_splat_ops(py::module &m) {
    namespace ops = fvdb::detail::ops;
    using RenderSettings  = fvdb::detail::ops::RenderSettings;
    using DistortionModel = fvdb::detail::ops::DistortionModel;
    using ProjectionMethod = fvdb::detail::ops::ProjectionMethod;

    // -----------------------------------------------------------------------
    // Data types needed by the functional ops
    // -----------------------------------------------------------------------

    py::class_<RenderSettings>(m, "RenderSettings")
        .def(py::init<>())
        .def_readwrite("image_width", &RenderSettings::imageWidth)
        .def_readwrite("image_height", &RenderSettings::imageHeight)
        .def_readwrite("image_origin_w", &RenderSettings::imageOriginW)
        .def_readwrite("image_origin_h", &RenderSettings::imageOriginH)
        .def_readwrite("near_plane", &RenderSettings::nearPlane)
        .def_readwrite("far_plane", &RenderSettings::farPlane)
        .def_readwrite("tile_size", &RenderSettings::tileSize)
        .def_readwrite("radius_clip", &RenderSettings::radiusClip)
        .def_readwrite("eps_2d", &RenderSettings::eps2d)
        .def_readwrite("antialias", &RenderSettings::antialias)
        .def_readwrite("sh_degree_to_use", &RenderSettings::shDegreeToUse)
        .def_readwrite("num_depth_samples", &RenderSettings::numDepthSamples)
        .def_readwrite("render_mode", &RenderSettings::renderMode);

    py::enum_<RenderSettings::RenderMode>(m, "RenderMode")
        .value("RGB", RenderSettings::RenderMode::RGB)
        .value("DEPTH", RenderSettings::RenderMode::DEPTH)
        .value("RGBD", RenderSettings::RenderMode::RGBD)
        .export_values();

    py::class_<fvdb::GaussianSplat3d::SparseProjectedGaussianSplats,
               fvdb::GaussianSplat3d::ProjectedGaussianSplats>(m, "SparseProjectedGaussianSplats")
        .def_readonly("active_tiles",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::activeTiles)
        .def_readonly("active_tile_mask",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::activeTileMask)
        .def_readonly("tile_pixel_mask",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::tilePixelMask)
        .def_readonly("tile_pixel_cumsum",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::tilePixelCumsum)
        .def_readonly("pixel_map",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::pixelMap)
        .def_readonly("inverse_indices",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::inverseIndices)
        .def_readonly("has_duplicates",
                      &fvdb::GaussianSplat3d::SparseProjectedGaussianSplats::hasDuplicates);

    // -----------------------------------------------------------------------
    // Validation & utility
    // -----------------------------------------------------------------------

    m.def("gsplat_check_state",
          &ops::checkGaussianState,
          py::arg("means"),
          py::arg("quats"),
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("sh0"),
          py::arg("shN"));

    // -----------------------------------------------------------------------
    // Spherical harmonics evaluation
    // -----------------------------------------------------------------------

    m.def("gsplat_eval_sh",
          &ops::evalSphericalHarmonics,
          py::arg("means"),
          py::arg("sh0"),
          py::arg("shN"),
          py::arg("sh_degree_to_use"),
          py::arg("world_to_camera_matrices"),
          py::arg("per_gaussian_projected_radii"));

    // -----------------------------------------------------------------------
    // Dense projection (no accumulator exposure -- pure functional)
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_project_gaussians_analytic",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel) {
            std::optional<torch::Tensor> gn, sc, mr;
            return ops::projectGaussiansAnalytic(
                means, quats, logScales, logitOpacities, sh0, shN, worldToCameraMatrices,
                projectionMatrices, settings, cameraModel, false, false, gn, sc, mr);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"));

    m.def("gsplat_project_gaussians_ut",
          &ops::projectGaussiansUT,
          py::arg("means"),
          py::arg("quats"),
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("sh0"),
          py::arg("shN"),
          py::arg("world_to_camera_matrices"),
          py::arg("projection_matrices"),
          py::arg("distortion_coeffs"),
          py::arg("settings"),
          py::arg("camera_model"));

    m.def(
        "gsplat_project_gaussians_for_camera",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel,
           ProjectionMethod projectionMethod,
           const std::optional<torch::Tensor> &distortionCoeffs) {
            std::optional<torch::Tensor> gn, sc, mr;
            return ops::projectGaussiansForCamera(
                means, quats, logScales, logitOpacities, sh0, shN, worldToCameraMatrices,
                projectionMatrices, settings, cameraModel, projectionMethod, distortionCoeffs,
                false, false, gn, sc, mr);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"),
        py::arg("projection_method"),
        py::arg("distortion_coeffs"));

    // -----------------------------------------------------------------------
    // Dense projection with accumulator support (for OO layer)
    // The accumulator tensors are mutated in-place during the backward pass
    // by the CUDA projection kernel. The Python OO layer owns these tensors
    // and passes them through so backward can find and update them.
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_project_gaussians_for_camera_with_accum",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel,
           ProjectionMethod projectionMethod,
           const std::optional<torch::Tensor> &distortionCoeffs,
           bool accumulateMean2dGradients,
           bool accumulateMax2dRadii,
           std::optional<torch::Tensor> accumGradNorms,
           std::optional<torch::Tensor> accumStepCounts,
           std::optional<torch::Tensor> accumMax2dRadii) {
            // Call the C++ function with mutable refs; accumulators may be
            // lazily initialized inside.
            auto result = ops::projectGaussiansForCamera(
                means, quats, logScales, logitOpacities, sh0, shN, worldToCameraMatrices,
                projectionMatrices, settings, cameraModel, projectionMethod, distortionCoeffs,
                accumulateMean2dGradients, accumulateMax2dRadii, accumGradNorms, accumStepCounts,
                accumMax2dRadii);
            // Return the projected state plus the (possibly newly allocated) accumulators
            return std::make_tuple(result, accumGradNorms, accumStepCounts, accumMax2dRadii);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"),
        py::arg("projection_method"),
        py::arg("distortion_coeffs"),
        py::arg("accumulate_mean_2d_gradients"),
        py::arg("accumulate_max_2d_radii"),
        py::arg("accum_grad_norms"),
        py::arg("accum_step_counts"),
        py::arg("accum_max_2d_radii"));

    // -----------------------------------------------------------------------
    // Sparse projection (no accumulator exposure -- pure functional)
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_sparse_project_gaussians_analytic",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel) {
            std::optional<torch::Tensor> gn, sc, mr;
            return ops::sparseProjectGaussiansAnalytic(
                pixelsToRender, means, quats, logScales, logitOpacities, sh0, shN,
                worldToCameraMatrices, projectionMatrices, settings, cameraModel, false, false, gn,
                sc, mr);
        },
        py::arg("pixels_to_render"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"));

    m.def("gsplat_sparse_project_gaussians_ut",
          &ops::sparseProjectGaussiansUT,
          py::arg("pixels_to_render"),
          py::arg("means"),
          py::arg("quats"),
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("sh0"),
          py::arg("shN"),
          py::arg("world_to_camera_matrices"),
          py::arg("projection_matrices"),
          py::arg("distortion_coeffs"),
          py::arg("settings"),
          py::arg("camera_model"));

    m.def(
        "gsplat_sparse_project_gaussians_for_camera",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel,
           ProjectionMethod projectionMethod,
           const std::optional<torch::Tensor> &distortionCoeffs) {
            std::optional<torch::Tensor> gn, sc, mr;
            return ops::sparseProjectGaussiansForCamera(
                pixelsToRender, means, quats, logScales, logitOpacities, sh0, shN,
                worldToCameraMatrices, projectionMatrices, settings, cameraModel, projectionMethod,
                distortionCoeffs, false, false, gn, sc, mr);
        },
        py::arg("pixels_to_render"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"),
        py::arg("projection_method"),
        py::arg("distortion_coeffs"));

    // -----------------------------------------------------------------------
    // Dense rasterization
    // -----------------------------------------------------------------------

    m.def("gsplat_render_crop_from_projected",
          &ops::renderCropFromProjected,
          py::arg("projected_gaussians"),
          py::arg("tile_size"),
          py::arg("crop_width"),
          py::arg("crop_height"),
          py::arg("crop_origin_w"),
          py::arg("crop_origin_h"),
          py::arg("backgrounds"),
          py::arg("masks"));

    // -----------------------------------------------------------------------
    // Sparse rendering (no accumulator exposure -- pure functional)
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_sparse_render",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const torch::Tensor &worldToCameraMatrices,
           const torch::Tensor &projectionMatrices,
           const RenderSettings &settings,
           DistortionModel cameraModel,
           ProjectionMethod projectionMethod,
           const std::optional<torch::Tensor> &distortionCoeffs,
           const std::optional<torch::Tensor> &backgrounds,
           const std::optional<torch::Tensor> &masks) {
            std::optional<torch::Tensor> gn, sc, mr;
            return ops::sparseRender(pixelsToRender, means, quats, logScales, logitOpacities, sh0,
                                     shN, worldToCameraMatrices, projectionMatrices, settings,
                                     cameraModel, projectionMethod, distortionCoeffs, backgrounds,
                                     masks, false, false, gn, sc, mr);
        },
        py::arg("pixels_to_render"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("world_to_camera_matrices"),
        py::arg("projection_matrices"),
        py::arg("settings"),
        py::arg("camera_model"),
        py::arg("projection_method"),
        py::arg("distortion_coeffs"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // -----------------------------------------------------------------------
    // From-world rasterization
    // -----------------------------------------------------------------------

    m.def("gsplat_rasterize_from_world",
          &ops::rasterizeFromWorld,
          py::arg("means"),
          py::arg("quats"),
          py::arg("log_scales"),
          py::arg("projected_state"),
          py::arg("world_to_camera_matrices"),
          py::arg("projection_matrices"),
          py::arg("distortion_coeffs"),
          py::arg("camera_model"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("tile_size"),
          py::arg("backgrounds"),
          py::arg("masks"));

    // -----------------------------------------------------------------------
    // Query operations
    // -----------------------------------------------------------------------

    m.def("gsplat_render_num_contributing",
          &ops::renderNumContributing,
          py::arg("state"),
          py::arg("settings"));

    m.def("gsplat_sparse_render_num_contributing",
          &ops::sparseRenderNumContributing,
          py::arg("state"),
          py::arg("pixels_to_render"),
          py::arg("settings"));

    m.def("gsplat_render_contributing_ids",
          &ops::renderContributingIds,
          py::arg("state"),
          py::arg("settings"),
          py::arg("num_contributing_gaussians"));

    m.def("gsplat_sparse_render_contributing_ids",
          &ops::sparseRenderContributingIds,
          py::arg("state"),
          py::arg("pixels_to_render"),
          py::arg("settings"),
          py::arg("num_contributing_gaussians"));
}
