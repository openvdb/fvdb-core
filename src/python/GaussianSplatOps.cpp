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
#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>
#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>

#include <torch/extension.h>

void
bind_gaussian_splat_ops(py::module &m) {
    namespace ops = fvdb::detail::ops;
    using RenderSettings  = fvdb::detail::ops::RenderSettings;
    using DistortionModel = fvdb::detail::ops::DistortionModel;
    using ProjectionMethod = fvdb::detail::ops::ProjectionMethod;
    using RollingShutterType = fvdb::detail::ops::RollingShutterType;

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

    // -----------------------------------------------------------------------
    // MCMC operations (thin dispatch wrappers)
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_relocate_gaussians",
        [](const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &ratios,
           const torch::Tensor &binomialCoeffs,
           int nMax,
           float minOpacity) {
            return FVDB_DISPATCH_KERNEL(logScales.device(), [&]() {
                return ops::dispatchGaussianRelocation<DeviceTag>(
                    logScales, logitOpacities, ratios, binomialCoeffs, nMax, minOpacity);
            });
        },
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("ratios"),
        py::arg("binomial_coeffs"),
        py::arg("n_max"),
        py::arg("min_opacity"));

    m.def(
        "gsplat_add_noise_to_means",
        [](torch::Tensor &means,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &quats,
           float noiseScale,
           float t,
           float k) {
            FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianMCMCAddNoise<DeviceTag>(
                    means, logScales, logitOpacities, quats, noiseScale, t, k);
            });
        },
        py::arg("means"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("quats"),
        py::arg("noise_scale"),
        py::arg("t"),
        py::arg("k"));

    // -----------------------------------------------------------------------
    // PLY I/O (wraps C++ PLY functions, creates temporary GaussianSplat3d internally)
    // -----------------------------------------------------------------------

    m.def(
        "gsplat_save_ply",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const std::string &filename,
           std::optional<std::unordered_map<std::string, fvdb::GaussianSplat3d::PlyMetadataTypes>>
               metadata) {
            fvdb::GaussianSplat3d gs(means, quats, logScales, logitOpacities, sh0, shN,
                                     false, false, false);
            gs.savePly(filename, metadata);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("filename"),
        py::arg("metadata"));

    m.def(
        "gsplat_load_ply",
        [](const std::string &filename,
           torch::Device device)
            -> std::tuple<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::unordered_map<std::string, fvdb::GaussianSplat3d::PlyMetadataTypes>> {
            auto [gs, metadata] = fvdb::GaussianSplat3d::fromPly(filename, device);
            return {gs.means(),
                    gs.quats(),
                    gs.logScales(),
                    gs.logitOpacities(),
                    gs.sh0(),
                    gs.shN(),
                    metadata};
        },
        py::arg("filename"),
        py::arg("device") = torch::kCPU);

    // ------- Raw forward/backward dispatch (for Python autograd) -------

    // 1. gsplat_projection_fwd
    m.def(
        "gsplat_projection_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           int64_t imageWidth,
           int64_t imageHeight,
           float eps2d,
           float nearPlane,
           float farPlane,
           float minRadius2d,
           bool calcCompensations,
           bool ortho) {
            return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianProjectionForward<DeviceTag>(
                    means, quats, scales, worldToCamMatrices, projectionMatrices,
                    imageWidth, imageHeight, eps2d, nearPlane, farPlane,
                    minRadius2d, calcCompensations, ortho);
            });
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("min_radius_2d"),
        py::arg("calc_compensations"),
        py::arg("ortho"));

    // 2. gsplat_projection_bwd
    m.def(
        "gsplat_projection_bwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const at::optional<torch::Tensor> &compensations,
           uint32_t imageWidth,
           uint32_t imageHeight,
           float eps2d,
           const torch::Tensor &radii,
           const torch::Tensor &conics,
           const torch::Tensor &dLossDMeans2d,
           const torch::Tensor &dLossDDepths,
           const torch::Tensor &dLossDConics,
           const at::optional<torch::Tensor> &dLossDCompensations,
           bool worldToCamMatricesRequiresGrad,
           bool ortho,
           at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum,
           at::optional<torch::Tensor> outNormalizedMaxRadiiAccum,
           at::optional<torch::Tensor> outGradientStepCounts) {
            return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianProjectionBackward<DeviceTag>(
                    means, quats, scales, worldToCamMatrices, projectionMatrices,
                    compensations, imageWidth, imageHeight, eps2d,
                    radii, conics, dLossDMeans2d, dLossDDepths, dLossDConics,
                    dLossDCompensations, worldToCamMatricesRequiresGrad, ortho,
                    outNormalizeddLossdMeans2dNormAccum,
                    outNormalizedMaxRadiiAccum,
                    outGradientStepCounts);
            });
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("compensations"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("radii"),
        py::arg("conics"),
        py::arg("d_loss_d_means2d"),
        py::arg("d_loss_d_depths"),
        py::arg("d_loss_d_conics"),
        py::arg("d_loss_d_compensations"),
        py::arg("world_to_cam_matrices_requires_grad"),
        py::arg("ortho"),
        py::arg("out_normalized_d_loss_d_means2d_norm_accum") = py::none(),
        py::arg("out_normalized_max_radii_accum") = py::none(),
        py::arg("out_gradient_step_counts") = py::none());

    // 3. gsplat_sh_eval_fwd
    m.def(
        "gsplat_sh_eval_fwd",
        [](int64_t shDegreeToUse,
           int64_t numCameras,
           const torch::Tensor &viewDirs,
           const torch::Tensor &sh0Coeffs,
           const torch::Tensor &shNCoeffs,
           const torch::Tensor &radii) {
            return FVDB_DISPATCH_KERNEL(sh0Coeffs.device(), [&]() {
                return ops::dispatchSphericalHarmonicsForward<DeviceTag>(
                    shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
            });
        },
        py::arg("sh_degree_to_use"),
        py::arg("num_cameras"),
        py::arg("view_dirs"),
        py::arg("sh0_coeffs"),
        py::arg("sh_n_coeffs"),
        py::arg("radii"));

    // 4. gsplat_sh_eval_bwd
    m.def(
        "gsplat_sh_eval_bwd",
        [](int64_t shDegreeToUse,
           int64_t numCameras,
           int64_t numGaussians,
           const torch::Tensor &viewDirs,
           const torch::Tensor &shNCoeffs,
           const torch::Tensor &dLossDColors,
           const torch::Tensor &radii,
           bool computeDLossDViewDirs) {
            return FVDB_DISPATCH_KERNEL(dLossDColors.device(), [&]() {
                return ops::dispatchSphericalHarmonicsBackward<DeviceTag>(
                    shDegreeToUse, numCameras, numGaussians,
                    viewDirs, shNCoeffs, dLossDColors, radii,
                    computeDLossDViewDirs);
            });
        },
        py::arg("sh_degree_to_use"),
        py::arg("num_cameras"),
        py::arg("num_gaussians"),
        py::arg("view_dirs"),
        py::arg("sh_n_coeffs"),
        py::arg("d_loss_d_colors"),
        py::arg("radii"),
        py::arg("compute_d_loss_d_view_dirs"));

    // 5. gsplat_rasterize_fwd
    m.def(
        "gsplat_rasterize_fwd",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           uint32_t imageWidth,
           uint32_t imageHeight,
           uint32_t imageOriginW,
           uint32_t imageOriginH,
           uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                const ops::RenderWindow2D renderWindow{imageWidth, imageHeight,
                                                       imageOriginW, imageOriginH};
                return ops::dispatchGaussianRasterizeForward<DeviceTag>(
                    means2d, conics, features, opacities,
                    renderWindow, tileSize, tileOffsets, tileGaussianIds,
                    backgrounds, masks);
            });
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 6. gsplat_rasterize_bwd
    m.def(
        "gsplat_rasterize_bwd",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           uint32_t imageWidth,
           uint32_t imageHeight,
           uint32_t imageOriginW,
           uint32_t imageOriginH,
           uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &renderedAlphas,
           const torch::Tensor &lastIds,
           const torch::Tensor &dLossDRenderedFeatures,
           const torch::Tensor &dLossDRenderedAlphas,
           bool absGrad,
           int64_t numSharedChannelsOverride,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                const ops::RenderWindow2D renderWindow{imageWidth, imageHeight,
                                                       imageOriginW, imageOriginH};
                return ops::dispatchGaussianRasterizeBackward<DeviceTag>(
                    means2d, conics, features, opacities,
                    renderWindow, tileSize, tileOffsets, tileGaussianIds,
                    renderedAlphas, lastIds,
                    dLossDRenderedFeatures, dLossDRenderedAlphas,
                    absGrad, numSharedChannelsOverride,
                    backgrounds, masks);
            });
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("rendered_alphas"),
        py::arg("last_ids"),
        py::arg("d_loss_d_rendered_features"),
        py::arg("d_loss_d_rendered_alphas"),
        py::arg("abs_grad"),
        py::arg("num_shared_channels_override") = -1,
        py::arg("backgrounds") = py::none(),
        py::arg("masks") = py::none());

    // 7. gsplat_rasterize_sparse_fwd
    m.def(
        "gsplat_rasterize_sparse_fwd",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           uint32_t imageWidth,
           uint32_t imageHeight,
           uint32_t imageOriginW,
           uint32_t imageOriginH,
           uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                const ops::RenderWindow2D renderWindow{imageWidth, imageHeight,
                                                       imageOriginW, imageOriginH};
                return ops::dispatchGaussianSparseRasterizeForward<DeviceTag>(
                    pixelsToRender, means2d, conics, features, opacities,
                    renderWindow, tileSize, tileOffsets, tileGaussianIds,
                    activeTiles, tilePixelMask, tilePixelCumsum, pixelMap,
                    backgrounds, masks);
            });
        },
        py::arg("pixels_to_render"),
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("active_tiles"),
        py::arg("tile_pixel_mask"),
        py::arg("tile_pixel_cumsum"),
        py::arg("pixel_map"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 8. gsplat_rasterize_sparse_bwd
    m.def(
        "gsplat_rasterize_sparse_bwd",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           uint32_t imageWidth,
           uint32_t imageHeight,
           uint32_t imageOriginW,
           uint32_t imageOriginH,
           uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const fvdb::JaggedTensor &renderedAlphas,
           const fvdb::JaggedTensor &lastIds,
           const fvdb::JaggedTensor &dLossDRenderedFeatures,
           const fvdb::JaggedTensor &dLossDRenderedAlphas,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           bool absGrad,
           int64_t numSharedChannelsOverride,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                const ops::RenderWindow2D renderWindow{imageWidth, imageHeight,
                                                       imageOriginW, imageOriginH};
                return ops::dispatchGaussianSparseRasterizeBackward<DeviceTag>(
                    pixelsToRender, means2d, conics, features, opacities,
                    renderWindow, tileSize, tileOffsets, tileGaussianIds,
                    renderedAlphas, lastIds,
                    dLossDRenderedFeatures, dLossDRenderedAlphas,
                    activeTiles, tilePixelMask, tilePixelCumsum, pixelMap,
                    absGrad, numSharedChannelsOverride,
                    backgrounds, masks);
            });
        },
        py::arg("pixels_to_render"),
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("rendered_alphas"),
        py::arg("last_ids"),
        py::arg("d_loss_d_rendered_features"),
        py::arg("d_loss_d_rendered_alphas"),
        py::arg("active_tiles"),
        py::arg("tile_pixel_mask"),
        py::arg("tile_pixel_cumsum"),
        py::arg("pixel_map"),
        py::arg("abs_grad"),
        py::arg("num_shared_channels_override") = -1,
        py::arg("backgrounds") = py::none(),
        py::arg("masks") = py::none());

    // 9. gsplat_rasterize_from_world_fwd
    m.def(
        "gsplat_rasterize_from_world_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const torch::Tensor &worldToCamMatricesStart,
           const torch::Tensor &worldToCamMatricesEnd,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           RollingShutterType rollingShutterType,
           DistortionModel cameraModel,
           const RenderSettings &settings,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianRasterizeFromWorld3DGSForward<DeviceTag>(
                    means, quats, logScales,
                    features, opacities,
                    worldToCamMatricesStart, worldToCamMatricesEnd,
                    projectionMatrices, distortionCoeffs,
                    rollingShutterType, cameraModel, settings,
                    tileOffsets, tileGaussianIds,
                    backgrounds, masks);
            });
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("world_to_cam_matrices_start"),
        py::arg("world_to_cam_matrices_end"),
        py::arg("projection_matrices"),
        py::arg("distortion_coeffs"),
        py::arg("rolling_shutter_type"),
        py::arg("camera_model"),
        py::arg("settings"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 10. gsplat_rasterize_from_world_bwd
    m.def(
        "gsplat_rasterize_from_world_bwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const torch::Tensor &worldToCamMatricesStart,
           const torch::Tensor &worldToCamMatricesEnd,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           RollingShutterType rollingShutterType,
           DistortionModel cameraModel,
           const RenderSettings &settings,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &renderedAlphas,
           const torch::Tensor &lastIds,
           const torch::Tensor &dLossDRenderedFeatures,
           const torch::Tensor &dLossDRenderedAlphas,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianRasterizeFromWorld3DGSBackward<DeviceTag>(
                    means, quats, logScales,
                    features, opacities,
                    worldToCamMatricesStart, worldToCamMatricesEnd,
                    projectionMatrices, distortionCoeffs,
                    rollingShutterType, cameraModel, settings,
                    tileOffsets, tileGaussianIds,
                    renderedAlphas, lastIds,
                    dLossDRenderedFeatures, dLossDRenderedAlphas,
                    backgrounds, masks);
            });
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("features"),
        py::arg("opacities"),
        py::arg("world_to_cam_matrices_start"),
        py::arg("world_to_cam_matrices_end"),
        py::arg("projection_matrices"),
        py::arg("distortion_coeffs"),
        py::arg("rolling_shutter_type"),
        py::arg("camera_model"),
        py::arg("settings"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("rendered_alphas"),
        py::arg("last_ids"),
        py::arg("d_loss_d_rendered_features"),
        py::arg("d_loss_d_rendered_alphas"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 11. gsplat_projection_jagged_fwd
    m.def(
        "gsplat_projection_jagged_fwd",
        [](const torch::Tensor &gSizes,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &cSizes,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           uint32_t imageWidth,
           uint32_t imageHeight,
           float eps2d,
           float nearPlane,
           float farPlane,
           float minRadius2d,
           bool ortho) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianProjectionJaggedForward<DeviceTag>(
                    gSizes, means, quats, scales, cSizes,
                    worldToCamMatrices, projectionMatrices,
                    imageWidth, imageHeight, eps2d,
                    nearPlane, farPlane, minRadius2d, ortho);
            });
        },
        py::arg("g_sizes"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("c_sizes"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("near_plane"),
        py::arg("far_plane"),
        py::arg("min_radius_2d"),
        py::arg("ortho"));

    // 12. gsplat_projection_jagged_bwd
    m.def(
        "gsplat_projection_jagged_bwd",
        [](const torch::Tensor &gSizes,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &cSizes,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           uint32_t imageWidth,
           uint32_t imageHeight,
           float eps2d,
           const torch::Tensor &radii,
           const torch::Tensor &conics,
           const torch::Tensor &dLossDMeans2d,
           const torch::Tensor &dLossDDepths,
           const torch::Tensor &dLossDConics,
           bool worldToCamMatricesRequiresGrad,
           bool ortho) {
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianProjectionJaggedBackward<DeviceTag>(
                    gSizes, means, quats, scales, cSizes,
                    worldToCamMatrices, projectionMatrices,
                    imageWidth, imageHeight, eps2d,
                    radii, conics,
                    dLossDMeans2d, dLossDDepths, dLossDConics,
                    worldToCamMatricesRequiresGrad, ortho);
            });
        },
        py::arg("g_sizes"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("c_sizes"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("radii"),
        py::arg("conics"),
        py::arg("d_loss_d_means2d"),
        py::arg("d_loss_d_depths"),
        py::arg("d_loss_d_conics"),
        py::arg("world_to_cam_matrices_requires_grad"),
        py::arg("ortho"));

    // ------- Tile intersection (non-differentiable) -------

    m.def(
        "gsplat_tile_intersection",
        [](const torch::Tensor &means2d,
           const torch::Tensor &radii,
           const torch::Tensor &depths,
           uint32_t numCameras,
           uint32_t tileSize,
           uint32_t numTilesH,
           uint32_t numTilesW) {
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                return ops::dispatchGaussianTileIntersection<DeviceTag>(
                    means2d, radii, depths, at::nullopt,
                    numCameras, tileSize, numTilesH, numTilesW);
            });
        },
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depths"),
        py::arg("num_cameras"),
        py::arg("tile_size"),
        py::arg("num_tiles_h"),
        py::arg("num_tiles_w"));
}
