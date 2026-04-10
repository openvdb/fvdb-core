// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Pybind11 bindings for Gaussian splat free-function ops.
// These expose the fvdb::detail::ops functions as module-level functions on
// _fvdb_cpp, enabling the Python functional layer.
//
// Design note on accumulator mutability:
// The C++ projection backward kernel mutates three accumulator tensors in-place
// (gradient norms, max 2D radii, step counts) via atomicAdd. These support
// Gaussian densification (split/clone/prune decisions during training).
// The backward binding (project_gaussians_analytic_bwd) accepts these as
// optional tensors. The Python GaussianSplat3d class owns the accumulators
// and passes them through to the C++ backward dispatch.

#include <pybind11/stl.h>

#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/ops/AddNoiseToGaussianMeans.h>
#include <fvdb/detail/ops/BuildSparseGaussianTileLayout.h>
#include <fvdb/detail/ops/CountContributingGaussians.h>
#include <fvdb/detail/ops/EvalGaussianShBackward.h>
#include <fvdb/detail/ops/EvalGaussianShForward.h>
#include <fvdb/detail/ops/IdentifyContributingGaussians.h>
#include <fvdb/detail/ops/IntersectGaussianTiles.h>
#include <fvdb/detail/ops/ProjectGaussiansAnalyticBackward.h>
#include <fvdb/detail/ops/ProjectGaussiansAnalyticForward.h>
#include <fvdb/detail/ops/ProjectGaussiansAnalyticJaggedBackward.h>
#include <fvdb/detail/ops/ProjectGaussiansAnalyticJaggedForward.h>
#include <fvdb/detail/ops/ProjectGaussiansUtForward.h>
#include <fvdb/detail/ops/RasterizeScreenSpaceGaussiansBackward.h>
#include <fvdb/detail/ops/RasterizeScreenSpaceGaussiansForward.h>
#include <fvdb/detail/ops/RasterizeWorldSpaceGaussiansBackward.h>
#include <fvdb/detail/ops/RasterizeWorldSpaceGaussiansForward.h>
#include <fvdb/detail/ops/RelocateGaussians.h>

#include <torch/extension.h>

void
bind_gaussian_splat_ops(py::module &m) {
    namespace ops            = fvdb::detail::ops;
    using DistortionModel    = fvdb::detail::ops::DistortionModel;
    using RollingShutterType = fvdb::detail::ops::RollingShutterType;

    // -----------------------------------------------------------------------
    // Enum types (moved from GaussianSplatBinding.cpp)
    // -----------------------------------------------------------------------

    py::enum_<fvdb::detail::ops::RollingShutterType>(m, "RollingShutterType")
        .value("NONE", fvdb::detail::ops::RollingShutterType::NONE)
        .value("VERTICAL", fvdb::detail::ops::RollingShutterType::VERTICAL)
        .value("HORIZONTAL", fvdb::detail::ops::RollingShutterType::HORIZONTAL)
        .export_values();

    py::enum_<fvdb::detail::ops::DistortionModel>(m, "CameraModel")
        .value("PINHOLE", fvdb::detail::ops::DistortionModel::PINHOLE)
        .value("OPENCV_RADTAN_5", fvdb::detail::ops::DistortionModel::OPENCV_RADTAN_5)
        .value("OPENCV_RATIONAL_8", fvdb::detail::ops::DistortionModel::OPENCV_RATIONAL_8)
        .value("OPENCV_RADTAN_THIN_PRISM_9",
               fvdb::detail::ops::DistortionModel::OPENCV_RADTAN_THIN_PRISM_9)
        .value("OPENCV_THIN_PRISM_12", fvdb::detail::ops::DistortionModel::OPENCV_THIN_PRISM_12)
        .value("ORTHOGRAPHIC", fvdb::detail::ops::DistortionModel::ORTHOGRAPHIC)
        .export_values();

    py::enum_<fvdb::detail::ops::ProjectionMethod>(m, "ProjectionMethod")
        .value("AUTO", fvdb::detail::ops::ProjectionMethod::AUTO)
        .value("ANALYTIC", fvdb::detail::ops::ProjectionMethod::ANALYTIC)
        .value("UNSCENTED", fvdb::detail::ops::ProjectionMethod::UNSCENTED)
        .export_values();

    // -----------------------------------------------------------------------
    // Data types needed by the functional ops
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Kernel-level bindings (for Python autograd and composition)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Analysis operations (call raw tensor dispatch functions directly)
    // -----------------------------------------------------------------------

    m.def("count_contributing_gaussians",
          &ops::count_contributing_gaussians,
          py::arg("means2d"),
          py::arg("conics"),
          py::arg("opacities"),
          py::arg("tile_offsets"),
          py::arg("tile_gaussian_ids"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("image_origin_w"),
          py::arg("image_origin_h"),
          py::arg("tile_size"));

    m.def("count_contributing_gaussians_sparse",
          &ops::count_contributing_gaussians_sparse,
          py::arg("means2d"),
          py::arg("conics"),
          py::arg("opacities"),
          py::arg("tile_offsets"),
          py::arg("tile_gaussian_ids"),
          py::arg("pixels_to_render"),
          py::arg("active_tiles"),
          py::arg("tile_pixel_mask"),
          py::arg("tile_pixel_cumsum"),
          py::arg("pixel_map"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("image_origin_w"),
          py::arg("image_origin_h"),
          py::arg("tile_size"));

    m.def("identify_contributing_gaussians",
          &ops::identify_contributing_gaussians,
          py::arg("means2d"),
          py::arg("conics"),
          py::arg("opacities"),
          py::arg("tile_offsets"),
          py::arg("tile_gaussian_ids"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("image_origin_w"),
          py::arg("image_origin_h"),
          py::arg("tile_size"),
          py::arg("num_depth_samples"),
          py::arg("num_contributing_gaussians") = py::none());

    m.def("identify_contributing_gaussians_sparse",
          &ops::identify_contributing_gaussians_sparse,
          py::arg("means2d"),
          py::arg("conics"),
          py::arg("opacities"),
          py::arg("tile_offsets"),
          py::arg("tile_gaussian_ids"),
          py::arg("pixels_to_render"),
          py::arg("active_tiles"),
          py::arg("tile_pixel_mask"),
          py::arg("tile_pixel_cumsum"),
          py::arg("pixel_map"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("image_origin_w"),
          py::arg("image_origin_h"),
          py::arg("tile_size"),
          py::arg("num_depth_samples"),
          py::arg("num_contributing_gaussians") = py::none());

    // -----------------------------------------------------------------------
    // MCMC operations (thin dispatch wrappers)
    // -----------------------------------------------------------------------

    m.def(
        "relocate_gaussians",
        [](const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &ratios,
           const torch::Tensor &binomialCoeffs,
           const int nMax,
           const float minOpacity) {
            return ops::relocate_gaussians(
                logScales, logitOpacities, ratios, binomialCoeffs, nMax, minOpacity);
        },
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("ratios"),
        py::arg("binomial_coeffs"),
        py::arg("n_max"),
        py::arg("min_opacity"));

    m.def(
        "add_noise_to_gaussian_means",
        [](torch::Tensor &means,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &quats,
           const float noiseScale,
           const float t,
           const float k) {
            ops::add_noise_to_gaussian_means(
                means, logScales, logitOpacities, quats, noiseScale, t, k);
        },
        py::arg("means"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("quats"),
        py::arg("noise_scale"),
        py::arg("t"),
        py::arg("k"));

    // -----------------------------------------------------------------------
    // PLY I/O (wraps C++ PLY functions directly with raw tensors)
    // -----------------------------------------------------------------------

    m.def(
        "save_gaussians_ply",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const std::string &filename,
           std::optional<std::unordered_map<std::string, fvdb::detail::io::PlyMetadataTypes>>
               metadata) {
            fvdb::detail::io::saveGaussianPly(
                filename, means, quats, logScales, logitOpacities, sh0, shN, metadata);
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
        "load_gaussians_ply",
        [](const std::string &filename, torch::Device device)
            -> std::tuple<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::unordered_map<std::string, fvdb::detail::io::PlyMetadataTypes>> {
            return fvdb::detail::io::loadGaussianPly(filename, device);
        },
        py::arg("filename"),
        py::arg("device") = torch::kCPU);

    // ------- Raw forward/backward dispatch (for Python autograd) -------

    // 1. project_gaussians_analytic_fwd
    m.def(
        "project_gaussians_analytic_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const int64_t imageWidth,
           const int64_t imageHeight,
           const float eps2d,
           const float nearPlane,
           const float farPlane,
           const float minRadius2d,
           const bool calcCompensations,
           const bool ortho) {
            return ops::project_gaussians_analytic_fwd(means,
                                                       quats,
                                                       scales,
                                                       worldToCamMatrices,
                                                       projectionMatrices,
                                                       imageWidth,
                                                       imageHeight,
                                                       eps2d,
                                                       nearPlane,
                                                       farPlane,
                                                       minRadius2d,
                                                       calcCompensations,
                                                       ortho);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("near"),
        py::arg("far"),
        py::arg("min_radius_2d"),
        py::arg("calc_compensations"),
        py::arg("ortho"));

    // 2. project_gaussians_analytic_bwd
    m.def(
        "project_gaussians_analytic_bwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const at::optional<torch::Tensor> &compensations,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const float eps2d,
           const torch::Tensor &radii,
           const torch::Tensor &conics,
           const torch::Tensor &dLossDMeans2d,
           const torch::Tensor &dLossDDepths,
           const torch::Tensor &dLossDConics,
           const at::optional<torch::Tensor> &dLossDCompensations,
           const bool worldToCamMatricesRequiresGrad,
           const bool ortho,
           at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum,
           at::optional<torch::Tensor> outNormalizedMaxRadiiAccum,
           at::optional<torch::Tensor> outGradientStepCounts) {
            return ops::project_gaussians_analytic_bwd(means,
                                                       quats,
                                                       scales,
                                                       worldToCamMatrices,
                                                       projectionMatrices,
                                                       compensations,
                                                       imageWidth,
                                                       imageHeight,
                                                       eps2d,
                                                       radii,
                                                       conics,
                                                       dLossDMeans2d,
                                                       dLossDDepths,
                                                       dLossDConics,
                                                       dLossDCompensations,
                                                       worldToCamMatricesRequiresGrad,
                                                       ortho,
                                                       outNormalizeddLossdMeans2dNormAccum,
                                                       outNormalizedMaxRadiiAccum,
                                                       outGradientStepCounts);
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
        py::arg("out_normalized_max_radii_accum")             = py::none(),
        py::arg("out_gradient_step_counts")                   = py::none());

    // 3. eval_gaussian_sh_fwd
    m.def(
        "eval_gaussian_sh_fwd",
        [](const int64_t shDegreeToUse,
           const int64_t numCameras,
           const torch::Tensor &viewDirs,
           const torch::Tensor &sh0Coeffs,
           const torch::Tensor &shNCoeffs,
           const torch::Tensor &radii) {
            return ops::eval_gaussian_sh_fwd(
                shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        },
        py::arg("sh_degree_to_use"),
        py::arg("num_cameras"),
        py::arg("view_dirs"),
        py::arg("sh0_coeffs"),
        py::arg("sh_n_coeffs"),
        py::arg("radii"));

    // 4. eval_gaussian_sh_bwd
    m.def(
        "eval_gaussian_sh_bwd",
        [](const int64_t shDegreeToUse,
           const int64_t numCameras,
           const int64_t numGaussians,
           const torch::Tensor &viewDirs,
           const torch::Tensor &shNCoeffs,
           const torch::Tensor &dLossDColors,
           const torch::Tensor &radii,
           const bool computeDLossDViewDirs) {
            return ops::eval_gaussian_sh_bwd(shDegreeToUse,
                                             numCameras,
                                             numGaussians,
                                             viewDirs,
                                             shNCoeffs,
                                             dLossDColors,
                                             radii,
                                             computeDLossDViewDirs);
        },
        py::arg("sh_degree_to_use"),
        py::arg("num_cameras"),
        py::arg("num_gaussians"),
        py::arg("view_dirs"),
        py::arg("sh_n_coeffs"),
        py::arg("d_loss_d_colors"),
        py::arg("radii"),
        py::arg("compute_d_loss_d_view_dirs"));

    // 5. rasterize_screen_space_gaussians_fwd
    m.def(
        "rasterize_screen_space_gaussians_fwd",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_screen_space_gaussians_fwd(means2d,
                                                             conics,
                                                             features,
                                                             opacities,
                                                             imageWidth,
                                                             imageHeight,
                                                             imageOriginW,
                                                             imageOriginH,
                                                             tileSize,
                                                             tileOffsets,
                                                             tileGaussianIds,
                                                             backgrounds,
                                                             masks);
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

    // 6. rasterize_screen_space_gaussians_bwd
    m.def(
        "rasterize_screen_space_gaussians_bwd",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &renderedAlphas,
           const torch::Tensor &lastIds,
           const torch::Tensor &dLossDRenderedFeatures,
           const torch::Tensor &dLossDRenderedAlphas,
           const bool absGrad,
           const int64_t numSharedChannelsOverride,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_screen_space_gaussians_bwd(means2d,
                                                             conics,
                                                             features,
                                                             opacities,
                                                             imageWidth,
                                                             imageHeight,
                                                             imageOriginW,
                                                             imageOriginH,
                                                             tileSize,
                                                             tileOffsets,
                                                             tileGaussianIds,
                                                             renderedAlphas,
                                                             lastIds,
                                                             dLossDRenderedFeatures,
                                                             dLossDRenderedAlphas,
                                                             absGrad,
                                                             numSharedChannelsOverride,
                                                             backgrounds,
                                                             masks);
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
        py::arg("backgrounds")                  = py::none(),
        py::arg("masks")                        = py::none());

    // 7. rasterize_screen_space_gaussians_sparse_fwd
    m.def(
        "rasterize_screen_space_gaussians_sparse_fwd",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_screen_space_gaussians_sparse_fwd(pixelsToRender,
                                                                    means2d,
                                                                    conics,
                                                                    features,
                                                                    opacities,
                                                                    imageWidth,
                                                                    imageHeight,
                                                                    imageOriginW,
                                                                    imageOriginH,
                                                                    tileSize,
                                                                    tileOffsets,
                                                                    tileGaussianIds,
                                                                    activeTiles,
                                                                    tilePixelMask,
                                                                    tilePixelCumsum,
                                                                    pixelMap,
                                                                    backgrounds,
                                                                    masks);
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

    // 8. rasterize_screen_space_gaussians_sparse_bwd
    m.def(
        "rasterize_screen_space_gaussians_sparse_bwd",
        [](const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
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
           const bool absGrad,
           const int64_t numSharedChannelsOverride,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_screen_space_gaussians_sparse_bwd(pixelsToRender,
                                                                    means2d,
                                                                    conics,
                                                                    features,
                                                                    opacities,
                                                                    imageWidth,
                                                                    imageHeight,
                                                                    imageOriginW,
                                                                    imageOriginH,
                                                                    tileSize,
                                                                    tileOffsets,
                                                                    tileGaussianIds,
                                                                    renderedAlphas,
                                                                    lastIds,
                                                                    dLossDRenderedFeatures,
                                                                    dLossDRenderedAlphas,
                                                                    activeTiles,
                                                                    tilePixelMask,
                                                                    tilePixelCumsum,
                                                                    pixelMap,
                                                                    absGrad,
                                                                    numSharedChannelsOverride,
                                                                    backgrounds,
                                                                    masks);
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
        py::arg("backgrounds")                  = py::none(),
        py::arg("masks")                        = py::none());

    // 9. rasterize_world_space_gaussians_fwd
    m.def(
        "rasterize_world_space_gaussians_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const torch::Tensor &worldToCamMatricesStart,
           const torch::Tensor &worldToCamMatricesEnd,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           const RollingShutterType rollingShutterType,
           const DistortionModel cameraModel,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_world_space_gaussians_fwd(means,
                                                            quats,
                                                            logScales,
                                                            features,
                                                            opacities,
                                                            worldToCamMatricesStart,
                                                            worldToCamMatricesEnd,
                                                            projectionMatrices,
                                                            distortionCoeffs,
                                                            rollingShutterType,
                                                            cameraModel,
                                                            imageWidth,
                                                            imageHeight,
                                                            imageOriginW,
                                                            imageOriginH,
                                                            tileSize,
                                                            tileOffsets,
                                                            tileGaussianIds,
                                                            backgrounds,
                                                            masks);
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
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 10. rasterize_world_space_gaussians_bwd
    m.def(
        "rasterize_world_space_gaussians_bwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &features,
           const torch::Tensor &opacities,
           const torch::Tensor &worldToCamMatricesStart,
           const torch::Tensor &worldToCamMatricesEnd,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           const RollingShutterType rollingShutterType,
           const DistortionModel cameraModel,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t imageOriginW,
           const uint32_t imageOriginH,
           const uint32_t tileSize,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &renderedAlphas,
           const torch::Tensor &lastIds,
           const torch::Tensor &dLossDRenderedFeatures,
           const torch::Tensor &dLossDRenderedAlphas,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::rasterize_world_space_gaussians_bwd(means,
                                                            quats,
                                                            logScales,
                                                            features,
                                                            opacities,
                                                            worldToCamMatricesStart,
                                                            worldToCamMatricesEnd,
                                                            projectionMatrices,
                                                            distortionCoeffs,
                                                            rollingShutterType,
                                                            cameraModel,
                                                            imageWidth,
                                                            imageHeight,
                                                            imageOriginW,
                                                            imageOriginH,
                                                            tileSize,
                                                            tileOffsets,
                                                            tileGaussianIds,
                                                            renderedAlphas,
                                                            lastIds,
                                                            dLossDRenderedFeatures,
                                                            dLossDRenderedAlphas,
                                                            backgrounds,
                                                            masks);
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
        py::arg("backgrounds"),
        py::arg("masks"));

    // 11. project_gaussians_analytic_jagged_fwd
    m.def(
        "project_gaussians_analytic_jagged_fwd",
        [](const torch::Tensor &gSizes,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &cSizes,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const float eps2d,
           const float nearPlane,
           const float farPlane,
           const float minRadius2d,
           const bool ortho) {
            return ops::project_gaussians_analytic_jagged_fwd(gSizes,
                                                              means,
                                                              quats,
                                                              scales,
                                                              cSizes,
                                                              worldToCamMatrices,
                                                              projectionMatrices,
                                                              imageWidth,
                                                              imageHeight,
                                                              eps2d,
                                                              nearPlane,
                                                              farPlane,
                                                              minRadius2d,
                                                              ortho);
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
        py::arg("near"),
        py::arg("far"),
        py::arg("min_radius_2d"),
        py::arg("ortho"));

    // 12. project_gaussians_analytic_jagged_bwd
    m.def(
        "project_gaussians_analytic_jagged_bwd",
        [](const torch::Tensor &gSizes,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &scales,
           const torch::Tensor &cSizes,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const float eps2d,
           const torch::Tensor &radii,
           const torch::Tensor &conics,
           const torch::Tensor &dLossDMeans2d,
           const torch::Tensor &dLossDDepths,
           const torch::Tensor &dLossDConics,
           const bool worldToCamMatricesRequiresGrad,
           const bool ortho) {
            return ops::project_gaussians_analytic_jagged_bwd(gSizes,
                                                              means,
                                                              quats,
                                                              scales,
                                                              cSizes,
                                                              worldToCamMatrices,
                                                              projectionMatrices,
                                                              imageWidth,
                                                              imageHeight,
                                                              eps2d,
                                                              radii,
                                                              conics,
                                                              dLossDMeans2d,
                                                              dLossDDepths,
                                                              dLossDConics,
                                                              worldToCamMatricesRequiresGrad,
                                                              ortho);
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
        "intersect_gaussian_tiles",
        [](const torch::Tensor &means2d,
           const torch::Tensor &radii,
           const torch::Tensor &depths,
           const uint32_t numCameras,
           const uint32_t tileSize,
           const uint32_t numTilesH,
           const uint32_t numTilesW,
           const at::optional<torch::Tensor> &cameraIds) {
            return ops::intersect_gaussian_tiles(
                means2d, radii, depths, cameraIds, numCameras, tileSize, numTilesH, numTilesW);
        },
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depths"),
        py::arg("num_cameras"),
        py::arg("tile_size"),
        py::arg("num_tiles_h"),
        py::arg("num_tiles_w"),
        py::arg("camera_ids") = py::none());

    // ------- Sparse tile intersection (non-differentiable) -------

    m.def(
        "intersect_gaussian_tiles_sparse",
        [](const torch::Tensor &means2d,
           const torch::Tensor &radii,
           const torch::Tensor &depths,
           const torch::Tensor &tileMask,
           const torch::Tensor &activeTiles,
           const uint32_t numCameras,
           const uint32_t tileSize,
           const uint32_t numTilesH,
           const uint32_t numTilesW,
           const at::optional<torch::Tensor> &cameraIds) {
            return ops::intersect_gaussian_tiles_sparse(means2d,
                                                        radii,
                                                        depths,
                                                        tileMask,
                                                        activeTiles,
                                                        cameraIds,
                                                        numCameras,
                                                        tileSize,
                                                        numTilesH,
                                                        numTilesW);
        },
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depths"),
        py::arg("tile_mask"),
        py::arg("active_tiles"),
        py::arg("num_cameras"),
        py::arg("tile_size"),
        py::arg("num_tiles_h"),
        py::arg("num_tiles_w"),
        py::arg("camera_ids") = py::none());

    // ------- Sparse tile layout (non-differentiable) -------

    m.def(
        "build_sparse_gaussian_tile_layout",
        [](const int32_t tileSideLength,
           const int32_t numTilesW,
           const int32_t numTilesH,
           const fvdb::JaggedTensor &pixelsToRender) {
            return ops::build_sparse_gaussian_tile_layout(
                tileSideLength, numTilesW, numTilesH, pixelsToRender);
        },
        py::arg("tile_side_length"),
        py::arg("num_tiles_w"),
        py::arg("num_tiles_h"),
        py::arg("pixels_to_render"));

    // ------- UT projection forward (non-differentiable) -------

    m.def(
        "project_gaussians_ut_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &worldToCamMatrices,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           const DistortionModel cameraModel,
           const int64_t imageWidth,
           const int64_t imageHeight,
           const float eps2d,
           const float nearPlane,
           const float farPlane,
           const float minRadius2d,
           const bool calcCompensations) {
            ops::UTParams utParams{};
            return ops::project_gaussians_ut_fwd(means,
                                                 quats,
                                                 logScales,
                                                 worldToCamMatrices,
                                                 worldToCamMatrices,
                                                 projectionMatrices,
                                                 ops::RollingShutterType::NONE,
                                                 utParams,
                                                 cameraModel,
                                                 distortionCoeffs,
                                                 imageWidth,
                                                 imageHeight,
                                                 eps2d,
                                                 nearPlane,
                                                 farPlane,
                                                 minRadius2d,
                                                 calcCompensations);
        },
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("world_to_cam_matrices"),
        py::arg("projection_matrices"),
        py::arg("distortion_coeffs"),
        py::arg("camera_model"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("near"),
        py::arg("far"),
        py::arg("min_radius_2d"),
        py::arg("calc_compensations"));
}
