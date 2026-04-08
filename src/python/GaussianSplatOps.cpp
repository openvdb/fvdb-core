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
// The C++ projection pipeline functions accept these as mutable
// std::optional<torch::Tensor>& refs. Here, we wrap those functions in lambdas
// that pass accumulateMean2dGradients=false and accumulateMax2dRadii=false,
// hiding the mutability and presenting a strictly functional interface to Python.
// The Python GaussianSplat3d class owns these accumulators and passes them
// through to the C++ backward dispatch.

#include <pybind11/stl.h>

#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/ops/gsplat/GaussianCameraValidation.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionPipeline.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionTypes.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeNumContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>

#include <torch/extension.h>

void
bind_gaussian_splat_ops(py::module &m) {
    namespace ops            = fvdb::detail::ops;
    using RenderSettings     = fvdb::detail::ops::RenderSettings;
    using DistortionModel    = fvdb::detail::ops::DistortionModel;
    using ProjectionMethod   = fvdb::detail::ops::ProjectionMethod;
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
        .value("FEATURES", RenderSettings::RenderMode::FEATURES)
        .value("DEPTH", RenderSettings::RenderMode::DEPTH)
        .value("FEATURES_AND_DEPTH", RenderSettings::RenderMode::FEATURES_AND_DEPTH)
        .export_values();


    // -----------------------------------------------------------------------
    // Kernel-level bindings (for Python autograd and composition)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Dense rasterization
    // -----------------------------------------------------------------------

    m.def(
        "render_crop_from_projected_gaussians",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &renderQuantities,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           int64_t imageWidth,
           int64_t imageHeight,
           int64_t tileSize,
           int64_t cropWidth,
           int64_t cropHeight,
           int64_t cropOriginW,
           int64_t cropOriginH,
           const std::optional<torch::Tensor> &backgrounds,
           const std::optional<torch::Tensor> &masks) {
            fvdb::ProjectedGaussianSplats s;
            s.perGaussian2dMean          = means2d;
            s.perGaussianConic           = conics;
            s.perGaussianRenderQuantity  = renderQuantities;
            s.perGaussianOpacity         = opacities;
            s.tileOffsets                = tileOffsets;
            s.tileGaussianIds            = tileGaussianIds;
            s.mRenderSettings.imageWidth  = imageWidth;
            s.mRenderSettings.imageHeight = imageHeight;
            return ops::renderCropFromProjected(
                s, tileSize, cropWidth, cropHeight, cropOriginW, cropOriginH, backgrounds, masks);
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("render_quantities"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("tile_size"),
        py::arg("crop_width"),
        py::arg("crop_height"),
        py::arg("crop_origin_w"),
        py::arg("crop_origin_h"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // (gsplat_sparse_render and gsplat_rasterize_from_world deleted —
    //  replaced by composing kernel-level bindings in Python)

    // -----------------------------------------------------------------------
    // Query operations
    // -----------------------------------------------------------------------

    m.def(
        "count_contributing_gaussians",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const RenderSettings &settings) {
            fvdb::ProjectedGaussianSplats s;
            s.perGaussian2dMean = means2d;
            s.perGaussianConic  = conics;
            s.perGaussianOpacity = opacities;
            s.tileOffsets        = tileOffsets;
            s.tileGaussianIds    = tileGaussianIds;
            return ops::renderNumContributing(s, settings);
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("settings"));

    m.def(
        "count_contributing_gaussians_sparse",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &activeTiles,
           const torch::Tensor &activeTileMask,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const torch::Tensor &inverseIndices,
           const fvdb::JaggedTensor &uniquePixelsToRender,
           const bool hasDuplicates,
           const fvdb::JaggedTensor &pixelsToRender,
           const RenderSettings &settings) {
            fvdb::SparseProjectedGaussianSplats ss;
            ss.perGaussian2dMean      = means2d;
            ss.perGaussianConic       = conics;
            ss.perGaussianOpacity     = opacities;
            ss.tileOffsets            = tileOffsets;
            ss.tileGaussianIds        = tileGaussianIds;
            ss.activeTiles            = activeTiles;
            ss.activeTileMask         = activeTileMask;
            ss.tilePixelMask          = tilePixelMask;
            ss.tilePixelCumsum        = tilePixelCumsum;
            ss.pixelMap               = pixelMap;
            ss.inverseIndices         = inverseIndices;
            ss.uniquePixelsToRender   = uniquePixelsToRender;
            ss.hasDuplicates          = hasDuplicates;
            return ops::sparseRenderNumContributing(ss, pixelsToRender, settings);
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("active_tiles"),
        py::arg("active_tile_mask"),
        py::arg("tile_pixel_mask"),
        py::arg("tile_pixel_cumsum"),
        py::arg("pixel_map"),
        py::arg("inverse_indices"),
        py::arg("unique_pixels_to_render"),
        py::arg("has_duplicates"),
        py::arg("pixels_to_render"),
        py::arg("settings"));

    m.def(
        "identify_contributing_gaussians",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const RenderSettings &settings,
           const std::optional<torch::Tensor> &numContributingGaussians) {
            fvdb::ProjectedGaussianSplats s;
            s.perGaussian2dMean  = means2d;
            s.perGaussianConic   = conics;
            s.perGaussianOpacity = opacities;
            s.tileOffsets        = tileOffsets;
            s.tileGaussianIds    = tileGaussianIds;
            return ops::renderContributingIds(s, settings, numContributingGaussians);
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("settings"),
        py::arg("num_contributing_gaussians"));

    m.def(
        "identify_contributing_gaussians_sparse",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &activeTiles,
           const torch::Tensor &activeTileMask,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const torch::Tensor &inverseIndices,
           const fvdb::JaggedTensor &uniquePixelsToRender,
           const bool hasDuplicates,
           const fvdb::JaggedTensor &pixelsToRender,
           const RenderSettings &settings,
           const std::optional<fvdb::JaggedTensor> &numContributingGaussians) {
            fvdb::SparseProjectedGaussianSplats ss;
            ss.perGaussian2dMean    = means2d;
            ss.perGaussianConic     = conics;
            ss.perGaussianOpacity   = opacities;
            ss.tileOffsets          = tileOffsets;
            ss.tileGaussianIds      = tileGaussianIds;
            ss.activeTiles          = activeTiles;
            ss.activeTileMask       = activeTileMask;
            ss.tilePixelMask        = tilePixelMask;
            ss.tilePixelCumsum      = tilePixelCumsum;
            ss.pixelMap             = pixelMap;
            ss.inverseIndices       = inverseIndices;
            ss.uniquePixelsToRender = uniquePixelsToRender;
            ss.hasDuplicates        = hasDuplicates;
            return ops::sparseRenderContributingIds(ss, pixelsToRender, settings, numContributingGaussians);
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("active_tiles"),
        py::arg("active_tile_mask"),
        py::arg("tile_pixel_mask"),
        py::arg("tile_pixel_cumsum"),
        py::arg("pixel_map"),
        py::arg("inverse_indices"),
        py::arg("unique_pixels_to_render"),
        py::arg("has_duplicates"),
        py::arg("pixels_to_render"),
        py::arg("settings"),
        py::arg("num_contributing_gaussians"));

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
            return ops::gaussianRelocation(
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
            ops::gaussianMCMCAddNoise(means, logScales, logitOpacities, quats, noiseScale, t, k);
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
           std::optional<std::unordered_map<std::string, fvdb::PlyMetadataTypes>> metadata) {
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
                          std::unordered_map<std::string, fvdb::PlyMetadataTypes>> {
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
            return ops::gaussianProjectionForward(means,
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
            return ops::gaussianProjectionBackward(means,
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
            return ops::sphericalHarmonicsForward(
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
            return ops::sphericalHarmonicsBackward(shDegreeToUse,
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
            return ops::gaussianRasterizeForward(means2d,
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
            return ops::gaussianRasterizeBackward(means2d,
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
            return ops::gaussianSparseRasterizeForward(pixelsToRender,
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
            return ops::gaussianSparseRasterizeBackward(pixelsToRender,
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
           const RenderSettings &settings,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::gaussianRasterizeFromWorldForward(means,
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
                                                          settings,
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
        py::arg("settings"),
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
           const RenderSettings &settings,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const torch::Tensor &renderedAlphas,
           const torch::Tensor &lastIds,
           const torch::Tensor &dLossDRenderedFeatures,
           const torch::Tensor &dLossDRenderedAlphas,
           const at::optional<torch::Tensor> &backgrounds,
           const at::optional<torch::Tensor> &masks) {
            return ops::gaussianRasterizeFromWorldBackward(means,
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
                                                           settings,
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
        py::arg("settings"),
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
            return ops::gaussianProjectionJaggedForward(gSizes,
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
            return ops::gaussianProjectionJaggedBackward(gSizes,
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
            return ops::gaussianTileIntersection(
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
            return ops::gaussianSparseTileIntersection(
                means2d, radii, depths, tileMask, activeTiles, cameraIds,
                numCameras, tileSize, numTilesH, numTilesW);
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
            return ops::computeSparseInfo(tileSideLength, numTilesW, numTilesH, pixelsToRender);
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
            return ops::gaussianProjectionForwardUT(means,
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
