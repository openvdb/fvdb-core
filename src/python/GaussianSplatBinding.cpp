// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"

#include <fvdb/FVDB.h>
#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/ops/gsplat/GaussianCameras.cuh>
#include <fvdb/detail/ops/gsplat/GaussianMCMCAddNoise.h>
#include <fvdb/detail/ops/gsplat/GaussianMCMCRelocation.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionJaggedForward.h>
#include <fvdb/detail/ops/gsplat/GaussianProjectionUT.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeFromWorldForward.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeNumContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeTopContributingGaussianIds.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/GaussianSplatSparse.h>
#include <fvdb/detail/ops/gsplat/GaussianTileIntersection.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/extension.h>

void
bind_gaussian_splat3d(py::module &m) {
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

    namespace ops        = fvdb::detail::ops;
    using RenderSettings = fvdb::detail::ops::RenderSettings;

    // -----------------------------------------------------------------------
    // Analysis rasterization dispatch functions
    // -----------------------------------------------------------------------

    // 15. rasterize_num_contributing_gaussians (non-differentiable)
    m.def(
        "rasterize_num_contributing_gaussians",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize) {
            ops::RenderSettings settings;
            settings.imageWidth  = imageWidth;
            settings.imageHeight = imageHeight;
            settings.tileSize    = tileSize;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianRasterizeNumContributingGaussians<DeviceTag>(
                    means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);
            });
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("tile_size"));

    // 16. sparse_rasterize_num_contributing_gaussians (non-differentiable)
    m.def(
        "sparse_rasterize_num_contributing_gaussians",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize) {
            ops::RenderSettings settings;
            settings.imageWidth  = imageWidth;
            settings.imageHeight = imageHeight;
            settings.tileSize    = tileSize;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseRasterizeNumContributingGaussians<DeviceTag>(
                    means2d,
                    conics,
                    opacities,
                    tileOffsets,
                    tileGaussianIds,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap,
                    settings);
            });
        },
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
        py::arg("tile_size"));

    // 17. rasterize_contributing_gaussian_ids (non-differentiable)
    m.def(
        "rasterize_contributing_gaussian_ids",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize,
           const at::optional<torch::Tensor> &numContributingGaussians,
           const int numDepthSamples) {
            ops::RenderSettings settings;
            settings.imageWidth      = imageWidth;
            settings.imageHeight     = imageHeight;
            settings.tileSize        = tileSize;
            settings.numDepthSamples = numDepthSamples;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianRasterizeContributingGaussianIds<DeviceTag>(
                    means2d,
                    conics,
                    opacities,
                    tileOffsets,
                    tileGaussianIds,
                    settings,
                    numContributingGaussians);
            });
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("tile_size"),
        py::arg("num_contributing_gaussians") = py::none(),
        py::arg("num_depth_samples")          = -1);

    // 18. rasterize_top_contributing_gaussian_ids (non-differentiable)
    m.def(
        "rasterize_top_contributing_gaussian_ids",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize,
           const int topK) {
            ops::RenderSettings settings;
            settings.imageWidth      = imageWidth;
            settings.imageHeight     = imageHeight;
            settings.tileSize        = tileSize;
            settings.numDepthSamples = topK;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianRasterizeTopContributingGaussianIds<DeviceTag>(
                    means2d, conics, opacities, tileOffsets, tileGaussianIds, settings);
            });
        },
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("opacities"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("tile_size"),
        py::arg("top_k"));

    // 19. sparse_rasterize_contributing_gaussian_ids (non-differentiable)
    m.def(
        "sparse_rasterize_contributing_gaussian_ids",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize,
           const at::optional<fvdb::JaggedTensor> &numContributingGaussians,
           const int numDepthSamples) {
            ops::RenderSettings settings;
            settings.imageWidth      = imageWidth;
            settings.imageHeight     = imageHeight;
            settings.tileSize        = tileSize;
            settings.numDepthSamples = numDepthSamples;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseRasterizeContributingGaussianIds<DeviceTag>(
                    means2d,
                    conics,
                    opacities,
                    tileOffsets,
                    tileGaussianIds,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap,
                    settings,
                    numContributingGaussians);
            });
        },
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
        py::arg("tile_size"),
        py::arg("num_contributing_gaussians") = py::none(),
        py::arg("num_depth_samples")          = -1);

    // 19b. sparse_rasterize_top_contributing_gaussian_ids (non-differentiable)
    m.def(
        "sparse_rasterize_top_contributing_gaussian_ids",
        [](const torch::Tensor &means2d,
           const torch::Tensor &conics,
           const torch::Tensor &opacities,
           const torch::Tensor &tileOffsets,
           const torch::Tensor &tileGaussianIds,
           const fvdb::JaggedTensor &pixelsToRender,
           const torch::Tensor &activeTiles,
           const torch::Tensor &tilePixelMask,
           const torch::Tensor &tilePixelCumsum,
           const torch::Tensor &pixelMap,
           const uint32_t imageWidth,
           const uint32_t imageHeight,
           const uint32_t tileSize,
           const int topK) {
            ops::RenderSettings settings;
            settings.imageWidth      = imageWidth;
            settings.imageHeight     = imageHeight;
            settings.tileSize        = tileSize;
            settings.numDepthSamples = topK;
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseRasterizeTopContributingGaussianIds<DeviceTag>(
                    means2d,
                    conics,
                    opacities,
                    tileOffsets,
                    tileGaussianIds,
                    pixelsToRender,
                    activeTiles,
                    tilePixelMask,
                    tilePixelCumsum,
                    pixelMap,
                    settings);
            });
        },
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
        py::arg("tile_size"),
        py::arg("top_k"));

    // -----------------------------------------------------------------------
    // MCMC helpers
    // -----------------------------------------------------------------------

    // 20. mcmc_relocate_gaussians
    m.def(
        "mcmc_relocate_gaussians",
        [](const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &ratios,
           const torch::Tensor &binomialCoeffs,
           const int nMax,
           const float minOpacity) -> std::tuple<torch::Tensor, torch::Tensor> {
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

    // 21. mcmc_add_noise_to_means (in-place)
    m.def(
        "mcmc_add_noise_to_means",
        [](torch::Tensor &means,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &quats,
           const float noiseScale,
           const float t,
           const float k) -> void {
            FVDB_DISPATCH_KERNEL(means.device(), [&]() -> int {
                ops::dispatchGaussianMCMCAddNoise<DeviceTag>(
                    means, logScales, logitOpacities, quats, noiseScale, t, k);
                return 0;
            });
        },
        py::arg("means"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("quats"),
        py::arg("noise_scale"),
        py::arg("t") = 0.005,
        py::arg("k") = 100.0);

    // -----------------------------------------------------------------------
    // PLY I/O (standalone, no GaussianSplat3d class dependency)
    // -----------------------------------------------------------------------

    // 22. save_gaussian_ply
    m.def(
        "save_gaussian_ply",
        [](const std::string &filename,
           const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &logitOpacities,
           const torch::Tensor &sh0,
           const torch::Tensor &shN,
           const std::optional<std::unordered_map<std::string, fvdb::detail::io::PlyMetadataTypes>>
               &metadata) {
            fvdb::detail::io::saveGaussianPly(
                filename, means, quats, logScales, logitOpacities, sh0, shN, metadata);
        },
        py::arg("filename"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("log_scales"),
        py::arg("logit_opacities"),
        py::arg("sh0"),
        py::arg("shN"),
        py::arg("metadata") = py::none());

    // 23. load_gaussian_ply -- returns (means, quats, logScales, logitOpacities, sh0, shN,
    // metadata)
    m.def("load_gaussian_ply",
          &fvdb::detail::io::loadGaussianPly,
          py::arg("filename"),
          py::arg("device") = torch::kCPU);

    // -----------------------------------------------------------------------
    // Raw forward/backward dispatch functions (for Python autograd wrappers)
    // -----------------------------------------------------------------------

    using RenderWindow2D  = fvdb::detail::ops::RenderWindow2D;
    using DistortionModel = fvdb::detail::ops::DistortionModel;

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
            return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianProjectionForward<DeviceTag>(means,
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
            return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianProjectionBackward<DeviceTag>(
                    means,
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
        py::arg("out_normalized_max_radii_accum")             = py::none(),
        py::arg("out_gradient_step_counts")                   = py::none());

    // 3. project_gaussians_analytic_jagged_fwd
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
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianProjectionJaggedForward<DeviceTag>(gSizes,
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
        py::arg("near"),
        py::arg("far"),
        py::arg("min_radius_2d"),
        py::arg("ortho"));

    // 4. project_gaussians_analytic_jagged_bwd
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
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianProjectionJaggedBackward<DeviceTag>(
                    gSizes,
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

    // 5. eval_gaussian_sh_fwd
    m.def(
        "eval_gaussian_sh_fwd",
        [](const int64_t shDegreeToUse,
           const int64_t numCameras,
           const at::optional<torch::Tensor> &viewDirs,
           const torch::Tensor &sh0Coeffs,
           const at::optional<torch::Tensor> &shNCoeffs,
           const torch::Tensor &radii) {
            const torch::Tensor vd  = viewDirs.value_or(torch::Tensor());
            const torch::Tensor shn = shNCoeffs.value_or(torch::Tensor());
            return FVDB_DISPATCH_KERNEL(sh0Coeffs.device(), [&]() {
                return ops::dispatchSphericalHarmonicsForward<DeviceTag>(
                    shDegreeToUse, numCameras, vd, sh0Coeffs, shn, radii);
            });
        },
        py::arg("sh_degree_to_use"),
        py::arg("num_cameras"),
        py::arg("view_dirs"),
        py::arg("sh0_coeffs"),
        py::arg("sh_n_coeffs"),
        py::arg("radii"));

    // 6. eval_gaussian_sh_bwd
    m.def(
        "eval_gaussian_sh_bwd",
        [](const int64_t shDegreeToUse,
           const int64_t numCameras,
           const int64_t numGaussians,
           const at::optional<torch::Tensor> &viewDirs,
           const at::optional<torch::Tensor> &shNCoeffs,
           const torch::Tensor &dLossDColors,
           const torch::Tensor &radii,
           const bool computeDLossDViewDirs) {
            const torch::Tensor vd  = viewDirs.value_or(torch::Tensor());
            const torch::Tensor shn = shNCoeffs.value_or(torch::Tensor());
            return FVDB_DISPATCH_KERNEL(dLossDColors.device(), [&]() {
                return ops::dispatchSphericalHarmonicsBackward<DeviceTag>(shDegreeToUse,
                                                                          numCameras,
                                                                          numGaussians,
                                                                          vd,
                                                                          shn,
                                                                          dLossDColors,
                                                                          radii,
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

    // 7. rasterize_screen_space_gaussians_fwd
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
            RenderWindow2D rw{imageWidth, imageHeight, imageOriginW, imageOriginH};
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                return ops::dispatchGaussianRasterizeForward<DeviceTag>(means2d,
                                                                        conics,
                                                                        features,
                                                                        opacities,
                                                                        rw,
                                                                        tileSize,
                                                                        tileOffsets,
                                                                        tileGaussianIds,
                                                                        backgrounds,
                                                                        masks);
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

    // 8. rasterize_screen_space_gaussians_bwd
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
            RenderWindow2D rw{imageWidth, imageHeight, imageOriginW, imageOriginH};
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                return ops::dispatchGaussianRasterizeBackward<DeviceTag>(means2d,
                                                                         conics,
                                                                         features,
                                                                         opacities,
                                                                         rw,
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
        py::arg("backgrounds")                  = py::none(),
        py::arg("masks")                        = py::none());

    // 9. rasterize_screen_space_gaussians_sparse_fwd
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
            RenderWindow2D rw{imageWidth, imageHeight, imageOriginW, imageOriginH};
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseRasterizeForward<DeviceTag>(pixelsToRender,
                                                                              means2d,
                                                                              conics,
                                                                              features,
                                                                              opacities,
                                                                              rw,
                                                                              tileSize,
                                                                              tileOffsets,
                                                                              tileGaussianIds,
                                                                              activeTiles,
                                                                              tilePixelMask,
                                                                              tilePixelCumsum,
                                                                              pixelMap,
                                                                              backgrounds,
                                                                              masks);
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

    // 10. rasterize_screen_space_gaussians_sparse_bwd
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
            RenderWindow2D rw{imageWidth, imageHeight, imageOriginW, imageOriginH};
            return FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseRasterizeBackward<DeviceTag>(
                    pixelsToRender,
                    means2d,
                    conics,
                    features,
                    opacities,
                    rw,
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
        py::arg("backgrounds")                  = py::none(),
        py::arg("masks")                        = py::none());

    // 11. rasterize_world_space_gaussians_fwd
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
           const fvdb::detail::ops::RollingShutterType rollingShutterType,
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
            RenderSettings settings;
            settings.imageWidth   = imageWidth;
            settings.imageHeight  = imageHeight;
            settings.imageOriginW = imageOriginW;
            settings.imageOriginH = imageOriginH;
            settings.tileSize     = tileSize;
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianRasterizeFromWorld3DGSForward<DeviceTag>(
                    means,
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
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("image_origin_w"),
        py::arg("image_origin_h"),
        py::arg("tile_size"),
        py::arg("tile_offsets"),
        py::arg("tile_gaussian_ids"),
        py::arg("backgrounds"),
        py::arg("masks"));

    // 12. rasterize_world_space_gaussians_bwd
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
           const fvdb::detail::ops::RollingShutterType rollingShutterType,
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
            RenderSettings settings;
            settings.imageWidth   = imageWidth;
            settings.imageHeight  = imageHeight;
            settings.imageOriginW = imageOriginW;
            settings.imageOriginH = imageOriginH;
            settings.tileSize     = tileSize;
            return FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
                return ops::dispatchGaussianRasterizeFromWorld3DGSBackward<DeviceTag>(
                    means,
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

    // 13. intersect_gaussian_tiles (non-differentiable)
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
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                return ops::dispatchGaussianTileIntersection<DeviceTag>(
                    means2d, radii, depths, cameraIds, numCameras, tileSize, numTilesH, numTilesW);
            });
        },
        py::arg("means2d"),
        py::arg("radii"),
        py::arg("depths"),
        py::arg("num_cameras"),
        py::arg("tile_size"),
        py::arg("num_tiles_h"),
        py::arg("num_tiles_w"),
        py::arg("camera_ids") = py::none());

    // 14. intersect_gaussian_tiles_sparse (non-differentiable)
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
            return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
                return ops::dispatchGaussianSparseTileIntersection<DeviceTag>(means2d,
                                                                              radii,
                                                                              depths,
                                                                              tileMask,
                                                                              activeTiles,
                                                                              cameraIds,
                                                                              numCameras,
                                                                              tileSize,
                                                                              numTilesH,
                                                                              numTilesW);
            });
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

    // 15. build_sparse_gaussian_tile_layout (non-differentiable)
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

    // 16. project_gaussians_ut_fwd (forward-only, non-differentiable)
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
            return FVDB_DISPATCH_KERNEL(means.device(), [&]() {
                return ops::dispatchGaussianProjectionForwardUT<DeviceTag>(
                    means,
                    quats,
                    logScales,
                    worldToCamMatrices,
                    worldToCamMatrices,
                    projectionMatrices,
                    fvdb::detail::ops::RollingShutterType::NONE,
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
            });
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
