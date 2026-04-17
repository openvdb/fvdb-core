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
// The backward binding (projectGaussiansAnalyticBwd) accepts these as
// optional tensors. The Python GaussianSplat3d class owns the accumulators
// and passes them through to the C++ backward dispatch.

#include <pybind11/stl.h>

#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/ops/gsplat/AddNoiseToGaussianMeans.h>
#include <fvdb/detail/ops/gsplat/BuildSparseGaussianTileLayout.h>
#include <fvdb/detail/ops/gsplat/CountContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsBackward.h>
#include <fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsForward.h>
#include <fvdb/detail/ops/gsplat/IdentifyContributingGaussians.h>
#include <fvdb/detail/ops/gsplat/IntersectGaussianTiles.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticBackward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticForward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedBackward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedForward.h>
#include <fvdb/detail/ops/gsplat/ProjectGaussiansUnscentedForward.h>
#include <fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansBackward.h>
#include <fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansForward.h>
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansBackward.h>
#include <fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansForward.h>
#include <fvdb/detail/ops/gsplat/RelocateGaussians.h>

#include <torch/extension.h>

void
bind_gaussian_splat_ops(py::module &m) {
    namespace ops         = fvdb::detail::ops;
    using DistortionModel = fvdb::detail::ops::DistortionModel;

    // -----------------------------------------------------------------------
    // Enum types
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
    // Analysis operations
    // -----------------------------------------------------------------------

    m.def("rasterize_num_contributing_gaussians",
          &ops::countContributingGaussians,
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

    m.def("sparse_rasterize_num_contributing_gaussians",
          &ops::countContributingGaussiansSparse,
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

    m.def("rasterize_contributing_gaussian_ids",
          &ops::identifyContributingGaussians,
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

    m.def("sparse_rasterize_contributing_gaussian_ids",
          &ops::identifyContributingGaussiansSparse,
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
    // MCMC operations
    // -----------------------------------------------------------------------

    m.def("mcmc_relocate_gaussians",
          &ops::relocateGaussians,
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("ratios"),
          py::arg("binomial_coeffs"),
          py::arg("n_max"),
          py::arg("min_opacity"));

    m.def("mcmc_add_noise_to_means",
          &ops::addNoiseToGaussianMeans,
          py::arg("means"),
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("quats"),
          py::arg("noise_scale"),
          py::arg("t"),
          py::arg("k"));

    // -----------------------------------------------------------------------
    // PLY I/O
    // -----------------------------------------------------------------------

    m.def("save_gaussian_ply",
          &fvdb::detail::io::saveGaussianPly,
          py::arg("filename"),
          py::arg("means"),
          py::arg("quats"),
          py::arg("log_scales"),
          py::arg("logit_opacities"),
          py::arg("sh0"),
          py::arg("shN"),
          py::arg("metadata") = py::none());

    m.def("load_gaussian_ply",
          &fvdb::detail::io::loadGaussianPly,
          py::arg("filename"),
          py::arg("device") = torch::kCPU);

    // ------- Raw forward/backward dispatch (for Python autograd) -------

    m.def("project_gaussians_analytic_fwd",
          &ops::projectGaussiansAnalyticFwd,
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

    m.def("project_gaussians_analytic_bwd",
          &ops::projectGaussiansAnalyticBwd,
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

    m.def("evaluate_spherical_harmonics_fwd",
          &ops::evaluateSphericalHarmonicsFwd,
          py::arg("sh_degree_to_use"),
          py::arg("num_cameras"),
          py::arg("view_dirs"),
          py::arg("sh0_coeffs"),
          py::arg("sh_n_coeffs"),
          py::arg("radii"));

    m.def("evaluate_spherical_harmonics_bwd",
          &ops::evaluateSphericalHarmonicsBwd,
          py::arg("sh_degree_to_use"),
          py::arg("num_cameras"),
          py::arg("num_gaussians"),
          py::arg("view_dirs"),
          py::arg("sh_n_coeffs"),
          py::arg("d_loss_d_colors"),
          py::arg("radii"),
          py::arg("compute_d_loss_d_view_dirs"));

    m.def("rasterize_screen_space_gaussians_fwd",
          &ops::rasterizeScreenSpaceGaussiansFwd,
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

    m.def("rasterize_screen_space_gaussians_bwd",
          &ops::rasterizeScreenSpaceGaussiansBwd,
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

    m.def("rasterize_screen_space_gaussians_sparse_fwd",
          &ops::rasterizeScreenSpaceGaussiansSparseFwd,
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

    m.def("rasterize_screen_space_gaussians_sparse_bwd",
          &ops::rasterizeScreenSpaceGaussiansSparseBwd,
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

    m.def("rasterize_world_space_gaussians_fwd",
          &ops::rasterizeWorldSpaceGaussiansFwd,
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

    m.def("rasterize_world_space_gaussians_bwd",
          &ops::rasterizeWorldSpaceGaussiansBwd,
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

    m.def("project_gaussians_analytic_jagged_fwd",
          &ops::projectGaussiansAnalyticJaggedFwd,
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

    m.def("project_gaussians_analytic_jagged_bwd",
          &ops::projectGaussiansAnalyticJaggedBwd,
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
            return ops::intersectGaussianTiles(
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
            return ops::intersectGaussianTilesSparse(means2d,
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

    m.def("build_sparse_gaussian_tile_layout",
          &ops::buildSparseGaussianTileLayout,
          py::arg("tile_side_length"),
          py::arg("num_tiles_w"),
          py::arg("num_tiles_h"),
          py::arg("pixels_to_render"));

    // ------- UT projection forward (non-differentiable) -------

    m.def(
        "project_gaussians_unscented_fwd",
        [](const torch::Tensor &means,
           const torch::Tensor &quats,
           const torch::Tensor &logScales,
           const torch::Tensor &worldToCamMatricesStart,
           const torch::Tensor &worldToCamMatricesEnd,
           const torch::Tensor &projectionMatrices,
           const torch::Tensor &distortionCoeffs,
           const DistortionModel cameraModel,
           const int64_t imageWidth,
           const int64_t imageHeight,
           const float eps2d,
           const float nearPlane,
           const float farPlane,
           const float minRadius2d,
           const bool calcCompensations,
           const ops::RollingShutterType rollingShutterType,
           const float utAlpha,
           const float utBeta,
           const float utKappa,
           const float utInImageMargin,
           const bool utRequireAllSigmaPointsInImage) {
            ops::UTParams utParams{
                utAlpha, utBeta, utKappa, utInImageMargin, utRequireAllSigmaPointsInImage};
            return ops::projectGaussiansUnscentedFwd(means,
                                                     quats,
                                                     logScales,
                                                     worldToCamMatricesStart,
                                                     worldToCamMatricesEnd,
                                                     projectionMatrices,
                                                     rollingShutterType,
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
        py::arg("world_to_cam_matrices_start"),
        py::arg("world_to_cam_matrices_end"),
        py::arg("projection_matrices"),
        py::arg("distortion_coeffs"),
        py::arg("camera_model"),
        py::arg("image_width"),
        py::arg("image_height"),
        py::arg("eps2d"),
        py::arg("near"),
        py::arg("far"),
        py::arg("min_radius_2d"),
        py::arg("calc_compensations"),
        py::arg("rolling_shutter_type")                 = ops::RollingShutterType::NONE,
        py::arg("ut_alpha")                             = 0.1f,
        py::arg("ut_beta")                              = 2.0f,
        py::arg("ut_kappa")                             = 0.0f,
        py::arg("ut_in_image_margin")                   = 0.1f,
        py::arg("ut_require_all_sigma_points_in_image") = true);
}
