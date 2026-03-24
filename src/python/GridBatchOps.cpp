// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>

#include <torch/extension.h>

// Interpolation
#include <fvdb/detail/ops/SampleGridBezier.h>
#include <fvdb/detail/ops/SampleGridBezierWithGrad.h>
#include <fvdb/detail/ops/SampleGridBezierWithGradBackward.h>
#include <fvdb/detail/ops/SampleGridTrilinear.h>
#include <fvdb/detail/ops/SampleGridTrilinearWithGrad.h>
#include <fvdb/detail/ops/SampleGridTrilinearWithGradBackward.h>
#include <fvdb/detail/ops/SplatIntoGridBezier.h>
#include <fvdb/detail/ops/SplatIntoGridTrilinear.h>

// Transforms
#include <fvdb/detail/ops/TransformPointToGrid.h>

// Pooling
#include <fvdb/detail/ops/DownsampleGridAvgPool.h>
#include <fvdb/detail/ops/DownsampleGridMaxPool.h>
#include <fvdb/detail/ops/UpsampleGridNearest.h>

// Dense I/O
#include <fvdb/detail/ops/Inject.h>
#include <fvdb/detail/ops/ReadFromDense.h>
#include <fvdb/detail/ops/ReadIntoDense.h>

// Spatial queries
#include <fvdb/detail/ops/ActiveGridGoords.h>
#include <fvdb/detail/ops/CoordsInGrid.h>
#include <fvdb/detail/ops/CubesInGrid.h>
#include <fvdb/detail/ops/IjkToIndex.h>
#include <fvdb/detail/ops/PointsInGrid.h>
#include <fvdb/detail/ops/VoxelNeighborhood.h>

// Rays
#include <fvdb/detail/ops/RayImplicitIntersection.h>
#include <fvdb/detail/ops/SampleRaysUniform.h>
#include <fvdb/detail/ops/SegmentsAlongRays.h>
#include <fvdb/detail/ops/VoxelsAlongRays.h>

// Meshing / TSDF
#include <fvdb/detail/ops/IntegrateTSDF.h>
#include <fvdb/detail/ops/MarchingCubes.h>

// Topology / misc
#include <fvdb/detail/ops/GridEdgeNetwork.h>
#include <fvdb/detail/ops/SerializeEncode.h>

namespace {

nanovdb::Coord
vecToCoord(const std::vector<int32_t> &v) {
    TORCH_CHECK_VALUE(v.size() == 3, "Expected a list of 3 integers, got ", v.size());
    return nanovdb::Coord(v[0], v[1], v[2]);
}

nanovdb::Vec3d
vecToVec3d(const std::vector<double> &v) {
    TORCH_CHECK_VALUE(v.size() == 3, "Expected a list of 3 doubles, got ", v.size());
    return nanovdb::Vec3d(v[0], v[1], v[2]);
}

} // namespace

void
bind_grid_batch_ops(py::module &m) {
    using GBI = fvdb::detail::GridBatchImpl;
    using JT  = fvdb::JaggedTensor;
    namespace ops = fvdb::detail::ops;

    // -----------------------------------------------------------------------
    // Interpolation: forward
    // -----------------------------------------------------------------------

    m.def("sample_trilinear_fwd",
          &ops::sampleGridTrilinear,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("sample_bezier_fwd",
          &ops::sampleGridBezier,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("splat_trilinear_fwd",
          &ops::splatIntoGridTrilinear,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("splat_bezier_fwd",
          &ops::splatIntoGridBezier,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    // Interpolation: with-gradient forward
    m.def("sample_trilinear_with_grad_fwd",
          &ops::sampleGridTrilinearWithGrad,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("sample_bezier_with_grad_fwd",
          &ops::sampleGridBezierWithGrad,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    // Interpolation: with-gradient backward
    m.def("sample_trilinear_with_grad_bwd",
          &ops::sampleGridTrilinearWithGradBackward,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"),
          py::arg("grad_out_features"),
          py::arg("grad_out_grad_features"));

    m.def("sample_bezier_with_grad_bwd",
          &ops::sampleGridBezierWithGradBackward,
          py::arg("grid"),
          py::arg("points"),
          py::arg("grad_out_features"),
          py::arg("grad_out_grad_features"),
          py::arg("data"));

    // -----------------------------------------------------------------------
    // Transforms
    // -----------------------------------------------------------------------

    m.def("transform_points_fwd",
          &ops::transformPointsToGrid,
          py::arg("grid"),
          py::arg("points"),
          py::arg("is_primal"));

    m.def("inv_transform_points_fwd",
          &ops::invTransformPointsToGrid,
          py::arg("grid"),
          py::arg("points"),
          py::arg("is_primal"));

    m.def("transform_points_bwd",
          &ops::transformPointsToGridBackward,
          py::arg("grid"),
          py::arg("grad_out"),
          py::arg("is_primal"));

    m.def("inv_transform_points_bwd",
          &ops::invTransformPointsToGridBackward,
          py::arg("grid"),
          py::arg("grad_out"),
          py::arg("is_primal"));

    // -----------------------------------------------------------------------
    // Pooling
    // -----------------------------------------------------------------------

    m.def(
        "max_pool_fwd",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::downsampleGridMaxPool(fine, coarse, data, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("fine_grid"),
        py::arg("coarse_grid"),
        py::arg("data"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "max_pool_bwd",
        [](const GBI &coarse,
           const GBI &fine,
           const torch::Tensor &fineData,
           const torch::Tensor &coarseGradOut,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::downsampleGridMaxPoolBackward(
                coarse, fine, fineData, coarseGradOut, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("fine_data"),
        py::arg("coarse_grad_out"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "avg_pool_fwd",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::downsampleGridAvgPool(fine, coarse, data, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("fine_grid"),
        py::arg("coarse_grid"),
        py::arg("data"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "avg_pool_bwd",
        [](const GBI &coarse,
           const GBI &fine,
           const torch::Tensor &fineData,
           const torch::Tensor &coarseGradOut,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::downsampleGridAvgPoolBackward(
                coarse, fine, fineData, coarseGradOut, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("fine_data"),
        py::arg("coarse_grad_out"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "upsample_nearest_fwd",
        [](const GBI &coarse,
           const GBI &fine,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor) {
            return ops::upsampleGridNearest(coarse, fine, data, vecToCoord(factor));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("data"),
        py::arg("factor"));

    m.def(
        "upsample_nearest_bwd",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &gradOut,
           const torch::Tensor &coarseData,
           const std::vector<int32_t> &factor) {
            return ops::upsampleGridNearestBackward(fine, coarse, gradOut, coarseData, vecToCoord(factor));
        },
        py::arg("fine_grid"),
        py::arg("coarse_grid"),
        py::arg("grad_out"),
        py::arg("coarse_data"),
        py::arg("factor"));

    // -----------------------------------------------------------------------
    // Dense I/O
    // -----------------------------------------------------------------------

    m.def("inject_op", &ops::inject, py::arg("dst_grid"), py::arg("src_grid"), py::arg("dst"), py::arg("src"));

    m.def("read_from_dense_cminor",
          &ops::readFromDenseCminor,
          py::arg("grid"),
          py::arg("dense_data"),
          py::arg("origins"));

    m.def("read_from_dense_cmajor",
          &ops::readFromDenseCmajor,
          py::arg("grid"),
          py::arg("dense_data"),
          py::arg("origins"));

    m.def(
        "write_to_dense_cminor",
        [](const GBI &grid,
           const torch::Tensor &sparseData,
           const torch::Tensor &origins,
           const std::vector<int32_t> &gridSize) {
            return ops::readIntoDenseCminor(grid, sparseData, origins, vecToCoord(gridSize));
        },
        py::arg("grid"),
        py::arg("sparse_data"),
        py::arg("origins"),
        py::arg("grid_size"));

    m.def(
        "write_to_dense_cmajor",
        [](const GBI &grid,
           const torch::Tensor &sparseData,
           const torch::Tensor &origins,
           const std::vector<int32_t> &gridSize) {
            return ops::readIntoDenseCmajor(grid, sparseData, origins, vecToCoord(gridSize));
        },
        py::arg("grid"),
        py::arg("sparse_data"),
        py::arg("origins"),
        py::arg("grid_size"));

    // -----------------------------------------------------------------------
    // Spatial queries (non-differentiable)
    // -----------------------------------------------------------------------

    m.def("points_in_grid", &ops::pointsInGrid, py::arg("grid"), py::arg("points"));

    m.def("coords_in_grid", &ops::coordsInGrid, py::arg("grid"), py::arg("coords"));

    m.def(
        "cubes_in_grid",
        [](const GBI &grid,
           const JT &cubeCenters,
           const std::vector<double> &padMin,
           const std::vector<double> &padMax) {
            return ops::cubesInGrid(grid, cubeCenters, vecToVec3d(padMin), vecToVec3d(padMax));
        },
        py::arg("grid"),
        py::arg("cube_centers"),
        py::arg("pad_min"),
        py::arg("pad_max"));

    m.def(
        "cubes_intersect_grid",
        [](const GBI &grid,
           const JT &cubeCenters,
           const std::vector<double> &padMin,
           const std::vector<double> &padMax) {
            return ops::cubesIntersectGrid(grid, cubeCenters, vecToVec3d(padMin), vecToVec3d(padMax));
        },
        py::arg("grid"),
        py::arg("cube_centers"),
        py::arg("pad_min"),
        py::arg("pad_max"));

    m.def("ijk_to_index", &ops::ijkToIndex, py::arg("grid"), py::arg("ijk"), py::arg("cumulative"));

    m.def("voxel_neighborhood",
          &ops::voxelNeighborhood,
          py::arg("grid"),
          py::arg("coords"),
          py::arg("extent"),
          py::arg("shift"));

    m.def("active_grid_coords", &ops::activeGridCoords, py::arg("grid"));

    // -----------------------------------------------------------------------
    // Ray ops (non-differentiable)
    // -----------------------------------------------------------------------

    m.def("voxels_along_rays",
          &ops::voxelsAlongRays,
          py::arg("grid"),
          py::arg("ray_origins"),
          py::arg("ray_directions"),
          py::arg("max_voxels"),
          py::arg("eps"),
          py::arg("return_ijk"),
          py::arg("cumulative"));

    m.def("segments_along_rays",
          &ops::segmentsAlongRays,
          py::arg("grid"),
          py::arg("ray_origins"),
          py::arg("ray_directions"),
          py::arg("max_segments"),
          py::arg("eps"));

    m.def("uniform_ray_samples",
          &ops::uniformRaySamples,
          py::arg("grid"),
          py::arg("ray_origins"),
          py::arg("ray_directions"),
          py::arg("t_min"),
          py::arg("t_max"),
          py::arg("min_step_size"),
          py::arg("cone_angle"),
          py::arg("include_end_segments"),
          py::arg("return_midpoint"),
          py::arg("eps"));

    m.def("ray_implicit_intersection",
          &ops::rayImplicitIntersection,
          py::arg("grid"),
          py::arg("ray_origins"),
          py::arg("ray_directions"),
          py::arg("grid_scalars"),
          py::arg("eps"));

    // -----------------------------------------------------------------------
    // Meshing / TSDF
    // -----------------------------------------------------------------------

    m.def("marching_cubes", &ops::marchingCubes, py::arg("grid"), py::arg("field"), py::arg("level"));

    m.def("integrate_tsdf",
          &ops::integrateTSDF,
          py::arg("grid"),
          py::arg("truncation_margin"),
          py::arg("projection_matrices"),
          py::arg("cam_to_world_matrices"),
          py::arg("tsdf"),
          py::arg("weights"),
          py::arg("depth_images"),
          py::arg("weight_images"));

    m.def("integrate_tsdf_with_features",
          &ops::integrateTSDFWithFeatures,
          py::arg("grid"),
          py::arg("truncation_margin"),
          py::arg("projection_matrices"),
          py::arg("cam_to_world_matrices"),
          py::arg("tsdf"),
          py::arg("features"),
          py::arg("weights"),
          py::arg("depth_images"),
          py::arg("feature_images"),
          py::arg("weight_images"));

    // -----------------------------------------------------------------------
    // Topology / misc
    // -----------------------------------------------------------------------

    m.def("grid_edge_network",
          &ops::gridEdgeNetwork,
          py::arg("grid"),
          py::arg("return_voxel_coordinates"));

    m.def(
        "serialize_encode",
        [](const GBI &grid, const std::string &order, const std::vector<int32_t> &offset) {
            fvdb::SpaceFillingCurveType orderType;
            if (order == "z" || order == "zorder" || order == "morton") {
                orderType = fvdb::SpaceFillingCurveType::ZOrder;
            } else if (order == "z_transposed" || order == "morton_zyx") {
                orderType = fvdb::SpaceFillingCurveType::ZOrderTransposed;
            } else if (order == "hilbert") {
                orderType = fvdb::SpaceFillingCurveType::Hilbert;
            } else if (order == "hilbert_transposed" || order == "hilbert_zyx") {
                orderType = fvdb::SpaceFillingCurveType::HilbertTransposed;
            } else {
                TORCH_CHECK_VALUE(false,
                                  "Unknown order type '",
                                  order,
                                  "'. Expected: z, morton, z_transposed, morton_zyx, "
                                  "hilbert, hilbert_transposed, hilbert_zyx");
            }
            return ops::serializeEncode(grid, orderType, vecToCoord(offset));
        },
        py::arg("grid"),
        py::arg("order"),
        py::arg("offset"));
}
