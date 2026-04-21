// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>

#include <torch/extension.h>

// Interpolation
#include <fvdb/detail/ops/SampleBezier.h>
#include <fvdb/detail/ops/SampleBezierWithGrad.h>
#include <fvdb/detail/ops/SampleBezierWithGradBackward.h>
#include <fvdb/detail/ops/SampleTrilinear.h>
#include <fvdb/detail/ops/SampleTrilinearWithGrad.h>
#include <fvdb/detail/ops/SampleTrilinearWithGradBackward.h>
#include <fvdb/detail/ops/SplatBezier.h>
#include <fvdb/detail/ops/SplatTrilinear.h>

// Transforms
#include <fvdb/detail/ops/VoxelToWorld.h>

// Pooling
#include <fvdb/detail/ops/AvgPool.h>
#include <fvdb/detail/ops/MaxPool.h>
#include <fvdb/detail/ops/Refine.h>

// Dense I/O
#include <fvdb/detail/ops/Inject.h>
#include <fvdb/detail/ops/InjectFromDense.h>
#include <fvdb/detail/ops/InjectToDense.h>

// Spatial queries
#include <fvdb/detail/ops/ActiveGridCoords.h>
#include <fvdb/detail/ops/CoordsInGrid.h>
#include <fvdb/detail/ops/CubesInGrid.h>
#include <fvdb/detail/ops/IjkToIndex.h>
#include <fvdb/detail/ops/IjkToInvIndex.h>
#include <fvdb/detail/ops/NeighborIndexes.h>
#include <fvdb/detail/ops/PointsInGrid.h>

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

// Grid construction / topology / batch ops
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildCoarseGridFromFine.h>
#include <fvdb/detail/ops/BuildDenseGrid.h>
#include <fvdb/detail/ops/BuildDilatedGrid.h>
#include <fvdb/detail/ops/BuildFineGridFromCoarse.h>
#include <fvdb/detail/ops/BuildGridForConv.h>
#include <fvdb/detail/ops/BuildGridForConvTranspose.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromMesh.h>
#include <fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/ops/BuildMergedGrids.h>
#include <fvdb/detail/ops/BuildPaddedGrid.h>
#include <fvdb/detail/ops/BuildPrunedGrid.h>
#include <fvdb/detail/ops/ClipGrid.h>
#include <fvdb/detail/ops/CloneGrid.h>
#include <fvdb/detail/ops/ConcatenateGrids.h>
#include <fvdb/detail/ops/IndexGrid.h>
#include <fvdb/detail/ops/MakeContiguous.h>
#include <fvdb/detail/ops/SerializeGrid.h>

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

std::vector<nanovdb::Vec3d>
vecsToVec3ds(const std::vector<std::vector<double>> &vecs) {
    std::vector<nanovdb::Vec3d> result;
    result.reserve(vecs.size());
    for (const auto &v: vecs) {
        result.push_back(vecToVec3d(v));
    }
    return result;
}

std::vector<nanovdb::Coord>
vecsToCoords(const std::vector<std::vector<int32_t>> &vecs) {
    std::vector<nanovdb::Coord> result;
    result.reserve(vecs.size());
    for (const auto &v: vecs) {
        result.push_back(vecToCoord(v));
    }
    return result;
}

} // namespace

void
bind_grid_batch_ops(py::module &m) {
    using GBI     = fvdb::GridBatchData;
    using JT      = fvdb::JaggedTensor;
    namespace ops = fvdb::detail::ops;

    // -----------------------------------------------------------------------
    // Interpolation: forward
    // -----------------------------------------------------------------------

    m.def("sample_trilinear",
          &ops::sampleTrilinear,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("sample_bezier", &ops::sampleBezier, py::arg("grid"), py::arg("points"), py::arg("data"));

    m.def("splat_trilinear",
          &ops::splatTrilinear,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("splat_bezier", &ops::splatBezier, py::arg("grid"), py::arg("points"), py::arg("data"));

    // Interpolation: with-gradient forward
    m.def("sample_trilinear_with_grad",
          &ops::sampleTrilinearWithGrad,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    m.def("sample_bezier_with_grad",
          &ops::sampleBezierWithGrad,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"));

    // Interpolation: with-gradient backward
    m.def("sample_trilinear_with_grad_bwd",
          &ops::sampleTrilinearWithGradBackward,
          py::arg("grid"),
          py::arg("points"),
          py::arg("data"),
          py::arg("grad_out_features"),
          py::arg("grad_out_grad_features"));

    m.def("sample_bezier_with_grad_bwd",
          &ops::sampleBezierWithGradBackward,
          py::arg("grid"),
          py::arg("points"),
          py::arg("grad_out_features"),
          py::arg("grad_out_grad_features"),
          py::arg("data"));

    // -----------------------------------------------------------------------
    // Transforms
    // -----------------------------------------------------------------------

    m.def("voxel_to_world",
          &ops::voxelToWorld,
          py::arg("grid"),
          py::arg("points"),
          py::arg("is_primal"));

    m.def("world_to_voxel",
          &ops::worldToVoxel,
          py::arg("grid"),
          py::arg("points"),
          py::arg("is_primal"));

    m.def("voxel_to_world_bwd",
          &ops::voxelToWorldBackward,
          py::arg("grid"),
          py::arg("grad_out"),
          py::arg("is_primal"));

    m.def("world_to_voxel_bwd",
          &ops::worldToVoxelBackward,
          py::arg("grid"),
          py::arg("grad_out"),
          py::arg("is_primal"));

    // -----------------------------------------------------------------------
    // Pooling
    // -----------------------------------------------------------------------

    m.def(
        "max_pool",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::maxPool(fine, coarse, data, vecToCoord(factor), vecToCoord(stride));
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
            return ops::maxPoolBackward(
                coarse, fine, fineData, coarseGradOut, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("fine_data"),
        py::arg("coarse_grad_out"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "avg_pool",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor,
           const std::vector<int32_t> &stride) {
            return ops::avgPool(fine, coarse, data, vecToCoord(factor), vecToCoord(stride));
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
            return ops::avgPoolBackward(
                coarse, fine, fineData, coarseGradOut, vecToCoord(factor), vecToCoord(stride));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("fine_data"),
        py::arg("coarse_grad_out"),
        py::arg("factor"),
        py::arg("stride"));

    m.def(
        "refine",
        [](const GBI &coarse,
           const GBI &fine,
           const torch::Tensor &data,
           const std::vector<int32_t> &factor) {
            return ops::refine(coarse, fine, data, vecToCoord(factor));
        },
        py::arg("coarse_grid"),
        py::arg("fine_grid"),
        py::arg("data"),
        py::arg("factor"));

    m.def(
        "refine_bwd",
        [](const GBI &fine,
           const GBI &coarse,
           const torch::Tensor &gradOut,
           const torch::Tensor &coarseData,
           const std::vector<int32_t> &factor) {
            return ops::refineBackward(fine, coarse, gradOut, coarseData, vecToCoord(factor));
        },
        py::arg("fine_grid"),
        py::arg("coarse_grid"),
        py::arg("grad_out"),
        py::arg("coarse_data"),
        py::arg("factor"));

    // -----------------------------------------------------------------------
    // Dense I/O
    // -----------------------------------------------------------------------

    m.def("inject_op",
          &ops::inject,
          py::arg("dst_grid"),
          py::arg("src_grid"),
          py::arg("dst"),
          py::arg("src"));

    m.def("inject_from_dense_cminor",
          &ops::injectFromDenseCminor,
          py::arg("grid"),
          py::arg("dense_data"),
          py::arg("origins"));

    m.def("inject_from_dense_cmajor",
          &ops::injectFromDenseCmajor,
          py::arg("grid"),
          py::arg("dense_data"),
          py::arg("origins"));

    m.def(
        "inject_to_dense_cminor",
        [](const GBI &grid,
           const torch::Tensor &sparseData,
           const torch::Tensor &origins,
           const std::vector<int32_t> &gridSize) {
            return ops::injectToDenseCminor(grid, sparseData, origins, vecToCoord(gridSize));
        },
        py::arg("grid"),
        py::arg("sparse_data"),
        py::arg("origins"),
        py::arg("grid_size"));

    m.def(
        "inject_to_dense_cmajor",
        [](const GBI &grid,
           const torch::Tensor &sparseData,
           const torch::Tensor &origins,
           const std::vector<int32_t> &gridSize) {
            return ops::injectToDenseCmajor(grid, sparseData, origins, vecToCoord(gridSize));
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
            return ops::cubesIntersectGrid(
                grid, cubeCenters, vecToVec3d(padMin), vecToVec3d(padMax));
        },
        py::arg("grid"),
        py::arg("cube_centers"),
        py::arg("pad_min"),
        py::arg("pad_max"));

    m.def("ijk_to_index", &ops::ijkToIndex, py::arg("grid"), py::arg("ijk"), py::arg("cumulative"));
    m.def("ijk_to_inv_index",
          &ops::ijkToInvIndex,
          py::arg("grid"),
          py::arg("ijk"),
          py::arg("cumulative"));

    m.def("neighbor_indexes",
          &ops::neighborIndexes,
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

    m.def(
        "marching_cubes", &ops::marchingCubes, py::arg("grid"), py::arg("field"), py::arg("level"));

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

    // -----------------------------------------------------------------------
    // Grid construction factories
    // -----------------------------------------------------------------------

    m.def(
        "create_from_ijk",
        [](const JT &ijk,
           const std::vector<std::vector<double>> &voxelSizes,
           const std::vector<std::vector<double>> &origins) {
            return ops::createNanoGridFromIJK(ijk, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
        },
        py::arg("ijk"),
        py::arg("voxel_sizes"),
        py::arg("origins"));

    m.def(
        "create_from_points",
        [](const JT &points,
           const std::vector<std::vector<double>> &voxelSizes,
           const std::vector<std::vector<double>> &origins) {
            return ops::buildGridFromPoints(
                points, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
        },
        py::arg("points"),
        py::arg("voxel_sizes"),
        py::arg("origins"));

    m.def(
        "create_from_mesh",
        [](const JT &vertices,
           const JT &faces,
           const std::vector<std::vector<double>> &voxelSizes,
           const std::vector<std::vector<double>> &origins) {
            return ops::buildGridFromMesh(
                vertices, faces, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
        },
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("voxel_sizes"),
        py::arg("origins"));

    m.def(
        "create_from_nearest_voxels_to_points",
        [](const JT &points,
           const std::vector<std::vector<double>> &voxelSizes,
           const std::vector<std::vector<double>> &origins) {
            return ops::buildGridFromNearestVoxelsToPoints(
                points, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
        },
        py::arg("points"),
        py::arg("voxel_sizes"),
        py::arg("origins"));

    m.def(
        "create_from_empty",
        [](const torch::Device &device,
           const std::vector<double> &voxelSize,
           const std::vector<double> &origin) {
            return fvdb::detail::makeEmptyGridBatchData(
                device, vecToVec3d(voxelSize), vecToVec3d(origin));
        },
        py::arg("device"),
        py::arg("voxel_size"),
        py::arg("origin"));

    m.def(
        "create_dense",
        [](int64_t numGrids,
           const torch::Device &device,
           const std::vector<int32_t> &denseDims,
           const std::vector<int32_t> &ijkMin,
           const std::vector<std::vector<double>> &voxelSizes,
           const std::vector<std::vector<double>> &origins,
           std::optional<torch::Tensor> mask) {
            return ops::createNanoGridFromDense(numGrids,
                                                vecToCoord(ijkMin),
                                                vecToCoord(denseDims),
                                                device,
                                                mask,
                                                vecsToVec3ds(voxelSizes),
                                                vecsToVec3ds(origins));
        },
        py::arg("num_grids"),
        py::arg("device"),
        py::arg("dense_dims"),
        py::arg("ijk_min"),
        py::arg("voxel_sizes"),
        py::arg("origins"),
        py::arg("mask") = std::nullopt);

    m.def("deserialize_grid", &ops::deserializeGrid, py::arg("serialized"));
    m.def("make_contiguous", &ops::makeContiguous, py::arg("input"));
    m.def("concatenate_grids", &ops::concatenateGrids, py::arg("elements"));

    // -----------------------------------------------------------------------
    // Topology ops
    // -----------------------------------------------------------------------

    m.def(
        "coarsen_grid",
        [](const GBI &grid, const std::vector<int32_t> &factor) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::buildCoarseGridFromFine(grid, vecToCoord(factor));
        },
        py::arg("grid"),
        py::arg("coarsening_factor"));

    m.def(
        "upsample_grid",
        [](const GBI &grid,
           const std::vector<int32_t> &factor,
           std::optional<fvdb::JaggedTensor> mask) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::buildFineGridFromCoarse(grid, vecToCoord(factor), mask);
        },
        py::arg("grid"),
        py::arg("upsample_factor"),
        py::arg("mask") = std::nullopt);

    m.def(
        "dual_grid",
        [](const GBI &grid, bool excludeBorder) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::buildPaddedGrid(grid, 0, 1, excludeBorder);
        },
        py::arg("grid"),
        py::arg("exclude_border") = false);

    m.def(
        "clip_grid",
        [](const GBI &grid,
           const std::vector<std::vector<int32_t>> &ijkMin,
           const std::vector<std::vector<int32_t>> &ijkMax) {
            return ops::clipGrid(grid, vecsToCoords(ijkMin), vecsToCoords(ijkMax));
        },
        py::arg("grid"),
        py::arg("ijk_min"),
        py::arg("ijk_max"));

    m.def(
        "clip_grid_with_mask",
        [](const GBI &grid,
           const std::vector<std::vector<int32_t>> &ijkMin,
           const std::vector<std::vector<int32_t>> &ijkMax) {
            return ops::clipGridWithMask(grid, vecsToCoords(ijkMin), vecsToCoords(ijkMax));
        },
        py::arg("grid"),
        py::arg("ijk_min"),
        py::arg("ijk_max"));

    m.def(
        "clip_grid_features_with_mask",
        [](const GBI &grid,
           const JT &features,
           const std::vector<std::vector<int32_t>> &ijkMin,
           const std::vector<std::vector<int32_t>> &ijkMax) {
            return ops::clipGridFeaturesWithMask(
                grid, features, vecsToCoords(ijkMin), vecsToCoords(ijkMax));
        },
        py::arg("grid"),
        py::arg("features"),
        py::arg("ijk_min"),
        py::arg("ijk_max"));

    m.def(
        "dilate_grid",
        [](const GBI &grid, const std::vector<int64_t> &dilation) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::dilateGrid(grid, dilation);
        },
        py::arg("grid"),
        py::arg("dilation"));

    m.def(
        "merge_grids",
        [](const GBI &grid1, const GBI &grid2) {
            if (grid1.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid1.device());
            }
            return ops::mergeGrids(grid1, grid2);
        },
        py::arg("grid1"),
        py::arg("grid2"));

    m.def(
        "prune_grid",
        [](const GBI &grid, const JT &mask) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::pruneGrid(grid, mask);
        },
        py::arg("grid"),
        py::arg("mask"));

    m.def(
        "conv_grid",
        [](const GBI &grid,
           const std::vector<int32_t> &kernelSize,
           const std::vector<int32_t> &stride) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::buildGridForConv(grid, vecToCoord(kernelSize), vecToCoord(stride));
        },
        py::arg("grid"),
        py::arg("kernel_size"),
        py::arg("stride"));

    m.def(
        "conv_transpose_grid",
        [](const GBI &grid,
           const std::vector<int32_t> &kernelSize,
           const std::vector<int32_t> &stride) {
            if (grid.batchSize() == 0) {
                return fvdb::detail::makeEmptyGridBatchData(grid.device());
            }
            return ops::buildGridForConvTranspose(grid, vecToCoord(kernelSize), vecToCoord(stride));
        },
        py::arg("grid"),
        py::arg("kernel_size"),
        py::arg("stride"));

    // -----------------------------------------------------------------------
    // Batch ops
    // -----------------------------------------------------------------------

    m.def(
        "clone_grid",
        [](const GBI &grid, const torch::Device &device) { return ops::cloneGrid(grid, device); },
        py::arg("grid"),
        py::arg("device"));

    m.def("serialize_grid", &ops::serializeGrid, py::arg("grid"));

    m.def(
        "index_grid_int",
        [](const GBI &grid, int64_t bi) { return ops::indexGrid(grid, bi); },
        py::arg("grid"),
        py::arg("index"));

    m.def(
        "index_grid_slice",
        [](const GBI &grid, ssize_t start, ssize_t stop, ssize_t step) {
            return ops::indexGrid(grid, start, stop, step);
        },
        py::arg("grid"),
        py::arg("start"),
        py::arg("stop"),
        py::arg("step"));

    m.def(
        "index_grid_tensor",
        [](const GBI &grid, const torch::Tensor &indices) { return ops::indexGrid(grid, indices); },
        py::arg("grid"),
        py::arg("indices"));

    m.def(
        "index_grid_int64_list",
        [](const GBI &grid, const std::vector<int64_t> &indices) {
            return ops::indexGrid(grid, indices);
        },
        py::arg("grid"),
        py::arg("indices"));

    m.def(
        "index_grid_bool_list",
        [](const GBI &grid, const std::vector<bool> &indices) {
            return ops::indexGrid(grid, indices);
        },
        py::arg("grid"),
        py::arg("indices"));
}
