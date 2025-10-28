// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "TypeCasters.h"
#include "fvdb/GridBatch.h"

#include <fvdb/FVDB.h>

#include <torch/extension.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

void
bind_grid_batch(py::module &m) {
    py::class_<fvdb::GridBatch>(m, "GridBatch")
        .def(py::init<const torch::Device &>(), py::arg("device") = torch::kCPU)
        .def(py::init([](const std::string &device) {
                 return fvdb::GridBatch(fvdb::parseDeviceString(device));
             }),
             py::arg("device") = "cpu")

        .def(py::init([](const torch::Device &device,
                         const torch::Tensor &voxelSizes,
                         const torch::Tensor &gridOrigins) {
                 TORCH_CHECK_VALUE(voxelSizes.dim() == 2 && voxelSizes.size(0) > 0 &&
                                       voxelSizes.size(1) == 3,
                                   "voxel_sizes must be a [num_grids, 3] tensor");
                 TORCH_CHECK_VALUE(gridOrigins.dim() == 2 && gridOrigins.size(0) > 0 &&
                                       gridOrigins.size(1) == 3,
                                   "grid_origins must be a [num_grids, 3] tensor");
                 TORCH_CHECK_VALUE(
                     voxelSizes.size(0) == gridOrigins.size(0),
                     "voxel_sizes and grid_origins must have the same number of grids");
                 TORCH_CHECK_VALUE(voxelSizes.size(1) == 3 && gridOrigins.size(1) == 3,
                                   "voxel_sizes and grid_origins must have shape [num_grids, 3]");
                 std::vector<nanovdb::Vec3d> voxelSizesVec;
                 std::vector<nanovdb::Vec3d> gridOriginsVec;
                 for (int64_t i = 0; i < voxelSizes.size(0); ++i) {
                     voxelSizesVec.emplace_back(voxelSizes[i][0].item<double>(),
                                                voxelSizes[i][1].item<double>(),
                                                voxelSizes[i][2].item<double>());
                     gridOriginsVec.emplace_back(gridOrigins[i][0].item<double>(),
                                                 gridOrigins[i][1].item<double>(),
                                                 gridOrigins[i][2].item<double>());
                 }
                 return fvdb::GridBatch(device, voxelSizesVec, gridOriginsVec);
             }),
             py::arg("device"),
             py::arg("voxel_sizes"),
             py::arg("grid_origins"))
        // Properties
        .def_property_readonly("total_voxels", &fvdb::GridBatch::total_voxels)
        .def_property_readonly("total_bbox", &fvdb::GridBatch::total_bbox)
        .def_property_readonly_static(
            "max_grids_per_batch",
            [](py::object) -> int64_t { return fvdb::GridBatch::MAX_GRIDS_PER_BATCH; })
        .def_property_readonly("device", &fvdb::GridBatch::device)
        .def_property_readonly("grid_count", &fvdb::GridBatch::grid_count)
        .def_property_readonly("num_voxels", &fvdb::GridBatch::num_voxels)
        .def_property_readonly("cum_voxels", &fvdb::GridBatch::cum_voxels)
        .def_property_readonly(
            "origins", [](const fvdb::GridBatch &self) { return self.origins(torch::kFloat32); })
        .def_property_readonly(
            "voxel_sizes",
            [](const fvdb::GridBatch &self) { return self.voxel_sizes(torch::kFloat32); })
        .def_property_readonly("total_bytes", &fvdb::GridBatch::total_bytes)
        .def_property_readonly("num_bytes", &fvdb::GridBatch::num_bytes)
        .def_property_readonly("total_leaf_nodes", &fvdb::GridBatch::total_leaf_nodes)
        .def_property_readonly("num_leaf_nodes", &fvdb::GridBatch::num_leaf_nodes)
        .def_property_readonly("jidx", &fvdb::GridBatch::jidx)
        .def_property_readonly("joffsets", &fvdb::GridBatch::joffsets)
        .def_property_readonly("ijk", &fvdb::GridBatch::ijk)
        .def("morton",
            &fvdb::GridBatch::morton,
            py::arg("offset"),
            R"_FVDB_(
           Return Morton codes (Z-order curve) for active voxels in this grid batch.

           Morton codes use xyz bit interleaving to create a space-filling curve that
           preserves spatial locality. This is useful for serialization, sorting, and
           spatial data structures.

           Args:
               offset (torch.Tensor): Offset to apply to voxel coordinates before encoding

           Returns:
               codes (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 1]` containing
                   the Morton codes for each active voxel in the batch.
        )_FVDB_")
        .def("morton_zyx",
                &fvdb::GridBatch::morton_zyx,
                py::arg("offset"),
                R"_FVDB_(
            Return transposed Morton codes (Z-order curve) for active voxels in this grid batch.

            Transposed Morton codes use zyx bit interleaving to create a space-filling curve.
            This variant can provide better spatial locality for certain access patterns.

            Args:
                offset (torch.Tensor): Offset to apply to voxel coordinates before encoding

            Returns:
                codes (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 1]` containing
                    the transposed Morton codes for each active voxel in the batch.
        )_FVDB_")
        .def("hilbert",
                &fvdb::GridBatch::hilbert,
                py::arg("offset"),
                R"_FVDB_(
            Return Hilbert curve codes for active voxels in this grid batch.

            Hilbert curves provide better spatial locality than Morton codes by ensuring
            that nearby points in 3D space are also nearby in the 1D curve ordering.

            Args:
                offset (torch.Tensor): Offset to apply to voxel coordinates before encoding

            Returns:
                codes (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 1]` containing
                    the Hilbert codes for each active voxel in the batch.
        )_FVDB_")
        .def("hilbert_zyx",
                &fvdb::GridBatch::hilbert_zyx,
                py::arg("offset"),
                R"_FVDB_(
            Return transposed Hilbert curve codes for active voxels in this grid batch.

            Transposed Hilbert curves use zyx ordering instead of xyz. This variant can
            provide better spatial locality for certain access patterns.

            Args:
                offset (torch.Tensor): Offset to apply to voxel coordinates before encoding

            Returns:
                codes (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 1]` containing
                    the transposed Hilbert codes for each active voxel in the batch.
        )_FVDB_")
        .def_property_readonly(
            "viz_edge_network",
            [](const fvdb::GridBatch &self) { return self.viz_edge_network(false); },
            "A pair of JaggedTensors `(gv, ge)` of shape [num_grids, -1, 3] and [num_grids, -1, 2] where `gv` are the corner positions of each voxel and `ge` are edge indices indexing into `gv`. This property is useful for visualizing the grid.")
        .def_property_readonly("grid_to_world_matrices",
                               [](const fvdb::GridBatch &self) {
                                   return self.grid_to_world_matrices(torch::kFloat32);
                               })
        .def_property_readonly("world_to_grid_matrices",
                               [](const fvdb::GridBatch &self) {
                                   return self.world_to_grid_matrices(torch::kFloat32);
                               })
        .def_property_readonly("bbox", &fvdb::GridBatch::bbox)
        .def_property_readonly("dual_bbox", &fvdb::GridBatch::dual_bbox)
        .def_property_readonly("address", &fvdb::GridBatch::address)

        // Read a property for a single grid in the batch
        .def("voxel_size_at",
             [](const fvdb::GridBatch &self, int64_t bi) {
                 return self.voxel_size_at(bi, torch::kFloat32);
             })
        .def("origin_at",
             [](const fvdb::GridBatch &self, int64_t bi) {
                 return self.origin_at(bi, torch::kFloat32);
             })
        .def("num_voxels_at", &fvdb::GridBatch::num_voxels_at)
        .def("cum_voxels_at", &fvdb::GridBatch::cum_voxels_at)
        .def("bbox_at", &fvdb::GridBatch::bbox_at, py::arg("bi"))
        .def("dual_bbox_at", &fvdb::GridBatch::dual_bbox_at)

        // Create a jagged tensor with the same offsets as this grid batch
        .def("jagged_like", &fvdb::GridBatch::jagged_like, py::arg("data"))

        // Deal with contiguity
        .def("contiguous", &fvdb::GridBatch::contiguous)
        .def("is_contiguous", &fvdb::GridBatch::is_contiguous)

        // Array indexing
        .def(
            "index_int",
            [](const fvdb::GridBatch &self, int64_t bi) { return self.index(bi); },
            py::arg("index"))
        .def(
            "index_slice",
            [](const fvdb::GridBatch &self, pybind11::slice slice) {
                ssize_t start, stop, step, len;
                if (!slice.compute(self.grid_count(), &start, &stop, &step, &len)) {
                    TORCH_CHECK_INDEX(false, "Invalid slice ", py::repr(slice).cast<std::string>());
                }
                TORCH_CHECK_INDEX(step != 0, "step cannot be 0");
                return self.index(start, stop, step);
            },
            py::arg("index"))
        .def(
            "index_list",
            [](const fvdb::GridBatch &self, std::vector<bool> bi) { return self.index(bi); },
            py::arg("index"))
        .def(
            "index_list",
            [](const fvdb::GridBatch &self, std::vector<int64_t> bi) { return self.index(bi); },
            py::arg("index"))
        .def(
            "index_tensor",
            [](const fvdb::GridBatch &self, torch::Tensor bi) { return self.index(bi); },
            py::arg("index"))
        .def("__getitem__", [](const fvdb::GridBatch &self, int64_t bi) { return self.index(bi); })
        .def("__getitem__",
             [](const fvdb::GridBatch &self, pybind11::slice slice) {
                 ssize_t start, stop, step, len;
                 if (!slice.compute(self.grid_count(), &start, &stop, &step, &len)) {
                     TORCH_CHECK_INDEX(
                         false, "Invalid slice ", py::repr(slice).cast<std::string>());
                 }
                 TORCH_CHECK_INDEX(step != 0, "step cannot be 0");
                 return self.index(start, stop, step);
             })
        .def("__getitem__",
             [](const fvdb::GridBatch &self, std::vector<bool> bi) { return self.index(bi); })
        .def("__getitem__",
             [](const fvdb::GridBatch &self, std::vector<int64_t> bi) { return self.index(bi); })
        .def("__getitem__",
             [](const fvdb::GridBatch &self, torch::Tensor bi) { return self.index(bi); })

        // length
        .def("__len__", &fvdb::GridBatch::grid_count)

        // Setting transformation
        .def("set_global_origin", &fvdb::GridBatch::set_global_origin, py::arg("origin"))
        .def(
            "set_global_voxel_size", &fvdb::GridBatch::set_global_voxel_size, py::arg("voxel_size"))

        // Grid construction
        .def("set_from_mesh",
             &fvdb::GridBatch::set_from_mesh,
             py::arg("mesh_vertices"),
             py::arg("mesh_faces"),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins")     = torch::zeros(3, torch::kInt32))
        .def("set_from_points",
             &fvdb::GridBatch::set_from_points,
             py::arg("points"),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins")     = torch::zeros(3, torch::kInt32))
        .def("set_from_dense_grid",
             &fvdb::GridBatch::set_from_dense_grid,
             py::arg("num_grids"),
             py::arg("dense_dims"),
             py::arg("ijk_min")     = torch::zeros(3, torch::kInt32),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins")     = torch::zeros(3),
             py::arg("mask")        = nullptr)
        .def("set_from_ijk",
             &fvdb::GridBatch::set_from_ijk,
             py::arg("ijk"),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins")     = torch::zeros(3))
        .def("set_from_nearest_voxels_to_points",
             &fvdb::GridBatch::set_from_nearest_voxels_to_points,
             py::arg("points"),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins")     = torch::zeros(3))

        // Interface with dense grids
        .def("write_to_dense_cminor",
             &fvdb::GridBatch::write_to_dense_cminor,
             py::arg("sparse_data"),
             py::arg("min_coord") = nullptr,
             py::arg("grid_size") = nullptr)

        .def("write_to_dense_cmajor",
             &fvdb::GridBatch::write_to_dense_cmajor,
             py::arg("sparse_data"),
             py::arg("min_coord") = nullptr,
             py::arg("grid_size") = nullptr)

        .def("read_from_dense_cminor",
             &fvdb::GridBatch::read_from_dense_cminor,
             py::arg("dense_data"),
             py::arg("dense_origins") = torch::zeros(3, torch::kInt32))

        .def("read_from_dense_cmajor",
             &fvdb::GridBatch::read_from_dense_cmajor,
             py::arg("dense_data"),
             py::arg("dense_origins") = torch::zeros(3, torch::kInt32))

        // Derived grids
        .def("dual_grid", &fvdb::GridBatch::dual_grid, py::arg("exclude_border") = false)
        .def("coarsened_grid", &fvdb::GridBatch::coarsened_grid, py::arg("coarsening_factor"))
        .def("refined_grid",
             &fvdb::GridBatch::refined_grid,
             py::arg("subdiv_factor"),
             py::arg("mask") = nullptr)
        .def("clipped_grid", &fvdb::GridBatch::clipped_grid, py::arg("ijk_min"), py::arg("ijk_max"))
        .def("conv_grid", &fvdb::GridBatch::conv_grid, py::arg("kernel_size"), py::arg("stride"))
        .def("dilated_grid", &fvdb::GridBatch::dilated_grid, py::arg("dilation"))
        .def("merged_grid", &fvdb::GridBatch::merged_grid, py::arg("other"))
        .def("pruned_grid", &fvdb::GridBatch::pruned_grid, py::arg("mask"))
        .def("inject_to",
             &fvdb::GridBatch::inject_to,
             py::arg("dst_grid"),
             py::arg("src"),
             py::arg("dst"))

        // Clipping to a bounding box
        .def("clip",
             &fvdb::GridBatch::clip,
             py::arg("features"),
             py::arg("ijk_min"),
             py::arg("ijk_max"))

        // Upsampling and pooling
        .def("max_pool",
             &fvdb::GridBatch::max_pool,
             py::arg("pool_factor"),
             py::arg("data"),
             py::arg("stride")      = 0,
             py::arg("coarse_grid") = nullptr)

        .def("avg_pool",
             &fvdb::GridBatch::avg_pool,
             py::arg("pool_factor"),
             py::arg("data"),
             py::arg("stride")      = 0,
             py::arg("coarse_grid") = nullptr)

        .def("refine",
             &fvdb::GridBatch::refine,
             py::arg("subdiv_factor"),
             py::arg("data"),
             py::arg("mask")      = nullptr,
             py::arg("fine_grid") = nullptr)

        // Grid intersects/contains objects
        .def("points_in_grid", &fvdb::GridBatch::points_in_grid, py::arg("points"))
        .def("coords_in_grid", &fvdb::GridBatch::coords_in_grid, py::arg("ijk"))
        .def("cubes_intersect_grid",
             &fvdb::GridBatch::cubes_intersect_grid,
             py::arg("cube_centers"),
             py::arg("cube_min") = 0.0,
             py::arg("cube_max") = 0.0)
        .def("cubes_in_grid",
             &fvdb::GridBatch::cubes_in_grid,
             py::arg("cube_centers"),
             py::arg("cube_min") = 0.0,
             py::arg("cube_max") = 0.0)

        // Indexing functions
        .def("ijk_to_index",
             &fvdb::GridBatch::ijk_to_index,
             py::arg("ijk"),
             py::arg("cumulative") = false)
        .def("ijk_to_inv_index",
             &fvdb::GridBatch::ijk_to_inv_index,
             py::arg("ijk"),
             py::arg("cumulative") = false)
        .def("neighbor_indexes",
             &fvdb::GridBatch::neighbor_indexes,
             py::arg("ijk"),
             py::arg("extent"),
             py::arg("bitshift") = 0)

        // Ray tracing
        .def("voxels_along_rays",
             &fvdb::GridBatch::voxels_along_rays,
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("max_voxels"),
             py::arg("eps")        = 0.0,
             py::arg("return_ijk") = true,
             py::arg("cumulative") = false)
        .def("segments_along_rays",
             &fvdb::GridBatch::segments_along_rays,
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("max_segments"),
             py::arg("eps") = 0.0)
        .def("uniform_ray_samples",
             &fvdb::GridBatch::uniform_ray_samples,
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("t_min"),
             py::arg("t_max"),
             py::arg("step_size"),
             py::arg("cone_angle")           = 0.0,
             py::arg("include_end_segments") = true,
             py::arg("return_midpoints")     = false,
             py::arg("eps")                  = 0.0)
        .def("ray_implicit_intersection",
             &fvdb::GridBatch::ray_implicit_intersection,
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("grid_scalars"),
             py::arg("eps") = 0.0)

        // Sparse grid operations
        .def("splat_trilinear",
             &fvdb::GridBatch::splat_trilinear,
             py::arg("points"),
             py::arg("points_data"))
        .def("splat_bezier",
             &fvdb::GridBatch::splat_bezier,
             py::arg("points"),
             py::arg("points_data"))
        .def("sample_trilinear",
             &fvdb::GridBatch::sample_trilinear,
             py::arg("points"),
             py::arg("voxel_data"))
        .def("sample_trilinear_with_grad",
             &fvdb::GridBatch::sample_trilinear_with_grad,
             py::arg("points"),
             py::arg("voxel_data"))
        .def("sample_bezier",
             &fvdb::GridBatch::sample_bezier,
             py::arg("points"),
             py::arg("voxel_data"))
        .def("sample_bezier_with_grad",
             &fvdb::GridBatch::sample_bezier_with_grad,
             py::arg("points"),
             py::arg("voxel_data"))

        // Marching cubes
        .def("marching_cubes",
             &fvdb::GridBatch::marching_cubes,
             py::arg("field"),
             py::arg("level") = 0.0)

        // Convolution
        .def("sparse_conv_halo",
             &fvdb::GridBatch::sparse_conv_halo,
             py::arg("input"),
             py::arg("weight"),
             py::arg("variant") = 8)

        // Coordinate transform
        .def("grid_to_world", &fvdb::GridBatch::grid_to_world, py::arg("ijk"))
        .def("world_to_grid", &fvdb::GridBatch::world_to_grid, py::arg("points"))

        // To device
        .def("to", &fvdb::GridBatch::to, py::arg("to_device"))
        .def(
            "to",
            [](const fvdb::GridBatch &self, const std::string &to_device) {
                return self.to(fvdb::parseDeviceString(to_device));
            },
            py::arg("to_device"))
        .def(
            "to",
            [](const fvdb::GridBatch &self, const torch::Tensor &to_tensor) {
                return self.to(to_tensor.device());
            },
            py::arg("to_tensor"))
        .def(
            "to",
            [](const fvdb::GridBatch &self, const fvdb::JaggedTensor &to_jtensor) {
                return self.to(to_jtensor.device());
            },
            py::arg("to_jtensor"))
        .def(
            "to",
            [](const fvdb::GridBatch &self, const fvdb::GridBatch &to_grid) {
                return self.to(to_grid.device());
            },
            py::arg("to_grid"))

        .def("cpu", [](const fvdb::GridBatch &self) { return self.to(torch::kCPU); })
        .def("cuda", [](const fvdb::GridBatch &self) { return self.to(torch::kCUDA); })

        // .def("clone", &fvdb::GridBatch::clone) // TODO: We totally want this

        .def("is_same", &fvdb::GridBatch::is_same, py::arg("other"))
        .def(
            "sparse_conv_kernel_map",
            [](fvdb::GridBatch &self,
               fvdb::Vec3iOrScalar kernelSize,
               fvdb::Vec3iOrScalar stride,
               std::optional<fvdb::GridBatch> targetGrid) {
                auto ret = fvdb::SparseConvPackInfo(kernelSize, stride, self, targetGrid);
                return std::make_tuple(ret, ret.targetGrid());
            },
            py::arg("kernel_size"),
            py::arg("stride"),
            py::arg("target_grid") = nullptr)
        .def("integrate_tsdf",
             &fvdb::GridBatch::integrate_tsdf,
             py::arg("voxel_truncation_distance"),
             py::arg("projection_matrices"),
             py::arg("cam_to_world_matrices"),
             py::arg("tsdf"),
             py::arg("weights"),
             py::arg("depth_images"),
             py::arg("weight_images") = py::none())
        .def("integrate_tsdf_with_features",
             &fvdb::GridBatch::integrate_tsdf_with_features,
             py::arg("voxel_truncation_distance"),
             py::arg("projection_matrices"),
             py::arg("cam_to_world_matrices"),
             py::arg("tsdf"),
             py::arg("features"),
             py::arg("weights"),
             py::arg("depth_images"),
             py::arg("feature_images"),
             py::arg("weight_images") = py::none())
        .def(py::pickle(
            [](const fvdb::GridBatch &batchHdl) {
                return batchHdl.serialize().to(batchHdl.device());
            },
            [](torch::Tensor t) { return fvdb::GridBatch::deserialize(t.cpu()).to(t.device()); }));
}
