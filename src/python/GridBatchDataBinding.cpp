// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include <fvdb/detail/GridBatchData.h>

#include <torch/extension.h>

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
    for (const auto &v : vecs) {
        result.push_back(vecToVec3d(v));
    }
    return result;
}

std::vector<nanovdb::Coord>
vecsToCoords(const std::vector<std::vector<int32_t>> &vecs) {
    std::vector<nanovdb::Coord> result;
    result.reserve(vecs.size());
    for (const auto &v : vecs) {
        result.push_back(vecToCoord(v));
    }
    return result;
}

} // namespace

void
bind_grid_batch_data(py::module &m) {
    using GBI = fvdb::detail::GridBatchData;

    py::class_<GBI, c10::intrusive_ptr<GBI>>(m, "GridBatchData")

        // -- Scalar properties --
        .def_property_readonly("grid_count", &GBI::batchSize)
        .def_property_readonly("total_voxels", &GBI::totalVoxels)
        .def_property_readonly("total_leaves", &GBI::totalLeaves)
        .def_property_readonly("total_bytes", &GBI::totalBytes)
        .def_property_readonly("max_voxels_per_grid", &GBI::maxVoxelsPerGrid)
        .def_property_readonly("max_leaves_per_grid", &GBI::maxLeavesPerGrid)
        .def_property_readonly("device", &GBI::device)
        .def_property_readonly("is_empty", &GBI::isEmpty)
        .def_property_readonly("is_contiguous", &GBI::isContiguous)
        .def_property_readonly_static(
            "MAX_GRIDS_PER_BATCH",
            [](py::object) -> int64_t { return GBI::MAX_GRIDS_PER_BATCH; })

        // -- Tensor properties --
        .def_property_readonly("joffsets", &GBI::voxelOffsets)
        .def_property_readonly("jlidx", &GBI::jlidx)
        .def_property_readonly("jidx", &GBI::jidx)
        .def_property_readonly("num_voxels", &GBI::numVoxelsPerGridTensor)
        .def_property_readonly("cum_voxels", &GBI::cumVoxelsPerGridTensor)
        .def_property_readonly("num_bytes", &GBI::numBytesPerGridTensor)
        .def_property_readonly("num_leaves", &GBI::numLeavesPerGridTensor)
        .def_property_readonly("voxel_sizes", &GBI::voxelSizesTensor)
        .def_property_readonly("origins", &GBI::voxelOriginsTensor)
        .def_property_readonly("bbox", &GBI::bboxPerGridTensor)
        .def_property_readonly("dual_bbox", &GBI::dualBBoxPerGridTensor)
        .def_property_readonly("total_bbox", &GBI::totalBBoxTensor)
        .def_property_readonly("voxel_to_world_matrices", &GBI::gridToWorldMatrixPerGrid)
        .def_property_readonly("world_to_voxel_matrices", &GBI::worldToGridMatrixPerGrid)

        // -- Per-grid queries --
        .def("num_voxels_at", &GBI::numVoxelsAt, py::arg("bi"))
        .def("cum_voxels_at", &GBI::cumVoxelsAt, py::arg("bi"))
        .def("num_bytes_at", &GBI::numBytesAt, py::arg("bi"))
        .def("num_leaves_at", &GBI::numLeavesAt, py::arg("bi"))
        .def("voxel_size_at", &GBI::voxelSizeAtTensor, py::arg("bi"))
        .def("origin_at", &GBI::voxelOriginAtTensor, py::arg("bi"))
        .def("bbox_at", &GBI::bboxAtTensor, py::arg("bi"))
        .def("dual_bbox_at", &GBI::dualBBoxAtTensor, py::arg("bi"))
        .def("voxel_to_world_matrix_at", &GBI::gridToWorldMatrixAt, py::arg("bi"))
        .def("world_to_voxel_matrix_at", &GBI::worldToGridMatrixAt, py::arg("bi"))

        // -- Utility --
        .def("jagged_like", &GBI::jaggedTensor, py::arg("data"))
        .def("serialize", &GBI::serialize)
        .def(
            "clone",
            [](const GBI &self, const torch::Device &device) { return self.clone(device); },
            py::arg("device"))
        .def(
            "is_same",
            [](const c10::intrusive_ptr<GBI> &self, const c10::intrusive_ptr<GBI> &other) {
                return self.get() == other.get();
            },
            py::arg("other"))

        // -- Indexing --
        .def(
            "index_int",
            [](const GBI &self, int64_t bi) { return self.index(bi); },
            py::arg("index"))
        .def(
            "index_slice",
            [](const GBI &self, ssize_t start, ssize_t stop, ssize_t step) {
                return self.index(start, stop, step);
            },
            py::arg("start"),
            py::arg("stop"),
            py::arg("step"))
        .def(
            "index_tensor",
            [](const GBI &self, const torch::Tensor &indices) { return self.index(indices); },
            py::arg("indices"))
        .def(
            "index_int64_list",
            [](const GBI &self, const std::vector<int64_t> &indices) {
                return self.index(indices);
            },
            py::arg("indices"))
        .def(
            "index_bool_list",
            [](const GBI &self, const std::vector<bool> &indices) { return self.index(indices); },
            py::arg("indices"))

        // -- Grid-deriving methods --
        .def(
            "coarsen",
            [](GBI &self, const std::vector<int32_t> &factor) {
                return self.coarsen(vecToCoord(factor));
            },
            py::arg("coarsening_factor"))
        .def(
            "upsample",
            [](GBI &self, const std::vector<int32_t> &factor, std::optional<fvdb::JaggedTensor> mask) {
                return self.upsample(vecToCoord(factor), mask);
            },
            py::arg("upsample_factor"),
            py::arg("mask") = std::nullopt)
        .def("dual", &GBI::dual, py::arg("exclude_border") = false)
        .def(
            "clip",
            [](GBI &self,
               const std::vector<std::vector<int32_t>> &ijkMin,
               const std::vector<std::vector<int32_t>> &ijkMax) {
                return self.clip(vecsToCoords(ijkMin), vecsToCoords(ijkMax));
            },
            py::arg("ijk_min"),
            py::arg("ijk_max"))
        .def(
            "clip_with_mask",
            [](GBI &self,
               const std::vector<std::vector<int32_t>> &ijkMin,
               const std::vector<std::vector<int32_t>> &ijkMax) {
                return self.clipWithMask(vecsToCoords(ijkMin), vecsToCoords(ijkMax));
            },
            py::arg("ijk_min"),
            py::arg("ijk_max"))
        .def(
            "clip_features_with_mask",
            [](GBI &self,
               const fvdb::JaggedTensor &features,
               const std::vector<std::vector<int32_t>> &ijkMin,
               const std::vector<std::vector<int32_t>> &ijkMax) {
                return self.clipFeaturesWithMask(features, vecsToCoords(ijkMin), vecsToCoords(ijkMax));
            },
            py::arg("features"),
            py::arg("ijk_min"),
            py::arg("ijk_max"))
        .def("dilate",
             py::overload_cast<const int64_t>(&GBI::dilate),
             py::arg("dilation"))
        .def(
            "dilate_per_grid",
            [](GBI &self, const std::vector<int64_t> &dilation) { return self.dilate(dilation); },
            py::arg("dilation"))
        .def("merge", &GBI::merge, py::arg("other"))
        .def("prune", &GBI::prune, py::arg("mask"))
        .def(
            "convolution_output",
            [](GBI &self, const std::vector<int32_t> &kernelSize, const std::vector<int32_t> &stride) {
                return self.convolutionOutput(vecToCoord(kernelSize), vecToCoord(stride));
            },
            py::arg("kernel_size"),
            py::arg("stride"))
        .def(
            "convolution_transpose_output",
            [](GBI &self, const std::vector<int32_t> &kernelSize, const std::vector<int32_t> &stride) {
                return self.convolutionTransposeOutput(vecToCoord(kernelSize), vecToCoord(stride));
            },
            py::arg("kernel_size"),
            py::arg("stride"))

        // -- Static factories --
        .def_static(
            "create_from_ijk",
            [](const fvdb::JaggedTensor &ijk,
               const std::vector<std::vector<double>> &voxelSizes,
               const std::vector<std::vector<double>> &origins) {
                return GBI::createFromIjk(ijk, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
            },
            py::arg("ijk"),
            py::arg("voxel_sizes"),
            py::arg("origins"))
        .def_static(
            "create_from_points",
            [](const fvdb::JaggedTensor &points,
               const std::vector<std::vector<double>> &voxelSizes,
               const std::vector<std::vector<double>> &origins) {
                return GBI::createFromPoints(
                    points, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
            },
            py::arg("points"),
            py::arg("voxel_sizes"),
            py::arg("origins"))
        .def_static(
            "create_from_mesh",
            [](const fvdb::JaggedTensor &vertices,
               const fvdb::JaggedTensor &faces,
               const std::vector<std::vector<double>> &voxelSizes,
               const std::vector<std::vector<double>> &origins) {
                return GBI::createFromMesh(
                    vertices, faces, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
            },
            py::arg("vertices"),
            py::arg("faces"),
            py::arg("voxel_sizes"),
            py::arg("origins"))
        .def_static(
            "create_from_nearest_voxels_to_points",
            [](const fvdb::JaggedTensor &points,
               const std::vector<std::vector<double>> &voxelSizes,
               const std::vector<std::vector<double>> &origins) {
                return GBI::createFromNeighborVoxelsToPoints(
                    points, vecsToVec3ds(voxelSizes), vecsToVec3ds(origins));
            },
            py::arg("points"),
            py::arg("voxel_sizes"),
            py::arg("origins"))
        .def_static(
            "create_from_empty",
            [](const torch::Device &device,
               const std::vector<double> &voxelSize,
               const std::vector<double> &origin) {
                return GBI::createFromEmpty(device, vecToVec3d(voxelSize), vecToVec3d(origin));
            },
            py::arg("device"),
            py::arg("voxel_size"),
            py::arg("origin"))
        .def_static(
            "create_dense",
            [](int64_t numGrids,
               const torch::Device &device,
               const std::vector<int32_t> &denseDims,
               const std::vector<int32_t> &ijkMin,
               const std::vector<std::vector<double>> &voxelSizes,
               const std::vector<std::vector<double>> &origins,
               std::optional<torch::Tensor> mask) {
                return GBI::dense(numGrids,
                                  device,
                                  vecToCoord(denseDims),
                                  vecToCoord(ijkMin),
                                  vecsToVec3ds(voxelSizes),
                                  vecsToVec3ds(origins),
                                  mask);
            },
            py::arg("num_grids"),
            py::arg("device"),
            py::arg("dense_dims"),
            py::arg("ijk_min"),
            py::arg("voxel_sizes"),
            py::arg("origins"),
            py::arg("mask") = std::nullopt)
        .def_static("deserialize", &GBI::deserialize, py::arg("serialized"))
        .def_static("make_contiguous", &GBI::contiguous, py::arg("input"))
        .def_static("concatenate", &GBI::concatenate, py::arg("elements"))

        // -- Pickle support --
        .def(py::pickle(
            [](const c10::intrusive_ptr<GBI> &self) {
                return self->serialize().to(self->device());
            },
            [](const torch::Tensor &t) {
                auto impl = GBI::deserialize(t.cpu());
                if (t.device() != torch::kCPU) {
                    impl = impl->clone(t.device());
                }
                return impl;
            }));
}
