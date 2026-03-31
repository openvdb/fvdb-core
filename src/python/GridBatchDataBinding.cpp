// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include <fvdb/detail/GridBatchData.h>
#include <fvdb/detail/ops/CloneGrid.h>
#include <fvdb/detail/ops/SerializeGrid.h>

#include <torch/extension.h>

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
            "MAX_GRIDS_PER_BATCH", [](py::object) -> int64_t { return GBI::MAX_GRIDS_PER_BATCH; })

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
        .def(
            "is_same",
            [](const c10::intrusive_ptr<GBI> &self, const c10::intrusive_ptr<GBI> &other) {
                return self.get() == other.get();
            },
            py::arg("other"))

        // -- Pickle support --
        .def(py::pickle(
            [](const c10::intrusive_ptr<GBI> &self) {
                namespace ops = fvdb::detail::ops;
                return ops::serializeGrid(*self).to(self->device());
            },
            [](const torch::Tensor &t) {
                namespace ops = fvdb::detail::ops;
                auto impl     = ops::deserializeGrid(t.cpu());
                if (t.device() != torch::kCPU) {
                    impl = ops::cloneGrid(*impl, t.device());
                }
                return impl;
            }));
}
