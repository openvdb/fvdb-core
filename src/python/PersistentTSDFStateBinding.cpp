// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/PersistentTSDFState.h>

#include <torch/extension.h>

#include <memory>
#include <optional>

namespace py = pybind11;

void
bind_persistent_tsdf_state(py::module &m) {
    using fvdb::GridBatchData;
    using fvdb::JaggedTensor;
    using fvdb::detail::ops::PersistentTSDFState;

    // Shared-pointer wrapper lets Python hold / pass the state around
    // with value semantics (i.e. mutating via one reference shows up
    // through all references). `PersistentTSDFState` is move-only in C++
    // (to avoid accidental sidecar aliasing on the C++ side), so pybind
    // must use a wrapping smart pointer here.
    py::class_<PersistentTSDFState, std::shared_ptr<PersistentTSDFState>>(
        m, "PersistentTSDFState")
        .def(py::init(
                 [](c10::intrusive_ptr<GridBatchData> grid,
                    torch::Tensor tsdf,
                    torch::Tensor weights,
                    std::optional<torch::Tensor> features) {
                     return std::make_shared<PersistentTSDFState>(
                         std::move(grid),
                         std::move(tsdf),
                         std::move(weights),
                         std::move(features));
                 }),
             py::arg("grid"),
             py::arg("tsdf"),
             py::arg("weights"),
             py::arg("features") = std::nullopt)
        .def(
            "grow",
            [](PersistentTSDFState &self, const JaggedTensor &ijks) { self.grow(ijks); },
            py::arg("ijks"))
        .def(
            "grow_from_grid",
            [](PersistentTSDFState &self, const c10::intrusive_ptr<GridBatchData> &shell) {
                self.growFromGrid(*shell);
            },
            py::arg("shell_grid"))
        .def("reset", &PersistentTSDFState::reset)
        .def_property_readonly("active_voxel_count", &PersistentTSDFState::activeVoxelCount)
        .def_property_readonly(
            "grid",
            [](const PersistentTSDFState &self) { return self.gridPtr(); })
        .def_property_readonly(
            "tsdf",
            [](const PersistentTSDFState &self) { return self.tsdf(); })
        .def_property_readonly(
            "weights",
            [](const PersistentTSDFState &self) { return self.weights(); })
        .def_property_readonly("has_features", &PersistentTSDFState::hasFeatures)
        .def_property_readonly(
            "features",
            [](const PersistentTSDFState &self) { return self.features(); });
}
