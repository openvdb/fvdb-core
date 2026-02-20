// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"

#include <fvdb/Config.h>
#include <fvdb/FVDB.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>

#include <c10/cuda/CUDAFunctions.h>
#include <torch/extension.h>

void bind_grid_batch(py::module &m);
void bind_jagged_tensor(py::module &m);
void bind_gaussian_splat3d(py::module &m);
void bind_viewer(py::module &m);

#define __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, LSHAPE_TYPE)                           \
    m.def(                                                                                \
        FUNC_STR,                                                                         \
        [](const LSHAPE_TYPE &lshape,                                                     \
           std::optional<const std::vector<int64_t>> &rshape,                             \
           std::optional<torch::ScalarType> dtype,                                        \
           std::optional<torch::Device> device,                                           \
           bool requires_grad,                                                            \
           bool pin_memory) {                                                             \
            const torch::Device device_     = device.value_or(torch::kCPU);               \
            const torch::ScalarType dtype_  = dtype.value_or(torch::kFloat32);            \
            const torch::TensorOptions opts = torch::TensorOptions()                      \
                                                  .dtype(dtype_)                          \
                                                  .device(device_)                        \
                                                  .requires_grad(requires_grad)           \
                                                  .pinned_memory(pin_memory);             \
            const std::vector<int64_t> rshape_ = rshape.value_or(std::vector<int64_t>()); \
            return fvdb::FUNC_NAME(lshape, rshape_, opts);                                \
        },                                                                                \
        py::arg("lshape"),                                                                \
        py::arg("rshape")        = std::nullopt,                                          \
        py::arg("dtype")         = std::nullopt,                                          \
        py::arg("device")        = std::nullopt,                                          \
        py::arg("requires_grad") = false,                                                 \
        py::arg("pin_memory")    = false);                                                   \
    m.def(                                                                                \
        FUNC_STR,                                                                         \
        [](const LSHAPE_TYPE &lshape,                                                     \
           std::optional<const std::vector<int64_t>> &rshape,                             \
           std::optional<torch::ScalarType> dtype,                                        \
           std::optional<std::string> device,                                             \
           bool requires_grad,                                                            \
           bool pin_memory) {                                                             \
            torch::Device device_(device.value_or("cpu"));                                \
            if (device_.is_cuda() && !device_.has_index()) {                              \
                device_.set_index(c10::cuda::current_device());                           \
            }                                                                             \
            const torch::ScalarType dtype_  = dtype.value_or(torch::kFloat32);            \
            const torch::TensorOptions opts = torch::TensorOptions()                      \
                                                  .dtype(dtype_)                          \
                                                  .device(device_)                        \
                                                  .requires_grad(requires_grad)           \
                                                  .pinned_memory(pin_memory);             \
            const std::vector<int64_t> rshape_ = rshape.value_or(std::vector<int64_t>()); \
            return fvdb::FUNC_NAME(lshape, rshape_, opts);                                \
        },                                                                                \
        py::arg("lshape"),                                                                \
        py::arg("rshape")        = std::nullopt,                                          \
        py::arg("dtype")         = std::nullopt,                                          \
        py::arg("device")        = std::nullopt,                                          \
        py::arg("requires_grad") = false,                                                 \
        py::arg("pin_memory")    = false);

#define __FVDB__BUILDER(FUNC_NAME, FUNC_STR)                         \
    __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, std::vector<int64_t>) \
    __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, std::vector<std::vector<int64_t>>)
void
bind_jt_build_functions(py::module &m){
    // clang-format off
    __FVDB__BUILDER(jrand, "jrand")
    __FVDB__BUILDER(jrandn, "jrandn")
    __FVDB__BUILDER(jzeros, "jzeros")
    __FVDB__BUILDER(jones, "jones")
    __FVDB__BUILDER(jempty, "jempty")
    // clang-format on
}
#undef __FVDB__BUILDER_INNER
#undef __FVDB__BUILDER

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Print types when the user passes in the wrong type
    py::class_<fvdb::Vec3i>(m, "Vec3i");
    py::class_<fvdb::Vec4i>(m, "Vec4i");
    py::class_<fvdb::Vec3d>(m, "Vec3d");
    py::class_<fvdb::Vec3dOrScalar>(m, "Vec3dOrScalar");
    py::class_<fvdb::Vec3iOrScalar>(m, "Vec3iOrScalar");
    py::class_<fvdb::Vec3dBatch>(m, "Vec3dBatch");
    py::class_<fvdb::Vec3dBatchOrScalar>(m, "Vec3dBatchOrScalar");
    py::class_<fvdb::Vec3iBatch>(m, "Vec3iBatch");

    bind_grid_batch(m);
    bind_jagged_tensor(m);
    bind_gaussian_splat3d(m);
    bind_viewer(m);

    //
    // Utility functions
    //

    // volume rendering
    // TODO: (@fwilliams) JaggedTensor interface
    m.def("volume_render",
          &fvdb::volumeRender,
          py::arg("sigmas"),
          py::arg("rgbs"),
          py::arg("deltaTs"),
          py::arg("ts"),
          py::arg("packInfo"),
          py::arg("transmittanceThresh"));

    // attention
    m.def("scaled_dot_product_attention",
          &fvdb::scaledDotProductAttention,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale"),
          R"_FVDB_(
      Computes scaled dot product attention on query, key and value tensors.
            Different SDP kernels could be chosen similar to
            https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

      Args:
            query (JaggedTensor): A JaggedTensor of shape [B, -1, H, E] of the query.
                  Here B is the batch size, H is the number of heads, E is the embedding size.
            key (JaggedTensor): A JaggedTensor of shape [B, -1, H, E] of the key.
            value (JaggedTensor): A JaggedTensor of shape [B, -1, H, V] of the value.
                  Here V is the value size. Note that the key and value should have the same shape.
            scale (float): The scale factor for the attention.

      Returns:
            out (JaggedTensor): Attention result of shape [B, -1, H, V].)_FVDB_");

    // Concatenate grids or jagged tensors
    m.def("jcat",
          py::overload_cast<const std::vector<fvdb::GridBatch> &>(&fvdb::jcat),
          py::arg("grid_batches"));
    m.def("jcat",
          py::overload_cast<const std::vector<fvdb::JaggedTensor> &, std::optional<int64_t>>(
              &fvdb::jcat),
          py::arg("jagged_tensors"),
          py::arg("dim") = std::nullopt);

    // Build a jagged tensor from a grid batch or another jagged tensor and a data tensor. They will
    // have the same offset structure m.def("jagged_like", py::overload_cast<fvdb::JaggedTensor,
    // torch::Tensor>(&fvdb::jagged_like), py::arg("like"), py::arg("data")); m.def("jagged_like",
    // py::overload_cast<fvdb::GridBatch, torch::Tensor>(&fvdb::jagged_like), py::arg("like"),
    // py::arg("data"));

    // Static grid construction
    m.def("gridbatch_from_points",
          &fvdb::gridbatch_from_points,
          py::arg("points"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_nearest_voxels_to_points",
          &fvdb::gridbatch_from_nearest_voxels_to_points,
          py::arg("points"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_ijk",
          &fvdb::gridbatch_from_ijk,
          py::arg("ijk"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_dense",
          &fvdb::gridbatch_from_dense,
          py::arg("num_grids"),
          py::arg("dense_dims"),
          py::arg("ijk_min")     = torch::zeros(3, torch::kInt32),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}),
          py::arg("mask")        = nullptr,
          py::arg("device")      = "cpu");

    m.def(
        "gridbatch_from_dense",
        [](const int64_t numGrids,
           const fvdb::Vec3i &denseDims,
           const fvdb::Vec3i &ijkMin,
           const fvdb::Vec3dBatchOrScalar &voxel_sizes,
           const fvdb::Vec3dBatch &origins,
           std::optional<torch::Tensor> mask,
           const std::string &device) {
            return fvdb::gridbatch_from_dense(numGrids,
                                              denseDims,
                                              ijkMin,
                                              voxel_sizes,
                                              origins,
                                              mask,
                                              fvdb::parseDeviceString(device));
        },
        py::arg("num_grids"),
        py::arg("dense_dims"),
        py::arg("ijk_min")     = torch::zeros(3, torch::kInt32),
        py::arg("voxel_sizes") = 1.0,
        py::arg("origins")     = torch::zeros({3}),
        py::arg("mask")        = nullptr,
        py::arg("device")      = "cpu");

    m.def("gridbatch_from_mesh",
          &fvdb::gridbatch_from_mesh,
          py::arg("vertices"),
          py::arg("faces"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));

    // Loading and saving grids
    m.def("load",
          py::overload_cast<const std::string &,
                            const std::vector<uint64_t> &,
                            const torch::Device &,
                            bool>(&fvdb::load),
          py::arg("path"),
          py::arg("indices"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path, uint64_t index, const torch::Device &device, bool verbose) {
            std::vector<uint64_t> indices{index};
            return fvdb::load(path, indices, device, verbose);
        },
        py::arg("path"),
        py::arg("index"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);
    m.def("load",
          py::overload_cast<const std::string &,
                            const std::vector<std::string> &,
                            const torch::Device &,
                            bool>(&fvdb::load),
          py::arg("path"),
          py::arg("names"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path,
           const std::string &name,
           const torch::Device &device,
           bool verbose) {
            std::vector<std::string> names{name};
            return fvdb::load(path, names, device, verbose);
        },
        py::arg("path"),
        py::arg("name"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);
    m.def("load",
          py::overload_cast<const std::string &, const torch::Device &, bool>(&fvdb::load),
          py::arg("path"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path,
           const std::vector<uint64_t> &indices,
           const std::string &device,
           bool verbose) {
            return fvdb::load(path, indices, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("indices"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path, uint64_t index, const std::string &device, bool verbose) {
            std::vector<uint64_t> indices{index};
            return fvdb::load(path, indices, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("index"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path,
           const std::vector<std::string> &names,
           const std::string &device,
           bool verbose) {
            return fvdb::load(path, names, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("names"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path,
           const std::string &name,
           const std::string &device,
           bool verbose) {
            std::vector<std::string> names{name};
            return fvdb::load(path, names, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("name"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path, const std::string &device, bool verbose) {
            return fvdb::load(path, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);

    m.def("save",
          py::overload_cast<const std::string &,
                            const fvdb::GridBatch &,
                            const std::optional<fvdb::JaggedTensor>,
                            const std::vector<std::string> &,
                            bool,
                            bool>(&fvdb::save),
          py::arg("path"),
          py::arg("grid_batch"),
          py::arg("data")       = py::none(),
          py::arg("names")      = std::vector<std::string>(),
          py::arg("compressed") = false,
          py::arg("verbose")    = false);
    m.def("save",
          py::overload_cast<const std::string &,
                            const fvdb::GridBatch &,
                            const std::optional<fvdb::JaggedTensor>,
                            const std::string &,
                            bool,
                            bool>(&fvdb::save),
          py::arg("path"),
          py::arg("grid_batch"),
          py::arg("data")       = py::none(),
          py::arg("name")       = std::string(),
          py::arg("compressed") = false,
          py::arg("verbose")    = false);

    m.def("morton", &fvdb::morton, py::arg("ijk"));
    m.def("hilbert", &fvdb::hilbert, py::arg("ijk"));

    /*
              py::overload_cast<const std::vector<int64_t>&,
                                std::optional<const std::vector<int64_t>>&,
                                std::optional<torch::ScalarType>,
                                std::optional<torch::Device>,
                                bool, bool>(
    */
    bind_jt_build_functions(m);

    // Global config
    py::class_<fvdb::Config>(m, "config")
        .def_property_static(
            "enable_ultra_sparse_acceleration",
            [](py::object) { return fvdb::Config::global().ultraSparseAccelerationEnabled(); },
            [](py::object, bool enabled) {
                fvdb::Config::global().setUltraSparseAcceleration(enabled);
            })
        .def_property_static(
            "pedantic_error_checking",
            [](py::object) { return fvdb::Config::global().pedanticErrorCheckingEnabled(); },
            [](py::object, bool enabled) {
                fvdb::Config::global().setPedanticErrorChecking(enabled);
            });

    // -----------------------------------------------------------------------
    // GatherScatterDefault convolution: Python-side autograd
    // -----------------------------------------------------------------------

    using GSDTopo = fvdb::detail::ops::GatherScatterDefaultTopology;

    py::class_<GSDTopo>(m, "GatherScatterDefaultTopology")
        .def_readonly("gather_indices", &GSDTopo::gatherIndices)
        .def_readonly("scatter_indices", &GSDTopo::scatterIndices)
        .def_readonly("offsets", &GSDTopo::offsets)
        .def_readonly("feature_total_voxels", &GSDTopo::featureTotalVoxels)
        .def_readonly("output_total_voxels", &GSDTopo::outputTotalVoxels)
        .def_readonly("kernel_volume", &GSDTopo::kernelVolume)
        .def_readonly("total_pairs", &GSDTopo::totalPairs)
        .def_property_readonly("kernel_size",
                               [](const GSDTopo &t) {
                                   return std::vector<int>{
                                       t.kernelSize[0], t.kernelSize[1], t.kernelSize[2]};
                               })
        .def_property_readonly("stride",
                               [](const GSDTopo &t) {
                                   return std::vector<int>{t.stride[0], t.stride[1], t.stride[2]};
                               })
        .def_property_readonly("is_transposed", [](const GSDTopo &t) {
            return t.direction == fvdb::detail::ops::ConvDirection::Transposed;
        });

    // --- Forward topology + conv ---

    m.def(
        "gs_build_topology",
        [](const fvdb::GridBatch &feature_grid,
           const fvdb::GridBatch &output_grid,
           fvdb::Vec3iOrScalar kernelSize,
           fvdb::Vec3iOrScalar stride) -> GSDTopo {
            return fvdb::GridBatch::buildGatherScatterDefaultTopology(
                feature_grid, output_grid, kernelSize, stride);
        },
        "Build the forward gather-scatter default topology.",
        py::arg("feature_grid"),
        py::arg("output_grid"),
        py::arg("kernel_size"),
        py::arg("stride"));

    m.def(
        "gs_conv",
        [](torch::Tensor features, torch::Tensor weights, const GSDTopo &topo) -> torch::Tensor {
            return fvdb::detail::ops::gatherScatterDefaultSparseConv(features, weights, topo);
        },
        "Gather-scatter default forward sparse convolution using precomputed topology.",
        py::arg("features"),
        py::arg("weights"),
        py::arg("topology"));

    m.def(
        "gs_conv_backward",
        [](torch::Tensor grad_output,
           torch::Tensor features,
           torch::Tensor weights,
           const GSDTopo &topo) -> std::tuple<torch::Tensor, torch::Tensor> {
            return fvdb::detail::ops::gatherScatterDefaultSparseConvBackward(
                grad_output, features, weights, topo);
        },
        "Gather-scatter default backward sparse convolution using precomputed topology.",
        py::arg("grad_output"),
        py::arg("features"),
        py::arg("weights"),
        py::arg("topology"));

    // --- Transposed topology + conv ---

    m.def(
        "gs_build_transpose_topology",
        [](const fvdb::GridBatch &feature_grid,
           const fvdb::GridBatch &output_grid,
           fvdb::Vec3iOrScalar kernelSize,
           fvdb::Vec3iOrScalar stride) -> GSDTopo {
            return fvdb::GridBatch::buildGatherScatterDefaultTransposeTopology(
                feature_grid, output_grid, kernelSize, stride);
        },
        "Build the transposed gather-scatter default topology.",
        py::arg("feature_grid"),
        py::arg("output_grid"),
        py::arg("kernel_size"),
        py::arg("stride"));

    m.def(
        "gs_conv_transpose",
        [](torch::Tensor features, torch::Tensor weights, const GSDTopo &topo) -> torch::Tensor {
            return fvdb::detail::ops::gatherScatterDefaultSparseConvTranspose(
                features, weights, topo);
        },
        "Gather-scatter default transposed sparse convolution using precomputed topology.",
        py::arg("features"),
        py::arg("weights"),
        py::arg("topology"));

    m.def(
        "gs_conv_transpose_backward",
        [](torch::Tensor grad_output,
           torch::Tensor features,
           torch::Tensor weights,
           const GSDTopo &topo) -> std::tuple<torch::Tensor, torch::Tensor> {
            return fvdb::detail::ops::gatherScatterDefaultSparseConvTransposeBackward(
                grad_output, features, weights, topo);
        },
        "Gather-scatter default transposed backward sparse convolution.",
        py::arg("grad_output"),
        py::arg("features"),
        py::arg("weights"),
        py::arg("topology"));
}

TORCH_LIBRARY(fvdb, m) {
    m.class_<fvdb::GridBatch>("GridBatch");
    m.class_<fvdb::JaggedTensor>("JaggedTensor");
    m.class_<fvdb::detail::GridBatchImpl>("GridBatchImpl");

    m.def(
        "_fused_ssim(float C1, float C2, Tensor img1, Tensor img2, bool train) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "_fused_ssim_backward(float C1, float C2, Tensor img1, Tensor img2, Tensor dL_dmap, Tensor dm_dmu1, Tensor dm_dsigma1_sq, Tensor dm_dsigma12) -> Tensor");
}
