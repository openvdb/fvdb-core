// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <fvdb/Config.h>
#include <fvdb/FVDB.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/PredGatherIGemm.h>
#include <fvdb/detail/utils/nanovdb/TorchNanoConversions.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAFunctions.h>
#include <torch/extension.h>
#include <torch/version.h>

#include <cuda_runtime_api.h>

void bind_grid_batch_data(py::module &m);
void bind_grid_batch_ops(py::module &m);
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
bind_jt_build_functions(py::module &m) {
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

nanovdb::Coord
pyToCoord(const torch::Tensor &t) {
    torch::Tensor s = t.squeeze().cpu();
    if (s.dim() == 0) {
        auto v = s.item().toLong();
        return nanovdb::Coord(v, v, v);
    }
    return fvdb::tensorToCoord(s);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    bind_grid_batch_data(m);
    bind_grid_batch_ops(m);
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
          py::overload_cast<const std::vector<c10::intrusive_ptr<fvdb::detail::GridBatchData>> &>(
              &fvdb::jcat),
          py::arg("grid_batches"));
    m.def("jcat",
          py::overload_cast<const std::vector<fvdb::JaggedTensor> &, std::optional<int64_t>>(
              &fvdb::jcat),
          py::arg("jagged_tensors"),
          py::arg("dim") = std::nullopt);

    // Static grid construction
    m.def(
        "gridbatch_from_points",
        [](const fvdb::JaggedTensor &points,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins) {
            return fvdb::gridbatch_from_points(
                points, vecsToVec3ds(voxel_sizes), vecsToVec3ds(origins));
        },
        py::arg("points"),
        py::arg("voxel_sizes"),
        py::arg("origins"));
    m.def(
        "gridbatch_from_nearest_voxels_to_points",
        [](const fvdb::JaggedTensor &points,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins) {
            return fvdb::gridbatch_from_nearest_voxels_to_points(
                points, vecsToVec3ds(voxel_sizes), vecsToVec3ds(origins));
        },
        py::arg("points"),
        py::arg("voxel_sizes"),
        py::arg("origins"));
    m.def(
        "gridbatch_from_ijk",
        [](const fvdb::JaggedTensor &ijk,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins) {
            return fvdb::gridbatch_from_ijk(ijk, vecsToVec3ds(voxel_sizes), vecsToVec3ds(origins));
        },
        py::arg("ijk"),
        py::arg("voxel_sizes"),
        py::arg("origins"));
    m.def(
        "gridbatch_from_dense",
        [](const int64_t numGrids,
           const std::vector<int32_t> &denseDims,
           const std::vector<int32_t> &ijkMin,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins,
           std::optional<torch::Tensor> mask,
           const std::string &device) {
            return fvdb::gridbatch_from_dense(numGrids,
                                              vecToCoord(denseDims),
                                              vecToCoord(ijkMin),
                                              vecsToVec3ds(voxel_sizes),
                                              vecsToVec3ds(origins),
                                              mask,
                                              fvdb::parseDeviceString(device));
        },
        py::arg("num_grids"),
        py::arg("dense_dims"),
        py::arg("ijk_min"),
        py::arg("voxel_sizes"),
        py::arg("origins"),
        py::arg("mask")   = std::nullopt,
        py::arg("device") = "cpu");

    m.def(
        "gridbatch_from_mesh",
        [](const fvdb::JaggedTensor &vertices,
           const fvdb::JaggedTensor &faces,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins) {
            return fvdb::gridbatch_from_mesh(
                vertices, faces, vecsToVec3ds(voxel_sizes), vecsToVec3ds(origins));
        },
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("voxel_sizes") = 1.0,
        py::arg("origins")     = torch::zeros({3}));

    // Empty grid batch construction
    m.def(
        "create_from_empty",
        [](const std::string &device) {
            return fvdb::detail::makeEmptyGridBatchData(fvdb::parseDeviceString(device));
        },
        py::arg("device") = "cpu");
    m.def(
        "create_from_empty",
        [](const std::string &device,
           const std::vector<double> &voxel_size,
           const std::vector<double> &origin) {
            return fvdb::detail::makeEmptyGridBatchData(
                fvdb::parseDeviceString(device), vecToVec3d(voxel_size), vecToVec3d(origin));
        },
        py::arg("device"),
        py::arg("voxel_size"),
        py::arg("origin"));
    m.def(
        "create_from_empty",
        [](const std::string &device,
           const std::vector<std::vector<double>> &voxel_sizes,
           const std::vector<std::vector<double>> &origins) {
            return fvdb::detail::makeEmptyGridBatchData(
                fvdb::parseDeviceString(device), vecsToVec3ds(voxel_sizes), vecsToVec3ds(origins));
        },
        py::arg("device"),
        py::arg("voxel_sizes"),
        py::arg("origins"));

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
                            const fvdb::detail::GridBatchData &,
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
                            const fvdb::detail::GridBatchData &,
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

    // Build-time version information exposed to Python
    auto version =
        m.def_submodule("version", "Build-time version information for fVDB dependencies.");
    version.attr("nanovdb") = py::str(std::to_string(NANOVDB_MAJOR_VERSION_NUMBER) + "." +
                                      std::to_string(NANOVDB_MINOR_VERSION_NUMBER) + "." +
                                      std::to_string(NANOVDB_PATCH_VERSION_NUMBER));
    version.attr("cuda")    = py::str(std::to_string(CUDART_VERSION / 1000) + "." +
                                   std::to_string((CUDART_VERSION % 1000) / 10));
    version.attr("torch")   = py::str(TORCH_VERSION);

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
        [](const fvdb::detail::GridBatchData &feature_grid,
           const fvdb::detail::GridBatchData &output_grid,
           const torch::Tensor &kernelSize,
           const torch::Tensor &stride) -> GSDTopo {
            return fvdb::detail::ops::gatherScatterDefaultSparseConvTopology(
                feature_grid, output_grid, pyToCoord(kernelSize), pyToCoord(stride));
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
        [](const fvdb::detail::GridBatchData &feature_grid,
           const fvdb::detail::GridBatchData &output_grid,
           const torch::Tensor &kernelSize,
           const torch::Tensor &stride) -> GSDTopo {
            return fvdb::detail::ops::gatherScatterDefaultSparseConvTransposeTopology(
                feature_grid, output_grid, pyToCoord(kernelSize), pyToCoord(stride));
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

    // -----------------------------------------------------------------------
    // PredGatherIGemm convolution (CUTLASS IGEMM, SM80+)
    // -----------------------------------------------------------------------

    m.def(
        "pred_gather_igemm_conv",
        [](torch::Tensor features,
           torch::Tensor weights,
           const fvdb::detail::GridBatchData &feature_grid,
           const fvdb::detail::GridBatchData &output_grid,
           int kernel_size,
           int stride) -> torch::Tensor {
            return fvdb::detail::ops::predGatherIGemmSparseConv(
                features, weights, feature_grid, output_grid, kernel_size, stride);
        },
        "PredGatherIGemm forward sparse convolution (SM80 CUTLASS IGEMM).",
        py::arg("features"),
        py::arg("weights"),
        py::arg("feature_grid"),
        py::arg("output_grid"),
        py::arg("kernel_size"),
        py::arg("stride"));
}

TORCH_LIBRARY(fvdb, m) {
    m.class_<fvdb::JaggedTensor>("JaggedTensor");
    m.class_<fvdb::detail::GridBatchData>("GridBatchData");

    m.def(
        "_fused_ssim(float C1, float C2, Tensor img1, Tensor img2, bool train) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "_fused_ssim_backward(float C1, float C2, Tensor img1, Tensor img2, Tensor dL_dmap, Tensor dm_dmu1, Tensor dm_dsigma1_sq, Tensor dm_dsigma12) -> Tensor");
}
