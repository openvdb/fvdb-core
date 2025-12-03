// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/FVDB.h>

// Autograd headers
#include <fvdb/detail/autograd/VolumeRender.h>

// Morton/hilbert
#include <fvdb/detail/ops/MortonHilbertFromIjk.h>

// IO headers
#include <fvdb/detail/io/LoadNanovdb.h>
#include <fvdb/detail/io/SaveNanoVDB.h>

#include <ATen/cuda/CUDAContext.h>

namespace fvdb {

torch::Device
parseDeviceString(const std::string &string) {
    torch::Device device(string);
    if (device.is_cuda() && !device.has_index()) {
        device.set_index(c10::cuda::current_device());
    }
    return device;
}

std::vector<torch::Tensor>
volumeRender(const torch::Tensor &sigmas,
             const torch::Tensor &rgbs,
             const torch::Tensor &deltaTs,
             const torch::Tensor &ts,
             const torch::Tensor &jOffsets,
             double transmittanceThresh) {
    return detail::autograd::VolumeRender::apply(
        sigmas, rgbs, deltaTs, ts, jOffsets, transmittanceThresh);
}

JaggedTensor
scaledDotProductAttention(const JaggedTensor &query,
                          const JaggedTensor &key,
                          const JaggedTensor &value,
                          float scale) {
    // https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    // - query: (N, ..., L, E)
    // - key: (N, ..., S, E)
    // - value: (N, ..., S, V)

    // Helper to create a zero-copy nested tensor view from a JaggedTensor
    // This is more efficient but only works with fused backends (flash/efficient attention)
    auto make_nested_view = [](const JaggedTensor &jt) -> torch::Tensor {
        const auto &data = jt.jdata(); // (Total, H, D)
        const int64_t H  = data.size(1);
        const int64_t D  = data.size(2);

        const int64_t stride_L = H * D;

        auto offsets_tensor       = jt.joffsets().cpu();
        const int64_t num_tensors = jt.num_tensors();

        // Compute lengths: offsets[1:] - offsets[:-1]
        auto lengths =
            offsets_tensor.slice(0, 1, num_tensors + 1) - offsets_tensor.slice(0, 0, num_tensors);

        // Construct nested_size: (N, 3) -> [L_i, H, D]
        auto nested_size = torch::empty({num_tensors, 3}, lengths.options());
        nested_size.select(1, 0).copy_(lengths);
        nested_size.select(1, 1).fill_(H);
        nested_size.select(1, 2).fill_(D);

        // Construct nested_strides: (N, 3) -> [H*D, D, 1]
        auto nested_strides = torch::empty({num_tensors, 3}, lengths.options());
        nested_strides.select(1, 0).fill_(stride_L);
        nested_strides.select(1, 1).fill_(D);
        nested_strides.select(1, 2).fill_(1);

        // Prepare offsets in elements for the buffer
        std::vector<int64_t> storage_offsets(num_tensors);
        auto offsets_acc = offsets_tensor.accessor<int64_t, 1>();
        for (int64_t i = 0; i < num_tensors; ++i) {
            storage_offsets[i] = offsets_acc[i] * stride_L;
        }

        auto offsets_arg = torch::tensor(storage_offsets, lengths.options().dtype(torch::kLong));

        return at::_nested_view_from_buffer(
            data.view({-1}), nested_size, nested_strides, offsets_arg);
    };

    // Helper to create a proper nested tensor (with copy) from a JaggedTensor
    // Required for math backend which needs contiguous tensors
    auto make_nested_tensor = [](const JaggedTensor &jt) -> torch::Tensor {
        const auto &data          = jt.jdata();
        auto offsets_tensor       = jt.joffsets();
        const int64_t num_tensors = jt.num_tensors();

        std::vector<torch::Tensor> tensor_list;
        tensor_list.reserve(num_tensors);

        auto offsets_cpu = offsets_tensor.cpu();
        auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
        for (int64_t i = 0; i < num_tensors; ++i) {
            int64_t start = offsets_acc[i];
            int64_t end   = offsets_acc[i + 1];
            tensor_list.push_back(data.slice(0, start, end).contiguous());
        }

        return at::_nested_tensor_from_tensor_list(tensor_list, {}, {}, {}, {});
    };

    torch::Tensor q_nested, k_nested, v_nested;

    // Check runtime context set by Python's sdpa_kernel() context manager
    // These reflect what backends are enabled/disabled at runtime
    auto &ctx                  = at::globalContext();
    bool flash_enabled         = ctx.userEnabledFlashSDP();
    bool mem_efficient_enabled = ctx.userEnabledMemEfficientSDP();
    bool math_enabled          = ctx.userEnabledMathSDP();
    bool cudnn_enabled         = ctx.userEnabledCuDNNSDP();

    // Different backends have different nested tensor requirements:
    // - Flash Attention: needs _nested_tensor_from_tensor_list (NOT contiguous after transpose)
    // - Efficient Attention: works with _nested_view_from_buffer (zero-copy)
    // - Math: needs _nested_tensor_from_tensor_list WITH contiguous after transpose

    bool math_only = math_enabled && !flash_enabled && !mem_efficient_enabled && !cudnn_enabled;

    if (math_only) {
        // Math backend requires contiguous nested tensors
        q_nested = make_nested_tensor(query).transpose(1, 2).contiguous();
        k_nested = make_nested_tensor(key).transpose(1, 2).contiguous();
        v_nested = make_nested_tensor(value).transpose(1, 2).contiguous();
    } else if (flash_enabled && !mem_efficient_enabled) {
        // Flash Attention needs proper nested tensors but NOT contiguous
        q_nested = make_nested_tensor(query).transpose(1, 2);
        k_nested = make_nested_tensor(key).transpose(1, 2);
        v_nested = make_nested_tensor(value).transpose(1, 2);
    } else {
        // Efficient attention can use zero-copy view
        q_nested = make_nested_view(query).transpose(1, 2);
        k_nested = make_nested_view(key).transpose(1, 2);
        v_nested = make_nested_view(value).transpose(1, 2);

        // Query which backend will actually be used based on tensor properties
        // SDPBackend: 0=error, 1=math, 2=flash_attention, 3=efficient_attention, 4=cudnn_attention
        int64_t backend =
            at::_fused_sdp_choice(q_nested, k_nested, v_nested, {}, 0.0, false, scale, false);

        // Handle fallback cases
        if (backend == 1) {
            // Math backend selected - need contiguous tensors
            q_nested = make_nested_tensor(query).transpose(1, 2).contiguous();
            k_nested = make_nested_tensor(key).transpose(1, 2).contiguous();
            v_nested = make_nested_tensor(value).transpose(1, 2).contiguous();
        } else if (backend == 2) {
            // Flash attention selected - need proper nested tensors (not view)
            q_nested = make_nested_tensor(query).transpose(1, 2);
            k_nested = make_nested_tensor(key).transpose(1, 2);
            v_nested = make_nested_tensor(value).transpose(1, 2);
        }
        // For efficient_attention (3) or cudnn (4), keep the zero-copy view
    }

    torch::Tensor out_nested = at::native::scaled_dot_product_attention(
        q_nested, k_nested, v_nested, {}, 0.0, false, scale);

    // out_nested is (N, H, L, D) nested tensor.
    // We need to convert back to JaggedTensor with shape (Total, H, D)
    std::vector<torch::Tensor> outList = out_nested.unbind();
    for (auto &t: outList) {
        t = t.permute({1, 0, 2}).contiguous();
    }

    return JaggedTensor(outList);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
from_nanovdb(nanovdb::GridHandle<nanovdb::HostBuffer> &handle) {
    return detail::io::fromNVDB(handle);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
to_nanovdb(const GridBatch &gridBatch,
           const std::optional<JaggedTensor> maybeData,
           const std::vector<std::string> &names) {
    return detail::io::toNVDB(gridBatch, maybeData, names);
}

GridBatch
jcat(const std::vector<GridBatch> &vec) {
    return GridBatch::concatenate(vec);
}

JaggedTensor
jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dim) {
    return JaggedTensor::jcat(vec, dim);
}

void
save(const std::string &path,
     const GridBatch &gridBatch,
     const std::optional<JaggedTensor> maybeData,
     const std::vector<std::string> &names,
     bool compressed,
     bool verbose) {
    detail::io::saveNVDB(path, gridBatch, maybeData, names, compressed, verbose);
}

void
save(const std::string &path,
     const GridBatch &gridBatch,
     const std::optional<JaggedTensor> maybeData,
     const std::string &name,
     bool compressed,
     bool verbose) {
    if (name.empty()) {
        detail::io::saveNVDB(path, gridBatch, maybeData, {}, compressed, verbose);
    } else {
        std::vector<std::string> names(gridBatch.grid_count());
        std::fill(names.begin(), names.end(), name);
        detail::io::saveNVDB(path, gridBatch, maybeData, names, compressed, verbose);
    }
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path,
     const std::vector<uint64_t> &indices,
     const torch::Device &device,
     bool verbose) {
    return detail::io::loadNVDB(path, indices, device, verbose);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path,
     const std::vector<std::string> &names,
     const torch::Device &device,
     bool verbose) {
    return detail::io::loadNVDB(path, names, device, verbose);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path, const torch::Device &device, bool verbose) {
    return detail::io::loadNVDB(path, device, verbose);
}

GridBatch
gridbatch_from_points(const JaggedTensor &points,
                      const Vec3dBatchOrScalar &voxel_sizes,
                      const Vec3dBatch &origins) {
    auto ret = GridBatch(points.device());
    ret.set_from_points(points, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_ijk(const JaggedTensor &ijk,
                   const Vec3dBatchOrScalar &voxel_sizes,
                   const Vec3dBatch &origins) {
    auto ret = GridBatch(ijk.device());
    ret.set_from_ijk(ijk, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_nearest_voxels_to_points(const JaggedTensor &points,
                                        const Vec3dBatchOrScalar &voxel_sizes,
                                        const Vec3dBatch &origins) {
    auto ret = GridBatch(points.device());
    ret.set_from_nearest_voxels_to_points(points, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_dense(const int64_t numGrids,
                     const Vec3i &denseDims,
                     const Vec3i &ijkMin,
                     const Vec3dBatchOrScalar &voxel_sizes,
                     const Vec3dBatch &origins,
                     std::optional<torch::Tensor> mask,
                     const torch::Device &device) {
    auto ret = GridBatch(device);
    ret.set_from_dense_grid(numGrids, denseDims, ijkMin, voxel_sizes, origins, mask);
    return ret;
}

GridBatch
gridbatch_from_mesh(const JaggedTensor &vertices,
                    const JaggedTensor &faces,
                    const Vec3dBatchOrScalar &voxel_sizes,
                    const Vec3dBatch &origins) {
    auto ret = GridBatch(vertices.device());
    ret.set_from_mesh(vertices, faces, voxel_sizes, origins);
    return ret;
}

std::vector<int64_t>
jdataShape1(const std::vector<int64_t> &lsizes, const std::vector<int64_t> &rsizes) {
    const int64_t totalElements = std::reduce(lsizes.begin(), lsizes.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalElements);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());
    return shape;
}

std::tuple<int64_t, std::vector<int64_t>>
jdataShape2(const std::vector<std::vector<int64_t>> &lsizes, const std::vector<int64_t> &rsizes) {
    std::vector<int64_t> elementCountsPerList;
    std::vector<int64_t> tensorCountsPerList;
    elementCountsPerList.reserve(lsizes.size());
    tensorCountsPerList.reserve(lsizes.size());
    for (const auto &l: lsizes) {
        elementCountsPerList.push_back(std::reduce(l.begin(), l.end()));
        tensorCountsPerList.push_back(l.size());
    }
    const int64_t totalSize = std::reduce(elementCountsPerList.begin(), elementCountsPerList.end());
    const int64_t totalTensors =
        std::reduce(tensorCountsPerList.begin(), tensorCountsPerList.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalSize);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());

    return std::make_tuple(totalTensors, shape);
}

#define __FVDB__BUILDER(FNAME, JFNAME)                                                       \
    JaggedTensor JFNAME(const std::vector<int64_t> &lsizes,                                  \
                        const std::vector<int64_t> rsizes,                                   \
                        at::TensorOptions options) {                                         \
        auto shape = jdataShape1(lsizes, rsizes);                                            \
        return JaggedTensor(lsizes, FNAME(shape, options));                                  \
    }                                                                                        \
                                                                                             \
    JaggedTensor JFNAME(const std::vector<std::vector<int64_t>> &lsizes,                     \
                        const std::vector<int64_t> rsizes,                                   \
                        at::TensorOptions options) {                                         \
        auto shape = jdataShape2(lsizes, rsizes);                                            \
        return JaggedTensor(lsizes, std::get<0>(shape), FNAME(std::get<1>(shape), options)); \
    }

__FVDB__BUILDER(torch::rand, jrand)
__FVDB__BUILDER(torch::randn, jrandn)
__FVDB__BUILDER(torch::zeros, jzeros)
__FVDB__BUILDER(torch::ones, jones)
__FVDB__BUILDER(torch::empty, jempty)

#undef __FVDB__BUILDER

torch::Tensor
morton(torch::Tensor const &ijk) {
    return detail::ops::mortonFromIjk(ijk);
}

torch::Tensor
hilbert(torch::Tensor const &ijk) {
    return detail::ops::hilbertFromIjk(ijk);
}

} // namespace fvdb
