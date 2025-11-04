// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_SIMPLEOPHELPER_H
#define FVDB_DETAIL_UTILS_SIMPLEOPHELPER_H

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <array>

namespace fvdb {
namespace detail {

template <torch::DeviceType DeviceTag, typename T, size_t... TailDims>
struct OutAccessorTypeHelper {
    static constexpr size_t N = sizeof...(TailDims) + 1;
    using type = decltype(tensorAccessor<DeviceTag, T, N>(std::declval<torch::Tensor &>()));
    type accessor;
};

template <torch::DeviceType DeviceTag, typename T, size_t... TailDims>
using OutAccessor_t = typename OutAccessorTypeHelper<DeviceTag, T, TailDims...>::type;

template <torch::DeviceType DeviceTag> struct DeviceValidationHelper;

template <> struct DeviceValidationHelper<torch::kCPU> {
    static void
    validate(torch::Device const &device) {
        TORCH_CHECK(device.is_cpu(), "Device must be CPU when DeviceTag is torch::kCPU");
    }
};

template <> struct DeviceValidationHelper<torch::kCUDA> {
    static void
    validate(torch::Device const &device) {
        TORCH_CHECK(device.is_cuda(), "Device must be CUDA when DeviceTag is torch::kCUDA");
    }
};

template <> struct DeviceValidationHelper<torch::kPrivateUse1> {
    static void
    validate(torch::Device const &device) {
        TORCH_CHECK(device.is_privateuseone(),
                    "Device must be PrivateUse1 when DeviceTag is torch::kPrivateUse1");
    }
};

template <torch::DeviceType DeviceTag>
void
validate_device(torch::Device const &device) {
    DeviceValidationHelper<DeviceTag>::validate(device);
}

template <torch::DeviceType DeviceTag, typename T, size_t... TailDims>
torch::Tensor
make_out_tensor(GridBatchImpl const &grid_batch) {
    grid_batch.checkNonEmptyGrid();
    validate_device<DeviceTag>(grid_batch.device());

    // Infer dtype from template type T
    auto const dtype = c10::CppTypeToScalarType<T>::value;
    auto const opts  = torch::TensorOptions().dtype(dtype).device(grid_batch.device());

    // Create shape array with variable length based on N
    static constexpr size_t N = sizeof...(TailDims) + 1;
    std::array<int64_t, N> const shape{grid_batch.totalVoxels(), TailDims...};
    return torch::empty(shape, opts);
}

template <torch::DeviceType DeviceTag, typename T, size_t... TailDims>
auto
make_out_accessor(torch::Tensor const &out_tensor) {
    static constexpr size_t N = sizeof...(TailDims) + 1;
    return tensorAccessor<DeviceTag, T, N>(out_tensor);
}

template <torch::DeviceType DeviceTag, typename Derived, typename T, size_t... TailDims>
struct BaseProcessor {
    using Out_t = OutAccessor_t<DeviceTag, T, TailDims...>;

    __hostdev__ void
    operator()(int64_t const batchIdx,
               int64_t const leafIdx,
               int64_t const voxelIdx,
               int64_t,
               GridBatchImpl::Accessor gridAccessor,
               Out_t out_accessor) const {
        auto const *grid      = gridAccessor.grid(batchIdx);
        auto const &leaf      = grid->tree().template getFirstNode<0>()[leafIdx];
        auto const baseOffset = gridAccessor.voxelOffset(batchIdx);

        auto const ijk = leaf.offsetToGlobalCoord(voxelIdx);
        if (leaf.isActive(voxelIdx)) {
            auto const feature_idx = static_cast<int64_t>(baseOffset + leaf.getValue(voxelIdx) - 1);
            static_cast<Derived const *>(this)->per_active_voxel(ijk, feature_idx, out_accessor);
        }
    }
};

template <torch::DeviceType DeviceTag, typename Derived, typename T, size_t... TailDims>
struct ExecutingBaseProcessor : public BaseProcessor<DeviceTag, Derived, T, TailDims...> {
    JaggedTensor
    execute(GridBatchImpl const &grid_batch, int const num_threads = 1024) const {
        auto out_tensor   = make_out_tensor<DeviceTag, T, TailDims...>(grid_batch);
        auto out_accessor = make_out_accessor<DeviceTag, T, TailDims...>(out_tensor);
        if constexpr (DeviceTag == torch::kCUDA) {
            forEachVoxelCUDA(num_threads, // num threads
                             1,
                             grid_batch,
                             *static_cast<Derived const *>(this),
                             out_accessor);
        } else if constexpr (DeviceTag == torch::kPrivateUse1) {
            forEachVoxelPrivateUse1(
                1, grid_batch, *static_cast<Derived const *>(this), out_accessor);
        } else if constexpr (DeviceTag == torch::kCPU) {
            forEachVoxelCPU(1, grid_batch, *static_cast<Derived const *>(this), out_accessor);
        }
        return grid_batch.jaggedTensor(out_tensor);
    }
};

} // namespace detail
} // namespace fvdb

#endif
