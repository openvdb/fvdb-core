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

template <typename T, size_t N> struct DynamicElementType {
    using value_type              = T;
    static constexpr size_t NDIMS = N;
    std::array<size_t, N> dynamic_shape;

    constexpr DynamicElementType(std::array<size_t, N> in_shape) : dynamic_shape(in_shape) {}

    constexpr std::array<size_t, N>
    shape() const {
        return dynamic_shape;
    }
};

template <typename T, size_t... Dims> struct FixedElementType {
    using value_type                                       = T;
    static constexpr size_t NDIMS                          = sizeof...(Dims);
    static constexpr std::array<size_t, NDIMS> fixed_shape = {Dims...};

    constexpr std::array<size_t, NDIMS>
    shape() const {
        return fixed_shape;
    }
};

template <typename T> using ScalarElementType = FixedElementType<T>;
static_assert(ScalarElementType<int64_t>::NDIMS == 0);

template <torch::DeviceType DeviceTag, typename ElementType> struct AccessorTypeHelper {
    using value_type                      = typename ElementType::value_type;
    static constexpr size_t ELEMENT_NDIMS = ElementType::NDIMS;
    using type = decltype(tensorAccessor<DeviceTag, value_type, ELEMENT_NDIMS + 1>(
        std::declval<torch::Tensor &>()));
};

template <torch::DeviceType DeviceTag, typename ElementType>
using Accessor_t = typename AccessorTypeHelper<DeviceTag, ElementType>::type;

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
validateDevice(torch::Device const &device) {
    DeviceValidationHelper<DeviceTag>::validate(device);
}

template <torch::DeviceType DeviceTag, typename ElementType>
torch::Tensor
makeOutTensorFromGridBatch(GridBatchImpl const &grid_batch, ElementType const &element_type) {
    grid_batch.checkNonEmptyGrid();
    validateDevice<DeviceTag>(grid_batch.device());

    using T            = typename ElementType::value_type;
    constexpr size_t N = 1 + ElementType::NDIMS;

    // Infer dtype from template type T
    auto const dtype = c10::CppTypeToScalarType<T>::value;
    auto const opts  = torch::TensorOptions().dtype(dtype).device(grid_batch.device());

    // Create shape array with variable length based on N
    auto const outer_dim = static_cast<int64_t>(grid_batch.totalVoxels());
    std::array<int64_t, N> shape;
    auto const tail_shape = element_type.shape();
    shape[0]              = outer_dim;
    for (size_t i = 1; i < N; ++i) {
        shape[i] = tail_shape[i - 1];
    }
    return torch::empty(shape, opts);
}

template <torch::DeviceType DeviceTag, typename ElementType>
torch::Tensor
makeOutTensorFromTensor(torch::Tensor const &in_tensor, ElementType const &element_type) {
    using T            = typename ElementType::value_type;
    constexpr size_t N = 1 + ElementType::NDIMS;
    validateDevice<DeviceTag>(in_tensor.device());

    // Infer dtype from template type T
    auto const dtype = c10::CppTypeToScalarType<T>::value;
    auto const opts  = torch::TensorOptions().dtype(dtype).device(in_tensor.device());

    // Create shape array with variable length based on N
    auto const outer_dim = static_cast<int64_t>(in_tensor.size(0));
    std::array<int64_t, N> shape;
    auto const tail_shape = element_type.shape();
    shape[0]              = outer_dim;
    for (size_t i = 1; i < N; ++i) {
        shape[i] = tail_shape[i - 1];
    }
    return torch::empty(shape, opts);
}

template <torch::DeviceType DeviceTag, typename ElementType>
auto
makeAccessor(torch::Tensor &tensor) {
    using T            = typename ElementType::value_type;
    constexpr size_t N = 1 + ElementType::NDIMS;
    return tensorAccessor<DeviceTag, T, N>(tensor);
}

/**
 * @brief CRTP base class for simple elementwise operations over grid voxels.
 *
 * This helper simplifies operations that share the same input/output topology
 * (i.e., same sparse voxel structure) but differ only in the rank and dtype
 * of the output features. Each active voxel is visited elementwise.
 *
 * @tparam DeviceTag The device type (torch::kCPU, torch::kCUDA, or torch::kPrivateUse1)
 * @tparam Derived The derived class (CRTP pattern)
 * @tparam OutElementType Element type descriptor that encapsulates both value type and shape.
 *                        Can be ScalarElementType<T> for scalar outputs,
 *                        FixedElementType<T, Dims...> for fixed-shape outputs (e.g., (N, 3)),
 *                        or DynamicElementType<T, N> for runtime-determined shapes.
 *
 * Usage example (from SerializeEncode.cu):
 * @code
 * template <torch::DeviceType DeviceTag>
 * struct Processor : public BasePerActiveVoxelProcessor<DeviceTag,
 *                                                        Processor<DeviceTag>,
 *                                                        ScalarElementType<int64_t>> {
 *     // Add any state needed for your operation
 *     nanovdb::Coord offset = nanovdb::Coord{0, 0, 0};
 *     SpaceFillingCurveType order_type = SpaceFillingCurveType::ZOrder;
 *
 *     // Implement this method to define per-voxel computation
 *     __hostdev__ void
 *     perActiveVoxel(nanovdb::Coord const &ijk, int64_t const feature_idx, auto out_accessor) const
 * {
 *         // Your voxel processing logic here
 *         // ijk: global 3D coordinates of the voxel
 *         // feature_idx: linear index into the output tensor for this voxel
 *         // out_accessor: tensor accessor for writing output
 *         out_accessor[feature_idx] = compute_value(ijk);
 *     }
 * };
 *
 * // Usage:
 * JaggedTensor result = Processor<torch::kCUDA>{...}.execute(gridBatch);
 * @endcode
 *
 * The base class handles:
 * - Output tensor allocation with correct shape based on OutElementType
 * - Dispatching to CPU/CUDA/PrivateUse1 implementations
 * - Iterating over all active voxels in the grid batch
 * - Wrapping output tensor as a JaggedTensor
 *
 * The derived class must implement:
 * - perActiveVoxel(): Computes output value(s) for each active voxel
 */
template <torch::DeviceType DeviceTag, typename Derived, typename OutElementType>
struct BasePerActiveVoxelProcessor {
    using Out_t = Accessor_t<DeviceTag, OutElementType>;

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
            static_cast<Derived const *>(this)->perActiveVoxel(ijk, feature_idx, out_accessor);
        }
    }

    JaggedTensor
    execute(GridBatchImpl const &grid_batch,
            OutElementType const &out_element = OutElementType{},
            int const num_threads             = 1024) const {
        auto out_tensor =
            makeOutTensorFromGridBatch<DeviceTag, OutElementType>(grid_batch, out_element);
        auto out_accessor = makeAccessor<DeviceTag, OutElementType>(out_tensor);
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

/**
 * @brief CRTP base class for simple elementwise operations over tensor elements.
 *
 * This helper simplifies operations that transform input tensor elements into output
 * tensor elements, preserving the first dimension (number of elements) while potentially
 * changing the element type, rank, or shape. Each element in the input tensor is processed
 * independently.
 *
 * @tparam DeviceTag The device type (torch::kCPU, torch::kCUDA, or torch::kPrivateUse1)
 * @tparam Derived The derived class (CRTP pattern)
 * @tparam InElementType Element type descriptor for input tensor.
 *                       Can be ScalarElementType<T>, FixedElementType<T, Dims...>,
 *                       or DynamicElementType<T, N>.
 * @tparam OutElementType Element type descriptor for output tensor.
 *                        Can be ScalarElementType<T>, FixedElementType<T, Dims...>,
 *                        or DynamicElementType<T, N>.
 *
 * Usage example:
 * @code
 * template <torch::DeviceType DeviceTag>
 * struct MyTransform : public BasePerElementProcessor<DeviceTag,
 *                                                      MyTransform<DeviceTag>,
 *                                                      ScalarElementType<float>,
 *                                                      FixedElementType<float, 3>> {
 *     // Add any state needed for your operation
 *     float scale = 1.0f;
 *
 *     // Implement this method to define per-element computation
 *     __hostdev__ void
 *     perElement(int64_t const element_idx, auto in_accessor, auto out_accessor) const {
 *         // Your element processing logic here
 *         // element_idx: index of the current element being processed
 *         // in_accessor: tensor accessor for reading input
 *         // out_accessor: tensor accessor for writing output
 *         float value = in_accessor[element_idx];
 *         out_accessor[element_idx][0] = value * scale;
 *         out_accessor[element_idx][1] = value * scale * 2.0f;
 *         out_accessor[element_idx][2] = value * scale * 3.0f;
 *     }
 * };
 *
 * // Usage:
 * torch::Tensor result = MyTransform<torch::kCUDA>{.scale = 2.0f}.execute(input_tensor);
 * @endcode
 *
 * The base class handles:
 * - Output tensor allocation with correct shape based on OutElementType
 * - Dispatching to CPU/CUDA/PrivateUse1 implementations
 * - Iterating over all elements in the input tensor
 *
 * The derived class must implement:
 * - perElement(): Computes output value(s) for each input element
 */
template <torch::DeviceType DeviceTag,
          typename Derived,
          typename InElementType,
          typename OutElementType>
struct BasePerElementProcessor {
    using In_t  = Accessor_t<DeviceTag, InElementType>;
    using Out_t = Accessor_t<DeviceTag, OutElementType>;

    __hostdev__ void
    operator()(int64_t const element_idx, int64_t, In_t in_accessor, Out_t out_accessor) const {
        static_cast<Derived const *>(this)->perElement(element_idx, in_accessor, out_accessor);
    }

    torch::Tensor
    execute(torch::Tensor const &in_tensor,
            OutElementType const &out_element = OutElementType{},
            int const num_threads             = 1024) const {
        auto out_tensor =
            makeOutTensorFromTensor<DeviceTag, OutElementType>(in_tensor, out_element);
        auto out_accessor         = makeAccessor<DeviceTag, OutElementType>(out_tensor);
        constexpr size_t IN_NDIMS = 1 + InElementType::NDIMS;
        using IN_T                = typename InElementType::value_type;
        if constexpr (DeviceTag == torch::kCUDA) {
            forEachTensorElementChannelCUDA<IN_T, IN_NDIMS>(num_threads,
                                                            1, // num channels, ignored
                                                            in_tensor,
                                                            *static_cast<Derived const *>(this),
                                                            out_accessor);
        } else if constexpr (DeviceTag == torch::kPrivateUse1) {
            forEachTensorElementChannelPrivateUse1<IN_T, IN_NDIMS>(
                1, in_tensor, *static_cast<Derived const *>(this), out_accessor);
        } else if constexpr (DeviceTag == torch::kCPU) {
            forEachTensorElementChannelCPU<IN_T, IN_NDIMS>(
                1, in_tensor, *static_cast<Derived const *>(this), out_accessor);
        }
        return out_tensor;
    }
};

} // namespace detail
} // namespace fvdb

#endif
