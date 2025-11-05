// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/convolution/backend/SparseConvolutionGroundTruth.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

inline __hostdev__ int64_t
flat_leaf_count(BatchGridAccessor grid_batch) {
    return grid_batch.totalLeaves();
}

inline __hostdev__ int64_t
batch_index_from_flat_leaf_index(BatchGridAccessor grid_batch, int64_t flat_leaf_index) {
    return grid_batch.leafBatchIndex(flat_leaf_index);
}

inline __hostdev__ auto
grid_from_grid_batch(BatchGridAccessor grid_batch, int64_t batch_index) {
    auto const *const grid_ptr = grid_batch.grid(batch_index);
    return grid_ptr->getAccessor();
}

template <typename... Types> struct Accumulation_type_helper;

template <typename T> struct Accumulation_type_helper<T> {
    using value_type = at::acc_type<T, true>;
};

template <typename T, typename... Ts> struct Accumulation_type_helper<T, Ts...> {
    // other_count = sizeof...(Ts)
    // if constexpr (other_count < 1) { ... }

    using S = typename Accumulation_type_helper<Ts...>::value_type;
    // Get the type that results from adding T to the Rest type
    using value_type = decltype(std::declval<T>() + std::declval<S>());
};

template <typename... Types>
using Accumulation_t = typename Accumulation_type_helper<Types...>::value_type;

template <typename Src_grid, Src, Wgt, Dst_grid, Dst>
__hostdev__ inline void
process_dst_leaf(Src_grid src_grid,
                 Src src_features,
                 Wgt weights,
                 Dst_grid dst_grid,
                 Dst dst_features,

                 int32_t dst_leaf_index,
                 int32_t dst_voxel_index,

                 Int3 const &kernel_size,
                 Int3 const &stride) {
    using S = typename Src::value_type;
    using W = typename Wgt::value_type;
    using D = typename Dst::value_type;
    using A = Accumulation_t<S, W, D>;

    auto const &dst_leaf = leaf_node_at(src_grid, dst_leaf_index);

    auto out = static_cast<A>(0);
}

                 int32_t batch_index,
                 int32_t dst_leaf_index,
                     int32_t dst_voxel_index,
                     int32_t _unused_channel_index,
                     GridBatchImpl::Accessor coarseBatchAccessor,
                     GridBatchImpl::Accessor fineBatchAccessor,
                     const TensorAccessor<Dtype, 2> fineData,
                     TensorAccessor<Dtype, 2> outCoarseData,
                     nanovdb::Coord poolingFactor,
                     nanovdb::Coord stride,
                     Dtype avgFactor) {
                     using accscalar_t                      = at::acc_type<Dtype, true>;
                     const nanovdb::OnIndexGrid *coarseGrid = coarseBatchAccessor.grid(batchIdx);
                     const nanovdb::OnIndexGrid *fineGrid   = fineBatchAccessor.grid(batchIdx);
                     const typename nanovdb::OnIndexGrid::LeafNodeType &coarseLeaf =
                         coarseGrid->tree().template getFirstNode<0>()[leafIdx];
                     const auto fineGridAcc         = fineGrid->getAccessor();
                     const int64_t coarseBaseOffset = coarseBatchAccessor.voxelOffset(batchIdx);
                     const int64_t fineBaseOffset   = fineBatchAccessor.voxelOffset(batchIdx);
                     const int64_t coarseVoxelIndex = coarseLeaf.getValue(voxelIdx);

                     if (coarseVoxelIndex == 0) {
                         return;
                     }
                     const nanovdb::Coord coarseIjk = coarseLeaf.offsetToGlobalCoord(voxelIdx);
                     const nanovdb::Coord fineIjk0(coarseIjk[0] * stride[0],
                                                   coarseIjk[1] * stride[1],
                                                   coarseIjk[2] * stride[2]);
                     const int64_t coarseIndex =
                         coarseVoxelIndex - static_cast<int64_t>(1) + coarseBaseOffset;
                     accscalar_t avgValue = static_cast<accscalar_t>(0.0);

                     for (nanovdb::Coord::ValueType i = 0; i < poolingFactor[0]; i += 1) {
                         for (nanovdb::Coord::ValueType j = 0; j < poolingFactor[1]; j += 1) {
                             for (nanovdb::Coord::ValueType k = 0; k < poolingFactor[2]; k += 1) {
                                 nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                                 if (!fineGridAcc.isActive(fineIjk)) {
                                     continue;
                                 }
                                 const int64_t fineIndex =
                                     (int64_t)fineGridAcc.getValue(fineIjk) + fineBaseOffset - 1;
                                 avgValue +=
                                     static_cast<accscalar_t>(fineData[fineIndex][channelIdx]);
                             }
                         }
                     }

                     outCoarseData[coarseIndex][channelIdx] =
                         static_cast<Dtype>(avgValue) * avgFactor;
                 }

                 } // End anonymous namespace

                 // GPU Implementation using GridBatchImpl
                 template <typename scalar_t>
                 __global__ void
                 sparseConvGroundTruthKernel(

                     scalar_t const *input_features, // (B, src_active_voxels, inC)
                     scalar_t *output_features,      // (B, dst_active_voxels, outC)
                     scalar_t const *weights,        // (outC, inC, Axis0, Axis1, Axis2)
                     BatchGridAccessor src_grid_acc,
                     BatchGridAccessor dst_grid_acc,
                     int const *kernel_size,         // (3,) - [Axis0, Axis1, Axis2]
                     int const *stride,              // (3,) - [Axis0, Axis1, Axis2]
                     int batch_size,
                     int in_channels,
                     int out_channels) {
                     int leafIdx = blockIdx.x;
                     int tid     = threadIdx.x;

                     if (leafIdx >= dst_grid_acc.totalLeaves())
                         return;

                     // Get destination leaf information
                     auto const batchIdx        = dst_grid_acc.leafBatchIndex(leafIdx);
                     auto const localLeafIdx    = leafIdx - dst_grid_acc.leafOffset(batchIdx);
                     auto const dst_base_offset = dst_grid_acc.voxelOffset(batchIdx);

                     auto const *dst_grid = dst_grid_acc.grid(batchIdx);
                     auto const &dst_leaf =
                         dst_grid->tree().template getFirstNode<0>()[localLeafIdx];
                     auto const dst_origin = dst_leaf.origin();
                     auto dst_accessor     = dst_grid->getAccessor();

                     // Get source grid accessor
                     auto const *src_grid = src_grid_acc.grid(batchIdx);
                     auto src_accessor    = src_grid->getAccessor();

                     // Iterating over the voxels in the destination leaf.
                     if (tid < 512) { // 8x8x8 = 512
                         int const d0 = (tid >> 6) & 0x7;
                         int const d1 = (tid >> 3) & 0x7;
                         int const d2 = tid & 0x7;

                         auto const dst_coord = dst_origin.offsetBy(d0, d1, d2);

                         if (dst_accessor.isActive(dst_coord)) {
                             int dst_voxel_idx =
                                 dst_accessor.getValue(dst_coord) - 1 + dst_base_offset;

                             // For each output channel
                             // While it is obviously painful for us to iterate over the channels
                             // outside the kernel neighbor voxel visits, the channel count can be
                             // quite large, and we won't necessarily be able to store
                             for (int out_channel = 0; out_channel < out_channels; out_channel++) {
                                 scalar_t sum = 0.0f;

                                 // For each kernel position
                                 for (int k0 = 0; k0 < kernel_size[0]; k0++) {
                                     for (int k1 = 0; k1 < kernel_size[1]; k1++) {
                                         for (int k2 = 0; k2 < kernel_size[2]; k2++) {
                                             // Calculate source coordinates
                                             nanovdb::Coord src_coord(
                                                 dst_coord[0] * stride[0] + k0 - kernel_size[0] / 2,
                                                 dst_coord[1] * stride[1] + k1 - kernel_size[1] / 2,
                                                 dst_coord[2] * stride[2] + k2 -
                                                     kernel_size[2] / 2);

                                             // Check if source voxel exists and get its index
                                             if (src_accessor.isActive(src_coord)) {
                                                 int src_voxel_idx =
                                                     src_accessor.getValue(src_coord) - 1 +
                                                     src_grid_acc.voxelOffset(batchIdx);

                                                 // Accumulate contribution
                                                 for (int in_channel = 0; in_channel < in_channels;
                                                      in_channel++) {
                                                     int input_offset =
                                                         batchIdx * src_grid_acc.totalVoxels() *
                                                             in_channels +
                                                         src_voxel_idx * in_channels + in_channel;

                                                     int weight_offset =
                                                         k2 +
                                                         kernel_size[2] *
                                                             (k1 +
                                                              kernel_size[1] *
                                                                  (k0 + kernel_size[0] *
                                                                            (in_channel +
                                                                             in_channels *
                                                                                 (out_channel))));

                                                     sum += input_features[input_offset] *
                                                            weights[weight_offset];
                                                 }
                                             }
                                         }
                                     }
                                 }

                                 // Store result
                                 int output_offset =
                                     batchIdx * dst_grid_acc.totalVoxels() * out_channels +
                                     dst_voxel_idx * out_channels + out_channel;
                                 output_features[output_offset] = sum;
                             }
                         }
                     }
                 }

                 template <>
                 void
                 dispatchSparseConvolutionGroundTruth<torch::kCUDA>(torch::Tensor input_features,
                                                                    torch::Tensor output_features,
                                                                    torch::Tensor weights,
                                                                    const GridBatchImpl &src_grid,
                                                                    const GridBatchImpl &dst_grid,
                                                                    torch::Tensor kernel_size,
                                                                    torch::Tensor stride,
                                                                    bool transpose) {
                     // Basic tensor validation
                     TORCH_CHECK(input_features.device().is_cuda(), "Input must be CUDA tensor");
                     TORCH_CHECK(input_features.device() == output_features.device(),
                                 "All tensors must be on same device");
                     TORCH_CHECK(input_features.device() == weights.device(),
                                 "All tensors must be on same device");

                     // GridBatchImpl validation
                     src_grid.checkDevice(input_features);
                     dst_grid.checkDevice(output_features);

                     // Validate tensor dimensions against GridBatchImpl
                     TORCH_CHECK(input_features.size(0) == src_grid.batchSize(),
                                 "Input batch size (",
                                 input_features.size(0),
                                 ") must match source grid batch size (",
                                 src_grid.batchSize(),
                                 ")");
                     TORCH_CHECK(output_features.size(0) == dst_grid.batchSize(),
                                 "Output batch size (",
                                 output_features.size(0),
                                 ") must match destination grid batch size (",
                                 dst_grid.batchSize(),
                                 ")");
                     TORCH_CHECK(input_features.size(1) == src_grid.totalVoxels(),
                                 "Input voxel count (",
                                 input_features.size(1),
                                 ") must match source grid total voxels (",
                                 src_grid.totalVoxels(),
                                 ")");
                     TORCH_CHECK(output_features.size(1) == dst_grid.totalVoxels(),
                                 "Output voxel count (",
                                 output_features.size(1),
                                 ") must match destination grid total voxels (",
                                 dst_grid.totalVoxels(),
                                 ")");

                     // Validate weight dimensions
                     TORCH_CHECK(weights.size(0) == output_features.size(2),
                                 "Weight output channels (",
                                 weights.size(0),
                                 ") must match output feature channels (",
                                 output_features.size(2),
                                 ")");
                     TORCH_CHECK(weights.size(1) == input_features.size(2),
                                 "Weight input channels (",
                                 weights.size(1),
                                 ") must match input feature channels (",
                                 input_features.size(2),
                                 ")");
                     TORCH_CHECK(weights.size(2) == kernel_size[0].item<int>(),
                                 "Weight axis 0 (",
                                 weights.size(2),
                                 ") must match kernel size 0 (",
                                 kernel_size[0].item<int>(),
                                 ")");
                     TORCH_CHECK(weights.size(3) == kernel_size[1].item<int>(),
                                 "Weight axis 1 (",
                                 weights.size(3),
                                 ") must match kernel size 1 (",
                                 kernel_size[1].item<int>(),
                                 ")");
                     TORCH_CHECK(weights.size(4) == kernel_size[2].item<int>(),
                                 "Weight axis 2 (",
                                 weights.size(4),
                                 ") must match kernel size 2 (",
                                 kernel_size[2].item<int>(),
                                 ")");

                     c10::cuda::CUDAGuard deviceGuard(input_features.device());

                     // Extract dimensions
                     int batch_size   = input_features.size(0);
                     int in_channels  = input_features.size(2);
                     int out_channels = output_features.size(2);

                     // Zero output
                     output_features.zero_();

                     // Get grid accessors
                     auto src_grid_acc = src_grid.deviceAccessor();
                     auto dst_grid_acc = dst_grid.deviceAccessor();

                     // Launch kernel - one block per destination leaf
                     int num_leaves = dst_grid.totalLeaves();

                     AT_DISPATCH_V2(input_features.scalar_type(),
                                    "sparseConvGroundTruth",
                                    AT_WRAP([&] {
                                        sparseConvGroundTruthKernel<scalar_t><<<num_leaves, 512>>>(
                                            input_features.data_ptr<scalar_t>(),
                                            output_features.data_ptr<scalar_t>(),
                                            weights.data_ptr<scalar_t>(),
                                            src_grid_acc,
                                            dst_grid_acc,
                                            kernel_size.data_ptr<int>(),
                                            stride.data_ptr<int>(),
                                            batch_size,
                                            in_channels,
                                            out_channels,
                                            transpose);
                                    }),
                                    AT_EXPAND(AT_FLOATING_TYPES),
                                    c10::kHalf,
                                    c10::kBFloat16);
                 }

                 // CPU Implementation using GridBatchImpl
                 template <typename scalar_t>
                 void
                 sparseConvGroundTruthCPU(const scalar_t *input_features,
                                          scalar_t *output_features,
                                          const scalar_t *weights,
                                          const GridBatchImpl::Accessor &src_grid_acc,
                                          const GridBatchImpl::Accessor &dst_grid_acc,
                                          const int *kernel_size,
                                          const int *stride,
                                          int batch_size,
                                          int in_channels,
                                          int out_channels,
                                          bool transpose) {
                     // Zero output
                     std::fill(output_features,
                               output_features +
                                   batch_size * dst_grid_acc.totalVoxels() * out_channels,
                               0.0f);

                     // For each destination leaf
                     for (int leafIdx = 0; leafIdx < dst_grid_acc.totalLeaves(); leafIdx++) {
                         const int64_t batchIdx     = dst_grid_acc.leafBatchIndex(leafIdx);
                         const int64_t localLeafIdx = leafIdx - dst_grid_acc.leafOffset(batchIdx);
                         const int64_t dst_base_offset = dst_grid_acc.voxelOffset(batchIdx);

                         const nanovdb::OnIndexGrid *dst_grid = dst_grid_acc.grid(batchIdx);
                         const nanovdb::OnIndexTree::LeafNodeType &dst_leaf =
                             dst_grid->tree().template getFirstNode<0>()[localLeafIdx];
                         const nanovdb::Coord dst_origin = dst_leaf.origin();
                         auto dst_accessor               = dst_grid->getAccessor();

                         // Get source grid accessor
                         const nanovdb::OnIndexGrid *src_grid = src_grid_acc.grid(batchIdx);
                         auto src_accessor                    = src_grid->getAccessor();

                         // Process each voxel in the destination leaf
                         for (int di = 0; di < 8; di++) {
                             for (int dj = 0; dj < 8; dj++) {
                                 for (int dk = 0; dk < 8; dk++) {
                                     auto dst_coord = dst_origin.offsetBy(di, dj, dk);

                                     if (dst_accessor.isActive(dst_coord)) {
                                         int dst_voxel_idx =
                                             dst_accessor.getValue(dst_coord) - 1 + dst_base_offset;

                                         // For each output channel
                                         for (int out_channel = 0; out_channel < out_channels;
                                              out_channel++) {
                                             scalar_t sum = 0.0f;

                                             // For each kernel position
                                             for (int kx = 0; kx < kernel_size[0]; kx++) {
                                                 for (int ky = 0; ky < kernel_size[1]; ky++) {
                                                     for (int kz = 0; kz < kernel_size[2]; kz++) {
                                                         // Calculate source coordinates
                                                         nanovdb::Coord src_coord;
                                                         if (transpose) {
                                                             src_coord = nanovdb::Coord(
                                                                 dst_coord.x() * stride[0] + kx -
                                                                     kernel_size[0] / 2,
                                                                 dst_coord.y() * stride[1] + ky -
                                                                     kernel_size[1] / 2,
                                                                 dst_coord.z() * stride[2] + kz -
                                                                     kernel_size[2] / 2);
                                                         } else {
                                                             src_coord = nanovdb::Coord(
                                                                 dst_coord.x() * stride[0] + kx -
                                                                     kernel_size[0] / 2,
                                                                 dst_coord.y() * stride[1] + ky -
                                                                     kernel_size[1] / 2,
                                                                 dst_coord.z() * stride[2] + kz -
                                                                     kernel_size[2] / 2);
                                                         }

                                                         // Check if source voxel exists
                                                         if (src_accessor.isActive(src_coord)) {
                                                             int src_voxel_idx =
                                                                 src_accessor.getValue(src_coord) -
                                                                 1 +
                                                                 src_grid_acc.voxelOffset(batchIdx);

                                                             // Accumulate contribution
                                                             for (int in_channel = 0;
                                                                  in_channel < in_channels;
                                                                  in_channel++) {
                                                                 int input_offset =
                                                                     batchIdx *
                                                                         src_grid_acc
                                                                             .totalVoxels() *
                                                                         in_channels +
                                                                     src_voxel_idx * in_channels +
                                                                     in_channel;
                                                                 int weight_offset =
                                                                     out_channel * in_channels *
                                                                         kernel_size[0] *
                                                                         kernel_size[1] *
                                                                         kernel_size[2] +
                                                                     in_channel * kernel_size[0] *
                                                                         kernel_size[1] *
                                                                         kernel_size[2] +
                                                                     kx * kernel_size[1] *
                                                                         kernel_size[2] +
                                                                     ky * kernel_size[2] + kz;

                                                                 sum +=
                                                                     input_features[input_offset] *
                                                                     weights[weight_offset];
                                                             }
                                                         }
                                                     }
                                                 }
                                             }

                                             // Store result
                                             int output_offset =
                                                 batchIdx * dst_grid_acc.totalVoxels() *
                                                     out_channels +
                                                 dst_voxel_idx * out_channels + out_channel;
                                             output_features[output_offset] = sum;
                                         }
                                     }
                                 }
                             }
                         }
                     }
                 }

                 template <>
                 void
                 dispatchSparseConvolutionGroundTruth<torch::kCPU>(torch::Tensor input_features,
                                                                   torch::Tensor output_features,
                                                                   torch::Tensor weights,
                                                                   const GridBatchImpl &src_grid,
                                                                   const GridBatchImpl &dst_grid,
                                                                   torch::Tensor kernel_size,
                                                                   torch::Tensor stride,
                                                                   bool transpose) {
                     // Basic tensor validation
                     TORCH_CHECK(input_features.device().is_cpu(), "Input must be CPU tensor");
                     TORCH_CHECK(output_features.device().is_cpu(), "Output must be CPU tensor");
                     TORCH_CHECK(weights.device().is_cpu(), "Weights must be CPU tensor");

                     // GridBatchImpl validation
                     src_grid.checkDevice(input_features);
                     dst_grid.checkDevice(output_features);

                     // Validate tensor dimensions against GridBatchImpl
                     TORCH_CHECK(input_features.size(0) == src_grid.batchSize(),
                                 "Input batch size (",
                                 input_features.size(0),
                                 ") must match source grid batch size (",
                                 src_grid.batchSize(),
                                 ")");
                     TORCH_CHECK(output_features.size(0) == dst_grid.batchSize(),
                                 "Output batch size (",
                                 output_features.size(0),
                                 ") must match destination grid batch size (",
                                 dst_grid.batchSize(),
                                 ")");
                     TORCH_CHECK(input_features.size(1) == src_grid.totalVoxels(),
                                 "Input voxel count (",
                                 input_features.size(1),
                                 ") must match source grid total voxels (",
                                 src_grid.totalVoxels(),
                                 ")");
                     TORCH_CHECK(output_features.size(1) == dst_grid.totalVoxels(),
                                 "Output voxel count (",
                                 output_features.size(1),
                                 ") must match destination grid total voxels (",
                                 dst_grid.totalVoxels(),
                                 ")");

                     // Validate weight dimensions
                     TORCH_CHECK(weights.size(0) == output_features.size(2),
                                 "Weight output channels (",
                                 weights.size(0),
                                 ") must match output feature channels (",
                                 output_features.size(2),
                                 ")");
                     TORCH_CHECK(weights.size(1) == input_features.size(2),
                                 "Weight input channels (",
                                 weights.size(1),
                                 ") must match input feature channels (",
                                 input_features.size(2),
                                 ")");
                     TORCH_CHECK(weights.size(2) == kernel_size[0].item<int>(),
                                 "Weight axis 0 (",
                                 weights.size(2),
                                 ") must match kernel size 0 (",
                                 kernel_size[0].item<int>(),
                                 ")");
                     TORCH_CHECK(weights.size(3) == kernel_size[1].item<int>(),
                                 "Weight axis 1 (",
                                 weights.size(3),
                                 ") must match kernel size 1 (",
                                 kernel_size[1].item<int>(),
                                 ")");
                     TORCH_CHECK(weights.size(4) == kernel_size[2].item<int>(),
                                 "Weight axis 2 (",
                                 weights.size(4),
                                 ") must match kernel size 2 (",
                                 kernel_size[2].item<int>(),
                                 ")");

                     // Extract dimensions
                     int batch_size   = input_features.size(0);
                     int in_channels  = input_features.size(2);
                     int out_channels = output_features.size(2);

                     // Get grid accessors
                     auto src_grid_acc = src_grid.hostAccessor();
                     auto dst_grid_acc = dst_grid.hostAccessor();

                     AT_DISPATCH_V2(input_features.scalar_type(),
                                    "sparseConvGroundTruthCPU",
                                    AT_WRAP([&] {
                                        sparseConvGroundTruthCPU<scalar_t>(
                                            input_features.data_ptr<scalar_t>(),
                                            output_features.data_ptr<scalar_t>(),
                                            weights.data_ptr<scalar_t>(),
                                            src_grid_acc,
                                            dst_grid_acc,
                                            kernel_size.data_ptr<int>(),
                                            stride.data_ptr<int>(),
                                            batch_size,
                                            in_channels,
                                            out_channels,
                                            transpose);
                                    }),
                                    AT_EXPAND(AT_FLOATING_TYPES),
                                    c10::kHalf,
                                    c10::kBFloat16);
                 }

                 } // namespace ops
                 } // namespace detail
                 } // namespace fvdb
