// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/utils/nanovdb/ActiveVoxelIterator.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <ATen/TensorUtils.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace fvdb::detail::ops {

template <>
void
dispatchInject<torch::kCUDA>(const GridBatchImpl &dstGridBatch,
                             const GridBatchImpl &srcGridBatch,
                             JaggedTensor &dst,
                             const JaggedTensor &src) {
    c10::cuda::CUDAGuard deviceGuard(dstGridBatch.device());

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, dstGridBatch.device());

    TORCH_CHECK_VALUE(dst.is_contiguous(), "Destination tensor must be contiguous");
    TORCH_CHECK_VALUE(src.is_contiguous(), "Source tensor must be contiguous");
    TORCH_CHECK_VALUE(dst.rdim() == src.rdim(),
                      "Source/Destination tensors should have matching dimensions");
    TORCH_CHECK_VALUE(dst.scalar_type() == src.scalar_type(),
                      "Source/Destination tensors should have matching scalar types");

    for (auto i = 1; i < dst.rdim(); i++) {
        TORCH_CHECK_VALUE(dst.rsize(i) == src.rsize(i),
                          "Source/Destination tensors should have matching feature dimensions");
    }

    int64_t featureDim = 1;
    for (auto j = 1; j < dst.rdim(); j++) {
        featureDim *= dst.rsize(j);
    }

    // Create a grid for each batch item and store the handles
    for (int i = 0; i < dstGridBatch.batchSize(); i += 1) {
        const nanovdb::OnIndexGrid *dstGrid =
            dstGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        const nanovdb::OnIndexGrid *srcGrid =
            srcGridBatch.nanoGridHandle().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(dstGrid, "Destination grid is null");
        TORCH_CHECK(srcGrid, "Source grid is null");

        torch::Tensor dstI       = dst.index(i).jdata();
        const torch::Tensor srcI = src.index(i).jdata();

        const auto srcLeafCount     = srcGridBatch.numLeavesAt(i);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(srcGridBatch.device().index());

        AT_DISPATCH_V2(src.scalar_type(),
                       "Inject",
                       AT_WRAP([&] {
                           using Op = nanovdb::util::cuda::
                               InjectGridFeatureFunctor<nanovdb::ValueOnIndex, scalar_t, -1>;
                           nanovdb::util::cuda::operatorKernel<Op>
                               <<<srcLeafCount, Op::MaxThreadsPerBlock, 0, stream.stream()>>>(
                                   srcGrid,
                                   dstGrid,
                                   srcI.const_data_ptr<scalar_t>(),
                                   dstI.data_ptr<scalar_t>(),
                                   featureDim);
                           C10_CUDA_KERNEL_LAUNCH_CHECK();
                       }),
                       AT_EXPAND(AT_ALL_TYPES),
                       torch::kFloat16);
    }
}

template <>
void
dispatchInject<torch::kCPU>(const GridBatchImpl &dstGridBatch,
                            const GridBatchImpl &srcGridBatch,
                            JaggedTensor &dst,
                            const JaggedTensor &src) {
    TORCH_CHECK_VALUE(dst.is_contiguous(), "Destination tensor must be contiguous");
    TORCH_CHECK_VALUE(src.is_contiguous(), "Source tensor must be contiguous");
    TORCH_CHECK_VALUE(dst.rdim() == src.rdim(),
                      "Source/Destination tensors should have matching dimensions");
    TORCH_CHECK_VALUE(dst.scalar_type() == src.scalar_type(),
                      "Source/Destination tensors should have matching scalar types");

    for (auto i = 1; i < dst.rdim(); i++) {
        TORCH_CHECK_VALUE(dst.rsize(i) == src.rsize(i),
                          "Source/Destination tensors should have matching feature dimensions");
    }

    const auto [srcJData, dstJData] = [&]() {
        if (src.rdim() == 1) {
            return std::make_tuple(src.jdata().unsqueeze(-1), dst.jdata().unsqueeze(-1));
        } else {
            return std::make_tuple(src.jdata().view({src.rsize(0), -1}),
                                   dst.jdata().view({dst.rsize(0), -1}));
        }
    }();

    const int64_t featureDim = srcJData.size(1);

    AT_DISPATCH_V2(src.scalar_type(),
                   "Inject",
                   AT_WRAP([&] {
                       const auto srcJDataAccessor = srcJData.accessor<scalar_t, 2>();
                       auto dstJDataAccessor       = dstJData.accessor<scalar_t, 2>();

                       for (auto i = 0; i < srcGridBatch.batchSize(); i += 1) {
                           const nanovdb::OnIndexGrid *grid =
                               srcGridBatch.nanoGridHandle().grid<nanovdb::ValueOnIndex>(i);
                           const nanovdb::OnIndexGrid *dstGrid =
                               dstGridBatch.nanoGridHandle().grid<nanovdb::ValueOnIndex>(i);
                           auto dstAccessor           = dstGrid->getAccessor();
                           const int64_t baseSrcIndex = srcGridBatch.cumVoxelsAt(i);
                           const int64_t baseDstIndex = dstGridBatch.cumVoxelsAt(i);
                           for (auto it = ActiveVoxelIterator<-1>(grid->tree()); it.isValid();
                                ++it) {
                               const nanovdb::Coord ijk = it->first;
                               const auto srcIndex      = it->second;

                               const int64_t dstIndex = int64_t(dstAccessor.getValue(ijk)) - 1;
                               if (dstIndex < 0) {
                                   continue; // Skip if the voxel is not in the destination grid
                               }

                               for (int c = 0; c < featureDim; ++c) {
                                   dstJDataAccessor[dstIndex + baseDstIndex][c] =
                                       srcJDataAccessor[srcIndex + baseSrcIndex][c];
                               }
                           }
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES),
                   torch::kFloat16);
}

} // namespace fvdb::detail::ops
