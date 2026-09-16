// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_LEGACYGRIDHANDLE_H
#define FVDB_DETAIL_UTILS_NANOVDB_LEGACYGRIDHANDLE_H

#include <fvdb/GridStorage.h>
#include <fvdb/TorchDeviceBuffer.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/HandleStorage.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstring>
#include <utility>

namespace fvdb::detail {

/// @brief Moves the grids of a dual-space TorchDeviceBuffer handle into GridStorage. Temporary:
///        this is the seam between the builders, which still produce GridHandle<TorchDeviceBuffer>
///        (#770 step 5 moves them onto GridStorage::deviceProto and HostBuffer directly), and
///        GridBatchData, which now holds GridStorage. It costs one copy of the grid bytes per
///        build; it disappears with the builders' conversion.
///
///        The handle's metadata was validated when it was constructed, so it is adopted. The copy
///        runs on the device's current torch stream and is synchronized before returning: the
///        legacy buffer's block is keyed to whatever stream was current when the builder allocated
///        it, which need not be current now and which the buffer does not record, so nothing
///        short of completion makes destroying the handle right after this safe.
inline GridStorage
adoptLegacyGridHandle(nanovdb::GridHandle<TorchDeviceBuffer> &&handle) {
    using HandleFactory        = nanovdb::cuda::detail::HandleFactory;
    const torch::Device device = handle.buffer().device();
    const uint64_t bytes       = handle.buffer().size();
    if (bytes == 0) {
        return GridStorage::empty(device);
    }
    if (device.is_cpu()) {
        nanovdb::HostBuffer buf(bytes);
        std::memcpy(buf.data(), handle.buffer().data(), bytes);
        return GridStorage(HandleFactory::make(std::move(buf), HandleFactory::meta(handle)));
    }
    c10::OptionalDeviceGuard deviceGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                          : std::nullopt);
    const cudaStream_t stream = detail::storageStream(device);
    DeviceGridBuffer buf(stream, TorchDeviceResource(device), bytes, nanovdb::cuda::noInit);
    C10_CUDA_CHECK(cudaMemcpyAsync(
        buf.data(), handle.buffer().deviceData(), bytes, cudaMemcpyDefault, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return GridStorage(HandleFactory::make(std::move(buf), HandleFactory::meta(handle)), device);
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_NANOVDB_LEGACYGRIDHANDLE_H
