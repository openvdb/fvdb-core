// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH
#define FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH

#include <fvdb/GridStorage.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/GridHeaderUtils.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/HandleStorage.h>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace fvdb::detail {

/// @brief Stamps a grid's position in its batch into its header and disables the checksum, which
///        the write invalidates. One thread. Shared by every site that lays grids into a buffer
///        by hand (merge, concatenate, compaction).
__global__ inline void
updateGridCountAndZeroChecksum(nanovdb::GridData *d_data, uint32_t gridIndex, uint32_t gridCount) {
    d_data->mGridIndex = gridIndex;
    d_data->mGridCount = gridCount;
    d_data->mChecksum.disable();
}

/// @brief Wraps device bytes whose layout is already known in a GridHandle without parsing them:
///        @p meta must list every grid in order, packed end to end from offset 0 at NanoVDB's data
///        alignment, which is the only layout the rest of the stack (the chain parse, the typed
///        accessors) can consume. That and the buffer's extent are checked; the grid types are
///        taken on trust, as they are when copyTo adopts metadata.
inline GridStorage::DeviceHandle
makeDeviceHandleFromLayout(DeviceGridBuffer &&buffer,
                           std::vector<nanovdb::GridHandleMetaData> meta) {
    const uint64_t extent = buffer.size_bytes();
    TORCH_CHECK(meta.empty() == (extent == 0),
                "makeDeviceHandleFromLayout: metadata and buffer disagree about emptiness");
    uint64_t end = 0;
    for (const auto &m: meta) {
        // Checked without forming offset + size, which can wrap.
        TORCH_CHECK(
            m.offset == end && m.offset % NANOVDB_DATA_ALIGNMENT == 0 &&
                m.size >= sizeof(nanovdb::GridData) && m.size <= extent - m.offset,
            "makeDeviceHandleFromLayout: grids must be packed, aligned and inside the buffer");
        end = m.offset + m.size;
    }
    return nanovdb::cuda::detail::HandleFactory::make(std::move(buffer), std::move(meta));
}

/// @brief Lays the grids of @p handles end to end in one allocation from @p proto's resource,
///        on @p stream, fixing each header's index/count: the single-space replacement for
///        nanovdb::cuda::mergeGridHandles. No parse kernel and no per-grid synchronization; the
///        metadata is assembled on the host from the sources'. All sources must live on the
///        prototype's device. Each source's retained stream is ordered before its copy and after
///        it, so sources may be destroyed as soon as this returns. Checksums are disabled, as the
///        CUDA MakeContiguous and ConcatenateGrids kernel above leaves them; the host paths
///        (nanovdb::mergeGrids and the CPU sides of those two ops, through tools::updateGridCount)
///        and the replaced nanovdb::cuda::mergeGridHandles update an existing checksum instead, so
///        a builder that computes one before merging loses it here (see #770 for whether step 5
///        recomputes or stops computing it).
inline GridStorage::DeviceHandle
mergeDeviceGridHandles(const std::vector<GridStorage::DeviceHandle> &handles,
                       const DeviceGridBuffer &proto,
                       cudaStream_t stream) {
    uint64_t totalBytes = 0;
    uint32_t gridCount  = 0;
    for (const auto &h: handles) {
        gridCount += h.gridCount();
        for (uint32_t n = 0; n < h.gridCount(); ++n) {
            totalBytes += h.gridSize(n);
        }
    }
    const torch::Device device = proto.resource().device();
    // The copies and header kernels below run on `stream`, a stream of the prototype's device;
    // the allocation and ordering helpers guard internally but restore the caller's device.
    c10::OptionalDeviceGuard deviceGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                          : std::nullopt);
    for (const auto &h: handles) {
        TORCH_CHECK(h.isEmpty() || h.buffer().resource().device() == device,
                    "mergeDeviceGridHandles: source on ",
                    h.buffer().resource().device(),
                    " cannot be merged onto ",
                    device);
    }
    if (totalBytes == 0) {
        // Keep the prototype's device and stream, as a non-empty result would.
        return nanovdb::cuda::detail::HandleFactory::make(
            DeviceGridBuffer(stream, proto.resource(), 0, nanovdb::cuda::noInit),
            std::vector<nanovdb::GridHandleMetaData>{});
    }

    DeviceGridBuffer buffer(stream, proto.resource(), totalBytes, nanovdb::cuda::noInit);
    std::vector<nanovdb::GridHandleMetaData> meta;
    meta.reserve(gridCount);

    uint64_t offset = 0;
    uint32_t index  = 0;
    for (const auto &h: handles) {
        if (h.isEmpty()) {
            continue;
        }
        const cudaStream_t srcStream = h.buffer().stream();
        orderStreamAfter(stream, device, srcStream, device);
        const auto &srcMeta = nanovdb::cuda::detail::HandleFactory::meta(h);
        for (uint32_t n = 0; n < h.gridCount(); ++n) {
            auto *dst = buffer.data() + offset;
            C10_CUDA_CHECK(
                cudaMemcpyAsync(dst,
                                static_cast<const std::byte *>(h.deviceData()) + srcMeta[n].offset,
                                srcMeta[n].size,
                                cudaMemcpyDeviceToDevice,
                                stream));
            updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(
                reinterpret_cast<nanovdb::GridData *>(dst), index, gridCount);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            meta.push_back(
                nanovdb::GridHandleMetaData{offset, srcMeta[n].size, srcMeta[n].gridType});
            offset += srcMeta[n].size;
            ++index;
        }
        // The source may be destroyed as soon as this returns: its block goes back to srcStream,
        // which now waits for the copies out of it.
        orderStreamAfter(srcStream, device, stream, device);
    }
    return nanovdb::cuda::detail::HandleFactory::make(std::move(buffer), std::move(meta));
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_NANOVDB_DEVICEGRIDHANDLEUTILS_CUH
