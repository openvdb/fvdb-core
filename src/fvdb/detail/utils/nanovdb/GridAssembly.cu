// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/StreamOrdering.h>
#include <fvdb/detail/utils/nanovdb/DeviceGridHandleUtils.cuh>

#include <nanovdb/HostBuffer.h>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <utility>

namespace fvdb::detail {

namespace {

struct StreamKey {
    torch::Device device;
    cudaStream_t stream;
    bool
    operator==(const StreamKey &o) const {
        return device == o.device && stream == o.stream;
    }
};

// The distinct (device, stream) pairs among the device-resident sources; CPU sources have no
// writers to order against.
std::vector<StreamKey>
distinctSourceStreams(const std::vector<GridSpanSource> &sources) {
    std::vector<StreamKey> keys;
    for (const auto &s: sources) {
        if (s.device.is_cpu() || s.spans.empty()) {
            continue;
        }
        const StreamKey k{s.device, s.stream};
        if (std::find(keys.begin(), keys.end(), k) == keys.end()) {
            keys.push_back(k);
        }
    }
    return keys;
}

// Copies one span to `dst` on `stream`. Between two different CUDA devices this must be a peer
// copy (torch's allocator backends do not support a plain memcpy there without peer access);
// every other pairing, host or unified memory included, is cudaMemcpyDefault.
void
copySpan(void *dst,
         const torch::Device &dstDevice,
         const GridSpan &span,
         const torch::Device &srcDevice,
         cudaStream_t stream) {
    if (srcDevice.is_cpu() && dstDevice.is_cpu()) {
        std::memcpy(dst, span.bytes, span.size);
        return;
    }
    if (srcDevice.is_cuda() && dstDevice.is_cuda() && srcDevice != dstDevice) {
        C10_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst, dstDevice.index(), span.bytes, srcDevice.index(), span.size, stream));
        return;
    }
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, span.bytes, span.size, cudaMemcpyDefault, stream));
}

// Destination bytes for device storage, in the buffer type the device's handle kind requires.
// PrivateUse1 gets a UnifiedGridBuffer, allocated synchronously. CUDA gets a DeviceGridBuffer
// that retains `stream`. Write through data(), then adopt() the finished layout.
class DestinationBuffer {
  public:
    DestinationBuffer(const torch::Device &device, cudaStream_t stream, uint64_t bytes)
        : mDevice(device) {
        if (device.is_privateuseone()) {
            mData = mUnified.emplace(bytes, nanovdb::cuda::noInit).data();
        } else {
            mData = mCuda.emplace(stream, TorchDeviceResource(device), bytes, nanovdb::cuda::noInit)
                        .data();
        }
    }

    std::byte *
    data() const {
        return mData;
    }

    GridStorage
    adopt(std::vector<nanovdb::GridHandleMetaData> meta) && {
        if (mUnified) {
            return GridStorage(makeDeviceHandleFromLayout(std::move(*mUnified), std::move(meta)),
                               mDevice);
        }
        return GridStorage(makeDeviceHandleFromLayout(std::move(*mCuda), std::move(meta)), mDevice);
    }

  private:
    torch::Device mDevice;
    std::optional<DeviceGridBuffer> mCuda;
    std::optional<UnifiedGridBuffer> mUnified;
    std::byte *mData = nullptr;
};

} // namespace

GridStorage
assembleGridStorage(const std::vector<GridSpanSource> &sources,
                    const torch::Device &requested,
                    cudaStream_t stream) {
    const torch::Device device = GridStorage::resolveDevice(requested);
    uint64_t totalBytes        = 0;
    uint32_t gridCount         = 0;
    for (const auto &s: sources) {
        for (const auto &span: s.spans) {
            // NanoVDB's own writers can emit a standalone grid whose blind payload is not padded
            // to the data alignment, so only the header's presence is required here; the device
            // layout check below rejects an unaligned grid that is not last, which cannot be
            // packed.
            TORCH_CHECK(span.size >= sizeof(nanovdb::GridData),
                        "assembleGridStorage: a grid of ",
                        span.size,
                        " bytes is not a valid NanoVDB grid");
            totalBytes += span.size;
            ++gridCount;
        }
    }
    if (gridCount == 0) {
        return GridStorage::empty(device);
    }
    // Packed end to end, every header but the first lands at the running offset, so every span
    // but the last must keep the data alignment; checked before any copy or header write, since
    // a misaligned header is not a valid GridData to write through on either side.
    {
        uint32_t seen = 0;
        for (const auto &s: sources) {
            for (const auto &span: s.spans) {
                ++seen;
                TORCH_CHECK(seen == gridCount || span.size % NANOVDB_DATA_ALIGNMENT == 0,
                            "assembleGridStorage: grid ",
                            seen - 1,
                            " of ",
                            gridCount,
                            " is ",
                            span.size,
                            " bytes, not a multiple of ",
                            NANOVDB_DATA_ALIGNMENT,
                            "; only the last grid may be unpadded");
            }
        }
    }

    const auto sourceStreams = distinctSourceStreams(sources);

    if (device.is_cpu()) {
        // Each device source is copied on its own retained stream, under its own device (a null
        // handle is device-relative), which is ordered after its writers by construction; the
        // stream is then waited for, since a copy into pageable memory is only *possibly*
        // synchronous and the header writes below need the bytes. Host sources are memcpy'd.
        nanovdb::HostBuffer buffer(totalBytes);
        auto *dstBase   = static_cast<uint8_t *>(buffer.data());
        uint64_t offset = 0;
        for (const auto &src: sources) {
            c10::OptionalDeviceGuard srcGuard(
                src.device.is_cuda() ? std::optional<torch::Device>(src.device) : std::nullopt);
            if (src.device.is_privateuseone() && !src.spans.empty()) {
                // Unified memory: its writers may sit on any device's null stream, which the
                // copy below (on this device's) is not ordered after. Wait for them first.
                synchronizeStream(src.stream, src.device);
            }
            for (const auto &span: src.spans) {
                copySpan(dstBase + offset, device, span, src.device, src.stream);
                offset += span.size;
            }
            if (!src.device.is_cpu() && !src.spans.empty()) {
                synchronizeStream(src.stream, src.device);
            }
        }
        offset         = 0;
        uint32_t index = 0;
        for (const auto &src: sources) {
            for (const auto &span: src.spans) {
                auto *data       = reinterpret_cast<nanovdb::GridData *>(dstBase + offset);
                data->mGridIndex = index++;
                data->mGridCount = gridCount;
                data->mChecksum.disable();
                offset += span.size;
            }
        }
        // Constructing the handle parses the chain, validating the headers just written.
        return GridStorage(GridStorage::HostHandle(std::move(buffer)));
    }

    c10::OptionalDeviceGuard deviceGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                          : std::nullopt);
    // The spans' writers are queued on their storages' retained streams; `stream` waits for them.
    for (const auto &k: sourceStreams) {
        orderStreamAfter(stream, device, k.stream, k.device);
    }

    DestinationBuffer buffer(device, stream, totalBytes);
    std::byte *base = buffer.data();
    std::vector<nanovdb::GridHandleMetaData> meta;
    meta.reserve(gridCount);
    uint64_t offset = 0;
    uint32_t index  = 0;
    for (const auto &s: sources) {
        for (const auto &span: s.spans) {
            auto *dst = base + offset;
            copySpan(dst, device, span, s.device, stream);
            updateGridCountAndZeroChecksum<<<1, 1, 0, stream>>>(
                nanovdb::util::PtrAdd<nanovdb::GridData>(base, static_cast<int64_t>(offset)),
                index,
                gridCount);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            meta.push_back(nanovdb::GridHandleMetaData{offset, span.size, span.type});
            offset += span.size;
            ++index;
        }
    }
    // The sources may be released as soon as this returns: their streams wait for the copies.
    for (const auto &k: sourceStreams) {
        if (k.device.is_cuda() && device.is_cuda()) {
            orderStreamAfter(k.stream, k.device, stream, device);
        }
    }
    const bool hostSources = std::any_of(
        sources.begin(), sources.end(), [](const GridSpanSource &s) { return s.device.is_cpu(); });
    if (device.is_privateuseone() || hostSources ||
        std::any_of(sourceStreams.begin(), sourceStreams.end(), [](const StreamKey &k) {
            return !k.device.is_cuda();
        })) {
        // Wait for everything enqueued above when the caller cannot order against it: unified
        // memory on either side is read by host code straight away and is not stream-ordered,
        // and a host source may be released or rewritten as soon as this returns while a copy
        // from it (pinned, or pageable and merely staged) could still be pending.
        synchronizeStream(stream, device);
    }
    return std::move(buffer).adopt(std::move(meta));
}

GridStorage
replicateGridStorage(const GridStorage &source,
                     uint32_t count,
                     const torch::Device &requested,
                     cudaStream_t stream) {
    const torch::Device device = GridStorage::resolveDevice(requested);
    TORCH_CHECK(!device.is_cpu(), "replicateGridStorage: destination must be a device");
    TORCH_CHECK(source.device() == device,
                "replicateGridStorage: source on ",
                source.device(),
                " cannot be replicated onto ",
                device);
    TORCH_CHECK(!source.isEmpty() && !source.isHost() && source.gridCount() == 1,
                "replicateGridStorage: source must hold one device-resident grid");
    TORCH_CHECK(count > 1, "replicateGridStorage: count must be at least 2");
    const uint64_t gridSize = source.gridSize(0);

    c10::OptionalDeviceGuard deviceGuard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                          : std::nullopt);
    orderStreamAfter(stream, device, source.stream(), source.device());

    DestinationBuffer buffer(device, stream, gridSize * count);
    std::byte *dst = buffer.data();

    // Copy the grid once, then double the filled prefix. Each copy reads only bytes already
    // written, so the copies never overlap.
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, source.deviceBytes(), gridSize, cudaMemcpyDefault, stream));
    for (uint64_t filled = 1; filled < count; filled *= 2) {
        const uint64_t n = std::min<uint64_t>(filled, count - filled);
        C10_CUDA_CHECK(
            cudaMemcpyAsync(dst + filled * gridSize, dst, n * gridSize, cudaMemcpyDefault, stream));
    }
    stampRepeatedGridHeaders<<<GET_BLOCKS(count, DEFAULT_BLOCK_DIM),
                               DEFAULT_BLOCK_DIM,
                               0,
                               stream>>>(dst, gridSize, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // The source may be released as soon as this returns.
    if (source.device().is_cuda() && device.is_cuda()) {
        orderStreamAfter(source.stream(), source.device(), stream, device);
    }
    if (device.is_privateuseone()) {
        // Unified memory is read by host code straight away.
        synchronizeStream(stream, device);
    }

    std::vector<nanovdb::GridHandleMetaData> meta(count);
    for (uint32_t i = 0; i < count; ++i) {
        meta[i] = nanovdb::GridHandleMetaData{i * gridSize, gridSize, source.gridType(0)};
    }
    return std::move(buffer).adopt(std::move(meta));
}

namespace {

// One thread per grid: 0 if the header agrees with the expected (index, count, size), else 1.
__global__ void
compareAdoptedHeaders(const std::byte *base,
                      const nanovdb::GridHandleMetaData *meta,
                      uint32_t count,
                      uint32_t *mismatches) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= count) {
        return;
    }
    const auto *header = reinterpret_cast<const nanovdb::GridData *>(base + meta[g].offset);
    mismatches[g]      = (header->mGridIndex != g || header->mGridCount != count ||
                     header->mGridSize != meta[g].size)
                             ? 1u
                             : 0u;
}

} // namespace

void
checkAdoptedLayoutDebug(const void *bytes,
                        const torch::Device &device,
                        const std::vector<nanovdb::GridHandleMetaData> &meta,
                        cudaStream_t stream) {
    if (meta.empty()) {
        return;
    }
    c10::OptionalDeviceGuard guard(device.is_cuda() ? std::optional<torch::Device>(device)
                                                    : std::nullopt);
    const uint32_t count = static_cast<uint32_t>(meta.size());
    BuilderBuffer<nanovdb::GridHandleMetaData> dMeta(stream, count, nanovdb::cuda::noInit);
    BuilderBuffer<uint32_t> dMismatch(stream, count, nanovdb::cuda::noInit);
    C10_CUDA_CHECK(cudaMemcpyAsync(dMeta.data(),
                                   meta.data(),
                                   count * sizeof(nanovdb::GridHandleMetaData),
                                   cudaMemcpyHostToDevice,
                                   stream));
    constexpr uint32_t kThreads = 128;
    compareAdoptedHeaders<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const std::byte *>(bytes), dMeta.data(), count, dMismatch.data());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<uint32_t> mismatches(count);
    C10_CUDA_CHECK(cudaMemcpyAsync(mismatches.data(),
                                   dMismatch.data(),
                                   count * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (uint32_t g = 0; g < count; ++g) {
        TORCH_CHECK(mismatches[g] == 0,
                    "makeDeviceHandleFromLayout: grid ",
                    g,
                    "'s header disagrees with the adopted layout (offset ",
                    meta[g].offset,
                    ", size ",
                    meta[g].size,
                    ")");
    }
}

} // namespace fvdb::detail
