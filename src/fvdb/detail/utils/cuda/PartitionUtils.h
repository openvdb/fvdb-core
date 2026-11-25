// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_DETAIL_UTILS_CUDA_PARTITIONUTILS_H
#define FVDB_DETAIL_UTILS_CUDA_PARTITIONUTILS_H

#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <tuple>

namespace fvdb::detail::partition {

// Match the balanced partition arithmetic in torch-dgx and gsplat.
// Alignment and device count must be positive; ranks must be less than devices.
C10_HOST_DEVICE inline size_t
units(size_t size, size_t alignment) {
    return size / alignment + (size % alignment != 0);
}

C10_HOST_DEVICE inline size_t
slot_units(size_t size, size_t alignment, size_t devices) {
    const auto num_units = units(size, alignment);
    return num_units / devices + (num_units % devices != 0);
}

// Logical ranges balance alignment units, assigning one extra unit to early ranks.
C10_HOST_DEVICE inline size_t
offset(size_t size, size_t alignment, size_t devices, size_t rank) {
    const auto num_units = units(size, alignment);
    const auto extra     = num_units % devices;
    const auto first     = rank * (num_units / devices) + (rank < extra ? rank : extra);
    // Clamp empty ranks before multiplying, including a partial final unit.
    return first == num_units ? size : first * alignment;
}

C10_HOST_DEVICE inline size_t
count(size_t size, size_t alignment, size_t devices, size_t rank) {
    const auto num_units   = units(size, alignment);
    const auto count_units = num_units / devices + (rank < num_units % devices);
    const auto available   = size - offset(size, alignment, devices, rank);
    return count_units > available / alignment ? available : count_units * alignment;
}

// Fixed-size slots are used only for NCCL. Callers must check these products for overflow.
C10_HOST_DEVICE inline size_t
slot_size(size_t size, size_t alignment, size_t devices) {
    return slot_units(size, alignment, devices) * alignment;
}

C10_HOST_DEVICE inline size_t
padded_size(size_t size, size_t alignment, size_t devices) {
    return devices * slot_size(size, alignment, devices);
}

C10_HOST_DEVICE inline bool
has_internal_padding(size_t size, size_t alignment, size_t devices) {
    const auto num_units = units(size, alignment);
    const auto extra     = num_units % devices;
    // A single smaller final rank or an empty tail needs only trailing padding.
    return num_units / devices > 0 && extra > 0 && extra + 1 < devices;
}

inline std::tuple<size_t, size_t>
aligned_range(size_t alignment, size_t size, c10::DeviceIndex device, size_t devices) {
    TORCH_CHECK(alignment > 0 && devices > 0 && device >= 0 &&
                    static_cast<size_t>(device) < devices,
                "aligned_range requires positive alignment and a valid CUDA device");
    return {offset(size, alignment, devices, device), count(size, alignment, devices, device)};
}

inline std::tuple<size_t, size_t>
aligned_range(size_t alignment, size_t size, c10::DeviceIndex device) {
    return aligned_range(alignment, size, device, c10::cuda::device_count());
}

inline std::tuple<size_t, size_t>
range(size_t size, c10::DeviceIndex device) {
    return aligned_range(1, size, device);
}

} // namespace fvdb::detail::partition

#endif // FVDB_DETAIL_UTILS_CUDA_PARTITIONUTILS_H
