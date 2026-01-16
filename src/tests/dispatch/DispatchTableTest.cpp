// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/SparseDispatchTable.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace fvdb {
namespace dispatch {

// =============================================================================
// =============================================================================
//
//  INFRASTRUCTURE / SUPPORT MACHINERY
//
//  This section contains helpers that would typically live in library headers
//  and be reused across many operators. An implementer of a specific op would
//  not need to write this code.
//
// =============================================================================
// =============================================================================

// -----------------------------------------------------------------------------
// Device and DType enums
// -----------------------------------------------------------------------------

enum class Device { CPU, GPU, Metal };
enum class DType { Float32, Float64, Int32 };
enum class ChannelPlacement { Major, Minor };

// -----------------------------------------------------------------------------
// DTypeSize - size in bytes of each dtype
// -----------------------------------------------------------------------------

template <DType dtype> struct DTypeSize;

template <> struct DTypeSize<DType::Float32> {
    static constexpr size_t value = 4;
};

template <> struct DTypeSize<DType::Float64> {
    static constexpr size_t value = 8;
};

template <> struct DTypeSize<DType::Int32> {
    static constexpr size_t value = 4;
};

template <DType dtype> inline constexpr size_t DTypeSize_v = DTypeSize<dtype>::value;

// Runtime version
inline size_t
dtypeSize(DType dtype) {
    switch (dtype) {
    case DType::Float32: return 4;
    case DType::Float64: return 8;
    case DType::Int32: return 4;
    }
    throw std::runtime_error("Unknown dtype");
}

// -----------------------------------------------------------------------------
// DTypeToCpp - map DType enum to C++ type
// -----------------------------------------------------------------------------

template <DType dtype> struct DTypeToCpp;

template <> struct DTypeToCpp<DType::Float32> {
    using type = float;
};

template <> struct DTypeToCpp<DType::Float64> {
    using type = double;
};

template <> struct DTypeToCpp<DType::Int32> {
    using type = int32_t;
};

template <DType dtype> using DTypeToCpp_t = typename DTypeToCpp<dtype>::type;

// -----------------------------------------------------------------------------
// Accessor - typed pointer wrapper for void* data
// -----------------------------------------------------------------------------

template <DType dtype> class Accessor {
  public:
    using value_type = DTypeToCpp_t<dtype>;

    __host__ __device__ explicit Accessor(void *data) : _data(static_cast<value_type *>(data)) {}

    __host__ __device__ value_type &
    operator[](size_t index) {
        return _data[index];
    }

    __host__ __device__ value_type const &
    operator[](size_t index) const {
        return _data[index];
    }

    __host__ __device__ value_type *
    data() {
        return _data;
    }

    __host__ __device__ value_type const *
    data() const {
        return _data;
    }

  private:
    value_type *_data;
};

// -----------------------------------------------------------------------------
// Index calculation helpers for channel-major and channel-minor layouts
// -----------------------------------------------------------------------------

// Channel-Major: channels are the outer dimension
// Layout: [channel_0_elem_0, channel_0_elem_1, ..., channel_1_elem_0, ...]
// Index = channel * element_count + element
__host__ __device__ inline size_t
channelMajorIndex(size_t channel, size_t element, size_t element_count) {
    return channel * element_count + element;
}

// Channel-Minor: elements are the outer dimension
// Layout: [elem_0_chan_0, elem_0_chan_1, ..., elem_1_chan_0, ...]
// Index = element * channel_count + channel
__host__ __device__ inline size_t
channelMinorIndex(size_t channel, size_t element, size_t channel_count) {
    return element * channel_count + channel;
}

// -----------------------------------------------------------------------------
// BasicArray - the array type we're creating
// -----------------------------------------------------------------------------

struct BasicArray {
    Device device                      = Device::CPU;
    DType dtype                        = DType::Float32;
    ChannelPlacement channel_placement = ChannelPlacement::Major;
    size_t channel_count               = 0;
    size_t element_count               = 0;
    void *data                         = nullptr;
};

// -----------------------------------------------------------------------------
// Memory allocation helpers
// -----------------------------------------------------------------------------

inline void *
allocateDeviceMemory(Device device, size_t bytes) {
    switch (device) {
    case Device::CPU: return std::malloc(bytes);
    case Device::GPU: {
        void *ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
    case Device::Metal: throw std::runtime_error("Metal not implemented");
    }
    throw std::runtime_error("Unknown device");
}

inline void
freeDeviceMemory(Device device, void *ptr) {
    if (ptr == nullptr)
        return;
    switch (device) {
    case Device::CPU: std::free(ptr); break;
    case Device::GPU: cudaFree(ptr); break;
    case Device::Metal: throw std::runtime_error("Metal not implemented");
    }
}

void
free_basic_array(BasicArray array) {
    freeDeviceMemory(array.device, array.data);
}

// -----------------------------------------------------------------------------
// is_float_dtype - concept helper
// -----------------------------------------------------------------------------

template <DType dtype>
inline constexpr bool is_float_dtype = dtype == DType::Float32 || dtype == DType::Float64;

// =============================================================================
// =============================================================================
//
//  CUDA KERNELS
//
//  These kernels would typically live in a .cu file and be compiled by nvcc.
//  They are the low-level implementations of the operation.
//
// =============================================================================
// =============================================================================

// -----------------------------------------------------------------------------
// GPU Float Kernels - Channel Major Layout
// -----------------------------------------------------------------------------

template <typename T>
__global__ void
basicFillKernel_FloatMajor(T *data, size_t channel_count, size_t element_count) {
    size_t const idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const total_elements = channel_count * element_count;

    if (idx >= total_elements)
        return;

    // In major layout: idx = channel * element_count + element
    size_t const channel = idx / element_count;
    // Fill with channel index (cast to T)
    data[idx] = static_cast<T>(channel);
}

// -----------------------------------------------------------------------------
// GPU Float Kernels - Channel Minor Layout
// -----------------------------------------------------------------------------

template <typename T>
__global__ void
basicFillKernel_FloatMinor(T *data, size_t channel_count, size_t element_count) {
    size_t const idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const total_elements = channel_count * element_count;

    if (idx >= total_elements)
        return;

    // In minor layout: idx = element * channel_count + channel
    size_t const channel = idx % channel_count;
    // Fill with channel index (cast to T)
    data[idx] = static_cast<T>(channel);
}

// -----------------------------------------------------------------------------
// GPU Int32 Kernel - Both Major and Minor layouts
// -----------------------------------------------------------------------------

template <ChannelPlacement channel_placement>
__global__ void
basicFillKernel_Int32(int32_t *data, size_t channel_count, size_t element_count) {
    size_t const idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const total_elements = channel_count * element_count;

    if (idx >= total_elements)
        return;

    size_t channel;
    if constexpr (channel_placement == ChannelPlacement::Major) {
        // Major layout: idx = channel * element_count + element
        channel = idx / element_count;
    } else {
        // Minor layout: idx = element * channel_count + channel
        channel = idx % channel_count;
    }
    data[idx] = static_cast<int32_t>(channel);
}

// =============================================================================
// =============================================================================
//
//  OPERATOR IMPLEMENTATIONS
//
//  Below this banner is where an actual op implementer would write their code.
//  They would use the infrastructure above, but the dispatch signatures and
//  the logic within are their responsibility.
//
// =============================================================================
// =============================================================================

// -----------------------------------------------------------------------------
// GPU Float Major Implementation
// -----------------------------------------------------------------------------

template <DType dtype>
    requires is_float_dtype<dtype>
BasicArray
basic_impl(Values<Device::GPU, dtype, ChannelPlacement::Major> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T = DTypeToCpp_t<dtype>;

    size_t const total_elements = channel_count * element_count;
    size_t const bytes          = total_elements * sizeof(T);

    void *data = allocateDeviceMemory(Device::GPU, bytes);

    // Launch kernel
    constexpr size_t block_size = 256;
    size_t const grid_size      = (total_elements + block_size - 1) / block_size;

    basicFillKernel_FloatMajor<T>
        <<<grid_size, block_size>>>(static_cast<T *>(data), channel_count, element_count);

    return BasicArray{
        .device            = Device::GPU,
        .dtype             = dtype,
        .channel_placement = ChannelPlacement::Major,
        .channel_count     = channel_count,
        .element_count     = element_count,
        .data              = data,
    };
}

// -----------------------------------------------------------------------------
// GPU Float Minor Implementation
// -----------------------------------------------------------------------------

template <DType dtype>
    requires is_float_dtype<dtype>
BasicArray
basic_impl(Values<Device::GPU, dtype, ChannelPlacement::Minor> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T = DTypeToCpp_t<dtype>;

    size_t const total_elements = channel_count * element_count;
    size_t const bytes          = total_elements * sizeof(T);

    void *data = allocateDeviceMemory(Device::GPU, bytes);

    // Launch kernel
    constexpr size_t block_size = 256;
    size_t const grid_size      = (total_elements + block_size - 1) / block_size;

    basicFillKernel_FloatMinor<T>
        <<<grid_size, block_size>>>(static_cast<T *>(data), channel_count, element_count);

    return BasicArray{
        .device            = Device::GPU,
        .dtype             = dtype,
        .channel_placement = ChannelPlacement::Minor,
        .channel_count     = channel_count,
        .element_count     = element_count,
        .data              = data,
    };
}

// -----------------------------------------------------------------------------
// GPU Int32 Implementation (handles both Major and Minor)
// -----------------------------------------------------------------------------

template <ChannelPlacement channel_placement>
BasicArray
basic_impl(Values<Device::GPU, DType::Int32, channel_placement> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T = int32_t;

    size_t const total_elements = channel_count * element_count;
    size_t const bytes          = total_elements * sizeof(T);

    void *data = allocateDeviceMemory(Device::GPU, bytes);

    // Launch kernel - single kernel template handles both layouts
    constexpr size_t block_size = 256;
    size_t const grid_size      = (total_elements + block_size - 1) / block_size;

    basicFillKernel_Int32<channel_placement>
        <<<grid_size, block_size>>>(static_cast<T *>(data), channel_count, element_count);

    return BasicArray{
        .device            = Device::GPU,
        .dtype             = DType::Int32,
        .channel_placement = channel_placement,
        .channel_count     = channel_count,
        .element_count     = element_count,
        .data              = data,
    };
}

// -----------------------------------------------------------------------------
// CPU Implementation (handles all dtypes and layouts in one function)
// -----------------------------------------------------------------------------

template <DType dtype, ChannelPlacement channel_placement>
BasicArray
basic_impl(Values<Device::CPU, dtype, channel_placement> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T = DTypeToCpp_t<dtype>;

    size_t const total_elements = channel_count * element_count;
    size_t const bytes          = total_elements * sizeof(T);

    void *data = allocateDeviceMemory(Device::CPU, bytes);

    Accessor<dtype> accessor(data);

    // Fill the array: each element gets the value of its channel index
    for (size_t c = 0; c < channel_count; ++c) {
        for (size_t e = 0; e < element_count; ++e) {
            size_t idx;
            if constexpr (channel_placement == ChannelPlacement::Major) {
                idx = channelMajorIndex(c, e, element_count);
            } else {
                idx = channelMinorIndex(c, e, channel_count);
            }
            accessor[idx] = static_cast<T>(c);
        }
    }

    return BasicArray{
        .device            = Device::CPU,
        .dtype             = dtype,
        .channel_placement = channel_placement,
        .channel_count     = channel_count,
        .element_count     = element_count,
        .data              = data,
    };
}

} // namespace dispatch
} // namespace fvdb
