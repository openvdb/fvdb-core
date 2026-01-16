// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/SparseDispatchTable.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace fvdb {
namespace dispatch {

// -----------------------------------------------------------------------------
// INFRASTRUCTURE
// -----------------------------------------------------------------------------

enum class Device { CPU, GPU, Metal };
enum class DType { Float32, Float64, Int32 };
enum class ChannelPlacement { Major, Minor };

// DType → C++ type mapping
template <DType> struct DTypeToCpp;
template <> struct DTypeToCpp<DType::Float32> {
    using type = float;
};
template <> struct DTypeToCpp<DType::Float64> {
    using type = double;
};
template <> struct DTypeToCpp<DType::Int32> {
    using type = int32_t;
};
template <DType D> using DTypeToCpp_t = typename DTypeToCpp<D>::type;

// Concept for float dtypes
template <DType D>
inline constexpr bool is_float_dtype = (D == DType::Float32 || D == DType::Float64);

// Index helpers
__host__ __device__ inline size_t
majorIdx(size_t c, size_t e, size_t E) {
    return c * E + e;
}
__host__ __device__ inline size_t
minorIdx(size_t c, size_t e, size_t C) {
    return e * C + c;
}

// BasicArray
struct BasicArray {
    Device device              = Device::CPU;
    DType dtype                = DType::Float32;
    ChannelPlacement placement = ChannelPlacement::Major;
    size_t channels = 0, elements = 0;
    void *data = nullptr;
};

// Memory
inline void *
allocMem(Device d, size_t bytes) {
    if (d == Device::CPU)
        return std::malloc(bytes);
    if (d == Device::GPU) {
        void *p = nullptr;
        cudaMalloc(&p, bytes);
        return p;
    }
    throw std::runtime_error("Unsupported device");
}
inline void
freeMem(Device d, void *p) {
    if (!p)
        return;
    if (d == Device::CPU)
        std::free(p);
    else if (d == Device::GPU)
        cudaFree(p);
}
inline void
freeArray(BasicArray const &a) {
    freeMem(a.device, a.data);
}

// -----------------------------------------------------------------------------
// CUDA KERNELS
// -----------------------------------------------------------------------------

template <typename T, ChannelPlacement P>
__global__ void
fillKernel(T *data, size_t C, size_t E) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * E)
        return;
    size_t chan = (P == ChannelPlacement::Major) ? idx / E : idx % C;
    data[idx]   = static_cast<T>(chan);
}

// -----------------------------------------------------------------------------
// OPERATOR IMPLEMENTATIONS (signatures preserved)
// -----------------------------------------------------------------------------

template <DType dtype>
    requires is_float_dtype<dtype>
BasicArray
basic_impl(Values<Device::GPU, dtype, ChannelPlacement::Major> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T    = DTypeToCpp_t<dtype>;
    size_t n   = channel_count * element_count;
    void *data = allocMem(Device::GPU, n * sizeof(T));
    fillKernel<T, ChannelPlacement::Major>
        <<<(n + 255) / 256, 256>>>(static_cast<T *>(data), channel_count, element_count);
    return {Device::GPU, dtype, ChannelPlacement::Major, channel_count, element_count, data};
}

BasicArray
basic_impl(Values<Device::GPU, DType::Int32, ChannelPlacement::Major> /* coord */,
           size_t channel_count,
           size_t element_count) {
    size_t n   = channel_count * element_count;
    void *data = allocMem(Device::GPU, n * sizeof(int32_t));
    fillKernel<int32_t, ChannelPlacement::Major>
        <<<(n + 255) / 256, 256>>>(static_cast<int32_t *>(data), channel_count, element_count);
    return {Device::GPU, DType::Int32, ChannelPlacement::Major, channel_count, element_count, data};
}

BasicArray
basic_impl(Values<Device::GPU, DType::Int32, ChannelPlacement::Minor> /* coord */,
           size_t channel_count,
           size_t element_count) {
    size_t n   = channel_count * element_count;
    void *data = allocMem(Device::GPU, n * sizeof(int32_t));
    fillKernel<int32_t, ChannelPlacement::Minor>
        <<<(n + 255) / 256, 256>>>(static_cast<int32_t *>(data), channel_count, element_count);
    return {Device::GPU, DType::Int32, ChannelPlacement::Minor, channel_count, element_count, data};
}

template <DType dtype, ChannelPlacement placement>
BasicArray
basic_impl(Values<Device::CPU, dtype, placement> /* coord */,
           size_t channel_count,
           size_t element_count) {
    using T  = DTypeToCpp_t<dtype>;
    size_t n = channel_count * element_count;
    T *data  = static_cast<T *>(allocMem(Device::CPU, n * sizeof(T)));
    for (size_t c = 0; c < channel_count; ++c)
        for (size_t e = 0; e < element_count; ++e) {
            size_t idx = (placement == ChannelPlacement::Major) ? majorIdx(c, e, element_count)
                                                                : minorIdx(c, e, channel_count);
            data[idx]  = static_cast<T>(c);
        }
    return {Device::CPU, dtype, placement, channel_count, element_count, data};
}

// -----------------------------------------------------------------------------
// DISPATCH TABLE
// -----------------------------------------------------------------------------

using BasicDeviceAxis    = Values<Device::CPU, Device::GPU>;
using BasicDtypeAxis     = Values<DType::Float32, DType::Float64, DType::Int32>;
using BasicPlacementAxis = Values<ChannelPlacement::Major, ChannelPlacement::Minor>;
using BasicSpace         = ValueAxes<BasicDeviceAxis, BasicDtypeAxis, BasicPlacementAxis>;

using BasicFnPtr = BasicArray (*)(size_t, size_t);

BasicArray
basic(Device dev, DType dtype, ChannelPlacement pl, size_t C, size_t E) {
    static DispatchTable<BasicSpace, BasicArray(size_t, size_t)> const table{
        [](auto coord) {
            return [](size_t c, size_t e) { return basic_impl(decltype(coord){}, c, e); };
        },
        // GPU: all dtypes major, plus int32 minor
        ValueAxes<Values<Device::GPU>, BasicDtypeAxis, Values<ChannelPlacement::Major>>{},
        Values<Device::GPU, DType::Int32, ChannelPlacement::Minor>{},
        // CPU: full space
        ValueAxes<Values<Device::CPU>, BasicDtypeAxis, BasicPlacementAxis>{}};
    return table(std::make_tuple(dev, dtype, pl), C, E);
}

// -----------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------

namespace {

template <typename T>
std::vector<T>
toHost(void *ptr, size_t n, Device d) {
    std::vector<T> v(n);
    if (d == Device::CPU)
        std::memcpy(v.data(), ptr, n * sizeof(T));
    else {
        cudaMemcpy(v.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    return v;
}

template <typename T>
bool
validate(std::vector<T> const &data, size_t C, size_t E, ChannelPlacement pl) {
    for (size_t c = 0; c < C; ++c)
        for (size_t e = 0; e < E; ++e) {
            size_t idx = (pl == ChannelPlacement::Major) ? majorIdx(c, e, E) : minorIdx(c, e, C);
            if (data[idx] != static_cast<T>(c))
                return false;
        }
    return true;
}

void
check(BasicArray const &a) {
    ASSERT_NE(a.data, nullptr);
    size_t n = a.channels * a.elements;
    bool ok  = false;
    switch (a.dtype) {
    case DType::Float32:
        ok = validate(toHost<float>(a.data, n, a.device), a.channels, a.elements, a.placement);
        break;
    case DType::Float64:
        ok = validate(toHost<double>(a.data, n, a.device), a.channels, a.elements, a.placement);
        break;
    case DType::Int32:
        ok = validate(toHost<int32_t>(a.data, n, a.device), a.channels, a.elements, a.placement);
        break;
    }
    EXPECT_TRUE(ok);
}

void
test(Device d, DType t, ChannelPlacement p, size_t C, size_t E) {
    BasicArray a = basic(d, t, p, C, E);
    EXPECT_EQ(a.device, d);
    EXPECT_EQ(a.dtype, t);
    EXPECT_EQ(a.placement, p);
    EXPECT_EQ(a.channels, C);
    EXPECT_EQ(a.elements, E);
    check(a);
    freeArray(a);
}

} // namespace

// GPU Major
TEST(DispatchTableBasicCuda, GPU_Float32_Major) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 3, 10);
}
TEST(DispatchTableBasicCuda, GPU_Float32_Major_Large) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 16, 1024);
}
TEST(DispatchTableBasicCuda, GPU_Float64_Major) {
    test(Device::GPU, DType::Float64, ChannelPlacement::Major, 4, 8);
}
TEST(DispatchTableBasicCuda, GPU_Float64_Major_Large) {
    test(Device::GPU, DType::Float64, ChannelPlacement::Major, 8, 2048);
}
TEST(DispatchTableBasicCuda, GPU_Int32_Major) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 5, 20);
}
TEST(DispatchTableBasicCuda, GPU_Int32_Major_Large) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 32, 512);
}

// GPU Minor (Int32 only)
TEST(DispatchTableBasicCuda, GPU_Int32_Minor) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 4, 16);
}
TEST(DispatchTableBasicCuda, GPU_Int32_Minor_Large) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 64, 256);
}

// CPU full coverage
TEST(DispatchTableBasicCuda, CPU_Float32_Major) {
    test(Device::CPU, DType::Float32, ChannelPlacement::Major, 3, 100);
}
TEST(DispatchTableBasicCuda, CPU_Float32_Minor) {
    test(Device::CPU, DType::Float32, ChannelPlacement::Minor, 3, 100);
}
TEST(DispatchTableBasicCuda, CPU_Float64_Major) {
    test(Device::CPU, DType::Float64, ChannelPlacement::Major, 5, 50);
}
TEST(DispatchTableBasicCuda, CPU_Float64_Minor) {
    test(Device::CPU, DType::Float64, ChannelPlacement::Minor, 5, 50);
}
TEST(DispatchTableBasicCuda, CPU_Int32_Major) {
    test(Device::CPU, DType::Int32, ChannelPlacement::Major, 7, 30);
}
TEST(DispatchTableBasicCuda, CPU_Int32_Minor) {
    test(Device::CPU, DType::Int32, ChannelPlacement::Minor, 7, 30);
}

// Edge cases
TEST(DispatchTableBasicCuda, SingleChannel) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 1, 128);
    test(Device::CPU, DType::Int32, ChannelPlacement::Minor, 1, 64);
}
TEST(DispatchTableBasicCuda, SingleElement) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 8, 1);
    test(Device::CPU, DType::Float64, ChannelPlacement::Major, 4, 1);
}
TEST(DispatchTableBasicCuda, LargeChannelCount) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 256, 16);
}

// Unsupported combos throw
TEST(DispatchTableBasicCuda, GPU_Float32_Minor_Throws) {
    EXPECT_THROW(basic(Device::GPU, DType::Float32, ChannelPlacement::Minor, 3, 10),
                 std::runtime_error);
}
TEST(DispatchTableBasicCuda, GPU_Float64_Minor_Throws) {
    EXPECT_THROW(basic(Device::GPU, DType::Float64, ChannelPlacement::Minor, 3, 10),
                 std::runtime_error);
}

} // namespace dispatch
} // namespace fvdb
