// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_DISPATCH_DISPATCHTABLEBASICUTILS_CUH
#define TESTS_DISPATCH_DISPATCHTABLEBASICUTILS_CUH

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
// BASIC FUNCTION PROTOTYPE
// -----------------------------------------------------------------------------
BasicArray basic(Device dev, DType dtype, ChannelPlacement pl, size_t C, size_t E);

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

} // namespace dispatch
} // namespace fvdb

#endif
