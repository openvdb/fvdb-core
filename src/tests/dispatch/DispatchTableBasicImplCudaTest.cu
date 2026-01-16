// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "DispatchTableBasicUtils.cuh"

#include <fvdb/detail/dispatch/SparseDispatchTable.h>
#include <fvdb/detail/dispatch/ValueSpace.h>
#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace fvdb {
namespace dispatch {

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
using BasicGPUFloatSubspace =
    ValueAxes<Values<Device::GPU>, BasicDtypeAxis, Values<ChannelPlacement::Major>>;
using BasicGPUInt32Subspace = Values<Device::GPU, DType::Int32, ChannelPlacement::Minor>;
using BasicCPUSubspace      = ValueAxes<Values<Device::CPU>, BasicDtypeAxis, BasicPlacementAxis>;
using BasicDispatcher       = DispatchTable<BasicSpace, BasicArray(size_t, size_t)>;

BasicArray
basic(Device dev, DType dtype, ChannelPlacement pl, size_t C, size_t E) {
    static BasicDispatcher const table{
        BasicDispatcher::from_visitor(
            [](auto coord, size_t c, size_t e) { return basic_impl(coord, c, e); }),
        BasicGPUFloatSubspace{},
        BasicGPUInt32Subspace{},
        BasicCPUSubspace{}};
    return table(std::make_tuple(dev, dtype, pl), C, E);
}

// -----------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------

// GPU Major
TEST(DispatchTableBasicImplCuda, GPU_Float32_Major) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 3, 10);
}
TEST(DispatchTableBasicImplCuda, GPU_Float32_Major_Large) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 16, 1024);
}
TEST(DispatchTableBasicImplCuda, GPU_Float64_Major) {
    test(Device::GPU, DType::Float64, ChannelPlacement::Major, 4, 8);
}
TEST(DispatchTableBasicImplCuda, GPU_Float64_Major_Large) {
    test(Device::GPU, DType::Float64, ChannelPlacement::Major, 8, 2048);
}
TEST(DispatchTableBasicImplCuda, GPU_Int32_Major) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 5, 20);
}
TEST(DispatchTableBasicImplCuda, GPU_Int32_Major_Large) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 32, 512);
}

// GPU Minor (Int32 only)
TEST(DispatchTableBasicImplCuda, GPU_Int32_Minor) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 4, 16);
}
TEST(DispatchTableBasicImplCuda, GPU_Int32_Minor_Large) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 64, 256);
}

// CPU full coverage
TEST(DispatchTableBasicImplCuda, CPU_Float32_Major) {
    test(Device::CPU, DType::Float32, ChannelPlacement::Major, 3, 100);
}
TEST(DispatchTableBasicImplCuda, CPU_Float32_Minor) {
    test(Device::CPU, DType::Float32, ChannelPlacement::Minor, 3, 100);
}
TEST(DispatchTableBasicImplCuda, CPU_Float64_Major) {
    test(Device::CPU, DType::Float64, ChannelPlacement::Major, 5, 50);
}
TEST(DispatchTableBasicImplCuda, CPU_Float64_Minor) {
    test(Device::CPU, DType::Float64, ChannelPlacement::Minor, 5, 50);
}
TEST(DispatchTableBasicImplCuda, CPU_Int32_Major) {
    test(Device::CPU, DType::Int32, ChannelPlacement::Major, 7, 30);
}
TEST(DispatchTableBasicImplCuda, CPU_Int32_Minor) {
    test(Device::CPU, DType::Int32, ChannelPlacement::Minor, 7, 30);
}

// Edge cases
TEST(DispatchTableBasicImplCuda, SingleChannel) {
    test(Device::GPU, DType::Float32, ChannelPlacement::Major, 1, 128);
    test(Device::CPU, DType::Int32, ChannelPlacement::Minor, 1, 64);
}
TEST(DispatchTableBasicImplCuda, SingleElement) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Major, 8, 1);
    test(Device::CPU, DType::Float64, ChannelPlacement::Major, 4, 1);
}
TEST(DispatchTableBasicImplCuda, LargeChannelCount) {
    test(Device::GPU, DType::Int32, ChannelPlacement::Minor, 256, 16);
}

// Unsupported combos throw
TEST(DispatchTableBasicImplCuda, GPU_Float32_Minor_Throws) {
    EXPECT_THROW(basic(Device::GPU, DType::Float32, ChannelPlacement::Minor, 3, 10),
                 std::runtime_error);
}
TEST(DispatchTableBasicImplCuda, GPU_Float64_Minor_Throws) {
    EXPECT_THROW(basic(Device::GPU, DType::Float64, ChannelPlacement::Minor, 3, 10),
                 std::runtime_error);
}

} // namespace dispatch
} // namespace fvdb
