// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::for_each and views (flat_in/flat_out, tensor_in/tensor_out).
//
// Covers:
//   - Rank-templated views: construction, multi-index access (Rank=1, Rank=2)
//   - Flat views: rank-free flat access, unravel correctness, broadcast via stride-0
//   - for_each on CPU and CUDA
//   - Empty and single-element ranges
//   - Example patterns: softplus (scalar elementwise, arbitrary rank),
//     morton (structured elements via tensor_in),
//     channel_scale (binary op with broadcasting via flat views and tensor_in)
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"
#include "examples/softplus.h"
#include "test_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace dispatch {
namespace {

using test::skip_if_no_cuda;

// =============================================================================
// Morton helper — simple bit interleave for testing
// =============================================================================

// Simple morton placeholder (bit interleave lowest bits only, for testing)
__hostdev__ int64_t
morton_encode(int32_t i, int32_t j, int32_t k) {
    // Simple 3-bit interleave for testing purposes
    int64_t result = 0;
    for (int b = 0; b < 10; ++b) {
        result |= (static_cast<int64_t>((i >> b) & 1) << (3 * b + 0));
        result |= (static_cast<int64_t>((j >> b) & 1) << (3 * b + 1));
        result |= (static_cast<int64_t>((k >> b) & 1) << (3 * b + 2));
    }
    return result;
}

// =============================================================================
// View tests (CPU only — views are constructed and accessed on host)
// =============================================================================

class ViewTest : public ::testing::Test {};

TEST_F(ViewTest, Rank1_Contiguous) {
    auto t = torch::arange(10, torch::kFloat32);
    auto v = tensor_in<torch::kCPU, torch::kFloat32, 1, contiguity::contiguous>(t);

    EXPECT_EQ(v.size(0), 10);
    EXPECT_EQ(v.numel(), 10);
    for (int64_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(v(i), static_cast<float>(i));
    }
}

TEST_F(ViewTest, Rank1_Strided) {
    // Create strided tensor: every other element
    auto full = torch::arange(20, torch::kFloat32);
    auto t    = full.slice(0, 0, 20, 2); // [0, 2, 4, 6, ..., 18]
    ASSERT_FALSE(t.is_contiguous());
    ASSERT_EQ(t.size(0), 10);

    auto v = tensor_in<torch::kCPU, torch::kFloat32, 1, contiguity::strided>(t);

    EXPECT_EQ(v.size(0), 10);
    EXPECT_EQ(v.stride(0), 2);
    for (int64_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(v(i), static_cast<float>(i * 2));
    }
}

TEST_F(ViewTest, Rank2_Contiguous) {
    // [4, 3] tensor with known values
    auto t = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            t[r][c] = static_cast<float>(r * 10 + c);

    auto v = tensor_in<torch::kCPU, torch::kFloat32, 2, contiguity::contiguous>(t);

    EXPECT_EQ(v.size(0), 4);
    EXPECT_EQ(v.size(1), 3);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(v(r, c), static_cast<float>(r * 10 + c));
}

TEST_F(ViewTest, Rank2_Strided_Transposed) {
    // Transposed [3, 4] tensor (strides are non-standard)
    auto t = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            t[r][c] = static_cast<float>(r * 10 + c);

    auto tt = t.t(); // [3, 4], strides = [1, 3]
    ASSERT_FALSE(tt.is_contiguous());

    auto v = tensor_in<torch::kCPU, torch::kFloat32, 2, contiguity::strided>(tt);

    EXPECT_EQ(v.size(0), 3);
    EXPECT_EQ(v.size(1), 4);
    // tt[c, r] = t[r, c] = r * 10 + c
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 4; ++r)
            EXPECT_FLOAT_EQ(v(c, r), static_cast<float>(r * 10 + c));
}

TEST_F(ViewTest, Rank2_Broadcast_Stride0) {
    // scale is [3], expanded to [4, 3] with strides [0, 1]
    auto scale    = torch::tensor({10.0f, 20.0f, 30.0f});
    auto input    = torch::ones({4, 3}, torch::kFloat32);
    auto expanded = scale.expand_as(input); // [4, 3], strides [0, 1]

    ASSERT_EQ(expanded.stride(0), 0);       // broadcast dimension
    ASSERT_EQ(expanded.stride(1), 1);

    auto v = tensor_in<torch::kCPU, torch::kFloat32, 2, contiguity::strided>(expanded);

    // Every row should see the same scale values
    for (int r = 0; r < 4; ++r) {
        EXPECT_FLOAT_EQ(v(r, 0), 10.0f);
        EXPECT_FLOAT_EQ(v(r, 1), 20.0f);
        EXPECT_FLOAT_EQ(v(r, 2), 30.0f);
    }
}

TEST_F(ViewTest, TensorOut_Rank1_Contiguous) {
    auto t = torch::zeros({5}, torch::kFloat32);
    auto v = tensor_out<torch::kCPU, torch::kFloat32, 1, contiguity::contiguous>(t);

    for (int64_t i = 0; i < 5; ++i)
        v(i) = static_cast<float>(i * 3);

    for (int64_t i = 0; i < 5; ++i)
        EXPECT_FLOAT_EQ(t[i].item<float>(), static_cast<float>(i * 3));
}

// =============================================================================
// Flat view tests (CPU only — views are constructed and accessed on host)
// =============================================================================

class FlatViewTest : public ::testing::Test {};

TEST_F(FlatViewTest, Rank1_Contiguous) {
    auto t = torch::arange(10, torch::kFloat32);
    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::contiguous>(t);

    for (int64_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
    }
}

TEST_F(FlatViewTest, Rank1_Strided) {
    auto full = torch::arange(20, torch::kFloat32);
    auto t    = full.slice(0, 0, 20, 2); // [0, 2, 4, ..., 18]
    ASSERT_FALSE(t.is_contiguous());

    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::strided>(t);

    for (int64_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i * 2));
    }
}

TEST_F(FlatViewTest, Rank2_Contiguous) {
    // [4, 3] tensor: flat access should visit elements in row-major order
    auto t = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            t[r][c] = static_cast<float>(r * 10 + c);

    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::contiguous>(t);

    // flat_idx 0 -> (0,0) = 0, flat_idx 1 -> (0,1) = 1, flat_idx 3 -> (1,0) = 10, etc.
    EXPECT_FLOAT_EQ(v[0], 0.0f);
    EXPECT_FLOAT_EQ(v[1], 1.0f);
    EXPECT_FLOAT_EQ(v[2], 2.0f);
    EXPECT_FLOAT_EQ(v[3], 10.0f);
    EXPECT_FLOAT_EQ(v[11], 32.0f); // (3,2) = 32
}

TEST_F(FlatViewTest, Rank2_Strided_Transposed) {
    // Original [4, 3] transposed to [3, 4]
    auto t = torch::zeros({4, 3}, torch::kFloat32);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            t[r][c] = static_cast<float>(r * 10 + c);

    auto tt = t.t(); // [3, 4], strides = [1, 3]
    ASSERT_FALSE(tt.is_contiguous());

    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::strided>(tt);

    // Flat access in row-major order of the transposed shape [3, 4]:
    // flat_idx 0 -> tt(0,0) = t(0,0) = 0
    // flat_idx 1 -> tt(0,1) = t(1,0) = 10
    // flat_idx 4 -> tt(1,0) = t(0,1) = 1
    // flat_idx 5 -> tt(1,1) = t(1,1) = 11
    EXPECT_FLOAT_EQ(v[0], 0.0f);
    EXPECT_FLOAT_EQ(v[1], 10.0f);
    EXPECT_FLOAT_EQ(v[4], 1.0f);
    EXPECT_FLOAT_EQ(v[5], 11.0f);
    EXPECT_FLOAT_EQ(v[11], 32.0f); // tt(2,3) = t(3,2) = 32
}

TEST_F(FlatViewTest, Rank2_Broadcast_Stride0) {
    // scale [3] expanded to [4, 3] with strides [0, 1]
    auto scale    = torch::tensor({10.0f, 20.0f, 30.0f});
    auto input    = torch::ones({4, 3}, torch::kFloat32);
    auto expanded = scale.expand_as(input);

    ASSERT_EQ(expanded.stride(0), 0);
    ASSERT_EQ(expanded.stride(1), 1);

    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::strided>(expanded);

    // flat_idx 0 -> (0,0) -> scale[0] = 10
    // flat_idx 1 -> (0,1) -> scale[1] = 20
    // flat_idx 3 -> (1,0) -> scale[0] = 10 (broadcast!)
    // flat_idx 4 -> (1,1) -> scale[1] = 20 (broadcast!)
    EXPECT_FLOAT_EQ(v[0], 10.0f);
    EXPECT_FLOAT_EQ(v[1], 20.0f);
    EXPECT_FLOAT_EQ(v[2], 30.0f);
    EXPECT_FLOAT_EQ(v[3], 10.0f);
    EXPECT_FLOAT_EQ(v[4], 20.0f);
    EXPECT_FLOAT_EQ(v[5], 30.0f);
    EXPECT_FLOAT_EQ(v[9], 10.0f);  // (3,0) -> scale[0]
    EXPECT_FLOAT_EQ(v[11], 30.0f); // (3,2) -> scale[2]
}

TEST_F(FlatViewTest, Rank3_Contiguous) {
    // [2, 3, 4] tensor — verify flat access matches numel() elements in order
    auto t = torch::arange(24, torch::kFloat32).reshape({2, 3, 4});

    auto v = flat_in<torch::kCPU, torch::kFloat32, contiguity::contiguous>(t);

    for (int64_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
    }
}

TEST_F(FlatViewTest, FlatOut_Rank1_Contiguous) {
    auto t = torch::zeros({5}, torch::kFloat32);
    auto v = flat_out<torch::kCPU, torch::kFloat32, contiguity::contiguous>(t);

    for (int64_t i = 0; i < 5; ++i)
        v[i] = static_cast<float>(i * 3);

    for (int64_t i = 0; i < 5; ++i)
        EXPECT_FLOAT_EQ(t[i].item<float>(), static_cast<float>(i * 3));
}

TEST_F(FlatViewTest, FlatOut_Rank2_Strided_Transposed) {
    auto t  = torch::zeros({4, 3}, torch::kFloat32);
    auto tt = t.t(); // [3, 4], strides [1, 3]
    ASSERT_FALSE(tt.is_contiguous());

    auto v = flat_out<torch::kCPU, torch::kFloat32, contiguity::strided>(tt);

    // Write via flat index in row-major order of [3, 4]
    for (int64_t i = 0; i < 12; ++i)
        v[i] = static_cast<float>(i);

    // Verify: v[5] wrote to tt(1,1) = t(1,1)
    EXPECT_FLOAT_EQ(t[0][0].item<float>(), 0.0f);  // tt(0,0)
    EXPECT_FLOAT_EQ(t[1][0].item<float>(), 1.0f);  // tt(0,1)
    EXPECT_FLOAT_EQ(t[0][1].item<float>(), 4.0f);  // tt(1,0)
    EXPECT_FLOAT_EQ(t[1][1].item<float>(), 5.0f);  // tt(1,1)
    EXPECT_FLOAT_EQ(t[3][2].item<float>(), 11.0f); // tt(2,3)
}

// =============================================================================
// for_each CPU tests
// =============================================================================

class ForEachCPUTest : public ::testing::Test {};

TEST_F(ForEachCPUTest, EmptyRange) {
    using Tag   = tag<torch::kCPU>;
    bool called = false;
    for_each(Tag{}, 0, [&](Tag, int64_t) { called = true; });
    EXPECT_FALSE(called);
}

TEST_F(ForEachCPUTest, SingleElement) {
    using Tag   = tag<torch::kCPU>;
    int64_t sum = 0;
    for_each(Tag{}, 1, [&](Tag, int64_t idx) { sum += idx + 1; });
    EXPECT_EQ(sum, 1);
}

TEST_F(ForEachCPUTest, Increment_Contiguous) {
    using Tag       = tag<torch::kCPU, torch::kFloat32, contiguity::contiguous>;
    int64_t const n = 10000;
    auto tensor     = torch::zeros({n}, torch::kFloat32);

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto out = flat_out<dev, stype, contig>(tensor);

    for_each(Tag{}, n, [=](Tag, int64_t i) { out[i] = static_cast<float>(i + 1); });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(tensor[i].item<float>(), static_cast<float>(i + 1)) << "at " << i;
    }
}

TEST_F(ForEachCPUTest, Increment_Strided) {
    using Tag       = tag<torch::kCPU, torch::kFloat32, contiguity::strided>;
    int64_t const n = 100;

    // Strided output: every other element of a 200-element tensor
    auto full   = torch::zeros({n * 2}, torch::kFloat32);
    auto sliced = full.slice(0, 0, n * 2, 2);
    ASSERT_EQ(sliced.size(0), n);
    ASSERT_FALSE(sliced.is_contiguous());

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto out = flat_out<dev, stype, contig>(sliced);

    for_each(Tag{}, n, [=](Tag, int64_t i) { out[i] = static_cast<float>(i + 1); });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(sliced[i].item<float>(), static_cast<float>(i + 1)) << "at " << i;
    }
    // Interleaved zeros should be untouched
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(full[2 * i + 1].item<float>(), 0.0f) << "gap at " << i;
    }
}

// =============================================================================
// Example: Softplus via dispatch_examples (scalar elementwise, all devices)
// =============================================================================

TEST_F(ForEachCPUTest, Softplus_ViaExample) {
    auto input  = torch::randn({1000}, torch::kFloat32);
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output, ref, 1e-5, 1e-5));
}

TEST_F(ForEachCPUTest, Softplus_Strided_ViaExample) {
    // Strided input: every other element of a 2000-element tensor
    auto full  = torch::randn({2000}, torch::kFloat32);
    auto input = full.slice(0, 0, 2000, 2); // 1000 elements, stride 2
    ASSERT_FALSE(input.is_contiguous());

    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output, ref, 1e-5, 1e-5));
}

// =============================================================================
// Example: Morton encoding (fixed types, structured elements, CPU)
// =============================================================================

TEST_F(ForEachCPUTest, Morton_Contiguous) {
    using Tag = tag<torch::kCPU, contiguity::contiguous>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    int64_t const n = 100;
    auto ijk        = torch::zeros({n, 3}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        ijk[i][0] = static_cast<int32_t>(i);
        ijk[i][1] = static_cast<int32_t>(i + 1);
        ijk[i][2] = static_cast<int32_t>(i + 2);
    }
    auto out = torch::empty({n}, torch::kInt64);

    auto ijk_v = tensor_in<dev, torch::kInt32, 2, contig>(ijk);
    auto out_v = tensor_out<dev, torch::kInt64, 1, contig>(out);

    for_each(Tag{}, n, [=](Tag, int64_t idx) {
        auto const i = ijk_v(idx, 0);
        auto const j = ijk_v(idx, 1);
        auto const k = ijk_v(idx, 2);
        out_v(idx)   = morton_encode(i, j, k);
    });

    // Verify a few known values
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t const expected = morton_encode(static_cast<int32_t>(idx),
                                               static_cast<int32_t>(idx + 1),
                                               static_cast<int32_t>(idx + 2));
        EXPECT_EQ(out[idx].item<int64_t>(), expected) << "at " << idx;
    }
}

// =============================================================================
// Example: Softplus on multi-rank tensors (proves flat views handle any rank)
// =============================================================================

TEST_F(ForEachCPUTest, Softplus_2D_Contiguous) {
    auto input  = torch::randn({100, 100}, torch::kFloat32);
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output, ref, 1e-5, 1e-5));
}

TEST_F(ForEachCPUTest, Softplus_3D_Contiguous) {
    auto input  = torch::randn({10, 20, 50}, torch::kFloat32);
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output, ref, 1e-5, 1e-5));
}

TEST_F(ForEachCPUTest, Softplus_2D_Transposed) {
    auto base  = torch::randn({50, 30}, torch::kFloat32);
    auto input = base.t(); // [30, 50], non-contiguous
    ASSERT_FALSE(input.is_contiguous());

    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output, ref, 1e-5, 1e-5));
}

// =============================================================================
// Example: Channel scale with broadcasting via flat views (CPU)
// =============================================================================
// This test demonstrates the recommended binary op pattern:
//   - The wrapper (not the caller) does expand_as, following PyTorch convention
//   - flat_in handles the broadcast tensor (stride-0 dims) via unravel
//   - No manual index decomposition (idx / c, idx % c) — operator[] does it

TEST_F(ForEachCPUTest, FlatBroadcast_ChannelScale) {
    using Tag = tag<torch::kCPU, torch::kFloat32, contiguity::strided>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    int64_t const n = 50;
    int64_t const c = 4;
    auto input      = torch::ones({n, c}, torch::kFloat32);
    auto scale      = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}); // [4], NOT pre-expanded
    auto output     = torch::empty({n, c}, torch::kFloat32);

    // Entry point responsibility: broadcast to matching shape
    auto scale_expanded = scale.expand_as(input); // [n, c], strides [0, 1]

    // Flat views — no Rank template parameter, no manual index decomposition
    auto in_v    = flat_in<dev, stype, contig>(input);
    auto scale_v = flat_in<dev, stype, contig>(scale_expanded);
    auto out_v   = flat_out<dev, stype, contig>(output);

    int64_t const total = input.numel();

    for_each(Tag{}, total, [=](Tag, int64_t idx) { out_v[idx] = in_v[idx] * scale_v[idx]; });

    // Verify: output[i, j] = 1.0 * scale[j]
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(output[i][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(output[i][1].item<float>(), 2.0f);
        EXPECT_FLOAT_EQ(output[i][2].item<float>(), 3.0f);
        EXPECT_FLOAT_EQ(output[i][3].item<float>(), 4.0f);
    }
}

// =============================================================================
// Example: Channel scale with broadcasting via tensor_in (structured access, CPU)
// =============================================================================
// Kept for comparison: uses rank-2 tensor_in with manual index decomposition.

TEST_F(ForEachCPUTest, ChannelScale_Broadcast_Structured) {
    using Tag = tag<torch::kCPU, torch::kFloat32, contiguity::strided>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    int64_t const n = 50;
    int64_t const c = 4;
    auto input      = torch::ones({n, c}, torch::kFloat32);
    auto scale      = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}); // [4]
    auto output     = torch::empty({n, c}, torch::kFloat32);

    auto scale_expanded = scale.expand_as(input);              // [n, c], strides [0, 1]

    auto in_v    = tensor_in<dev, stype, 2, contig>(input);
    auto scale_v = tensor_in<dev, stype, 2, contig>(scale_expanded);
    auto out_v   = tensor_out<dev, stype, 2, contig>(output);

    int64_t const total = n * c;

    for_each(Tag{}, total, [=](Tag, int64_t idx) {
        int64_t const i = idx / c;
        int64_t const j = idx % c;
        out_v(i, j)     = in_v(i, j) * scale_v(i, j);
    });

    // Verify: output[i, j] = 1.0 * scale[j]
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(output[i][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(output[i][1].item<float>(), 2.0f);
        EXPECT_FLOAT_EQ(output[i][2].item<float>(), 3.0f);
        EXPECT_FLOAT_EQ(output[i][3].item<float>(), 4.0f);
    }
}

// =============================================================================
// CUDA for_each tests
// =============================================================================
// CUDA device lambdas must be in free functions, not inside TEST_F methods.

void
cuda_for_each_increment_contiguous(float *ptr, int64_t n) {
    using Tag = tag<torch::kCUDA, torch::kFloat32, contiguity::contiguous>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto t = torch::from_blob(
        ptr, {n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto out = flat_out<dev, stype, contig>(t);

    for_each(Tag{}, n, [=] __device__(Tag, int64_t i) { out[i] = static_cast<float>(i + 1); });
}

void
cuda_for_each_increment_strided(torch::Tensor sliced) {
    using Tag = tag<torch::kCUDA, torch::kFloat32, contiguity::strided>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto out        = flat_out<dev, stype, contig>(sliced);
    int64_t const n = sliced.size(0);

    for_each(Tag{}, n, [=] __device__(Tag, int64_t i) { out[i] = static_cast<float>(i + 1); });
}

void
cuda_for_each_morton(torch::Tensor ijk, torch::Tensor out) {
    using Tag = tag<torch::kCUDA, contiguity::contiguous>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto ijk_v = tensor_in<dev, torch::kInt32, 2, contig>(ijk);
    auto out_v = tensor_out<dev, torch::kInt64, 1, contig>(out);

    int64_t const n = ijk.size(0);

    for_each(Tag{}, n, [=] __device__(Tag, int64_t idx) {
        auto const i = ijk_v(idx, 0);
        auto const j = ijk_v(idx, 1);
        auto const k = ijk_v(idx, 2);
        out_v(idx)   = morton_encode(i, j, k);
    });
}

// Flat broadcast channel_scale: operator[] handles the unravel + broadcast
void
cuda_for_each_flat_channel_scale(torch::Tensor input,
                                 torch::Tensor scale, // un-expanded [C]
                                 torch::Tensor output) {
    using Tag = tag<torch::kCUDA, torch::kFloat32, contiguity::strided>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    // Entry point responsibility: broadcast
    auto scale_expanded = scale.expand_as(input);

    auto in_v    = flat_in<dev, stype, contig>(input);
    auto scale_v = flat_in<dev, stype, contig>(scale_expanded);
    auto out_v   = flat_out<dev, stype, contig>(output);

    int64_t const total = input.numel();

    for_each(
        Tag{}, total, [=] __device__(Tag, int64_t idx) { out_v[idx] = in_v[idx] * scale_v[idx]; });
}

// Structured channel_scale: tensor_in with manual index decomposition (kept for comparison)
void
cuda_for_each_channel_scale(torch::Tensor input,
                            torch::Tensor scale_expanded,
                            torch::Tensor output,
                            int64_t c) {
    using Tag = tag<torch::kCUDA, torch::kFloat32, contiguity::strided>;

    constexpr auto dev    = tag_get<torch::DeviceType>(Tag{});
    constexpr auto stype  = tag_get<torch::ScalarType>(Tag{});
    constexpr auto contig = tag_get<contiguity>(Tag{});

    auto in_v    = tensor_in<dev, stype, 2, contig>(input);
    auto scale_v = tensor_in<dev, stype, 2, contig>(scale_expanded);
    auto out_v   = tensor_out<dev, stype, 2, contig>(output);

    int64_t const total = input.numel();

    for_each(Tag{}, total, [=] __device__(Tag, int64_t idx) {
        int64_t const i = idx / c;
        int64_t const j = idx % c;
        out_v(i, j)     = in_v(i, j) * scale_v(i, j);
    });
}

void
cuda_for_each_empty() {
    using Tag = tag<torch::kCUDA>;
    for_each(Tag{}, 0, [=] __device__(Tag, int64_t) {});
}

class ForEachCUDATest : public ::testing::Test {};

TEST_F(ForEachCUDATest, EmptyRange) {
    skip_if_no_cuda();
    cuda_for_each_empty();
}

TEST_F(ForEachCUDATest, Increment_Contiguous) {
    skip_if_no_cuda();

    int64_t const n = 10000;
    auto tensor =
        torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cuda_for_each_increment_contiguous(tensor.data_ptr<float>(), n);

    auto cpu = tensor.cpu();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(cpu[i].item<float>(), static_cast<float>(i + 1)) << "at " << i;
    }
}

TEST_F(ForEachCUDATest, Increment_Strided) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto full =
        torch::zeros({n * 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto sliced = full.slice(0, 0, n * 2, 2);
    ASSERT_EQ(sliced.size(0), n);
    ASSERT_FALSE(sliced.is_contiguous());

    cuda_for_each_increment_strided(sliced);

    auto cpu_sliced = sliced.cpu();
    auto cpu_full   = full.cpu();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(cpu_sliced[i].item<float>(), static_cast<float>(i + 1)) << "at " << i;
    }
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(cpu_full[2 * i + 1].item<float>(), 0.0f) << "gap at " << i;
    }
}

TEST_F(ForEachCUDATest, Softplus_ViaExample) {
    skip_if_no_cuda();

    auto input =
        torch::randn({10000}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output.cpu(), ref.cpu(), 1e-5, 1e-5));
}

TEST_F(ForEachCUDATest, Softplus_Strided_ViaExample) {
    skip_if_no_cuda();

    auto full_in =
        torch::randn({20000}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto input = full_in.slice(0, 0, 20000, 2); // 10000 elements, stride 2
    ASSERT_FALSE(input.is_contiguous());

    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output.cpu(), ref.cpu(), 1e-5, 1e-5));
}

TEST_F(ForEachCUDATest, Morton_Contiguous) {
    skip_if_no_cuda();

    int64_t const n = 100;
    auto ijk =
        torch::zeros({n, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    // Fill on CPU, then copy
    auto ijk_cpu = torch::zeros({n, 3}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        ijk_cpu[i][0] = static_cast<int32_t>(i);
        ijk_cpu[i][1] = static_cast<int32_t>(i + 1);
        ijk_cpu[i][2] = static_cast<int32_t>(i + 2);
    }
    ijk = ijk_cpu.cuda();

    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    cuda_for_each_morton(ijk, out);

    auto out_cpu = out.cpu();
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t const expected = morton_encode(static_cast<int32_t>(idx),
                                               static_cast<int32_t>(idx + 1),
                                               static_cast<int32_t>(idx + 2));
        EXPECT_EQ(out_cpu[idx].item<int64_t>(), expected) << "at " << idx;
    }
}

TEST_F(ForEachCUDATest, Softplus_2D_Contiguous) {
    skip_if_no_cuda();

    auto input  = torch::randn({100, 100},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output.cpu(), ref.cpu(), 1e-5, 1e-5));
}

TEST_F(ForEachCUDATest, Softplus_3D_Contiguous) {
    skip_if_no_cuda();

    auto input  = torch::randn({10, 20, 50},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output.cpu(), ref.cpu(), 1e-5, 1e-5));
}

TEST_F(ForEachCUDATest, Softplus_2D_Transposed) {
    skip_if_no_cuda();

    auto base =
        torch::randn({50, 30}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto input = base.t(); // [30, 50], non-contiguous
    ASSERT_FALSE(input.is_contiguous());

    auto output = dispatch_examples::example_softplus(input, 1.0, 20.0);

    auto ref = torch::nn::functional::softplus(
        input, torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    EXPECT_TRUE(torch::allclose(output.cpu(), ref.cpu(), 1e-5, 1e-5));
}

TEST_F(ForEachCUDATest, FlatBroadcast_ChannelScale) {
    skip_if_no_cuda();

    int64_t const n = 50;
    int64_t const c = 4;
    auto input =
        torch::ones({n, c}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto scale = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).cuda(); // [4], NOT pre-expanded
    auto output =
        torch::empty({n, c}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cuda_for_each_flat_channel_scale(input, scale, output);

    auto out_cpu = output.cpu();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out_cpu[i][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][1].item<float>(), 2.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][2].item<float>(), 3.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][3].item<float>(), 4.0f);
    }
}

TEST_F(ForEachCUDATest, ChannelScale_Broadcast_Structured) {
    skip_if_no_cuda();

    int64_t const n = 50;
    int64_t const c = 4;
    auto input =
        torch::ones({n, c}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto scale = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).cuda();
    auto output =
        torch::empty({n, c}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto scale_expanded = scale.expand_as(input);

    cuda_for_each_channel_scale(input, scale_expanded, output, c);

    auto out_cpu = output.cpu();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out_cpu[i][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][1].item<float>(), 2.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][2].item<float>(), 3.0f);
        EXPECT_FLOAT_EQ(out_cpu[i][3].item<float>(), 4.0f);
    }
}

} // namespace
} // namespace dispatch
