// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::atomic_fetch_increment_i32: every participating thread must receive a
// distinct pre-increment value, and each counter's values must form [0, count) with no gaps.
//

#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"

#include <fvdb/detail/dispatch/AtomicAdd.cuh>

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using fvdb::detail::dispatch::atomic_fetch_increment_i32;

// Thread i increments counter (i * stride) % numCounters when i % skip != 0, and records the
// returned value in slots[i] (-1 for threads that skip).
template <torch::DeviceType Device>
void
runIncrements(int64_t n, int64_t numCounters, int64_t stride, int64_t skip, torch::Device device) {
    using Tag = ::dispatch::tag<Device>;

    auto counters = torch::zeros({numCounters}, torch::dtype(torch::kInt32).device(device));
    auto slots    = torch::full({n}, -1, torch::dtype(torch::kInt32).device(device));
    auto *cPtr    = counters.data_ptr<int32_t>();
    auto *sPtr    = slots.data_ptr<int32_t>();

    ::dispatch::for_each(Tag{}, n, [=] __hostdev__(Tag tg, int64_t i) {
        if (i % skip != 0) {
            sPtr[i] = atomic_fetch_increment_i32(tg, &cPtr[(i * stride) % numCounters]);
        }
    });

    auto const idx       = torch::arange(n, torch::dtype(torch::kInt64).device(device));
    auto const counter   = (idx * stride) % numCounters;
    auto const active    = (idx % skip) != 0;
    auto const countsCpu = counters.cpu();
    for (int64_t k = 0; k < numCounters; ++k) {
        auto const mine = slots.masked_select(active & (counter == k)).cpu().to(torch::kInt64);
        ASSERT_EQ(countsCpu[k].item<int64_t>(), mine.numel()) << "counter " << k;
        auto const sorted = std::get<0>(torch::sort(mine));
        EXPECT_TRUE(torch::equal(sorted, torch::arange(mine.numel(), torch::kInt64)))
            << "counter " << k << " values are not a permutation of [0, count)";
    }
}

} // namespace

TEST(AtomicFetchIncrementTest, CudaOneCounterPerWarp) {
    // stride 0: every lane of every warp hits counter 0, so each group is a full warp.
    runIncrements<torch::kCUDA>(1 << 20, 1, 0, 1 << 30, torch::kCUDA);
}

TEST(AtomicFetchIncrementTest, CudaMixedCountersWithinWarp) {
    // 27 counters interleaved across lanes, so each warp splits into several address groups.
    runIncrements<torch::kCUDA>(1 << 20, 27, 1, 1 << 30, torch::kCUDA);
}

TEST(AtomicFetchIncrementTest, CudaPartialWarps) {
    // Every third lane skips the atomic, so groups are formed from a divergent active mask.
    runIncrements<torch::kCUDA>(1 << 20, 27, 7, 3, torch::kCUDA);
}

TEST(AtomicFetchIncrementTest, Cpu) {
    runIncrements<torch::kCPU>(1 << 16, 27, 7, 3, torch::kCPU);
}
