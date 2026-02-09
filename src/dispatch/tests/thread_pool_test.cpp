// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Tests for dispatch::thread_pool safety invariants.
// Covers both scheduling::uniform (broadcast_pool) and scheduling::adaptive (work_stealing_pool).
//

#include "dispatch/thread_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dispatch {
namespace {

// =============================================================================
// Test fixture parameterized over scheduling strategy
// =============================================================================

class ThreadPoolTest : public ::testing::TestWithParam<scheduling> {
  protected:
    // Run parallel_for on the pool selected by the test parameter.
    template <typename Func>
    void
    pool_parallel_for(int64_t start, int64_t end, Func &&f) {
        switch (GetParam()) {
        case scheduling::uniform:
            broadcast_pool::instance().parallel_for(start, end, std::forward<Func>(f));
            break;
        case scheduling::adaptive:
            work_stealing_pool::instance().parallel_for(start, end, std::forward<Func>(f));
            break;
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

TEST_P(ThreadPoolTest, EmptyRange) {
    // Must not crash or invoke the functor.
    bool called = false;
    pool_parallel_for(0, 0, [&](int64_t, int64_t) { called = true; });
    EXPECT_FALSE(called);
}

TEST_P(ThreadPoolTest, SingleElement) {
    // Single-element range should work correctly.
    std::atomic<int64_t> sum{0};
    pool_parallel_for(0, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            sum.fetch_add(i + 1, std::memory_order_relaxed);
        }
    });
    EXPECT_EQ(sum.load(), 1);
}

TEST_P(ThreadPoolTest, CorrectPartitioning) {
    // Every element in a large range is visited exactly once.
    constexpr int64_t n = 100000;
    std::vector<std::atomic<int>> counts(n);
    for (auto &c: counts) {
        c.store(0, std::memory_order_relaxed);
    }

    pool_parallel_for(int64_t{0}, n, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            counts[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(counts[static_cast<std::size_t>(i)].load(), 1) << "at index " << i;
    }
}

TEST_P(ThreadPoolTest, NestedParallelForRunsSerially) {
    // Nested parallel_for must not deadlock. The inner call should run serially.
    constexpr int64_t outer_n = 100;
    constexpr int64_t inner_n = 100;
    std::atomic<int64_t> total{0};

    pool_parallel_for(int64_t{0}, outer_n, [&](int64_t outer_begin, int64_t outer_end) {
        for (int64_t o = outer_begin; o < outer_end; ++o) {
            // This inner call should run serially on the same thread.
            pool_parallel_for(int64_t{0}, inner_n, [&](int64_t inner_begin, int64_t inner_end) {
                total.fetch_add(inner_end - inner_begin, std::memory_order_relaxed);
            });
        }
    });

    EXPECT_EQ(total.load(), outer_n * inner_n);
}

TEST_P(ThreadPoolTest, ExceptionPropagation) {
    // An exception thrown in a task must be rethrown to the caller.
    EXPECT_THROW(
        pool_parallel_for(int64_t{0}, int64_t{100000}, [](int64_t begin, int64_t) {
            if (begin == 0) {
                throw std::runtime_error("test exception");
            }
        }),
        std::runtime_error);
}

// =============================================================================
// Instantiate for both pool types
// =============================================================================

INSTANTIATE_TEST_SUITE_P(BroadcastPool,
                         ThreadPoolTest,
                         ::testing::Values(scheduling::uniform),
                         [](auto const &info) { return "uniform"; });

INSTANTIATE_TEST_SUITE_P(WorkStealingPool,
                         ThreadPoolTest,
                         ::testing::Values(scheduling::adaptive),
                         [](auto const &info) { return "adaptive"; });

} // namespace
} // namespace dispatch
