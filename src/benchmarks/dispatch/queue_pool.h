// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// queue_pool.h - Retired thread pool implementation preserved for benchmark comparison.
//
// Simple queue-based pool using std::jthread, std::latch, and std::function.
// Superseded by dispatch::thread_pool<scheduling::uniform> (broadcast_pool) and
// dispatch::thread_pool<scheduling::adaptive> (work_stealing_pool).
//
// This is NOT part of the dispatch library. It exists only as a benchmark baseline
// alongside spin_pool.h, to demonstrate why broadcast_pool was chosen.
//
// Safety mechanisms (matching broadcast_pool/work_stealing_pool for fair comparison):
//   - Nested parallel_for runs serially (deadlock prevention)
//   - Exception propagation from worker tasks to caller
//   - Constructor barrier (all workers started before returning)
//   - Destructor assertion for outstanding work
//
#ifndef DISPATCH_BENCH_QUEUE_POOL_H
#define DISPATCH_BENCH_QUEUE_POOL_H

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <latch>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace dispatch_bench {

class queue_pool {
  public:
    static queue_pool &
    instance() {
        static queue_pool pool;
        return pool;
    }

    queue_pool(queue_pool const &)            = delete;
    queue_pool &operator=(queue_pool const &) = delete;
    queue_pool(queue_pool &&)                 = delete;
    queue_pool &operator=(queue_pool &&)      = delete;

    template <typename Index, typename Func>
        requires std::integral<Index> && std::invocable<Func, Index, Index>
    void
    parallel_for(Index start, Index end, Index grain_size, Func &&func) {
        if (start >= end)
            return;

        // Nested parallel_for runs serially to avoid deadlock.
        if (tls_in_parallel_region_) {
            func(start, end);
            return;
        }

        auto const range       = static_cast<std::size_t>(end - start);
        auto const num_workers = workers_.size();

        if (range <= static_cast<std::size_t>(grain_size) || num_workers == 0) {
            func(start, end);
            return;
        }

        std::size_t const num_chunks =
            std::min(num_workers,
                     (range + static_cast<std::size_t>(grain_size) - 1) /
                         static_cast<std::size_t>(grain_size));
        auto const chunk_size = (range + num_chunks - 1) / num_chunks;

        active_jobs_.fetch_add(1, std::memory_order_relaxed);

        std::latch work_done(static_cast<std::ptrdiff_t>(num_chunks));

        // Shared exception state
        std::exception_ptr eptr;
        std::mutex eptr_mutex;
        std::atomic<bool> cancelled{false};

        for (std::size_t i = 0; i < num_chunks; ++i) {
            auto const chunk_start = start + static_cast<Index>(i * chunk_size);
            auto const chunk_end   = std::min(chunk_start + static_cast<Index>(chunk_size), end);

            enqueue([&func, chunk_start, chunk_end, &work_done, &eptr, &eptr_mutex, &cancelled]() {
                try {
                    if (!cancelled.load(std::memory_order_relaxed)) {
                        func(chunk_start, chunk_end);
                    }
                } catch (...) {
                    if (!cancelled.exchange(true, std::memory_order_relaxed)) {
                        std::lock_guard<std::mutex> lock(eptr_mutex);
                        eptr = std::current_exception();
                    }
                }
                work_done.count_down();
            });
        }

        work_done.wait();

        active_jobs_.fetch_sub(1, std::memory_order_relaxed);

        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }

    template <typename Index, typename Func>
    void
    parallel_for(Index start, Index end, Func &&func) {
        constexpr Index default_grain_size = 2048;
        parallel_for(start, end, default_grain_size, std::forward<Func>(func));
    }

    [[nodiscard]] std::size_t
    num_threads() const noexcept {
        return workers_.size();
    }

  private:
    explicit queue_pool(std::size_t num_threads = std::thread::hardware_concurrency())
        : stop_flag_(false) {
        if (num_threads == 0)
            num_threads = 1;

        workers_.reserve(num_threads);
        std::atomic<std::size_t> started{0};

        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this, &started](std::stop_token st) {
                tls_in_parallel_region_ = true;
                started.fetch_add(1, std::memory_order_release);
                worker_loop(st);
            });
        }

        // Constructor barrier: spin until all workers have started.
        while (started.load(std::memory_order_acquire) != num_threads) {
            std::this_thread::yield();
        }
    }

    ~queue_pool() {
        assert(active_jobs_.load(std::memory_order_relaxed) == 0 &&
               "queue_pool destroyed with outstanding work");

        {
            std::scoped_lock lock(mutex_);
            stop_flag_ = true;
        }
        cv_.notify_all();
    }

    void
    worker_loop(std::stop_token st) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this, &st] {
                    return stop_flag_ || !tasks_.empty() || st.stop_requested();
                });

                if ((stop_flag_ || st.stop_requested()) && tasks_.empty())
                    return;

                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    void
    enqueue(std::function<void()> &&task) {
        {
            std::scoped_lock lock(mutex_);
            tasks_.emplace(std::move(task));
        }
        cv_.notify_one();
    }

    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_flag_;

    std::atomic<std::size_t> active_jobs_{0};
    inline static thread_local bool tls_in_parallel_region_ = false;
};

} // namespace dispatch_bench

#endif // DISPATCH_BENCH_QUEUE_POOL_H
