// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// basic_thread_pool.h - Minimal C++20 thread pool for CPU parallelization
//
// This is the original "cooking with gas" version that outperformed torch
// on large tensors with a simple fixed grain_size = 2048 approach.
//
// Provides a simple parallel_for implementation without requiring OpenMP or TBB.
// Uses std::jthread (auto-join), std::latch (efficient barrier), and concepts.
//
#ifndef DISPATCH_DISPATCH_BASIC_THREAD_POOL_H
#define DISPATCH_DISPATCH_BASIC_THREAD_POOL_H

#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <latch>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace dispatch {

/// @brief Minimal thread pool for CPU parallelization.
///
/// Provides a parallel_for interface compatible with ATen's signature.
/// Uses Meyer's singleton pattern for safe, lazy initialization.
///
/// Example:
/// @code
///     basic_thread_pool::instance().parallel_for(0, n, grain_size,
///         [&](int64_t begin, int64_t end) {
///             for (int64_t i = begin; i < end; ++i) { ... }
///         });
/// @endcode
class basic_thread_pool {
  public:
    /// @brief Get the global thread pool instance (lazy-initialized, thread-safe).
    static basic_thread_pool &
    instance() {
        static basic_thread_pool pool;
        return pool;
    }

    // Non-copyable, non-movable
    basic_thread_pool(const basic_thread_pool &)            = delete;
    basic_thread_pool &operator=(const basic_thread_pool &) = delete;
    basic_thread_pool(basic_thread_pool &&)                 = delete;
    basic_thread_pool &operator=(basic_thread_pool &&)      = delete;

    /// @brief Execute a function in parallel over a range [start, end).
    ///
    /// Divides the range into chunks of approximately `grain_size` elements
    /// and distributes them across worker threads. Blocks until all chunks complete.
    ///
    /// @tparam Index Integral index type
    /// @tparam Func  Callable with signature void(Index begin, Index end)
    /// @param start      Beginning of range (inclusive)
    /// @param end        End of range (exclusive)
    /// @param grain_size Minimum elements per chunk (controls granularity)
    /// @param func       Function to execute on each chunk
    template <typename Index, typename Func>
        requires std::integral<Index> && std::invocable<Func, Index, Index>
    void
    parallel_for(Index start, Index end, Index grain_size, Func &&func) {
        if (start >= end) {
            return;
        }

        const auto range       = static_cast<size_t>(end - start);
        const auto num_workers = workers_.size();

        // For small ranges, just run sequentially
        if (range <= static_cast<size_t>(grain_size) || num_workers == 0) {
            func(start, end);
            return;
        }

        // Calculate number of chunks based on grain_size
        const size_t num_chunks = std::min(num_workers,
                                           (range + static_cast<size_t>(grain_size) - 1) /
                                               static_cast<size_t>(grain_size));
        const auto chunk_size   = (range + num_chunks - 1) / num_chunks;

        std::latch work_done(static_cast<std::ptrdiff_t>(num_chunks));

        for (size_t i = 0; i < num_chunks; ++i) {
            const auto chunk_start = start + static_cast<Index>(i * chunk_size);
            const auto chunk_end   = std::min(chunk_start + static_cast<Index>(chunk_size), end);

            enqueue([&func, chunk_start, chunk_end, &work_done]() {
                func(chunk_start, chunk_end);
                work_done.count_down();
            });
        }

        work_done.wait();
    }

    // a parallel for that uses a default grain size
    template <typename Index, typename Func>
    void
    parallel_for(Index start, Index end, Func &&func) {
        constexpr Index default_grain_size = 2048;
        parallel_for(start, end, default_grain_size, std::forward<Func>(func));
    }

    /// @brief Get the number of worker threads.
    [[nodiscard]] size_t
    num_threads() const noexcept {
        return workers_.size();
    }

  private:
    explicit basic_thread_pool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_flag_(false) {
        // Ensure at least 1 thread if hardware_concurrency returns 0
        if (num_threads == 0) {
            num_threads = 1;
        }

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this](std::stop_token st) { worker_loop(st); });
        }
    }

    ~basic_thread_pool() {
        {
            std::scoped_lock lock(mutex_);
            stop_flag_ = true;
        }
        cv_.notify_all();
        // std::jthread destructor will request_stop() and join() automatically
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

                if ((stop_flag_ || st.stop_requested()) && tasks_.empty()) {
                    return;
                }

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
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_BASIC_THREAD_POOL_H
