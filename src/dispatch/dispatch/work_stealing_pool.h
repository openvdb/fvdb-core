// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// work_stealing_pool.h
//
// True work-stealing thread pool using Chase-Lev deques for dynamic load balancing.
//
// Key design points:
//  - Persistent worker threads with per-worker work-stealing deques.
//  - Chase-Lev deque: owner pushes/pops from bottom, thieves steal from top.
//  - Dynamic load balancing via work stealing - ideal for unbalanced workloads.
//  - "Hot wait" (spin) before sleeping to reduce wake latency.
//  - Nested parallel_for runs serially to prevent deadlock/oversubscription.
//
// Interface:
//   dispatch::work_stealing_pool::instance().parallel_for(start, end, grain, func);
//
// Notes:
//  - Better for unbalanced workloads than static partitioning (broadcast_pool).
//  - Higher overhead than broadcast_pool for uniform workloads.
//  - Use broadcast_pool for balanced workloads, this for unbalanced.

#ifndef DISPATCH_DISPATCH_WORK_STEALING_POOL_H
#define DISPATCH_DISPATCH_WORK_STEALING_POOL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define DISPATCH_WS_SPIN_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define DISPATCH_WS_SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define DISPATCH_WS_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define DISPATCH_WS_SPIN_PAUSE() std::this_thread::yield()
#endif

namespace dispatch {

// =============================================================================
// Chase-Lev Work-Stealing Deque
// =============================================================================
// Lock-free deque supporting:
//   - push_bottom / pop_bottom: called only by the owning worker (single-producer)
//   - steal: called by other workers (multi-consumer)
//
// Based on "Dynamic Circular Work-Stealing Deque" by Chase and Lev (SPAA 2005)
// Simplified fixed-size buffer (no dynamic resizing needed for our use case).

template <typename T> class work_stealing_deque {
  public:
    static constexpr std::size_t kDefaultCapacity = 8192;

    explicit work_stealing_deque(std::size_t capacity = kDefaultCapacity)
        : capacity_(capacity), mask_(capacity - 1),
          buffer_(std::make_unique<std::atomic<T>[]>(capacity)), top_(0), bottom_(0) {
        // Capacity must be power of 2 for efficient masking
        // assert removed for header-only, caller responsibility
    }

    work_stealing_deque(work_stealing_deque const &)            = delete;
    work_stealing_deque &operator=(work_stealing_deque const &) = delete;

    // Owner: push task to bottom of deque
    void
    push(T item) noexcept {
        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        buffer_[static_cast<std::size_t>(b) & mask_].store(item, std::memory_order_relaxed);
        // Release fence ensures the item is visible before we publish the new bottom
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    // Owner: pop task from bottom of deque (LIFO for owner)
    std::optional<T>
    pop() noexcept {
        std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        // Full fence to ensure the bottom update is visible to thieves
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty
            T item = buffer_[static_cast<std::size_t>(b) & mask_].load(std::memory_order_relaxed);
            if (t == b) {
                // Last item - race with thieves
                if (!top_.compare_exchange_strong(
                        t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // Lost race to thief
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        } else {
            // Empty
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    // Thief: steal task from top of deque (FIFO for thieves)
    std::optional<T>
    steal() noexcept {
        std::int64_t t = top_.load(std::memory_order_acquire);
        // Acquire fence pairs with release in push
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            // Non-empty
            T item = buffer_[static_cast<std::size_t>(t) & mask_].load(std::memory_order_relaxed);
            if (!top_.compare_exchange_strong(
                    t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                // Lost race to another thief or owner
                return std::nullopt;
            }
            return item;
        }
        return std::nullopt;
    }

    [[nodiscard]] bool
    empty() const noexcept {
        std::int64_t t = top_.load(std::memory_order_acquire);
        std::int64_t b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }

    void
    clear() noexcept {
        bottom_.store(0, std::memory_order_relaxed);
        top_.store(0, std::memory_order_relaxed);
    }

  private:
    std::size_t capacity_;
    std::size_t mask_;
    std::unique_ptr<std::atomic<T>[]> buffer_;
    alignas(64) std::atomic<std::int64_t> top_;
    alignas(64) std::atomic<std::int64_t> bottom_;
};

// =============================================================================
// Work-Stealing Pool
// =============================================================================

class work_stealing_pool {
  public:
    static work_stealing_pool &
    instance() {
        static work_stealing_pool pool;
        return pool;
    }

    work_stealing_pool(work_stealing_pool const &)            = delete;
    work_stealing_pool &operator=(work_stealing_pool const &) = delete;

    template <typename Index, typename Func>
        requires std::integral<Index> && std::invocable<std::decay_t<Func> &, Index, Index>
    void
    parallel_for(Index start, Index end, Index grain_size, Func &&func) {
        if (start >= end)
            return;

        // Nested parallel_for runs serially to avoid deadlock and oversubscription.
        if (tls_in_parallel_region_) {
            auto f = std::forward<Func>(func);
            std::invoke(f, start, end);
            return;
        }

        if (num_workers_ == 0) {
            auto f = std::forward<Func>(func);
            std::invoke(f, start, end);
            return;
        }

        using F = std::decay_t<Func>;
        F f(std::forward<Func>(func));

        Index const n = end - start;
        Index grain   = normalize_grain(grain_size, n, num_workers_ + 1);

        // Calculate number of tasks
        std::size_t const num_tasks = static_cast<std::size_t>((n + grain - 1) / grain);
        if (num_tasks <= 1) {
            std::invoke(f, start, end);
            return;
        }

        // Create typed job context on stack
        job<Index, F> job_ctx(start, end, grain, num_tasks, std::move(f));

        // Submit job - serialize access
        std::unique_lock<std::mutex> submit_lock(submit_mutex_);

        // Reset and populate worker deques with initial task distribution
        // Round-robin distribution: worker i gets tasks i, i+W, i+2W, ...
        std::size_t const total_workers = num_workers_ + 1;
        for (std::size_t w = 0; w < num_workers_; ++w) {
            worker_deques_[w]->clear();
            for (std::size_t t = w; t < num_tasks; t += total_workers) {
                worker_deques_[w]->push(t);
            }
        }

        // Main thread's tasks go into a temporary deque
        main_deque_.clear();
        for (std::size_t t = num_workers_; t < num_tasks; t += total_workers) {
            main_deque_.push(t);
        }

        // Publish job to workers
        current_job_.store(&job_ctx, std::memory_order_release);
        job_epoch_.fetch_add(1, std::memory_order_release);

        // Wake up workers
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        // Main thread participates in work
        tls_in_parallel_region_ = true;
        execute_work(&job_ctx, main_deque_, num_workers_);
        tls_in_parallel_region_ = false;

        // Wait for all tasks to complete
        job_ctx.wait();

        // Clear job
        current_job_.store(nullptr, std::memory_order_release);

        // Propagate exception if any
        if (job_ctx.eptr) {
            std::rethrow_exception(job_ctx.eptr);
        }
    }

    template <typename Index, typename F>
    void
    parallel_for(Index start, Index end, F &&f) {
        parallel_for(start, end, Index{0}, std::forward<F>(f));
    }

    [[nodiscard]] std::size_t
    num_threads() const noexcept {
        return num_workers_;
    }

  private:
    // -------------------------------------------------------------------------
    // Job base (type-erased)
    // -------------------------------------------------------------------------

    struct job_base {
        using exec_fn_t = void (*)(job_base *, std::size_t task_id) noexcept;

        exec_fn_t exec        = nullptr;
        std::size_t num_tasks = 0;
        std::latch done_latch;

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex eptr_mutex;

        explicit job_base(std::size_t n)
            : num_tasks(n), done_latch(static_cast<std::ptrdiff_t>(n)) {}

        void
        complete_task() noexcept {
            done_latch.count_down();
        }

        void
        wait() {
            done_latch.wait();
        }
    };

    // -------------------------------------------------------------------------
    // Typed job
    // -------------------------------------------------------------------------

    template <typename Index, typename F> struct job final : job_base {
        Index start;
        Index end;
        Index grain;
        F func;

        job(Index s, Index e, Index g, std::size_t n, F &&f)
            : job_base(n), start(s), end(e), grain(g), func(std::move(f)) {
            exec = &job::exec_impl;
        }

        static void
        exec_impl(job_base *base, std::size_t task_id) noexcept {
            auto *self = static_cast<job *>(base);
            try {
                if (!self->cancelled.load(std::memory_order_relaxed)) {
                    Index task_start = self->start + static_cast<Index>(task_id) * self->grain;
                    Index task_end   = task_start + self->grain;
                    // Don't exceed original end (last task may be partial)
                    if (task_end > self->end) {
                        task_end = self->end;
                    }
                    if (task_start < task_end) {
                        std::invoke(self->func, task_start, task_end);
                    }
                }
            } catch (...) {
                if (!self->cancelled.exchange(true, std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(self->eptr_mutex);
                    self->eptr = std::current_exception();
                }
            }
            self->complete_task();
        }
    };

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    template <typename Index>
    static Index
    normalize_grain(Index grain_size, Index n, std::size_t threads_hint) {
        if (grain_size > Index{0})
            return grain_size;

        // Aim for ~8 tasks per thread for good stealing opportunities
        using U                        = std::make_unsigned_t<Index>;
        U un                           = static_cast<U>(n);
        std::size_t const target_tasks = std::max<std::size_t>(1, threads_hint * 8);
        U g                            = static_cast<U>(un / target_tasks);
        if (g == 0)
            g = 1;
        return static_cast<Index>(g);
    }

    static std::size_t
    choose_default_workers() {
        std::size_t hw = std::thread::hardware_concurrency();
        if (hw == 0)
            hw = 1;
        if (hw <= 1)
            return 0;
        return hw - 1;
    }

    // -------------------------------------------------------------------------
    // Work execution with stealing
    // -------------------------------------------------------------------------

    void
    execute_work(job_base *ctx,
                 work_stealing_deque<std::size_t> &my_deque,
                 std::size_t my_id) noexcept {
        // XorShift RNG for random victim selection
        std::uint32_t rng = static_cast<std::uint32_t>(my_id + 1) * 2654435761u;
        auto xorshift     = [&rng]() -> std::uint32_t {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };

        std::size_t const total_workers  = num_workers_ + 1;
        std::size_t consecutive_failures = 0;

        while (true) {
            std::optional<std::size_t> task_id;

            // 1. Try local deque first (LIFO - good cache locality)
            task_id = my_deque.pop();

            // 2. If local empty, try stealing (FIFO from victims)
            if (!task_id) {
                // Try stealing from all workers
                for (std::size_t attempt = 0; attempt < total_workers && !task_id; ++attempt) {
                    std::size_t victim = xorshift() % total_workers;

                    if (victim == my_id)
                        continue;

                    if (victim < num_workers_) {
                        task_id = worker_deques_[victim]->steal();
                    } else if (victim == num_workers_) {
                        // Steal from main thread's deque
                        task_id = main_deque_.steal();
                    }
                }
            }

            // 3. Execute if we got a task
            if (task_id) {
                ctx->exec(ctx, *task_id);
                consecutive_failures = 0;
            } else {
                // No task found - check if we're done
                ++consecutive_failures;

                // After many failures, check latch state
                // The latch wait is the authoritative completion check
                if (consecutive_failures > total_workers * 4) {
                    // Try to observe completion non-destructively
                    // Since std::latch doesn't have try_wait, we use a heuristic:
                    // If all deques are empty and we've failed many times, we're likely done
                    bool all_empty = my_deque.empty();
                    for (std::size_t w = 0; w < num_workers_ && all_empty; ++w) {
                        if (!worker_deques_[w]->empty())
                            all_empty = false;
                    }
                    if (all_empty && main_deque_.empty()) {
                        return;               // All work is done
                    }
                    consecutive_failures = 0; // Reset and keep trying
                }

                DISPATCH_WS_SPIN_PAUSE();
            }
        }
    }

    // -------------------------------------------------------------------------
    // Worker thread main loop
    // -------------------------------------------------------------------------

    void
    worker_thread_loop(std::size_t worker_index) {
        tls_in_parallel_region_   = true;
        std::uint64_t local_epoch = 0;
        constexpr auto kBlockTime = std::chrono::microseconds(500);

        while (!stop_.load(std::memory_order_acquire)) {
            // Wait for new job
            if (job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                auto const start_wait = std::chrono::steady_clock::now();
                int spins             = 0;

                while (!stop_.load(std::memory_order_relaxed) &&
                       job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                    if ((std::chrono::steady_clock::now() - start_wait) < kBlockTime) {
                        if ((++spins & 0xFF) == 0) {
                            std::this_thread::yield();
                        } else {
                            DISPATCH_WS_SPIN_PAUSE();
                        }
                        continue;
                    }

                    std::unique_lock<std::mutex> lk(cv_mutex_);
                    cv_.wait(lk, [&] {
                        return stop_.load(std::memory_order_acquire) ||
                               job_epoch_.load(std::memory_order_acquire) != local_epoch;
                    });
                    break;
                }
            }

            if (stop_.load(std::memory_order_acquire))
                break;

            std::uint64_t const e = job_epoch_.load(std::memory_order_acquire);
            if (e == local_epoch)
                continue;
            local_epoch = e;

            job_base *ctx = current_job_.load(std::memory_order_acquire);
            if (!ctx)
                continue;

            // Execute work-stealing loop with my deque
            execute_work(ctx, *worker_deques_[worker_index], worker_index);
        }
    }

    // -------------------------------------------------------------------------
    // Pool construction and destruction
    // -------------------------------------------------------------------------

    work_stealing_pool() : num_workers_(choose_default_workers()), main_deque_() {
        if (num_workers_ == 0)
            return;

        // Create per-worker deques
        worker_deques_.reserve(num_workers_);
        for (std::size_t i = 0; i < num_workers_; ++i) {
            worker_deques_.push_back(std::make_unique<work_stealing_deque<std::size_t>>());
        }

        workers_.reserve(num_workers_);
        std::atomic<std::size_t> started{0};

        for (std::size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this, i, &started] {
                started.fetch_add(1, std::memory_order_release);
                worker_thread_loop(i);
            });
        }

        // Wait for all workers to start
        while (started.load(std::memory_order_acquire) != num_workers_) {
            DISPATCH_WS_SPIN_PAUSE();
        }
    }

    ~work_stealing_pool() {
        stop_.store(true, std::memory_order_release);
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        for (auto &t: workers_) {
            if (t.joinable())
                t.join();
        }
    }

  private:
    std::size_t num_workers_{0};
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<work_stealing_deque<std::size_t>>> worker_deques_;
    work_stealing_deque<std::size_t> main_deque_;

    std::atomic<bool> stop_{false};
    std::atomic<job_base *> current_job_{nullptr};
    std::atomic<std::uint64_t> job_epoch_{0};

    std::mutex cv_mutex_;
    std::condition_variable cv_;
    std::mutex submit_mutex_;

    inline static thread_local bool tls_in_parallel_region_ = false;
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_WORK_STEALING_POOL_H
