// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// work_stealing_pool.h
//
// True work-stealing thread pool using Chase-Lev deques for dynamic load balancing.

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

template <typename T> class work_stealing_deque {
  public:
    static constexpr std::size_t kDefaultCapacity = 65536;

    explicit work_stealing_deque(std::size_t capacity = kDefaultCapacity)
        : capacity_(capacity), mask_(capacity - 1),
          buffer_(std::make_unique<std::atomic<T>[]>(capacity)), top_(0), bottom_(0) {}

    work_stealing_deque(work_stealing_deque const &)            = delete;
    work_stealing_deque &operator=(work_stealing_deque const &) = delete;

    void
    push(T item) noexcept {
        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        buffer_[static_cast<std::size_t>(b) & mask_].store(item, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    std::optional<T>
    pop() noexcept {
        std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_seq_cst);
        std::int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            T item = buffer_[static_cast<std::size_t>(b) & mask_].load(std::memory_order_relaxed);
            if (t == b) {
                if (!top_.compare_exchange_strong(
                        t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        } else {
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    std::optional<T>
    steal() noexcept {
        std::int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            T item = buffer_[static_cast<std::size_t>(t) & mask_].load(std::memory_order_relaxed);
            if (!top_.compare_exchange_strong(
                    t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return std::nullopt;
            }
            return item;
        }
        return std::nullopt;
    }

    [[nodiscard]] bool
    empty() const noexcept {
        std::int64_t t = top_.load(std::memory_order_relaxed);
        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        return t >= b;
    }

    void
    clear() noexcept {
        bottom_.store(0, std::memory_order_relaxed);
        top_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t
    capacity() const noexcept {
        return capacity_;
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

        Index const n             = end - start;
        std::size_t total_workers = num_workers_ + 1;

        Index grain = normalize_grain_safe(grain_size, n, total_workers);

        std::size_t const num_tasks = static_cast<std::size_t>((n + grain - 1) / grain);
        if (num_tasks <= 1) {
            std::invoke(f, start, end);
            return;
        }

        // Create job context on stack with latch for synchronization
        job<Index, F> job_ctx(start, end, grain, num_tasks, total_workers, std::move(f));

        // Serialize job submission
        std::unique_lock<std::mutex> submit_lock(submit_mutex_);

        // Distribute tasks round-robin
        for (std::size_t w = 0; w < num_workers_; ++w) {
            worker_deques_[w]->clear();
            for (std::size_t t = w; t < num_tasks; t += total_workers) {
                worker_deques_[w]->push(t);
            }
        }

        main_deque_.clear();
        for (std::size_t t = num_workers_; t < num_tasks; t += total_workers) {
            main_deque_.push(t);
        }

        // Publish job
        current_job_.store(&job_ctx, std::memory_order_release);
        job_epoch_.fetch_add(1, std::memory_order_release);

        // Wake workers
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        // Main thread participates
        tls_in_parallel_region_ = true;
        execute_work(&job_ctx, main_deque_, num_workers_);
        tls_in_parallel_region_ = false;

        // Signal main thread is done working
        job_ctx.participant_done();

        // Wait for all participants (workers + main) to finish
        job_ctx.wait();

        // Clear job pointer (workers already exited based on remaining == 0)
        current_job_.store(nullptr, std::memory_order_release);

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
    struct job_base {
        using exec_fn_t = void (*)(job_base *, std::size_t task_id) noexcept;

        exec_fn_t exec        = nullptr;
        std::size_t num_tasks = 0;
        std::atomic<std::size_t> remaining{0};
        std::latch done_latch; // Tracks when all participants have exited

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex eptr_mutex;

        explicit job_base(std::size_t n, std::size_t participants)
            : num_tasks(n), remaining(n), done_latch(static_cast<std::ptrdiff_t>(participants)) {}

        void
        complete_task() noexcept {
            remaining.fetch_sub(1, std::memory_order_release);
        }

        void
        participant_done() noexcept {
            done_latch.count_down();
        }

        void
        wait() {
            done_latch.wait();
        }

        [[nodiscard]] bool
        all_tasks_done() const noexcept {
            return remaining.load(std::memory_order_acquire) == 0;
        }
    };

    template <typename Index, typename F> struct job final : job_base {
        Index start;
        Index end;
        Index grain;
        F func;

        job(Index s, Index e, Index g, std::size_t n, std::size_t participants, F &&f)
            : job_base(n, participants), start(s), end(e), grain(g), func(std::move(f)) {
            exec = &job::exec_impl;
        }

        static void
        exec_impl(job_base *base, std::size_t task_id) noexcept {
            auto *self = static_cast<job *>(base);
            try {
                if (!self->cancelled.load(std::memory_order_relaxed)) {
                    Index task_start = self->start + static_cast<Index>(task_id) * self->grain;
                    Index task_end   = task_start + self->grain;
                    if (task_end > self->end)
                        task_end = self->end;
                    if (task_start < task_end)
                        std::invoke(self->func, task_start, task_end);
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

    template <typename Index>
    Index
    normalize_grain_safe(Index grain_size, Index n, std::size_t total_workers) {
        using U = std::make_unsigned_t<Index>;
        U un    = static_cast<U>(n);

        std::size_t const capacity        = work_stealing_deque<std::size_t>::kDefaultCapacity;
        std::size_t const safe_per_worker = (capacity * 9) / 10;
        std::size_t const max_tasks_total = total_workers * safe_per_worker;

        U min_grain_for_capacity =
            (un + static_cast<U>(max_tasks_total) - 1) / static_cast<U>(max_tasks_total);
        if (min_grain_for_capacity == 0)
            min_grain_for_capacity = 1;

        U grain = static_cast<U>(grain_size);
        if (grain == 0) {
            std::size_t const target_tasks = std::max<std::size_t>(1, total_workers * 8);
            grain                          = un / static_cast<U>(target_tasks);
            if (grain == 0)
                grain = 1;
        }

        if (grain < min_grain_for_capacity)
            grain = min_grain_for_capacity;

        return static_cast<Index>(grain);
    }

    static std::size_t
    choose_default_workers() {
        std::size_t hw = std::thread::hardware_concurrency();
        if (hw == 0)
            hw = 1;
        return (hw > 1) ? hw - 1 : 0;
    }

    // Unified execute_work for both main thread and workers
    // Workers exit when remaining == 0; main thread also uses this
    void
    execute_work(job_base *ctx,
                 work_stealing_deque<std::size_t> &my_deque,
                 std::size_t my_id) noexcept {
        std::uint32_t rng = static_cast<std::uint32_t>(my_id + 1) * 2654435761u;
        auto xorshift     = [&rng]() -> std::uint32_t {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };

        std::size_t const total_workers = num_workers_ + 1;

        while (true) {
            // Exit when all tasks are done - this is the key optimization!
            // When remaining == 0, all exec() calls have completed, so no task is in-flight.
            if (ctx->all_tasks_done()) {
                return;
            }

            std::optional<std::size_t> task_id = my_deque.pop();

            if (!task_id) {
                // Try stealing from others
                for (std::size_t attempt = 0; attempt < total_workers && !task_id; ++attempt) {
                    std::size_t victim = xorshift() % total_workers;
                    if (victim == my_id)
                        continue;

                    if (victim < num_workers_) {
                        task_id = worker_deques_[victim]->steal();
                    } else {
                        task_id = main_deque_.steal();
                    }
                }
            }

            if (task_id) {
                ctx->exec(ctx, *task_id);
            } else {
                // No work found, brief pause before retrying
                DISPATCH_WS_SPIN_PAUSE();
            }
        }
    }

    void
    worker_thread_loop(std::size_t worker_index) {
        tls_in_parallel_region_   = true;
        std::uint64_t local_epoch = 0;
        constexpr auto kBlockTime = std::chrono::microseconds(500);

        while (!stop_.load(std::memory_order_acquire)) {
            if (job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                auto const start_wait = std::chrono::steady_clock::now();
                int spins             = 0;

                while (!stop_.load(std::memory_order_relaxed) &&
                       job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                    if ((std::chrono::steady_clock::now() - start_wait) < kBlockTime) {
                        if ((++spins & 0xFF) == 0)
                            std::this_thread::yield();
                        else
                            DISPATCH_WS_SPIN_PAUSE();
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

            // Execute work - exits when remaining == 0
            execute_work(ctx, *worker_deques_[worker_index], worker_index);

            // Signal this participant is done
            ctx->participant_done();
        }
    }

    work_stealing_pool() : num_workers_(choose_default_workers()), main_deque_() {
        if (num_workers_ == 0)
            return;

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
