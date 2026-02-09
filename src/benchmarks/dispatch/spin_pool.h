// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// spin_pool.h - Retired thread pool implementation preserved for benchmark comparison.
//
// Generation-counter spin-wait pool with static partitioning.
// Superseded by dispatch::thread_pool<scheduling::uniform> (broadcast_pool).
//
// This is NOT part of the dispatch library. It exists only as a benchmark baseline
// alongside queue_pool.h, to demonstrate why broadcast_pool was chosen.
//
// Safety mechanisms (matching broadcast_pool/work_stealing_pool for fair comparison):
//   - Nested parallel_for runs serially (deadlock prevention)
//   - Exception propagation from worker tasks to caller
//   - Atomic ref-count completion (use-after-free prevention, same as broadcast_pool)
//   - Job serialization via submit_mutex
//   - Constructor barrier (all workers started before returning)
//   - Destructor assertion for outstanding work
//
#ifndef DISPATCH_BENCH_SPIN_POOL_H
#define DISPATCH_BENCH_SPIN_POOL_H

#include "dispatch/macros.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace dispatch_bench {

class spin_pool {
    static constexpr std::size_t kMaxWorkers = 128;
    static constexpr int kSpinCount          = 10000;

    inline static thread_local bool tls_in_parallel_region_ = false;

    // -------------------------------------------------------------------------
    // Job: type-erased work unit with exception propagation and safe completion
    // -------------------------------------------------------------------------

    struct job_base {
        using run_fn_t = void (*)(job_base *, int64_t begin, int64_t end) noexcept;

        run_fn_t run                   = nullptr;
        std::size_t num_workers_to_use = 0;

        // Atomic ref-count instead of std::latch (same pattern as broadcast_pool).
        // Prevents use-after-free: raw atomic decrement has no side effects after
        // hitting zero, unlike std::latch which may notify waiters.
        alignas(64) std::atomic<std::ptrdiff_t> ref_count{0};

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex eptr_mutex;

        explicit job_base(std::ptrdiff_t count) : ref_count(count) {}

        void
        finish_participant() noexcept {
            ref_count.fetch_sub(1, std::memory_order_release);
        }

        void
        wait() {
            while (ref_count.load(std::memory_order_acquire) > 0) {
                DISPATCH_SPIN_PAUSE();
            }
        }
    };

    template <typename Index, typename F> struct job final : job_base {
        F func;

        job(std::size_t participants, F &&f)
            : job_base(static_cast<std::ptrdiff_t>(participants)), func(std::move(f)) {
            num_workers_to_use = participants;
            run                = &job::run_impl;
        }

        static void
        run_impl(job_base *base, int64_t begin, int64_t end) noexcept {
            auto *self = static_cast<job *>(base);

            struct region_guard {
                bool prev;
                region_guard() : prev(tls_in_parallel_region_) { tls_in_parallel_region_ = true; }
                ~region_guard() { tls_in_parallel_region_ = prev; }
            } guard;

            try {
                if (!self->cancelled.load(std::memory_order_relaxed)) {
                    self->func(static_cast<Index>(begin), static_cast<Index>(end));
                }
            } catch (...) {
                if (!self->cancelled.exchange(true, std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(self->eptr_mutex);
                    self->eptr = std::current_exception();
                }
            }

            self->finish_participant();
        }
    };

    // -------------------------------------------------------------------------
    // Per-worker slot (cache-line aligned)
    // -------------------------------------------------------------------------

    struct alignas(64) worker_slot {
        std::atomic<int64_t> begin{0};
        std::atomic<int64_t> end{0};
    };

    // -------------------------------------------------------------------------
    // Pool state
    // -------------------------------------------------------------------------

    worker_slot slots_[kMaxWorkers];
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::size_t num_workers_{0};

    std::atomic<job_base *> current_job_{nullptr};
    std::atomic<std::uint64_t> job_epoch_{0};

    std::mutex cv_mutex_;
    std::condition_variable cv_;

    std::mutex submit_mutex_;
    std::atomic<std::size_t> active_jobs_{0};

    // -------------------------------------------------------------------------
    // Worker thread loop
    // -------------------------------------------------------------------------

    void
    worker_loop(std::size_t my_idx) {
        tls_in_parallel_region_ = true;
        std::uint64_t last_gen  = 0;

        while (!stop_.load(std::memory_order_relaxed)) {
            std::uint64_t gen;

            int spins = 0;
            while ((gen = job_epoch_.load(std::memory_order_acquire)) == last_gen) {
                if (stop_.load(std::memory_order_relaxed))
                    return;

                if (spins < kSpinCount) {
                    DISPATCH_SPIN_PAUSE();
                    ++spins;
                } else {
                    std::unique_lock lock(cv_mutex_);
                    cv_.wait(lock, [&] {
                        gen = job_epoch_.load(std::memory_order_acquire);
                        return gen != last_gen || stop_.load(std::memory_order_relaxed);
                    });
                    break;
                }
            }

            if (stop_.load(std::memory_order_relaxed))
                return;

            // Double-dip epoch check (same as broadcast_pool)
            std::uint64_t const e = job_epoch_.load(std::memory_order_acquire);
            if (e == last_gen)
                continue;

            job_base *j = current_job_.load(std::memory_order_acquire);
            if (!j)
                continue;

            if (job_epoch_.load(std::memory_order_relaxed) != e)
                continue;

            last_gen = e;

            if (my_idx < j->num_workers_to_use) {
                int64_t const b = slots_[my_idx].begin.load(std::memory_order_relaxed);
                int64_t const e = slots_[my_idx].end.load(std::memory_order_relaxed);

                if (b < e) {
                    j->run(j, b, e);
                } else {
                    j->finish_participant();
                }
            }
        }
    }

  public:
    static spin_pool &
    instance() {
        static spin_pool pool;
        return pool;
    }

    spin_pool(spin_pool const &)            = delete;
    spin_pool &operator=(spin_pool const &) = delete;

    [[nodiscard]] std::size_t
    num_threads() const noexcept {
        return num_workers_;
    }

    template <typename Index, typename Func>
    void
    parallel_for(Index start, Index end, Func &&f) {
        static_assert(std::is_integral_v<Index>, "Index must be integral");

        if (tls_in_parallel_region_) {
            f(start, end);
            return;
        }

        if (start >= end)
            return;

        Index const range = end - start;

        if (num_workers_ == 0 || range < 2048) {
            f(start, end);
            return;
        }

        using F = std::decay_t<Func>;
        F func_copy(std::forward<Func>(f));

        std::size_t const total_threads = num_workers_ + 1;
        Index const chunk_size =
            (range + static_cast<Index>(total_threads) - 1) / static_cast<Index>(total_threads);

        // Serialize jobs
        std::unique_lock<std::mutex> submit_lock(submit_mutex_);

        active_jobs_.fetch_add(1, std::memory_order_relaxed);

        // Create typed job on stack (participants = num_workers_, caller runs separately)
        job<Index, F> j(num_workers_, std::move(func_copy));

        // Assign ranges to workers
        for (std::size_t i = 0; i < num_workers_; ++i) {
            Index const worker_start = start + static_cast<Index>(i + 1) * chunk_size;
            Index const worker_end   = std::min(worker_start + chunk_size, end);

            slots_[i].begin.store(static_cast<int64_t>(worker_start), std::memory_order_relaxed);
            slots_[i].end.store(static_cast<int64_t>(worker_end), std::memory_order_relaxed);
        }

        // Publish job
        job_epoch_.fetch_add(1, std::memory_order_release);
        current_job_.store(&j, std::memory_order_release);

        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        // Caller participates on chunk 0
        {
            struct region_guard {
                bool prev;
                region_guard() : prev(tls_in_parallel_region_) { tls_in_parallel_region_ = true; }
                ~region_guard() { tls_in_parallel_region_ = prev; }
            } guard;

            Index const caller_end = std::min(start + chunk_size, end);

            try {
                if (!j.cancelled.load(std::memory_order_relaxed)) {
                    j.func(start, caller_end);
                }
            } catch (...) {
                if (!j.cancelled.exchange(true, std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(j.eptr_mutex);
                    j.eptr = std::current_exception();
                }
            }
        }

        // Wait for all workers via atomic ref-count
        j.wait();

        current_job_.store(nullptr, std::memory_order_release);

        active_jobs_.fetch_sub(1, std::memory_order_relaxed);

        if (j.eptr) {
            std::rethrow_exception(j.eptr);
        }
    }

    template <typename Index, typename Func>
    void
    parallel_for(Index start, Index end, Index /*grain_size*/, Func &&f) {
        parallel_for(start, end, std::forward<Func>(f));
    }

  private:
    spin_pool() {
        std::size_t const hw = std::thread::hardware_concurrency();
        num_workers_         = (hw > 1) ? (hw - 1) : 0;
        if (num_workers_ > kMaxWorkers)
            num_workers_ = kMaxWorkers;

        if (num_workers_ == 0)
            return;

        workers_.reserve(num_workers_);
        std::atomic<std::size_t> started{0};

        for (std::size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this, i, &started] {
                started.fetch_add(1, std::memory_order_release);
                worker_loop(i);
            });
        }

        // Constructor barrier
        while (started.load(std::memory_order_acquire) != num_workers_) {
            DISPATCH_SPIN_PAUSE();
        }
    }

    ~spin_pool() {
        assert(active_jobs_.load(std::memory_order_relaxed) == 0 &&
               "spin_pool destroyed with outstanding work");

        stop_.store(true, std::memory_order_release);
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        for (auto &t: workers_) {
            if (t.joinable())
                t.join();
        }
    }
};

} // namespace dispatch_bench

#endif // DISPATCH_BENCH_SPIN_POOL_H
