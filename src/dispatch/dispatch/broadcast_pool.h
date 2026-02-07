// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// broadcast_pool.h
//
// Practical, low-overhead CPU parallel_for for C++20 (no TBB, no OpenMP).
//
// Key design points:
//  - Persistent worker threads.
//  - Broadcast-style job dispatch (no per-call task queue fanout).
//  - Static partitioning (OpenMP-like "static") with optional grain alignment.
//  - "Hot wait" (blocktime) before sleeping to avoid OS wake/sleep costs
//    dominating tight benchmark loops.
//  - Nested parallel_for inside a parallel region runs serially to prevent
//    deadlocks and "overlapping jobs" complexity.
//
// Interface preserved:
//   dispatch::broadcast_pool::instance().parallel_for(start, end, grain, func);
//
// IMPORTANT USAGE NOTES:
//
// 1. Pool lifetime: The pool must outlive all parallel_for() calls. Since this
//    is a singleton, this is normally satisfied. Destroying the pool with
//    outstanding work will deadlock.
//
// 2. Nested parallel_for: Calling parallel_for() from within a task is SAFE.
//    The nested call executes synchronously (serially) on the calling thread.
//    This avoids deadlock and is often the desired behavior when the outer
//    loop already saturates available cores.
//
// 3. No cross-thread nested submission: Tasks must NOT spawn new threads that
//    call parallel_for() and then wait/join on them. This causes deadlock
//    because submit_mutex_ is held for the job duration. Same-thread nested
//    calls (point 2 above) are safe.
//
// Notes:
//  - This is intentionally range-parallel, not a general task scheduler.
//  - If you need nested *parallel* regions (true nested parallelism), you need a
//    tasking runtime (work-stealing deques, etc.). This is the "solid baseline"
//    that should not be orders-of-magnitude slower than a good reference.

#ifndef DISPATCH_DISPATCH_BROADCAST_POOL_H
#define DISPATCH_DISPATCH_BROADCAST_POOL_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define DISPATCH_BROADCAST_SPIN_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define DISPATCH_BROADCAST_SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define DISPATCH_BROADCAST_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define DISPATCH_BROADCAST_SPIN_PAUSE() std::this_thread::yield()
#endif

namespace dispatch {

class broadcast_pool {
  public:
    static broadcast_pool &
    instance() {
        static broadcast_pool pool;
        return pool;
    }

    broadcast_pool(broadcast_pool const &)            = delete;
    broadcast_pool &operator=(broadcast_pool const &) = delete;

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

        // Auto grain if caller passes <= 0 (signed) or == 0 (unsigned).
        Index grain = normalize_grain<Index>(grain_size, n, num_workers_ + 1);

        // Choose participants based on available work.
        std::size_t const max_participants = num_workers_ + 1;

        std::size_t const blocks = blocks_for(n, grain);
        if (blocks <= 1) {
            std::invoke(f, start, end);
            return;
        }

        std::size_t participants = std::min(max_participants, blocks);

        // Heuristic: don't use more threads than useful for the amount of work.
        constexpr std::size_t kMinGrainsPerThread = 4;
        {
            std::size_t const max_by_work = max_participants_by_work(n, grain, kMinGrainsPerThread);
            participants = std::min(participants, std::max<std::size_t>(1, max_by_work));
        }

        if (participants <= 1) {
            std::invoke(f, start, end);
            return;
        }

        // Serialize jobs: this runtime runs exactly one broadcast job at a time.
        std::unique_lock<std::mutex> submit_lock(submit_mutex_);

        // Track active jobs for destructor safety check.
        active_jobs_.fetch_add(1, std::memory_order_relaxed);

        // job lives on stack; workers only use it until we wait for completion.
        job<Index, F> j(start, end, grain, participants, std::move(f));

        // Publish job to workers.
        // Increment epoch BEFORE storing pointer for acquire/release pairing.
        // This ensures workers that see the new epoch will also see the new pointer.
        job_epoch_.fetch_add(1, std::memory_order_release);
        current_job_.store(&j, std::memory_order_release);

        // Wake sleepers (hot threads observe epoch change without kernel wake).
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        // Caller participates as the last participant id.
        std::size_t const caller_pid = participants - 1;
        j.run_participant(caller_pid);

        // Wait for all participants to finish.
        j.wait();

        // Clear published job.
        current_job_.store(nullptr, std::memory_order_release);

        active_jobs_.fetch_sub(1, std::memory_order_relaxed);

        // Propagate exception if any participant threw.
        if (j.eptr) {
            std::rethrow_exception(j.eptr);
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
    // Helper types and functions
    // -------------------------------------------------------------------------

    template <typename Index>
    using unsigned_index_t = std::make_unsigned_t<std::remove_cv_t<Index>>;

    template <typename Index>
    static constexpr bool
    is_nonpositive(Index v) {
        if constexpr (std::is_signed_v<Index>)
            return v <= Index{0};
        else
            return v == Index{0};
    }

    template <typename Index>
    static Index
    normalize_grain(Index grain_size, Index n, std::size_t threads_hint) {
        if (!is_nonpositive(grain_size))
            return grain_size;

        // Aim for ~4 chunks per thread, clamp to at least 1.
        unsigned_index_t<Index> un = static_cast<unsigned_index_t<Index>>(n);
        if (un == 0)
            return Index{1};

        std::size_t const denom = std::max<std::size_t>(1, threads_hint * 4);
        unsigned_index_t<Index> g =
            static_cast<unsigned_index_t<Index>>(un / static_cast<unsigned_index_t<Index>>(denom));
        if (g == 0)
            g = 1;
        return static_cast<Index>(g);
    }

    template <typename Index>
    static std::size_t
    blocks_for(Index n, Index grain) {
        // blocks = ceil(n / grain)
        unsigned_index_t<Index> un = static_cast<unsigned_index_t<Index>>(n);
        unsigned_index_t<Index> ug = static_cast<unsigned_index_t<Index>>(grain);
        if (ug == 0)
            ug = 1;
        unsigned_index_t<Index> q = un / ug;
        unsigned_index_t<Index> r = un % ug;
        unsigned_index_t<Index> b = q + (r ? 1 : 0);

        if (b > static_cast<unsigned_index_t<Index>>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(b);
    }

    template <typename Index>
    static std::size_t
    max_participants_by_work(Index n, Index grain, std::size_t min_grains_per_thread) {
        // Limit threads so each gets >= min_grains_per_thread * grain elements.
        unsigned_index_t<Index> un = static_cast<unsigned_index_t<Index>>(n);
        unsigned_index_t<Index> ug = static_cast<unsigned_index_t<Index>>(grain);
        if (ug == 0)
            ug = 1;

        unsigned_index_t<Index> min_elems = ug;
        for (std::size_t i = 1; i < min_grains_per_thread; ++i) {
            if (min_elems > (std::numeric_limits<unsigned_index_t<Index>>::max() - ug)) {
                min_elems = std::numeric_limits<unsigned_index_t<Index>>::max();
                break;
            }
            min_elems += ug;
        }
        if (min_elems == 0)
            min_elems = 1;

        unsigned_index_t<Index> q = un / min_elems;
        unsigned_index_t<Index> r = un % min_elems;
        unsigned_index_t<Index> t = q + (r ? 1 : 0);
        if (t == 0)
            t = 1;

        if (t > static_cast<unsigned_index_t<Index>>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(t);
    }

    static std::size_t
    choose_default_workers() {
        // Default to hw-1 so the caller can participate without oversubscription.
        std::size_t hw = std::thread::hardware_concurrency();
        if (hw == 0)
            hw = 1;
        if (hw <= 1)
            return 0;
        return hw - 1;
    }

    // -------------------------------------------------------------------------
    // job_base: type-erased run function
    // -------------------------------------------------------------------------

    struct job_base {
        using run_fn_t = void (*)(job_base *, std::size_t pid) noexcept;

        run_fn_t run             = nullptr;
        std::size_t participants = 0;
        std::size_t blocks       = 0;

        // Atomic ref-count instead of std::latch. This fixes potential use-after-free:
        // unlike std::latch, a raw atomic decrement has no complex internal logic
        // (like notifying waiters) that runs after the counter hits zero.
        // Aligned to prevent false sharing with the read-only fields above.
        alignas(64) std::atomic<std::ptrdiff_t> ref_count{0};

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex eptr_mutex;

        explicit job_base(std::ptrdiff_t count) : ref_count(count) {}

        void
        finish_participant() noexcept {
            // Atomic decrement is safe as the final instruction. Unlike std::latch,
            // this has no side effects (like notifying waiters) that could race
            // against the main thread destroying 'this'.
            ref_count.fetch_sub(1, std::memory_order_release);
        }

        void
        wait() {
            // Spin-wait until all participants have fully checked out.
            while (ref_count.load(std::memory_order_acquire) > 0) {
                DISPATCH_BROADCAST_SPIN_PAUSE();
            }
        }

      protected:
        static std::size_t
        mul_div(std::size_t a, std::size_t b, std::size_t d) {
#if defined(__SIZEOF_INT128__)
            return static_cast<std::size_t>((static_cast<__uint128_t>(a) * b) / d);
#elif defined(_MSC_VER) && defined(_M_X64)
            // Use MSVC intrinsics to prevent overflow on 64-bit Windows.
            unsigned __int64 high;
            unsigned __int64 low = _umul128(a, b, &high);
            unsigned __int64 rem;
            return _udiv128(high, low, d, &rem);
#else
            // 32-bit fallback - may overflow for huge inputs.
            return (a * b) / d;
#endif
        }
    };

    // -------------------------------------------------------------------------
    // job: typed job with static partition and grain-aligned boundaries
    // -------------------------------------------------------------------------

    template <typename Index, typename F> struct job final : job_base {
        Index start;
        Index end;
        Index grain;
        F func;

        job(Index s, Index e, Index g, std::size_t parts, F &&f)
            : job_base(static_cast<std::ptrdiff_t>(std::max<std::size_t>(1, parts))), start(s),
              end(e), grain(g), func(std::move(f)) {
            participants = std::max<std::size_t>(1, parts);
            blocks       = blocks_for(end - start, grain);
            run          = &job::run_impl;
        }

        static void
        run_impl(job_base *base, std::size_t pid) noexcept {
            static_cast<job *>(base)->run_participant(pid);
        }

        void
        run_participant(std::size_t pid) noexcept {
            // Mark this thread as inside a parallel region so nested calls run serially.
            struct region_guard {
                bool prev;
                region_guard() : prev(tls_in_parallel_region_) { tls_in_parallel_region_ = true; }
                ~region_guard() { tls_in_parallel_region_ = prev; }
            } guard;

            try {
                if (!cancelled.load(std::memory_order_relaxed)) {
                    std::size_t const P = participants;
                    std::size_t const B = blocks;

                    if (pid >= P || B == 0) {
                        finish_participant();
                        return;
                    }

                    std::size_t const b0 = mul_div(B, pid, P);
                    std::size_t const b1 = mul_div(B, pid + 1, P);

                    // Use wide arithmetic to prevent integer overflow.
                    // (b0 * grain) can overflow 32-bit Index types for large ranges.
                    using wide_index_t =
                        std::conditional_t<std::is_signed_v<Index>, std::int64_t, std::uint64_t>;

                    wide_index_t const w_start = static_cast<wide_index_t>(start);
                    wide_index_t const w_end   = static_cast<wide_index_t>(end);
                    wide_index_t const w_grain = static_cast<wide_index_t>(grain);

                    wide_index_t begin_w  = w_start + static_cast<wide_index_t>(b0) * w_grain;
                    wide_index_t finish_w = w_start + static_cast<wide_index_t>(b1) * w_grain;

                    // Clamp to valid range.
                    if (finish_w > w_end)
                        finish_w = w_end;

                    Index begin  = static_cast<Index>(begin_w);
                    Index finish = static_cast<Index>(finish_w);

                    if (begin < finish) {
                        std::invoke(func, begin, finish);
                    }
                }
            } catch (...) {
                if (!cancelled.exchange(true, std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(eptr_mutex);
                    eptr = std::current_exception();
                }
            }

            finish_participant();
        }
    };

    // -------------------------------------------------------------------------
    // Pool construction and destruction
    // -------------------------------------------------------------------------

    broadcast_pool() : num_workers_(choose_default_workers()) {
        if (num_workers_ == 0)
            return;

        workers_.reserve(num_workers_);

        std::atomic<std::size_t> started{0};

        for (std::size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this, i, &started] {
                tls_worker_index_ = static_cast<int>(i);
                started.fetch_add(1, std::memory_order_release);
                worker_loop(i);
                tls_worker_index_ = -1;
            });
        }

        // Spin until all workers have started (constructor barrier).
        while (started.load(std::memory_order_acquire) != num_workers_) {
            DISPATCH_BROADCAST_SPIN_PAUSE();
        }
    }

    ~broadcast_pool() {
        // Debug check: pool must not be destroyed with outstanding work.
        // If this fires, a job is still in progress when the pool is being destroyed,
        // which will cause deadlock (workers waiting on job, destructor waiting on workers).
        assert(active_jobs_.load(std::memory_order_relaxed) == 0 &&
               "broadcast_pool destroyed with outstanding work - this will deadlock");

        stop_.store(true, std::memory_order_release);
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        for (auto &t: workers_) {
            if (t.joinable())
                t.join();
        }
    }

    // -------------------------------------------------------------------------
    // Worker thread main loop
    // -------------------------------------------------------------------------

    void
    worker_loop(std::size_t worker_index) {
        // Start with epoch 0 rather than loading the current value. This
        // prevents a race where a worker starts, loads an already-incremented
        // epoch as its local_epoch, then waits forever because the job has
        // already been published.
        std::uint64_t local_epoch = 0;

        // Stay hot briefly before sleeping to avoid kernel wake/sleep costs
        // dominating tight benchmark loops.
        constexpr auto kBlockTime = std::chrono::microseconds(500);

        while (!stop_.load(std::memory_order_acquire)) {
            // Wait for a new epoch (spin/yield for blocktime, then sleep).
            if (job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                auto const start_wait = std::chrono::steady_clock::now();
                int spins             = 0;

                while (!stop_.load(std::memory_order_relaxed) &&
                       job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                    // Hot wait for blocktime.
                    if ((std::chrono::steady_clock::now() - start_wait) < kBlockTime) {
                        if ((++spins & 0xFF) == 0) {
                            std::this_thread::yield();
                        } else {
                            DISPATCH_BROADCAST_SPIN_PAUSE();
                        }
                        continue;
                    }

                    // Sleep until epoch changes or stop is signaled.
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

            // Observe new job.
            std::uint64_t const e = job_epoch_.load(std::memory_order_acquire);
            if (e == local_epoch)
                continue;

            job_base *j = current_job_.load(std::memory_order_acquire);

            // If pointer is null, the publisher may still be in the window between
            // incrementing epoch and storing the pointer. Retry without committing
            // to this epoch, otherwise we'd miss the job entirely.
            if (!j)
                continue;

            // Double-dip epoch check: Verify epoch didn't change while loading pointer.
            // If a worker is preempted between reading the epoch and loading the pointer,
            // it could load a job pointer for a different epoch than it recorded, leading
            // to double-execution of the new job.
            if (job_epoch_.load(std::memory_order_relaxed) != e)
                continue;

            // Only update local_epoch after we have BOTH a consistent epoch AND
            // a non-null pointer. This ensures we don't skip a job.
            local_epoch = e;

            // Workers get participant IDs [0, participants-2]; caller is participants-1.
            std::size_t const workers_to_use = (j->participants > 0) ? (j->participants - 1) : 0;
            if (worker_index < workers_to_use) {
                j->run(j, worker_index);
            }
        }
    }

  private:
    std::size_t num_workers_{0};
    std::vector<std::thread> workers_;

    std::atomic<bool> stop_{false};

    // Single-job broadcast state.
    std::atomic<job_base *> current_job_{nullptr};
    std::atomic<std::uint64_t> job_epoch_{0};

    // Sleep/wake for idle workers.
    std::mutex cv_mutex_;
    std::condition_variable cv_;

    // Serialize jobs (one at a time).
    std::mutex submit_mutex_;

    // Debug counter for active jobs - used to detect destruction during active work.
    std::atomic<std::size_t> active_jobs_{0};

    // TLS: worker id (>= 0 if pool thread).
    inline static thread_local int tls_worker_index_ = -1;
    // TLS: true if currently executing inside a parallel_for region.
    inline static thread_local bool tls_in_parallel_region_ = false;
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_BROADCAST_POOL_H
