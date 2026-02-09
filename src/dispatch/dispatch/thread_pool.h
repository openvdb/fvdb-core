// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// thread_pool.h
//
// Consolidated CPU thread pool with two scheduling strategies:
//
//   thread_pool<scheduling::uniform>  - Static partitioning (broadcast-style).
//   thread_pool<scheduling::adaptive> - Work-stealing (Chase-Lev deques).
//
// Named aliases for convenience:
//   broadcast_pool     = thread_pool<scheduling::uniform>
//   work_stealing_pool = thread_pool<scheduling::adaptive>
//   default_thread_pool = broadcast_pool
//
// Interface (both specializations):
//   static thread_pool& instance();
//   void parallel_for(start, end, grain, func);
//   void parallel_for(start, end, func);
//   std::size_t num_threads() const noexcept;
//
// IMPORTANT USAGE NOTES:
//
// 1. Pool lifetime: Each pool is a singleton and must outlive all parallel_for()
//    calls. Destroying a pool with outstanding work will deadlock.
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

#ifndef DISPATCH_DISPATCH_THREAD_POOL_H
#define DISPATCH_DISPATCH_THREAD_POOL_H

#include "dispatch/enums.h"
#include "dispatch/macros.h"

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
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace dispatch {

// =============================================================================
// Primary template (never instantiated directly)
// =============================================================================

template <scheduling S> class thread_pool;

// =============================================================================
// thread_pool<scheduling::uniform> — broadcast-style static partitioning
// =============================================================================
//
// Key design points:
//  - Persistent worker threads with broadcast-style job dispatch.
//  - Static partitioning with optional grain alignment.
//  - "Hot wait" (blocktime) before sleeping to avoid OS wake/sleep costs.
//  - Caller participates as the last worker (N+1 threads for N workers).
//  - Nested parallel_for runs serially to prevent deadlocks.

template <> class thread_pool<scheduling::uniform> {
  public:
    static thread_pool &
    instance() {
        static thread_pool pool;
        return pool;
    }

    thread_pool(thread_pool const &)            = delete;
    thread_pool &operator=(thread_pool const &) = delete;

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

      protected:
        static std::size_t
        mul_div(std::size_t a, std::size_t b, std::size_t d) {
#if defined(__SIZEOF_INT128__)
            return static_cast<std::size_t>((static_cast<__uint128_t>(a) * b) / d);
#elif defined(_MSC_VER) && defined(_M_X64)
            unsigned __int64 high;
            unsigned __int64 low = _umul128(a, b, &high);
            unsigned __int64 rem;
            return _udiv128(high, low, d, &rem);
#else
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

                    using wide_index_t =
                        std::conditional_t<std::is_signed_v<Index>, std::int64_t, std::uint64_t>;

                    wide_index_t const w_start = static_cast<wide_index_t>(start);
                    wide_index_t const w_end   = static_cast<wide_index_t>(end);
                    wide_index_t const w_grain = static_cast<wide_index_t>(grain);

                    wide_index_t begin_w  = w_start + static_cast<wide_index_t>(b0) * w_grain;
                    wide_index_t finish_w = w_start + static_cast<wide_index_t>(b1) * w_grain;

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

    thread_pool() : num_workers_(choose_default_workers()) {
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

        while (started.load(std::memory_order_acquire) != num_workers_) {
            DISPATCH_SPIN_PAUSE();
        }
    }

    ~thread_pool() {
        assert(active_jobs_.load(std::memory_order_relaxed) == 0 &&
               "thread_pool<uniform> destroyed with outstanding work - this will deadlock");

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
        std::uint64_t local_epoch = 0;
        constexpr auto kBlockTime = std::chrono::microseconds(500);

        while (!stop_.load(std::memory_order_acquire)) {
            if (job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                auto const start_wait = std::chrono::steady_clock::now();
                int spins             = 0;

                while (!stop_.load(std::memory_order_relaxed) &&
                       job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                    if ((std::chrono::steady_clock::now() - start_wait) < kBlockTime) {
                        if ((++spins & 0xFF) == 0) {
                            std::this_thread::yield();
                        } else {
                            DISPATCH_SPIN_PAUSE();
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

            job_base *j = current_job_.load(std::memory_order_acquire);

            if (!j)
                continue;

            if (job_epoch_.load(std::memory_order_relaxed) != e)
                continue;

            local_epoch = e;

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

    std::atomic<job_base *> current_job_{nullptr};
    std::atomic<std::uint64_t> job_epoch_{0};

    std::mutex cv_mutex_;
    std::condition_variable cv_;

    std::mutex submit_mutex_;

    std::atomic<std::size_t> active_jobs_{0};

    inline static thread_local int tls_worker_index_        = -1;
    inline static thread_local bool tls_in_parallel_region_ = false;
};

// =============================================================================
// Chase-Lev Work-Stealing Deque (internal to adaptive pool)
// =============================================================================

template <typename T> class work_stealing_deque {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable for atomic<T>");

  public:
    static constexpr std::size_t kDefaultCapacity = 65536;

    explicit work_stealing_deque(std::size_t capacity = kDefaultCapacity)
        : capacity_(round_up_to_power_of_two(capacity)), mask_(capacity_ - 1),
          buffer_(std::make_unique<std::atomic<T>[]>(capacity_)), top_(0), bottom_(0) {
        assert(is_power_of_two(capacity_) && "Chase-Lev deque capacity must be a power of two");
    }

    work_stealing_deque(work_stealing_deque const &)            = delete;
    work_stealing_deque &operator=(work_stealing_deque const &) = delete;

    void
    push(T item) noexcept {
        std::int64_t b                  = bottom_.load(std::memory_order_relaxed);
        [[maybe_unused]] std::int64_t t = top_.load(std::memory_order_relaxed);
        assert(static_cast<std::size_t>(b - t) < capacity_ && "Chase-Lev deque overflow");
        buffer_[static_cast<std::size_t>(b) & mask_].store(item, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    std::optional<T>
    pop() noexcept {
        std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
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
    [[nodiscard]] static constexpr bool
    is_power_of_two(std::size_t x) noexcept {
        return x != 0 && (x & (x - 1)) == 0;
    }

    [[nodiscard]] static constexpr std::size_t
    round_up_to_power_of_two(std::size_t x) noexcept {
        if (x == 0)
            return 1;
        if (is_power_of_two(x))
            return x;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        if constexpr (sizeof(std::size_t) > 4) {
            x |= x >> 32;
        }
        return x + 1;
    }

    std::size_t capacity_;
    std::size_t mask_;
    std::unique_ptr<std::atomic<T>[]> buffer_;
    alignas(64) std::atomic<std::int64_t> top_;
    alignas(64) std::atomic<std::int64_t> bottom_;
};

// =============================================================================
// thread_pool<scheduling::adaptive> — Chase-Lev work-stealing
// =============================================================================
//
// Key design points:
//  - Per-worker Chase-Lev deques for lock-free task distribution.
//  - Randomized victim selection for steal attempts.
//  - Caller participates as a worker (N+1 threads for N workers).
//  - Optimal for imbalanced workloads where static partitioning leaves threads idle.

template <> class thread_pool<scheduling::adaptive> {
  public:
    static thread_pool &
    instance() {
        static thread_pool pool;
        return pool;
    }

    thread_pool(thread_pool const &)            = delete;
    thread_pool &operator=(thread_pool const &) = delete;

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

        // Track active jobs for destructor safety check
        active_jobs_.fetch_add(1, std::memory_order_relaxed);

        // Reset worker exit counter
        worker_exit_count_.store(0, std::memory_order_relaxed);

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
        job_epoch_.fetch_add(1, std::memory_order_release);
        current_job_.store(&job_ctx, std::memory_order_release);

        // Wake workers
        { std::lock_guard<std::mutex> lk(cv_mutex_); }
        cv_.notify_all();

        // Main thread participates
        tls_in_parallel_region_ = true;
        execute_work(&job_ctx, main_deque_, num_workers_);
        tls_in_parallel_region_ = false;

        // Signal main thread is done working
        job_ctx.participant_done();

        // Wait for all participants
        job_ctx.wait();

        // Wait for all workers to fully exit participant_done()
        while (worker_exit_count_.load(std::memory_order_acquire) < num_workers_) {
            DISPATCH_SPIN_PAUSE();
        }

        current_job_.store(nullptr, std::memory_order_release);

        active_jobs_.fetch_sub(1, std::memory_order_relaxed);

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

        alignas(64) std::atomic<std::size_t> remaining{0};

        alignas(64) std::latch done_latch;

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex eptr_mutex;

        explicit job_base(std::size_t n, std::size_t participants)
            : num_tasks(n), remaining(n), done_latch(static_cast<std::ptrdiff_t>(participants)) {}

        void
        complete_task() noexcept {
            remaining.fetch_sub(1, std::memory_order_relaxed);
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
            return remaining.load(std::memory_order_relaxed) == 0;
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
                    using wide_index_t =
                        std::conditional_t<std::is_signed_v<Index>, std::int64_t, std::uint64_t>;

                    wide_index_t const wide_start = static_cast<wide_index_t>(self->start);
                    wide_index_t const wide_end   = static_cast<wide_index_t>(self->end);
                    wide_index_t const wide_grain = static_cast<wide_index_t>(self->grain);

                    wide_index_t task_start_wide =
                        wide_start + static_cast<wide_index_t>(task_id) * wide_grain;
                    wide_index_t task_end_wide = task_start_wide + wide_grain;

                    if (task_start_wide > wide_end)
                        task_start_wide = wide_end;
                    if (task_end_wide > wide_end)
                        task_end_wide = wide_end;

                    Index task_start = static_cast<Index>(task_start_wide);
                    Index task_end   = static_cast<Index>(task_end_wide);

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
            if (ctx->all_tasks_done())
                return;

            std::optional<std::size_t> task_id = my_deque.pop();

            if (!task_id) {
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
                DISPATCH_SPIN_PAUSE();
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
                            DISPATCH_SPIN_PAUSE();
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

            job_base *ctx = current_job_.load(std::memory_order_acquire);
            if (!ctx)
                continue;

            if (job_epoch_.load(std::memory_order_relaxed) != e)
                continue;

            local_epoch = e;

            execute_work(ctx, *worker_deques_[worker_index], worker_index);

            ctx->participant_done();

            worker_exit_count_.fetch_add(1, std::memory_order_release);
        }
    }

    thread_pool() : num_workers_(choose_default_workers()), main_deque_() {
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
            DISPATCH_SPIN_PAUSE();
        }
    }

    ~thread_pool() {
        assert(active_jobs_.load(std::memory_order_relaxed) == 0 &&
               "thread_pool<adaptive> destroyed with outstanding work - this will deadlock");

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

    alignas(64) std::atomic<std::size_t> worker_exit_count_{0};

    std::atomic<std::size_t> active_jobs_{0};

    std::mutex cv_mutex_;
    std::condition_variable cv_;
    std::mutex submit_mutex_;

    inline static thread_local bool tls_in_parallel_region_ = false;
};

// =============================================================================
// Named aliases
// =============================================================================

using broadcast_pool      = thread_pool<scheduling::uniform>;
using work_stealing_pool  = thread_pool<scheduling::adaptive>;
using default_thread_pool = work_stealing_pool;

} // namespace dispatch

#endif // DISPATCH_DISPATCH_THREAD_POOL_H
