// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// work_stealing_pool.h
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
//   dispatch::thread_pool::instance().parallel_for(start, end, grain, func);
//
// Notes:
//  - This is intentionally range-parallel, not a general task scheduler.
//  - If you need nested *parallel* regions (true nested parallelism), you need a
//    tasking runtime (work-stealing deques, etc.). This is the “solid baseline”
//    that should not be orders-of-magnitude slower than a good reference.

#ifndef DISPATCH_DISPATCH_WORK_STEALING_POOL_H
#define DISPATCH_DISPATCH_WORK_STEALING_POOL_H

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <latch>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#define DISPATCH_SPIN_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define DISPATCH_SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define DISPATCH_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define DISPATCH_SPIN_PAUSE() std::this_thread::yield()
#endif

namespace dispatch {

class work_stealing_pool {
public:
    static work_stealing_pool& instance() {
        static work_stealing_pool pool;
        return pool;
    }

    work_stealing_pool(const work_stealing_pool&)            = delete;
    work_stealing_pool& operator=(const work_stealing_pool&) = delete;

    template <typename Index, typename Func>
        requires std::integral<Index> && std::invocable<std::decay_t<Func>&, Index, Index>
    void parallel_for(Index start, Index end, Index grain_size, Func&& func) {
        if (start >= end) return;

        // If we're already executing inside one of *our* parallel regions,
        // run nested regions serially. This avoids:
        //  - deadlock (if we serialized jobs with a mutex)
        //  - overlapping "broadcast jobs" (this pool runs 1 job at a time)
        //  - oversubscription explosions
        if (tls_in_parallel_region_) {
            auto f = std::forward<Func>(func);
            std::invoke(f, start, end);
            return;
        }

        // No worker threads configured -> always serial.
        if (num_workers_ == 0) {
            auto f = std::forward<Func>(func);
            std::invoke(f, start, end);
            return;
        }

        using F = std::decay_t<Func>;
        F f(std::forward<Func>(func));

        const Index n = end - start;

        // "Auto grain" if caller passes <= 0 (signed) or ==0 (unsigned).
        // This is a conservative heuristic: aim for ~4 chunks per hardware thread.
        Index grain = normalize_grain<Index>(grain_size, n, num_workers_ + 1);

        // Choose participants (threads) based on available work:
        // - cap by pool size (+ caller)
        // - cap by number of grain-sized blocks
        // - cap so each participant gets >= kMinGrainsPerThread * grain elements
        const std::size_t max_participants = num_workers_ + 1; // workers + caller

        const std::size_t blocks = blocks_for(n, grain);
        if (blocks <= 1) {
            std::invoke(f, start, end);
            return;
        }

        std::size_t participants = std::min(max_participants, blocks);

        // Heuristic: don't use more threads than "useful" for the amount of work.
        // This prevents thrash on small-ish ranges and mirrors what good runtimes do.
        constexpr std::size_t kMinGrainsPerThread = 4; // tune: 2..8
        {
            const std::size_t max_by_work = max_participants_by_work(n, grain, kMinGrainsPerThread);
            participants = std::min(participants, std::max<std::size_t>(1, max_by_work));
        }

        if (participants <= 1) {
            std::invoke(f, start, end);
            return;
        }

        // Serialize jobs: this runtime runs exactly one broadcast job at a time.
        // (You can lift this later with a real job queue/tasking scheduler.)
        std::unique_lock<std::mutex> submit_lock(submit_mutex_);

        // Job object lives on the stack; workers only use it until we wait for completion.
        Job<Index, F> job(start, end, grain, participants, std::move(f));

        // Publish job to workers.
        current_job_.store(&job, std::memory_order_release);
        job_epoch_.fetch_add(1, std::memory_order_release);

        // Wake sleepers (hot threads will observe epoch change without needing kernel wake).
        {
            std::lock_guard<std::mutex> lk(cv_mutex_);
            // Nothing else needed; we just pair notify with the mutex to reduce races.
        }
        cv_.notify_all();

        // Caller participates as the last participant id.
        const std::size_t caller_pid = participants - 1;
        job.run_participant(caller_pid);

        // Wait for all participants to finish (spin-then-sleep).
        job.wait();

        // Clear published job.
        current_job_.store(nullptr, std::memory_order_release);

        // Propagate exception if any participant threw.
        if (job.eptr) {
            std::rethrow_exception(job.eptr);
        }
    }

    template <typename Index, typename F>
    void parallel_for(Index start, Index end, F&& f) {
        parallel_for(start, end, Index{0}, std::forward<F>(f));
    }

    [[nodiscard]] std::size_t num_threads() const noexcept {
        return num_workers_;
    }

private:
    // -----------------------------
    // Small helpers
    // -----------------------------
    template <typename Index>
    using UIndex = std::make_unsigned_t<std::remove_cv_t<Index>>;

    template <typename Index>
    static constexpr bool is_nonpositive(Index v) {
        if constexpr (std::is_signed_v<Index>) return v <= Index{0};
        else return v == Index{0};
    }

    template <typename Index>
    static Index normalize_grain(Index grain_size, Index n, std::size_t threads_hint) {
        if (!is_nonpositive(grain_size)) return grain_size;

        // Aim for ~4 chunks per thread: grain ~= n / (threads * 4)
        // Clamp to at least 1.
        // Note: This is intentionally simple; user can always pass an explicit grain.
        UIndex<Index> un = static_cast<UIndex<Index>>(n);
        if (un == 0) return Index{1};

        const std::size_t denom = std::max<std::size_t>(1, threads_hint * 4);
        UIndex<Index> g = static_cast<UIndex<Index>>(un / static_cast<UIndex<Index>>(denom));
        if (g == 0) g = 1;
        return static_cast<Index>(g);
    }

    template <typename Index>
    static std::size_t blocks_for(Index n, Index grain) {
        // blocks = ceil(n / grain) computed as: n/grain + (n%grain != 0)
        UIndex<Index> un = static_cast<UIndex<Index>>(n);
        UIndex<Index> ug = static_cast<UIndex<Index>>(grain);
        if (ug == 0) ug = 1;
        UIndex<Index> q = un / ug;
        UIndex<Index> r = un % ug;
        UIndex<Index> b = q + (r ? 1 : 0);

        if (b > static_cast<UIndex<Index>>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(b);
    }

    template <typename Index>
    static std::size_t max_participants_by_work(Index n, Index grain, std::size_t min_grains_per_thread) {
        // limit threads so each gets >= min_grains_per_thread * grain elements
        UIndex<Index> un = static_cast<UIndex<Index>>(n);
        UIndex<Index> ug = static_cast<UIndex<Index>>(grain);
        if (ug == 0) ug = 1;

        UIndex<Index> min_elems = ug;
        for (std::size_t i = 1; i < min_grains_per_thread; ++i) {
            if (min_elems > (std::numeric_limits<UIndex<Index>>::max() - ug)) {
                // overflow -> just clamp
                min_elems = std::numeric_limits<UIndex<Index>>::max();
                break;
            }
            min_elems += ug;
        }
        if (min_elems == 0) min_elems = 1;

        UIndex<Index> q = un / min_elems;
        UIndex<Index> r = un % min_elems;
        UIndex<Index> t = q + (r ? 1 : 0);
        if (t == 0) t = 1;

        if (t > static_cast<UIndex<Index>>(std::numeric_limits<std::size_t>::max())) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(t);
    }

    static std::size_t choose_default_workers() {
        // Default to "hw-1" so the caller can participate without oversubscription.
        std::size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (hw <= 1) return 0;
        return hw - 1;
    }

    // -----------------------------
    // Job base (type-erased run fn)
    // -----------------------------
    struct OLD_JobBase {
        using RunFn = void (*)(OLD_JobBase*, std::size_t pid) noexcept;

        RunFn run = nullptr;

        std::size_t participants = 0;
        std::size_t blocks       = 0;

        std::atomic<int> remaining{0};

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex         eptr_mutex;

        std::mutex              done_mutex;
        std::condition_variable done_cv;

        void finish_participant() noexcept {
            // Last participant wakes the caller.
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // notify without holding lock is fine; caller uses predicate.
                done_cv.notify_one();
            }
        }

        void wait_done_spin_then_sleep() {
            // Fast path: spin briefly (no kernel transition for tiny jobs).
            int spins = 0;
            while (remaining.load(std::memory_order_acquire) != 0 && spins < 2000) {
                DISPATCH_SPIN_PAUSE();
                ++spins;
            }

            if (remaining.load(std::memory_order_acquire) == 0) return;

            std::unique_lock<std::mutex> lk(done_mutex);
            done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
        }
    };

    // -----------------------------
    // Job base (type-erased run fn)
    // -----------------------------
    struct JobBase {
        using RunFn = void (*)(JobBase*, std::size_t pid) noexcept;

        RunFn run = nullptr;

        std::size_t participants = 0;
        std::size_t blocks       = 0;

        // NEW: The robust C++20 synchronization primitive
        std::latch done_latch;

        std::atomic<bool> cancelled{false};
        std::exception_ptr eptr;
        std::mutex         eptr_mutex;

        // Initialize latch with the number of expected participants
        explicit JobBase(std::ptrdiff_t count) : done_latch(count) {}

        void finish_participant() noexcept {
            // Safely decrements the counter.
            // Unlike std::condition_variable, this is safe even if the
            // parent thread destroys the 'Job' object immediately after this returns.
            done_latch.count_down();
        }

        void wait() {
            // Blocks until internal counter reaches 0.
            // C++ implementations typically use an efficient spin-wait strategy
            // internally before falling back to OS sleep, preserving your performance.
            done_latch.wait();
        }

    protected:
        // Moved here. Protected so 'Job' can use it.
        static std::size_t mul_div(std::size_t a, std::size_t b, std::size_t d) {
#if defined(__SIZEOF_INT128__)
            return static_cast<std::size_t>((static_cast<__uint128_t>(a) * b) / d);
#else
            return (a * b) / d;
#endif
        }
    };

    // -----------------------------
    // Typed job: static partition, grain-aligned boundaries
    // -----------------------------
    template <typename Index, typename F>
    struct OLD_Job final : OLD_JobBase {
        Index start;
        Index end;
        Index grain;
        F     func;

        OLD_Job(Index s, Index e, Index g, std::size_t parts, F&& f)
            : start(s), end(e), grain(g), func(std::move(f)) {

            participants = std::max<std::size_t>(1, parts);
            blocks       = blocks_for(end - start, grain);
            remaining.store(static_cast<int>(participants), std::memory_order_relaxed);

            run = &OLD_Job::run_impl;
        }

        static void run_impl(OLD_JobBase* base, std::size_t pid) noexcept {
            static_cast<OLD_Job*>(base)->run_participant(pid);
        }

        void run_participant(std::size_t pid) noexcept {
            // Mark this thread as "inside a parallel region" so nested calls run serially.
            struct Guard {
                bool prev;
                Guard() : prev(tls_in_parallel_region_) { tls_in_parallel_region_ = true; }
                ~Guard() { tls_in_parallel_region_ = prev; }
            } guard;

            try {
                if (!cancelled.load(std::memory_order_relaxed)) {
                    // Compute this participant's block range [b0, b1)
                    const std::size_t P = participants;
                    const std::size_t B = blocks;

                    if (pid >= P || B == 0) {
                        finish_participant();
                        return;
                    }

                    const std::size_t b0 = mul_div(B, pid, P);
                    const std::size_t b1 = mul_div(B, pid + 1, P);

                    // Convert blocks -> element indices, align to grain.
                    Index begin = start + static_cast<Index>(b0) * grain;
                    Index finish = start + static_cast<Index>(b1) * grain;
                    if (finish > end) finish = end;

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

        void wait() {
            wait_done_spin_then_sleep();
        }

    private:
        static std::size_t mul_div(std::size_t a, std::size_t b, std::size_t d) {
#if defined(__SIZEOF_INT128__)
            return static_cast<std::size_t>((static_cast<__uint128_t>(a) * b) / d);
#else
            // Fallback: safe for typical tensor sizes; if you expect extreme sizes,
            // compile with a compiler that supports __int128 on 64-bit targets.
            return (a * b) / d;
#endif
        }
    };

    // -----------------------------
    // Typed job: static partition, grain-aligned boundaries
    // -----------------------------
    template <typename Index, typename F>
    struct Job final : JobBase {
        Index start;
        Index end;
        Index grain;
        F     func;

        Job(Index s, Index e, Index g, std::size_t parts, F&& f)
            // Initialize the latch with the participant count
            : JobBase(static_cast<std::ptrdiff_t>(std::max<std::size_t>(1, parts)))
            , start(s), end(e), grain(g), func(std::move(f)) {

            participants = std::max<std::size_t>(1, parts);
            blocks       = blocks_for(end - start, grain);
            // 'remaining' logic is removed; handled by JobBase::done_latch

            run = &Job::run_impl;
        }

        static void run_impl(JobBase* base, std::size_t pid) noexcept {
            static_cast<Job*>(base)->run_participant(pid);
        }

        void run_participant(std::size_t pid) noexcept {
            // ... (Copy existing implementation of run_participant here) ...
            // ... (It is unchanged, except it calls the updated finish_participant) ...

            // Just ensure you keep the existing body logic exactly as is.
            // The existing call to finish_participant() at the end is perfect.

            struct Guard {
                bool prev;
                Guard() : prev(tls_in_parallel_region_) { tls_in_parallel_region_ = true; }
                ~Guard() { tls_in_parallel_region_ = prev; }
            } guard;

            try {
                if (!cancelled.load(std::memory_order_relaxed)) {
                    const std::size_t P = participants;
                    const std::size_t B = blocks;

                    if (pid >= P || B == 0) {
                        finish_participant();
                        return;
                    }

                    const std::size_t b0 = mul_div(B, pid, P);
                    const std::size_t b1 = mul_div(B, pid + 1, P);

                    Index begin = start + static_cast<Index>(b0) * grain;
                    Index finish = start + static_cast<Index>(b1) * grain;
                    if (finish > end) finish = end;

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

        // wait_done_spin_then_sleep() is no longer needed in Job
        // We just use JobBase::wait() which wraps the latch.
    };

    // -----------------------------
    // Pool implementation
    // -----------------------------
    work_stealing_pool()
        : num_workers_(choose_default_workers()) {

        if (num_workers_ == 0) return;

        workers_.reserve(num_workers_);

        // Start worker threads and wait until they’re ready.
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
            DISPATCH_SPIN_PAUSE();
        }
    }

    ~work_stealing_pool() {
        stop_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(cv_mutex_);
        }
        cv_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void worker_loop(std::size_t worker_index) {
        // Start with epoch 0 (the initial value of job_epoch_) rather than loading.
        // This prevents a race where:
        //   1. Constructor waits for workers to call started.fetch_add()
        //   2. Constructor exits, parallel_for() increments epoch to 1
        //   3. Worker finally loads epoch = 1 as local_epoch
        //   4. Worker sees epoch == local_epoch, waits for next epoch
        //   5. Deadlock: worker misses the job, remaining never reaches 0
        std::uint64_t local_epoch = 0;

        // "Blocktime": stay hot briefly before sleeping.
        // This is *the* big win vs naive "sleep immediately" pools in tight loops.
        constexpr auto kBlockTime = std::chrono::microseconds(500);

        while (!stop_.load(std::memory_order_acquire)) {
            // 1) Wait for a new epoch (spin/yield for blocktime, then sleep).
            if (job_epoch_.load(std::memory_order_acquire) == local_epoch) {
                const auto start_wait = std::chrono::steady_clock::now();
                int spins = 0;

                while (!stop_.load(std::memory_order_relaxed) &&
                       job_epoch_.load(std::memory_order_acquire) == local_epoch) {

                    // Hot wait for blocktime:
                    if ((std::chrono::steady_clock::now() - start_wait) < kBlockTime) {
                        if ((++spins & 0xFF) == 0) {
                            std::this_thread::yield();
                        } else {
                            DISPATCH_SPIN_PAUSE();
                        }
                        continue;
                    }

                    // Sleep until epoch changes or stop.
                    std::unique_lock<std::mutex> lk(cv_mutex_);
                    cv_.wait(lk, [&] {
                        return stop_.load(std::memory_order_acquire) ||
                               job_epoch_.load(std::memory_order_acquire) != local_epoch;
                    });
                    break;
                }
            }

            if (stop_.load(std::memory_order_acquire)) break;

            // 2) Observe new job.
            const std::uint64_t e = job_epoch_.load(std::memory_order_acquire);
            if (e == local_epoch) continue;
            local_epoch = e;

            JobBase* job = current_job_.load(std::memory_order_acquire);
            if (!job) continue;

            // This worker participates only if it's within [0, participants-2].
            // The caller is participant (participants-1).
            const std::size_t workers_to_use = (job->participants > 0) ? (job->participants - 1) : 0;
            if (worker_index < workers_to_use) {
                job->run(job, worker_index);
            }
        }
    }

private:
    std::size_t num_workers_{0};
    std::vector<std::thread> workers_;

    std::atomic<bool> stop_{false};

    // Single-job broadcast state
    std::atomic<JobBase*>   current_job_{nullptr};
    std::atomic<std::uint64_t> job_epoch_{0};

    // Sleep/wake for idle workers
    std::mutex cv_mutex_;
    std::condition_variable cv_;

    // Serialize jobs (one at a time)
    std::mutex submit_mutex_;

    // TLS: worker id (>=0 if pool thread)
    inline static thread_local int  tls_worker_index_ = -1;
    // TLS: "am I currently executing inside a parallel_for region?"
    inline static thread_local bool tls_in_parallel_region_ = false;
};

} // namespace dispatch

#endif // DISPATCH_DISPATCH_WORK_STEALING_POOL_H
