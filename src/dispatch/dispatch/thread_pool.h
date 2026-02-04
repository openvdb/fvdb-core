#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define DISPATCH_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define DISPATCH_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define DISPATCH_PAUSE() __asm__ __volatile__("yield")
#else
#define DISPATCH_PAUSE() std::this_thread::yield()
#endif

namespace dispatch {

// ============================================================================
// thread_pool - Static Distribution with Spin-Wait
//
// Design: Workers spin on a generation counter, ready for instant wake-up.
// Falls back to condition variable only after extended idle periods.
//
// Key optimizations:
//   - No mutex on the hot path (just atomic generation bump)
//   - Workers spin aggressively (~10k iterations before sleeping)
//   - Static partitioning (one chunk per thread, no queuing overhead)
//   - Caller participates (N+1 threads for N workers)
// ============================================================================

class thread_pool {
    static constexpr size_t kMaxWorkers = 128;

    // Spin count before falling back to CV sleep.
    // At ~10ns per pause, 10000 spins = ~100us of hot waiting.
    // This covers typical benchmark inter-call gaps.
    static constexpr int kSpinCount = 10000;

    // TLS: Prevent recursion / nested parallel_for
    inline static thread_local bool tls_in_parallel_region_ = false;

    // Type-erased function pointer for the current parallel_for
    using RangeFunc = void (*)(int64_t begin, int64_t end, void* ctx);

    // Per-worker slot (cache-line aligned to prevent false sharing)
    struct alignas(64) WorkerSlot {
        std::atomic<int64_t> begin{0};
        std::atomic<int64_t> end{0};
    };

    // Shared state for current parallel_for (all fields atomic for safety)
    struct alignas(64) SharedState {
        std::atomic<uint64_t> generation{0};
        std::atomic<RangeFunc> func{nullptr};
        std::atomic<void*> ctx{nullptr};
        std::atomic<std::latch*> done{nullptr};
    };

    SharedState state_;
    WorkerSlot slots_[kMaxWorkers];
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    size_t num_workers_{0};

    // Condition variable for fallback blocking (only used after extended idle)
    std::mutex cv_mutex_;
    std::condition_variable cv_;

    void worker_loop(size_t my_idx) {
        tls_in_parallel_region_ = true; // Mark worker as "in a region"
        uint64_t last_gen = 0;

        while (!stop_.load(std::memory_order_relaxed)) {
            uint64_t gen;

            // Phase 1: Aggressive spin for fast wake-up
            int spins = 0;
            while ((gen = state_.generation.load(std::memory_order_acquire)) == last_gen) {
                if (stop_.load(std::memory_order_relaxed)) return;

                if (spins < kSpinCount) {
                    DISPATCH_PAUSE();
                    ++spins;
                } else {
                    // Phase 2: Fall back to condition variable
                    std::unique_lock lock(cv_mutex_);
                    cv_.wait(lock, [&] {
                        gen = state_.generation.load(std::memory_order_acquire);
                        return gen != last_gen || stop_.load(std::memory_order_relaxed);
                    });
                    break;
                }
            }

            if (stop_.load(std::memory_order_relaxed)) return;
            last_gen = gen;

            // Read my assigned range (relaxed OK, synchronized by generation acquire)
            int64_t b = slots_[my_idx].begin.load(std::memory_order_relaxed);
            int64_t e = slots_[my_idx].end.load(std::memory_order_relaxed);

            // Read shared state (relaxed OK, synchronized by generation acquire)
            RangeFunc func = state_.func.load(std::memory_order_relaxed);
            void* ctx = state_.ctx.load(std::memory_order_relaxed);
            std::latch* done = state_.done.load(std::memory_order_relaxed);

            // Execute if valid range
            if (b < e && func) {
                func(b, e, ctx);
            }

            // Signal completion
            if (done) {
                done->count_down();
            }
        }
    }

public:
    static thread_pool& instance() {
        static thread_pool pool;
        return pool;
    }

    size_t num_threads() const noexcept {
        return num_workers_;
    }

    thread_pool() {
        size_t hw = std::thread::hardware_concurrency();
        num_workers_ = (hw > 1) ? (hw - 1) : 0;
        if (num_workers_ > kMaxWorkers) num_workers_ = kMaxWorkers;

        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~thread_pool() {
        // Signal stop
        stop_.store(true, std::memory_order_release);

        // Wake up everyone with the lock held
        {
            std::lock_guard<std::mutex> lock(cv_mutex_);
            state_.generation.fetch_add(1, std::memory_order_release);
            cv_.notify_all();
        }

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    // -------------------------------------------------------------------------
    // parallel_for - Direct Work Distribution with Spin Wake
    // -------------------------------------------------------------------------
    template <typename Index, typename Func>
    void parallel_for(Index start, Index end, Func&& f) {
        static_assert(std::is_integral_v<Index>, "Index must be integral");

        // 1. Recursion Guard - run nested parallel_for serially
        if (tls_in_parallel_region_) {
            f(start, end);
            return;
        }

        // 2. Early exit for empty range
        if (start >= end) return;

        const Index range = end - start;

        // 3. Serial fallback for small work or no workers
        // Threshold tuned for typical SIMD workloads (~2K elements is break-even)
        if (num_workers_ == 0 || range < 2048) {
            f(start, end);
            return;
        }

        // 4. Mark main thread as "in region" for the duration
        struct Guard {
            Guard() { tls_in_parallel_region_ = true; }
            ~Guard() { tls_in_parallel_region_ = false; }
        } guard;

        // Static partitioning: one chunk per thread
        const size_t total_threads = num_workers_ + 1;
        const Index chunk_size = (range + static_cast<Index>(total_threads) - 1)
                                / static_cast<Index>(total_threads);

        std::latch done(static_cast<std::ptrdiff_t>(num_workers_));

        // Type-erase the functor via a trampoline
        using FuncDecay = std::decay_t<Func>;
        FuncDecay func_copy = std::forward<Func>(f);

        auto trampoline = [](int64_t b, int64_t e, void* ctx) {
            (*static_cast<FuncDecay*>(ctx))(static_cast<Index>(b), static_cast<Index>(e));
        };

        // Assign ranges to workers (workers get chunks 1..N, caller gets chunk 0)
        for (size_t i = 0; i < num_workers_; ++i) {
            Index worker_start = start + static_cast<Index>(i + 1) * chunk_size;
            Index worker_end = std::min(worker_start + chunk_size, end);

            slots_[i].begin.store(static_cast<int64_t>(worker_start), std::memory_order_relaxed);
            slots_[i].end.store(static_cast<int64_t>(worker_end), std::memory_order_relaxed);
        }

        // Set up shared state (relaxed stores, synchronized by generation release below)
        state_.func.store(trampoline, std::memory_order_relaxed);
        state_.ctx.store(&func_copy, std::memory_order_relaxed);
        state_.done.store(&done, std::memory_order_relaxed);

        // --------------------------------------------------------------------
        // CRITICAL FIX: Hold lock during update and notify
        // --------------------------------------------------------------------
        {
            std::lock_guard<std::mutex> lock(cv_mutex_);

            // Now we bump generation. If a worker is currently inside cv_.wait(),
            // it holds the lock (or is trying to), so we are synchronized.
            state_.generation.fetch_add(1, std::memory_order_release);

            cv_.notify_all();
        }
        // --------------------------------------------------------------------

        // Caller executes chunk 0 immediately
        Index caller_end = std::min(start + chunk_size, end);
        func_copy(start, caller_end);

        // Wait for workers to complete
        done.wait();
    }

    // Convenience overload with grain_size parameter (ignored for static scheduling)
    template <typename Index, typename Func>
    void parallel_for(Index start, Index end, Index /*grain_size*/, Func&& f) {
        parallel_for(start, end, std::forward<Func>(f));
    }
};

} // namespace dispatch
