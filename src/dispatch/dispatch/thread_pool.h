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
// thread_pool - Hybrid Direct Distribution
//
// Design: Workers use a two-phase wait:
//   1. Brief spin on generation counter (fast wake-up for small/medium work)
//   2. Fall back to condition variable (no interference for large work)
//
// This gives us the best of both worlds:
//   - Low latency for frequent small parallel_for calls
//   - No CPU burn / cache pollution for large workloads
// ============================================================================

class thread_pool {
    static constexpr size_t kMaxWorkers = 128;
    static constexpr int kSpinCount = 64;  // Brief spin before blocking

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

    // Condition variable for fallback blocking (reduces CPU burn on large workloads)
    std::mutex cv_mutex_;
    std::condition_variable cv_;

    void worker_loop(size_t my_idx) {
        tls_in_parallel_region_ = true; // Mark worker as "in a region"
        uint64_t last_gen = 0;

        while (!stop_.load(std::memory_order_relaxed)) {
            uint64_t gen;

            // Phase 1: Brief spin for fast wake-up
            int spins = 0;
            while ((gen = state_.generation.load(std::memory_order_acquire)) == last_gen) {
                if (stop_.load(std::memory_order_relaxed)) return;

                if (spins < kSpinCount) {
                    DISPATCH_PAUSE();
                    ++spins;
                } else {
                    // Phase 2: Fall back to condition variable (no CPU burn)
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
        stop_.store(true, std::memory_order_release);
        // Bump generation and notify to wake all workers
        // Note: Must hold cv_mutex_ while incrementing to prevent lost wakeups.
        {
            std::lock_guard lock(cv_mutex_);
            state_.generation.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    // -------------------------------------------------------------------------
    // parallel_for - Direct Work Distribution with Hybrid Wake
    // -------------------------------------------------------------------------
    template <typename Index, typename Func>
    void parallel_for(Index start, Index end, Func&& f) {
        // 1. Recursion Guard
        if (tls_in_parallel_region_) {
            f(start, end);
            return;
        }

        // 2. Serial fallback (existing logic)
        if (num_workers_ == 0 || (end - start) < 32768) {
            f(start, end);
            return;
        }

        // 3. Mark main thread as "in region" for the duration
        struct Guard {
            Guard() { tls_in_parallel_region_ = true; }
            ~Guard() { tls_in_parallel_region_ = false; }
        } guard;

        static_assert(std::is_integral_v<Index>, "Index must be integral");

        if (start >= end) return;

        const Index range = end - start;

        // Serial path for small work or no workers
        if (num_workers_ == 0 || range < 32768) {
            f(start, end);
            return;
        }

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

        // Set up shared state (relaxed stores, will be synchronized by generation release)
        state_.func.store(trampoline, std::memory_order_relaxed);
        state_.ctx.store(&func_copy, std::memory_order_relaxed);
        state_.done.store(&done, std::memory_order_relaxed);

        // Release barrier: make all stores visible, then bump generation
        // Note: Must hold cv_mutex_ while incrementing to prevent lost wakeups.
        // Without this, a worker checking the predicate (gen == last_gen) could
        // see the old value, then we increment + notify, then worker blocks forever.
        {
            std::lock_guard lock(cv_mutex_);
            state_.generation.fetch_add(1, std::memory_order_release);
        }

        // Wake any workers that fell asleep (for large workloads)
        cv_.notify_all();

        // Caller executes chunk 0 immediately
        Index caller_end = std::min(start + chunk_size, end);
        func_copy(start, caller_end);

        // Wait for workers to complete
        done.wait();

        // Note: No need to clear state - next parallel_for will overwrite before
        // bumping generation, and workers only read after seeing new generation.
    }

    // Convenience overload with grain_size parameter (ignored for static scheduling)
    template <typename Index, typename Func>
    void parallel_for(Index start, Index end, Index /*grain_size*/, Func&& f) {
        parallel_for(start, end, std::forward<Func>(f));
    }
};

} // namespace dispatch
