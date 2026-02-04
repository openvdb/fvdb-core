// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ================================================================================================
// gelu_for_each.cu - GELU implementation using the for_each primitive
// ================================================================================================
//
// This example demonstrates using dispatch::for_each with flat views for element-wise
// operations. The implementation is optimized for contiguous GPU tensors using vectorized
// loads/stores (float4/double2) for improved memory throughput.
//
// Dispatch paths:
//   - CUDA contiguous: Vectorized loads (128-bit transactions) + scalar tail
//   - CUDA strided: Scalar access via offset computation
//   - CPU: Scalar access (no vectorization benefit on CPU for this op)
//
// ================================================================================================

#include "examples/gelu_for_each.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/thread_pool.h"
#include "dispatch/basic_thread_pool.h"
#include "dispatch/work_stealing_pool.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/views.h"
#include "examples/gelu_scalar.h"


#include <ATen/cpu/vec/vec.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#include "examples/gelu_vectorized.cuh"
#endif

namespace dispatch_examples {

using namespace dispatch;

//------------------------------------------------------------------------------
// CUDA contiguous: vectorized path (float4/double2)
//------------------------------------------------------------------------------
#ifdef __CUDACC__

template <torch::ScalarType stype>
void
gelu_cuda_contiguous(torch::Tensor input, torch::Tensor output) {
    using scalar_t = torch_scalar_cpp_type_t<stype>;
    using Tag      = tag<torch::kCUDA, stype>;

    scalar_t const *in_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr      = output.data_ptr<scalar_t>();
    int64_t const numel    = input.numel();

    // Vector configuration
    constexpr int64_t vec_size = vector_traits<scalar_t>::vector_size;
    int64_t const num_vecs     = numel / vec_size;
    int64_t const tail_start   = num_vecs * vec_size;
    int64_t const tail_count   = numel - tail_start;

    // Process vectorized portion
    if (num_vecs > 0) {
        for_each(Tag{}, num_vecs, [=] __device__(Tag, int64_t vec_idx) {
            int64_t const offset = vec_idx * vec_size;
            auto vec_in          = load_vector<scalar_t>(in_ptr + offset);
            auto vec_out         = gelu_vector<scalar_t>(vec_in);
            store_vector<scalar_t>(out_ptr + offset, vec_out);
        });
    }

    // Process tail (always less than vec_size elements)
    if (tail_count > 0) {
        for_each(Tag{}, tail_count, [=] __device__(Tag, int64_t i) {
            int64_t const idx = tail_start + i;
            out_ptr[idx]      = gelu_scalar(in_ptr[idx]);
        });
    }
}

#endif // __CUDACC__

//------------------------------------------------------------------------------
// Generic scalar path (strided CUDA, all CPU)
//------------------------------------------------------------------------------

struct gelu_op {
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig>, torch::Tensor input, torch::Tensor output) {
#ifdef __CUDACC__
        // Use vectorized path for contiguous CUDA float/double
        if constexpr (dev == torch::kCUDA && contig == contiguity::contiguous &&
                      (stype == torch::kFloat32 || stype == torch::kFloat64)) {
            gelu_cuda_contiguous<stype>(input, output);
            return;
        }
#endif

        // Scalar fallback for strided CUDA, half types, and all CPU
        using Tag       = tag<dev, stype>;
        using ConstView = flat_const_view<dev, stype>;
        using MutView   = flat_mutable_view<dev, stype>;

        ConstView in{input};
        MutView out{output};
        int64_t const numel = in.numel;

        for_each(
            Tag{}, numel, [=] __hostdev__(Tag, int64_t idx) { out[idx] = gelu_scalar(in[idx]); });
    }

    using space = axes<torch_full_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using subspaces  = coverage<space>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor)>;
};

// Internal implementation that operates on pre-allocated input/output
void
example_gelu_for_each_kernel(torch::Tensor input, torch::Tensor output) {
    static auto const table = dispatch_table_from_op<gelu_op>();

    if (input.numel() == 0) {
        return;
    }

    auto const dev            = input.device().type();
    auto const st             = input.scalar_type();
    auto const contig         = combined_contiguity(input, output);
    auto const dispatch_coord = std::make_tuple(dev, st, contig);

    torch_dispatch("example_gelu_for_each", table, dispatch_coord, input, output);
}

torch::Tensor
example_gelu_for_each_(torch::Tensor input) {
    example_gelu_for_each_kernel(input, input);
    return input;
}

//------------------------------------------------------------------------------
// ALTERNATE
//------------------------------------------------------------------------------

// -------------------------------------------------------------------------
// 1. Vector Alignment Traits (The "128-bit" Policy)
// -------------------------------------------------------------------------

template <typename T> struct VectorizedConfig {};

// Float: 128-bit = 4 elements
template <> struct VectorizedConfig<float> {
    using Storage                          = float4;
    static constexpr int N                 = 4;
    static constexpr const char *asm_load  = "ld.global.f32";
    static constexpr const char *asm_store = "st.global.f32";
};

// Double: 128-bit = 2 elements
template <> struct VectorizedConfig<double> {
    using Storage                          = double2;
    static constexpr int N                 = 2;
    static constexpr const char *asm_load  = "ld.global.f64";
    static constexpr const char *asm_store = "st.global.f64";
};

// Half: 128-bit = 8 elements
template <> struct VectorizedConfig<at::Half> {
    using Storage                          = uint4;           // Use raw bits for storage
    static constexpr int N                 = 8;
    static constexpr const char *asm_load  = "ld.global.u16"; // Load raw 16 bits
    static constexpr const char *asm_store = "st.global.u16";
};

// BFloat16: 128-bit = 8 elements
template <> struct VectorizedConfig<at::BFloat16> {
    using Storage                          = uint4;
    static constexpr int N                 = 8;
    static constexpr const char *asm_load  = "ld.global.u16";
    static constexpr const char *asm_store = "st.global.u16";
};

// -------------------------------------------------------------------------
// 2. Templated Predicated Load/Store
// -------------------------------------------------------------------------

template <typename T, typename Config>
__device__ __forceinline__ typename Config::Storage
masked_load(const T *ptr, int valid_count) {
    typename Config::Storage data;
    // Treat the storage as a flat array of T (or compatible raw bits)
    T *flat_data = reinterpret_cast<T *>(&data);

    // Zero-init (crucial for ensuring no garbage in padded lanes)
    // We use a simple memset-like approach for the vector type
    DISPATCH_UNROLL
    for (int i = 0; i < Config::N; ++i) {
        if constexpr (sizeof(T) == 2)
            *reinterpret_cast<uint16_t *>(&flat_data[i]) = 0;
        else if constexpr (sizeof(T) == 4)
            *reinterpret_cast<uint32_t *>(&flat_data[i]) = 0;
        else
            *reinterpret_cast<uint64_t *>(&flat_data[i]) = 0;
    }

    // Generate predicate mask
    unsigned mask = 0;
    DISPATCH_UNROLL
    for (int i = 0; i < Config::N; ++i) {
        if (valid_count > i)
            mask |= (1 << i);
    }

    // Unroll the predicated loads
    DISPATCH_UNROLL
    for (int i = 0; i < Config::N; ++i) {
        if (mask & (1 << i)) {
            // We use extended inline ASM to handle generic types safely
            if constexpr (sizeof(T) == 8) { // Double
                asm("ld.global.f64 %0, [%1];"
                    : "=d"(*reinterpret_cast<double *>(&flat_data[i]))
                    : "l"(ptr + i));
            } else if constexpr (sizeof(T) == 4) { // Float
                asm("ld.global.f32 %0, [%1];"
                    : "=f"(*reinterpret_cast<float *>(&flat_data[i]))
                    : "l"(ptr + i));
            } else { // Half/BF16 (16-bit)
                // We load into a u16 register, then cast to T
                uint16_t raw;
                asm("ld.global.u16 %0, [%1];" : "=h"(raw) : "l"(ptr + i));
                *reinterpret_cast<uint16_t *>(&flat_data[i]) = raw;
            }
        }
    }
    return data;
}

template <typename T, typename Config>
__device__ __forceinline__ void
masked_store(T *ptr, typename Config::Storage data, int valid_count) {
    T *flat_data  = reinterpret_cast<T *>(&data);
    unsigned mask = 0;
    DISPATCH_UNROLL
    for (int i = 0; i < Config::N; ++i) {
        if (valid_count > i)
            mask |= (1 << i);
    }

    DISPATCH_UNROLL
    for (int i = 0; i < Config::N; ++i) {
        if (mask & (1 << i)) {
            if constexpr (sizeof(T) == 8) {
                asm("st.global.f64 [%0], %1;" ::"l"(ptr + i),
                    "d"(*reinterpret_cast<double *>(&flat_data[i])));
            } else if constexpr (sizeof(T) == 4) {
                asm("st.global.f32 [%0], %1;" ::"l"(ptr + i),
                    "f"(*reinterpret_cast<float *>(&flat_data[i])));
            } else {
                uint16_t raw = *reinterpret_cast<uint16_t *>(&flat_data[i]);
                asm("st.global.u16 [%0], %1;" ::"l"(ptr + i), "h"(raw));
            }
        }
    }
}

// -------------------------------------------------------------------------
// 3. Generic Math Logic
// -------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T
generic_gelu_op(T x) {
    // Cast to float for calculation (except double)
    float val = static_cast<float>(x);

    constexpr float kAlpha = 0.79788456080286535587989211986876f;
    constexpr float kBeta  = 0.044715f;

    float x_sq     = val * val;
    float inner    = __fmaf_rn(kBeta, x_sq * val, val);
    float tanh_res = tanhf(kAlpha * inner);

    return static_cast<T>(0.5f * val * (1.0f + tanh_res));
}

// Specialization for Double to maintain precision
template <>
__device__ __forceinline__ double
generic_gelu_op<double>(double x) {
    constexpr double kAlpha = 0.79788456080286535587989211986876;
    constexpr double kBeta  = 0.044715;
    double x_sq             = x * x;
    double inner            = kBeta * x_sq * x + x; // fma is also available for double if needed
    double tanh_res         = tanh(kAlpha * inner);
    return 0.5 * x * (1.0 + tanh_res);
}

// -------------------------------------------------------------------------
// 4. The Kernel
// -------------------------------------------------------------------------

template <typename T>
__global__ void __launch_bounds__(256)
generic_gelu_kernel(const T *__restrict__ in, T *__restrict__ out, int64_t numel) {
    using Config    = VectorizedConfig<T>;
    using Storage   = typename Config::Storage;
    constexpr int N = Config::N; // Elements per thread (2, 4, or 8)

    // Index refers to the element index, but we jump by N
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * N;

    if (idx >= numel)
        return;

    int64_t remaining = numel - idx;

    Storage v;

    // --- Fast Path vs Tail Path ---
    if (__builtin_expect(remaining >= N, 1)) {
        // Optimized 128-bit aligned load
        v = reinterpret_cast<const Storage *>(in + idx)[0];
    } else {
        // Predicated Load
        v = masked_load<T, Config>(in + idx, remaining);
    }

    // --- Compute ---
    // We treat 'v' as an array of T elements
    T *elems = reinterpret_cast<T *>(&v);
    DISPATCH_UNROLL
    for (int i = 0; i < N; ++i) {
        elems[i] = generic_gelu_op(elems[i]);
    }

    // --- Store ---
    if (__builtin_expect(remaining >= N, 1)) {
        reinterpret_cast<Storage *>(out + idx)[0] = v;
    } else {
        masked_store<T, Config>(out + idx, v, remaining);
    }
}

// -------------------------------------------------------------------------
// 5. Host Dispatcher
// -------------------------------------------------------------------------

void
fast_gelu(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(out.is_cuda(), "Output must be CUDA");

    const int64_t numel = in.numel();
    if (numel == 0)
        return;

    // Use PyTorch's dispatcher to handle types
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in.scalar_type(), "fast_gelu", ([&] {
            using Config = VectorizedConfig<scalar_t>;

            const int block_size         = 256;
            const int elems_per_thread   = Config::N;
            const int elements_per_block = block_size * elems_per_thread;
            const int grid_size          = (numel + elements_per_block - 1) / elements_per_block;

            generic_gelu_kernel<scalar_t>
                <<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                    in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), numel);
        }));

    // Note: BFloat16 requires a separate dispatch macro in older PyTorch versions,
    // or AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, ...
}


// -------------------------------------------------------------------------
// 1. Vectorized Math Logic (Polynomial Approx)
// -------------------------------------------------------------------------
// We port the EXACT same math from CUDA to CPU SIMD.
// CPU Tanh is slow (libm call), so we MUST use the polynomial approximation
// or the Vectorized<T>::tanh() if available (which is often fast).
// Here we manually implement the polynomial for maximum control and parity.

template <typename T>
inline at::vec::Vectorized<T> gelu_simd_op(const at::vec::Vectorized<T>& x) {
    using Vec = at::vec::Vectorized<T>;

    // Constants broadcasted to vector registers
    // Note: Vec is not a literal type (uses SIMD intrinsics), so cannot be constexpr
    const Vec kAlpha(static_cast<T>(0.79788456080286535587989211986876));
    const Vec kBeta(static_cast<T>(0.044715));
    const Vec kOne(static_cast<T>(1.0));
    const Vec kHalf(static_cast<T>(0.5));

    // x^3 term:  inner = 0.044715 * x^3 + x
    // Fused Multiply-Add (FMA) is crucial for speed here
    Vec x_sq = x * x;
    Vec inner = at::vec::fmadd(kBeta, x_sq * x, x);

    // tanh(sqrt(2/pi) * inner)
    // Note: PyTorch's Vectorized class usually provides a fast, vectorized tanh approximation
    Vec tanh_res = (kAlpha * inner).tanh();

    // Final result: 0.5 * x * (1.0 + tanh_res)
    return kHalf * x * (kOne + tanh_res);
}

// -------------------------------------------------------------------------
// 2. The Kernel
// -------------------------------------------------------------------------

template <typename scalar_t>
void gelu_cpu_kernel(const scalar_t* in, scalar_t* out, int64_t numel) {
    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_len = Vec::size(); // e.g., 8 for AVX2 float, 16 for AVX512

    // 1. Threading Strategy
    // at::parallel_for manages the thread pool (OpenMP or TBB).
    // GRAIN_SIZE is a heuristic: don't spawn a thread for less than ~2048 elements
    // to avoid overhead swamping the gain.
    int64_t grain_size = 2048;

    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
        int64_t i = begin;

        // 2. Main Vector Loop
        // Process as many full vectors as possible within this thread's chunk
        // We ensure we don't read past 'end' with the vectorized step.
        int64_t limit = end - (end - begin) % vec_len;

        for (; i < limit; i += vec_len) {
            // Load full vector (unaligned load usually safe/fast on modern x86)
            Vec data = Vec::loadu(in + i);

            // Compute
            data = gelu_simd_op(data);

            // Store
            data.store(out + i);
        }

        // 3. Tail Handling (The "Clean" CPU Way)
        // CPU vector libraries often support masked load/store for tails automatically.
        // If not, we fall back to a scalar loop.
        // PyTorch's Vectorized class handles 'loadu(ptr, count)' efficiently.
        if (i < end) {
            int64_t remaining = end - i;

            // Masked Load: only loads 'remaining' elements, pads rest with 0
            Vec data = Vec::loadu(in + i, remaining);

            data = gelu_simd_op(data);

            // Masked Store: only writes 'remaining' elements
            data.store(out + i, remaining);
        }
    });
}

// -------------------------------------------------------------------------
// 3. Host Dispatcher
// -------------------------------------------------------------------------

void fast_gelu_cpu_old(torch::Tensor in, torch::Tensor out) {
    // Basic checks
    TORCH_CHECK(!in.is_cuda(), "Input must be a CPU tensor");
    TORCH_CHECK(!out.is_cuda(), "Output must be a CPU tensor");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "Tensors must be contiguous");

    const int64_t numel = in.numel();
    if (numel == 0) return;

    // Dispatcher handles float, double, etc.
    // Note: at::vec::Vectorized supports Half/BFloat16 on modern builds,
    // but usually requires specific compiler flags (AVX512_BF16 etc).
    // For standard safety, we usually dispatch float/double.
    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "fast_gelu_cpu", ([&] {
        gelu_cpu_kernel<scalar_t>(
            in.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel
        );
    }));
}


// // Heuristic for element-wise operations (add, mul, gelu, relu)
// inline int64_t
// calc_optimal_grain_size(int64_t numel) {
//     // 1. Minimum Threshold:
//     // Don't create tiny tasks where overhead > payload.
//     // Conservative value would be 32768, but for bandwidth-bound ops like GELU,
//     // lower values allow better thread utilization at mid-sized tensors.
//     //constexpr int64_t MIN_GRAIN_SIZE = 4096; // (conservative: 32768)
//     constexpr int64_t MIN_GRAIN_SIZE = 64;

//     // 2. Concurrency Target:
//     // How many threads do we actually have?
//     const auto num_threads = dispatch::thread_pool::instance().num_threads();

//     // 3. Static Scheduling:
//     // Use 1 chunk per thread (like OpenMP's 'static' schedule) to minimize
//     // synchronization overhead. For pure "for_each", this maximizes throughput.
//     // Calculate ideal grain size to split work evenly among threads.
//     const int64_t grain_size = numel / static_cast<int64_t>(num_threads);

//     // Clamp to the minimum to ensure we don't create tiny tasks
//     return std::max(MIN_GRAIN_SIZE, grain_size);
// }


void fast_gelu_cpu_simd(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(!in.is_cuda(), "Input must be a CPU tensor");
    TORCH_CHECK(!out.is_cuda(), "Output must be a CPU tensor");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "Tensors must be contiguous");

    const int64_t numel = in.numel();
    if (numel == 0) return;

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "fast_gelu_cpu", [&] {
        using Vec = at::vec::Vectorized<scalar_t>;
        constexpr int64_t vec_len = Vec::size();

        const scalar_t* __restrict__ in_ptr  = in.data_ptr<scalar_t>();
        scalar_t* __restrict__       out_ptr = out.data_ptr<scalar_t>();

        // Work-stealing grain size tuning:
        // - Small enough for good load balancing (more steal opportunities)
        // - Large enough for cache efficiency (~4-8KB per task fits in L1)
        // - Multiple of vector width for clean SIMD boundaries
        // 1024 floats = 4KB, good balance for work-stealing overhead
        //constexpr int64_t grain_size = 1024;

        dispatch::thread_pool::instance().parallel_for(
            int64_t{0}, numel,
            [in_ptr, out_ptr, vec_len](int64_t begin, int64_t end) {
                // Process full vectors
                int64_t i     = begin;
                int64_t limit = end - (end - begin) % vec_len;

                // Main SIMD loop
                for (; i + vec_len <= limit; i += vec_len) {
                    Vec data0 = Vec::loadu(in_ptr + i);
                    data0 = gelu_simd_op(data0);
                    data0.store(out_ptr + i);
                }

                // Tail elements
                if (i < end) {
                    int64_t remaining = end - i;
                    Vec     data      = Vec::loadu(in_ptr + i, remaining);
                    data              = gelu_simd_op(data);
                    data.store(out_ptr + i, remaining);
                }
            });
    });
}

// =============================================================================
// fast_tanh: Polynomial approximation (avoids slow libm tanh)
// Uses Padé [7,6] rational approximation, accurate to ~1e-5 for |x| < 5
// =============================================================================

inline float fast_tanh(float x) {
    // Clamp to asymptotic region
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;

    float x2 = x * x;
    // Padé rational approximation coefficients
    float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return num / den;
}

inline double fast_tanh(double x) {
    if (x > 5.0) return 1.0;
    if (x < -5.0) return -1.0;

    double x2 = x * x;
    double num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    double den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    return num / den;
}

// =============================================================================
// fast_gelu_host: Host-side scalar GELU for benchmarking
// =============================================================================

inline float fast_gelu_host(float x) {
    constexpr float kAlpha = 0.79788456080286535587989211986876f;
    constexpr float kBeta  = 0.044715f;

    float x_sq     = x * x;
    float inner    = std::fma(kBeta, x_sq * x, x);
    float tanh_res = fast_tanh(kAlpha * inner);

    return 0.5f * x * (1.0f + tanh_res);
}

inline double fast_gelu_host(double x) {
    constexpr double kAlpha = 0.79788456080286535587989211986876;
    constexpr double kBeta  = 0.044715;

    double x_sq     = x * x;
    double inner    = std::fma(kBeta, x_sq * x, x);
    double tanh_res = fast_tanh(kAlpha * inner);

    return 0.5 * x * (1.0 + tanh_res);
}

// Half types: compute in float, cast back
inline at::Half fast_gelu_host(at::Half x) {
    return static_cast<at::Half>(fast_gelu_host(static_cast<float>(x)));
}

inline at::BFloat16 fast_gelu_host(at::BFloat16 x) {
    return static_cast<at::BFloat16>(fast_gelu_host(static_cast<float>(x)));
}


void fast_gelu_cpu(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(!in.is_cuda(), "Input must be a CPU tensor");
    TORCH_CHECK(!out.is_cuda(), "Output must be a CPU tensor");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "Tensors must be contiguous");

    const int64_t numel = in.numel();
    if (numel == 0) return;

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "fast_gelu_cpu", [&] {

        const scalar_t* __restrict__ in_ptr  = in.data_ptr<scalar_t>();
        scalar_t* __restrict__       out_ptr = out.data_ptr<scalar_t>();

        // Work-stealing grain size tuning:
        // - Small enough for good load balancing (more steal opportunities)
        // - Large enough for cache efficiency (~4-8KB per task fits in L1)
        // - Multiple of vector width for clean SIMD boundaries
        // 1024 floats = 4KB, good balance for work-stealing overhead
        //constexpr int64_t grain_size = 1024;

        //dispatch::work_stealing_pool::instance().parallel_for(
        //dispatch::thread_pool::instance().parallel_for(
        dispatch::basic_thread_pool::instance().parallel_for(
            int64_t{0}, numel,
            [in_ptr, out_ptr](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; i++) {
                    out_ptr[i] = fast_gelu_host(in_ptr[i]);
                }
            });
    });
}

torch::Tensor
example_gelu_for_each(torch::Tensor input) {
    if (input.numel() == 0) {
        return torch::empty_like(input);
    }
    auto output = torch::empty_like(input);

    if (input.device().type() == torch::kCUDA && input.is_contiguous() && output.is_contiguous()) {
        fast_gelu(input, output);
    } else if (input.device().type() == torch::kCPU && input.is_contiguous() && output.is_contiguous()) {
        fast_gelu_cpu(input, output);
    } else {
        example_gelu_for_each_kernel(input, output);
    }
    return output;
}

void
example_gelu_for_each_out(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.sizes() == output.sizes(),
                "example_gelu_for_each_out: input and output must have the same shape");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                "example_gelu_for_each_out: input and output must have the same dtype");
    TORCH_CHECK(input.device() == output.device(),
                "example_gelu_for_each_out: input and output must be on the same device");
    if (input.device().type() == torch::kCUDA && input.is_contiguous() && output.is_contiguous()) {
        fast_gelu(input, output);
    } else if (input.device().type() == torch::kCPU && input.is_contiguous() && output.is_contiguous()) {
        fast_gelu_cpu(input, output);
    } else {
        example_gelu_for_each_kernel(input, output);
    }
}

} // namespace dispatch_examples
