// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Ternary affine transformation: y = R @ x + T
//
// This example demonstrates structured-elementwise dispatch:
//   - Iteration shape: (N,) - loop over N points
//   - Element shapes: R is (3,3), T is (3,), x is (3,), y is (3,)
//
// Unlike unary_elementwise which applies a scalar function to each element,
// this operation applies a structured element operation (matrix-vector multiply
// plus vector add) at each iteration index.
//

#include "examples/affine_xform_ternary.h"
#include "examples/common.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/macros.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstdint>

namespace dispatch_examples {

using namespace dispatch;

//------------------------------------------------------------------------------
// Element operation: 3x3 matrix-vector multiply + vector add
//------------------------------------------------------------------------------
// Computes y = R @ x + t where:
//   R: 3x3 matrix (9 elements, row-major)
//   t: 3-vector (translation)
//   x: 3-vector (input position)
//   y: 3-vector (output position)
//
// This is the "structured element" operation - it processes elements with
// shape (3,3), (3,), (3,) rather than scalars.

template <typename T>
__hostdev__ void
affine_xform_element(T const *R, T const *t, T const *x, T *y) {
    // y[i] = sum_j(R[i,j] * x[j]) + t[i]
    DISPATCH_UNROLL
    for (int i = 0; i < 3; ++i) {
        T sum = t[i];
        DISPATCH_UNROLL
        for (int j = 0; j < 3; ++j) {
            sum += R[i * 3 + j] * x[j];
        }
        y[i] = sum;
    }
}

//------------------------------------------------------------------------------
// Accessor for structured elements with broadcast support
//------------------------------------------------------------------------------
// Provides pointer access to elements at a given iteration index, handling:
//   - Contiguous vs strided storage
//   - Broadcast on iteration dimension (size(0) == 1)
//
// Template parameters:
//   Stype      - torch::ScalarType
//   Contig     - contiguity enum
//   ElementRank - rank of element shape (1 for vectors, 2 for matrices)

template <torch::ScalarType Stype, contiguity Contig, int64_t ElementRank>
struct element_accessor;

// Contiguous specialization: direct pointer arithmetic
template <torch::ScalarType Stype, int64_t ElementRank>
struct element_accessor<Stype, contiguity::contiguous, ElementRank> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    value_type *data;
    int64_t element_size;  // product of element shape dimensions
    int64_t iter_size;     // size of iteration dimension (for broadcast check)

    static element_accessor
    from_tensor(torch::Tensor t) {
        element_accessor acc;
        acc.data      = t.data_ptr<value_type>();
        acc.iter_size = t.size(0);

        // Compute element size as product of dimensions after the first
        acc.element_size = 1;
        for (int64_t d = 1; d < t.dim(); ++d) {
            acc.element_size *= t.size(d);
        }
        return acc;
    }

    // Get pointer to element at iteration index n
    __hostdev__ value_type *
    at(int64_t n) const {
        // Handle broadcast: if iter_size == 1, always use index 0
        int64_t const idx = (iter_size == 1) ? 0 : n;
        return data + idx * element_size;
    }
};

// Strided specialization: use strides for addressing
template <torch::ScalarType Stype, int64_t ElementRank>
struct element_accessor<Stype, contiguity::strided, ElementRank> {
    using value_type = torch_scalar_cpp_type_t<Stype>;

    static constexpr int64_t total_rank = 1 + ElementRank; // iteration dim + element dims

    value_type *data;
    int64_t strides[total_rank];
    int64_t sizes[total_rank];

    static element_accessor
    from_tensor(torch::Tensor t) {
        element_accessor acc;
        acc.data = t.data_ptr<value_type>();
        for (int64_t d = 0; d < total_rank; ++d) {
            acc.strides[d] = t.stride(d);
            acc.sizes[d]   = t.size(d);
        }
        return acc;
    }

    // Get pointer to start of element at iteration index n
    // For strided tensors, this returns a pointer to the first value of the element
    // The caller must use inner strides to access within the element
    __hostdev__ value_type *
    at(int64_t n) const {
        // Handle broadcast: if sizes[0] == 1, always use index 0
        int64_t const idx = (sizes[0] == 1) ? 0 : n;
        return data + idx * strides[0];
    }

    // Get stride for element dimension d (0-indexed within element shape)
    __hostdev__ int64_t
    element_stride(int64_t d) const {
        return strides[1 + d];
    }
};

//------------------------------------------------------------------------------
// CUDA kernel (must be free function, not member)
//------------------------------------------------------------------------------

#if defined(__CUDACC__)
template <torch::ScalarType stype, contiguity contig>
__global__ void
affine_xform_cuda_kernel(element_accessor<stype, contig, 2> R_acc,
                         element_accessor<stype, contig, 1> T_acc,
                         element_accessor<stype, contig, 1> x_acc,
                         element_accessor<stype, contig, 1> out_acc,
                         int64_t N) {
    using T = torch_scalar_cpp_type_t<stype>;

    int64_t const n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    if constexpr (contig == contiguity::contiguous) {
        T const *R_ptr   = R_acc.at(n);
        T const *T_ptr   = T_acc.at(n);
        T const *x_ptr   = x_acc.at(n);
        T *out_ptr       = out_acc.at(n);

        affine_xform_element(R_ptr, T_ptr, x_ptr, out_ptr);
    } else {
        // Strided path: gather, compute, scatter
        T R_local[9], T_local[3], x_local[3], out_local[3];

        T const *R_base = R_acc.at(n);
        T const *T_base = T_acc.at(n);
        T const *x_base = x_acc.at(n);

        // Gather R (3x3 matrix)
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            DISPATCH_UNROLL
            for (int j = 0; j < 3; ++j) {
                R_local[i * 3 + j] =
                    R_base[i * R_acc.element_stride(0) + j * R_acc.element_stride(1)];
            }
        }

        // Gather T (3-vector)
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            T_local[i] = T_base[i * T_acc.element_stride(0)];
        }

        // Gather x (3-vector)
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            x_local[i] = x_base[i * x_acc.element_stride(0)];
        }

        // Compute
        affine_xform_element(R_local, T_local, x_local, out_local);

        // Scatter output
        T *out_base = out_acc.at(n);
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            out_base[i * out_acc.element_stride(0)] = out_local[i];
        }
    }
}
#endif // __CUDACC__

//------------------------------------------------------------------------------
// Affine transform operation struct for dispatch
//------------------------------------------------------------------------------

struct affine_xform_op_t {
    // CPU implementation
    template <torch::ScalarType stype, contiguity contig>
    static void
    op(tag<torch::kCPU, stype, contig>,
       torch::Tensor R_tensor,
       torch::Tensor T_tensor,
       torch::Tensor x_tensor,
       torch::Tensor out_tensor,
       int64_t N) {
        using T = torch_scalar_cpp_type_t<stype>;

        // Create accessors for each tensor
        // R has element rank 2 (3x3), T/x/out have element rank 1 (3,)
        auto R_acc   = element_accessor<stype, contig, 2>::from_tensor(R_tensor);
        auto T_acc   = element_accessor<stype, contig, 1>::from_tensor(T_tensor);
        auto x_acc   = element_accessor<stype, contig, 1>::from_tensor(x_tensor);
        auto out_acc = element_accessor<stype, contig, 1>::from_tensor(out_tensor);

        if constexpr (contig == contiguity::contiguous) {
            // Contiguous path: elements are densely packed
            for (int64_t n = 0; n < N; ++n) {
                T const *R_ptr   = R_acc.at(n);
                T const *T_ptr   = T_acc.at(n);
                T const *x_ptr   = x_acc.at(n);
                T *out_ptr       = out_acc.at(n);

                affine_xform_element(R_ptr, T_ptr, x_ptr, out_ptr);
            }
        } else {
            // Strided path: need to gather elements respecting strides
            for (int64_t n = 0; n < N; ++n) {
                // For strided tensors, we need to copy element data accounting for strides
                T R_local[9], T_local[3], x_local[3], out_local[3];

                T const *R_base = R_acc.at(n);
                T const *T_base = T_acc.at(n);
                T const *x_base = x_acc.at(n);

                // Gather R (3x3 matrix)
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        R_local[i * 3 + j] =
                            R_base[i * R_acc.element_stride(0) + j * R_acc.element_stride(1)];
                    }
                }

                // Gather T (3-vector)
                for (int i = 0; i < 3; ++i) {
                    T_local[i] = T_base[i * T_acc.element_stride(0)];
                }

                // Gather x (3-vector)
                for (int i = 0; i < 3; ++i) {
                    x_local[i] = x_base[i * x_acc.element_stride(0)];
                }

                // Compute
                affine_xform_element(R_local, T_local, x_local, out_local);

                // Scatter output
                T *out_base = out_acc.at(n);
                for (int i = 0; i < 3; ++i) {
                    out_base[i * out_acc.element_stride(0)] = out_local[i];
                }
            }
        }
    }

#if defined(__CUDACC__)
    // CUDA implementation
    template <torch::ScalarType stype, contiguity contig>
    static void
    op(tag<torch::kCUDA, stype, contig>,
       torch::Tensor R_tensor,
       torch::Tensor T_tensor,
       torch::Tensor x_tensor,
       torch::Tensor out_tensor,
       int64_t N) {
        c10::cuda::CUDAGuard device_guard(x_tensor.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        auto R_acc   = element_accessor<stype, contig, 2>::from_tensor(R_tensor);
        auto T_acc   = element_accessor<stype, contig, 1>::from_tensor(T_tensor);
        auto x_acc   = element_accessor<stype, contig, 1>::from_tensor(x_tensor);
        auto out_acc = element_accessor<stype, contig, 1>::from_tensor(out_tensor);

        int constexpr block_size = 256;
        int const num_blocks     = static_cast<int>((N + block_size - 1) / block_size);

        affine_xform_cuda_kernel<stype, contig><<<num_blocks, block_size, 0, stream>>>(
            R_acc, T_acc, x_acc, out_acc, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#endif // __CUDACC__

    using space      = axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using dispatcher = dispatch_table<space, void(torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, int64_t)>;
};

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

torch::Tensor
example_affine_xform(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    static affine_xform_op_t::dispatcher const table{
        affine_xform_op_t::dispatcher::from_op<affine_xform_op_t>(),
        affine_xform_op_t::space{}};

    // Validate tensor ranks
    TORCH_CHECK_VALUE(R.dim() == 3, "example_affine_xform: R must be 3D (N, 3, 3), got ", R.dim(), "D");
    TORCH_CHECK_VALUE(T.dim() == 2, "example_affine_xform: T must be 2D (N, 3), got ", T.dim(), "D");
    TORCH_CHECK_VALUE(x.dim() == 2, "example_affine_xform: x must be 2D (N, 3), got ", x.dim(), "D");

    // Validate element shapes
    TORCH_CHECK_VALUE(R.size(1) == 3 && R.size(2) == 3,
                      "example_affine_xform: R element shape must be (3, 3), got (",
                      R.size(1), ", ", R.size(2), ")");
    TORCH_CHECK_VALUE(T.size(1) == 3, "example_affine_xform: T element shape must be (3,), got (", T.size(1), ",)");
    TORCH_CHECK_VALUE(x.size(1) == 3, "example_affine_xform: x element shape must be (3,), got (", x.size(1), ",)");

    // Determine iteration dimension size N
    int64_t const N = x.size(0);

    // Validate broadcast compatibility on iteration dimension
    TORCH_CHECK_VALUE(R.size(0) == N || R.size(0) == 1,
                      "example_affine_xform: R iteration dim must be ", N, " or 1, got ", R.size(0));
    TORCH_CHECK_VALUE(T.size(0) == N || T.size(0) == 1,
                      "example_affine_xform: T iteration dim must be ", N, " or 1, got ", T.size(0));

    // Validate all tensors have same dtype
    TORCH_CHECK_VALUE(R.scalar_type() == T.scalar_type() && T.scalar_type() == x.scalar_type(),
                      "example_affine_xform: all tensors must have same dtype, got R=",
                      R.scalar_type(), ", T=", T.scalar_type(), ", x=", x.scalar_type());

    // Validate all tensors on same device
    TORCH_CHECK_VALUE(R.device() == T.device() && T.device() == x.device(),
                      "example_affine_xform: all tensors must be on same device");

    // Handle empty case
    if (N == 0) {
        return torch::empty({0, 3}, x.options());
    }

    // Allocate output
    auto output = torch::empty({N, 3}, x.options());

    // Dispatch
    auto const dev   = x.device().type();
    auto const stype = x.scalar_type();
    auto const contig = combined_contiguity(R, T, x, output);

    torch_dispatch("example_affine_xform", table, std::make_tuple(dev, stype, contig),
                   R, T, x, output, N);

    return output;
}

} // namespace dispatch_examples

/*
================================================================================
TODO: Design Pass for Generic Structured-Elementwise Abstraction
================================================================================

CONTEXT:
This file (affine_xform_ternary.cu) is a proof-of-concept for "structured
elementwise" operations - where we iterate over an outer dimension N and apply
a fixed computation on structured elements (matrices, vectors) rather than
scalars.

The goal is to extract a generic abstraction so that op-writers only need to
specify the pure element computation (like affine_xform_element), and the
infrastructure handles dispatch, iteration, gather/scatter, and broadcast.

--------------------------------------------------------------------------------
CURRENT STATE:
--------------------------------------------------------------------------------

The affine_xform_element function is the "task-specific" code:

    template <typename T>
    __hostdev__ void affine_xform_element(T const *R, T const *t, T const *x, T *y) {
        for (int i = 0; i < 3; ++i) {
            y[i] = t[i];
            for (int j = 0; j < 3; ++j)
                y[i] += R[i * 3 + j] * x[j];
        }
    }

It assumes contiguous memory. The infrastructure ensures this via:
- Contiguous path: accessor.at(n) returns pointer to contiguous element
- Strided path: gather into local buffer, compute, scatter back

--------------------------------------------------------------------------------
KEY ABSTRACTIONS TO BUILD:
--------------------------------------------------------------------------------

1. element_shape<Dims...>
   - Compile-time shape declaration
   - Enables constexpr buffer sizes for gather/scatter
   - Example: element_shape<3, 3> for a 3x3 matrix

2. element_view<T, Dims...>
   - Contiguous view passed to element function
   - operator()(i, j) for matrices, operator[](i) for vectors
   - Always contiguous - the op-writer never sees strides

3. gather<Dims...>(local, base, strides)
   - Recursive template to copy strided -> contiguous
   - Rank-1: for (i < D0) local[i] = base[i * stride[0]]
   - Rank-2: nested loops for matrices

4. scatter<Dims...>(base, strides, local)
   - Inverse of gather for output tensors
   - Only needed for strided outputs

5. element_accessor<Shape, Contig>
   - Combines iteration indexing + element access
   - Handles broadcast (size[0] == 1 -> always index 0)

--------------------------------------------------------------------------------
DESIRED OP-WRITER INTERFACE:
--------------------------------------------------------------------------------

    struct affine_xform_op {
        // Declare element shapes and input/output roles
        using R = element<shape<3, 3>, input>;
        using t = element<shape<3>, input>;
        using x = element<shape<3>, input>;
        using y = element<shape<3>, output>;

        // Pure computation on contiguous views
        template <typename T>
        __hostdev__ static void apply(
            R::view<T> R_, t::view<T> t_, x::view<T> x_, y::view<T> y_
        ) {
            for (int i = 0; i < 3; ++i) {
                y_[i] = t_[i];
                for (int j = 0; j < 3; ++j)
                    y_[i] += R_(i, j) * x_[j];
            }
        }
    };

    // Usage - infrastructure handles everything else
    torch::Tensor result = structured_elementwise<affine_xform_op>::map(R, t, x, "affine_xform");

--------------------------------------------------------------------------------
DESIGN QUESTIONS:
--------------------------------------------------------------------------------

1. Shape declaration syntax
   - Nested types in op struct? Separate traits? Constexpr tuple?

2. View type design
   - Generic view<T, Dims...> or named aliases (mat3x3<T>, vec3<T>)?

3. Output allocation
   - Infrastructure allocates from element shapes, or caller provides?

4. Validation generation
   - Auto-generate shape checks from declarations?
   - How to provide meaningful error messages with tensor names?

5. CUDA kernel generation
   - Kernel must be free function, how to reference op's apply()?
   - Template the kernel on Op type, call Op::apply() inside?

6. APL/J/K connection
   - This is the "rank operator" - element shapes define cell rank
   - Iteration happens over the "frame" (leading dimensions)
   - Broadcast is implicit frame agreement
   - Can we express this as a higher-order combinator?

--------------------------------------------------------------------------------
IMPLEMENTATION PLAN:
--------------------------------------------------------------------------------

1. Implement element_shape<Dims...> and element_view<T, Dims...>
2. Implement gather<Dims...> and scatter<Dims...> as recursive templates
3. Refactor affine_xform_ternary to use these primitives
4. Extract common iteration + dispatch into structured_elementwise<Op>
5. Validate by implementing a second op (e.g., batched dot product)

--------------------------------------------------------------------------------
NOTES:
--------------------------------------------------------------------------------

- Jaggedness is out of scope - assume valid tensor hyperrectangles
- Contiguity is a single boolean across all tensors (combined_contiguity)
- Broadcast handled by size[0] == 1 check in accessor
- Dispatch axes unchanged: device x stype x contiguity

================================================================================
*/
