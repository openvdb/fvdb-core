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
// Uses the for_each machinery to eliminate host/device code duplication.
// The gather/scatter pattern handles both contiguous and strided tensors.
//

#include "examples/affine_xform_ternary.h"

#include "dispatch/dispatch_table.h"
#include "dispatch/macros.h"
#include "dispatch/torch/accessors.h"
#include "dispatch/torch/dispatch.h"
#include "dispatch/torch/for_each.h"
#include "dispatch/torch/types.h"
#include "dispatch/types.h"
#include "examples/common.h"

#include <array>
#include <cstdint>

namespace dispatch_examples {

using namespace dispatch;

//------------------------------------------------------------------------------
// Gatherer: encapsulates input accessors
//------------------------------------------------------------------------------
// Holds accessors for R, t, x and provides gather(tag, n) to produce input element.
// Tag-based dispatch handles contiguous vs strided specialization.

template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig> struct affine_gatherer;

// Contiguous specialization
template <torch::DeviceType dev, torch::ScalarType stype>
struct affine_gatherer<dev, stype, contiguity::contiguous> {
    using T        = torch_scalar_cpp_type_t<stype>;
    using R_acc_t  = typename element_accessor<2>::template contiguous<stype>;
    using Tx_acc_t = typename element_accessor<1>::template contiguous<stype>;

    struct element {
        T const *R; // 9 elements, row-major
        T const *t; // 3 elements
        T const *x; // 3 elements
    };

    R_acc_t R_acc;
    Tx_acc_t t_acc;
    Tx_acc_t x_acc;

    __hostdev__ element
    get(int64_t n) const {
        return {R_acc.at(n), t_acc.at(n), x_acc.at(n)};
    }
};

// Strided specialization
template <torch::DeviceType dev, torch::ScalarType stype>
struct affine_gatherer<dev, stype, contiguity::strided> {
    using T        = torch_scalar_cpp_type_t<stype>;
    using R_acc_t  = typename element_accessor<2>::template strided<stype>;
    using Tx_acc_t = typename element_accessor<1>::template strided<stype>;

    struct element {
        T R[9]; // gathered into contiguous local
        T t[3];
        T x[3];
    };

    R_acc_t R_acc;
    Tx_acc_t t_acc;
    Tx_acc_t x_acc;

    __hostdev__ element
    get(int64_t n) const {
        element elem;

        // Gather R (3x3 matrix)
        T const *R_base = R_acc.at(n);
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            DISPATCH_UNROLL
            for (int j = 0; j < 3; ++j) {
                elem.R[i * 3 + j] =
                    R_base[i * R_acc.element_stride(0) + j * R_acc.element_stride(1)];
            }
        }

        // Gather t (3-vector)
        T const *t_base = t_acc.at(n);
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            elem.t[i] = t_base[i * t_acc.element_stride(0)];
        }

        // Gather x (3-vector)
        T const *x_base = x_acc.at(n);
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            elem.x[i] = x_base[i * x_acc.element_stride(0)];
        }

        return elem;
    }
};

// Factory: create gatherer from tensors (contiguous)
template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
auto
make_gatherer(tag<dev, stype, contig> t, torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    return affine_gatherer<dev, stype, contig>{element_accessor<2>::from_tensor(t, R),
                                               element_accessor<1>::from_tensor(t, T),
                                               element_accessor<1>::from_tensor(t, x)};
}

//------------------------------------------------------------------------------
// Scatterer: encapsulates output accessor
//------------------------------------------------------------------------------
template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
struct affine_scatterer;

template <torch::DeviceType dev, torch::ScalarType stype>
struct affine_scatterer<dev, stype, contiguity::contiguous> {
    using T     = torch_scalar_cpp_type_t<stype>;
    using acc_t = typename element_accessor<1>::template contiguous<stype>;

    acc_t out_acc;

    struct element {
        T *y; // 3 elements
    };

    __hostdev__ element
    at(int64_t n) const {
        return {out_acc.at(n)};
    }

    __hostdev__ void
    set(int64_t n, element const &elem) const {
        // Nothing, pointer was written directly to memory
    }
};

template <torch::DeviceType dev, torch::ScalarType stype>
struct affine_scatterer<dev, stype, contiguity::strided> {
    using T     = torch_scalar_cpp_type_t<stype>;
    using acc_t = typename element_accessor<1>::template strided<stype>;

    acc_t out_acc;

    struct element {
        T y[3];
    };

    __hostdev__ element
    at(int64_t n) const {
        return element{};
    }

    __hostdev__ void
    set(int64_t n, element const &elem) const {
        T *base = out_acc.at(n);
        DISPATCH_UNROLL
        for (int i = 0; i < 3; ++i) {
            base[i * out_acc.element_stride(0)] = elem.y[i];
        }
    }
};

// Factory: create scatterer from tensor
template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
auto
make_scatterer(tag<dev, stype, contig> t, torch::Tensor out) {
    return affine_scatterer<dev, stype, contig>{element_accessor<1>::from_tensor(t, out)};
}

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

template <typename Tag, typename T>
    requires tag_like<Tag>
__hostdev__ void
affine_xform_element(Tag, T const *R, T const *t, T const *x, T *y) {
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
// Affine transform operation struct for dispatch
//------------------------------------------------------------------------------
// Uses for_each to provide unified CPU/CUDA implementation.
// The lambda handles gather/compute/scatter for both contiguous and strided.

struct affine_xform {
    //--------------------------------------------------------------------------
    // Unified op: works for all device/contiguity combinations
    //--------------------------------------------------------------------------
    template <torch::DeviceType dev, torch::ScalarType stype, contiguity contig>
    static void
    op(tag<dev, stype, contig> t,
       torch::Tensor R_tensor,
       torch::Tensor T_tensor,
       torch::Tensor x_tensor,
       torch::Tensor out_tensor) {
        auto guard   = make_device_guard(t, R_tensor);
        auto gather  = make_gatherer(t, R_tensor, T_tensor, x_tensor);
        auto scatter = make_scatterer(t, out_tensor);

        int64_t const N = x_tensor.size(0);

        for_each(
            t, N, [t, gather, scatter] __hostdev__(tag<dev, stype, contig>, int64_t n) mutable {
                auto in  = gather.get(n);
                auto out = scatter.at(n);
                affine_xform_element(t, in.R, in.t, in.x, out.y);
                scatter.set(n, out);
            });
    }

    using space =
        axes<torch_cpu_cuda_device_axis, torch_full_float_stype_axis, full_contiguity_axis>;
    using dispatcher =
        dispatch_table<space, void(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)>;
};

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

torch::Tensor
example_affine_xform(torch::Tensor R, torch::Tensor T, torch::Tensor x) {
    static affine_xform::dispatcher const table{affine_xform::dispatcher::from_op<affine_xform>(),
                                                affine_xform::space{}};

    // Validate tensor ranks
    TORCH_CHECK_VALUE(
        R.dim() == 3, "example_affine_xform: R must be 3D (N, 3, 3), got ", R.dim(), "D");
    TORCH_CHECK_VALUE(
        T.dim() == 2, "example_affine_xform: T must be 2D (N, 3), got ", T.dim(), "D");
    TORCH_CHECK_VALUE(
        x.dim() == 2, "example_affine_xform: x must be 2D (N, 3), got ", x.dim(), "D");

    // Validate element shapes
    TORCH_CHECK_VALUE(R.size(1) == 3 && R.size(2) == 3,
                      "example_affine_xform: R element shape must be (3, 3), got (",
                      R.size(1),
                      ", ",
                      R.size(2),
                      ")");
    TORCH_CHECK_VALUE(T.size(1) == 3,
                      "example_affine_xform: T element shape must be (3,), got (",
                      T.size(1),
                      ",)");
    TORCH_CHECK_VALUE(x.size(1) == 3,
                      "example_affine_xform: x element shape must be (3,), got (",
                      x.size(1),
                      ",)");

    // Determine iteration dimension size N
    int64_t const N = x.size(0);

    // Validate broadcast compatibility on iteration dimension
    TORCH_CHECK_VALUE(R.size(0) == N || R.size(0) == 1,
                      "example_affine_xform: R iteration dim must be ",
                      N,
                      " or 1, got ",
                      R.size(0));
    TORCH_CHECK_VALUE(T.size(0) == N || T.size(0) == 1,
                      "example_affine_xform: T iteration dim must be ",
                      N,
                      " or 1, got ",
                      T.size(0));

    // Validate all tensors have same dtype
    TORCH_CHECK_VALUE(R.scalar_type() == T.scalar_type() && T.scalar_type() == x.scalar_type(),
                      "example_affine_xform: all tensors must have same dtype, got R=",
                      R.scalar_type(),
                      ", T=",
                      T.scalar_type(),
                      ", x=",
                      x.scalar_type());

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
    auto const dev    = x.device().type();
    auto const stype  = x.scalar_type();
    auto const contig = combined_contiguity(R, T, x, output);

    torch_dispatch(
        "example_affine_xform", table, std::make_tuple(dev, stype, contig), R, T, x, output);

    return output;
}

} // namespace dispatch_examples
