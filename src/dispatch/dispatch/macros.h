// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Common macros for host/device portability and CUDA-specific features.
//
#ifndef DISPATCH_DISPATCH_MACROS_H
#define DISPATCH_DISPATCH_MACROS_H

//------------------------------------------------------------------------------
// __hostdev__ macro for host/device portability
//------------------------------------------------------------------------------
// Marks functions as callable from both host and device code when compiling
// with CUDA, or as regular functions when compiling without CUDA.

#ifndef __hostdev__
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif
#endif

//------------------------------------------------------------------------------
// DISPATCH_UNROLL macro for portable loop unrolling
//------------------------------------------------------------------------------
// #pragma unroll is CUDA-specific; gcc warns about unknown pragmas.
// Use DISPATCH_UNROLL before a loop to request unrolling in CUDA code.
//
// Note: We use __CUDA_ARCH__ (not __CUDACC__) because __CUDACC__ is defined
// during both host and device compilation passes when using nvcc, but
// __CUDA_ARCH__ is only defined during device code compilation.

#ifndef DISPATCH_UNROLL
#if defined(__CUDA_ARCH__)
#define DISPATCH_UNROLL _Pragma("unroll")
#else
#define DISPATCH_UNROLL
#endif
#endif

#endif // DISPATCH_DISPATCH_MACROS_H
