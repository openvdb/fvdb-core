// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_H
#define FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_H

#include <nanovdb/NanoVDB.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

void mainSparseConvolutionIGEMM(const std::vector<nanovdb::Coord> &inputPoints,
                                const std::vector<nanovdb::Coord> &outputPoints,
                                uint32_t benchmark_iters);

int test_sparse_convolution_igemm_nanovdb_cuda(int benchmark_iters = 10);

/// Non-template wrapper around the Sifakis reference scatter-gather convolution
/// (SparseConvolveCudaReference) with hardcoded IGEMM_Geometry (3x3x3, C=64, K=128).
///
/// All pointer arguments must reside in device memory.
/// @param outputLeafCount  Number of leaf nodes in the output grid.
/// @param inputGrid        NanoVDB ValueOnIndex grid for the input (device pointer).
/// @param outputGrid       NanoVDB ValueOnIndex grid for the output (device pointer).
/// @param d_filterData     Filter weights in Sifakis layout [3][3][3][128][64].
/// @param d_inputData      Input features in [valueCount][64] layout (index 0 = background).
/// @param d_outputData     Output features in [valueCount][128] layout (written by kernel).
/// @param stream           CUDA stream for the kernel launch.
void sifakisRefSparseConv(uint32_t outputLeafCount,
                          const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                          const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                          const float *d_filterData,
                          const float *d_inputData,
                          float *d_outputData,
                          cudaStream_t stream = 0);

/// Non-template wrapper around the CUTLASS implicit-GEMM sparse convolution
/// kernel (AmperePredicatedFprop) with hardcoded IGEMM_Geometry (3x3x3, C=64, K=128).
///
/// Same interface as sifakisRefSparseConv but launches the optimized IGEMM
/// instead of the naive reference.  All pointer arguments must reside in
/// device memory.
void sifakisIGemmConv(uint32_t outputLeafCount,
                      const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                      const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                      const float *d_filterData,
                      const float *d_inputData,
                      float *d_outputData,
                      cudaStream_t stream = 0);

#endif // FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_H
