// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_2_H
#define FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_2_H

#include <nanovdb/NanoVDB.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

void mainSparseConvolutionIGEMM_2(const std::vector<nanovdb::Coord> &inputPoints,
                                  const std::vector<nanovdb::Coord> &outputPoints,
                                  uint32_t benchmark_iters);

int test_sparse_convolution_igemm_nanovdb_cuda_2(int benchmark_iters = 10);

/// Non-template wrapper around the Sifakis v2 reference scatter-gather
/// convolution (SparseConvolveCudaReference) with hardcoded geometry
/// (3x3x3, C=64, K=128, stride 1).
///
/// All pointer arguments must reside in device memory.
/// @param outputLeafCount  Number of leaf nodes in the output grid.
/// @param inputGrid        NanoVDB ValueOnIndex grid for the input (device pointer).
/// @param outputGrid       NanoVDB ValueOnIndex grid for the output (device pointer).
/// @param d_filterData     Filter weights in Sifakis layout [3][3][3][128][64].
/// @param d_inputData      Input features in [valueCount][64] layout (index 0 = background).
/// @param d_outputData     Output features in [valueCount][128] layout (written by kernel).
/// @param stream           CUDA stream for the kernel launch.
void sifakisRefSparseConv_2(uint32_t outputLeafCount,
                            const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                            const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                            const float *d_filterData,
                            const float *d_inputData,
                            float *d_outputData,
                            cudaStream_t stream = 0);

/// Non-template wrapper around the CUTLASS implicit-GEMM sparse convolution
/// kernel (SparseFpropSm80Strided) with hardcoded geometry (3x3x3, C=64,
/// K=128, stride 1).
///
/// Same interface as sifakisRefSparseConv_2 but launches the optimized IGEMM
/// instead of the naive reference.  All pointer arguments must reside in
/// device memory.
void sifakisIGemmConv_2(uint32_t outputLeafCount,
                        const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *inputGrid,
                        const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *outputGrid,
                        const float *d_filterData,
                        const float *d_inputData,
                        float *d_outputData,
                        cudaStream_t stream = 0);

#endif // FVDB_DETAIL_OPS_CONVOLUTION_SIFAKIS_REF_2_H
