// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAMATRIXUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAMATRIXUTILS_CUH

#include <nanovdb/math/Math.h>

#include <cuda/std/tuple>

namespace fvdb::detail::ops {

template <typename T>
inline __host__ __device__ cuda::std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
loadWorldToCamRtRowMajor4x4(const T *m44) {
    // Row-major 4x4 with last column = translation.
    return {nanovdb::math::Mat3<T>(m44[0],
                                   m44[1],
                                   m44[2],   // 1st row
                                   m44[4],
                                   m44[5],
                                   m44[6],   // 2nd row
                                   m44[8],
                                   m44[9],
                                   m44[10]), // 3rd row
            nanovdb::math::Vec3<T>(m44[3], m44[7], m44[11])};
}

template <typename T, typename Acc44>
inline __device__ cuda::std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
loadWorldToCamRtFromAccessor44(const Acc44 &m44 /* [C,4,4] */, const int64_t camId) {
    return {nanovdb::math::Mat3<T>(m44[camId][0][0],
                                   m44[camId][0][1],
                                   m44[camId][0][2],  // 1st row
                                   m44[camId][1][0],
                                   m44[camId][1][1],
                                   m44[camId][1][2],  // 2nd row
                                   m44[camId][2][0],
                                   m44[camId][2][1],
                                   m44[camId][2][2]), // 3rd row
            nanovdb::math::Vec3<T>(m44[camId][0][3], m44[camId][1][3], m44[camId][2][3])};
}

template <typename T, typename Acc33>
inline __device__ nanovdb::math::Mat3<T>
loadMat3FromAccessor33(const Acc33 &m33 /* [C,3,3] */, const int64_t camId) {
    return nanovdb::math::Mat3<T>(m33[camId][0][0],
                                  m33[camId][0][1],
                                  m33[camId][0][2],
                                  m33[camId][1][0],
                                  m33[camId][1][1],
                                  m33[camId][1][2],
                                  m33[camId][2][0],
                                  m33[camId][2][1],
                                  m33[camId][2][2]);
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAMATRIXUTILS_CUH
