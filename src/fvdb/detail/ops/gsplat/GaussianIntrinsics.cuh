// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANINTRINSICS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANINTRINSICS_CUH

#include <nanovdb/math/Math.h>

namespace fvdb::detail::ops {

template <typename T> struct CameraIntrinsics {
    T fx, fy, cx, cy;
};

template <typename T>
inline __host__ __device__ CameraIntrinsics<T>
loadIntrinsics(const nanovdb::math::Mat3<T> &K) {
    return CameraIntrinsics<T>{K[0][0], K[1][1], K[0][2], K[1][2]};
}

template <typename T>
inline __host__ __device__ CameraIntrinsics<T>
loadIntrinsicsRowMajor3x3(const T *K9) {
    return CameraIntrinsics<T>{K9[0], K9[4], K9[2], K9[5]};
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANINTRINSICS_CUH
