// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAINTRINSICS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAINTRINSICS_CUH

#include <nanovdb/math/Math.h>

namespace fvdb::detail::ops {

template <typename T> struct CameraIntrinsics {
    T fx, fy, cx, cy;
    inline __host__ __device__ CameraIntrinsics() = default;
    inline __host__ __device__ CameraIntrinsics(T fx_, T fy_, T cx_, T cy_)
        : fx(fx_)
        , fy(fy_)
        , cx(cx_)
        , cy(cy_) {}
    inline __host__ __device__ explicit CameraIntrinsics(const nanovdb::math::Mat3<T> &K)
        : fx(K[0][0])
        , fy(K[1][1])
        , cx(K[0][2])
        , cy(K[1][2]) {}
    inline __host__ __device__ explicit CameraIntrinsics(const T *rowMajorK9)
        : fx(rowMajorK9[0])
        , fy(rowMajorK9[4])
        , cx(rowMajorK9[2])
        , cy(rowMajorK9[5]) {}
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAINTRINSICS_CUH
