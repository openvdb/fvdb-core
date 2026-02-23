// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANROLLINGSHUTTER_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANROLLINGSHUTTER_CUH

#include <fvdb/detail/ops/gsplat/GaussianCameraModels.h>

#include <cuda/std/cmath>

namespace fvdb::detail::ops {

template <typename T>
inline __device__ T
clamp01(const T x) {
    return (x < T(0)) ? T(0) : ((x > T(1)) ? T(1) : x);
}

/// @brief Convert a pixel coordinate to rolling-shutter time in \([0,1]\).
///
/// FVDB convention:
/// - vertical:    time varies with image row (y)
/// - horizontal:  time varies with image column (x)
/// - none:        time is 0
///
/// Pixel coordinates are in image space (not normalized), and we use `floor` (matching existing
/// kernels) before normalizing by (dim-1).
template <typename T>
inline __device__ T
rollingShutterTimeFromPixel(const RollingShutterType rollingShutterType,
                            const T px,
                            const T py,
                            const int64_t imageWidth,
                            const int64_t imageHeight) {
    if (rollingShutterType == RollingShutterType::NONE) {
        return T(0);
    }

    T u     = T(0);
    T denom = T(1);
    if (rollingShutterType == RollingShutterType::VERTICAL) {
        denom = (imageHeight > 1) ? T(imageHeight - 1) : T(1);
        u     = ::cuda::std::floor(py) / denom;
    } else if (rollingShutterType == RollingShutterType::HORIZONTAL) {
        denom = (imageWidth > 1) ? T(imageWidth - 1) : T(1);
        u     = ::cuda::std::floor(px) / denom;
    }
    return clamp01(u);
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANROLLINGSHUTTER_CUH
