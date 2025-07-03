// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH

#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>

#include <nanovdb/math/Math.h>

namespace fvdb::detail::ops {

template <typename ScalarType> struct alignas(32) Gaussian2D { // 28 bytes
    using vec2t = nanovdb::math::Vec2<ScalarType>;
    using vec3t = nanovdb::math::Vec3<ScalarType>;

    int32_t id;         // 4 bytes
    vec2t xy;           // 8 bytes
    ScalarType opacity; // 4 bytes
    vec3t conic;        // 12 bytes

    inline __device__ vec2t
    delta(const ScalarType px, const ScalarType py) const {
        return {xy[0] - px, xy[1] - py};
    }

    inline __device__ ScalarType
    sigma(const vec2t delta) const {
        return ScalarType{0.5} * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1]) +
               conic[1] * delta[0] * delta[1];
    }

    inline __device__ ScalarType
    sigma(const ScalarType px, const ScalarType py) const {
        return sigma(delta(px, py));
    }
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH
