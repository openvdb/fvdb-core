// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIAN2D_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIAN2D_CUH

#include <nanovdb/math/Math.h>

namespace fvdb::detail::ops {

template <typename ScalarType> struct alignas(32) Gaussian2D { // 32 bytes
    using vec2t = nanovdb::math::Vec2<ScalarType>;
    using vec3t = nanovdb::math::Vec3<ScalarType>;

    // First 16 bytes: id(4) + opacity(4) + xy(8, 8-byte aligned)
    int32_t id;         // 4 bytes  (offset 0)
    ScalarType opacity; // 4 bytes  (offset 4)
    vec2t xy;           // 8 bytes  (offset 8)

    // Second 16 bytes: conic(12) + pad(4)
    vec3t conic;     // 12 bytes (offset 16)
    ScalarType _pad; // 4 bytes  (offset 28)

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

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIAN2D_CUH
