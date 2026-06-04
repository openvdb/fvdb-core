// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_MATH_ROTATION_CUH
#define FVDB_DETAIL_UTILS_CUDA_MATH_ROTATION_CUH

#include <nanovdb/math/Math.h>

namespace fvdb {
namespace detail {

/// Safely normalize a 3D vector.
///
/// Returns `v / ||v||` when `||v|| > 0`, otherwise returns zero.
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafe(const nanovdb::math::Vec3<T> &v) {
    const T n2 = v.dot(v);
    if (n2 > T(0)) {
        return v * (T(1) / sqrt(n2));
    }
    return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
}

/// Vector-Jacobian product for `y = normalizeSafe(x)`.
///
/// Given upstream gradient `v_y = dL/dy`, returns `dL/dx`.
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
normalizeSafeVJP(const nanovdb::math::Vec3<T> &x, const nanovdb::math::Vec3<T> &v_y) {
    const T n2 = x.dot(x);
    if (!(n2 > T(0))) {
        return nanovdb::math::Vec3<T>(T(0), T(0), T(0));
    }
    const T n     = sqrt(n2);
    const T invn  = T(1) / n;
    const T invn3 = invn * invn * invn;
    const T xdotv = x.dot(v_y);
    return v_y * invn - x * (xdotv * invn3);
}

/// Clamp a scalar to [0, 1].
template <typename T>
inline __device__ T
clamp01(const T x) {
    return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

/// @brief Converts a 3x3 rotation matrix to a quaternion [w,x,y,z].
///
/// Uses a branch-based algorithm for numerical robustness. Degenerate inputs
/// fall back to the identity quaternion.
template <typename T>
__host__ __device__ nanovdb::math::Vec4<T>
rotationMatrixToQuaternion(const nanovdb::math::Mat3<T> &R) {
    T trace = R[0][0] + R[1][1] + R[2][2];
    T x, y, z, w;

    const T s_min = (sizeof(T) == sizeof(float)) ? T(1e-8) : T(1e-12);

    if (trace > 0) {
        T t = trace + T(1);
        t   = (t > T(0)) ? t : T(0);
        T s = sqrt(t) * T(2);
        if (!(s > s_min)) {
            w = T(1);
            x = y = z = T(0);
        } else {
            w = T(0.25) * s;
            x = (R[2][1] - R[1][2]) / s;
            y = (R[0][2] - R[2][0]) / s;
            z = (R[1][0] - R[0][1]) / s;
        }
    } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
        T t = T(1) + R[0][0] - R[1][1] - R[2][2];
        t   = (t > T(0)) ? t : T(0);
        T s = sqrt(t) * T(2);
        if (!(s > s_min)) {
            w = T(1);
            x = y = z = T(0);
        } else {
            w = (R[2][1] - R[1][2]) / s;
            x = T(0.25) * s;
            y = (R[0][1] + R[1][0]) / s;
            z = (R[0][2] + R[2][0]) / s;
        }
    } else if (R[1][1] > R[2][2]) {
        T t = T(1) + R[1][1] - R[0][0] - R[2][2];
        t   = (t > T(0)) ? t : T(0);
        T s = sqrt(t) * T(2);
        if (!(s > s_min)) {
            w = T(1);
            x = y = z = T(0);
        } else {
            w = (R[0][2] - R[2][0]) / s;
            x = (R[0][1] + R[1][0]) / s;
            y = T(0.25) * s;
            z = (R[1][2] + R[2][1]) / s;
        }
    } else {
        T t = T(1) + R[2][2] - R[0][0] - R[1][1];
        t   = (t > T(0)) ? t : T(0);
        T s = sqrt(t) * T(2);
        if (!(s > s_min)) {
            w = T(1);
            x = y = z = T(0);
        } else {
            w = (R[1][0] - R[0][1]) / s;
            x = (R[0][2] + R[2][0]) / s;
            y = (R[1][2] + R[2][1]) / s;
            z = T(0.25) * s;
        }
    }

    const T norm2 = (w * w + x * x + y * y + z * z);
    if (norm2 > T(0)) {
        const T invNorm = T(1) / sqrt(norm2);
        w *= invNorm;
        x *= invNorm;
        y *= invNorm;
        z *= invNorm;
    } else {
        w = T(1);
        x = y = z = T(0);
    }

    if (w < T(0)) {
        w = -w;
        x = -x;
        y = -y;
        z = -z;
    }
    return nanovdb::math::Vec4<T>(w, x, y, z);
}

/// @brief Converts a quaternion [w,x,y,z] to a 3x3 rotation matrix.
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
quaternionToRotationMatrix(nanovdb::math::Vec4<T> const &quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T inverseNormalization = rsqrt(x * x + y * y + z * z + w * w);
    x *= inverseNormalization;
    y *= inverseNormalization;
    z *= inverseNormalization;
    w *= inverseNormalization;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return nanovdb::math::Mat3<T>((1.f - 2.f * (y2 + z2)),
                                  (2.f * (xy - wz)),
                                  (2.f * (xz + wy)),
                                  (2.f * (xy + wz)),
                                  (1.f - 2.f * (x2 + z2)),
                                  (2.f * (yz - wx)),
                                  (2.f * (xz - wy)),
                                  (2.f * (yz + wx)),
                                  (1.f - 2.f * (x2 + y2)));
}

/// @brief Normalizes a quaternion to unit length.
///
/// If the quaternion is zero, returns the identity quaternion.
template <typename T>
inline __host__ __device__ nanovdb::math::Vec4<T>
normalizeQuaternionSafe(nanovdb::math::Vec4<T> q) {
    const T n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (n2 > T(0)) {
        const T invN = T(1) / sqrt(n2);
        q[0] *= invN;
        q[1] *= invN;
        q[2] *= invN;
        q[3] *= invN;
    } else {
        q[0] = T(1);
        q[1] = q[2] = q[3] = T(0);
    }
    return q;
}

/// @brief NLERP between two quaternions along the shortest arc.
template <typename T>
inline __host__ __device__ nanovdb::math::Vec4<T>
nlerpQuaternionShortestPath(const nanovdb::math::Vec4<T> &q0,
                            nanovdb::math::Vec4<T> q1,
                            const T u) {
    T dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if (dot < T(0)) {
        q1[0] = -q1[0];
        q1[1] = -q1[1];
        q1[2] = -q1[2];
        q1[3] = -q1[3];
    }

    const T s = T(1) - u;
    return normalizeQuaternionSafe<T>(nanovdb::math::Vec4<T>(s * q0[0] + u * q1[0],
                                                             s * q0[1] + u * q1[1],
                                                             s * q0[2] + u * q1[2],
                                                             s * q0[3] + u * q1[3]));
}

/// @brief VJP for quaternion-to-rotation-matrix conversion.
///
/// Given dL/dR, computes dL/dq.
template <typename T>
inline __device__ nanovdb::math::Vec4<T>
quaternionToRotationMatrixVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat,
                                                const nanovdb::math::Mat3<T> &dLossDRotation) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    const T inverseNormalization = rsqrt(x * x + y * y + z * z + w * w);
    x *= inverseNormalization;
    y *= inverseNormalization;
    z *= inverseNormalization;
    w *= inverseNormalization;
    const nanovdb::math::Vec4<T> dLossDQuatNormalized(
        2.f * (x * (dLossDRotation[2][1] - dLossDRotation[1][2]) +
               y * (dLossDRotation[0][2] - dLossDRotation[2][0]) +
               z * (dLossDRotation[1][0] - dLossDRotation[0][1])),
        2.f * (-2.f * x * (dLossDRotation[1][1] + dLossDRotation[2][2]) +
               y * (dLossDRotation[1][0] + dLossDRotation[0][1]) +
               z * (dLossDRotation[2][0] + dLossDRotation[0][2]) +
               w * (dLossDRotation[2][1] - dLossDRotation[1][2])),
        2.f * (x * (dLossDRotation[1][0] + dLossDRotation[0][1]) -
               2.f * y * (dLossDRotation[0][0] + dLossDRotation[2][2]) +
               z * (dLossDRotation[2][1] + dLossDRotation[1][2]) +
               w * (dLossDRotation[0][2] - dLossDRotation[2][0])),
        2.f * (x * (dLossDRotation[2][0] + dLossDRotation[0][2]) +
               y * (dLossDRotation[2][1] + dLossDRotation[1][2]) -
               2.f * z * (dLossDRotation[0][0] + dLossDRotation[1][1]) +
               w * (dLossDRotation[1][0] - dLossDRotation[0][1])));

    const nanovdb::math::Vec4<T> quatNormalized(w, x, y, z);
    return (dLossDQuatNormalized - dLossDQuatNormalized.dot(quatNormalized) * quatNormalized) *
           inverseNormalization;
}

/// @brief Rigid transform (cached rotation + translation).
///
/// Quaternion is stored as [w,x,y,z]. The rotation matrix is cached.
template <typename T> struct RigidTransform {
    nanovdb::math::Mat3<T> R;
    nanovdb::math::Vec4<T> q;
    nanovdb::math::Vec3<T> t;

    /// Construct from quaternion and translation.
    inline __host__ __device__
    RigidTransform(const nanovdb::math::Vec4<T> &q_in, const nanovdb::math::Vec3<T> &t_in)
        : R(quaternionToRotationMatrix<T>(q_in)), q(q_in), t(t_in) {}

    /// Construct from rotation matrix and translation.
    inline __host__ __device__
    RigidTransform(const nanovdb::math::Mat3<T> &R_in, const nanovdb::math::Vec3<T> &t_in)
        : R(R_in), q(rotationMatrixToQuaternion<T>(R_in)), t(t_in) {}

    /// Apply the transform: R * p + t.
    inline __host__ __device__ nanovdb::math::Vec3<T>
    apply(const nanovdb::math::Vec3<T> &p_world) const {
        return R * p_world + t;
    }

    /// Interpolate between two rigid transforms (linear t, NLERP q).
    inline static __host__ __device__ RigidTransform<T>
    interpolate(const T u, const RigidTransform<T> &start, const RigidTransform<T> &end) {
        const nanovdb::math::Vec3<T> t_interp = start.t + u * (end.t - start.t);
        const nanovdb::math::Vec4<T> q_interp = nlerpQuaternionShortestPath<T>(start.q, end.q, u);
        return RigidTransform<T>(q_interp, t_interp);
    }
};

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_MATH_ROTATION_CUH
